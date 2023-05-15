#pragma once

#ifndef ROW_LAYERNORM_DELTAXP_V1_H
#define ROW_LAYERNORM_DELTAXP_V1_H

//(1) field_num % A.width ==0
//(2) field_num -> field_stride = field_num / width * stride
//(3) A.length % feature_num == 0
//(4) A.lengthv % fieled_stride == 0
//(5) row_num = A.lengthv / field_stride
//(6) [y, x] -> [row_num, field_stride]
//(7) [N, M] = [row_num, field_stride], A.lengthv = N*M
#ifndef ROW_LAYERNORM_DELTAXP_V1_CALL
#define ROW_LAYERNORM_DELTAXP_V1_CALL

//LBX>=4
#define row_layernorm_deltaXp_v1_fast16(stream, LBY, LBX, LTY, LTX, deltaY, Y, X_mean, X_square_mean, eps, N, M, deltaXp1, deltaXp2, SV, width, stride) \
	row_layernorm_deltaXp_v1_kernel_fast_16<LBY, LBX>\
		<<< dim3((M >> LBX >> LTX), ((N + (1<<LBY<<LTY) - 1)>>LBY>>LTY) ),\
			dim3(1<<LBX, 1<<LBY), 0, stream >>>\
			(deltaY, Y, X_mean, X_square_mean, eps, N, M, deltaXp1, deltaXp2, SV, width, stride)

//LBX>=3
#define row_layernorm_deltaXp_v1_fast8(stream, LBY, LBX, LTY, LTX, deltaY, Y, X_mean, X_square_mean, eps, N, M, deltaXp1, deltaXp2, SV, width, stride) \
	row_layernorm_deltaXp_v1_kernel_fast_8<LBY, LBX>\
		<<< dim3((M >> LBX >> LTX), ((N + (1<<LBY<<LTY) - 1)>>LBY>>LTY) ),\
			dim3(1<<LBX, 1<<LBY), 0, stream >>> \
			(deltaY, Y, X_mean, X_square_mean, eps, N, M, deltaXp1, deltaXp2, SV, width, stride)

//LBX=5, LTX=0, used for: M<=32
#define row_layernorm_deltaXp_v1_small(stream, LBY, LTY, deltaY, Y, X_mean, X_square_mean, eps, N, M, deltaXp1, deltaXp2, SV, width, stride) \
	row_layernorm_deltaXp_v1_kernel_slow_16<LBY, 5>\
		<<< dim3(1, ((N + (1<<LBY<<LTY) - 1)>>LBY>>LTY) ),\
		    dim3(32, 1<<LBY), 0, stream >>>\
			(deltaY, Y, X_mean, X_square_mean, eps, N, M, deltaXp1, deltaXp2, SV, width, stride)

#endif


#ifndef ROW_LAYERNORM_DELTAXP_V1_FAST_16
#define ROW_LAYERNORM_DELTAXP_V1_FAST_16

//affined = false
//V1: holdY(), Y is not changed
//[1] deltaY[N, M], Y[N, M]
//[2] X_mean[N]: mean of each row of X
//[3] X_squareMean[N]: mean of each row of X^2
//[4] X_std[N] = sqrt(X_squareMean - X_mean^2 + eps)
//[5] deltaXp1[N] = row_sum: deltaY * (Y*X_mean - X_std)
//    deltaXp2[N] = row_sum: deltaY * Y
template<int LBY, int LBX>
__global__ void row_layernorm_deltaXp_v1_kernel_fast_16(
	const float* __restrict__ deltaY,
	const float* __restrict__ Y,
	const float* __restrict__ X_mean,
	const float* __restrict__ X_square_mean, float eps,
	int N, int M,
	float* __restrict__ deltaXp1,
	float* __restrict__ deltaXp2, int SV,//stride of V
	int width, int stride)
{
	int by = blockIdx.y, bx = blockIdx.x;
	int ty = threadIdx.y, tx = threadIdx.x;

	__shared__ float2 As[1 << LBY << LBX];//[BLOCK_SIZE_Y, BLOCK_SIZE_X]
	int As_yx = (ty << LBX) + tx;//[ty, tx]

	int y = (by << LBY) + ty, offsetX = ((bx << LBX) + tx) << 2;
	const int stepY = (gridDim.y << LBY);//go to the next row for this thread
	const int stepX4 = (gridDim.x << LBX) << 2; //go to the next element of the current row

	int offset = y * M;
	for (deltaY += offset, Y += offset; y < N; y += stepY)
	{
		//X_std = sqrt(X_squareMean - X_mean^2 + eps)
		float x_mean = X_mean[y];
		float x_smean = X_square_mean[y];
		float x_std = sqrtf(x_smean - x_mean * x_mean + eps);

		float c1 = 0.0f, c2 = 0.0f;//solve the error
		float v1 = 0.0f, v2 = 0.0f;
		for (int x = offsetX; x < M; x += stepX4)//thread local reduce
		{
			float4 dy = *(float4*)(deltaY + x);//deltaY[y, x]
			float4 yv = *(float4*)(Y + x);//Y[y, x]

			float4 dx1;//<1> deltaXp1 = deltaY * (Y*X_mean - X_std)
			dx1.x = dy.x * (yv.x*x_mean - x_std);
			dx1.y = dy.y * (yv.y*x_mean - x_std);
			dx1.z = dy.z * (yv.z*x_mean - x_std);
			dx1.w = dy.w * (yv.w*x_mean - x_std);
			
			float4 dx2;//<2> deltaXp2 = deltaY * Y
			dx2.x = dy.x * yv.x;
			dx2.y = dy.y * yv.y;
			dx2.z = dy.z * yv.z;
			dx2.w = dy.w * yv.w;

			within_width4(dx1, x, stride, width);
			within_width4(dx2, x, stride, width);

			Kahan_sum4(v1, dx1, c1);//v1 += dx1
			Kahan_sum4(v2, dx2, c2);//v2 += dx2;
		}
		As[As_yx] = make_float2(v1, v2);//save: v1, v2
		int move = stepY * M;
		deltaY += move; Y += move;
		__syncthreads();

		//block global reduce
		if (LBX >= 7) {//128 -> 64
			if (tx < 64) simdAdd2(As[As_yx], As[As_yx], As[As_yx + 64]);
			__syncthreads();
		}
		if (LBX >= 6) {//64 -> 32
			if (tx < 32) simdAdd2(As[As_yx], As[As_yx], As[As_yx + 32]);
			__syncthreads();
		}
		if (LBX >= 5) {//32 -> 16
			if (tx < 16) simdAdd2(As[As_yx], As[As_yx], As[As_yx + 16]);
			__syncthreads();
		}
		if (tx < 8) warp_simdSum2_8(As, As_yx);//LBX >= 4: 16 -> 1
		if (tx == 0) {//save
			float2 result = As[As_yx];//transposed: [y, bx] -> [bx, y]
			get(deltaXp1, bx, y, SV) = result.x;
			get(deltaXp2, bx, y, SV) = result.y;
		}
		__syncthreads();
	}
}

#endif


#ifndef ROW_LAYERNORM_DELTAXP_V1_FAST_8
#define ROW_LAYERNORM_DELTAXP_V1_FAST_8

//affined = false
//V1: holdY(), Y is not changed
//[1] deltaY[N, M], Y[N, M]
//[2] X_mean[N]: mean of each row of X
//[3] X_squareMean[N]: mean of each row of X^2
//[4] X_std[N] = sqrt(X_squareMean - X_mean^2 + eps)
//[5] deltaXp1[N] = row_sum: deltaY * (Y*X_mean - X_std)
//    deltaXp2[N] = row_sum: deltaY * Y
template<int LBY, int LBX>
__global__ void row_layernorm_deltaXp_v1_kernel_fast_8(
	const float* __restrict__ deltaY,
	const float* __restrict__ Y,
	const float* __restrict__ X_mean,
	const float* __restrict__ X_square_mean, float eps,
	int N, int M,
	float* __restrict__ deltaXp1,
	float* __restrict__ deltaXp2, int SV,//stride of V
	int width, int stride)
{
	int by = blockIdx.y, bx = blockIdx.x;
	int ty = threadIdx.y, tx = threadIdx.x;

	__shared__ float2 As[1 << LBY << LBX];//[BLOCK_SIZE_Y, BLOCK_SIZE_X]
	int As_yx = (ty << LBX) + tx;//[ty, tx]

	int y = (by << LBY) + ty, offsetX = ((bx << LBX) + tx) << 2;
	const int stepY = (gridDim.y << LBY);//go to the next row for this thread
	const int stepX4 = (gridDim.x << LBX) << 2; //go to the next element of the current row

	int offset = y * M;
	for (deltaY += offset, Y += offset; y < N; y += stepY)
	{
		//X_std = sqrt(X_squareMean - X_mean^2 + eps)
		float x_mean = X_mean[y];
		float x_smean = X_square_mean[y];
		float x_std = sqrtf(x_smean - x_mean * x_mean + eps);

		float c1 = 0.0f, c2 = 0.0f;//solve the error
		float v1 = 0.0f, v2 = 0.0f;
		for (int x = offsetX; x < M; x += stepX4)//thread local reduce
		{
			float4 dy = *(float4*)(deltaY + x);//deltaY[y, x]
			float4 yv = *(float4*)(Y + x);//Y[y, x]

			float4 dx1;//<1> deltaXp1 = deltaY * (Y*X_mean - X_std)
			dx1.x = dy.x * (yv.x*x_mean - x_std);
			dx1.y = dy.y * (yv.y*x_mean - x_std);
			dx1.z = dy.z * (yv.z*x_mean - x_std);
			dx1.w = dy.w * (yv.w*x_mean - x_std);

			float4 dx2;//<2> deltaXp2 = deltaY * Y
			dx2.x = dy.x * yv.x;
			dx2.y = dy.y * yv.y;
			dx2.z = dy.z * yv.z;
			dx2.w = dy.w * yv.w;

			within_width4(dx1, x, stride, width);
			within_width4(dx2, x, stride, width);

			Kahan_sum4(v1, dx1, c1);//v1 += dx1
			Kahan_sum4(v2, dx2, c2);//v2 += dx2;
		}
		As[As_yx] = make_float2(v1, v2);//save: v1, v2
		int move = stepY * M;
		deltaY += move; Y += move;
		__syncthreads();

		//block global reduce
		if (LBX >= 7) {//128 -> 64
			if (tx < 64) simdAdd2(As[As_yx], As[As_yx], As[As_yx + 64]);
			__syncthreads();
		}
		if (LBX >= 6) {//64 -> 32
			if (tx < 32) simdAdd2(As[As_yx], As[As_yx], As[As_yx + 32]);
			__syncthreads();
		}
		if (LBX >= 5) {//32 -> 16
			if (tx < 16) simdAdd2(As[As_yx], As[As_yx], As[As_yx + 16]);
			__syncthreads();
		}
		if (LBX >= 4) {//16 -> 8
			if (tx < 16) simdAdd2(As[As_yx], As[As_yx], As[As_yx + 8]);
			__syncthreads();
		}
		if (tx < 4) warp_simdSum2_4(As, As_yx);//LBX >= 4: 8 -> 1
		if (tx == 0) {//save
			float2 result = As[As_yx];//transposed: [y, bx] -> [bx, y]
			get(deltaXp1, bx, y, SV) = result.x;
			get(deltaXp2, bx, y, SV) = result.y;
		}
		__syncthreads();
	}
}

#endif


#ifndef ROW_LAYERNORM_DELTAXP_V1_SLOW
#define ROW_LAYERNORM_DELTAXP_V1_SLOW

//affined = false
//V1: holdY(), Y is not changed
//[1] deltaY[N, M], Y[N, M]
//[2] X_mean[N]: mean of each row of X
//[3] X_squareMean[N]: mean of each row of X^2
//[4] X_std[N] = sqrt(X_squareMean - X_mean^2 + eps)
//[5] deltaXp1[N] = row_sum: deltaY * (Y*X_mean - X_std)
//    deltaXp2[N] = row_sum: deltaY * Y
template<int LBY, int LBX>
__global__ void row_layernorm_deltaXp_v1_kernel_slow_16(
	const float* __restrict__ deltaY,
	const float* __restrict__ Y,
	const float* __restrict__ X_mean,
	const float* __restrict__ X_square_mean, float eps,
	int N, int M,
	float* __restrict__ deltaXp1,
	float* __restrict__ deltaXp2, int SV,//stride of V
	int width, int stride)
{
	int by = blockIdx.y, bx = blockIdx.x;
	int ty = threadIdx.y, tx = threadIdx.x;

	__shared__ float2 As[1 << LBY << LBX];//[BLOCK_SIZE_Y, BLOCK_SIZE_X]
	int As_yx = (ty << LBX) + tx;//[ty, tx]

	int y = (by << LBY) + ty, offsetX = (bx << LBX) + tx;
	const int stepY = (gridDim.y << LBY), stepX = (gridDim.x << LBX);

	int offset = y * M;
	for (deltaY += offset, Y += offset; y < N; y += stepY)
	{
		//X_std = sqrt(X_squareMean - X_mean^2 + eps)
		float x_mean = X_mean[y];
		float x_smean = X_square_mean[y];
		float x_std = sqrtf(x_smean - x_mean * x_mean + eps);

		float c1 = 0.0f, c2 = 0.0f;//solve the error
		float v1 = 0.0f, v2 = 0.0f;
		for (int x = offsetX; x < M; x += stepX)//thread local reduce
		{
			float dy = deltaY[x];//deltaY[y, x]
			float yv = Y[x];//Y[y, x]

			//<1> deltaXp1 = deltaY * (Y*X_mean - X_std)
			float dx1 = dy * (yv*x_mean - x_std);

			//<2> deltaXp2 = deltaY * Y
			float dx2 = dy * yv;

			bool within_width = ((x % stride) < width);
			dx1 *= within_width;//within_width
			dx2 *= within_width;//within_width

			Kahan_sum1(v1, dx1, c1);//v1 += dv1
			Kahan_sum1(v2, dx2, c2);//v2 += dv2;
		}
		As[As_yx] = make_float2(v1, v2);//save: v1, v2
		int move = stepY * M;
		deltaY += move; Y += move;
		__syncthreads();

		//block global reduce
		if (LBX >= 7) {//128 -> 64
			if (tx < 64) simdAdd2(As[As_yx], As[As_yx], As[As_yx + 64]);
			__syncthreads();
		}
		if (LBX >= 6) {//64 -> 32
			if (tx < 32) simdAdd2(As[As_yx], As[As_yx], As[As_yx + 32]);
			__syncthreads();
		}
		if (LBX >= 5) {//32 -> 16
			if (tx < 16) simdAdd2(As[As_yx], As[As_yx], As[As_yx + 16]);
			__syncthreads();
		}
		if (tx < 8) warp_simdSum2_8(As, As_yx);//LBX >= 4: 16 -> 1
		if (tx == 0) {
			float2 result = As[As_yx];//transposed: [y, bx] -> [bx, y]
			get(deltaXp1, bx, y, SV) = result.x;
			get(deltaXp2, bx, y, SV) = result.y;
		}
		__syncthreads();
	}
}

#endif


#ifndef ROW_LAYERNORM_DELTAXP_V1_STAGE
#define ROW_LAYERNORM_DELTAXP_V1_STAGE

//HV = nextRowReduceStageHeight(M)
//V[HV, N], A:[N, M]
//V[i] is the ith field vector of V
//A[i] is the ith row vector of A
//sum(V[i], 0, HV) = sum(A[i], 0, M)
//M%4 == 0, M >= 4 
void __row_layernorm_deltaXp_v1_stage(cudaStream_t stream,
	const float* deltaY, const float* Y,
	const float* X_mean, const float* X_square_mean, float eps,
	int N, int M,
	float* deltaXp1, float* deltaXp2, int SV,//stride of V
	int width, int stride)
{
	if (M > 255) {
		if (N > 31) { row_layernorm_deltaXp_v1_fast16(stream, 1, 4, 4, 4, deltaY, Y, X_mean, X_square_mean, eps, N, M, deltaXp1, deltaXp2, SV, width, stride); return; }
		if (N > 15) { row_layernorm_deltaXp_v1_fast16(stream, 1, 4, 3, 4, deltaY, Y, X_mean, X_square_mean, eps, N, M, deltaXp1, deltaXp2, SV, width, stride); return; }
		if (N >  7) { row_layernorm_deltaXp_v1_fast16(stream, 1, 4, 2, 4, deltaY, Y, X_mean, X_square_mean, eps, N, M, deltaXp1, deltaXp2, SV, width, stride); return; }
		if (N >  3) { row_layernorm_deltaXp_v1_fast16(stream, 1, 4, 1, 4, deltaY, Y, X_mean, X_square_mean, eps, N, M, deltaXp1, deltaXp2, SV, width, stride); return; }
		row_layernorm_deltaXp_v1_fast16(stream, 1, 4, 0, 4, deltaY, Y, X_mean, X_square_mean, eps, N, M, deltaXp1, deltaXp2, SV, width, stride); return;
	}

	if (M > 127) {
		if (N > 31) { row_layernorm_deltaXp_v1_fast8(stream, 2, 3, 3, 4, deltaY, Y, X_mean, X_square_mean, eps, N, M, deltaXp1, deltaXp2, SV, width, stride); return; }
		if (N > 15) { row_layernorm_deltaXp_v1_fast8(stream, 2, 3, 2, 4, deltaY, Y, X_mean, X_square_mean, eps, N, M, deltaXp1, deltaXp2, SV, width, stride); return; }
		if (N >  7) { row_layernorm_deltaXp_v1_fast8(stream, 2, 3, 1, 4, deltaY, Y, X_mean, X_square_mean, eps, N, M, deltaXp1, deltaXp2, SV, width, stride); return; }
		if (N >  3) { row_layernorm_deltaXp_v1_fast8(stream, 2, 3, 0, 4, deltaY, Y, X_mean, X_square_mean, eps, N, M, deltaXp1, deltaXp2, SV, width, stride); return; }
		row_layernorm_deltaXp_v1_fast8(stream, 1, 3, 0, 4, deltaY, Y, X_mean, X_square_mean, eps, N, M, deltaXp1, deltaXp2, SV, width, stride); return;
	}

	if (M > 63) {
		if (N > 31) { row_layernorm_deltaXp_v1_fast8(stream, 2, 3, 3, 3, deltaY, Y, X_mean, X_square_mean, eps, N, M, deltaXp1, deltaXp2, SV, width, stride); return; }
		if (N > 15) { row_layernorm_deltaXp_v1_fast8(stream, 2, 3, 2, 3, deltaY, Y, X_mean, X_square_mean, eps, N, M, deltaXp1, deltaXp2, SV, width, stride); return; }
		if (N >  7) { row_layernorm_deltaXp_v1_fast8(stream, 2, 3, 1, 3, deltaY, Y, X_mean, X_square_mean, eps, N, M, deltaXp1, deltaXp2, SV, width, stride); return; }
		if (N >  3) { row_layernorm_deltaXp_v1_fast8(stream, 2, 3, 0, 3, deltaY, Y, X_mean, X_square_mean, eps, N, M, deltaXp1, deltaXp2, SV, width, stride); return; }
		row_layernorm_deltaXp_v1_fast8(stream, 1, 3, 0, 3, deltaY, Y, X_mean, X_square_mean, eps, N, M, deltaXp1, deltaXp2, SV, width, stride); return;
	}

	if (M > 31) {//2^7, M >= 128
		if (N > 31) { row_layernorm_deltaXp_v1_fast8(stream, 2, 3, 3, 2, deltaY, Y, X_mean, X_square_mean, eps, N, M, deltaXp1, deltaXp2, SV, width, stride); return; }
		if (N > 15) { row_layernorm_deltaXp_v1_fast8(stream, 2, 3, 2, 2, deltaY, Y, X_mean, X_square_mean, eps, N, M, deltaXp1, deltaXp2, SV, width, stride); return; }
		if (N >  7) { row_layernorm_deltaXp_v1_fast8(stream, 2, 3, 1, 2, deltaY, Y, X_mean, X_square_mean, eps, N, M, deltaXp1, deltaXp2, SV, width, stride); return; }
		if (N >  3) { row_layernorm_deltaXp_v1_fast8(stream, 2, 3, 0, 2, deltaY, Y, X_mean, X_square_mean, eps, N, M, deltaXp1, deltaXp2, SV, width, stride); return; }
		row_layernorm_deltaXp_v1_fast8(stream, 1, 3, 0, 2, deltaY, Y, X_mean, X_square_mean, eps, N, M, deltaXp1, deltaXp2, SV, width, stride); return;
	}

	if (N > 31) { row_layernorm_deltaXp_v1_small(stream, 1, 4, deltaY, Y, X_mean, X_square_mean, eps, N, M, deltaXp1, deltaXp2, SV, width, stride); return; }
	if (N > 15) { row_layernorm_deltaXp_v1_small(stream, 1, 3, deltaY, Y, X_mean, X_square_mean, eps, N, M, deltaXp1, deltaXp2, SV, width, stride); return; }
	if (N >  7) { row_layernorm_deltaXp_v1_small(stream, 1, 2, deltaY, Y, X_mean, X_square_mean, eps, N, M, deltaXp1, deltaXp2, SV, width, stride); return; }
	if (N >  3) { row_layernorm_deltaXp_v1_small(stream, 1, 1, deltaY, Y, X_mean, X_square_mean, eps, N, M, deltaXp1, deltaXp2, SV, width, stride); return; }
	row_layernorm_deltaXp_v1_small(stream, 1, 0, deltaY, Y, X_mean, X_square_mean, eps, N, M, deltaXp1, deltaXp2, SV, width, stride);
}

#endif


#ifndef ROW_LAYERNORM_DELTAXP_V1
#define ROW_LAYERNORM_DELTAXP_V1

//new fashion
//V used a template buffer to store median value for: A -> Y
//Y: must 1D tensor, no inner zero between results
int __row_layernorm_deltaXp_v1(JNIEnv *env, cudaStream_t stream1, cudaStream_t stream2,
	const float* deltaY, const float* Y,
	const float* X_mean, const float* X_square_mean, float eps,
	int N, int M,
	float* deltaXp1, float* deltaXp2,
	int width, int stride,
	int partNum)
{
	int nextM = row_nextM(M);
	if (nextM <= partNum) {//only 1 stage: directly write result to Y tightly, so SV = N
		__row_layernorm_deltaXp_v1_stage(stream1, deltaY, Y, X_mean, X_square_mean, eps, N, M, deltaXp1, deltaXp2, N, width, stride);
		return nextM;
	}

	//at least 2 stages: tansposed: A[N, M] -> V[nextM, SV]
	int SV = (N + 3) >> 2 << 2;///make sure: SV >= 4, SV % 4 == 0
	__row_layernorm_deltaXp_v1_stage(stream1, deltaY, Y, X_mean, X_square_mean, eps, N, M, deltaXp1, deltaXp2, SV, width, stride);

	//====stream2 shold wait stream1======================================================
	cudaEvent_t event; cudaError_t error;
	error = cudaEventCreate(&event, cudaEventDisableTiming); handleError(error);
	error = cudaEventRecord(event, stream1); handleError(error);
	error = cudaStreamWaitEvent(stream2, event, cudaEventWaitDefault); handleError(error);
	//====stream2 shold wait stream1======================================================

	M = nextM, nextM = field_nextN(M, SV);
	while (nextM > partNum)//end: nextN <= partNum
	{
		__field_sum_stage(stream1, deltaXp1, M, SV, deltaXp1, N, SV);//width = N, stride = SV
		__field_sum_stage(stream2, deltaXp2, M, SV, deltaXp2, N, SV);//width = N, stride = SV
		M = nextM, nextM = field_nextN(M, SV);
	}
	__field_sum_stage(stream1, deltaXp1, M, SV, deltaXp1, N, SV);
	__field_sum_stage(stream2, deltaXp2, M, SV, deltaXp2, N, SV);

	error = cudaEventDestroy(event); handleError(error);
	return nextM;
}

#endif

#endif