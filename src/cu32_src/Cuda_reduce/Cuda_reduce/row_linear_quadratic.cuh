#pragma once

#ifndef ROW_LINEAR_QUADRATIC_H
#define ROW_LINEAR_QUADRATIC_H

//(1) field_num % A.width ==0
//(2) field_num -> field_stride = field_num / width * stride
//(3) A.length % feature_num == 0
//(4) A.lengthv % fieled_stride == 0
//(5) row_num = A.lengthv / field_stride
//(6) [y, x] -> [row_num, field_stride]
//(7) [N, M] = [row_num, field_stride], A.lengthv = N*M
#ifndef ROW_LINEAR_QUADRATIC_CALL
#define ROW_LINEAR_QUADRATIC_CALL

//LBX>=4
#define row_linear_quadratic_fast16(stream, LBY, LBX, LTY, LTX, A, alpha1, beta1, alpha2, beta2, gamma2, N, M, V1, V2, SV, width, stride) \
	row_linear_quadratic_kernel_fast_16<LBY, LBX>\
		<<< dim3((M >> LBX >> LTX), ((N + (1<<LBY<<LTY) - 1)>>LBY>>LTY) ),\
			dim3(1<<LBX, 1<<LBY), 0, stream >>>\
			(A, alpha1, beta1, alpha2, beta2, gamma2, N, M, V1, V2, SV, width, stride)

//LBX>=3
#define row_linear_quadratic_fast8(stream, LBY, LBX, LTY, LTX, A, alpha1, beta1, alpha2, beta2, gamma2, N, M, V1, V2, SV, width, stride) \
	row_linear_quadratic_kernel_fast_8<LBY, LBX>\
		<<< dim3((M >> LBX >> LTX), ((N + (1<<LBY<<LTY) - 1)>>LBY>>LTY) ),\
			dim3(1<<LBX, 1<<LBY), 0, stream >>> \
			(A, alpha1, beta1, alpha2, beta2, gamma2, N, M, V1, V2, SV, width, stride)

//LBX=5, LTX=0, used for: M<=32
#define row_linear_quadratic_small(stream, LBY, LTY, A, alpha1, beta1, alpha2, beta2, gamma2, N, M, V1, V2, SV, width, stride) \
	row_linear_quadratic_kernel_slow_16<LBY, 5>\
		<<< dim3(1, ((N + (1<<LBY<<LTY) - 1)>>LBY>>LTY) ),\
		    dim3(32, 1<<LBY), 0, stream >>>\
			(A, alpha1, beta1, alpha2, beta2, gamma2, N, M, V1, V2, SV, width, stride)

#endif


#ifndef ROW_LINEAR_QUADRATIC_FAST_16
#define ROW_LINEAR_QUADRATIC_FAST_16

//M = row_lengthv
//N = field_lengthv
template<int LBY, int LBX>
__global__ void row_linear_quadratic_kernel_fast_16(
	const float* __restrict__ A,
	float alpha1, float beta1,
	float alpha2, float beta2, float gamma2,
	int N, int M,
	float* __restrict__ V1, 
	float* __restrict__ V2, int SV,//stride of V
	int width, int stride)
{
	int by = blockIdx.y, bx = blockIdx.x;
	int ty = threadIdx.y, tx = threadIdx.x;

	__shared__ float2 As[1 << LBY << LBX];//[BLOCK_SIZE_Y, BLOCK_SIZE_X]
	int As_yx = (ty << LBX) + tx;//[ty, tx]

	int y = (by << LBY) + ty, offsetX = ((bx << LBX) + tx) << 2;
	const int stepY = (gridDim.y << LBY);//go to the next row for this thread
	const int stepX4 = (gridDim.x << LBX) << 2; //go to the next element of the current row

	for (A += y * M; y < N; y += stepY)
	{
		float c1 = 0.0f, c2 = 0.0f;//solve the error
		float v1 = 0.0f, v2 = 0.0f;
		for (int x = offsetX; x < M; x += stepX4)//thread local reduce
		{
			float4 a = *(float4*)(A + x);
			float4 dv1; simdLinear4(dv1, alpha1, a, beta1);//dv1 = alpha1*a + beta1
			float4 dv2; simdQuadratic4(dv2, alpha2, a, beta2, gamma2);//dv2 = alpha2*a^2 + beta2*a + gamma2
			
			within_width4(dv1, x, stride, width);
			within_width4(dv2, x, stride, width);

			Kahan_sum4(v1, dv1, c1);//v1 += dv1
			Kahan_sum4(v2, dv2, c2);//v2 += dv2;
		}
		As[As_yx] = make_float2(v1, v2);//save: v1, v2
		A += stepY * M;//A[Y + stepY][0]
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
			get(V1, bx, y, SV) = result.x;
			get(V2, bx, y, SV) = result.y;
		}
		__syncthreads();
	}
}

#endif


#ifndef ROW_LINEAR_QUADRATIC_FAST_8
#define ROW_LINEAR_QUADRATIC_FAST_8

//M = row_lengthv
//N = field_lengthv
template<int LBY, int LBX>
__global__ void row_linear_quadratic_kernel_fast_8(
	const float* __restrict__ A,
	float alpha1, float beta1,
	float alpha2, float beta2, float gamma2,
	int N, int M,
	float* __restrict__ V1,
	float* __restrict__ V2, int SV,//stride of V
	int width, int stride)
{
	int by = blockIdx.y, bx = blockIdx.x;
	int ty = threadIdx.y, tx = threadIdx.x;

	__shared__ float2 As[1 << LBY << LBX];//[BLOCK_SIZE_Y, BLOCK_SIZE_X]
	int As_yx = (ty << LBX) + tx;//(ty, tx)

	int y = (by << LBY) + ty, offsetX = ((bx << LBX) + tx) << 2;
	const int stepY = (gridDim.y << LBY);//go to the next row for this thread
	const int stepX4 = (gridDim.x << LBX) << 2; //go to the next element of the current row

	for (A += y * M; y < N; y += stepY)
	{
		float c1 = 0.0f, c2 = 0.0f;//solve the error
		float v1 = 0.0f, v2 = 0.0f;
		for (int x = offsetX; x < M; x += stepX4)//thread local reduce
		{
			float4 a = *(float4*)(A + x);
			float4 dv1; simdLinear4(dv1, alpha1, a, beta1);//dv1 = alpha1*a + beta1
			float4 dv2; simdQuadratic4(dv2, alpha2, a, beta2, gamma2);//dv2 = alpha2*a^2 + beta2*a + gamma2

			within_width4(dv1, x, stride, width);
			within_width4(dv2, x, stride, width);

			Kahan_sum4(v1, dv1, c1);//v1 += dv1
			Kahan_sum4(v2, dv2, c2);//v2 += dv2;
		}
		As[As_yx] = make_float2(v1, v2);
		A += stepY * M;//A[Y + stepY][0]
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
			if (tx < 8) simdAdd2(As[As_yx], As[As_yx], As[As_yx + 8]);
			__syncthreads();
		}
		if (tx < 4) warp_simdSum2_4(As, As_yx);//LBX >= 3: 8 -> 1
		if (tx == 0) {//save
			float2 result = As[As_yx];//transposed: [y, bx] -> [bx, y]
			get(V1, bx, y, SV) = result.x;
			get(V2, bx, y, SV) = result.y;
		}
		__syncthreads();
	}
}

#endif


#ifndef ROW_LINEAR_QUADRATIC_SLOW
#define ROW_LINEAR_QUADRATIC_SLOW

template<int LBY, int LBX>
__global__ void row_linear_quadratic_kernel_slow_16(
	const float* __restrict__ A,
	float alpha1, float beta1,
	float alpha2, float beta2, float gamma2,
	int N, int M,
	float* __restrict__ V1,
	float* __restrict__ V2, int SV,//stride of V
	int width, int stride)
{
	int by = blockIdx.y, bx = blockIdx.x;
	int ty = threadIdx.y, tx = threadIdx.x;

	__shared__ float2 As[1 << LBY << LBX];//[BLOCK_SIZE_Y, BLOCK_SIZE_X]
	int As_yx = (ty << LBX) + tx;//[ty, tx]

	int y = (by << LBY) + ty, offsetX = (bx << LBX) + tx;
	const int stepY = (gridDim.y << LBY), stepX = (gridDim.x << LBX);

	for (A += y * M; y < N; y += stepY)
	{
		float c1 = 0.0f, c2 = 0.0f;//solve the error
		float v1 = 0.0f, v2 = 0.0f;
		for (int x = offsetX; x < M; x += stepX)//thread local reduce
		{
			float a = A[x];
			float dv1 = alpha1 * a + beta1;
			float dv2 = alpha2 * a*a + beta2 * a + gamma2;
			
			bool within_width = ((x % stride) < width);
			dv1 *= within_width;//within_width
			dv2 *= within_width;//within_width

			Kahan_sum1(v1, dv1, c1);//v1 += dv1
			Kahan_sum1(v2, dv2, c2);//v2 += dv2;
		}
		As[As_yx] = make_float2(v1, v2);//save: v1, v2
		A += stepY * M;
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
			get(V1, bx, y, SV) = result.x;
			get(V2, bx, y, SV) = result.y;
		}
		__syncthreads();
	}
}

#endif


#ifndef ROW_LINEAR_QUADRATIC_STAGE
#define ROW_LINEAR_QUADRATIC_STAGE

//HV = nextRowReduceStageHeight(M)
//V[HV, N], A:[N, M]
//V[i] is the ith field vector of V
//A[i] is the ith row vector of A
//sum(V[i], 0, HV) = sum(A[i], 0, M)
//M%4 == 0, M >= 4 

void __row_linear_quadratic_stage(cudaStream_t stream,
	const float * __restrict__ A,
	float alpha1, float beta1,
	float alpha2, float beta2, float gamma2,
	int N, int M,
	float* __restrict__ V1,
	float* __restrict__ V2, int SV,//stride of V
	int width, int stride)
{
	if (M > 255) {
		if (N > 31) { row_linear_quadratic_fast16(stream, 1, 4, 4, 4, A, alpha1, beta1, alpha2, beta2, gamma2, N, M, V1, V2, SV, width, stride); return; }
		if (N > 15) { row_linear_quadratic_fast16(stream, 1, 4, 3, 4, A, alpha1, beta1, alpha2, beta2, gamma2, N, M, V1, V2, SV, width, stride); return; }
		if (N >  7) { row_linear_quadratic_fast16(stream, 1, 4, 2, 4, A, alpha1, beta1, alpha2, beta2, gamma2, N, M, V1, V2, SV, width, stride); return; }
		if (N >  3) { row_linear_quadratic_fast16(stream, 1, 4, 1, 4, A, alpha1, beta1, alpha2, beta2, gamma2, N, M, V1, V2, SV, width, stride); return; }
		row_linear_quadratic_fast16(stream, 1, 4, 0, 4, A, alpha1, beta1, alpha2, beta2, gamma2, N, M, V1, V2, SV, width, stride); return;
	}

	if (M > 127) {
		if (N > 31) { row_linear_quadratic_fast8(stream, 2, 3, 3, 4, A, alpha1, beta1, alpha2, beta2, gamma2, N, M, V1, V2, SV, width, stride); return; }
		if (N > 15) { row_linear_quadratic_fast8(stream, 2, 3, 2, 4, A, alpha1, beta1, alpha2, beta2, gamma2, N, M, V1, V2, SV, width, stride); return; }
		if (N >  7) { row_linear_quadratic_fast8(stream, 2, 3, 1, 4, A, alpha1, beta1, alpha2, beta2, gamma2, N, M, V1, V2, SV, width, stride); return; }
		if (N >  3) { row_linear_quadratic_fast8(stream, 2, 3, 0, 4, A, alpha1, beta1, alpha2, beta2, gamma2, N, M, V1, V2, SV, width, stride); return; }
		row_linear_quadratic_fast8(stream, 1, 3, 0, 4, A, alpha1, beta1, alpha2, beta2, gamma2, N, M, V1, V2, SV, width, stride); return;
	}

	if (M > 63) {
		if (N > 31) { row_linear_quadratic_fast8(stream, 2, 3, 3, 3, A, alpha1, beta1, alpha2, beta2, gamma2, N, M, V1, V2, SV, width, stride); return; }
		if (N > 15) { row_linear_quadratic_fast8(stream, 2, 3, 2, 3, A, alpha1, beta1, alpha2, beta2, gamma2, N, M, V1, V2, SV, width, stride); return; }
		if (N >  7) { row_linear_quadratic_fast8(stream, 2, 3, 1, 3, A, alpha1, beta1, alpha2, beta2, gamma2, N, M, V1, V2, SV, width, stride); return; }
		if (N >  3) { row_linear_quadratic_fast8(stream, 2, 3, 0, 3, A, alpha1, beta1, alpha2, beta2, gamma2, N, M, V1, V2, SV, width, stride); return; }
		row_linear_quadratic_fast8(stream, 1, 3, 0, 3, A, alpha1, beta1, alpha2, beta2, gamma2, N, M, V1, V2, SV, width, stride); return;
	}

	if (M > 31) {//2^7, M >= 128
		if (N > 31) { row_linear_quadratic_fast8(stream, 2, 3, 3, 2, A, alpha1, beta1, alpha2, beta2, gamma2, N, M, V1, V2, SV, width, stride); return; }
		if (N > 15) { row_linear_quadratic_fast8(stream, 2, 3, 2, 2, A, alpha1, beta1, alpha2, beta2, gamma2, N, M, V1, V2, SV, width, stride); return; }
		if (N >  7) { row_linear_quadratic_fast8(stream, 2, 3, 1, 2, A, alpha1, beta1, alpha2, beta2, gamma2, N, M, V1, V2, SV, width, stride); return; }
		if (N >  3) { row_linear_quadratic_fast8(stream, 2, 3, 0, 2, A, alpha1, beta1, alpha2, beta2, gamma2, N, M, V1, V2, SV, width, stride); return; }
		row_linear_quadratic_fast8(stream, 1, 3, 0, 2, A, alpha1, beta1, alpha2, beta2, gamma2, N, M, V1, V2, SV, width, stride); return;
	}

	if (N > 31) { row_linear_quadratic_small(stream, 1, 4, A, alpha1, beta1, alpha2, beta2, gamma2, N, M, V1, V2, SV, width, stride); return; }
	if (N > 15) { row_linear_quadratic_small(stream, 1, 3, A, alpha1, beta1, alpha2, beta2, gamma2, N, M, V1, V2, SV, width, stride); return; }
	if (N >  7) { row_linear_quadratic_small(stream, 1, 2, A, alpha1, beta1, alpha2, beta2, gamma2, N, M, V1, V2, SV, width, stride); return; }
	if (N >  3) { row_linear_quadratic_small(stream, 1, 1, A, alpha1, beta1, alpha2, beta2, gamma2, N, M, V1, V2, SV, width, stride); return; }
	row_linear_quadratic_small(stream, 1, 0, A, alpha1, beta1, alpha2, beta2, gamma2, N, M, V1, V2, SV, width, stride);
}

#endif


#ifndef ROW_LINEAR_QUADRATIC
#define ROW_LINEAR_QUADRATIC

//new fashion
//V used a template buffer to store median value for: A -> Y
//Y: must 1D tensor, no inner zero between results
int __row_linear_quadratic(JNIEnv *env, cudaStream_t stream1, cudaStream_t stream2,
	const float* A,
	float alpha1, float beta1,
	float alpha2, float beta2, float gamma2,
	int N, int M,
	float* V1, float *Y1,
	float* V2, float *Y2,
	int width, int stride,
	int partNum)
{
	int nextM = row_nextM(M);
	if (nextM <= partNum) {//only 1 stage: directly write result to Y tightly, so SV = N
		__row_linear_quadratic_stage(stream1, A, alpha1, beta1, alpha2, beta2, gamma2, N, M, Y1, Y2, N, width, stride);
		return nextM;
	}

	//at least 2 stages: tansposed: A[N, M] -> V[nextM, SV]
	int SV = (N + 3) >> 2 << 2;///make sure: SV >= 4, SV % 4 == 0
	__row_linear_quadratic_stage(stream1, A, alpha1, beta1, alpha2, beta2, gamma2, N, M, V1, V2, SV, width, stride);

	//====stream2 shold wait stream1======================================================
	cudaEvent_t event; cudaError_t error;
	error = cudaEventCreate(&event, cudaEventDisableTiming); handleError(error);
	error = cudaEventRecord(event, stream1); handleError(error);
	error = cudaStreamWaitEvent(stream2, event, cudaEventWaitDefault); handleError(error);
	//====stream2 shold wait stream1======================================================

	M = nextM, nextM = field_nextN(M, SV);
	while (nextM > partNum)//end: nextN <= partNum
	{
		__field_sum_stage(stream1, V1, M, SV, V1, N, SV);//width = N, stride = SV
		__field_sum_stage(stream2, V2, M, SV, V2, N, SV);//width = N, stride = SV
		M = nextM, nextM = field_nextN(M, SV);
	}
	__field_sum_stage(stream1, V1, M, SV, Y1, N, SV);
	__field_sum_stage(stream2, V2, M, SV, Y2, N, SV);

	error = cudaEventDestroy(event); handleError(error);
	return nextM;
}

#endif

#endif