#pragma once

#ifndef ROW_QUADRATIC_H
#define ROW_QUADRATIC_H

//(1) field_num % A.width ==0
//(2) field_num -> field_stride = field_num / width * stride
//(3) A.length % feature_num == 0
//(4) A.lengthv % fieled_stride == 0
//(5) row_num = A.lengthv / field_stride
//(6) [y, x] -> [row_num, field_stride]
//(7) [N, M] = [row_num, field_stride], A.lengthv = N*M
#ifndef ROW_QUADRATIC_CALL
#define ROW_QUADRATIC_CALL

//LBX>=4
#define row_quadratic_fast16(stream, LBY, LBX, LTY, LTX, A, alpha, beta, gamma, N, M, V, SV, width, stride) \
	row_quadratic_kernel_fast_16<LBY, LBX>\
		<<< dim3((M >> LBX >> LTX), ((N + (1<<LBY<<LTY) - 1)>>LBY>>LTY) ),\
			dim3(1<<LBX, 1<<LBY), 0, stream >>>\
			(A, alpha, beta, gamma, N, M, V, SV, width, stride)

//LBX>=3
#define row_quadratic_fast8(stream, LBY, LBX, LTY, LTX, A, alpha, beta, gamma, N, M, V, SV,width, stride) \
	row_quadratic_kernel_fast_8<LBY, LBX>\
		<<< dim3((M >> LBX >> LTX), ((N + (1<<LBY<<LTY) - 1)>>LBY>>LTY) ),\
			dim3(1<<LBX, 1<<LBY), 0, stream >>> \
			(A, alpha, beta, gamma, N, M, V, SV, width, stride)

//LBX=5, LTX=0, used for: M<=32
#define row_quadratic_small(stream, LBY, LTY, A, alpha, beta, gamma, N, M, V, SV, width, stride) \
	row_quadratic_kernel_slow_16<LBY, 5>\
		<<< dim3(1, ((N + (1<<LBY<<LTY) - 1)>>LBY>>LTY) ),\
		    dim3(32, 1<<LBY), 0, stream >>>\
			(A, alpha, beta, gamma, N, M, V, SV, width, stride)

#endif


#ifndef ROW_QUADRATIC_FAST_16
#define ROW_QUADRATIC_FAST_16

//M = row_lengthv
//N = field_lengthv
template<int LBY, int LBX>
__global__ void row_quadratic_kernel_fast_16(
	const float* __restrict__ A,
	float alpha, float beta, float gamma,
	int N, int M,
	float* __restrict__ V, int SV,//stride of V
	int width, int stride)
{
	int by = blockIdx.y, bx = blockIdx.x;
	int ty = threadIdx.y, tx = threadIdx.x;

	__shared__ float As[1 << LBY << LBX];//(1<<LBY, 1<<LBX)
	int As_yx = (ty << LBX) + tx;//(ty, tx)

	int y = (by << LBY) + ty, offsetX = ((bx << LBX) + tx) << 2;
	const int stepY = (gridDim.y << LBY);//go to the next row for this thread
	const int stepX4 = (gridDim.x << LBX) << 2; //go to the next element of the current row

	for (A += y * M; y < N; y += stepY)
	{
		float c = 0.0f;//solve the error
		float v = 0;
		for (int x = offsetX; x < M; x += stepX4)//thread local reduce
		{
			float4 a = *(float4*)(A + x);
			simdQuadratic4(a, alpha, a, beta, gamma);
			within_width4(a, x, stride, width);

			float dv = (a.x + a.y + a.z + a.w) - c;
			float t = v + dv;
			c = (t - v) - dv;
			v = t;//v += a.x + a.y + a.z + a.w;
		}
		As[As_yx] = v;

		A += stepY * M;//A[Y + stepY][0]
		__syncthreads();

		//block global reduce
		if (LBX >= 7) {
			if (tx < 64) As[As_yx] += As[As_yx + 64];
			__syncthreads();
		}
		if (LBX >= 6) {
			if (tx < 32) As[As_yx] += As[As_yx + 32];
			__syncthreads();
		}
		if (LBX >= 5) {
			if (tx < 16) As[As_yx] += As[As_yx + 16];
			__syncthreads();
		}
		if (tx < 8) warp_sum_8(As, As_yx);//LBX >= 4
		if (tx == 0) get(V, bx, y, SV) = As[As_yx];
		__syncthreads();
	}
}

#endif


#ifndef ROW_QUADRATIC_FAST_8
#define ROW_QUADRATIC_FAST_8

template<int LBY, int LBX>
__global__ void row_quadratic_kernel_fast_8(
	const float* __restrict__ A,
	float alpha, float beta, float gamma,
	int N, int M, 
	float* __restrict__ V, int SV,//stride of V
	int width, int stride)
{
	int by = blockIdx.y, bx = blockIdx.x;
	int ty = threadIdx.y, tx = threadIdx.x;

	__shared__ float As[1 << LBY << LBX];//(1<<LBY, 1<<LBX)
	int As_yx = (ty << LBX) + tx;//(ty, tx)
	int y = (by << LBY) + ty, offsetX = ((bx << LBX) + tx) << 2;
	const int stepY = (gridDim.y << LBY);//go to the next row for this thread
	const int stepX4 = (gridDim.x << LBX) << 2; //go to the next element of the current row

	for (A += y * M; y < N; y += stepY)
	{
		float c = 0.0f;//solve the error
		float v = 0.0f;
		for (int x = offsetX; x < M; x += stepX4)//thread local reduce
		{
			float4 a = *(float4*)(A + x);
			simdQuadratic4(a, alpha, a, beta, gamma);
			within_width4(a, x, stride, width);
			
			float dv = (a.x + a.y + a.z + a.w) - c;
			float t = v + dv;
			c = (t - v) - dv;
			v = t;//v += a.x + a.y + a.z + a.w;
		}
		As[As_yx] = v;

		A += stepY * M;//A[Y + stepY][0]
		__syncthreads();

		//block global reduce
		if (LBX >= 7) {
			if (tx < 64) As[As_yx] += As[As_yx + 64];
			__syncthreads();
		}
		if (LBX >= 6) {
			if (tx < 32) As[As_yx] += As[As_yx + 32];
			__syncthreads();
		}
		if (LBX >= 5) {
			if (tx < 16) As[As_yx] += As[As_yx + 16];
			__syncthreads();
		}
		if (LBX >= 4) {
			if (tx < 8) As[As_yx] += As[As_yx + 8];
			__syncthreads();
		}
		if (tx < 4) warp_sum_4(As, As_yx);//LBX >= 3
		if (tx == 0) get(V, bx, y, SV) = As[As_yx];
		__syncthreads();
	}
}

#endif


#ifndef ROW_QUADRATIC_SLOW
#define ROW_QUADRATIC_SLOW

template<int LBY, int LBX>
__global__ void row_quadratic_kernel_slow_16(
	const float* __restrict__ A,
	float alpha, float beta, float gamma,
	int N, int M, 
	float* __restrict__ V, int SV,//stride of V
	int width, int stride)
{
	int by = blockIdx.y, bx = blockIdx.x;
	int ty = threadIdx.y, tx = threadIdx.x;

	__shared__ float As[1 << LBY << LBX];//(1<<LBY, 1<<LBX)
	int As_yx = (ty << LBX) + tx;//(ty, tx)

	int y = (by << LBY) + ty, offsetX = (bx << LBX) + tx;
	const int stepY = (gridDim.y << LBY), stepX = (gridDim.x << LBX);

	for (A += y * M; y < N; y += stepY)
	{
		float c = 0.0f;//solve the error
		float v = 0;
		for (int x = offsetX; x < M; x += stepX)//thread local reduce
		{
			float a = A[x];
			a = alpha * (a*a) + beta * a + gamma;
			a *= ((x % stride) < width);//within_width

			float dv = a - c;
			float t = v + dv;
			c = (t - v) - dv;
			v = t;//v += a
		}
		As[As_yx] = v;

		A += stepY * M;
		__syncthreads();

		//block global reduce
		if (LBX >= 7) {
			if (tx < 64) As[As_yx] += As[As_yx + 64];
			__syncthreads();
		}
		if (LBX >= 6) {
			if (tx < 32) As[As_yx] += As[As_yx + 32];
			__syncthreads();
		}
		if (LBX >= 5) {
			if (tx < 16) As[As_yx] += As[As_yx + 16];
			__syncthreads();
		}
		if (tx < 8) warp_sum_8(As, As_yx);
		if (tx == 0) get(V, bx, y, SV) = As[As_yx];
		__syncthreads();
	}
}

#endif


#ifndef ROW_QUADRATIC_STAGE
#define ROW_QUADRATIC_STAGE

//HV = nextRowReduceStageHeight(M)
//V[HV, N], A:[N, M]
//V[i] is the ith field vector of V
//A[i] is the ith row vector of A
//sum(V[i], 0, HV) = sum(A[i], 0, M)
//M%4 == 0, M >= 4  

void __row_quadratic_stage(cudaStream_t stream,
	const float*  A,
	float alpha, float beta, float gamma,
	int N, int M,
	float* V, int SV,//stride of V
	int width, int stride)
{
	if (M > 255) {
		if (N > 31) { row_quadratic_fast16(stream, 1, 4, 4, 4, A, alpha, beta, gamma, N, M, V, SV, width, stride); return; }
		if (N > 15) { row_quadratic_fast16(stream, 1, 4, 3, 4, A, alpha, beta, gamma, N, M, V, SV, width, stride); return; }
		if (N >  7) { row_quadratic_fast16(stream, 1, 4, 2, 4, A, alpha, beta, gamma, N, M, V, SV, width, stride); return; }
		if (N >  3) { row_quadratic_fast16(stream, 1, 4, 1, 4, A, alpha, beta, gamma, N, M, V, SV, width, stride); return; }
		row_quadratic_fast16(stream, 1, 4, 0, 4, A, alpha, beta, gamma, N, M, V, SV, width, stride); return;
	}

	if (M > 127) {
		if (N > 31) { row_quadratic_fast8(stream, 2, 3, 3, 4, A, alpha, beta, gamma, N, M, V, SV, width, stride); return; }
		if (N > 15) { row_quadratic_fast8(stream, 2, 3, 2, 4, A, alpha, beta, gamma, N, M, V, SV, width, stride); return; }
		if (N >  7) { row_quadratic_fast8(stream, 2, 3, 1, 4, A, alpha, beta, gamma, N, M, V, SV, width, stride); return; }
		if (N >  3) { row_quadratic_fast8(stream, 2, 3, 0, 4, A, alpha, beta, gamma, N, M, V, SV, width, stride); return; }
		row_quadratic_fast8(stream, 1, 3, 0, 4, A, alpha, beta, gamma, N, M, V, SV, width, stride); return;
	}
	
	if (M > 63) {
		if (N > 31) { row_quadratic_fast8(stream, 2, 3, 3, 3, A, alpha, beta, gamma, N, M, V, SV, width, stride); return; }
		if (N > 15) { row_quadratic_fast8(stream, 2, 3, 2, 3, A, alpha, beta, gamma, N, M, V, SV, width, stride); return; }
		if (N >  7) { row_quadratic_fast8(stream, 2, 3, 1, 3, A, alpha, beta, gamma, N, M, V, SV, width, stride); return; }
		if (N >  3) { row_quadratic_fast8(stream, 2, 3, 0, 3, A, alpha, beta, gamma, N, M, V, SV, width, stride); return; }
		row_quadratic_fast8(stream, 1, 3, 0, 3, A, alpha, beta, gamma, N, M, V, SV, width, stride); return;
	}

	if (M > 31) {//2^7, M >= 128
		if (N > 31) { row_quadratic_fast8(stream, 2, 3, 3, 2, A, alpha, beta, gamma, N, M, V, SV, width, stride); return; }
		if (N > 15) { row_quadratic_fast8(stream, 2, 3, 2, 2, A, alpha, beta, gamma, N, M, V, SV, width, stride); return; }
		if (N >  7) { row_quadratic_fast8(stream, 2, 3, 1, 2, A, alpha, beta, gamma, N, M, V, SV, width, stride); return; }
		if (N >  3) { row_quadratic_fast8(stream, 2, 3, 0, 2, A, alpha, beta, gamma, N, M, V, SV, width, stride); return; }
		row_quadratic_fast8(stream, 1, 3, 0, 2, A, alpha, beta, gamma, N, M, V, SV, width, stride); return;
	}

	if (N > 31) { row_quadratic_small(stream, 1, 4, A, alpha, beta, gamma, N, M, V, SV, width, stride); return; }
	if (N > 15) { row_quadratic_small(stream, 1, 3, A, alpha, beta, gamma, N, M, V, SV, width, stride); return; }
	if (N >  7) { row_quadratic_small(stream, 1, 2, A, alpha, beta, gamma, N, M, V, SV, width, stride); return; }
	if (N >  3) { row_quadratic_small(stream, 1, 1, A, alpha, beta, gamma, N, M, V, SV, width, stride); return; }
	row_quadratic_small(stream, 1, 0, A, alpha, beta, gamma, N, M, V, SV, width, stride);
}
	
#endif


#ifndef ROW_QUADRATIC
#define ROW_QUADRATIC

//new fashion
//V used a template buffer to store median value for: A -> Y
//Y: must 1D tensor, no inner zero between results
int __row_quadratic(cudaStream_t stream,
	const float*  A,
	float alpha, float beta, float gamma,
	int N, int M,
	float* V, float *Y,
	int width, int stride,
	int partNum)
{
	int nextM = row_nextM(M);
	if (nextM <= partNum) {//only 1 stage: directly write result to Y tightly, so SV = N
		__row_quadratic_stage(stream, A, alpha, beta, gamma, N, M, Y, N, width, stride);
		return nextM;
	}

	//at least 2 stages: tansposed: //A[N, M] -> V[nextM, SV]
	int SV = (N + 3) >> 2 << 2;///make sure: SV >= 4, SV % 4 == 0
	__row_quadratic_stage(stream, A, alpha, beta, gamma, N, M, V, SV, width, stride);

	M = nextM, nextM = field_nextN(M, SV);
	while (nextM > partNum)//end: nextN <= partNum
	{
		__field_sum_stage(stream, V, M, SV, V, N, SV);//width = N, stride = SV
		M = nextM, nextM = field_nextN(M, SV);
	}
	__field_sum_stage(stream, V, M, SV, Y, N, SV);
	return nextM;
}

#endif

#endif