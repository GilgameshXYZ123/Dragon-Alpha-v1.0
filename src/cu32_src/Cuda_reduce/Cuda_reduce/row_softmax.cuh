#pragma once

#ifndef ROW_SOFTMAX_H
#define ROW_SOFTMAX_H

//(1) field_num % A.width ==0
//(2) field_num -> field_stride = field_num / width * stride
//(3) A.length % feature_num == 0
//(4) A.lengthv % fieled_stride == 0
//(5) row_num = A.lengthv / field_stride
//(6) [y, x] -> [row_num, field_stride]
//(7) [N, M] = [row_num, field_stride], A.lengthv = N*M
#ifndef ROW_SOFTMAX_CALL
#define ROW_SOFTMAX_CALL

//LBX>=4
#define row_softmax_fast16(stream, LBY, LBX, LTY, LTX, A, maxA, expA, N, M, V, SV, width, stride) \
	row_softmax_kernel_fast_16<LBY, LBX>\
		<<< dim3((M >>LBX >> LTX), ((N + (1<<LBY<<LTY) - 1)>>LBY>>LTY) ),\
			dim3(1<<LBX, 1<<LBY), 0, stream >>>\
			(A, maxA, expA, N, M, V, SV, width, stride)

//LBX>=3
#define row_softmax_fast8(stream, LBY, LBX, LTY, LTX, A, maxA, expA, N, M, V, SV, width, stride) \
	row_softmax_kernel_fast_8<LBY, LBX>\
		<<< dim3((M >> LBX >> LTX), ((N + (1<<LBY<<LTY) - 1)>>LBY>>LTY) ),\
			dim3(1<<LBX, 1<<LBY), 0, stream >>> \
			(A, maxA, expA, N, M, V, SV, width, stride)

//LBX=5, LTX=0, used for: M<=32
#define row_softmax_small(stream, LBY, LTY, A, maxA, expA, N, M, V, SV, width, stride) \
	row_softmax_kernel_slow_16<LBY, 5>\
		<<< dim3(1, ((N + (1<<LBY<<LTY) - 1)>>LBY>>LTY) ),\
		    dim3(32, 1<<LBY), 0, stream >>>\
			(A, maxA, expA, N, M, V, SV, width, stride)

#endif


#ifndef ROW_SOFTMAX_FAST_16
#define ROW_SOFTMAX_FAST_16

//maxA: max(exp(X)) of each row to avoid numerical overflow
template<int LBY, int LBX>
__global__ void row_softmax_kernel_fast_16(
	const float* __restrict__ A,
	const float* __restrict__ maxA,
	float* __restrict__ expA,
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

	int offset = y * M;
	for (A += offset, expA += offset; y < N; y += stepY)
	{
		float c = 0.0f;//solve the error
		float v = 0, ma = maxA[y];
		for (int x = offsetX; x < M; x += stepX4)//thread local reduce
		{
			float4 a = *(float4*)(A + x);//Y = exp(X - ma)
			a.x = expf(a.x - ma);
			a.y = expf(a.y - ma);
			a.z = expf(a.z - ma);
			a.w = expf(a.w - ma);
			
			within_width4(a, x, stride, width);
			*(float4*)(expA + x) = a;

			float dv = (a.x + a.y + a.z + a.w) - c;
			float t = v + dv;
			c = (t - v) - dv;
			v = t;//v += a.x + a.y + a.z + a.w;
		}
		As[As_yx] = v;

		int move = stepY * M;
		A    += move;//   A[Y + stepY][0]
		expA += move;//expA[Y + stepY][0]
		__syncthreads();

		if (LBX >= 7) {//block global reduce
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


#ifndef ROW_SOFTMAX_FAST_8
#define ROW_SOFTMAX_FAST_8

template<int LBY, int LBX>
__global__ void row_softmax_kernel_fast_8(
	const float* __restrict__ A,
	const float* __restrict__ maxA,
	float* __restrict__ expA,
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

	int offset = y * M;
	for (A += offset, expA += offset; y < N; y += stepY)
	{
		float c = 0.0f;//solve the error
		float v = 0, ma = maxA[y];
		for (int x = offsetX; x < M; x += stepX4)//thread local reduce
		{
			float4 a = *(float4*)(A + x);
			a.x = expf(a.x - ma);
			a.y = expf(a.y - ma);
			a.z = expf(a.z - ma);
			a.w = expf(a.w - ma);

			within_width4(a, x, stride, width);
			*(float4*)(expA + x) = a;

			float dv = (a.x + a.y + a.z + a.w) - c;
			float t = v + dv;
			c = (t - v) - dv;
			v = t;//v += a.x + a.y + a.z + a.w;
		}
		As[As_yx] = v;

		int move = stepY * M;
		A    += move;//   A[Y + stepY][0]
		expA += move;//expA[Y + stepY][0]
		__syncthreads();

		if (LBX >= 7) {//block global reduce
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
		if (tx < 4) warp_sum_4(As, As_yx);
		if (tx == 0) get(V, bx, y, SV) = As[As_yx];
		__syncthreads();
	}
}

#endif


#ifndef ROW_SOFTMAX_SLOW
#define ROW_SOFTMAX_SLOW

template<int LBY, int LBX>
__global__ void row_softmax_kernel_slow_16(
	const float* __restrict__ A, 
	const float* __restrict__ maxA,
	float* __restrict__ expA,  
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

	int offset = y * M;
	for (A += offset, expA += offset; y < N; y += stepY)
	{
		float c = 0.0f;//solve the error
		float v = 0, ma = maxA[y];
		for (int x = offsetX; x < M; x += stepX)//thread local reduce
		{
			float a = expf(A[x] - ma);//Y = exp(X -M)
			a *= ((x % stride) < width);
			expA[x] = a;

			float dv = a - c;
			float t = v + dv;
			c = (t - v) - dv;
			v = t;//v += a;
		}
		As[As_yx] = v;
		int move = stepY * M;
		A    += move;
		expA += move;
		__syncthreads();

		if (LBX >= 7) {//block global reduce
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


#ifndef ROW_SOFTMAX_STAGE
#define ROW_SOFTMAX_STAGE

//HV = nextRowReduceStageHeight(M)
//M%4 == 0, M >= 4  
void __row_softmax_stage(cudaStream_t stream,
	const float* A, float* maxA,
	float* expA, 
	int N, int M, 
	float* V, int SV,//stride of V
	int width, int stride)
{
	if (M > 255) {
		if (N > 31) { row_softmax_fast16(stream, 1, 4, 4, 4, A, maxA, expA, N, M, V, SV, width, stride); return; }
		if (N > 15) { row_softmax_fast16(stream, 1, 4, 3, 4, A, maxA, expA, N, M, V, SV, width, stride); return; }
		if (N >  7) { row_softmax_fast16(stream, 1, 4, 2, 4, A, maxA, expA, N, M, V, SV, width, stride); return; }
		if (N >  3) { row_softmax_fast16(stream, 1, 4, 1, 4, A, maxA, expA, N, M, V, SV, width, stride); return; }
		row_softmax_fast16(stream, 1, 4, 0, 4, A, maxA, expA, N, M, V, SV, width, stride); return;
	}

	if (M > 127) {
		if (N > 31) { row_softmax_fast8(stream, 2, 3, 3, 4, A, maxA, expA, N, M, V, SV, width, stride); return; }
		if (N > 15) { row_softmax_fast8(stream, 2, 3, 2, 4, A, maxA, expA, N, M, V, SV, width, stride); return; }
		if (N >  7) { row_softmax_fast8(stream, 2, 3, 1, 4, A, maxA, expA, N, M, V, SV, width, stride); return; }
		if (N >  3) { row_softmax_fast8(stream, 2, 3, 0, 4, A, maxA, expA, N, M, V, SV, width, stride); return; }
		row_softmax_fast8(stream, 1, 3, 0, 4, A, maxA, expA, N, M, V, SV, width, stride); return;
	}

	if (M > 63) {
		if (N > 31) { row_softmax_fast8(stream, 2, 3, 3, 3, A, maxA, expA, N, M, V, SV, width, stride); return; }
		if (N > 15) { row_softmax_fast8(stream, 2, 3, 2, 3, A, maxA, expA, N, M, V, SV, width, stride); return; }
		if (N > 7) { row_softmax_fast8(stream, 2, 3, 1, 3, A, maxA, expA, N, M, V, SV, width, stride); return; }
		if (N > 3) { row_softmax_fast8(stream, 2, 3, 0, 3, A, maxA, expA, N, M, V, SV, width, stride); return; }
		row_softmax_fast8(stream, 1, 3, 0, 3, A, maxA, expA, N, M, V, SV, width, stride); return;
	}

	if (M > 31) {//2^7, M >= 128
		if (N > 31) { row_softmax_fast8(stream, 2, 3, 3, 2, A, maxA, expA, N, M, V, SV, width, stride); return; }
		if (N > 15) { row_softmax_fast8(stream, 2, 3, 2, 2, A, maxA, expA, N, M, V, SV, width, stride); return; }
		if (N > 7) { row_softmax_fast8(stream, 2, 3, 1, 2, A, maxA, expA, N, M, V, SV, width, stride); return; }
		if (N > 3) { row_softmax_fast8(stream, 2, 3, 0, 2, A, maxA, expA, N, M, V, SV, width, stride); return; }
		row_softmax_fast8(stream, 1, 3, 0, 2, A, maxA, expA, N, M, V, SV, width, stride); return;
	}

	if (N > 31) { row_softmax_small(stream, 1, 4, A, maxA, expA, N, M, V, SV, width, stride); return; }
	if (N > 15) { row_softmax_small(stream, 1, 3, A, maxA, expA, N, M, V, SV, width, stride); return; }
	if (N > 7) { row_softmax_small(stream, 1, 2, A, maxA, expA, N, M, V, SV, width, stride); return; }
	if (N > 3) { row_softmax_small(stream, 1, 1, A, maxA, expA, N, M, V, SV, width, stride); return; }
	row_softmax_small(stream, 1, 0, A, maxA, expA, N, M, V, SV, width, stride);
}

#endif


#ifndef ROW_SOFTMAX
#define ROW_SOFTMAX

//new fashion
//V: use a template buffer to store median value for: A -> Y
//Y: must 1D tensor, no inner zero between results
int __row_softmax(cudaStream_t stream,
	const float* A, float*  maxA,
	float* expA, 
	int N, int M,
	float*  V,
	int width, int stride,
	int partNum)
{
	int nextM = row_nextM(M);
	int SV = (nextM <= partNum ? N : ((N + 3) >> 2 << 2));

	__row_softmax_stage(stream, A, maxA, expA, N, M, V, SV, width, stride);
	
	for (M = nextM; M > partNum; M = field_nextN(M, SV)) {
		__field_sum_stage(stream, V, M, SV, V, N, SV);//width = N, stride = SV
	}
	return M;
}

#endif

#endif