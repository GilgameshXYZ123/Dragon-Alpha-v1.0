#pragma once

#ifndef ROW_QUADRATIC_DUAL_H
#define ROW_QUADRATIC_DUAL_H

//(1) field_num % [A.width, B.width] ==0
//(2) field_num -> field_stride = field_num / width * stride
//(3) [A.length,  B.length ] % feature_num == 0
//(4) [A.lengthv, B.lengthv] % fieled_stride == 0
//(5) row_num = A.lengthv / field_stride
//(6) [y, x] -> [row_num, field_stride]
//(7) [N, M] = [row_num, field_stride], A.lengthv = N*M
#ifndef ROW_QUADRATIC_DUAL_CALL
#define ROW_QUADRATIC_DUAL_CALL

//LBX>=4
#define row_quadratic_dual_fast16(stream, LBY, LBX, LTY, LTX, A, B, k11, k12, k22, k1, k2, C, N, M, V, SV, width, stride) \
	row_quadratic_dual_kernel_fast_16<LBY, LBX>\
		<<< dim3((M >> LBX >> LTX), ((N + (1<<LBY<<LTY) - 1)>>LBY>>LTY) ),\
			dim3(1<<LBX, 1<<LBY), 0, stream >>>\
			(A, B, k11, k12, k22, k1, k2, C, N, M, V, SV, width, stride)

//LBX>=3
#define row_quadratic_dual_fast8(stream, LBY, LBX, LTY, LTX, A, B, k11, k12, k22, k1, k2, C, N, M, V, SV,width, stride) \
	row_quadratic_dual_kernel_fast_8<LBY, LBX>\
		<<< dim3((M >> LBX >> LTX), ((N + (1<<LBY<<LTY) - 1)>>LBY>>LTY) ),\
			dim3(1<<LBX, 1<<LBY), 0, stream >>> \
			(A, B, k11, k12, k22, k1, k2, C, N, M, V, SV, width, stride)

//LBX=5, LTX=0, used for: M<=32
#define row_quadratic_dual_small(stream, LBY, LTY, A, B, k11, k12, k22, k1, k2, C, N, M, V, SV, width, stride) \
	row_quadratic_dual_kernel_slow_16<LBY, 5>\
		<<< dim3(1, ((N + (1<<LBY<<LTY) - 1)>>LBY>>LTY) ),\
		    dim3(32, 1<<LBY), 0, stream >>>\
			(A, B, k11, k12, k22, k1, k2, C, N, M, V, SV, width, stride)

#endif


#ifndef ROW_QUADRATIC_DUAL_FAST_16
#define ROW_QUADRATIC_DUAL_FAST_16

//M = row_lengthv
//N = field_lengthv
template<int LBY, int LBX>
__global__ void row_quadratic_dual_kernel_fast_16(
	const float* __restrict__ A,
	const float* __restrict__ B,
	float k11, float k12, float k22,
	float k1, float k2, float C,
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
	for (A += offset, B += offset; y < N; y += stepY)
	{
		float c = 0.0f;//solve the error
		float v = 0;
		for (int x = offsetX; x < M; x += stepX4)//thread local reduce
		{
			float4 a = *(float4*)(A + x);
			float4 b = *(float4*)(B + x);

			float4 y;
			y.x = (k11 * a.x*a.x) + (k12 * a.x*b.x) + (k22 * b.x*b.x) + (k1 * a.x) + (k2 * b.x) + C;
			y.y = (k11 * a.y*a.y) + (k12 * a.y*b.y) + (k22 * b.y*b.y) + (k1 * a.y) + (k2 * b.y) + C;
			y.z = (k11 * a.z*a.z) + (k12 * a.z*b.z) + (k22 * b.z*b.z) + (k1 * a.z) + (k2 * b.z) + C;
			y.w = (k11 * a.w*a.w) + (k12 * a.w*b.w) + (k22 * b.w*b.w) + (k1 * a.w) + (k2 * b.w) + C;
			within_width4(y, x, stride, width);

			float dv = (y.x + y.y + y.z + y.w) - c;
			float t = v + dv;
			c = (t - v) - dv;
			v = t;//v += y.x + y.y + y.z + y.w;
		}
		As[As_yx] = v;

		int move = stepY * M;
		A += move;//A[Y + stepY][0]
		B += move;//B[Y + stepY][0]
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


#ifndef ROW_QUADRATIC_DUAL_FAST_8
#define ROW_QUADRATIC_DUAL_FAST_8

template<int LBY, int LBX>
__global__ void row_quadratic_dual_kernel_fast_8(
	const float* __restrict__ A,
	const float* __restrict__ B,
	float k11, float k12, float k22,
	float k1, float k2, float C,
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
	for (A += offset, B += offset; y < N; y += stepY)
	{
		float c = 0.0f;//solve the error
		float v = 0.0f;
		for (int x = offsetX; x < M; x += stepX4)//thread local reduce
		{
			float4 a = *(float4*)(A + x);
			float4 b = *(float4*)(B + x);

			float4 y;
			y.x = (k11 * a.x*a.x) + (k12 * a.x*b.x) + (k22 * b.x*b.x) + (k1 * a.x) + (k2 * b.x) + C;
			y.y = (k11 * a.y*a.y) + (k12 * a.y*b.y) + (k22 * b.y*b.y) + (k1 * a.y) + (k2 * b.y) + C;
			y.z = (k11 * a.z*a.z) + (k12 * a.z*b.z) + (k22 * b.z*b.z) + (k1 * a.z) + (k2 * b.z) + C;
			y.w = (k11 * a.w*a.w) + (k12 * a.w*b.w) + (k22 * b.w*b.w) + (k1 * a.w) + (k2 * b.w) + C;
			within_width4(y, x, stride, width);

			float dv = (y.x + y.y + y.z + y.w) - c;
			float t = v + dv;
			c = (t - v) - dv;
			v = t;//v += y.x + y.y + y.z + y.w;
		}
		As[As_yx] = v;

		int move = stepY * M;
		A += move;//A[Y + stepY][0]
		B += move;//B[Y + stepY][0]
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


#ifndef ROW_QUADRATIC_DUAL_SLOW
#define ROW_QUADRATIC_DUAL_SLOW

template<int LBY, int LBX>
__global__ void row_quadratic_dual_kernel_slow_16(
	const float* __restrict__ A,
	const float* __restrict__ B,
	float k11, float k12, float k22,
	float k1, float k2, float C,
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
	for (A += offset, B += offset; y < N; y += stepY)
	{
		float c = 0.0f;//solve the error
		float v = 0;
		for (int x = offsetX; x < M; x += stepX)//thread local reduce
		{
			float a = A[x];
			float b = B[x];

			float y = (k11 * a*a) + (k12 * a*b) + (k22 *b*b) + (k1*a) + (k2*b) + C;
			y *= ((x % stride) < width);//within_width

			float dv = y - c;
			float t = v + dv;
			c = (t - v) - dv;
			v = t;//v += a
		}
		As[As_yx] = v;

		int move = stepY * M;
		A += move;
		B += move;
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


#ifndef ROW_QUADRATIC_DUAL_STAGE
#define ROW_QUADRATIC_DUAL_STAGE

//HV = nextRowReduceStageHeight(M)
void __row_quadratic_dual_stage(cudaStream_t stream,
	const float* A, const float* B,
	float k11, float k12, float k22,
	float k1, float k2, float C,
	int N, int M,
	float* V, int SV,//stride of V
	int width, int stride)
{
	if (M > 255) {
		if (N > 31) { row_quadratic_dual_fast16(stream, 1, 4, 4, 4, A, B, k11, k12, k22, k1, k2, C, N, M, V, SV, width, stride); return; }
		if (N > 15) { row_quadratic_dual_fast16(stream, 1, 4, 3, 4, A, B, k11, k12, k22, k1, k2, C, N, M, V, SV, width, stride); return; }
		if (N >  7) { row_quadratic_dual_fast16(stream, 1, 4, 2, 4, A, B, k11, k12, k22, k1, k2, C, N, M, V, SV, width, stride); return; }
		if (N >  3) { row_quadratic_dual_fast16(stream, 1, 4, 1, 4, A, B, k11, k12, k22, k1, k2, C, N, M, V, SV, width, stride); return; }
		row_quadratic_dual_fast16(stream, 1, 4, 0, 4, A, B, k11, k12, k22, k1, k2, C, N, M, V, SV, width, stride); return;
	}

	if (M > 127) {
		if (N > 31) { row_quadratic_dual_fast8(stream, 2, 3, 3, 4, A, B, k11, k12, k22, k1, k2, C, N, M, V, SV, width, stride); return; }
		if (N > 15) { row_quadratic_dual_fast8(stream, 2, 3, 2, 4, A, B, k11, k12, k22, k1, k2, C, N, M, V, SV, width, stride); return; }
		if (N >  7) { row_quadratic_dual_fast8(stream, 2, 3, 1, 4, A, B, k11, k12, k22, k1, k2, C, N, M, V, SV, width, stride); return; }
		if (N >  3) { row_quadratic_dual_fast8(stream, 2, 3, 0, 4, A, B, k11, k12, k22, k1, k2, C, N, M, V, SV, width, stride); return; }
		row_quadratic_dual_fast8(stream, 1, 3, 0, 4, A, B, k11, k12, k22, k1, k2, C, N, M, V, SV, width, stride); return;
	}

	if (M > 63) {
		if (N > 31) { row_quadratic_dual_fast8(stream, 2, 3, 3, 3, A, B, k11, k12, k22, k1, k2, C, N, M, V, SV, width, stride); return; }
		if (N > 15) { row_quadratic_dual_fast8(stream, 2, 3, 2, 3, A, B, k11, k12, k22, k1, k2, C, N, M, V, SV, width, stride); return; }
		if (N >  7) { row_quadratic_dual_fast8(stream, 2, 3, 1, 3, A, B, k11, k12, k22, k1, k2, C, N, M, V, SV, width, stride); return; }
		if (N >  3) { row_quadratic_dual_fast8(stream, 2, 3, 0, 3, A, B, k11, k12, k22, k1, k2, C, N, M, V, SV, width, stride); return; }
		row_quadratic_dual_fast8(stream, 1, 3, 0, 3, A, B, k11, k12, k22, k1, k2, C, N, M, V, SV, width, stride); return;
	}

	if (M > 31) {//2^7, M >= 128
		if (N > 31) { row_quadratic_dual_fast8(stream, 2, 3, 3, 2, A, B, k11, k12, k22, k1, k2, C, N, M, V, SV, width, stride); return; }
		if (N > 15) { row_quadratic_dual_fast8(stream, 2, 3, 2, 2, A, B, k11, k12, k22, k1, k2, C, N, M, V, SV, width, stride); return; }
		if (N >  7) { row_quadratic_dual_fast8(stream, 2, 3, 1, 2, A, B, k11, k12, k22, k1, k2, C, N, M, V, SV, width, stride); return; }
		if (N >  3) { row_quadratic_dual_fast8(stream, 2, 3, 0, 2, A, B, k11, k12, k22, k1, k2, C, N, M, V, SV, width, stride); return; }
		row_quadratic_dual_fast8(stream, 1, 3, 0, 2, A, B, k11, k12, k22, k1, k2, C, N, M, V, SV, width, stride); return;
	}

	if (N > 31) { row_quadratic_dual_small(stream, 1, 4, A, B, k11, k12, k22, k1, k2, C, N, M, V, SV, width, stride); return; }
	if (N > 15) { row_quadratic_dual_small(stream, 1, 3, A, B, k11, k12, k22, k1, k2, C, N, M, V, SV, width, stride); return; }
	if (N >  7) { row_quadratic_dual_small(stream, 1, 2, A, B, k11, k12, k22, k1, k2, C, N, M, V, SV, width, stride); return; }
	if (N >  3) { row_quadratic_dual_small(stream, 1, 1, A, B, k11, k12, k22, k1, k2, C, N, M, V, SV, width, stride); return; }
	row_quadratic_dual_small(stream, 1, 0, A, B, k11, k12, k22, k1, k2, C, N, M, V, SV, width, stride);
}

#endif


#ifndef ROW_QUADRATIC_DUAL
#define ROW_QUADRATIC_DUAL

//new fashion
//V used a template buffer to store median value for: A -> Y
//Y: must 1D tensor, no inner zero between results
int __row_quadratic_dual(cudaStream_t stream,
	const float* A, const float* B,
	float k11, float k12, float k22, 
	float k1, float k2, float C,
	int N, int M,
	float* V, float *Y,
	int width, int stride,
	int partNum)
{
	int nextM = row_nextM(M);
	if (nextM <= partNum) {//only 1 stage: directly write result to Y tightly, so SV = N
		__row_quadratic_dual_stage(stream, A, B, k11, k12, k22, k1, k2, C, N, M, Y, N, width, stride);
		return nextM;
	}

	//at least 2 stages: tansposed: //A[N, M] -> V[nextM, SV]
	int SV = (N + 3) >> 2 << 2;///make sure: SV >= 4, SV % 4 == 0
	__row_quadratic_dual_stage(stream, A, B, k11, k12, k22, k1, k2, C, N, M, V, SV, width, stride);

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