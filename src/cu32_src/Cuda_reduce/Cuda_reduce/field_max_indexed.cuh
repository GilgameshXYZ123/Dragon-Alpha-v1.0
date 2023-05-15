#pragma once

#ifndef FIELD_MAX_INDEXED_H
#define FIELD_MAX_INDEXED_H 

//(1) field_num % A.width ==0
//(2) field_num -> field_stride = field_num / width * stride
//(3) A.length % feature_num == 0
//(4) A.lengthv % fieled_stride == 0
//(5) row_num = A.lengthv / field_stride
//(6) [y, x] -> [row_num, field_stride]
//(7) [N, M] = [row_num, field_stride], A.lengthv = N*M
#ifndef FIELD_MAX_INDEXED_CALL
#define FIELD_MAX_INDEXED_CALL

//LTX >=2
#define field_max_indexed4(stream, LBY, LBX, LTY, LTX, A, N, M, V, Index, width, stride) \
	field_max_indexed_kernel_4<LBY, LBX>\
		<<< dim3(M>>LBX>>LTX, N>>LBY>>LTY), dim3(1<<LBX, 1<<LBY), 0, stream >>>\
			(A, N, M, V, Index, width, stride)

//[16, 4]
#define field_max_indexed4_small(stream, A, N, M, V, Index, width, stride) \
	field_max_indexed_kernel_4<3, 3>\
		<<< dim3((M + 31)>>5, (N + 63)>>6), dim3(8, 8), 0, stream>>>\
			(A, N, M, V, Index, width, stride)

//=====================================================================================================
//LTX >=2
#define field_max_indexed4_next(stream, LBY, LBX, LTY, LTX, A, VIndex, N, M, V, Index, width, stride) \
	field_max_indexed_kernel_4_next<LBY, LBX>\
		<<< dim3(M>>LBX>>LTX, N>>LBY>>LTY), dim3(1<<LBX, 1<<LBY), 0, stream >>>\
			(A, VIndex, N, M, V, Index, width, stride)

//[16, 4]
#define field_max_indexed4_small_next(stream, A, VIndex, N, M, V, Index, width, stride) \
	field_max_indexed_kernel_4_next<3, 3>\
		<<< dim3((M + 31)>>5, (N + 63)>>6), dim3(8, 8), 0, stream>>>\
			(A, VIndex, N, M, V, Index, width, stride)

#endif


#ifndef FIELD_MAX_INDEXED_KERNEL_4
#define FIELD_MAX_INDEXED_KERNEL_4

template<int LBY, int LBX>
__global__ void field_max_indexed_kernel_4(
	const float* __restrict__ A,
	int N, int M,
	float* __restrict__ V, 
	int* __restrict__ Index, 
	int width, int stride)
{
	int by = blockIdx.y, bx = blockIdx.x;
	int ty = threadIdx.y, tx = threadIdx.x;

	__shared__ float2 As[2 << LBX << LBY];//[BX, 2*BY]
	__shared__ int2   Ps[2 << LBX << LBY];//[BX, 2*BY]

	const int offsetY = (by << LBY) + ty, stepY = (gridDim.y << LBY);
	const int offsetX = (bx << LBX) + tx, stepX = (gridDim.x << LBX);
	const int As_xy = ((tx << LBY) + ty) << 1;//(tx, ty*2)

	//parallel field num = 4
	int x4 = offsetX << 2, stepX4 = stepX << 2, M4 = (M >> 2) << 2;
	for (; x4 < M4; x4 += stepX4)
	{
		//thread reduce: 4 local result
		float4 v = MIN_FLOAT4;
		int4 pos = make_int4(0, 0, 0, 0);
		for (int y = offsetY; y < N; y += stepY) 
		{
			float4 a = *(float4*)(&A[y*M + x4]);

			pos.x = pos.x + (v.x < a.x)*(y - pos.x); v.x = fmaxf(v.x, a.x);
			pos.y = pos.y + (v.y < a.y)*(y - pos.y); v.y = fmaxf(v.y, a.y);
			pos.z = pos.z + (v.z < a.z)*(y - pos.z); v.z = fmaxf(v.z, a.z);
			pos.w = pos.w + (v.w < a.w)*(y - pos.w); v.w = fmaxf(v.w, a.w);
		}
		*(float4*)(As + As_xy) = v;
		*(int4*)(Ps + As_xy) = pos;
		__syncthreads();

		int As_index;//block reduce: get 4 global result
		if (LBY >= 6) {
			As_index = (((tx << LBY) + (ty & 31)) << 1) + (ty >> 5);
			if (ty < 64) {
				float2 a1 = As[As_index], a2 = As[As_index + 64];
				int2   p1 = Ps[As_index], p2 = Ps[As_index + 64];
				Ps[As_index].x = p1.x + (a1.x < a2.x) *(p2.x - p1.x); As[As_index].x = fmaxf(a1.x, a2.x);
				Ps[As_index].y = p1.y + (a1.y < a2.y) *(p2.y - p1.y); As[As_index].y = fmaxf(a1.y, a2.y);
			}
			__syncthreads();
		}
		if (LBY >= 5) {
			As_index = (((tx << LBY) + (ty & 15)) << 1) + (ty >> 4);
			if (ty < 32) {
				float2 a1 = As[As_index], a2 = As[As_index + 32];
				int2   p1 = Ps[As_index], p2 = Ps[As_index + 32];
				Ps[As_index].x = p1.x + (a1.x < a2.x) *(p2.x - p1.x); As[As_index].x = fmaxf(a1.x, a2.x);
				Ps[As_index].y = p1.y + (a1.y < a2.y) *(p2.y - p1.y); As[As_index].y = fmaxf(a1.y, a2.y);
			}
			__syncthreads();
		}
		if (LBY >= 4) {
			As_index = (((tx << LBY) + (ty & 7)) << 1) + (ty >> 3);
			if (ty < 16) { 
				float2 a1 = As[As_index], a2 = As[As_index + 16];
				int2   p1 = Ps[As_index], p2 = Ps[As_index + 16];
				Ps[As_index].x = p1.x + (a1.x < a2.x) *(p2.x - p1.x); As[As_index].x = fmaxf(a1.x, a2.x);
				Ps[As_index].y = p1.y + (a1.y < a2.y) *(p2.y - p1.y); As[As_index].y = fmaxf(a1.y, a2.y);
			}
			__syncthreads();
		}

		As_index = (((tx << LBY) + (ty & 3)) << 1) + (ty >> 2);
		if (ty < 8) { 
			float2 a1 = As[As_index], a2 = As[As_index + 8];
			int2   p1 = Ps[As_index], p2 = Ps[As_index + 8];
			Ps[As_index].x = p1.x + (a1.x < a2.x) *(p2.x - p1.x); As[As_index].x = fmaxf(a1.x, a2.x);
			Ps[As_index].y = p1.y + (a1.y < a2.y) *(p2.y - p1.y); As[As_index].y = fmaxf(a1.y, a2.y);
		}
		__syncthreads();

		As_index = (((tx << LBY) + (ty & 1)) << 1) + (ty >> 1);
		if (ty < 4) { 
			float2 a1 = As[As_index], a2 = As[As_index + 4];
			int2   p1 = Ps[As_index], p2 = Ps[As_index + 4];
			Ps[As_index].x = p1.x + (a1.x < a2.x) *(p2.x - p1.x); As[As_index].x = fmaxf(a1.x, a2.x);
			Ps[As_index].y = p1.y + (a1.y < a2.y) *(p2.y - p1.y); As[As_index].y = fmaxf(a1.y, a2.y);
		}
		__syncthreads();

		As_index = (((tx << LBY) + (ty & 0)) << 1) + ty;
		if (ty < 2) {
			float2 a1 = As[As_index], a2 = As[As_index + 2];
			int2   p1 = Ps[As_index], p2 = Ps[As_index + 2];
			Ps[As_index].x = p1.x + (a1.x < a2.x) *(p2.x - p1.x); As[As_index].x = fmaxf(a1.x, a2.x);
			Ps[As_index].y = p1.y + (a1.y < a2.y) *(p2.y - p1.y); As[As_index].y = fmaxf(a1.y, a2.y);
		}
		__syncthreads();

		if (ty < 2) {
			float2 result = As[As_index];
			int2 position = Ps[As_index];

			int xindex2 = x4 + (ty << 1);
			within_width2(result  , xindex2, stride, width);
			within_width2(position, xindex2, stride, width);
			*(float2*)(&get(V, by, x4 + (ty << 1), M)) = result;
			*(int2*)(&get(Index, by, x4 + (ty << 1), M)) = position;
		}
		__syncthreads();
	}
}

#endif


#ifndef FIELD_MAX_INDEXED_KERNEL_4_NEXT
#define FIELD_MAX_INDEXED_KERNEL_4_NEXT

template<int LBY, int LBX>
__global__ void field_max_indexed_kernel_4_next(
	const float* __restrict__ A,
	const int* __restrict VIndex,
	int N, int M,
	float* __restrict__ V,
	int* __restrict__ Index,
	int width, int stride)
{
	int by = blockIdx.y, bx = blockIdx.x;
	int ty = threadIdx.y, tx = threadIdx.x;

	__shared__ float2 As[2 << LBX << LBY];//[BX, 2*BY]
	__shared__ int2   Ps[2 << LBX << LBY];//[BX, 2*BY]

	const int offsetY = (by << LBY) + ty, stepY = (gridDim.y << LBY);
	const int offsetX = (bx << LBX) + tx, stepX = (gridDim.x << LBX);
	const int As_xy = ((tx << LBY) + ty) << 1;//(tx, ty*2)

	//parallel field num = 4
	int x4 = offsetX << 2, stepX4 = stepX << 2, M4 = (M >> 2) << 2;
	for (; x4 < M4; x4 += stepX4)
	{
		//thread reduce: 4 local result
		float4 v = MIN_FLOAT4;
		int4 pos = make_int4(0, 0, 0, 0);
		for (int y = offsetY; y < N; y += stepY)
		{
			int y_x4 = y * M + x4;
			float4 a = *(float4*)(A + y_x4);
			int4 npos = *(int4*)(VIndex + y_x4);

			pos.x = pos.x + (v.x < a.x)*(npos.x - pos.x); v.x = fmaxf(v.x, a.x);
			pos.y = pos.y + (v.y < a.y)*(npos.y - pos.y); v.y = fmaxf(v.y, a.y);
			pos.z = pos.z + (v.z < a.z)*(npos.z - pos.z); v.z = fmaxf(v.z, a.z);
			pos.w = pos.w + (v.w < a.w)*(npos.w - pos.w); v.w = fmaxf(v.w, a.w);
		}
		*(float4*)(As + As_xy) = v;
		*(int4*)(Ps + As_xy) = pos;
		__syncthreads();

		int As_index;//block reduce: get 4 global result
		if (LBY >= 6) {
			As_index = (((tx << LBY) + (ty & 31)) << 1) + (ty >> 5);
			if (ty < 64) {
				float2 a1 = As[As_index], a2 = As[As_index + 64];
				int2   p1 = Ps[As_index], p2 = Ps[As_index + 64];
				Ps[As_index].x = p1.x + (a1.x < a2.x) *(p2.x - p1.x); As[As_index].x = fmaxf(a1.x, a2.x);
				Ps[As_index].y = p1.y + (a1.y < a2.y) *(p2.y - p1.y); As[As_index].y = fmaxf(a1.y, a2.y);
			}
			__syncthreads();
		}
		if (LBY >= 5) {
			As_index = (((tx << LBY) + (ty & 15)) << 1) + (ty >> 4);
			if (ty < 32) {
				float2 a1 = As[As_index], a2 = As[As_index + 32];
				int2   p1 = Ps[As_index], p2 = Ps[As_index + 32];
				Ps[As_index].x = p1.x + (a1.x < a2.x) *(p2.x - p1.x); As[As_index].x = fmaxf(a1.x, a2.x);
				Ps[As_index].y = p1.y + (a1.y < a2.y) *(p2.y - p1.y); As[As_index].y = fmaxf(a1.y, a2.y);
			}
			__syncthreads();
		}
		if (LBY >= 4) {
			As_index = (((tx << LBY) + (ty & 7)) << 1) + (ty >> 3);
			if (ty < 16) {
				float2 a1 = As[As_index], a2 = As[As_index + 16];
				int2   p1 = Ps[As_index], p2 = Ps[As_index + 16];
				Ps[As_index].x = p1.x + (a1.x < a2.x) *(p2.x - p1.x); As[As_index].x = fmaxf(a1.x, a2.x);
				Ps[As_index].y = p1.y + (a1.y < a2.y) *(p2.y - p1.y); As[As_index].y = fmaxf(a1.y, a2.y);
			}
			__syncthreads();
		}

		As_index = (((tx << LBY) + (ty & 3)) << 1) + (ty >> 2);
		if (ty < 8) {
			float2 a1 = As[As_index], a2 = As[As_index + 8];
			int2   p1 = Ps[As_index], p2 = Ps[As_index + 8];
			Ps[As_index].x = p1.x + (a1.x < a2.x) *(p2.x - p1.x); As[As_index].x = fmaxf(a1.x, a2.x);
			Ps[As_index].y = p1.y + (a1.y < a2.y) *(p2.y - p1.y); As[As_index].y = fmaxf(a1.y, a2.y);
		}
		__syncthreads();

		As_index = (((tx << LBY) + (ty & 1)) << 1) + (ty >> 1);
		if (ty < 4) {
			float2 a1 = As[As_index], a2 = As[As_index + 4];
			int2   p1 = Ps[As_index], p2 = Ps[As_index + 4];
			Ps[As_index].x = p1.x + (a1.x < a2.x) *(p2.x - p1.x); As[As_index].x = fmaxf(a1.x, a2.x);
			Ps[As_index].y = p1.y + (a1.y < a2.y) *(p2.y - p1.y); As[As_index].y = fmaxf(a1.y, a2.y);
		}
		__syncthreads();

		As_index = (((tx << LBY) + (ty & 0)) << 1) + ty;
		if (ty < 2) {
			float2 a1 = As[As_index], a2 = As[As_index + 2];
			int2   p1 = Ps[As_index], p2 = Ps[As_index + 2];
			Ps[As_index].x = p1.x + (a1.x < a2.x) *(p2.x - p1.x); As[As_index].x = fmaxf(a1.x, a2.x);
			Ps[As_index].y = p1.y + (a1.y < a2.y) *(p2.y - p1.y); As[As_index].y = fmaxf(a1.y, a2.y);
		}
		__syncthreads();

		if (ty < 2) {
			float2 result = As[As_index];
			int2 position = Ps[As_index];

			int xindex2 = x4 + (ty << 1);
			within_width2(result  , xindex2, stride, width);
			within_width2(position, xindex2, stride, width);
			*(float2*)(&get(V, by, x4 + (ty << 1), M)) = result;
			*(int2*)(&get(Index, by, x4 + (ty << 1), M)) = position;
		}
		__syncthreads();
	}
}

#endif


#ifndef FIELD_MAX_INDEXED_STAGE
#define FIELD_MAX_INDEXED_STAGE

//M % 4 == 0, M >= 4
void __field_max_indexed_stage(cudaStream_t stream,
	const float* A, int N, int M,
	float*  V, int* Index,
	int width, int stride)
{
	if (M > 15) {
		if (N > 63) { field_max_indexed4(stream, 3, 2, 3, 2, A, N, M, V, Index, width, stride); return; }//[64, 16]
		if (N > 31) { field_max_indexed4(stream, 3, 2, 2, 2, A, N, M, V, Index, width, stride); return; }//[32, 16]
		if (N > 15) { field_max_indexed4(stream, 3, 2, 1, 2, A, N, M, V, Index, width, stride); return; }//[16, 16]
	}
	if (M > 7) {
		if (N > 127) { field_max_indexed4(stream, 4, 1, 3, 2, A, N, M, V, Index, width, stride); return; }//[ 64, 8]
		if (N >  63) { field_max_indexed4(stream, 4, 1, 2, 2, A, N, M, V, Index, width, stride); return; }//[ 32, 8]
		if (N >  31) { field_max_indexed4(stream, 4, 1, 1, 2, A, N, M, V, Index, width, stride); return; }//[ 16, 8]
	}
	field_max_indexed4_small(stream, A, N, M, V, Index, width, stride);
}

#endif 


#ifndef FIELD_MAX_INDEXED_STAGE_NEXT
#define FIELD_MAX_INDEXED_STAGE_NEXT

//M % 4 == 0, M >= 4
void __field_max_indexed_stage_next(cudaStream_t stream,
	const float* A, const int* VIndex, 
	int N, int M,
	float*  V, int* Index,
	int width, int stride)
{
	if (M > 15) {
		if (N > 63) { field_max_indexed4_next(stream, 3, 2, 3, 2, A, VIndex, N, M, V, Index, width, stride); return; }//[64, 16]
		if (N > 31) { field_max_indexed4_next(stream, 3, 2, 2, 2, A, VIndex, N, M, V, Index, width, stride); return; }//[32, 16]
		if (N > 15) { field_max_indexed4_next(stream, 3, 2, 1, 2, A, VIndex, N, M, V, Index, width, stride); return; }//[16, 16]
	}
	if (M > 7) {
		if (N > 127) { field_max_indexed4_next(stream, 4, 1, 3, 2, A, VIndex, N, M, V, Index, width, stride); return; }//[ 64, 8]
		if (N >  63) { field_max_indexed4_next(stream, 4, 1, 2, 2, A, VIndex, N, M, V, Index, width, stride); return; }//[ 32, 8]
		if (N >  31) { field_max_indexed4_next(stream, 4, 1, 1, 2, A, VIndex, N, M, V, Index, width, stride); return; }//[ 16, 8]
	}
	field_max_indexed4_small_next(stream, A, VIndex, N, M, V, Index, width, stride);
}

#endif 


#ifndef FIELD_MAX_INDEXED
#define FIELD_MAX_INDEXED

int __field_max_indexed(cudaStream_t stream,
	const float* A, int N, int M,
	float* V, int* VIndex,//VIndex is the buffer of Index
	float* Y, int* Index,
	int width, int stride,
	int partNum)
{
	int nextN = field_nextN(N, M);
	if (nextN <= partNum) {//V.address = NULL, only 1 stage, write result to Y
		__field_max_indexed_stage(stream, A, N, M, Y, Index, width, stride);
		return nextN;
	}

	//V.address != NULL, at least 2 stages
	__field_max_indexed_stage(stream, A, N, M, V, VIndex, width, stride);

	N = nextN, nextN = field_nextN(N, M);
	while (nextN > partNum)//end: nextN <= partNum
	{
		__field_max_indexed_stage_next(stream, V, VIndex, N, M, V, VIndex, width, stride);
		N = nextN, nextN = field_nextN(nextN, M);
	}
	__field_max_indexed_stage_next(stream, V, VIndex, N, M, Y, Index, width, stride); //the last stage
	return nextN;

}

#endif

#endif