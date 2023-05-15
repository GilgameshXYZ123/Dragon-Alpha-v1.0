#pragma once

#ifndef STRAIGHT_MAX_INDEX_H
#define STRAIGHT_MAX_INDEX_H

//index % 4 == 0
//lengthv % 4(stride) == 0
#ifndef STRAIGHT_MAX_INDEXED_CALL
#define STRAIGHT_MAX_INDEXED_CALL

#define straight_max_indexed4(stream, LB, LT, X, lengthv, V, Index, width, stride) \
	straight_max_indexed_kernel_4<LB>\
		<<< (lengthv >> LB >> LT), (1<<LB), 0, stream >>>\
			(X, lengthv, V, Index, width, stride)

//2 << 7 = 128, for lengthv <= 128, 128 >> 2 = 32
#define straight_max_indexed4_small(stream, X, lengthv, V, Index, width, stride) \
	straight_max_indexed_kernel_4<5>\
		<<< 1, 32, 0, stream>>> \
			(X, lengthv, V, Index, width, stride)

//==================================================================
#define straight_max_indexed4_next(stream, LB, LT, X, lengthv, V, Index) \
	straight_max_indexed_kernel_4_next<LB>\
		<<< (lengthv >> LB >> LT), (1<<LB), 0, stream >>>\
			(X, lengthv, V, Index)

//2 << 7 = 128, for lengthv <= 128, 128 >> 2 = 32
#define straight_max_indexed4_small_next(stream, X, lengthv, V, Index) \
	straight_max_indexed_kernel_4_next<5>\
		<<< 1, 32, 0, stream>>> \
			(X, lengthv, V, Index)

#endif


#ifndef STRAIGHT_MAX_INDEXED_KERNEL_4
#define STRAIGHT_MAX_INDEXED_KERNEL_4

template<int LB>
__global__ void straight_max_indexed_kernel_4(
	const float* __restrict__ X, int lengthv,
	float* __restrict__ V,
	int* __restrict__ Index,
	int width, int stride)
{
	__shared__ float As[1 << LB];//solve the max value
	__shared__ int   Ps[1 << LB];//solve the Index of max value

	int bx = blockIdx.x, tx = threadIdx.x;
	int index = (bx << LB) + tx;
	int step = (gridDim.x << LB), step4 = step << 2;

	float v = FLOAT_MIN; int pos = 0;
	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 a = *(float4*)(intptr_t)(X + index4);
		EXCEED_width4_TO_MIN(a, index4, stride, width);

		int4 npos = INDEX4(index4);
		pos = pos + (v < a.x)*(npos.x - pos); v = fmaxf(v, a.x);
		pos = pos + (v < a.y)*(npos.y - pos); v = fmaxf(v, a.y);
		pos = pos + (v < a.z)*(npos.z - pos); v = fmaxf(v, a.z);
		pos = pos + (v < a.w)*(npos.w - pos); v = fmaxf(v, a.w);
	}
	As[tx] = v;
	Ps[tx] = pos;
	__syncthreads();

	if (LB >= 7) {
		if (tx < 64) {
			float a1 = As[tx], a2 = As[tx + 64];
			int   p1 = Ps[tx], p2 = Ps[tx + 64];
			Ps[tx] = p1 + (a1 < a2)*(p2 - p1);
			As[tx] = fmaxf(a1, a2);
		}
		__syncthreads();
	}
	if (LB >= 6) {
		if (tx < 32) {
			float a1 = As[tx], a2 = As[tx + 32];
			int   p1 = Ps[tx], p2 = Ps[tx + 32];
			Ps[tx] = p1 + (a1 < a2)*(p2 - p1);
			As[tx] = fmaxf(a1, a2);
		}
		__syncthreads();
	}
	if (LB >= 5) {
		if (tx < 16) {
			float a1 = As[tx], a2 = As[tx + 16];
			int   p1 = Ps[tx], p2 = Ps[tx + 16];
			Ps[tx] = p1 + (a1 < a2)*(p2 - p1);
			As[tx] = fmaxf(a1, a2);
		}
		__syncthreads();
	}
	if (tx < 8) warp_max_indexed_8(As, Ps, tx);//LBX >= 4
	if (tx == 0) { V[bx] = As[0]; Index[bx] = Ps[0]; }
}

#endif


#ifndef STRAIGHT_MAX_INDEXED_KERNEL_4_NEXT
#define STRAIGHT_MAX_INDEXED_KERNEL_4_NEXT

template<int LB>
__global__ void straight_max_indexed_kernel_4_next(
	const float* __restrict__ X, int lengthv,
	float* __restrict__ V,
	int* __restrict__ Index)
{
	__shared__ float As[1 << LB];//solve the max value
	__shared__ int   Ps[1 << LB];//solve the Index of max value

	int bx = blockIdx.x, tx = threadIdx.x;
	int index = (bx << LB) + tx;
	int step = (gridDim.x << LB), step4 = step << 2;

	int stride = ((lengthv + 3) >> 2) << 2;
	float v = FLOAT_MIN; int pos = 0;
	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 a = *(float4*)(X + index4);
		int4 npos = *(int4*)(Index + index4);
		EXCEED_width4_TO_MIN(a, index4, stride, lengthv);

		pos = pos + (v < a.x)*(npos.x - pos); v = fmaxf(v, a.x);
		pos = pos + (v < a.y)*(npos.y - pos); v = fmaxf(v, a.y);
		pos = pos + (v < a.z)*(npos.z - pos); v = fmaxf(v, a.z);
		pos = pos + (v < a.w)*(npos.w - pos); v = fmaxf(v, a.w);
	}
	As[tx] = v; 
	Ps[tx] = pos;
	__syncthreads();

	if (LB >= 7) {
		if (tx < 64) {
			float a1 = As[tx], a2 = As[tx + 64];
			int   p1 = Ps[tx], p2 = Ps[tx + 64];
			Ps[tx] = p1 + (a1 < a2)*(p2 - p1);
			As[tx] = fmaxf(a1, a2);
		}
		__syncthreads();
	}
	if (LB >= 6) {
		if (tx < 32) {
			float a1 = As[tx], a2 = As[tx + 32];
			int   p1 = Ps[tx], p2 = Ps[tx + 32];
			Ps[tx] = p1 + (a1 < a2)*(p2 - p1);
			As[tx] = fmaxf(a1, a2);
		}
		__syncthreads();
	}
	if (LB >= 5) {
		if (tx < 16) {
			float a1 = As[tx], a2 = As[tx + 16];
			int   p1 = Ps[tx], p2 = Ps[tx + 16];
			Ps[tx] = p1 + (a1 < a2)*(p2 - p1);
			As[tx] = fmaxf(a1, a2);
		}
		__syncthreads();
	}
	if (tx < 8) warp_max_indexed_8(As, Ps, tx);//LBX >= 4
	if (tx == 0) { V[bx] = As[0]; Index[bx] = Ps[0]; }
}

#endif


#ifndef STRAIGHT_MAX_INDEXED_STAGE
#define STRAIGHT_MAX_INDEXED_STAGE

void __straight_max_indexed_stage(cudaStream_t stream,
	const float *X, int lengthv,
	float *V, int *Index,
	int width, int stride)
{
	if (lengthv <   128) { straight_max_indexed4_small(stream, X, lengthv, V, Index, width, stride); return; }
	if (lengthv >= 8192) { straight_max_indexed4(stream, 6, 7, X, lengthv, V, Index, width, stride); return; }
	if (lengthv >= 4096) { straight_max_indexed4(stream, 6, 6, X, lengthv, V, Index, width, stride); return; }
	if (lengthv >= 2048) { straight_max_indexed4(stream, 6, 5, X, lengthv, V, Index, width, stride); return; }
	if (lengthv >= 1024) { straight_max_indexed4(stream, 6, 4, X, lengthv, V, Index, width, stride); return; }
	if (lengthv >=  512) { straight_max_indexed4(stream, 6, 3, X, lengthv, V, Index, width, stride); return; }
	if (lengthv >=  256) { straight_max_indexed4(stream, 6, 2, X, lengthv, V, Index, width, stride); return; }
	straight_max_indexed4(stream, 5, 2, X, lengthv, V, Index, width, stride);//lengthv >= 128
}

#endif


#ifndef STRAIGHT_MAX_INDEXED_STAGE_NEXT
#define STRAIGHT_MAX_INDEXED_STAGE_NEXT

void __straight_max_indexed_stage_next(cudaStream_t stream,
	const float *X, int lengthv,
	float *V, int *Index)
{
	if (lengthv <   128) { straight_max_indexed4_small_next(stream, X, lengthv, V, Index); return; }
	if (lengthv >= 8192) { straight_max_indexed4_next(stream, 6, 7, X, lengthv, V, Index); return; }
	if (lengthv >= 4096) { straight_max_indexed4_next(stream, 6, 6, X, lengthv, V, Index); return; }
	if (lengthv >= 2048) { straight_max_indexed4_next(stream, 6, 5, X, lengthv, V, Index); return; }
	if (lengthv >= 1024) { straight_max_indexed4_next(stream, 6, 4, X, lengthv, V, Index); return; }
	if (lengthv >=  512) { straight_max_indexed4_next(stream, 6, 3, X, lengthv, V, Index); return; }
	if (lengthv >=  256) { straight_max_indexed4_next(stream, 6, 2, X, lengthv, V, Index); return; }
	straight_max_indexed4_next(stream, 5, 2, X, lengthv, V, Index);//lengthv >= 128
}

#endif


#ifndef STRAIGHT_MAX_INDEXED
#define STRAIGHT_MAX_INDEXED

int __straight_max_indexed(cudaStream_t stream,
	const float *X, int lengthv,
	float *V, int *Index,
	int width, int stride,
	int partNum)
{
	__straight_max_indexed_stage(stream, X, lengthv, V, Index, width, stride);
	lengthv = straight_nextLengthV(lengthv);

	while (lengthv > partNum) {
		__straight_max_indexed_stage_next(stream, V, lengthv, V, Index);
		lengthv = straight_nextLengthV(lengthv);
	}
	return lengthv;
}

#endif

#endif
