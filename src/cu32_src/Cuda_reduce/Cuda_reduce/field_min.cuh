#pragma once

#ifndef FIELD_MIN_H
#define FIELD_MIN_H

//(1) field_num % A.width ==0
//(2) field_num -> field_stride = field_num / width * stride
//(3) A.length % feature_num == 0
//(4) A.lengthv % fieled_stride == 0
//(5) row_num = A.lengthv / field_stride
//(6) [y, x] -> [row_num, field_stride]
//(7) [N, M] = [row_num, field_stride], A.lengthv = N*M

#ifndef FIELD_MIN_CALL
#define FIELD_MIN_CALL

//LTX >=2
#define field_min4(stream, LBY, LBX, LTY, LTX, A, N, M, V, width, stride) \
	field_min_kernel_4<LBY, LBX>\
		<<< dim3(M>>LBX>>LTX, N>>LBY>>LTY), dim3(1<<LBX, 1<<LBY), 0, stream >>>\
			(A, N, M, V, width, stride)

//[16, 4]
#define field_min4_small(stream, A, N, M, V, width, stride) \
	field_min_kernel_4<3, 3>\
		<<< dim3((M + 31)>>5, (N + 63)>>6), dim3(8, 8), 0, stream>>>\
			(A, N, M, V, width, stride)

#endif


#ifndef FIELD_MIN_KERNEL_4
#define FIELD_MIN_KERNEL_4

template<int LBY, int LBX>
__global__ void field_min_kernel_4(
	const float* __restrict__ A,
	int N, int M,
	float* __restrict__ V,
	int width, int stride)
{
	int by = blockIdx.y, bx = blockIdx.x;
	int ty = threadIdx.y, tx = threadIdx.x;

	__shared__ float2 As[2 << LBX << LBY];//[BX, 2*BY]

	const int offsetY = (by << LBY) + ty, stepY = (gridDim.y << LBY);
	const int offsetX = (bx << LBX) + tx, stepX = (gridDim.x << LBX);
	const int As_xy = ((tx << LBY) + ty) << 1;//(tx, ty*2)

	//parallel field num = 4
	int x4 = offsetX << 2, stepX4 = stepX << 2, M4 = (M >> 2) << 2;
	for (; x4 < M4; x4 += stepX4)
	{
		float4 v = MAX_FLOAT4;//thread reduce: 4 local result
		for (int y = offsetY; y < N; y += stepY) {
			float4 a = *(float4*)(&A[y*M + x4]);
			simdMin4(v, v, a);
		}
		*(float4*)(As + As_xy) = v;
		__syncthreads();

		int As_index;//block reduce: get 4 global result
		if (LBY >= 6) {
			As_index = (((tx << LBY) + (ty & 31)) << 1) + (ty >> 5);
			if (ty < 64) { simdMin2(As[As_index], As[As_index], As[As_index + 64]); }
			__syncthreads();
		}
		if (LBY >= 5) {
			As_index = (((tx << LBY) + (ty & 15)) << 1) + (ty >> 4);
			if (ty < 32) { simdMin2(As[As_index], As[As_index], As[As_index + 32]); }
			__syncthreads();
		}
		if (LBY >= 4) {
			As_index = (((tx << LBY) + (ty & 7)) << 1) + (ty >> 3);
			if (ty < 16) { simdMin2(As[As_index], As[As_index], As[As_index + 16]); }
			__syncthreads();
		}
		
		As_index = (((tx << LBY) + (ty & 3)) << 1) + (ty >> 2);
		if (ty < 8) { simdMin2(As[As_index], As[As_index], As[As_index + 8]); }
		__syncthreads();

		As_index = (((tx << LBY) + (ty & 1)) << 1) + (ty >> 1);
		if (ty < 4) { simdMin2(As[As_index], As[As_index], As[As_index + 4]); }
		__syncthreads();

		As_index = (((tx << LBY) + (ty & 0)) << 1) + ty;
		if (ty < 2) { simdMin2(As[As_index], As[As_index], As[As_index + 2]); }
		__syncthreads();

		if (ty < 2) { 
			float2 result = As[As_index];
			int xindex2 = x4 + (ty << 1);
			within_width2(result, xindex2, stride, width);
			*(float2*)(&get(V, by, x4 + (ty << 1), M)) = result;
		}
		__syncthreads();
	}
}

#endif


#ifndef FIELD_MIN_STAGE
#define FIELD_MIN_STAGE

//M % 4 == 0, M >= 4
void __field_min_stage(cudaStream_t stream,
	const float* A, int N, int M,
	float*  V, int width, int stride) 
{
	if (M > 15) {
		if (N > 63) { field_min4(stream, 3, 2, 3, 2, A, N, M, V, width, stride); return; }//[64, 16]
		if (N > 31) { field_min4(stream, 3, 2, 2, 2, A, N, M, V, width, stride); return; }//[32, 16]
		if (N > 15) { field_min4(stream, 3, 2, 1, 2, A, N, M, V, width, stride); return; }//[16, 16]
	}
	if (M > 7) {
		if (N > 127) { field_min4(stream, 4, 1, 3, 2, A, N, M, V, width, stride); return; }//[ 64, 8]
		if (N >  63) { field_min4(stream, 4, 1, 2, 2, A, N, M, V, width, stride); return; }//[ 32, 8]
		if (N >  31) { field_min4(stream, 4, 1, 1, 2, A, N, M, V, width, stride); return; }//[ 16, 8]
	}
	if (N > 31) { field_min4(stream, 5, 0, 0, 2, A, N, M, V, width, stride); return; }//[ 32, 4]
	field_min4_small(stream, A, N, M, V, width, stride);
}

#endif 


#ifndef FIELD_MIN
#define FIELD_MIN

int __field_min(cudaStream_t stream,
	const float* A, int N, int M,
	float*  V, float *Y,
	int width, int stride,
	int partNum)
{
	int nextN = field_nextN(N, M);
	if (nextN <= partNum) {//V.address = NULL, only 1 stage, write result to Y
		__field_min_stage(stream, A, N, M, Y, width, stride);
		return nextN;
	}

	//V.address != NULL, at least 2 stages
	__field_min_stage(stream, A, N, M, V, width, stride);

	N = nextN, nextN = field_nextN(N, M);
	while (nextN > partNum)//end: nextN <= partNum
	{
		__field_min_stage(stream, V, N, M, V, width, stride);
		N = nextN, nextN = field_nextN(nextN, M);
	}
	__field_min_stage(stream, V, N, M, Y, width, stride); //the last stage
	return nextN;
}
#endif

#endif