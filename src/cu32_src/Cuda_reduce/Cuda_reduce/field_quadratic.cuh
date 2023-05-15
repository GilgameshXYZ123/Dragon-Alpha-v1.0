#pragma once

#ifndef FIELD_QUADRATIC_H
#define FIELD_QUADRATIC_H

//(1) field_num % A.width ==0
//(2) field_num -> field_stride = field_num / width * stride
//(3) A.length % feature_num == 0
//(4) A.lengthv % fieled_stride == 0
//(5) row_num = A.lengthv / field_stride
//(6) [y, x] -> [row_num, field_stride]
//(7) [N, M] = [row_num, field_stride], A.lengthv = N*M
#ifndef FIELD_QUADRATIC_CALL
#define FIELD_QUADRATIC_CALL

//LTX >=2
#define field_quadratic4(stream, LBY, LBX, LTY, LTX, A, alpha, beta, gamma, N, M, V, width, stride) \
	field_quadratic_kernel_4<LBY, LBX>\
		<<< dim3(M>>LBX>>LTX, N>>LBY>>LTY), dim3(1<<LBX, 1<<LBY), 0, stream >>>\
			(A, alpha, beta, gamma, N, M, V, width, stride)

//[16, 4]
#define field_quadratic4_small(stream, A, alpha, beta, gamma, N, M, V, width, stride) \
	field_quadratic_kernel_4<3, 3>\
		<<< dim3((M + 31)>>5, (N + 63)>>6), dim3(8, 8), 0, stream>>>\
			(A, alpha, beta, gamma, N, M, V, width, stride)

#endif


#ifndef FIELD_QUADRATIC_KERNEL_4
#define FIELD_QUADRATIC_KERNEL_4

//<3, 2, 3, 4>: Time = 0.047     mesc, Speed = 85.7089GB/s
//<3, 2, 2, 4>: Time = 0.0506667 mesc, Speed = 81.9156GB/s
//<3, 2, 1, 4>: Time = 0.075     mesc, Speed = 58.5937GB/s
//<3, 2, 2, 3>: Time = 0.049     mesc, Speed = 84.7019GB/s
//<3, 2, 4, 2>: Time = 0.0463333 mesc, Speed = 85.6249GB/s
template<int LBY, int LBX>
__global__ void field_quadratic_kernel_4(
	const float* __restrict__ A,
	float alpha, float beta, float gamma,
	int N, int M,
	float* __restrict__ V,
	int width, int stride)
{
	int by = blockIdx.y, bx = blockIdx.x;
	int ty = threadIdx.y, tx = threadIdx.x;

	__shared__ float2 As[1 << LBX][(2 << LBY) + 2];

	//parallel field num = 4
	const int offsetY = (by << LBY) + ty, stepY = (gridDim.y << LBY);
	const int offsetX = (bx << LBX) + tx, stepX = (gridDim.x << LBX);
	int x4 = offsetX << 2, stepX4 = stepX << 2, M4 = (M >> 2) << 2;

	for (; x4 < M4; x4 += stepX4)
	{
		float4 c = make_float4(0, 0, 0, 0);//solve the errors
		float4 v = make_float4(0, 0, 0, 0);//thread reduce: 4 local result

		for (int y = offsetY; y < N; y += stepY)
		{
			float4 a = *(float4*)(A + y * M + x4);//A[y, x4]
			simdQuadratic4(a, alpha, a, beta, gamma);

			Kahan_simdAdd4(v, a, c);//v = v + a
		}
		*(float4*)(&As[tx][ty << 1]) = v;
		__syncthreads();

		if (LBY >= 6) {//block reduce: 4 global result
			if (ty < 64) {//128 -> 64
				int yIdx = ((ty & 31) << 1) + (ty >> 5);
				float2 v1 = As[tx][yIdx], v2 = As[tx][yIdx + 64];
				simdAdd2(v1, v1, v2); As[tx][yIdx] = v1;
			}
			__syncthreads();
		}
		if (LBY >= 5) {
			if (ty < 32) {//64 -> 32
				int yIdx = ((ty & 15) << 1) + (ty >> 4);
				float2 v1 = As[tx][yIdx], v2 = As[tx][yIdx + 32];
				simdAdd2(v1, v1, v2); As[tx][yIdx] = v1;
			}
			__syncthreads();
		}
		if (LBY >= 4) {
			if (ty < 16) {//32 -> 16
				int yIdx = ((ty & 7) << 1) + (ty >> 3);
				float2 v1 = As[tx][yIdx], v2 = As[tx][yIdx + 16];
				simdAdd2(v1, v1, v2); As[tx][yIdx] = v1;
			}
			__syncthreads();
		}

		if (ty < 8) {//16 -> 8
			int yIdx = ((ty & 3) << 1) + (ty >> 2);
			float2 v1 = As[tx][yIdx], v2 = As[tx][yIdx + 8];
			simdAdd2(v1, v1, v2); As[tx][yIdx] = v1;
		}
		__syncthreads();

		if (ty < 4) {//8 -> 4
			int yIdx = ((ty & 1) << 1) + (ty >> 1);
			float2 v1 = As[tx][yIdx], v2 = As[tx][yIdx + 4];
			simdAdd2(v1, v1, v2); As[tx][yIdx] = v1;
		}
		__syncthreads();

		if (ty < 2) {//4 -> 2, save, 
			int yIdx = ((ty & 0) << 1) + ty;
			float2 v1 = As[tx][yIdx], v2 = As[tx][yIdx + 2];
			simdAdd2(v1, v1, v2);

			int xindex2 = x4 + (ty << 1);
			within_width2(v1, xindex2, stride, width);
			*(float2*)(&get(V, by, xindex2, M)) = v1;
		}
		__syncthreads();
	}
}

#endif


#ifndef FIELD_QUADRATIC_STAGE
#define FIELD_QUADRATIC_STAGE

#define __field_square_sum_stage(stream, A, N, M, V, width, stride)\
	__field_quadratic_stage(stream, A, 1.0f, 0.0f, 0.0f, N, M, V, width, stride)

//M % 4 == 0, M >= 4
void __field_quadratic_stage(cudaStream_t stream,
	const float* A,
	float alpha, float beta, float gamma,
	int N, int M,
	float* V, int width, int stride)
{
	if (M > 15) {
		if (N >  63) { field_quadratic4(stream, 3, 2, 3, 2, A, alpha, beta, gamma, N, M, V, width, stride); return; }//[64, 16]
		if (N >  31) { field_quadratic4(stream, 3, 2, 2, 2, A, alpha, beta, gamma, N, M, V, width, stride); return; }//[32, 16]
		if (N >  15) { field_quadratic4(stream, 3, 2, 1, 2, A, alpha, beta, gamma, N, M, V, width, stride); return; }//[16, 16]
	}
	if (M >  7) {
		if (N > 127) { field_quadratic4(stream, 4, 1, 3, 2, A, alpha, beta, gamma, N, M, V, width, stride); return; }//[ 64, 8]
		if (N >  63) { field_quadratic4(stream, 4, 1, 2, 2, A, alpha, beta, gamma, N, M, V, width, stride); return; }//[ 32, 8]
		if (N >  31) { field_quadratic4(stream, 4, 1, 1, 2, A, alpha, beta, gamma, N, M, V, width, stride); return; }//[ 16, 8]
	}
	field_quadratic4_small(stream, A, alpha, beta, gamma, N, M, V, width, stride);
}

#endif 


#ifndef FIELD_QUADRATIC
#define FIELD_QUADRATIC

//old fashion
//int __field_quadratic(cudaStream_t stream,
//	const float* A, float alpha, float beta, float gamma,
//	int N, int M,
//	float*  V,
//	int width, int stride,
//	int partNum)
//{
//	__field_quadratic_stage(stream, A, alpha, beta, gamma, N, M, V, width, stride);
//	int HV = field_nextN(N, M);//N <= HV
//
//	while(HV > partNum) //end : HV <= partNum
//	{
//		__field_sum_stage(stream, V, HV, M, V, width, stride);
//		HV = field_nextN(HV, M);
//	}
//	return HV;
//}

//new fashion
int __field_quadratic(cudaStream_t stream,
	const float* A, float alpha, float beta, float gamma,
	int N, int M,
	float* V, float *Y, 
	int width, int stride,
	int partNum)
{
	int nextN = field_nextN(N, M);//N <= HV
	if (nextN <= partNum) {//V.address = NULL, only 1 stage, write result to Y
		__field_quadratic_stage(stream, A, alpha, beta, gamma, N, M, Y, width, stride); 
		return nextN;
	}

	//V.address != NULL, at least 2 stages
	__field_quadratic_stage(stream, A, alpha, beta, gamma, N, M, V, width, stride);

	N = nextN, nextN = field_nextN(N, M);
	while (nextN > partNum)//end: nextN <= partNum
	{
		__field_sum_stage(stream, V, N, M, V, width, stride);
		N = nextN, nextN = field_nextN(nextN, M);
	}
	__field_sum_stage(stream, V, N, M, Y, width, stride);//the last stage
	return nextN;
}

#endif

#endif