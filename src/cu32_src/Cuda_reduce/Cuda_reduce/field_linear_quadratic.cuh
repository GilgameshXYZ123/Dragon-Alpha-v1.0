#pragma once

#ifndef FIELD_LINEAR_QUADRATIC_H
#define FIELD_LINEAR_QUADRATIC_H

//(1) field_num % A.width ==0
//(2) field_num -> field_stride = field_num / width * stride
//(3) A.length % feature_num == 0
//(4) A.lengthv % fieled_stride == 0
//(5) row_num = A.lengthv / field_stride
//(6) [y, x] -> [row_num, field_stride]
//(7) [N, M] = [row_num, field_stride], A.lengthv = N*M
//(8) For BN.deltaX: V1: holdY(), Y is not changed \\ affine = false
#ifndef FIELD_LINEAR_QUADRATIC_CALL
#define FIELD_LINEAR_QUADRATIC_CALL

//LTX >=2
#define field_linear_quadratic4(stream, LBY, LBX, LTY, LTX, A, alpha1, beta1, alpha2, beta2, gamma2, N, M, V1, V2, width, stride) \
	field_linear_quadratic_kernel_4<LBY, LBX>\
		<<< dim3(M>>LBX>>LTX, N>>LBY>>LTY), dim3(1<<LBX, 1<<LBY), 0, stream >>>\
			(A, alpha1, beta1, alpha2, beta2, gamma2, N, M, V1, V2, width, stride)

//[16, 4]
#define field_linear_quadratic4_small(stream, A, alpha1, beta1, alpha2, beta2, gamma2, N, M, V1, V2, width, stride) \
	field_linear_quadratic_kernel_4<3, 3>\
		<<< dim3((M + 31)>>5, (N + 63)>>6), dim3(8, 8), 0, stream>>>\
			(A, alpha1, beta1, alpha2, beta2, gamma2, N, M, V1, V2, width, stride)

#endif


#ifndef FIELD_LINEAR_QUADRATIC_KERNEL_4
#define FIELD_LINERA_QUADRATIC_KERNEL_4

//[1] A belons to Mat[N, M]
//[2] V1[M] = field_sum: alpha1 * A + beta1,
//[3] V2[M] = field_sum: alpha2 * A^2 + beta2* + gamma2 
template<int LBY, int LBX>
__global__ void field_linear_quadratic_kernel_4(
	const float* __restrict__ A,
	float alpha1, float beta1,
	float alpha2, float beta2, float gamma2,
	int N, int M,
	float* __restrict__ V1,
	float* __restrict__ V2,
	int width, int stride)
{
	int by = blockIdx.y, bx = blockIdx.x;
	int ty = threadIdx.y, tx = threadIdx.x;

	__shared__ float4 As[1 << LBX][(2 << LBY) + 1];

	//[parallel field num = 4]-----------------------------------------
	const int offsetY = (by << LBY) + ty, stepY = (gridDim.y << LBY);
	const int offsetX = (bx << LBX) + tx, stepX = (gridDim.x << LBX);
	const int stepX4 = stepX << 2, M4 = (M >> 2) << 2;
	for (int x4 = offsetX << 2; x4 < M4; x4 += stepX4)
	{
		//compute area[thread reduce: 4 local result]------------------
		float4 v0 = F32_4_0, c0 = F32_4_0;//(deltaXp1 = deltaB)
		float4 v1 = F32_4_0, c1 = F32_4_0;//(deltaXp2 = deltaA)
		for (int y = offsetY; y < N; y += stepY)
		{
			float4 a = *(float4*)(A + y * M + x4);//A[y, x4]
			float4 dv0; simdLinear4(dv0, alpha1, a, beta1);//dv0 = alpha1*A + beta1
			float4 dv1; simdQuadratic4(dv1, alpha2, a, beta2, gamma2);//dv1 = alpha2*A^2 + beta2*A + gamma

			Kahan_simdAdd4(v0, dv0, c0);//v1 = v1 + dv1
			Kahan_simdAdd4(v1, dv1, c1);//v2 = v2 + dv2
		}

		As[tx][(ty << 1)    ] = float4{ v0.x, v0.y, v1.x, v1.y };
		As[tx][(ty << 1) + 1] = float4{ v0.z, v0.w, v1.z, v1.w };
		__syncthreads();

		//compute area[block reduce: 4 global result]------------------
		int Ax;
		if (LBY >= 6) {
			Ax = ((ty & 31) << 1) + (ty >> 5);
			if (ty < 64) { SMEM_simdAdd4(As[tx][Ax], As[tx][Ax], As[tx][Ax + 64]); }
			__syncthreads();
		}
		if (LBY >= 5) {
			Ax = ((ty & 15) << 1) + (ty >> 4);
			if (ty < 32) { SMEM_simdAdd4(As[tx][Ax], As[tx][Ax], As[tx][Ax + 32]); }
			__syncthreads();
		}
		if (LBY >= 4) {
			Ax = ((ty & 7) << 1) + (ty >> 3);
			if (ty < 16) { SMEM_simdAdd4(As[tx][Ax], As[tx][Ax], As[tx][Ax + 16]); }
			__syncthreads();
		}

		Ax = ((ty & 3) << 1) + (ty >> 2);//in all cases: LBY >= 3
		if (ty < 8) { SMEM_simdAdd4(As[tx][Ax], As[tx][Ax], As[tx][Ax + 8]); }
		__syncthreads();

		Ax = ((ty & 1) << 1) + (ty >> 1);
		if (ty < 4) { SMEM_simdAdd4(As[tx][Ax], As[tx][Ax], As[tx][Ax + 4]); }
		__syncthreads();

		Ax = ((ty & 0) << 1) + (ty >> 0);
		if (ty < 2) { SMEM_simdAdd4(As[tx][Ax], As[tx][Ax], As[tx][Ax + 2]); }
		__syncthreads();

		if (ty < 2) {
			float4 R = As[tx][Ax];
			float2 RV1 = float2{ R.x, R.y };
			float2 RV2 = float2{ R.z, R.w };

			int xindex2 = x4 + (ty << 1);
			bool wrt0 = ((xindex2    ) % stride) < width;
			bool wrt1 = ((xindex2 + 1) % stride) < width;
			RV1.x *= wrt0; RV1.y *= wrt1;//within_width2(RV1, xindex2, stride, width);
			RV2.x *= wrt0; RV2.y *= wrt1;//within_width2(RV2, xindex2, stride, width);

			int dst_index = by * M + xindex2;//[by, xindex2]
			*(float2*)(V1 + dst_index) = RV1;//deltaXp1 = deltaB
			*(float2*)(V2 + dst_index) = RV2;//deltaXp2 = deltaA
		}
		__syncthreads();
	}
}

#endif


#ifndef FIELD_LINEAR_QUADRATIC_STAGE
#define FIELD_LINEAR_QUADRATIC_STAGE

//M % 4 == 0, M >= 4
void __field_linear_quadratic_stage(cudaStream_t stream,
	const float* A,
	float alpha1, float beta1,
	float alpha2, float beta2, float gamma2,
	int N, int M,
	float* V1, 
	float* V2,
	int width, int stride)
{
	if (M > 15) {
		if (N > 63) { field_linear_quadratic4(stream, 3, 2, 3, 2, A, alpha1, beta1, alpha2, beta2, gamma2, N, M, V1, V2, width, stride); return; }//[64, 16]
		if (N > 31) { field_linear_quadratic4(stream, 3, 2, 2, 2, A, alpha1, beta1, alpha2, beta2, gamma2, N, M, V1, V2, width, stride); return; }//[32, 16]
		if (N > 15) { field_linear_quadratic4(stream, 3, 2, 1, 2, A, alpha1, beta1, alpha2, beta2, gamma2, N, M, V1, V2, width, stride); return; }//[16, 16]
	}
	if (M > 7) {
		if (N > 127) { field_linear_quadratic4(stream, 4, 1, 3, 2, A, alpha1, beta1, alpha2, beta2, gamma2, N, M, V1, V2, width, stride); return; }//[ 64, 8]
		if (N >  63) { field_linear_quadratic4(stream, 4, 1, 2, 2, A, alpha1, beta1, alpha2, beta2, gamma2, N, M, V1, V2, width, stride); return; }//[ 32, 8]
		if (N >  31) { field_linear_quadratic4(stream, 4, 1, 1, 2, A, alpha1, beta1, alpha2, beta2, gamma2, N, M, V1, V2, width, stride); return; }//[ 16, 8]
	}
	field_linear_quadratic4_small(stream, A, alpha1, beta1, alpha2, beta2, gamma2, N, M, V1, V2, width, stride);
}

#endif 


#ifndef FIELD_LINEAR_QUADRATIC
#define FIELD_LINEAR_QUADRATIC

//new fashion
int __field_linear_quadratic(JNIEnv *env, cudaStream_t stream1, cudaStream_t stream2,
	const float* A,
	float alpha1, float beta1, 
	float alpha2, float beta2, float gamma2,
	int N, int M,
	float* V1, float *Y1,//V1 = Y1.buf
	float* V2, float *Y2,//V2 = Y2.buf
	int width, int stride,
	int partNum)
{
	int nextN = field_nextN(N, M);//N <= HV
	if (nextN <= partNum) {//V.address = NULL, only 1 stage, write result to Y
		__field_linear_quadratic_stage(stream1, A, 
			alpha1, beta1, alpha2, beta2, gamma2, N, M, 
			Y1,//V1 = Y1.buf
			Y2,//V2 = Y2.buf
			width, stride);
		return nextN;
	}

	//V.address != NULL, at least 2 stages
	__field_linear_quadratic_stage(stream1, A, 
		alpha1, beta1, alpha2, beta2, gamma2, N, M, 
		V1,//V1 = Y1.buf
		V2,//V2 = Y2.buf
		width, stride);

	//====stream2 shold wait stream1======================================================
	cudaEvent_t event; cudaError_t error;
	error = cudaEventCreate(&event, cudaEventDisableTiming); handleError(error);
	error = cudaEventRecord(event, stream1); handleError(error);
	error = cudaStreamWaitEvent(stream2, event, cudaEventWaitDefault); handleError(error);
	//====stream2 shold wait stream1======================================================

	N = nextN, nextN = field_nextN(N, M);
	while (nextN > partNum) {//end: nextN <= partNum
		__field_sum_stage(stream1, V1, N, M, V1, width, stride);
		__field_sum_stage(stream2, V2, N, M, V2, width, stride);
		N = nextN, nextN = field_nextN(nextN, M);
	}
	__field_sum_stage(stream1, V1, N, M, Y1, width, stride);//the last stage
	__field_sum_stage(stream2, V2, N, M, Y2, width, stride);//the last stage

	error = cudaEventDestroy(event); handleError(error);
	return nextN;
}

#endif

#endif