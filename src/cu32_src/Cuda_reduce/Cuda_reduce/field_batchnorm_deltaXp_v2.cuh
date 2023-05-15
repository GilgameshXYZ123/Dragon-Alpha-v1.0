#pragma once

#ifndef FIELD_BATCH_NORM_DELTAXP_V2_H
#define FIELD_BATCH_NORM_DELTAXP_V2_H

//(1) field_num % A.width ==0
//(2) field_num -> field_stride = field_num / width * stride
//(3) A.length % feature_num == 0
//(4) A.lengthv % fieled_stride == 0
//(5) row_num = A.lengthv / field_stride
//(6) [y, x] -> [row_num, field_stride]
//(7) [N, M] = [row_num, field_stride], A.lengthv = N*M
//V2: holdX(), X is not changed
#ifndef FIELD_BATCH_NORM_DELTAXP_V2_CALL
#define FIELD_BATCH_NORM_DELTAXP_V2_CALL

//LTX >=2
#define field_batchnorm_deltaXp_v2_k4(stream1, LBY, LBX, LTY, LTX, deltaY, X, X_mean, X_square_mean, eps, N, M, deltaXp1, deltaXp2, width, stride) \
	field_batchnorm_deltaXp_v2_kernel_4<LBY, LBX>\
		<<< dim3(M>>LBX>>LTX, N>>LBY>>LTY), dim3(1<<LBX, 1<<LBY), 0, stream1 >>>\
			(deltaY, X, X_mean, X_square_mean, eps, N, M, deltaXp1, deltaXp2, width, stride)

//[16, 4]
#define field_batchnorm_deltaXp_v2_k4_small(stream1, deltaY, X, X_mean, X_square_mean, eps, N, M, deltaXp1, deltaXp2, width, stride) \
	field_batchnorm_deltaXp_v2_kernel_4<3, 3>\
		<<< dim3((M + 31)>>5, (N + 63)>>6), dim3(8, 8), 0, stream1>>>\
			(deltaY, X, X_mean, X_square_mean, eps, N, M, deltaXp1, deltaXp2, width, stride)

#endif


#ifndef FIELD_BATCH_NORM_DELTAXP_V2_KERNEL_4
#define FIELD_BATCH_NORM_DELTAXP_V2_KERNEL_4

//affined = false or true
//V2: holdX(), X is not changed
//[1] deltaY[N, M], X[N, M]
//[2] X_mean[M], X_squareMean[M]: mean of each field of [X, X^2]
//[3] deltaX_p1[M] = field_sum: deltaY * (X*X_mean - X_squareMean - eps) 
//[4] deltaX_p2[M] = field_sum: deltaY * (X - X_mean) 
template<int LBY, int LBX>
__global__ void field_batchnorm_deltaXp_v2_kernel_4(
	const float* __restrict__ deltaY,
	const float* __restrict__ X,
	const float* __restrict__ X_mean,
	const float* __restrict__ X_square_mean, float eps,
	int N, int M,
	float* __restrict__ deltaXp1,
	float* __restrict__ deltaXp2,
	int width, int stride)
{
	int by = blockIdx.y, bx = blockIdx.x;
	int ty = threadIdx.y, tx = threadIdx.x;

	__shared__ float2 As1[2 << LBX << LBY];//[BX, 2*BY]
	__shared__ float2 As2[2 << LBX << LBY];//[BX, 2*BY]
	const int As_xy = ((tx << LBY) + ty) << 1;//(tx, ty*2)
	
	//parallel field num = 4
	const int offsetY = (by << LBY) + ty, stepY = (gridDim.y << LBY);
	const int offsetX = (bx << LBX) + tx, stepX = (gridDim.x << LBX);
	int x4 = offsetX << 2, stepX4 = stepX << 2, M4 = (M >> 2) << 2;

	for (; x4 < M4; x4 += stepX4)
	{
		float4 x_mean  = *(float4*)(X_mean + x4);
		float4 x_smean = *(float4*)(X_square_mean + x4);
		
		float4 c1 = make_float4(0, 0, 0, 0);//solve the errors
		float4 c2 = make_float4(0, 0, 0, 0);//solve the errors 
		float4 v1 = make_float4(0, 0, 0, 0);//thread reduce: 4 local result
		float4 v2 = make_float4(0, 0, 0, 0);//thread reduce: 4 local result

		for (int y = offsetY; y < N; y += stepY)
		{
			int y_x4 = y * M + x4;//(y, x4)
			float4 dy = *(float4*)(deltaY + y_x4);//deltaY[y, x4]
			float4 xv = *(float4*)(X + y_x4);//deltaX[y, x4]

			float4 dx1;//deltaXp1 = deltaY * (X*X_mean - X_squareMean - eps) 
			dx1.x = dy.x * (xv.x * x_mean.x - x_smean.x - eps);
			dx1.y = dy.y * (xv.y * x_mean.y - x_smean.y - eps);
			dx1.z = dy.z * (xv.z * x_mean.z - x_smean.z - eps);
			dx1.w = dy.w * (xv.w * x_mean.w - x_smean.w - eps);

			float4 dx2;//deltaXp2 = deltaY * (X - X_mean) 
			dx2.x = dy.x * (xv.x - x_mean.x);
			dx2.y = dy.y * (xv.y - x_mean.y);
			dx2.z = dy.z * (xv.z - x_mean.z);
			dx2.w = dy.w * (xv.w - x_mean.w);

			Kahan_simdAdd4(v1, dx1, c1);//field sum for deltaXp1: v1 += dx1
			Kahan_simdAdd4(v2, dx2, c2);//field sum for deltaXp2: v2 += dx2
		}
		*(float4*)(As1 + As_xy) = v1;
		*(float4*)(As2 + As_xy) = v2;
		__syncthreads();

		int As_index;
		if (LBY >= 6) {//block reduce: get 4 global result
			As_index = (((tx << LBY) + (ty & 31)) << 1) + (ty >> 5);
			if (ty < 64) {
				simdAdd2(As1[As_index], As1[As_index], As1[As_index + 64]); 
				simdAdd2(As2[As_index], As2[As_index], As2[As_index + 64]);
			}
			__syncthreads();
		}
		if (LBY >= 5) {
			As_index = (((tx << LBY) + (ty & 15)) << 1) + (ty >> 4);
			if (ty < 32) {
				simdAdd2(As1[As_index], As1[As_index], As1[As_index + 32]);
				simdAdd2(As2[As_index], As2[As_index], As2[As_index + 32]);
			}
			__syncthreads();
		}
		if (LBY >= 4) {
			As_index = (((tx << LBY) + (ty & 7)) << 1) + (ty >> 3);
			if (ty < 16) { 
				simdAdd2(As1[As_index], As1[As_index], As1[As_index + 16]); 
				simdAdd2(As2[As_index], As2[As_index], As2[As_index + 16]);
			}
			__syncthreads();
		}

		As_index = (((tx << LBY) + (ty & 3)) << 1) + (ty >> 2);
		if (ty < 8) {//in all cases: LBY >= 3
			simdAdd2(As1[As_index], As1[As_index], As1[As_index + 8]); 
			simdAdd2(As2[As_index], As2[As_index], As2[As_index + 8]);
		}
		__syncthreads();

		As_index = (((tx << LBY) + (ty & 1)) << 1) + (ty >> 1);
		if (ty < 4) { 
			simdAdd2(As1[As_index], As1[As_index], As1[As_index + 4]);
			simdAdd2(As2[As_index], As2[As_index], As2[As_index + 4]);
		}
		__syncthreads();

		As_index = (((tx << LBY) + (ty & 0)) << 1) + ty;
		if (ty < 2) { 
			simdAdd2(As1[As_index], As1[As_index], As1[As_index + 2]); 
			simdAdd2(As2[As_index], As2[As_index], As2[As_index + 2]);
		}
		__syncthreads();

		if (ty < 2) {
			float2 result1 = As1[As_index];
			float2 result2 = As2[As_index];

			int xindex2 = x4 + (ty << 1);
			within_width2(result1, xindex2, stride, width);
			within_width2(result2, xindex2, stride, width);

			*(float2*)(&get(deltaXp1, by, xindex2, M)) = result1;
			*(float2*)(&get(deltaXp2, by, xindex2, M)) = result2;
		}
		__syncthreads();
	}
}

#endif


#ifndef FIELD_BATCH_NORM_DELTAXP_V2_STAGE
#define FIELD_BATCH_NORM_DELTAXP_V2_STAGE

//M % 4 == 0, M >= 4
void __field_batchnorm_deltaXp_v2_stage(cudaStream_t stream,
	const float* deltaY, const float* X,
	const float* X_mean,
	const float* X_square_mean, float eps,
	int N, int M,
	float* deltaXp1, float* deltaXp2,
	int width, int stride)
{
	if (M > 15) {
		if (N > 63) { field_batchnorm_deltaXp_v2_k4(stream, 3, 2, 3, 2, deltaY, X, X_mean, X_square_mean, eps, N, M, deltaXp1, deltaXp2, width, stride); return; }//[64, 16]
		if (N > 31) { field_batchnorm_deltaXp_v2_k4(stream, 3, 2, 2, 2, deltaY, X, X_mean, X_square_mean, eps, N, M, deltaXp1, deltaXp2, width, stride); return; }//[32, 16]
		if (N > 15) { field_batchnorm_deltaXp_v2_k4(stream, 3, 2, 1, 2, deltaY, X, X_mean, X_square_mean, eps, N, M, deltaXp1, deltaXp2, width, stride); return; }//[16, 16]
	}
	if (M > 7) {
		if (N > 127) { field_batchnorm_deltaXp_v2_k4(stream, 4, 1, 3, 2, deltaY, X, X_mean, X_square_mean, eps, N, M, deltaXp1, deltaXp2, width, stride); return; }//[ 64, 8]
		if (N >  63) { field_batchnorm_deltaXp_v2_k4(stream, 4, 1, 2, 2, deltaY, X, X_mean, X_square_mean, eps, N, M, deltaXp1, deltaXp2, width, stride); return; }//[ 32, 8]
		if (N >  31) { field_batchnorm_deltaXp_v2_k4(stream, 4, 1, 1, 2, deltaY, X, X_mean, X_square_mean, eps, N, M, deltaXp1, deltaXp2, width, stride); return; }//[ 16, 8]
	}
	field_batchnorm_deltaXp_v2_k4_small(stream, deltaY, X, X_mean, X_square_mean, eps, N, M, deltaXp1, deltaXp2, width, stride);
}

#endif 


#ifndef FIELD_BATCH_NORM_DELTAXP_V2
#define FIELD_BATCH_NORM_DELTAXP_V2

//(correct)
int __field_batchnorm_deltaXp_v2(JNIEnv *env, cudaStream_t stream1, cudaStream_t stream2,
	const float* deltaY, const float* X,
	const float* X_mean,
	const float* X_square_mean, float eps,
	int N, int M,
	float* deltaXp1, float* deltaXp2,
	int width, int stride,
	int partNum)
{
	int nextN = field_nextN(N, M);//N <= HV
	if (nextN <= partNum) {//V.address = NULL, only 1 stage, write result to Y
		__field_batchnorm_deltaXp_v2_stage(stream1, deltaY, X, X_mean, X_square_mean, eps, N, M, deltaXp1, deltaXp2, width, stride);
		return nextN;
	}

	//V.address != NULL, at least 2 stages
	__field_batchnorm_deltaXp_v2_stage(stream1, deltaY, X, X_mean, X_square_mean, eps, N, M, deltaXp1, deltaXp2, width, stride);

	//====stream2 shold wait stream1======================================================
	cudaEvent_t event; cudaError_t error;
	error = cudaEventCreate(&event, cudaEventDisableTiming); handleError(error);
	error = cudaEventRecord(event, stream1); handleError(error);
	error = cudaStreamWaitEvent(stream2, event, cudaEventWaitDefault); handleError(error);
	//====stream2 shold wait stream1======================================================

	N = nextN, nextN = field_nextN(N, M);//N % 4 == 0, N >= 4
	while (nextN > partNum)//end: nextN <= partNum
	{
		__field_sum_stage(stream1, deltaXp1, N, M, deltaXp1, width, stride);
		__field_sum_stage(stream2, deltaXp2, N, M, deltaXp2, width, stride);
		N = nextN, nextN = field_nextN(nextN, M);
	}
	__field_sum_stage(stream1, deltaXp1, N, M, deltaXp1, width, stride);//the last stage
	__field_sum_stage(stream2, deltaXp2, N, M, deltaXp2, width, stride);//the last stage

	error = cudaEventDestroy(event); handleError(error);
	return nextN;
}

#endif

#endif
