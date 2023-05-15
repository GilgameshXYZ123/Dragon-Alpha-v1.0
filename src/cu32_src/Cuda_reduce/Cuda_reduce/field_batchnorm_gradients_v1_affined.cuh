#pragma once

#ifndef FIELD_BATCH_NORM_AFFINED_GRADIENTS_V1_H
#define FIELD_BATCH_NORM_AFFINED_GRADIENTS_V1_H

//(1) field_num % A.width ==0
//(2) field_num -> field_stride = field_num / width * stride
//(3) A.length % feature_num == 0
//(4) A.lengthv % fieled_stride == 0
//(5) row_num = A.lengthv / field_stride
//(6) [y, x] -> [row_num, field_stride]
//(7) [N, M] = [row_num, field_stride], A.lengthv = N*M
//V1: holdY(), Y is not changed
#ifndef FIELD_BATCH_NORM_AFFINED_GRADIENTS_V1_CALL
#define FIELD_BATCH_NORM_AFFINED_GRADIENTS_V1_CALL

//LTX >=2
#define field_batchnorm_affined_gradients_v1_k4(stream1, LBY, LBX, LTY, LTX, deltaY, Y, A, B, N, M, deltaXp1, deltaXp2, width, stride) \
	field_batchnorm_affined_gradients_v1_kernel_4<LBY, LBX>\
		<<< dim3(M>>LBX>>LTX, N>>LBY>>LTY), dim3(1<<LBX, 1<<LBY), 0, stream1 >>>\
			(deltaY, Y, A, B, N, M, deltaXp1, deltaXp2, width, stride)

//[16, 4]
#define field_batchnorm_affined_gradients_v1_k4_small(stream1, deltaY, Y, A, B, N, M, deltaXp1, deltaXp2, width, stride) \
	field_batchnorm_affined_gradients_v1_kernel_4<3, 3>\
		<<< dim3((M + 31)>>5, (N + 63)>>6), dim3(8, 8), 0, stream1>>>\
			(deltaY, Y, A, B, N, M, deltaXp1, deltaXp2, width, stride)

#endif


#ifndef FIELD_BATCH_NORM_AFFINED_GRADIENTS_V1_KERNEL_4
#define FIELD_BATCH_NORM_AFFINED_GRADIENTS_V1_KERNEL_4

//======[Document]=============================================================================
//forward:
//<1> std = sqrt(sqmean - mean^2 + eps)
//<2> Y = A * (X - mean) / std + B
//backward:
//<1> deltaXp1 = field_sum: deltaY
//<2> deltaXp2 = field_sum: deltaY * ((X - mean) / std)
//<3> deltaB   = field_sum: deltaY
//<4> deltaA   = field_sum: deltaY * ((X - mean) / std)
//<5> deltaX = (A / std) * { deltaY - (1 / N) * { deltaXp1 + ((X - mean) / std) * deltaXp2 } }
//STEP: 
//<1> rstd = rsqrt(sqmean - mean^2 + eps)
//<2> X_norm = (Y - B) / A
//<3> (deltaXp2 = deltaA) = field_sum: deltaY * X_norm
//<4> (deltaXp1 = deltaB) = field_sum: deltaY
//======[Document]============================================================================

template<int LBY, int LBX>
__global__ void field_batchnorm_affined_gradients_v1_kernel_4(
	const float* __restrict__ deltaY,
	const float* __restrict__ Y,
	const float* __restrict__ A,
	const float* __restrict__ B,
	int N, int M,
	float* __restrict__ deltaXp1,//deltaB_buf
	float* __restrict__ deltaXp2,//deltaA_buf
	int width, int stride)
{
	int by = blockIdx.y, bx = blockIdx.x;
	int ty = threadIdx.y, tx = threadIdx.x;

	__shared__ float4 As[1 << LBX][(2 << LBY) + 1];

	//parallel field num = 4
	const int offsetY = (by << LBY) + ty, stepY = (gridDim.y << LBY);
	const int offsetX = (bx << LBX) + tx, stepX = (gridDim.x << LBX);
	const int stepX4 = stepX << 2, M4 = (M >> 2) << 2;

	float2 table[2]; table[0] = F32_2_0;//(A == 0) may cause NaN
	for (int x4 = offsetX << 2; x4 < M4; x4 += stepX4)
	{
		const float4 a = *(float4*)(A + x4);
		const float4 b = *(float4*)(B + x4);

		//compute area[thread reduce: 4 local result]------------------
		float4 v0  = F32_4_0, c0  = F32_4_0;//[deltaXp1, deltaB]
		float4 v1a = F32_4_0, c1a = F32_4_0;//[deltaXp2, deltaA]
		float4 v1b = F32_4_0, c1b = F32_4_0;//[deltaXp2, deltaA]
		for (int y = offsetY; y < N; y += stepY)
		{
			int src_offset = y * M + x4;//[y, x4]
			const float4 dy = *(float4*)(deltaY + src_offset);//deltaY[y, x4]
			const float4 yv = *(float4*)(Y + src_offset);//Y[y, x4]
			
			float4 k;
			k.x = __fdividef(dy.x, a.x);
			k.y = __fdividef(dy.y, a.y);
			k.z = __fdividef(dy.z, a.z);
			k.w = __fdividef(dy.w, a.w);

			float4 dxp2a;//(deltaXp2 = deltaA) = deltaY * X_norm = deltaY * ((Y - B) / A)
			dxp2a.x = k.x * yv.x;
			dxp2a.y = k.y * yv.y;
			dxp2a.z = k.z * yv.z;
			dxp2a.w = k.w * yv.w;

			float4 dxp2b;//(deltaXp2 = deltaA) = deltaY * X_norm = deltaY * ((Y - B) / A)
			dxp2b.x = -k.x * b.x;
			dxp2b.y = -k.y * b.y;
			dxp2b.z = -k.z * b.z;
			dxp2b.w = -k.w * b.w;

			Kahan_simdAdd4(v0,     dy,  c0);//field_sum for: deltaXp1, deltaB
			Kahan_simdAdd4(v1a, dxp2a, c1a);//field sum for: deltaXp2, deltaA
			Kahan_simdAdd4(v1b, dxp2b, c1b);//field sum for: deltaXp2, deltaA
		}

		float4 v1; simdAdd4(v1, v1a, v1b);
		As[tx][(ty << 1)    ] = float4{ v0.x, v0.y, v1.x, v1.y };//[deltaXp1, deltaB]
		As[tx][(ty << 1) + 1] = float4{ v0.z, v0.w, v1.z, v1.w };//[deltaXp2, deltaA]
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
			float4 R = As[tx][Ax];//[deltaXp1 = deltaB, deltaXp2 = deltaA]
			float2 result0 = float2{ R.x, R.y };//deltaXp1, deltaB
			float2 result1 = float2{ R.z, R.w };//deltaXp2, deltaA

			int xindex2 = x4 + (ty << 1);
			within_width_zero_nan2(result0, xindex2, table, stride, width);
			within_width_zero_nan2(result1, xindex2, table, stride, width);

			int dst_index = by * M + xindex2;//[by, xindex2]
			*(float2*)(deltaXp1 + dst_index) = result0;//deltaXp1, deltaB
			*(float2*)(deltaXp2 + dst_index) = result1;//deltaXp2, deltaA
		}
		__syncthreads();
	}
}

#endif


#ifndef FIELD_BATCH_NORM_AFFINED_GRADIENTS_V1_STAGE
#define FIELD_BATCH_NORM_AFFINED_GRADIENTS_V1_STAGE

//M % 4 == 0, M >= 4
void __field_batchnorm_affined_gradients_v1_stage(cudaStream_t stream,
	const float* deltaY, const float* Y,
	const float* A, const float* B,
	int N, int M,
	float* deltaXp1,//used as deltaB_buf
	float* deltaXp2,//used as deltaA_buf
	int width, int stride)
{
	if (M > 15) {
		if (N > 63) { field_batchnorm_affined_gradients_v1_k4(stream, 3, 2, 3, 2, deltaY, Y, A, B, N, M, deltaXp1, deltaXp2, width, stride); return; }//[64, 16]
		if (N > 31) { field_batchnorm_affined_gradients_v1_k4(stream, 3, 2, 2, 2, deltaY, Y, A, B, N, M, deltaXp1, deltaXp2, width, stride); return; }//[32, 16]
		if (N > 15) { field_batchnorm_affined_gradients_v1_k4(stream, 3, 2, 1, 2, deltaY, Y, A, B, N, M, deltaXp1, deltaXp2, width, stride); return; }//[16, 16]
	}
	if (M > 7) {
		if (N > 127) { field_batchnorm_affined_gradients_v1_k4(stream, 4, 1, 3, 2, deltaY, Y, A, B, N, M, deltaXp1, deltaXp2, width, stride); return; }//[ 64, 8]
		if (N > 63) { field_batchnorm_affined_gradients_v1_k4(stream, 4, 1, 2, 2, deltaY, Y, A, B, N, M, deltaXp1, deltaXp2, width, stride); return; }//[ 32, 8]
		if (N > 31) { field_batchnorm_affined_gradients_v1_k4(stream, 4, 1, 1, 2, deltaY, Y, A, B, N, M, deltaXp1, deltaXp2, width, stride); return; }//[ 16, 8]
	}
	field_batchnorm_affined_gradients_v1_k4_small(stream, deltaY, Y, A, B, N, M, deltaXp1, deltaXp2, width, stride);
}

#endif 


#ifndef FIELD_BATCH_NORM_AFFINED_GRADIENTS_V1
#define FIELD_BATCH_NORM_AFFINED_GRADIENTS_V1

//(correct)
int __field_batchnorm_affined_gradients_v1(JNIEnv *env,
	cudaStream_t stream1, cudaStream_t stream2,
	const float* deltaY, const float* Y,
	const float* A, const float *B,
	int N, int M,
	float* deltaXp1,//used as deltaB_buf
	float* deltaXp2,//used as deltaA_buf
	float* deltaA,
	float* deltaB,
	int width, int stride,
	int partNum)
{
	int nextN = field_nextN(N, M);//N <= HV
	if (nextN <= partNum) {//V.address = NULL, only 1 stage, write result to Y
		__field_batchnorm_affined_gradients_v1_stage(stream1,
			deltaY, Y, A, B, N, M,
			deltaB,//deltaXp1 = deltaB
			deltaA,//deltaXp2 = deltaA
			width, stride);
		return nextN;
	}

	//V.address != NULL, at least 2 stages
	__field_batchnorm_affined_gradients_v1_stage(stream1,
		deltaY, Y, A, B, N, M,
		deltaXp1,//used as deltaB_buf
		deltaXp2,//used as deltaA_buf
		width, stride);

	//====stream2 shold wait stream1======================================================
	cudaEvent_t event; cudaError_t error;
	error = cudaEventCreate(&event, cudaEventDisableTiming); handleError(error);
	error = cudaEventRecord(event, stream1); handleError(error);
	error = cudaStreamWaitEvent(stream2, event, cudaEventWaitDefault); handleError(error);
	//====stream2 shold wait stream1======================================================

	N = nextN, nextN = field_nextN(N, M);//N % 4 == 0, N >= 4
	while (nextN > partNum) {//end: nextN <= partNum
		__field_sum_stage(stream1, deltaXp1, N, M, deltaXp1, width, stride);
		__field_sum_stage(stream2, deltaXp2, N, M, deltaXp2, width, stride);
		N = nextN, nextN = field_nextN(nextN, M);
	}
	__field_sum_stage(stream1, deltaXp1, N, M, deltaB, width, stride);//deltaXp1 = deltaB
	__field_sum_stage(stream2, deltaXp2, N, M, deltaA, width, stride);//deltaXp2 = deltaA

	error = cudaEventDestroy(event); handleError(error);
	return nextN;
}

#endif

#endif