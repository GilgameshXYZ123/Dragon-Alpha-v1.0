#pragma once

#ifndef FIELD_BATCH_NORM_AFFINED_GRADIENTS_V2_H
#define FIELD_BATCH_NORM_AFFINED_GRADIENTS_V2_H

//(1) field_num % A.width ==0
//(2) field_num -> field_stride = field_num / width * stride
//(3) A.length % feature_num == 0
//(4) A.lengthv % fieled_stride == 0
//(5) row_num = A.lengthv / field_stride
//(6) [y, x] -> [row_num, field_stride]
//(7) [N, M] = [row_num, field_stride], A.lengthv = N*M
//V2: holdX(), X is not changed
#ifndef FIELD_BATCH_NORM_AFFINED_GRADIENTS_V2_CALL
#define FIELD_BATCH_NORM_AFFINED_GRADIENTS_V2_CALL

//LTX >=2
#define field_batchnorm_affined_gradients_v2_k4(stream1, LBY, LBX, LTY, LTX, deltaY, X, X_mean, X_square_mean, eps, N, M, deltaXp1, deltaXp2, deltaA, deltaB, width, stride) \
	field_batchnorm_affined_gradients_v2_kernel_4<LBY, LBX>\
		<<< dim3(M>>LBX>>LTX, N>>LBY>>LTY), dim3(1<<LBX, 1<<LBY), 0, stream1 >>>\
			(deltaY, X, X_mean, X_square_mean, eps, N, M, deltaXp1, deltaXp2, deltaA, deltaB, width, stride)

//[16, 4]
#define field_batchnorm_affined_gradients_v2_k4_small(stream1, deltaY, X, X_mean, X_square_mean, eps, N, M, deltaXp1, deltaXp2, deltaA, deltaB, width, stride) \
	field_batchnorm_affined_gradients_v2_kernel_4<3, 3>\
		<<< dim3((M + 31)>>5, (N + 63)>>6), dim3(8, 8), 0, stream1>>>\
			(deltaY, X, X_mean, X_square_mean, eps, N, M, deltaXp1, deltaXp2, deltaA, deltaB, width, stride)

#endif


#define SMEM_simdAdd4_f64(c, a, b) { double4 v1 = a; double4 v2 = b; simdAdd4(v1, v1, v2); c = v1; }

//v = v + a
#define Kahan_simdAdd4_f64(v, a, c)	{\
	double4 dv;\
	dv.x = a.x - c.x;\
	dv.y = a.y - c.y;\
	dv.z = a.z - c.z;\
	dv.w = a.w - c.w;\
	double4 t;\
	t.x = v.x + dv.x;\
	t.y = v.y + dv.y;\
	t.z = v.z + dv.z;\
	t.w = v.w + dv.w;\
	c.x = (t.x - v.x) - dv.x;\
	c.y = (t.y - v.y) - dv.y;\
	c.z = (t.z - v.z) - dv.z;\
	c.w = (t.w - v.w) - dv.w;\
	v = t; }

#ifndef FIELD_BATCH_NORM_AFFINED_GRADIENTS_V2_KERNEL_4
#define FIELD_BATCH_NORM_AFFINED_GRADIENTS_V2_KERNEL_4

//affined = false or true
//V2: holdX(), X is not changed
//[1] deltaY[N, M], X[N, M]
//[2] X_mean[M], X_squareMean[M]: mean of each field of [X, X^2]
//<1> deltaX_p1[M] = field_sum: deltaY * (X*X_mean - X_squareMean - eps) 
//<2> deltaX_p2[M] = field_sum: deltaY * (X - X_mean) 
//<3> deltaA[M] = field_sum: deltaY[i] * (X[i] - X_mean) / sqrt(X_var)
//<4> deltaB[M] = field_sum: deltaY[i]
template<int LBY, int LBX>
__global__ void field_batchnorm_affined_gradients_v2_kernel_4(
	const float* __restrict__ deltaY,
	const float* __restrict__ X,
	const float* __restrict__ X_mean,
	const float* __restrict__ X_square_mean, float eps,
	int N, int M,
	float* __restrict__ deltaXp1,
	float* __restrict__ deltaXp2,
	float* __restrict__ deltaA,
	float* __restrict__ deltaB,
	int width, int stride)
{
	int by = blockIdx.y, bx = blockIdx.x;
	int ty = threadIdx.y, tx = threadIdx.x;

	__shared__ double4 As0[1 << LBX][(2 << LBY) + 1];//[BX, 2*BY]
	__shared__ double4 As1[1 << LBX][(2 << LBY) + 1];//[BX, 2*BY]

	//parallel field num = 4
	const int offsetY = (by << LBY) + ty, stepY = (gridDim.y << LBY);
	const int offsetX = (bx << LBX) + tx, stepX = (gridDim.x << LBX);
	int x4 = offsetX << 2, stepX4 = stepX << 2, M4 = (M >> 2) << 2;

	float2 table[2]; table[0] = F32_2_0;
	for (; x4 < M4; x4 += stepX4)
	{
		float4 fx_mean = *(float4*)(X_mean + x4);
		float4 fx_smean = *(float4*)(X_square_mean + x4);
		float4 x_mean; COPY4(x_mean, fx_mean);
		float4 x_smean; COPY4(x_smean, fx_smean);

		double4 x_rstd;//rstd = 1 / sqrt(square_mean - mean^2 + eps)
		x_rstd.x = rsqrt(x_smean.x - x_mean.x * x_mean.x + eps);
		x_rstd.y = rsqrt(x_smean.y - x_mean.y * x_mean.y + eps);
		x_rstd.z = rsqrt(x_smean.z - x_mean.z * x_mean.z + eps);
		x_rstd.w = rsqrt(x_smean.w - x_mean.w * x_mean.w + eps);

		//compute area[thread reduce: 4 local result]------------------
		double4 v0 = F64_4_0;//deltaXp1
		double4 v1 = F64_4_0;//deltaXp2
		double4 v2 = F64_4_0;//deltaA
		double4 v3 = F64_4_0;//deltaB

		for (int y = offsetY; y < N; y += stepY)
		{
			int y_x4 = y * M + x4;//(y, x4)
			float4 fdy = *(float4*)(deltaY + y_x4);//deltaY[y, x4]
			float4 fxv = *(float4*)(X + y_x4);//deltaX[y, x4]
			double4 xv; COPY4(xv, fxv);
			double4 dy; COPY4(dy, fdy);

			//F64--------------------------------------------------
			//<1> deltaXp1 = deltaY * (X*X_mean - X_squareMean - eps) 
			v0.x += dy.x * (xv.x * x_mean.x - x_smean.x - eps);
			v0.y += dy.y * (xv.y * x_mean.y - x_smean.y - eps);
			v0.z += dy.z * (xv.z * x_mean.z - x_smean.z - eps);
			v0.w += dy.w * (xv.w * x_mean.w - x_smean.w - eps);

			//<2> deltaXp2 = deltaY * (X - X_mean) 
			v1.x += dy.x * (xv.x - x_mean.x);
			v1.y += dy.y * (xv.y - x_mean.y);
			v1.z += dy.z * (xv.z - x_mean.z);
			v1.w += dy.w * (xv.w - x_mean.w);

			//<3> deltaA = deltaY * (X - X_mean) / sqrt(X_var)
			v2.x += dy.x * (xv.x - x_mean.x) * x_rstd.x;
			v2.y += dy.y * (xv.y - x_mean.y) * x_rstd.y;
			v2.z += dy.z * (xv.z - x_mean.z) * x_rstd.z;
			v2.w += dy.w * (xv.w - x_mean.w) * x_rstd.w;

			simdAdd4(v3, v3, dy);//field sum for deltaB
		}

		//F64-------------------------------------------------------
		As0[tx][(ty << 1)] = double4{ v0.x, v0.y, v1.x, v1.y };//[deltaXp1, deltaXp2]
		As0[tx][(ty << 1) + 1] = double4{ v0.z, v0.w, v1.z, v1.w };//[deltaXp1, deltaXp2]
		As1[tx][(ty << 1)] = double4{ v2.x, v2.y, v3.x, v3.y };//[deltaA, deltaB]
		As1[tx][(ty << 1) + 1] = double4{ v2.z, v2.w, v3.z, v3.w };//[deltaA, deltaB]
		__syncthreads();

		//compute area[block reduce: 4 global result]------------------
		int Ax;
		if (LBY >= 6) {
			Ax = ((ty & 31) << 1) + (ty >> 5);
			if (ty < 64) {
				SMEM_simdAdd4_f64(As0[tx][Ax], As0[tx][Ax], As0[tx][Ax + 64]);
				SMEM_simdAdd4_f64(As1[tx][Ax], As1[tx][Ax], As1[tx][Ax + 64]);
			}
			__syncthreads();
		}
		if (LBY >= 5) {
			Ax = ((ty & 15) << 1) + (ty >> 4);
			if (ty < 32) {
				SMEM_simdAdd4_f64(As0[tx][Ax], As0[tx][Ax], As0[tx][Ax + 32]);
				SMEM_simdAdd4_f64(As1[tx][Ax], As1[tx][Ax], As1[tx][Ax + 32]);
			}
			__syncthreads();
		}
		if (LBY >= 4) {
			Ax = ((ty & 7) << 1) + (ty >> 3);
			if (ty < 16) {
				SMEM_simdAdd4_f64(As0[tx][Ax], As0[tx][Ax], As0[tx][Ax + 16]);
				SMEM_simdAdd4_f64(As1[tx][Ax], As1[tx][Ax], As1[tx][Ax + 16]);
			}
			__syncthreads();
		}

		Ax = ((ty & 3) << 1) + (ty >> 2);
		if (ty < 8) {//in all cases: LBY >= 3
			SMEM_simdAdd4_f64(As0[tx][Ax], As0[tx][Ax], As0[tx][Ax + 8]);
			SMEM_simdAdd4_f64(As1[tx][Ax], As1[tx][Ax], As1[tx][Ax + 8]);
		}
		__syncthreads();

		Ax = ((ty & 1) << 1) + (ty >> 1);
		if (ty < 4) {
			SMEM_simdAdd4_f64(As0[tx][Ax], As0[tx][Ax], As0[tx][Ax + 4]);
			SMEM_simdAdd4_f64(As1[tx][Ax], As1[tx][Ax], As1[tx][Ax + 4]);
		}
		__syncthreads();

		Ax = ((ty & 0) << 1) + (ty >> 0);
		if (ty < 2) {
			SMEM_simdAdd4_f64(As0[tx][Ax], As0[tx][Ax], As0[tx][Ax + 2]);
			SMEM_simdAdd4_f64(As1[tx][Ax], As1[tx][Ax], As1[tx][Ax + 2]);
		}
		__syncthreads();

		if (ty < 2) {
			double4 R0 = As0[tx][Ax];//[deltaXp1, deltaXp2]
			double4 R1 = As1[tx][Ax];//[deltaA, deltaB]

			//F64----------------------------------------------------------
			float2 result0 = float2{ (float)R0.x, (float)R0.y };//deltaXp1
			float2 result1 = float2{ (float)R0.z, (float)R0.w };//deltaXp2
			float2 result2 = float2{ (float)R1.x, (float)R1.y };//deltaA
			float2 result3 = float2{ (float)R1.z, (float)R1.w };//deltaB

			//F32----------------------------------------------------------
			//float2 result0 = float2{ R0.x, R0.y };//deltaXp1
			//float2 result1 = float2{ R0.z, R0.w };//deltaXp2
			//float2 result2 = float2{ R1.x, R1.y };//deltaA
			//float2 result3 = float2{ R1.z, R1.w };//deltaB

			int xindex2 = x4 + (ty << 1);
			within_width2(result0, xindex2, stride, width);
			within_width2(result1, xindex2, stride, width);
			within_width_zero_nan2(result2, xindex2, table, stride, width);
			within_width_zero_nan2(result3, xindex2, table, stride, width);

			*(float2*)(&get(deltaXp1, by, xindex2, M)) = result0;
			*(float2*)(&get(deltaXp2, by, xindex2, M)) = result1;
			*(float2*)(&get(deltaA, by, xindex2, M)) = result2;
			*(float2*)(&get(deltaB, by, xindex2, M)) = result3;
		}
		__syncthreads();
	}
}

#endif


#ifndef FIELD_BATCH_NORM_AFFINED_GRADIENTS_V2_STAGE
#define FIELD_BATCH_NORM_AFFINED_GRADIENTS_V2_STAGE

//M % 4 == 0, M >= 4
void __field_batchnorm_affined_gradients_v2_stage(cudaStream_t stream,
	const float* deltaY, const float* X,
	const float* X_mean,
	const float* X_square_mean, float eps,
	int N, int M,
	float* deltaXp1,
	float* deltaXp2,
	float* deltaA,
	float* deltaB,
	int width, int stride)
{
	if (M > 15) {
		if (N > 63) { field_batchnorm_affined_gradients_v2_k4(stream, 3, 2, 3, 2, deltaY, X, X_mean, X_square_mean, eps, N, M, deltaXp1, deltaXp2, deltaA, deltaB, width, stride); return; }//[64, 16]
		if (N > 31) { field_batchnorm_affined_gradients_v2_k4(stream, 3, 2, 2, 2, deltaY, X, X_mean, X_square_mean, eps, N, M, deltaXp1, deltaXp2, deltaA, deltaB, width, stride); return; }//[32, 16]
		if (N > 15) { field_batchnorm_affined_gradients_v2_k4(stream, 3, 2, 1, 2, deltaY, X, X_mean, X_square_mean, eps, N, M, deltaXp1, deltaXp2, deltaA, deltaB, width, stride); return; }//[16, 16]
	}
	if (M > 7) {
		if (N > 127) { field_batchnorm_affined_gradients_v2_k4(stream, 4, 1, 3, 2, deltaY, X, X_mean, X_square_mean, eps, N, M, deltaXp1, deltaXp2, deltaA, deltaB, width, stride); return; }//[ 64, 8]
		if (N > 63) { field_batchnorm_affined_gradients_v2_k4(stream, 4, 1, 2, 2, deltaY, X, X_mean, X_square_mean, eps, N, M, deltaXp1, deltaXp2, deltaA, deltaB, width, stride); return; }//[ 32, 8]
		if (N > 31) { field_batchnorm_affined_gradients_v2_k4(stream, 4, 1, 1, 2, deltaY, X, X_mean, X_square_mean, eps, N, M, deltaXp1, deltaXp2, deltaA, deltaB, width, stride); return; }//[ 16, 8]
	}
	field_batchnorm_affined_gradients_v2_k4_small(stream, deltaY, X, X_mean, X_square_mean, eps, N, M, deltaXp1, deltaXp2, deltaA, deltaB, width, stride);
}

#endif 


#ifndef FIELD_BATCH_NORM_AFFINED_GRADIENTS_V2
#define FIELD_BATCH_NORM_AFFINED_GRADIENTS_V2

//(correct)
int __field_batchnorm_affined_gradients_v2(JNIEnv *env,
	cudaStream_t stream1, cudaStream_t stream2,
	cudaStream_t stream3, cudaStream_t stream4,
	const float* deltaY, const float* X,
	const float* X_mean,
	const float* X_square_mean, float eps,
	int N, int M,
	float* deltaXp1,
	float* deltaXp2,
	float* deltaA_buf, float* deltaA,
	float* deltaB_buf, float* deltaB,
	int width, int stride,
	int partNum)
{
	int nextN = field_nextN(N, M);//N <= HV
	if (nextN <= partNum) {//V.address = NULL, only 1 stage, write result to Y
		__field_batchnorm_affined_gradients_v2_stage(stream1,
			deltaY, X, X_mean, X_square_mean, eps, N, M,
			deltaXp1,
			deltaXp2,
			deltaA,
			deltaB,
			width, stride);
		return nextN;
	}

	//V.address != NULL, at least 2 stages
	__field_batchnorm_affined_gradients_v2_stage(stream1,
		deltaY, X, X_mean, X_square_mean, eps, N, M,
		deltaXp1,
		deltaXp2,
		deltaA_buf,
		deltaB_buf,
		width, stride);

	//====stream{2, 3, 4} shold wait stream1===============================================
	cudaEvent_t event; cudaError_t error;
	error = cudaEventCreate(&event, cudaEventDisableTiming); handleError(error);
	error = cudaEventRecord(event, stream1); handleError(error);
	error = cudaStreamWaitEvent(stream2, event, cudaEventWaitDefault); handleError(error);
	error = cudaStreamWaitEvent(stream3, event, cudaEventWaitDefault); handleError(error);
	error = cudaStreamWaitEvent(stream4, event, cudaEventWaitDefault); handleError(error);
	//====stream{2, 3, 4} shold wait stream1===============================================

	N = nextN, nextN = field_nextN(N, M);//N % 4 == 0, N >= 4
	while (nextN > partNum) {//end: nextN <= partNum
		__field_sum_stage(stream1, deltaXp1, N, M, deltaXp1, width, stride);
		__field_sum_stage(stream2, deltaXp2, N, M, deltaXp2, width, stride);
		__field_sum_stage(stream3, deltaA_buf, N, M, deltaA_buf, width, stride);
		__field_sum_stage(stream4, deltaB_buf, N, M, deltaB_buf, width, stride);
		N = nextN, nextN = field_nextN(nextN, M);
	}

	__field_sum_stage(stream1, deltaXp1, N, M, deltaXp1, width, stride);//the last stage
	__field_sum_stage(stream2, deltaXp2, N, M, deltaXp2, width, stride);//the last stage
	__field_sum_stage(stream3, deltaA_buf, N, M, deltaA, width, stride);//the last stage
	__field_sum_stage(stream4, deltaB_buf, N, M, deltaB, width, stride);//the last stage

	error = cudaEventDestroy(event); handleError(error);
	return nextN;
}

#endif

#endif