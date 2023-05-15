#pragma once

#ifndef FIELD_LAYER_NORM_DELTAA_V2_H
#define FIELD_LAYER_NORM_DELTAA_V2_H

//(1) field_num % A.width ==0
//(2) field_num -> field_stride = field_num / width * stride
//(3) A.length % feature_num == 0
//(4) A.lengthv % fieled_stride == 0
//(5) row_num = A.lengthv / field_stride
//(6) [y, x] -> [row_num, field_stride]
//(7) [N, M] = [row_num, field_stride], A.lengthv = N*M
//V2: holdX(), X is not changed
#ifndef FIELD_LAYER_NORM_DELTAA_V2_CALL
#define FIELD_LAYER_NORM_DELTAA_V2_CALL

//LTX >=2
#define field_layernorm_deltaA_v2_k4(stream, LBY, LBX, LTY, LTX, deltaY, X, X_mean, X_square_mean, eps, N, M, deltaA, width, stride) \
	field_layernorm_deltaA_v2_kernel_4<LBY, LBX>\
		<<< dim3(M>>LBX>>LTX, N>>LBY>>LTY), dim3(1<<LBX, 1<<LBY), 0, stream >>>\
			(deltaY, X, X_mean, X_square_mean, eps, N, M, deltaA, width, stride)

//[16, 4]
#define field_layernorm_deltaA_v2_k4_small(stream, deltaY, X, X_mean, X_square_mean, eps, N, M, deltaA, width, stride) \
	field_layernorm_deltaA_v2_kernel_4<3, 3>\
		<<< dim3((M + 31)>>5, (N + 63)>>6), dim3(8, 8), 0, stream>>>\
			(deltaY, X, X_mean, X_square_mean, eps, N, M, deltaA, width, stride)

#endif


#ifndef FIELD_LAYER_NORM_DELTAA_V2_KERNEL_4
#define FIELD_LAYER_NORM_DELTAA_V2_KERNEL_4

//X_mean & X_square_mean: each row
//Y = X_norm * A + B => X_norm = (Y - B)/A
//for each row:
//[V1] deltaA = field_sum: deltaY[i] * (Y[i] - B)/A
//[V2] deltaA = field_sum: deltaY[i] * (X[i] - X_mean) / (sqrt(X_var) + eps)

template<int LBY, int LBX>
__global__ void field_layernorm_deltaA_v2_kernel_4(
	const float* __restrict__ deltaY,
	const float* __restrict__ X,
	const float* __restrict__ X_mean,
	const float* __restrict__ X_square_mean, float eps,
	int N, int M, 
	float* __restrict__ deltaA,
	int width, int stride)
{
	int by = blockIdx.y, bx = blockIdx.x;
	int ty = threadIdx.y, tx = threadIdx.x;

	__shared__ float2 As[2 << LBX << LBY];//[BX, 2*BY]
	const int As_xy = ((tx << LBY) + ty) << 1;//[tx, ty*2]

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
			//stddev^2 = (square_mean - mean*mean)
			float x_mean = X_mean[y];
			float x_smean = X_square_mean[y];
			float x_rstd = rsqrtf(x_smean - x_mean * x_mean + eps);

			int y_x4 = y * M + x4;//(y, x4)
			float4 dy = *(float4*)(deltaY + y_x4);
			float4 x = *(float4*)(X + y_x4);

			float4 da;//deltaA = deltaY[i] * (X[i] - X_mean)/ sqrt(X_var + eps)
			da.x = dy.x * (x.x - x_mean) * x_rstd;
			da.y = dy.y * (x.y - x_mean) * x_rstd;
			da.z = dy.z * (x.z - x_mean) * x_rstd;
			da.w = dy.w * (x.w - x_mean) * x_rstd;

			float4 dv;//Kahan: simdAdd4(v, v, da);
			dv.x = da.x - c.x;
			dv.y = da.y - c.y;
			dv.z = da.z - c.z;
			dv.w = da.w - c.w;

			float4 t;
			t.x = v.x + dv.x;
			t.y = v.y + dv.y;
			t.z = v.z + dv.z;
			t.w = v.w + dv.w;

			c.x = (t.x - v.x) - dv.x;
			c.y = (t.y - v.y) - dv.y;
			c.z = (t.z - v.z) - dv.z;
			c.w = (t.w - v.w) - dv.w;

			v = t;//Kahan: simdAdd4(v, v, da);
		}
		*(float4*)(As + As_xy) = v;
		__syncthreads();

		int As_index;
		if (LBY >= 6) {//block reduce: get 4 global result
			As_index = (((tx << LBY) + (ty & 31)) << 1) + (ty >> 5);
			if (ty < 64) { simdAdd2(As[As_index], As[As_index], As[As_index + 64]); }
			__syncthreads();
		}
		if (LBY >= 5) {
			As_index = (((tx << LBY) + (ty & 15)) << 1) + (ty >> 4);
			if (ty < 32) { simdAdd2(As[As_index], As[As_index], As[As_index + 32]); }
			__syncthreads();
		}
		if (LBY >= 4) {
			As_index = (((tx << LBY) + (ty & 7)) << 1) + (ty >> 3);
			if (ty < 16) { simdAdd2(As[As_index], As[As_index], As[As_index + 16]); }
			__syncthreads();
		}

		As_index = (((tx << LBY) + (ty & 3)) << 1) + (ty >> 2);
		if (ty < 8) { simdAdd2(As[As_index], As[As_index], As[As_index + 8]); }
		__syncthreads();

		As_index = (((tx << LBY) + (ty & 1)) << 1) + (ty >> 1);
		if (ty < 4) { simdAdd2(As[As_index], As[As_index], As[As_index + 4]); }
		__syncthreads();

		As_index = (((tx << LBY) + (ty & 0)) << 1) + ty;
		if (ty < 2) { simdAdd2(As[As_index], As[As_index], As[As_index + 2]); }
		__syncthreads();

		if (ty < 2) {
			float2 result = As[As_index];
			int xindex2 = x4 + (ty << 1);
			within_width2(result, xindex2, stride, width);
			*(float2*)(&get(deltaA, by, xindex2, M)) = result;
		}
		__syncthreads();
	}
}

#endif


#ifndef FIELD_LAYER_NORM_DELTAA_V2_STAGE
#define FIELD_LAYER_NORM_DELTAA_V2_STAGE

//M % 4 == 0, M >= 4
void __field_layernorm_deltaA_v2_stage(cudaStream_t stream,
	const float* deltaY, const float* X,
	const float* X_mean,
	const float* X_square_mean, float eps,
	int N, int M, 
	float* deltaA, int width, int stride)
{
	if (M > 15) {
		if (N > 63) { field_layernorm_deltaA_v2_k4(stream, 3, 2, 3, 2, deltaY, X, X_mean, X_square_mean, eps, N, M, deltaA, width, stride); return; }//[64, 16]
		if (N > 31) { field_layernorm_deltaA_v2_k4(stream, 3, 2, 2, 2, deltaY, X, X_mean, X_square_mean, eps, N, M, deltaA, width, stride); return; }//[32, 16]
		if (N > 15) { field_layernorm_deltaA_v2_k4(stream, 3, 2, 1, 2, deltaY, X, X_mean, X_square_mean, eps, N, M, deltaA, width, stride); return; }//[16, 16]
	}
	if (M > 7) {
		if (N > 127) { field_layernorm_deltaA_v2_k4(stream, 4, 1, 3, 2, deltaY, X, X_mean, X_square_mean, eps, N, M, deltaA, width, stride); return; }//[ 64, 8]
		if (N > 63) { field_layernorm_deltaA_v2_k4(stream, 4, 1, 2, 2, deltaY, X, X_mean, X_square_mean, eps, N, M, deltaA, width, stride); return; }//[ 32, 8]
		if (N > 31) { field_layernorm_deltaA_v2_k4(stream, 4, 1, 1, 2, deltaY, X, X_mean, X_square_mean, eps, N, M, deltaA, width, stride); return; }//[ 16, 8]
	}
	field_layernorm_deltaA_v2_k4_small(stream, deltaY, X, X_mean, X_square_mean, eps, N, M, deltaA, width, stride);
}

#endif 


#ifndef FIELD_LAYER_NORM_DELTAA_V2
#define FIELD_LAYER_NORM_DELTAA_V2

int __field_layernorm_deltaA_v2(cudaStream_t stream,
	const float* deltaY, const float* X,
	const float* X_mean,
	const float* X_square_mean, float eps,
	int N, int M, 
	float* deltaA_buf, float* deltaA,
	int width, int stride,
	int partNum)
{
	int nextN = field_nextN(N, M);//N <= HV
	if (nextN <= partNum) {//V.address = NULL, only 1 stage, write result to Y
		__field_layernorm_deltaA_v2_stage(stream, deltaY, X, X_mean, X_square_mean, eps, N, M, deltaA, width, stride);
		return nextN;
	}

	//V.address != NULL, at least 2 stages
	__field_layernorm_deltaA_v2_stage(stream, deltaY, X, X_mean, X_square_mean, eps, N, M, deltaA_buf, width, stride);

	N = nextN, nextN = field_nextN(N, M);//N % 4 == 0, N >= 4
	while (nextN > partNum)//end: nextN <= partNum
	{
		__field_sum_stage(stream, deltaA_buf, N, M, deltaA_buf, width, stride);
		N = nextN, nextN = field_nextN(nextN, M);
	}
	__field_sum_stage(stream, deltaA_buf, N, M, deltaA, width, stride);//the last stage
	return nextN;
}

#endif

#endif