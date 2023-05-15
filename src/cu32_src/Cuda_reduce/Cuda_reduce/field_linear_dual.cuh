#pragma once

#ifndef FIELD_LINEAR_DUAL_H
#define FIELD_LINEAR_DUAL_H

//(1) field_num % A.width ==0
//(2) field_num -> field_stride = field_num / width * stride
//(3) A.length % feature_num == 0
//(4) A.lengthv % fieled_stride == 0
//(5) row_num = A.lengthv / field_stride
//(6) [y, x] -> [row_num, field_stride]
//(7) [N, M] = [row_num, field_stride], A.lengthv = N*M
#ifndef FIELD_LINEAR_DUAL_CALL
#define FIELD_LINEAR_DUAL_CALL

//LTX >=2
#define field_linear_dual4(stream, LBY, LBX, LTY, LTX, X1, X2, alpha, beta, gamma, N, M, V, width, stride) \
	field_linear_dual_kernel_4<LBY, LBX>\
		<<< dim3(M>>LBX>>LTX, N>>LBY>>LTY), dim3(1<<LBX, 1<<LBY), 0, stream >>>\
			(X1, X2, alpha, beta, gamma, N, M, V, width, stride)

//[16, 4]
#define field_linear_dual4_small(stream, X1, X2, alpha, beta, gamma, N, M, V, width, stride) \
	field_linear_dual_kernel_4<3, 3>\
		<<< dim3((M + 31)>>5, (N + 63)>>6), dim3(8, 8), 0, stream>>>\
			(X1, X2, alpha, beta, gamma, N, M, V, width, stride)

#endif


#ifndef FIELD_LINEAR_DUAL_KERNEL_4
#define FIELD_LINEAR_DUAL_KERNEL_4

//V = field_sum: alpha*X1 + beta*X2 + gamma
template<int LBY, int LBX>
__global__ void field_linear_dual_kernel_4(
	const float* __restrict__ X1,
	const float* __restrict__ X2,
	float alpha, float beta, float gamma,
	int N, int M,
	float* __restrict__ V,
	int width, int stride)
{
	int by = blockIdx.y, bx = blockIdx.x;
	int ty = threadIdx.y, tx = threadIdx.x;

	__shared__ float2 As[2 << LBX << LBY];//[BX, 2*BY]
	const int As_xy = ((tx << LBY) + ty) << 1;//(tx, ty*2)

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
			int y_x4 = y * M + x4;
			float4 x1 = *(float4*)(X1 + y_x4);//X1[y, x4]
			float4 x2 = *(float4*)(X2 + y_x4);//X2[y, x4]

			float4 a;//a = alpha*x1 + beta*x2 + gamma
			a.x = alpha * x1.x + beta * x2.x + gamma;
			a.y = alpha * x1.y + beta * x2.y + gamma;
			a.z = alpha * x1.z + beta * x2.z + gamma;
			a.w = alpha * x1.w + beta * x2.w + gamma;

			float4 dv;//Kahan simdAdd4(v, v, a) 
			dv.x = a.x - c.x;
			dv.y = a.y - c.y;
			dv.z = a.z - c.z;
			dv.w = a.w - c.w;

			float4 t;
			t.x = v.x + dv.x;
			t.y = v.y + dv.y;
			t.z = v.z + dv.z;
			t.w = v.w + dv.w;

			c.x = (t.x - v.x) - dv.x;
			c.y = (t.y - v.y) - dv.y;
			c.z = (t.z - v.z) - dv.z;
			c.w = (t.w - v.w) - dv.w;

			v = t;//Kahan simdAdd4(v, v, a) 
		}
		*(float4*)(As + As_xy) = v;
		__syncthreads();

		int As_index;//block reduce: 4 global result
		if (LBY >= 6) {
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
			*(float2*)(&get(V, by, xindex2, M)) = result;
		}
		__syncthreads();
	}
}

#endif


#ifndef FIELD_LINEAR_DUAL_STAGE
#define FIELD_LINEAR_DUAL_STAGE

//M % 4 == 0, M >= 4
void __field_linear_dual_stage(cudaStream_t stream,
	const float* X1, const float* X2,
	float alpha, float beta, float gamma,
	int N, int M,
	float*  V, int width, int stride)
{
	if (M > 15) {
		if (N > 63) { field_linear_dual4(stream, 3, 2, 3, 2, X1, X2, alpha, beta, gamma, N, M, V, width, stride); return; }//[64, 16]
		if (N > 31) { field_linear_dual4(stream, 3, 2, 2, 2, X1, X2, alpha, beta, gamma, N, M, V, width, stride); return; }//[32, 16]
		if (N > 15) { field_linear_dual4(stream, 3, 2, 1, 2, X1, X2, alpha, beta, gamma, N, M, V, width, stride); return; }//[16, 16]
	}
	if (M > 7) {
		if (N > 127) { field_linear_dual4(stream, 4, 1, 3, 2, X1, X2, alpha, beta, gamma, N, M, V, width, stride); return; }//[ 64, 8]
		if (N >  63) { field_linear_dual4(stream, 4, 1, 2, 2, X1, X2, alpha, beta, gamma, N, M, V, width, stride); return; }//[ 32, 8]
		if (N >  31) { field_linear_dual4(stream, 4, 1, 1, 2, X1, X2, alpha, beta, gamma, N, M, V, width, stride); return; }//[ 16, 8]
	}
	field_linear_dual4_small(stream, X1, X2, alpha, beta, gamma, N, M, V, width, stride);
}

#endif 


#ifndef FIELD_LINEAR_DUAL
#define FIELD_LINEAR_DUAL

//new fashion
int __field_linear_dual(cudaStream_t stream,
	const float* X1, const float* X2,
	float alpha, float beta, float gamma,
	int N, int M,
	float*  V, float *Y,
	int width, int stride,
	int partNum)
{
	int nextN = field_nextN(N, M);//N <= HV
	if (nextN <= partNum) {//V.address = NULL, only 1 stage, write result to Y
		__field_linear_dual_stage(stream, X1, X2, alpha, beta, gamma, N, M, Y, width, stride);
		return nextN;
	}

	//V.address != NULL, at least 2 stages
	__field_linear_dual_stage(stream, X1, X2, alpha, beta, gamma, N, M, V, width, stride);

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