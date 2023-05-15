#pragma once

#ifndef FIELD_QUADRATIC_DUAL_H
#define FIELD_QUADRATIC_DUAL_H

//(1) field_num % A.width ==0
//(2) field_num -> field_stride = field_num / width * stride
//(3) A.length % feature_num == 0
//(4) A.lengthv % fieled_stride == 0
//(5) row_num = A.lengthv / field_stride
//(6) [y, x] -> [row_num, field_stride]
//(7) [N, M] = [row_num, field_stride], A.lengthv = N*M
#ifndef FIELD_QUADRATIC_DUAL_CALL
#define FIELD_QUADRATIC_DUAL_CALL

//LTX >=2
#define field_quadratic_dual4(stream, LBY, LBX, LTY, LTX, X1, X2, k11, k12, k22, k1, k2, C, N, M, V, width, stride) \
	field_quadratic_dual_kernel_4<LBY, LBX>\
		<<< dim3(M>>LBX>>LTX, N>>LBY>>LTY), dim3(1<<LBX, 1<<LBY), 0, stream >>>\
			(X1, X2, k11, k12, k22, k1, k2, C, N, M, V, width, stride)

//[16, 4]
#define field_quadratic_dual4_small(stream, X1, X2, k11, k12, k22, k1, k2, C, N, M, V, width, stride) \
	field_quadratic_dual_kernel_4<3, 3>\
		<<< dim3((M + 31)>>5, (N + 63)>>6), dim3(8, 8), 0, stream>>>\
			(X1, X2, k11, k12, k22, k1, k2, C, N, M, V, width, stride)

#endif


#ifndef FIELD_QUADRATIC_DUAL_KERNEL_4
#define FIELD_QUADRATIC_DUAL_KERNEL_4

//<3, 2, 3, 4>: Time = 0.047     mesc, Speed = 85.7089GB/s
//<3, 2, 2, 4>: Time = 0.0506667 mesc, Speed = 81.9156GB/s
//<3, 2, 1, 4>: Time = 0.075     mesc, Speed = 58.5937GB/s
//<3, 2, 2, 3>: Time = 0.049     mesc, Speed = 84.7019GB/s
//<3, 2, 4, 2>: Time = 0.0463333 mesc, Speed = 85.6249GB/s
template<int LBY, int LBX>
__global__ void field_quadratic_dual_kernel_4(
	const float* __restrict__ X1,
	const float* __restrict__ X2,
	float k11, float k12, float k22,
	float k1, float k2, float C,
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
		float4 c = make_float4(0, 0, 0, 0);//solve the errors
		float4 v = make_float4(0, 0, 0, 0);//thread reduce: 4 local result
		for (int y = offsetY; y < N; y += stepY)
		{
			float4 x1 = *(float4*)(&X1[y*M + x4]);
			float4 x2 = *(float4*)(&X2[y*M + x4]);

			float4 dv;
			dv.x = k11 * (x1.x * x1.x) + k12 * (x1.x * x2.x) + k22 * (x2.x * x2.x) + k1 * x1.x + k2 * x2.x + C - c.x;
			dv.y = k11 * (x1.y * x1.y) + k12 * (x1.y * x2.y) + k22 * (x2.y * x2.y) + k1 * x1.y + k2 * x2.y + C - c.y;
			dv.z = k11 * (x1.z * x1.z) + k12 * (x1.z * x2.z) + k22 * (x2.z * x2.z) + k1 * x1.z + k2 * x2.z + C - c.z;
			dv.w = k11 * (x1.w * x1.w) + k12 * (x1.w * x2.w) + k22 * (x2.w * x2.w) + k1 * x1.w + k2 * x2.w + C - c.w;

			float4 t;
			t.x = v.x + dv.x;
			t.y = v.y + dv.y;
			t.z = v.z + dv.z;
			t.w = v.w + dv.w;

			c.x = (t.x - v.x) - dv.x;
			c.y = (t.y - v.y) - dv.y;
			c.z = (t.z - v.z) - dv.z;
			c.w = (t.w - v.w) - dv.w;

			v = t;//simdAdd4(v, v, dv);
		}
		*(float4*)(&As[As_xy]) = v;
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


#ifndef FIELD_QUADRATIC_DUAL_STAGE
#define FIELD_QUADRATIC_DUAL_STAGE

//M % 4 == 0, M >= 4
void __field_quadratic_dual_stage(cudaStream_t stream,
	const float* X1, const float *X2,
	float k11, float k12, float k22,
	float k1, float k2, float C,
	int N, int M,
	float*  V, int width, int stride)
{
	if (M > 15) {
		if (N > 63) { field_quadratic_dual4(stream, 3, 2, 3, 2, X1, X2, k11, k12, k22, k1, k2, C, N, M, V, width, stride); return; }//[64, 16]
		if (N > 31) { field_quadratic_dual4(stream, 3, 2, 2, 2, X1, X2, k11, k12, k22, k1, k2, C, N, M, V, width, stride); return; }//[32, 16]
		if (N > 15) { field_quadratic_dual4(stream, 3, 2, 1, 2, X1, X2, k11, k12, k22, k1, k2, C, N, M, V, width, stride); return; }//[16, 16]
	}
	if (M > 7) {
		if (N > 127) { field_quadratic_dual4(stream, 4, 1, 3, 2, X1, X2, k11, k12, k22, k1, k2, C, N, M, V, width, stride); return; }//[ 64, 8]
		if (N > 63) { field_quadratic_dual4(stream, 4, 1, 2, 2, X1, X2, k11, k12, k22, k1, k2, C, N, M, V, width, stride); return; }//[ 32, 8]
		if (N > 31) { field_quadratic_dual4(stream, 4, 1, 1, 2, X1, X2, k11, k12, k22, k1, k2, C, N, M, V, width, stride); return; }//[ 16, 8]
	}
	field_quadratic_dual4_small(stream, X1, X2, k11, k12, k22, k1, k2, C, N, M, V, width, stride);
}

#endif 


#ifndef FIELD_QUADRATIC_DUAL
#define FIELD_QUADRATIC_DUAL

//new fashion
int __field_quadratic_dual(cudaStream_t stream,
	const float* X1, const float* X2, 
	float k11, float k12, float k22,
	float k1, float k2, float C,
	int N, int M,
	float*  V, float *Y,
	int width, int stride,
	int partNum)
{
	int nextN = field_nextN(N, M);//N <= HV
	if (nextN <= partNum) {//V.address = NULL, only 1 stage, write result to Y
		__field_quadratic_dual_stage(stream, X1, X2, k11, k12, k22, k1, k2, C, N, M, Y, width, stride);
		return nextN;
	}

	//V.address != NULL, at least 2 stages
	__field_quadratic_dual_stage(stream, X1, X2, k11, k12, k22, k1, k2, C, N, M, V, width, stride);

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