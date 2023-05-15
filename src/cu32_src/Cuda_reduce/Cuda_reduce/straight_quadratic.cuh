#pragma once

#ifndef STRAIGHT_QUADRATIC_H
#define STRAIGHT_QUADRATIC_H

//index % 4 == 0
//lengthv % 4(stride) == 0
#ifndef STRAIGHT_QUADRATIC_CALL
#define STRAIGHT_QUADRATIC_CALL

#define straight_quadratic4(stream, LB, LT, X, alpha, beta, gamma, lengthv, V, width, stride) \
	straight_quadratic_kernel_4<LB>\
		<<< (lengthv >> LB >> LT), (1<<LB), 0, stream >>>\
			(X, alpha, beta, gamma, lengthv, V, width, stride);

//2 << 7 = 128, for lengthv <= 128, 128 >> 2 = 32
#define straight_quadratic4_small(stream, X, alpha, beta, gamma, lengthv, V, width, stride) \
	straight_quadratic_kernel_4<5>\
		<<< 1, 32, 0, stream>>> \
			(X, alpha, beta, gamma, lengthv, V, width, stride);

#endif


#ifndef STRAIGHT_QUADRATIC_KERNEL_4
#define STRAIGHT_QUADRATIC_KERNEL_4

template<int LB>
__global__ void straight_quadratic_kernel_4(
	const float* __restrict__ X,
	float alpha, float beta, float gamma,
	int lengthv,
	float* __restrict__ V,
	int width, int stride)
{
	__shared__ float As[1 << LB];

	int bx = blockIdx.x, tx = threadIdx.x;
	int index = (bx << LB) + tx;
	int step = (gridDim.x << LB), step4 = step << 2;

	float v = 0.0f;
	float c = 0.0f;//solve the error
	for (int index4 = index << 2; index4 < lengthv; index4 += step4) 
	{
		float4 a = *(float4*)(X + index4);
		simdQuadratic4(a, alpha, a, beta, gamma);
		within_width4(a, index4, stride, width);

		float dv = (a.x + a.y + a.z + a.w) - c;
		float t = v + dv;
		c = (t - v) - dv;
		v = t;//v += a.x + a.y + a.z + a.w;
	}
	As[tx] = v;
	__syncthreads();

	if (LB >= 7) {
		if (tx < 64) As[tx] += As[tx + 64];
		__syncthreads();
	}
	if (LB >= 6) {
		if (tx < 32) As[tx] += As[tx + 32];
		__syncthreads();
	}
	if (LB >= 5) {
		if (tx < 16) As[tx] += As[tx + 16];
		__syncthreads();
	}
	if (tx < 8) warp_sum_8(As, tx);//LBX >= 4
	if (tx == 0) V[bx] = As[0];
}

#endif


#ifndef STRAIGHT_QUADRATIC_STAGE
#define STRAIGHT_QUADRATIC_STAGE

void __straight_quadratic_stage(cudaStream_t stream,
	const float *X,
	float alpha, float beta, float gamma,
	int lengthv,
	float *V, int width, int stride)
{
	if (lengthv <   128) { straight_quadratic4_small(stream, X, alpha, beta, gamma, lengthv, V, width, stride); return;}
	if (lengthv >= 8192) { straight_quadratic4(stream, 6, 7, X, alpha, beta, gamma, lengthv, V, width, stride); return; }
	if (lengthv >= 4096) { straight_quadratic4(stream, 6, 6, X, alpha, beta, gamma, lengthv, V, width, stride); return; }
	if (lengthv >= 2048) { straight_quadratic4(stream, 6, 5, X, alpha, beta, gamma, lengthv, V, width, stride); return; }
	if (lengthv >= 1024) { straight_quadratic4(stream, 6, 4, X, alpha, beta, gamma, lengthv, V, width, stride); return; }
	if (lengthv >=  512) { straight_quadratic4(stream, 6, 3, X, alpha, beta, gamma, lengthv, V, width, stride); return; }
	if (lengthv >=  256) { straight_quadratic4(stream, 6, 2, X, alpha, beta, gamma, lengthv, V, width, stride); return; }
	straight_quadratic4(stream, 5, 2, X, alpha, beta, gamma, lengthv, V, width, stride);//lengthv >= 128
}

#endif


#ifndef STRAIGHT_QUADRATIC
#define STRAIGHT_QUADRATIC

int __straight_quadratic(cudaStream_t stream,
	const float *X,
	float alpha, float beta, float gamma,
	int lengthv,
	float *V, int width, int stride,
	int partNum)
{
	__straight_quadratic_stage(stream, X, alpha, beta, gamma, lengthv, V, width, stride);
	lengthv = straight_nextLengthV(lengthv);

	while(lengthv > partNum) {
		__straight_sum_stage(stream, V, lengthv, V, lengthv, ((lengthv + 3) >> 2) << 2);
		lengthv = straight_nextLengthV(lengthv);
	}
	return lengthv;
}

#endif

#endif
