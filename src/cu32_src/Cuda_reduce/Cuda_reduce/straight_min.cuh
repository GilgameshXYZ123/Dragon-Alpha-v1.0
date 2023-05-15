#pragma once

#ifndef STRAIGHT_MIN_H
#define STRAIGHT_MIN_H

//index % 4 == 0
//lengthv % 4(stride) == 0
#ifndef STRAIGHT_MIN_CALL
#define STRAIGHT_MIN_CALL

#define straight_min4(stream, LB, LT, X, lengthv, V, width, stride) \
	straight_min_kernel_4<LB>\
		<<< (lengthv >> LB >> LT), (1<<LB), 0, stream >>>\
			(X, lengthv, V, width, stride)

//2 << 7 = 128, for lengthv <= 128, 128 >> 2 = 32
#define straight_min4_small(stream, X, lengthv, V, width, stride) \
	straight_min_kernel_4<5>\
		<<< 1, 32, 0, stream>>> \
			(X, lengthv, V, width, stride)

#endif


#ifndef STRAIGHT_MIN_KERNEL_4
#define STRAIGHT_MIN_KERNEL_4

template<int LB>
__global__ void straight_min_kernel_4(
	const float* __restrict__ X, int lengthv,
	float* __restrict__ V,
	int width, int stride)
{
	__shared__ float As[1 << LB];

	int bx = blockIdx.x, tx = threadIdx.x;
	int index = (bx << LB) + tx;
	int step = (gridDim.x << LB), step4 = step << 2;

	float v = FLOAT_MAX;
	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 a = *(float4*)(X + index4);
		EXCEED_width4_TO_MAX(a, index4, stride, width);

		v = fminf(v, a.x);
		v = fminf(v, a.y);
		v = fminf(v, a.z);
		v = fminf(v, a.w);
	}
	As[tx] = v;
	__syncthreads();

	if (LB >= 7) {
		if (tx < 64) As[tx] = fminf(As[tx], As[tx + 64]);
		__syncthreads();
	}
	if (LB >= 6) {
		if (tx < 32) As[tx] = fminf(As[tx], As[tx + 32]);
		__syncthreads();
	}
	if (LB >= 5) {
		if (tx < 16) As[tx] = fminf(As[tx], As[tx + 16]);
		__syncthreads();
	}
	if (tx < 8) warp_min_8(As, tx);//LBX >= 4
	if (tx == 0) V[bx] = As[0];
}

#endif


#ifndef STRAIGHT_MIN_STAGE
#define STRAIGHT_MIN_STAGE

void __straight_min_stage(cudaStream_t stream,
	const float *X, int lengthv,
	float *V, int width, int stride)
{
	if (lengthv <   128) { straight_min4_small(stream, X, lengthv, V, width, stride); return; }
	if (lengthv >= 8192) { straight_min4(stream, 6, 7, X, lengthv, V, width, stride); return; }
	if (lengthv >= 4096) { straight_min4(stream, 6, 6, X, lengthv, V, width, stride); return; }
	if (lengthv >= 2048) { straight_min4(stream, 6, 5, X, lengthv, V, width, stride); return; }
	if (lengthv >= 1024) { straight_min4(stream, 6, 4, X, lengthv, V, width, stride); return; }
	if (lengthv >=  512) { straight_min4(stream, 6, 3, X, lengthv, V, width, stride); return; }
	if (lengthv >=  256) { straight_min4(stream, 6, 2, X, lengthv, V, width, stride); return; }
	straight_min4(stream, 5, 2, X, lengthv, V, width, stride);//lengthv >= 128
}

#endif


#ifndef STRAIGHT_MIN
#define STRAIGHT_MIN

int __straight_min(cudaStream_t stream,
	const float *X, int lengthv,
	float *V, int width, int stride,
	int partNum)
{
	__straight_min_stage(stream, X, lengthv, V, width, stride);
	lengthv = straight_nextLengthV(lengthv);

	while (lengthv > partNum) {
		__straight_min_stage(stream, V, lengthv, V, lengthv, ((lengthv + 3) >> 2) << 2);
		lengthv = straight_nextLengthV(lengthv);
	}
	return lengthv;
}

#endif

#endif
