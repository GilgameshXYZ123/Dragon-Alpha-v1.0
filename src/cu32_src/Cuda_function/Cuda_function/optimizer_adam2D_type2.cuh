#pragma once

#ifndef ADAM_2D_H_TYPE2
#define ADAM_2D_H_TYPE2

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef ADAM_2D_TYPE2_CALL
#define ADAM_2D_TYPE2_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define adam2d_k4_type2(stream, LB, LT, W, V, a1, a2, Uv, S, b1, b2, eps, Us, deltaW, lr, lengthv, width, stride)\
	adam2D_kernel_4_type2\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(W, V,a1,a2,Uv, S,b1,b2,eps,Us, deltaW,lr, lengthv,width,stride)

#define adam2d_k4_small_type2(stream, W, V, a1, a2, Uv, S, b1, b2, eps, Us, deltaW, lr, lengthv, width, stride)\
	adam2D_kernel_4_type2\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(W, V,a1,a2,Uv, S,b1,b2,eps,Us, deltaW,lr, lengthv,width,stride)

#endif


#ifndef ADAM_2D_TYPE2_KERNEL
#define ADAM_2D_TYPE2_KERNEL

__global__ void adam2D_kernel_4_type2(
	const float* __restrict__ W,
	float* __restrict__ V, float a1, float a2, float Uv, 
	float* __restrict__ S, float b1, float b2, float eps, float Us,
	const float* __restrict__ deltaW, float lr,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	Us = rsqrtf(1.0f - Us);
	lr = lr / (1.0f - Uv);
	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 dwf = *(float4*)(&deltaW[index4]); double4 dw; COPY4(dw, dwf);
		float4 vf = *(float4*)(&V[index4]); double4 v; COPY4(v, vf);
		float4 sf = *(float4*)(&S[index4]); double4 s; COPY4(s, sf);
		float4 w = *(float4*)(&W[index4]);//W = W - lr_t * V/(sqrt(S) + eps)

		//V = a1 * V + a2 * deltaW
		v.x = a1 * v.x + a2 * dw.x;
		v.y = a1 * v.y + a2 * dw.y;
		v.z = a1 * v.z + a2 * dw.z;
		v.w = a1 * v.w + a2 * dw.w;

		//S = b1 * S + b2 * deltaW^2
		s.x = b1 * s.x + b2 * (dw.x * dw.x);
		s.y = b1 * s.y + b2 * (dw.y * dw.y);
		s.z = b1 * s.z + b2 * (dw.z * dw.z);
		s.w = b1 * s.w + b2 * (dw.w * dw.w);

		//denom = sqrt(s) / sqrt(1 - Us) + eps
		double4 dnorm;
		dnorm.x = sqrtf(s.x)*Us + eps;
		dnorm.y = sqrtf(s.y)*Us + eps;
		dnorm.z = sqrtf(s.z)*Us + eps;
		dnorm.w = sqrtf(s.w)*Us + eps;

		w.x -= lr * v.x / dnorm.x;
		w.y -= lr * v.y / dnorm.y;
		w.z -= lr * v.z / dnorm.z;
		w.w -= lr * v.w / dnorm.w;

		within_width(s, index4, stride, width);
		within_width(v, index4, stride, width);
		within_width(w, index4, stride, width);
		COPY4(vf, v); *(float4*)(V + index4) = vf;//update the standard deviation
		COPY4(sf, s); *(float4*)(S + index4) = sf;//update the velocity
		*(float4*)(&W[index4]) = w;//update weight
	}
}

#endif


void __adam2D_type2(cudaStream_t stream,
	const float* W,
	float* V, float a1, float a2, float Uv,
	float* S, float b1, float b2, float eps, float Us,
	const float* deltaW, float lr,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { adam2d_k4_small_type2(stream, W, V, a1, a2, Uv, S, b1, b2, eps, Us, deltaW, lr, lengthv, width, stride); return; }
	adam2d_k4_type2(stream, 5, 2, W, V, a1, a2, Uv, S, b1, b2, eps, Us, deltaW, lr, lengthv, width, stride);
}


#endif