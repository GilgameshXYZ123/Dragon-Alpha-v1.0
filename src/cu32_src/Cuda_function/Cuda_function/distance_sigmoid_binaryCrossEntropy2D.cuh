#pragma once

#ifndef SIGMOID_BINARY_CROSS_ENTROPY_2D_H
#define SIGMOID_BINARY_CROSS_ENTROPY_2D_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef SIGMOID_BINARY_CROSS_ENTROPY_2D_CALL
#define SIGMOID_BINARY_CROSS_ENTROPY_2D_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define sigmoid_binaryCrossEntropy_2d_k4(stream, LB, LT, Y, X, alpha, beta, L, lengthv, width, stride)\
	sigmoid_binaryCrossEntropy_2D_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(Y, X, alpha, beta, L, lengthv, width, stride)

#define sigmoid_binaryCrossEntropy_2d_k4_small(stream, Y, X, alpha, beta, L, lengthv, width, stride)\
	sigmoid_binaryCrossEntropy_2D_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(Y, X, alpha, beta, L, lengthv, width, stride)

#endif


#ifndef SIGMOID_BINARY_CROSS_ENTROPY_2D_KERNEL
#define SIGMOID_BINARY_CROSS_ENTROPY_2D_KERNEL

//[1] Yh = sigmoid(X) = exp(X) / (1 + exp(X))
//[2] L = binaryCrossEntropy(Y, Yh) 
//      = -alpha*Y*log(Yh) + beta*(Y - 1)*log(1 - Yh)
//We have: 
//L = -alpha * Y * log[exp(X) / (1 + exp(X)] + 
//    beta * (Y - 1) * log[1 / (1+exp(X))]
//L = alpha * Y * {log[1 + exp(X)] - X}
//    beta * (1 - Y) * log[(1+exp(X))]
//L = -alpha * Y * X + {alpha*Y + beta*(1 - Y)} * log(1 + exp(X))
//L = -alpha * Y * X + {(alpha - beta)*Y + beta} * log(1 + exp(X))
//when: alpha = beta = 1
//L = -Y * X + log[1 + exp(X)]

__global__ void sigmoid_binaryCrossEntropy_2D_kernel_4(
	const float* __restrict__ Y,
	const float* __restrict__ X,
	float alpha, float beta,
	float* __restrict__ L,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	
	float gamma = alpha - beta;
	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 x = *(float4*)(X + index4);
		float4 y = *(float4*)(Y + index4);
		
		float4 loss;//L = -alpha * Y * X + {(alpha - beta)*Y + beta} * log(1 + exp(X))
		loss.x = -alpha * (y.x * x.x) + (gamma*y.x + beta) * logf(1.0f + expf(x.x));
		loss.y = -alpha * (y.y * x.y) + (gamma*y.y + beta) * logf(1.0f + expf(x.y));
		loss.z = -alpha * (y.z * x.z) + (gamma*y.z + beta) * logf(1.0f + expf(x.z));
		loss.w = -alpha * (y.w * x.w) + (gamma*y.w + beta) * logf(1.0f + expf(x.w));

		within_width(loss, index4, stride, width);
		*(float4*)(L + index4) = loss;
	}
}

#endif


void __sigmoid_binaryCrossEntropy_2D(cudaStream_t stream,
	const float* Y, const float* X, 
	float alpha, float beta,
	float* L,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { sigmoid_binaryCrossEntropy_2d_k4_small(stream, Y, X, alpha, beta, L, lengthv, width, stride); return; }
	sigmoid_binaryCrossEntropy_2d_k4(stream, 5, 2, Y, X, alpha, beta, L, lengthv, width, stride);
}

#endif