#pragma once

#ifndef SGD_MN_2D_H
#define SGD_MN_2D_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
//(1) M: Momentum
//(2) N: Nesterov
#ifndef SGD_MN_2D_CALL
#define SGD_MN_2D_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define sgdmn2d_k4(stream, LB, LT, W, V, momentum, dampen, nesterov, deltaW, lr, lengthv, width, stride)\
	sgdmn2D_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(W, V, momentum, dampen, nesterov, deltaW, lr, lengthv, width, stride)

#define sgdmn2d_k4_small(stream, W, V, momentum, dampen, nesterov, deltaW, lr, lengthv, width, stride)\
	sgdmn2D_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(W, V, momentum, dampen, nesterov, deltaW, lr, lengthv, width, stride)

#endif


#ifndef SGD_MN_2D_KERNEL
#define SGD_MN_2D_KERNEL

//if (nestrov == 1): step = deltaW + momentum*V
//if (nestrov == 0): step = V;
//So: step = nesterov * deltaW + (nesterov * momentum + (1 - nesterov))*V 
//Step:
//<1> K = nesterov * momentum + 1 - nesterov
//<2> V = momentum*V + (1 - dampening)*deltaW
//<3> step = (nesterov * deltaW) + (K * V)
//<4> W = W - lr*step

__global__ void sgdmn2D_kernel_4(
	const float* __restrict__ W,
	float* __restrict__ V, 
	float momentum,
	float dampen,//to suppress the gradient
	float nesterov,//0 or 1
	const float* __restrict__ deltaW, float lr,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	float K = (nesterov * momentum) + (1.0f - nesterov);//<1>
	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		//load data------------------------------------------------
		float4 dw = *(float4*)(deltaW + index4);
		float4 v = *(float4*)(V + index4);
		float4 w = *(float4*)(W + index4);

		//compute result--------------------------------------------
		//<3> V = momentum*V + (1 - dampening)*deltaW
		v.x = (momentum * v.x) + dw.x - (dampen * dw.x);
		v.y = (momentum * v.y) + dw.y - (dampen * dw.y);
		v.z = (momentum * v.z) + dw.z - (dampen * dw.z);
		v.w = (momentum * v.w) + dw.w - (dampen * dw.w);

		float4 step;//<4> step = nesterov * deltaW + K * V
		step.x = (nesterov * dw.x) + (K * v.x);
		step.y = (nesterov * dw.y) + (K * v.y);
		step.z = (nesterov * dw.z) + (K * v.z);
		step.w = (nesterov * dw.w) + (K * v.w);

		//<5> W = W - lr * step
		w.x -= lr * step.x;
		w.y -= lr * step.y;
		w.z -= lr * step.z;
		w.w -= lr * step.w;

		//write data-----------------------------------------------
		within_width(v, index4, stride, width);
		within_width(w, index4, stride, width);
		*(float4*)(V + index4) = v;//update velocity
		*(float4*)(W + index4) = w;//update weight
	}
}

#endif


void __sgdmn2D(cudaStream_t stream,
	const float* W,
	float* V,  float momentum, float dampen, float nesterov,
	const float* deltaW, float lr,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { sgdmn2d_k4_small(stream, W, V, momentum, dampen, nesterov, deltaW, lr, lengthv, width, stride); return; }
	sgdmn2d_k4(stream, 5, 2, W, V, momentum, dampen, nesterov, deltaW, lr, lengthv, width, stride);
}

#endif