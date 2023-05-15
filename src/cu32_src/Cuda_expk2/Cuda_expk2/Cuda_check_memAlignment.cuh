#pragma once

#ifndef CHECK_MEM_ALIGNMENT_H
#define CHECK_MEM_ALIGNMENT_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0

#ifndef CHECK_MEM_ALIGNMENT_CALL
#define CHECK_MEM_ALIGNMENT_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define check_memAlign_k4(stream, LB, LT, X, lengthv, width, stride)\
	check_mem_alignment_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X, lengthv, width, stride)

#define check_memAlign_k4_small(stream, X, lengthv, width, stride)\
	check_mem_alignment_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(X, lengthv, width, stride)

#endif


#ifndef CHECK_MEM_ALIGNMENT_KERNEL_4
#define CHECK_MEM_ALIGNMENT_KERNEL_4

__global__ void check_mem_alignment_kernel_4(
	const float* __restrict__ X,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 x = *(float4*)(&X[index4]);

		bool flag0 = (((index4    ) % stride) >= width) && (x.x != 0);
		bool flag1 = (((index4 + 1) % stride) >= width) && (x.y != 0);
		bool flag2 = (((index4 + 2) % stride) >= width) && (x.z != 0);
		bool flag3 = (((index4 + 3) % stride) >= width) && (x.w != 0);

		bool flag = flag0 || flag1 || flag2 || flag3;
		if (flag) { printf("error"); return; }
	}
}

#endif

void __check_mem_alignment(cudaStream_t stream,
	const float* X,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { check_memAlign_k4_small(stream, X, lengthv, width, stride); return; }
	check_memAlign_k4(stream, 5, 2, X, lengthv, width, stride);
}


//Method:    check_mem_alignment
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_Cuda_1expk2_check_1mem_1alignment(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dX_address,
	jint lengthv, jint stride, jint width)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)(dX_address);
	__check_mem_alignment(stream, dX, lengthv, width, stride);
}

#endif