#pragma once

#ifndef GAPPED_MEMCPY_2D_H
#define GAPPED_MEMCPY_2D_H


#ifndef GAPPED_MEMCPY_CALL
#define GAPPED_MEMCPY_CALL

//float1===========================================================
#define gappedMemcpy_k1_small(stream, X, strideX, Y, strideY, width, length) \
	gappedMemcpy_kernel1\
		<<< 1, length, 0, stream >>> \
			(X, strideX, Y, strideY, width, length)

#define gappedMemcpy_k1(stream, LB, LT, X, strideX, Y, strideY, width, length) \
	gappedMemcpy_kernel1\
		<<< (length >> LB >> LT), (1<<LB), 0, stream>>>\
			(X, strideX, Y, strideY, width, length)

//float4===========================================================
#define gappedMemcpy_k4_small(stream, X, strideX, Y, strideY, width, length) \
	gappedMemcpy_kernel4\
		<<< 1, length, 0, stream >>> \
			(X, strideX, Y, strideY, width, length)

#define gappedMemcpy_k4(stream, LB, LT, X, strideX, Y, strideY, width, length) \
	gappedMemcpy_kernel4\
		<<< (length >> LB >> LT), (1<<LB), 0, stream>>>\
			(X, strideX, Y, strideY, width, length)

#endif


#ifndef GAPPED_MEMCPY_KERNEL4
#define GAPPED_MEMCPY_KERNEL4

__global__ void gappedMemcpy_kernel4(
	const float* __restrict__ X, int strideX,
	      float* __restrict__ Y, int strideY,
	int width, int length)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x, index4 = index << 2;
	int step = (blockDim.x * gridDim.x), step4 = step << 2;

	for (; index4 < length; index4 += step4)
	{
		//index4 = y*width + x
		int y = index4 / width;
		int x = index4 - y * width;

		int Xindex = y * strideX + x;
		int Yindex = y * strideY + x;
		*(float4*)(Y + Yindex) = *(float4*)(X + Xindex);
	}
}

#endif


#ifndef GAPPED_MEMCPY_KERNEL1
#define GAPPED_MEMCPY_KERNEL1

__global__ void gappedMemcpy_kernel1(
	const float* __restrict__ X, int strideX,
	      float* __restrict__ Y, int strideY,
	int width, int length)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	int step = (blockDim.x * gridDim.x);

	for (; index < length; index += step)
	{
		//index = y*width + x
		int y = index / width;
		int x = index - y * width;
		
		int Xindex = y * strideX + x;
		int Yindex = y * strideY + x;
		Y[Yindex] = X[Xindex];
	}
}

#endif


#ifndef GAPPED_MEMCPY_2D_FUNCTION
#define GAPPED_MEMCPY_2D_FUNCTION

//total_stride >= copy_stride
//length % copy_stride == 0
//X(src) -> Y(dst)
void __gappedMemcpy(cudaStream_t stream,
	const float* X, int Xstart, int strideX,
	      float* Y, int Ystart, int strideY,
	int width, int length)
{
	X += Xstart; Y += Ystart;
	if (!(width & 3)) {//copy_stride % 4 == 0
		if (length < 256) { gappedMemcpy_k4_small(stream, X, strideX, Y, strideY, width, length); return; }
		gappedMemcpy_k4(stream, 5, 2, X, strideX, Y, strideY, width, length);
		return;
	}

	if (length < 256) { gappedMemcpy_k1_small(stream, X, strideX, Y, strideY, width, length); return; }
	gappedMemcpy_k1(stream, 5, 2, X, strideX, Y, strideY, width, length);
}


#endif


//Method:    gappedMemcpy2D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_Cuda_1expk2_gappedMemcpy2D(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dX_address, jint Xstart, jint strideX,
	jlong dY_address, jint Ystart, jint strideY,
	jint width, jint length) 
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float* dX = (float*)(intptr_t)dX_address;
	float* dY = (float*)(intptr_t)dY_address;
	__gappedMemcpy(stream, dX, Xstart, strideX, dY, Ystart, strideY, width, length);
}

#endif