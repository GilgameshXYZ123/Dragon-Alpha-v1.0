#pragma once

#ifndef CUDA_TRANSPOSE_H
#define CUDA_TRANSPOSE_H

#include "Cuda_transpose_batchMat.cuh"
#include "Cuda_transpose4D.cuh"
#include "Cuda_transpose3D.cuh"
#include "Cuda_transpose2D.cuh"


//Method:    transpose2D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_Cuda_1expk2_transpose2D(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dX_address,
	jlong dY_address,
	jint Xdim1, jint Ydim1,
	jint strideX, jint strideY,
	jint length)
{
	cudaStream_t stream = (cudaStream_t)stream_address;
	float *dX = (float*)dX_address;
	float *dY = (float*)dY_address;
	__transpose2d(stream, dX, dY, Xdim1, Ydim1, strideX, strideY, length);
}

//Method:    transpose3D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_Cuda_1expk2_transpose3D(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dX_address,
	jlong dY_address,
	jint Xdim1, jint Xdim2,
	jint Ydim1, jint Ydim2,
	jint dimIndex1, jint dimIndex2,
	jint strideX, jint strideY,
	jint length)
{
	cudaStream_t stream = (cudaStream_t)stream_address;
	float *dX = (float*)dX_address;
	float *dY = (float*)dY_address;
	__transpose3d(stream, dX, dY,
		Xdim1, Xdim2, 
		Ydim1, Ydim2,
		dimIndex1, dimIndex2, 
		strideX, strideY, 
		length);
}

//Method:    transposed4D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_Cuda_1expk2_transpose4D(JNIEnv *env, jclass cls,
	jlong stream_address, 
	jlong dX_address,
	jlong dY_address,
	jint Xdim1, jint Xdim2, jint Xdim3, 
	jint Ydim1, jint Ydim2, jint Ydim3,
	jint dimIndex1, jint dimIndex2, 
	jint strideX, jint strideY,
	jint length) 
{
	cudaStream_t stream = (cudaStream_t)stream_address;
	float *dX = (float*)dX_address;
	float *dY = (float*)dY_address;
	__transpose4d(stream, dX, dY,
		Xdim1, Xdim2, Xdim3,
		Ydim1, Ydim2, Ydim3,
		dimIndex1, dimIndex2,
		strideX, strideY,
		length);
}

#endif