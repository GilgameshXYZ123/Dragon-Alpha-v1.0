#pragma once

#ifndef INDEXED_MEMCPY_H
#define INDEXED_MEMCPY_H

#include "Cuda_indexedMemcpy_src.cuh"
#include "Cuda_indexedMemcpy_dst.cuh"

//Method:    srcIndexedMemcpy
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_Cuda_1expk2_srcIndexedMemcpy(JNIEnv *env, jclass cls,
	jlong stream_address, 
	jlong dX_address, jlong dIndex_address,
	jlong dY_address, 
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float* dX = (float*)(intptr_t)dX_address;
	int* dIndex = (int*)(intptr_t)dIndex_address;
	float* dY = (float*)(intptr_t)dY_address;
	__srcIndexedMemcpy(stream, dX, dIndex, dY, 
		lengthv, width, stride);
}

//Method:    dstIndexedMemcpy
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_Cuda_1expk2_dstIndexedMemcpy(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dX_address,
	jlong dIndex_address, 
	jlong dY_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float* dX = (float*)(intptr_t)dX_address;
	int* dIndex = (int*)(intptr_t)dIndex_address;
	float* dY = (float*)(intptr_t)dY_address;
	__dstIndexedMemcpy(stream, dX, dIndex, dY,
		lengthv, width, stride);
}

#endif