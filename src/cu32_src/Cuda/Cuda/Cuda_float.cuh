#pragma once

#ifndef CUDA_FLOAT_H
#define CUDA_FLOAT_H

#include "Cuda_float_set1D.cuh"
#include "Cuda_float_set2D.cuh"
#include "Cuda_float_set1to2.cuh"
#include "Cuda_float_set2to1.cuh"
#include "Cuda_float_set2to2.cuh"

#ifndef CUDA_FLOAT_GET
#define CUDA_FLOAT_GET

//Method:    get1D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_Cuda_get1D(JNIEnv *env, jclass cls, 
	jlong stream_address, 
	jlong address, jfloatArray value, 
	jint length)
{
	//CUDA to C
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)address;
	float *X = new float[length];
	cudaError_t error = cudaMemcpyAsync(X, dX, (length << 2L), 
		cudaMemcpyDeviceToHost, stream); handleError(error);
	error = cudaStreamSynchronize(stream); handleError(error);

	//C to Java
	env->SetFloatArrayRegion(value, 0, length, X);
	delete X;
}

//Method:    get1D_v2
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_Cuda_get1D_1v2(JNIEnv *env, jclass cls,
	jlong stream_address, 
	jlong address, jfloatArray value, 
	jlong buf_address,
	jint length)
{
	//CUDA to C
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)address;
	float *X = (float*)(intptr_t)buf_address;
	cudaError_t error = cudaMemcpyAsync(X, dX, (length << 2L),
		cudaMemcpyDeviceToHost, stream); handleError(error);
	error = cudaStreamSynchronize(stream); handleError(error);

	//C to Java
	env->SetFloatArrayRegion(value, 0, length, X);
}


//Method:    get2D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_Cuda_get2D(JNIEnv *env, jclass cls, 
	jlong stream_address, 
	jlong address, jfloatArray value, 
	jint height, jint width, jint stride)
{
	//Java to C: stride > width
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	int lengthv = height * stride;
	float *dX = (float*)(intptr_t)address;
	float *X = new float[lengthv];
	
	//CUDA to C
	cudaError_t error = cudaMemcpyAsync(X, dX, lengthv << 2L,
		cudaMemcpyDeviceToHost, stream); handleError(error);
	error = cudaStreamSynchronize(stream); handleError(error);
	
	//C to Java
	if (width > 63) 
	{
		float *tX = X;
		for (int i = 0, start = 0; i < height; i++) 
		{
			env->SetFloatArrayRegion(value, start, width, tX);
			start += width; tX += stride;
		}
	}
	else 
	{
		//compact: lengthv(height * stride) -> length(height * width)
		float *X1 = X + width, *X2 = X + stride;
		for (int i = 1; i < height; i++)
		{
			for (int j = 0; j < width; j++) X1[j] = X2[j];
			X1 += width, X2 += stride;
		}
		env->SetFloatArrayRegion(value, 0, height * width, X);
	}
	delete X;
}


//Method:    get2D_v2
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_Cuda_get2D_1v2(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong address, jfloatArray value,
	jlong buf_address,
	jint height, jint width, jint stride)
{
	//Java to C: stride = (width + 3) >> 2 << 2
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	int lengthv = height * stride;
	float *dX = (float*)(intptr_t)address;
	float *X = (float*)(intptr_t)buf_address;

	//CUDA to C
	cudaError_t error = cudaMemcpyAsync(X, dX, lengthv << 2L,
		cudaMemcpyDeviceToHost, stream); handleError(error);
	error = cudaStreamSynchronize(stream); handleError(error);

	//C to Java
	if (width > 63) 
	{
		float *tX = X;
		for (int i = 0, start = 0; i < height; i++) 
		{
			env->SetFloatArrayRegion(value, start, width, tX);
			start += width; tX += stride;
		}
	}
	else
	{
		//compact: lengthv(height * stride) -> length(height * width)
		float *X1 = X + width, *X2 = X + stride;
		for (int i = 1; i < height; i++) 
		{
			for (int j = 0; j < width; j++) X1[j] = X2[j];
			X1 += width; X2 += stride;
		}
		env->SetFloatArrayRegion(value, 0, height * width, X);
	}
}


#endif


#ifndef CUDA_FLOAT_SET
#define CUDA_FLOAT_SET

//Method:    set1D(float v)
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_Cuda_set1D__JJFI(JNIEnv *env, jclass cls,
	jlong stream_address, 
	jlong address, jfloat value,
	jint length)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)address;
	__set1D(stream, dX, value, length);//return syncer
}

//2D: stride > width
//Method:    set2D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_Cuda_set2D__JJFIII(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong address, jfloat value,
	jint height, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)address;
	__set2D(stream, dX, value, height, width, stride);//return syncer
}

#endif


#ifndef CUDA_FLOAT_SET_CPU_2_GPU_1D
#define CUDA_FLOAT_SET_CPU_2_GPU_1D

//Method:    set1D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_Cuda_set1D__JJ_3BI(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong address, jbyteArray value,
	jint length)
{
	//Java to C
	char *BX = (char*)env->GetByteArrayElements(value, NULL), *tBX = BX;
	float *X = new float[length], *tX = X;
	for (int i = 0; i < length; i++) *tX++ = *tBX++;

	//C to CUDA
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)address;
	cudaError_t error = cudaMemcpyAsync(dX, X, length << 2L,
		cudaMemcpyHostToDevice, stream); handleError(error);
	error = cudaStreamSynchronize(stream); handleError(error);

	delete X;
	env->ReleaseByteArrayElements(value, (jbyte*)BX, JNI_ABORT);
}

//Method:    set1D_v2
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_Cuda_set1D_1v2__JJ_3BJI(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong address, jbyteArray value,
	jlong buf_address,
	jint length)
{
	//Java to C, X: pinned buffer to speed up memcpy(DMA)
	char *BX = (char*)env->GetByteArrayElements(value, NULL), *tBX = BX;
	float *X = (float*)(intptr_t)buf_address, *tX = X;
	for (int i = 0; i < length; i++) *tX++ = *tBX++;
	
	//C to CUDA
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)address;
	cudaError_t error = cudaMemcpyAsync(dX, X, length << 2L,
		cudaMemcpyHostToDevice, stream); handleError(error);
	error = cudaStreamSynchronize(stream); handleError(error);

	env->ReleaseByteArrayElements(value, (jbyte*)BX, JNI_ABORT);
}


//Method:    set1D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_Cuda_set1D__JJ_3FI(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong address, jfloatArray value,
	jint length)
{
	//Java to C
	float *X = env->GetFloatArrayElements(value, NULL);

	//C to CUDA
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)address;
	cudaError_t error = cudaMemcpyAsync(dX, X, length << 2L,
		cudaMemcpyHostToDevice, stream); handleError(error);
	error = cudaStreamSynchronize(stream); handleError(error);

	env->ReleaseFloatArrayElements(value, X, JNI_ABORT);
}

//Method:    set1D_v2
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_Cuda_set1D_1v2__JJ_3FJI(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong address, jfloatArray value, 
	jlong buf_address, 
	jint length) 
{
	//Java to C
	float *X = (float*)(intptr_t)buf_address;
	env->GetFloatArrayRegion(value, 0, length, X);
		
	//C to CUDA
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)address;
	cudaError_t error = cudaMemcpyAsync(dX, X, length << 2L,
		cudaMemcpyHostToDevice, stream); handleError(error);
	error = cudaStreamSynchronize(stream); handleError(error);
}

#endif


#ifndef CUDA_FLOAT_SET_CPU_2_GPU_2D
#define CUDA_FLOAT_SET_CPU_2_GPU_2D

//Method:    set2D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_Cuda_set2D__JJ_3BIII(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong address, jbyteArray value,
	jint height, jint width, jint stride)
{
	//Java to C
	int lengthv = height * stride;
	char *BX = (char*)env->GetByteArrayElements(value, NULL), *tBX = BX;
	float *X = new float[lengthv], *tX = X;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) *tX++ = *tBX++;
		for (int j = width; j < stride; j++) *tX++ = 0.0f;
	}

	//C to CUDA
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)address;
	cudaError_t error = cudaMemcpyAsync(dX, X, lengthv << 2L,
		cudaMemcpyHostToDevice, stream); handleError(error);
	error = cudaStreamSynchronize(stream); handleError(error);

	delete X;
	env->ReleaseByteArrayElements(value, (jbyte*)BX, JNI_ABORT);
}

//Method:    set2D_v2
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_Cuda_set2D_1v2__JJ_3BJIII(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong address, jbyteArray value, 
	jlong buf_address,
	jint height, jint width, jint stride)
{
	//Java to C
	int lengthv = height * stride;
	char *BX = (char*)env->GetByteArrayElements(value, NULL), *tBX = BX;
	float *X = (float*)(intptr_t)buf_address, *tX = X;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) *tX++ = *tBX++;
		for (int j = width; j < stride; j++) *tX++ = 0.0f;
	}

	//C to CUDA
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)address;
	cudaError_t error = cudaMemcpyAsync(dX, X, lengthv << 2L,
		cudaMemcpyHostToDevice, stream); handleError(error);
	error = cudaStreamSynchronize(stream); handleError(error);

	env->ReleaseByteArrayElements(value, (jbyte*)BX, JNI_ABORT);
}

//Method:    set2D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_Cuda_set2D__JJ_3FIII(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong address, jfloatArray value,
	jint height, jint width, jint stride)
{
	//Java to C
	int length = height * width, lengthv = height * stride;
	float *X = new float[lengthv];
	env->GetFloatArrayRegion(value, 0, length, X);

	float *X1 = X + lengthv, *X2 = X + length;//go to the end of Tensor
	for (int i = 0; i < height; i++) {
		for (int j = width; j < stride; j++) *(--X1) = 0.0f;
		for (int j = 0; j < width; j++) *(--X1) = *(--X2);
	}

	//C to CUDA
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)address;
	cudaError_t error = cudaMemcpyAsync(dX, X, lengthv << 2L,
		cudaMemcpyHostToDevice, stream); handleError(error);
	error = cudaStreamSynchronize(stream); handleError(error);

	delete X;
}

//Method:    set2D_v2
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_Cuda_set2D_1v2__JJ_3FJIII(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong address, jfloatArray value,
	jlong buf_address,
	jint height, jint width, jint stride) 
{
	int length = height * width, lengthv = height * stride;
	float* X = (float*)(intptr_t)buf_address;

	env->GetFloatArrayRegion(value, 0, length, X);

	float *X1 = X + lengthv, *X2 = X + length;//go to the end of Tensor
	for (int i = 0; i < height; i++) {
		for (int j = width; j < stride; j++) *(--X1) = 0.0f;
		for (int j = 0; j < width; j++) *(--X1) = *(--X2);
	}
	
	//C to CUDA
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)address;
	cudaError_t error = cudaMemcpyAsync(dX, X, lengthv << 2L,
		cudaMemcpyHostToDevice, stream); handleError(error);
	error = cudaStreamSynchronize(stream); handleError(error);
}

#endif


#ifndef CUDA_FLOAT_SET_FROM_TO
#define CUDA_FLOAT_SET_FROM_TO

//Method:    setFrom1Dto2D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_Cuda_setFrom1Dto2D(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong src_address, jint src_length, 
	jlong dst_address, jint dst_height, jint dst_width, jint dst_stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)src_address;
	float *dY = (float*)(intptr_t)dst_address;
	__set1to2(stream, 
		dX, src_length, 
		dY, dst_height, dst_width, dst_stride);
	//return syncer
}

//Method:    setFrom2Dto1D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_Cuda_setFrom2Dto1D(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong src_address, jint src_height, jint src_width, jint src_stride,
	jlong dst_address, jint dst_length)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)src_address;
	float *dY = (float*)(intptr_t)dst_address;
	__set2to1(stream,
		dX, src_height, src_width, src_stride,
		dY, dst_length);
	//return syncer
}

//Method:    setFrom2Dto2D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_Cuda_setFrom2Dto2D(JNIEnv *env, jclass cls,
	jlong stream_address, 
	jlong src_address, jint src_height, jint src_width, jint src_stride,
	jlong dst_address, jint dst_height, jint dst_width, jint dst_stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)src_address;
	float *dY = (float*)(intptr_t)dst_address;
	__set2to2(stream,
		dX, src_height, src_width, src_stride,
		dY, dst_height, dst_width, dst_stride);
	//return syncer
}

#endif

#endif