#pragma once

#ifndef CUDA_INT_H
#define CUDA_INT_H


#ifndef CUDA_INT_GET_1D
#define CUDA_INT_GET_1D

//Method:    get1D_int
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_Cuda_get1D_1int(JNIEnv *env, jclass cls,
	jlong stream_address, 
	jlong address, jintArray value,
	jint length)
{
	//CUDA to C
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	int *dX = (int*)(intptr_t)address;
	int *X = new int[length];
	cudaError_t error = cudaMemcpyAsync(X, dX, (length << 2L),
		cudaMemcpyDeviceToHost, stream); handleError(error);
	error = cudaStreamSynchronize(stream); handleError(error);

	//C to Java
	env->SetIntArrayRegion(value, 0, length, (jint*)X);
	delete X;
}

//Method:    get1D_v2_int
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_Cuda_get1D_1v2_1int(JNIEnv *env, jclass cls,
	jlong stream_address, 
	jlong address, jintArray value,
	jlong buf_address,
	jint length)
{
	//CUDA to C
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	int *dX = (int*)(intptr_t)address;
	int *X = (int*)(intptr_t)buf_address;
	cudaError_t error = cudaMemcpyAsync(X, dX, (length << 2L),
		cudaMemcpyDeviceToHost, stream); handleError(error);
	error = cudaStreamSynchronize(stream); handleError(error);

	//C to Java
	env->SetIntArrayRegion(value, 0, length, (jint*)X);
}

#endif


#ifndef CUDA_INT_GET_2D
#define CUDA_INT_GET_2D

//Method:    get2D_int
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_Cuda_get2D_1int(JNIEnv *env, jclass cls,
	jlong stream_address, 
	jlong address,jintArray value,
	jint height, jint width, jint stride)
{
	//Java to C: stride > width
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	int lengthv = height * stride;
	int *dX = (int*)(intptr_t)address;
	int *X = new int[lengthv];

	//CUDA to C
	cudaError_t error = cudaMemcpyAsync(X, dX, lengthv << 2L,
		cudaMemcpyDeviceToHost, stream); handleError(error);
	error = cudaStreamSynchronize(stream); handleError(error);

	//C to Java
	if (width > 63)
	{
		int *tX = X;
		for (int i = 0, start = 0; i < height; i++) 
		{
			env->SetIntArrayRegion(value, start, width, (jint*)tX);
			start += width; tX += stride;
		}
	}
	else
	{
		//compact: lengthv(height * stride) -> length(height * width)
		int *X1 = X + width, *X2 = X + stride;
		for (int i = 1; i < height; i++)
		{
			for (int j = 0; j < width; j++) X1[j] = X2[j];
			X1 += width; X2 += stride;
		}
		env->SetIntArrayRegion(value, 0, height * width, (jint*)X);
	}
	delete X;
}

//Method:    get2D_v2_int
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_Cuda_get2D_1v2_1int(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong address, jintArray value, 
	jlong buf_address,
	jint height, jint width, jint stride)
{
	//Java to C: stride = (width + 3) >> 2 << 2
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	int lengthv = height * stride;
	int *dX = (int*)(intptr_t)address;
	int *X = (int*)(intptr_t)buf_address;

	//CUDA to C
	cudaError_t error = cudaMemcpyAsync(X, dX, lengthv << 2L,
		cudaMemcpyDeviceToHost, stream); handleError(error);
	error = cudaStreamSynchronize(stream); handleError(error);

	//C to Java
	if (width > 63)
	{
		int *tX = X;
		for (int i = 0, start = 0; i < height; i++) 
		{
			env->SetIntArrayRegion(value, start, width, (jint*)tX);
			start += width; tX += stride;
		}
	}
	else
	{
		//compact: lengthv(height * stride) -> length(height * width)
		int *X1 = X + width, *X2 = X + stride;
		for (int i = 1; i < height; i++) 
		{
			for (int j = 0; j < width; j++) X1[j] = X2[j];
			X1 += width; X2 += stride;
		}
		env->SetIntArrayRegion(value, 0, height * width, (jint*)X);
	}
}

#endif


//transfer data: CPU -> GPU(1D)
#ifndef CUDA_INT_SET_CPU2GPU_1D
#define CUDA_INT_SET_CPU2GPU_1D

//Method:    set1D_int
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_Cuda_set1D_1int(JNIEnv *env, jclass cls,
	jlong stream_address, 
	jlong address, jintArray value, 
	jint length)
{
	//Java to C
	int *X = (int*)env->GetIntArrayElements(value, NULL);

	//C to CUDA
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	int* dX = (int*)(intptr_t)address;
	cudaError_t error = cudaMemcpyAsync(dX, X, length << 2L,
		cudaMemcpyHostToDevice, stream); handleError(error);
	error = cudaStreamSynchronize(stream); handleError(error);

	env->ReleaseIntArrayElements(value, (jint*)X, JNI_ABORT);
}

//Method:    set1D_v2_int
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_Cuda_set1D_1v2_1int(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong address, jintArray value,
	jlong buf_address,
	jint length)
{
	//Java to C
	int *X = (int*)(intptr_t)buf_address;
	env->GetIntArrayRegion(value, 0, length, (jint*)X);

	//C to CUDA
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	int *dX = (int*)(intptr_t)address;
	cudaError_t error = cudaMemcpyAsync(dX, X, length << 2L,
		cudaMemcpyHostToDevice, stream); handleError(error);
	error = cudaStreamSynchronize(stream); handleError(error);
}

#endif


//transfer data: CPU -> GPU(2D) with zero padding
#ifndef CUDA_INT_SET_CPU2GPU_2D
#define CUDA_INT_SET_CPU2GPU_2D

//Method:    set2D_int
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_Cuda_set2D_1int(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong address, jintArray value,
	jint height, jint width, jint stride)
{
	//Java to C
	int length = height * width, lengthv = height * stride;
	int *X = new int[lengthv];
	env->GetIntArrayRegion(value, 0, length, (jint*)X);

	int *X1 = X + lengthv, *X2 = X + length;//go to the end of Tensor
	for (int i = 0; i < height; i++) {
		for (int j = width; j < stride; j++) *(--X1) = 0;
		for (int j = 0; j < width; j++) *(--X1) = *(--X2);
	}

	//C to CUDA
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	int *dX = (int*)(intptr_t)address;
	cudaError_t error = cudaMemcpyAsync(dX, X, lengthv << 2L,
		cudaMemcpyHostToDevice, stream); handleError(error);
	error = cudaStreamSynchronize(stream); handleError(error);

	delete X;
}

//Method:    set2D_v2_int
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_Cuda_set2D_1v2_1int(JNIEnv *env, jclass cls,
	jlong stream_address, 
	jlong address, jintArray value,
	jlong buf_address,
	jint height, jint width, jint stride)
{
	int length = height * width, lengthv = height * stride;
	int* X = (int*)(intptr_t)buf_address;

	env->GetIntArrayRegion(value, 0, length, (jint*)X);

	int *X1 = X + lengthv, *X2 = X + length;//go to the end of Tensor
	for (int i = 0; i < height; i++) {
		for (int j = width; j < stride; j++) *(--X1) = 0;
		for (int j = 0; j < width; j++) *(--X1) = *(--X2);
	}

	//C to CUDA
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	int *dX = (int*)(intptr_t)address;
	cudaError_t error = cudaMemcpyAsync(dX, X, lengthv << 2L,
		cudaMemcpyHostToDevice, stream); handleError(error);
	error = cudaStreamSynchronize(stream); handleError(error);
}

#endif

#endif