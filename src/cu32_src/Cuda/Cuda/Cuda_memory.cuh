#pragma once

//Method:    malloc
JNIEXPORT jlong JNICALL Java_z_dragon_engine_cuda_impl_Cuda_malloc(JNIEnv *env, jclass cls, 
	jlong size) 
{
	void *dp = NULL;
	cudaError_t error = cudaMalloc((void**)&dp, size); handleError(error);
	return (intptr_t)dp;
}

//Method:    free
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_Cuda_free(JNIEnv * env, jclass cls, 
	jlong address)
{
	void *p = (void*)(intptr_t)address;
	cudaError_t error = cudaFree(p); handleError(error);
}

//Method:    mallocHost
JNIEXPORT jlong JNICALL Java_z_dragon_engine_cuda_impl_Cuda_mallocHost(JNIEnv *env, jclass cls,
	jlong size)
{
	void *dp = NULL;
	cudaError_t error = cudaMallocHost((void**)&dp, size); handleError(error);
	return (intptr_t)dp;
}

//Method:    freeHost
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_Cuda_freeHost(JNIEnv *env, jclass cls, 
	jlong address)
{
	void *dp = (void*)(intptr_t)address;
	cudaError_t error = cudaFreeHost(dp); handleError(error);
}

//Method:    memsetAsync
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_Cuda_memsetAsync(JNIEnv *env, jclass cls,
	jlong stream_address, jlong address, jint value, jlong size)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	void *dp = (void*)(intptr_t)address;
	cudaError_t error = cudaMemsetAsync(dp, value, size, stream); handleError(error)
	//return syncer
}

//Method:    memcpyAsyncHostToDevice
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_Cuda_memcpyAsyncHostToDevice(JNIEnv *env, jclass cls,
	jlong stream_address, jlong dst_address, jlong src_address, jlong size)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	void *dst = (void*)(intptr_t)dst_address;
	void *src = (void*)(intptr_t)src_address;
	cudaError_t error = cudaMemcpyAsync(dst, src, size, 
		cudaMemcpyHostToDevice, stream); handleError(error);
	//return syncer
}

//Method:    memcpyAsyncHostToHost
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_Cuda_memcpyAsyncHostToHost(JNIEnv *env, jclass cls,
	jlong stream_address, jlong dst_address, jlong src_address, jlong size)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	void *dst = (void*)(intptr_t)dst_address;
	void *src = (void*)(intptr_t)src_address;
	cudaError_t error = cudaMemcpyAsync(dst, src, size, 
		cudaMemcpyHostToHost, stream); handleError(error);
	//return syncer
}

//Method:    memcpyAsyncDeviceToHost
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_Cuda_memcpyAsyncDeviceToHost(JNIEnv *env, jclass cls,
	jlong stream_address, jlong dst_address, jlong src_address, jlong size)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	void *dst = (void*)(intptr_t)dst_address;
	void *src = (void*)(intptr_t)src_address;
	cudaError_t error = cudaMemcpyAsync(dst, src, size, 
		cudaMemcpyDeviceToHost, stream); handleError(error);
	//return syncer
}

//Method:    memcpyAsyncDeviceToDevice
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_Cuda_memcpyAsyncDeviceToDevice
(JNIEnv *env, jclass cls,
	jlong stream_address, jlong dst_address, jlong src_address, jlong size)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	void *dst = (void*)(intptr_t)dst_address;
	void *src = (void*)(intptr_t)src_address;
	cudaError_t error = cudaMemcpyAsync(dst, src, size, 
		cudaMemcpyDeviceToDevice, stream); handleError(error);
	//return syncer
}

//Method:    memcpyAsyncDefault
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_Cuda_memcpyAsyncDefault
(JNIEnv *env, jclass cls,
	jlong stream_address, jlong dst_address, jlong src_address, jlong size)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	void *dst = (void*)(intptr_t)dst_address;
	void *src = (void*)(intptr_t)src_address;
	cudaError_t error = cudaMemcpyAsync(dst, src, size, 
		cudaMemcpyDefault, stream); handleError(error);
	//return syncer
}