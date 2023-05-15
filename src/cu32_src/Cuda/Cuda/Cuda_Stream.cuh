#pragma once

//Method:    getStreamPriorityRange
JNIEXPORT jintArray JNICALL Java_z_dragon_engine_cuda_impl_Cuda_getStreamPriorityRange(JNIEnv *env, jclass cls)
{
	int low, high;
	cudaError_t error = cudaDeviceGetStreamPriorityRange(&low, &high); handleError(error);
	jintArray arr = env->NewIntArray(2);
	long range[2] = { low, high };
	env->SetIntArrayRegion(arr, 0, 2, range);
	return arr;
}

//Method:    newStream_Blocking
JNIEXPORT jlong JNICALL Java_z_dragon_engine_cuda_impl_Cuda_newStream_1Blocking__(JNIEnv *env, jclass cls)
{
	cudaStream_t dp = NULL;
	cudaError_t error = cudaStreamCreate(&dp); handleError(error);
	return (intptr_t)dp;
}

//Method:    newStream_Blocking
JNIEXPORT jlong JNICALL Java_z_dragon_engine_cuda_impl_Cuda_newStream_1Blocking__I(JNIEnv *env, jclass cls, 
	jint priority)
{
	cudaStream_t dp = NULL;
	cudaError_t error = cudaStreamCreateWithPriority(&dp, 
		cudaStreamDefault, priority); handleError(error);
	return (intptr_t)dp;
}

//Method:    newStream_NonBlocking
JNIEXPORT jlong JNICALL Java_z_dragon_engine_cuda_impl_Cuda_newStream_1NonBlocking__(JNIEnv *env, jclass cls)
{
	cudaStream_t dp = NULL;
	cudaError_t error = cudaStreamCreateWithFlags(&dp, 
		cudaStreamNonBlocking); handleError(error);
	return (intptr_t)dp;
}

//Method:    newStream_NonBlocking
JNIEXPORT jlong JNICALL Java_z_dragon_engine_cuda_impl_Cuda_newStream_1NonBlocking__I(JNIEnv *env, jclass cls, 
	jint priority)
{
	cudaStream_t dp = NULL;
	cudaError_t error = cudaStreamCreateWithPriority(&dp,
		cudaStreamNonBlocking, priority); handleError(error);
	return (intptr_t)dp;
}

//Method:    deleteStream
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_Cuda_deleteStream__J(JNIEnv *env, jclass cls, 
	jlong stream_address)
{
	if (!stream_address) return;
	cudaStream_t dp = (cudaStream_t)(intptr_t)stream_address;
	cudaError_t error = cudaStreamDestroy(dp); handleError(error);
}

//Method:	deleteStream
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_Cuda_deleteStream___3JI(JNIEnv *env, jclass cls,
	jlongArray stream_arr, jint length)
{
	jlong *stream_addrs = env->GetLongArrayElements(stream_arr, NULL);
	for (int i = 0; i < length; i++) 
	{
		if (!stream_addrs[i]) continue;
		cudaStream_t stream = (cudaStream_t)(intptr_t)stream_addrs[i];
		cudaError_t error = cudaStreamDestroy(stream); handleError(error);
	}
	env->ReleaseLongArrayElements(stream_arr, stream_addrs, JNI_ABORT);
}

//Method:    streamQuery
JNIEXPORT jboolean JNICALL Java_z_dragon_engine_cuda_impl_Cuda_streamQuery(JNIEnv *env, jclass cls,
	jlong stream_address)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	cudaError_t error = cudaStreamQuery(stream); 
	if (error == cudaSuccess) return true;
	if (error == cudaErrorNotReady) return false;
	handleError(error);
	return false;
}

//Method:    streamSynchronize
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_Cuda_streamSynchronize__J(JNIEnv *env, jclass cls,
	jlong stream_address)
{
	cudaStream_t dp = (cudaStream_t)(intptr_t)stream_address;
	cudaError_t error = cudaStreamSynchronize(dp); handleError(error);
}

//Method:	streamSynchronize
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_Cuda_streamSynchronize___3JI(JNIEnv *env, jclass cls,
	jlongArray stream_arr, jint length)
{
	jlong *streamAddrs = env->GetLongArrayElements(stream_arr, NULL);
	for (int i = 0; i < length; i++) {
		cudaStream_t stream = (cudaStream_t)(intptr_t)streamAddrs[i];
		cudaError_t error = cudaStreamSynchronize(stream); handleError(error);
	}
	env->ReleaseLongArrayElements(stream_arr, streamAddrs, JNI_ABORT);
}