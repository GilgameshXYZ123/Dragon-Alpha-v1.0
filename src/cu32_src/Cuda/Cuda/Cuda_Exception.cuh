#pragma once

//Method:    getExceptionName
JNIEXPORT jstring JNICALL Java_z_dragon_engine_cuda_impl_Cuda_getExceptionName(JNIEnv * env, jclass cls,
	jint type)
{
	cudaError_t error = (cudaError_t)type;
	return env->NewStringUTF(cudaGetErrorName(error));
}

//Method:    getExceptionInfo
JNIEXPORT jstring JNICALL Java_z_dragon_engine_cuda_impl_Cuda_getExceptionInfo(JNIEnv *env, jclass cls,
	jint type)
{
	cudaError_t error = (cudaError_t)type;
	return env->NewStringUTF(cudaGetErrorString(error));
}

//Method:    getLastExceptionType
JNIEXPORT jint JNICALL Java_z_dragon_engine_cuda_impl_Cuda_getLastExceptionType(JNIEnv *env, jclass cls)
{
	return cudaGetLastError();
}