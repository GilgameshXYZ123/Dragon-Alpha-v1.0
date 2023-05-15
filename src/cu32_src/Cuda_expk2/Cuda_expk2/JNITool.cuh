#pragma once

#ifndef JCUDA_TOOL_H
#define JCUDA_TOOL_H

#define printError(error) \
	{if(error!=cudaSuccess) printf("error: %d  -  %s\n", error, cudaGetErrorName(error));}

#define handleError(error) \
	{if(error!=cudaSuccess) throwCudaException(env, error);}

#define handleError(error) \
	{if(error!=cudaSuccess) throwCudaException(env, error);}

//passed
JNIEXPORT void JNICALL throwCudaException
(JNIEnv* env, cudaError_t stat)
{
	jclass cls = env->FindClass("z/jcuda/exception/CudaException");
	jmethodID constructor = env->GetMethodID(cls, "<init>", "(I)V");
	jobject e = env->NewObject(cls, constructor, stat);
	env->Throw((jthrowable)e);
}

//passed
JNIEXPORT void JNICALL throwCudaException
(JNIEnv* env, cudaError_t stat, const char *msg)
{
	jclass cls = env->FindClass("z/jcuda/exception/CudaException");
	jmethodID constructor = env->GetMethodID(cls, "<init>", "(ILjava/lang/String;)V");
	jobject e = env->NewObject(cls, constructor, stat, env->NewStringUTF(msg));
	env->Throw((jthrowable)e);
}
#endif
