#include "frame.cuh"
#include "Cuda_matMul.cuh"
#include "JNITool.cuh"
#include "micro.cuh"

#include "SKbufsum.cuh"
#include "matMul.cuh"
#include "matMulT1.cuh"
#include "matMulT2.cuh"

#include "test.cuh"


#ifndef JNI_SKBUF_SUMMARY
#define JNI_SKBUF_SUMMARY

//Method:    SKbuf_summary
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1matMul_SKbuf_1summary(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dCbuf_address, jlong dC_address, 
	jint part, jint sizeC)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float* dCbuf = (float*)(intptr_t)dCbuf_address;
	float* dC = (float*)(intptr_t)dC_address;
	SKbuf_summary(stream, dCbuf, dC, part, sizeC);
}

#endif


#ifndef JNI_MATMUL
#define JNI_MATMUL

//Method:    matMul
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1matMul_matMul(JNIEnv *env, jclass cls, 
	jlongArray streamArray, jint length,
	jlong dA_address,
	jlong dB_address,
	jlong dC_address, 
	jint N, jint M, jint K)
{
	jlong streams[10];
	env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *dA = (float*)(intptr_t)dA_address;
	float *dB = (float*)(intptr_t)dB_address;
	float *dC = (float*)(intptr_t)dC_address;
	int index = 0;
	matMul4x(streams, index, length, dA, dB, dC, N, M, K, M);
}

#endif


#ifndef JNI_MATMUL_T1
#define JNI_MATMUL_T1

//Method:    matMulT1
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1matMul_matMulT1(JNIEnv *env, jclass cls,
	jlongArray streamArray, jint length,
	jlong dA_address,
	jlong dB_address,
	jlong dC_address,
	jint N, jint M, jint K)
{
	jlong streams[10];
	env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *dA = (float*)(intptr_t)dA_address;
	float *dB = (float*)(intptr_t)dB_address;
	float *dC = (float*)(intptr_t)dC_address;
	int index = 0;
	matMul4x_T1(streams, index, length, dA, dB, dC, N, M, K, N, M);
}

//Method:    matMulT1SK
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1matMul_matMulT1SK(JNIEnv *env, jclass cls,
	jlongArray streamArray, jint length, jint GridZ, 
	jlong dA_address,
	jlong dB_address,
	jlong dC_address, jlong dCbuf_address,
	jint N, jint M, jint K)
{
	jlong streams[10];
	env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *dA = (float*)(intptr_t)dA_address;
	float *dB = (float*)(intptr_t)dB_address;
	float *dC = (float*)(intptr_t)dC_address;
	float *dCbuf = (float*)(intptr_t)dCbuf_address;
	
	int K_slice = SK_K_slice(K, GridZ);
	int index = 0;
	matMul4x_T1_SK(streams, index, length, GridZ,
		dA, dB, dC, dCbuf,
		N, M, K, K_slice, N, M);
}

#endif


#ifndef JNI_MATMUL_T2
#define JNI_MATMUL_T2

//Method:    matMulT2
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1matMul_matMulT2(JNIEnv *env, jclass cls,
	jlongArray streamArray, jint length,
	jlong dA_address,
	jlong dB_address,
	jlong dC_address,
	jint N, jint M, jint K)
{
	jlong streams[10];
	env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *dA = (float*)(intptr_t)dA_address;
	float *dB = (float*)(intptr_t)dB_address;
	float *dC = (float*)(intptr_t)dC_address;
	int index = 0;
	matMul4x_T2(streams, index, length, dA, dB, dC, N, M, K, M);
}

#endif