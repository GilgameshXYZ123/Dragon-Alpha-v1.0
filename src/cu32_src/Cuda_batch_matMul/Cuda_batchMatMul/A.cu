#include "frame.cuh"
#include "Cuda_batchMatMul.cuh"
#include "JNITool.cuh"
#include "micro.cuh"
#include "texture.cuh"

#include "batchMatMul.cuh"
#include "batchMatMulT1.cuh"
#include "batchMatMulT2.cuh"

#include "test.cuh"

#ifndef JNI_BATCH_MATMUL 
#define JNI_BATCH_MATMUL

//Method:    batchMatMul
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1batchMatMul_batchMatMul(JNIEnv *env, jclass cls, 
	jlongArray streamArray, jint length,
	jlong dA_address,
	jlong dB_address,
	jlong dC_address,
	jint Batch, jint N, jint M, jint BK, jint AK)
{
	jlong streams[10];
	env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *dA = (float*)(intptr_t)dA_address;
	float *dB = (float*)(intptr_t)dB_address;
	float *dC = (float*)(intptr_t)dC_address;

	int index = 0;
	__batch_matMul(1, 1, streams, index, length,
		dA, dB, dC, 
		Batch, N, M, BK, AK);

}

//Method:    batchMatMul_texture
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1batchMatMul_batchMatMul_1texture(JNIEnv *env, jclass cls,
	jlongArray streamArray, jint length,
	jlong dA_address,
	jlong dB_address,
	jlong dC_address,
	jint Batch, jint N, jint M, jint BK, jint AK)
{
	jlong streams[10];
	env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *dA = (float*)(intptr_t)dA_address;
	float *dB = (float*)(intptr_t)dB_address;
	float *dC = (float*)(intptr_t)dC_address;

	int index = 0;
	cudaTextureObject_t texA = floatTexture(dA, (Batch * N * AK), env);
	__batch_matMul_tex(1, 1, streams, index, length,
		dA, texA, dB, dC, Batch,
		N, M, BK, AK);
	cudaError_t error = cudaDestroyTextureObject(texA); handleError(error);
}

#endif


#ifndef JNI_BATCH_MATMUL_T1
#define JNI_BATCH_MATMUL_T1

//Method:    batchMatMulT1
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1batchMatMul_batchMatMulT1(JNIEnv *env, jclass cls,
	jlongArray streamArray, jint length,
	jlong dA_address,
	jlong dB_address,
	jlong dC_address,
	jint Batch, jint CN, jint AN, jint M, jint K)
{
	jlong streams[10];
	env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *dA = (float*)(intptr_t)dA_address;
	float *dB = (float*)(intptr_t)dB_address;
	float *dC = (float*)(intptr_t)dC_address;

	int index = 0;
	__batch_matMulT1(1, 1, streams, index, length, 
		dA, dB, dC, 
		Batch, CN, AN, M, K);
}

#endif


#ifndef JNI_BATCH_MATMUL_T2
#define JNI_BATCH_MATMUL_T2

//Method:    batchMatMulT2
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1batchMatMul_batchMatMulT2(JNIEnv *env, jclass cls,
	jlongArray streamArray, jint length,
	jlong dA_address,
	jlong dB_address,
	jlong dC_address,
	jint Batch, jint N, jint CM, jint BM, jint K)
{
	jlong streams[10];
	env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *dA = (float*)(intptr_t)dA_address;
	float *dB = (float*)(intptr_t)dB_address;
	float *dC = (float*)(intptr_t)dC_address;

	int index = 0;
	__batch_matMulT2(1, 1, streams, index, length, 
		dA, dB, dC, 
		Batch, N, CM, BM, K);
}

//Method:    batchMatMulT2_texture
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1batchMatMul_batchMatMulT2_1texture(JNIEnv *env, jclass cls,
	jlongArray streamArray, jint length,
	jlong dA_address,
	jlong dB_address,
	jlong dC_address,
	jint Batch, jint N, jint CM, jint BM, jint K)
{
	jlong streams[10];
	env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *dA = (float*)(intptr_t)dA_address;
	float *dB = (float*)(intptr_t)dB_address;
	float *dC = (float*)(intptr_t)dC_address;

	cudaTextureObject_t texA = floatTexture(dA, (Batch * N * K), env);
	cudaTextureObject_t texB = floatTexture(dB, (Batch * K * CM), env);
	int index = 0;
	__batch_matMulT2_tex(1, 1, streams, index, length,
		dA, texA, dB, texB, dC, 
		Batch, N, CM, BM, K);

	cudaError_t error;
	error = cudaDestroyTextureObject(texA); handleError(error);
	error = cudaDestroyTextureObject(texB); handleError(error);
}

#endif