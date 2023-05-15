#include "frame.cuh"
#include "Cuda_random.cuh"
#include "JNITool.cuh"
#include "micro.cuh"

#include "random_uniform2D.cuh"
#include "random_sparse_uniform2D.cuh"

#include "random_bernouli2D.cuh"
#include "random_bernouli_mul2D.cuh"

#include "random_gaussian2D.cuh"
#include "random_sparse_gaussian2D.cuh"

#include "test.cuh"


//Method:    bernouli
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1random_bernouli2D(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dX_address, 
	jint seed, 
	jfloat p, jfloat v1, jfloat v2,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)dX_address;
	__bernouli2D(stream, dX, seed, p, v1, v2, lengthv, width, stride);
}

//Method:    bernouli_mul2D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1random_bernouli_1mul2D(JNIEnv *env, jclass cls, 
	jlong stream_address, 
	jlong dX_address,
	jlong dR_address, jlong dY_address,
	jint seed,
	jfloat p, jfloat v1, jfloat v2,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)dX_address;
	float *dR = (float*)(intptr_t)dR_address;
	float *dY = (float*)(intptr_t)dY_address;
	__bernouli_mul2D(stream, dX, dR, dY,
		seed, p, v1, v2,
		lengthv, width, stride);
}

//Method:    uniform2D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1random_uniform2D(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dX_address,
	jint seed,
	jfloat vmin, jfloat vmax, 
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)dX_address;
	__uniform2D(stream, dX, seed, vmin, vmax, lengthv, width, stride);
}

//Method:    sparse_uniform2D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1random_sparse_1uniform2D(JNIEnv *env, jclass cls,
	jlong stream_address, 
	jlong dX_address,
	jint seed1, jint seed2,
	jfloat p, jfloat vmin, jfloat vmax,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)dX_address;
	__sparse_uniform2D(stream, dX, seed1, seed2, p, vmin, vmax, lengthv, width, stride);
}

//Method:    gaussian2D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1random_gaussian2D(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dX_address,
	jint seed1, jint seed2, 
	jfloat mu, jfloat sigma, 
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)dX_address;
	__gaussian2D(stream, dX, seed1, seed2, mu, sigma, lengthv, width, stride);
}

//Method:    sparse_gaussian2D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1random_sparse_1gaussian2D(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dX_address,
	jint seed1, jint seed2, jint seed3,
	jfloat p, jfloat mu, jfloat sigma,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)dX_address;
	__sparse_gaussian2D(stream, dX, seed1, seed2, seed3, p, mu, sigma, lengthv, width, stride);
}