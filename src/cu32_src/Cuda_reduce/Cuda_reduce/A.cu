#include "frame.cuh"
#include "Cuda_reduce.cuh"
#include "JNITool.cuh"
#include "micro.cuh"
#include "R_straight.cuh"
#include "R_field.cuh"
#include "R_row.cuh"
#include "test.cuh"


#ifndef JNI_STRAIGHT_REDUCE
#define JNI_STRAIGHT_REDUCE

//Method:    straight_linear
JNIEXPORT jint JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1reduce_straight_1linear(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dX_address,
	jfloat alpha, jfloat beta, 
	jint lengthv, 
	jlong dV_address, 
	jint width, jint stride,
	jint partNum)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)dX_address;
	float *dV = (float*)(intptr_t)dV_address;
	return __straight_linear(stream, dX, alpha, beta, lengthv, dV, 
		width, stride, partNum);
}

//Method:    straight_quadratic
JNIEXPORT jint JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1reduce_straight_1quadratic(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dX_address, 
	jfloat alpha, jfloat beta, jfloat gamma,
	jint lengthv, 
	jlong dV_address, 
	jint width, jint stride,
	jint partNum)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)dX_address;
	float *dV = (float*)(intptr_t)dV_address;
	return __straight_quadratic(stream, dX, alpha, beta, gamma, lengthv, dV, 
		width, stride, partNum);
}

//Method:    straight_max
JNIEXPORT jint JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1reduce_straight_1max(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dX_address, jint lengthv, 
	jlong dV_address,
	jint width, jint stride,
	jint partNum)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)dX_address;
	float *dV = (float*)(intptr_t)dV_address;
	return __straight_max(stream, dX, lengthv, dV,
		width, stride, partNum);
}

//Method:    straight_min
JNIEXPORT jint JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1reduce_straight_1min(JNIEnv *env, jclass cls,
	jlong stream_address, 
	jlong dX_address, jint lengthv, 
	jlong dV_address,
	jint width, jint stride,
	jint partNum)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)dX_address;
	float *dV = (float*)(intptr_t)dV_address;
	return __straight_min(stream, dX, lengthv, dV,
		width, stride, partNum);
}

//Method:    straight_max_indexed
JNIEXPORT jint JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1reduce_straight_1max_1indexed(JNIEnv *env, jclass cls,
	jlong stream_address, 
	jlong dX_address, jint lengthv, 
	jlong dV_address, jlong dIndex_address,
	jint width, jint stride,
	jint partNum)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)dX_address;
	float *dV = (float*)(intptr_t)dV_address;
	int *dIndex = (int*)(intptr_t)dIndex_address;
	return __straight_max_indexed(stream, dX, lengthv, dV, dIndex, 
		width, stride, partNum);
}

//Method:    straight_min_indexed
JNIEXPORT jint JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1reduce_straight_1min_1indexed(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dX_address, jint lengthv, 
	jlong dV_address, jlong dIndex_address,
	jint width, jint stride,
	jint partNum)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)dX_address;
	float *dV = (float*)(intptr_t)dV_address;
	int *dIndex = (int*)(intptr_t)dIndex_address;
	return __straight_min_indexed(stream, dX, lengthv, dV, dIndex, 
		width, stride, partNum);
}

#endif


//------[field reduce area]--------------------------------------
#ifndef JNI_FIELD_REDUCE
#define JNI_FIELD_REDUCE

//Method:    field_linear
JNIEXPORT jint JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1reduce_field_1linear(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dX_address,
	jfloat alpha, jfloat beta,
	jint N, jint M,
	jlong dV_address, jlong dY_address,
	jint width, jint stride,
	jint partNum)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)dX_address;
	float *dV = (float*)(intptr_t)dV_address;
	float *dY = (float*)(intptr_t)dY_address;
	return __field_linear(stream, dX, alpha, beta, N, M, dV, dY,
		width, stride, partNum);
}

//Method:    field_linear_dual
JNIEXPORT jint JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1reduce_field_1linear_1dual(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dX1_address, jlong dX2_address,
	jfloat alpha, jfloat beta, jfloat gamma, 
	jint N, jint M,
	jlong dV_address, jlong dY_address,
	jint width, jint stride,
	jint partNum)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX1 = (float*)(intptr_t)dX1_address;
	float *dX2 = (float*)(intptr_t)dX2_address;
	float *dV = (float*)(intptr_t)dV_address;
	float *dY = (float*)(intptr_t)dY_address;
	return __field_linear_dual(stream, dX1, dX2, alpha, beta, gamma, N, M, dV, dY, 
		width, stride, partNum);
}

//Method:    field_quadratic
JNIEXPORT jint JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1reduce_field_1quadratic(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dX_address,
	jfloat alpha, jfloat beta, jfloat gamma,
	jint N, jint M,
	jlong dV_address, jlong dY_address, 
	jint width, jint stride, 
	jint partNum)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)dX_address;
	float *dV = (float*)(intptr_t)dV_address;
	float *dY = (float*)(intptr_t)dY_address;
	return __field_quadratic(stream, dX, alpha, beta, gamma, N, M, dV, dY,
		width, stride, partNum);
}

//Method:    field_quadratic_dual
JNIEXPORT jint JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1reduce_field_1quadratic_1dual(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dX1_address, jlong dX2_address,
	jfloat k11, jfloat k12, jfloat k22,
	jfloat k1, jfloat k2, jfloat C,
	jint N, jint M,
	jlong dV_address, jlong dY_address,
	jint width, jint stride,
	jint partNum)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX1 = (float*)(intptr_t)dX1_address;
	float *dX2 = (float*)(intptr_t)dX2_address;
	float *dV = (float*)(intptr_t)dV_address;
	float *dY = (float*)(intptr_t)dY_address;
	return __field_quadratic_dual(stream, dX1, dX2, k11, k12, k22, k1, k2, C, N, M, dV, dY,
		width, stride, partNum);
}

//Method:    field_linear_quadratic
JNIEXPORT jint JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1reduce_field_1linear_1quadratic(JNIEnv *env, jclass cls,
	jlong stream1_address, jlong stream2_address, 
	jlong dX_address,
	jfloat alpha1, jfloat beta1,
	jfloat alpha2, jfloat beta2, jfloat gamma2,
	jint N, jint M,
	jlong dV1_address, jlong dY1_address, 
	jlong dV2_address, jlong dY2_address,
	jint width, jint stride,
	jint partNum)
{
	cudaStream_t stream1 = (cudaStream_t)(intptr_t)stream1_address;
	cudaStream_t stream2 = (cudaStream_t)(intptr_t)stream2_address;

	float* dX = (float*)(intptr_t)dX_address;
	float* dV1 = (float*)(intptr_t)dV1_address;
	float* dV2 = (float*)(intptr_t)dV2_address;
	float* dY1 = (float*)(intptr_t)dY1_address;
	float* dY2 = (float*)(intptr_t)dY2_address;

	return __field_linear_quadratic(env, stream1, stream2,
		dX, alpha1, beta1, alpha2, beta2, gamma2, N, M,
		dV1, dY1, dV2, dY2,
		stride, width, partNum);
}

//Method:    field_max
JNIEXPORT jint JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1reduce_field_1max(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dX_address,
	jint N, jint M,
	jlong dV_address, jlong dY_address,
	jint width, jint stride, 
	jint partNum)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)dX_address;
	float *dV = (float*)(intptr_t)dV_address;
	float *dY = (float*)(intptr_t)dY_address;
	return __field_max(stream, dX, N, M, dV, dY, 
		width, stride, partNum);
}

//Method:    field_min
JNIEXPORT jint JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1reduce_field_1min(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dX_address,
	jint N, jint M,
	jlong dV_address, jlong dY_address,
	jint width, jint stride, 
	jint partNum)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)dX_address;
	float *dV = (float*)(intptr_t)dV_address;
	float *dY = (float*)(intptr_t)dY_address;
	return __field_min(stream, dX, N, M, dV, dY, 
		width, stride, partNum);
}

//Method:    field_max_indexed
JNIEXPORT jint JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1reduce_field_1max_1indexed(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dX_address,
	jint N, jint M,
	jlong dV_address, jlong dVIndex_address,
	jlong dY_address, jlong dIndex_address,
	jint width, jint stride, 
	jint partNum)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)dX_address;
	float *dV = (float*)(intptr_t)dV_address;
	float *dY = (float*)(intptr_t)dY_address;
	int *dVIndex = (int*)(intptr_t)dVIndex_address;
	int *dIndex = (int*)(intptr_t)dIndex_address;
	return __field_max_indexed(stream, dX, N, M, dV, dVIndex, dY, dIndex,
		width, stride, 1);
}

//Method:    field_min_indexed
JNIEXPORT jint JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1reduce_field_1min_1indexed(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dX_address,
	jint N, jint M,
	jlong dV_address, jlong dVIndex_address,
	jlong dY_address, jlong dIndex_address,
	jint width, jint stride, 
	jint partNum)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)dX_address;
	float *dV = (float*)(intptr_t)dV_address;
	float *dY = (float*)(intptr_t)dY_address;
	int *dVIndex = (int*)(intptr_t)dVIndex_address;
	int *dIndex = (int*)(intptr_t)dIndex_address;
	return __field_min_indexed(stream, dX, N, M, dV, dVIndex, dY, dIndex,
		width, stride, 1);
}

#endif


//affine, sqBatchNorm, batchNorm.backward
#ifndef JNI_FIELD_AFFINE_REDUCE
#define JNI_FIELD_AFFINE_REDUCE

//Method:    field_affine_deltaA_v1
JNIEXPORT jint JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1reduce_field_1affine_1deltaA_1v1(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong d_deltaY_address, 
	jlong dY_address,//V1: holdY(), Y is not changed
	jlong dA_address, jlong dB_address, 
	jint N, jint M,
	jlong d_deltaA_buf_address, jlong d_deltaA_address,//deltaA = deltaXp2
	jint width, jint stride, 
	jint partNum)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float *dY = (float*)(intptr_t)dY_address;
	float *dA = (float*)(intptr_t)dA_address;
	float *dB = (float*)(intptr_t)dB_address;

	float *d_deltaA_buf = (float*)(intptr_t)d_deltaA_buf_address;
	float *d_deltaA = (float*)(intptr_t)d_deltaA_address;

	return __field_affine_deltaA_v1(stream, 
		d_deltaY, dY, dA, dB, N, M,
		d_deltaA_buf, d_deltaA,
		width, stride, partNum);
}

// Method:    field_affine_deltaAB_v1
JNIEXPORT jint JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1reduce_field_1affine_1deltaAB_1v1(JNIEnv *env, jclass cls,
	jlong stream1_address, jlong stream2_address,
	jlong d_deltaY_address, 
	jlong dY_address,//V1: holdY(), Y is not changed
	jlong dA_address, jlong dB_address,
	jint N, jint M, 
	jlong d_deltaA_buf_address, jlong d_deltaA_address,//deltaA = deltaXp2
	jlong d_deltaB_buf_address, jlong d_deltaB_address,//deltaB = deltaXp1
	jint width, jint stride,
	jint partNum)
{
	cudaStream_t stream1 = (cudaStream_t)(intptr_t)stream1_address;
	cudaStream_t stream2 = (cudaStream_t)(intptr_t)stream2_address;
	float *d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float *dY = (float*)(intptr_t)dY_address;
	float *dA = (float*)(intptr_t)dA_address;
	float *dB = (float*)(intptr_t)dB_address;

	float *d_deltaA_buf = (float*)(intptr_t)d_deltaA_buf_address;
	float *d_deltaA = (float*)(intptr_t)d_deltaA_address;
	float *d_deltaB_buf = (float*)(intptr_t)d_deltaB_buf_address;
	float *d_deltaB = (float*)(intptr_t)d_deltaB_address;

	return __field_affine_deltaAB_v1(env, stream1, stream2,
		d_deltaY, dY, dA, dB, N, M,
		d_deltaA_buf, d_deltaA,
		d_deltaB_buf, d_deltaB,
		width, stride, partNum);
}

//Method:    field_affine_deltaAB_v2
JNIEXPORT jint JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1reduce_field_1affine_1deltaAB_1v2(JNIEnv *env, jclass cls,
	jlong stream1_address, jlong stream2_address,
	jlong d_deltaY_address,
	jlong dX_address,//(V2: X for Affine || V1: Y for Norm)
	jint N, jint M,
	jlong d_deltaA_buf_address, jlong d_deltaA_address,
	jlong d_deltaB_buf_address, jlong d_deltaB_address,
	jint width, jint stride, jint partNum)
{
	cudaStream_t stream1 = (cudaStream_t)(intptr_t)stream1_address;
	cudaStream_t stream2 = (cudaStream_t)(intptr_t)stream2_address;
	float *d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float *dX = (float*)(intptr_t)dX_address;

	float *d_deltaA_buf = (float*)(intptr_t)d_deltaA_buf_address;
	float *d_deltaA = (float*)(intptr_t)d_deltaA_address;
	float *d_deltaB_buf = (float*)(intptr_t)d_deltaB_buf_address;
	float *d_deltaB = (float*)(intptr_t)d_deltaB_address;

	return __field_affine_deltaAB_v2(env, stream1, stream2,
		d_deltaY, dX, N, M,
		d_deltaA_buf, d_deltaA,
		d_deltaB_buf, d_deltaB, 
		width, stride, 1);
}

#endif


//sqBatchNorm.backward
#ifndef JNI_FIELD_SQUARE_BATCH_NORM_REDUCE
#define JNI_FIELD_SQUARE_BATCH_NORM_REDUCE

//Method:    field_sqBatchNorm_deltaA_v2
JNIEXPORT jint JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1reduce_field_1sqBatchNorm_1deltaA_1v2(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong d_deltaY_address, 
	jlong dX_address,//V2: holdX(), X is not changed
	jlong dX_mean_address, 
	jlong dX_sqmean_address, jfloat eps,
	jint N, jint M, 
	jlong d_deltaA_buf_address, jlong d_deltaA_address,//deltaA = deltaXp2
	jint width, jint stride,
	jint partNum)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float *dX = (float*)(intptr_t)dX_address;
	float *dX_mean = (float*)(intptr_t)dX_mean_address;
	float *dX_sqmean = (float*)(intptr_t)dX_sqmean_address;

	float *d_deltaA_buf = (float*)(intptr_t)d_deltaA_buf_address;
	float *d_deltaA = (float*)(intptr_t)d_deltaA_address;

	return __field_sqBatchNorm_deltaA_v2(stream,
		d_deltaY, dX, dX_mean, dX_sqmean, eps, N, M,
		d_deltaA_buf, d_deltaA,
		width, stride, partNum);
}

//Method:    field_sqBatchNorm_deltaAB_v2
JNIEXPORT jint JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1reduce_field_1sqBatchNorm_1deltaAB_1v2(JNIEnv *env, jclass cls,
	jlong stream1_address, jlong stream2_address,
	jlong d_deltaY_address, 
	jlong dX_address,//V2: holdX(), X is not changed
	jlong dX_mean_address,
	jlong dX_sqmean_address, jfloat eps,
	jint N, jint M,
	jlong d_deltaA_buf_address, jlong d_deltaA_address,//deltaA = deltaXp2
	jlong d_deltaB_buf_address, jlong d_deltaB_address,//deltaB = deltaXp1
	jint width, jint stride,
	jint partNum)
{
	cudaStream_t stream1 = (cudaStream_t)(intptr_t)stream1_address;
	cudaStream_t stream2 = (cudaStream_t)(intptr_t)stream2_address;

	float *d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float *dX = (float*)(intptr_t)dX_address;
	float *dX_mean = (float*)(intptr_t)dX_mean_address;
	float *dX_sqmean = (float*)(intptr_t)dX_sqmean_address;
	
	float *d_deltaA_buf = (float*)(intptr_t)d_deltaA_buf_address;
	float *d_deltaA = (float*)(intptr_t)d_deltaA_address;
	float *d_deltaB_buf = (float*)(intptr_t)d_deltaB_buf_address;
	float *d_deltaB = (float*)(intptr_t)d_deltaB_address;

	return __field_sqBatchNorm_deltaAB_v2(env, stream1, stream2,
		d_deltaY, dX, dX_mean, dX_sqmean, eps, N, M,
		d_deltaA_buf, d_deltaA,
		d_deltaB_buf, d_deltaB,
		width, stride, partNum);
}

#endif


//batchNorm.backward
#ifndef JNI_FIELD_BATCH_NORM_REDUCE
#define JNI_FIELD_BATCH_NORM_REDUCE

//Method:    field_batchNorm_deltaA_v2
JNIEXPORT jint JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1reduce_field_1batchNorm_1deltaA_1v2(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong d_deltaY_address,
	jlong dX_address,//V2: holdX(), X is not changed
	jlong dX_mean_address,
	jlong dX_var_address, jfloat eps,//deltaA = deltaXp2
	jint N, jint M, 
	jlong d_deltaA_buf_address, jlong d_deltaA_address,
	jint width, jint stride,
	jint partNum)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float *dX = (float*)(intptr_t)dX_address;
	float *dX_mean = (float*)(intptr_t)dX_mean_address;
	float *dX_var = (float*)(intptr_t)dX_var_address;

	float *d_deltaA_buf = (float*)(intptr_t)d_deltaA_buf_address;
	float *d_deltaA = (float*)(intptr_t)d_deltaA_address;

	return __field_batchNorm_deltaA_v2(stream,
		d_deltaY, dX, dX_mean, dX_var, eps, N, M,
		d_deltaA_buf, d_deltaA,
		width, stride, partNum);
}

//Method:    field_batchNorm_deltaAB_v2
JNIEXPORT jint JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1reduce_field_1batchNorm_1deltaAB_1v2(JNIEnv *env, jclass cls,
	jlong stream1_address, jlong stream2_address,
	jlong d_deltaY_address,
	jlong dX_address,//V2: holdX(), X is not changed
	jlong dX_mean_address,
	jlong dX_var_address, jfloat eps,
	jint N, jint M, 
	jlong d_deltaA_buf_address, jlong d_deltaA_address,//deltaA = deltaXp2
	jlong d_deltaB_buf_address, jlong d_deltaB_address,//deltaB = deltaXp1
	jint width, jint stride, 
	jint partNum)
{
	cudaStream_t stream1 = (cudaStream_t)(intptr_t)stream1_address;
	cudaStream_t stream2 = (cudaStream_t)(intptr_t)stream2_address;

	float *d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float *dX = (float*)(intptr_t)dX_address;
	float *dX_mean = (float*)(intptr_t)dX_mean_address;
	float *dX_var = (float*)(intptr_t)dX_var_address;

	float *d_deltaA_buf = (float*)(intptr_t)d_deltaA_buf_address;
	float *d_deltaA = (float*)(intptr_t)d_deltaA_address;
	float *d_deltaB_buf = (float*)(intptr_t)d_deltaB_buf_address;
	float *d_deltaB = (float*)(intptr_t)d_deltaB_address;

	return __field_batchNorm_deltaAB_v2(env, stream1, stream2,
		d_deltaY, dX, dX_mean, dX_var, eps, N, M,
		d_deltaA_buf, d_deltaA,
		d_deltaB_buf, d_deltaB,
		width, stride, partNum);
}

#endif


//layerNorm.backward
#ifndef JNI_FIELD_LAYER_NORM_REDUCE
#define JNI_FIELD_LAYER_NORM_REDUCE

// Method:    field_layerNorm_deltaA_v2
JNIEXPORT jint JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1reduce_field_1layerNorm_1deltaA_1v2(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong d_deltaY_address, jlong dX_address,//V2: holdX(), X is not changed
	jlong dX_mean_address, 
	jlong dX_square_mean_address, jfloat eps,
	jint N, jint M, 
	jlong d_deltaA_buf_address, 
	jlong d_deltaA_address,
	jint width, jint stride, 
	jint partNum)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float *dX = (float*)(intptr_t)dX_address;
	float *dX_mean = (float*)(intptr_t)dX_mean_address;
	float *dX_square_mean = (float*)(intptr_t)dX_square_mean_address;
	float *d_deltaA_buf = (float*)(intptr_t)d_deltaA_buf_address;
	float *d_deltaA = (float*)(intptr_t)d_deltaA_address;

	return __field_layernorm_deltaA_v2(stream, 
		d_deltaY, dX, dX_mean, dX_square_mean, eps, N, M,
		d_deltaA_buf, d_deltaA,
		width, stride, partNum);
}

//Method:    field_layerNorm_deltaAB_v2
JNIEXPORT jint JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1reduce_field_1layerNorm_1deltaAB_1v2(JNIEnv *env, jclass cls,
	jlong stream1_address, jlong stream2_address, 
	jlong d_deltaY_address, jlong dX_address,//V2: holdX(), X is not changed
	jlong dX_mean_address, 
	jlong dX_square_mean_address, jfloat eps,
	jint N, jint M,
	jlong d_deltaA_buf_address, jlong d_deltaA_address,
	jlong d_deltaB_buf_address, jlong d_deltaB_address,
	jint width, jint stride,
	jint partNum)
{
	cudaStream_t stream1 = (cudaStream_t)(intptr_t)stream1_address;
	cudaStream_t stream2 = (cudaStream_t)(intptr_t)stream2_address;

	float *d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float *dX = (float*)(intptr_t)dX_address;
	float *dX_mean = (float*)(intptr_t)dX_mean_address;
	float *dX_square_mean = (float*)(intptr_t)dX_square_mean_address;

	float *d_deltaA_buf = (float*)(intptr_t)d_deltaA_buf_address;
	float *d_deltaA = (float*)(intptr_t)d_deltaA_address;
	float *d_deltaB_buf = (float*)(intptr_t)d_deltaB_buf_address;
	float *d_deltaB = (float*)(intptr_t)d_deltaB_address;

	return __field_layernorm_deltaAB_v2(env, stream1, stream2,
		d_deltaY, dX, dX_mean, dX_square_mean, eps,
		N, M,
		d_deltaA_buf, d_deltaA,
		d_deltaB_buf, d_deltaB,
		width, stride, partNum);
}

#endif


//------[row reduce area]----------------------------------------
#ifndef JNI_ROW_REDUCE
#define JNI_ROW_REDUCE

//Method:    row_linear
JNIEXPORT jint JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1reduce_row_1linear(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dX_address, 
	jfloat alpha, jfloat beta,
	jint N, jint M,
	jlong dV_address, jlong dY_address,
	jint width, jint stride,
	jint partNum)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)dX_address;
	float *dV = (float*)(intptr_t)dV_address;
	float *dY = (float*)(intptr_t)dY_address;
	return __row_linear(stream, dX, alpha, beta, N, M, dV, dY, 
		width, stride, partNum);
}

//Method:    row_linear_dual
JNIEXPORT jint JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1reduce_row_1linear_1dual(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dX1_address,
	jlong dX2_address, 
	jfloat alpha, jfloat beta, jfloat gamma, 
	jint N, jint M,
	jlong dV_address, jlong dY_address,
	jint width, jint stride, 
	jint partNum)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX1 = (float*)(intptr_t)dX1_address;
	float *dX2 = (float*)(intptr_t)dX2_address;
	float *dV = (float*)(intptr_t)dV_address;
	float *dY = (float*)(intptr_t)dY_address;
	return __row_linear_dual(stream, dX1, dX2, alpha, beta, gamma, N, M, dV, dY, 
		width, stride, partNum);
}

//Method:    row_quadratic
JNIEXPORT jint JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1reduce_row_1quadratic(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dX_address,
	jfloat alpha, jfloat beta, jfloat gamma,
	jint N, jint M,
	jlong dV_address, jlong dY_address,
	jint width, jint stride,
	jint partNum)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)dX_address;
	float *dV = (float*)(intptr_t)dV_address;
	float *dY = (float*)(intptr_t)dY_address;
	return __row_quadratic(stream, dX, alpha, beta, gamma, N, M, dV, dY,
		width, stride, partNum);
}

//Method:    row_quadratic_dual
JNIEXPORT jint JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1reduce_row_1quadratic_1dual(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dX1_address,
	jlong dX2_address, 
	jfloat k11, jfloat k12, jfloat k22,
	jfloat k1, jfloat k2, jfloat C,
	jint N, jint M,
	jlong dV_address, jlong dY_address,
	jint width, jint stride,
	jint partNum)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX1 = (float*)(intptr_t)dX1_address;
	float *dX2 = (float*)(intptr_t)dX2_address;
	float *dV = (float*)(intptr_t)dV_address;
	float *dY = (float*)(intptr_t)dY_address;
	return __row_quadratic_dual(stream, dX1, dX2,
		k11, k12, k22, k1, k2, C,
		N, M, dV, dY, 
		width, stride, partNum);
}

//Method:    row_linear_quadratic
JNIEXPORT jint JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1reduce_row_1linear_1quadratic(JNIEnv *env, jclass cls,
	jlong stream1_address, jlong stream2_address,
	jlong dX_address,
	jfloat alpha1, jfloat beta1,
	jfloat alpha2, jfloat beta2, jfloat gamma2,
	jint N, jint M,
	jlong dV1_address, jlong dY1_address,
	jlong dV2_address, jlong dY2_address,
	jint width, jint stride,
	jint partNum)
{
	cudaStream_t stream1 = (cudaStream_t)(intptr_t)stream1_address;
	cudaStream_t stream2 = (cudaStream_t)(intptr_t)stream2_address;

	float* dX = (float*)(intptr_t)dX_address;
	float* dV1 = (float*)(intptr_t)dV1_address;
	float* dV2 = (float*)(intptr_t)dV2_address;
	float* dY1 = (float*)(intptr_t)dY1_address;
	float* dY2 = (float*)(intptr_t)dY2_address;
	
	return __row_linear_quadratic(env, stream1, stream2,
		dX, alpha1, beta1, alpha2, beta2, gamma2, N, M,
		dV1, dY1, dV2, dY2, 
		width, stride, 1);
}

//Method:    row_max
JNIEXPORT jint JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1reduce_row_1max(JNIEnv *env, jclass cls, 
	jlong stream_address, 
	jlong dX_address,
	jint N, jint M, 
	jlong dV_address, jlong dY_address, 
	jint width, jint stride,
	jint partNum)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)dX_address;
	float *dV = (float*)(intptr_t)dV_address;
	float *dY = (float*)(intptr_t)dY_address;
	return __row_max(stream, dX, N, M, dV, dY,
		width, stride, partNum);
}

//Method:    row_min
JNIEXPORT jint JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1reduce_row_1min(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dX_address,
	jint N, jint M,
	jlong dV_address, jlong dY_address,
	jint width, jint stride,
	jint partNum)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)dX_address;
	float *dV = (float*)(intptr_t)dV_address;
	float *dY = (float*)(intptr_t)dY_address;
	return __row_min(stream, dX, N, M, dV, dY,
		width, stride, partNum);
}

//Method:    row_max_indexed
JNIEXPORT jint JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1reduce_row_1max_1indexed(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dX_address, 
	jint N, jint M,
	jlong dV_address, jlong dVIndex_address,
	jlong dY_address, jlong dIndex_address,
	jint width, jint stride, 
	jint partNum)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)dX_address;
	float *dV = (float*)(intptr_t)dV_address;
	float *dY = (float*)(intptr_t)dY_address;
	int *dVIndex = (int*)(intptr_t)dVIndex_address;
	int *dIndex = (int*)(intptr_t)dIndex_address;
	return __row_max_indexed(stream, dX, N, M, dV, dVIndex, dY, dIndex,
		width, stride, partNum);
}

//Method:    row_min_indexed
JNIEXPORT jint JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1reduce_row_1min_1indexed(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dX_address,
	jint N, jint M,
	jlong dV_address, jlong dVIndex_address,
	jlong dY_address, jlong dIndex_address,
	jint width, jint stride, jint partNum)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)dX_address;
	float *dV = (float*)(intptr_t)dV_address;
	float *dY = (float*)(intptr_t)dY_address;
	int *dVIndex = (int*)(intptr_t)dVIndex_address;
	int *dIndex = (int*)(intptr_t)dIndex_address;
	return __row_min_indexed(stream, dX, N, M, dV, dVIndex, dY, dIndex,
		width, stride, partNum);
}

#endif


#ifndef JNI_ROW_REDUCE_SOFTMAX
#define JNI_ROW_REDUCE_SOFTMAX

//Method:    row_softmax
JNIEXPORT jint JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1reduce_row_1softmax(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dX_address, jlong d_maxX_address,
	jlong d_expX_address,
	jint N, jint M,
	jlong dV_address, 
	jint width, jint stride, 
	jint partNum)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)dX_address;
	float *d_maxX = (float*)(intptr_t)d_maxX_address;
	float *d_expX = (float*)(intptr_t)d_expX_address;
	float *dV = (float*)(intptr_t)dV_address;
	return __row_softmax(stream, dX, d_maxX, d_expX, N, M, dV, 
		width, stride, partNum);
}

//Method:    row_softmaxCrossEntropy_stage1
JNIEXPORT jint JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1reduce_row_1softmaxCrossEntropy_1stage1(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dX_address, jlong d_maxX_address,
	jint N, jint M, 
	jlong dV_address,
	jint width, jint stride,
	jint partNum) 
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)dX_address;
	float *d_maxX = (float*)(intptr_t)d_maxX_address;
	float *dV = (float*)(intptr_t)dV_address;
	return __row_softmax_crossEntropy_stage1(stream, dX, d_maxX, N, M, dV,
		width, stride, partNum);
}

#endif


#ifndef JNI_ROW_REDUCE_LAYER_NORM
#define JNI_ROW_REDUCE_LAYER_NORM

// Method:    row_layernorm_deltaXp_v1
JNIEXPORT jint JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1reduce_row_1layernorm_1deltaXp_1v1(JNIEnv *env, jclass cls,
	jlong stream1_address, jlong stream2_address,
	jlong d_deltaY_address, jlong dY_address,//V1: holdY(), Y is not changed
	jlong dX_mean_address, 
	jlong dX_square_mean_address, jfloat eps,
	jint N, jint M,
	jlong d_deltaXp1_address,
	jlong d_deltaXp2_address,
	jint width, jint stride, 
	jint partNum)
{
	cudaStream_t stream1 = (cudaStream_t)(intptr_t)stream1_address;
	cudaStream_t stream2 = (cudaStream_t)(intptr_t)stream2_address;

	float* d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float* dY = (float*)(intptr_t)dY_address;
	float* dX_mean = (float*)(intptr_t)dX_mean_address;
	float* dX_square_mean = (float*)(intptr_t)dX_square_mean_address;
	float* d_deltaXp1 = (float*)(intptr_t)d_deltaXp1_address;
	float* d_deltaXp2 = (float*)(intptr_t)d_deltaXp2_address;

	return __row_layernorm_deltaXp_v1(env, stream1, stream2,
		d_deltaY, dY,
		dX_mean, dX_square_mean, eps,
		N, M,
		d_deltaXp1, d_deltaXp2,
		width, stride, partNum);
}

//Method:    row_layernorm_affined_deltaXp_v1
JNIEXPORT jint JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1reduce_row_1layernorm_1affined_1deltaXp_1v1(JNIEnv *env, jclass cls,
	jlong stream1_address, jlong stream2_address,
	jlong d_deltaY_address, jlong dY_address,//V1: holdY(), Y is not changed
	jlong dX_mean_address, jlong dX_square_mean_address, jfloat eps,
	jlong dA_address, jlong dB_address,
	jint N, jint M,
	jlong d_deltaXp1_address,
	jlong d_deltaXp2_address, 
	jint width, jint stride, jint partNum)
{
	cudaStream_t stream1 = (cudaStream_t)(intptr_t)stream1_address;
	cudaStream_t stream2 = (cudaStream_t)(intptr_t)stream2_address;

	float* d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float* dY = (float*)(intptr_t)dY_address;
	float* dX_mean = (float*)(intptr_t)dX_mean_address;
	float* dX_square_mean = (float*)(intptr_t)dX_square_mean_address;
	float* dA = (float*)(intptr_t)dA_address;
	float* dB = (float*)(intptr_t)dB_address;
	float* d_deltaXp1 = (float*)(intptr_t)d_deltaXp1_address;
	float* d_deltaXp2 = (float*)(intptr_t)d_deltaXp2_address;

	return __row_layernorm_affined_deltaXp_v1(env, stream1, stream2,
		d_deltaY, dY,
		dX_mean, dX_square_mean, eps, dA, dB,
		N, M,
		d_deltaXp1, d_deltaXp2,
		width, stride, partNum);
}

//Method:    row_layernorm_deltaXp_v2
JNIEXPORT jint JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1reduce_row_1layernorm_1deltaXp_1v2(JNIEnv *env, jclass cls,
	jlong stream1_address, jlong stream2_address,
	jlong d_deltaY_address, jlong dX_address,//V2: holdX(), X is not changed
	jlong dX_mean_address, 
	jlong dX_square_mean_address, jfloat eps,
	jint N, jint M,
	jlong d_deltaXp1_address, jlong d_deltaXp2_address,
	jint width, jint stride,
	jint partNum)
{
	cudaStream_t stream1 = (cudaStream_t)(intptr_t)stream1_address;
	cudaStream_t stream2 = (cudaStream_t)(intptr_t)stream2_address;

	float* d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float* dX = (float*)(intptr_t)dX_address;
	float* dX_mean = (float*)(intptr_t)dX_mean_address;
	float* dX_square_mean = (float*)(intptr_t)dX_square_mean_address;
	float* d_deltaXp1 = (float*)(intptr_t)d_deltaXp1_address;
	float* d_deltaXp2 = (float*)(intptr_t)d_deltaXp2_address;

	return __row_layernorm_deltaXp_v2(env, stream1, stream2,
		d_deltaY, dX, 
		dX_mean, dX_square_mean, eps,
		N, M,
		d_deltaXp1, d_deltaXp2, 
		width, stride, partNum);
}

//Method:    row_layernorm_affined_deltaXp_v2
JNIEXPORT jint JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1reduce_row_1layernorm_1affined_1deltaXp_1v2(JNIEnv *env, jclass cls,
	jlong stream1_address, jlong stream2_address,
	jlong d_deltaY_address, jlong dX_address,
	jlong dX_mean_address, jlong dX_square_mean_address, jfloat eps,
	jlong dA_address,
	jint N, jint M, 
	jlong d_deltaXp1_address, jlong d_deltaXp2_address,
	jint width, jint stride,
	jint partNum)
{
	cudaStream_t stream1 = (cudaStream_t)(intptr_t)stream1_address;
	cudaStream_t stream2 = (cudaStream_t)(intptr_t)stream2_address;

	float* d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float* dX = (float*)(intptr_t)dX_address;
	float* dX_mean = (float*)(intptr_t)dX_mean_address;
	float* dX_square_mean = (float*)(intptr_t)dX_square_mean_address;
	float* dA = (float*)(intptr_t)dA_address;
	float* d_deltaXp1 = (float*)(intptr_t)d_deltaXp1_address;
	float* d_deltaXp2 = (float*)(intptr_t)d_deltaXp2_address;

	return __row_layernorm_affined_deltaXp_v2(env, stream1, stream2,
		d_deltaY, dX,
		dX_mean, dX_square_mean, eps, dA,
		N, M,
		d_deltaXp1, d_deltaXp2,
		width, stride, 1);
}

#endif
