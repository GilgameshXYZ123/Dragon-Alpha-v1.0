#include "frame.cuh"
#include "Cuda_function.cuh"
#include "JNITool.cuh"
#include "micro.cuh"
#include "Afunction.cuh"
#include "test.cuh"


//linear, linear_dual, rpl, div, quadratic, quadratic_dual
#ifndef JNI_TYPE1
#define JNI_TYPE1

#ifndef JNI_COMPARE
#define JNI_COMPARE

//Method:    equal_abs2D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_equal_1abs2D(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dX1_address, jlong dX2_address,
	jfloat min, jfloat max,
	jlong dY_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX1 = (float*)(intptr_t)dX1_address;
	float *dX2 = (float*)(intptr_t)dX2_address;
	float *dY = (float*)(intptr_t)dY_address;
	__equal_abs2D(stream, dX1, dX2, min, max, dY,
		lengthv, width, stride);
}

//Method:    equal_abs2D_char
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_equal_1abs2D_1char(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dX1_address, jlong dX2_address,
	jbyte min, jbyte max,
	jlong dY_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	char *dX1 = (char*)(intptr_t)dX1_address;
	char *dX2 = (char*)(intptr_t)dX2_address;
	float *dY = (float*)(intptr_t)dY_address;
	__equal_abs2D_char(stream, dX1, dX2, min, max, dY, 
		lengthv, width, stride);
}

//Method:    equal_abs2D_int
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_equal_1abs2D_1int(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dX1_address, jlong dX2_address,
	jint min, jint max,
	jlong dY_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	int *dX1 = (int*)(intptr_t)dX1_address;
	int *dX2 = (int*)(intptr_t)dX2_address;
	float *dY = (float*)(intptr_t)dY_address;
	__equal_abs2D_int(stream, dX1, dX2, min, max, dY, 
		lengthv, width, stride);
}

//Method:    linear_greater2D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_linear_1greater2D(JNIEnv *env, jclass cls, 
	jlong stream_address,
	jfloat alpha, jlong dX_address, jfloat beta,
	jlong dY_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float* dX = (float*)(intptr_t)dX_address;
	float* dY = (float*)(intptr_t)dY_address;
	__linear_greater2D(stream, alpha, dX, beta, dY, 
		lengthv, width, stride);

}

//Method:    linear_greater_dual2D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_linear_1greater_1dual2D(JNIEnv *env, jclass cls,
	jlong stream_address, 
	jlong dX1_address, jlong dX2_address,
	jfloat alpha, jfloat beta, jfloat gamma,
	jlong dY_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float* dX1 = (float*)(intptr_t)dX1_address;
	float* dX2 = (float*)(intptr_t)dX2_address;
	float* dY = (float*)(intptr_t)dY_address;
	__linear_greater_dual2D(stream, dX1, dX2, alpha, beta, gamma, dY,
		lengthv, width, stride);
}

#endif


#ifndef JNI_LINEAR
#define JNI_LINEAR

//Method:    linear2D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_linear2D(JNIEnv *env, jclass cls,
	jlong stream_address,
	jfloat alpha, jlong dX_address, jfloat beta,
	jlong dY_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)dX_address;
	float *dY = (float*)(intptr_t)dY_address;
	__linear2D(stream, alpha, dX, beta, dY, 
		lengthv, width, stride);
}


//Method:    linear_dual_out2D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_linear_1dual_1out2D(JNIEnv *env, jclass cls, 
	jlong stream_address, 
	jlong dX_address, 
	jfloat alpha1, jfloat beta1,
	jfloat alpha2, jfloat beta2,
	jlong dY1_address, jlong dY2_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)dX_address;
	float *dY1 = (float*)(intptr_t)dY1_address;
	float *dY2 = (float*)(intptr_t)dY2_address;
	__linear_dual_out2D(stream, dX, alpha1, beta1, alpha2, beta2, dY1, dY2, 
		lengthv, width, stride);
}


//Method:    linear2D_char2float
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_linear2D_1char2float(JNIEnv *env, jclass cls,
	jlong stream_address,
	jfloat alpha, jlong dX_address, jfloat beta,
	jlong dY_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	char  *dX =  (char*)(intptr_t)dX_address;
	float *dY = (float*)(intptr_t)dY_address;
	__linear2D_char2float(stream, alpha, dX, beta, dY,
		lengthv, width, stride);
}

//Method:    linear2D_float2char
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_linear2D_1float2char(JNIEnv *env, jclass cls,
	jlong stream_address,
	jfloat alpha, jlong dX_address, jfloat beta, 
	jlong dY_address, 
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)dX_address;
	char  *dY =  (char*)(intptr_t)dY_address;
	__linear2D_float2char(stream, alpha, dX, beta, dY, 
		lengthv, width, stride);
}

//Method:    linear2D_int2float
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_linear2D_1int2float(JNIEnv *env, jclass cls,
	jlong stream_address,
	jfloat alpha, jlong dX_address, jfloat beta,
	jlong dY_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	int   *dX =   (int*)(intptr_t)dX_address;
	float *dY = (float*)(intptr_t)dY_address;
	__linear2D_int2float(stream, alpha, dX, beta, dY,
		lengthv, width, stride);
}

//Method:    linear2D_float2int
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_linear2D_1float2int(JNIEnv *env, jclass cls,
	jlong stream_address,
	jfloat alpha, jlong dX_address, jfloat beta,
	jlong dY_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)dX_address;
	int   *dY =   (int*)(intptr_t)dY_address;
	__linear2D_float2int(stream, alpha, dX, beta, dY,
		lengthv, width, stride);
}

#endif


#ifndef JNI_LINEAR_DUAL
#define JNI_LINEAR_DUAL

//Method:    linear_dual2D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_linear_1dual2D(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dX1_address,
	jlong dX2_address, 
	jfloat alpha, jfloat beta, jfloat gamma,
	jlong dY_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX1 = (float*)(intptr_t)dX1_address;
	float *dX2 = (float*)(intptr_t)dX2_address;
	float *dY = (float*)(intptr_t)dY_address;

	__linear_dual2D(stream, dX1, dX2, alpha, beta, gamma, dY,
		lengthv, width, stride);
}

//Method:    linear_dual2D_row
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_linear_1dual2D_1row(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dX1_address,
	jlong dX2_address, jint row_lengthv,
	jfloat alpha, jfloat beta, jfloat gamma,
	jlong dY_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX1 = (float*)(intptr_t)dX1_address;
	float *dX2 = (float*)(intptr_t)dX2_address;
	float *dY = (float*)(intptr_t)dY_address;

	__linear_dual2D_row(stream, dX1, dX2, row_lengthv, alpha, beta, gamma, dY,
		lengthv, width, stride);
}

//Method:    linear_dual2D_field
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_linear_1dual2D_1field(JNIEnv *env, jclass cls,
	jlong stream_address, 
	jlong dX1_address,
	jlong dX2_address, jint row_lengthv,
	jfloat alpha, jfloat beta, jfloat gamma,
	jlong dY_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX1 = (float*)(intptr_t)dX1_address;
	float *dX2 = (float*)(intptr_t)dX2_address;
	float *dY = (float*)(intptr_t)dY_address;

	__linear_dual2D_field(stream, dX1, dX2, row_lengthv, alpha, beta, gamma, dY,
		lengthv, width, stride);
}

#endif


#ifndef JNI_QUADRATIC
#define JNI_QUADRATIC

//Method:    quadratic2D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_quadratic2D(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dX_address, 
	jfloat alpha, jfloat beta, jfloat gamma,
	jlong dY_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)dX_address;
	float *dY = (float*)(intptr_t)dY_address;
	__quadratic2D(stream, dX, alpha, beta, gamma, dY, lengthv, width, stride);
}

//Method:    quadratic2D_deltaX
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_quadratic2D_1deltaX(JNIEnv *env, jclass cls,
	jlong stream_address, 
	jlong d_deltaX_address,
	jlong d_deltaY_address,
	jlong dX_address, jfloat alpha, jfloat beta,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *d_deltaX = (float*)(intptr_t)d_deltaX_address;
	float *d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float *dX = (float*)(intptr_t)dX_address;
	__quadratic2D_deltaX(stream, d_deltaX, d_deltaY, dX, alpha, beta, lengthv, width, stride);
}
#endif


#ifndef JNI_QUADRATIC_DUAL
#define JNI_QUADRATIC_DUAL

//Method:    quadratic_dual2D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_quadratic_1dual2D(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dX1_address,
	jlong dX2_address,
	jfloat k11, jfloat k12, jfloat k22,
	jfloat k1, jfloat k2,
	jfloat C,
	jlong dY_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX1 = (float*)(intptr_t)dX1_address;
	float *dX2 = (float*)(intptr_t)dX2_address;
	float *dY = (float*)(intptr_t)dY_address;
	__quadratic_dual2D(stream, dX1, dX2, k11, k12, k22, k1, k2, C, dY, lengthv, width, stride);
}

//Method:    quadratic_dual2D_deltaX
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_quadratic_1dual2D_1deltaX(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong d_deltaX1_address, jlong d_deltaX2_address,
	jlong d_deltaY_address,
	jlong dX1_address, jlong dX2_address,
	jfloat k11, jfloat k12, jfloat k22,
	jfloat k1, jfloat k2,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *d_deltaX1 = (float*)(intptr_t)d_deltaX1_address;
	float *d_deltaX2 = (float*)(intptr_t)d_deltaX2_address;
	float *d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float *dX1 = (float*)(intptr_t)dX1_address;
	float *dX2 = (float*)(intptr_t)dX2_address;
	__quadratic_dual2D_deltaX(stream, d_deltaX1, d_deltaX2, d_deltaY,
		dX1, dX2, k11, k12, k22, k1, k2,
		lengthv, width, stride);
}


//Method:    quadratic_dual2D_row
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_quadratic_1dual2D_1row(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dX1_address,
	jlong dX2_address, jint X2_lengthv,
	jfloat k11, jfloat k12, jfloat k22,
	jfloat k1, jfloat k2,
	jfloat C,
	jlong dY_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX1 = (float*)(intptr_t)dX1_address;
	float *dX2 = (float*)(intptr_t)dX2_address;
	float *dY = (float*)(intptr_t)dY_address;
	__quadratic_dual2D_row(stream, dX1, dX2, X2_lengthv, k11, k12, k22, k1, k2, C, dY,
		lengthv, width, stride);
}

//Method:    quadratic_dual2D_field
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_quadratic_1dual2D_1field(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dX1_address,
	jlong dX2_address, jint row_lengthv,
	jfloat k11, jfloat k12, jfloat k22,
	jfloat k1, jfloat k2,
	jfloat C,
	jlong dY_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX1 = (float*)(intptr_t)dX1_address;
	float *dX2 = (float*)(intptr_t)dX2_address;
	float *dY = (float*)(intptr_t)dY_address;
	__quadratic_dual2D_field(stream, dX1, dX2, row_lengthv, 
		k11, k12, k22, k1, k2, C, dY,
		lengthv, width, stride);
}

//Method:    variance2D_f64
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_variance2D_1f64(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dX_mean_address,
	jlong dX_sqmean_address,
	jlong dX_var_address, 
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float* dX_mean = (float*)(intptr_t)dX_mean_address;
	float* dX_sqmean = (float*)(intptr_t)dX_sqmean_address;
	float* dX_var = (float*)(intptr_t)dX_var_address;
	__variance2D_f64(stream, dX_mean, dX_sqmean, dX_var,
		lengthv, width, stride);
}

#endif


#ifndef JNI_RPL
#define JNI_RPL

//Method:    rpl2D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_rpl2D(JNIEnv *env, jclass cls,
	jlong stream_address,
	jfloat alpha, jlong dX_address, jfloat beta, jfloat gamma,
	jlong dY_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)dX_address;
	float *dY = (float*)(intptr_t)dY_address;
	__rpl2D(stream, alpha, dX, beta, gamma, dY, lengthv, width, stride);
}

//Method:    rpl2D_deltaX
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_rpl2D_1deltaX(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong d_deltaX_address,
	jlong d_deltaY_address,
	jlong dY_address, jfloat alpha, jfloat gamma,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dY = (float*)(intptr_t)dY_address;
	float *d_deltaX = (float*)(intptr_t)d_deltaX_address;
	float *d_deltaY = (float*)(intptr_t)d_deltaY_address;
	__rpl2D_deltaX(stream, d_deltaX, d_deltaY, dY, alpha, gamma, lengthv, width, stride);
}

#endif


#ifndef JNI_DIV
#define JNI_DIV

//Method:    div2D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_div2D(JNIEnv *env, jclass cls,
	jlong stream_address,
	jfloat alpha1, jlong dX1_address, jfloat beta1, 
	jfloat alpha2, jlong dX2_address, jfloat beta2,
	jfloat gamma,
	jlong dY_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX1 = (float*)(intptr_t)dX1_address;
	float *dX2 = (float*)(intptr_t)dX2_address;
	float *dY = (float*)(intptr_t)dY_address;
	__div2D(stream, alpha1, dX1, beta1, alpha2, dX2, beta2, gamma, dY,
		lengthv, width, stride);
}

//Method:    div2D_deltaX
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_div2D_1deltaX(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong d_deltaX1_address, jlong d_deltaX2_address,
	jlong d_deltaY_address,
	jlong dX1_address, jfloat alpha1, jfloat beta1,
	jlong dX2_address, jfloat alpha2, jfloat beta2,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *d_deltaX1 = (float*)(intptr_t)d_deltaX1_address;
	float *d_deltaX2 = (float*)(intptr_t)d_deltaX2_address;
	float *d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float *dX1 = (float*)(intptr_t)dX1_address;
	float *dX2 = (float*)(intptr_t)dX2_address;
	__div2D_deltaX(stream, d_deltaX1, d_deltaX2, d_deltaY, 
		dX1, alpha1, beta1, 
		dX2, alpha2, beta2, 
		lengthv, width, stride);
}

#endif


#ifndef JNI_DIV_ROW_FIELD
#define JNI_DIV_ROW_FIELD

//Method:    div2D_row
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_div2D_1row(JNIEnv *env, jclass cls,
	jlong stream_address,
	jfloat alpha1, jlong dX1_address, jfloat beta1,
	jfloat alpha2, jlong dX2_address, jfloat beta2,
	jfloat gamma, jint row_lengthv,
	jlong dY_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX1 = (float*)(intptr_t)dX1_address;
	float *dX2 = (float*)(intptr_t)dX2_address;
	float *dY = (float*)(intptr_t)dY_address;
	__div2D_row(stream, alpha1, dX1, beta1, alpha2, dX2, beta2, gamma, row_lengthv, dY,
		lengthv, width, stride);
}

//Method:    div2D_field
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_div2D_1field(JNIEnv *env, jclass cls,
	jlong stream_address,
	jfloat alpha1, jlong dX1_address, jfloat beta1,
	jfloat alpha2, jlong dX2_address, jfloat beta2,
	jfloat gamma, jint row_lengthv,
	jlong dY_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX1 = (float*)(intptr_t)dX1_address;
	float *dX2 = (float*)(intptr_t)dX2_address;
	float *dY = (float*)(intptr_t)dY_address;
	__div2D_field(stream, alpha1, dX1, beta1, alpha2, dX2, beta2, gamma, row_lengthv, dY,
		lengthv, width, stride);
}

#endif


#ifndef JNI_ADD_DIV_ROW_FIELD
#define JNI_ADD_DIV_ROW_FIELD

//Method:    add_div2D_row
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_add_1div2D_1row(JNIEnv *env, jclass cls,
	jlong stream_address, 
	jlong dX1_address,
	jlong dX2_address,
	jlong dX3_address, jint row_lengthv,
	jfloat alpha, jfloat beta, jfloat gamma, jfloat delta,
	jlong dY_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX1 = (float*)(intptr_t)dX1_address;
	float *dX2 = (float*)(intptr_t)dX2_address;
	float *dX3 = (float*)(intptr_t)dX3_address;
	float *dY = (float*)(intptr_t)dY_address;
	__add_div2D_row(stream, dX1, dX2, dX3, row_lengthv,
		alpha, beta, gamma, delta, dY,
		lengthv, width, stride);
}

//Method:    add_div2D_field
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_add_1div2D_1field(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dX1_address,
	jlong dX2_address,
	jlong dX3_address, jint row_lengthv,
	jfloat alpha, jfloat beta, jfloat gamma, jfloat delta,
	jlong dY_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX1 = (float*)(intptr_t)dX1_address;
	float *dX2 = (float*)(intptr_t)dX2_address;
	float *dX3 = (float*)(intptr_t)dX3_address;
	float *dY = (float*)(intptr_t)dY_address;
	__add_div2D_field(stream, dX1, dX2, dX3, row_lengthv,
		alpha, beta, gamma, delta, dY,
		lengthv, width, stride);
}

#endif

#endif


//sign, ceil, floor, abs, zero_nan, sqrt
#ifndef JNI_TYPE2
#define JNI_TYPE2

//Method:    sign2D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_sign2D(JNIEnv *env, jclass cls,
	jlong stream_address,
	jfloat alpha, jlong dX_address, jfloat beta, 
	jlong dY_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)dX_address;
	float *dY = (float*)(intptr_t)dY_address;
	__sign2D(stream, alpha, dX, beta, dY, lengthv, width, stride);
}

//Method:    ceil2D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_ceil2D(JNIEnv *env, jclass cls,
	jlong stream_address,
	jfloat alpha, jlong dX_address, jfloat beta,
	jlong dY_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)dX_address;
	float *dY = (float*)(intptr_t)dY_address;
	__ceil2D(stream, alpha, dX, beta, dY, lengthv, width, stride);
}

//Method:    floor2D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_floor2D(JNIEnv *env, jclass cls,
	jlong stream_address,
	jfloat alpha, jlong dX_address, jfloat beta,
	jlong dY_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)dX_address;
	float *dY = (float*)(intptr_t)dY_address;
	__floor2D(stream, alpha, dX, beta, dY, lengthv, width, stride);
}

#ifndef JNI_ABS
#define JNI_ABS

//Method:    abs2D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_abs2D(JNIEnv *env, jclass cls,
	jlong stream_address,
	jfloat alpha, jlong dX_address, jfloat beta, 
	jlong dY_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)dX_address;
	float *dY = (float*)(intptr_t)dY_address;
	__abs2D(stream, alpha, dX, beta, dY, lengthv, width, stride);
}

//Method:    abs2D_deltaX
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_abs2D_1deltaX(JNIEnv *env, jclass cls,
	jlong stream_address, 
	jlong d_deltaX_address,
	jlong d_deltaY_address,
	jlong dX_address, jfloat alpha, jfloat beta,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)dX_address;
	float *d_deltaX = (float*)(intptr_t)d_deltaX_address;
	float *d_deltaY = (float*)(intptr_t)d_deltaY_address;
	__abs2D_deltaX(stream, d_deltaX, d_deltaY, dX, alpha, beta,
		lengthv, width, stride);
}

#endif

//Method:    zero_nan2D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_zero_1nan2D(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dX_address,
	jlong dY_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)dX_address;
	float *dY = (float*)(intptr_t)dY_address;
	__zero_nan2D(stream, dX, dY, lengthv, width, stride);
}

//Method:    sqrt2D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_sqrt2D(JNIEnv *env, jclass cls,
	jlong stream_address,
	jfloat alpha, jlong dX_address, jfloat beta,
	jlong dY_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)dX_address;
	float *dY = (float*)(intptr_t)dY_address;
	__sqrt2D(stream, alpha, dX, beta, dY, lengthv, width, stride);
}

//Method:    sqrt_quadratic_dual2D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_sqrt_1quadratic_1dual2D(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dX1_address, jlong dX2_address,
	jfloat k11, jfloat k12, jfloat k22, 
	jfloat k1, jfloat k2, 
	jfloat C,
	jlong dY_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX1 = (float*)(intptr_t)dX1_address;
	float *dX2 = (float*)(intptr_t)dX2_address;
	float *dY = (float*)(intptr_t)dY_address;
	__sqrt_quadratic_dual2D(stream, dX1, dX2,
		k11, k12, k22, k1, k2, C, dY,
		lengthv, width, stride);
}

#endif


//min, min_dual, max, max_dual, clip
#ifndef JNI_TYPE3
#define JNI_TYPE3

#ifndef JNI_MIN_MIN_DUAL
#define JNI_MIN_MIN_DUAL

//Method:    min2D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_min2D(JNIEnv *env, jclass cls,
	jlong stream_address, 
	jfloat alpha, jlong dX_address, jfloat beta, 
	jfloat vmin,
	jlong dY_address, 
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)dX_address;
	float *dY = (float*)(intptr_t)dY_address;
	__min2D(stream, alpha, dX, beta, vmin, dY, lengthv, width, stride);
}

//Method:    min_dual2D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_min_1dual2D(JNIEnv *env, jclass cls,
	jlong stream_address,
	jfloat alpha1, jlong dX1_address, jfloat beta1,
	jfloat alpha2, jlong dX2_address, jfloat beta2,
	jlong dY_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX1 = (float*)(intptr_t)dX1_address;
	float *dX2 = (float*)(intptr_t)dX2_address;
	float *dY = (float*)(intptr_t)dY_address;
	__min_dual2D(stream, alpha1, dX1, beta1, alpha2, dX2, beta2, dY,
		lengthv, width, stride);
}

#endif


#ifndef JNI_MAX_MAX_DUAL
#define JNI_MAX_MAX_DUAL

//Method:    max2D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_max2D(JNIEnv *env, jclass cls,
	jlong stream_address,
	jfloat alpha, jlong dX_address, jfloat beta,
	jfloat vmax,
	jlong dY_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)dX_address;
	float *dY = (float*)(intptr_t)dY_address;
	__max2D(stream, alpha, dX, beta, vmax, dY, lengthv, width, stride);
}

//Method:    max_dual2D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_max_1dual2D(JNIEnv *env, jclass cls,
	jlong stream_address,
	jfloat alpha1, jlong dX1_address, jfloat beta1, 
	jfloat alpha2, jlong dX2_address, jfloat beta2,
	jlong dY_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX1 = (float*)(intptr_t)dX1_address;
	float *dX2 = (float*)(intptr_t)dX2_address;
	float *dY = (float*)(intptr_t)dY_address;
	__max_dual2D(stream, alpha1, dX1, beta1, alpha2, dX2, beta2, dY,
		lengthv, width, stride);
}

#endif

//Method:    clip2D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_clip2D(JNIEnv *env, jclass cls,
	jlong stream_address,
	jfloat alpha, jlong dX_address, jfloat beta,
	jfloat vmin, jfloat vmax, 
	jlong dY_address, 
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)dX_address;
	float *dY = (float*)(intptr_t)dY_address;
	__clip2D(stream, alpha, dX, beta, vmin, vmax, dY, lengthv, width, stride);
}

#endif


//semi-linear unit functions
#ifndef JNI_TYPE4_SEMI_LINEAR
#define JNI_TYPE4_SEMI_LINEAR

#ifndef JNI_LOG_EXP
#define JNI_LOG_EXP

//Method:    exp2D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_exp2D(JNIEnv *env, jclass cls,
	jlong stream_address, 
	jfloat alpha, jlong dX_address, jfloat beta,
	jlong dY_address,
	jint lengthv, jint width, jint stride) 
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)dX_address;
	float *dY = (float*)(intptr_t)dY_address;
	__expf2D(stream, alpha, dX, beta, dY, lengthv, width, stride);
}

#endif


#ifndef JNI_LOG
#define JNI_LOG

// Method:    log2D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_log2D(JNIEnv *env, jclass cls, 
	jlong stream_address, 
	jfloat alpha, jlong dX_address, jfloat beta,
	jlong dY_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)dX_address;
	float *dY = (float*)(intptr_t)dY_address;
	__log2D(stream, alpha, dX, beta, dY, 
		lengthv, width, stride);
}

//Method:    log2D_deltaX
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_log2D_1deltaX(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong d_deltaX_address,
	jlong d_deltaY_address,
	jlong dY_address, jfloat alpha,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *d_deltaX = (float*)(intptr_t)d_deltaX_address;
	float *d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float *dY = (float*)(intptr_t)dY_address;
	__log2D_deltaX(stream, d_deltaX, d_deltaY, dY, alpha, 
		lengthv, width, stride);
}

#endif


#ifndef JNI_RELU
#define JNI_RELU

//Method:    relu2D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_relu2D(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dX_address,
	jlong dY_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)dX_address;
	float *dY = (float*)(intptr_t)dY_address;
	__relu2D(stream, dX, dY, lengthv, width, stride);
}

//Method:    relu2D_deltaX_v1
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_relu2D_1deltaX_1v1(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong d_deltaX_address,
	jlong d_deltaY_address,
	jlong dY_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *d_deltaX = (float*)(intptr_t)d_deltaX_address;
	float *d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float *dY = (float*)(intptr_t)dY_address;
	__relu2D_deltaX_v1(stream, d_deltaX, d_deltaY, dY, lengthv, width, stride);
}


//Method:    relu2D_deltaX_v2
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_relu2D_1deltaX_1v2(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong d_deltaX_address,
	jlong d_deltaY_address, 
	jlong dX_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *d_deltaX = (float*)(intptr_t)d_deltaX_address;
	float *d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float *dX = (float*)dX_address;
	__relu2D_deltaX_v2(stream, d_deltaX, d_deltaY, dX, lengthv, width, stride);
}

#endif


#ifndef JNI_LEAKY_RELU
#define JNI_LEAKY_RELU

//Method:    leakyRelu2D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_leakyRelu2D(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dX_address, jfloat k, 
	jlong dY_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)dX_address;
	float *dY = (float*)(intptr_t)dY_address;
	__leakyRelu2D(stream, dX, k, dY, lengthv, width, stride);
}

//Method:    leakyRelu2D_deltaX_v1
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_leakyRelu2D_1deltaX_1v1(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong d_deltaX_address,
	jlong d_deltaY_address,
	jlong dY_address, jfloat k,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *d_deltaX = (float*)(intptr_t)d_deltaX_address;
	float *d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float *dY = (float*)(intptr_t)dY_address;
	__leakyRelu2D_deltaX_v1(stream, d_deltaX, d_deltaY, dY, k,
		lengthv, width, stride);
}

//Method:    leakyRelu2D_deltaX_v2
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_leakyRelu2D_1deltaX_1v2(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong d_deltaX_address,
	jlong d_deltaY_address,
	jlong dX_address, jfloat k,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *d_deltaX = (float*)(intptr_t)d_deltaX_address;
	float *d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float *dX = (float*)(intptr_t)dX_address;
	__leakyRelu2D_deltaX_v2(stream, d_deltaX, d_deltaY, dX, k, 
		lengthv, width, stride);
}

#endif


#ifndef JNI_ELU
#define JNI_ELU

//Method:    elu2D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_elu2D(JNIEnv *env, jclass cls,
	jlong stream_address, 
	jlong dX_address, jfloat alpha, jfloat k,
	jlong dY_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)dX_address;
	float *dY = (float*)(intptr_t)dY_address;
	__elu2D(stream, dX, alpha, k, dY, 
		lengthv, width, stride);
}

//Method:    elu2D_deltaX_v1
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_elu2D_1deltaX_1v1(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong d_deltaX_address,
	jlong d_deltaY_address,
	jlong dY_address, jfloat alpha, jfloat k,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *d_deltaX = (float*)(intptr_t)d_deltaX_address;
	float *d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float *dY = (float*)(intptr_t)dY_address;
	__elu2D_deltaX_v1(stream, d_deltaX, d_deltaY, dY, alpha, k, 
		lengthv, width, stride);
}

//Method:    elu2D_deltaX_v2
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_elu2D_1deltaX_1v2(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong d_deltaX_address,
	jlong d_deltaY_address,
	jlong dX_address, jfloat alpha, jfloat k,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *d_deltaX = (float*)(intptr_t)d_deltaX_address;
	float *d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float *dX = (float*)(intptr_t)dX_address;
	__elu2D_deltaX_v2(stream, d_deltaX, d_deltaY, dX, alpha, k, 
		lengthv, width, stride);
}

#endif


#ifndef JNI_SOFTPLUS
#define JNI_SOFTPLUS

//Method:    softPlus2D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_softPlus2D(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dX_address,
	jlong dY_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)dX_address;
	float *dY = (float*)(intptr_t)dY_address;
	__softplus2D(stream, dX, dY,
		lengthv, width, stride);
}

//Method:    softPlus2D_deltaX_v1
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_softPlus2D_1deltaX_1v1(JNIEnv *env, jclass cls,
	jlong stream_address, 
	jlong d_deltaX_address,
	jlong d_deltaY_address, 
	jlong dY_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *d_deltaX = (float*)(intptr_t)d_deltaX_address;
	float *d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float *dY = (float*)(intptr_t)dY_address;
	__softplus2D_deltaX_v1(stream, d_deltaX, d_deltaY, dY,
		lengthv, width, stride);
}

//Method:    softPlus2D_deltaX_v2
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_softPlus2D_1deltaX_1v2(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong d_deltaX_address,
	jlong d_deltaY_address,
	jlong dX_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *d_deltaX = (float*)(intptr_t)d_deltaX_address;
	float *d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float *dX = (float*)(intptr_t)dX_address;
	__softplus2D_deltaX_v2(stream, d_deltaX, d_deltaY, dX, 
		lengthv, width, stride);
}

#endif

#endif


//hypherbolic functions
#ifndef JNI_TYPE5_HYPHERBOLIC
#define JNI_TYPE5_HYPHERBOLIC

#ifndef JNI_TANH
#define JNI_TANH

//Method:    tanh2D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_tanh2D(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dX_address,
	jlong dY_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)dX_address;
	float *dY = (float*)(intptr_t)dY_address;
	__tanh2D(stream, dX, dY, 
		lengthv, width, stride);
}

//Method:    tanh2D_deltaX_v1
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_tanh2D_1deltaX_1v1(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong d_deltaX_address,
	jlong d_deltaY_address,
	jlong dY_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *d_deltaX = (float*)(intptr_t)d_deltaX_address;
	float *d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float *dY = (float*)(intptr_t)dY_address;
	__tanh2D_deltaX_v1(stream, d_deltaX, d_deltaY, dY,
		lengthv, width, stride);
}

//Method:    tanh2D_deltaX_v2
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_tanh2D_1deltaX_1v2(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong d_deltaX_address,
	jlong d_deltaY_address,
	jlong dX_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *d_deltaX = (float*)(intptr_t)d_deltaX_address;
	float *d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float *dX = (float*)(intptr_t)dX_address;
	__tanh2D_deltaX_v2(stream, d_deltaX, d_deltaY, dX,
		lengthv, width, stride);
}

#endif


#ifndef JNI_SIGMOID
#define JNI_SIGMOID

//Method:    sigmoid2D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_sigmoid2D(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dX_address,
	jlong dY_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)dX_address;
	float *dY = (float*)(intptr_t)dY_address;
	__sigmoid2D(stream, dX, dY,
		lengthv, width, stride);
}

//Method:    sigmoid2D_deltaX_v1
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_sigmoid2D_1deltaX_1v1(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong d_deltaX_address,
	jlong d_deltaY_address,
	jlong dY_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *d_deltaX = (float*)(intptr_t)d_deltaX_address;
	float *d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float *dY = (float*)(intptr_t)dY_address;
	__sigmoid2D_deltaX_v1(stream, d_deltaX, d_deltaY, dY, 
		lengthv, width, stride);
}

//Method:    sigmoid2D_deltaX_v2
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_sigmoid2D_1deltaX_1v2(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong d_deltaX_address,
	jlong d_deltaY_address,
	jlong dX_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *d_deltaX = (float*)(intptr_t)d_deltaX_address;
	float *d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float *dX = (float*)(intptr_t)dX_address;
	__sigmoid2D_deltaX_v2(stream, d_deltaX, d_deltaY, dX,
		lengthv, width, stride);
}

#endif

//Method:    softmax2D_deltaX
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_softmax2D_1deltaX(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong d_deltaX_address,
	jlong d_deltaY_address,
	jlong dY_address,
	jlong d_deltaY_Y_rowSum_address, jint row_lengthv,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *d_deltaX = (float*)(intptr_t)d_deltaX_address;
	float *d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float *dY = (float*)(intptr_t)dY_address;
	float *d_deltaY_Y_rowSum = (float*)(intptr_t)d_deltaY_Y_rowSum_address;
	__softmax2D_deltaX(stream, d_deltaX, d_deltaY, dY,
		d_deltaY_Y_rowSum, row_lengthv,
		lengthv, width, stride);
}

#ifndef JNI_LOG_SOFTMAX
#define JNI_LOG_SOFTMAX

//Method:    logsoftmax2D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_logsoftmax2D(JNIEnv *env, jclass cls,
	jlong stream_address, 
	jlong dX_address,
	jlong d_maxX_address, 
	jlong d_expXm_max_rowSum_address, jint row_lengthv, 
	jlong dY_address, 
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)dX_address;
	float *dY = (float*)(intptr_t)dY_address;
	float *d_maxX = (float*)(intptr_t)d_maxX_address;
	float *d_expXm_max_rowSum = (float*)(intptr_t)d_expXm_max_rowSum_address;
	__logsoftmax2D(stream, dX, d_maxX, d_expXm_max_rowSum, row_lengthv, 
		dY, lengthv, width, stride);
}

//Method:    logsoftmax2D_deltaX
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_logsoftmax2D_1deltaX(JNIEnv *env, jclass cls,
	jlong stream_address, 
	jlong d_deltaX_address, 
	jlong d_deltaY_address,
	jlong dY_address,
	jlong d_deltaY_rowSum_address, jint row_lengthv,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *d_deltaX = (float*)(intptr_t)d_deltaX_address;
	float *d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float *dY = (float*)(intptr_t)dY_address;
	float *d_deltaY_rowSum = (float*)(intptr_t)d_deltaY_rowSum_address;
	__logsoftmax2D_deltaX(stream, d_deltaX, d_deltaY, dY, 
		d_deltaY_rowSum, row_lengthv, 
		lengthv, width, stride);
}

#endif

#endif


//trigonometric functions
#ifndef JNI_TYPE6_TRIGONOMETRIC
#define JNI_TYPE6_TRIGONOMETRIC

#ifndef JNI_SIN
#define JNI_SIN

//Method:    sin2D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_sin2D(JNIEnv *env, jclass cls,
	jlong stream_address,
	jfloat alpha, jlong dX_address, jfloat beta,
	jlong dY_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)dX_address;
	float *dY = (float*)(intptr_t)dY_address;
	__sin2D(stream, alpha, dX, beta, dY, lengthv, width, stride);
}

//Method:    sin2D_deltaX
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_sin2D_1deltaX(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong d_deltaX_address,
	jlong d_deltaY_address,
	jlong dX_address, jfloat alpha, jfloat beta,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *d_deltaX = (float*)(intptr_t)d_deltaX_address;
	float *d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float *dX = (float*)(intptr_t)dX_address;
	__sin2D_deltaX(stream, d_deltaX, d_deltaY, dX, alpha, beta, lengthv, width, stride);
}

#endif


#ifndef JNI_TAN
#define JNI_TAN

//Method:    tan2D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_tan2D(JNIEnv *env, jclass cls,
	jlong stream_address,
	jfloat alpha, jlong dX_address, jfloat beta,
	jlong dY_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)dX_address;
	float *dY = (float*)(intptr_t)dY_address;
	__tan2D(stream, alpha, dX, beta, dY, 
		lengthv, width, stride);
}

//Method:    tan2D_deltaX
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_tan2D_1deltaX(JNIEnv *env, jclass cls,
	jlong stream_address, 
	jlong d_deltaX_address,
	jlong d_deltaY_address,
	jlong dY_address, jfloat alpha,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *d_deltaX = (float*)(intptr_t)d_deltaX_address;
	float *d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float *dY = (float*)(intptr_t)dY_address;
	__tan2D_deltaX(stream, d_deltaX, d_deltaY, dY, alpha, 
		lengthv, width, stride);
}

#endif


#ifndef JNI_HALF_SIN
#define JNI_HALF_SIN

//Method:    halfSin2D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_halfSin2D(JNIEnv *env, jclass cls,
	jlong stream_address,
	jfloat Amp, jfloat alpha, jlong dX_address, jfloat beta,
	jlong dY_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)dX_address;
	float *dY = (float*)(intptr_t)dY_address;
	__halfsin2D(stream, Amp, alpha, dX, beta, dY, lengthv, width, stride);
}

//Method:    halfSin2D_deltaX
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_halfSin2D_1deltaX(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong d_deltaX_address,
	jlong d_deltaY_address,
	jlong dY_address, jfloat Amp, jfloat alpha,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *d_deltaX = (float*)(intptr_t)d_deltaX_address;
	float *d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float *dY = (float*)(intptr_t)dY_address;
	__halfsin2D_deltaX(stream, d_deltaX, d_deltaY, dY, Amp, alpha, lengthv, width, stride);
}

#endif


#ifndef JNI_ARCSIN
#define JNI_ARCSIN

//Method:    arcsin2D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_arcsin2D(JNIEnv *env, jclass cls,
	jlong stream_address,
	jfloat alpha, jlong dX_address, jfloat beta,
	jlong dY_address, 
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)dX_address;
	float *dY = (float*)(intptr_t)dY_address;
	__arcsin2D(stream, alpha, dX, beta, dY, 
		lengthv, width, stride);
}

//Method:    arcsin2D_deltaX
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_arcsin2D_1deltaX(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong d_deltaX_address,
	jlong d_deltaY_address,
	jlong dY_address, jfloat alpha,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *d_deltaX = (float*)(intptr_t)d_deltaX_address;
	float *d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float *dY = (float*)(intptr_t)dY_address;
	return __arcsin2D_deltaX(stream, d_deltaX, d_deltaY, dY, alpha,
		lengthv, width, stride);
}

#endif


#ifndef JNI_ARCTAN
#define JNI_ARCTAN

//Method:    arctan2D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_arctan2D(JNIEnv *env, jclass cls,
	jlong stream_address,
	jfloat alpha, jlong dX_address, jfloat beta,
	jlong dY_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)dX_address;
	float *dY = (float*)(intptr_t)dY_address;
	__arctan2D(stream, alpha, dX, beta, dY, 
		lengthv, width, stride);
}

//Method:    arctan2D_deltaX
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_arctan2D_1deltaX(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong d_deltaX_address, 
	jlong d_deltaY_address,
	jlong dY_address, jfloat alpha, 
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *d_deltaX = (float*)(intptr_t)d_deltaX_address;
	float *d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float *dY = (float*)(intptr_t)dY_address;
	return __arctan2D_deltaX(stream, d_deltaX, d_deltaY, dY, alpha, 
		lengthv, width, stride);
}

#endif

#endif


//distance & loss function
#ifndef JNI_DISTANCE
#define JNI_DISTANCE

#ifndef JNI_L1
#define JNI_L1

//Method:    L1_2D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_L1_12D(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dY_address, jlong dYh_address,
	jlong dL_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dY = (float*)(intptr_t)dY_address;
	float *dYh = (float*)(intptr_t)dYh_address;
	float *dL = (float*)(intptr_t)dL_address;
	__L1_2D(stream, dY, dYh, dL, 
		lengthv, width, stride);
}

//Method:    L1_2D_deltaYh
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_L1_12D_1deltaYh(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dY_address, jlong dYh_address,
	jlong d_deltaYh_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dY = (float*)(intptr_t)dY_address;
	float *dYh = (float*)(intptr_t)dYh_address;
	float *d_deltaYh = (float*)(intptr_t)d_deltaYh_address;
	__L1_2D_deltaYh(stream, dY, dYh, d_deltaYh, 
		lengthv, width, stride);
}

#endif


#ifndef JNI_L2
#define JNI_L2

//Method:    L2_2D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_L2_12D(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dY_address, jlong dYh_address,
	jlong dL_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dY = (float*)(intptr_t)dY_address;
	float *dYh = (float*)(intptr_t)dYh_address;
	float *dL = (float*)(intptr_t)dL_address;
	__L2_2D(stream, dY, dYh, dL, 
		lengthv, width, stride);
}

//Method:    L2_2D_deltaYh
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_L2_12D_1deltaYh(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dY_address, jlong dYh_address,
	jlong d_deltaYh_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dY = (float*)(intptr_t)dY_address;
	float *dYh = (float*)(intptr_t)dYh_address;
	float *d_deltaYh = (float*)(intptr_t)d_deltaYh_address;
	__L2_2D_deltaYh(stream, dY, dYh, d_deltaYh, 
		lengthv, width, stride);
}

#endif


#ifndef JNI_SMOOTH_L1
#define JNI_SMOOTH_L1

//Method:    smoothL1_2D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_smoothL1_12D(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dY_address, jlong dYh_address,
	jlong dL_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dY = (float*)(intptr_t)dY_address;
	float *dYh = (float*)(intptr_t)dYh_address;
	float *dL = (float*)(intptr_t)dL_address;
	__smoothL1_2D(stream, dY, dYh, dL, 
		lengthv, width, stride);
}

//Method:    smoothL1_2D_deltaYh
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_smoothL1_12D_1deltaYh(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dY_address, jlong dYh_address,
	jlong d_deltaYh_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dY = (float*)(intptr_t)dY_address;
	float *dYh = (float*)(intptr_t)dYh_address;
	float *d_deltaYh = (float*)(intptr_t)d_deltaYh_address;
	__smoothL1_2D_deltaYh(stream, dY, dYh, d_deltaYh,
		lengthv, width, stride);
}

#endif


#ifndef JNI_BINARY_CROSS_ENTROPY
#define JNI_BINARY_CROSS_ENTROPY

//Method:    binaryCrossEntropy2D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_binaryCrossEntropy2D(JNIEnv *env, jclass cls,
	jlong stream_address, 
	jlong dY_address, jlong dYh_address, 
	jfloat alpha, jfloat beta,
	jlong dL_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dY = (float*)(intptr_t)dY_address;
	float *dYh = (float*)(intptr_t)dYh_address;
	float *dL = (float*)(intptr_t)dL_address;
	__binaryCrossEntropy_2D(stream, dY, dYh, alpha, beta, dL, 
		lengthv, width, stride);
}

//Method:    binaryCrossEntropy2D_deltaYh
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_binaryCrossEntropy2D_1deltaYh(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dY_address, jlong dYh_address,
	jfloat alpha, jfloat beta,
	jlong d_deltaYh_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dY = (float*)(intptr_t)dY_address;
	float *dYh = (float*)(intptr_t)dYh_address;
	float *d_deltaYh = (float*)(intptr_t)d_deltaYh_address;
	__binaryCrossEntropy_2D_deltaYh(stream, dY, dYh, alpha, beta, d_deltaYh,
		lengthv, width, stride);
}

#endif


#ifndef JNI_SIGMOID_BINARY_CROSS_ENTROPY
#define JNI_SIGMOID_BINARY_CROSS_ENTROPY

//Method:    sigmoid_binaryCrossEntropy2D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_sigmoid_1binaryCrossEntropy2D(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dY_address, jlong dX_address,
	jfloat alpha, jfloat beta,
	jlong dL_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dY = (float*)(intptr_t)dY_address;
	float *dX = (float*)(intptr_t)dX_address;
	float *dL = (float*)(intptr_t)dL_address;
	__sigmoid_binaryCrossEntropy_2D(stream, dY, dX, alpha, beta, dL,
		lengthv, width, stride);
}

//Method:    sigmoid_binaryCrossEntropy2D_deltaX
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_sigmoid_1binaryCrossEntropy2D_1deltaX(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dY_address, jlong dX_address,
	jfloat alpha, jfloat beta,
	jlong d_deltaX_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dY = (float*)(intptr_t)dY_address;
	float *dX = (float*)(intptr_t)dX_address;
	float *d_deltaX = (float*)(intptr_t)d_deltaX_address;
	__sigmoid_crossEntropy_2D_deltaX(stream, dY, dX, alpha, beta, d_deltaX,
		lengthv, width, stride);
}

#endif


#ifndef JNI_CROSS_ENTROPY
#define JNI_CROSS_ENTROPY

//Method:    crossEntropy2D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_crossEntropy2D(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dY_address, jlong dYh_address,
	jlong dL_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dY = (float*)(intptr_t)dY_address;
	float *dYh = (float*)(intptr_t)dYh_address;
	float *dL = (float*)(intptr_t)dL_address;
	__crossEntropy_2D(stream, dY, dYh, dL,
		lengthv, width, stride);
}

//Method:    crossEntropy2D_deltaYh
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_crossEntropy2D_1deltaYh(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dY_address, jlong dYh_address,
	jlong d_deltaYh_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dY = (float*)(intptr_t)dY_address;
	float *dYh = (float*)(intptr_t)dYh_address;
	float *d_deltaYh = (float*)(intptr_t)d_deltaYh_address;
	__crossEntropy_2D_deltaYh(stream, dY, dYh, d_deltaYh, 
		lengthv, width, stride);
}

#endif


#ifndef JNI_SOFTMAX_CROSS_ENTROPY
#define JNI_SOFTMAX_CROSS_ENTROPY

//Method:    softmax_crossEntropy2D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_softmax_1crossEntropy2D(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dY_address, jlong dX_address,
	jlong d_maxX_address,
	jlong d_expXm_max_rowSum_address, jint row_lengthv,
	jlong dL_address, 
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dY = (float*)(intptr_t)dY_address;
	float *dX = (float*)(intptr_t)dX_address;
	float *d_maxX = (float*)(intptr_t)d_maxX_address;
	float *d_expXm_max_rowSum = (float*)(intptr_t)d_expXm_max_rowSum_address;
	float *dL = (float*)(intptr_t)dL_address;
	__softmax_crossEntropy_2D(stream, dY, dX, 
		d_maxX, d_expXm_max_rowSum, row_lengthv, dL, 
		lengthv, width, stride);
}

//Method:    softmax_crossEntropy2D_deltaX
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_softmax_1crossEntropy2D_1deltaX(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dY_address, jlong dX_address,
	jlong d_maxX_address, 
	jlong d_expXm_max_rowSum_address,
	jlong Y_rowSum_address, jint row_lengthv,
	jlong d_deltaX_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dY = (float*)(intptr_t)dY_address;
	float *dX = (float*)(intptr_t)dX_address;
	float *d_maxX = (float*)(intptr_t)d_maxX_address;
	float *d_expXm_max_rowSum = (float*)(intptr_t)d_expXm_max_rowSum_address;
	float *dY_rowSum = (float*)(intptr_t)Y_rowSum_address;
	float *d_deltaX = (float*)(intptr_t)d_deltaX_address;
	__softmax_crossEntropy_2D_deltaX(stream, dY, dX,
		d_maxX, d_expXm_max_rowSum, dY_rowSum, row_lengthv, d_deltaX,
		lengthv, width, stride);
}

#endif

#endif


//optimizers
#ifndef JNI_OPTIMIZER
#define JNI_OPTIMIZER

#ifndef JNI_MOMENTUM
#define JNI_MOMENTUM

//Method:    momentum2D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_momentum2D(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dW_address,
	jlong dV_address, jfloat a1, jfloat a2,
	jlong d_deltaW_address, jfloat lr_t, 
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dW = (float*)(intptr_t)dW_address;
	float *dV = (float*)(intptr_t)dV_address;
	float *d_deltaW = (float*)(intptr_t)d_deltaW_address;
	__momentum2D(stream, dW, 
		dV, a1, a2,
		d_deltaW, lr_t, 
		lengthv, width, stride);
}

//Method:    momentum2D_decay
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_momentum2D_1decay(JNIEnv *env, jclass cls, 
	jlong stream_address,
	jlong dW_address,
	jlong dV_address, jfloat a1, jfloat a2,
	jlong d_deltaW_address, jfloat lr_t,
	jfloat L1coef, jfloat L2coef, 
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dW = (float*)(intptr_t)dW_address;
	float *dV = (float*)(intptr_t)dV_address;
	float *d_deltaW = (float*)(intptr_t)d_deltaW_address;
	__momentum2D_decay(stream, dW,
		dV, a1, a2,
		d_deltaW, lr_t, 
		L1coef, L2coef, 
		lengthv, width, stride);
}

#endif


#ifndef JNI_SGDMN
#define JNI_SGDMN

//Method:    sgdmn2D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_sgdmn2D(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dW_address,
	jlong dV_address, jfloat momentum, jfloat dampen, jfloat nesterov,
	jlong d_deltaW_address, jfloat lr,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dW = (float*)(intptr_t)dW_address;
	float *dV = (float*)(intptr_t)dV_address;
	float *d_deltaW = (float*)(intptr_t)d_deltaW_address;
	__sgdmn2D(stream, dW,
		dV, momentum, dampen, nesterov,
		d_deltaW, lr,
		lengthv, width, stride);
}

//Method:    sgdmn2D_decay
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_sgdmn2D_1decay(JNIEnv *env, jclass cls,
	jlong stream_address, 
	jlong dW_address,
	jlong dV_address, jfloat momentum, jfloat dampen, jfloat nesterov,
	jlong d_deltaW_address, jfloat lr,
	jfloat L1coef, jfloat L2coef, 
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dW = (float*)(intptr_t)dW_address;
	float *dV = (float*)(intptr_t)dV_address;
	float *d_deltaW = (float*)(intptr_t)d_deltaW_address;
	__sgdmn2D_decay(stream, dW,
		dV, momentum, dampen, nesterov,
		d_deltaW, lr,
		L1coef, L2coef,
		lengthv, width, stride);
}

#endif


#ifndef JNI_RMSPROP
#define JNI_RMSPROP

//Method:    rmsprop2D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_rmsprop2D(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dW_address, 
	jlong dS_address, jfloat a1, jfloat a2, jfloat eps_t,
	jlong d_deltaW_address, jfloat lr_t,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dW = (float*)(intptr_t)dW_address;
	float *dS = (float*)(intptr_t)dS_address;
	float *d_deltaW = (float*)(intptr_t)d_deltaW_address;
	__rmsprop2D(stream, dW, 
		dS, a1, a2, eps_t,
		d_deltaW, lr_t, 
		lengthv, width, stride);
}

//Method:    rmsprop2D_decay
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_rmsprop2D_1decay(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dW_address, 
	jlong dS_address, jfloat a1, jfloat a2, jfloat eps_t,
	jlong d_deltaW_address, jfloat lr_t, 
	jfloat L1coef, jfloat L2coef, 
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dW = (float*)(intptr_t)dW_address;
	float *dS = (float*)(intptr_t)dS_address;
	float *d_deltaW = (float*)(intptr_t)d_deltaW_address;
	__rmsprop2D_decay(stream, dW,
		dS, a1, a2, eps_t, 
		d_deltaW, lr_t,
		L1coef, L2coef, 
		lengthv, width, stride);
}

#endif


#ifndef JNI_ADAM
#define JNI_ADAM

//Method:    adam2D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_adam2D(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dW_address,
	jlong dV_address, jfloat a1, jfloat a2,
	jlong dS_address, jfloat b1, jfloat b2, jfloat eps,
	jlong d_deltaW_address, jfloat lr_t,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dW = (float*)(intptr_t)dW_address;
	float *dV = (float*)(intptr_t)dV_address;
	float *dS = (float*)(intptr_t)dS_address;
	float *d_deltaW = (float*)(intptr_t)d_deltaW_address;
	__adam2D(stream, dW, 
		dV, a1, a2, 
		dS, b1, b2, eps, 
		d_deltaW, lr_t, 
		lengthv, width, stride);
}

//Method:    adam2D_type2
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_adam2D_1type2(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dW_address,
	jlong dV_address, jfloat a1, jfloat a2, jfloat Uv,
	jlong dS_address, jfloat b1, jfloat b2, jfloat eps, jfloat Us,
	jlong d_deltaW_address, jfloat lr,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dW = (float*)(intptr_t)dW_address;
	float *dV = (float*)(intptr_t)dV_address;
	float *dS = (float*)(intptr_t)dS_address;
	float *d_deltaW = (float*)(intptr_t)d_deltaW_address;
	__adam2D_type2(stream, dW,
		dV, a1, a2, Uv,
		dS, b1, b2, eps, Us,
		d_deltaW, lr,
		lengthv, width, stride);
}

//Method:    adam2D_decay
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_adam2D_1decay(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dW_address,
	jlong dV_address, jfloat a1, jfloat a2,
	jlong dS_address, jfloat b1, jfloat b2, jfloat eps_t,
	jlong d_deltaW_address, jfloat lr_t,
	jfloat L1coef, jfloat L2coef,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dW = (float*)(intptr_t)dW_address;
	float *dV = (float*)(intptr_t)dV_address;
	float *dS = (float*)(intptr_t)dS_address;
	float *d_deltaW = (float*)(intptr_t)d_deltaW_address;
	__adam2D_decay(stream, dW, 
		dV, a1, a2, 
		dS, b1, b2, eps_t,
		d_deltaW, lr_t, 
		L1coef, L2coef, 
		lengthv, width, stride);
}

#endif


#ifndef JNI_ADAMAX
#define JNI_ADAMAX

//Method:    adamax2D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_adamax2D(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dW_address,
	jlong dV_address, jfloat a1, jfloat a2,
	jlong dS_address, jfloat b1, jfloat eps,
	jlong d_deltaW_address, jfloat lr_t,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dW = (float*)(intptr_t)dW_address;
	float *dV = (float*)(intptr_t)dV_address;
	float *dS = (float*)(intptr_t)dS_address;
	float *d_deltaW = (float*)(intptr_t)d_deltaW_address;
	__adamax2D(stream, dW, 
		dV, a1, a2,
		dS, b1, eps,
		d_deltaW, lr_t,
		lengthv, width, stride);
}

//Method:    adamax2D_decay
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_adamax2D_1decay(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dW_address,
	jlong dV_address, jfloat a1, jfloat a2, 
	jlong dS_address, jfloat b1, jfloat eps, 
	jlong d_deltaW_address, jfloat lr_t, 
	jfloat L1coef, jfloat L2coef,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dW = (float*)(intptr_t)dW_address;
	float *dV = (float*)(intptr_t)dV_address;
	float *dS = (float*)(intptr_t)dS_address;
	float *d_deltaW = (float*)(intptr_t)d_deltaW_address;
	__adamax2D_decay(stream, dW, 
		dV, a1, a2, 
		dS, b1, eps, 
		d_deltaW, lr_t, 
		L1coef, L2coef, 
		lengthv, width, stride);
}

#endif


#ifndef JNI_ADAMW
#define JNI_ADAMW

//Method:    adamW2D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_adamW2D(JNIEnv *env, jclass cls,
	jlong stream_address, 
	jlong dW_address, 
	jlong dV_address, jfloat a1, jfloat a2,
	jlong dS_address, jfloat b1, jfloat b2, jfloat eps,
	jlong d_deltaW_address, jfloat lr_t, jfloat lr,
	jfloat L1coef, jfloat L2coef,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dW = (float*)(intptr_t)dW_address;
	float *dV = (float*)(intptr_t)dV_address;
	float *dS = (float*)(intptr_t)dS_address;
	float *d_deltaW = (float*)(intptr_t)d_deltaW_address;
	__adamW2D(stream, dW,
		dV, a1, a2, 
		dS, b1, b2, eps,
		d_deltaW, lr_t, lr,
		L1coef, L2coef,
		lengthv, width, stride);
}

#endif


#ifndef JNI_ADAMOD
#define JNI_ADAMOD

//Method:    adamod2D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_adamod2D(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dW_address, 
	jlong dV_address, jfloat a1, jfloat a2, 
	jlong dS_address, jfloat b1, jfloat b2, jfloat eps_t,
	jlong dG_address, jfloat c1, jfloat c2, 
	jlong d_deltaW_address, jfloat lr_t, 
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dW = (float*)(intptr_t)dW_address;
	float *dV = (float*)(intptr_t)dV_address;
	float *dS = (float*)(intptr_t)dS_address;
	float *dG = (float*)(intptr_t)dG_address;
	float *d_deltaW = (float*)(intptr_t)d_deltaW_address;
	__adamod2D(stream, dW,
		dV, a1, a2, 
		dS, b1, b2, eps_t,
		dG, c1, c2,
		d_deltaW, lr_t,
		lengthv, width, stride);
}

//Method:    adamod2D_decay
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_adamod2D_1decay(JNIEnv *env, jclass cls, jlong stream_address,
	jlong dW_address,
	jlong dV_address, jfloat a1, jfloat a2, 
	jlong dS_address, jfloat b1, jfloat b2, jfloat eps_t,
	jlong dG_address, jfloat c1, jfloat c2,
	jlong d_deltaW_address, jfloat lr_t,
	jfloat L1coef, jfloat L2coef,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dW = (float*)(intptr_t)dW_address;
	float *dV = (float*)(intptr_t)dV_address;
	float *dS = (float*)(intptr_t)dS_address;
	float *dG = (float*)(intptr_t)dG_address;
	float *d_deltaW = (float*)(intptr_t)d_deltaW_address;
	__adamod2D_decay(stream, dW,
		dV, a1, a2, 
		dS, b1, b2, eps_t,
		dG, c1, c2,
		d_deltaW, lr_t,
		L1coef, L2coef,
		lengthv, width, stride);
}

#endif

#endif


//affine, batchnorm, layernorm
#ifndef JNI_AFFINES
#define JNI_AFFINES


#ifndef JNI_AFFINE
#define JNI_AFFINE

//Method:    affine2D_row
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_affine2D_1row(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dX_address,
	jlong dA_address,
	jlong dB_address, jint row_lengthv,
	jlong dY_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)(dX_address);
	float *dA = (float*)(intptr_t)(dA_address);
	float *dB = (float*)(intptr_t)(dB_address);
	float *dY = (float*)(intptr_t)(dY_address);
	__affine2D_row(stream, dX, dA, dB, row_lengthv, dY,
		lengthv, width, stride);
}

#endif


//sqBatchNorm.forward_prop
#ifndef JNI_SQUARE_BATCH_NORM_FORWARD
#define JNI_SQUARE_BATCH_NORM_FORWARD

//Method:    sqBatchNorm2D_row
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_sqBatchNorm2D_1row(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dX_address, 
	jlong dX_mean_address,
	jlong dX_sqmean_address, jfloat eps, jint row_lengthv, 
	jlong dY_address, 
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)(dX_address);
	float *dX_mean = (float*)(intptr_t)(dX_mean_address);
	float *dX_sqmean = (float*)(intptr_t)(dX_sqmean_address);
	float *dY = (float*)(intptr_t)(dY_address);

	__sqBatchNorm2D_row(stream, dX, 
		dX_mean, dX_sqmean, eps, row_lengthv, dY,
		lengthv, width, stride);
}

//Method:    sqBatchNorm_affined2D_row
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_sqBatchNorm_1affined2D_1row(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dX_address,
	jlong dX_mean_address,
	jlong dX_sqmean_address, jfloat eps,
	jlong dA_address, jlong dB_address, jint row_lengthv,
	jlong dY_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)(dX_address);
	float *dX_mean = (float*)(intptr_t)(dX_mean_address);
	float *dX_sqmean = (float*)(intptr_t)(dX_sqmean_address);
	float *dA = (float*)(intptr_t)(dA_address);
	float *dB = (float*)(intptr_t)(dB_address);
	float *dY = (float*)(intptr_t)(dY_address);

	__sqBatchNorm_affined2D_row(stream, dX,
		dX_mean, dX_sqmean, eps, dA, dB, row_lengthv, dY,
		lengthv, width, stride);
}

#endif


//sqBatchNorm.backward_prop
#ifndef JNI_SQUARE_BATCH_NORM_BACKWARD
#define JNI_SQUARE_BATCH_NORM_BACKWARD

//Method:    sqBatchNorm2D_row_deltaX_v1
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_sqBatchNorm2D_1row_1deltaX_1v1(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong d_deltaY_address, 
	jlong dY_address,//V1: holdY(), Y is not changed
	jlong dX_mean_address, 
	jlong dX_sqmean_address, jfloat eps, 
	jlong d_deltaXp1_address, 
	jlong d_deltaXp2_address, jint row_lengthv,
	jlong d_deltaX_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float* d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float* dY = (float*)dY_address;
	float* dX_mean = (float*)(intptr_t)dX_mean_address;
	float* dX_sqmean = (float*)(intptr_t)dX_sqmean_address;

	float* d_deltaXp1 = (float*)(intptr_t)d_deltaXp1_address;
	float* d_deltaXp2 = (float*)(intptr_t)d_deltaXp2_address;
	float* d_deltaX = (float*)(intptr_t)d_deltaX_address;

	__sqBatchNorm2D_row_deltaX_v1(stream,
		d_deltaY, dY,
		dX_mean, dX_sqmean, eps,
		d_deltaXp1, d_deltaXp2, row_lengthv,
		d_deltaX, 
		lengthv, width, stride);
}

//Method:    sqBatchNorm2D_row_deltaX_v2
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_sqBatchNorm2D_1row_1deltaX_1v2(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong d_deltaY_address, 
	jlong dX_address,//V2: holdX(), X is not changed
	jlong dX_mean_address,
	jlong dX_sqmean_address, jfloat eps,
	jlong d_deltaXp1_address,
	jlong d_deltaXp2_address, jint row_lengthv,
	jlong d_deltaX_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float* d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float* dX = (float*)(intptr_t)dX_address;
	float* dX_mean = (float*)(intptr_t)dX_mean_address;
	float* dX_sqmean = (float*)(intptr_t)dX_sqmean_address;

	float* d_deltaXp1 = (float*)(intptr_t)d_deltaXp1_address;
	float* d_deltaXp2 = (float*)(intptr_t)d_deltaXp2_address;
	float* d_deltaX = (float*)(intptr_t)d_deltaX_address;

	__sqBatchNorm2D_row_deltaX_v2(stream,
		d_deltaY, dX,
		dX_mean, dX_sqmean, eps,
		d_deltaXp1, d_deltaXp2, row_lengthv,
		d_deltaX,
		lengthv, width, stride);
}

#endif


//sqBatchNorm.backward_prop(affined)
#ifndef JNI_SQUARE_BATCH_NORM_BACKWARD_AFFINED
#define JNI_SQUARE_BATCH_NORM_BACKWARD_AFFINED

//Method:    sqBatchNorm_affined2D_row_deltaX_v1
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_sqBatchNorm_1affined2D_1row_1deltaX_1v1(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong d_deltaY_address, 
	jlong dY_address,//V1: holdY(), Y is not changed
	jlong dX_mean_address, 
	jlong dX_sqmean_address, jfloat eps, 
	jlong dA_address, jlong dB_address,
	jlong d_deltaXp1_address, 
	jlong d_deltaXp2_address, jint row_lengthv,
	jlong d_deltaX_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float* d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float* dY = (float*)dY_address;
	float* dX_mean = (float*)(intptr_t)dX_mean_address;
	float* dX_sqmean = (float*)(intptr_t)dX_sqmean_address;
	float* dA = (float*)(intptr_t)dA_address;
	float* dB = (float*)(intptr_t)dB_address; 

	float* d_deltaXp1 = (float*)(intptr_t)d_deltaXp1_address;
	float* d_deltaXp2 = (float*)(intptr_t)d_deltaXp2_address;
	float* d_deltaX = (float*)(intptr_t)d_deltaX_address;

	__sqBatchNorm_affined2D_row_deltaX_v1(stream,
		d_deltaY, dY,
		dX_mean, dX_sqmean, eps, dA, dB, 
		d_deltaXp1, d_deltaXp2, row_lengthv,
		d_deltaX, 
		lengthv, width, stride);
}

//Method:    sqBatchNorm_affined2D_row_deltaX_v2
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_sqBatchNorm_1affined2D_1row_1deltaX_1v2(JNIEnv *env, jclass cls,
	jlong stream_address, 
	jlong d_deltaY_address, 
	jlong dX_address,//V2: holdX(), X is not changed
	jlong dX_mean_address, 
	jlong dX_sqmean_address, jfloat eps,
	jlong dA_address, 
	jlong d_deltaXp1_address, 
	jlong d_deltaXp2_address, jint row_lengthv,
	jlong d_deltaX_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float* d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float* dX = (float*)(intptr_t)dX_address;
	float* dX_mean = (float*)(intptr_t)dX_mean_address;
	float* dX_sqmean = (float*)(intptr_t)dX_sqmean_address;
	float* dA = (float*)(intptr_t)dA_address;

	float* d_deltaXp1 = (float*)(intptr_t)d_deltaXp1_address;
	float* d_deltaXp2 = (float*)(intptr_t)d_deltaXp2_address;
	float* d_deltaX = (float*)(intptr_t)d_deltaX_address;

	__sqBatchNorm_affined2D_row_deltaX_v2(stream,
		d_deltaY, dX,
		dX_mean, dX_sqmean, eps, dA,
		d_deltaXp1, d_deltaXp2, row_lengthv,
		d_deltaX,
		lengthv, width, stride);
}

#endif


//batchNorm.forward_prop
#ifndef JNI_BATCH_NORM_FORWARD
#define JNI_BATCH_NORM_FORWARD

//Method:    batchNorm2D_row
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_batchNorm2D_1row(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dX_address,
	jlong dX_mean_address,
	jlong dX_var_address, jfloat eps, jint row_lengthv, 
	jlong dY_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)(dX_address);
	float *dX_mean = (float*)(intptr_t)(dX_mean_address);
	float *dX_var = (float*)(intptr_t)(dX_var_address);
	float *dY = (float*)(intptr_t)(dY_address);
	
	__batchNorm2D_row(stream, dX,
		dX_mean, dX_var, eps, row_lengthv, dY,
		lengthv, width, stride);
}

//Method:    batchNorm_affined2D_row
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_batchNorm_1affined2D_1row(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dX_address,
	jlong dX_mean_address,
	jlong dX_var_address, jfloat eps,
	jlong dA_address, jlong dB_address, jint row_lengthv,
	jlong dY_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)(dX_address);
	float *dX_mean = (float*)(intptr_t)(dX_mean_address);
	float *dX_var = (float*)(intptr_t)(dX_var_address);
	float *dA = (float*)(intptr_t)(dA_address);
	float *dB = (float*)(intptr_t)(dB_address);
	float *dY = (float*)(intptr_t)(dY_address);

	__batchNorm_affined2D_row(stream, dX,
		dX_mean, dX_var, eps, dA, dB, row_lengthv, dY,
		lengthv, width, stride);
}

#endif


//batchNorm.backward_prop
#ifndef JNI_BATCH_NORM_BACKWARD
#define JNI_BATCH_NORM_BACKWARD

//Method:    batchNorm2D_row_deltaX_v1
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_batchNorm2D_1row_1deltaX_1v1(JNIEnv *env, jclass cls,
	jlong stream_address, 
	jlong d_deltaY_address, 
	jlong dY_address,//V1: holdY(), Y is not changed
	jlong dX_var_address, jfloat eps,
	jlong d_deltaXp1_address,
	jlong d_deltaXp2_address, jint row_lengthv, 
	jlong d_deltaX_address, 
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float* d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float* dY = (float*)dY_address;
	float* dX_var = (float*)(intptr_t)dX_var_address;

	float* d_deltaXp1 = (float*)(intptr_t)d_deltaXp1_address;
	float* d_deltaXp2 = (float*)(intptr_t)d_deltaXp2_address;
	float* d_deltaX = (float*)(intptr_t)d_deltaX_address;

	__batchNorm2D_row_deltaX_v1(stream, d_deltaY,
		dY, dX_var, eps,
		d_deltaXp1, d_deltaXp2, row_lengthv,
		d_deltaX,
		lengthv, width, stride);
}

//Method:    batchNorm2D_row_deltaX_v2
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_batchNorm2D_1row_1deltaX_1v2(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong d_deltaY_address,
	jlong dX_address,//V2: holdX(), X is not changed
	jlong dX_mean_address,
	jlong dX_var_address, jfloat eps,
	jlong d_deltaXp1_address,
	jlong d_deltaXp2_address, jint row_lengthv,
	jlong d_deltaX_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float* d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float* dX = (float*)(intptr_t)dX_address;
	float* dX_mean = (float*)(intptr_t)dX_mean_address;
	float* dX_var = (float*)(intptr_t)dX_var_address;

	float* d_deltaXp1 = (float*)(intptr_t)d_deltaXp1_address;
	float* d_deltaXp2 = (float*)(intptr_t)d_deltaXp2_address;
	float* d_deltaX = (float*)(intptr_t)d_deltaX_address;

	__batchNorm2D_row_deltaX_v2(stream,
		d_deltaY, dX,
		dX_mean, dX_var, eps,
		d_deltaXp1, d_deltaXp2, row_lengthv,
		d_deltaX,
		lengthv, width, stride);
}

#endif


//batchNorm.backward_prop(affined)
#ifndef JNI_BATCH_NORM_BACKWARD_AFFINED
#define JNI_BATCH_NORM_BACKWARD_AFFINED

//Method:    batchNorm_affined2D_row_deltaX_v1
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_batchNorm_1affined2D_1row_1deltaX_1v1(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong d_deltaY_address, 
	jlong dY_address,//V1: holdY(), Y is not changed
	jlong dX_var_address, jfloat eps,
	jlong dA_address, jlong dB_address,
	jlong d_deltaXp1_address, 
	jlong d_deltaXp2_address, jint row_lengthv, 
	jlong d_deltaX_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float* d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float* dY = (float*)dY_address;
	float* dX_var = (float*)(intptr_t)dX_var_address;
	float* dA = (float*)(intptr_t)dA_address;
	float* dB = (float*)(intptr_t)dB_address;

	float* d_deltaXp1 = (float*)(intptr_t)d_deltaXp1_address;
	float* d_deltaXp2 = (float*)(intptr_t)d_deltaXp2_address;
	float* d_deltaX = (float*)(intptr_t)d_deltaX_address;

	__batchNorm_affined2D_row_deltaX_v1(stream,
		d_deltaY, dY,
		dX_var, eps, dA, dB,
		d_deltaXp1, d_deltaXp2, row_lengthv,
		d_deltaX, 
		lengthv, width, stride);
}

//Method:    batchNorm_affined2D_row_deltaX_v2
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_batchNorm_1affined2D_1row_1deltaX_1v2(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong d_deltaY_address, 
	jlong dX_address,//V2: holdX(), X is not changed
	jlong dX_mean_address,
	jlong dX_var_address, jfloat eps,
	jlong dA_address,
	jlong d_deltaXp1_address,
	jlong d_deltaXp2_address, jint row_lengthv,
	jlong d_deltaX_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float* d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float* dX = (float*)(intptr_t)dX_address;
	float* dX_mean = (float*)(intptr_t)dX_mean_address;
	float* dX_var = (float*)(intptr_t)dX_var_address;
	float* dA = (float*)(intptr_t)dA_address;

	float* d_deltaXp1 = (float*)(intptr_t)d_deltaXp1_address;
	float* d_deltaXp2 = (float*)(intptr_t)d_deltaXp2_address;
	float* d_deltaX = (float*)(intptr_t)d_deltaX_address;

	__batchNorm_affined2D_row_deltaX_v2(stream,
		d_deltaY, dX,
		dX_mean, dX_var, eps, dA,
		d_deltaXp1, d_deltaXp2, row_lengthv,
		d_deltaX,
		lengthv, width, stride);
}

#endif


#ifndef JNI_LAYER_NORM
#define JNI_LAYER_NORM

//Method:    layerNorm2D_row
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_layerNorm2D_1row(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dX_address,
	jlong dX_mean_address,
	jlong dX_square_mean_address, jfloat eps, jint row_lengthv,
	jlong dY_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)(dX_address);
	float *dX_mean = (float*)(intptr_t)(dX_mean_address);
	float *dX_square_mean = (float*)(intptr_t)(dX_square_mean_address);
	float *dY = (float*)(intptr_t)(dY_address);
	__layernorm2D_row(stream, dX, 
		dX_mean, dX_square_mean, eps, row_lengthv, dY, 
		lengthv, width, stride);
}

//Method:    layerNorm2D_row_deltaX_v1
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_layerNorm2D_1row_1deltaX_1v1(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong d_deltaY_address, jlong dY_address,//V1: holdY(), Y is not changed
	jlong dX_mean_address, 
	jlong dX_square_mean_address, jfloat eps,
	jlong d_deltaXp1_address,
	jlong d_deltaXp2_address, jint row_lengthv,
	jlong d_deltaX_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *d_deltaY = (float*)(intptr_t)(d_deltaY_address);
	float *dY = (float*)(intptr_t)dY_address;
	float *dX_mean = (float*)(intptr_t)(dX_mean_address);
	float *dX_square_mean = (float*)(intptr_t)(dX_square_mean_address);
	float *d_deltaX = (float*)(intptr_t)(d_deltaX_address);
	float *d_deltaXp1 = (float*)(intptr_t)(d_deltaXp1_address);
	float *d_deltaXp2 = (float*)(intptr_t)(d_deltaXp2_address);
	__layernorm2D_row_deltaX_v1(stream,
		d_deltaY, dY,
		dX_mean, dX_square_mean, eps,
		d_deltaXp1, d_deltaXp2, row_lengthv,
		d_deltaX, 
		lengthv, width, stride);
}

//Method:    layerNorm2D_row_deltaX_v2
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_layerNorm2D_1row_1deltaX_1v2(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong d_deltaY_address, jlong dX_address,//V2: holdY(), X is not changed
	jlong dX_mean_address, 
	jlong dX_square_mean_address, jfloat eps,
	jlong d_deltaXp1_address, 
	jlong d_deltaXp2_address, jint row_lengthv, 
	jlong d_deltaX_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *d_deltaY = (float*)(intptr_t)(d_deltaY_address);
	float *dX = (float*)(intptr_t)dX_address;
	float *dX_mean = (float*)(intptr_t)(dX_mean_address);
	float *dX_square_mean = (float*)(intptr_t)(dX_square_mean_address);
	float *d_deltaX = (float*)(intptr_t)(d_deltaX_address);
	float *d_deltaXp1 = (float*)(intptr_t)(d_deltaXp1_address);
	float *d_deltaXp2 = (float*)(intptr_t)(d_deltaXp2_address);
	__layernorm2D_row_deltaX_v2(stream,
		d_deltaY, dX,
		dX_mean, dX_square_mean, eps,
		d_deltaXp1, d_deltaXp2, row_lengthv,
		d_deltaX,
		lengthv, width, stride);
}

#endif


#ifndef JNI_LAYER_NORM_AFFINED
#define JNI_LAYER_NORM_AFFINED

//Method:    layerNorm_affined2D_row
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_layerNorm_1affined2D_1row(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dX_address,
	jlong dX_mean_address, 
	jlong dX_square_mean_address, jfloat eps,
	jlong dA_address, jlong dB_address, jint row_lengthv,
	jlong dY_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)(dX_address);
	float *dX_mean = (float*)(intptr_t)(dX_mean_address);
	float *dX_square_mean = (float*)(intptr_t)(dX_square_mean_address);
	float *dA = (float*)(intptr_t)(dA_address);
	float *dB = (float*)(intptr_t)(dB_address);
	float *dY = (float*)(intptr_t)(dY_address);
	__layernorm_affined2D_row(stream, dX, 
		dX_mean, dX_square_mean, eps, dA, dB, row_lengthv, dY, 
		lengthv, width, stride);
}


//Method:    layerNorm_affined2D_row_deltaX_v1
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_layerNorm_1affined2D_1row_1deltaX_1v1(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong d_deltaY_address, jlong dY_address,//V1: holdY(), Y is not changed
	jlong dX_mean_address, jlong dX_square_mean_address, jfloat eps, 
	jlong dA_address, jlong dB_address,
	jlong d_deltaXp1_address, jlong d_deltaXp2_address, jint row_lengthv, 
	jlong d_deltaX_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *d_deltaY = (float*)(intptr_t)(d_deltaY_address);
	float *dY = (float*)(intptr_t)dY_address;
	float *dX_mean = (float*)(intptr_t)(dX_mean_address);
	float *dX_square_mean = (float*)(intptr_t)(dX_square_mean_address);
	float *dA = (float*)(intptr_t)(dA_address);
	float *dB = (float*)(intptr_t)(dB_address);
	float *d_deltaXp1 = (float*)(intptr_t)d_deltaXp1_address;
	float* d_deltaXp2 = (float*)(intptr_t)d_deltaXp2_address;
	float *d_deltaX = (float*)(intptr_t)d_deltaX_address;

	__layernorm_affined2D_row_deltaX_v1(stream,
		d_deltaY, dY,
		dX_mean, dX_square_mean, eps, dA, dB,
		d_deltaXp1, d_deltaXp2, row_lengthv,
		d_deltaX, lengthv, width, stride);
}

//Method:    layerNorm_affined2D_row_deltaX_v2
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_layerNorm_1affined2D_1row_1deltaX_1v2(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong d_deltaY_address, jlong dX_address,//V2: holdX(), X is not changed
	jlong dX_mean_address, jlong dX_square_mean_address, jfloat eps, 
	jlong dA_address, 
	jlong d_deltaXp1_address, jlong d_deltaXp2_address, jint row_lengthv,
	jlong d_deltaX_address, 
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *d_deltaY = (float*)(intptr_t)(d_deltaY_address);
	float *dX = (float*)(intptr_t)dX_address;
	float *dX_mean = (float*)(intptr_t)(dX_mean_address);
	float *dX_square_mean = (float*)(intptr_t)(dX_square_mean_address);
	float *dA = (float*)(intptr_t)(dA_address);
	float *d_deltaXp1 = (float*)(intptr_t)d_deltaXp1_address;
	float* d_deltaXp2 = (float*)(intptr_t)d_deltaXp2_address;
	float *d_deltaX = (float*)(intptr_t)d_deltaX_address;

	__layernorm_affined2D_row_deltaX_v2(stream,
		d_deltaY, dX,
		dX_mean, dX_square_mean, eps, dA,
		d_deltaXp1, d_deltaXp2, row_lengthv,
		d_deltaX, lengthv, width, stride);
}
#endif

#endif


#ifndef JNI_ONE_HOT
#define JNI_ONE_HOT

//Method:    onehot2D_row_int
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_onehot2D_1row_1int(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dX_address, 
	jfloat alpha, jfloat beta, jint row_lengthv, 
	jlong dY_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	int *dX = (int*)(intptr_t)(dX_address);
	float *dY = (float*)(intptr_t)(dY_address);
	__onehot2D_row_int(stream, dX, alpha, beta, row_lengthv, dY, 
		lengthv, width, stride);
}

//Method:    onehot2D_row_char
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_onehot2D_1row_1char(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dX_address,
	jfloat alpha, jfloat beta, jint row_lengthv,
	jlong dY_address, 
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	char *dX = (char*)(intptr_t)(dX_address);
	float *dY = (float*)(intptr_t)(dY_address);
	__onehot2D_row_char(stream, dX, alpha, beta, row_lengthv, dY,
		lengthv, width, stride);
}

#endif


#ifndef JNI_PIX2TENSOR
#define JNI_PIX2TENSOR

//Method:    pix2tensor2D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1function_pix2tensor2D(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dX_address,
	jlong dY_address,
	jint lengthv, jint width, jint stride)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	char *dX = (char*)(intptr_t)(dX_address);
	float *dY = (float*)(intptr_t)(dY_address);
	__pix2tensor2D(stream, dX, dY, lengthv, width, stride);
}

#endif