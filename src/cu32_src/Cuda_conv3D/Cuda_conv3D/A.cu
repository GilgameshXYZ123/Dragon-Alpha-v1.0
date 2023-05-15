#include "frame.cuh"
#include "Cuda_conv3D.cuh"
#include "JNITool.cuh"
#include "micro.cuh"
#include "texture.cuh"
#include "conv3D.cuh"
#include "test.cuh"


#ifndef CONV3D_JNI_GEMM
#define CONV3D_JNI_GEMM

//Method:    conv3D
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1conv3D_conv3D(JNIEnv *env, jclass cls,
	jlongArray streamArray, jint length,
	jlong dX_address, jint IH, jint IW,
	jlong dW_address, jint FH, jint FW,
	jlong dY_address, jint OH, jint OW,
	jint N, jint IC, jint OC,
	jint sh, jint sw, jint ph, jint pw)
{
	jlong streams[10]; 
	env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *dX = (float*)(intptr_t)dX_address;
	float *dW = (float*)(intptr_t)dW_address;
	float *dY = (float*)(intptr_t)dY_address;

	int index = 0;
	__conv3D_Gemm(streams, index, length, 
		dX, IH, IW, 
		dW, FH, FW, 
		dY, OH, OW, 
		N, IC, OC, 
		sh, sw, ph, pw);
}

//Method:    conv3D_texture
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1conv3D_conv3D_1texture(JNIEnv *env, jclass cls,
	jlongArray streamArray, jint length,
	jlong dX_address, jint IH, jint IW,
	jlong dW_address, jint FH, jint FW,
	jlong dY_address, jint OH, jint OW,
	jint N, jint IC, jint OC,
	jint sh, jint sw, jint ph, jint pw)
{
	jlong streams[10];
	env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *dX = (float*)(intptr_t)dX_address;
	float *dW = (float*)(intptr_t)dW_address;
	float *dY = (float*)(intptr_t)dY_address;

	int index = 0;
	cudaTextureObject_t texX = floatTexture(dX, (N*IH*IW*IC), env);
	__conv3D_Gemm_tex(streams, index, length,
		texX, dX, IH, IW,
		dW, FH, FW,
		dY, OH, OW,
		N, IC, OC,
		sh, sw, ph, pw);
	cudaError_t error = cudaDestroyTextureObject(texX); handleError(error);
}

//Method:    conv3DV2
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1conv3D_conv3DV2(JNIEnv *env, jclass cls,
	jboolean useTexture, jlongArray streamArray, jint length,
	jlong dX_address, jint IH, jint IW,
	jlong dW_address, jint FH, jint FW,
	jlong dY_address, jint OH, jint OW,
	jint N, jint IC, jint OC,
	jint sh, jint sw, jint ph, jint pw)
{
	jlong streams[10];
	env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *dX = (float*)(intptr_t)dX_address;
	float *dW = (float*)(intptr_t)dW_address;
	float *dY = (float*)(intptr_t)dY_address;

	int index = 0;
	if (!useTexture) env = NULL;
	__conv3D_GemmV2(env, streams, index, length,
		dX, IH, IW, (N*IH*IW*IC),//sizeX = N * IH * IW * IC
		dW, FH, FW,
		dY, OH, OW,
		IC, OC,
		sh, sw, ph, pw);
}

// Method:    conv3D_np
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1conv3D_conv3D_1np(JNIEnv *env, jclass cls, 
	jlongArray streamArray, jint length, 
	jlong dX_address, jint IH, jint IW,
	jlong dW_address, jint FH, jint FW, 
	jlong dY_address, jint OH, jint OW,
	jint N, jint IC, jint OC,
	jint sh, jint sw)
{
	jlong streams[10];
	env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *dX = (float*)(intptr_t)dX_address;
	float *dW = (float*)(intptr_t)dW_address;
	float *dY = (float*)(intptr_t)dY_address;
	int index = 0;
	__conv3D_Gemm_np(streams, index, length,
		dX, IH, IW,
		dW, FH, FW,
		dY, OH, OW,
		IC, OC,
		sh, sw);
}

//Method:    conv3D_W1
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1conv3D_conv3D_1W1(JNIEnv *env, jclass cls,
	jlongArray streamArray, jint length, 
	jlong dX_address, jint IH, jint IW,
	jlong dW_address, 
	jlong dY_address, 
	jint N, jint IC, jint OC)
{
	jlong streams[10];
	env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *dX = (float*)(intptr_t)dX_address;
	float *dW = (float*)(intptr_t)dW_address;
	float *dY = (float*)(intptr_t)dY_address;
	int index = 0;
	__conv3D_W1(streams, index, length,
		dX, IH, IW, dW, dY, 
		N, IC, OC);
}

#endif


#ifndef CONV3D_JNI_GEMMR
#define CONV3D_JNI_GEMMR

//Method:    kernel_remode
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1conv3D_kernel_1remode(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dW_address, jlong dCW_address,
	jint FH, jint FW, jint OC, jint IC)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float* dW = (float*)(intptr_t)dW_address;
	float* dCW = (float*)(intptr_t)dCW_address;
	__kernel_remode(stream, dW, dCW, FH, FW, OC, IC);
}

//Method:    conv3D_GemmR
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1conv3D_conv3D_1GemmR(JNIEnv *env, jclass cls,
	jlongArray streamArray, jint length,
	jlong dX_address, jint IH, jint IW,
	jlong dW_address, jlong dCW_address, jint FH, jint FW,
	jlong dY_address, jint OH, jint OW,
	jint N, jint IC, jint OC,
	jint sh, jint sw, jint ph, jint pw)
{
	jlong streams[10];
	env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *dX = (float*)(intptr_t)dX_address;
	float *dW = (float*)(intptr_t)dW_address;
	float *dCW = (float*)(intptr_t)dCW_address;
	float *dY = (float*)(intptr_t)dY_address;

	int index = 0;
	__conv3D_GemmR(streams, index, length,
		dX, IH, IW,
		dW, dCW, FH, FW,
		dY, OH, OW,
		N, IC, OC,
		sh, sw, ph, pw);
}

//Method:    conv3D_GemmR_texture
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1conv3D_conv3D_1GemmR_1texture(JNIEnv *env, jclass cls,
	jlongArray streamArray, jint length,
	jlong dX_address, jint IH, jint IW,
	jlong dW_address, jlong dCW_address, jint FH, jint FW,
	jlong dY_address, jint OH, jint OW,
	jint N, jint IC, jint OC,
	jint sh, jint sw, jint ph, jint pw)
{
	jlong streams[10];
	env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *dX = (float*)(intptr_t)dX_address;
	float *dW = (float*)(intptr_t)dW_address;
	float *dCW = (float*)(intptr_t)dCW_address;
	float *dY = (float*)(intptr_t)dY_address;

	int index = 0;
	cudaTextureObject_t texX = floatTexture(dX, (N*IH*IW*IC), env);
	__conv3D_GemmR_tex(streams, index, length,
		texX, dX, IH, IW,
		dW, dCW, FH, FW,
		dY, OH, OW,
		N, IC, OC,
		sh, sw, ph, pw);
	cudaError_t error = cudaDestroyTextureObject(texX); handleError(error);
}

//Method:    conv3D_GemmV2R
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1conv3D_conv3D_1GemmV2R(JNIEnv *env, jclass cls,
	jboolean useTexture, jlongArray streamArray, jint length, 
	jlong dX_address, jint IH, jint IW,
	jlong dW_address, jlong dCW_address, jint FH, jint FW,
	jlong dY_address, jint OH, jint OW, 
	jint N, jint IC, jint OC,
	jint sh, jint sw, jint ph, jint pw)
{
	jlong streams[10];
	env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *dX = (float*)(intptr_t)dX_address;
	float *dW = (float*)(intptr_t)dW_address;
	float *dCW = (float*)(intptr_t)dCW_address;
	float *dY = (float*)(intptr_t)dY_address;
	
	int index = 0;
	if (!useTexture) env = NULL;
	__conv3D_GemmV2R(env, streams, index, length,
		dX, IH, IW, (N*IH*IW*IC),//sizeX = N * IH * IW * IC
		dW, dCW, FH, FW,
		dY, OH, OW,
		IC, OC,
		sh, sw, ph, pw);
}

//Method:    conv3D_GemmR_W1
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1conv3D_conv3D_1GemmR_1W1(JNIEnv *env, jclass cls,
	jlongArray streamArray, jint length,
	jlong dX_address, jint IH, jint IW,
	jlong dW_address, jlong dCW_address,
	jlong dY_address,
	jint N, jint IC, jint OC)
{
	jlong streams[10];
	env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *dX = (float*)(intptr_t)dX_address;
	float *dW = (float*)(intptr_t)dW_address;
	float* dCW = (float*)(intptr_t)dCW_address;
	float *dY = (float*)(intptr_t)dY_address;
	int index = 0;
	__conv3D_GemmR_W1(streams, index, length,
		dX, IH, IW, dW, dCW, dY,
		N, IC, OC);
}

#endif