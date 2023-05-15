#include "frame.cuh"
#include "JNITool.cuh"
#include "Cuda_dconv3D_deltaX.cuh"
#include "micro.cuh"
#include "texture.cuh"
#include "dconv3D_dX.cuh"
#include "test.cuh"


#ifndef JNI_DECONV3D_DELTAX_ZERO_PADDING
#define JNI_DECONV3D_DELTAX_ZERO_PADDING

//Method:    dconv3D_deltaX_s1
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1dconv3D_1deltaX_dconv3D_1deltaX_1s1(JNIEnv *env, jclass cls,
	jlongArray streamArray, jint length,
	jlong d_deltaY_address, jint OH, jint OW,
	jlong dW_address, jint FH, jint FW,
	jlong d_deltaX_address, jint IH, jint IW,
	jint N, jint IC, jint OC,
	jint ph, jint pw)
{
	jlong streams[10];
	env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float *dW = (float*)(intptr_t)dW_address;
	float *d_deltaX = (float*)(intptr_t)d_deltaX_address;
	
	int index = 0;
	__dconv3D_deltaX_ZeroPadding_s1(streams, index, length,
		d_deltaY, OH, OW,
		dW, FH, FW,
		d_deltaX, IH, IW,
		N, IC, OC,
		ph, pw);
}

//Method:    dconv3D_deltaX_s1_texture
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1dconv3D_1deltaX_dconv3D_1deltaX_1s1_1texture(JNIEnv *env, jclass cls,
	jlongArray streamArray, jint length,
	jlong d_deltaY_address, jint OH, jint OW,
	jlong dW_address, jint FH, jint FW,
	jlong d_deltaX_address, jint IH, jint IW,
	jint N, jint IC, jint OC,
	jint ph, jint pw)
{
	jlong streams[10];
	env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float *dW = (float*)(intptr_t)dW_address;
	float *d_deltaX = (float*)(intptr_t)d_deltaX_address;
	
	int index = 0;
	cudaTextureObject_t texDy = floatTexture(d_deltaY, (N*OC*OH*OW), env);
	__dconv3D_deltaX_ZeroPadding_s1_tex(streams, index, length,
		texDy, d_deltaY, OH, OW,
		dW, FH, FW,
		d_deltaX, IH, IW,
		N, IC, OC,
		ph, pw);
	cudaError_t error = cudaDestroyTextureObject(texDy); handleError(error);
}

//Method:    dconv3D_deltaX_V2_s1
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1dconv3D_1deltaX_dconv3D_1deltaX_1V2_1s1(JNIEnv *env, jclass cls,
	jboolean useTexture, jlongArray streamArray, jint length,
	jlong d_deltaY_address, jint OH, jint OW, 
	jlong dW_address, jint FH, jint FW,
	jlong d_deltaX_address, jint IH, jint IW,
	jint N, jint IC, jint OC, jint ph, jint pw)
{
	jlong streams[10];
	env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float *dW = (float*)(intptr_t)dW_address;
	float *d_deltaX = (float*)(intptr_t)d_deltaX_address;
	
	int index = 0;
	if (!useTexture) env = NULL;//useTexture = false, env = null
	__dconv3D_deltaX_ZeroPaddingV2_s1(env, streams, index, length,
		d_deltaY, OH, OW, (N*OH*OW*OC),//sizeY = N*OH*OW*OC
		dW, FH, FW,
		d_deltaX, IH, IW,
		N, IC, OC,
		ph, pw);
}

//Method:    dconv3D_deltaX_W1
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1dconv3D_1deltaX_dconv3D_1deltaX_1W1(JNIEnv *env, jclass cls,
	jlongArray streamArray, jint length,
	jlong d_deltaY_address,
	jlong dW_address,
	jlong d_deltaX_address, jint IH, jint IW,
	jint N, jint IC, jint OC)
{
	jlong streams[10];
	env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float *dW = (float*)(intptr_t)dW_address;
	float *d_deltaX = (float*)(intptr_t)d_deltaX_address;
	int index = 0;
	__dconv3D_deltaX_W1(streams, index, length,
		d_deltaY,
		dW, 
		d_deltaX, IH, IW, 
		N, IC, OC);
}

#endif


#ifndef JNI_DECONV3D_DELTAX_KERNEL_SPLIT
#define JNI_DECONV3D_DELTAX_KERNEL_SPLIT

//Method:    ks_remode
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1dconv3D_1deltaX_ks_1remode(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dW_address, jint FH, jint FW,
	jlong dCW_address, jint CFH, jint CFW,
	jint OC, jint IC, jint sh, jint sw)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float* dW = (float*)(intptr_t)dW_address;
	float* dCW = (float*)(intptr_t)dCW_address;
	__ks_remode(stream, dW, FH, FW, dCW, CFH, CFW, OC, IC, sh, sw);
}

//Method:    ks_remodev2
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1dconv3D_1deltaX_ks_1remodev2(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dW_address, jint FH, jint FW,
	jlong dCW_address, jint CFH, jint CFW, 
	jint OC, jint IC, jint sh, jint sw)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float* dW = (float*)(intptr_t)dW_address;
	float* dCW = (float*)(intptr_t)dCW_address;
	__ks_remodev2(stream, dW, FH, FW, dCW, CFH, CFW, OC, IC, sh, sw);
}

//Method:    dconv3D_deltaX_kernelSplit
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1dconv3D_1deltaX_dconv3D_1deltaX_1kernelSplit(JNIEnv *env, jclass cls,
	jlongArray streamArray, jint length, 
	jlong d_deltaY_address, jint OH, jint OW,
	jlong dCW_address, jint FH, jint FW,
	jlong d_deltaX_address, jint IH, jint IW,
	jint N, jint IC, jint OC, 
	jint sh, jint sw, jint ph, jint pw)
{
	jlong streams[10];
	env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float *dCW = (float*)(intptr_t)dCW_address;
	float *d_deltaX = (float*)(intptr_t)d_deltaX_address;
	
	int index = 0;
	KS_init(N, IH, IW, FH, FW, OC, IC, sh, sw);
	__dconv3D_deltaX_ksR(streams, index, length,
		d_deltaY, OH, OW,
		dCW, FH, FW, CWstride,
		d_deltaX, IH, IW, 
		N, IC, OC,
		sh, sw, ph, pw);
}

//Method:    dconv3D_deltaX_ksImsR
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1dconv3D_1deltaX_dconv3D_1deltaX_1ksImsR(JNIEnv *env, jclass cls,
	jlongArray streamArray, jint length, 
	jlong d_deltaY_address, jint OH, jint OW,
	jlong dCW_address, jint FH, jint FW,
	jlong d_deltaX_address, jint IH, jint IW,
	jint N, jint IC, jint OC, 
	jint sh, jint sw, jint ph, jint pw)
{
	jlong streams[10];
	env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float *dCW = (float*)(intptr_t)dCW_address;
	float *d_deltaX = (float*)(intptr_t)d_deltaX_address;

	Ims_init(N, IH, IW, FH, FW, OC, IC, sh, sw);
	int index = 0;
	__dconv3D_deltaX_ksImsR(streams, index, length,
		d_deltaY, OH, OW,
		dCW, FH, FW, CWstride,
		d_deltaX, IH_slice, IW_slice,
		IC, OC,
		sh, sw, ph, pw);
}

//Method:    dconv3D_deltaX_ksImsR_texture
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1dconv3D_1deltaX_dconv3D_1deltaX_1ksImsR_1texture(JNIEnv *env, jclass cls,
	jlongArray streamArray, jint length,
	jlong d_deltaY_address, jint OH, jint OW,
	jlong dCW_address, jint FH, jint FW,
	jlong d_deltaX_address, jint IH, jint IW,
	jint N, jint IC, jint OC,
	jint sh, jint sw, jint ph, jint pw)
{
	jlong streams[10];
	env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float *dCW = (float*)(intptr_t)dCW_address;
	float *d_deltaX = (float*)(intptr_t)d_deltaX_address;

	Ims_init(N, IH, IW, FH, FW, OC, IC, sh, sw);
	int index = 0;
	cudaTextureObject_t texDy = floatTexture(d_deltaY, (N*OH*OW*OC), env);
	__dconv3D_deltaX_ksImsR_tex(streams, index, length,
		texDy, d_deltaY, OH, OW,
		dCW, FH, FW, CWstride,
		d_deltaX, IH_slice, IW_slice,
		IC, OC, 
		sh, sw, ph, pw);
	cudaError_t error = cudaDestroyTextureObject(texDy); handleError(error);
}

//Method:    dconv3D_deltaX_ksIms2R
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1dconv3D_1deltaX_dconv3D_1deltaX_1ksIms2R(JNIEnv *env, jclass cls,
	jlongArray streamArray, jint length, 
	jlong d_deltaY_address, jint OH, jint OW,
	jlong dCW_address, jint FH, jint FW,
	jlong d_deltaX_address, jint IH, jint IW,
	jint N, jint IC, jint OC,
	jint ph, jint pw)
{
	jlong streams[10];
	env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float *dCW = (float*)(intptr_t)dCW_address;
	float *d_deltaX = (float*)(intptr_t)d_deltaX_address;

	Ims2_init(N, IH, IW, FH, FW, OC, IC);
	int index = 0;
	__dconv3D_deltaX_ksIms2R(streams, index, length,
		d_deltaY, OH, OW,
		dCW, FH, FW, CWstride,
		d_deltaX, IH_slice, IW_slice,
		N, IC, OC,
		ph, pw);
}

//Method:    dconv3D_deltaX_ksIms2R_texture
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1dconv3D_1deltaX_dconv3D_1deltaX_1ksIms2R_1texture(JNIEnv *env, jclass cls,
	jlongArray streamArray, jint length,
	jlong d_deltaY_address, jint OH, jint OW,
	jlong dCW_address, jint FH, jint FW,
	jlong d_deltaX_address, jint IH, jint IW,
	jint N, jint IC, jint OC,
	jint ph, jint pw)
{
	jlong streams[10];
	env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float *dCW = (float*)(intptr_t)dCW_address;
	float *d_deltaX = (float*)(intptr_t)d_deltaX_address;

	Ims2_init(N, IH, IW, FH, FW, OC, IC);
	int index = 0;
	cudaTextureObject_t texDy = floatTexture(d_deltaY, (N*OH*OW*OC), env);
	__dconv3D_deltaX_ksIms2R_tex(streams, index, length,
		texDy, d_deltaY, OH, OW,
		dCW, FH, FW, CWstride,
		d_deltaX, IH_slice, IW_slice,
		IC, OC,
		ph, pw);
	cudaError_t error = cudaDestroyTextureObject(texDy); handleError(error);
}

//Method:    dconv3D_deltaX_ksV2_Ims2R
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1dconv3D_1deltaX_dconv3D_1deltaX_1ksV2_1Ims2R(JNIEnv *env, jclass cls,
	jboolean useTexture, jlongArray streamArray, jint length,
	jlong d_deltaY_address, jint OH, jint OW,
	jlong dCW_address, jint FH, jint FW,
	jlong d_deltaX_address, jint IH, jint IW,
	jint N, jint IC, jint OC, 
	jint ph, jint pw)
{
	jlong streams[10];
	env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float *dCW = (float*)(intptr_t)dCW_address;
	float *d_deltaX = (float*)(intptr_t)d_deltaX_address;

	V2_Ims2_init(N, IH, IW, FH, FW, OC, IC);
	int index = 0;
	__dconv3D_deltaX_ksV2_Ims2R(env, streams, index, length,
		d_deltaY, OH, OW, (N*OH*OW*OC),//sizeY = N*OH*OW*OC
		dCW, FH, FW, CWstride,
		d_deltaX, IH_slice, IW_slice,
		IC, OC, ph, pw);
}

#endif


#ifndef JNI_DECONV3D_DELTAX_CROSS_ADD
#define JNI_DECONV3D_DELTAX_CROSS_ADD

//Method:    dconv3D_deltaX_crossAdd
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1dconv3D_1deltaX_dconv3D_1deltaX_1crossAdd(JNIEnv *env, jclass cls,
	jlongArray streamArray, jint length,
	jlong d_deltaY_address, jint OH, jint OW,
	jlong dW_address, jint FH, jint FW,
	jlong d_deltaX_address , jint IH, jint IW,
	jint N, jint IC, jint OC,
	jint sh, jint sw, jint ph, jint pw)
{
	jlong streams[7];
	env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float *dW = (float*)(intptr_t)dW_address;
	float *d_deltaX = (float*)(intptr_t)d_deltaX_address;

	cudaStream_t stream2 = (cudaStream_t)(intptr_t)(streams[0]);//2L = sizeof(float)
	cudaError_t error = cudaMemsetAsync(d_deltaX, 0, (N*IH*IW*IC) << 2L, stream2); handleError(error);
	error = cudaStreamSynchronize(stream2); handleError(error);

	int index = 0;
	__dconv3D_deltaX_CrossAdd(streams, index, length,
		d_deltaY, OH, OW, 
		dW, FH, FW, 
		d_deltaX, IH, IW, 
		N, IC, OC, 
		sh, sw, ph, pw);
}

#endif
