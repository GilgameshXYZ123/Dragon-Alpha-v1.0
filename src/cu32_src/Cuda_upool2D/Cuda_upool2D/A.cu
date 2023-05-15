#include "frame.cuh"
#include "Cuda_upool2D.cuh"
#include "JNITool.cuh"
#include "micro.cuh" 
#include "upool2D.cuh"
#include "test.cuh"

//Method:    upool2D_max
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1upool2D_upool2D_1max(JNIEnv *env, jclass cls,
	jlongArray streamArray, jint length,
	jlong d_deltaY_address, jlong dY_address, jint OH, jint OW, 
	jint FH, jint FW,
	jlong d_deltaX_address, jlong dX_address, jint IH, jint IW,
	jint N, jint IC, 
	jint sh, jint sw, jint ph, jint pw)
{
	jlong streams[4];//the max streamsize is 4
	env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float *d_deltaX = (float*)(intptr_t)d_deltaX_address;
	float *dY = (float*)(intptr_t)dY_address;
	float *dX = (float*)(intptr_t)dX_address;
	int index = 0;
	__upool2D_max(streams, index, length,
		d_deltaY, dY, OH, OW, 
		FH, FW,
		d_deltaX, dX, IH, IW, 
		N, IC, 
		sh, sw, ph, pw);
}

//Method:    upool2D_max_Indexed
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1upool2D_upool2D_1max_1Indexed(JNIEnv *env, jclass cls, 
	jlongArray streamArray, jint length,
	jlong d_deltaY_address, jlong dIndex_address, jint OH, jint OW,
	jint FH, jint FW,
	jlong d_deltaX_address, jint IH, jint IW,
	jint N, jint IC, 
	jint sh, jint sw, jint ph, jint pw)
{
	jlong streams[4];//the max streamsize is 4
	env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float *d_deltaX = (float*)(intptr_t)d_deltaX_address;
	int *dIndex = (int*)(intptr_t)dIndex_address;
	int index = 0;
	__upool2D_max_indexed(streams, index, length, 
		d_deltaY, dIndex, OH, OW,
		FH, FW, 
		d_deltaX, IH, IW, 
		N, IC,
		sh, sw, ph, pw);
}


//Method:    upool2D_avg
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1upool2D_upool2D_1avg(JNIEnv *env, jclass cls,
	jlongArray streamArray, jint length,
	jlong d_deltaY_address, jint OH, jint OW,
	jint FH, jint FW,
	jlong d_deltaX_address, jint IH, jint IW,
	jint N, jint IC,
	jint sh, jint sw, jint ph, jint pw)
{
	jlong streams[4];//the max streamsize is 4
	env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float *d_deltaX = (float*)(intptr_t)d_deltaX_address;
	int index = 0;
	__upool2D_avg(streams, index, length, 
		d_deltaY, OH, OW, 
		FH, FW, 
		d_deltaX, IH, IW,
		N, IC, 
		sh, sw, ph, pw);
}

//Method:    upool2D_avg_tiled
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1upool2D_upool2D_1avg_1tiled(JNIEnv *env, jclass cls,
	jlongArray streamArray, jint length,
	jlong d_deltaY_address, jint OH, jint OW,
	jint FH, jint FW,
	jlong d_deltaX_address, jint IH, jint IW,
	jint N, jint IC,
	jint sh, jint sw, jint ph, jint pw)
{
	jlong streams[4];//the max streamsize is 4
	env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float *d_deltaX = (float*)(intptr_t)d_deltaX_address;
	int index = 0;
	__unpool2D_avg_tiled(streams, index, length,
		d_deltaY, OH, OW,
		FH, FW,
		d_deltaX, IH, IW,
		N, IC,
		sh, sw, ph, pw);
}


//Method:    upool2D_avg_ignore_padding
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1upool2D_upool2D_1avg_1ignore_1padding(JNIEnv *env, jclass cls,
	jlongArray streamArray, jint length, 
	jlong d_deltaY_address, jint OH, jint OW, 
	jint FH, jint FW, 
	jlong d_deltaX_address, jint IH, jint IW, 
	jint N, jint IC, 
	jint sh, jint sw, jint ph, jint pw)
{
	jlong streams[4];//the max streamsize is 4
	env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float *d_deltaX = (float*)(intptr_t)d_deltaX_address;
	int index = 0;
	__upool2D_avg_ip(streams, index, length,
		d_deltaY, OH, OW,
		FH, FW,
		d_deltaX, IH, IW,
		N, IC,
		sh, sw, ph, pw);
}

//Method:    upool2D_avg_ignore_padding_tiled
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1upool2D_upool2D_1avg_1ignore_1padding_1tiled(JNIEnv *env, jclass cls,
	jlongArray streamArray, jint length,
	jlong d_deltaY_address, jint OH, jint OW,
	jint FH, jint FW, 
	jlong d_deltaX_address, jint IH, jint IW,
	jint N, jint IC, 
	jint sh, jint sw, jint ph, jint pw)
{
	jlong streams[4];//the max streamsize is 4
	env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float *d_deltaX = (float*)(intptr_t)d_deltaX_address;
	int index = 0;
	__unpool2D_avg_ip_tiled(streams, index, length,
		d_deltaY, OH, OW,
		FH, FW,
		d_deltaX, IH, IW,
		N, IC,
		sh, sw, ph, pw);
}