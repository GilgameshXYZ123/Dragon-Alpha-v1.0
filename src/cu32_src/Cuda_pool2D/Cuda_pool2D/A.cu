#include "frame.cuh"
#include "JNITool.cuh"
#include "Cuda_pool2D.cuh"
#include "micro.cuh"
#include "pool2D.cuh"
#include "test.cuh"


//Method:    pool2D_max
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1pool2D_pool2D_1max(JNIEnv *env, jclass cls,
	jlongArray streamArray, jint length,
	jlong dX_address, jint IH, jint IW,
	jint FH, jint FW,
	jlong dY_address, jint OH, jint OW,
	jint N, jint IC,
	jint sh, jint sw, jint ph, jint pw)
{
	jlong streams[4];
	env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *dX = (float*)(intptr_t)dX_address;
	float *dY = (float*)(intptr_t)dY_address;
	int index = 0;
	__pool2D_max(streams, index, length,
		dX, IH, IW, 
		FH, FW, 
		dY, OH, OW, 
		N, IC, 
		sh, sw, ph, pw);
}

//Method:    pool2D_max_indexed
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1pool2D_pool2D_1max_1indexed(JNIEnv *env, jclass cls,
	jlongArray streamArray, jint length,
	jlong dX_address, jint IH, jint IW,
	jint FH, jint FW,
	jlong dY_address, jlong dIndex_address, jint OH, jint OW, 
	jint N, jint IC, 
	jint sh, jint sw, jint ph, jint pw)
{
	jlong streams[4];
	env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *dX = (float*)(intptr_t)dX_address;
	float *dY = (float*)(intptr_t)dY_address;
	int *dIndex = (int*)(intptr_t)dIndex_address;
	int index = 0;
	__pool2D_max_indexed(streams, index, length, 
		dX, IH, IW, 
		FH, FW, 
		dY, dIndex, OH, OW, 
		N, IC, 
		sh, sw, ph, pw);
}

//Method:    pool2D_avg
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1pool2D_pool2D_1avg(JNIEnv *env, jclass cls,
	jlongArray streamArray, jint length,
	jlong dX_address, jint IH, jint IW,
	jint FH, jint FW,
	jlong dY_address, jint OH, jint OW,
	jint N, jint IC,
	jint sh, jint sw, jint ph, jint pw)
{
	jlong streams[4];
	env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *dX = (float*)(intptr_t)dX_address;
	float *dY = (float*)(intptr_t)dY_address;
	int index = 0;
	__pool2D_avg(streams, index, length, 
		dX, IH, IW,
		FH, FW,
		dY, OH, OW,
		N, IC,
		sh, sw, ph, pw);
}

//Method:    pool2D_avg_ignore_padding
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1pool2D_pool2D_1avg_1ignore_1padding(JNIEnv *env, jclass cls,
	jlongArray streamArray, jint length, 
	jlong dX_address, jint IH, jint IW,
	jint FH, jint FW, 
	jlong dY_address,
	jint OH, jint OW, 
	jint N, jint IC, 
	jint sh, jint sw, jint ph, jint pw)
{
	jlong streams[4];
	env->GetLongArrayRegion(streamArray, 0, length, streams);
	float *dX = (float*)(intptr_t)dX_address;
	float *dY = (float*)(intptr_t)dY_address;
	int index = 0;
	__pool2D_avg_ip(streams, index, length,
		dX, IH, IW,
		FH, FW,
		dY, OH, OW,
		N, IC,
		sh, sw, ph, pw);
}