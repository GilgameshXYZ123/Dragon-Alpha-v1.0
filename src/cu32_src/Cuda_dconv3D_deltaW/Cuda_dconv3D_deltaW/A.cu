#include "frame.cuh"
#include "JNITool.cuh"
#include "micro.cuh"
#include "complex.cuh"
#include "texture.cuh"

#include "Cuda_deconv3D_deltaW.cuh"
#include "dconv3d_dW.cuh"
#include "test.cuh"


#ifndef JNI_D3DW_GEMM
#define JNI_D3DW_GEMM

//Method:    dconv3D_deltaW
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1dconv3D_1deltaW_dconv3D_1deltaW(JNIEnv *env, jclass cls,
	jlongArray streamArray, jint length, 
	jlong dX_address, jint IH, jint IW,
	jlong d_deltaY_address, jint OH, jint OW,
	jlong d_deltaW_address, jint FH, jint FW, 
	jint N, jint IC, jint OC,
	jint sh, jint sw, jint ph, jint pw)
{
	jlong streams[12];
	env->GetLongArrayRegion(streamArray, 0, length, streams);
	float* dX = (float*)(intptr_t)dX_address;
	float* d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float* d_deltaW = (float*)(intptr_t)d_deltaW_address;
	int index = 0;
	__dconv3D_deltaW_Gemm(streams, index, length,
		dX, IH, IW,
		d_deltaY, OH, OW,
		d_deltaW, FH, FW,
		N, IC, OC,
		sh, sw, ph, pw);
}

//Method:    dconv3D_deltaW_W1
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1dconv3D_1deltaW_dconv3D_1deltaW_1W1(JNIEnv *env, jclass cls,
	jlongArray streamArray, jint length,
	jlong dX_address, jint IH, jint IW,
	jlong d_deltaY_address, 
	jlong d_deltaW_address,
	jint N, jint IC, jint OC)
{
	jlong streams[12];
	env->GetLongArrayRegion(streamArray, 0, length, streams);
	float* dX = (float*)(intptr_t)dX_address;
	float* d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float* d_deltaW = (float*)(intptr_t)d_deltaW_address;
	int index = 0;
	__dconv3D_deltaW_W1(streams, index, length,
		dX, IH, IW,
		d_deltaY,
		d_deltaW,
		N, IC, OC);
}

#endif


#ifndef JNI_D3DW_GEMMSK
#define JNI_D3DW_GEMMSK

//Method:    buf_summary
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1dconv3D_1deltaW_buf_1summary(JNIEnv *env, jclass cls,
	jlong stream_address, 
	jlong d_deltaW_buf_address,
	jlong d_deltaW_address,
	jint part, jint sizeW)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float* d_deltaW = (float*)(intptr_t)d_deltaW_address;
	float* d_deltaW_buf = (float*)(intptr_t)d_deltaW_buf_address;
	buf_summary(stream, d_deltaW_buf, d_deltaW, part, sizeW);
}

//Method:    dconv3D_deltaW_GemmSK
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1dconv3D_1deltaW_dconv3D_1deltaW_1GemmSK(JNIEnv *env, jclass cls,
	jlongArray streamArray, jint length, jint GridZ,
	jlong dX_address, jint IH, jint IW,
	jlong d_deltaY_address, jint OH, jint OW,
	jlong d_deltaW_address,
	jlong d_deltaW_buf_address, jint FH, jint FW,
	jint N, jint IC, jint OC,
	jint sh, jint sw, jint ph, jint pw)
{
	jlong streams[12];
	env->GetLongArrayRegion(streamArray, 0, length, streams);
	float* dX = (float*)(intptr_t)dX_address;
	float* d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float* d_deltaW = (float*)(intptr_t)d_deltaW_address;
	float* d_deltaW_buf = (float*)(intptr_t)d_deltaW_buf_address;

	GEMMSK_init(GridZ, OH, OW, FH, FW, N, IC, OC);
	int index = 0; 
	__dconv3D_deltaW_GemmSK(streams, index, length, GridZ,
		dX, IH, IW,
		d_deltaY, OH, OW,
		d_deltaW, d_deltaW_buf, FH, FW,
		N, IC, OC,
		sh, sw, ph, pw);
}

//Method:    dconv3D_deltaW_GemmV2SK
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1dconv3D_1deltaW_dconv3D_1deltaW_1GemmV2SK(JNIEnv *env, jclass cls,
	jlongArray streamArray, jint length, jint GridZ,
	jlong dX_address, jint IH, jint IW,
	jlong d_deltaY_address, jint OH, jint OW,
	jlong d_deltaW_address,
	jlong d_deltaW_buf_address, jint FH, jint FW,
	jint N, jint IC, jint OC, 
	jint sh, jint sw, jint ph, jint pw)
{
	jlong streams[12];
	env->GetLongArrayRegion(streamArray, 0, length, streams);
	float* dX = (float*)(intptr_t)dX_address;
	float* d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float* d_deltaW = (float*)(intptr_t)d_deltaW_address;
	float* d_deltaW_buf = (float*)(intptr_t)d_deltaW_buf_address;

	int index = 0;
	__dconv3D_deltaW_GemmV2SK(streams, index, length, GridZ,
		dX, IH, IW,
		d_deltaY, OH, OW,
		d_deltaW, d_deltaW_buf, FH, FW,
		N, IC, OC,
		sh, sw, ph, pw);
}

//Method:    dconv3D_deltaW_GemmSK_W1
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_math_Cuda_1dconv3D_1deltaW_dconv3D_1deltaW_1GemmSK_1W1(JNIEnv *env, jclass cls,
	jlongArray streamArray, jint length, jint GridZ, 
	jlong dX_address, jint IH, jint IW, 
	jlong d_deltaY_address, 
	jlong d_deltaW_address,
	jlong d_deltaW_buf_address,
	jint N, jint IC, jint OC)
{
	jlong streams[12];
	env->GetLongArrayRegion(streamArray, 0, length, streams);
	float* dX = (float*)(intptr_t)dX_address;
	float* d_deltaY = (float*)(intptr_t)d_deltaY_address;
	float* d_deltaW = (float*)(intptr_t)d_deltaW_address;
	float* d_deltaW_buf = (float*)(intptr_t)d_deltaW_buf_address;

	GEMMSK_init(GridZ, IH, IW, 1, 1, N, IC, OC);
	int index = 0;
	__dconv3D_deltaW_GemmSK_W1(streams, index, length, GridZ,
		dX, IH, IW,
		d_deltaY,
		d_deltaW, d_deltaW_buf,
		IC, OC);
}

#endif