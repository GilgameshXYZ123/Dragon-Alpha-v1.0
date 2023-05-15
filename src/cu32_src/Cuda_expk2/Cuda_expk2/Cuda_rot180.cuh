#pragma once

#ifndef CUDA_ROT_180
#define CUDA_ROT_180


#ifndef CUDA_ROT_180_CALL
#define CUDA_ROT_180_CALL

#define rot180_k4_small(stream, X, Y, IH, IW, IC, length) \
	rot180_kernel4\
		<<< 1, length, 0, stream >>>\
			(X, Y, IH, IW, IC, length)

#define rot180_k4(stream, LB, LT, X, Y, IH, IW, IC, length)\
	rot180_kernel4\
		<<< (length >> LB >> LT), (1<<LB), 0, stream>>>\
			(X, Y, IH, IW, IC, length)

#endif


#ifndef CUDA_ROT_180_KERNEL
#define CUDA_ROT_180_KERNEL

__global__ void rot180_kernel4(
	const float* __restrict__ X,
	      float* __restrict__ Y,
	int IH, int IW, int IC,
	int length)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x, index4 = index << 2;
	int step = (blockDim.x * gridDim.x), step4 = step << 2;

	const int IW_IC = IW * IC;
	const int IH_IW_IC = IH * IW_IC;
	for (; index4 < length; index4 += step4) 
	{
		int offset = index4;
		int n = offset / IH_IW_IC; offset -= n * IH_IW_IC;
		int ih = offset / IW_IC; offset -= ih * IW_IC;
		int iw = offset / IC, ic = offset - iw * IC;
		
		ih = IH - ih - 1; iw = IW - iw - 1;
		int rot_offset = ((n*IH + ih)*IW + iw)*IC + ic;
		*(float4*)(Y + rot_offset) = *(float4*)(X + index4);
	}
}

#endif 


#ifndef CUDA_ROT_180_FUNCTION
#define CUDA_ROT_180_FUNCTION

//IC % 4 == 0, IC >= 4
//so: length = X.lengthv = Y.lengthv = N*IH*IW*IC % 4 ==0
void __rot180(cudaStream_t stream,
	const float* X, float* Y,
	int IH, int IW, int IC,
	int length)
{
	if (length < 256) { rot180_k4_small(stream, X, Y, IH, IW, IC, length); return; }
	rot180_k4(stream, 5, 2, X, Y, IH, IW, IC, length);
}

#endif


//Method:    rot180
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_Cuda_1expk2_rot180(JNIEnv *env, jclass cls,
	jlong stream_address,
	jlong dX_address,
	jlong dY_address, 
	jint IH, jint IW, jint IC,
	jint length)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	float *dX = (float*)(intptr_t)dX_address;
	float *dY = (float*)(intptr_t)dY_address;
	__rot180(stream, dX, dY, IH, IW, IC, length);
}

#endif 
