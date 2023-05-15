#pragma once

#ifndef DECONV3D_DW_GEMMSK_BUF_SUM_EX_H
#define DECONV3D_DW_GEMMSK_BUF_SUM_EX_H

#ifndef DECONV3D_DW_GEMMSK_BUF_SUM_EX_CALL
#define DECONV3D_DW_GEMMSK_BUF_SUM_EX_CALL

//LBX = log2(BLOCK_SIZE_X)
//LBY = log2(BLOCK_SIZE_Y)
//Wsize >> LB, part = (1 << LGZ) - 1

#define gbuf_sum4(stream, LBY, LBX, LTX, deltaW_buf, deltaW, part, sizeW)\
	general_bufsum_kernel4<LBY, LBX>\
		<<< (sizeW>>LBX>>LTX), dim3(1<<LBY, 1<<LBX), 0, stream >>>\
			(deltaW_buf, deltaW, part, sizeW)

#define gbuf_sum4_small(stream, LBY, deltaW_buf, deltaW, part, sizeW)\
	general_bufsum_kernel4<LBY, 4>\
		<<< (sizeW + 15) >> 4 , dim3(1<<LBY, 1<<4), 0, stream >>>\
			(deltaW_buf, deltaW, part, sizeW)

#endif


#ifndef DECONV3D_DW_GEMMSK_GENERAL_BUF_SUM_KERNEL
#define DECONV3D_DW_GEMMSK_GENERAL_BUF_SUM_KERNEL

template<int LBY, int LBX>
__global__ void general_bufsum_kernel4(
	const float* __restrict__ deltaW_buf,
	float* __restrict__ deltaW,
	int part, int sizeW)
{
	int ty = threadIdx.y, tx = threadIdx.x;
	__shared__ float4 Ws[1 << LBX][1 << LBY];

	int pstart = ty, pstride = blockDim.y;
	int step = (gridDim.x*blockDim.x), step4 = step << 2;
	int index = (blockIdx.x*blockDim.x) + tx;

	for (int index4 = index << 2; index4 < sizeW; index4 += step4)
	{
		float4 dw = *(float4*)(deltaW + index4);
		for (int p = pstart; p < part; p += pstride)
		{
			float4 dw_buf = *(float4*)(&deltaW_buf[(p * sizeW) + index4]);
			dw.x += dw_buf.x;
			dw.y += dw_buf.y;
			dw.z += dw_buf.z;
			dw.w += dw_buf.w;
		}
		Ws[tx][ty] = dw;
		__syncthreads();

		if (LBY >= 3) {//BLOCK_SIZE_Y = 8
			if (ty < 4) simdAdd4(Ws[tx][ty], Ws[tx][ty], Ws[tx][ty + 4]);
			__syncthreads();
		}
		if (LBY >= 2) {//BLOCK_SIZE_Y = 4
			if (ty < 2) simdAdd4(Ws[tx][ty], Ws[tx][ty], Ws[tx][ty + 2]);
			__syncthreads();
		}
		if (LBY >= 1) {//BLOCK_SIZE_Y = 2
			if (ty < 1) simdAdd4(Ws[tx][ty], Ws[tx][ty], Ws[tx][ty + 1]);
			__syncthreads();
		}
		if (ty == 0)*(float4*)(deltaW + index4) = Ws[tx][ty];
	}
}

#endif

void general_buf_summary(cudaStream_t stream,
	const float* __restrict__ deltaW_buf,
	float* __restrict__ deltaW,
	int part, int sizeW)
{
	if (sizeW < 256) { 
		if (part < 16) { buf_sum4_small(stream, deltaW_buf, deltaW, part, sizeW); return; }
		if (part < 32) { gbuf_sum4_small(stream, 1, deltaW_buf, deltaW, part, sizeW); return; }
		if (part < 64) { gbuf_sum4_small(stream, 2, deltaW_buf, deltaW, part, sizeW); return; }
		gbuf_sum4_small(stream, 3, deltaW_buf, deltaW, part, sizeW); return;
	}
	if (part < 16) { buf_sum4(stream, 5, 2, deltaW_buf, deltaW, part, sizeW); return; }
	if (part < 32) { gbuf_sum4(stream, 1, 5, 2, deltaW_buf, deltaW, part, sizeW); return; }
	if (part < 64) { gbuf_sum4(stream, 2, 5, 2, deltaW_buf, deltaW, part, sizeW); return; }
	gbuf_sum4(stream, 3, 5, 2, deltaW_buf, deltaW, part, sizeW);
}

#endif