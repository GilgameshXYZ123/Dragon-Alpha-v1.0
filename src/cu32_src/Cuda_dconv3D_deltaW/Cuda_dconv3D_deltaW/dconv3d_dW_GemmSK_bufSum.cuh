#pragma once

#ifndef DECONV3D_DW_GEMMSK_BUF_SUM_H
#define DECONV3D_DW_GEMMSK_BUF_SUM_H

//part = GZ - 1:
//<1> deltaW_buf[part, OC, FH, FW, IC]
//<2> deltaW[1, OC, FH, FW, IC]
//<3> deltaW_buf + deltaW = deltaW[OC, FH, FW, IC]
#ifndef DECONV3D_DW_GEMMSK_BUF_SUM_CALL
#define DECONV3D_DW_GEMMSK_BUF_SUM_CALL

//LB = log2(BLOCK_SIZE)
//GZ = Wsize >> LB, part = (1 << LGZ) - 1

#define buf_sum4(stream, LB, LT, deltaW_buf, deltaW, part, sizeW)\
	buf_summary_kernel4\
		<<< (sizeW>>LB>>LT), (1<<LB), 0, stream >>>\
			(deltaW_buf, deltaW, part, sizeW)

#define buf_sum4_small(stream, deltaW_buf, deltaW, part, sizeW)\
	buf_summary_kernel4\
		<<< 1, ((sizeW + 3) >> 2), 0, stream >>>\
			(deltaW_buf, deltaW, part, sizeW)

#endif


#ifndef DECONV3D_DW_GEMMSK_BUF_SUM_KERNEL
#define DECONV3D_DW_GEMMSK_BUF_SUM_KERNEL

__global__ void buf_summary_kernel4(
	const float* __restrict__ deltaW_buf,
	float* __restrict__ deltaW,
	int part, int sizeW)
{
	int step = (gridDim.x*blockDim.x), step4 = step << 2;
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	for (int index4 = index << 2; index4 < sizeW; index4 += step4)
	{
		float4 dw = *(float4*)(deltaW + index4);
		float4 c = make_float4(0, 0, 0, 0);
		for (int p = 0; p < part; p++)
		{
			float4 dw_buf = *(float4*)(&deltaW_buf[(p * sizeW) + index4]);

			//two small number Minus the error of the last summation
			dw_buf.x -= c.x;
			dw_buf.y -= c.y;
			dw_buf.z -= c.z;
			dw_buf.w -= c.w;

			float4 t;
			t.x = dw.x + dw_buf.x;
			t.y = dw.y + dw_buf.y;
			t.z = dw.z + dw_buf.z;
			t.w = dw.w + dw_buf.w;

			//minus the big number first, to find the error
			c.x = (t.x - dw.x) - dw_buf.x;
			c.y = (t.y - dw.y) - dw_buf.y;
			c.z = (t.z - dw.z) - dw_buf.z;
			c.w = (t.w - dw.w) - dw_buf.w;

			dw = t;
		}
		*(float4*)(deltaW + index4) = dw;
	}
}

#endif


void buf_summary(cudaStream_t stream,
	const float* __restrict__ deltaW_buf,
	float* __restrict__ deltaW,
	int part, int sizeW)
{
	if (sizeW < 256) { buf_sum4_small(stream, deltaW_buf, deltaW, part, sizeW); return; }
	buf_sum4(stream, 5, 2, deltaW_buf, deltaW, part, sizeW);
}

#endif