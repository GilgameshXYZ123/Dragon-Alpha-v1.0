#pragma once

#ifndef SK_BUF_SUM_H
#define SK_BUF_SUM_H


#ifndef SK_BUF_SUM_CALL
#define SK_BUF_SUM_CALL

//LB = log2(BLOCK_SIZE)
//GZ = Wsize >> LB, part = (1 << LGZ) - 1

#define buf_sum4(stream, LB, LT, Cbuf, C, part, sizeC)\
	buf_summary_kernel4\
		<<< (sizeC>>LB>>LT), (1<<LB), 0, stream >>>\
			(Cbuf, C, part, sizeC)

#define buf_sum4_small(stream, Cbuf, C, part, sizeC)\
	buf_summary_kernel4\
		<<< 1, ((sizeC + 3) >> 2), 0, stream >>>\
			(Cbuf, C, part, sizeC)

#endif


#ifndef SK_BUF_SUM_KERNEL
#define SK_BUF_SUM_KERNEL

__global__ void buf_summary_kernel4(
	const float* __restrict__ Cbuf,
	float* __restrict__ C,
	int part, int sizeC)
{
	int step = (gridDim.x*blockDim.x), step4 = step << 2;
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	for (int index4 = index << 2; index4 < sizeC; index4 += step4)
	{
		float4 v = *(float4*)(C + index4);
		float4 c = make_float4(0, 0, 0, 0);
		for (int p = 0; p < part; p++)
		{
			float4 vbuf = *(float4*)(&Cbuf[(p * sizeC) + index4]);

			//two small number Minus the error of the last summation
			vbuf.x -= c.x;
			vbuf.y -= c.y;
			vbuf.z -= c.z;
			vbuf.w -= c.w;

			float4 t;
			t.x = v.x + vbuf.x;
			t.y = v.y + vbuf.y;
			t.z = v.z + vbuf.z;
			t.w = v.w + vbuf.w;

			//minus the big number first, to find the error
			c.x = (t.x - v.x) - vbuf.x;
			c.y = (t.y - v.y) - vbuf.y;
			c.z = (t.z - v.z) - vbuf.z;
			c.w = (t.w - v.w) - vbuf.w;

			v = t;
		}
		*(float4*)(C + index4) = v;
	}
}


#endif


void SKbuf_summary(cudaStream_t stream,
	const float* __restrict__ Cbuf,
	float* __restrict__ C,
	int part, int sizeC)
{
	if (sizeC < 256) { buf_sum4_small(stream, Cbuf, C, part, sizeC); return; }
	buf_sum4(stream, 5, 2, Cbuf, C, part, sizeC);
}

#endif