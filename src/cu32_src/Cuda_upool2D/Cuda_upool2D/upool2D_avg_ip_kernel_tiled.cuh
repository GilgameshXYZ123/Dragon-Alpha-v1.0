#pragma once

#ifndef UNPOOL_2D_AVERAGE_IGNORE_PADDING_KERNEL_TILED_H
#define UNPOOL_2D_AVERAGE_IGNORE_PADDING_KERNEL_TILED_H

//(1) FH * FW >= 2;
//(2) GN = IC; GN % 4 == 0, GN >= 4
//(3) GM = N * OH * OW;
//(4) GK = FH * FW >= 2
//(5) the memory padding alignment only effects N, IC, OC
#ifndef UNPOOL_2D_AVERAGE_IGNORE_PADDING_KERNEL_TILED_CALL
#define UNPOOL_2D_AVERAGE_IGNORE_PADDING_KERNEL_TILED_CALL

#define kavg_ip_tiled4(stream, LBY, LBX, ic_index, j_index, deltaY, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM)\
	avg_kernel_ignore_padding_tiled4\
		<<< dim3(GM>>LBX , GN>>LBY>>2), dim3(1<<LBX, 1<<LBY), 0, stream>>>\
			(deltaY, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, ic_index, j_index)

#define kavg_ip_tiled2(stream, LBY, LBX, ic_index, j_index, deltaY, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM)\
	avg_kernel_ignore_padding_tiled2\
		<<< dim3(GM>>LBX , GN>>LBY>>1), dim3(1<<LBX, 1<<LBY), 0, stream>>>\
			(deltaY, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, ic_index, j_index)

#define kavg_ip_tiled1(stream, LBY, LBX, ic_index, j_index, deltaY, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM)\
	avg_kernel_ignore_padding_tiled1\
		<<< dim3(GM>>LBX , GN>>LBY), dim3(1<<LBX, 1<<LBY), 0, stream>>>\
			(deltaY, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, ic_index, j_index)

#endif


//(Y: BLOCK_SIZE * 4)
#ifndef UNPOOL2D_AVG_IGNORE_PADDING_KERNEL_TIELD_4
#define UNPOOL2D_AVG_IGNORE_PADDING_KERNEL_TIELD_4

//<5, 0>: Size = 0.250000, Time = 0.750000 msec, Peformance = 357.913940 GFlop/s
//<4, 1>: Size = 0.250000, Time = 0.740000 msec, Peformance = 362.750580 GFlop/s
//<3, 2>: Size = 0.250000, Time = 1.464000 msec, Peformance = 183.357529 GFlop/s
//<2, 3>: Size = 0.250000, Time = 1.456000 msec, Peformance = 184.365005 GFlop/s
//FH = sh, FW = sw
__global__ void avg_kernel_ignore_padding_tiled4(
	const float* __restrict__ deltaY, int OH, int OW,
	int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC,
	int sh, int sw, int ph, int pw,
	int ic_index, int j_index)
{
	//prepare for GN = IC
	int ic = (((blockIdx.y*blockDim.y) + threadIdx.y) << 2) + ic_index;

	//prepare for GM = N * OH * OW
	int j = (blockIdx.x * blockDim.x) + threadIdx.x + j_index;
	const int OH_OW = OH * OW; get_n_oh_ow(j, n, oh, ow);
	const int toh = oh * sh - ph, tow = ow * sw - pw;
	deltaX += n * IH * IW * IC;

	//deltaY -> deltaX
	float4 dy = *(float4*)(&deltaY[j*IC + ic]);
	float4 dx; float alpha; get_alpha(oh, ow);
	dx.x = dy.x * alpha;
	dx.y = dy.y * alpha;
	dx.z = dy.z * alpha;
	dx.w = dy.w * alpha;
	
	for (int fh = 0; fh < FH; fh++) {
		int ih = toh + fh;
		for (int fw = 0; fw < FW; fw++) 
		{
			int iw = tow + fw;
			if ((ih < 0) || (ih >= IH) || (iw < 0) || (iw >= IW)) continue;
			*(float4*)(&get3d(deltaX, ih, iw, ic, IW, IC)) = dx;
		}
	}
}

#endif


//(Y: BLOCK_SIZE * 2)
#ifndef UNPOOL2D_AVG_IGNORE_PADDING_KERNEL_TIELD_2
#define UNPOOL2D_AVG_IGNORE_PADDING_KERNEL_TIELD_2

//<5, 0>: Size = 0.250000, Time = 0.750000 msec, Peformance = 357.913940 GFlop/s
//<4, 1>: Size = 0.250000, Time = 0.750000 msec, Peformance = 357.913940 GFlop/s
//<3, 2>: Size = 0.250000, Time = 2.958000 msec, Peformance =  90.748962 GFlop/s
//<2, 3>: Size = 0.250000, Time = 2.704000 msec, Peformance =  99.273460 GFlop/s
//FH = sh, FW = sw
__global__ void avg_kernel_ignore_padding_tiled2(
	const float* __restrict__ deltaY, int OH, int OW,
	int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC,
	int sh, int sw, int ph, int pw,
	int ic_index, int j_index)
{
	//prepare for GN = IC
	int ic = (((blockIdx.y*blockDim.y) + threadIdx.y) << 1) + ic_index;

	//prepare for GM = N * OH * OW
	int j = (blockIdx.x * blockDim.x) + threadIdx.x + j_index;
	const int OH_OW = OH * OW; get_n_oh_ow(j, n, oh, ow);
	const int toh = oh * sh - ph, tow = ow * sw - pw;
	deltaX += n * IH * IW * IC;

	//deltaY -> deltaX
	float2 dy = *(float2*)(&deltaY[j*IC + ic]);
	float2 dx; float alpha; get_alpha(oh, ow);
	dx.x = dy.x * alpha;
	dx.y = dy.y * alpha;

	for (int fh = 0; fh < FH; fh++) {
		int ih = toh + fh;
		for (int fw = 0; fw < FW; fw++)
		{
			int iw = tow + fw;
			if ((ih < 0) || (ih >= IH) || (iw < 0) || (iw >= IW)) continue;
			*(float2*)(&get3d(deltaX, ih, iw, ic, IW, IC)) = dx;
		}
	}
}

#endif


//(Y: BLOCK_SIZE * 1)
#ifndef UNPOOL2D_AVG_IGNORE_PADDING_KERNEL_TIELD_1
#define UNPOOL2D_AVG_IGNORE_PADDING_KERNEL_TIELD_1

//<5, 0>: Size = 0.250000, Time = 0.754000 msec, Peformance = 356.015167 GFlop/s
//<4, 1>: Size = 0.250000, Time = 0.800000 msec, Peformance = 335.544281 GFlop/s
//<3, 2>: Size = 0.250000, Time = 2.838000 msec, Peformance =  94.586128 GFlop/s
//<2, 3>: Size = 0.250000, Time = 2.934000 msec, Peformance =  91.491287 GFlop/s
//FH = sh, FW = sw
__global__ void avg_kernel_ignore_padding_tiled1(
	const float* __restrict__ deltaY, int OH, int OW,
	int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC,
	int sh, int sw, int ph, int pw,
	int ic_index, int j_index)
{
	//prepare for GN = IC
	int ic = ((blockIdx.y*blockDim.y) + threadIdx.y) + ic_index;

	//prepare for GM = N * OH * OW
	int j = (blockIdx.x * blockDim.x) + threadIdx.x + j_index;
	const int OH_OW = OH * OW; get_n_oh_ow(j, n, oh, ow);
	const int toh = oh * sh - ph, tow = ow * sw - pw;
	deltaX += n * IH * IW * IC;

	//deltaY -> deltaX
	float dy = deltaY[j*IC + ic];
	float dx; float alpha; get_alpha(oh, ow);
	dx = dy * alpha;

	for (int fh = 0; fh < FH; fh++) {
		int ih = toh + fh;
		for (int fw = 0; fw < FW; fw++)
		{
			int iw = tow + fw;
			if ((ih < 0) || (ih >= IH) || (iw < 0) || (iw >= IW)) continue;
			get3d(deltaX, ih, iw, ic, IW, IC) = dx;
		}
	}
}

#endif

#endif