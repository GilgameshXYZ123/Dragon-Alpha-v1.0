#pragma once

#ifndef POOL2D_AVG_IGNORE_PADDING_KERNEL_H
#define POOL2D_AVG_IGNORE_PADDING_KERNEL_H

//(1) FH * FW >= 2;
//(2) GN = IC; GN % 4 == 0, GN >= 4
//(3) GM = N * OH * OW;
//(4) GK = FH * FW >= 2
//(5) the memory padding alignment only effects N, IC, OC

#ifndef POOL2D_AVG_IGNORE_PADDING_KERNEL_CALL
#define POOL2D_AVG_IGNORE_PADDING_KERNEL_CALL

#define kavg4_ip(stream, LBY, LBX, X, IH, IW, FH, FW, Y, OH, OW, sh, sw, ph, pw, GN, GM, ic_index, j_index)\
	avg_kernel_4_ignore_padding\
		<<< dim3(GM>>LBX , GN>>LBY>>2), dim3(1<<LBX, 1<<LBY), 0, stream>>>\
			(X, IH, IW, FH, FW, Y, OH, OW, IC, sh, sw, ph, pw, ic_index, j_index)

#define kavg2_ip(stream, LBY, LBX, X, IH, IW, FH, FW, Y, OH, OW, sh, sw, ph, pw, GN, GM, ic_index, j_index)\
	avg_kernel_2_ignore_padding\
		<<< dim3(GM>>LBX , GN>>LBY>>1), dim3(1<<LBX, 1<<LBY), 0, stream>>>\
			(X, IH, IW, FH, FW, Y, OH, OW, IC, sh, sw, ph, pw, ic_index, j_index)

#define kavg1_ip(stream, LBY, LBX, X, IH, IW, FH, FW, Y, OH, OW, sh, sw, ph, pw, GN, GM, ic_index, j_index)\
	avg_kernel_1_ignore_padding\
		<<< dim3(GM>>LBX , GN>>LBY), dim3(1<<LBX, 1<<LBY), 0, stream>>>\
			(X, IH, IW, FH, FW, Y, OH, OW, IC, sh, sw, ph, pw, ic_index, j_index)

#endif


//(Y: BLOCK_SIZE * 4)
#ifndef POOL2D_AVG_IGNORE_PADDING_KERNEL_4
#define POOL2D_AVG_IGNORE_PADDING_KERNEL_4

//<2, 2>: Size = 0.062500, Time = 1.538000 msec, Performance = 43.633850 GFlop/s [ 16, 4]
//<2, 3>: Size = 0.125000, Time = 2.796000 msec, Performance = 48.003475 GFlop/s [ 16, 8]
//<3, 2>: Size = 0.125000, Time = 2.312000 msec, Performance = 58.052647 GFlop/s [ 32, 4]
//<4, 1>: Size = 0.125000, Time = 2.208000 msec, Performance = 60.787010 GFlop/ss [ 64, 2]
//<5, 0>: Size = 0.125000, Time = 2.024000 msec, Performance = 66.313103 GFlop/s [128, 1]
__global__ void avg_kernel_4_ignore_padding(
	const float* __restrict__ X, int IH, int IW,
	int FH, int FW,
	float* __restrict__ Y, int OH, int OW,
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
	X += n * IH * IW * IC;

	float4 v = make_float4(0, 0, 0, 0);
	int count = 0;
	for (int fh = 0; fh < FH; fh++)
	{
		int ih = toh + fh;
		bool lflag = (ih >= 0) && (ih < IH);
		for (int fw = 0; fw < FW; fw++)
		{
			int iw = tow + fw;
			if (lflag && iw >= 0 && iw < IW) {
				float4 x = *(float4*)(&get3d(X, ih, iw, ic, IW, IC));
				simdAdd4(v, v, x);
				count++;
			}
		}
	}
	simdSDiv4(v, v, count);
	*(float4*)(&Y[j*IC + ic]) = v;
}

#endif


//(Y: BLOCK_SIZE * 2)
#ifndef POOL2D_AVG_IGNORE_PADDING_KERNEL_2
#define POOL2D_AVG_IGNORE_PADDING_KERNEL_2

//<2, 2>: Size = 0.062500, Time = 2.796000 msec, Performance = 24.001738 GFlop/s [ 4, 8]
//<2, 3>: Size = 0.062500, Time = 1.862000 msec, Performance = 36.041279 GFlop/s [ 8, 8]
//<3, 2>: Size = 0.062500, Time = 1.740000 msec, Performance = 38.568310 GFlop/s [16, 4]
//<4, 1>: Size = 0.062500, Time = 1.626000 msec, Performance = 41.272358 GFlop/s [32, 2]
//<5, 0>: Size = 0.062500, Time = 1.516000 msec, Performance = 44.267056 GFlop/s [64, 1]
__global__ void avg_kernel_2_ignore_padding(
	const float* __restrict__ X, int IH, int IW,
	int FH, int FW,
	float* __restrict__ Y, int OH, int OW,
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
	X += n * IH * IW * IC;

	float2 v = make_float2(0, 0);
	int count = 0;
	for (int fh = 0; fh < FH; fh++)
	{
		int ih = toh + fh;
		bool lflag = (ih >= 0) && (ih < IH);
		for (int fw = 0; fw < FW; fw++)
		{
			int iw = tow + fw;
			if (lflag && iw >= 0 && iw < IW) {
				float2 x = *(float2*)(&get3d(X, ih, iw, ic, IW, IC));
				simdAdd2(v, v, x);
				count++;
			}
		}
	}
	simdSDiv2(v, v, count);
	*(float2*)(&Y[j*IC + ic]) = v;
}

#endif


//(Y: BLOCK_SIZE * 1)
#ifndef POOL2D_AVG_IGNORE_PADDING_KERNEL_1
#define POOL2D_AVG_IGNORE_PADDING_KERNEL_1

//<2, 2>: Size = 0.062500, Time = 5.292000 msec, Performance = 12.681190 GFlop/s [ 4, 4]
//<2, 3>: Size = 0.062500, Time = 3.634000 msec, Performance = 18.466940 GFlop/s [ 4, 8]
//<3, 2>: Size = 0.062500, Time = 2.842000 msec, Performance = 23.613251 GFlop/s [ 8, 4]
//<4, 1>: Size = 0.062500, Time = 2.562000 msec, Performance = 26.193933 GFlop/s [16, 2]
//<5, 0>: Size = 0.062500, Time = 2.664000 msec, Performance = 25.191011 GFlop/s [32, 1]
__global__ void avg_kernel_1_ignore_padding(
	const float* __restrict__ X, int IH, int IW,
	int FH, int FW,
	float* __restrict__ Y, int OH, int OW,
	int IC,
	int sh, int sw, int ph, int pw,
	int ic_index, int j_index)
{
	//prepare for GN = IC
	int ic = (blockIdx.y*blockDim.y) + threadIdx.y + ic_index;

	//prepare for GM = N * OH * OW
	int j = (blockIdx.x * blockDim.x) + threadIdx.x + j_index;
	const int OH_OW = OH * OW; get_n_oh_ow(j, n, oh, ow);
	const int toh = oh * sh - ph, tow = ow * sw - pw;
	X += n * IH * IW * IC;

	float v = 0.0f;
	int count = 0;
	for (int fh = 0; fh < FH; fh++)
	{
		int ih = toh + fh;
		bool lflag = (ih >= 0) && (ih < IH);
		for (int fw = 0; fw < FW; fw++)
		{
			int iw = tow + fw;
			if (lflag && iw >= 0 && iw < IW) {
				float x = get3d(X, ih, iw, ic, IW, IC);
				v += x; count++;
			}
		}
	}
	Y[j*IC + ic] = v / count;
}

#endif

#endif