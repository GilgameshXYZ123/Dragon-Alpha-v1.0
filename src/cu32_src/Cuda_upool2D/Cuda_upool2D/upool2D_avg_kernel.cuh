#pragma once

#ifndef UNPOOL2D_AVERAGE_KERNEL_H
#define UNPOOL2D_AVERAGE_KERNEL_H

//GN = IC
//GM = N * IH * IW
//GK = FH * FW
//LBY = log2(blockDim.y)
//LBX = log2(blockDim.x)
//We have:
//(1) FH * FW >=2
//(2) GN % 4==0, GN >= 4
//(3) GM % 4==0, GM >= 4
//(4) GK = FH * FW >= 2
#ifndef UNPOOL_2D_AVERAGE_KERNEL_CALL
#define UNPOOL_2D_AVERAGE_KERNEL_CALL

#define kavg81(stream, LBY, LBX, ic_index, j_index, deltaY, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM)\
	avg_kernel_8_1\
		<<< dim3(GM>>LBX, GN>>LBY>>3), dim3(1<<LBX, 1<<LBY), 0, stream >>>\
			(deltaY,OH,OW, FH, FW, deltaX,IH,IW, IC, sh,sw,ph,pw, ic_index, j_index)

#define kavg41(stream, LBY, LBX, ic_index, j_index, deltaY, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM)\
	avg_kernel_4_1\
		<<< dim3(GM>>LBX, GN>>LBY>>2), dim3(1<<LBX, 1<<LBY), 0, stream >>>\
			(deltaY,OH,OW, FH, FW, deltaX,IH,IW, IC, sh,sw,ph,pw, ic_index, j_index)

#define kavg21(stream, LBY, LBX, ic_index, j_index, deltaY, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM)\
	avg_kernel_2_1\
		<<< dim3(GM>>LBX, GN>>LBY>>1), dim3(1<<LBX, 1<<LBY), 0, stream >>>\
			(deltaY,OH,OW, FH, FW, deltaX,IH,IW, IC, sh,sw,ph,pw, ic_index, j_index)

#define kavg11(stream, LBY, LBX, ic_index, j_index, deltaY, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM)\
	avg_kernel_1_1\
		<<< dim3(GM>>LBX, GN>>LBY), dim3(1<<LBX, 1<<LBY), 0, stream >>>\
			(deltaY,OH,OW, FH, FW, deltaX,IH,IW, IC, sh,sw,ph,pw, ic_index, j_index)

#endif


//(Y: BLOCK_SIZE*8)
#ifndef UNPOOL2D_AVG_KERNEL_8_1
#define UNPOOL2D_AVG_KERNEL_8_1

//<4, 1>: Size = 1.000000, Time = 1.948000 msec, Peformance = 551.202148 GFlop/s [128,  2]
//<3, 3>: Size = 1.000000, Time = 2.094000 msec, Peformance = 512.770630 GFlop/s [ 64,  8]
//<2, 4>: Size = 1.000000, Time = 2.126000 msec, Peformance = 505.052551 GFlop/s [ 32, 16]
//<1, 5>: Size = 1.000000, Time = 2.082000 msec, Peformance = 515.726135 GFlop/s [ 16, 32]

//<3, 2>: Size = 1.000000, Time = 2.146000 msec, Peformance = 500.345642 GFlop/s [ 64,  4]
//<2, 3>: Size = 1.000000, Time = 2.120000 msec, Peformance = 506.481964 GFlop/s [ 32,  8]
//<1, 4>: Size = 1.000000, Time = 2.138000 msec, Peformance = 502.217804 GFlop/s [ 16, 16]
//<0, 5>: Size = 1.000000, Time = 2.166000 msec, Peformance = 495.725677 GFlop/s [  8, 32]
__global__ void avg_kernel_8_1(
	const float* __restrict__ deltaY, int OH, int OW,
	int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC,
	int sh, int sw, int ph, int pw,
	int ic_index, int j_index)
{
	//prepare for GN = IC
	const int ic0 = (((blockIdx.y*blockDim.y) + threadIdx.y) << 3) + ic_index, ic4 = ic0 + 4;

	//prepare fo GM = N * IH * IW
	int j = ((blockIdx.x*blockDim.x) + threadIdx.x) + j_index;
	const int oph = FH - ph - 1, opw = FW - pw - 1;
	const int IH_IW = IH * IW; get_n_ih_iw(j, n, ih, iw); j *= IC;
	const int OHp = OH * sh - sh + 1, OWp = OW * sw - sw + 1;
	const int tih = ih - oph, tiw = iw - opw;
	deltaY += n * OH * OW * IC;

	//find (fhs, fws) to compress deltaXpe -> deltaX
	int fhs = 0, fws = 0; FIND_FHS_FWS(fhs, fws, tih, tiw, max_kernel_8_1_end);
	const int FHr = (FH - fhs + sh - 1) / sh;
	const int FWr = (FW - fws + sw - 1) / sw;
	const int GKr = FHr * FWr;
	const int toh = (tih + fhs) / sh, tow = (tiw + fws) / sw;

	//compute area------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0);
	float4 v1 = make_float4(0, 0, 0, 0);
	float alpha = 1.0f / (FH * FW);
	for (int k = 0; k < GKr; k++)
	{
		int fhr = k / FWr, fwr = k - fhr * FWr;
		int oh = toh + fhr, ow = tow + fwr;
		if (oh < OH && ow < OW) {
			float4 dy0 = *(float4*)(&get3d(deltaY, oh, ow, ic0, OW, IC));
			float4 dy1 = *(float4*)(&get3d(deltaY, oh, ow, ic4, OW, IC));
			simdMM4(v0, alpha, dy0);
			simdMM4(v1, alpha, dy1);
		}
	}

	*(float4*)(&deltaX[j + ic0]) = v0;
	*(float4*)(&deltaX[j + ic4]) = v1;
}

#endif


//(Y: BLOCK_SIZE*4)
#ifndef UNPOOL2D_AVG_KERNEL_4_1
#define UNPOOL2D_AVG_KERNEL_4_1

//<3, 2>: Size = 1.000000, Time = 3.812000 msec, Peformance = 281.674133 GFlop/s [32,  4]
//<2, 3>: Size = 1.000000, Time = 3.828000 msec, Peformance = 280.496796 GFlop/s [16,  8]
//<1, 4>: Size = 1.000000, Time = 3.872000 msec, Peformance = 277.309326 GFlop/s [ 8, 16]
//<0, 5>: Size = 1.000000, Time = 3.960000 msec, Peformance = 271.146881 GFlop/s [ 4, 32]
__global__ void avg_kernel_4_1(
	const float* __restrict__ deltaY, int OH, int OW,
	int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC,
	int sh, int sw, int ph, int pw,
	int ic_index, int j_index)
{
	//prepare for GN = IC
	const int ic0 = (((blockIdx.y*blockDim.y) + threadIdx.y) << 2) + ic_index;

	//prepare fo GM = N * IH * IW
	int j = ((blockIdx.x*blockDim.x) + threadIdx.x) + j_index;
	const int oph = FH - ph - 1, opw = FW - pw - 1;
	const int IH_IW = IH * IW; get_n_ih_iw(j, n, ih, iw); j *= IC;
	const int OHp = OH * sh - sh + 1, OWp = OW * sw - sw + 1;
	const int tih = ih - oph, tiw = iw - opw;
	deltaY += n * OH * OW * IC;

	//find (fhs, fws) to compress deltaXpe -> deltaX
	int fhs = 0, fws = 0; FIND_FHS_FWS(fhs, fws, tih, tiw, max_kernel_8_1_end);
	const int FHr = (FH - fhs + sh - 1) / sh;
	const int FWr = (FW - fws + sw - 1) / sw;
	const int GKr = FHr * FWr;
	const int toh = (tih + fhs) / sh, tow = (tiw + fws) / sw;

	//compute area------------------------------------------
	float4 v = make_float4(0, 0, 0, 0);
	float alpha = 1.0f / (FH * FW);
	for (int k = 0; k < GKr; k++)
	{
		int fhr = k / FWr, fwr = k - fhr * FWr;
		int oh = toh + fhr, ow = tow + fwr;
		if (oh < OH && ow < OW) {
			float4 dy = *(float4*)(&get3d(deltaY, oh, ow, ic0, OW, IC));
			simdMM4(v, alpha, dy);
		}
	}

	*(float4*)(&deltaX[j + ic0]) = v;
}

#endif


//(Y: BLOCK_SIZE*2)
#ifndef UNPOOL2D_AVG_KERNEL_2_1
#define UNPOOL2D_AVG_KERNEL_2_1

//<3, 2>: Size = 1.000000, Time = 7.446000 msec, Peformance = 144.203827 GFlop/s [64,  4]
//<2, 3>: Size = 1.000000, Time = 7.494000 msec, Peformance = 143.280197 GFlop/s [32,  8]
//<1, 4>: Size = 1.000000, Time = 7.568000 msec, Peformance = 141.879196 GFlop/s [16, 16]

__global__ void avg_kernel_2_1(
	const float* __restrict__ deltaY, int OH, int OW,
	int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC,
	int sh, int sw, int ph, int pw,
	int ic_index, int j_index)
{
	//prepare for GN = IC
	const int ic0 = (((blockIdx.y*blockDim.y) + threadIdx.y) << 1) + ic_index;

	//prepare fo GM = N * IH * IW
	int j = ((blockIdx.x*blockDim.x) + threadIdx.x) + j_index;
	const int oph = FH - ph - 1, opw = FW - pw - 1;
	const int IH_IW = IH * IW; get_n_ih_iw(j, n, ih, iw); j *= IC;
	const int OHp = OH * sh - sh + 1, OWp = OW * sw - sw + 1;
	const int tih = ih - oph, tiw = iw - opw;
	deltaY += n * OH * OW * IC;

	//find (fhs, fws) to compress deltaXpe -> deltaX
	int fhs = 0, fws = 0; FIND_FHS_FWS(fhs, fws, tih, tiw, max_kernel_8_1_end);
	const int FHr = (FH - fhs + sh - 1) / sh;
	const int FWr = (FW - fws + sw - 1) / sw;
	const int GKr = FHr * FWr;
	const int toh = (tih + fhs) / sh, tow = (tiw + fws) / sw;

	//compute area------------------------------------------
	float2 v = make_float2(0, 0);
	float alpha = 1.0f / (FH * FW);
	for (int k = 0; k < GKr; k++)
	{
		int fhr = k / FWr, fwr = k - fhr * FWr;
		int oh = toh + fhr, ow = tow + fwr;
		if (oh < OH && ow < OW) {
			float2 dy = *(float2*)(&get3d(deltaY, oh, ow, ic0, OW, IC));
			simdMM2(v, alpha, dy);
		}
	}

	*(float2*)(&deltaX[j + ic0]) = v;
}

#endif


//(Y: BLOCK_SIZE*2)
#ifndef UNPOOL2D_AVG_KERNEL_1_1
#define UNPOOL2D_AVG_KERNEL_1_1

//<3, 2>: Size = 1.000000, Time = 14.712000 msec, Peformance = 72.984077 GFlop/s [64,  4]
//<2, 3>: Size = 1.000000, Time = 14.868000 msec, Peformance = 72.218307 GFlop/s [32,  8]
//<1, 4>: Size = 1.000000, Time = 15.160000 msec, Peformance = 70.827293 GFlop/s [16, 16]
__global__ void avg_kernel_1_1(
	const float* __restrict__ deltaY, int OH, int OW,
	int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC,
	int sh, int sw, int ph, int pw,
	int ic_index, int j_index)
{
	//prepare for GN = IC
	const int ic0 = ((blockIdx.y*blockDim.y) + threadIdx.y) + ic_index;

	//prepare fo GM = N * IH * IW
	int j = ((blockIdx.x*blockDim.x) + threadIdx.x) + j_index;
	const int oph = FH - ph - 1, opw = FW - pw - 1;
	const int IH_IW = IH * IW; get_n_ih_iw(j, n, ih, iw); j *= IC;
	const int OHp = OH * sh - sh + 1, OWp = OW * sw - sw + 1;
	const int tih = ih - oph, tiw = iw - opw;
	deltaY += n * OH * OW * IC;

	//find (fhs, fws) to compress deltaXpe -> deltaX
	int fhs = 0, fws = 0; FIND_FHS_FWS(fhs, fws, tih, tiw, max_kernel_8_1_end);
	const int FHr = (FH - fhs + sh - 1) / sh;
	const int FWr = (FW - fws + sw - 1) / sw;
	const int GKr = FHr * FWr;
	const int toh = (tih + fhs) / sh, tow = (tiw + fws) / sw;

	//compute area------------------------------------------
	float v = 0.0f;
	float alpha = 1.0f / (FH * FW);
	for (int k = 0; k < GKr; k++)
	{
		int fhr = k / FWr, fwr = k - fhr * FWr;
		int oh = toh + fhr, ow = tow + fwr;
		if (oh < OH && ow < OW) {
			float dy = get3d(deltaY, oh, ow, ic0, OW, IC);
			v += alpha * dy;
		}
	}

	deltaX[j + ic0] = v;
}

#endif

#endif