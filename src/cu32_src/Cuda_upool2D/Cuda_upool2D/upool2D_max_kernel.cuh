#pragma once

#ifndef UNPOOL2D_MAX_KERNEL_H
#define UNPOOL2D_MAX_KERNEL_H

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
#ifndef UNPOOL2D_MAX_KERNEL_CALL
#define UNPOOL2D_MAX_KERNEL_CALL

#define kmax81(stream, LBY, LBX, ic_index, j_index, deltaY, Y, OH, OW, FH, FW, deltaX, X, IH, IW, IC, sh, sw, ph, pw, GN, GM)\
	max_kernel_8_1\
		<<< dim3(GM>>LBX, GN>>LBY>>3), dim3(1<<LBX, 1<<LBY), 0, stream >>>\
			(deltaY,Y,OH,OW, FH,FW, deltaX,X,IH,IW, IC, sh,sw,ph,pw, ic_index, j_index)

#define kmax41(stream, LBY, LBX, ic_index, j_index, deltaY, Y, OH, OW, FH, FW, deltaX, X, IH, IW, IC, sh, sw, ph, pw, GN, GM)\
	max_kernel_4_1\
		<<< dim3(GM>>LBX, GN>>LBY>>2), dim3(1<<LBX, 1<<LBY), 0, stream >>>\
			(deltaY,Y,OH,OW, FH,FW, deltaX,X,IH,IW, IC, sh,sw,ph,pw, ic_index, j_index)

#define kmax21(stream, LBY, LBX, ic_index, j_index, deltaY, Y, OH, OW, FH, FW, deltaX, X, IH, IW, IC, sh, sw, ph, pw, GN, GM)\
	max_kernel_2_1\
		<<< dim3(GM>>LBX, GN>>LBY>>1), dim3(1<<LBX, 1<<LBY), 0, stream >>>\
			(deltaY,Y,OH,OW, FH,FW, deltaX,X,IH,IW, IC, sh,sw,ph,pw, ic_index, j_index)

#define kmax11(stream, LBY, LBX, ic_index, j_index, deltaY, Y, OH, OW, FH, FW, deltaX, X, IH, IW, IC, sh, sw, ph, pw, GN, GM)\
	max_kernel_1_1\
		<<< dim3(GM>>LBX, GN>>LBY), dim3(1<<LBX, 1<<LBY), 0, stream >>>\
			(deltaY,Y,OH,OW, FH,FW, deltaX,X,IH,IW, IC, sh,sw,ph,pw, ic_index, j_index)

#endif


//(Y: BLOCK_SIZE*8)
#ifndef UNPOOL2D_MAX_KERNEL_8_1
#define UNPOOL2D_MAX_KERNEL_8_1

//<4, 1>: Size = 1.000000, Time = 1.956000 msec, Peformance = 548.947754 GFlop/s [128,  2]
//<3, 3>: Size = 1.000000, Time = 2.088000 msec, Peformance = 514.244141 GFlop/s [ 64,  8]
//<2, 4>: Size = 1.000000, Time = 2.116000 msec, Peformance = 507.439392 GFlop/s [ 32, 16]
//<1, 5>: Size = 1.000000, Time = 2.082000 msec, Peformance = 515.726135 GFlop/s [ 16, 32]

//<3, 2>: Size = 1.000000, Time = 2.146000 msec, Peformance = 500.345642 GFlop/s [64,  4]
//<2, 3>: Size = 1.000000, Time = 2.128000 msec, Peformance = 504.577881 GFlop/s [32,  8]
//<1, 4>: Size = 1.000000, Time = 2.216000 msec, Peformance = 484.540497 GFlop/s [16, 16]
//<0, 5>: Size = 1.000000, Time = 2.246000 msec, Peformance = 478.068451 GFlop/s [ 8, 32]
__global__ void max_kernel_8_1(
	const float* __restrict__ deltaY,
	const float* __restrict__ Y, int OH, int OW,
	int FH, int FW,	
	float* __restrict__ deltaX,
	const float* __restrict__ X, int IH, int IW,
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
	float4 x0 = *(float4*)(&X[j + ic0]), x1 = *(float4*)(&X[j + ic4]);
	const int OHp = OH * sh - sh + 1, OWp = OW * sw - sw + 1;
	const int tih = ih - oph, tiw = iw - opw;
	int offsetY = n * OH * OW * IC; Y += offsetY; deltaY += offsetY;

	//find (fhs, fws) to compress deltaXpe -> deltaX
	int fhs = 0, fws = 0; FIND_FHS_FWS(fhs, fws, tih, tiw, max_kernel_8_1_end);
	const int FHr = (FH - fhs + sh - 1) / sh;
	const int FWr = (FW - fws + sw - 1) / sw;
	const int GKr = FHr * FWr;
	const int toh = (tih + fhs) / sh, tow = (tiw + fws) / sw;

	//compute area------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0);
	float4 v1 = make_float4(0, 0, 0, 0);
	for (int k = 0; k < GKr; k++)
	{
		int fhr = k / FWr, fwr = k - fhr * FWr;
		int oh = toh + fhr, ow = tow + fwr;
		if (oh < OH && ow < OW)
		{
			float4 y0 = *(float4*)(&get3d(Y, oh, ow, ic0, OW, IC));
			float4 y1 = *(float4*)(&get3d(Y, oh, ow, ic4, OW, IC));
			float4 dy0 = *(float4*)(&get3d(deltaY, oh, ow, ic0, OW, IC));
			float4 dy1 = *(float4*)(&get3d(deltaY, oh, ow, ic4, OW, IC));

			v0.x += (x0.x >= y0.x) * dy0.x;
			v0.y += (x0.y >= y0.y) * dy0.y;
			v0.z += (x0.z >= y0.z) * dy0.z;
			v0.w += (x0.w >= y0.w) * dy0.w;

			v1.x += (x1.x >= y1.x) * dy1.x;
			v1.y += (x1.y >= y1.y) * dy1.y; 
			v1.z += (x1.z >= y1.z) * dy1.z;
			v1.w += (x1.w >= y1.w) * dy1.w;
		}
	}

	*(float4*)(&deltaX[j + ic0]) = v0;
	*(float4*)(&deltaX[j + ic4]) = v1;
}

#endif


//(Y: BLOCK_SIZE*4)
#ifndef UNPOOL2D_MAX_KERNEL_4_1
#define UNPOOL2D_MAX_KERNEL_4_1

//<3, 2>: Size = 1.000000, Time = 3.436000 msec, Peformance = 312.497589 GFlop/s [32,  4]
//<2, 3>: Size = 1.000000, Time = 3.488000 msec, Peformance = 307.838806 GFlop/s [16,  8]
//<1, 4>: Size = 1.000000, Time = 3.736000 msec, Peformance = 287.404114 GFlop/s [ 8, 16]
//<0, 5>: Size = 1.000000, Time = 4.550000 msec, Peformance = 235.987198 GFlop/s [ 4, 32]
__global__ void max_kernel_4_1(
	const float* __restrict__ deltaY,
	const float* __restrict__ Y, int OH, int OW,
	int FH, int FW,
	float* __restrict__ deltaX,
	const float* __restrict__ X, int IH, int IW,
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
	float4 x = *(float4*)(&X[j + ic0]);
	const int OHp = OH * sh - sh + 1, OWp = OW * sw - sw + 1;
	const int tih = ih - oph, tiw = iw - opw;
	int offsetY = n * OH * OW * IC; Y += offsetY; deltaY += offsetY;

	//find (fhs, fws) to compress deltaXpe -> deltaX
	int fhs = 0, fws = 0; FIND_FHS_FWS(fhs, fws, tih, tiw, max_kernel_8_1_end);
	const int FHr = (FH - fhs + sh - 1) / sh;
	const int FWr = (FW - fws + sw - 1) / sw;
	const int GKr = FHr * FWr;
	const int toh = (tih + fhs) / sh, tow = (tiw + fws) / sw;

	//compute area------------------------------------------
	float4 v = make_float4(0, 0, 0, 0);
	for (int k = 0; k < GKr; k++)
	{
		int fhr = k / FWr, fwr = k - fhr * FWr;
		int oh = toh + fhr, ow = tow + fwr;
		if (oh < OH && ow < OW)
		{
			float4 y = *(float4*)(&get3d(Y, oh, ow, ic0, OW, IC));
			float4 dy = *(float4*)(&get3d(deltaY, oh, ow, ic0, OW, IC));

			v.x += (x.x >= y.x) * dy.x;
			v.y += (x.y >= y.y) * dy.y;
			v.z += (x.z >= y.z) * dy.z;
			v.w += (x.w >= y.w) * dy.w;
		}
	}

	*(float4*)(&deltaX[j + ic0]) = v;
}

#endif


//(Y: BLOCK_SIZE*2)
#ifndef UNPOOL2D_MAX_KERNEL_2_1
#define UNPOOL2D_MAX_KERNEL_2_1

//<3, 2>: Size = 1.000000, Time = 6.424000 msec, Peformance = 167.145355 GFlop/s [16,  4]
//<2, 3>: Size = 1.000000, Time = 6.480000 msec, Peformance = 165.700882 GFlop/s [ 8,  8]
//<1, 4>: Size = 1.000000, Time = 3.736000 msec, Peformance = 287.404114 GFlop/s [ 4, 16]
__global__ void max_kernel_2_1(
	const float* __restrict__ deltaY,
	const float* __restrict__ Y, int OH, int OW,
	int FH, int FW,
	float* __restrict__ deltaX,
	const float* __restrict__ X, int IH, int IW,
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
	float2 x = *(float2*)(&X[j + ic0]);
	const int OHp = OH * sh - sh + 1, OWp = OW * sw - sw + 1;
	const int tih = ih - oph, tiw = iw - opw;
	int offsetY = n * OH * OW * IC; Y += offsetY; deltaY += offsetY;

	//find (fhs, fws) to compress deltaXpe -> deltaX
	int fhs = 0, fws = 0; FIND_FHS_FWS(fhs, fws, tih, tiw, max_kernel_8_1_end);
	const int FHr = (FH - fhs + sh - 1) / sh;
	const int FWr = (FW - fws + sw - 1) / sw;
	const int GKr = FHr * FWr;
	const int toh = (tih + fhs) / sh, tow = (tiw + fws) / sw;

	//compute area------------------------------------------
	float2 v = make_float2(0, 0);
	for (int k = 0; k < GKr; k++)
	{
		int fhr = k / FWr, fwr = k - fhr * FWr;
		int oh = toh + fhr, ow = tow + fwr;
		if (oh < OH && ow < OW) {
			float2 y = *(float2*)(&get3d(Y, oh, ow, ic0, OW, IC));
			float2 dy = *(float2*)(&get3d(deltaY, oh, ow, ic0, OW, IC));
			v.x += (x.x >= y.x) * dy.x;
			v.y += (x.y >= y.y) * dy.y;
		}
	}

	*(float2*)(&deltaX[j + ic0]) = v;
}

#endif


//(Y: BLOCK_SIZE*1)
#ifndef UNPOOL2D_MAX_KERNEL_1_1
#define UNPOOL2D_MAX_KERNEL_1_1

//<3, 2>: Size = 0.500000, Time = 6.174000 msec, Peformance = 86.956741 GFlop/s [8,  4]
//<2, 3>: Size = 0.500000, Time = 6.270000 msec, Peformance = 85.625336 GFlop/s [4,  8]
__global__ void max_kernel_1_1(
	const float* __restrict__ deltaY,
	const float* __restrict__ Y, int OH, int OW,
	int FH, int FW,
	float* __restrict__ deltaX,
	const float* __restrict__ X, int IH, int IW,
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
	float x = X[j + ic0];
	const int OHp = OH * sh - sh + 1, OWp = OW * sw - sw + 1;
	const int tih = ih - oph, tiw = iw - opw;
	int offsetY = n * OH * OW * IC; Y += offsetY; deltaY += offsetY;

	//find (fhs, fws) to compress deltaXpe -> deltaX
	int fhs = 0, fws = 0; FIND_FHS_FWS(fhs, fws, tih, tiw, max_kernel_8_1_end);
	const int FHr = (FH - fhs + sh - 1) / sh;
	const int FWr = (FW - fws + sw - 1) / sw;
	const int GKr = FHr * FWr;
	const int toh = (tih + fhs) / sh, tow = (tiw + fws) / sw;

	//compute area------------------------------------------
	float v = 0.0f;
	for (int k = 0; k < GKr; k++)
	{
		int fhr = k / FWr, fwr = k - fhr * FWr;
		int oh = toh + fhr, ow = tow + fwr;
		if (oh < OH && ow < OW) {
			float y = get3d(Y, oh, ow, ic0, OW, IC);
			float dy = get3d(deltaY, oh, ow, ic0, OW, IC);
			v += (x >= y) * dy;
		}
	}

	deltaX[j + ic0] = v;
}

#endif

#endif