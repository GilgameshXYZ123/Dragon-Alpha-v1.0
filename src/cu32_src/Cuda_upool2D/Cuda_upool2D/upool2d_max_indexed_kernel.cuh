#pragma once

#ifndef UNPOOL2D_MAX_INDEXED_KERNEL_H
#define UNPOOL2D_MAX_INDEXED_KERNEL_H

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
#ifndef UNPOOL2D_MAX_INDEXED_KERNEL_CALL
#define UNPOOL2D_MAX_INDEXED_KERNEL_CALL

#define kmaxIdx81(stream, LBY, LBX, ic_index, j_index, deltaY, Index, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM)\
	max_indexed_kernel_8_1\
		<<< dim3(GM>>LBX, GN>>LBY>>3), dim3(1<<LBX, 1<<LBY), 0, stream >>>\
			(deltaY,Index,OH,OW, FH,FW, deltaX,IH,IW, IC, sh,sw,ph,pw, ic_index, j_index)

#define kmaxIdx41(stream, LBY, LBX, ic_index, j_index, deltaY, Index, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM)\
	max_indexed_kernel_4_1\
		<<< dim3(GM>>LBX, GN>>LBY>>2), dim3(1<<LBX, 1<<LBY), 0, stream >>>\
			(deltaY,Index,OH,OW, FH,FW, deltaX,IH,IW, IC, sh,sw,ph,pw, ic_index, j_index)

#define kmaxIdx21(stream, LBY, LBX, ic_index, j_index, deltaY, Index, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM)\
	max_indexed_kernel_2_1\
		<<< dim3(GM>>LBX, GN>>LBY>>1), dim3(1<<LBX, 1<<LBY), 0, stream >>>\
			(deltaY,Index,OH,OW, FH,FW, deltaX,IH,IW, IC, sh,sw,ph,pw, ic_index, j_index)

#define kmaxIdx11(stream, LBY, LBX, ic_index, j_index, deltaY, Index, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM)\
	max_indexed_kernel_1_1\
		<<< dim3(GM>>LBX, GN>>LBY), dim3(1<<LBX, 1<<LBY), 0, stream >>>\
			(deltaY,Index,OH,OW, FH,FW, deltaX,IH,IW, IC, sh,sw,ph,pw, ic_index, j_index)

#endif


//(Y: BLOCK_SIZE*8)
#ifndef UNPOOL2D_MAX_INDEXED_KERNEL_8_1
#define UNPOOL2D_MAX_INDEXED_KERNEL_8_1

__global__ void max_indexed_kernel_8_1(
	const float* __restrict__ deltaY,
	const int * __restrict__ Index, int OH, int OW,
	int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC,
	int sh, int sw, int ph, int pw,
	int ic_index, int j_index)
{
	//prepare for GN = IC
	const int ic0 = (((blockIdx.y*blockDim.y) + threadIdx.y) << 3) + ic_index;
	const int ic4 = ic0 + 4;

	//prepare fo GM = N * IH * IW
	int j = ((blockIdx.x*blockDim.x) + threadIdx.x) + j_index;
	const int oph = FH - ph - 1, opw = FW - pw - 1;
	const int IH_IW = IH * IW; get_n_ih_iw(j, n, ih, iw); j *= IC;
	const int xoffset = j + ic0;

	const int OHp = OH * sh - sh + 1, OWp = OW * sw - sw + 1;
	const int tih = ih - oph, tiw = iw - opw;
	int offsetY = n * OH * OW * IC; 
	Index += offsetY; deltaY += offsetY;

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
			int yoffset = (oh*OW + ow)*IC;
			int4 idx0 = *(int4*)(Index + yoffset + ic0);
			int4 idx1 = *(int4*)(Index + yoffset + ic4);
			float4 dy0 = *(float4*)(deltaY + yoffset + ic0);
			float4 dy1 = *(float4*)(deltaY + yoffset + ic4);
				
			v0.x += (idx0.x == xoffset    ) * dy0.x;
			v0.y += (idx0.y == xoffset + 1) * dy0.y;
			v0.z += (idx0.z == xoffset + 2) * dy0.z;
			v0.w += (idx0.w == xoffset + 3) * dy0.w;
			v1.x += (idx1.x == xoffset + 4) * dy1.x;
			v1.y += (idx1.y == xoffset + 5) * dy1.y;
			v1.z += (idx1.z == xoffset + 6) * dy1.z;
			v1.w += (idx1.w == xoffset + 7) * dy1.w;
		}
	}

	*(float4*)(&deltaX[j + ic0]) = v0;
	*(float4*)(&deltaX[j + ic4]) = v1;
}

#endif


//(Y: BLOCK_SIZE*4)
#ifndef UNPOOL2D_MAX_INDEXED_KERNEL_4_1
#define UNPOOL2D_MAX_INDEXED_KERNEL_4_1

__global__ void max_indexed_kernel_4_1(
	const float* __restrict__ deltaY,
	const int* __restrict__ Index, int OH, int OW,
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
	const int xoffset = j + ic0;

	const int OHp = OH * sh - sh + 1, OWp = OW * sw - sw + 1;
	const int tih = ih - oph, tiw = iw - opw;
	int offsetY = n * OH * OW * IC;
	Index += offsetY; deltaY += offsetY;

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
			int yoffset = (oh*OW + ow)*IC;
			int4 idx = *(int4*)(Index + yoffset + ic0);
			float4 dy = *(float4*)(deltaY + yoffset + ic0);

			v.x += (idx.x == xoffset    ) * dy.x;
			v.y += (idx.y == xoffset + 1) * dy.y;
			v.z += (idx.z == xoffset + 2) * dy.z;
			v.w += (idx.w == xoffset + 3) * dy.w;
		}
	}

	*(float4*)(&deltaX[j + ic0]) = v;
}

#endif


//(Y: BLOCK_SIZE*2)
#ifndef UNPOOL2D_MAX_INDEXED_KERNEL_2_1
#define UNPOOL2D_MAX_INDEXED_KERNEL_2_1

__global__ void max_indexed_kernel_2_1(
	const float* __restrict__ deltaY,
	const int* __restrict__ Index, int OH, int OW,
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
	const int xoffset = j + ic0;

	const int OHp = OH * sh - sh + 1, OWp = OW * sw - sw + 1;
	const int tih = ih - oph, tiw = iw - opw;
	int offsetY = n * OH * OW * IC; 
	Index += offsetY; deltaY += offsetY;

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
		if (oh < OH && ow < OW) 
		{
			int yoffset = (oh*OW + ow)*IC;
			int2 idx = *(int2*)(Index + yoffset + ic0);
			float2 dy = *(float2*)(deltaY + yoffset + ic0);

			v.x += (idx.x == xoffset    ) * dy.x;
			v.y += (idx.y == xoffset + 1) * dy.y;
		}
	}

	*(float2*)(&deltaX[j + ic0]) = v;
}

#endif


//(Y: BLOCK_SIZE*1)
#ifndef UNPOOL2D_MAX_INDEXED_KERNEL_1_1
#define UNPOOL2D_MAX_INDEXED_KERNEL_1_1

__global__ void max_indexed_kernel_1_1(
	const float* __restrict__ deltaY,
	const int* __restrict__ Index, int OH, int OW,
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
	const int xoffset = j + ic0;
	
	const int OHp = OH * sh - sh + 1, OWp = OW * sw - sw + 1;
	const int tih = ih - oph, tiw = iw - opw;
	int offsetY = n * OH * OW * IC; 
	Index += offsetY; deltaY += offsetY;

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
		if (oh < OH && ow < OW)
		{
			int yoffset = (oh*OW + ow)*IC;
			int idx = Index[yoffset + ic0];
			int dy = deltaY[yoffset + ic0];

			v += (idx == xoffset) * dy;
		}
	}

	deltaX[j + ic0] = v;
}

#endif

#endif