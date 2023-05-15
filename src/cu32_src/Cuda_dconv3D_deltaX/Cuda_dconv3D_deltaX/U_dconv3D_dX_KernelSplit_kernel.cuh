#pragma once

#ifndef DCONV3D_DX_KERNEL_SPLIT_KENREL_H
#define DCONV3D_DX_KERNEL_SPLIT_KERNEL_H

#define kernelSplit_kv1(stream, LB, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, CFH, CFW, GN, GM) \
	kernelSplit_kernel_v1<LB>\
		<<< dim3(GM>>LB, GN>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, sh,sw,ph,pw, CFH, CFW)

template<int LB>
__global__ void kernelSplit_kernel_v1(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW, 
		  float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int CFH, int CFW)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	int ic = (blockIdx.y << LB) + ty;
	int j = (blockIdx.x << LB) + tx;//[n, ohs, ows]
	int oph = CFH - 1, opw = CFW - 1;
	int OHS = OH - CFH + (oph << 1) + 1;
	int OWS = OW - CFW + (opw << 1) + 1;
	int OHS_OWS = OHS * OWS;
	int n = j / OHS_OWS; j %= OHS_OWS;
	int ohs = (j / OWS) - oph, ows = (j % OWS) - opw;

	for (int y = 0; y < sh; y++)
	for (int x = 0; x < sw; x++)
	{
		int ih = y + (ohs + oph)*sh - ph;
		int iw = x + (ows + opw)*sw - pw;
		if (ih < 0 || iw < 0 || ih >= IH || iw >= IW) continue;

		float dx = 0;
		for (int fhr = 0; fhr < CFH; fhr++)
		for (int fwr = 0; fwr < CFW; fwr++)
		for (int oc = 0; oc < OC; oc++)
		{
			int oh = ohs + fhr;
			int ow = ows + fwr;
			float dy = (oh < 0 || ow < 0 || oh >= OH || ow >= OW) ? 0 : get4d(deltaY, n, oh, ow, oc, OH, OW, OC);

			int fh = y + (CFH - 1 - fhr)*sh;
			int fw = x + (CFW - 1 - fwr)*sw;
			float w = (fh < 0 || fw < 0 || fh >= FH || fw >= FW) ? 0 : get4d(W, oc, fh, fw, ic, FH, FW, IC);

			dx += w * dy;
		}
		get4d(deltaX, n, ih, iw, ic, IH, IW, IC) = dx;
	}
}


#define kernelSplit_kv2(stream, LB, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, CFH, CFW, GN, GM) \
	kernelSplit_kernel_v2<LB>\
		<<< dim3((GM + (1<<LB) - 1)>>LB, GN>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, sh,sw,ph,pw, CFH, CFW)

template<int LB>
__global__ void kernelSplit_kernel_v2(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW, 
		  float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int CFH, int CFW)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	int ic = (blockIdx.y << LB) + ty;
	int j = (blockIdx.x << LB) + tx;//[n, ohs, ows]
	int oph = CFH - 1, opw = CFW - 1;
	int OHS = OH - CFH + (oph << 1) + 1;
	int OWS = OW - CFW + (opw << 1) + 1;
	int OHS_OWS = OHS * OWS;
	int n = j / OHS_OWS; j -= n * OHS_OWS;
	int ohs = j / OWS, ows = j - ohs * OWS;
	ohs -= oph; ows -= opw;

	for (int y = 0; y < sh; y++)
		for (int x = 0; x < sw; x++)
		{
			int ih = y + (ohs + oph)*sh - ph;
			int iw = x + (ows + opw)*sw - pw;
			if (ih < 0 || iw < 0 || ih >= IH || iw >= IW) continue;

			float dx = 0;
			for (int fhr = 0; fhr < CFH; fhr++)
				for (int fwr = 0; fwr < CFW; fwr++)
					for (int oc = 0; oc < OC; oc++)
					{
						int oh = ohs + fhr;
						int ow = ows + fwr;

						float dy = (oh < 0 || ow < 0 || oh >= OH || ow >= OW) ? 0 : get4d(deltaY, n, oh, ow, oc, OH, OW, OC);

						int fh = y + (CFH - 1 - fhr)*sh;
						int fw = x + (CFW - 1 - fwr)*sw;
						float w = (fh < 0 || fw < 0 || fh >= FH || fw >= FW) ? 0 : get4d(W, oc, fh, fw, ic, FH, FW, IC);

						dx += w * dy;
					}
			get4d(deltaX, n, ih, iw, ic, IH, IW, IC) = dx;
		}
}


#define kernelSplit_kv3(stream, LB, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, CFH, CFW, GN, GM) \
	kernelSplit_kernel_v3<LB>\
		<<< dim3((GM + (1<<LB) - 1)>>LB, GN>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, sh,sw,ph,pw, CFH, CFW)

template<int LB>
__global__ void kernelSplit_kernel_v3(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int CFH, int CFW)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	int ic = (blockIdx.y << LB) + ty;

	int j = (blockIdx.x << LB) + tx;//[n, ohs, ows]
	int oph = CFH - 1, opw = CFW - 1;
	int OHS = OH - CFH + (oph << 1) + 1;
	int OWS = OW - CFW + (opw << 1) + 1;
	int OHS_OWS = OHS * OWS;
	int n = j / OHS_OWS; j -= n * OHS_OWS;
	int ohs = j / OWS, ows = j - ohs * OWS;
	ohs -= oph; ows -= opw;
	int tih = (ohs + oph)*sh - ph, tiw = (ows + opw)*sw - pw;//need the same -> same tx

	for (int y = 0; y < sh; y++) 
	{
		int ih = y + tih;
		for (int x = 0; x < sw; x++)
		{
			int iw = x + tiw;
			bool write_dx = (ih >= 0) && (iw >= 0) && (ih < IH) && (iw < IW);

			float dx = 0;
			for (int fhr = 0; fhr < CFH; fhr++) 
			{
				int oh = ohs + fhr;
				int fh = y + (CFH - 1 - fhr)*sh;
				for (int fwr = 0; fwr < CFW; fwr++) 
				{
					int ow = ows + fwr;
					int fw = x + (CFW - 1 - fwr)*sw;
					bool load_dy = (oh >= 0) && (ow >= 0) && (oh < OH) && (ow < OW) && write_dx;
					bool load_w = (fh >= 0) && (fw >= 0) && (fh < FH) && (fw < FW) && write_dx;
					for (int oc = 0; oc < OC; oc++)
					{
						float dy = load_dy ? get4d(deltaY, n, oh, ow, oc, OH, OW, OC) : 0;
						float w = load_w ? get4d(W, oc, fh, fw, ic, FH, FW, IC) : 0;
						dx += w * dy;
					}
				}
			}
			if(write_dx) get4d(deltaX, n, ih, iw, ic, IH, IW, IC) = dx;
		}
	}
}


#define kernelSplit_kv4(stream, LB, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, CFH, CFW, GN, GM) \
	kernelSplit_kernel_v4<LB>\
		<<< dim3((GM + (1<<LB) - 1)>>LB, GN>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, sh,sw,ph,pw, CFH, CFW, GM)

template<int LB>
__global__ void kernelSplit_kernel_v4(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int CFH, int CFW, int GM)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	int ic = (blockIdx.y << LB) + ty;
	int j = (blockIdx.x << LB) + tx;//[n, ohs, ows]

	int oph = CFH - 1, opw = CFW - 1;
	int OHS = OH - CFH + (oph << 1) + 1;
	int OWS = OW - CFW + (opw << 1) + 1;
	int OHS_OWS = OHS * OWS;
	int n = j / OHS_OWS; j -= n * OHS_OWS;
	int ohs = j / OWS, ows = j - ohs * OWS;
	ohs -= oph; ows -= opw;
	int tih = (ohs + oph)*sh - ph, tiw = (ows + opw)*sw - pw;//need the same -> same tx

	for (int y = 0; y < sh; y++)
	{
		int ih = y + tih;
		for (int x = 0; x < sw; x++)
		{
			int iw = x + tiw;
			bool write_dx = (ih >= 0) && (iw >= 0) && (ih < IH) && (iw < IW);

			float dx = 0;
			for (int fhr = 0; fhr < CFH; fhr++)
			{
				int oh = ohs + fhr;
				int fh = y + (CFH - 1 - fhr)*sh;
				for (int fwr = 0; fwr < CFW; fwr++)
				{
					int ow = ows + fwr;
					int fw = x + (CFW - 1 - fwr)*sw;
					bool load_dy = (oh >= 0) && (ow >= 0) && (oh < OH) && (ow < OW) && write_dx;
					bool load_w = (fh >= 0) && (fw >= 0) && (fh < FH) && (fw < FW) && write_dx;
					for (int oc = 0; oc < OC; oc++)
					{
						float dy = *(float*)ADDRESS_WITHIN(load_dy, get4d(deltaY, n, oh, ow, oc, OH, OW, OC));
						float w = *(float*)ADDRESS_WITHIN(load_w, get4d(W, oc, fh, fw, ic, FH, FW, IC));
						dx += w * dy;
					}
				}
			}
			if (write_dx) get4d(deltaX, n, ih, iw, ic, IH, IW, IC) = dx;
		}
	}
}


#define kernelSplit_kv5(stream, LB, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, CFH, CFW, GN, GM) \
	kernelSplit_kernel_v5<LB, (1<<LB)>\
		<<< dim3((GM + (1<<LB) - 1)>>LB, GN>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, sh,sw,ph,pw, CFH, CFW)

//Size = 0.234619, Time = 1.84 msec, Performace = 273.826 GFlop/s
template<int LB, int STEP>
__global__ void kernelSplit_kernel_v5(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int CFH, int CFW)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	__shared__ float  Ws[1 << LB][(1 << LB) + 1];
	__shared__ float dYs[1 << LB][(1 << LB) + 1];

	int ic = (blockIdx.y << LB) + ty;

	int j = (blockIdx.x << LB) + tx;//[n, ohs, ows]

	int oph = CFH - 1, opw = CFW - 1;
	int OHS = OH - CFH + (oph << 1) + 1;
	int OWS = OW - CFW + (opw << 1) + 1;
	int OHS_OWS = OHS * OWS;
	int n = j / OHS_OWS; j -= n * OHS_OWS;
	int ohs = j / OWS, ows = j - ohs * OWS;
	ohs -= oph; ows -= opw;
	int tih = (ohs + oph)*sh - ph;
	int tiw = (ows + opw)*sw - pw;//need the same -> same tx

	const int OOC = OC >> LB;
	for (int y = 0; y < sh; y++)
	{
		int ih = y + tih;
		for (int x = 0; x < sw; x++)
		{
			int iw = x + tiw;
			float dx = 0;
			for (int fhr = 0; fhr < CFH; fhr++)
			{
				int oh = ohs + fhr;
				int fh = y + (CFH - 1 - fhr)*sh;
				for (int fwr = 0; fwr < CFW; fwr++)
				{
					int ow = ows + fwr;
					int fw = x + (CFW - 1 - fwr)*sw;
					bool load_dy = (oh >= 0) && (ow >= 0) && (oh < OH) && (ow < OW);
					bool load_w = (fh >= 0) && (fw >= 0) && (fh < FH) && (fw < FW);

					for (int ooc = 0; ooc < OOC; ooc++)
					{
						int W_oc = (ooc << LB) + tx;
						Ws[tx][ty] = load_w ? get4d(W, W_oc, fh, fw, ic, FH, FW, IC) : 0; //with the same ty;

						int dY_oc = (ooc << LB) + ty;
						dYs[ty][tx] = load_dy ? get4d(deltaY, n, oh, ow, dY_oc, OH, OW, OC) : 0;//with the same tx;
						__syncthreads();

#pragma unroll
						for (int ik = 0; ik < STEP; ik++)
						{
							int oc = (ooc << LB) + ik;
							float w = Ws[ik][ty];
							float dy = dYs[ik][tx];
							dx += w * dy;
						}
						__syncthreads();
					}
				}
			}

			bool write_dx = (ih >= 0) && (iw >= 0) && (ih < IH) && (iw < IW);
			if (write_dx) get4d(deltaX, n, ih, iw, ic, IH, IW, IC) = dx;
		}
	}
}



#define kernelSplit_kv6(stream, LB, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, CFH, CFW, GN, GM) \
	kernelSplit_kernel_v6<LB, (1<<LB)>\
		<<< dim3((GM + (1<<LB) - 1)>>LB, GN>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, sh,sw,ph,pw, CFH, CFW)

//Size = 0.234619, Time = 1.71667 msec, Performace = 293.499 GFlop/s
template<int LB, int STEP>
__global__ void kernelSplit_kernel_v6(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int CFH, int CFW)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float  Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float dYs[2][1 << LB][(1 << LB) + 1];

	int ic = (blockIdx.y << LB) + ty;
	int j = (blockIdx.x << LB) + tx;//[n, ohs, ows]

	int oph = CFH - 1, opw = CFW - 1;
	int OHS = OH - CFH + (oph << 1) + 1;
	int OWS = OW - CFW + (opw << 1) + 1;
	int OHS_OWS = OHS * OWS;
	int n = j / OHS_OWS; j -= n * OHS_OWS;
	int ohs = j / OWS, ows = j - ohs * OWS;
	ohs -= oph; ows -= opw;
	int tih = (ohs + oph)*sh - ph;
	int tiw = (ows + opw)*sw - pw;//need the same -> same tx

	const int OOC = OC >> LB;
	for (int y = 0; y < sh; y++)
	{
		int ih = y + tih;
		for (int x = 0; x < sw; x++)
		{
			int iw = x + tiw;
			float dx = 0;
			for (int fhr = 0; fhr < CFH; fhr++)
			{
				int oh = ohs + fhr;
				int fh = y + (CFH - 1 - fhr)*sh;
				for (int fwr = 0; fwr < CFW; fwr++)
				{
					int ow = ows + fwr;
					int fw = x + (CFW - 1 - fwr)*sw;
					bool load_dy = (oh >= 0) && (ow >= 0) && (oh < OH) && (ow < OW);
					bool load_w = (fh >= 0) && (fw >= 0) && (fh < FH) && (fw < FW);

					int W_oc = tx;
					Ws[buf][tx][ty] = load_w ? get4d(W, W_oc, fh, fw, ic, FH, FW, IC) : 0; //with the same ty;

					int dY_oc = ty;
					dYs[buf][ty][tx] = load_dy ? get4d(deltaY, n, oh, ow, dY_oc, OH, OW, OC) : 0;//with the same tx;
					__syncthreads();

					for (int ooc = 1; ooc < OOC; ooc++) {
#pragma unroll
						for (int ik = 0; ik < STEP; ik++) {
							float w = Ws[buf][ik][ty];
							float dy = dYs[buf][ik][tx];
							dx += w * dy;
						}
						buf ^= 1;

						int W_oc = (ooc << LB) + tx;
						Ws[buf][tx][ty] = load_w ? get4d(W, W_oc, fh, fw, ic, FH, FW, IC) : 0; //with the same ty;

						int dY_oc = (ooc << LB) + ty;
						dYs[buf][ty][tx] = load_dy ? get4d(deltaY, n, oh, ow, dY_oc, OH, OW, OC) : 0;//with the same tx;
						__syncthreads();
					}
#pragma unroll
					for (int ik = 0; ik < STEP; ik++) {
						float w = Ws[buf][ik][ty];
						float dy = dYs[buf][ik][tx];
						dx += w * dy;
					}
					buf ^= 1;
				}
			}

			bool write_dx = (ih >= 0) && (iw >= 0) && (ih < IH) && (iw < IW);
			if (write_dx) get4d(deltaX, n, ih, iw, ic, IH, IW, IC) = dx;
		}
	}
}


#define kernelSplit_kv7(stream, LB, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, CFH, CFW, GN, GM) \
	kernelSplit_kernel_v7<LB, (1<<LB)>\
		<<< dim3((GM + (2<<LB) - 1)>>LB>>1, GN>>LB>>1), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, sh,sw,ph,pw, CFH, CFW)

//OC is power of 4
//Size = 0.938477, Time = 2.05333 msec, Performace = 981.508 GFlop/s
//LB = 4: OC % 16 == 0
//LB = 3: OC % 8 == 0
template<int LB, int STEP>
__global__ void kernelSplit_kernel_v7(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int CFH, int CFW)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2  Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float2 dYs[2][1 << LB][(1 << LB) + 1];

	int ic0 = ((blockIdx.y << LB) + ty) << 1;
	int j0 = ((blockIdx.x << LB) + tx) << 1, j1 = j0 + 1;//[n, ohs, ows]
	int oph = --CFH, opw = --CFW;
	const int OHS = OH + CFH;//OHS = OH - CFH + (oph << 1) + 1
	const int OWS = OW + CFW;//OWS = OW - CFW + (opw << 1) + 1;
	const int OHS_OWS = OHS * OWS;
	get_n_ohs_ows(j0, n0, ohs0, ows0); ohs0 -= oph; ows0 -= opw;
	get_n_ohs_ows(j1, n1, ohs1, ows1); ohs1 -= oph; ows1 -= opw;
	const int tih0 = (ohs0 + oph)*sh - ph, tiw0 = (ows0 + opw)*sw - pw;
	const int tih1 = (ohs1 + oph)*sh - ph, tiw1 = (ows1 + opw)*sw - pw;

	const int OOC = OC >> LB;
	const int GK = FH * FW * IC;
	for (int y = 0; y < sh; y++)
	{
		int ih0 = y + tih0;
		int ih1 = y + tih1;
		for (int x = 0; x < sw; x++)
		{
			int iw0 = x + tiw0;
			int iw1 = x + tiw1;

			float2 v0 = make_float2(0, 0);
			float2 v1 = make_float2(0, 0);
			for (int fhr = 0; fhr <= CFH; fhr++)
			{
				int oh0 = ohs0 + fhr;
				int oh1 = ohs1 + fhr;
				int fh = y + (CFH - fhr)*sh;

				for (int fwr = 0; fwr <= CFW; fwr++)
				{
					int fw = x + (CFW - fwr)*sw;
					bool load_w = (fh >= 0) && (fw >= 0) && (fh < FH) && (fw < FW);

					int W_oc = tx;
					const int W_offset = (fh*FW + fw)*IC + ic0;
					Ws[buf][tx][ty] = load_w ? *(float2*)(&W[W_oc*GK + W_offset]) : make_float2(0, 0);

					int ow0 = ows0 + fwr;
					int ow1 = ows1 + fwr;
					bool load_dy0 = (oh0 >= 0) && (ow0 >= 0) && (oh0 < OH) && (ow0 < OW);
					bool load_dy1 = (oh1 >= 0) && (ow1 >= 0) && (oh1 < OH) && (ow1 < OW);
					const int dY_offset0 = ((n0*OH + oh0)*OW + ow0)*OC;
					const int dY_offset1 = ((n1*OH + oh1)*OW + ow1)*OC;

					int dY_oc = ty;
					dYs[buf][ty][tx].x = load_dy0 ? deltaY[dY_offset0 + dY_oc] : 0;
					dYs[buf][ty][tx].y = load_dy1 ? deltaY[dY_offset1 + dY_oc] : 0;
					__syncthreads();

					for (int ooc = 1; ooc < OOC; ooc++) {
#pragma unroll
						for (int ik = 0; ik < STEP; ik++) {
							float2 w = Ws[buf][ik][ty];
							float2 dy = dYs[buf][ik][tx];
							simdMM2(v0, dy.x, w);
							simdMM2(v1, dy.y, w);
						}
						buf ^= 1;

						int W_oc = (ooc << LB) + tx;
						Ws[buf][tx][ty] = load_w ? *(float2*)(&W[W_oc*GK + W_offset]) : make_float2(0, 0);

						int dY_oc = (ooc << LB) + ty;
						dYs[buf][ty][tx].x = load_dy0 ? deltaY[dY_offset0 + dY_oc] : 0;
						dYs[buf][ty][tx].y = load_dy1 ? deltaY[dY_offset1 + dY_oc] : 0;
						__syncthreads();
					}
#pragma unroll
					for (int ik = 0; ik < STEP; ik++) {
						float2 w = Ws[buf][ik][ty];
						float2 dy = dYs[buf][ik][tx];
						simdMM2(v0, dy.x, w);
						simdMM2(v1, dy.y, w);
					}
					buf ^= 1;
				}
			}

			if ((ih0 >= 0) && (ih0 < IH) && (iw0 < IW) && (iw0 >= 0)) {
				*(float2*)(&get4d(deltaX, n0, ih0, iw0, ic0, IH, IW, IC)) = v0;
			}
			if ((ih1 >= 0) && (iw1 < IW) && (iw1 >= 0) && (ih1 < IH) ) {
				*(float2*)(&get4d(deltaX, n1, ih1, iw1, ic0, IH, IW, IC)) = v1;
			}
		}
	}
}


#define kernelSplit_kv8(stream, LB, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, CFH, CFW, GN, GM) \
	kernelSplit_kernel_v8<LB, (1<<LB>>1)>\
		<<< dim3((GM + (4<<LB) - 1)>>LB>>2, GN>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, N, IC,OC, sh,sw,ph,pw, CFH, CFW)

//Size = 0.938477, Time = 1.76667 msec, Performace = 1140.77 GFlop/s
//Size = 1.87695, Time = 3.43333 msec, Performace = 1174 GFlop/s
template<int LB, int STEP>
__global__ void kernelSplit_kernel_v8(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw,
	int CFH, int CFW)
{
	int by = blockIdx.y, bx = blockIdx.x;
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2  Ws[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2 dYs[2][1 << LB >> 1][(2 << LB) + 2];

	const int ic0 = (((by << LB) + ty) << 2);
	const int tic0 = ((tx & 1) << 1) + ic0;

	const int j0 = (((bx << LB) + tx) << 2);
	const int j1 = j0 + 1, j2 = j0 + 2, j3 = j0 + 3;
	const int oph = --CFH, opw = --CFW;
	const int OHS = OH + CFH;//OHS = OH - CFH + (oph << 1) + 1
	const int OWS = OW + CFW;//OWS = OW - CFW + (opw << 1) + 1;
	const int OHS_OWS = OHS * OWS;
	get_n_ohs_ows(j0, n0, ohs0, ows0);
	get_n_ohs_ows(j1, n1, ohs1, ows1);
	get_n_ohs_ows(j2, n2, ohs2, ows2);
	get_n_ohs_ows(j3, n3, ohs3, ows3);
	int tih0 = ohs0 * sh - ph; ohs0 -= oph;
	int tih1 = ohs1 * sh - ph; ohs1 -= oph;
	int tih2 = ohs2 * sh - ph; ohs2 -= oph;
	int tih3 = ohs3 * sh - ph; ohs3 -= oph;
	int tiw0 = ows0 * sw - pw; ows0 -= opw;
	int tiw1 = ows1 * sw - pw; ows1 -= opw;
	int tiw2 = ows2 * sw - pw; ows2 -= opw;
	int tiw3 = ows3 * sw - pw; ows3 -= opw;
	bool flagY = (ty & 1);
	int tn0 = (n2 - n0)*flagY + n0;
	int tn1 = (n3 - n1)*flagY + n1;
	const int tohs0 = (ohs2 - ohs0)*flagY + ohs0;
	const int tohs1 = (ohs3 - ohs1)*flagY + ohs1;
	const int tows0 = (ows2 - ows0)*flagY + ows0;
	const int tows1 = (ows3 - ows1)*flagY + ows1;

	//((n*IH + ih)*IW + iw)*IC + ic
	//((n*IH + (tih + y))*IW + (tiw + x))*IC + ic
	//n*IH*IW*IC + (tih + y)*IW*C + (tiw+x)*IC + ic
	//(n*IH*IW*IC + tih*IW*IC + tiw*IC) + y*IW*IC + x*IC + ic

	const int OOC = OC << 1 >> LB, GK = FH * FW * IC;
	const int Ws_x = (tx >> 1), Ws_y = (ty << 1) + (tx & 1);
	const int dYs_y = (ty >> 1), dYs_x = (tx << 1) + (ty & 1);
	for (int y = 0; y < sh; y++)
	{
		for (int x = 0; x < sw; x++)
		{
			float4 v0 = make_float4(0, 0, 0, 0);
			float4 v1 = make_float4(0, 0, 0, 0);
			float4 v2 = make_float4(0, 0, 0, 0);
			float4 v3 = make_float4(0, 0, 0, 0);
			for (int fhr = 0; fhr <= CFH; fhr++)
			{
				int fh = y + (CFH - fhr)*sh;
				int toh0 = tohs0 + fhr;
				int toh1 = tohs1 + fhr;
				for (int fwr = 0; fwr <= CFW; fwr++)
				{
					int fw = x + (CFW - fwr)*sw;
					int W_offset = (fh*FW + fw)*IC + tic0;
					bool load_w = (fh >= 0) && (fw >= 0) && (fh < FH) && (fw < FW);
					int W_oc = tx >> 1;
					Ws[buf][Ws_x][Ws_y] = load_w ? *(float2*)(&W[W_oc*GK + W_offset]) : make_float2(0, 0);

					int tow0 = tows0 + fwr;
					int tow1 = tows1 + fwr;
					int dY_offset0 = ((tn0*OH + toh0)*OW + tow0)*OC;
					int dY_offset1 = ((tn1*OW + toh1)*OW + tow1)*OC;
					bool load_dy0 = (toh0 >= 0) && (tow0 >= 0) && (toh0 < OH) && (tow0 < OW);
					bool load_dy1 = (toh1 >= 0) && (tow1 >= 0) && (toh1 < OH) && (tow1 < OW);
					int dY_oc = ty >> 1;
					dYs[buf][dYs_y][dYs_x].x = load_dy0 ? deltaY[dY_offset0 + dY_oc] : 0;
					dYs[buf][dYs_y][dYs_x].y = load_dy1 ? deltaY[dY_offset1 + dY_oc] : 0;
					__syncthreads();

					for (int ooc = 1; ooc < OOC; ooc++) {
#pragma unroll
						for (int ik = 0; ik < STEP; ik++)
						{
							float4 w = *(float4*)(&Ws[buf][ik][ty << 1]);
							float4 dy = *(float4*)(&dYs[buf][ik][tx << 1]);

							simdMM4(v0, dy.x, w);
							simdMM4(v1, dy.y, w);
							simdMM4(v2, dy.z, w);
							simdMM4(v3, dy.w, w);
						}
						buf ^= 1;

						W_oc = ((ooc << LB) + tx) >> 1;
						Ws[buf][Ws_x][Ws_y] = load_w ? *(float2*)(&W[W_oc*GK + W_offset]) : make_float2(0, 0);

						dY_oc = ((ooc << LB) + ty) >> 1;
						dYs[buf][dYs_y][dYs_x].x = load_dy0 ? deltaY[dY_offset0 + dY_oc] : 0;
						dYs[buf][dYs_y][dYs_x].y = load_dy1 ? deltaY[dY_offset1 + dY_oc] : 0;
						__syncthreads();
					}
#pragma unroll
					for (int ik = 0; ik < STEP; ik++) {
						float4 w = *(float4*)(&Ws[buf][ik][ty << 1]);
						float4 dy = *(float4*)(&dYs[buf][ik][tx << 1]);

						simdMM4(v0, dy.x, w);
						simdMM4(v1, dy.y, w);
						simdMM4(v2, dy.z, w);
						simdMM4(v3, dy.w, w);
					}
					buf ^= 1;
				}
			}
			//((n*IH + ih)*IW + iw)*IC + ic
			//((n*IH + (tih + y))*IW + (tiw + x))*IC + ic
			//n*IH*IW*IC + (tih + y)*IW*C + (tiw+x)*IC + ic
			//(n*IH*IW*IC + tih*IW*IC + tiw*IC) + y*IW*IC + x*IC + ic
			int ih0 = y + tih0, iw0 = x + tiw0; bool wrt0 = (ih0 >= 0) && (ih0 < IH) && (iw0 >= 0) && (iw0 < IW) && (n0 < N);
			int ih1 = y + tih1, iw1 = x + tiw1; bool wrt1 = (ih1 >= 0) && (ih1 < IH) && (iw1 >= 0) && (iw1 < IW) && (n1 < N);
			int ih2 = y + tih2, iw2 = x + tiw2; bool wrt2 = (ih2 >= 0) && (ih2 < IH) && (iw2 >= 0) && (iw2 < IW) && (n2 < N);
			int ih3 = y + tih3, iw3 = x + tiw3; bool wrt3 = (ih3 >= 0) && (ih3 < IH) && (iw3 >= 0) && (iw3 < IW) && (n3 < N);

			if (wrt0) *(float4*)(&get4d(deltaX, n0, ih0, iw0, ic0, IH, IW, IC)) = v0;
			if (wrt1) *(float4*)(&get4d(deltaX, n1, ih1, iw1, ic0, IH, IW, IC)) = v1;
			if (wrt2) *(float4*)(&get4d(deltaX, n2, ih2, iw2, ic0, IH, IW, IC)) = v2;
			if (wrt3) *(float4*)(&get4d(deltaX, n3, ih3, iw3, ic0, IH, IW, IC)) = v3;
		}
	}
}



#define kernelSplit_kv9(stream, LB, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, CFH, CFW, GN, GM) \
	kernelSplit_kernel_v9<LB, (1<<LB>>1)>\
		<<< dim3((GM + (4<<LB) - 1)>>LB>>2, GN>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, N, IC,OC, sh,sw,ph,pw, CFH, CFW)


//Size = 0.938477, Time = 1.77667 msec, Performace = 1134.35 GFlop/s
//Size = 1.87695, Time = 3.15 msec, Performace = 1279.6 GFlop/s
template<int LB, int STEP>
__global__ void kernelSplit_kernel_v9(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw,
	int CFH, int CFW)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2  Ws[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2 dYs[2][1 << LB >> 1][(2 << LB) + 2];

	const int ic0 = ((blockIdx.y << LB) + ty) << 2;
	const int tic0 = ((tx & 1) << 1) + ic0;

	const int j0 = ((blockIdx.x << LB) + tx) << 2;//[n, ohs, ows]
	const int j1 = j0 + 1, j2 = j0 + 2, j3 = j0 + 3;
	const int oph = --CFH, opw = --CFW;
	const int OHS = OH + CFH;//OHS = OH - CFH + (oph << 1) + 1
	const int OWS = OW + CFW;//OWS = OW - CFW + (opw << 1) + 1;
	const int OHS_OWS = OHS * OWS;
	get_n_ohs_ows(j0, n0, ohs0, ows0);
	get_n_ohs_ows(j1, n1, ohs1, ows1);
	get_n_ohs_ows(j2, n2, ohs2, ows2);
	get_n_ohs_ows(j3, n3, ohs3, ows3);
	const int tih0 = ohs0 * sh - ph; ohs0 -= oph;
	const int tih1 = ohs1 * sh - ph; ohs1 -= oph;
	const int tih2 = ohs2 * sh - ph; ohs2 -= oph;
	const int tih3 = ohs3 * sh - ph; ohs3 -= oph;
	const int tiw0 = ows0 * sw - pw; ows0 -= opw;
	const int tiw1 = ows1 * sw - pw; ows1 -= opw;
	const int tiw2 = ows2 * sw - pw; ows2 -= opw;
	const int tiw3 = ows3 * sw - pw; ows3 -= opw;
	const int X_offset0 = ((n0*IH + tih0)*IW + tiw0)*IC + ic0;
	const int X_offset1 = ((n1*IH + tih1)*IW + tiw1)*IC + ic0;
	const int X_offset2 = ((n2*IH + tih2)*IW + tiw2)*IC + ic0;
	const int X_offset3 = ((n3*IH + tih3)*IW + tiw3)*IC + ic0;
	bool flagY = (ty & 1);
	const int tn0 = ((n2 - n0)*flagY + n0) * OH;
	const int tn1 = ((n3 - n1)*flagY + n1)*OH;
	const int tohs0 = (ohs2 - ohs0)*flagY + ohs0;
	const int tohs1 = (ohs3 - ohs1)*flagY + ohs1;
	const int tows0 = (ows2 - ows0)*flagY + ows0;
	const int tows1 = (ows3 - ows1)*flagY + ows1;

	const int OOC = OC << 1 >> LB, GK = FH * FW * IC;
	const int Ws_x = (tx >> 1), Ws_y = (ty << 1) + (tx & 1);
	const int dYs_y = (ty >> 1), dYs_x = (tx << 1) + (ty & 1);
	for (int y = 0; y < sh; y++) {//y*IW*IC -> y++ -> ih++, 
		for (int x = 0; x < sw; x++)//x*IC -> x++ -> iw++,  
		{
			float4 v0 = make_float4(0, 0, 0, 0);
			float4 v1 = make_float4(0, 0, 0, 0);
			float4 v2 = make_float4(0, 0, 0, 0);
			float4 v3 = make_float4(0, 0, 0, 0);
			for (int fhr = 0; fhr <= CFH; fhr++)
			{
				int fh = y + (CFH - fhr)*sh;
				int toh0 = tohs0 + fhr;
				int toh1 = tohs1 + fhr;
				for (int fwr = 0; fwr <= CFW; ++fwr)
				{
					int fw = x + (CFW - fwr)*sw;
					int W_offset = (fh*FW + fw)*IC + tic0;
					bool lw = (fh >= 0) && (fw >= 0) && (fh < FH) && (fw < FW);
					int W_oc = tx >> 1;
					Ws[buf][Ws_x][Ws_y] = lw ? *(float2*)(&W[W_oc*GK + W_offset]) : make_float2(0, 0);

					int tow0 = tows0 + fwr;
					int tow1 = tows1 + fwr;
					int Y_offset0 = ((tn0 + toh0)*OW + tow0)*OC;
					int Y_offset1 = ((tn1 + toh1)*OW + tow1)*OC;
					bool ldy0 = (toh0 >= 0) && (tow0 >= 0) && (toh0 < OH) && (tow0 < OW);
					bool ldy1 = (toh1 >= 0) && (tow1 >= 0) && (toh1 < OH) && (tow1 < OW);
					int dY_oc = ty >> 1;
					dYs[buf][dYs_y][dYs_x].x = ldy0 ? deltaY[Y_offset0 + dY_oc] : 0;
					dYs[buf][dYs_y][dYs_x].y = ldy1 ? deltaY[Y_offset1 + dY_oc] : 0;
					__syncthreads();

					for (int ooc = 1; ooc < OOC; ++ooc) {
#pragma unroll
						for (int ik = 0; ik < STEP; ik++)
						{
							float4 w = *(float4*)(&Ws[buf][ik][ty << 1]);
							float4 dy = *(float4*)(&dYs[buf][ik][tx << 1]);

							simdMM4(v0, dy.x, w);
							simdMM4(v1, dy.y, w);
							simdMM4(v2, dy.z, w);
							simdMM4(v3, dy.w, w);
						}
						buf ^= 1;

						W_oc = ((ooc << LB) + tx) >> 1;
						Ws[buf][Ws_x][Ws_y] = lw ? *(float2*)(&W[W_oc*GK + W_offset]) : make_float2(0, 0);

						dY_oc = ((ooc << LB) + ty) >> 1;
						dYs[buf][dYs_y][dYs_x].x = ldy0 ? deltaY[Y_offset0 + dY_oc] : 0;
						dYs[buf][dYs_y][dYs_x].y = ldy1 ? deltaY[Y_offset1 + dY_oc] : 0;
						__syncthreads();
					}
#pragma unroll
					for (int ik = 0; ik < STEP; ik++) {
						float4 w = *(float4*)(&Ws[buf][ik][ty << 1]);
						float4 dy = *(float4*)(&dYs[buf][ik][tx << 1]);

						simdMM4(v0, dy.x, w);
						simdMM4(v1, dy.y, w);
						simdMM4(v2, dy.z, w);
						simdMM4(v3, dy.w, w);
					}
					buf ^= 1;
				}
			}
			//((n*IH + ih)*IW + iw)*IC + ic
			//((n*IH + (tih + y))*IW + (tiw + x))*IC + ic
			//n*IH*IW*IC + (tih + y)*IW*C + (tiw+x)*IC + ic
			//(n*IH*IW*IC + tih*IW*IC + tiw*IC) + y*IW*IC + x*IC + ic

			bool wrt0 = (tih0 >= -y) && (tih0 < IH - y) && (tiw0 >= -x) && (tiw0 < IW - x) && (n0 < N);
			bool wrt1 = (tih1 >= -y) && (tih1 < IH - y) && (tiw1 >= -x) && (tiw1 < IW - x) && (n1 < N);
			bool wrt2 = (tih2 >= -y) && (tih2 < IH - y) && (tiw2 >= -x) && (tiw2 < IW - x) && (n2 < N);
			bool wrt3 = (tih3 >= -y) && (tih3 < IH - y) && (tiw3 >= -x) && (tiw3 < IW - x) && (n3 < N);

			int offset = (y * IW + x) * IC;
			if (wrt0) *(float4*)(&deltaX[X_offset0 + offset]) = v0;
			if (wrt1) *(float4*)(&deltaX[X_offset1 + offset]) = v1;
			if (wrt2) *(float4*)(&deltaX[X_offset2 + offset]) = v2;
			if (wrt3) *(float4*)(&deltaX[X_offset3 + offset]) = v3;
		}
	}
}


//Size = 1.87695, Time = 3.52 msec, Performace = 1145.09 GFlop/s
#define kernelSplit_kv10(stream, LB, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, CFH, CFW, GN, GM) \
	kernelSplit_kernel_v10<LB, (1<<LB>>1)>\
		<<< dim3((GM + (8<<LB) - 1)>>LB>>3, GN>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, N,IC,OC, sh,sw,ph,pw, CFH, CFW)

template<int LB, int STEP>
__global__ void kernelSplit_kernel_v10(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw,
	int CFH, int CFW)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4  Ws[2][1 << LB >> 1][(2 << LB) + 1];
	__shared__ float4 dYs[2][1 << LB >> 1][(2 << LB) + 1];

	const int ic0 = ((blockIdx.y << LB) + ty) << 3;
	const int tic0 = ((tx & 1) << 2) + ic0;

	const int j0 = ((blockIdx.x << LB) + tx) << 3;//[n, ohs, ows]
	const int j1 = j0 + 1, j2 = j0 + 2, j3 = j0 + 3;
	const int j4 = j0 + 4, j5 = j0 + 5, j6 = j0 + 6, j7 = j0 + 7;
	const int oph = --CFH, opw = --CFW;
	const int OHS = OH + CFH;//OHS = OH - CFH + (oph << 1) + 1
	const int OWS = OW + CFW;//OWS = OW - CFW + (opw << 1) + 1;
	const int OHS_OWS = OHS * OWS;
	get_n_ohs_ows(j0, n0, ohs0, ows0);
	get_n_ohs_ows(j1, n1, ohs1, ows1);
	get_n_ohs_ows(j2, n2, ohs2, ows2);
	get_n_ohs_ows(j3, n3, ohs3, ows3);
	const int tih0 = ohs0 * sh - ph; ohs0 -= oph;
	const int tih1 = ohs1 * sh - ph; ohs1 -= oph;
	const int tih2 = ohs2 * sh - ph; ohs2 -= oph;
	const int tih3 = ohs3 * sh - ph; ohs3 -= oph;
	const int tiw0 = ows0 * sw - pw; ows0 -= opw;
	const int tiw1 = ows1 * sw - pw; ows1 -= opw;
	const int tiw2 = ows2 * sw - pw; ows2 -= opw;
	const int tiw3 = ows3 * sw - pw; ows3 -= opw;
	int X_offset0 = ((n0*IH + tih0)*IW + tiw0)*IC + ic0;
	int X_offset1 = ((n1*IH + tih1)*IW + tiw1)*IC + ic0;
	int X_offset2 = ((n2*IH + tih2)*IW + tiw2)*IC + ic0;
	int X_offset3 = ((n3*IH + tih3)*IW + tiw3)*IC + ic0;

	get_n_ohs_ows(j4, n4, ohs4, ows4);
	get_n_ohs_ows(j5, n5, ohs5, ows5);
	get_n_ohs_ows(j6, n6, ohs6, ows6);
	get_n_ohs_ows(j7, n7, ohs7, ows7);
	const int tih4 = ohs4 * sh - ph; ohs4 -= oph;
	const int tih5 = ohs5 * sh - ph; ohs5 -= oph;
	const int tih6 = ohs6 * sh - ph; ohs6 -= oph;
	const int tih7 = ohs7 * sh - ph; ohs7 -= oph;
	const int tiw4 = ows4 * sw - pw; ows4 -= opw;
	const int tiw5 = ows5 * sw - pw; ows5 -= opw;
	const int tiw6 = ows6 * sw - pw; ows6 -= opw;
	const int tiw7 = ows7 * sw - pw; ows7 -= opw;
	int X_offset4 = ((n4*IH + tih4)*IW + tiw4)*IC + ic0;
	int X_offset5 = ((n5*IH + tih5)*IW + tiw5)*IC + ic0;
	int X_offset6 = ((n6*IH + tih6)*IW + tiw6)*IC + ic0;
	int X_offset7 = ((n7*IH + tih7)*IW + tiw7)*IC + ic0;

	bool flagY = (ty & 1);
	const int tn0 = ((n4 - n0)*flagY + n0)*OH;
	const int tn1 = ((n5 - n1)*flagY + n1)*OH;
	const int tn2 = ((n6 - n2)*flagY + n2)*OH;
	const int tn3 = ((n7 - n3)*flagY + n3)*OH;

	const int tohs0 = (ohs4 - ohs0)*flagY + ohs0;
	const int tohs1 = (ohs5 - ohs1)*flagY + ohs1;
	const int tohs2 = (ohs6 - ohs2)*flagY + ohs2;
	const int tohs3 = (ohs7 - ohs3)*flagY + ohs3;

	const int tows0 = (ows4 - ows0)*flagY + ows0;
	const int tows1 = (ows5 - ows1)*flagY + ows1;
	const int tows2 = (ows6 - ows2)*flagY + ows2;
	const int tows3 = (ows7 - ows3)*flagY + ows3;

	const int OOC = OC << 1 >> LB, GK = FH * FW * IC;
	const int Ws_x = (tx >> 1), Ws_y = (ty << 1) + (tx & 1);
	const int dYs_y = (ty >> 1), dYs_x = (tx << 1) + (ty & 1);
	for (int y = 0; y < sh; y++)//y*IW*IC -> y++ -> ih++, 
	{
		for (int x = 0; x < sw; x++)//x*IC -> x++ -> iw++,  
		{
			float4 v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
			float4 v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
			float4 v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
			float4 v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
			float4 v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
			float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
			float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
			float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);
			for (int fhr = 0; fhr <= CFH; fhr++)
			{
				int fh = y + (CFH - fhr)*sh;
				int toh0 = tohs0 + fhr;
				int toh1 = tohs1 + fhr;
				int toh2 = tohs2 + fhr;
				int toh3 = tohs3 + fhr;
				for (int fwr = 0; fwr <= CFW; ++fwr)
				{
					int fw = x + (CFW - fwr)*sw;
					int W_offset = (fh*FW + fw)*IC + tic0;
					bool lw = (fh >= 0) && (fw >= 0) && (fh < FH) && (fw < FW);
					int W_oc = tx >> 1;
					Ws[buf][Ws_x][Ws_y] = lw ? *(float4*)(&W[W_oc*GK + W_offset]) : make_float4(0, 0, 0, 0);

					int tow0 = tows0 + fwr;
					int tow1 = tows1 + fwr;
					int tow2 = tows2 + fwr;
					int tow3 = tows3 + fwr;
					int Y_offset0 = ((tn0 + toh0)*OW + tow0)*OC;
					int Y_offset1 = ((tn1 + toh1)*OW + tow1)*OC;
					int Y_offset2 = ((tn2 + toh2)*OW + tow2)*OC;
					int Y_offset3 = ((tn3 + toh3)*OW + tow3)*OC;
					bool ldy0 = (toh0 >= 0) && (tow0 >= 0) && (toh0 < OH) && (tow0 < OW);
					bool ldy1 = (toh1 >= 0) && (tow1 >= 0) && (toh1 < OH) && (tow1 < OW);
					bool ldy2 = (toh2 >= 0) && (tow2 >= 0) && (toh2 < OH) && (tow2 < OW);
					bool ldy3 = (toh3 >= 0) && (tow3 >= 0) && (toh3 < OH) && (tow3 < OW);
					int dY_oc = ty >> 1;
					dYs[buf][dYs_y][dYs_x].x = ldy0 ? deltaY[Y_offset0 + dY_oc] : 0;
					dYs[buf][dYs_y][dYs_x].y = ldy1 ? deltaY[Y_offset1 + dY_oc] : 0;
					dYs[buf][dYs_y][dYs_x].z = ldy2 ? deltaY[Y_offset2 + dY_oc] : 0;
					dYs[buf][dYs_y][dYs_x].w = ldy3 ? deltaY[Y_offset3 + dY_oc] : 0;
					__syncthreads();

					for (int ooc = 1; ooc < OOC; ++ooc) {
#pragma unroll
						for (int ik = 0; ik < STEP; ik++)
						{
							float4 dy0 = dYs[buf][ik][(tx << 1)]; 
							float4 dy1 = dYs[buf][ik][(tx << 1) + 1];
							float4  w0 =  Ws[buf][ik][(ty << 1)];
							float4  w1 =  Ws[buf][ik][(ty << 1) + 1];

							simdMM4(v0, dy0.x, w0); simdMM4(v1, dy0.x, w1);
							simdMM4(v2, dy0.y, w0); simdMM4(v3, dy0.y, w1);
							simdMM4(v4, dy0.z, w0); simdMM4(v5, dy0.z, w1);
							simdMM4(v6, dy0.w, w0); simdMM4(v7, dy0.w, w1);
							simdMM4(v8, dy1.x, w0); simdMM4(v9, dy1.x, w1);
							simdMM4(v10, dy1.y, w0); simdMM4(v11, dy1.y, w1);
							simdMM4(v12, dy1.z, w0); simdMM4(v13, dy1.z, w1);
							simdMM4(v14, dy1.w, w0); simdMM4(v15, dy1.w, w1);
						}
						buf ^= 1;

						W_oc = ((ooc << LB) + tx) >> 1;
						Ws[buf][Ws_x][Ws_y] = lw ? *(float4*)(&W[W_oc*GK + W_offset]) : make_float4(0, 0, 0, 0);

						dY_oc = ((ooc << LB) + ty) >> 1;
						dYs[buf][dYs_y][dYs_x].x = ldy0 ? deltaY[Y_offset0 + dY_oc] : 0;
						dYs[buf][dYs_y][dYs_x].y = ldy1 ? deltaY[Y_offset1 + dY_oc] : 0;
						dYs[buf][dYs_y][dYs_x].z = ldy2 ? deltaY[Y_offset2 + dY_oc] : 0;
						dYs[buf][dYs_y][dYs_x].w = ldy3 ? deltaY[Y_offset3 + dY_oc] : 0;
						__syncthreads();
					}
#pragma unroll
					for (int ik = 0; ik < STEP; ik++) 
					{
						float4 dy0 = dYs[buf][ik][(tx << 1)];
						float4 dy1 = dYs[buf][ik][(tx << 1) + 1];
						float4  w0 = Ws[buf][ik][(ty << 1)];
						float4  w1 = Ws[buf][ik][(ty << 1) + 1];

						simdMM4(v0, dy0.x, w0); simdMM4(v1, dy0.x, w1);
						simdMM4(v2, dy0.y, w0); simdMM4(v3, dy0.y, w1);
						simdMM4(v4, dy0.z, w0); simdMM4(v5, dy0.z, w1);
						simdMM4(v6, dy0.w, w0); simdMM4(v7, dy0.w, w1);
						simdMM4(v8, dy1.x, w0); simdMM4(v9, dy1.x, w1);
						simdMM4(v10, dy1.y, w0); simdMM4(v11, dy1.y, w1);
						simdMM4(v12, dy1.z, w0); simdMM4(v13, dy1.z, w1);
						simdMM4(v14, dy1.w, w0); simdMM4(v15, dy1.w, w1);
					}
					buf ^= 1;
				}
			}

			bool wrt0 = (tih0 >= -y) && (tih0 < IH - y) && (tiw0 >= -x) && (tiw0 < IW - x) && (n0 < N);
			bool wrt1 = (tih1 >= -y) && (tih1 < IH - y) && (tiw1 >= -x) && (tiw1 < IW - x) && (n1 < N);
			bool wrt2 = (tih2 >= -y) && (tih2 < IH - y) && (tiw2 >= -x) && (tiw2 < IW - x) && (n2 < N);
			bool wrt3 = (tih3 >= -y) && (tih3 < IH - y) && (tiw3 >= -x) && (tiw3 < IW - x) && (n3 < N);
			bool wrt4 = (tih4 >= -y) && (tih4 < IH - y) && (tiw4 >= -x) && (tiw4 < IW - x) && (n4 < N);
			bool wrt5 = (tih5 >= -y) && (tih5 < IH - y) && (tiw5 >= -x) && (tiw5 < IW - x) && (n5 < N);
			bool wrt6 = (tih6 >= -y) && (tih6 < IH - y) && (tiw6 >= -x) && (tiw6 < IW - x) && (n6 < N);
			bool wrt7 = (tih7 >= -y) && (tih7 < IH - y) && (tiw7 >= -x) && (tiw7 < IW - x) && (n7 < N);

			int offset = (y * IW + x) * IC;
			if (wrt0) { *(float4*)(&deltaX[X_offset0 + offset]) = v0;  *(float4*)(&deltaX[X_offset0 + offset + 4]) = v1; }
			if (wrt1) { *(float4*)(&deltaX[X_offset1 + offset]) = v2;  *(float4*)(&deltaX[X_offset1 + offset + 4]) = v3; }
			if (wrt2) { *(float4*)(&deltaX[X_offset2 + offset]) = v4;  *(float4*)(&deltaX[X_offset2 + offset + 4]) = v5; }
			if (wrt3) { *(float4*)(&deltaX[X_offset3 + offset]) = v6;  *(float4*)(&deltaX[X_offset3 + offset + 4]) = v7; }
			if (wrt4) { *(float4*)(&deltaX[X_offset4 + offset]) = v8;  *(float4*)(&deltaX[X_offset4 + offset + 4]) = v9; }
			if (wrt5) { *(float4*)(&deltaX[X_offset5 + offset]) = v10; *(float4*)(&deltaX[X_offset5 + offset + 4]) = v11; }
			if (wrt6) { *(float4*)(&deltaX[X_offset6 + offset]) = v12; *(float4*)(&deltaX[X_offset6 + offset + 4]) = v13; }
			if (wrt7) { *(float4*)(&deltaX[X_offset7 + offset]) = v14; *(float4*)(&deltaX[X_offset7 + offset + 4]) = v15; }
		}
	}
}


#define kernelSplit_kv11(stream, LB, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, CFH, CFW, GN, GM) \
	kernelSplit_kernel_v11<LB, (1<<LB>>1)>\
		<<< dim3((GM + (4<<LB) - 1)>>LB>>2, GN>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, N,IC,OC, sh,sw,ph,pw, CFH, CFW)

//Size = 0.938477, Time = 1.74667 msec, Performace = 1153.83 GFlop/s
template<int LB, int STEP>
__global__ void kernelSplit_kernel_v11(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw,
	int CFH, int CFW)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2  Ws[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2 dYs[2][1 << LB >> 1][(2 << LB) + 2];

	const int ic0 = ((blockIdx.y << LB) + ty) << 2;
	const int tic0 = ((tx & 1) << 1) + ic0;

	const int j0 = ((blockIdx.x << LB) + tx) << 2;//[n, ohs, ows]
	const int j1 = j0 + 1, j2 = j0 + 2, j3 = j0 + 3;
	const int oph = --CFH, opw = --CFW;
	const int OHS = OH + CFH;//OHS = OH - CFH + (oph << 1) + 1
	const int OWS = OW + CFW;//OWS = OW - CFW + (opw << 1) + 1;
	const int OHS_OWS = OHS * OWS;
	get_n_ohs_ows(j0, n0, ohs0, ows0);
	get_n_ohs_ows(j1, n1, ohs1, ows1);
	get_n_ohs_ows(j2, n2, ohs2, ows2);
	get_n_ohs_ows(j3, n3, ohs3, ows3);
	const int tih0 = ohs0 * sh - ph; ohs0 -= oph;
	const int tih1 = ohs1 * sh - ph; ohs1 -= oph;
	const int tih2 = ohs2 * sh - ph; ohs2 -= oph;
	const int tih3 = ohs3 * sh - ph; ohs3 -= oph;
	const int tiw0 = ows0 * sw - pw; ows0 -= opw;
	const int tiw1 = ows1 * sw - pw; ows1 -= opw;
	const int tiw2 = ows2 * sw - pw; ows2 -= opw;
	const int tiw3 = ows3 * sw - pw; ows3 -= opw;
	const int X_offset0 = ((n0*IH + tih0)*IW + tiw0)*IC + ic0;
	const int X_offset1 = ((n1*IH + tih1)*IW + tiw1)*IC + ic0;
	const int X_offset2 = ((n2*IH + tih2)*IW + tiw2)*IC + ic0;
	const int X_offset3 = ((n3*IH + tih3)*IW + tiw3)*IC + ic0;

	bool flagY = (ty & 1);
	const int tn0 = ((n2 - n0)*flagY + n0) * OH, tn1 = ((n3 - n1)*flagY + n1)*OH;
	const int tohs0 = (ohs2 - ohs0)*flagY + ohs0, tohs1 = (ohs3 - ohs1)*flagY + ohs1;
	const int tows0 = (ows2 - ows0)*flagY + ows0, tows1 = (ows3 - ows1)*flagY + ows1;

	const int OOC = OC << 1 >> LB, GK = FH * FW * IC, strideY = (IW - sw)*IC;
	const int Ws_x = (tx >> 1), Ws_y = (ty << 1) + (tx & 1);
	const int dYs_y = (ty >> 1), dYs_x = (tx << 1) + (ty & 1);
	for (int y = 0; y < sh; y++, deltaX += strideY)//y*IW*IC -> y++ -> ih++, 
	{
		for (int x = 0; x < sw; x++, deltaX += IC)//x*IC -> x++ -> iw++,  
		{
			float4 v0 = make_float4(0, 0, 0, 0);
			float4 v1 = make_float4(0, 0, 0, 0);
			float4 v2 = make_float4(0, 0, 0, 0);
			float4 v3 = make_float4(0, 0, 0, 0);
			for (int fhr = 0; fhr <= CFH; fhr++)
			{
				int fh = y + (CFH - fhr)*sh;
				int toh0 = tohs0 + fhr;
				int toh1 = tohs1 + fhr;
				for (int fwr = 0; fwr <= CFW; fwr++)
				{
					int fw = x + (CFW - fwr)*sw;
					int W_offset = (fh*FW + fw)*IC + tic0;
					bool load_w = (fh >= 0) && (fw >= 0) && (fh < FH) && (fw < FW);
					int W_oc = tx >> 1;
					Ws[buf][Ws_x][Ws_y] = load_w ? *(float2*)(&W[W_oc*GK + W_offset]) : make_float2(0, 0);

					int tow0 = tows0 + fwr;
					int tow1 = tows1 + fwr;
					int Y_offset0 = ((tn0 + toh0)*OW + tow0)*OC;
					int Y_offset1 = ((tn1 + toh1)*OW + tow1)*OC;
					bool load_dy0 = (toh0 >= 0) && (tow0 >= 0) && (toh0 < OH) && (tow0 < OW);
					bool load_dy1 = (toh1 >= 0) && (tow1 >= 0) && (toh1 < OH) && (tow1 < OW);
					int dY_oc = ty >> 1;
					dYs[buf][dYs_y][dYs_x].x = load_dy0 ? deltaY[Y_offset0 + dY_oc] : 0;
					dYs[buf][dYs_y][dYs_x].y = load_dy1 ? deltaY[Y_offset1 + dY_oc] : 0;
					__syncthreads();

					for (int ooc = 1; ooc < OOC; ooc++) {
#pragma unroll
						for (int ik = 0; ik < STEP; ik++)
						{
							float4 w = *(float4*)(&Ws[buf][ik][ty << 1]);
							float4 dy = *(float4*)(&dYs[buf][ik][tx << 1]);

							simdMM4(v0, dy.x, w);
							simdMM4(v1, dy.y, w);
							simdMM4(v2, dy.z, w);
							simdMM4(v3, dy.w, w);
						}
						buf ^= 1;

						W_oc = ((ooc << LB) + tx) >> 1;
						Ws[buf][Ws_x][Ws_y] = load_w ? *(float2*)(&W[W_oc*GK + W_offset]) : make_float2(0, 0);

						dY_oc = ((ooc << LB) + ty) >> 1;
						dYs[buf][dYs_y][dYs_x].x = load_dy0 ? deltaY[Y_offset0 + dY_oc] : 0;
						dYs[buf][dYs_y][dYs_x].y = load_dy1 ? deltaY[Y_offset1 + dY_oc] : 0;
						__syncthreads();
					}
#pragma unroll
					for (int ik = 0; ik < STEP; ik++)
					{
						float4 w = *(float4*)(&Ws[buf][ik][ty << 1]);
						float4 dy = *(float4*)(&dYs[buf][ik][tx << 1]);

						simdMM4(v0, dy.x, w);
						simdMM4(v1, dy.y, w);
						simdMM4(v2, dy.z, w);
						simdMM4(v3, dy.w, w);
					}
					buf ^= 1;
				}
			}
			
			bool wrt0 = (tih0 >= -y) && (tih0 < IH - y) && (tiw0 >= -x) && (tiw0 < IW - x) && (n0 < N);
			bool wrt1 = (tih1 >= -y) && (tih1 < IH - y) && (tiw1 >= -x) && (tiw1 < IW - x) && (n1 < N);
			bool wrt2 = (tih2 >= -y) && (tih2 < IH - y) && (tiw2 >= -x) && (tiw2 < IW - x) && (n2 < N);
			bool wrt3 = (tih3 >= -y) && (tih3 < IH - y) && (tiw3 >= -x) && (tiw3 < IW - x) && (n3 < N);
		/*	if (wrt0) *(float4*)(&deltaX[X_offset0]) = v0;
			if (wrt1) *(float4*)(&deltaX[X_offset1]) = v1;
			if (wrt2) *(float4*)(&deltaX[X_offset2]) = v2;
			if (wrt3) *(float4*)(&deltaX[X_offset3]) = v3;*/

			intptr_t hole0 = (intptr_t)(&_HOLE[X_offset0 & 1023]);
			intptr_t hole1 = (intptr_t)(&_HOLE[X_offset1 & 1023]);
			intptr_t hole2 = (intptr_t)(&_HOLE[X_offset2 & 1023]);
			intptr_t hole3 = (intptr_t)(&_HOLE[X_offset3 & 1023]);
			intptr_t addr0 = wrt0 * ((intptr_t)(&deltaX[X_offset0]) - hole0) + hole0;
			intptr_t addr1 = wrt1 * ((intptr_t)(&deltaX[X_offset1]) - hole1) + hole1;
			intptr_t addr2 = wrt2 * ((intptr_t)(&deltaX[X_offset2]) - hole2) + hole2;
			intptr_t addr3 = wrt3 * ((intptr_t)(&deltaX[X_offset3]) - hole3) + hole3;
			*(float4*)(addr0) = v0;
			*(float4*)(addr1) = v1;
			*(float4*)(addr2) = v2;
			*(float4*)(addr3) = v3;
		}
	}
}


#define kernelSplit_kv12(stream, LB, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, CFH, CFW, GN, GM) \
	kernelSplit_kernel_v12<LB, (1<<LB>>1)>\
		<<< dim3((GM + (8<<LB) - 1)>>LB>>3, GN>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, N,IC,OC, sh,sw,ph,pw, CFH, CFW)

//Size = 0.938477, Time = 1.50333 msec, Performace = 1340.6 GFlop/s
template<int LB, int STEP>
__global__ void kernelSplit_kernel_v12(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw,
	int CFH, int CFW)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4  Ws[2][1 << LB >> 1][(2 << LB) + 1];
	__shared__ float4 dYs[2][1 << LB >> 1][(2 << LB) + 1];

	const int ic0 = ((blockIdx.y << LB) + ty) << 3;
	const int tic0 = ((tx & 1) << 2) + ic0;

	const int j0 = ((blockIdx.x << LB) + tx) << 3;//[n, ohs, ows]
	const int j1 = j0 + 1, j2 = j0 + 2, j3 = j0 + 3;
	const int j4 = j0 + 4, j5 = j0 + 5, j6 = j0 + 6, j7 = j0 + 7;
	const int oph = --CFH, opw = --CFW;
	const int OHS = OH + CFH;//OHS = OH - CFH + (oph << 1) + 1
	const int OWS = OW + CFW;//OWS = OW - CFW + (opw << 1) + 1;
	const int OHS_OWS = OHS * OWS;
	get_n_ohs_ows(j0, n0, ohs0, ows0);
	get_n_ohs_ows(j1, n1, ohs1, ows1);
	get_n_ohs_ows(j2, n2, ohs2, ows2);
	get_n_ohs_ows(j3, n3, ohs3, ows3);
	get_n_ohs_ows(j4, n4, ohs4, ows4);
	get_n_ohs_ows(j5, n5, ohs5, ows5);
	get_n_ohs_ows(j6, n6, ohs6, ows6);
	get_n_ohs_ows(j7, n7, ohs7, ows7);
	const int tih0 = ohs0 * sh - ph; ohs0 -= oph;
	const int tih1 = ohs1 * sh - ph; ohs1 -= oph;
	const int tih2 = ohs2 * sh - ph; ohs2 -= oph;
	const int tih3 = ohs3 * sh - ph; ohs3 -= oph;
	const int tiw0 = ows0 * sw - pw; ows0 -= opw;
	const int tiw1 = ows1 * sw - pw; ows1 -= opw;
	const int tiw2 = ows2 * sw - pw; ows2 -= opw;
	const int tiw3 = ows3 * sw - pw; ows3 -= opw;
	const int tih4 = ohs4 * sh - ph; ohs4 -= oph;
	const int tih5 = ohs5 * sh - ph; ohs5 -= oph;
	const int tih6 = ohs6 * sh - ph; ohs6 -= oph;
	const int tih7 = ohs7 * sh - ph; ohs7 -= oph;
	const int tiw4 = ows4 * sw - pw; ows4 -= opw;
	const int tiw5 = ows5 * sw - pw; ows5 -= opw;
	const int tiw6 = ows6 * sw - pw; ows6 -= opw;
	const int tiw7 = ows7 * sw - pw; ows7 -= opw;
	int X_offset0 = ((n0*IH + tih0)*IW + tiw0)*IC + ic0;
	int X_offset1 = ((n1*IH + tih1)*IW + tiw1)*IC + ic0;
	int X_offset2 = ((n2*IH + tih2)*IW + tiw2)*IC + ic0;
	int X_offset3 = ((n3*IH + tih3)*IW + tiw3)*IC + ic0;
	int X_offset4 = ((n4*IH + tih4)*IW + tiw4)*IC + ic0;
	int X_offset5 = ((n5*IH + tih5)*IW + tiw5)*IC + ic0;
	int X_offset6 = ((n6*IH + tih6)*IW + tiw6)*IC + ic0;
	int X_offset7 = ((n7*IH + tih7)*IW + tiw7)*IC + ic0;
	bool flagY = (ty & 1);
	const int tn0 = ((n4 - n0)*flagY + n0)*OH;
	const int tn1 = ((n5 - n1)*flagY + n1)*OH;
	const int tn2 = ((n6 - n2)*flagY + n2)*OH;
	const int tn3 = ((n7 - n3)*flagY + n3)*OH;
	const int tohs0 = (ohs4 - ohs0)*flagY + ohs0;
	const int tohs1 = (ohs5 - ohs1)*flagY + ohs1;
	const int tohs2 = (ohs6 - ohs2)*flagY + ohs2;
	const int tohs3 = (ohs7 - ohs3)*flagY + ohs3;
	const int tows0 = (ows4 - ows0)*flagY + ows0;
	const int tows1 = (ows5 - ows1)*flagY + ows1;
	const int tows2 = (ows6 - ows2)*flagY + ows2;
	const int tows3 = (ows7 - ows3)*flagY + ows3;

	const int OOC = OC << 1 >> LB, GK = FH * FW * IC, strideY = (IW - sw)*IC;
	const int Ws_x = (tx >> 1), Ws_y = (ty << 1) + (tx & 1);
	const int dYs_y = (ty >> 1), dYs_x = (tx << 1) + (ty & 1);
	for (int y = 0; y < sh; y++, deltaX += strideY) {
		for (int x = 0; x < sw; x++, deltaX += IC)
		{
			float4 v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
			float4 v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
			float4 v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
			float4 v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
			float4 v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
			float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
			float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
			float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);
			for (int fhr = 0; fhr <= CFH; fhr++) {
				for (int fwr = 0; fwr <= CFW; ++fwr)
				{
					int fh = y + (CFH - fhr)*sh, fw = x + (CFW - fwr)*sw;
					int W_offset = (fh*FW + fw)*IC + tic0;
					bool lw = (fh >= 0) && (fw >= 0) && (fh < FH) && (fw < FW);
					int W_oc = tx >> 1;
					Ws[buf][Ws_x][Ws_y] = lw ? *(float4*)(&W[W_oc*GK + W_offset]) : make_float4(0, 0, 0, 0);
					
					int toh0 = tohs0 + fhr, tow0 = tows0 + fwr, Y_offset0 = ((tn0 + toh0)*OW + tow0)*OC;
					int toh1 = tohs1 + fhr, tow1 = tows1 + fwr, Y_offset1 = ((tn1 + toh1)*OW + tow1)*OC;
					int toh2 = tohs2 + fhr, tow2 = tows2 + fwr, Y_offset2 = ((tn2 + toh2)*OW + tow2)*OC;
					int toh3 = tohs3 + fhr, tow3 = tows3 + fwr, Y_offset3 = ((tn3 + toh3)*OW + tow3)*OC;
					bool ldy0 = (toh0 >= 0) && (tow0 >= 0) && (toh0 < OH) && (tow0 < OW);
					bool ldy1 = (toh1 >= 0) && (tow1 >= 0) && (toh1 < OH) && (tow1 < OW);
					bool ldy2 = (toh2 >= 0) && (tow2 >= 0) && (toh2 < OH) && (tow2 < OW);
					bool ldy3 = (toh3 >= 0) && (tow3 >= 0) && (toh3 < OH) && (tow3 < OW);
					int dY_oc = ty >> 1;
					dYs[buf][dYs_y][dYs_x].x = ldy0 ? deltaY[Y_offset0 + dY_oc] : 0;
					dYs[buf][dYs_y][dYs_x].y = ldy1 ? deltaY[Y_offset1 + dY_oc] : 0;
					dYs[buf][dYs_y][dYs_x].z = ldy2 ? deltaY[Y_offset2 + dY_oc] : 0;
					dYs[buf][dYs_y][dYs_x].w = ldy3 ? deltaY[Y_offset3 + dY_oc] : 0;
					__syncthreads();

					for (int ooc = 1; ooc < OOC; ++ooc) {
#pragma unroll
						for (int ik = 0; ik < STEP; ik++)
						{
							float4 dy0 = dYs[buf][ik][(tx << 1)];
							float4 dy1 = dYs[buf][ik][(tx << 1) + 1];
							float4  w0 = Ws[buf][ik][(ty << 1)];
							float4  w1 = Ws[buf][ik][(ty << 1) + 1];

							simdMM4(v0, dy0.x, w0); simdMM4(v1, dy0.x, w1);
							simdMM4(v2, dy0.y, w0); simdMM4(v3, dy0.y, w1);
							simdMM4(v4, dy0.z, w0); simdMM4(v5, dy0.z, w1);
							simdMM4(v6, dy0.w, w0); simdMM4(v7, dy0.w, w1);
							simdMM4(v8, dy1.x, w0); simdMM4(v9, dy1.x, w1);
							simdMM4(v10, dy1.y, w0); simdMM4(v11, dy1.y, w1);
							simdMM4(v12, dy1.z, w0); simdMM4(v13, dy1.z, w1);
							simdMM4(v14, dy1.w, w0); simdMM4(v15, dy1.w, w1);
						}
						buf ^= 1;

						W_oc = ((ooc << LB) + tx) >> 1;
						Ws[buf][Ws_x][Ws_y] = lw ? *(float4*)(&W[W_oc*GK + W_offset]) : make_float4(0, 0, 0, 0);

						dY_oc = ((ooc << LB) + ty) >> 1;
						dYs[buf][dYs_y][dYs_x].x = ldy0 ? deltaY[Y_offset0 + dY_oc] : 0;
						dYs[buf][dYs_y][dYs_x].y = ldy1 ? deltaY[Y_offset1 + dY_oc] : 0;
						dYs[buf][dYs_y][dYs_x].z = ldy2 ? deltaY[Y_offset2 + dY_oc] : 0;
						dYs[buf][dYs_y][dYs_x].w = ldy3 ? deltaY[Y_offset3 + dY_oc] : 0;
						__syncthreads();
					}
#pragma unroll
					for (int ik = 0; ik < STEP; ik++)
					{
						float4 dy0 = dYs[buf][ik][(tx << 1)];
						float4 dy1 = dYs[buf][ik][(tx << 1) + 1];
						float4  w0 = Ws[buf][ik][(ty << 1)];
						float4  w1 = Ws[buf][ik][(ty << 1) + 1];

						simdMM4(v0, dy0.x, w0); simdMM4(v1, dy0.x, w1);
						simdMM4(v2, dy0.y, w0); simdMM4(v3, dy0.y, w1);
						simdMM4(v4, dy0.z, w0); simdMM4(v5, dy0.z, w1);
						simdMM4(v6, dy0.w, w0); simdMM4(v7, dy0.w, w1);
						simdMM4(v8, dy1.x, w0); simdMM4(v9, dy1.x, w1);
						simdMM4(v10, dy1.y, w0); simdMM4(v11, dy1.y, w1);
						simdMM4(v12, dy1.z, w0); simdMM4(v13, dy1.z, w1);
						simdMM4(v14, dy1.w, w0); simdMM4(v15, dy1.w, w1);
					}
					buf ^= 1;
				}
			}

			bool wrt0 = (tih0 >= -y) && (tih0 < IH - y) && (tiw0 >= -x) && (tiw0 < IW - x) && (n0 < N);
			bool wrt1 = (tih1 >= -y) && (tih1 < IH - y) && (tiw1 >= -x) && (tiw1 < IW - x) && (n1 < N);
			bool wrt2 = (tih2 >= -y) && (tih2 < IH - y) && (tiw2 >= -x) && (tiw2 < IW - x) && (n2 < N);
			bool wrt3 = (tih3 >= -y) && (tih3 < IH - y) && (tiw3 >= -x) && (tiw3 < IW - x) && (n3 < N);
			bool wrt4 = (tih4 >= -y) && (tih4 < IH - y) && (tiw4 >= -x) && (tiw4 < IW - x) && (n4 < N);
			bool wrt5 = (tih5 >= -y) && (tih5 < IH - y) && (tiw5 >= -x) && (tiw5 < IW - x) && (n5 < N);
			bool wrt6 = (tih6 >= -y) && (tih6 < IH - y) && (tiw6 >= -x) && (tiw6 < IW - x) && (n6 < N);
			bool wrt7 = (tih7 >= -y) && (tih7 < IH - y) && (tiw7 >= -x) && (tiw7 < IW - x) && (n7 < N);

			int offset = (y * IW + x) * IC;
			if (wrt0) { *(float4*)(&deltaX[X_offset0]) = v0;  *(float4*)(&deltaX[X_offset0 + 4]) = v1; }
			if (wrt1) { *(float4*)(&deltaX[X_offset1]) = v2;  *(float4*)(&deltaX[X_offset1 + 4]) = v3; }
			if (wrt2) { *(float4*)(&deltaX[X_offset2]) = v4;  *(float4*)(&deltaX[X_offset2 + 4]) = v5; }
			if (wrt3) { *(float4*)(&deltaX[X_offset3]) = v6;  *(float4*)(&deltaX[X_offset3 + 4]) = v7; }
			if (wrt4) { *(float4*)(&deltaX[X_offset4]) = v8;  *(float4*)(&deltaX[X_offset4 + 4]) = v9; }
			if (wrt5) { *(float4*)(&deltaX[X_offset5]) = v10; *(float4*)(&deltaX[X_offset5 + 4]) = v11; }
			if (wrt6) { *(float4*)(&deltaX[X_offset6]) = v12; *(float4*)(&deltaX[X_offset6 + 4]) = v13; }
			if (wrt7) { *(float4*)(&deltaX[X_offset7]) = v14; *(float4*)(&deltaX[X_offset7 + 4]) = v15; }
		}
	}
}


#define kernelSplit_kv13(stream, LB, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, CFH, CFW, GN, GM) \
	kernelSplit_kernel_v13<LB, (1<<LB>>1)>\
		<<< dim3((GM + (8<<LB) - 1)>>LB>>3, GN>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, N,IC,OC, sh,sw,ph,pw, CFH, CFW)

//Size = 0.938477, Time = 1.39667 msec, Performace = 1442.98 GFlop/s
template<int LB, int STEP>
__global__ void kernelSplit_kernel_v13(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw,
	int CFH, int CFW)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4  Ws[2][1 << LB >> 1][(2 << LB) + 1];
	__shared__ float4 dYs[2][1 << LB >> 1][(2 << LB) + 1];

	const int ic0 = ((blockIdx.y << LB) + ty) << 3;
	const int tic0 = ((tx & 1) << 2) + ic0;

	const int j0 = ((blockIdx.x << LB) + tx) << 3;//[n, ohs, ows]
	const int j1 = j0 + 1, j2 = j0 + 2, j3 = j0 + 3;
	const int j4 = j0 + 4, j5 = j0 + 5, j6 = j0 + 6, j7 = j0 + 7;
	const int oph = --CFH, opw = --CFW;
	const int OHS = OH + CFH;//OHS = OH - CFH + (oph << 1) + 1
	const int OWS = OW + CFW;//OWS = OW - CFW + (opw << 1) + 1;
	const int OHS_OWS = OHS * OWS;
	get_n_ohs_ows(j0, n0, ohs0, ows0);
	get_n_ohs_ows(j1, n1, ohs1, ows1);
	get_n_ohs_ows(j2, n2, ohs2, ows2);
	get_n_ohs_ows(j3, n3, ohs3, ows3);
	get_n_ohs_ows(j4, n4, ohs4, ows4);
	get_n_ohs_ows(j5, n5, ohs5, ows5);
	get_n_ohs_ows(j6, n6, ohs6, ows6);
	get_n_ohs_ows(j7, n7, ohs7, ows7);
	const int tih0 = ohs0 * sh - ph; ohs0 -= oph;
	const int tih1 = ohs1 * sh - ph; ohs1 -= oph;
	const int tih2 = ohs2 * sh - ph; ohs2 -= oph;
	const int tih3 = ohs3 * sh - ph; ohs3 -= oph;
	const int tiw0 = ows0 * sw - pw; ows0 -= opw;
	const int tiw1 = ows1 * sw - pw; ows1 -= opw;
	const int tiw2 = ows2 * sw - pw; ows2 -= opw;
	const int tiw3 = ows3 * sw - pw; ows3 -= opw;
	const int tih4 = ohs4 * sh - ph; ohs4 -= oph;
	const int tih5 = ohs5 * sh - ph; ohs5 -= oph;
	const int tih6 = ohs6 * sh - ph; ohs6 -= oph;
	const int tih7 = ohs7 * sh - ph; ohs7 -= oph;
	const int tiw4 = ows4 * sw - pw; ows4 -= opw;
	const int tiw5 = ows5 * sw - pw; ows5 -= opw;
	const int tiw6 = ows6 * sw - pw; ows6 -= opw;
	const int tiw7 = ows7 * sw - pw; ows7 -= opw;
	int X_offset0 = ((n0*IH + tih0)*IW + tiw0)*IC + ic0;
	int X_offset1 = ((n1*IH + tih1)*IW + tiw1)*IC + ic0;
	int X_offset2 = ((n2*IH + tih2)*IW + tiw2)*IC + ic0;
	int X_offset3 = ((n3*IH + tih3)*IW + tiw3)*IC + ic0;
	int X_offset4 = ((n4*IH + tih4)*IW + tiw4)*IC + ic0;
	int X_offset5 = ((n5*IH + tih5)*IW + tiw5)*IC + ic0;
	int X_offset6 = ((n6*IH + tih6)*IW + tiw6)*IC + ic0;
	int X_offset7 = ((n7*IH + tih7)*IW + tiw7)*IC + ic0;

	bool flagY = (ty & 1);
	const int tn0 = (n4 - n0)*flagY + n0;
	const int tn1 = (n5 - n1)*flagY + n1;
	const int tn2 = (n6 - n2)*flagY + n2;
	const int tn3 = (n7 - n3)*flagY + n3;
	const int tohs0 = (ohs4 - ohs0)*flagY + ohs0;
	const int tohs1 = (ohs5 - ohs1)*flagY + ohs1;
	const int tohs2 = (ohs6 - ohs2)*flagY + ohs2;
	const int tohs3 = (ohs7 - ohs3)*flagY + ohs3;
	const int tows0 = (ows4 - ows0)*flagY + ows0;
	const int tows1 = (ows5 - ows1)*flagY + ows1;
	const int tows2 = (ows6 - ows2)*flagY + ows2;
	const int tows3 = (ows7 - ows3)*flagY + ows3;
	int Y_offset0 = ((tn0*OH + tohs0)*OW + tows0)*OC;
	int Y_offset1 = ((tn1*OH + tohs1)*OW + tows1)*OC;
	int Y_offset2 = ((tn2*OH + tohs2)*OW + tows2)*OC;
	int Y_offset3 = ((tn3*OH + tohs3)*OW + tows3)*OC;

	const int GK = FH * FW * IC, strideY = (IW - sw)*IC;
	const int Ws_x = (tx >> 1), Ws_y = (ty << 1) + (tx & 1);
	const int dYs_y = (ty >> 1), dYs_x = (tx << 1) + (ty & 1);

	for (int y = 0; y < sh; y++, deltaX += strideY) {
		for (int x = 0; x < sw; x++, deltaX += IC)
		{
			float4 v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
			float4 v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
			float4 v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
			float4 v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
			float4 v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
			float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
			float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
			float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);
			for (int fhr = 0; fhr <= CFH; fhr++) {
				for (int fwr = 0; fwr <= CFW; ++fwr)
				{
					int fh = y + (CFH - fhr)*sh, fw = x + (CFW - fwr)*sw;
					int W_offset = (fh*FW + fw)*IC + tic0;
					bool lw = (fh >= 0) && (fw >= 0) && (fh < FH) && (fw < FW);
					int W_oc = tx >> 1;
					Ws[buf][Ws_x][Ws_y] = lw ? *(float4*)(&W[W_oc*GK + W_offset]) : make_float4(0, 0, 0, 0);

					bool ldy0 = (tohs0 >= -fhr) && (tohs0 < OH - fhr) && (tows0 >= -fwr) && (tows0 < OW - fwr);
					bool ldy1 = (tohs1 >= -fhr) && (tohs1 < OH - fhr) && (tows1 >= -fwr) && (tows1 < OW - fwr);
					bool ldy2 = (tohs2 >= -fhr) && (tohs2 < OH - fhr) && (tows2 >= -fwr) && (tows2 < OW - fwr);
					bool ldy3 = (tohs3 >= -fhr) && (tohs3 < OH - fhr) && (tows3 >= -fwr) && (tows3 < OW - fwr);
					int Y_oc = ty >> 1;
					int Y_offset = (fhr*OW + fwr)*OC;
					dYs[buf][dYs_y][dYs_x].x = ldy0 ? deltaY[Y_offset0 + Y_offset + Y_oc] : 0;
					dYs[buf][dYs_y][dYs_x].y = ldy1 ? deltaY[Y_offset1 + Y_offset + Y_oc] : 0;
					dYs[buf][dYs_y][dYs_x].z = ldy2 ? deltaY[Y_offset2 + Y_offset + Y_oc] : 0;
					dYs[buf][dYs_y][dYs_x].w = ldy3 ? deltaY[Y_offset3 + Y_offset + Y_oc] : 0;
					__syncthreads();

					for (int ooc = 1, OOC = OC << 1 >> LB; ooc < OOC; ++ooc) {
#pragma unroll
						for (int ik = 0; ik < STEP; ik++)
						{
							float4 dy0 = dYs[buf][ik][(tx << 1)];
							float4 dy1 = dYs[buf][ik][(tx << 1) + 1];
							float4  w0 = Ws[buf][ik][(ty << 1)];
							float4  w1 = Ws[buf][ik][(ty << 1) + 1];

							simdMM4(v0, dy0.x, w0); simdMM4(v1, dy0.x, w1);
							simdMM4(v2, dy0.y, w0); simdMM4(v3, dy0.y, w1);
							simdMM4(v4, dy0.z, w0); simdMM4(v5, dy0.z, w1);
							simdMM4(v6, dy0.w, w0); simdMM4(v7, dy0.w, w1);
							simdMM4(v8, dy1.x, w0); simdMM4(v9, dy1.x, w1);
							simdMM4(v10, dy1.y, w0); simdMM4(v11, dy1.y, w1);
							simdMM4(v12, dy1.z, w0); simdMM4(v13, dy1.z, w1);
							simdMM4(v14, dy1.w, w0); simdMM4(v15, dy1.w, w1);
						}
						buf ^= 1;

						W_oc = ((ooc << LB) + tx) >> 1;
						Ws[buf][Ws_x][Ws_y] = lw ? *(float4*)(&W[W_oc*GK + W_offset]) : make_float4(0, 0, 0, 0);

						Y_oc = ((ooc << LB) + ty) >> 1;
						dYs[buf][dYs_y][dYs_x].x = ldy0 ? deltaY[Y_offset0 + Y_offset + Y_oc] : 0;
						dYs[buf][dYs_y][dYs_x].y = ldy1 ? deltaY[Y_offset1 + Y_offset + Y_oc] : 0;
						dYs[buf][dYs_y][dYs_x].z = ldy2 ? deltaY[Y_offset2 + Y_offset + Y_oc] : 0;
						dYs[buf][dYs_y][dYs_x].w = ldy3 ? deltaY[Y_offset3 + Y_offset + Y_oc] : 0;
						__syncthreads();
					}
#pragma unroll
					for (int ik = 0; ik < STEP; ik++)
					{
						float4 dy0 = dYs[buf][ik][(tx << 1)];
						float4 dy1 = dYs[buf][ik][(tx << 1) + 1];
						float4  w0 = Ws[buf][ik][(ty << 1)];
						float4  w1 = Ws[buf][ik][(ty << 1) + 1];

						simdMM4(v0, dy0.x, w0); simdMM4(v1, dy0.x, w1);
						simdMM4(v2, dy0.y, w0); simdMM4(v3, dy0.y, w1);
						simdMM4(v4, dy0.z, w0); simdMM4(v5, dy0.z, w1);
						simdMM4(v6, dy0.w, w0); simdMM4(v7, dy0.w, w1);
						simdMM4(v8, dy1.x, w0); simdMM4(v9, dy1.x, w1);
						simdMM4(v10, dy1.y, w0); simdMM4(v11, dy1.y, w1);
						simdMM4(v12, dy1.z, w0); simdMM4(v13, dy1.z, w1);
						simdMM4(v14, dy1.w, w0); simdMM4(v15, dy1.w, w1);
					}
					buf ^= 1;
				}
			}

			bool wrt0 = (tih0 >= -y) && (tih0 < IH - y) && (tiw0 >= -x) && (tiw0 < IW - x) && (n0 < N);
			bool wrt1 = (tih1 >= -y) && (tih1 < IH - y) && (tiw1 >= -x) && (tiw1 < IW - x) && (n1 < N);
			bool wrt2 = (tih2 >= -y) && (tih2 < IH - y) && (tiw2 >= -x) && (tiw2 < IW - x) && (n2 < N);
			bool wrt3 = (tih3 >= -y) && (tih3 < IH - y) && (tiw3 >= -x) && (tiw3 < IW - x) && (n3 < N);
			bool wrt4 = (tih4 >= -y) && (tih4 < IH - y) && (tiw4 >= -x) && (tiw4 < IW - x) && (n4 < N);
			bool wrt5 = (tih5 >= -y) && (tih5 < IH - y) && (tiw5 >= -x) && (tiw5 < IW - x) && (n5 < N);
			bool wrt6 = (tih6 >= -y) && (tih6 < IH - y) && (tiw6 >= -x) && (tiw6 < IW - x) && (n6 < N);
			bool wrt7 = (tih7 >= -y) && (tih7 < IH - y) && (tiw7 >= -x) && (tiw7 < IW - x) && (n7 < N);

			int offset = (y * IW + x) * IC;
			if (wrt0) { *(float4*)(&deltaX[X_offset0]) = v0;  *(float4*)(&deltaX[X_offset0 + 4]) = v1; }
			if (wrt1) { *(float4*)(&deltaX[X_offset1]) = v2;  *(float4*)(&deltaX[X_offset1 + 4]) = v3; }
			if (wrt2) { *(float4*)(&deltaX[X_offset2]) = v4;  *(float4*)(&deltaX[X_offset2 + 4]) = v5; }
			if (wrt3) { *(float4*)(&deltaX[X_offset3]) = v6;  *(float4*)(&deltaX[X_offset3 + 4]) = v7; }
			if (wrt4) { *(float4*)(&deltaX[X_offset4]) = v8;  *(float4*)(&deltaX[X_offset4 + 4]) = v9; }
			if (wrt5) { *(float4*)(&deltaX[X_offset5]) = v10; *(float4*)(&deltaX[X_offset5 + 4]) = v11; }
			if (wrt6) { *(float4*)(&deltaX[X_offset6]) = v12; *(float4*)(&deltaX[X_offset6 + 4]) = v13; }
			if (wrt7) { *(float4*)(&deltaX[X_offset7]) = v14; *(float4*)(&deltaX[X_offset7 + 4]) = v15; }
		}
	}
}



#define kernelSplit_kv14(stream, LB, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, CFH, CFW, GN, GM) \
	kernelSplit_kernel_v14<LB, (1<<LB>>1)>\
		<<< dim3((GM + (8<<LB) - 1)>>LB>>3, GN>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, N,IC,OC, sh,sw,ph,pw, CFH, CFW)

//Size = 0.938477, Time = 1.39667 msec, Performace = 1442.98 GFlop/s
template<int LB, int STEP>
__global__ void kernelSplit_kernel_v14(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw,
	int CFH, int CFW)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4  Ws[2][1 << LB >> 1][(2 << LB) + 1];
	__shared__ float4 dYs[2][1 << LB >> 1][(2 << LB) + 1];

	const int ic0 = ((blockIdx.y << LB) + ty) << 3;
	const int tic0 = ((tx & 1) << 2) + ic0;

	const int j0 = ((blockIdx.x << LB) + tx) << 3;//[n, ohs, ows]
	const int j1 = j0 + 1, j2 = j0 + 2, j3 = j0 + 3;
	const int j4 = j0 + 4, j5 = j0 + 5, j6 = j0 + 6, j7 = j0 + 7;
	const int oph = --CFH, opw = --CFW;
	const int OHS = OH + CFH;//OHS = OH - CFH + (oph << 1) + 1
	const int OWS = OW + CFW;//OWS = OW - CFW + (opw << 1) + 1;
	const int OHS_OWS = OHS * OWS;
	get_n_ohs_ows(j0, n0, ohs0, ows0);
	get_n_ohs_ows(j1, n1, ohs1, ows1);
	get_n_ohs_ows(j2, n2, ohs2, ows2);
	get_n_ohs_ows(j3, n3, ohs3, ows3);
	get_n_ohs_ows(j4, n4, ohs4, ows4);
	get_n_ohs_ows(j5, n5, ohs5, ows5);
	get_n_ohs_ows(j6, n6, ohs6, ows6);
	get_n_ohs_ows(j7, n7, ohs7, ows7);
	const int tih0 = ohs0 * sh - ph; ohs0 -= oph;
	const int tih1 = ohs1 * sh - ph; ohs1 -= oph;
	const int tih2 = ohs2 * sh - ph; ohs2 -= oph;
	const int tih3 = ohs3 * sh - ph; ohs3 -= oph;
	const int tih4 = ohs4 * sh - ph; ohs4 -= oph;
	const int tih5 = ohs5 * sh - ph; ohs5 -= oph;
	const int tih6 = ohs6 * sh - ph; ohs6 -= oph;
	const int tih7 = ohs7 * sh - ph; ohs7 -= oph;
	const int tiw0 = ows0 * sw - pw; ows0 -= opw;
	const int tiw1 = ows1 * sw - pw; ows1 -= opw;
	const int tiw2 = ows2 * sw - pw; ows2 -= opw;
	const int tiw3 = ows3 * sw - pw; ows3 -= opw;
	const int tiw4 = ows4 * sw - pw; ows4 -= opw;
	const int tiw5 = ows5 * sw - pw; ows5 -= opw;
	const int tiw6 = ows6 * sw - pw; ows6 -= opw;
	const int tiw7 = ows7 * sw - pw; ows7 -= opw;
	int X_offset0 = ((n0*IH + tih0)*IW + tiw0)*IC + ic0;
	int X_offset1 = ((n1*IH + tih1)*IW + tiw1)*IC + ic0;
	int X_offset2 = ((n2*IH + tih2)*IW + tiw2)*IC + ic0;
	int X_offset3 = ((n3*IH + tih3)*IW + tiw3)*IC + ic0;
	int X_offset4 = ((n4*IH + tih4)*IW + tiw4)*IC + ic0;
	int X_offset5 = ((n5*IH + tih5)*IW + tiw5)*IC + ic0;
	int X_offset6 = ((n6*IH + tih6)*IW + tiw6)*IC + ic0;
	int X_offset7 = ((n7*IH + tih7)*IW + tiw7)*IC + ic0;

	bool flagY = (ty & 1);
	const int tn0 = (n4 - n0)*flagY + n0;
	const int tn1 = (n5 - n1)*flagY + n1;
	const int tn2 = (n6 - n2)*flagY + n2;
	const int tn3 = (n7 - n3)*flagY + n3;
	const int tohs0 = (ohs4 - ohs0)*flagY + ohs0;
	const int tohs1 = (ohs5 - ohs1)*flagY + ohs1;
	const int tohs2 = (ohs6 - ohs2)*flagY + ohs2;
	const int tohs3 = (ohs7 - ohs3)*flagY + ohs3;
	const int tows0 = (ows4 - ows0)*flagY + ows0;
	const int tows1 = (ows5 - ows1)*flagY + ows1;
	const int tows2 = (ows6 - ows2)*flagY + ows2;
	const int tows3 = (ows7 - ows3)*flagY + ows3;
	int Y_offset0 = ((tn0*OH + tohs0)*OW + tows0)*OC;
	int Y_offset1 = ((tn1*OH + tohs1)*OW + tows1)*OC;
	int Y_offset2 = ((tn2*OH + tohs2)*OW + tows2)*OC;
	int Y_offset3 = ((tn3*OH + tohs3)*OW + tows3)*OC;

	const int GK = FH * FW * IC, strideY = (IW - sw)*IC;
	const int Ws_x = (tx >> 1), Ws_y = (ty << 1) + (tx & 1);
	const int dYs_y = (ty >> 1), dYs_x = (tx << 1) + (ty & 1);

	for (int y = 0; y < sh; y++, deltaX += strideY) {
		for (int x = 0; x < sw; x++, deltaX += IC)
		{
			float4 v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
			float4 v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
			float4 v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
			float4 v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
			float4 v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
			float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
			float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
			float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);
			for (int fhr = 0; fhr <= CFH; fhr++) {
				for (int fwr = 0; fwr <= CFW; ++fwr)
				{
					int fh = y + (CFH - fhr)*sh, fw = x + (CFW - fwr)*sw;
					int W_offset = (fh*FW + fw)*IC + tic0;
					bool lw = (fh >= 0) && (fw >= 0) && (fh < FH) && (fw < FW);
					int W_oc = tx >> 1;
					Ws[buf][Ws_x][Ws_y] = lw ? *(float4*)(&W[W_oc*GK + W_offset]) : make_float4(0, 0, 0, 0);

					bool ldy0 = (tohs0 >= -fhr) && (tohs0 < OH - fhr) && (tows0 >= -fwr) && (tows0 < OW - fwr);
					bool ldy1 = (tohs1 >= -fhr) && (tohs1 < OH - fhr) && (tows1 >= -fwr) && (tows1 < OW - fwr);
					bool ldy2 = (tohs2 >= -fhr) && (tohs2 < OH - fhr) && (tows2 >= -fwr) && (tows2 < OW - fwr);
					bool ldy3 = (tohs3 >= -fhr) && (tohs3 < OH - fhr) && (tows3 >= -fwr) && (tows3 < OW - fwr);
					int Y_oc = ty >> 1;
					int Y_offset = (fhr*OW + fwr)*OC;
					deltaY += Y_offset + Y_oc;
					dYs[buf][dYs_y][dYs_x].x = ldy0 ? deltaY[Y_offset0] : 0;
					dYs[buf][dYs_y][dYs_x].y = ldy1 ? deltaY[Y_offset1] : 0;
					dYs[buf][dYs_y][dYs_x].z = ldy2 ? deltaY[Y_offset2] : 0;
					dYs[buf][dYs_y][dYs_x].w = ldy3 ? deltaY[Y_offset3] : 0;
					deltaY -= Y_offset + Y_oc;
					__syncthreads();

					for (int ooc = 1, OOC = OC << 1 >> LB; ooc < OOC; ++ooc) {
#pragma unroll
						for (int ik = 0; ik < STEP; ik++)
						{
							float4 dy0 = dYs[buf][ik][(tx << 1)];
							float4 dy1 = dYs[buf][ik][(tx << 1) + 1];
							float4  w0 = Ws[buf][ik][(ty << 1)];
							float4  w1 = Ws[buf][ik][(ty << 1) + 1];

							simdMM4(v0, dy0.x, w0); simdMM4(v1, dy0.x, w1);
							simdMM4(v2, dy0.y, w0); simdMM4(v3, dy0.y, w1);
							simdMM4(v4, dy0.z, w0); simdMM4(v5, dy0.z, w1);
							simdMM4(v6, dy0.w, w0); simdMM4(v7, dy0.w, w1);
							simdMM4(v8, dy1.x, w0); simdMM4(v9, dy1.x, w1);
							simdMM4(v10, dy1.y, w0); simdMM4(v11, dy1.y, w1);
							simdMM4(v12, dy1.z, w0); simdMM4(v13, dy1.z, w1);
							simdMM4(v14, dy1.w, w0); simdMM4(v15, dy1.w, w1);
						}
						buf ^= 1;

						W_oc = ((ooc << LB) + tx) >> 1;
						Ws[buf][Ws_x][Ws_y] = lw ? *(float4*)(&W[W_oc*GK + W_offset]) : make_float4(0, 0, 0, 0);

						Y_oc = ((ooc << LB) + ty) >> 1;
						deltaY += Y_offset + Y_oc;
						dYs[buf][dYs_y][dYs_x].x = ldy0 ? deltaY[Y_offset0] : 0;
						dYs[buf][dYs_y][dYs_x].y = ldy1 ? deltaY[Y_offset1] : 0;
						dYs[buf][dYs_y][dYs_x].z = ldy2 ? deltaY[Y_offset2] : 0;
						dYs[buf][dYs_y][dYs_x].w = ldy3 ? deltaY[Y_offset3] : 0;
						deltaY -= Y_offset + Y_oc;
						__syncthreads();
					}
#pragma unroll
					for (int ik = 0; ik < STEP; ik++)
					{
						float4 dy0 = dYs[buf][ik][(tx << 1)];
						float4 dy1 = dYs[buf][ik][(tx << 1) + 1];
						float4  w0 = Ws[buf][ik][(ty << 1)];
						float4  w1 = Ws[buf][ik][(ty << 1) + 1];

						simdMM4(v0, dy0.x, w0); simdMM4(v1, dy0.x, w1);
						simdMM4(v2, dy0.y, w0); simdMM4(v3, dy0.y, w1);
						simdMM4(v4, dy0.z, w0); simdMM4(v5, dy0.z, w1);
						simdMM4(v6, dy0.w, w0); simdMM4(v7, dy0.w, w1);
						simdMM4(v8, dy1.x, w0); simdMM4(v9, dy1.x, w1);
						simdMM4(v10, dy1.y, w0); simdMM4(v11, dy1.y, w1);
						simdMM4(v12, dy1.z, w0); simdMM4(v13, dy1.z, w1);
						simdMM4(v14, dy1.w, w0); simdMM4(v15, dy1.w, w1);
					}
					buf ^= 1;
				}
			}

			bool wrt0 = (tih0 >= -y) && (tih0 < IH - y) && (tiw0 >= -x) && (tiw0 < IW - x) && (n0 < N);
			bool wrt1 = (tih1 >= -y) && (tih1 < IH - y) && (tiw1 >= -x) && (tiw1 < IW - x) && (n1 < N);
			bool wrt2 = (tih2 >= -y) && (tih2 < IH - y) && (tiw2 >= -x) && (tiw2 < IW - x) && (n2 < N);
			bool wrt3 = (tih3 >= -y) && (tih3 < IH - y) && (tiw3 >= -x) && (tiw3 < IW - x) && (n3 < N);
			if (wrt0) { *(float4*)(&deltaX[X_offset0]) = v0;  *(float4*)(&deltaX[X_offset0 + 4]) = v1; }
			if (wrt1) { *(float4*)(&deltaX[X_offset1]) = v2;  *(float4*)(&deltaX[X_offset1 + 4]) = v3; }
			if (wrt2) { *(float4*)(&deltaX[X_offset2]) = v4;  *(float4*)(&deltaX[X_offset2 + 4]) = v5; }
			if (wrt3) { *(float4*)(&deltaX[X_offset3]) = v6;  *(float4*)(&deltaX[X_offset3 + 4]) = v7; }

			bool wrt4 = (tih4 >= -y) && (tih4 < IH - y) && (tiw4 >= -x) && (tiw4 < IW - x) && (n4 < N);
			bool wrt5 = (tih5 >= -y) && (tih5 < IH - y) && (tiw5 >= -x) && (tiw5 < IW - x) && (n5 < N);
			bool wrt6 = (tih6 >= -y) && (tih6 < IH - y) && (tiw6 >= -x) && (tiw6 < IW - x) && (n6 < N);
			bool wrt7 = (tih7 >= -y) && (tih7 < IH - y) && (tiw7 >= -x) && (tiw7 < IW - x) && (n7 < N);
			if (wrt4) { *(float4*)(&deltaX[X_offset4]) = v8;  *(float4*)(&deltaX[X_offset4 + 4]) = v9; }
			if (wrt5) { *(float4*)(&deltaX[X_offset5]) = v10; *(float4*)(&deltaX[X_offset5 + 4]) = v11; }
			if (wrt6) { *(float4*)(&deltaX[X_offset6]) = v12; *(float4*)(&deltaX[X_offset6 + 4]) = v13; }
			if (wrt7) { *(float4*)(&deltaX[X_offset7]) = v14; *(float4*)(&deltaX[X_offset7 + 4]) = v15; }
		}
	}
}


#define kernelSplit_kv15(stream, LB, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, CFH, CFW, GN, GM) \
	kernelSplit_kernel_v15<LB, (1<<LB>>1)>\
		<<< dim3((GM + (8<<LB) - 1)>>LB>>3, GN>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, N,IC,OC, sh,sw,ph,pw, CFH, CFW)



//Size = 0.938477, Time = 1.39667 msec, Performace = 1442.98 GFlop/s
template<int LB, int STEP>
__global__ void kernelSplit_kernel_v15(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw,
	int CFH, int CFW)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4  Ws[2][1 << LB >> 1][(2 << LB) + 1];
	__shared__ float4 dYs[2][1 << LB >> 1][(2 << LB) + 1];

	const int ic0 = ((blockIdx.y << LB) + ty) << 3;
	const int tic0 = ((tx & 1) << 2) + ic0;

	const int j0 = ((blockIdx.x << LB) + tx) << 3;//[n, ohs, ows]
	const int j1 = j0 + 1, j2 = j0 + 2, j3 = j0 + 3;
	const int j4 = j0 + 4, j5 = j0 + 5, j6 = j0 + 6, j7 = j0 + 7;
	const int oph = --CFH, opw = --CFW;
	const int OHS = OH + CFH;//OHS = OH - CFH + (oph << 1) + 1
	const int OWS = OW + CFW;//OWS = OW - CFW + (opw << 1) + 1;
	const int OHS_OWS = OHS * OWS;
	get_n_ohs_ows(j0, n0, ohs0, ows0);
	get_n_ohs_ows(j1, n1, ohs1, ows1);
	get_n_ohs_ows(j2, n2, ohs2, ows2);
	get_n_ohs_ows(j3, n3, ohs3, ows3);
	get_n_ohs_ows(j4, n4, ohs4, ows4);
	get_n_ohs_ows(j5, n5, ohs5, ows5);
	get_n_ohs_ows(j6, n6, ohs6, ows6);
	get_n_ohs_ows(j7, n7, ohs7, ows7);
	const int tih0 = ohs0 * sh - ph; ohs0 -= oph;
	const int tih1 = ohs1 * sh - ph; ohs1 -= oph;
	const int tih2 = ohs2 * sh - ph; ohs2 -= oph;
	const int tih3 = ohs3 * sh - ph; ohs3 -= oph;
	const int tih4 = ohs4 * sh - ph; ohs4 -= oph;
	const int tih5 = ohs5 * sh - ph; ohs5 -= oph;
	const int tih6 = ohs6 * sh - ph; ohs6 -= oph;
	const int tih7 = ohs7 * sh - ph; ohs7 -= oph;
	const int tiw0 = ows0 * sw - pw; ows0 -= opw;
	const int tiw1 = ows1 * sw - pw; ows1 -= opw;
	const int tiw2 = ows2 * sw - pw; ows2 -= opw;
	const int tiw3 = ows3 * sw - pw; ows3 -= opw;
	const int tiw4 = ows4 * sw - pw; ows4 -= opw;
	const int tiw5 = ows5 * sw - pw; ows5 -= opw;
	const int tiw6 = ows6 * sw - pw; ows6 -= opw;
	const int tiw7 = ows7 * sw - pw; ows7 -= opw;
	int X_offset0 = ((n0*IH + tih0)*IW + tiw0)*IC + ic0;
	int X_offset1 = ((n1*IH + tih1)*IW + tiw1)*IC + ic0;
	int X_offset2 = ((n2*IH + tih2)*IW + tiw2)*IC + ic0;
	int X_offset3 = ((n3*IH + tih3)*IW + tiw3)*IC + ic0;
	int X_offset4 = ((n4*IH + tih4)*IW + tiw4)*IC + ic0;
	int X_offset5 = ((n5*IH + tih5)*IW + tiw5)*IC + ic0;
	int X_offset6 = ((n6*IH + tih6)*IW + tiw6)*IC + ic0;
	int X_offset7 = ((n7*IH + tih7)*IW + tiw7)*IC + ic0;

	bool flagY = (ty & 1);
	const int tn0 = (n4 - n0)*flagY + n0;
	const int tn1 = (n5 - n1)*flagY + n1;
	const int tn2 = (n6 - n2)*flagY + n2;
	const int tn3 = (n7 - n3)*flagY + n3;
	const int tohs0 = (ohs4 - ohs0)*flagY + ohs0;
	const int tohs1 = (ohs5 - ohs1)*flagY + ohs1;
	const int tohs2 = (ohs6 - ohs2)*flagY + ohs2;
	const int tohs3 = (ohs7 - ohs3)*flagY + ohs3;
	const int tows0 = (ows4 - ows0)*flagY + ows0;
	const int tows1 = (ows5 - ows1)*flagY + ows1;
	const int tows2 = (ows6 - ows2)*flagY + ows2;
	const int tows3 = (ows7 - ows3)*flagY + ows3;
	int Y_offset0 = ((tn0*OH + tohs0)*OW + tows0)*OC;
	int Y_offset1 = ((tn1*OH + tohs1)*OW + tows1)*OC;
	int Y_offset2 = ((tn2*OH + tohs2)*OW + tows2)*OC;
	int Y_offset3 = ((tn3*OH + tohs3)*OW + tows3)*OC;

	const int GK = FH * FW * IC, strideY = (IW - sw)*IC;
	const int Ws_x = (tx >> 1), Ws_y = (ty << 1) + (tx & 1);
	const int dYs_y = (ty >> 1), dYs_x = (tx << 1) + (ty & 1);

	for (int y = 0; y < sh; y++, deltaX += strideY) {
		for (int x = 0; x < sw; x++, deltaX += IC)
		{
			float4 v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
			float4 v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
			float4 v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
			float4 v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
			float4 v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
			float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
			float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
			float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);
			for (int fhr = 0; fhr <= CFH; fhr++) {
				for (int fwr = 0; fwr <= CFW; ++fwr)
				{
					int fh = y + (CFH - fhr)*sh, fw = x + (CFW - fwr)*sw;
					int W_offset = (fh*FW + fw)*IC + tic0;
					bool lw = (fh >= 0) && (fw >= 0) && (fh < FH) && (fw < FW);
					int W_oc = tx >> 1;
					Ws[buf][Ws_x][Ws_y] = lw ? *(float4*)(&W[W_oc*GK + W_offset]) : make_float4(0, 0, 0, 0);

					bool ldy0 = (tohs0 >= -fhr) && (tohs0 < OH - fhr) && (tows0 >= -fwr) && (tows0 < OW - fwr);
					bool ldy1 = (tohs1 >= -fhr) && (tohs1 < OH - fhr) && (tows1 >= -fwr) && (tows1 < OW - fwr);
					bool ldy2 = (tohs2 >= -fhr) && (tohs2 < OH - fhr) && (tows2 >= -fwr) && (tows2 < OW - fwr);
					bool ldy3 = (tohs3 >= -fhr) && (tohs3 < OH - fhr) && (tows3 >= -fwr) && (tows3 < OW - fwr);
					int Y_oc = ty >> 1;
					int Y_offset = (fhr*OW + fwr)*OC;
					deltaY += Y_offset + Y_oc;
					intptr_t zero = (intptr_t)(&_ZERO);
					intptr_t addr0 = ldy0 * ((intptr_t)(&deltaY[Y_offset0]) - zero) + zero;
					intptr_t addr1 = ldy1 * ((intptr_t)(&deltaY[Y_offset1]) - zero) + zero;
					intptr_t addr2 = ldy2 * ((intptr_t)(&deltaY[Y_offset2]) - zero) + zero;
					intptr_t addr3 = ldy3 * ((intptr_t)(&deltaY[Y_offset3]) - zero) + zero;
					deltaY -= Y_offset + Y_oc;
					dYs[buf][dYs_y][dYs_x].x = *(float*)addr0;
					dYs[buf][dYs_y][dYs_x].y = *(float*)addr1;
					dYs[buf][dYs_y][dYs_x].z = *(float*)addr2;
					dYs[buf][dYs_y][dYs_x].w = *(float*)addr3;
					__syncthreads();

					for (int ooc = 1, OOC = OC << 1 >> LB; ooc < OOC; ++ooc) {
#pragma unroll
						for (int ik = 0; ik < STEP; ik++)
						{
							float4 dy0 = dYs[buf][ik][(tx << 1)];
							float4 dy1 = dYs[buf][ik][(tx << 1) + 1];
							float4  w0 = Ws[buf][ik][(ty << 1)];
							float4  w1 = Ws[buf][ik][(ty << 1) + 1];

							simdMM4(v0, dy0.x, w0); simdMM4(v1, dy0.x, w1);
							simdMM4(v2, dy0.y, w0); simdMM4(v3, dy0.y, w1);
							simdMM4(v4, dy0.z, w0); simdMM4(v5, dy0.z, w1);
							simdMM4(v6, dy0.w, w0); simdMM4(v7, dy0.w, w1);
							simdMM4(v8, dy1.x, w0); simdMM4(v9, dy1.x, w1);
							simdMM4(v10, dy1.y, w0); simdMM4(v11, dy1.y, w1);
							simdMM4(v12, dy1.z, w0); simdMM4(v13, dy1.z, w1);
							simdMM4(v14, dy1.w, w0); simdMM4(v15, dy1.w, w1);
						}
						buf ^= 1;

						W_oc = ((ooc << LB) + tx) >> 1;
						Ws[buf][Ws_x][Ws_y] = lw ? *(float4*)(&W[W_oc*GK + W_offset]) : make_float4(0, 0, 0, 0);

						Y_oc = ((ooc << LB) + ty) >> 1;
						deltaY += Y_offset + Y_oc;
						intptr_t zero = (intptr_t)(&_ZERO);
						intptr_t addr0 = ldy0 * ((intptr_t)(&deltaY[Y_offset0]) - zero) + zero;
						intptr_t addr1 = ldy1 * ((intptr_t)(&deltaY[Y_offset1]) - zero) + zero;
						intptr_t addr2 = ldy2 * ((intptr_t)(&deltaY[Y_offset2]) - zero) + zero;
						intptr_t addr3 = ldy3 * ((intptr_t)(&deltaY[Y_offset3]) - zero) + zero;
						deltaY -= Y_offset + Y_oc;
						dYs[buf][dYs_y][dYs_x].x = *(float*)addr0;
						dYs[buf][dYs_y][dYs_x].y = *(float*)addr1;
						dYs[buf][dYs_y][dYs_x].z = *(float*)addr2;
						dYs[buf][dYs_y][dYs_x].w = *(float*)addr3;
						__syncthreads();
					}
#pragma unroll
					for (int ik = 0; ik < STEP; ik++)
					{
						float4 dy0 = dYs[buf][ik][(tx << 1)];
						float4 dy1 = dYs[buf][ik][(tx << 1) + 1];
						float4  w0 = Ws[buf][ik][(ty << 1)];
						float4  w1 = Ws[buf][ik][(ty << 1) + 1];

						simdMM4(v0, dy0.x, w0); simdMM4(v1, dy0.x, w1);
						simdMM4(v2, dy0.y, w0); simdMM4(v3, dy0.y, w1);
						simdMM4(v4, dy0.z, w0); simdMM4(v5, dy0.z, w1);
						simdMM4(v6, dy0.w, w0); simdMM4(v7, dy0.w, w1);
						simdMM4(v8, dy1.x, w0); simdMM4(v9, dy1.x, w1);
						simdMM4(v10, dy1.y, w0); simdMM4(v11, dy1.y, w1);
						simdMM4(v12, dy1.z, w0); simdMM4(v13, dy1.z, w1);
						simdMM4(v14, dy1.w, w0); simdMM4(v15, dy1.w, w1);
					}
					buf ^= 1;
				}
			}

			bool wrt0 = (tih0 >= -y) && (tih0 < IH - y) && (tiw0 >= -x) && (tiw0 < IW - x) && (n0 < N);
			bool wrt1 = (tih1 >= -y) && (tih1 < IH - y) && (tiw1 >= -x) && (tiw1 < IW - x) && (n1 < N);
			bool wrt2 = (tih2 >= -y) && (tih2 < IH - y) && (tiw2 >= -x) && (tiw2 < IW - x) && (n2 < N);
			bool wrt3 = (tih3 >= -y) && (tih3 < IH - y) && (tiw3 >= -x) && (tiw3 < IW - x) && (n3 < N);
			if (wrt0) { *(float4*)(&deltaX[X_offset0]) = v0;  *(float4*)(&deltaX[X_offset0 + 4]) = v1; }
			if (wrt1) { *(float4*)(&deltaX[X_offset1]) = v2;  *(float4*)(&deltaX[X_offset1 + 4]) = v3; }
			if (wrt2) { *(float4*)(&deltaX[X_offset2]) = v4;  *(float4*)(&deltaX[X_offset2 + 4]) = v5; }
			if (wrt3) { *(float4*)(&deltaX[X_offset3]) = v6;  *(float4*)(&deltaX[X_offset3 + 4]) = v7; }

			bool wrt4 = (tih4 >= -y) && (tih4 < IH - y) && (tiw4 >= -x) && (tiw4 < IW - x) && (n4 < N);
			bool wrt5 = (tih5 >= -y) && (tih5 < IH - y) && (tiw5 >= -x) && (tiw5 < IW - x) && (n5 < N);
			bool wrt6 = (tih6 >= -y) && (tih6 < IH - y) && (tiw6 >= -x) && (tiw6 < IW - x) && (n6 < N);
			bool wrt7 = (tih7 >= -y) && (tih7 < IH - y) && (tiw7 >= -x) && (tiw7 < IW - x) && (n7 < N);
			if (wrt4) { *(float4*)(&deltaX[X_offset4]) = v8;  *(float4*)(&deltaX[X_offset4 + 4]) = v9; }
			if (wrt5) { *(float4*)(&deltaX[X_offset5]) = v10; *(float4*)(&deltaX[X_offset5 + 4]) = v11; }
			if (wrt6) { *(float4*)(&deltaX[X_offset6]) = v12; *(float4*)(&deltaX[X_offset6 + 4]) = v13; }
			if (wrt7) { *(float4*)(&deltaX[X_offset7]) = v14; *(float4*)(&deltaX[X_offset7 + 4]) = v15; }
		}
	}
}


#endif