

#define X_IH_slice(IH, sh) ((IH + sh - 1)/sh)
#define X_IW_slice(IW, sw) ((IW + sw - 1)/sw)

#define X_GN(IC) (IC)
#define X_GM(N, IH_slice, IW_slice) (N*IH_slice*IW_slice)
#define X_GK(CFH, CFW, OC)  (CFH*CFW*OC)

#define init_X(N, IH, IW, FH, FW, sh, sw, ph, pw) \
	float Nh0 = KernelSplit_Nh0(FH, sh);\
	float Nw0 = KernelSplit_Nw0(FW, sw);\
	bool lh5 = KernelSplit_lh5(Nh0);\
	bool lw5 = KernelSplit_lw5(Nw0);\
	int CFH = KernelSplit_CFH(Nh0, lh5);\
	int CFW = KernelSplit_CFW(Nw0, lw5);\
	int IH_slice = X_IH_slice(IH, sh);\
	int IW_slice = X_IW_slice(IW, sw);\
	int GN = X_GN(IC); \
	int GM = X_GM(N, IH_slice, IW_slice);\
	int GK = X_GK(CFH, CFW, OC);

#define X_n_ih_iw(j, n, ih, iw) \
	int n, ih, iw; {\
	n = j / IH_IW_slice; int jr = j - n*IH_IW_slice;\
	ih = jr / IW_slice, iw = jr - ih*IW_slice;}

#define X_write8(deltaX, n, ih, iw, ic, v0, v1) \
	if ((ih < IH) && (iw < IW)) {\
		int Xoffset = ((n*IH + ih)*IW + iw)*IC + ic;\
		*(float4*)(deltaX + Xoffset) = v0; *(float4*)(deltaX + Xoffset + 4) = v1; }


#ifndef X_KERNEL_V1
#define X_KERNEL_V1

#define X_kv1(stream, LB, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM) \
	X_kernel_v1<LB>\
		<<< dim3(GM>>LB, GN>>LB, (sh*sw)), dim3(1<<LB, 1<<LB) >>>\
			(deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw,\
			 CFH, CFW, IH_slice, IW_slice, GK)

//LB = 4: Size = 0.5, Time = 31.46 msec, Performace = 34.1304 GFlop/s
template<int LB>
__global__ void X_kernel_v1(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw,
	int CFH, int CFW,
	int IH_slice, int IW_slice, int GK)
{
	//prepare for GZ = sh*sw
	int bz = blockIdx.z;
	int y = bz / sw, x = bz % sw;
	int ihs = (y - ph); if(ihs < 0) ihs += (ph - y + sh - 1) / sh * sh;
	int iws = (x - pw); if(iws < 0) iws += (pw - x + sw - 1) / sw * sw;

	//prepare for GN = IC
	int ic = (blockIdx.y << LB) + threadIdx.y;

	//prepare_for_GM = N*IH_slice*IW_slice
	int j = (blockIdx.x << LB) + threadIdx.x;
	int IH_IW_slice = IH_slice * IW_slice;
	int n = j / IH_IW_slice, jr = j % IH_IW_slice;
	int ih = jr / IW_slice, iw = jr % IW_slice;
	
	ih = ih * sh + ihs;
	iw = iw * sw + iws;

	const int oph = CFH - 1, opw = CFW - 1;
	int ohs = (ih + ph - y) / sh - oph;
	int ows = (iw + pw - x) / sw - opw;

	const int CFW_OC = CFW * OC;

	float dx = 0;
	for (int k = 0; k < GK; k++) 
	{
		int fhr = k / CFW_OC, kr = k % CFW_OC;
		int fwr = kr / OC, oc = kr % OC;

		int oh = ohs + fhr;
		int ow = ows + fwr;
		bool ldy = (oh >= 0) && (oh < OH) && (ow >= 0) && (ow < OW);
		float dy = (ldy ? get4d(deltaY, n, oh, ow, oc, OH, OW, OC) : 0);

		int fh = y + (CFH - 1 - fhr)*sh;
		int fw = x + (CFW - 1 - fwr)*sw;
		bool lw = (fh < FH) && (fw < FW);
		float w = (lw ? get4d(W, oc, fh, fw, ic, FH, FW, IC) : 0);
			
		dx += w * dy;
	}

	bool wrt = (ih < IH) && (iw < IW);
	if (wrt) get4d(deltaX, n, ih, iw, ic,IH, IW, IC) = dx;
}

#endif


#ifndef X_KERNEL_V2
#define X_KERNEL_V2

#define X_kv2(stream, LB, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM) \
	X_kernel_v2<LB>\
		<<< dim3(GM>>LB, GN>>LB, (sh*sw)), dim3(1<<LB, 1<<LB) >>>\
			(deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw,\
			 CFH, CFW, IH_slice, IW_slice, GK)

//LB = 4: Size = 0.5, Time = 31.46 msec, Performace = 34.1304 GFlop/s
template<int LB>
__global__ void X_kernel_v2(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw,
	int CFH, int CFW,
	int IH_slice, int IW_slice, int GK)
{
	//prepare for GZ = sh*sw
	int bz = blockIdx.z;
	int y = bz / sw, x = bz - y * sw;
	int ihs = (y - ph); ihs += (ihs < 0)*(ph - y + sh - 1) / sh * sh;
	int iws = (x - pw); iws += (iws < 0)*(pw - x + sw - 1) / sw * sw;

	//prepare for GN = IC
	int ic = (blockIdx.y << LB) + threadIdx.y;

	//prepare_for_GM = N*IH_slice*IW_slice
	int j = (blockIdx.x << LB) + threadIdx.x;
	int IH_IW_slice = IH_slice * IW_slice;
	int n = j / IH_IW_slice, jr = j - n * IH_IW_slice;
	int ih = jr / IW_slice, iw = jr - ih * IW_slice;

	ih = ih * sh + ihs;
	iw = iw * sw + iws;

	const int oph = CFH - 1, opw = CFW - 1;
	int ohs = (ih + ph - y) / sh - oph;
	int ows = (iw + pw - x) / sw - opw;

	const int CFW_OC = CFW * OC;

	float dx = 0;
	for (int k = 0; k < GK; k++)
	{
		int fhr = k / CFW_OC, kr = k - fhr * CFW_OC;
		int fwr = kr / OC, oc = kr - fwr * OC;

		int oh = ohs + fhr;
		int ow = ows + fwr;
		bool ldy = (oh >= 0) && (oh < OH) && (ow >= 0) && (ow < OW);
		float dy = (ldy ? get4d(deltaY, n, oh, ow, oc, OH, OW, OC) : 0);

		int fh = y + (CFH - 1 - fhr)*sh;
		int fw = x + (CFW - 1 - fwr)*sw;
		bool lw = (fh < FH) && (fw < FW);
		float w = (lw ? get4d(W, oc, fh, fw, ic, FH, FW, IC) : 0);

		dx += w * dy;
	}

	bool wrt = (ih < IH) && (iw < IW);
	if (wrt) get4d(deltaX, n, ih, iw, ic, IH, IW, IC) = dx;
}

#endif


#ifndef X_KERNEL_V3
#define X_KERNEL_V3

#define X_kv3(stream, LB, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM) \
	X_kernel_v3<LB, (1<<LB)>\
		<<< dim3(GM>>LB, GN>>LB, (sh*sw)), dim3(1<<LB, 1<<LB) >>>\
			(deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw,\
			 CFH, CFW, IH_slice, IW_slice, GK)

//LB = 4: Size = 0.5, Time = 5.58 msec, Performace = 192.427 GFlop/s
template<int LB, int STEP>
__global__ void X_kernel_v3(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw,
	int CFH, int CFW,
	int IH_slice, int IW_slice, int GK)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	__shared__ float  Ws[1 << LB][(1 << LB) + 1];
	__shared__ float dYs[1 << LB][(1 << LB) + 1];

	//prepare for GZ = sh*sw
	int bz = blockIdx.z;
	int y = bz / sw, x = bz - y * sw;
	int ihs = (y - ph); ihs += (ihs < 0)*(ph - y + sh - 1) / sh * sh;
	int iws = (x - pw); iws += (iws < 0)*(pw - x + sw - 1) / sw * sw;

	//prepare for GN = IC
	int ic = (blockIdx.y << LB) + ty;

	//prepare_for_GM = N*IH_slice*IW_slice
	int j = (blockIdx.x << LB) + tx;
	int IH_IW_slice = IH_slice * IW_slice;
	int n = j / IH_IW_slice, jr = j - n * IH_IW_slice;
	int ih = jr / IW_slice, iw = jr - ih * IW_slice;

	ih = ih * sh + ihs;
	iw = iw * sw + iws;

	const int oph = CFH - 1, opw = CFW - 1;
	int ohs = (ih + ph - y) / sh - oph;
	int ows = (iw + pw - x) / sw - opw;

	const int CFW_OC = CFW * OC;

	float dx = 0;
	for (int ok = 0, OK = (GK >> LB); ok < OK; ok++)
	{
		int W_k = (ok << LB) + tx;
		int W_fhr = W_k / CFW_OC, W_kr = W_k - W_fhr * CFW_OC;
		int W_fwr = W_kr / OC, W_oc = W_kr - W_fwr * OC;

		int fh = y + (CFH - 1 - W_fhr)*sh;
		int fw = x + (CFW - 1 - W_fwr)*sw;
		bool lw = (fh < FH) && (fw < FW);
		Ws[tx][ty] = (lw ? get4d(W, W_oc, fh, fw, ic, FH, FW, IC) : 0);

		int dY_k = (ok << LB) + ty;
		int dY_fhr = dY_k / CFW_OC, dY_kr = dY_k - dY_fhr * CFW_OC;
		int dY_fwr = dY_kr / OC, dY_oc = dY_kr - dY_fwr * OC;
		int oh = ohs + dY_fhr;
		int ow = ows + dY_fwr;
		bool ldy = (oh >= 0) && (oh < OH) && (ow >= 0) && (ow < OW);
		dYs[ty][tx] = (ldy ? get4d(deltaY, n, oh, ow, dY_oc, OH, OW, OC) : 0);
		__syncthreads();

#pragma once
		for (int ik = 0; ik < STEP; ik++)
		{
			float w  = Ws[ik][ty];
			float dy = dYs[ik][tx];

			dx += w * dy;
		}
		__syncthreads();
	}

	bool wrt = (ih < IH) && (iw < IW);
	if (wrt) get4d(deltaX, n, ih, iw, ic, IH, IW, IC) = dx;
}

#endif


#ifndef X_KERNEL_V4
#define X_KERNEL_V4

#define X_kv4(stream, LB, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM) \
	X_kernel_v4<LB, (1<<LB)>\
		<<< dim3(GM>>LB, GN>>LB, (sh*sw)), dim3(1<<LB, 1<<LB) >>>\
			(deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw,\
			 CFH, CFW, IH_slice, IW_slice, GK)

//LB = 4: Size = 0.5, Time = 4.446 msec, Performace = 241.507 GFlop/s
template<int LB, int STEP>
__global__ void X_kernel_v4(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw,
	int CFH, int CFW,
	int IH_slice, int IW_slice, int GK)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float  Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float dYs[2][1 << LB][(1 << LB) + 1];

	//prepare for GZ = sh*sw
	int bz = blockIdx.z;
	int y = bz / sw, x = bz - y * sw;
	int ihs = (y - ph); ihs += (ihs < 0)*(ph - y + sh - 1) / sh * sh;
	int iws = (x - pw); iws += (iws < 0)*(pw - x + sw - 1) / sw * sw;

	//prepare for GN = IC
	int ic = (blockIdx.y << LB) + ty;

	//prepare_for_GM = N*IH_slice*IW_slice
	int j = (blockIdx.x << LB) + tx;
	int IH_IW_slice = IH_slice * IW_slice;
	int n = j / IH_IW_slice, jr = j - n * IH_IW_slice;
	int ih = jr / IW_slice, iw = jr - ih * IW_slice;

	ih = ih * sh + ihs;
	iw = iw * sw + iws;

	const int oph = CFH - 1, opw = CFW - 1;
	int ohs = (ih + ph - y) / sh - oph;
	int ows = (iw + pw - x) / sw - opw;

	const int CFW_OC = CFW * OC;

	//load 1 elem from W[OC, FH, FW, IC]
	int W_k =  tx;
	int W_fhr = W_k / CFW_OC; W_k -= W_fhr * CFW_OC;
	int W_fwr = W_k / OC, W_oc = W_k - W_fwr * OC;
	int fh = y + (CFH - 1 - W_fhr)*sh;
	int fw = x + (CFW - 1 - W_fwr)*sw;
	bool lw = (fh < FH) && (fw < FW);
	Ws[buf][tx][ty] = (lw ? get4d(W, W_oc, fh, fw, ic, FH, FW, IC) : 0);

	//load 1 elem from deltaY[N, FH, FW, OC]
	int dY_k = ty;
	int dY_fhr = dY_k / CFW_OC; dY_k -= dY_fhr * CFW_OC;
	int dY_fwr = dY_k / OC, dY_oc = dY_k - dY_fwr * OC;
	int oh = ohs + dY_fhr;
	int ow = ows + dY_fwr;
	bool ldy = (oh >= 0) && (oh < OH) && (ow >= 0) && (ow < OW);
	dYs[buf][ty][tx] = (ldy ? get4d(deltaY, n, oh, ow, dY_oc, OH, OW, OC) : 0);
	__syncthreads();

	//compure_area----------------------------------------------
	float dx = 0;
	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)
	{
#pragma once
		for (int ik = 0; ik < STEP; ik++)
		{
			float w  =  Ws[buf][ik][ty];
			float dy = dYs[buf][ik][tx];

			dx += w * dy;
		}
		buf ^= 1;

		//load 1 elem from W[OC, FH, FW, IC]
		int W_k = (ok << LB) + tx;
		int W_fhr = W_k / CFW_OC; W_k -= W_fhr * CFW_OC;
		int W_fwr = W_k / OC, W_oc = W_k - W_fwr * OC;

		int fh = y + (CFH - 1 - W_fhr)*sh;
		int fw = x + (CFW - 1 - W_fwr)*sw;
		bool lw = (fh < FH) && (fw < FW);
		Ws[buf][tx][ty] = (lw ? get4d(W, W_oc, fh, fw, ic, FH, FW, IC) : 0);

		//load 1 elem from deltaY[N, FH, FW, OC]
		int dY_k = (ok << LB) + ty;
		int dY_fhr = dY_k / CFW_OC; dY_k -= dY_fhr * CFW_OC;
		int dY_fwr = dY_k / OC, dY_oc = dY_k - dY_fwr * OC;

		int oh = ohs + dY_fhr;
		int ow = ows + dY_fwr;
		bool ldy = (oh >= 0) && (oh < OH) && (ow >= 0) && (ow < OW);
		dYs[buf][ty][tx] = (ldy ? get4d(deltaY, n, oh, ow, dY_oc, OH, OW, OC) : 0);
		__syncthreads();
	}
#pragma once
	for (int ik = 0; ik < STEP; ik++)
	{
		float w  =  Ws[buf][ik][ty];
		float dy = dYs[buf][ik][tx];

		dx += w * dy;
	}

	bool wrt = (ih < IH) && (iw < IW);
	if (wrt) get4d(deltaX, n, ih, iw, ic, IH, IW, IC) = dx;
}

#endif


//(2 * 2)
#ifndef X_KERNEL_V5
#define X_KERNEL_V5

#define X_kv5(stream, LB, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM) \
	X_kernel_v5<LB, (1<<LB)>\
		<<< dim3((GM>>LB>>1), (GN>>LB>>1), (sh*sw)), dim3(1<<LB, 1<<LB) >>>\
			(deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw,\
			 CFH, CFW, IH_slice, IW_slice, GK)

//LB = 4: Size = 1, Time = 3.63 msec, Performace = 591.593 GFlop/s
template<int LB, int STEP>
__global__ void X_kernel_v5(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw,
	int CFH, int CFW,
	int IH_slice, int IW_slice, int GK)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2  Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float2 dYs[2][1 << LB][(1 << LB) + 1];

	//prepare for GZ = sh*sw
	int bz = blockIdx.z;
	int y = bz / sw, x = bz - y * sw;
	int ihs = (y - ph); ihs += (ihs < 0)*(ph - y + sh - 1) / sh * sh;
	int iws = (x - pw); iws += (iws < 0)*(pw - x + sw - 1) / sw * sw;

	//prepare for GN = IC
	int ic0 = ((blockIdx.y << LB) + ty) << 1;

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = ((blockIdx.x << LB) + tx) << 1, j1 = j0 + 1;
	int IH_IW_slice = IH_slice * IW_slice;
	X_n_ih_iw(j0, n0, ih0, iw0);
	X_n_ih_iw(j1, n1, ih1, iw1);
	ih0 = ih0 * sh + ihs, iw0 = iw0 * sw + iws;
	ih1 = ih1 * sh + ihs, iw1 = iw1 * sw + iws;

	const int oph = CFH - 1, opw = CFW - 1;
	int ohs0 = (ih0 + ph - y) / sh - oph, ows0 = (iw0 + pw - x) / sw - opw;
	int ohs1 = (ih1 + ph - y) / sh - oph, ows1 = (iw1 + pw - x) / sw - opw;

	const int CFW_OC = CFW * OC;

	//load 1 elem from W[OC, FH, FW, IC]
	int W_k = tx;
	int W_fhr = W_k / CFW_OC; W_k -= W_fhr * CFW_OC;
	int W_fwr = W_k / OC, W_oc = W_k - W_fwr * OC;

	int fh = y + (CFH - 1 - W_fhr)*sh;
	int fw = x + (CFW - 1 - W_fwr)*sw;
	bool lw = (fh < FH) && (fw < FW);
	Ws[buf][tx][ty] = (lw ? *(float2*)(&get4d(W, W_oc, fh, fw, ic0, FH, FW, IC)) : make_float2(0, 0));

	//load 1 elem from deltaY[N, FH, FW, OC]
	int dY_k = ty;
	int dY_fhr = dY_k / CFW_OC; dY_k -= dY_fhr * CFW_OC;
	int dY_fwr = dY_k / OC, dY_oc = dY_k - dY_fwr * OC;

	int oh0 = ohs0 + dY_fhr, ow0 = ows0 + dY_fwr;
	int oh1 = ohs1 + dY_fhr, ow1 = ows1 + dY_fwr;
	bool ldy0 = (oh0 >= 0) && (oh0 < OH) && (ow0 >= 0) && (ow0 < OW);
	bool ldy1 = (oh1 >= 0) && (oh1 < OH) && (ow1 >= 0) && (ow1 < OW);
	dYs[buf][ty][tx].x = (ldy0 ? get4d(deltaY, n0, oh0, ow0, dY_oc, OH, OW, OC) : 0);
	dYs[buf][ty][tx].y = (ldy1 ? get4d(deltaY, n1, oh1, ow1, dY_oc, OH, OW, OC) : 0);
	__syncthreads();

	//compure_area----------------------------------------------
	float2 v0 = make_float2(0, 0);
	float2 v1 = make_float2(0, 0);
	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)
	{
#pragma once
		for (int ik = 0; ik < STEP; ik++)
		{
			float2 w  =  Ws[buf][ik][ty];
			float2 dy = dYs[buf][ik][tx];

			simdMM2(v0, dy.x, w);
			simdMM2(v1, dy.y, w);
		}
		buf ^= 1;

		//load 1 elem from W[OC, FH, FW, IC]
		int W_k = (ok << LB) + tx;
		int W_fhr = W_k / CFW_OC; W_k -= W_fhr * CFW_OC;
		int W_fwr = W_k / OC, W_oc = W_k - W_fwr * OC;

		int fh = y + (CFH - 1 - W_fhr)*sh;
		int fw = x + (CFW - 1 - W_fwr)*sw;
		bool lw = (fh < FH) && (fw < FW);
		Ws[buf][tx][ty] = (lw ? *(float2*)(&get4d(W, W_oc, fh, fw, ic0, FH, FW, IC)) : make_float2(0, 0));

		//load 1 elem from deltaY[N, FH, FW, OC]
		int dY_k = (ok << LB) + ty;
		int dY_fhr = dY_k / CFW_OC; dY_k -= dY_fhr * CFW_OC;
		int dY_fwr = dY_k / OC, dY_oc = dY_k - dY_fwr * OC;

		int oh0 = ohs0 + dY_fhr, ow0 = ows0 + dY_fwr;
		int oh1 = ohs1 + dY_fhr, ow1 = ows1 + dY_fwr;
		bool ldy0 = (oh0 >= 0) && (oh0 < OH) && (ow0 >= 0) && (ow0 < OW);
		bool ldy1 = (oh1 >= 0) && (oh1 < OH) && (ow1 >= 0) && (ow1 < OW);
		dYs[buf][ty][tx].x = (ldy0 ? get4d(deltaY, n0, oh0, ow0, dY_oc, OH, OW, OC) : 0);
		dYs[buf][ty][tx].y = (ldy1 ? get4d(deltaY, n1, oh1, ow1, dY_oc, OH, OW, OC) : 0);
		__syncthreads();
	}
#pragma once
	for (int ik = 0; ik < STEP; ik++)
	{
		float2 w = Ws[buf][ik][ty];
		float2 dy = dYs[buf][ik][tx];

		simdMM2(v0, dy.x, w);
		simdMM2(v1, dy.y, w);
	}

	bool wrt0 = (ih0 < IH) && (iw0 < IW);
	bool wrt1 = (ih1 < IH) && (iw1 < IW);
	if (wrt0) *(float2*)(&get4d(deltaX, n0, ih0, iw0, ic0, IH, IW, IC)) = v0;
	if (wrt1) *(float2*)(&get4d(deltaX, n1, ih1, iw1, ic0, IH, IW, IC)) = v1;
}

#endif


//(2 * 2)
#ifndef X_KERNEL_V6
#define X_KERNEL_V6

#define X_kv6(stream, LB, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM) \
	X_kernel_v6<LB, (1<<LB)>\
		<<< dim3((GM>>LB>>1), (GN>>LB>>1), (sh*sw)), dim3(1<<LB, 1<<LB) >>>\
			(deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw,\
			 CFH, CFW, IH_slice, IW_slice, GK)

//LB = 4: Size = 1, Time = 3.47 msec, Performace = 618.871 GFlop/s
template<int LB, int STEP>
__global__ void X_kernel_v6(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw,
	int CFH, int CFW,
	int IH_slice, int IW_slice, int GK)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2  Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float2 dYs[2][1 << LB][(1 << LB) + 1];

	//prepare for GZ = sh*sw
	int bz = blockIdx.z;
	int y = bz / sw, x = bz - y * sw;
	int ihs = (y - ph); ihs += (ihs < 0)*(ph - y + sh - 1) / sh * sh;
	int iws = (x - pw); iws += (iws < 0)*(pw - x + sw - 1) / sw * sw;

	//prepare for GN = IC
	int ic0 = ((blockIdx.y << LB) + ty) << 1;

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = ((blockIdx.x << LB) + tx) << 1, j1 = j0 + 1;
	int IH_IW_slice = IH_slice * IW_slice;
	X_n_ih_iw(j0, n0, ih0, iw0);
	X_n_ih_iw(j1, n1, ih1, iw1);
	ih0 = ih0 * sh + ihs, iw0 = iw0 * sw + iws;
	ih1 = ih1 * sh + ihs, iw1 = iw1 * sw + iws;

	const int oph = CFH - 1, opw = CFW - 1;
	int ohs0 = (ih0 + ph - y) / sh - oph, ows0 = (iw0 + pw - x) / sw - opw;
	int ohs1 = (ih1 + ph - y) / sh - oph, ows1 = (iw1 + pw - x) / sw - opw;

	const int CFW_OC = CFW * OC;

	//load 1 elem from W[OC, FH, FW, IC]
	int W_k = tx;
	int W_fhr = W_k / CFW_OC; W_k -= W_fhr * CFW_OC;
	int W_fwr = W_k / OC, W_oc = W_k - W_fwr * OC;

	int fh = y + (CFH - 1 - W_fhr)*sh;
	int fw = x + (CFW - 1 - W_fwr)*sw;
	bool lw = (fh < FH) && (fw < FW);
	int Woffset = ((W_oc*FH + fh)*FW + fw)*IC + ic0;
	Ws[buf][tx][ty] = (lw ? *(float2*)(W + Woffset) : make_float2(0, 0));

	//load 1 elem from deltaY[N, FH, FW, OC]
	int dY_k = ty;
	int dY_fhr = dY_k / CFW_OC; dY_k -= dY_fhr * CFW_OC;
	int dY_fwr = dY_k / OC, dY_oc = dY_k - dY_fwr * OC;

	int oh0 = ohs0 + dY_fhr, ow0 = ows0 + dY_fwr;
	int oh1 = ohs1 + dY_fhr, ow1 = ows1 + dY_fwr;
	bool ldy0 = (oh0 >= 0) && (oh0 < OH) && (ow0 >= 0) && (ow0 < OW);
	bool ldy1 = (oh1 >= 0) && (oh1 < OH) && (ow1 >= 0) && (ow1 < OW);
	int Yoffset0 = ((n0*OH + oh0)*OW + ow0)*OC;
	int Yoffset1 = ((n1*OH + oh1)*OW + ow1)*OC;
	dYs[buf][ty][tx].x = (ldy0 ? deltaY[Yoffset0 + dY_oc] : 0);
	dYs[buf][ty][tx].y = (ldy1 ? deltaY[Yoffset1 + dY_oc] : 0);
	__syncthreads();

	//compure_area----------------------------------------------
	float2 v0 = make_float2(0, 0);
	float2 v1 = make_float2(0, 0);
	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)
	{
#pragma once
		for (int ik = 0; ik < STEP; ik++)
		{
			float2 dy = dYs[buf][ik][tx];
			float2 w = Ws[buf][ik][ty];

			simdMM2(v0, dy.x, w);
			simdMM2(v1, dy.y, w);
		}
		buf ^= 1;

		//load 1 elem from W[OC, FH, FW, IC]
		int W_k = (ok << LB) + tx;
		int W_fhr = W_k / CFW_OC; W_k -= W_fhr * CFW_OC;
		int W_fwr = W_k / OC, W_oc = W_k - W_fwr * OC;

		int fh = y + (CFH - 1 - W_fhr)*sh;
		int fw = x + (CFW - 1 - W_fwr)*sw;
		bool lw = (fh < FH) && (fw < FW);
		int Woffset = ((W_oc*FH + fh)*FW + fw)*IC + ic0;
		Ws[buf][tx][ty] = (lw ? *(float2*)(W + Woffset) : make_float2(0, 0));

		//load 1 elem from deltaY[N, FH, FW, OC]
		int dY_k = (ok << LB) + ty;
		int dY_fhr = dY_k / CFW_OC; dY_k -= dY_fhr * CFW_OC;
		int dY_fwr = dY_k / OC, dY_oc = dY_k - dY_fwr * OC;

		int oh0 = ohs0 + dY_fhr, ow0 = ows0 + dY_fwr;
		int oh1 = ohs1 + dY_fhr, ow1 = ows1 + dY_fwr;
		bool ldy0 = (oh0 >= 0) && (oh0 < OH) && (ow0 >= 0) && (ow0 < OW);
		bool ldy1 = (oh1 >= 0) && (oh1 < OH) && (ow1 >= 0) && (ow1 < OW);
		int Yoffset0 = ((n0*OH + oh0)*OW + ow0)*OC;
		int Yoffset1 = ((n1*OH + oh1)*OW + ow1)*OC;
		dYs[buf][ty][tx].x = (ldy0 ? deltaY[Yoffset0 + dY_oc] : 0);
		dYs[buf][ty][tx].y = (ldy1 ? deltaY[Yoffset1 + dY_oc] : 0);
		__syncthreads();
	}
#pragma once
	for (int ik = 0; ik < STEP; ik++)
	{
		float2 dy = dYs[buf][ik][tx];
		float2 w = Ws[buf][ik][ty];

		simdMM2(v0, dy.x, w);
		simdMM2(v1, dy.y, w);
	}

	bool wrt0 = (ih0 < IH) && (iw0 < IW);
	bool wrt1 = (ih1 < IH) && (iw1 < IW);
	if (wrt0) *(float2*)(&get4d(deltaX, n0, ih0, iw0, ic0, IH, IW, IC)) = v0;
	if (wrt1) *(float2*)(&get4d(deltaX, n1, ih1, iw1, ic0, IH, IW, IC)) = v1;
}

#endif


//(4 * 4)
#ifndef X_KERNEL_V7
#define X_KERNEL_V7

#define X_kv7(stream, LB, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM) \
	X_kernel_v7<LB, (1<<LB>>1)>\
		<<< dim3((GM>>LB>>2), (GN>>LB>>2), (sh*sw)), dim3(1<<LB, 1<<LB) >>>\
			(deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw,\
			 CFH, CFW, IH_slice, IW_slice, GK)

//LB = 4: Size = 1, Time = 2.12333 msec, Performace = 1011.37 GFlop/s
template<int LB, int STEP>
__global__ void X_kernel_v7(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw,
	int CFH, int CFW,
	int IH_slice, int IW_slice, int GK)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2  Ws[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2 dYs[2][1 << LB >> 1][(2 << LB) + 2];

	//prepare for GZ = sh*sw
	int bz = blockIdx.z;
	int y = bz / sw, x = bz - y * sw;
	int ihs = (y - ph); ihs += (ihs < 0)*(ph - y + sh - 1) / sh * sh;
	int iws = (x - pw); iws += (iws < 0)*(pw - x + sw - 1) / sw * sw;

	//prepare for GN = IC
	int ic0 = ((blockIdx.y << LB) + ty) << 2;
	const int tic0 = ((tx & 1) << 1) + ic0;

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = ((blockIdx.x << LB) + tx) << 2;
	int j1 = j0 + 1, j2 = j0 + 2, j3 = j0 + 3;
	int IH_IW_slice = IH_slice * IW_slice;
	X_n_ih_iw(j0, n0, ih0, iw0);
	X_n_ih_iw(j1, n1, ih1, iw1);
	X_n_ih_iw(j2, n2, ih2, iw2);
	X_n_ih_iw(j3, n3, ih3, iw3);
	ih0 = ih0 * sh + ihs, iw0 = iw0 * sw + iws;
	ih1 = ih1 * sh + ihs, iw1 = iw1 * sw + iws;
	ih2 = ih2 * sh + ihs, iw2 = iw2 * sw + iws;
	ih3 = ih3 * sh + ihs, iw3 = iw3 * sw + iws;

	const int oph = CFH - 1, opw = CFW - 1;
	int ohs0 = (ih0 + ph - y) / sh - oph, ows0 = (iw0 + pw - x) / sw - opw;
	int ohs1 = (ih1 + ph - y) / sh - oph, ows1 = (iw1 + pw - x) / sw - opw;
	int ohs2 = (ih2 + ph - y) / sh - oph, ows2 = (iw2 + pw - x) / sw - opw;
	int ohs3 = (ih3 + ph - y) / sh - oph, ows3 = (iw3 + pw - x) / sw - opw;
	bool flagY = (ty & 1);
	const int tn0 = (n2 - n0)*flagY + n0;
	const int tn1 = (n3 - n1)*flagY + n1;
	const int tohs0 = (ohs2 - ohs0)*flagY + ohs0;
	const int tohs1 = (ohs3 - ohs1)*flagY + ohs1;
	const int tows0 = (ows2 - ows0)*flagY + ows0;
	const int tows1 = (ows3 - ows1)*flagY + ows1;

	const int CFW_OC = CFW * OC;

	//load 2 elem from W[OC, FH, FW, IC]
	int W_k = tx >> 1;
	int W_fhr = W_k / CFW_OC; W_k -= W_fhr * CFW_OC;
	int W_fwr = W_k / OC, W_oc = W_k - W_fwr * OC;

	int fh = y + (CFH - 1 - W_fhr)*sh;
	int fw = x + (CFW - 1 - W_fwr)*sw;
	bool lw = (fh < FH) && (fw < FW);
	int Woffset = ((W_oc*FH + fh)*FW + fw)*IC + tic0;
	const int Ws_x = (tx >> 1), Ws_y = (ty << 1) + (tx & 1);
	Ws[buf][Ws_x][Ws_y] = (lw ? *(float2*)(W + Woffset) : make_float2(0, 0));

	//load 2 elem from deltaY[N, FH, FW, OC]
	int dY_k = ty >> 1;
	int dY_fhr = dY_k / CFW_OC; dY_k -= dY_fhr * CFW_OC;
	int dY_fwr = dY_k / OC, dY_oc = dY_k - dY_fwr * OC;

	int oh0 = tohs0 + dY_fhr, ow0 = tows0 + dY_fwr;
	int oh1 = tohs1 + dY_fhr, ow1 = tows1 + dY_fwr;
	bool ldy0 = (oh0 >= 0) && (oh0 < OH) && (ow0 >= 0) && (ow0 < OW);
	bool ldy1 = (oh1 >= 0) && (oh1 < OH) && (ow1 >= 0) && (ow1 < OW);
	int Yoffset0 = ((tn0*OH + oh0)*OW + ow0)*OC;
	int Yoffset1 = ((tn1*OH + oh1)*OW + ow1)*OC;
	const int dYs_y = (ty >> 1), dYs_x = (tx << 1) + (ty & 1);
	dYs[buf][dYs_y][dYs_x].x = (ldy0 ? deltaY[Yoffset0 + dY_oc] : 0);
	dYs[buf][dYs_y][dYs_x].y = (ldy1 ? deltaY[Yoffset1 + dY_oc] : 0);
	__syncthreads();

	//compure_area----------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0);
	float4 v1 = make_float4(0, 0, 0, 0);
	float4 v2 = make_float4(0, 0, 0, 0);
	float4 v3 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)
	{
#pragma once
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 dy = *(float4*)(&dYs[buf][ik][tx << 1]);
			float4 w  = *(float4*)(&Ws[buf][ik][ty << 1]);

			simdMM4(v0, dy.x, w);
			simdMM4(v1, dy.y, w);
			simdMM4(v2, dy.z, w);
			simdMM4(v3, dy.w, w);
		}
		buf ^= 1;

		//load 1 elem from W[OC, FH, FW, IC]
		int W_k = ((ok << LB) + tx) >> 1;
		int W_fhr = W_k / CFW_OC; W_k -= W_fhr * CFW_OC;
		int W_fwr = W_k / OC, W_oc = W_k - W_fwr * OC;

		int fh = y + (CFH - 1 - W_fhr)*sh;
		int fw = x + (CFW - 1 - W_fwr)*sw;
		bool lw = (fh < FH) && (fw < FW);
		int Woffset = ((W_oc*FH + fh)*FW + fw)*IC + tic0;
		Ws[buf][Ws_x][Ws_y] = (lw ? *(float2*)(W + Woffset) : make_float2(0, 0));

		//load 1 elem from deltaY[N, FH, FW, OC]
		int dY_k = ((ok << LB) + ty) >> 1;
		int dY_fhr = dY_k / CFW_OC; dY_k -= dY_fhr * CFW_OC;
		int dY_fwr = dY_k / OC, dY_oc = dY_k - dY_fwr * OC;

		int oh0 = tohs0 + dY_fhr, ow0 = tows0 + dY_fwr;
		int oh1 = tohs1 + dY_fhr, ow1 = tows1 + dY_fwr;
		bool ldy0 = (oh0 >= 0) && (oh0 < OH) && (ow0 >= 0) && (ow0 < OW);
		bool ldy1 = (oh1 >= 0) && (oh1 < OH) && (ow1 >= 0) && (ow1 < OW);
		int Yoffset0 = ((tn0*OH + oh0)*OW + ow0)*OC;
		int Yoffset1 = ((tn1*OH + oh1)*OW + ow1)*OC;
		dYs[buf][dYs_y][dYs_x].x = (ldy0 ? deltaY[Yoffset0 + dY_oc] : 0);
		dYs[buf][dYs_y][dYs_x].y = (ldy1 ? deltaY[Yoffset1 + dY_oc] : 0);
		__syncthreads();
	}
#pragma once
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 dy = *(float4*)(&dYs[buf][ik][tx << 1]);
		float4  w = *(float4*)(&Ws[buf][ik][ty << 1]);

		simdMM4(v0, dy.x, w);
		simdMM4(v1, dy.y, w);
		simdMM4(v2, dy.z, w);
		simdMM4(v3, dy.w, w);
	}

	bool wrt0 = (ih0 < IH) && (iw0 < IW);
	bool wrt1 = (ih1 < IH) && (iw1 < IW);
	bool wrt2 = (ih2 < IH) && (iw2 < IW);
	bool wrt3 = (ih3 < IH) && (iw3 < IW);
	if (wrt0) *(float4*)(&get4d(deltaX, n0, ih0, iw0, ic0, IH, IW, IC)) = v0;
	if (wrt1) *(float4*)(&get4d(deltaX, n1, ih1, iw1, ic0, IH, IW, IC)) = v1;
	if (wrt2) *(float4*)(&get4d(deltaX, n2, ih2, iw2, ic0, IH, IW, IC)) = v2;
	if (wrt3) *(float4*)(&get4d(deltaX, n3, ih3, iw3, ic0, IH, IW, IC)) = v3;
}

#endif


//(4 * 4)
#ifndef X_KERNEL_V8
#define X_KERNEL_V8

#define X_kv8(stream, LB, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM) \
	X_kernel_v8<LB, (1<<LB>>1)>\
		<<< dim3((GM>>LB>>2), (GN>>LB>>2), (sh*sw)), dim3(1<<LB, 1<<LB) >>>\
			(deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw,\
			 CFH, CFW, IH_slice, IW_slice, GK)

//LB = 4: Size = 1, Time = 2.09333 msec, Performace = 1025.87 GFlop/s
template<int LB, int STEP>
__global__ void X_kernel_v8(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw,
	int CFH, int CFW,
	int IH_slice, int IW_slice, int GK)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2  Ws[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2 dYs[2][1 << LB >> 1][(2 << LB) + 2];

	//prepare for GZ = sh*sw
	int bz = blockIdx.z;
	int y = bz / sw, x = bz - y * sw;
	int ihs = (y - ph); ihs += (ihs < 0)*(ph - y + sh - 1) / sh * sh;
	int iws = (x - pw); iws += (iws < 0)*(pw - x + sw - 1) / sw * sw;

	//prepare for GN = IC
	int ic0 = ((blockIdx.y << LB) + ty) << 2;
	const int tic0 = ((tx & 1) << 1) + ic0;

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = ((blockIdx.x << LB) + tx) << 2;
	int j1 = j0 + 1, j2 = j0 + 2, j3 = j0 + 3;
	int IH_IW_slice = IH_slice * IW_slice;
	X_n_ih_iw(j0, n0, ih0, iw0);
	X_n_ih_iw(j1, n1, ih1, iw1);
	X_n_ih_iw(j2, n2, ih2, iw2);
	X_n_ih_iw(j3, n3, ih3, iw3);
	ih0 = ih0 * sh + ihs, iw0 = iw0 * sw + iws;
	ih1 = ih1 * sh + ihs, iw1 = iw1 * sw + iws;
	ih2 = ih2 * sh + ihs, iw2 = iw2 * sw + iws;
	ih3 = ih3 * sh + ihs, iw3 = iw3 * sw + iws;

	bool flagY = (ty & 1);
	const int oph = CFH - 1, opw = CFW - 1;
	const int tohs0 = (((ih2 - ih0)*flagY + ih0) + ph - y) / sh - oph;
	const int tohs1 = (((ih3 - ih1)*flagY + ih1) + ph - y) / sh - oph;
	const int tows0 = (((iw2 - iw0)*flagY + iw0) + pw - x) / sw - opw;
	const int tows1 = (((iw3 - iw1)*flagY + iw1) + pw - x) / sw - opw;
	int Yoffset0 = ((((n2 - n0)*flagY + n0)*OH + tohs0)*OW + tows0)*OC;
	int Yoffset1 = ((((n3 - n1)*flagY + n1)*OH + tohs1)*OW + tows1)*OC;

	const int CFW_OC = CFW * OC;

	//load 2 elem from W[OC, FH, FW, IC]
	int W_k = tx >> 1;
	int W_fhr = W_k / CFW_OC; W_k -= W_fhr * CFW_OC;
	int W_fwr = W_k / OC, W_oc = W_k - W_fwr * OC;

	int fh = y + (CFH - 1 - W_fhr)*sh;
	int fw = x + (CFW - 1 - W_fwr)*sw;
	bool lw = (fh < FH) && (fw < FW);
	int Woffset = ((W_oc*FH + fh)*FW + fw)*IC + tic0;
	const int Ws_x = (tx >> 1), Ws_y = (ty << 1) + (tx & 1);
	Ws[buf][Ws_x][Ws_y] = (lw ? *(float2*)(W + Woffset) : make_float2(0, 0));

	//load 2 elem from deltaY[N, FH, FW, OC]
	int dY_k = ty >> 1;
	int dY_fhr = dY_k / CFW_OC; dY_k -= dY_fhr * CFW_OC;
	int dY_fwr = dY_k / OC; 

	bool ldy0 = (tohs0 >= -dY_fhr) && (tohs0 < OH - dY_fhr) && (tows0 >= -dY_fwr) && (tows0 < OW - dY_fwr);
	bool ldy1 = (tohs1 >= -dY_fhr) && (tohs1 < OH - dY_fhr) && (tows1 >= -dY_fwr) && (tows1 < OW - dY_fwr);
	const int OW_OC = OW * OC, yoffset = dY_fhr * OW_OC + dY_k;
	const int dYs_y = (ty >> 1), dYs_x = (tx << 1) + (ty & 1);
	dYs[buf][dYs_y][dYs_x].x = (ldy0 ? deltaY[Yoffset0 + yoffset] : 0);
	dYs[buf][dYs_y][dYs_x].y = (ldy1 ? deltaY[Yoffset1 + yoffset] : 0);
	__syncthreads();

	//compure_area----------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0);
	float4 v1 = make_float4(0, 0, 0, 0);
	float4 v2 = make_float4(0, 0, 0, 0);
	float4 v3 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)
	{
#pragma once
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 dy = *(float4*)(&dYs[buf][ik][tx << 1]);
			float4 w = *(float4*)(&Ws[buf][ik][ty << 1]);

			simdMM4(v0, dy.x, w);
			simdMM4(v1, dy.y, w);
			simdMM4(v2, dy.z, w);
			simdMM4(v3, dy.w, w);
		}
		buf ^= 1;

		//load 1 elem from W[OC, FH, FW, IC]
		int W_k = ((ok << LB) + tx) >> 1;
		int W_fhr = W_k / CFW_OC; W_k -= W_fhr * CFW_OC;
		int W_fwr = W_k / OC, W_oc = W_k - W_fwr * OC;

		int fh = y + (CFH - 1 - W_fhr)*sh;
		int fw = x + (CFW - 1 - W_fwr)*sw;
		bool lw = (fh < FH) && (fw < FW);
		int Woffset = ((W_oc*FH + fh)*FW + fw)*IC + tic0;
		Ws[buf][Ws_x][Ws_y] = (lw ? *(float2*)(W + Woffset) : make_float2(0, 0));

		//load 1 elem from deltaY[N, FH, FW, OC]
		int dY_k = ((ok << LB) + ty) >> 1;
		int dY_fhr = dY_k / CFW_OC; dY_k -= dY_fhr * CFW_OC;
		int dY_fwr = dY_k / OC;

		bool ldy0 = (tohs0 >= -dY_fhr) && (tohs0 < OH - dY_fhr) && (tows0 >= -dY_fwr) && (tows0 < OW - dY_fwr);
		bool ldy1 = (tohs1 >= -dY_fhr) && (tohs1 < OH - dY_fhr) && (tows1 >= -dY_fwr) && (tows1 < OW - dY_fwr);
		const int yoffset = dY_fhr * OW_OC + dY_k;
		dYs[buf][dYs_y][dYs_x].x = (ldy0 ? deltaY[Yoffset0 + yoffset] : 0);
		dYs[buf][dYs_y][dYs_x].y = (ldy1 ? deltaY[Yoffset1 + yoffset] : 0);
		__syncthreads();
	}
#pragma once
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 dy = *(float4*)(&dYs[buf][ik][tx << 1]);
		float4  w = *(float4*)(&Ws[buf][ik][ty << 1]);

		simdMM4(v0, dy.x, w);
		simdMM4(v1, dy.y, w);
		simdMM4(v2, dy.z, w);
		simdMM4(v3, dy.w, w);
	}

	bool wrt0 = (ih0 < IH) && (iw0 < IW);
	bool wrt1 = (ih1 < IH) && (iw1 < IW);
	bool wrt2 = (ih2 < IH) && (iw2 < IW);
	bool wrt3 = (ih3 < IH) && (iw3 < IW);
	if (wrt0) *(float4*)(&get4d(deltaX, n0, ih0, iw0, ic0, IH, IW, IC)) = v0;
	if (wrt1) *(float4*)(&get4d(deltaX, n1, ih1, iw1, ic0, IH, IW, IC)) = v1;
	if (wrt2) *(float4*)(&get4d(deltaX, n2, ih2, iw2, ic0, IH, IW, IC)) = v2;
	if (wrt3) *(float4*)(&get4d(deltaX, n3, ih3, iw3, ic0, IH, IW, IC)) = v3;
}

#endif


//(8 * 8)
#ifndef X_KERNEL_V9
#define X_KERNEL_V9

#define X_kv9(stream, LB, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM) \
	X_kernel_v9<LB, (1<<LB>>1)>\
		<<< dim3((GM>>LB>>3), (GN>>LB>>3), (sh*sw)), dim3(1<<LB, 1<<LB) >>>\
			(deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw,\
			 CFH, CFW, IH_slice, IW_slice, GK)

//LB = 4: Size = 1, Time = 1.918 msec, Performace = 1119.65 GFlop/s
template<int LB, int STEP>
__global__ void X_kernel_v9(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw,
	int CFH, int CFW,
	int IH_slice, int IW_slice, int GK)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4  Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 dYs[2][1 << LB][(1 << LB) + 1];

	//prepare for GZ = sh*sw
	int bz = blockIdx.z;
	int y = bz / sw, x = bz - y * sw;
	int ihs = (y - ph); ihs += (ihs < 0)*(ph - y + sh - 1) / sh * sh;
	int iws = (x - pw); iws += (iws < 0)*(pw - x + sw - 1) / sw * sw;

	//prepare for GN = IC
	int ic0 = ((blockIdx.y << LB) + ty) << 3;
	const int tic0 = ic0 + ((tx >= STEP) << 2);

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = ((blockIdx.x << LB) + tx) << 3;
	int j1 = j0 + 1, j2 = j0 + 2, j3 = j0 + 3;
	int j4 = j0 + 4, j5 = j0 + 5, j6 = j0 + 6, j7 = j0 + 7;
	int IH_IW_slice = IH_slice * IW_slice;
	X_n_ih_iw(j0, n0, ih0, iw0);
	X_n_ih_iw(j1, n1, ih1, iw1);
	X_n_ih_iw(j2, n2, ih2, iw2);
	X_n_ih_iw(j3, n3, ih3, iw3);
	X_n_ih_iw(j4, n4, ih4, iw4);
	X_n_ih_iw(j5, n5, ih5, iw5);
	X_n_ih_iw(j6, n6, ih6, iw6);
	X_n_ih_iw(j7, n7, ih7, iw7);
	ih0 = ih0 * sh + ihs, iw0 = iw0 * sw + iws;
	ih1 = ih1 * sh + ihs, iw1 = iw1 * sw + iws;
	ih2 = ih2 * sh + ihs, iw2 = iw2 * sw + iws;
	ih3 = ih3 * sh + ihs, iw3 = iw3 * sw + iws;
	ih4 = ih4 * sh + ihs, iw4 = iw4 * sw + iws;
	ih5 = ih5 * sh + ihs, iw5 = iw5 * sw + iws;
	ih6 = ih6 * sh + ihs, iw6 = iw6 * sw + iws;
	ih7 = ih7 * sh + ihs, iw7 = iw7 * sw + iws;

	bool flagY = (ty >= STEP);
	const int oph = CFH - 1, opw = CFW - 1;
	const int tohs0 = (((ih4 - ih0)*flagY + ih0) + ph - y) / sh - oph;
	const int tohs1 = (((ih5 - ih1)*flagY + ih1) + ph - y) / sh - oph;
	const int tohs2 = (((ih6 - ih2)*flagY + ih2) + ph - y) / sh - oph;
	const int tohs3 = (((ih7 - ih3)*flagY + ih3) + ph - y) / sh - oph;
	const int tows0 = (((iw4 - iw0)*flagY + iw0) + pw - x) / sw - opw;
	const int tows1 = (((iw5 - iw1)*flagY + iw1) + pw - x) / sw - opw;
	const int tows2 = (((iw6 - iw2)*flagY + iw2) + pw - x) / sw - opw;
	const int tows3 = (((iw7 - iw3)*flagY + iw3) + pw - x) / sw - opw;
	int Yoffset0 = ((((n4 - n0)*flagY + n0)*OH + tohs0)*OW + tows0)*OC;
	int Yoffset1 = ((((n5 - n1)*flagY + n1)*OH + tohs1)*OW + tows1)*OC;
	int Yoffset2 = ((((n6 - n2)*flagY + n2)*OH + tohs2)*OW + tows2)*OC;
	int Yoffset3 = ((((n7 - n3)*flagY + n3)*OH + tohs3)*OW + tows3)*OC;

	const int CFW_OC = CFW * OC;

	//load 2 elem from W[OC, FH, FW, IC]
	int W_k = tx - ((tx >= STEP) << LB >> 1);
	int W_fhr = W_k / CFW_OC; W_k -= W_fhr * CFW_OC;
	int W_fwr = W_k / OC, W_oc = W_k - W_fwr * OC;

	int fh = y + (CFH - 1 - W_fhr)*sh;
	int fw = x + (CFW - 1 - W_fwr)*sw;
	bool lw = (fh < FH) && (fw < FW);
	int Woffset = ((W_oc*FH + fh)*FW + fw)*IC + tic0;
	Ws[buf][tx][ty] = (lw ? *(float4*)(W + Woffset) : FLOAT_ZERO4);

	//load 2 elem from deltaY[N, FH, FW, OC]
	int dY_k = ty - ((ty >= STEP) << LB >> 1);
	int dY_fhr = dY_k / CFW_OC; dY_k -= dY_fhr * CFW_OC;
	int dY_fwr = dY_k / OC;

	bool ldy0 = (tohs0 >= -dY_fhr) && (tohs0 < OH - dY_fhr) && (tows0 >= -dY_fwr) && (tows0 < OW - dY_fwr);
	bool ldy1 = (tohs1 >= -dY_fhr) && (tohs1 < OH - dY_fhr) && (tows1 >= -dY_fwr) && (tows1 < OW - dY_fwr);
	bool ldy2 = (tohs2 >= -dY_fhr) && (tohs2 < OH - dY_fhr) && (tows2 >= -dY_fwr) && (tows2 < OW - dY_fwr);
	bool ldy3 = (tohs3 >= -dY_fhr) && (tohs3 < OH - dY_fhr) && (tows3 >= -dY_fwr) && (tows3 < OW - dY_fwr);
	const int OW_OC = OW * OC, yoffset = dY_fhr * OW_OC + dY_k;
	dYs[buf][ty][tx].x = (ldy0 ? deltaY[Yoffset0 + yoffset] : 0);
	dYs[buf][ty][tx].y = (ldy1 ? deltaY[Yoffset1 + yoffset] : 0);
	dYs[buf][ty][tx].z = (ldy2 ? deltaY[Yoffset2 + yoffset] : 0);
	dYs[buf][ty][tx].w = (ldy3 ? deltaY[Yoffset3 + yoffset] : 0);
	__syncthreads();

	//compure_area----------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)
	{
#pragma once
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 dy0 = dYs[buf][ik][tx], dy1 = dYs[buf][ik + STEP][tx];
			float4  w0 =  Ws[buf][ik][ty],  w1 =  Ws[buf][ik + STEP][ty];

			simdMM4( v0, dy0.x, w0); simdMM4( v1, dy0.x, w1);
			simdMM4( v2, dy0.y, w0); simdMM4( v3, dy0.y, w1);
			simdMM4( v4, dy0.z, w0); simdMM4( v5, dy0.z, w1);
			simdMM4( v6, dy0.w, w0); simdMM4( v7, dy0.w, w1);
			simdMM4( v8, dy1.x, w0); simdMM4( v9, dy1.x, w1);
			simdMM4(v10, dy1.y, w0); simdMM4(v11, dy1.y, w1);
			simdMM4(v12, dy1.z, w0); simdMM4(v13, dy1.z, w1);
			simdMM4(v14, dy1.w, w0); simdMM4(v15, dy1.w, w1);
		}
		buf ^= 1;

		//load 1 elem from W[OC, FH, FW, IC]
		int W_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int W_fhr = W_k / CFW_OC; W_k -= W_fhr * CFW_OC;
		int W_fwr = W_k / OC, W_oc = W_k - W_fwr * OC;

		int fh = y + (CFH - 1 - W_fhr)*sh;
		int fw = x + (CFW - 1 - W_fwr)*sw;
		bool lw = (fh < FH) && (fw < FW);
		int Woffset = ((W_oc*FH + fh)*FW + fw)*IC + tic0;
		Ws[buf][tx][ty] = (lw ? *(float4*)(W + Woffset) : FLOAT_ZERO4);

		//load 1 elem from deltaY[N, FH, FW, OC]
		int dY_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int dY_fhr = dY_k / CFW_OC; dY_k -= dY_fhr * CFW_OC;
		int dY_fwr = dY_k / OC;

		bool ldy0 = (tohs0 >= -dY_fhr) && (tohs0 < OH - dY_fhr) && (tows0 >= -dY_fwr) && (tows0 < OW - dY_fwr);
		bool ldy1 = (tohs1 >= -dY_fhr) && (tohs1 < OH - dY_fhr) && (tows1 >= -dY_fwr) && (tows1 < OW - dY_fwr);
		bool ldy2 = (tohs2 >= -dY_fhr) && (tohs2 < OH - dY_fhr) && (tows2 >= -dY_fwr) && (tows2 < OW - dY_fwr);
		bool ldy3 = (tohs3 >= -dY_fhr) && (tohs3 < OH - dY_fhr) && (tows3 >= -dY_fwr) && (tows3 < OW - dY_fwr);
		int yoffset = dY_fhr * OW_OC + dY_k;
		dYs[buf][ty][tx].x = (ldy0 ? deltaY[Yoffset0 + yoffset] : 0);
		dYs[buf][ty][tx].y = (ldy1 ? deltaY[Yoffset1 + yoffset] : 0);
		dYs[buf][ty][tx].z = (ldy2 ? deltaY[Yoffset2 + yoffset] : 0);
		dYs[buf][ty][tx].w = (ldy3 ? deltaY[Yoffset3 + yoffset] : 0);
		__syncthreads();
	}
#pragma once
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 dy0 = dYs[buf][ik][tx], dy1 = dYs[buf][ik + STEP][tx];
		float4  w0 =  Ws[buf][ik][ty],  w1 =  Ws[buf][ik + STEP][ty];

		simdMM4( v0, dy0.x, w0); simdMM4( v1, dy0.x, w1);
		simdMM4( v2, dy0.y, w0); simdMM4( v3, dy0.y, w1);
		simdMM4( v4, dy0.z, w0); simdMM4( v5, dy0.z, w1);
		simdMM4( v6, dy0.w, w0); simdMM4( v7, dy0.w, w1);
		simdMM4( v8, dy1.x, w0); simdMM4( v9, dy1.x, w1);
		simdMM4(v10, dy1.y, w0); simdMM4(v11, dy1.y, w1);
		simdMM4(v12, dy1.z, w0); simdMM4(v13, dy1.z, w1);
		simdMM4(v14, dy1.w, w0); simdMM4(v15, dy1.w, w1);
	}

	bool wrt0 = (ih0 < IH) && (iw0 < IW);
	bool wrt1 = (ih1 < IH) && (iw1 < IW);
	bool wrt2 = (ih2 < IH) && (iw2 < IW);
	bool wrt3 = (ih3 < IH) && (iw3 < IW);
	bool wrt4 = (ih4 < IH) && (iw4 < IW);
	bool wrt5 = (ih5 < IH) && (iw5 < IW);
	bool wrt6 = (ih6 < IH) && (iw6 < IW);
	bool wrt7 = (ih7 < IH) && (iw7 < IW);
	if (wrt0) { *(float4*)(&get4d(deltaX, n0, ih0, iw0, ic0, IH, IW, IC)) = v0;  *(float4*)(&get4d(deltaX, n0, ih0, iw0, ic0 + 4, IH, IW, IC)) = v1; }
	if (wrt1) { *(float4*)(&get4d(deltaX, n1, ih1, iw1, ic0, IH, IW, IC)) = v2;  *(float4*)(&get4d(deltaX, n1, ih1, iw1, ic0 + 4, IH, IW, IC)) = v3; }
	if (wrt2) { *(float4*)(&get4d(deltaX, n2, ih2, iw2, ic0, IH, IW, IC)) = v4;  *(float4*)(&get4d(deltaX, n2, ih2, iw2, ic0 + 4, IH, IW, IC)) = v5; }
	if (wrt3) { *(float4*)(&get4d(deltaX, n3, ih3, iw3, ic0, IH, IW, IC)) = v6;  *(float4*)(&get4d(deltaX, n3, ih3, iw3, ic0 + 4, IH, IW, IC)) = v7; }
	if (wrt4) { *(float4*)(&get4d(deltaX, n4, ih4, iw4, ic0, IH, IW, IC)) = v8;  *(float4*)(&get4d(deltaX, n4, ih4, iw4, ic0 + 4, IH, IW, IC)) = v9; }
	if (wrt5) { *(float4*)(&get4d(deltaX, n5, ih5, iw5, ic0, IH, IW, IC)) = v10; *(float4*)(&get4d(deltaX, n5, ih5, iw5, ic0 + 4, IH, IW, IC)) = v11; }
	if (wrt6) { *(float4*)(&get4d(deltaX, n6, ih6, iw6, ic0, IH, IW, IC)) = v12; *(float4*)(&get4d(deltaX, n6, ih6, iw6, ic0 + 4, IH, IW, IC)) = v13; }
	if (wrt7) { *(float4*)(&get4d(deltaX, n7, ih7, iw7, ic0, IH, IW, IC)) = v14; *(float4*)(&get4d(deltaX, n7, ih7, iw7, ic0 + 4, IH, IW, IC)) = v15; }
}

#endif


//(8 * 8)
#ifndef X_KERNEL_V10
#define X_KERNEL_V10

#define X_kv10(stream, LB, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM) \
	X_kernel_v10<LB, (1<<LB>>1)>\
		<<< dim3((GM>>LB>>3), (GN>>LB>>3), (sh*sw)), dim3(1<<LB, 1<<LB) >>>\
			(deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw,\
			 CFH, CFW, IH_slice, IW_slice, GK)

//LB = 4: Size = 1, Time = 1.932 msec, Performace = 1111.53 GFlop/s
template<int LB, int STEP>
__global__ void X_kernel_v10(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw,
	int CFH, int CFW,
	int IH_slice, int IW_slice, int GK)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4  Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 dYs[2][1 << LB][(1 << LB) + 1];

	//prepare for GZ = sh*sw
	int bz = blockIdx.z;
	int y = bz / sw, x = bz - y * sw;
	int ihs = (y - ph); ihs += (ihs < 0)*(ph - y + sh - 1) / sh * sh;
	int iws = (x - pw); iws += (iws < 0)*(pw - x + sw - 1) / sw * sw;

	//prepare for GN = IC
	int ic0 = ((blockIdx.y << LB) + ty) << 3;
	const int tic0 = ic0 + ((tx >= STEP) << 2);

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = ((blockIdx.x << LB) + tx) << 3;
	int j1 = j0 + 1, j2 = j0 + 2, j3 = j0 + 3;
	int j4 = j0 + 4, j5 = j0 + 5, j6 = j0 + 6, j7 = j0 + 7;
	int IH_IW_slice = IH_slice * IW_slice;
	X_n_ih_iw(j0, n0, ih0, iw0);
	X_n_ih_iw(j1, n1, ih1, iw1);
	X_n_ih_iw(j2, n2, ih2, iw2);
	X_n_ih_iw(j3, n3, ih3, iw3);
	X_n_ih_iw(j4, n4, ih4, iw4);
	X_n_ih_iw(j5, n5, ih5, iw5);
	X_n_ih_iw(j6, n6, ih6, iw6);
	X_n_ih_iw(j7, n7, ih7, iw7);

	bool flagY = (ty >= STEP);
	const int oph = CFH - 1, opw = CFW - 1; ph -= y; pw -= x;
	const int tohs0 = (((ih4 - ih0)*flagY + ih0) + ph) / sh - oph;
	const int tohs1 = (((ih5 - ih1)*flagY + ih1) + ph) / sh - oph;
	const int tohs2 = (((ih6 - ih2)*flagY + ih2) + ph) / sh - oph;
	const int tohs3 = (((ih7 - ih3)*flagY + ih3) + ph) / sh - oph;
	const int tows0 = (((iw4 - iw0)*flagY + iw0) + pw) / sw - opw;
	const int tows1 = (((iw5 - iw1)*flagY + iw1) + pw) / sw - opw;
	const int tows2 = (((iw6 - iw2)*flagY + iw2) + pw) / sw - opw;
	const int tows3 = (((iw7 - iw3)*flagY + iw3) + pw) / sw - opw;
	int Yoffset0 = ((((n4 - n0)*flagY + n0)*OH + tohs0)*OW + tows0)*OC;
	int Yoffset1 = ((((n5 - n1)*flagY + n1)*OH + tohs1)*OW + tows1)*OC;
	int Yoffset2 = ((((n6 - n2)*flagY + n2)*OH + tohs2)*OW + tows2)*OC;
	int Yoffset3 = ((((n7 - n3)*flagY + n3)*OH + tohs3)*OW + tows3)*OC;

	const int CFW_OC = CFW * OC;

	//load 2 elem from W[OC, FH, FW, IC]
	int W_k = tx - ((tx >= STEP) << LB >> 1);
	int W_fhr = W_k / CFW_OC; W_k -= W_fhr * CFW_OC;
	int W_fwr = W_k / OC, W_oc = W_k - W_fwr * OC;

	int fh = y + (CFH - 1 - W_fhr)*sh;
	int fw = x + (CFW - 1 - W_fwr)*sw;

	bool lw = (fh < FH) && (fw < FW);
	int Woffset = ((W_oc*FH + fh)*FW + fw)*IC + tic0;
	Ws[buf][tx][ty] = (lw ? *(float4*)(W + Woffset) : FLOAT_ZERO4);

	//load 2 elem from deltaY[N, FH, FW, OC]
	int dY_k = ty - ((ty >= STEP) << LB >> 1);
	int dY_fhr = dY_k / CFW_OC; dY_k -= dY_fhr * CFW_OC;
	int dY_fwr = dY_k / OC;

	bool ldy0 = (tohs0 >= -dY_fhr) && (tohs0 < OH - dY_fhr) && (tows0 >= -dY_fwr) && (tows0 < OW - dY_fwr);
	bool ldy1 = (tohs1 >= -dY_fhr) && (tohs1 < OH - dY_fhr) && (tows1 >= -dY_fwr) && (tows1 < OW - dY_fwr);
	bool ldy2 = (tohs2 >= -dY_fhr) && (tohs2 < OH - dY_fhr) && (tows2 >= -dY_fwr) && (tows2 < OW - dY_fwr);
	bool ldy3 = (tohs3 >= -dY_fhr) && (tohs3 < OH - dY_fhr) && (tows3 >= -dY_fwr) && (tows3 < OW - dY_fwr);
	const int OW_OC = OW * OC, yoffset = dY_fhr * OW_OC + dY_k;
	dYs[buf][ty][tx].x = (ldy0 ? deltaY[Yoffset0 + yoffset] : 0);
	dYs[buf][ty][tx].y = (ldy1 ? deltaY[Yoffset1 + yoffset] : 0);
	dYs[buf][ty][tx].z = (ldy2 ? deltaY[Yoffset2 + yoffset] : 0);
	dYs[buf][ty][tx].w = (ldy3 ? deltaY[Yoffset3 + yoffset] : 0);
	__syncthreads();

	//compure_area----------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)
	{
#pragma once
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 dy0 = dYs[buf][ik][tx], dy1 = dYs[buf][ik + STEP][tx];
			float4  w0 = Ws[buf][ik][ty], w1 = Ws[buf][ik + STEP][ty];

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

		//load 1 elem from W[OC, FH, FW, IC]
		int W_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int W_fhr = W_k / CFW_OC; W_k -= W_fhr * CFW_OC;
		int W_fwr = W_k / OC, W_oc = W_k - W_fwr * OC;

		int fh = y + (CFH - 1 - W_fhr)*sh;
		int fw = x + (CFW - 1 - W_fwr)*sw;
		bool lw = (fh < FH) && (fw < FW);
		int Woffset = ((W_oc*FH + fh)*FW + fw)*IC + tic0;
		Ws[buf][tx][ty] = (lw ? *(float4*)(W + Woffset) : FLOAT_ZERO4);

		//load 1 elem from deltaY[N, FH, FW, OC]
		int dY_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int dY_fhr = dY_k / CFW_OC; dY_k -= dY_fhr * CFW_OC;
		int dY_fwr = dY_k / OC;

		bool ldy0 = (tohs0 >= -dY_fhr) && (tohs0 < OH - dY_fhr) && (tows0 >= -dY_fwr) && (tows0 < OW - dY_fwr);
		bool ldy1 = (tohs1 >= -dY_fhr) && (tohs1 < OH - dY_fhr) && (tows1 >= -dY_fwr) && (tows1 < OW - dY_fwr);
		bool ldy2 = (tohs2 >= -dY_fhr) && (tohs2 < OH - dY_fhr) && (tows2 >= -dY_fwr) && (tows2 < OW - dY_fwr);
		bool ldy3 = (tohs3 >= -dY_fhr) && (tohs3 < OH - dY_fhr) && (tows3 >= -dY_fwr) && (tows3 < OW - dY_fwr);
		int yoffset = dY_fhr * OW_OC + dY_k;
		dYs[buf][ty][tx].x = (ldy0 ? deltaY[Yoffset0 + yoffset] : 0);
		dYs[buf][ty][tx].y = (ldy1 ? deltaY[Yoffset1 + yoffset] : 0);
		dYs[buf][ty][tx].z = (ldy2 ? deltaY[Yoffset2 + yoffset] : 0);
		dYs[buf][ty][tx].w = (ldy3 ? deltaY[Yoffset3 + yoffset] : 0);
		__syncthreads();
	}
#pragma once
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 dy0 = dYs[buf][ik][tx], dy1 = dYs[buf][ik + STEP][tx];
		float4  w0 = Ws[buf][ik][ty], w1 = Ws[buf][ik + STEP][ty];

		simdMM4(v0, dy0.x, w0); simdMM4(v1, dy0.x, w1);
		simdMM4(v2, dy0.y, w0); simdMM4(v3, dy0.y, w1);
		simdMM4(v4, dy0.z, w0); simdMM4(v5, dy0.z, w1);
		simdMM4(v6, dy0.w, w0); simdMM4(v7, dy0.w, w1);
		simdMM4(v8, dy1.x, w0); simdMM4(v9, dy1.x, w1);
		simdMM4(v10, dy1.y, w0); simdMM4(v11, dy1.y, w1);
		simdMM4(v12, dy1.z, w0); simdMM4(v13, dy1.z, w1);
		simdMM4(v14, dy1.w, w0); simdMM4(v15, dy1.w, w1);
	}

	X_write8(deltaX, n0, ih0, iw0, ic0,  v0,  v1);
	X_write8(deltaX, n1, ih1, iw1, ic0,  v2,  v3);
	X_write8(deltaX, n2, ih2, iw2, ic0,  v4,  v5);
	X_write8(deltaX, n3, ih3, iw3, ic0,  v6,  v7);
	X_write8(deltaX, n4, ih4, iw4, ic0,  v8,  v9);
	X_write8(deltaX, n5, ih5, iw5, ic0, v10, v11);
	X_write8(deltaX, n6, ih6, iw6, ic0, v12, v13);
	X_write8(deltaX, n7, ih7, iw7, ic0, v14, v15);
}

#endif


//(8 * 8)
#ifndef X_KERNEL_V11
#define X_KERNEL_V11

#define X_kv11(stream, LB, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM) \
	X_kernel_v11<LB, (1<<LB>>1)>\
		<<< dim3((GM>>LB>>3), (GN>>LB>>3), (sh*sw)), dim3(1<<LB, 1<<LB) >>>\
			(deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw,\
			 CFH, CFW, IH_slice, IW_slice, GK)

//LB = 4: Size = 1, Time = 1.786 msec, Performace = 1202.4 GFlop/s
template<int LB, int STEP>
__global__ void X_kernel_v11(
	cudaTextureObject_t deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw,
	int CFH, int CFW,
	int IH_slice, int IW_slice, int GK)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4  Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 dYs[2][1 << LB][(1 << LB) + 1];

	//prepare for GZ = sh*sw
	int bz = blockIdx.z;
	int y = bz / sw, x = bz - y * sw;
	int ihs = (y - ph); ihs += (ihs < 0)*(ph - y + sh - 1) / sh * sh;
	int iws = (x - pw); iws += (iws < 0)*(pw - x + sw - 1) / sw * sw;

	//prepare for GN = IC
	int ic0 = ((blockIdx.y << LB) + ty) << 3;
	const int tic0 = ic0 + ((tx >= STEP) << 2);

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = ((blockIdx.x << LB) + tx) << 3;
	int j1 = j0 + 1, j2 = j0 + 2, j3 = j0 + 3;
	int j4 = j0 + 4, j5 = j0 + 5, j6 = j0 + 6, j7 = j0 + 7;
	int IH_IW_slice = IH_slice * IW_slice;
	X_n_ih_iw(j0, n0, ih0, iw0);
	X_n_ih_iw(j1, n1, ih1, iw1);
	X_n_ih_iw(j2, n2, ih2, iw2);
	X_n_ih_iw(j3, n3, ih3, iw3);
	X_n_ih_iw(j4, n4, ih4, iw4);
	X_n_ih_iw(j5, n5, ih5, iw5);
	X_n_ih_iw(j6, n6, ih6, iw6);
	X_n_ih_iw(j7, n7, ih7, iw7);
	ih0 = (ih0 * sh) + ihs, iw0 = (iw0 * sw) + iws;
	ih1 = (ih1 * sh) + ihs, iw1 = (iw1 * sw) + iws;
	ih2 = (ih2 * sh) + ihs, iw2 = (iw2 * sw) + iws;
	ih3 = (ih3 * sh) + ihs, iw3 = (iw3 * sw) + iws;
	ih4 = (ih4 * sh) + ihs, iw4 = (iw4 * sw) + iws;
	ih5 = (ih5 * sh) + ihs, iw5 = (iw5 * sw) + iws;
	ih6 = (ih6 * sh) + ihs, iw6 = (iw6 * sw) + iws;
	ih7 = (ih7 * sh) + ihs, iw7 = (iw7 * sw) + iws;

	bool flagY = (ty >= STEP);
	const int oph = CFH - 1, opw = CFW - 1; ph -= y; pw -= x;
	const int tohs0 = (((ih4 - ih0)*flagY + ih0) + ph) / sh - oph;
	const int tohs1 = (((ih5 - ih1)*flagY + ih1) + ph) / sh - oph;
	const int tohs2 = (((ih6 - ih2)*flagY + ih2) + ph) / sh - oph;
	const int tohs3 = (((ih7 - ih3)*flagY + ih3) + ph) / sh - oph;
	const int tows0 = (((iw4 - iw0)*flagY + iw0) + pw) / sw - opw;
	const int tows1 = (((iw5 - iw1)*flagY + iw1) + pw) / sw - opw;
	const int tows2 = (((iw6 - iw2)*flagY + iw2) + pw) / sw - opw;
	const int tows3 = (((iw7 - iw3)*flagY + iw3) + pw) / sw - opw;
	int Yoffset0 = ((((n4 - n0)*flagY + n0)*OH + tohs0)*OW + tows0)*OC;
	int Yoffset1 = ((((n5 - n1)*flagY + n1)*OH + tohs1)*OW + tows1)*OC;
	int Yoffset2 = ((((n6 - n2)*flagY + n2)*OH + tohs2)*OW + tows2)*OC;
	int Yoffset3 = ((((n7 - n3)*flagY + n3)*OH + tohs3)*OW + tows3)*OC;

	const int CFW_OC = CFW * OC;

	//load 2 elem from W[OC, FH, FW, IC]
	int W_k = tx - ((tx >= STEP) << LB >> 1);
	int W_fhr = W_k / CFW_OC; W_k -= W_fhr * CFW_OC;
	int W_fwr = W_k / OC, W_oc = W_k - W_fwr * OC;

	int fh = y + (CFH - 1 - W_fhr)*sh;
	int fw = x + (CFW - 1 - W_fwr)*sw;

	bool lw = (fh < FH) && (fw < FW);
	int Woffset = ((W_oc*FH + fh)*FW + fw)*IC + tic0;
	Ws[buf][tx][ty] = (lw ? *(float4*)(W + Woffset) : FLOAT_ZERO4);

	//load 2 elem from deltaY[N, FH, FW, OC]
	int dY_k = ty - ((ty >= STEP) << LB >> 1);
	int dY_fhr = dY_k / CFW_OC; dY_k -= dY_fhr * CFW_OC;
	int dY_fwr = dY_k / OC;
	bool ldy0 = (tohs0 >= -dY_fhr) && (tohs0 < OH - dY_fhr) && (tows0 >= -dY_fwr) && (tows0 < OW - dY_fwr);
	bool ldy1 = (tohs1 >= -dY_fhr) && (tohs1 < OH - dY_fhr) && (tows1 >= -dY_fwr) && (tows1 < OW - dY_fwr);
	bool ldy2 = (tohs2 >= -dY_fhr) && (tohs2 < OH - dY_fhr) && (tows2 >= -dY_fwr) && (tows2 < OW - dY_fwr);
	bool ldy3 = (tohs3 >= -dY_fhr) && (tohs3 < OH - dY_fhr) && (tows3 >= -dY_fwr) && (tows3 < OW - dY_fwr);
	const int OW_OC = OW * OC, yoffset = dY_fhr * OW_OC + dY_k;
	dYs[buf][ty][tx].x = ldy0 * tex1Dfetch<float>(deltaY, Yoffset0 + yoffset);
	dYs[buf][ty][tx].y = ldy1 * tex1Dfetch<float>(deltaY, Yoffset1 + yoffset);
	dYs[buf][ty][tx].z = ldy2 * tex1Dfetch<float>(deltaY, Yoffset2 + yoffset);
	dYs[buf][ty][tx].w = ldy3 * tex1Dfetch<float>(deltaY, Yoffset3 + yoffset);
	__syncthreads();

	//compure_area----------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)
	{
#pragma once
		for (int ik = 0; ik < STEP; ik++)
		{
			float4  w0 = Ws[buf][ik][ty], w1 = Ws[buf][ik + STEP][ty];
			float4 dy0 = dYs[buf][ik][tx], dy1 = dYs[buf][ik + STEP][tx];

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

		//load 1 elem from W[OC, FH, FW, IC]
		int W_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int W_fhr = W_k / CFW_OC; W_k -= W_fhr * CFW_OC;
		int W_fwr = W_k / OC, W_oc = W_k - W_fwr * OC;

		int fh = y + (CFH - 1 - W_fhr)*sh;
		int fw = x + (CFW - 1 - W_fwr)*sw;
		bool lw = (fh < FH) && (fw < FW);
		int Woffset = ((W_oc*FH + fh)*FW + fw)*IC + tic0;
		Ws[buf][tx][ty] = (lw ? *(float4*)(W + Woffset) : FLOAT_ZERO4);

		//load 1 elem from deltaY[N, FH, FW, OC]
		int dY_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int dY_fhr = dY_k / CFW_OC; dY_k -= dY_fhr * CFW_OC;
		int dY_fwr = dY_k / OC;

		bool ldy0 = (tohs0 >= -dY_fhr) && (tohs0 < OH - dY_fhr) && (tows0 >= -dY_fwr) && (tows0 < OW - dY_fwr);
		bool ldy1 = (tohs1 >= -dY_fhr) && (tohs1 < OH - dY_fhr) && (tows1 >= -dY_fwr) && (tows1 < OW - dY_fwr);
		bool ldy2 = (tohs2 >= -dY_fhr) && (tohs2 < OH - dY_fhr) && (tows2 >= -dY_fwr) && (tows2 < OW - dY_fwr);
		bool ldy3 = (tohs3 >= -dY_fhr) && (tohs3 < OH - dY_fhr) && (tows3 >= -dY_fwr) && (tows3 < OW - dY_fwr);
		int yoffset = dY_fhr * OW_OC + dY_k;
		dYs[buf][ty][tx].x = ldy0 * tex1Dfetch<float>(deltaY, Yoffset0 + yoffset);
		dYs[buf][ty][tx].y = ldy1 * tex1Dfetch<float>(deltaY, Yoffset1 + yoffset);
		dYs[buf][ty][tx].z = ldy2 * tex1Dfetch<float>(deltaY, Yoffset2 + yoffset);
		dYs[buf][ty][tx].w = ldy3 * tex1Dfetch<float>(deltaY, Yoffset3 + yoffset);
		__syncthreads();
	}
#pragma once
	for (int ik = 0; ik < STEP; ik++)
	{
		float4  w0 = Ws[buf][ik][ty], w1 = Ws[buf][ik + STEP][ty];
		float4 dy0 = dYs[buf][ik][tx], dy1 = dYs[buf][ik + STEP][tx];

		simdMM4(v0, dy0.x, w0); simdMM4(v1, dy0.x, w1);
		simdMM4(v2, dy0.y, w0); simdMM4(v3, dy0.y, w1);
		simdMM4(v4, dy0.z, w0); simdMM4(v5, dy0.z, w1);
		simdMM4(v6, dy0.w, w0); simdMM4(v7, dy0.w, w1);
		simdMM4(v8, dy1.x, w0); simdMM4(v9, dy1.x, w1);
		simdMM4(v10, dy1.y, w0); simdMM4(v11, dy1.y, w1);
		simdMM4(v12, dy1.z, w0); simdMM4(v13, dy1.z, w1);
		simdMM4(v14, dy1.w, w0); simdMM4(v15, dy1.w, w1);
	}

	X_write8(deltaX, n0, ih0, iw0, ic0, v0, v1);
	X_write8(deltaX, n1, ih1, iw1, ic0, v2, v3);
	X_write8(deltaX, n2, ih2, iw2, ic0, v4, v5);
	X_write8(deltaX, n3, ih3, iw3, ic0, v6, v7);
	X_write8(deltaX, n4, ih4, iw4, ic0, v8, v9);
	X_write8(deltaX, n5, ih5, iw5, ic0, v10, v11);
	X_write8(deltaX, n6, ih6, iw6, ic0, v12, v13);
	X_write8(deltaX, n7, ih7, iw7, ic0, v14, v15);
}

#endif


//(8 * 8), sh = sw = 2
#ifndef X_KERNEL_V12
#define X_KERNEL_V12

#define X_kv12(stream, LB, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM) \
	X_kernel_v12<LB, (1<<LB>>1)>\
		<<< dim3((GM>>LB>>3), (GN>>LB>>3), (sh*sw)), dim3(1<<LB, 1<<LB) >>>\
			(deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, ph, pw,\
			 CFH, CFW, IH_slice, IW_slice, GK)

//LB = 4: Size = 1, Time = 1.768 msec, Performace = 1214.64 GFlop/s
template<int LB, int STEP>
__global__ void X_kernel_v12(
	cudaTextureObject_t deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int OC,
	int ph, int pw,
	int CFH, int CFW,
	int IH_slice, int IW_slice, int GK)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4  Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 dYs[2][1 << LB][(1 << LB) + 1];

	//prepare for GZ = sh*sw
	int bz = blockIdx.z;
	int y = bz >> 1, x = bz & 1;
	int ihs = (y - ph); ihs += (ihs < 0)*((ph - y + 1) >> 1 << 1);
	int iws = (x - pw); iws += (iws < 0)*((pw - x + 1) >> 1 << 1);

	//prepare for GN = IC
	int ic0 = ((blockIdx.y << LB) + ty) << 3;
	const int tic0 = ic0 + ((tx >= STEP) << 2);

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = ((blockIdx.x << LB) + tx) << 3;
	int j1 = j0 + 1, j2 = j0 + 2, j3 = j0 + 3;
	int j4 = j0 + 4, j5 = j0 + 5, j6 = j0 + 6, j7 = j0 + 7;
	int IH_IW_slice = IH_slice * IW_slice;
	X_n_ih_iw(j0, n0, ih0, iw0);
	X_n_ih_iw(j1, n1, ih1, iw1);
	X_n_ih_iw(j2, n2, ih2, iw2);
	X_n_ih_iw(j3, n3, ih3, iw3);
	X_n_ih_iw(j4, n4, ih4, iw4);
	X_n_ih_iw(j5, n5, ih5, iw5);
	X_n_ih_iw(j6, n6, ih6, iw6);
	X_n_ih_iw(j7, n7, ih7, iw7);
	ih0 = (ih0 << 1) + ihs, iw0 = (iw0 << 1) + iws;
	ih1 = (ih1 << 1) + ihs, iw1 = (iw1 << 1) + iws;
	ih2 = (ih2 << 1) + ihs, iw2 = (iw2 << 1) + iws;
	ih3 = (ih3 << 1) + ihs, iw3 = (iw3 << 1) + iws;
	ih4 = (ih4 << 1) + ihs, iw4 = (iw4 << 1) + iws;
	ih5 = (ih5 << 1) + ihs, iw5 = (iw5 << 1) + iws;
	ih6 = (ih6 << 1) + ihs, iw6 = (iw6 << 1) + iws;
	ih7 = (ih7 << 1) + ihs, iw7 = (iw7 << 1) + iws;

	bool flagY = (ty >= STEP);
	const int oph = CFH - 1, opw = CFW - 1; ph -= y; pw -= x;
	const int tohs0 = (((ih4 - ih0)*flagY + ih0 + ph) >> 1) - oph;
	const int tohs1 = (((ih5 - ih1)*flagY + ih1 + ph) >> 1) - oph;
	const int tohs2 = (((ih6 - ih2)*flagY + ih2 + ph) >> 1) - oph;
	const int tohs3 = (((ih7 - ih3)*flagY + ih3 + ph) >> 1) - oph;
	const int tows0 = (((iw4 - iw0)*flagY + iw0 + pw) >> 1) - opw;
	const int tows1 = (((iw5 - iw1)*flagY + iw1 + pw) >> 1) - opw;
	const int tows2 = (((iw6 - iw2)*flagY + iw2 + pw) >> 1) - opw;
	const int tows3 = (((iw7 - iw3)*flagY + iw3 + pw) >> 1) - opw;
	int Yoffset0 = ((((n4 - n0)*flagY + n0)*OH + tohs0)*OW + tows0)*OC;
	int Yoffset1 = ((((n5 - n1)*flagY + n1)*OH + tohs1)*OW + tows1)*OC;
	int Yoffset2 = ((((n6 - n2)*flagY + n2)*OH + tohs2)*OW + tows2)*OC;
	int Yoffset3 = ((((n7 - n3)*flagY + n3)*OH + tohs3)*OW + tows3)*OC;

	const int CFW_OC = CFW * OC;

	//load 2 elem from W[OC, FH, FW, IC]
	int W_k = tx - ((tx >= STEP) << LB >> 1);
	int W_fhr = W_k / CFW_OC; W_k -= W_fhr * CFW_OC;
	int W_fwr = W_k / OC, W_oc = W_k - W_fwr * OC;

	int fh = y + ((CFH - 1 - W_fhr) << 1);
	int fw = x + ((CFW - 1 - W_fwr) << 1);

	bool lw = (fh < FH) && (fw < FW);
	int Woffset = ((W_oc*FH + fh)*FW + fw)*IC + tic0;
	Ws[buf][tx][ty] = (lw ? *(float4*)(W + Woffset) : FLOAT_ZERO4);

	//load 2 elem from deltaY[N, FH, FW, OC]
	int dY_k = ty - ((ty >= STEP) << LB >> 1);
	int dY_fhr = dY_k / CFW_OC; dY_k -= dY_fhr * CFW_OC;
	int dY_fwr = dY_k / OC;
	bool ldy0 = (tohs0 >= -dY_fhr) && (tohs0 < OH - dY_fhr) && (tows0 >= -dY_fwr) && (tows0 < OW - dY_fwr);
	bool ldy1 = (tohs1 >= -dY_fhr) && (tohs1 < OH - dY_fhr) && (tows1 >= -dY_fwr) && (tows1 < OW - dY_fwr);
	bool ldy2 = (tohs2 >= -dY_fhr) && (tohs2 < OH - dY_fhr) && (tows2 >= -dY_fwr) && (tows2 < OW - dY_fwr);
	bool ldy3 = (tohs3 >= -dY_fhr) && (tohs3 < OH - dY_fhr) && (tows3 >= -dY_fwr) && (tows3 < OW - dY_fwr);
	const int OW_OC = OW * OC, yoffset = dY_fhr * OW_OC + dY_k;
	dYs[buf][ty][tx].x = ldy0 * tex1Dfetch<float>(deltaY, Yoffset0 + yoffset);
	dYs[buf][ty][tx].y = ldy1 * tex1Dfetch<float>(deltaY, Yoffset1 + yoffset);
	dYs[buf][ty][tx].z = ldy2 * tex1Dfetch<float>(deltaY, Yoffset2 + yoffset);
	dYs[buf][ty][tx].w = ldy3 * tex1Dfetch<float>(deltaY, Yoffset3 + yoffset);
	__syncthreads();

	//compure_area----------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)
	{
#pragma once
		for (int ik = 0; ik < STEP; ik++)
		{
			float4  w0 = Ws[buf][ik][ty], w1 = Ws[buf][ik + STEP][ty];
			float4 dy0 = dYs[buf][ik][tx], dy1 = dYs[buf][ik + STEP][tx];

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

		//load 1 elem from W[OC, FH, FW, IC]
		int W_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int W_fhr = W_k / CFW_OC; W_k -= W_fhr * CFW_OC;
		int W_fwr = W_k / OC, W_oc = W_k - W_fwr * OC;

		int fh = y + ((CFH - 1 - W_fhr) << 1);
		int fw = x + ((CFW - 1 - W_fwr) << 1);
		bool lw = (fh < FH) && (fw < FW);
		int Woffset = ((W_oc*FH + fh)*FW + fw)*IC + tic0;
		Ws[buf][tx][ty] = (lw ? *(float4*)(W + Woffset) : FLOAT_ZERO4);

		//load 1 elem from deltaY[N, FH, FW, OC]
		int dY_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int dY_fhr = dY_k / CFW_OC; dY_k -= dY_fhr * CFW_OC;
		int dY_fwr = dY_k / OC;

		bool ldy0 = (tohs0 >= -dY_fhr) && (tohs0 < OH - dY_fhr) && (tows0 >= -dY_fwr) && (tows0 < OW - dY_fwr);
		bool ldy1 = (tohs1 >= -dY_fhr) && (tohs1 < OH - dY_fhr) && (tows1 >= -dY_fwr) && (tows1 < OW - dY_fwr);
		bool ldy2 = (tohs2 >= -dY_fhr) && (tohs2 < OH - dY_fhr) && (tows2 >= -dY_fwr) && (tows2 < OW - dY_fwr);
		bool ldy3 = (tohs3 >= -dY_fhr) && (tohs3 < OH - dY_fhr) && (tows3 >= -dY_fwr) && (tows3 < OW - dY_fwr);
		int yoffset = dY_fhr * OW_OC + dY_k;
		dYs[buf][ty][tx].x = ldy0 * tex1Dfetch<float>(deltaY, Yoffset0 + yoffset);
		dYs[buf][ty][tx].y = ldy1 * tex1Dfetch<float>(deltaY, Yoffset1 + yoffset);
		dYs[buf][ty][tx].z = ldy2 * tex1Dfetch<float>(deltaY, Yoffset2 + yoffset);
		dYs[buf][ty][tx].w = ldy3 * tex1Dfetch<float>(deltaY, Yoffset3 + yoffset);
		__syncthreads();
	}
#pragma once
	for (int ik = 0; ik < STEP; ik++)
	{
		float4  w0 = Ws[buf][ik][ty], w1 = Ws[buf][ik + STEP][ty];
		float4 dy0 = dYs[buf][ik][tx], dy1 = dYs[buf][ik + STEP][tx];

		simdMM4(v0, dy0.x, w0); simdMM4(v1, dy0.x, w1);
		simdMM4(v2, dy0.y, w0); simdMM4(v3, dy0.y, w1);
		simdMM4(v4, dy0.z, w0); simdMM4(v5, dy0.z, w1);
		simdMM4(v6, dy0.w, w0); simdMM4(v7, dy0.w, w1);
		simdMM4(v8, dy1.x, w0); simdMM4(v9, dy1.x, w1);
		simdMM4(v10, dy1.y, w0); simdMM4(v11, dy1.y, w1);
		simdMM4(v12, dy1.z, w0); simdMM4(v13, dy1.z, w1);
		simdMM4(v14, dy1.w, w0); simdMM4(v15, dy1.w, w1);
	}

	X_write8(deltaX, n0, ih0, iw0, ic0, v0, v1);
	X_write8(deltaX, n1, ih1, iw1, ic0, v2, v3);
	X_write8(deltaX, n2, ih2, iw2, ic0, v4, v5);
	X_write8(deltaX, n3, ih3, iw3, ic0, v6, v7);
	X_write8(deltaX, n4, ih4, iw4, ic0, v8, v9);
	X_write8(deltaX, n5, ih5, iw5, ic0, v10, v11);
	X_write8(deltaX, n6, ih6, iw6, ic0, v12, v13);
	X_write8(deltaX, n7, ih7, iw7, ic0, v14, v15);
}

#endif


//(8 * 8),
#ifndef X_KERNEL_V13
#define X_KERNEL_V13

#define X_kv13(stream, LB, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM) \
	X_kernel_v13<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>3), (GM>>LB>>3), (sh*sw)), dim3(1<<LB, 1<<LB) >>>\
			(deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw,\
			 CFH, CFW, IH_slice, IW_slice, GK)

//LB = 4: Size = 1.05579, Time = 1.606 msec, Performace = 1411.76 GFlop/s
//LB = 3: Size = 1.05579, Time = 1.264 msec, Performace = 1793.74 GFlop/s
template<int LB, int STEP>
__global__ void X_kernel_v13(
	cudaTextureObject_t deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw,
	int CFH, int CFW, int IH_slice, int IW_slice, int GK)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4  Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 dYs[2][1 << LB][(1 << LB) + 1];

	//prepare for GZ = sh*sw
	int bz = blockIdx.z;
	int y = bz / sw, x = bz - y * sw;
	int ihs = (y - ph); ihs += (ihs < 0)*(ph - y + sh - 1) / sh * sh;
	int iws = (x - pw); iws += (iws < 0)*(pw - x + sw - 1) / sw * sw;

	//prepare for GN = IC
	int ic0 = ((blockIdx.x << LB) + tx) << 3;
	const int tic0 = ic0 + ((ty >= STEP) << 2);

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = ((blockIdx.y << LB) + ty) << 3;
	int j1 = j0 + 1, j2 = j0 + 2, j3 = j0 + 3;
	int j4 = j0 + 4, j5 = j0 + 5, j6 = j0 + 6, j7 = j0 + 7;
	int IH_IW_slice = IH_slice * IW_slice;
	X_n_ih_iw(j0, n0, ih0, iw0);
	X_n_ih_iw(j1, n1, ih1, iw1);
	X_n_ih_iw(j2, n2, ih2, iw2);
	X_n_ih_iw(j3, n3, ih3, iw3);
	X_n_ih_iw(j4, n4, ih4, iw4);
	X_n_ih_iw(j5, n5, ih5, iw5);
	X_n_ih_iw(j6, n6, ih6, iw6);
	X_n_ih_iw(j7, n7, ih7, iw7);
	ih0 = (ih0 * sh) + ihs, iw0 = (iw0 * sw) + iws;
	ih1 = (ih1 * sh) + ihs, iw1 = (iw1 * sw) + iws;
	ih2 = (ih2 * sh) + ihs, iw2 = (iw2 * sw) + iws;
	ih3 = (ih3 * sh) + ihs, iw3 = (iw3 * sw) + iws;
	ih4 = (ih4 * sh) + ihs, iw4 = (iw4 * sw) + iws;
	ih5 = (ih5 * sh) + ihs, iw5 = (iw5 * sw) + iws;
	ih6 = (ih6 * sh) + ihs, iw6 = (iw6 * sw) + iws;
	ih7 = (ih7 * sh) + ihs, iw7 = (iw7 * sw) + iws;


	bool flagY = (tx >= STEP);
	const int oph = CFH - 1, opw = CFW - 1; ph -= y; pw -= x;
	const int tohs0 = (((ih4 - ih0)*flagY + ih0) + ph) / sh - oph;
	const int tohs1 = (((ih5 - ih1)*flagY + ih1) + ph) / sh - oph;
	const int tohs2 = (((ih6 - ih2)*flagY + ih2) + ph) / sh - oph;
	const int tohs3 = (((ih7 - ih3)*flagY + ih3) + ph) / sh - oph;
	const int tows0 = (((iw4 - iw0)*flagY + iw0) + pw) / sw - opw;
	const int tows1 = (((iw5 - iw1)*flagY + iw1) + pw) / sw - opw;
	const int tows2 = (((iw6 - iw2)*flagY + iw2) + pw) / sw - opw;
	const int tows3 = (((iw7 - iw3)*flagY + iw3) + pw) / sw - opw;
	int Yoffset0 = ((((n4 - n0)*flagY + n0)*OH + tohs0)*OW + tows0)*OC;
	int Yoffset1 = ((((n5 - n1)*flagY + n1)*OH + tohs1)*OW + tows1)*OC;
	int Yoffset2 = ((((n6 - n2)*flagY + n2)*OH + tohs2)*OW + tows2)*OC;
	int Yoffset3 = ((((n7 - n3)*flagY + n3)*OH + tohs3)*OW + tows3)*OC;

	const int CFW_OC = CFW * OC;

	//load 2 elem from W[OC, FH, FW, IC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);
	int W_fhr = W_k / CFW_OC; W_k -= W_fhr * CFW_OC;
	int W_fwr = W_k / OC, W_oc = W_k - W_fwr * OC;

	int fh = y + (CFH - 1 - W_fhr)*sh;
	int fw = x + (CFW - 1 - W_fwr)*sw;

	bool lw = (fh < FH) && (fw < FW);
	int Woffset = ((W_oc*FH + fh)*FW + fw)*IC + tic0;
	Ws[buf][ty][tx] = (lw ? *(float4*)(W + Woffset) : FLOAT_ZERO4);

	//load 2 elem from deltaY[N, FH, FW, OC]
	int dY_k = tx - ((tx >= STEP) << LB >> 1);
	int dY_fhr = dY_k / CFW_OC; dY_k -= dY_fhr * CFW_OC;
	int dY_fwr = dY_k / OC;
	bool ldy0 = (tohs0 >= -dY_fhr) && (tohs0 < OH - dY_fhr) && (tows0 >= -dY_fwr) && (tows0 < OW - dY_fwr);
	bool ldy1 = (tohs1 >= -dY_fhr) && (tohs1 < OH - dY_fhr) && (tows1 >= -dY_fwr) && (tows1 < OW - dY_fwr);
	bool ldy2 = (tohs2 >= -dY_fhr) && (tohs2 < OH - dY_fhr) && (tows2 >= -dY_fwr) && (tows2 < OW - dY_fwr);
	bool ldy3 = (tohs3 >= -dY_fhr) && (tohs3 < OH - dY_fhr) && (tows3 >= -dY_fwr) && (tows3 < OW - dY_fwr);
	const int OW_OC = OW * OC, yoffset = dY_fhr * OW_OC + dY_k;
	dYs[buf][tx][ty].x = ldy0 * tex1Dfetch<float>(deltaY, Yoffset0 + yoffset);
	dYs[buf][tx][ty].y = ldy1 * tex1Dfetch<float>(deltaY, Yoffset1 + yoffset);
	dYs[buf][tx][ty].z = ldy2 * tex1Dfetch<float>(deltaY, Yoffset2 + yoffset);
	dYs[buf][tx][ty].w = ldy3 * tex1Dfetch<float>(deltaY, Yoffset3 + yoffset);
	__syncthreads();

	//compure_area----------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)
	{
#pragma once
		for (int ik = 0; ik < STEP; ik++)
		{
			float4  w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];
			float4 dy0 = dYs[buf][ik][ty], dy1 = dYs[buf][ik + STEP][ty];

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

		//load 1 elem from W[OC, FH, FW, IC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int W_fhr = W_k / CFW_OC; W_k -= W_fhr * CFW_OC;
		int W_fwr = W_k / OC, W_oc = W_k - W_fwr * OC;

		int fh = y + (CFH - 1 - W_fhr)*sh;
		int fw = x + (CFW - 1 - W_fwr)*sw;
		bool lw = (fh < FH) && (fw < FW);
		int Woffset = ((W_oc*FH + fh)*FW + fw)*IC + tic0;
		Ws[buf][ty][tx] = (lw ? *(float4*)(W + Woffset) : FLOAT_ZERO4);

		//load 1 elem from deltaY[N, FH, FW, OC]
		int dY_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int dY_fhr = dY_k / CFW_OC; dY_k -= dY_fhr * CFW_OC;
		int dY_fwr = dY_k / OC;

		bool ldy0 = (tohs0 >= -dY_fhr) && (tohs0 < OH - dY_fhr) && (tows0 >= -dY_fwr) && (tows0 < OW - dY_fwr);
		bool ldy1 = (tohs1 >= -dY_fhr) && (tohs1 < OH - dY_fhr) && (tows1 >= -dY_fwr) && (tows1 < OW - dY_fwr);
		bool ldy2 = (tohs2 >= -dY_fhr) && (tohs2 < OH - dY_fhr) && (tows2 >= -dY_fwr) && (tows2 < OW - dY_fwr);
		bool ldy3 = (tohs3 >= -dY_fhr) && (tohs3 < OH - dY_fhr) && (tows3 >= -dY_fwr) && (tows3 < OW - dY_fwr);
		int yoffset = dY_fhr * OW_OC + dY_k;
		dYs[buf][tx][ty].x = ldy0 * tex1Dfetch<float>(deltaY, Yoffset0 + yoffset);
		dYs[buf][tx][ty].y = ldy1 * tex1Dfetch<float>(deltaY, Yoffset1 + yoffset);
		dYs[buf][tx][ty].z = ldy2 * tex1Dfetch<float>(deltaY, Yoffset2 + yoffset);
		dYs[buf][tx][ty].w = ldy3 * tex1Dfetch<float>(deltaY, Yoffset3 + yoffset);
		__syncthreads();
	}
#pragma once
	for (int ik = 0; ik < STEP; ik++)
	{
		float4  w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];
		float4 dy0 = dYs[buf][ik][ty], dy1 = dYs[buf][ik + STEP][ty];

		simdMM4(v0, dy0.x, w0); simdMM4(v1, dy0.x, w1);
		simdMM4(v2, dy0.y, w0); simdMM4(v3, dy0.y, w1);
		simdMM4(v4, dy0.z, w0); simdMM4(v5, dy0.z, w1);
		simdMM4(v6, dy0.w, w0); simdMM4(v7, dy0.w, w1);
		simdMM4(v8, dy1.x, w0); simdMM4(v9, dy1.x, w1);
		simdMM4(v10, dy1.y, w0); simdMM4(v11, dy1.y, w1);
		simdMM4(v12, dy1.z, w0); simdMM4(v13, dy1.z, w1);
		simdMM4(v14, dy1.w, w0); simdMM4(v15, dy1.w, w1);
	}

	X_write8(deltaX, n0, ih0, iw0, ic0, v0, v1);
	X_write8(deltaX, n1, ih1, iw1, ic0, v2, v3);
	X_write8(deltaX, n2, ih2, iw2, ic0, v4, v5);
	X_write8(deltaX, n3, ih3, iw3, ic0, v6, v7);
	X_write8(deltaX, n4, ih4, iw4, ic0, v8, v9);
	X_write8(deltaX, n5, ih5, iw5, ic0, v10, v11);
	X_write8(deltaX, n6, ih6, iw6, ic0, v12, v13);
	X_write8(deltaX, n7, ih7, iw7, ic0, v14, v15);
}

#endif


//(8 * 8),
#ifndef X_KERNEL_V14
#define X_KERNEL_V14

#define X_kv14(stream, LB, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM) \
	X_kernel_v14<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>3), (GM>>LB>>3), (sh*sw)), dim3(1<<LB, 1<<LB) >>>\
			(deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw,\
			 CFH, CFW, IH_slice, IW_slice, GK)

//LB = 4: Size = 1.05579, Time = 1.592 msec, Performace = 1424.17 GFlop/s
//LB = 3: Size = 1.05579, Time = 1.264 msec, Performace = 1793.74 GFlop/s
template<int LB, int STEP>
__global__ void X_kernel_v14(
	cudaTextureObject_t deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw,
	int CFH, int CFW, int IH_slice, int IW_slice, int GK)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4  Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 dYs[2][1 << LB][(1 << LB) + 1];

	//prepare for GZ = sh*sw
	int bz = blockIdx.z;
	int y = bz / sw, x = bz - y * sw; 
	ph = ph - y; pw = pw - x;
	int ihs = -ph; ihs += (ihs < 0)*(ph + sh - 1) / sh * sh;
	int iws = -pw; iws += (iws < 0)*(pw + sw - 1) / sw * sw;

	//prepare for GN = IC
	int ic0 = ((blockIdx.x << LB) + tx) << 3;
	const int tic0 = ic0 + ((ty >= STEP) << 2);

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = ((blockIdx.y << LB) + ty) << 3;
	int j1 = j0 + 1, j2 = j0 + 2, j3 = j0 + 3;
	int j4 = j0 + 4, j5 = j0 + 5, j6 = j0 + 6, j7 = j0 + 7;
	int IH_IW_slice = IH_slice * IW_slice;
	X_n_ih_iw(j0, n0, ih0, iw0);
	X_n_ih_iw(j1, n1, ih1, iw1);
	X_n_ih_iw(j2, n2, ih2, iw2);
	X_n_ih_iw(j3, n3, ih3, iw3);
	X_n_ih_iw(j4, n4, ih4, iw4);
	X_n_ih_iw(j5, n5, ih5, iw5);
	X_n_ih_iw(j6, n6, ih6, iw6);
	X_n_ih_iw(j7, n7, ih7, iw7);
	ih0 = (ih0 * sh) + ihs, iw0 = (iw0 * sw) + iws;
	ih1 = (ih1 * sh) + ihs, iw1 = (iw1 * sw) + iws;
	ih2 = (ih2 * sh) + ihs, iw2 = (iw2 * sw) + iws;
	ih3 = (ih3 * sh) + ihs, iw3 = (iw3 * sw) + iws;
	ih4 = (ih4 * sh) + ihs, iw4 = (iw4 * sw) + iws;
	ih5 = (ih5 * sh) + ihs, iw5 = (iw5 * sw) + iws;
	ih6 = (ih6 * sh) + ihs, iw6 = (iw6 * sw) + iws;
	ih7 = (ih7 * sh) + ihs, iw7 = (iw7 * sw) + iws;
	bool flagY = (tx >= STEP);
	const int oph = CFH - 1, opw = CFW - 1; 
	const int tohs0 = (((ih4 - ih0)*flagY + ih0) + ph) / sh - oph;
	const int tohs1 = (((ih5 - ih1)*flagY + ih1) + ph) / sh - oph;
	const int tohs2 = (((ih6 - ih2)*flagY + ih2) + ph) / sh - oph;
	const int tohs3 = (((ih7 - ih3)*flagY + ih3) + ph) / sh - oph;
	const int tows0 = (((iw4 - iw0)*flagY + iw0) + pw) / sw - opw;
	const int tows1 = (((iw5 - iw1)*flagY + iw1) + pw) / sw - opw;
	const int tows2 = (((iw6 - iw2)*flagY + iw2) + pw) / sw - opw;
	const int tows3 = (((iw7 - iw3)*flagY + iw3) + pw) / sw - opw;
	int Yoffset0 = ((((n4 - n0)*flagY + n0)*OH + tohs0)*OW + tows0)*OC;
	int Yoffset1 = ((((n5 - n1)*flagY + n1)*OH + tohs1)*OW + tows1)*OC;
	int Yoffset2 = ((((n6 - n2)*flagY + n2)*OH + tohs2)*OW + tows2)*OC;
	int Yoffset3 = ((((n7 - n3)*flagY + n3)*OH + tohs3)*OW + tows3)*OC;

	const int CFW_OC = CFW * OC;

	//load 2 elem from W[OC, FH, FW, IC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);
	int W_fhr = W_k / CFW_OC; W_k -= W_fhr * CFW_OC;
	int W_fwr = W_k / OC;
	int fh = y + (CFH - 1 - W_fhr)*sh;
	int fw = x + (CFW - 1 - W_fwr)*sw;
	bool lw = (fh < FH) && (fw < FW);
	int Woffset = (((W_k - W_fwr * OC)*FH + fh)*FW + fw)*IC + tic0;
	Ws[buf][ty][tx] = (lw ? *(float4*)(W + Woffset) : FLOAT_ZERO4);

	//load 2 elem from deltaY[N, FH, FW, OC]
	int dY_k = tx - ((tx >= STEP) << LB >> 1);
	int dY_fhr = dY_k / CFW_OC; dY_k -= dY_fhr * CFW_OC;
	int dY_fwr = dY_k / OC;
	const int OW_OC = OW * OC, yoffset = dY_fhr * OW_OC + dY_k;
	bool ldy0 = (tohs0 >= -dY_fhr) && (tohs0 < OH - dY_fhr) && (tows0 >= -dY_fwr) && (tows0 < OW - dY_fwr);
	bool ldy1 = (tohs1 >= -dY_fhr) && (tohs1 < OH - dY_fhr) && (tows1 >= -dY_fwr) && (tows1 < OW - dY_fwr);
	bool ldy2 = (tohs2 >= -dY_fhr) && (tohs2 < OH - dY_fhr) && (tows2 >= -dY_fwr) && (tows2 < OW - dY_fwr);
	bool ldy3 = (tohs3 >= -dY_fhr) && (tohs3 < OH - dY_fhr) && (tows3 >= -dY_fwr) && (tows3 < OW - dY_fwr);
	dYs[buf][tx][ty].x = ldy0 * tex1Dfetch<float>(deltaY, Yoffset0 + yoffset);
	dYs[buf][tx][ty].y = ldy1 * tex1Dfetch<float>(deltaY, Yoffset1 + yoffset);
	dYs[buf][tx][ty].z = ldy2 * tex1Dfetch<float>(deltaY, Yoffset2 + yoffset);
	dYs[buf][tx][ty].w = ldy3 * tex1Dfetch<float>(deltaY, Yoffset3 + yoffset);
	__syncthreads();

	//compure_area----------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)
	{
#pragma once
		for (int ik = 0; ik < STEP; ik++)
		{
			float4  w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];
			float4 dy0 = dYs[buf][ik][ty], dy1 = dYs[buf][ik + STEP][ty];

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

		//load 1 elem from W[OC, FH, FW, IC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int W_fhr = W_k / CFW_OC; W_k -= W_fhr * CFW_OC;
		int W_fwr = W_k / OC;

		int fh = y + (CFH - 1 - W_fhr)*sh;
		int fw = x + (CFW - 1 - W_fwr)*sw;
		bool lw = (fh < FH) && (fw < FW);
		int Woffset = (((W_k - W_fwr * OC)*FH + fh)*FW + fw)*IC + tic0;
		Ws[buf][ty][tx] = (lw ? *(float4*)(W + Woffset) : FLOAT_ZERO4);

		//load 1 elem from deltaY[N, FH, FW, OC]
		int dY_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int dY_fhr = dY_k / CFW_OC; dY_k -= dY_fhr * CFW_OC;
		int dY_fwr = dY_k / OC; 
		int yoffset = dY_fhr * OW_OC + dY_k;
		bool ldy0 = (tohs0 >= -dY_fhr) && (tohs0 < OH - dY_fhr) && (tows0 >= -dY_fwr) && (tows0 < OW - dY_fwr);
		bool ldy1 = (tohs1 >= -dY_fhr) && (tohs1 < OH - dY_fhr) && (tows1 >= -dY_fwr) && (tows1 < OW - dY_fwr);
		bool ldy2 = (tohs2 >= -dY_fhr) && (tohs2 < OH - dY_fhr) && (tows2 >= -dY_fwr) && (tows2 < OW - dY_fwr);
		bool ldy3 = (tohs3 >= -dY_fhr) && (tohs3 < OH - dY_fhr) && (tows3 >= -dY_fwr) && (tows3 < OW - dY_fwr);
		dYs[buf][tx][ty].x = ldy0 * tex1Dfetch<float>(deltaY, Yoffset0 + yoffset);
		dYs[buf][tx][ty].y = ldy1 * tex1Dfetch<float>(deltaY, Yoffset1 + yoffset);
		dYs[buf][tx][ty].z = ldy2 * tex1Dfetch<float>(deltaY, Yoffset2 + yoffset);
		dYs[buf][tx][ty].w = ldy3 * tex1Dfetch<float>(deltaY, Yoffset3 + yoffset);
		__syncthreads();
	}
#pragma once
	for (int ik = 0; ik < STEP; ik++)
	{
		float4  w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];
		float4 dy0 = dYs[buf][ik][ty], dy1 = dYs[buf][ik + STEP][ty];

		simdMM4(v0, dy0.x, w0); simdMM4(v1, dy0.x, w1);
		simdMM4(v2, dy0.y, w0); simdMM4(v3, dy0.y, w1);
		simdMM4(v4, dy0.z, w0); simdMM4(v5, dy0.z, w1);
		simdMM4(v6, dy0.w, w0); simdMM4(v7, dy0.w, w1);
		simdMM4(v8, dy1.x, w0); simdMM4(v9, dy1.x, w1);
		simdMM4(v10, dy1.y, w0); simdMM4(v11, dy1.y, w1);
		simdMM4(v12, dy1.z, w0); simdMM4(v13, dy1.z, w1);
		simdMM4(v14, dy1.w, w0); simdMM4(v15, dy1.w, w1);
	}

	X_write8(deltaX, n0, ih0, iw0, ic0, v0, v1);
	X_write8(deltaX, n1, ih1, iw1, ic0, v2, v3);
	X_write8(deltaX, n2, ih2, iw2, ic0, v4, v5);
	X_write8(deltaX, n3, ih3, iw3, ic0, v6, v7);
	X_write8(deltaX, n4, ih4, iw4, ic0, v8, v9);
	X_write8(deltaX, n5, ih5, iw5, ic0, v10, v11);
	X_write8(deltaX, n6, ih6, iw6, ic0, v12, v13);
	X_write8(deltaX, n7, ih7, iw7, ic0, v14, v15);
}

#endif


//sh, sw = 2
#ifndef X_KERNEL_V15
#define X_KERNEL_V15

#define X_kv15(stream, LB, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM) \
	X_kernel_v15<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>3), (GM>>LB>>3), (sh*sw)), dim3(1<<LB, 1<<LB) >>>\
			(deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, ph, pw,\
			 CFH, CFW, IH_slice, IW_slice, GK)

//LB = 4: Size = 1.05579, Time = 1.586 msec, Performace = 1429.56 GFlop/s
//LB = 3: Size = 1.05579, Time = 1.264 msec, Performace = 1793.74 GFlop/s
template<int LB, int STEP>
__global__ void X_kernel_v15(
	cudaTextureObject_t deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int OC,
	int ph, int pw,
	int CFH, int CFW, int IH_slice, int IW_slice, int GK)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4  Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 dYs[2][1 << LB][(1 << LB) + 1];

	//prepare for GZ = sh*sw
	int y = blockIdx.z >> 1, x = blockIdx.z &1;
	ph = ph - y; pw = pw - x;
	int ihs = -ph; ihs += (ihs < 0)*((ph + 1) << 1 >> 1);
	int iws = -pw; iws += (iws < 0)*((pw + 1) << 1 >> 1);

	//prepare for GN = IC
	int ic0 = ((blockIdx.x << LB) + tx) << 3;
	const int tic0 = ic0 + ((ty >= STEP) << 2);

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = ((blockIdx.y << LB) + ty) << 3;
	int j1 = j0 + 1, j2 = j0 + 2, j3 = j0 + 3;
	int j4 = j0 + 4, j5 = j0 + 5, j6 = j0 + 6, j7 = j0 + 7;
	int IH_IW_slice = IH_slice * IW_slice;
	X_n_ih_iw(j0, n0, ih0, iw0);
	X_n_ih_iw(j1, n1, ih1, iw1);
	X_n_ih_iw(j2, n2, ih2, iw2);
	X_n_ih_iw(j3, n3, ih3, iw3);
	X_n_ih_iw(j4, n4, ih4, iw4);
	X_n_ih_iw(j5, n5, ih5, iw5);
	X_n_ih_iw(j6, n6, ih6, iw6);
	X_n_ih_iw(j7, n7, ih7, iw7);
	ih0 = (ih0 << 1) + ihs, iw0 = (iw0 << 1) + iws;
	ih1 = (ih1 << 1) + ihs, iw1 = (iw1 << 1) + iws;
	ih2 = (ih2 << 1) + ihs, iw2 = (iw2 << 1) + iws;
	ih3 = (ih3 << 1) + ihs, iw3 = (iw3 << 1) + iws;
	ih4 = (ih4 << 1) + ihs, iw4 = (iw4 << 1) + iws;
	ih5 = (ih5 << 1) + ihs, iw5 = (iw5 << 1) + iws;
	ih6 = (ih6 << 1) + ihs, iw6 = (iw6 << 1) + iws;
	ih7 = (ih7 << 1) + ihs, iw7 = (iw7 << 1) + iws;
	
	bool flagX = (tx >= STEP);
	const int oph = CFH - 1, opw = CFW - 1;
	const int tohs0 = (((ih4 - ih0)*flagX + ih0 + ph) >> 1) - oph;
	const int tohs1 = (((ih5 - ih1)*flagX + ih1 + ph) >> 1) - oph;
	const int tohs2 = (((ih6 - ih2)*flagX + ih2 + ph) >> 1) - oph;
	const int tohs3 = (((ih7 - ih3)*flagX + ih3 + ph) >> 1) - oph;
	const int tows0 = (((iw4 - iw0)*flagX + iw0 + pw) >> 1) - opw;
	const int tows1 = (((iw5 - iw1)*flagX + iw1 + pw) >> 1) - opw;
	const int tows2 = (((iw6 - iw2)*flagX + iw2 + pw) >> 1) - opw;
	const int tows3 = (((iw7 - iw3)*flagX + iw3 + pw) >> 1) - opw;
	int Yoffset0 = ((((n4 - n0)*flagX + n0)*OH + tohs0)*OW + tows0)*OC;
	int Yoffset1 = ((((n5 - n1)*flagX + n1)*OH + tohs1)*OW + tows1)*OC;
	int Yoffset2 = ((((n6 - n2)*flagX + n2)*OH + tohs2)*OW + tows2)*OC;
	int Yoffset3 = ((((n7 - n3)*flagX + n3)*OH + tohs3)*OW + tows3)*OC;

	const int CFW_OC = CFW * OC;

	//load 2 elem from W[OC, FH, FW, IC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);
	int W_fhr = W_k / CFW_OC; W_k -= W_fhr * CFW_OC;
	int W_fwr = W_k / OC;
	int fh = y + ((CFH - 1 - W_fhr) << 1);
	int fw = x + ((CFW - 1 - W_fwr) << 1);
	bool lw = (fh < FH) && (fw < FW);
	int Woffset = (((W_k - W_fwr * OC)*FH + fh)*FW + fw)*IC + tic0;
	Ws[buf][ty][tx] = (lw ? *(float4*)(W + Woffset) : FLOAT_ZERO4);

	//load 2 elem from deltaY[N, FH, FW, OC]
	int dY_k = tx - ((tx >= STEP) << LB >> 1);
	int dY_fhr = dY_k / CFW_OC; dY_k -= dY_fhr * CFW_OC;
	int dY_fwr = dY_k / OC;
	const int OW_OC = OW * OC, yoffset = dY_fhr * OW_OC + dY_k;
	bool ldy0 = (tohs0 >= -dY_fhr) && (tohs0 < OH - dY_fhr) && (tows0 >= -dY_fwr) && (tows0 < OW - dY_fwr);
	bool ldy1 = (tohs1 >= -dY_fhr) && (tohs1 < OH - dY_fhr) && (tows1 >= -dY_fwr) && (tows1 < OW - dY_fwr);
	bool ldy2 = (tohs2 >= -dY_fhr) && (tohs2 < OH - dY_fhr) && (tows2 >= -dY_fwr) && (tows2 < OW - dY_fwr);
	bool ldy3 = (tohs3 >= -dY_fhr) && (tohs3 < OH - dY_fhr) && (tows3 >= -dY_fwr) && (tows3 < OW - dY_fwr);
	dYs[buf][tx][ty].x = ldy0 * tex1Dfetch<float>(deltaY, Yoffset0 + yoffset);
	dYs[buf][tx][ty].y = ldy1 * tex1Dfetch<float>(deltaY, Yoffset1 + yoffset);
	dYs[buf][tx][ty].z = ldy2 * tex1Dfetch<float>(deltaY, Yoffset2 + yoffset);
	dYs[buf][tx][ty].w = ldy3 * tex1Dfetch<float>(deltaY, Yoffset3 + yoffset);
	__syncthreads();

	//compure_area----------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)
	{
#pragma once
		for (int ik = 0; ik < STEP; ik++)
		{
			float4  w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];
			float4 dy0 = dYs[buf][ik][ty], dy1 = dYs[buf][ik + STEP][ty];

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

		//load 1 elem from W[OC, FH, FW, IC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int W_fhr = W_k / CFW_OC; W_k -= W_fhr * CFW_OC;
		int W_fwr = W_k / OC;
		int fh = y + ((CFH - 1 - W_fhr) << 1);
		int fw = x + ((CFW - 1 - W_fwr) << 1);
		bool lw = (fh < FH) && (fw < FW);
		int Woffset = (((W_k - W_fwr * OC)*FH + fh)*FW + fw)*IC + tic0;
		Ws[buf][ty][tx] = (lw ? *(float4*)(W + Woffset) : FLOAT_ZERO4);

		//load 1 elem from deltaY[N, FH, FW, OC]
		int dY_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int dY_fhr = dY_k / CFW_OC; dY_k -= dY_fhr * CFW_OC;
		int dY_fwr = dY_k / OC;
		int yoffset = dY_fhr * OW_OC + dY_k;
		bool ldy0 = (tohs0 >= -dY_fhr) && (tohs0 < OH - dY_fhr) && (tows0 >= -dY_fwr) && (tows0 < OW - dY_fwr);
		bool ldy1 = (tohs1 >= -dY_fhr) && (tohs1 < OH - dY_fhr) && (tows1 >= -dY_fwr) && (tows1 < OW - dY_fwr);
		bool ldy2 = (tohs2 >= -dY_fhr) && (tohs2 < OH - dY_fhr) && (tows2 >= -dY_fwr) && (tows2 < OW - dY_fwr);
		bool ldy3 = (tohs3 >= -dY_fhr) && (tohs3 < OH - dY_fhr) && (tows3 >= -dY_fwr) && (tows3 < OW - dY_fwr);
		dYs[buf][tx][ty].x = ldy0 * tex1Dfetch<float>(deltaY, Yoffset0 + yoffset);
		dYs[buf][tx][ty].y = ldy1 * tex1Dfetch<float>(deltaY, Yoffset1 + yoffset);
		dYs[buf][tx][ty].z = ldy2 * tex1Dfetch<float>(deltaY, Yoffset2 + yoffset);
		dYs[buf][tx][ty].w = ldy3 * tex1Dfetch<float>(deltaY, Yoffset3 + yoffset);
		__syncthreads();
	}
#pragma once
	for (int ik = 0; ik < STEP; ik++)
	{
		float4  w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];
		float4 dy0 = dYs[buf][ik][ty], dy1 = dYs[buf][ik + STEP][ty];

		simdMM4(v0, dy0.x, w0); simdMM4(v1, dy0.x, w1);
		simdMM4(v2, dy0.y, w0); simdMM4(v3, dy0.y, w1);
		simdMM4(v4, dy0.z, w0); simdMM4(v5, dy0.z, w1);
		simdMM4(v6, dy0.w, w0); simdMM4(v7, dy0.w, w1);
		simdMM4(v8, dy1.x, w0); simdMM4(v9, dy1.x, w1);
		simdMM4(v10, dy1.y, w0); simdMM4(v11, dy1.y, w1);
		simdMM4(v12, dy1.z, w0); simdMM4(v13, dy1.z, w1);
		simdMM4(v14, dy1.w, w0); simdMM4(v15, dy1.w, w1);
	}

	X_write8(deltaX, n0, ih0, iw0, ic0, v0, v1);
	X_write8(deltaX, n1, ih1, iw1, ic0, v2, v3);
	X_write8(deltaX, n2, ih2, iw2, ic0, v4, v5);
	X_write8(deltaX, n3, ih3, iw3, ic0, v6, v7);
	X_write8(deltaX, n4, ih4, iw4, ic0, v8, v9);
	X_write8(deltaX, n5, ih5, iw5, ic0, v10, v11);
	X_write8(deltaX, n6, ih6, iw6, ic0, v12, v13);
	X_write8(deltaX, n7, ih7, iw7, ic0, v14, v15);
}

#endif


//(IH, IW) % (sh, sw) = 0
//sh, sw = 2
#ifndef X_KERNEL_V16
#define X_KERNEL_V16

#define X_kv16(stream, LB, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM) \
	X_kernel_v16<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>3), (GM>>LB>>3), (sh*sw)), dim3(1<<LB, 1<<LB) >>>\
			(deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, ph, pw,\
			 CFH, CFW, IH_slice, IW_slice, GK)

//LB = 4: Size = 1.125, Time = 1.218 msec, Performace = 1983.51 GFlop/s
//LB = 3: Size = 1.125, Time = 1.53  msec, Performace = 1579.03 GFlop/s
template<int LB, int STEP>
__global__ void X_kernel_v16(
	cudaTextureObject_t deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int OC,
	int ph, int pw,
	int CFH, int CFW, int IH_slice, int IW_slice, int GK)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4  Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 dYs[2][1 << LB][(1 << LB) + 1];

	//prepare for GZ = sh*sw
	int y = blockIdx.z >> 1, x = blockIdx.z & 1;
	ph = ph - y; pw = pw - x;
	int ihs = -ph; ihs += (ihs < 0)*((ph + 1) << 1 >> 1);
	int iws = -pw; iws += (iws < 0)*((pw + 1) << 1 >> 1);

	//prepare for GN = IC
	int ic0 = ((blockIdx.x << LB) + tx) << 3;
	const int tic0 = ic0 + ((ty >= STEP) << 2);

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = ((blockIdx.y << LB) + ty) << 3;
	int j1 = j0 + 1, j2 = j0 + 2, j3 = j0 + 3;
	int j4 = j0 + 4, j5 = j0 + 5, j6 = j0 + 6, j7 = j0 + 7;
	int IH_IW_slice = IH_slice * IW_slice;
	X_n_ih_iw(j0, n0, ih0, iw0);
	X_n_ih_iw(j1, n1, ih1, iw1);
	X_n_ih_iw(j2, n2, ih2, iw2);
	X_n_ih_iw(j3, n3, ih3, iw3);
	X_n_ih_iw(j4, n4, ih4, iw4);
	X_n_ih_iw(j5, n5, ih5, iw5);
	X_n_ih_iw(j6, n6, ih6, iw6);
	X_n_ih_iw(j7, n7, ih7, iw7);
	ih0 = (ih0 << 1) + ihs, iw0 = (iw0 << 1) + iws;
	ih1 = (ih1 << 1) + ihs, iw1 = (iw1 << 1) + iws;
	ih2 = (ih2 << 1) + ihs, iw2 = (iw2 << 1) + iws;
	ih3 = (ih3 << 1) + ihs, iw3 = (iw3 << 1) + iws;
	ih4 = (ih4 << 1) + ihs, iw4 = (iw4 << 1) + iws;
	ih5 = (ih5 << 1) + ihs, iw5 = (iw5 << 1) + iws;
	ih6 = (ih6 << 1) + ihs, iw6 = (iw6 << 1) + iws;
	ih7 = (ih7 << 1) + ihs, iw7 = (iw7 << 1) + iws;

	bool flagX = (tx >= STEP);
	const int oph = CFH - 1, opw = CFW - 1;
	const int tohs0 = (((ih4 - ih0)*flagX + ih0 + ph) >> 1) - oph;
	const int tohs1 = (((ih5 - ih1)*flagX + ih1 + ph) >> 1) - oph;
	const int tohs2 = (((ih6 - ih2)*flagX + ih2 + ph) >> 1) - oph;
	const int tohs3 = (((ih7 - ih3)*flagX + ih3 + ph) >> 1) - oph;
	const int tows0 = (((iw4 - iw0)*flagX + iw0 + pw) >> 1) - opw;
	const int tows1 = (((iw5 - iw1)*flagX + iw1 + pw) >> 1) - opw;
	const int tows2 = (((iw6 - iw2)*flagX + iw2 + pw) >> 1) - opw;
	const int tows3 = (((iw7 - iw3)*flagX + iw3 + pw) >> 1) - opw;
	int Yoffset0 = ((((n4 - n0)*flagX + n0)*OH + tohs0)*OW + tows0)*OC;
	int Yoffset1 = ((((n5 - n1)*flagX + n1)*OH + tohs1)*OW + tows1)*OC;
	int Yoffset2 = ((((n6 - n2)*flagX + n2)*OH + tohs2)*OW + tows2)*OC;
	int Yoffset3 = ((((n7 - n3)*flagX + n3)*OH + tohs3)*OW + tows3)*OC;

	const int CFW_OC = CFW * OC;

	//load 2 elem from W[OC, FH, FW, IC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);
	int W_fhr = W_k / CFW_OC; W_k -= W_fhr * CFW_OC;
	int W_fwr = W_k / OC;
	int fh = y + ((CFH - 1 - W_fhr) << 1);
	int fw = x + ((CFW - 1 - W_fwr) << 1);
	bool lw = (fh < FH) && (fw < FW);
	int Woffset = (((W_k - W_fwr * OC)*FH + fh)*FW + fw)*IC + tic0;
	Ws[buf][ty][tx] = (lw ? *(float4*)(W + Woffset) : FLOAT_ZERO4);

	//load 2 elem from deltaY[N, FH, FW, OC]
	int dY_k = tx - ((tx >= STEP) << LB >> 1);
	int dY_fhr = dY_k / CFW_OC; dY_k -= dY_fhr * CFW_OC;
	int dY_fwr = dY_k / OC;
	const int OW_OC = OW * OC, yoffset = dY_fhr * OW_OC + dY_k;
	bool ldy0 = (tohs0 >= -dY_fhr) && (tohs0 < OH - dY_fhr) && (tows0 >= -dY_fwr) && (tows0 < OW - dY_fwr);
	bool ldy1 = (tohs1 >= -dY_fhr) && (tohs1 < OH - dY_fhr) && (tows1 >= -dY_fwr) && (tows1 < OW - dY_fwr);
	bool ldy2 = (tohs2 >= -dY_fhr) && (tohs2 < OH - dY_fhr) && (tows2 >= -dY_fwr) && (tows2 < OW - dY_fwr);
	bool ldy3 = (tohs3 >= -dY_fhr) && (tohs3 < OH - dY_fhr) && (tows3 >= -dY_fwr) && (tows3 < OW - dY_fwr);
	dYs[buf][tx][ty].x = ldy0 * tex1Dfetch<float>(deltaY, Yoffset0 + yoffset);
	dYs[buf][tx][ty].y = ldy1 * tex1Dfetch<float>(deltaY, Yoffset1 + yoffset);
	dYs[buf][tx][ty].z = ldy2 * tex1Dfetch<float>(deltaY, Yoffset2 + yoffset);
	dYs[buf][tx][ty].w = ldy3 * tex1Dfetch<float>(deltaY, Yoffset3 + yoffset);
	__syncthreads();

	//compure_area----------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)
	{
#pragma once
		for (int ik = 0; ik < STEP; ik++)
		{
			float4  w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];
			float4 dy0 = dYs[buf][ik][ty], dy1 = dYs[buf][ik + STEP][ty];

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

		//load 1 elem from W[OC, FH, FW, IC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int W_fhr = W_k / CFW_OC; W_k -= W_fhr * CFW_OC;
		int W_fwr = W_k / OC;
		int fh = y + ((CFH - 1 - W_fhr) << 1);
		int fw = x + ((CFW - 1 - W_fwr) << 1);
		bool lw = (fh < FH) && (fw < FW);
		int Woffset = (((W_k - W_fwr * OC)*FH + fh)*FW + fw)*IC + tic0;
		Ws[buf][ty][tx] = (lw ? *(float4*)(W + Woffset) : FLOAT_ZERO4);

		//load 1 elem from deltaY[N, FH, FW, OC]
		int dY_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int dY_fhr = dY_k / CFW_OC; dY_k -= dY_fhr * CFW_OC;
		int dY_fwr = dY_k / OC;
		int yoffset = dY_fhr * OW_OC + dY_k;
		bool ldy0 = (tohs0 >= -dY_fhr) && (tohs0 < OH - dY_fhr) && (tows0 >= -dY_fwr) && (tows0 < OW - dY_fwr);
		bool ldy1 = (tohs1 >= -dY_fhr) && (tohs1 < OH - dY_fhr) && (tows1 >= -dY_fwr) && (tows1 < OW - dY_fwr);
		bool ldy2 = (tohs2 >= -dY_fhr) && (tohs2 < OH - dY_fhr) && (tows2 >= -dY_fwr) && (tows2 < OW - dY_fwr);
		bool ldy3 = (tohs3 >= -dY_fhr) && (tohs3 < OH - dY_fhr) && (tows3 >= -dY_fwr) && (tows3 < OW - dY_fwr);
		dYs[buf][tx][ty].x = ldy0 * tex1Dfetch<float>(deltaY, Yoffset0 + yoffset);
		dYs[buf][tx][ty].y = ldy1 * tex1Dfetch<float>(deltaY, Yoffset1 + yoffset);
		dYs[buf][tx][ty].z = ldy2 * tex1Dfetch<float>(deltaY, Yoffset2 + yoffset);
		dYs[buf][tx][ty].w = ldy3 * tex1Dfetch<float>(deltaY, Yoffset3 + yoffset);
		__syncthreads();
	}
#pragma once
	for (int ik = 0; ik < STEP; ik++)
	{
		float4  w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];
		float4 dy0 = dYs[buf][ik][ty], dy1 = dYs[buf][ik + STEP][ty];

		simdMM4(v0, dy0.x, w0); simdMM4(v1, dy0.x, w1);
		simdMM4(v2, dy0.y, w0); simdMM4(v3, dy0.y, w1);
		simdMM4(v4, dy0.z, w0); simdMM4(v5, dy0.z, w1);
		simdMM4(v6, dy0.w, w0); simdMM4(v7, dy0.w, w1);
		simdMM4(v8, dy1.x, w0); simdMM4(v9, dy1.x, w1);
		simdMM4(v10, dy1.y, w0); simdMM4(v11, dy1.y, w1);
		simdMM4(v12, dy1.z, w0); simdMM4(v13, dy1.z, w1);
		simdMM4(v14, dy1.w, w0); simdMM4(v15, dy1.w, w1);
	}

	int Xoffset0 = ((n0*IH + ih0)*IW + iw0)*IC + ic0; 
	int Xoffset1 = ((n1*IH + ih1)*IW + iw1)*IC + ic0;
	int Xoffset2 = ((n2*IH + ih2)*IW + iw2)*IC + ic0;
	int Xoffset3 = ((n3*IH + ih3)*IW + iw3)*IC + ic0;
	int Xoffset4 = ((n4*IH + ih4)*IW + iw4)*IC + ic0;
	int Xoffset5 = ((n5*IH + ih5)*IW + iw5)*IC + ic0;
	int Xoffset6 = ((n6*IH + ih6)*IW + iw6)*IC + ic0;
	int Xoffset7 = ((n7*IH + ih7)*IW + iw7)*IC + ic0;
	*(float4*)(deltaX + Xoffset0) = v0;  *(float4*)(deltaX + Xoffset0 + 4) = v1;
	*(float4*)(deltaX + Xoffset1) = v2;  *(float4*)(deltaX + Xoffset1 + 4) = v3;
	*(float4*)(deltaX + Xoffset2) = v4;  *(float4*)(deltaX + Xoffset2 + 4) = v5;
	*(float4*)(deltaX + Xoffset3) = v6;  *(float4*)(deltaX + Xoffset3 + 4) = v7;
	*(float4*)(deltaX + Xoffset4) = v8;  *(float4*)(deltaX + Xoffset4 + 4) = v9;
	*(float4*)(deltaX + Xoffset5) = v10; *(float4*)(deltaX + Xoffset5 + 4) = v11;
	*(float4*)(deltaX + Xoffset6) = v12; *(float4*)(deltaX + Xoffset6 + 4) = v13;
	*(float4*)(deltaX + Xoffset7) = v14; *(float4*)(deltaX + Xoffset7 + 4) = v15;
}

#endif