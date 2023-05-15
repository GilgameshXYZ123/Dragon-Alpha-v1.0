


#ifndef X_KERNEL_V1
#define X_KERNEL_V1

#define X_kv1(stream, LB, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM) \
	ksn_kernel_v1<LB>\
		<<< dim3(GM>>LB, GN>>LB, (sh*sw)), dim3(1<<LB, 1<<LB) >>>\
			(deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw)

//LB = 4: Size = 0.0703125, Time = 2.5 msec, Performace = 60.398 GFlop/s
template<int LB>
__global__ void ksn_kernel_v1(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw)
{
	//prepare for GK(bz(y, x)) = CFW * CFH * OC
	int bz = blockIdx.z;
	int y = bz / sw, x = bz % sw;

	int CFH = (FH - y + sh - 1) / sh, oph = CFH - 1;
	int CFW = (FW - x + sw - 1) / sw, opw = CFW - 1;
	int GK = CFH * CFW * OC;

	int ihs = (y - ph); if (ihs < 0) ihs += (ph - y + sh - 1) / sh * sh;
	int iws = (x - pw); if (iws < 0) iws += (pw - x + sw - 1) / sw * sw;

	//prepare for GN = IC
	int ic = (blockIdx.y << LB) + threadIdx.y;

	//prepare_for_GM = N*IH_slice*IW_slice
	int IH_slice = (IH + sh - 1) / sh;
	int IW_slice = (IW + sw - 1) / sw;
	int IH_IW_slice = IH_slice * IW_slice;

	int j = (blockIdx.x << LB) + threadIdx.x;
	int n = j / IH_IW_slice, jr = j % IH_IW_slice;
	int ih = jr / IW_slice, iw = jr % IW_slice;

	ih = ih * sh + ihs;
	iw = iw * sw + iws;
	
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


	bool wrt = (ih >= 0) && (iw >= 0) && (ih < IH) && (iw < IW);
	if (wrt) get4d(deltaX, n, ih, iw, ic, IH, IW, IC) = dx;
}

#endif


//(IH % sh == 0) && (IW % sw == 0)
#ifndef X_KERNEL_V2
#define X_KERNEL_V2

#define X_kv2(stream, LB, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM) \
	ksn_kernel_v2<LB>\
		<<< dim3(GN>>LB, GM>>LB, (sh*sw)), dim3(1<<LB, 1<<LB) >>>\
			(deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw)

//LB = 4: Size = 0.140625, Time = 4.78 msec, Performace = 63.1778 GFlop/s
template<int LB>
__global__ void ksn_kernel_v2(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw)
{
	//prepare for GK(bz(y, x)) = CFW * CFH * OC
	int bz = blockIdx.z;
	int y = bz / sw, x = bz % sw;

	int CFH = (FH - y + sh - 1) / sh, oph = CFH - 1;
	int CFW = (FW - x + sw - 1) / sw, opw = CFW - 1;
	int GK = CFH * CFW * OC;

	int ihs = (y - ph); if (ihs < 0) ihs += (ph - y + sh - 1) / sh * sh;
	int iws = (x - pw); if (iws < 0) iws += (pw - x + sw - 1) / sw * sw;

	//prepare for GN = IC
	int ic = (blockIdx.x << LB) + threadIdx.x;

	//prepare_for_GM = N*IH_slice*IW_slice
	int IH_slice = (IH + sh - 1) / sh;
	int IW_slice = (IW + sw - 1) / sw;
	int IH_IW_slice = IH_slice * IW_slice;

	int j = (blockIdx.y << LB) + threadIdx.y;
	int n = j / IH_IW_slice, jr = j % IH_IW_slice;
	int ih = jr / IW_slice, iw = jr % IW_slice;

	ih = ih * sh + ihs;
	iw = iw * sw + iws;

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

	get4d(deltaX, n, ih, iw, ic, IH, IW, IC) = dx;
}

#endif


//(IH % sh == 0) && (IW % sw == 0)
#ifndef X_KERNEL_V3
#define X_KERNEL_V3

#define X_kv3(stream, LB, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM) \
	ksn_kernel_v3<LB>\
		<<< dim3(GN>>LB, GM>>LB, (sh*sw)), dim3(1<<LB, 1<<LB) >>>\
			(deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw)

//LB = 4: Size = 0.140625, Time = 4.76 msec, Performace = 63.4432 GFlop/s
template<int LB>
__global__ void ksn_kernel_v3(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw)
{
	//prepare for GK(bz(y, x)) = CFW * CFH * OC
	int bz = blockIdx.z;
	int y = bz / sw, x = bz - y * sw;//x = bz % sw

	int CFH = (FH - y + sh - 1) / sh, oph = CFH - 1;
	int CFW = (FW - x + sw - 1) / sw, opw = CFW - 1;
	int GK = CFH * CFW * OC;

	ph = ph - y; pw = pw - x;
	int ihs = -ph + (-(ph > 0) & ((ph - y + sh - 1) / sh * sh));
	int iws = -pw + (-(pw > 0) & ((pw - x + sw - 1) / sw * sw));

	//prepare for GN = IC
	int ic = (blockIdx.x << LB) + threadIdx.x;

	//prepare_for_GM = N*IH_slice*IW_slice
	int IH_slice = (IH + sh - 1) / sh;
	int IW_slice = (IW + sw - 1) / sw;
	int IH_IW_slice = IH_slice * IW_slice;

	int j = (blockIdx.y << LB) + threadIdx.y;
	int n = j / IH_IW_slice, jr = j - n * IH_IW_slice;//jr = j % (IH_slice * IW_slice)
	int ih = jr / IW_slice, iw = jr - ih * IW_slice;//iw = jr % IW_slice

	ih = ih * sh + ihs;
	iw = iw * sw + iws;

	int ohs = (ih + ph) / sh - oph;
	int ows = (iw + pw) / sw - opw;

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

	get4d(deltaX, n, ih, iw, ic, IH, IW, IC) = dx;
}

#endif


//(IH % sh == 0) && (IW % sw == 0)
#ifndef X_KERNEL_V4
#define X_KERNEL_V4

#define X_kv4(stream, LB, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM) \
	ksn_kernel_v4<LB>\
		<<< dim3(GN>>LB, GM>>LB, (sh*sw)), dim3(1<<LB, 1<<LB) >>>\
			(deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw)

//LB = 4: Size = 0.140625, Time = 4.76 msec, Performace = 63.4432 GFlop/s
template<int LB>
__global__ void ksn_kernel_v4(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw)
{
	//prepare for GK(bz(y, x)) = CFW * CFH * OC
	int bz = blockIdx.z;
	int y = bz / sw, x = bz - y * sw;//x = bz % sw

	int CFH = (FH - y + sh - 1) / sh, oph = CFH - 1;
	int CFW = (FW - x + sw - 1) / sw, opw = CFW - 1;
	int GK = CFH * CFW * OC;

	ph = ph - y; pw = pw - x;
	int ihs = -ph + (-(ph > 0) & ((ph - y + sh - 1) / sh * sh));
	int iws = -pw + (-(pw > 0) & ((pw - x + sw - 1) / sw * sw));

	//prepare for GN = IC
	int ic = (blockIdx.x << LB) + threadIdx.x;

	//prepare_for_GM = N*IH_slice*IW_slice
	int IH_slice = (IH + sh - 1) / sh;
	int IW_slice = (IW + sw - 1) / sw;
	int IH_IW_slice = IH_slice * IW_slice;

	int j = (blockIdx.y << LB) + threadIdx.y;
	int n = j / IH_IW_slice, jr = j - n * IH_IW_slice;//jr = j % (IH_slice * IW_slice)
	int ih = jr / IW_slice, iw = jr - ih * IW_slice;//iw = jr % IW_slice

	ih = ih * sh + ihs;
	iw = iw * sw + iws;

	int ohs = (ih + ph) / sh - oph;
	int ows = (iw + pw) / sw - opw;

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

		//As GK is compressed: we have (fh < FH && FW < fw)
		int fh = y + (CFH - 1 - fhr)*sh;
		int fw = x + (CFW - 1 - fwr)*sw;
		float w = get4d(W, oc, fh, fw, ic, FH, FW, IC);

		dx += w * dy;
	}

	get4d(deltaX, n, ih, iw, ic, IH, IW, IC) = dx;
}

#endif


//(IH % sh == 0) && (IW % sw == 0), shared_memory
#ifndef X_KERNEL_V5
#define X_KERNEL_V5

#define X_kv5(stream, LB, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM) \
	ksn_kernel_v5<LB, (1<<LB)>\
		<<< dim3(GN>>LB, GM>>LB, (sh*sw)), dim3(1<<LB, 1<<LB) >>>\
			(deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw)

//LB = 4: Size = 0.140625, Time = 0.84 msec, Performace = 359.512 GFlop/s
template<int LB, int STEP>
__global__ void ksn_kernel_v5(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	__shared__ float Ws[1 << LB][(1 << LB) + 1];
	__shared__ float Ys[1 << LB][(1 << LB) + 1];

	//prepare for GK(bz(y, x)) = CFW * CFH * OC
	int bz = blockIdx.z;
	int y = bz / sw, x = bz - y * sw;//x = bz % sw
	int CFH = (FH - y + sh - 1) / sh, oph = CFH - 1;
	int CFW = (FW - x + sw - 1) / sw, opw = CFW - 1;
	const int CFH_CFW = CFH * CFW, GK = CFH_CFW * OC;

	ph = ph - y; pw = pw - x;
	int ihs = -ph + (-(ph > 0) & ((ph - y + sh - 1) / sh * sh));
	int iws = -pw + (-(pw > 0) & ((pw - x + sw - 1) / sw * sw));

	//prepare for GN = IC
	int ic = (blockIdx.x << LB) + tx;

	//prepare_for_GM = N*IH_slice*IW_slice
	int IH_slice = (IH + sh - 1) / sh;
	int IW_slice = (IW + sw - 1) / sw;
	int IH_IW_slice = IH_slice * IW_slice;

	int j = (blockIdx.y << LB) + ty;
	int n = j / IH_IW_slice, jr = j - n * IH_IW_slice;//jr = j % (IH_slice * IW_slice)
	int ih = jr / IW_slice, iw = jr - ih * IW_slice;//iw = jr % IW_slice

	ih = ih * sh + ihs;
	iw = iw * sw + iws;

	int ohs = (ih + ph) / sh - oph;
	int ows = (iw + pw) / sw - opw;
	
	float v = 0;
	for (int ok = 0, OK = (GK >> LB); ok < OK; ok++)
	{
		int W_k = (ok << LB) + ty;
		Ims_oc_fhr_fwr(W_k, W_oc, W_fhr, W_fwr);
		int fh = y + (oph - W_fhr)*sh;
		int fw = x + (opw - W_fwr)*sw;
		Ws[ty][tx] = get4d(W, W_oc, fh, fw, ic, FH, FW, IC);

		int Y_k = (ok << LB) + tx;
		Ims_oc_fhr_fwr(Y_k, Y_oc, Y_fhr, Y_fwr);
		int oh = ohs + Y_fhr;
		int ow = ows + Y_fwr;
		bool ldy = (oh >= 0) && (oh < OH) && (ow >= 0) && (ow < OW);
		Ys[tx][ty] = (ldy ? get4d(deltaY, n, oh, ow, Y_oc, OH, OW, OC) : 0);
		__syncthreads();

#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float y = Ys[ik][ty];
			float w = Ws[ik][tx];
			v += y * w;
		}
		__syncthreads();
	}

	get4d(deltaX, n, ih, iw, ic, IH, IW, IC) = v;
}

#endif


//(IH % sh == 0) && (IW % sw == 0), double buffered shared_memory
#ifndef X_KERNEL_V6
#define X_KERNEL_V6

#define X_kv6(stream, LB, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM) \
	ksn_kernel_v6<LB, (1<<LB)>\
		<<< dim3(GN>>LB, GM>>LB, (sh*sw)), dim3(1<<LB, 1<<LB) >>>\
			(deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw)

//LB = 4: Size = 0.5625, Time = 3.26 msec, Performace = 370.54 GFlop/s
template<int LB, int STEP>
__global__ void ksn_kernel_v6(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float Ys[2][1 << LB][(1 << LB) + 1];

	//prepare for GK(bz(y, x)) = CFW * CFH * OC
	int bz = blockIdx.z;
	int y = bz / sw, x = bz - y * sw;//x = bz % sw
	int CFH = (FH - y + sh - 1) / sh, oph = CFH - 1;
	int CFW = (FW - x + sw - 1) / sw, opw = CFW - 1;
	const int CFH_CFW = CFH * CFW, GK = CFH_CFW * OC;

	ph = ph - y; pw = pw - x;
	int ihs = -ph + (-(ph > 0) & ((ph - y + sh - 1) / sh * sh));
	int iws = -pw + (-(pw > 0) & ((pw - x + sw - 1) / sw * sw));

	//prepare for GN = IC
	int ic = (blockIdx.x << LB) + tx;

	//prepare_for_GM = N*IH_slice*IW_slice
	int IH_slice = (IH + sh - 1) / sh;
	int IW_slice = (IW + sw - 1) / sw;
	int IH_IW_slice = IH_slice * IW_slice;

	int j = (blockIdx.y << LB) + ty;
	int n = j / IH_IW_slice, jr = j - n * IH_IW_slice;//jr = j % (IH_slice * IW_slice)
	int ih = jr / IW_slice, iw = jr - ih * IW_slice;//iw = jr % IW_slice

	ih = ih * sh + ihs;
	iw = iw * sw + iws;

	int ohs = (ih + ph) / sh - oph;
	int ows = (iw + pw) / sw - opw;

	//load 1 elem from W
	int W_k = ty;
	Ims_oc_fhr_fwr(W_k, W_oc, W_fhr, W_fwr);
	int fh = y + (oph - W_fhr)*sh;
	int fw = x + (opw - W_fwr)*sw;
	Ws[buf][ty][tx] = get4d(W, W_oc, fh, fw, ic, FH, FW, IC);

	//load 1 elem from deltaY
	int Y_k = tx;
	Ims_oc_fhr_fwr(Y_k, Y_oc, Y_fhr, Y_fwr);
	int oh = ohs + Y_fhr;
	int ow = ows + Y_fwr;
	bool ldy = (oh >= 0) && (oh < OH) && (ow >= 0) && (ow < OW);
	Ys[buf][tx][ty] = (ldy ? get4d(deltaY, n, oh, ow, Y_oc, OH, OW, OC) : 0);
	__syncthreads();

	//compute area-------------------------------------------------
	float v = 0;
	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float y = Ys[buf][ik][ty];
			float w = Ws[buf][ik][tx];
			v += y * w;
		}
		buf ^= 1;

		//load 1 elem from W
		int W_k = (ok << LB) + ty;
		Ims_oc_fhr_fwr(W_k, W_oc, W_fhr, W_fwr);
		int fh = y + (oph - W_fhr)*sh;
		int fw = x + (opw - W_fwr)*sw;
		Ws[buf][ty][tx] = get4d(W, W_oc, fh, fw, ic, FH, FW, IC);

		//load 1 elem from Y
		int Y_k = (ok << LB) + tx;
		Ims_oc_fhr_fwr(Y_k, Y_oc, Y_fhr, Y_fwr);
		int oh = ohs + Y_fhr;
		int ow = ows + Y_fwr;
		bool ldy = (oh >= 0) && (oh < OH) && (ow >= 0) && (ow < OW);
		Ys[buf][tx][ty] = (ldy ? get4d(deltaY, n, oh, ow, Y_oc, OH, OW, OC) : 0);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float y = Ys[buf][ik][ty];
		float w = Ws[buf][ik][tx];
		v += y * w;
	}

	get4d(deltaX, n, ih, iw, ic, IH, IW, IC) = v;
}

#endif


//(BLOCK_SIZE*2, BLOCK_SIZE*2)
//(IH % sh == 0) && (IW % sw == 0), double buffered shared_memory
#ifndef X_KERNEL_V7
#define X_KERNEL_V7

#define X_kv7(stream, LB, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM) \
	ksn_kernel_v7<LB, (1<<LB)>\
		<<< dim3((GN>>LB>>1), (GM>>LB>>1), (sh*sw)), dim3(1<<LB, 1<<LB) >>>\
			(deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw)

//LB = 4: Size = 1.125, Time = 2.06 msec, Performace = 1172.78 GFlop/s
template<int LB, int STEP>
__global__ void ksn_kernel_v7(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float2 Ys[2][1 << LB][(1 << LB) + 1];

	//prepare for GK(bz(y, x)) = CFW * CFH * OC
	int bz = blockIdx.z;
	int y = bz / sw, x = bz - y * sw;//x = bz % sw
	int CFH = (FH - y + sh - 1) / sh, oph = CFH - 1;
	int CFW = (FW - x + sw - 1) / sw, opw = CFW - 1;
	const int CFH_CFW = CFH * CFW, GK = CFH_CFW * OC;

	ph = ph - y; pw = pw - x;
	int ihs = -ph + (-(ph > 0) & ((ph - y + sh - 1) / sh * sh));
	int iws = -pw + (-(pw > 0) & ((pw - x + sw - 1) / sw * sw));

	//prepare for GN = IC
	int ic0 = ((blockIdx.x << LB) + tx) << 1;

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = ((blockIdx.y << LB) + ty) << 1, j1 = j0 + 1;
	int IW_slice = IW / sw;
	int IH_IW_slice = (IH / sh)* IW_slice;
	Ims_n_ih_iw(j0, n0, ih0, iw0);
	Ims_n_ih_iw(j1, n1, ih1, iw1);

	int ohs0 = (ih0 + ph) / sh - oph, ows0 = (iw0 + pw) / sw - opw;
	int ohs1 = (ih1 + ph) / sh - oph, ows1 = (iw1 + pw) / sw - opw;

	//load 2 elem from W
	int W_k = ty;
	Ims_oc_fhr_fwr(W_k, W_oc, W_fhr, W_fwr);
	int fh = y + (oph - W_fhr)*sh;
	int fw = x + (opw - W_fwr)*sw;
	Ws[buf][ty][tx] = *(float2*)(&get4d(W, W_oc, fh, fw, ic0, FH, FW, IC));

	//load 2 elem from deltaY
	int Y_k = tx;
	Ims_oc_fhr_fwr(Y_k, Y_oc, Y_fhr, Y_fwr);
	int oh0 = ohs0 + Y_fhr, ow0 = ows0 + Y_fwr;
	int oh1 = ohs1 + Y_fhr, ow1 = ows1 + Y_fwr;
	bool ly0 = (oh0 >= 0) && (oh0 < OH) && (ow0 >= 0) && (ow0 < OW);
	bool ly1 = (oh1 >= 0) && (oh1 < OH) && (ow1 >= 0) && (ow1 < OW);
	Ys[buf][tx][ty].x = (ly0 ? get4d(deltaY, n0, oh0, ow0, Y_oc, OH, OW, OC) : 0);
	Ys[buf][tx][ty].y = (ly1 ? get4d(deltaY, n1, oh1, ow1, Y_oc, OH, OW, OC) : 0);
	__syncthreads();

	//compute area-------------------------------------------------
	float2 v0 = make_float2(0, 0);
	float2 v1 = make_float2(0, 0);
	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float2 y = Ys[buf][ik][ty];
			float2 w = Ws[buf][ik][tx];
			simdMM2(v0, y.x, w);
			simdMM2(v1, y.y, w);
		}
		buf ^= 1;

		//load 2 elem from W
		int W_k = (ok << LB) + ty;
		Ims_oc_fhr_fwr(W_k, W_oc, W_fhr, W_fwr);
		int fh = y + (oph - W_fhr)*sh;
		int fw = x + (opw - W_fwr)*sw;
		Ws[buf][ty][tx] = *(float2*)(&get4d(W, W_oc, fh, fw, ic0, FH, FW, IC));

		//load 2 elem from Y
		int Y_k = (ok << LB) + tx;
		Ims_oc_fhr_fwr(Y_k, Y_oc, Y_fhr, Y_fwr);
		int oh0 = ohs0 + Y_fhr, ow0 = ows0 + Y_fwr;
		int oh1 = ohs1 + Y_fhr, ow1 = ows1 + Y_fwr;
		bool ly0 = (oh0 >= 0) && (oh0 < OH) && (ow0 >= 0) && (ow0 < OW);
		bool ly1 = (oh1 >= 0) && (oh1 < OH) && (ow1 >= 0) && (ow1 < OW);
		Ys[buf][tx][ty].x = (ly0 ? get4d(deltaY, n0, oh0, ow0, Y_oc, OH, OW, OC) : 0);
		Ys[buf][tx][ty].y = (ly1 ? get4d(deltaY, n1, oh1, ow1, Y_oc, OH, OW, OC) : 0);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float2 y = Ys[buf][ik][ty];
		float2 w = Ws[buf][ik][tx];
		simdMM2(v0, y.x, w);
		simdMM2(v1, y.y, w);
	}

	*(float2*)(&get4d(deltaX, n0, ih0, iw0, ic0, IH, IW, IC)) = v0;
	*(float2*)(&get4d(deltaX, n1, ih1, iw1, ic0, IH, IW, IC)) = v1;
}

#endif


//(BLOCK_SIZE*2, BLOCK_SIZE*2)
//(IH % sh == 0) && (IW % sw == 0), double buffered shared_memory
#ifndef X_KERNEL_V8
#define X_KERNEL_V8

#define X_kv8(stream, LB, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM) \
	ksn_kernel_v8<LB, (1<<LB)>\
		<<< dim3((GN>>LB>>1), (GM>>LB>>1), (sh*sw)), dim3(1<<LB, 1<<LB) >>>\
			(deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw)

//LB = 4: Size = 1.125, Time = 2.06 msec, Performace = 1172.78 GFlop/s
template<int LB, int STEP>
__global__ void ksn_kernel_v8(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float2 Ys[2][1 << LB][(1 << LB) + 1];

	//prepare for GK(bz(y, x)) = CFW * CFH * OC
	int bz = blockIdx.z;
	int y = bz / sw, x = bz - y * sw;//x = bz % sw
	int CFH = (FH - y + sh - 1) / sh, oph = CFH - 1;
	int CFW = (FW - x + sw - 1) / sw, opw = CFW - 1;
	const int CFH_CFW = CFH * CFW, GK = CFH_CFW * OC;

	ph = ph - y; pw = pw - x;
	int ihs = -ph + (-(ph > 0) & ((ph - y + sh - 1) / sh * sh));
	int iws = -pw + (-(pw > 0) & ((pw - x + sw - 1) / sw * sw));

	//prepare for GN = IC
	int ic0 = ((blockIdx.x << LB) + tx) << 1;
	deltaX += ic0;

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = ((blockIdx.y << LB) + ty) << 1, j1 = j0 + 1;
	int IW_slice = IW / sw;
	int IH_IW_slice = (IH / sh)* IW_slice;
	Ims_n_ih_iw(j0, n0, ih0, iw0);
	Ims_n_ih_iw(j1, n1, ih1, iw1);
	int X0 = ((n0*IH + ih0)*IW + iw0) * IC;
	int X1 = ((n1*IH + ih1)*IW + iw1) * IC;

	int ohs0 = (ih0 + ph) / sh - oph, ows0 = (iw0 + pw) / sw - opw;
	int ohs1 = (ih1 + ph) / sh - oph, ows1 = (iw1 + pw) / sw - opw;
	int Y0 = ((n0*OH + ohs0)*OW + ows0)*OC;
	int Y1 = ((n1*OH + ohs1)*OW + ows1)*OC;

	//load 2 elem from W
	int W_k = ty;
	Ims_oc_fhr_fwr(W_k, W_oc, W_fhr, W_fwr);
	int fh = y + (oph - W_fhr)*sh;
	int fw = x + (opw - W_fwr)*sw;
	Ws[buf][ty][tx] = *(float2*)(&get4d(W, W_oc, fh, fw, ic0, FH, FW, IC));

	//load 2 elem from deltaY
	int Y_k = tx;
	Ims_oc_fhr_fwr(Y_k, Y_oc, Y_fhr, Y_fwr);
	bool ly0 = (ohs0 >= -Y_fhr) && (ohs0 < OH - Y_fhr) && (ows0 >= -Y_fwr) && (ows0 < OW - Y_fwr);
	bool ly1 = (ohs1 >= -Y_fhr) && (ohs1 < OH - Y_fhr) && (ows1 >= -Y_fwr) && (ows1 < OW - Y_fwr);
	int yoffset = ((Y_fhr * OW) + Y_fwr)*OC + Y_oc;
	Ys[buf][tx][ty].x = (ly0 ? deltaY[Y0 + yoffset] : 0);
	Ys[buf][tx][ty].y = (ly1 ? deltaY[Y1 + yoffset] : 0);
	__syncthreads();

	//compute area-------------------------------------------------
	float2 v0 = make_float2(0, 0);
	float2 v1 = make_float2(0, 0);
	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float2 y = Ys[buf][ik][ty];
			float2 w = Ws[buf][ik][tx];
			simdMM2(v0, y.x, w);
			simdMM2(v1, y.y, w);
		}
		buf ^= 1;

		//load 2 elem from W
		int W_k = (ok << LB) + ty;
		Ims_oc_fhr_fwr(W_k, W_oc, W_fhr, W_fwr);
		int fh = y + (oph - W_fhr)*sh;
		int fw = x + (opw - W_fwr)*sw;
		Ws[buf][ty][tx] = *(float2*)(&get4d(W, W_oc, fh, fw, ic0, FH, FW, IC));

		//load 2 elem from Y
		int Y_k = (ok << LB) + tx;
		Ims_oc_fhr_fwr(Y_k, Y_oc, Y_fhr, Y_fwr);
		bool ly0 = (ohs0 >= -Y_fhr) && (ohs0 < OH - Y_fhr) && (ows0 >= -Y_fwr) && (ows0 < OW - Y_fwr);
		bool ly1 = (ohs1 >= -Y_fhr) && (ohs1 < OH - Y_fhr) && (ows1 >= -Y_fwr) && (ows1 < OW - Y_fwr);
		int yoffset = ((Y_fhr * OW) + Y_fwr)*OC + Y_oc;
		Ys[buf][tx][ty].x = (ly0 ? deltaY[Y0 + yoffset] : 0);
		Ys[buf][tx][ty].y = (ly1 ? deltaY[Y1 + yoffset] : 0);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float2 y = Ys[buf][ik][ty];
		float2 w = Ws[buf][ik][tx];
		simdMM2(v0, y.x, w);
		simdMM2(v1, y.y, w);
	}

	*(float2*)(deltaX + X0) = v0;
	*(float2*)(deltaX + X1) = v1;
}

#endif


//(BLOCK_SIZE*4, BLOCK_SIZE*4)
//(IH % sh == 0) && (IW % sw == 0), double buffered shared_memory
#ifndef X_KERNEL_V9
#define X_KERNEL_V9

#define X_kv9(stream, LB, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM) \
	ksn_kernel_v9<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>2), (GM>>LB>>2), (sh*sw)), dim3(1<<LB, 1<<LB) >>>\
			(deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw)

//LB = 4: Size = 1.125, Time = 1.07333 msec, Performace = 2250.86 GFlop/s
template<int LB, int STEP>
__global__ void ksn_kernel_v9(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2 Ys[2][1 << LB >> 1][(2 << LB) + 2];

	//prepare for GK(bz(y, x)) = CFW * CFH * OC
	int bz = blockIdx.z;
	int y = bz / sw, x = bz - y * sw;//x = bz % sw
	int CFH = (FH - y + sh - 1) / sh, oph = CFH - 1;
	int CFW = (FW - x + sw - 1) / sw, opw = CFW - 1;
	const int CFH_CFW = CFH * CFW, GK = CFH_CFW * OC;

	ph = ph - y; pw = pw - x;
	int ihs = -ph + (-(ph > 0) & ((ph - y + sh - 1) / sh * sh));
	int iws = -pw + (-(pw > 0) & ((pw - x + sw - 1) / sw * sw));

	//prepare for GN = IC
	int ic0 = ((blockIdx.x << LB) + tx) << 2;
	int tic0 = ic0 + ((ty & 1) << 1);
	deltaX += ic0;

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = ((blockIdx.y << LB) + ty) << 2;
	int j1 = j0 + 1, j2 = j0 + 2, j3 = j0 + 3;
	int IW_slice = IW / sw, IH_IW_slice = (IH / sh) * IW_slice;
	Ims_n_ih_iw(j0, n0, ih0, iw0);
	Ims_n_ih_iw(j1, n1, ih1, iw1);
	Ims_n_ih_iw(j2, n2, ih2, iw2);
	Ims_n_ih_iw(j3, n3, ih3, iw3);
	int X0 = ((n0*IH + ih0)*IW + iw0) * IC;
	int X1 = ((n1*IH + ih1)*IW + iw1) * IC;
	int X2 = ((n2*IH + ih2)*IW + iw2) * IC;
	int X3 = ((n3*IH + ih3)*IW + iw3) * IC;

	bool flagX = (tx & 1);
	int tohs0 = ((IF_int(flagX, ih2, ih0) + ph) >> 1) - oph;
	int tohs1 = ((IF_int(flagX, ih3, ih1) + ph) >> 1) - oph;
	int tows0 = ((IF_int(flagX, iw2, iw0) + pw) >> 1) - opw;
	int tows1 = ((IF_int(flagX, iw3, iw1) + pw) >> 1) - opw;
	int Y0 = ((IF_int(flagX, n2, n0)*OH + tohs0)*OW + tows0) * OC;
	int Y1 = ((IF_int(flagX, n3, n1)*OH + tohs1)*OW + tows1) * OC;

	//load 2 elem from W
	int W_k = ty >> 1;
	Ims_oc_fhr_fwr(W_k, W_oc, W_fhr, W_fwr);
	int fh = y + (oph - W_fhr)*sh;
	int fw = x + (opw - W_fwr)*sw;
	const int Ws_y = (ty >> 1), Ws_x = (tx << 1) + (ty & 1);
	Ws[buf][Ws_y][Ws_x] = *(float2*)(&get4d(W, W_oc, fh, fw, tic0, FH, FW, IC));

	//load 2 elem from deltaY
	int Y_k = tx >> 1;
	Ims_oc_fhr_fwr(Y_k, Y_oc, Y_fhr, Y_fwr);
	int yoffset = ((Y_fhr * OW) + Y_fwr)*OC + Y_oc;
	bool ly0 = (tohs0 >= -Y_fhr) && (tohs0 < OH - Y_fhr) && (tows0 >= -Y_fwr) && (tows0 < OW - Y_fwr);
	bool ly1 = (tohs1 >= -Y_fhr) && (tohs1 < OH - Y_fhr) && (tows1 >= -Y_fwr) && (tows1 < OW - Y_fwr);
	const int Ys_x = (tx >> 1), Ys_y = (ty << 1) + (tx & 1);
	Ys[buf][Ys_x][Ys_y].x = (ly0 ? deltaY[Y0 + yoffset] : 0);
	Ys[buf][Ys_x][Ys_y].y = (ly1 ? deltaY[Y1 + yoffset] : 0);
	__syncthreads();

	//compute area-------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0);
	float4 v1 = make_float4(0, 0, 0, 0);
	float4 v2 = make_float4(0, 0, 0, 0);
	float4 v3 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 y = *(float4*)(&Ys[buf][ik][ty << 1]);
			float4 w = *(float4*)(&Ws[buf][ik][tx << 1]);

			simdMM4(v0, y.x, w);
			simdMM4(v1, y.y, w);
			simdMM4(v2, y.z, w);
			simdMM4(v3, y.w, w);
		}
		buf ^= 1;

		//load 2 elem from W
		int W_k = ((ok << LB) + ty) >> 1;
		Ims_oc_fhr_fwr(W_k, W_oc, W_fhr, W_fwr);
		int fh = y + (oph - W_fhr)*sh;
		int fw = x + (opw - W_fwr)*sw;
		Ws[buf][Ws_y][Ws_x] = *(float2*)(&get4d(W, W_oc, fh, fw, tic0, FH, FW, IC));

		//load 2 elem from Y
		int Y_k = ((ok << LB) + tx) >> 1;
		Ims_oc_fhr_fwr(Y_k, Y_oc, Y_fhr, Y_fwr);
		int yoffset = ((Y_fhr * OW) + Y_fwr)*OC + Y_oc;
		bool ly0 = (tohs0 >= -Y_fhr) && (tohs0 < OH - Y_fhr) && (tows0 >= -Y_fwr) && (tows0 < OW - Y_fwr);
		bool ly1 = (tohs1 >= -Y_fhr) && (tohs1 < OH - Y_fhr) && (tows1 >= -Y_fwr) && (tows1 < OW - Y_fwr);
		Ys[buf][Ys_x][Ys_y].x = (ly0 ? deltaY[Y0 + yoffset] : 0);
		Ys[buf][Ys_x][Ys_y].y = (ly1 ? deltaY[Y1 + yoffset] : 0);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 y = *(float4*)(&Ys[buf][ik][ty << 1]);
		float4 w = *(float4*)(&Ws[buf][ik][tx << 1]);

		simdMM4(v0, y.x, w);
		simdMM4(v1, y.y, w);
		simdMM4(v2, y.z, w);
		simdMM4(v3, y.w, w);
	}

	*(float4*)(deltaX + X0) = v0;
	*(float4*)(deltaX + X1) = v1;
	*(float4*)(deltaX + X2) = v2;
	*(float4*)(deltaX + X3) = v3;
}

#endif


//(BLOCK_SIZE*4, BLOCK_SIZE*4)
//(IH % sh == 0) && (IW % sw == 0), double buffered shared_memory
#ifndef X_KERNEL_V10
#define X_KERNEL_V10

#define X_kv10(stream, LB, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM) \
	ksn_kernel_v10<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>2), (GM>>LB>>2), (sh*sw)), dim3(1<<LB, 1<<LB) >>>\
			(deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw)

//LB = 4: Size = 1.125, Time = 1.018 msec, Performace = 2373.2 GFlop/s
template<int LB, int STEP>
__global__ void ksn_kernel_v10(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2 Ys[2][1 << LB >> 1][(2 << LB) + 2];

	//prepare for GK(bz(y, x)) = CFW * CFH * OC
	int bz = blockIdx.z;
	int y = bz / sw, x = bz - y * sw;//x = bz % sw
	int CFH = (FH - y + sh - 1) / sh, oph = CFH - 1;
	int CFW = (FW - x + sw - 1) / sw, opw = CFW - 1;
	const int CFH_CFW = CFH * CFW, GK = CFH_CFW * OC;

	ph = ph - y; pw = pw - x;
	int ihs = -ph + (-(ph > 0) & ((ph - y + sh - 1) / sh * sh));
	int iws = -pw + (-(pw > 0) & ((pw - x + sw - 1) / sw * sw));

	//prepare for GN = IC
	int ic0 = ((blockIdx.x << LB) + tx) << 2;
	int tic0 = ic0 + ((ty & 1) << 1);
	deltaX += ic0;
	W += tic0;

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = ((blockIdx.y << LB) + ty) << 2;
	int j1 = j0 + 1, j2 = j0 + 2, j3 = j0 + 3;
	int IW_slice = IW / sw, IH_IW_slice = (IH / sh) * IW_slice;
	Ims_n_ih_iw(j0, n0, ih0, iw0);
	Ims_n_ih_iw(j1, n1, ih1, iw1);
	Ims_n_ih_iw(j2, n2, ih2, iw2);
	Ims_n_ih_iw(j3, n3, ih3, iw3);
	int X0 = ((n0*IH + ih0)*IW + iw0) * IC;
	int X1 = ((n1*IH + ih1)*IW + iw1) * IC;
	int X2 = ((n2*IH + ih2)*IW + iw2) * IC;
	int X3 = ((n3*IH + ih3)*IW + iw3) * IC;

	bool flagX = (tx & 1);
	int tohs0 = ((IF_int(flagX, ih2, ih0) + ph) >> 1) - oph;
	int tohs1 = ((IF_int(flagX, ih3, ih1) + ph) >> 1) - oph;
	int tows0 = ((IF_int(flagX, iw2, iw0) + pw) >> 1) - opw;
	int tows1 = ((IF_int(flagX, iw3, iw1) + pw) >> 1) - opw;
	int Y0 = ((IF_int(flagX, n2, n0)*OH + tohs0)*OW + tows0) * OC;
	int Y1 = ((IF_int(flagX, n3, n1)*OH + tohs1)*OW + tows1) * OC;

	//load 2 elem from W
	int W_k = ty >> 1;
	Ims_oc_fhr_fwr(W_k, W_oc, W_fhr, W_fwr);
	int fh = y + (oph - W_fhr)*sh;
	int fw = x + (opw - W_fwr)*sw;
	int woffset = ((W_oc*FH + fh)*FW + fw) * IC;
	const int Ws_y = (ty >> 1), Ws_x = (tx << 1) + (ty & 1);
	Ws[buf][Ws_y][Ws_x] = *(float2*)(W + woffset);

	//load 2 elem from deltaY
	int Y_k = tx >> 1;
	Ims_oc_fhr_fwr(Y_k, Y_oc, Y_fhr, Y_fwr);
	int yoffset = ((Y_fhr * OW) + Y_fwr)*OC + Y_oc;
	bool ly0 = (tohs0 >= -Y_fhr) && (tohs0 < OH - Y_fhr) && (tows0 >= -Y_fwr) && (tows0 < OW - Y_fwr);
	bool ly1 = (tohs1 >= -Y_fhr) && (tohs1 < OH - Y_fhr) && (tows1 >= -Y_fwr) && (tows1 < OW - Y_fwr);
	const int Ys_x = (tx >> 1), Ys_y = (ty << 1) + (tx & 1);
	Ys[buf][Ys_x][Ys_y].x = (ly0 ? deltaY[Y0 + yoffset] : 0);
	Ys[buf][Ys_x][Ys_y].y = (ly1 ? deltaY[Y1 + yoffset] : 0);
	__syncthreads();

	//compute area-------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0);
	float4 v1 = make_float4(0, 0, 0, 0);
	float4 v2 = make_float4(0, 0, 0, 0);
	float4 v3 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 y = *(float4*)(&Ys[buf][ik][ty << 1]);
			float4 w = *(float4*)(&Ws[buf][ik][tx << 1]);

			simdMM4(v0, y.x, w);
			simdMM4(v1, y.y, w);
			simdMM4(v2, y.z, w);
			simdMM4(v3, y.w, w);
		}
		buf ^= 1;

		//load 2 elem from W
		int W_k = ((ok << LB) + ty) >> 1;
		Ims_oc_fhr_fwr(W_k, W_oc, W_fhr, W_fwr);
		int fh = y + (oph - W_fhr)*sh;
		int fw = x + (opw - W_fwr)*sw;
		int woffset = ((W_oc*FH + fh)*FW + fw) * IC;
		Ws[buf][Ws_y][Ws_x] = *(float2*)(W + woffset);

		//load 2 elem from Y
		int Y_k = ((ok << LB) + tx) >> 1;
		Ims_oc_fhr_fwr(Y_k, Y_oc, Y_fhr, Y_fwr);
		int yoffset = ((Y_fhr * OW) + Y_fwr)*OC + Y_oc;
		bool ly0 = (tohs0 >= -Y_fhr) && (tohs0 < OH - Y_fhr) && (tows0 >= -Y_fwr) && (tows0 < OW - Y_fwr);
		bool ly1 = (tohs1 >= -Y_fhr) && (tohs1 < OH - Y_fhr) && (tows1 >= -Y_fwr) && (tows1 < OW - Y_fwr);
		Ys[buf][Ys_x][Ys_y].x = (ly0 ? deltaY[Y0 + yoffset] : 0);
		Ys[buf][Ys_x][Ys_y].y = (ly1 ? deltaY[Y1 + yoffset] : 0);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 y = *(float4*)(&Ys[buf][ik][ty << 1]);
		float4 w = *(float4*)(&Ws[buf][ik][tx << 1]);

		simdMM4(v0, y.x, w);
		simdMM4(v1, y.y, w);
		simdMM4(v2, y.z, w);
		simdMM4(v3, y.w, w);
	}

	*(float4*)(deltaX + X0) = v0;
	*(float4*)(deltaX + X1) = v1;
	*(float4*)(deltaX + X2) = v2;
	*(float4*)(deltaX + X3) = v3;
}

#endif


//(BLOCK_SIZE*8, BLOCK_SIZE*8)
//(IH % sh == 0) && (IW % sw == 0), double buffered shared_memory
#ifndef X_KERNEL_V11
#define X_KERNEL_V11

#define X_kv11(stream, LB, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM) \
	ksn_kernel_v11<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>3), (GM>>LB>>3), (sh*sw)), dim3(1<<LB, 1<<LB) >>>\
			(deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw)

//LB = 4: Size = 1.125, Time = 1.128 msec, Performace = 2141.77 GFlop/s
//LB = 3: Size = 1.125, Time = 0.924 msec, Performace = 2614.63 GFlop/s
template<int LB, int STEP>
__global__ void ksn_kernel_v11(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];

	//prepare for GZ = sh*sw 
	int bz = blockIdx.z;
	int y = bz / sw, x = bz - y * sw;//x = bz % sw
	ph = ph - y; pw = pw - x;
	int ihs = -ph + (-(ph > 0) & ((ph - y + sh - 1) / sh * sh));
	int iws = -pw + (-(pw > 0) & ((pw - x + sw - 1) / sw * sw));

	//prepare for GK(bz(y, x)) = CFW * CFH * OC
	int CFH = (FH - y + sh - 1) / sh;
	int CFW = (FW - x + sw - 1) / sw;
	const int CFH_CFW = CFH * CFW, GK = CFH_CFW * OC;

	//prepare for GN = IC
	int ic0 = ((blockIdx.x << LB) + tx) << 3;
	W += ic0 + ((ty >= STEP) << 2);
	deltaX += ic0;

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = ((blockIdx.y << LB) + ty) << 3;
	int j1 = j0 + 1, j2 = j0 + 2, j3 = j0 + 3;
	int j4 = j0 + 4, j5 = j0 + 5, j6 = j0 + 6, j7 = j0 + 7;
	int IW_slice = IW / sw, IH_IW_slice = (IH / sh) * IW_slice;
	Ims_n_ih_iw(j0, n0, ih0, iw0);
	Ims_n_ih_iw(j1, n1, ih1, iw1);
	Ims_n_ih_iw(j2, n2, ih2, iw2);
	Ims_n_ih_iw(j3, n3, ih3, iw3);
	Ims_n_ih_iw(j4, n4, ih4, iw4);
	Ims_n_ih_iw(j5, n5, ih5, iw5);
	Ims_n_ih_iw(j6, n6, ih6, iw6);
	Ims_n_ih_iw(j7, n7, ih7, iw7);
	int X0 = ((n0*IH + ih0)*IW + iw0) * IC;
	int X1 = ((n1*IH + ih1)*IW + iw1) * IC;
	int X2 = ((n2*IH + ih2)*IW + iw2) * IC;
	int X3 = ((n3*IH + ih3)*IW + iw3) * IC;
	int X4 = ((n4*IH + ih4)*IW + iw4) * IC;
	int X5 = ((n5*IH + ih5)*IW + iw5) * IC;
	int X6 = ((n6*IH + ih6)*IW + iw6) * IC;
	int X7 = ((n7*IH + ih7)*IW + iw7) * IC;

	bool flagX = (tx >= STEP);
	int oph = CFH - 1, opw = CFW - 1;
	int tohs0 = ((IF_int(flagX, ih4, ih0) + ph) >> 1) - oph;
	int tohs1 = ((IF_int(flagX, ih5, ih1) + ph) >> 1) - oph;
	int tohs2 = ((IF_int(flagX, ih6, ih2) + ph) >> 1) - oph;
	int tohs3 = ((IF_int(flagX, ih7, ih3) + ph) >> 1) - oph;
	int tows0 = ((IF_int(flagX, iw4, iw0) + pw) >> 1) - opw;
	int tows1 = ((IF_int(flagX, iw5, iw1) + pw) >> 1) - opw;
	int tows2 = ((IF_int(flagX, iw6, iw2) + pw) >> 1) - opw;
	int tows3 = ((IF_int(flagX, iw7, iw3) + pw) >> 1) - opw;
	int Y0 = ((IF_int(flagX, n4, n0)*OH + tohs0)*OW + tows0) * OC;
	int Y1 = ((IF_int(flagX, n5, n1)*OH + tohs1)*OW + tows1) * OC;
	int Y2 = ((IF_int(flagX, n6, n2)*OH + tohs2)*OW + tows2) * OC;
	int Y3 = ((IF_int(flagX, n7, n3)*OH + tohs3)*OW + tows3) * OC;

	//load 4 elem from W
	int W_k = ty - ((ty >= STEP) << LB >> 1);
	Ims_oc_fhr_fwr(W_k, W_oc, W_fhr, W_fwr);
	int fh = y + (oph - W_fhr)*sh;
	int fw = x + (opw - W_fwr)*sw;

	//int fh = y + (oph - W_fhr)*sh;
	//int fw = x + (opw - W_fwr)*sw;
	//(y*FW + x) * IC
	int woffset = ((W_oc*FH + fh)*FW + fw) * IC;
	Ws[buf][ty][tx] = *(float4*)(W + woffset);

	//load 4 elem from deltaY
	int Y_k = tx - ((tx >= STEP) << LB >> 1);
	Ims_oc_fhr_fwr(Y_k, Y_oc, Y_fhr, Y_fwr);
	int yoffset = ((Y_fhr * OW) + Y_fwr)*OC + Y_oc;
	bool ly0 = (tohs0 >= -Y_fhr) && (tohs0 < OH - Y_fhr) && (tows0 >= -Y_fwr) && (tows0 < OW - Y_fwr);
	bool ly1 = (tohs1 >= -Y_fhr) && (tohs1 < OH - Y_fhr) && (tows1 >= -Y_fwr) && (tows1 < OW - Y_fwr);
	bool ly2 = (tohs2 >= -Y_fhr) && (tohs2 < OH - Y_fhr) && (tows2 >= -Y_fwr) && (tows2 < OW - Y_fwr);
	bool ly3 = (tohs3 >= -Y_fhr) && (tohs3 < OH - Y_fhr) && (tows3 >= -Y_fwr) && (tows3 < OW - Y_fwr);
	Ys[buf][tx][ty].x = (ly0 ? deltaY[Y0 + yoffset] : 0);
	Ys[buf][tx][ty].y = (ly1 ? deltaY[Y1 + yoffset] : 0);
	Ys[buf][tx][ty].z = (ly2 ? deltaY[Y2 + yoffset] : 0);
	Ys[buf][tx][ty].w = (ly3 ? deltaY[Y3 + yoffset] : 0);
	__syncthreads();

	//compute area-------------------------------------------------
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
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];
			float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];

			simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
			simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
			simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
			simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
			simdMM4(v8, y1.x, w0); simdMM4(v9, y1.x, w1);
			simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
			simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
			simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
		}
		buf ^= 1;

		//load 4 elem from W
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		Ims_oc_fhr_fwr(W_k, W_oc, W_fhr, W_fwr);
		int fh = y + (oph - W_fhr)*sh;
		int fw = x + (opw - W_fwr)*sw;
		int woffset = ((W_oc*FH + fh)*FW + fw) * IC;
		Ws[buf][ty][tx] = *(float4*)(W + woffset);

		//load 4 elem from Y
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		Ims_oc_fhr_fwr(Y_k, Y_oc, Y_fhr, Y_fwr);
		int yoffset = ((Y_fhr * OW) + Y_fwr)*OC + Y_oc;
		bool ly0 = (tohs0 >= -Y_fhr) && (tohs0 < OH - Y_fhr) && (tows0 >= -Y_fwr) && (tows0 < OW - Y_fwr);
		bool ly1 = (tohs1 >= -Y_fhr) && (tohs1 < OH - Y_fhr) && (tows1 >= -Y_fwr) && (tows1 < OW - Y_fwr);
		bool ly2 = (tohs2 >= -Y_fhr) && (tohs2 < OH - Y_fhr) && (tows2 >= -Y_fwr) && (tows2 < OW - Y_fwr);
		bool ly3 = (tohs3 >= -Y_fhr) && (tohs3 < OH - Y_fhr) && (tows3 >= -Y_fwr) && (tows3 < OW - Y_fwr);
		Ys[buf][tx][ty].x = (ly0 ? deltaY[Y0 + yoffset] : 0);
		Ys[buf][tx][ty].y = (ly1 ? deltaY[Y1 + yoffset] : 0);
		Ys[buf][tx][ty].z = (ly2 ? deltaY[Y2 + yoffset] : 0);
		Ys[buf][tx][ty].w = (ly3 ? deltaY[Y3 + yoffset] : 0);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];
		float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];

		simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
		simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
		simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
		simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
		simdMM4(v8, y1.x, w0); simdMM4(v9, y1.x, w1);
		simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
		simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
		simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
	}

	*(float4*)(deltaX + X0) = v0;  *(float4*)(deltaX + X0 + 4) = v1;
	*(float4*)(deltaX + X1) = v2;  *(float4*)(deltaX + X1 + 4) = v3;
	*(float4*)(deltaX + X2) = v4;  *(float4*)(deltaX + X2 + 4) = v5;
	*(float4*)(deltaX + X3) = v6;  *(float4*)(deltaX + X3 + 4) = v7;
	*(float4*)(deltaX + X4) = v8;  *(float4*)(deltaX + X4 + 4) = v9;
	*(float4*)(deltaX + X5) = v10; *(float4*)(deltaX + X5 + 4) = v11;
	*(float4*)(deltaX + X6) = v12; *(float4*)(deltaX + X6 + 4) = v13;
	*(float4*)(deltaX + X7) = v14; *(float4*)(deltaX + X7 + 4) = v15;
}

#endif


//(BLOCK_SIZE*8, BLOCK_SIZE*8)
//(IH % sh == 0) && (IW % sw == 0), double buffered shared_memory
#ifndef X_KERNEL_V12
#define X_KERNEL_V12

#define X_kv12(stream, LB, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM) \
	ksn_kernel_v12<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>3), (GM>>LB>>3), (sh*sw)), dim3(1<<LB, 1<<LB) >>>\
			(deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw)

//LB = 4: Size = 1.125, Time = 1.108 msec, Performace = 2180.43 GFlop/s
//LB = 3: Size = 1.125, Time = 0.924 msec, Performace = 2614.63 GFlop/s
template<int LB, int STEP>
__global__ void ksn_kernel_v12(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];

	//prepare for GZ = sh*sw 
	int bz = blockIdx.z;
	int y = bz / sw, x = bz - y * sw;//x = bz % sw
	W = W + (y*FW + x) * IC;

	ph = ph - y; pw = pw - x;
	int ihs = -ph + (-(ph > 0) & ((ph - y + sh - 1) / sh * sh));
	int iws = -pw + (-(pw > 0) & ((pw - x + sw - 1) / sw * sw));

	//prepare for GK(bz(y, x)) = CFW * CFH * OC
	int CFH = (FH - y + sh - 1) / sh;
	int CFW = (FW - x + sw - 1) / sw;
	const int CFH_CFW = CFH * CFW, GK = CFH_CFW * OC;

	//prepare for GN = IC
	int ic0 = ((blockIdx.x << LB) + tx) << 3;
	W += ic0 + ((ty >= STEP) << 2);
	deltaX += ic0;

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = ((blockIdx.y << LB) + ty) << 3;
	int j1 = j0 + 1, j2 = j0 + 2, j3 = j0 + 3;
	int j4 = j0 + 4, j5 = j0 + 5, j6 = j0 + 6, j7 = j0 + 7;
	int IW_slice = IW / sw, IH_IW_slice = (IH / sh) * IW_slice;
	Ims_n_ih_iw(j0, n0, ih0, iw0);
	Ims_n_ih_iw(j1, n1, ih1, iw1);
	Ims_n_ih_iw(j2, n2, ih2, iw2);
	Ims_n_ih_iw(j3, n3, ih3, iw3);
	Ims_n_ih_iw(j4, n4, ih4, iw4);
	Ims_n_ih_iw(j5, n5, ih5, iw5);
	Ims_n_ih_iw(j6, n6, ih6, iw6);
	Ims_n_ih_iw(j7, n7, ih7, iw7);
	int X0 = ((n0*IH + ih0)*IW + iw0) * IC;
	int X1 = ((n1*IH + ih1)*IW + iw1) * IC;
	int X2 = ((n2*IH + ih2)*IW + iw2) * IC;
	int X3 = ((n3*IH + ih3)*IW + iw3) * IC;
	int X4 = ((n4*IH + ih4)*IW + iw4) * IC;
	int X5 = ((n5*IH + ih5)*IW + iw5) * IC;
	int X6 = ((n6*IH + ih6)*IW + iw6) * IC;
	int X7 = ((n7*IH + ih7)*IW + iw7) * IC;

	bool flagX = (tx >= STEP);
	int oph = CFH - 1, opw = CFW - 1;
	int tohs0 = ((IF_int(flagX, ih4, ih0) + ph) >> 1) - oph;
	int tohs1 = ((IF_int(flagX, ih5, ih1) + ph) >> 1) - oph;
	int tohs2 = ((IF_int(flagX, ih6, ih2) + ph) >> 1) - oph;
	int tohs3 = ((IF_int(flagX, ih7, ih3) + ph) >> 1) - oph;
	int tows0 = ((IF_int(flagX, iw4, iw0) + pw) >> 1) - opw;
	int tows1 = ((IF_int(flagX, iw5, iw1) + pw) >> 1) - opw;
	int tows2 = ((IF_int(flagX, iw6, iw2) + pw) >> 1) - opw;
	int tows3 = ((IF_int(flagX, iw7, iw3) + pw) >> 1) - opw;
	int Y0 = ((IF_int(flagX, n4, n0)*OH + tohs0)*OW + tows0) * OC;
	int Y1 = ((IF_int(flagX, n5, n1)*OH + tohs1)*OW + tows1) * OC;
	int Y2 = ((IF_int(flagX, n6, n2)*OH + tohs2)*OW + tows2) * OC;
	int Y3 = ((IF_int(flagX, n7, n3)*OH + tohs3)*OW + tows3) * OC;

	//((W_oc*FH + (y + (oph - W_fhr)*sh))*FW + (x + (opw - W_fwr)*sw)) * IC;
	//((W_oc*FH + (oph - W_fhr)*sh)*FW + (opw - W_fwr) + y*sh*FW + x*sw) * IC
	//So: W = W + (y*FW + x)*IC
	//load 4 elem from W
	int W_k = ty - ((ty >= STEP) << LB >> 1);
	Ims_oc_fhr_fwr(W_k, W_oc, W_fhr, W_fwr);
	int fh = (oph - W_fhr)*sh;
	int fw = (opw - W_fwr)*sw;
	int woffset = ((W_oc*FH + fh)*FW + fw) * IC;
	Ws[buf][ty][tx] = *(float4*)(W + woffset);

	//load 4 elem from deltaY
	int Y_k = tx - ((tx >= STEP) << LB >> 1);
	Ims_oc_fhr_fwr(Y_k, Y_oc, Y_fhr, Y_fwr);
	int yoffset = ((Y_fhr * OW) + Y_fwr)*OC + Y_oc;
	bool ly0 = (tohs0 >= -Y_fhr) && (tohs0 < OH - Y_fhr) && (tows0 >= -Y_fwr) && (tows0 < OW - Y_fwr);
	bool ly1 = (tohs1 >= -Y_fhr) && (tohs1 < OH - Y_fhr) && (tows1 >= -Y_fwr) && (tows1 < OW - Y_fwr);
	bool ly2 = (tohs2 >= -Y_fhr) && (tohs2 < OH - Y_fhr) && (tows2 >= -Y_fwr) && (tows2 < OW - Y_fwr);
	bool ly3 = (tohs3 >= -Y_fhr) && (tohs3 < OH - Y_fhr) && (tows3 >= -Y_fwr) && (tows3 < OW - Y_fwr);
	Ys[buf][tx][ty].x = (ly0 ? deltaY[Y0 + yoffset] : 0);
	Ys[buf][tx][ty].y = (ly1 ? deltaY[Y1 + yoffset] : 0);
	Ys[buf][tx][ty].z = (ly2 ? deltaY[Y2 + yoffset] : 0);
	Ys[buf][tx][ty].w = (ly3 ? deltaY[Y3 + yoffset] : 0);
	__syncthreads();

	//compute area-------------------------------------------------
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
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];
			float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];

			simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
			simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
			simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
			simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
			simdMM4(v8, y1.x, w0); simdMM4(v9, y1.x, w1);
			simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
			simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
			simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
		}
		buf ^= 1;

		//load 4 elem from W
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		Ims_oc_fhr_fwr(W_k, W_oc, W_fhr, W_fwr);
		int fh = (oph - W_fhr)*sh;
		int fw = (opw - W_fwr)*sw;
		int woffset = ((W_oc*FH + fh)*FW + fw) * IC;
		Ws[buf][ty][tx] = *(float4*)(W + woffset);

		//load 4 elem from Y
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		Ims_oc_fhr_fwr(Y_k, Y_oc, Y_fhr, Y_fwr);
		int yoffset = ((Y_fhr * OW) + Y_fwr)*OC + Y_oc;
		bool ly0 = (tohs0 >= -Y_fhr) && (tohs0 < OH - Y_fhr) && (tows0 >= -Y_fwr) && (tows0 < OW - Y_fwr);
		bool ly1 = (tohs1 >= -Y_fhr) && (tohs1 < OH - Y_fhr) && (tows1 >= -Y_fwr) && (tows1 < OW - Y_fwr);
		bool ly2 = (tohs2 >= -Y_fhr) && (tohs2 < OH - Y_fhr) && (tows2 >= -Y_fwr) && (tows2 < OW - Y_fwr);
		bool ly3 = (tohs3 >= -Y_fhr) && (tohs3 < OH - Y_fhr) && (tows3 >= -Y_fwr) && (tows3 < OW - Y_fwr);
		Ys[buf][tx][ty].x = (ly0 ? deltaY[Y0 + yoffset] : 0);
		Ys[buf][tx][ty].y = (ly1 ? deltaY[Y1 + yoffset] : 0);
		Ys[buf][tx][ty].z = (ly2 ? deltaY[Y2 + yoffset] : 0);
		Ys[buf][tx][ty].w = (ly3 ? deltaY[Y3 + yoffset] : 0);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];
		float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];

		simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
		simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
		simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
		simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
		simdMM4(v8, y1.x, w0); simdMM4(v9, y1.x, w1);
		simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
		simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
		simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
	}

	*(float4*)(deltaX + X0) = v0;  *(float4*)(deltaX + X0 + 4) = v1;
	*(float4*)(deltaX + X1) = v2;  *(float4*)(deltaX + X1 + 4) = v3;
	*(float4*)(deltaX + X2) = v4;  *(float4*)(deltaX + X2 + 4) = v5;
	*(float4*)(deltaX + X3) = v6;  *(float4*)(deltaX + X3 + 4) = v7;
	*(float4*)(deltaX + X4) = v8;  *(float4*)(deltaX + X4 + 4) = v9;
	*(float4*)(deltaX + X5) = v10; *(float4*)(deltaX + X5 + 4) = v11;
	*(float4*)(deltaX + X6) = v12; *(float4*)(deltaX + X6 + 4) = v13;
	*(float4*)(deltaX + X7) = v14; *(float4*)(deltaX + X7 + 4) = v15;
}

#endif


//(BLOCK_SIZE*8, BLOCK_SIZE*8)
//(IH % sh == 0) && (IW % sw == 0), double buffered shared_memory
#ifndef X_KERNEL_V13
#define X_KERNEL_V13

#define X_kv13(stream, LB, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM) \
	ksn_kernel_v13<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>3), (GM>>LB>>3), (sh*sw)), dim3(1<<LB, 1<<LB) >>>\
			(deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw)

//LB = 4: Size = 1.125, Time = 1.042 msec, Performace = 2318.54 GFlop/s
//LB = 3: Size = 1.125, Time = 0.902 msec, Performace = 2678.4 GFlop/s
template<int LB, int STEP>
__global__ void ksn_kernel_v13(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];

	//prepare for GZ = sh*sw 
	int bz = blockIdx.z;
	int y = bz / sw, x = bz - y * sw;//x = bz % sw

	ph = ph - y; pw = pw - x;
	int ihs = -ph + (-(ph > 0) & ((ph - y + sh - 1) / sh * sh));
	int iws = -pw + (-(pw > 0) & ((pw - x + sw - 1) / sw * sw));

	//prepare for GK(bz(y, x)) = CFW * CFH * OC
	int CFH = (FH - y + sh - 1) / sh, oph = CFH - 1;
	int CFW = (FW - x + sw - 1) / sw, opw = CFW - 1;
	const int CFH_CFW = CFH * CFW, GK = CFH_CFW * OC;
	W += ((y + oph * sh)*FW + (opw * sw + x)) * IC;

	//prepare for GN = IC
	int ic0 = ((blockIdx.x << LB) + tx) << 3;
	W += ic0 + ((ty >= STEP) << 2);
	deltaX += ic0;

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = ((blockIdx.y << LB) + ty) << 3;
	int j1 = j0 + 1, j2 = j0 + 2, j3 = j0 + 3;
	int j4 = j0 + 4, j5 = j0 + 5, j6 = j0 + 6, j7 = j0 + 7;
	int IW_slice = IW / sw, IH_IW_slice = (IH / sh) * IW_slice;
	Ims_n_ih_iw(j0, n0, ih0, iw0);
	Ims_n_ih_iw(j1, n1, ih1, iw1);
	Ims_n_ih_iw(j2, n2, ih2, iw2);
	Ims_n_ih_iw(j3, n3, ih3, iw3);
	Ims_n_ih_iw(j4, n4, ih4, iw4);
	Ims_n_ih_iw(j5, n5, ih5, iw5);
	Ims_n_ih_iw(j6, n6, ih6, iw6);
	Ims_n_ih_iw(j7, n7, ih7, iw7);
	int X0 = ((n0*IH + ih0)*IW + iw0) * IC;
	int X1 = ((n1*IH + ih1)*IW + iw1) * IC;
	int X2 = ((n2*IH + ih2)*IW + iw2) * IC;
	int X3 = ((n3*IH + ih3)*IW + iw3) * IC;
	int X4 = ((n4*IH + ih4)*IW + iw4) * IC;
	int X5 = ((n5*IH + ih5)*IW + iw5) * IC;
	int X6 = ((n6*IH + ih6)*IW + iw6) * IC;
	int X7 = ((n7*IH + ih7)*IW + iw7) * IC;

	bool flagX = (tx >= STEP);
	int tohs0 = ((IF_int(flagX, ih4, ih0) + ph) >> 1) - oph;
	int tohs1 = ((IF_int(flagX, ih5, ih1) + ph) >> 1) - oph;
	int tohs2 = ((IF_int(flagX, ih6, ih2) + ph) >> 1) - oph;
	int tohs3 = ((IF_int(flagX, ih7, ih3) + ph) >> 1) - oph;
	int tows0 = ((IF_int(flagX, iw4, iw0) + pw) >> 1) - opw;
	int tows1 = ((IF_int(flagX, iw5, iw1) + pw) >> 1) - opw;
	int tows2 = ((IF_int(flagX, iw6, iw2) + pw) >> 1) - opw;
	int tows3 = ((IF_int(flagX, iw7, iw3) + pw) >> 1) - opw;
	int Y0 = ((IF_int(flagX, n4, n0)*OH + tohs0)*OW + tows0) * OC;
	int Y1 = ((IF_int(flagX, n5, n1)*OH + tohs1)*OW + tows1) * OC;
	int Y2 = ((IF_int(flagX, n6, n2)*OH + tohs2)*OW + tows2) * OC;
	int Y3 = ((IF_int(flagX, n7, n3)*OH + tohs3)*OW + tows3) * OC;

	//woffset = ((W_oc*FH + fh)*FW + fw) * IC;
	//woffset = ((W_oc*FH + (oph - W_fhr)*sh)*FW + (opw - W_fwr)*sw) * IC;
	//((W_oc*FH + (oph - W_fhr)*sh)*FW + (opw - W_fwr)*sw) * IC;
	//W_oc*FH*FW*IC + (oph - W_fhr)*sh*FW*IC + (opw - W_fwr)*sw*IC
	//W_oc*FH*FW*IC + oph*sh*FW*IC + opw*sw*IC - W_fhr*sh*FW*IC - W_fwr*sw*IC
	//let: U = oph*sh*FW*IC + opw*sw*IC
	//woffset = U + W_oc*FH*FW*IC - W_fhr*sh*FW*IC - W_fwr*sw*IC
	//woffset = U + ((W_oc*FH - W_fhr*sh)*FW - W_fwr*sw)*IC
	//W = W + ((y + oph * sh)*FW + (opw * sw + x)) * IC;

	//load 4 elem from W
	int W_k = ty - ((ty >= STEP) << LB >> 1);
	Ims_oc_fhr_fwr(W_k, W_oc, W_fhr, W_fwr);
	int woffset = ((W_oc*FH - W_fhr * sh)*FW - W_fwr * sw)*IC;
	Ws[buf][ty][tx] = *(float4*)(W + woffset);

	//load 4 elem from deltaY
	int Y_k = tx - ((tx >= STEP) << LB >> 1);
	Ims_oc_fhr_fwr(Y_k, Y_oc, Y_fhr, Y_fwr);
	int yoffset = ((Y_fhr * OW) + Y_fwr)*OC + Y_oc;
	bool ly0 = (tohs0 >= -Y_fhr) && (tohs0 < OH - Y_fhr) && (tows0 >= -Y_fwr) && (tows0 < OW - Y_fwr);
	bool ly1 = (tohs1 >= -Y_fhr) && (tohs1 < OH - Y_fhr) && (tows1 >= -Y_fwr) && (tows1 < OW - Y_fwr);
	bool ly2 = (tohs2 >= -Y_fhr) && (tohs2 < OH - Y_fhr) && (tows2 >= -Y_fwr) && (tows2 < OW - Y_fwr);
	bool ly3 = (tohs3 >= -Y_fhr) && (tohs3 < OH - Y_fhr) && (tows3 >= -Y_fwr) && (tows3 < OW - Y_fwr);
	Ys[buf][tx][ty].x = (ly0 ? deltaY[Y0 + yoffset] : 0);
	Ys[buf][tx][ty].y = (ly1 ? deltaY[Y1 + yoffset] : 0);
	Ys[buf][tx][ty].z = (ly2 ? deltaY[Y2 + yoffset] : 0);
	Ys[buf][tx][ty].w = (ly3 ? deltaY[Y3 + yoffset] : 0);
	__syncthreads();

	//compute area-------------------------------------------------
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
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];
			float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];

			simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
			simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
			simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
			simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
			simdMM4(v8, y1.x, w0); simdMM4(v9, y1.x, w1);
			simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
			simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
			simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
		}
		buf ^= 1;

		//load 4 elem from W
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		Ims_oc_fhr_fwr(W_k, W_oc, W_fhr, W_fwr);
		int woffset = ((W_oc*FH - W_fhr * sh)*FW - W_fwr * sw)*IC;
		Ws[buf][ty][tx] = *(float4*)(W + woffset);

		//load 4 elem from Y
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		Ims_oc_fhr_fwr(Y_k, Y_oc, Y_fhr, Y_fwr);
		int yoffset = ((Y_fhr * OW) + Y_fwr)*OC + Y_oc;
		bool ly0 = (tohs0 >= -Y_fhr) && (tohs0 < OH - Y_fhr) && (tows0 >= -Y_fwr) && (tows0 < OW - Y_fwr);
		bool ly1 = (tohs1 >= -Y_fhr) && (tohs1 < OH - Y_fhr) && (tows1 >= -Y_fwr) && (tows1 < OW - Y_fwr);
		bool ly2 = (tohs2 >= -Y_fhr) && (tohs2 < OH - Y_fhr) && (tows2 >= -Y_fwr) && (tows2 < OW - Y_fwr);
		bool ly3 = (tohs3 >= -Y_fhr) && (tohs3 < OH - Y_fhr) && (tows3 >= -Y_fwr) && (tows3 < OW - Y_fwr);
		Ys[buf][tx][ty].x = (ly0 ? deltaY[Y0 + yoffset] : 0);
		Ys[buf][tx][ty].y = (ly1 ? deltaY[Y1 + yoffset] : 0);
		Ys[buf][tx][ty].z = (ly2 ? deltaY[Y2 + yoffset] : 0);
		Ys[buf][tx][ty].w = (ly3 ? deltaY[Y3 + yoffset] : 0);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];
		float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];

		simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
		simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
		simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
		simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
		simdMM4(v8, y1.x, w0); simdMM4(v9, y1.x, w1);
		simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
		simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
		simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
	}

	*(float4*)(deltaX + X0) = v0;  *(float4*)(deltaX + X0 + 4) = v1;
	*(float4*)(deltaX + X1) = v2;  *(float4*)(deltaX + X1 + 4) = v3;
	*(float4*)(deltaX + X2) = v4;  *(float4*)(deltaX + X2 + 4) = v5;
	*(float4*)(deltaX + X3) = v6;  *(float4*)(deltaX + X3 + 4) = v7;
	*(float4*)(deltaX + X4) = v8;  *(float4*)(deltaX + X4 + 4) = v9;
	*(float4*)(deltaX + X5) = v10; *(float4*)(deltaX + X5 + 4) = v11;
	*(float4*)(deltaX + X6) = v12; *(float4*)(deltaX + X6 + 4) = v13;
	*(float4*)(deltaX + X7) = v14; *(float4*)(deltaX + X7 + 4) = v15;
}

#endif


//(BLOCK_SIZE*8, BLOCK_SIZE*8)
//(IH % sh == 0) && (IW % sw == 0), double buffered shared_memory
#ifndef X_KERNEL_V14
#define X_KERNEL_V14

#define X_kv14(stream, LB, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM) \
	ksn_kernel_v14<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>3), (GM>>LB>>3), (sh*sw)), dim3(1<<LB, 1<<LB) >>>\
			(deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw)

//LB = 4: Size = 1.125, Time = 1.01 msec, Performace = 2392 GFlop/s
//LB = 3: Size = 1.125, Time = 0.902 msec, Performace = 2678.4 GFlop/s
template<int LB, int STEP>
__global__ void ksn_kernel_v14(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];

	//prepare for GZ = sh*sw 
	int bz = blockIdx.z;
	int y = bz / sw, x = bz - y * sw;//x = bz % sw

	ph = ph - y; pw = pw - x;
	int ihs = -ph + (-(ph > 0) & ((ph - y + sh - 1) / sh * sh));
	int iws = -pw + (-(pw > 0) & ((pw - x + sw - 1) / sw * sw));

	//prepare for GK(bz(y, x)) = CFW * CFH * OC
	int CFH = (FH - y + sh - 1) / sh, oph = CFH - 1;
	int CFW = (FW - x + sw - 1) / sw, opw = CFW - 1;
	const int CFW_OC = CFW * OC, GK = CFH * CFW_OC;
	W += ((y + oph * sh)*FW + (opw * sw + x)) * IC;

	//prepare for GN = IC
	int ic0 = ((blockIdx.x << LB) + tx) << 3;
	W += ic0 + ((ty >= STEP) << 2);
	deltaX += ic0;

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = ((blockIdx.y << LB) + ty) << 3;
	int j1 = j0 + 1, j2 = j0 + 2, j3 = j0 + 3;
	int j4 = j0 + 4, j5 = j0 + 5, j6 = j0 + 6, j7 = j0 + 7;
	int IW_slice = IW / sw, IH_IW_slice = (IH / sh) * IW_slice;
	Ims_n_ih_iw(j0, n0, ih0, iw0);
	Ims_n_ih_iw(j1, n1, ih1, iw1);
	Ims_n_ih_iw(j2, n2, ih2, iw2);
	Ims_n_ih_iw(j3, n3, ih3, iw3);
	Ims_n_ih_iw(j4, n4, ih4, iw4);
	Ims_n_ih_iw(j5, n5, ih5, iw5);
	Ims_n_ih_iw(j6, n6, ih6, iw6);
	Ims_n_ih_iw(j7, n7, ih7, iw7);
	int X0 = ((n0*IH + ih0)*IW + iw0) * IC;
	int X1 = ((n1*IH + ih1)*IW + iw1) * IC;
	int X2 = ((n2*IH + ih2)*IW + iw2) * IC;
	int X3 = ((n3*IH + ih3)*IW + iw3) * IC;
	int X4 = ((n4*IH + ih4)*IW + iw4) * IC;
	int X5 = ((n5*IH + ih5)*IW + iw5) * IC;
	int X6 = ((n6*IH + ih6)*IW + iw6) * IC;
	int X7 = ((n7*IH + ih7)*IW + iw7) * IC;

	bool flagX = (tx >= STEP);
	int tohs0 = ((IF_int(flagX, ih4, ih0) + ph) >> 1) - oph;
	int tohs1 = ((IF_int(flagX, ih5, ih1) + ph) >> 1) - oph;
	int tohs2 = ((IF_int(flagX, ih6, ih2) + ph) >> 1) - oph;
	int tohs3 = ((IF_int(flagX, ih7, ih3) + ph) >> 1) - oph;
	int tows0 = ((IF_int(flagX, iw4, iw0) + pw) >> 1) - opw;
	int tows1 = ((IF_int(flagX, iw5, iw1) + pw) >> 1) - opw;
	int tows2 = ((IF_int(flagX, iw6, iw2) + pw) >> 1) - opw;
	int tows3 = ((IF_int(flagX, iw7, iw3) + pw) >> 1) - opw;
	int Y0 = ((IF_int(flagX, n4, n0)*OH + tohs0)*OW + tows0) * OC;
	int Y1 = ((IF_int(flagX, n5, n1)*OH + tohs1)*OW + tows1) * OC;
	int Y2 = ((IF_int(flagX, n6, n2)*OH + tohs2)*OW + tows2) * OC;
	int Y3 = ((IF_int(flagX, n7, n3)*OH + tohs3)*OW + tows3) * OC;

	//load 4 elem from W
	int W_k = ty - ((ty >= STEP) << LB >> 1);
	Ims_fhr_fwr(W_k, W_fhr, W_fwr); int W_oc = W_k - W_fwr * OC;
	int woffset = ((W_oc*FH - W_fhr * sh)*FW - W_fwr * sw)*IC;
	Ws[buf][ty][tx] = *(float4*)(W + woffset);

	//yoffset = ((Y_fhr * OW) + Y_fwr)*OC + Y_oc;
	//yoffset = Y_fhr * OW*OC + (Y_fwr*OC + Y_oc);
	//yoffset = Y_fhr * OW*OC + Y_k;
	//load 4 elem from deltaY
	int Y_k = tx - ((tx >= STEP) << LB >> 1);
	Ims_fhr_fwr(Y_k, Y_fhr, Y_fwr);
	int yoffset = (Y_fhr * OW * OC) + Y_k;
	bool ly0 = (tohs0 >= -Y_fhr) && (tohs0 < OH - Y_fhr) && (tows0 >= -Y_fwr) && (tows0 < OW - Y_fwr);
	bool ly1 = (tohs1 >= -Y_fhr) && (tohs1 < OH - Y_fhr) && (tows1 >= -Y_fwr) && (tows1 < OW - Y_fwr);
	bool ly2 = (tohs2 >= -Y_fhr) && (tohs2 < OH - Y_fhr) && (tows2 >= -Y_fwr) && (tows2 < OW - Y_fwr);
	bool ly3 = (tohs3 >= -Y_fhr) && (tohs3 < OH - Y_fhr) && (tows3 >= -Y_fwr) && (tows3 < OW - Y_fwr);
	Ys[buf][tx][ty].x = (ly0 ? deltaY[Y0 + yoffset] : 0);
	Ys[buf][tx][ty].y = (ly1 ? deltaY[Y1 + yoffset] : 0);
	Ys[buf][tx][ty].z = (ly2 ? deltaY[Y2 + yoffset] : 0);
	Ys[buf][tx][ty].w = (ly3 ? deltaY[Y3 + yoffset] : 0);
	__syncthreads();

	//compute area-------------------------------------------------
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
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];
			float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];

			simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
			simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
			simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
			simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
			simdMM4(v8, y1.x, w0); simdMM4(v9, y1.x, w1);
			simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
			simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
			simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
		}
		buf ^= 1;

		//load 4 elem from W
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		Ims_fhr_fwr(W_k, W_fhr, W_fwr); int W_oc = W_k - W_fwr * OC;
		int woffset = ((W_oc*FH - W_fhr * sh)*FW - W_fwr * sw)*IC;
		Ws[buf][ty][tx] = *(float4*)(W + woffset);

		//load 4 elem from Y
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		Ims_fhr_fwr(Y_k, Y_fhr, Y_fwr);
		int yoffset = (Y_fhr * OW * OC) + Y_k;
		bool ly0 = (tohs0 >= -Y_fhr) && (tohs0 < OH - Y_fhr) && (tows0 >= -Y_fwr) && (tows0 < OW - Y_fwr);
		bool ly1 = (tohs1 >= -Y_fhr) && (tohs1 < OH - Y_fhr) && (tows1 >= -Y_fwr) && (tows1 < OW - Y_fwr);
		bool ly2 = (tohs2 >= -Y_fhr) && (tohs2 < OH - Y_fhr) && (tows2 >= -Y_fwr) && (tows2 < OW - Y_fwr);
		bool ly3 = (tohs3 >= -Y_fhr) && (tohs3 < OH - Y_fhr) && (tows3 >= -Y_fwr) && (tows3 < OW - Y_fwr);
		Ys[buf][tx][ty].x = (ly0 ? deltaY[Y0 + yoffset] : 0);
		Ys[buf][tx][ty].y = (ly1 ? deltaY[Y1 + yoffset] : 0);
		Ys[buf][tx][ty].z = (ly2 ? deltaY[Y2 + yoffset] : 0);
		Ys[buf][tx][ty].w = (ly3 ? deltaY[Y3 + yoffset] : 0);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];
		float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];

		simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
		simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
		simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
		simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
		simdMM4(v8, y1.x, w0); simdMM4(v9, y1.x, w1);
		simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
		simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
		simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
	}

	*(float4*)(deltaX + X0) = v0;  *(float4*)(deltaX + X0 + 4) = v1;
	*(float4*)(deltaX + X1) = v2;  *(float4*)(deltaX + X1 + 4) = v3;
	*(float4*)(deltaX + X2) = v4;  *(float4*)(deltaX + X2 + 4) = v5;
	*(float4*)(deltaX + X3) = v6;  *(float4*)(deltaX + X3 + 4) = v7;
	*(float4*)(deltaX + X4) = v8;  *(float4*)(deltaX + X4 + 4) = v9;
	*(float4*)(deltaX + X5) = v10; *(float4*)(deltaX + X5 + 4) = v11;
	*(float4*)(deltaX + X6) = v12; *(float4*)(deltaX + X6 + 4) = v13;
	*(float4*)(deltaX + X7) = v14; *(float4*)(deltaX + X7 + 4) = v15;
}

#endif

//=================================================================
//remode: W -> CW
//(BLOCK_SIZE*8, BLOCK_SIZE*8)
//(IH % sh == 0) && (IW % sw == 0), double buffered shared_memory
#ifndef X_KERNEL_V15
#define X_KERNEL_V15

#define X_kv15(stream, LB, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM) \
	ksn_kernel_v15<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>3), (GM>>LB>>3), (sh*sw)), dim3(1<<LB, 1<<LB) >>>\
			(deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, CFH, CFW)

//LB = 4: Size = 1.53125, Time = 0.794 msec, Performace = 4141.48 GFlop/s
//LB = 4: Size = 1.125, Time = 0.716 msec, Performace = 3374.19 GFlop/s
//LB = 3:Size = 1.125, Time = 0.876 msec, Performace = 2757.9 GFlop/s
template<int LB, int STEP>
__global__ void ksn_kernel_v15(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ CW, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw,
	int mCFH, int mCFW)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];

	//prepare for GZ = sh*sw 
	int bz = blockIdx.z;
	CW += bz * (mCFH * mCFW * OC * IC);//CW[y, x]

	int y = bz / sw, x = bz - y * sw;//x = bz % sw
	ph = ph - y; pw = pw - x;
	int ihs = -ph + (-(ph > 0) & ((ph - y + sh - 1) / sh * sh));
	int iws = -pw + (-(pw > 0) & ((pw - x + sw - 1) / sw * sw));

	//prepare for GK(bz(y, x)) = CFW * CFH * OC
	int CFH = (FH - y + sh - 1) / sh;
	int CFW = (FW - x + sw - 1) / sw;
	const int CFH_CFW = CFH * CFW, GK = CFH_CFW * OC;

	//prepare for GN = IC
	int ic0 = ((blockIdx.x << LB) + tx) << 3;
	CW += ic0 + ((ty >= STEP) << 2);
	deltaX += ic0;

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = ((blockIdx.y << LB) + ty) << 3;
	int j1 = j0 + 1, j2 = j0 + 2, j3 = j0 + 3;
	int j4 = j0 + 4, j5 = j0 + 5, j6 = j0 + 6, j7 = j0 + 7;
	int IW_slice = IW / sw, IH_IW_slice = (IH / sh) * IW_slice;
	Ims_n_ih_iw(j0, n0, ih0, iw0);
	Ims_n_ih_iw(j1, n1, ih1, iw1);
	Ims_n_ih_iw(j2, n2, ih2, iw2);
	Ims_n_ih_iw(j3, n3, ih3, iw3);
	Ims_n_ih_iw(j4, n4, ih4, iw4);
	Ims_n_ih_iw(j5, n5, ih5, iw5);
	Ims_n_ih_iw(j6, n6, ih6, iw6);
	Ims_n_ih_iw(j7, n7, ih7, iw7);
	int X0 = ((n0*IH + ih0)*IW + iw0) * IC;
	int X1 = ((n1*IH + ih1)*IW + iw1) * IC;
	int X2 = ((n2*IH + ih2)*IW + iw2) * IC;
	int X3 = ((n3*IH + ih3)*IW + iw3) * IC;
	int X4 = ((n4*IH + ih4)*IW + iw4) * IC;
	int X5 = ((n5*IH + ih5)*IW + iw5) * IC;
	int X6 = ((n6*IH + ih6)*IW + iw6) * IC;
	int X7 = ((n7*IH + ih7)*IW + iw7) * IC;

	bool flagX = (tx >= STEP);
	int oph = CFH - 1, opw = CFW - 1;
	int tohs0 = ((IF_int(flagX, ih4, ih0) + ph) >> 1) - oph;
	int tohs1 = ((IF_int(flagX, ih5, ih1) + ph) >> 1) - oph;
	int tohs2 = ((IF_int(flagX, ih6, ih2) + ph) >> 1) - oph;
	int tohs3 = ((IF_int(flagX, ih7, ih3) + ph) >> 1) - oph;
	int tows0 = ((IF_int(flagX, iw4, iw0) + pw) >> 1) - opw;
	int tows1 = ((IF_int(flagX, iw5, iw1) + pw) >> 1) - opw;
	int tows2 = ((IF_int(flagX, iw6, iw2) + pw) >> 1) - opw;
	int tows3 = ((IF_int(flagX, iw7, iw3) + pw) >> 1) - opw;
	int Y0 = ((IF_int(flagX, n4, n0)*OH + tohs0)*OW + tows0) * OC;
	int Y1 = ((IF_int(flagX, n5, n1)*OH + tohs1)*OW + tows1) * OC;
	int Y2 = ((IF_int(flagX, n6, n2)*OH + tohs2)*OW + tows2) * OC;
	int Y3 = ((IF_int(flagX, n7, n3)*OH + tohs3)*OW + tows3) * OC;

	//load 4 elem from W
	int W_k = ty - ((ty >= STEP) << LB >> 1);
	Ws[buf][ty][tx] = *(float4*)(CW + W_k * IC);

	//load 4 elem from deltaY
	int Y_k = tx - ((tx >= STEP) << LB >> 1);
	Ims_oc_fhr_fwr(Y_k, Y_oc, Y_fhr, Y_fwr);
	int yoffset = ((Y_fhr * OW) + Y_fwr)*OC + Y_oc;
	bool ly0 = (tohs0 >= -Y_fhr) && (tohs0 < OH - Y_fhr) && (tows0 >= -Y_fwr) && (tows0 < OW - Y_fwr);
	bool ly1 = (tohs1 >= -Y_fhr) && (tohs1 < OH - Y_fhr) && (tows1 >= -Y_fwr) && (tows1 < OW - Y_fwr);
	bool ly2 = (tohs2 >= -Y_fhr) && (tohs2 < OH - Y_fhr) && (tows2 >= -Y_fwr) && (tows2 < OW - Y_fwr);
	bool ly3 = (tohs3 >= -Y_fhr) && (tohs3 < OH - Y_fhr) && (tows3 >= -Y_fwr) && (tows3 < OW - Y_fwr);
	Ys[buf][tx][ty].x = (ly0 ? deltaY[Y0 + yoffset] : 0);
	Ys[buf][tx][ty].y = (ly1 ? deltaY[Y1 + yoffset] : 0);
	Ys[buf][tx][ty].z = (ly2 ? deltaY[Y2 + yoffset] : 0);
	Ys[buf][tx][ty].w = (ly3 ? deltaY[Y3 + yoffset] : 0);
	__syncthreads();

	//compute area-------------------------------------------------
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
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];
			float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];

			simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
			simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
			simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
			simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
			simdMM4(v8, y1.x, w0); simdMM4(v9, y1.x, w1);
			simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
			simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
			simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
		}
		buf ^= 1;

		//load 4 elem from W
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		Ws[buf][ty][tx] = *(float4*)(CW + W_k * IC);

		//load 4 elem from Y
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		Ims_oc_fhr_fwr(Y_k, Y_oc, Y_fhr, Y_fwr);
		int yoffset = ((Y_fhr * OW) + Y_fwr)*OC + Y_oc;
		bool ly0 = (tohs0 >= -Y_fhr) && (tohs0 < OH - Y_fhr) && (tows0 >= -Y_fwr) && (tows0 < OW - Y_fwr);
		bool ly1 = (tohs1 >= -Y_fhr) && (tohs1 < OH - Y_fhr) && (tows1 >= -Y_fwr) && (tows1 < OW - Y_fwr);
		bool ly2 = (tohs2 >= -Y_fhr) && (tohs2 < OH - Y_fhr) && (tows2 >= -Y_fwr) && (tows2 < OW - Y_fwr);
		bool ly3 = (tohs3 >= -Y_fhr) && (tohs3 < OH - Y_fhr) && (tows3 >= -Y_fwr) && (tows3 < OW - Y_fwr);
		Ys[buf][tx][ty].x = (ly0 ? deltaY[Y0 + yoffset] : 0);
		Ys[buf][tx][ty].y = (ly1 ? deltaY[Y1 + yoffset] : 0);
		Ys[buf][tx][ty].z = (ly2 ? deltaY[Y2 + yoffset] : 0);
		Ys[buf][tx][ty].w = (ly3 ? deltaY[Y3 + yoffset] : 0);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];
		float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];

		simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
		simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
		simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
		simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
		simdMM4(v8, y1.x, w0); simdMM4(v9, y1.x, w1);
		simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
		simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
		simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
	}

	*(float4*)(deltaX + X0) = v0;  *(float4*)(deltaX + X0 + 4) = v1;
	*(float4*)(deltaX + X1) = v2;  *(float4*)(deltaX + X1 + 4) = v3;
	*(float4*)(deltaX + X2) = v4;  *(float4*)(deltaX + X2 + 4) = v5;
	*(float4*)(deltaX + X3) = v6;  *(float4*)(deltaX + X3 + 4) = v7;
	*(float4*)(deltaX + X4) = v8;  *(float4*)(deltaX + X4 + 4) = v9;
	*(float4*)(deltaX + X5) = v10; *(float4*)(deltaX + X5 + 4) = v11;
	*(float4*)(deltaX + X6) = v12; *(float4*)(deltaX + X6 + 4) = v13;
	*(float4*)(deltaX + X7) = v14; *(float4*)(deltaX + X7 + 4) = v15;
}

#endif



//remode: W -> CW
//(BLOCK_SIZE*8, BLOCK_SIZE*8)
//(IH % sh == 0) && (IW % sw == 0), double buffered shared_memory
#ifndef X_KERNEL_V16
#define X_KERNEL_V16

#define X_kv16(stream, LB, deltaY, OH, OW, CW, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM) \
	ksn_kernel_v16<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>3), (GM>>LB>>3), (sh*sw)), dim3(1<<LB, 1<<LB) >>>\
			(deltaY, OH, OW, CW, FH, FW, (CFH*CFW*OC*IC), deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw)

//LB = 4: Size = 1.125, Time = 0.716 msec, Performace = 3374.19 GFlop/s
//LB = 3:Size = 1.125, Time = 0.876 msec, Performace = 2757.9 GFlop/s
template<int LB, int STEP>
__global__ void ksn_kernel_v16(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ CW, int FH, int FW, int CWstride,
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];

	//prepare for GZ = sh*sw 
	int bz = blockIdx.z; CW += bz * CWstride;//CW[y, x]

	int y = bz / sw, x = bz - y * sw;//x = bz % sw
	ph = ph - y; pw = pw - x;
	int ihs = -ph + (-(ph > 0) & ((ph - y + sh - 1) / sh * sh));
	int iws = -pw + (-(pw > 0) & ((pw - x + sw - 1) / sw * sw));


	//CFH = (FH - y + sh - 1) / sh
	//CFH = (FH + sh - 1 - y) / sh

	//prepare for GK(bz(y, x)) = CFW * CFH * OC
	int CFH = (FH - y + sh - 1) / sh;
	int CFW = (FW - x + sw - 1) / sw;
	const int CFH_CFW = CFH * CFW, GK = CFH_CFW * OC;

	//prepare for GN = IC
	int ic0 = ((blockIdx.x << LB) + tx) << 3;
	CW += ic0 + ((ty >= STEP) << 2);
	deltaX += ic0;

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = ((blockIdx.y << LB) + ty) << 3;
	int j1 = j0 + 1, j2 = j0 + 2, j3 = j0 + 3;
	int j4 = j0 + 4, j5 = j0 + 5, j6 = j0 + 6, j7 = j0 + 7;
	int IW_slice = IW / sw, IH_IW_slice = (IH / sh) * IW_slice;
	Ims_n_ih_iw(j0, n0, ih0, iw0);
	Ims_n_ih_iw(j1, n1, ih1, iw1);
	Ims_n_ih_iw(j2, n2, ih2, iw2);
	Ims_n_ih_iw(j3, n3, ih3, iw3);
	Ims_n_ih_iw(j4, n4, ih4, iw4);
	Ims_n_ih_iw(j5, n5, ih5, iw5);
	Ims_n_ih_iw(j6, n6, ih6, iw6);
	Ims_n_ih_iw(j7, n7, ih7, iw7);
	int X0 = ((n0*IH + ih0)*IW + iw0) * IC;
	int X1 = ((n1*IH + ih1)*IW + iw1) * IC;
	int X2 = ((n2*IH + ih2)*IW + iw2) * IC;
	int X3 = ((n3*IH + ih3)*IW + iw3) * IC;
	int X4 = ((n4*IH + ih4)*IW + iw4) * IC;
	int X5 = ((n5*IH + ih5)*IW + iw5) * IC;
	int X6 = ((n6*IH + ih6)*IW + iw6) * IC;
	int X7 = ((n7*IH + ih7)*IW + iw7) * IC;

	bool flagX = (tx >= STEP);
	int oph = CFH - 1, opw = CFW - 1;
	int tohs0 = ((IF_int(flagX, ih4, ih0) + ph) >> 1) - oph;
	int tohs1 = ((IF_int(flagX, ih5, ih1) + ph) >> 1) - oph;
	int tohs2 = ((IF_int(flagX, ih6, ih2) + ph) >> 1) - oph;
	int tohs3 = ((IF_int(flagX, ih7, ih3) + ph) >> 1) - oph;
	int tows0 = ((IF_int(flagX, iw4, iw0) + pw) >> 1) - opw;
	int tows1 = ((IF_int(flagX, iw5, iw1) + pw) >> 1) - opw;
	int tows2 = ((IF_int(flagX, iw6, iw2) + pw) >> 1) - opw;
	int tows3 = ((IF_int(flagX, iw7, iw3) + pw) >> 1) - opw;
	int Y0 = ((IF_int(flagX, n4, n0)*OH + tohs0)*OW + tows0) * OC;
	int Y1 = ((IF_int(flagX, n5, n1)*OH + tohs1)*OW + tows1) * OC;
	int Y2 = ((IF_int(flagX, n6, n2)*OH + tohs2)*OW + tows2) * OC;
	int Y3 = ((IF_int(flagX, n7, n3)*OH + tohs3)*OW + tows3) * OC;

	//load 4 elem from W
	int W_k = ty - ((ty >= STEP) << LB >> 1);
	Ws[buf][ty][tx] = *(float4*)(CW + W_k * IC);

	//load 4 elem from deltaY
	int Y_k = tx - ((tx >= STEP) << LB >> 1);
	Ims_oc_fhr_fwr(Y_k, Y_oc, Y_fhr, Y_fwr);
	int yoffset = ((Y_fhr * OW) + Y_fwr)*OC + Y_oc;
	bool ly0 = (tohs0 >= -Y_fhr) && (tohs0 < OH - Y_fhr) && (tows0 >= -Y_fwr) && (tows0 < OW - Y_fwr);
	bool ly1 = (tohs1 >= -Y_fhr) && (tohs1 < OH - Y_fhr) && (tows1 >= -Y_fwr) && (tows1 < OW - Y_fwr);
	bool ly2 = (tohs2 >= -Y_fhr) && (tohs2 < OH - Y_fhr) && (tows2 >= -Y_fwr) && (tows2 < OW - Y_fwr);
	bool ly3 = (tohs3 >= -Y_fhr) && (tohs3 < OH - Y_fhr) && (tows3 >= -Y_fwr) && (tows3 < OW - Y_fwr);
	Ys[buf][tx][ty].x = (ly0 ? deltaY[Y0 + yoffset] : 0);
	Ys[buf][tx][ty].y = (ly1 ? deltaY[Y1 + yoffset] : 0);
	Ys[buf][tx][ty].z = (ly2 ? deltaY[Y2 + yoffset] : 0);
	Ys[buf][tx][ty].w = (ly3 ? deltaY[Y3 + yoffset] : 0);
	__syncthreads();

	//compute area-------------------------------------------------
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
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];
			float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];

			simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
			simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
			simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
			simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
			simdMM4(v8, y1.x, w0); simdMM4(v9, y1.x, w1);
			simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
			simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
			simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
		}
		buf ^= 1;

		//load 4 elem from W
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		Ws[buf][ty][tx] = *(float4*)(CW + W_k * IC);

		//load 4 elem from Y
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		Ims_oc_fhr_fwr(Y_k, Y_oc, Y_fhr, Y_fwr);
		int yoffset = ((Y_fhr * OW) + Y_fwr)*OC + Y_oc;
		bool ly0 = (tohs0 >= -Y_fhr) && (tohs0 < OH - Y_fhr) && (tows0 >= -Y_fwr) && (tows0 < OW - Y_fwr);
		bool ly1 = (tohs1 >= -Y_fhr) && (tohs1 < OH - Y_fhr) && (tows1 >= -Y_fwr) && (tows1 < OW - Y_fwr);
		bool ly2 = (tohs2 >= -Y_fhr) && (tohs2 < OH - Y_fhr) && (tows2 >= -Y_fwr) && (tows2 < OW - Y_fwr);
		bool ly3 = (tohs3 >= -Y_fhr) && (tohs3 < OH - Y_fhr) && (tows3 >= -Y_fwr) && (tows3 < OW - Y_fwr);
		Ys[buf][tx][ty].x = (ly0 ? deltaY[Y0 + yoffset] : 0);
		Ys[buf][tx][ty].y = (ly1 ? deltaY[Y1 + yoffset] : 0);
		Ys[buf][tx][ty].z = (ly2 ? deltaY[Y2 + yoffset] : 0);
		Ys[buf][tx][ty].w = (ly3 ? deltaY[Y3 + yoffset] : 0);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];
		float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];

		simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
		simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
		simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
		simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
		simdMM4(v8, y1.x, w0); simdMM4(v9, y1.x, w1);
		simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
		simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
		simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
	}

	*(float4*)(deltaX + X0) = v0;  *(float4*)(deltaX + X0 + 4) = v1;
	*(float4*)(deltaX + X1) = v2;  *(float4*)(deltaX + X1 + 4) = v3;
	*(float4*)(deltaX + X2) = v4;  *(float4*)(deltaX + X2 + 4) = v5;
	*(float4*)(deltaX + X3) = v6;  *(float4*)(deltaX + X3 + 4) = v7;
	*(float4*)(deltaX + X4) = v8;  *(float4*)(deltaX + X4 + 4) = v9;
	*(float4*)(deltaX + X5) = v10; *(float4*)(deltaX + X5 + 4) = v11;
	*(float4*)(deltaX + X6) = v12; *(float4*)(deltaX + X6 + 4) = v13;
	*(float4*)(deltaX + X7) = v14; *(float4*)(deltaX + X7 + 4) = v15;
}

#endif


//remode: W -> CW
//(BLOCK_SIZE*8, BLOCK_SIZE*8)
//(IH % sh == 0) && (IW % sw == 0), double buffered shared_memory
#ifndef X_KERNEL_V17
#define X_KERNEL_V17

#define X_kv17(stream, LB, deltaY, OH, OW, CW, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM) \
	ksn_kernel_v17<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>3), (GM>>LB>>3), (sh*sw)), dim3(1<<LB, 1<<LB) >>>\
			(deltaY, OH, OW, CW, FH, FW, (CFH*CFW*OC*IC), deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw)

//LB = 4: Size = 1.125, Time = 0.71 msec, Performace = 3402.7 GFlop/s
//LB = 3: Size = 1.125, Time = 0.876 msec, Performace = 2757.9 GFlop/s
template<int LB, int STEP>
__global__ void ksn_kernel_v17(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ CW, int FH, int FW, int CWstride,
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];

	//prepare for GZ = sh*sw 
	int bz = blockIdx.z; CW += bz * CWstride;//CW[y, x]

	int y = bz / sw, x = bz - y * sw;//x = bz % sw
	ph = ph - y; pw = pw - x;
	int ihs = -ph + (-(ph > 0) & ((ph - y + sh - 1) / sh * sh));
	int iws = -pw + (-(pw > 0) & ((pw - x + sw - 1) / sw * sw));

	//prepare for GK(bz(y, x)) = CFW * CFH * OC
	int CFH = (FH - y + sh - 1) / sh, oph = CFH - 1;
	int CFW = (FW - x + sw - 1) / sw, opw = CFW - 1;
	const int CFH_CFW = CFH * CFW, GK = CFH_CFW * OC;

	//prepare for GN = IC
	int ic0 = ((blockIdx.x << LB) + tx) << 3;
	CW += ic0 + ((ty >= STEP) << 2);
	deltaX += ic0;

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = ((blockIdx.y << LB) + ty) << 3;
	int j1 = j0 + 1, j2 = j0 + 2, j3 = j0 + 3;
	int j4 = j0 + 4, j5 = j0 + 5, j6 = j0 + 6, j7 = j0 + 7;
	int IW_slice = IW / sw, IH_IW_slice = (IH / sh) * IW_slice;
	Ims_n_ih_iw(j0, n0, ih0, iw0);
	Ims_n_ih_iw(j1, n1, ih1, iw1);
	Ims_n_ih_iw(j2, n2, ih2, iw2);
	Ims_n_ih_iw(j3, n3, ih3, iw3);
	Ims_n_ih_iw(j4, n4, ih4, iw4);
	Ims_n_ih_iw(j5, n5, ih5, iw5);
	Ims_n_ih_iw(j6, n6, ih6, iw6);
	Ims_n_ih_iw(j7, n7, ih7, iw7);
	int X0 = ((n0*IH + ih0)*IW + iw0) * IC;
	int X1 = ((n1*IH + ih1)*IW + iw1) * IC;
	int X2 = ((n2*IH + ih2)*IW + iw2) * IC;
	int X3 = ((n3*IH + ih3)*IW + iw3) * IC;
	int X4 = ((n4*IH + ih4)*IW + iw4) * IC;
	int X5 = ((n5*IH + ih5)*IW + iw5) * IC;
	int X6 = ((n6*IH + ih6)*IW + iw6) * IC;
	int X7 = ((n7*IH + ih7)*IW + iw7) * IC;

	bool flagX = (tx >= STEP);
	int tohs0 = ((IF_int(flagX, ih4, ih0) + ph) >> 1) - oph;
	int tohs1 = ((IF_int(flagX, ih5, ih1) + ph) >> 1) - oph;
	int tohs2 = ((IF_int(flagX, ih6, ih2) + ph) >> 1) - oph;
	int tohs3 = ((IF_int(flagX, ih7, ih3) + ph) >> 1) - oph;
	int tows0 = ((IF_int(flagX, iw4, iw0) + pw) >> 1) - opw;
	int tows1 = ((IF_int(flagX, iw5, iw1) + pw) >> 1) - opw;
	int tows2 = ((IF_int(flagX, iw6, iw2) + pw) >> 1) - opw;
	int tows3 = ((IF_int(flagX, iw7, iw3) + pw) >> 1) - opw;
	int Y0 = ((IF_int(flagX, n4, n0)*OH + tohs0)*OW + tows0) * OC;
	int Y1 = ((IF_int(flagX, n5, n1)*OH + tohs1)*OW + tows1) * OC;
	int Y2 = ((IF_int(flagX, n6, n2)*OH + tohs2)*OW + tows2) * OC;
	int Y3 = ((IF_int(flagX, n7, n3)*OH + tohs3)*OW + tows3) * OC;

	//load 4 elem from W
	int W_k = ty - ((ty >= STEP) << LB >> 1);
	Ws[buf][ty][tx] = *(float4*)(CW + W_k * IC);

	//load 4 elem from deltaY
	int Y_k = tx - ((tx >= STEP) << LB >> 1);
	Ims_oc_fhr_fwr(Y_k, Y_oc, Y_fhr, Y_fwr);
	int yoffset = ((Y_fhr * OW) + Y_fwr)*OC + Y_oc;
	bool ly0 = (tohs0 >= -Y_fhr) && (tohs0 < OH - Y_fhr) && (tows0 >= -Y_fwr) && (tows0 < OW - Y_fwr);
	bool ly1 = (tohs1 >= -Y_fhr) && (tohs1 < OH - Y_fhr) && (tows1 >= -Y_fwr) && (tows1 < OW - Y_fwr);
	bool ly2 = (tohs2 >= -Y_fhr) && (tohs2 < OH - Y_fhr) && (tows2 >= -Y_fwr) && (tows2 < OW - Y_fwr);
	bool ly3 = (tohs3 >= -Y_fhr) && (tohs3 < OH - Y_fhr) && (tows3 >= -Y_fwr) && (tows3 < OW - Y_fwr);
	Ys[buf][tx][ty].x = (ly0 ? deltaY[Y0 + yoffset] : 0);
	Ys[buf][tx][ty].y = (ly1 ? deltaY[Y1 + yoffset] : 0);
	Ys[buf][tx][ty].z = (ly2 ? deltaY[Y2 + yoffset] : 0);
	Ys[buf][tx][ty].w = (ly3 ? deltaY[Y3 + yoffset] : 0);
	__syncthreads();

	//compute area-------------------------------------------------
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
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];
			float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];

			simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
			simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
			simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
			simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
			simdMM4(v8, y1.x, w0); simdMM4(v9, y1.x, w1);
			simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
			simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
			simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
		}
		buf ^= 1;

		//load 4 elem from W
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		Ws[buf][ty][tx] = *(float4*)(CW + W_k * IC);

		//load 4 elem from Y
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		Ims_oc_fhr_fwr(Y_k, Y_oc, Y_fhr, Y_fwr);
		int yoffset = ((Y_fhr * OW) + Y_fwr)*OC + Y_oc;
		bool ly0 = (tohs0 >= -Y_fhr) && (tohs0 < OH - Y_fhr) && (tows0 >= -Y_fwr) && (tows0 < OW - Y_fwr);
		bool ly1 = (tohs1 >= -Y_fhr) && (tohs1 < OH - Y_fhr) && (tows1 >= -Y_fwr) && (tows1 < OW - Y_fwr);
		bool ly2 = (tohs2 >= -Y_fhr) && (tohs2 < OH - Y_fhr) && (tows2 >= -Y_fwr) && (tows2 < OW - Y_fwr);
		bool ly3 = (tohs3 >= -Y_fhr) && (tohs3 < OH - Y_fhr) && (tows3 >= -Y_fwr) && (tows3 < OW - Y_fwr);
		Ys[buf][tx][ty].x = (ly0 ? deltaY[Y0 + yoffset] : 0);
		Ys[buf][tx][ty].y = (ly1 ? deltaY[Y1 + yoffset] : 0);
		Ys[buf][tx][ty].z = (ly2 ? deltaY[Y2 + yoffset] : 0);
		Ys[buf][tx][ty].w = (ly3 ? deltaY[Y3 + yoffset] : 0);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];
		float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];

		simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
		simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
		simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
		simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
		simdMM4(v8, y1.x, w0); simdMM4(v9, y1.x, w1);
		simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
		simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
		simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
	}

	*(float4*)(deltaX + X0) = v0;  *(float4*)(deltaX + X0 + 4) = v1;
	*(float4*)(deltaX + X1) = v2;  *(float4*)(deltaX + X1 + 4) = v3;
	*(float4*)(deltaX + X2) = v4;  *(float4*)(deltaX + X2 + 4) = v5;
	*(float4*)(deltaX + X3) = v6;  *(float4*)(deltaX + X3 + 4) = v7;
	*(float4*)(deltaX + X4) = v8;  *(float4*)(deltaX + X4 + 4) = v9;
	*(float4*)(deltaX + X5) = v10; *(float4*)(deltaX + X5 + 4) = v11;
	*(float4*)(deltaX + X6) = v12; *(float4*)(deltaX + X6 + 4) = v13;
	*(float4*)(deltaX + X7) = v14; *(float4*)(deltaX + X7 + 4) = v15;
}

#endif


//(Y: BLOCK_SIZE*8, X:BLOCK_SIZE*8), OC % (BLOCK_SIZE/2) == 0
#ifndef DCONV3D_DX_KERNEL_SPLIT_IMS_KERNEL_8_8R
#define DCONV3D_DX_KERNEL_SPLIT_IMS_KERNEL_8_8R

//LB = 4: Size = 1.53125, Time = 0.778 msec, Performace = 4226.65 GFlop/s
//LB = 4: Size = 1.125, Time = 0.702 msec, Performace = 3441.48 GFlop/s
//LB = 3: Size = 1.125, Time = 0.876 msec, Performace = 2757.9 GFlop/s
template<int LB, int STEP>
__global__ void ksIms_kernel_8_8R_copy(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ CW, int FH, int FW, int CWstride,
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];

	//prepare for GZ = sh*sw 
	int bz = blockIdx.z; CW += bz * CWstride;//CW[y, x]

	int y = bz / sw, x = bz - y * sw;//x = bz % sw
	ph = ph - y; pw = pw - x;
	int ihs = -ph + (-(ph > 0) & ((ph - y + sh - 1) / sh * sh));
	int iws = -pw + (-(pw > 0) & ((pw - x + sw - 1) / sw * sw));

	//prepare for GK(bz(y, x)) = CFW * CFH * OC
	int CFH = (FH - y + sh - 1) / sh, oph = CFH - 1;
	int CFW = (FW - x + sw - 1) / sw, opw = CFW - 1;
	const int CFW_OC = CFW * OC, GK = CFH * CFW_OC;

	//prepare for GN = IC
	int ic0 = ((blockIdx.x << LB) + tx) << 3;
	CW += ic0 + ((ty >= STEP) << 2);
	deltaX += ic0;

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = ((blockIdx.y << LB) + ty) << 3;
	int j1 = j0 + 1, j2 = j0 + 2, j3 = j0 + 3;
	int j4 = j0 + 4, j5 = j0 + 5, j6 = j0 + 6, j7 = j0 + 7;
	int IW_slice = IW / sw, IH_IW_slice = (IH / sh) * IW_slice;
	Ims_n_ih_iw(j0, n0, ih0, iw0);
	Ims_n_ih_iw(j1, n1, ih1, iw1);
	Ims_n_ih_iw(j2, n2, ih2, iw2);
	Ims_n_ih_iw(j3, n3, ih3, iw3);
	Ims_n_ih_iw(j4, n4, ih4, iw4);
	Ims_n_ih_iw(j5, n5, ih5, iw5);
	Ims_n_ih_iw(j6, n6, ih6, iw6);
	Ims_n_ih_iw(j7, n7, ih7, iw7);
	int X0 = ((n0*IH + ih0)*IW + iw0) * IC;
	int X1 = ((n1*IH + ih1)*IW + iw1) * IC;
	int X2 = ((n2*IH + ih2)*IW + iw2) * IC;
	int X3 = ((n3*IH + ih3)*IW + iw3) * IC;
	int X4 = ((n4*IH + ih4)*IW + iw4) * IC;
	int X5 = ((n5*IH + ih5)*IW + iw5) * IC;
	int X6 = ((n6*IH + ih6)*IW + iw6) * IC;
	int X7 = ((n7*IH + ih7)*IW + iw7) * IC;

	bool flagX = (tx >= STEP);
	int tohs0 = ((IF_int(flagX, ih4, ih0) + ph) >> 1) - oph;
	int tohs1 = ((IF_int(flagX, ih5, ih1) + ph) >> 1) - oph;
	int tohs2 = ((IF_int(flagX, ih6, ih2) + ph) >> 1) - oph;
	int tohs3 = ((IF_int(flagX, ih7, ih3) + ph) >> 1) - oph;
	int tows0 = ((IF_int(flagX, iw4, iw0) + pw) >> 1) - opw;
	int tows1 = ((IF_int(flagX, iw5, iw1) + pw) >> 1) - opw;
	int tows2 = ((IF_int(flagX, iw6, iw2) + pw) >> 1) - opw;
	int tows3 = ((IF_int(flagX, iw7, iw3) + pw) >> 1) - opw;
	int Y0 = ((IF_int(flagX, n4, n0)*OH + tohs0)*OW + tows0) * OC;
	int Y1 = ((IF_int(flagX, n5, n1)*OH + tohs1)*OW + tows1) * OC;
	int Y2 = ((IF_int(flagX, n6, n2)*OH + tohs2)*OW + tows2) * OC;
	int Y3 = ((IF_int(flagX, n7, n3)*OH + tohs3)*OW + tows3) * OC;

	//load 4 elem from W
	int W_k = ty - ((ty >= STEP) << LB >> 1);
	Ws[buf][ty][tx] = *(float4*)(CW + W_k * IC);

	//load 4 elem from deltaY
	int Y_k = tx - ((tx >= STEP) << LB >> 1);
	Ims_fhr_fwr(Y_k, Y_fhr, Y_fwr);
	int OW_OC = OW * OC, yoffset = Y_fhr * OW_OC + Y_k;
	bool ly0 = (tohs0 >= -Y_fhr) && (tohs0 < OH - Y_fhr) && (tows0 >= -Y_fwr) && (tows0 < OW - Y_fwr);
	bool ly1 = (tohs1 >= -Y_fhr) && (tohs1 < OH - Y_fhr) && (tows1 >= -Y_fwr) && (tows1 < OW - Y_fwr);
	bool ly2 = (tohs2 >= -Y_fhr) && (tohs2 < OH - Y_fhr) && (tows2 >= -Y_fwr) && (tows2 < OW - Y_fwr);
	bool ly3 = (tohs3 >= -Y_fhr) && (tohs3 < OH - Y_fhr) && (tows3 >= -Y_fwr) && (tows3 < OW - Y_fwr);
	Ys[buf][tx][ty].x = (ly0 ? deltaY[Y0 + yoffset] : 0);
	Ys[buf][tx][ty].y = (ly1 ? deltaY[Y1 + yoffset] : 0);
	Ys[buf][tx][ty].z = (ly2 ? deltaY[Y2 + yoffset] : 0);
	Ys[buf][tx][ty].w = (ly3 ? deltaY[Y3 + yoffset] : 0);
	__syncthreads();

	//compute area-------------------------------------------------
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
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];
			float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];

			simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
			simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
			simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
			simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
			simdMM4(v8, y1.x, w0); simdMM4(v9, y1.x, w1);
			simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
			simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
			simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
		}
		buf ^= 1;

		//load 4 elem from W
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		Ws[buf][ty][tx] = *(float4*)(CW + W_k * IC);

		//load 4 elem from Y
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		Ims_fhr_fwr(Y_k, Y_fhr, Y_fwr);
		int yoffset = Y_fhr * OW_OC + Y_k;
		bool ly0 = (tohs0 >= -Y_fhr) && (tohs0 < OH - Y_fhr) && (tows0 >= -Y_fwr) && (tows0 < OW - Y_fwr);
		bool ly1 = (tohs1 >= -Y_fhr) && (tohs1 < OH - Y_fhr) && (tows1 >= -Y_fwr) && (tows1 < OW - Y_fwr);
		bool ly2 = (tohs2 >= -Y_fhr) && (tohs2 < OH - Y_fhr) && (tows2 >= -Y_fwr) && (tows2 < OW - Y_fwr);
		bool ly3 = (tohs3 >= -Y_fhr) && (tohs3 < OH - Y_fhr) && (tows3 >= -Y_fwr) && (tows3 < OW - Y_fwr);
		Ys[buf][tx][ty].x = (ly0 ? deltaY[Y0 + yoffset] : 0);
		Ys[buf][tx][ty].y = (ly1 ? deltaY[Y1 + yoffset] : 0);
		Ys[buf][tx][ty].z = (ly2 ? deltaY[Y2 + yoffset] : 0);
		Ys[buf][tx][ty].w = (ly3 ? deltaY[Y3 + yoffset] : 0);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];
		float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];

		simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
		simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
		simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
		simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
		simdMM4(v8, y1.x, w0); simdMM4(v9, y1.x, w1);
		simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
		simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
		simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
	}

	*(float4*)(deltaX + X0) = v0;  *(float4*)(deltaX + X0 + 4) = v1;
	*(float4*)(deltaX + X1) = v2;  *(float4*)(deltaX + X1 + 4) = v3;
	*(float4*)(deltaX + X2) = v4;  *(float4*)(deltaX + X2 + 4) = v5;
	*(float4*)(deltaX + X3) = v6;  *(float4*)(deltaX + X3 + 4) = v7;
	*(float4*)(deltaX + X4) = v8;  *(float4*)(deltaX + X4 + 4) = v9;
	*(float4*)(deltaX + X5) = v10; *(float4*)(deltaX + X5 + 4) = v11;
	*(float4*)(deltaX + X6) = v12; *(float4*)(deltaX + X6 + 4) = v13;
	*(float4*)(deltaX + X7) = v14; *(float4*)(deltaX + X7 + 4) = v15;
}

#endif


//(IH_slice, IW_slice) % 8 == 0
//(Y: BLOCK_SIZE*8, X:BLOCK_SIZE*8), OC % (BLOCK_SIZE/2) == 0
#ifndef DCONV3D_DX_KERNEL_SPLIT_IMS_KERNEL_8_8R8
#define DCONV3D_DX_KERNEL_SPLIT_IMS_KERNEL_8_8R8

#define ksImsR_88R8(stream, LB, deltaY, OH, OW, CW, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM) \
	ksIms_kernel_8_8R8<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>3), (GM>>LB>>3), (sh*sw)), dim3(1<<LB, 1<<LB) >>>\
			(deltaY, OH, OW, CW, FH, FW, (CFH*CFW*OC*IC), deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw)

//LB = 4: Size = 1.53125, Time = 0.752 msec, Performace = 4372.78 GFlop/s
//LB = 4: Size = 1.125, Time = 0.68 msec, Performace = 3552.82 GFlop/s
//LB = 3: Size = 1.125, Time = 0.876 msec, Performace = 2757.9 GFlop/s
template<int LB, int STEP>
__global__ void ksIms_kernel_8_8R8_copy(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ CW, int FH, int FW, int CWstride,
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];

	//prepare for GZ = sh*sw 
	int bz = blockIdx.z; CW += bz * CWstride;//CW[y, x]

	int y = bz / sw, x = bz - y * sw;//x = bz % sw
	ph = ph - y; pw = pw - x;
	int ihs = -ph + (-(ph > 0) & ((ph - y + sh - 1) / sh * sh));
	int iws = -pw + (-(pw > 0) & ((pw - x + sw - 1) / sw * sw));

	//prepare for GK(bz(y, x)) = CFW * CFH * OC
	int CFH = (FH - y + sh - 1) / sh, oph = CFH - 1;
	int CFW = (FW - x + sw - 1) / sw, opw = CFW - 1;
	const int CFW_OC = CFW * OC, GK = CFH * CFW_OC;

	//prepare for GN = IC
	int ic0 = ((blockIdx.x << LB) + tx) << 3;
	CW += ic0 + ((ty >= STEP) << 2);

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = ((blockIdx.y << LB) + ty) << 3;
	int IW_slice = IW / sw, IH_IW_slice = (IH / sh) * IW_slice;
	Ims_n_ih_iw(j0, n0, ih0, iw0);
	int X0 = ((n0*IH + ih0)*IW + iw0)*IC + ic0;

	int tohs0 = (ih0 + ph) / sh - oph;
	int tows0 = (iw0 + ((tx >= STEP) << 3) + pw) / sw - opw;
	int Y1 = ((n0*OH + tohs0)*OW + tows0 + 1) * OC;
	OH -= tohs0;

	//load 4 elem from W
	int W_k = ty - ((ty >= STEP) << LB >> 1);
	Ws[buf][ty][tx] = *(float4*)(CW + W_k * IC);

	//load 4 elem from deltaY
	int Y_k = tx - ((tx >= STEP) << LB >> 1);
	Ims_fhr_fwr(Y_k, Y_fhr, Y_fwr);
	int OW_OC = OW * OC, yoffset = Y_fhr * OW_OC + Y_k;
	bool ly = (tohs0 >= -Y_fhr) && (Y_fhr < OH);
	bool ly0 = ly && (tows0 >= -Y_fwr) && (tows0 < OW - Y_fwr);
	bool ly1 = ly && (tows0 + 1 >= -Y_fwr) && (tows0 + 1 < OW - Y_fwr);
	bool ly2 = ly && (tows0 + 2 >= -Y_fwr) && (tows0 + 2 < OW - Y_fwr);
	bool ly3 = ly && (tows0 + 3 >= -Y_fwr) && (tows0 + 3 < OW - Y_fwr);
	Ys[buf][tx][ty].x = (ly0 ? deltaY[Y1 - OC + yoffset] : 0);
	Ys[buf][tx][ty].y = (ly1 ? deltaY[Y1 + yoffset] : 0);
	Ys[buf][tx][ty].z = (ly2 ? deltaY[Y1 + OC + yoffset] : 0);
	Ys[buf][tx][ty].w = (ly3 ? deltaY[Y1 + (OC << 1) + yoffset] : 0);
	__syncthreads();

	//compute area-------------------------------------------------
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
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];
			float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];

			simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
			simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
			simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
			simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
			simdMM4(v8, y1.x, w0); simdMM4(v9, y1.x, w1);
			simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
			simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
			simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
		}
		buf ^= 1;


		//load 4 elem from Y
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		Ims_fhr_fwr(Y_k, Y_fhr, Y_fwr);
		int yoffset = Y_fhr * OW_OC + Y_k;
		bool ly = (tohs0 >= -Y_fhr) && (Y_fhr < OH);
		bool ly0 = ly && (tows0 >= -Y_fwr) && (tows0 < OW - Y_fwr);
		bool ly1 = ly && (tows0 + 1 >= -Y_fwr) && (tows0 + 1 < OW - Y_fwr);
		bool ly2 = ly && (tows0 + 2 >= -Y_fwr) && (tows0 + 2 < OW - Y_fwr);
		bool ly3 = ly && (tows0 + 3 >= -Y_fwr) && (tows0 + 3 < OW - Y_fwr);
		Ys[buf][tx][ty].x = (ly0 ? deltaY[Y1 - OC + yoffset] : 0);
		Ys[buf][tx][ty].y = (ly1 ? deltaY[Y1 + yoffset] : 0);
		Ys[buf][tx][ty].z = (ly2 ? deltaY[Y1 + OC + yoffset] : 0);
		Ys[buf][tx][ty].w = (ly3 ? deltaY[Y1 + (OC << 1) + yoffset] : 0);

		//load 4 elem from W
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		Ws[buf][ty][tx] = *(float4*)(CW + W_k * IC);

		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];
		float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];

		simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
		simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
		simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
		simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
		simdMM4(v8, y1.x, w0); simdMM4(v9, y1.x, w1);
		simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
		simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
		simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
	}

	IC *= sw;//IC * sw
	int X1 = X0 + IC, X2 = X1 + IC, X3 = X2 + IC;
	int X4 = X3 + IC, X5 = X4 + IC, X6 = X5 + IC, X7 = X6 + IC;

	*(float4*)(deltaX + X0) = v0;  *(float4*)(deltaX + X0 + 4) = v1;
	*(float4*)(deltaX + X1) = v2;  *(float4*)(deltaX + X1 + 4) = v3;
	*(float4*)(deltaX + X2) = v4;  *(float4*)(deltaX + X2 + 4) = v5;
	*(float4*)(deltaX + X3) = v6;  *(float4*)(deltaX + X3 + 4) = v7;
	*(float4*)(deltaX + X4) = v8;  *(float4*)(deltaX + X4 + 4) = v9;
	*(float4*)(deltaX + X5) = v10; *(float4*)(deltaX + X5 + 4) = v11;
	*(float4*)(deltaX + X6) = v12; *(float4*)(deltaX + X6 + 4) = v13;
	*(float4*)(deltaX + X7) = v14; *(float4*)(deltaX + X7 + 4) = v15;

	*(float4*)(deltaX + X0) = v0;  *(float4*)(deltaX + X0 + 4) = v1;
	*(float4*)(deltaX + X1) = v2;  *(float4*)(deltaX + X1 + 4) = v3;
	*(float4*)(deltaX + X2) = v4;  *(float4*)(deltaX + X2 + 4) = v5;
	*(float4*)(deltaX + X3) = v6;  *(float4*)(deltaX + X3 + 4) = v7;
	*(float4*)(deltaX + X4) = v8;  *(float4*)(deltaX + X4 + 4) = v9;
	*(float4*)(deltaX + X5) = v10; *(float4*)(deltaX + X5 + 4) = v11;
	*(float4*)(deltaX + X6) = v12; *(float4*)(deltaX + X6 + 4) = v13;
	*(float4*)(deltaX + X7) = v14; *(float4*)(deltaX + X7 + 4) = v15;
}

#endif