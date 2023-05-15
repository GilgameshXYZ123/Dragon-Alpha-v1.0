
#define crossAdd_kv1(stream, LB, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM) \
	crossAdd_kernel_v1<LB>\
		<<< dim3(GM>>LB, GN>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, sh,sw,ph,pw)

//GM = N*OH*OW
//GN = OC
//38.5
//k11suv: 90
//k11: 77
template<int LB>
__global__ void crossAdd_kernel_v1(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int sh, int sw, int ph, int pw)
{
	int bx = blockIdx.x, by = blockIdx.y;
	int tx = threadIdx.x, ty = threadIdx.y;

	int oc = (by << LB) + ty;
	int j = (bx << LB) + tx;
	const int OH_OW = OH * OW;
	get_n_oh_ow(j, n, oh, ow);
	float dy = get4d(deltaY, n, oh, ow, oc, OH, OW, OC);

	for (int fh = 0; fh < FH; fh++)
		for (int fw = 0; fw < FW; fw++)
		{
			int ih = oh * sh - ph + fh;
			int iw = ow * sw - pw + fw;

			if (ih < 0 || iw < 0 || ih >= IH || iw >= IW) continue;

			for (int ic = 0; ic < IC; ic++)
			{
				float w = get4d(W, oc, fh, fw, ic, FH, FW, IC);
				atomicAdd(&get4d(deltaX, n, ih, iw, ic, IH, IW, IC), w*dy);
			}
		}
}

#define crossAdd_kv2(stream, LB, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM) \
	crossAdd_kernel_v2<LB>\
		<<< dim3(GM>>LB, GN>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, sh,sw,ph,pw)

//GM = N*OH*OW
//GN = OC
//LB = 3: 31.6551 GFlop/s
//k11suv: 40.394 GFlop/s
//k11: 26.8861 GFlop/s
template<int LB>
__global__ void crossAdd_kernel_v2(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int sh, int sw, int ph, int pw)
{
	int bx = blockIdx.x, by = blockIdx.y;
	int tx = threadIdx.x, ty = threadIdx.y;

	int oc = (by << LB) + ty;
	int j = (bx << LB) + tx;
	const int OH_OW = OH * OW;
	get_n_oh_ow(j, n, oh, ow);
	float dy = deltaY[j*OC + oc];

	for (int fh = 0; fh < FH; fh++)
	{
		int ih = oh * sh - ph + fh;
		for (int fw = 0; fw < FW; fw++)
		{
			int iw = ow * sw - pw + fw;

			if (ih < 0 || iw < 0 || ih >= IH || iw >= IW) continue;

			for (int ic = 0; ic < IC; ic++)
			{
				float w = get4d(W, oc, fh, fw, ic, FH, FW, IC);
				float dx = w * dy;
				atomicAdd(&get4d(deltaX, n, ih, iw, ic, IH, IW, IC), dx);
			}
		}
	}
}


#define crossAdd_kv3(stream, LB, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM) \
	crossAdd_kernel_v3<LB, (1<<LB)>\
		<<< dim3(GM>>LB, GN>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, sh,sw,ph,pw)

//GM = N*OH*OW
//GN = OC
//LB = 3: 34.6551 GFlop/s
//k11suv: 40.394 GFlop/s
//k11: 26.8861 GFlop/s
template<int LB, int STEP>
__global__ void crossAdd_kernel_v3(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int sh, int sw, int ph, int pw)
{
	int bx = blockIdx.x, by = blockIdx.y;
	int tx = threadIdx.x, ty = threadIdx.y;

	int oc = (by << LB) + ty;
	int j = (bx << LB) + tx;
	const int OH_OW = OH * OW;
	get_n_oh_ow(j, n, oh, ow);
	const int tih = oh * sh - ph;
	const int tiw = ow * sw - pw;
	float dy = deltaY[j*OC + oc];

	const int FW_IC = FW * IC, GK = FH * FW_IC;
	for (int k = 0; k < GK; k++)
	{
		int fh = k / FW_IC, kr = k % FW_IC;
		int fw = kr / IC, ic = kr % IC;

		float w = get4d(W, oc, fh, fw, ic, FH, FW, IC);

		int ih = tih + fh;
		int iw = tiw + fw;
		if (ih < 0 || iw < 0 || ih >= IH || iw >= IW) continue;

		float dx = w * dy;
		atomicAdd(&get4d(deltaX, n, ih, iw, ic, IH, IW, IC), dx);
	}
}


#define crossAdd_kv4(stream, LB, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM) \
	crossAdd_kernel_v4<LB, (1<<LB)>\
		<<< dim3(GM>>LB, GN>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, sh,sw,ph,pw)

//GM = N*OH*OW
//GN = OC
//LB = 3: 34.6551 GFlop/s
//k11suv: 40.394 GFlop/s
//k11: 26.8861 GFlop/s
//Size = 0.03125, Time = 2.09 msec, Performace = 32.1095 GFlop/s
template<int LB, int STEP>
__global__ void crossAdd_kernel_v4(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int sh, int sw, int ph, int pw)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	int oc = (blockIdx.y << LB) + ty;
	int j = (blockIdx.x << LB) + tx;

	const int OH_OW = OH * OW;
	get_n_oh_ow(j, n, oh, ow);
	const int tih = oh * sh - ph;
	const int tiw = ow * sw - pw;
	float dy = deltaY[j*OC + oc];

	__shared__ float  Ws[1 << LB][1 << LB];
	__shared__ int Xoffsets[1 << LB][1 << LB];

	const int FW_IC = FW * IC, GK = FH * FW_IC;
	for (int ok = 0, OK = GK >> LB; ok < OK; ok++)
	{
		const int W_k = (ok << LB) + tx;//threads with the same ty
		Ws[tx][ty] = W[oc*GK + W_k];//W_k(fh, fw, ic)

		int dX_k = (ok << LB) + ty;
		int X_fh = dX_k / FW_IC; dX_k -= X_fh * FW_IC;//with the same tx
		int X_fw = dX_k / IC, X_ic = dX_k - X_fw * IC;
		int ih = tih + X_fh, iw = tiw + X_fw;
		bool write = (ih >= 0 && ih < IH && iw >= 0 && iw < IW);
		Xoffsets[ty][tx] = write * (((((n*IH) + ih)*IW) + iw)*IC + X_ic) - !write;
		__syncthreads();

		for (int ik = 0; ik < STEP; ik++)
		{
			int Xoffset = Xoffsets[ik][tx];
			if (Xoffset == -1) continue;

			float w = Ws[ik][ty];
			float dx = w * dy;//deltaX; the same n, ih, iw -> the same
			atomicAdd(&deltaX[Xoffset], dx);
		}
		__syncthreads();
	}

}


#define crossAdd_kv5(stream, LB, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM) \
	crossAdd_kernel_v5<LB, (1<<LB)>\
		<<< dim3(GM>>LB, GN>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, sh,sw,ph,pw)

//GM = N*OH*OW
//GN = OC
//LB = 3: 34.6551 GFlop/s
//k11suv: 40.394 GFlop/s
//k11: ize = 0.0625, Time = 0.35 msec, Performace = 383.479 GFlop/s
template<int LB, int STEP>
__global__ void crossAdd_kernel_v5(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int sh, int sw, int ph, int pw)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	int oc = (blockIdx.y << LB) + ty;
	int j = (blockIdx.x << LB) + tx;

	const int OH_OW = OH * OW;
	get_n_oh_ow(j, n, oh, ow);
	const int tih = oh * sh - ph;
	const int tiw = ow * sw - pw;
	float dy = deltaY[j*OC + oc];

	__shared__ float   Ws[1 << LB][1 << LB];
	__shared__ float  dXs[1 << LB][1 << LB];
	__shared__ int Xaddrs[1 << LB][1 << LB];

	const int FW_IC = FW * IC, GK = FH * FW_IC;
	for (int ok = 0, OK = GK >> LB; ok < OK; ok++)
	{
		const int W_k = (ok << LB) + tx;//threads with the same ty
		Ws[tx][ty] = W[oc*GK + W_k];//W_k(fh, fw, ic)

		int dX_k = (ok << LB) + ty;
		int X_fh = dX_k / FW_IC; dX_k -= X_fh * FW_IC;//with the same tx
		int X_fw = dX_k / IC, X_ic = dX_k - X_fw * IC;
		int ih = tih + X_fh, iw = tiw + X_fw;

		bool write = (ih >= 0 && ih < IH && iw >= 0 && iw < IW);
		Xaddrs[ty][tx] = write * (((((n*IH) + ih)*IW) + iw)*IC + X_ic) - !write;
		dXs[ty][tx] = 0.0f;//deltaX; the same n, ih, iw -> the same
		__syncthreads();

		for (int ik = 0; ik < STEP; ik++)
		{
			float w = Ws[ik][ty] * (Xaddrs[ik][tx] != -1);
			float dx = w * dy;
			atomicAdd(&dXs[ik][tx], dx);
		}
		__syncthreads();

		atomicAdd(&deltaX[Xaddrs[ty][tx]], dXs[ty][tx]);
		__syncthreads();
	}
}


#define crossAdd_kv6(stream, LB, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM) \
	crossAdd_kernel_v6<LB, (1<<LB)>\
		<<< dim3(GM>>LB>>1, GN>>LB>>1), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, sh,sw,ph,pw)

//GM = N*OH*OW
//GN = OC
//LB = 3: 34.6551 GFlop/s
//k11suv: 40.394 GFlop/s
//k11: Size = 0.0625, Time = 0.353333 msec, Performace = 379.861 GFlop/s
template<int LB, int STEP>
__global__ void crossAdd_kernel_v6(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int sh, int sw, int ph, int pw)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	int oc0 = ((blockIdx.y << LB) + ty) << 1, oc1 = oc0 + 1;
	int j0 = ((blockIdx.x << LB) + tx) << 1, j1 = j0 + 1;
	const int OH_OW = OH * OW;
	get_n_oh_ow(j0, n0, oh0, ow0);
	get_n_oh_ow(j1, n1, oh1, ow1);
	const int tih0 = oh0 * sh - ph, tih1 = oh1 * sh - ph;
	const int tiw0 = ow0 * sw - pw, tiw1 = ow1 * sw - pw;
	float2 dy0 = *(float2*)(&deltaY[j0*OC + oc0]);
	float2 dy1 = *(float2*)(&deltaY[j1*OC + oc0]);

	__shared__ float2   Ws[1 << LB][1 << LB];
	__shared__ float2  dXs[1 << LB][1 << LB];
	__shared__ int2 Xaddrs[1 << LB][1 << LB];

	const int FW_IC = FW * IC, GK = FH * FW_IC;
	float2 dx;
	for (int ok = 0, OK = GK >> LB; ok < OK; ok++)
	{
		const int W_k = (ok << LB) + tx;//threads with the same ty
		Ws[tx][ty].x = W[oc0*GK + W_k];//for: oc0
		Ws[tx][ty].y = W[oc1*GK + W_k];//for: oc1

		int dX_k = (ok << LB) + ty;
		int X_fh = dX_k / FW_IC; dX_k -= X_fh * FW_IC;//with the same tx
		int X_fw = dX_k / IC, X_ic = dX_k - X_fw * IC;
		int ih0 = tih0 + X_fh, iw0 = tiw0 + X_fw;
		int ih1 = tih1 + X_fh, iw1 = tiw1 + X_fw;
		bool write0 = (ih0 >= 0) && (ih0 < IH) && (iw0 >= 0) && (iw0 < IW);
		bool write1 = (ih1 >= 0) && (ih1 < IH) && (iw1 >= 0) && (iw1 < IW);
		Xaddrs[ty][tx].x = write0 * (((((n0*IH) + ih0)*IW) + iw0)*IC + X_ic) - !write0;
		Xaddrs[ty][tx].y = write1 * (((((n1*IH) + ih1)*IW) + iw1)*IC + X_ic) - !write1;
		dXs[ty][tx] = make_float2(0, 0);//deltaX; the same n, ih, iw -> the same
		__syncthreads();

		for (int ik = 0; ik < STEP; ik++)
		{
			float2 w = Ws[ik][ty];
			int2 xaddr = Xaddrs[ik][tx];

			//for: oc0 -> X(n0, ih0, iw0, ic)
			dx.x = (xaddr.x != -1) * (w.x*dy0.x + w.y*dy0.y);
			dx.y = (xaddr.y != -1) * (w.x*dy1.x + w.y*dy1.y);
			atomicAdd(&(dXs[ik][tx].x), dx.x);
			atomicAdd(&(dXs[ik][tx].y), dx.y);
		}
		__syncthreads();

		atomicAdd(&deltaX[Xaddrs[ty][tx].x], dXs[ty][tx].x);//(oc0, oc1) -> (n0, ih0, iw0)
		atomicAdd(&deltaX[Xaddrs[ty][tx].y], dXs[ty][tx].y);//(oc0, oc1) -> (n1, ih1, iw1)
		__syncthreads();
	}
}


#define crossAdd_kv7(stream, LB, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM) \
	crossAdd_kernel_v7<LB, (1<<LB)>\
		<<< dim3(GM>>LB, GN>>LB>>1), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, sh,sw,ph,pw)

//Size = 0.0625, Time = 0.336667 msec, Performace = 398.667 GFlop/s
template<int LB, int STEP>
__global__ void crossAdd_kernel_v7(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int sh, int sw, int ph, int pw)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	__shared__ float2   Ws[1 << LB][1 << LB];
	__shared__ float   dXs[1 << LB][1 << LB];
	__shared__ int  Xaddrs[1 << LB][1 << LB];

	int oc0 = ((blockIdx.y << LB) + ty) << 1, oc1 = oc0 + 1;
	int j = (blockIdx.x << LB) + tx;
	const int OH_OW = OH * OW;
	get_n_oh_ow(j, n, oh, ow);
	const int tih = oh * sh - ph, tiw = ow * sh - ph;

	float2 dy = *(float2*)(&deltaY[j*OC + oc0]);

	const int FW_IC = FW * IC, GK = FH * FW_IC;
	for (int ok = 0, OK = GK >> LB; ok < OK; ok++)
	{
		const int W_k = (ok << LB) + tx;//threads with the same ty
		Ws[tx][ty].x = W[oc0*GK + W_k];//for: oc0
		Ws[tx][ty].y = W[oc1*GK + W_k];//for: oc1

		int dX_k = (ok << LB) + ty;
		int X_fh = dX_k / FW_IC; dX_k -= X_fh * FW_IC;//with the same tx
		int X_fw = dX_k / IC, X_ic = dX_k - X_fw * IC;

		int ih = tih + X_fh, iw = tiw + X_fw;
		bool write = (ih >= 0) && (ih < IH) && (iw >= 0) && (iw < IW);
		Xaddrs[ty][tx] = write * (((((n*IH) + ih)*IW) + iw)*IC + X_ic) - !write;
		dXs[ty][tx] = 0;//deltaX; the same n, ih, iw -> the same
		__syncthreads();

		for (int ik = 0; ik < STEP; ik++)
		{
			float2 w = Ws[ik][ty];
			int xaddr = Xaddrs[ik][tx];

			//for: oc0 -> X(n0, ih0, iw0, ic)
			float dx = (xaddr != -1) * (w.x*dy.x + w.y*dy.y);
			atomicAdd(&dXs[ik][tx], dx);
		}
		__syncthreads();

		atomicAdd(&deltaX[Xaddrs[ty][tx]], dXs[ty][tx]);//(oc0, oc1) -> (n0, ih0, iw0)
		__syncthreads();
	}
}


#define crossAdd_kv8(stream, LB, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM) \
	crossAdd_kernel_v8<LB, (1<<LB)>\
		<<< dim3(GM>>LB, GN>>LB>>1), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, sh,sw,ph,pw)

//Size = 0.25, Time = 1.25 msec, Performace = 429.497 GFlop/s
template<int LB, int STEP>
__global__ void crossAdd_kernel_v8(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int sh, int sw, int ph, int pw)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	__shared__ float2   Ws[1 << LB << LB];
	__shared__ float   dXs[1 << LB << LB];
	__shared__ int  Xaddrs[1 << LB << LB];

	int oc0 = ((blockIdx.y << LB) + ty) << 1;
	int j = (blockIdx.x << LB) + tx;
	const int OH_OW = OH * OW;
	get_n_oh_ow(j, n, oh, ow);
	const int tih = oh * sh - ph, tiw = ow * sh - ph;

	float2 dy = *(float2*)(&deltaY[j*OC + oc0]);
	const int FW_IC = FW * IC, GK = FH * FW_IC;
	oc0 *= GK; int oc1 = oc0 + GK;

	for (int ok = 0, OK = GK >> LB; ok < OK; ok++)
	{
		const int W_k = (ok << LB) + tx;//threads with the same ty
		Ws[(tx << LB) + ty].x = W[oc0 + W_k];//for: oc0
		Ws[(tx << LB) + ty].y = W[oc1 + W_k];//for: oc1

		int dX_k = (ok << LB) + ty;
		int X_fh = dX_k / FW_IC; dX_k -= X_fh * FW_IC;//with the same tx
		int X_fw = dX_k / IC, X_ic = dX_k - X_fw * IC;

		int ih = tih + X_fh, iw = tiw + X_fw;
		bool write = (ih >= 0) && (ih < IH) && (iw >= 0) && (iw < IW);
		Xaddrs[(ty << LB) + tx] = write * (((((n*IH) + ih)*IW) + iw)*IC + X_ic) - !write;
		dXs[(ty << LB) + tx] = 0;//deltaX; the same n, ih, iw -> the same
		__syncthreads();

		for (int ik = 0; ik < STEP; ik++)
		{
			float2 w = Ws[(ik << LB) + ty];
			int xaddr = Xaddrs[(ik << LB) + tx];

			//for: oc0 -> X(n0, ih0, iw0, ic)
			float dx = (xaddr != -1) * (w.x*dy.x + w.y*dy.y);
			atomicAdd(&dXs[(ik << LB) + tx], dx);
		}
		__syncthreads();

		atomicAdd(&deltaX[Xaddrs[(ty << LB) + tx]], dXs[(ty << LB) + tx]);//(oc0, oc1) -> (n0, ih0, iw0)
		__syncthreads();
	}
}

#define crossAdd_kv9(stream, LB, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM) \
	crossAdd_kernel_v9<LB, (1<<LB)>\
		<<< dim3(GM>>LB, GN>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, sh,sw,ph,pw)

//Size = 0.25, Time = 0.736667 msec, Performace = 728.784 GFlop/s
//Size = 0.5, Time = 1.24667 msec, Performace = 861.29 GFlop/s
template<int LB, int STEP>
__global__ void crossAdd_kernel_v9(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int sh, int sw, int ph, int pw)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	__shared__ float4   Ws[1 << LB << LB];
	__shared__ float   dXs[1 << LB << LB];
	__shared__ int  Xaddrs[1 << LB << LB];

	int oc0 = ((blockIdx.y << LB) + ty) << 2;
	int j = (blockIdx.x << LB) + tx;
	const int OH_OW = OH * OW;
	get_n_oh_ow(j, n, oh, ow);
	const int tih = oh * sh - ph, tiw = ow * sh - ph;

	float4 dy = *(float4*)(&deltaY[j*OC + oc0]);
	const int FW_IC = FW * IC, GK = FH * FW_IC;
	oc0 *= GK; int oc1 = oc0 + GK, oc2 = oc1 + GK, oc3 = oc2 + GK;

	for (int ok = 0, OK = GK >> LB; ok < OK; ok++)
	{
		const int W_k = (ok << LB) + tx;//threads with the same ty
		Ws[(tx << LB) + ty].x = W[oc0 + W_k];//for: oc0
		Ws[(tx << LB) + ty].y = W[oc1 + W_k];//for: oc1
		Ws[(tx << LB) + ty].z = W[oc2 + W_k];//for: oc2
		Ws[(tx << LB) + ty].w = W[oc3 + W_k];//for: oc3

		int dX_k = (ok << LB) + ty;
		int X_fh = dX_k / FW_IC; dX_k -= X_fh * FW_IC;//with the same tx
		int X_fw = dX_k / IC, X_ic = dX_k - X_fw * IC;
		int ih = tih + X_fh, iw = tiw + X_fw;

		bool write = (ih >= 0) && (ih < IH) && (iw >= 0) && (iw < IW);
		Xaddrs[(ty << LB) + tx] = write * (((((n*IH) + ih)*IW) + iw)*IC + X_ic) - !write;
		dXs[(ty << LB) + tx] = 0.0f;
		__syncthreads();

		for (int ik = 0; ik < STEP; ik++)
		{
			float4 w = Ws[(ik << LB) + ty]; int xaddr = Xaddrs[(ik << LB) + tx];
			float dx = (xaddr != -1) * (w.x*dy.x + w.y*dy.y + w.z*dy.z + w.w *dy.w);
			atomicAdd(&dXs[(ik << LB) + tx], dx);
		}
		__syncthreads();

		atomicAdd(&deltaX[Xaddrs[(ty << LB) + tx]], dXs[(ty << LB) + tx]);//(oc0, oc1) -> (n0, ih0, iw0)
		__syncthreads();
	}
}



#define crossAdd_kv10(stream, LB, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM) \
	crossAdd_kernel_v10<LB, (1<<LB)>\
		<<< dim3(GM>>LB, GN>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, sh,sw,ph,pw)

//Size = 0.25, Time = 0.736667 msec, Performace = 728.784 GFlop/s
//Size = 0.5, Time = 0.986667 msec, Performace = 1088.25 GFlop/s
template<int LB, int STEP>
__global__ void crossAdd_kernel_v10(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int sh, int sw, int ph, int pw)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	__shared__ float4   Ws[2 << LB << LB];
	__shared__ float   dXs[1 << LB << LB];
	__shared__ int  Xaddrs[1 << LB << LB];

	int oc0 = ((blockIdx.y << LB) + ty) << 3;
	int j = (blockIdx.x << LB) + tx;
	const int OH_OW = OH * OW;
	get_n_oh_ow(j, n, oh, ow);
	const int tih = oh * sh - ph, tiw = ow * sh - ph;

	j = j * OC + oc0;
	float4 dy0 = *(float4*)(&deltaY[j]);
	float4 dy1 = *(float4*)(&deltaY[j + 4]);

	const int FW_IC = FW * IC, GK = FH * FW_IC;
	oc0 *= GK; int oc1 = oc0 + GK, oc2 = oc1 + GK, oc3 = oc2 + GK;
	int oc4 = oc3 + GK, oc5 = oc4 + GK, oc6 = oc5 + GK, oc7 = oc6 + GK;

	const int Ws_xy = ((tx << LB) + ty) << 1;
	const int Xs_xy = (ty << LB) + tx;
	for (int ok = 0, OK = GK >> LB; ok < OK; ok++)
	{
		const int W_k = (ok << LB) + tx;//threads with the same ty
		Ws[Ws_xy].x = W[oc0 + W_k];//for: oc0
		Ws[Ws_xy].y = W[oc1 + W_k];//for: oc1
		Ws[Ws_xy].z = W[oc2 + W_k];//for: oc2
		Ws[Ws_xy].w = W[oc3 + W_k];//for: oc3
		Ws[Ws_xy + 1].x = W[oc4 + W_k];//for: oc4
		Ws[Ws_xy + 1].y = W[oc5 + W_k];//for: oc5
		Ws[Ws_xy + 1].z = W[oc6 + W_k];//for: oc6
		Ws[Ws_xy + 1].w = W[oc7 + W_k];//for: oc7

		int dX_k = (ok << LB) + ty;
		int X_fh = dX_k / FW_IC; dX_k -= X_fh * FW_IC;//with the same tx
		int X_fw = dX_k / IC, X_ic = dX_k - X_fw * IC;
		int ih = tih + X_fh, iw = tiw + X_fw;

		bool write = (ih >= 0) && (ih < IH) && (iw >= 0) && (iw < IW);
		Xaddrs[Xs_xy] = write * (((((n*IH) + ih)*IW) + iw)*IC + X_ic) - !write;
		dXs[Xs_xy] = 0.0f;
		__syncthreads();

		for (int ik = 0; ik < STEP; ++ik)
		{
			float4 w0 = Ws[(((ik << LB) + ty) << 1)];
			float4 w1 = Ws[(((ik << LB) + ty) << 1) + 1];

			float dx0 = w0.x*dy0.x + w0.y*dy0.y + w0.z*dy0.z + w0.w *dy0.w;
			float dx1 = w1.x*dy1.x + w1.y*dy1.y + w1.z*dy1.z + w1.w *dy1.w;
			float dx = (Xaddrs[(ik << LB) + tx] != -1) * (dx0 + dx1);
			atomicAdd(&dXs[(ik << LB) + tx], dx);
		}
		__syncthreads();

		atomicAdd(deltaX + Xaddrs[Xs_xy], dXs[Xs_xy]);//(oc0, oc1) -> (n0, ih0, iw0)
		__syncthreads();
	}
}


#define crossAdd_kv11(stream, LB, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, GN, GM) \
	crossAdd_kernel_v11<LB, (1<<LB)>\
		<<< dim3(GM>>LB, GN>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, sh,sw,ph,pw)

//Size = 0.25, Time = 0.736667 msec, Performace = 728.784 GFlop/s
//Size = 0.5, Time = 1.08333 msec, Performace = 991.146 GFlop/s
//LB = 3: Size = 0.5, Time = 1.63333 msec, Performace = 657.393 GFlop/s
template<int LB, int STEP>
__global__ void crossAdd_kernel_v11(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int sh, int sw, int ph, int pw)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4   Ws[2][2 << LB << LB];
	__shared__ float   dXs[2][1 << LB << LB];
	__shared__ int  Xaddrs[2][1 << LB << LB];

	int oc0 = ((blockIdx.y << LB) + ty) << 3, oc4 = oc0 + 4;
	int j = (blockIdx.x << LB) + tx;
	const int OH_OW = OH * OW;
	get_n_oh_ow(j, n, oh, ow);
	const int tih = oh * sh - ph, tiw = ow * sh - ph;

	j *= OC;
	float4 dy0 = *(float4*)(&deltaY[j + oc0]);
	float4 dy1 = *(float4*)(&deltaY[j + oc4]);

	const int FW_IC = FW * IC, GK = FH * FW_IC;
	oc0 *= GK; int oc1 = oc0 + GK, oc2 = oc1 + GK, oc3 = oc2 + GK;
	oc4 *= GK; int oc5 = oc4 + GK, oc6 = oc5 + GK, oc7 = oc6 + GK;

	const int W_k = tx;//preload 8 elements from W
	const int Ws_xy = ((tx << LB) + ty) << 1;
	Ws[buf][Ws_xy].x = W[oc0 + W_k];//for: oc0
	Ws[buf][Ws_xy].y = W[oc1 + W_k];//for: oc1
	Ws[buf][Ws_xy].z = W[oc2 + W_k];//for: oc2
	Ws[buf][Ws_xy].w = W[oc3 + W_k];//for: oc3
	Ws[buf][Ws_xy + 1].x = W[oc4 + W_k];//for: oc4
	Ws[buf][Ws_xy + 1].y = W[oc5 + W_k];//for: oc5
	Ws[buf][Ws_xy + 1].z = W[oc6 + W_k];//for: oc6
	Ws[buf][Ws_xy + 1].w = W[oc7 + W_k];//for: oc7

	int dX_k = ty;//precompute for Xaddr
	int X_fh = dX_k / FW_IC; dX_k -= X_fh * FW_IC;//with the same tx
	int X_fw = dX_k / IC, X_ic = dX_k - X_fw * IC;
	int ih = tih + X_fh, iw = tiw + X_fw;

	bool write = (ih >= 0) && (ih < IH) && (iw >= 0) && (iw < IW);
	int offset = ((((n*IH) + ih)*IW) + iw)*IC + X_ic;
	Xaddrs[buf][(ty << LB) + tx] = write * offset - !write;
	dXs[buf][(ty << LB) + tx] = 0.0f;
	__syncthreads();

	for (int ok = 1, OK = GK >> LB; ok < OK; ok++)
	{
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 w0 = Ws[buf][(((ik << LB) + ty) << 1)];
			float4 w1 = Ws[buf][(((ik << LB) + ty) << 1) + 1];
			float dx0 = w0.x*dy0.x + w0.y*dy0.y + w0.z*dy0.z + w0.w *dy0.w;
			float dx1 = w1.x*dy1.x + w1.y*dy1.y + w1.z*dy1.z + w1.w *dy1.w;

			float dx = (Xaddrs[buf][(ik << LB) + tx] != -1) * (dx0 + dx1);
			atomicAdd(&dXs[buf][(ik << LB) + tx], dx);
		}
		__syncthreads();
		atomicAdd(deltaX + Xaddrs[buf][(ty << LB) + tx], dXs[buf][(ty << LB) + tx]);

		buf ^= 1;
		int W_k = (ok << LB) + tx;
		Ws[buf][Ws_xy].x = W[oc0 + W_k];//for: oc0
		Ws[buf][Ws_xy].y = W[oc1 + W_k];//for: oc1
		Ws[buf][Ws_xy].z = W[oc2 + W_k];//for: oc2
		Ws[buf][Ws_xy].w = W[oc3 + W_k];//for: oc3
		Ws[buf][Ws_xy + 1].x = W[oc4 + W_k];//for: oc4
		Ws[buf][Ws_xy + 1].y = W[oc5 + W_k];//for: oc5
		Ws[buf][Ws_xy + 1].z = W[oc6 + W_k];//for: oc6
		Ws[buf][Ws_xy + 1].w = W[oc7 + W_k];//for: oc7

		int dX_k = (ok << LB) + ty;
		int X_fh = dX_k / FW_IC; dX_k -= X_fh * FW_IC;//with the same tx
		int X_fw = dX_k / IC, X_ic = dX_k - X_fw * IC;
		int ih = tih + X_fh, iw = tiw + X_fw;

		bool write = (ih >= 0) && (ih < IH) && (iw >= 0) && (iw < IW);
		int offset = ((((n*IH) + ih)*IW) + iw)*IC + X_ic;
		Xaddrs[buf][(ty << LB) + tx] = write * offset - !write;
		dXs[buf][(ty << LB) + tx] = 0.0f;
		__syncthreads();

	}
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 w0 = Ws[buf][(((ik << LB) + ty) << 1)];
		float4 w1 = Ws[buf][(((ik << LB) + ty) << 1) + 1];
		float dx0 = w0.x*dy0.x + w0.y*dy0.y + w0.z*dy0.z + w0.w *dy0.w;
		float dx1 = w1.x*dy1.x + w1.y*dy1.y + w1.z*dy1.z + w1.w *dy1.w;

		float dx = (Xaddrs[buf][(ik << LB) + tx] != -1) * (dx0 + dx1);
		atomicAdd(&dXs[buf][(ik << LB) + tx], dx);
	}
	__syncthreads();
	atomicAdd(deltaX + Xaddrs[buf][(ty << LB) + tx], dXs[buf][(ty << LB) + tx]);
}


	//GK = FH * FW * IC
//LB = 4: GK % 16 == 0: Size = 0.234619, Time = 0.404 msec, Performace = 1247.13 GFlop/s
	template<int LB, int STEP>
	__global__ void crossAdd_kernel_8_2(
		const float* __restrict__ deltaY, int OH, int OW,
		const float* __restrict__ W, int FH, int FW,
		float* __restrict__ deltaX, int IH, int IW,
		int IC, int OC,
		int sh, int sw, int ph, int pw)
	{
		int tx = threadIdx.x, ty = threadIdx.y;

		bool buf = 0;
		__shared__ float4   Ws[2][1 << LB][(2 << LB) + 1];
		__shared__ float2  dXs[2][1 << LB][(1 << LB) + 1];
		__shared__ int2 Xaddrs[2][1 << LB][(1 << LB) + 1];

		int oc0 = ((blockIdx.y << LB) + ty) << 3;

		int j0 = ((blockIdx.x << LB) + tx) << 1, j1 = j0 + 1;
		const int OH_OW = OH * OW;
		get_n_oh_ow(j0, n0, oh0, ow0);
		get_n_oh_ow(j1, n1, oh1, ow1);
		const int tih0 = oh0 * sh - ph, tiw0 = ow0 * sw - pw; j0 *= OC;
		const int tih1 = oh1 * sh - ph, tiw1 = ow1 * sw - pw; j1 *= OC;
		float4 dy00 = *(float4*)(&deltaY[j0 + oc0]), dy01 = *(float4*)(&deltaY[j0 + oc0 + 4]);
		float4 dy10 = *(float4*)(&deltaY[j1 + oc0]), dy11 = *(float4*)(&deltaY[j1 + oc0 + 4]);

		const int FW_IC = FW * IC, GK = FH * FW_IC; oc0 *= GK;
		int oc1 = oc0 + GK, oc2 = oc1 + GK, oc3 = oc2 + GK;
		int oc4 = oc3 + GK, oc5 = oc4 + GK, oc6 = oc5 + GK, oc7 = oc6 + GK;

		//load 8 elements from W[OC, FH, FW, IC]
		const int W_k = tx;
		Ws[buf][tx][(ty << 1)].x = W[oc0 + W_k];
		Ws[buf][tx][(ty << 1)].y = W[oc1 + W_k];
		Ws[buf][tx][(ty << 1)].z = W[oc2 + W_k];
		Ws[buf][tx][(ty << 1)].w = W[oc3 + W_k];
		Ws[buf][tx][(ty << 1) + 1].x = W[oc4 + W_k];
		Ws[buf][tx][(ty << 1) + 1].y = W[oc5 + W_k];
		Ws[buf][tx][(ty << 1) + 1].z = W[oc6 + W_k];
		Ws[buf][tx][(ty << 1) + 1].w = W[oc7 + W_k];

		//compute 2 addresses for Xaddr[M, IH, IW, IC]
		int dX_k = ty;
		int X_fh = dX_k / FW_IC; dX_k -= X_fh * FW_IC;//with the same tx
		int X_fw = dX_k / IC, X_ic = dX_k - X_fw * IC;
		int ih0 = tih0 + X_fh, iw0 = tiw0 + X_fw;
		int ih1 = tih1 + X_fh, iw1 = tiw1 + X_fw;
		bool write0 = (ih0 >= 0) && (ih0 < IH) && (iw0 >= 0) && (iw0 < IW);
		bool write1 = (ih1 >= 0) && (ih1 < IH) && (iw1 >= 0) && (iw1 < IW);
		int offset0 = ((((n0*IH) + ih0)*IW) + iw0)*IC + X_ic;
		int offset1 = ((((n1*IH) + ih1)*IW) + iw1)*IC + X_ic;
		Xaddrs[buf][ty][tx].x = (write0 * offset0) - !write0;
		Xaddrs[buf][ty][tx].y = (write1 * offset1) - !write1;
		dXs[buf][ty][tx] = make_float2(0, 0);
		__syncthreads();

		for (int ok = 1, OK = GK >> LB; ok < OK; ok++)
		{
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 w0 = Ws[buf][ik][(ty << 1)], w1 = Ws[buf][ik][(ty << 1) + 1];
				int2 xaddr = Xaddrs[buf][ik][tx];

				float dx0 = (CrossAdd_SUM4(w0, dy00) + CrossAdd_SUM4(w1, dy01));
				float dx1 = (CrossAdd_SUM4(w0, dy10) + CrossAdd_SUM4(w1, dy11));

				atomicAdd_block(&(dXs[buf][ik][tx].x), (xaddr.x != -1)*dx0);
				atomicAdd_block(&(dXs[buf][ik][tx].y), (xaddr.y != -1)*dx1);
			}
			__syncthreads();
			atomicAdd(deltaX + Xaddrs[buf][ty][tx].x, dXs[buf][ty][tx].x);
			atomicAdd(deltaX + Xaddrs[buf][ty][tx].y, dXs[buf][ty][tx].y);
			buf ^= 1;

			//load 8 elements from W[OC, FH, FW, IC]
			int W_k = (ok << LB) + tx;
			Ws[buf][tx][(ty << 1)].x = W[oc0 + W_k];
			Ws[buf][tx][(ty << 1)].y = W[oc1 + W_k];
			Ws[buf][tx][(ty << 1)].z = W[oc2 + W_k];
			Ws[buf][tx][(ty << 1)].w = W[oc3 + W_k];
			Ws[buf][tx][(ty << 1) + 1].x = W[oc4 + W_k];
			Ws[buf][tx][(ty << 1) + 1].y = W[oc5 + W_k];
			Ws[buf][tx][(ty << 1) + 1].z = W[oc6 + W_k];
			Ws[buf][tx][(ty << 1) + 1].w = W[oc7 + W_k];

			//compute 2 addresses for Xaddr[M, IH, IW, IC]
			int dX_k = (ok << LB) + ty;
			int X_fh = dX_k / FW_IC; dX_k -= X_fh * FW_IC;//with the same tx
			int X_fw = dX_k / IC, X_ic = dX_k - X_fw * IC;
			int ih0 = tih0 + X_fh, iw0 = tiw0 + X_fw;
			int ih1 = tih1 + X_fh, iw1 = tiw1 + X_fw;
			bool write0 = (ih0 >= 0) && (ih0 < IH) && (iw0 >= 0) && (iw0 < IW);
			bool write1 = (ih1 >= 0) && (ih1 < IH) && (iw1 >= 0) && (iw1 < IW);
			int offset0 = ((((n0*IH) + ih0)*IW) + iw0)*IC + X_ic;
			int offset1 = ((((n1*IH) + ih1)*IW) + iw1)*IC + X_ic;
			Xaddrs[buf][ty][tx].x = (write0 * offset0) - !write0;
			Xaddrs[buf][ty][tx].y = (write1 * offset1) - !write1;
			dXs[buf][ty][tx] = make_float2(0, 0);
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 w0 = Ws[buf][ik][(ty << 1)], w1 = Ws[buf][ik][(ty << 1) + 1];
			int2 xaddr = Xaddrs[buf][ik][tx];

			float dx0 = (CrossAdd_SUM4(w0, dy00) + CrossAdd_SUM4(w1, dy01));
			float dx1 = (CrossAdd_SUM4(w0, dy10) + CrossAdd_SUM4(w1, dy11));

			atomicAdd_block(&(dXs[buf][ik][tx].x), (xaddr.x != -1)*dx0);
			atomicAdd_block(&(dXs[buf][ik][tx].y), (xaddr.y != -1)*dx1);
		}
		__syncthreads();
		atomicAdd(deltaX + Xaddrs[buf][ty][tx].x, dXs[buf][ty][tx].x);
		atomicAdd(deltaX + Xaddrs[buf][ty][tx].y, dXs[buf][ty][tx].y);
	}



	//((((tn0*IH) + ih0)*IW) + iw0)*IC + X_ic;
	//tn0*IH*IW*IC + ih0*IW*IC + iw0*IC + X_ic
	//tn0*IH*IW*IC + (tih0 + X_fh)*IW*IC + (tiw0 + X_fw)*IC + X_ic
	//((tn0*IH + tih0)*IW + tiw0)*IC + (X_fh*IW + X_fw)*IC + X_ic
	//GK = FH * FW * IC
	//LB = 4: GK % 16 == 0: Size = 0.234619, Time = 0.404 msec, Performace = 1247.13 GFlop/s
	template<int LB, int STEP>
	__global__ void crossAdd_kernel_8_4(
		const float* __restrict__ deltaY, int OH, int OW,
		const float* __restrict__ W, int FH, int FW,
		float* __restrict__ deltaX, int IH, int IW,
		int IC, int OC,
		int sh, int sw, int ph, int pw)
	{
		int tx = threadIdx.x, ty = threadIdx.y;

		bool buf = 0;
		__shared__ float4  Ws[2][1 << LB >> 1][(2 << LB) + 1];
		__shared__ float  dXs[2][1 << LB >> 1][(2 << LB) + 2];
		__shared__ int Xaddrs[2][1 << LB >> 1][(2 << LB) + 2];

		//prepare for GN = OC
		int oc0 = ((blockIdx.y << LB) + ty) << 3;
		const int FW_IC = FW * IC, GK = FH * FW_IC;
		const int toc0 = (((tx & 1) << 2) + oc0)*GK;
		const int toc1 = toc0 + GK, toc2 = toc1 + GK, toc3 = toc2 + GK;

		//prepared for GM = N * OH * OW
		int j0 = ((blockIdx.x << LB) + tx) << 1, j1 = j0 + 1;
		const int OH_OW = OH * OW;
		get_n_oh_ow(j0, n0, oh0, ow0);
		get_n_oh_ow(j1, n1, oh1, ow1);
		bool flagY = (ty & 1);
		const int tn0 = (n1 - n0)*flagY + n0;
		const int tih0 = ((oh1 - oh0)*flagY + oh0) * sh - ph;
		const int tiw0 = ((ow1 - ow0)*flagY + ow0) * sw - pw;
		const int Xoffset0 = ((tn0*IH + tih0)*IW + tiw0)*IC;

		//load 4 elements from W[OC, FH, FW, IC]
		const int W_k = tx >> 1;
		const int Ws_x = (tx >> 1), Ws_y = (ty << 1) + (tx & 1);
		Ws[buf][Ws_x][Ws_y].x = W[toc0 + W_k];
		Ws[buf][Ws_x][Ws_y].y = W[toc1 + W_k];
		Ws[buf][Ws_x][Ws_y].z = W[toc2 + W_k];
		Ws[buf][Ws_x][Ws_y].w = W[toc3 + W_k];

		//compute 1 address for Xaddr[M, IH, IW, IC]
		int dX_k = ty >> 1;
		int X_fh = dX_k / FW_IC; dX_k -= X_fh * FW_IC;//with the same tx
		int X_fw = dX_k / IC, X_ic = dX_k - X_fw * IC;
		int ih0 = tih0 + X_fh, iw0 = tiw0 + X_fw;
		bool write0 = (tih0 >= -X_fh) && (tih0 < IH - X_fh) && (tiw0 >= -X_fw) && (tiw0 < IW - X_fw);
		int xoffset = Xoffset0 + (X_fh*IW + X_fw)*IC + X_ic;

		const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);
		Xaddrs[buf][Xs_y][Xs_x] = (write0 * xoffset) - !write0;
		dXs[buf][Xs_y][Xs_x] = 0;
		__syncthreads();

		//compute area-----------------------------------------
		j0 = j0 * OC + oc0; j1 = j0 + OC;
		float4 dy00 = *(float4*)(&deltaY[j0]), dy01 = *(float4*)(&deltaY[j0 + 4]);
		float4 dy10 = *(float4*)(&deltaY[j1]), dy11 = *(float4*)(&deltaY[j1 + 4]);
		for (int ok = 1, OK = GK << 1 >> LB; ok < OK; ok++)
		{
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 w0 = Ws[buf][ik][(ty << 1)];
				float4 w1 = Ws[buf][ik][(ty << 1) + 1];
				int2 xaddr = *(int2*)(&Xaddrs[buf][ik][tx << 1]);

				float dx0 = CrossAdd_SUM4(w0, dy00) + CrossAdd_SUM4(w1, dy01);
				float dx1 = CrossAdd_SUM4(w0, dy10) + CrossAdd_SUM4(w1, dy11);
				dx0 *= (xaddr.x != -1);
				dx1 *= (xaddr.y != -1);

				atomicAdd_block(&(dXs[buf][ik][tx << 1]), dx0);
				atomicAdd_block(&(dXs[buf][ik][(tx << 1) + 1]), dx1);
			}
			__syncthreads();
			atomicAdd(deltaX + Xaddrs[buf][Xs_y][Xs_x], dXs[buf][Xs_y][Xs_x]);
			buf ^= 1;

			//load 4 elements from W[OC, FH, FW, IC]
			int W_k = ((ok << LB) + tx) >> 1;
			Ws[buf][Ws_x][Ws_y].x = W[toc0 + W_k];
			Ws[buf][Ws_x][Ws_y].y = W[toc1 + W_k];
			Ws[buf][Ws_x][Ws_y].z = W[toc2 + W_k];
			Ws[buf][Ws_x][Ws_y].w = W[toc3 + W_k];

			//compute 1 address for Xaddr[M, IH, IW, IC]
			int dX_k = ((ok << LB) + ty) >> 1;
			int X_fh = dX_k / FW_IC; dX_k -= X_fh * FW_IC;//with the same tx
			int X_fw = dX_k / IC, X_ic = dX_k - X_fw * IC;
			int ih0 = tih0 + X_fh, iw0 = tiw0 + X_fw;
			bool write0 = (tih0 >= -X_fh) && (tih0 < IH - X_fh) && (tiw0 >= -X_fw) && (tiw0 < IW - X_fw);
			int xoffset = Xoffset0 + (X_fh*IW + X_fw)*IC + X_ic;
			Xaddrs[buf][Xs_y][Xs_x] = (write0 * xoffset) - !write0;
			dXs[buf][Xs_y][Xs_x] = 0;
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 w0 = Ws[buf][ik][(ty << 1)];
			float4 w1 = Ws[buf][ik][(ty << 1) + 1];
			int2 xaddr = *(int2*)(&Xaddrs[buf][ik][tx << 1]);

			float dx0 = CrossAdd_SUM4(w0, dy00) + CrossAdd_SUM4(w1, dy01);
			float dx1 = CrossAdd_SUM4(w0, dy10) + CrossAdd_SUM4(w1, dy11);
			dx0 *= (xaddr.x != -1);
			dx1 *= (xaddr.y != -1);

			atomicAdd_block(&(dXs[buf][ik][tx << 1]), dx0);
			atomicAdd_block(&(dXs[buf][ik][(tx << 1) + 1]), dx1);
		}
		__syncthreads();
		atomicAdd(deltaX + Xaddrs[buf][Xs_y][Xs_x], dXs[buf][Xs_y][Xs_x]);
	}

#endif