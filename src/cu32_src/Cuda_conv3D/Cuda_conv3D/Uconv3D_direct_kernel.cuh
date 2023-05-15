

#define conv3dGemm_k88_f3(stream, LB, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_kernel_8_8_f3<LB, (1<<LB>>1), FH, FW>\
		<<< dim3(GM>>LB>>3, GN>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, W, Y, OH, OW, IC, OC, sh, sw, ph, pw, oc_index, j_index)

#ifndef CONV_3D_GEMM_KERNEL_8_8_F3
#define CONV_3D_GEMM_KERNEL_8_8_F3

//LB = 4: Size = 1, Time = 1.694 msec, Performace = 1267.7 GFlop/s
//LB = 3: Size = 1, Time = 1.998 msec, Performace = 1074.82 GFlop/s
template<int LB, int STEP, int FH, int FW>
__global__ void conv3dGemm_kernel_8_8_f3(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W,
	float* __restrict__ Y, int OH, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GK = FH * FW * IC
	const int FW_IC = FW * IC, GK = FH * FW_IC;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.y << LB) + ty) << 3) + oc_index;
	int toc0 = (oc0 + ((tx >= STEP) << 2)) * GK;
	int toc1 = toc0 + GK, toc2 = toc1 + GK, toc3 = toc2 + GK;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.x << LB) + tx) << 3) + j_index;
	int tj0 = j0 + ((ty >= STEP) << 2);
	int tj1 = tj0 + 1, tj2 = tj0 + 2, tj3 = tj0 + 3;
	int OH_OW = OH * OW;
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	get_n_oh_ow(tj1, tn1, toh1, tow1);
	get_n_oh_ow(tj2, tn2, toh2, tow2);
	get_n_oh_ow(tj3, tn3, toh3, tow3);
	toh0 = toh0 * sh - ph, tow0 = tow0 * sw - pw;
	toh1 = toh1 * sh - ph, tow1 = tow1 * sw - pw;
	toh2 = toh2 * sh - ph, tow2 = tow2 * sw - pw;
	toh3 = toh3 * sh - ph, tow3 = tow3 * sw - pw;
	const int Xoffset0 = ((tn0*IH + toh0)*IW + tow0)*IC;
	const int Xoffset1 = ((tn1*IH + toh1)*IW + tow1)*IC;
	const int Xoffset2 = ((tn2*IH + toh2)*IW + tow2)*IC;
	const int Xoffset3 = ((tn3*IH + toh3)*IW + tow3)*IC;

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = tx - ((tx >= STEP) << LB >> 1);//k = ((fh*FW) + fw)*IC + ic
	Ws[buf][tx][ty].x = W[toc0 + W_k];
	Ws[buf][tx][ty].y = W[toc1 + W_k];
	Ws[buf][tx][ty].z = W[toc2 + W_k];
	Ws[buf][tx][ty].w = W[toc3 + W_k];

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = ty - ((ty >= STEP) << LB >> 1), X_fh, X_fw;
	get_X_fh_fw(X_k, X_fh, X_fw);
	bool lx0 = (toh0 >= -X_fh) && (toh0 < IH - X_fh) && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
	bool lx1 = (toh1 >= -X_fh) && (toh1 < IH - X_fh) && (tow1 >= -X_fw) && (tow1 < IW - X_fw);
	bool lx2 = (toh2 >= -X_fh) && (toh2 < IH - X_fh) && (tow2 >= -X_fw) && (tow2 < IW - X_fw);
	bool lx3 = (toh3 >= -X_fh) && (toh3 < IH - X_fh) && (tow3 >= -X_fw) && (tow3 < IW - X_fw);
	int IW_IC = IW * IC, xoffset = X_fh * IW_IC + X_k;//X[n, ih, iw, ic]
	Xs[buf][ty][tx].x = (lx0 ? X[Xoffset0 + xoffset] : 0);
	Xs[buf][ty][tx].y = (lx1 ? X[Xoffset1 + xoffset] : 0);
	Xs[buf][ty][tx].z = (lx2 ? X[Xoffset2 + xoffset] : 0);
	Xs[buf][ty][tx].w = (lx3 ? X[Xoffset3 + xoffset] : 0);
	__syncthreads();

	//compute area------------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)//OK = GK / STEP
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];
			float4 a0 = Ws[buf][ik][ty], a1 = Ws[buf][ik + STEP][ty];

			simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
			simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
			simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
			simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
			simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);
			simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
			simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
			simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
		}
		buf ^= 1;

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		Ws[buf][tx][ty].x = W[toc0 + W_k];
		Ws[buf][tx][ty].y = W[toc1 + W_k];
		Ws[buf][tx][ty].z = W[toc2 + W_k];
		Ws[buf][tx][ty].w = W[toc3 + W_k];

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		get_X_fh_fw(X_k, X_fh, X_fw);
		bool lx0 = (toh0 >= -X_fh) && (toh0 < IH - X_fh) && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
		bool lx1 = (toh1 >= -X_fh) && (toh1 < IH - X_fh) && (tow1 >= -X_fw) && (tow1 < IW - X_fw);
		bool lx2 = (toh2 >= -X_fh) && (toh2 < IH - X_fh) && (tow2 >= -X_fw) && (tow2 < IW - X_fw);
		bool lx3 = (toh3 >= -X_fh) && (toh3 < IH - X_fh) && (tow3 >= -X_fw) && (tow3 < IW - X_fw);
		int xoffset = X_fh * IW_IC + X_k;
		Xs[buf][ty][tx].x = (lx0 ? X[Xoffset0 + xoffset] : 0);
		Xs[buf][ty][tx].y = (lx1 ? X[Xoffset1 + xoffset] : 0);
		Xs[buf][ty][tx].z = (lx2 ? X[Xoffset2 + xoffset] : 0);
		Xs[buf][ty][tx].w = (lx3 ? X[Xoffset3 + xoffset] : 0);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];
		float4 a0 = Ws[buf][ik][ty], a1 = Ws[buf][ik + STEP][ty];

		simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
		simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
		simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
		simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
		simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);
		simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
		simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
		simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
	}

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;

	*(float4*)(Y + j0) = v0;  *(float4*)(Y + j0 + 4) = v1;
	*(float4*)(Y + j1) = v2;  *(float4*)(Y + j1 + 4) = v3;
	*(float4*)(Y + j2) = v4;  *(float4*)(Y + j2 + 4) = v5;
	*(float4*)(Y + j3) = v6;  *(float4*)(Y + j3 + 4) = v7;
	*(float4*)(Y + j4) = v8;  *(float4*)(Y + j4 + 4) = v9;
	*(float4*)(Y + j5) = v10; *(float4*)(Y + j5 + 4) = v11;
	*(float4*)(Y + j6) = v12; *(float4*)(Y + j6 + 4) = v13;
	*(float4*)(Y + j7) = v14; *(float4*)(Y + j7 + 4) = v15;
}

#endif


#define conv3d_directv1(stream, LB, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw)\
	conv3d_direct_kernelv1<LB>\
		<<< dim3((OH*OW)>>LB, (N * OC)>>LB), dim3(1<<LB, 1<<LB), 0, stream>>>\
			(X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw)


#define conv3d_directv2(stream, LB, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw)\
	conv3d_direct_kernelv2<LB>\
		<<< dim3((OH*OW)>>LB, (N * OC)>>LB), dim3(1<<LB, 1<<LB), 0, stream>>>\
			(X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw)


#define conv3d_directv3(stream, LB, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw)\
	conv3d_direct_kernelv3<LB>\
		<<< dim3((N*OH*OW)>>LB, (OC)>>LB), dim3(1<<LB, 1<<LB), 0, stream>>>\
			(X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw)



template<int LB>
__global__ void conv3d_direct_kernelv1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W, int FH, int FW,
		  float* __restrict__ Y, int OH, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw)
{
	int by = blockIdx.y, bx = blockIdx.x;
	int ty = threadIdx.y, tx = threadIdx.x;

	int yindex = (by << LB) + ty;//N, OC
	int xindex = (bx << LB) + tx;//OH, OW

	int n = yindex / OC, oc = yindex % OC;//Y[N, OC]
	int oh = xindex / OW, ow = xindex % OW;//X[OH, OW]
	
	float v = 0;
	int ihs = oh * sh - ph, iws = ow * sw - pw;

	for (int fh = 0; fh < FH; fh++)
	for (int fw = 0; fw < FW; fw++)
	for (int ic = 0; ic < IC; ic++)
	{
		int ih = ihs + fh, iw = iws + fw;
		if (ih < 0 || iw < 0 || ih >= IH || iw >= IW) continue;

		float x = get4d(X, n, ih, iw, ic, IH, IW, IC);
		float w = get4d(W, oc, fh, fw, ic, FH, FW, IC);
		v += x * w;
	}
	get4d(Y, n, oh, ow, oc, OH, OW, OC) = v;
}


template<int LB>
__global__ void conv3d_direct_kernelv2(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ Y, int OH, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw)
{

	int by = blockIdx.y, bx = blockIdx.x;
	int ty = threadIdx.y, tx = threadIdx.x;

	int yindex = (by << LB) + ty;//N, OC
	int xindex = (bx << LB) + tx;//OH, OW

	int n = yindex / OC, oc = yindex - n * OC;//Y[N, OC]
	int oh = xindex / OW, ow = xindex - oh * OW;//X[OH, OW]

	float v = 0;
	int ihs = oh * sh - ph, iws = ow * sw - pw;

	for (int fh = 0; fh < FH; fh++)
	for (int fw = 0; fw < FW; fw++)
	for (int ic = 0; ic < IC; ic += 4)
	{
		int ih = ihs + fh, iw = iws + fw;
		if (ih < 0 || iw < 0 || ih >= IH || iw >= IW) continue;

		float4 x = *(float4*)(&get4d(X, n, ih, iw, ic, IH, IW, IC));
		float4 w = *(float4*)(&get4d(W, oc, fh, fw, ic, FH, FW, IC));
		v += x.x*w.x + x.y*w.y + x.z*w.z + x.w*w.w;
	}
	get4d(Y, n, oh, ow, oc, OH, OW, OC) = v;
}

template<int LB>
__global__ void conv3d_direct_kernelv3(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ Y, int OH, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw)
{

	int by = blockIdx.y, bx = blockIdx.x;
	int ty = threadIdx.y, tx = threadIdx.x;

	int yindex = (by << LB) + ty;//N, OC
	int xindex = (bx << LB) + tx;//OH, OW

	int oc = yindex;

	int OH_OW = OH * OW;
	int n = xindex / OH_OW, nres = xindex % OH_OW;//Y[OC]
	int oh = nres / OW, ow = nres % OW;//X[N, OH, OW]

	float v = 0;
	int ihs = oh * sh - ph, iws = ow * sw - pw;
	//same tx: (n, oh, ow) -> the same (n, ih, iw)
	for (int fh = 0; fh < FH; fh++)
	for (int fw = 0; fw < FW; fw++)
	for (int ic = 0; ic < IC; ic += 4)
	{
		int ih = ihs + fh, iw = iws + fw;
		if (ih < 0 || iw < 0 || ih >= IH || iw >= IW) continue;

		float4 x = *(float4*)(&get4d(X, n, ih, iw, ic, IH, IW, IC));
		float4 w = *(float4*)(&get4d(W, oc, fh, fw, ic, FH, FW, IC));
		v += x.x*w.x + x.y*w.y + x.z*w.z + x.w*w.w;
	}
	get4d(Y, n, oh, ow, oc, OH, OW, OC) = v;
}



#define conv3d_directv4(stream, LB, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw)\
	conv3d_direct_kernelv4<LB>\
		<<< dim3((N*OH*OW)>>LB, (OC)>>LB), dim3(1<<LB, 1<<LB), 0, stream>>>\
			(X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw)

//Size = 1, Time = 12.05 msec, Performace = 178.214 GFlop/s
template<int LB>
__global__ void conv3d_direct_kernelv4(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ Y, int OH, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	__shared__ float Xs[1 << LB][1 << LB];
	__shared__ float Ws[1 << LB][1 << LB];

	int oc = (blockIdx.y << LB) + ty;

	int j = (blockIdx.x << LB) + tx;
	int OH_OW = OH * OW;
	get_n_oh_ow(j, n, oh, ow);

	float v = 0;
	const int OIC = IC >> LB, STEP = 1 << LB;
	int ihs = oh * sh - ph, iws = ow * sw - pw;
	for (int fh = 0; fh < FH; fh++) 
	for (int fw = 0; fw < FW; fw++)
	{ 
		int ih = ihs + fh, iw = iws + fw;
		for (int oic = 0; oic < OIC; oic++)
		{
			int Wic = (oic << LB) + tx;//oc: with the same ty
			Ws[tx][ty] = get4d(W, oc, fh, fw, Wic, FH, FW, IC);
			             
			int Xic = (oic << LB) + ty;//n, ih, iw: with the same tx
			if (ih < 0 || iw < 0 || ih >= IH || iw >= IW) Xs[ty][tx] = 0;
			else Xs[ty][tx] = get4d(X, n, ih, iw, Xic, IH, IW, IC);
			__syncthreads();

#pragma unroll
			for (int ik = 0; ik < STEP; ik++) {
				int ic = (oic << LB) + ik;
				float w = Ws[ik][ty];
				float x = Xs[ik][tx];
				v += w * x;
			}
			__syncthreads();
		}
	}
	get4d(Y, n, oh, ow, oc, OH, OW, OC) = v;
}



#define conv3d_directv5(stream, LB, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw)\
	conv3d_direct_kernelv5<LB>\
		<<< dim3((N*OH*OW)>>LB, (OC)>>LB), dim3(1<<LB, 1<<LB), 0, stream>>>\
			(X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw)

//Size = 1, Time = 10 msec, Performace = 214.748 GFlop/s
//Size = 1, Time = 12.05 msec, Performace = 178.214 GFlop/s
template<int LB>
__global__ void conv3d_direct_kernelv5(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ Y, int OH, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	__shared__ float4 Xs[1 << LB][1 << LB];
	__shared__ float4 Ws[1 << LB][1 << LB];

	int oc = (blockIdx.y << LB) + ty;

	int j = (blockIdx.x << LB) + tx;
	int OH_OW = OH * OW;
	get_n_oh_ow(j, n, oh, ow);

	float v = 0;
	const int OIC = IC >> LB >> 2, STEP = 1 << LB;
	int ihs = oh * sh - ph, iws = ow * sw - pw;
	for (int fh = 0; fh < FH; fh++) 
	{
		int ih = ihs + fh;
		for (int fw = 0; fw < FW; fw++)
		{
			int iw = iws + fw;
			for (int oic = 0; oic < OIC; oic++)
			{
				int Wic = ((oic << LB) + tx) << 2;//oc: with the same ty
				Ws[tx][ty] = *(float4*)(&get4d(W, oc, fh, fw, Wic, FH, FW, IC));

				int Xic = ((oic << LB) + ty) << 2;//n, ih, iw: with the same tx
				if (ih < 0 || iw < 0 || ih >= IH || iw >= IW) Xs[ty][tx] = make_float4(0, 0, 0, 0);
				else Xs[ty][tx] = *(float4*)(&get4d(X, n, ih, iw, Xic, IH, IW, IC));
				__syncthreads();

#pragma unroll
				for (int ik = 0; ik < STEP; ik++) {
					float4 w = Ws[ik][ty];
					float4 x = Xs[ik][tx];
					v += w.x * x.x;
					v += w.y * x.y;
					v += w.z * x.z;
					v += w.w * x.w;
				}
				__syncthreads();
			}
		}
	}
	
	get4d(Y, n, oh, ow, oc, OH, OW, OC) = v;
}


#define conv3d_directv6(stream, LB, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw)\
	conv3d_direct_kernelv6<LB>\
		<<< dim3((N*OH*OW)>>LB, (OC)>>LB), dim3(1<<LB, 1<<LB), 0, stream>>>\
			(X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw)

//Size = 1, Time = 10 msec, Performace = 214.748 GFlop/s
//Size = 1, Time = 12.05 msec, Performace = 178.214 GFlop/s
template<int LB>
__global__ void conv3d_direct_kernelv6(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ Y, int OH, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	__shared__ float2 Xs[1 << LB][1 << LB];
	__shared__ float2 Ws[1 << LB][1 << LB];

	int oc = (blockIdx.y << LB) + ty;

	int j = (blockIdx.x << LB) + tx;
	int OH_OW = OH * OW;
	get_n_oh_ow(j, n, oh, ow);

	float v = 0;
	const int OIC = IC >> LB >> 1, STEP = 1 << LB;
	const int ihs = oh * sh - ph, iws = ow * sw - pw;
	for (int fh = 0; fh < FH; fh++)
	for (int fw = 0, ih = ihs + fh; fw < FW; fw++)
	{
		for (int oic = 0, iw = iws + fw; oic < OIC; oic++)
		{
			int Wic = ((oic << LB) + tx) << 1;//oc: with the same ty
			Ws[tx][ty] = *(float2*)(&get4d(W, oc, fh, fw, Wic, FH, FW, IC));

			int Xic = ((oic << LB) + ty) << 1;//n, ih, iw: with the same tx
			if (ih < 0 || iw < 0 || ih >= IH || iw >= IW) Xs[ty][tx] = make_float2(0, 0);
			else Xs[ty][tx] = *(float2*)(&get4d(X, n, ih, iw, Xic, IH, IW, IC));
			__syncthreads();

#pragma unroll
			for (int ik = 0; ik < STEP; ik++) {
				float2 w = Ws[ik][ty];
				float2 x = Xs[ik][tx];
				v += w.x * x.x + w.y * x.y;
			}
			__syncthreads();
		}
	}

	get4d(Y, n, oh, ow, oc, OH, OW, OC) = v;
}


#define conv3d_directv7(stream, LB, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw)\
	conv3d_direct_kernelv7<LB>\
		<<< dim3((N*OH*OW)>>LB, (OC)>>LB), dim3(1<<LB, 1<<LB), 0, stream>>>\
			(X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw)


//Size = 1, Time = 10.918 msec, Performace = 196.692 GFlop/s
template<int LB>
__global__ void conv3d_direct_kernelv7(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W, int FH, int FW,
	      float* __restrict__ Y, int OH, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 Xs[2][1 << LB][1 << LB];
	__shared__ float2 Ws[2][1 << LB][1 << LB];

	int oc = (blockIdx.y << LB) + ty;

	int j = (blockIdx.x << LB) + tx;
	int OH_OW = OH * OW; get_n_oh_ow(j, n, oh, ow);

	float v = 0;
	const int OIC = IC >> LB >> 1, STEP = 1 << LB;
	const int ihs = oh * sh - ph, iws = ow * sw - pw;
	for (int fh = 0; fh < FH; fh++)
	for (int fw = 0; fw < FW; fw++)
	{
		int ih = ihs + fh;
		int iw = iws + fw;

		int Wic = tx << 1;
		Ws[buf][tx][ty] = *(float2*)(&get4d(W, oc, fh, fw, Wic, FH, FW, IC));

		int Xic = ty << 1;
		if (ih < 0 || iw < 0 || ih >= IH || iw >= IW) Xs[buf][ty][tx] = make_float2(0, 0);
		else Xs[buf][ty][tx] = *(float2*)(&get4d(X, n, ih, iw, Xic, IH, IW, IC));
		__syncthreads();

		for (int oic = 1; oic < OIC; oic++)
		{
#pragma unroll
			for (int ik = 0; ik < STEP; ik++) {
				float2 w = Ws[buf][ik][ty];
				float2 x = Xs[buf][ik][tx];
				v += w.x * x.x + w.y * x.y;
			}

			buf ^= 1;
			int Wic = ((oic << LB) + tx) << 1;//oc: with the same ty
			Ws[buf][tx][ty] = *(float2*)(&get4d(W, oc, fh, fw, Wic, FH, FW, IC));

			int Xic = ((oic << LB) + ty) << 1;//n, ih, iw: with the same tx
			if (ih < 0 || iw < 0 || ih >= IH || iw >= IW) Xs[buf][ty][tx] = make_float2(0, 0);
			else Xs[buf][ty][tx] = *(float2*)(&get4d(X, n, ih, iw, Xic, IH, IW, IC));
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float2 w = Ws[buf][ik][ty];
			float2 x = Xs[buf][ik][tx];
			v += w.x * x.x + w.y * x.y;
		}
		__syncthreads();
	}

	get4d(Y, n, oh, ow, oc, OH, OW, OC) = v;
}



#define conv3d_directv8(stream, LB, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw)\
	conv3d_direct_kernelv8<LB, 1<<LB>\
		<<< dim3((N*OH*OW)>>LB>>1, (OC)>>LB>>1), dim3(1<<LB, 1<<LB), 0, stream>>>\
			(X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw)

//Size = 1, Time = 4.726 msec, Performace = 454.398 GFlop/s
template<int LB, int STEP>
__global__ void conv3d_direct_kernelv8(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ Y, int OH, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 Xs[2][1 << LB][2 << LB];
	__shared__ float2 Ws[2][1 << LB][2 << LB];

	int oc0 = ((blockIdx.y << LB) + ty) << 1, oc1 = oc0 + 1;
	int j0 = ((blockIdx.x << LB) + tx) << 1, j1 = j0 + 1;
	int OH_OW = OH * OW;
	get_n_oh_ow(j0, n0, oh0, ow0);
	get_n_oh_ow(j1, n1, oh1, ow1);

	float2 v0 = make_float2(0, 0);
	float2 v1 = make_float2(0, 0);
	const int OIC = IC >> LB >> 1, STEP = 1 << LB;
	const int ihs0 = oh0 * sh - ph, iws0 = ow0 * sw - pw;
	const int ihs1 = oh1 * sh - ph, iws1 = ow1 * sw - pw;
	for (int fh = 0; fh < FH; fh++)
	{
		int ih0 = ihs0 + fh, ih1 = ihs1 + fh;
		for (int fw = 0; fw < FW; fw++)
		{
			int iw0 = iws0 + fw, iw1 = iws1 + fw;
			bool unload0 = ih0 < 0 || iw0 < 0 || ih0 >= IH || iw0 >= IW;
			bool unload1 = ih1 < 0 || iw1 < 0 || ih1 >= IH || iw1 >= IW;

			int Wic = tx << 1;
			Ws[buf][tx][(ty << 1)    ] = *(float2*)(&get4d(W, oc0, fh, fw, Wic, FH, FW, IC));
			Ws[buf][tx][(ty << 1) + 1] = *(float2*)(&get4d(W, oc1, fh, fw, Wic, FH, FW, IC));

			int Xic = ty << 1;
			Xs[buf][ty][(tx << 1)    ] = unload0 ? make_float2(0, 0) : *(float2*)(&get4d(X, n0, ih0, iw0, Xic, IH, IW, IC));
			Xs[buf][ty][(tx << 1) + 1] = unload1 ? make_float2(0, 0) : *(float2*)(&get4d(X, n1, ih1, iw1, Xic, IH, IW, IC));
			__syncthreads();

			for (int oic = 1; oic < OIC; oic++)
			{
#pragma unroll
				for (int ik = 0; ik < STEP; ik++) {
					float4 w = *(float4*)(&Ws[buf][ik][ty << 1]);
					float4 x = *(float4*)(&Xs[buf][ik][tx << 1]);
					v0.x += w.x * x.x + w.y * x.y; v0.y += w.z * x.x + w.w * x.y;
					v1.x += w.x * x.z + w.y * x.w; v1.y += w.z * x.z + w.w * x.w;
				}

				buf ^= 1;
				int Wic = ((oic << LB) + tx) << 1;
				Ws[buf][tx][(ty << 1)    ] = *(float2*)(&get4d(W, oc0, fh, fw, Wic, FH, FW, IC));
		     	Ws[buf][tx][(ty << 1) + 1] = *(float2*)(&get4d(W, oc1, fh, fw, Wic, FH, FW, IC));

				int Xic = ((oic << LB) + ty) << 1;
				if (unload0) Xs[buf][ty][(tx << 1)    ] = make_float2(0, 0);
				Xs[buf][ty][(tx << 1)    ] = unload0 ? make_float2(0, 0) : *(float2*)(&get4d(X, n0, ih0, iw0, Xic, IH, IW, IC));
				Xs[buf][ty][(tx << 1) + 1] = unload1 ? make_float2(0, 0) : *(float2*)(&get4d(X, n1, ih1, iw1, Xic, IH, IW, IC));
				__syncthreads();
			}
#pragma unroll
			for (int ik = 0; ik < STEP; ik++) {
				float4 w = *(float4*)(&Ws[buf][ik][ty << 1]);
				float4 x = *(float4*)(&Xs[buf][ik][tx << 1]);
				v0.x += w.x * x.x + w.y * x.y; v0.y += w.z * x.x + w.w * x.y;
				v1.x += w.x * x.z + w.y * x.w; v1.y += w.z * x.z + w.w * x.w;
			}
			__syncthreads();
		}
	}

	j0 *= OC; j1 = j0 + OC;
	*(float2*)(&Y[j0 + oc0]) = v0;
	*(float2*)(&Y[j1 + oc0]) = v1;
}


#define conv3d_directv9(stream, LB, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw)\
	conv3d_direct_kernelv9<LB, (1<<LB>>1)>\
		<<< dim3((N*OH*OW)>>LB>>2, (OC)>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream>>>\
			(X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw)


//Size = 1, Time = 2.486 msec, Performace = 863.831 GFlop/s
template<int LB, int STEP>
__global__ void conv3d_direct_kernelv9(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ Y, int OH, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 Xs[2][1 << LB << LB];//[1<<LB][2<<LB]
	__shared__ float2 Ws[2][1 << LB << LB];//[1<<LB][2<<LB]

	const int oc0 = ((blockIdx.y << LB) + ty) << 2;
	const int GK = FH * FW * IC;
	const int toc0 = (((tx & 1) << 1) + oc0) * GK, toc1 = toc0 + GK;

	int j0 = ((blockIdx.x << LB) + tx) << 2;
	int j1 = j0 + 1, j2 = j0 + 2, j3 = j0 + 3;
	const int OH_OW = OH * OW;
	get_n_oh_ow(j0, n0, oh0, ow0);
	get_n_oh_ow(j1, n1, oh1, ow1);
	get_n_oh_ow(j2, n2, oh2, ow2);
	get_n_oh_ow(j3, n3, oh3, ow3);
	bool flagY = (ty & 1);
	const int tn0 = (n2 - n0)*flagY + n0;
	const int tn1 = (n3 - n1)*flagY + n1;
	const int tihs0 = ((oh2 - oh0)*flagY + oh0) * sh - ph;
	const int tihs1 = ((oh3 - oh1)*flagY + oh1) * sh - ph;
	const int tiws0 = ((ow2 - ow0)*flagY + ow0) * sw - pw;
	const int tiws1 = ((ow3 - ow1)*flagY + ow1) * sw - pw;
	const int Xoffset0 = (((tn0 *IH) + tihs0) * IW + tiws0) * IC;
	const int Xoffset1 = (((tn1 *IH) + tihs1) * IW + tiws1) * IC;

	//compute area----------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0);
	float4 v1 = make_float4(0, 0, 0, 0);
	float4 v2 = make_float4(0, 0, 0, 0);
	float4 v3 = make_float4(0, 0, 0, 0);

	const int Ws_xy = ((tx >> 1) << 1 << LB) + (ty << 1) + (tx & 1);
	const int Xs_yx = ((ty >> 1) << 1 << LB) + (tx << 1) + (ty & 1);
	for (int fh = 0; fh < FH; fh++, X += (IW - FW) *IC)
	for (int fw = 0; fw < FW; fw++, X += IC, W += IC)
		{
			int Wic = tx >> 1;
			Ws[buf][Ws_xy].x = W[toc0 + Wic];
			Ws[buf][Ws_xy].y = W[toc1 + Wic];

			int Xic = ty >> 1;
			bool load0 = (tihs0 >= -fh) && (tihs0 < IH - fh) && (tiws0 >= -fw) && (tiws0 < IW - fw);
			bool load1 = (tihs1 >= -fh) && (tihs1 < IH - fh) && (tiws1 >= -fw) && (tiws1 < IW - fw);
			Xs[buf][Xs_yx].x = load0 ? X[Xoffset0 + Xic] : 0;
			Xs[buf][Xs_yx].y = load1 ? X[Xoffset1 + Xic] : 0;
			__syncthreads();

			for (int oic = 1, OIC = IC << 1 >> LB; oic < OIC; oic++) {
#pragma unroll
				for (int ik = 0; ik < STEP; ik++)
				{
					float4 b = *(float4*)(&Xs[buf][((ik << LB) + tx) << 1]);
					float4 a = *(float4*)(&Ws[buf][((ik << LB) + ty) << 1]);
					simdMM4(v0, b.x, a);
					simdMM4(v1, b.y, a);
					simdMM4(v2, b.z, a);
					simdMM4(v3, b.w, a);
				}
				buf ^= 1;

				Wic = ((oic << LB) + tx) >> 1;
				Ws[buf][Ws_xy].x = W[toc0 + Wic];
				Ws[buf][Ws_xy].y = W[toc1 + Wic];

				Xic = ((oic << LB) + ty) >> 1;
				Xs[buf][Xs_yx].x = load0 ? X[Xoffset0 + Xic] : 0;
				Xs[buf][Xs_yx].y = load1 ? X[Xoffset1 + Xic] : 0;
				__syncthreads();
			}
#pragma unroll
			for (int ik = 0; ik < STEP; ik++) {
				float4 b = *(float4*)(&Xs[buf][((ik << LB) + tx) << 1]);
				float4 a = *(float4*)(&Ws[buf][((ik << LB) + ty) << 1]);

				simdMM4(v0, b.x, a);
				simdMM4(v1, b.y, a);
				simdMM4(v2, b.z, a);
				simdMM4(v3, b.w, a);
			}
			buf ^= 1;
		}

	j0 *= OC; j1 = j0 + OC; j2 = j1 + OC; j3 = j2 + OC;
	*(float4*)(&Y[j0 + oc0]) = v0;
	*(float4*)(&Y[j1 + oc0]) = v1;
	*(float4*)(&Y[j2 + oc0]) = v2;
	*(float4*)(&Y[j3 + oc0]) = v3;
}


#define conv3d_directv10(stream, LB, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw)\
	conv3d_direct_kernelv10<LB, (1<<LB>>1)>\
		<<< dim3((N*OH*OW)>>LB>>3, (OC)>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream>>>\
			(X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw)

//Size = 1, Time = 2.146 msec, Performace = 1000.69 GFlop/s
template<int LB, int STEP>
__global__ void conv3d_direct_kernelv10(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ Y, int OH, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 Xs[2][1 << LB << LB];//[1<<LB][2<<LB]
	__shared__ float4 Ws[2][1 << LB << LB];//[1<<LB][2<<LB]

	const int oc0 = ((blockIdx.y << LB) + ty) << 3;
	const int GK = FH * FW * IC;
	const int toc0 = (((tx & 1) << 2) + oc0) * GK;
	const int toc1 = toc0 + GK, toc2 = toc1 + GK, toc3 = toc2 + GK;

	int j0 = ((blockIdx.x << LB) + tx) << 3;
	int j1 = j0 + 1, j2 = j0 + 2, j3 = j0 + 3;
	int j4 = j0 + 4, j5 = j0 + 5, j6 = j0 + 6, j7 = j0 + 7;
	const int OH_OW = OH * OW;
	get_n_oh_ow(j0, n0, oh0, ow0);
	get_n_oh_ow(j1, n1, oh1, ow1);
	get_n_oh_ow(j2, n2, oh2, ow2);
	get_n_oh_ow(j3, n3, oh3, ow3);
	get_n_oh_ow(j4, n4, oh4, ow4);
	get_n_oh_ow(j5, n5, oh5, ow5);
	get_n_oh_ow(j6, n6, oh6, ow6);
	get_n_oh_ow(j7, n7, oh7, ow7);
	bool flagY = (ty & 1);
	const int tn0 = (n4 - n0)*flagY + n0;
	const int tn1 = (n5 - n1)*flagY + n1;
	const int tn2 = (n6 - n2)*flagY + n2;
	const int tn3 = (n7 - n3)*flagY + n3;
	const int tihs0 = ((oh4 - oh0)*flagY + oh0) * sh - ph;
	const int tihs1 = ((oh5 - oh1)*flagY + oh1) * sh - ph;
	const int tihs2 = ((oh6 - oh2)*flagY + oh2) * sh - ph;
	const int tihs3 = ((oh7 - oh3)*flagY + oh3) * sh - ph;
	const int tiws0 = ((ow4 - ow0)*flagY + ow0) * sw - pw;
	const int tiws1 = ((ow5 - ow1)*flagY + ow1) * sw - pw;
	const int tiws2 = ((ow6 - ow2)*flagY + ow2) * sw - pw;
	const int tiws3 = ((ow7 - ow3)*flagY + ow3) * sw - pw;
	int Xoffset0 = (((tn0 *IH) + tihs0) * IW + tiws0) * IC;
	int Xoffset1 = (((tn1 *IH) + tihs1) * IW + tiws1) * IC;
	int Xoffset2 = (((tn2 *IH) + tihs2) * IW + tiws2) * IC;
	int Xoffset3 = (((tn3 *IH) + tihs3) * IW + tiws3) * IC;

	//compute area----------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4 v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4 v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4 v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4 v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	const int Ws_xy = ((tx >> 1) << 1 << LB) + (ty << 1) + (tx & 1);
	const int Xs_yx = ((ty >> 1) << 1 << LB) + (tx << 1) + (ty & 1);
	const int OIC = IC << 1 >> LB;
	for (int fh = 0; fh < FH; fh++, X += (IW - FW) *IC)
	{
		for (int fw = 0; fw < FW; fw++, X += IC, W += IC)
		{
			int Wic = tx >> 1;
			Ws[buf][Ws_xy].x = W[toc0 + Wic];
			Ws[buf][Ws_xy].y = W[toc1 + Wic];
			Ws[buf][Ws_xy].z = W[toc2 + Wic];
			Ws[buf][Ws_xy].w = W[toc3 + Wic];

			int Xic = ty >> 1;
			bool unload0 = (tihs0 < -fh || tihs0 >= IH - fh || tiws0 < -fw || tiws0 >= IW - fw);
			bool unload1 = (tihs1 < -fh || tihs1 >= IH - fh || tiws1 < -fw || tiws1 >= IW - fw);
			bool unload2 = (tihs2 < -fh || tihs2 >= IH - fh || tiws2 < -fw || tiws2 >= IW - fw);
			bool unload3 = (tihs3 < -fh || tihs3 >= IH - fh || tiws3 < -fw || tiws3 >= IW - fw);
			Xs[buf][Xs_yx].x = unload0 ? 0 : X[Xoffset0 + Xic];
			Xs[buf][Xs_yx].y = unload1 ? 0 : X[Xoffset1 + Xic];
			Xs[buf][Xs_yx].z = unload2 ? 0 : X[Xoffset2 + Xic];
			Xs[buf][Xs_yx].w = unload3 ? 0 : X[Xoffset3 + Xic];
			__syncthreads();

			for (int oic = 1; oic < OIC; oic++) {
#pragma unroll
				for (int ik = 0; ik < STEP; ik++)
				{
					float4 b0 = Xs[buf][(((ik << LB) + tx) << 1)	];
					float4 b1 = Xs[buf][(((ik << LB) + tx) << 1) + 1];
					float4 a0 = Ws[buf][(((ik << LB) + ty) << 1)];
					float4 a1 = Ws[buf][(((ik << LB) + ty) << 1) + 1];
					
					simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
					simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
					simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
					simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
					simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);
					simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
					simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
					simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
				}
				buf ^= 1;

				Wic = ((oic << LB) + tx) >> 1;
				Ws[buf][Ws_xy].x = W[toc0 + Wic];
				Ws[buf][Ws_xy].y = W[toc1 + Wic];
				Ws[buf][Ws_xy].z = W[toc2 + Wic];
				Ws[buf][Ws_xy].w = W[toc3 + Wic];

				Xic = ((oic << LB) + ty) >> 1;
				Xs[buf][Xs_yx].x = unload0 ? 0 : X[Xoffset0 + Xic];
				Xs[buf][Xs_yx].y = unload1 ? 0 : X[Xoffset1 + Xic];
				Xs[buf][Xs_yx].z = unload2 ? 0 : X[Xoffset2 + Xic];
				Xs[buf][Xs_yx].w = unload3 ? 0 : X[Xoffset3 + Xic];
				__syncthreads();
			}
#pragma unroll
			for (int ik = 0; ik < STEP; ik++) {
				float4 b0 = Xs[buf][(((ik << LB) + tx) << 1)];
				float4 b1 = Xs[buf][(((ik << LB) + tx) << 1) + 1];
				float4 a0 = Ws[buf][(((ik << LB) + ty) << 1)];
				float4 a1 = Ws[buf][(((ik << LB) + ty) << 1) + 1];

				simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
				simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
				simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
				simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
				simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);
				simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
				simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
				simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
			}
			buf ^= 1; 
		}
	}

	int oc4 = oc0 + 4; j0 *= OC;
	j1 = j0 + OC; j2 = j1 + OC; j3 = j2 + OC;
	j4 = j3 + OC; j5 = j4 + OC; j6 = j5 + OC; j7 = j6 + OC;

	*(float4*)(&Y[j0 + oc0]) = v0; *(float4*)(&Y[j0 + oc4]) = v1;
	*(float4*)(&Y[j1 + oc0]) = v2; *(float4*)(&Y[j1 + oc4]) = v3;
	*(float4*)(&Y[j2 + oc0]) = v4; *(float4*)(&Y[j2 + oc4]) = v5;
	*(float4*)(&Y[j3 + oc0]) = v6; *(float4*)(&Y[j3 + oc4]) = v7;
	*(float4*)(&Y[j4 + oc0]) = v8; *(float4*)(&Y[j4 + oc4]) = v9;
	*(float4*)(&Y[j5 + oc0]) = v10; *(float4*)(&Y[j5 + oc4]) = v11;
	*(float4*)(&Y[j6 + oc0]) = v12; *(float4*)(&Y[j6 + oc4]) = v13;
	*(float4*)(&Y[j7 + oc0]) = v14; *(float4*)(&Y[j7 + oc4]) = v15;
}


#define conv3d_directv11(stream, LB, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw)\
	conv3d_direct_kernelv11<LB, (1<<LB>>1)>\
		<<< dim3((N*OH*OW)>>LB>>2, (OC)>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream>>>\
			(X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw)

//Size = 1, Time = 2.146 msec, Performace = 1000.69 GFlop/s
template<int LB, int STEP>
__global__ void conv3d_direct_kernelv11(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ Y, int OH, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 Xs[2][1 << LB << LB];//[1<<LB][2<<LB]
	__shared__ float4 Ws[2][1 << LB << LB];//[1<<LB][2<<LB]

	const int oc0 = ((blockIdx.y << LB) + ty) << 3;
	const int GK = FH * FW * IC;
	const int toc0 = (((tx & 1) << 2) + oc0) * GK;
	const int toc1 = toc0 + GK, toc2 = toc1 + GK, toc3 = toc2 + GK;

	int j0 = ((blockIdx.x << LB) + tx) << 2;
	int j1 = j0 + 1, j2 = j0 + 2, j3 = j0 + 3;
	const int OH_OW = OH * OW;
	get_n_oh_ow(j0, n0, oh0, ow0);
	get_n_oh_ow(j1, n1, oh1, ow1);
	get_n_oh_ow(j2, n2, oh2, ow2);
	get_n_oh_ow(j3, n3, oh3, ow3);
	bool flagY = (ty & 1);
	const int tn0 = (n2 - n0)*flagY + n0;
	const int tn1 = (n3 - n1)*flagY + n1;
	const int tihs0 = ((oh2 - oh0)*flagY + oh0) * sh - ph;
	const int tihs1 = ((oh3 - oh1)*flagY + oh1) * sh - ph;
	const int tiws0 = ((ow2 - ow0)*flagY + ow0) * sw - pw;
	const int tiws1 = ((ow3 - ow1)*flagY + ow1) * sw - pw;
	int Xoffset0 = (((tn0 *IH) + tihs0) * IW + tiws0) * IC;
	int Xoffset1 = (((tn1 *IH) + tihs1) * IW + tiws1) * IC;

	//compute area----------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4 v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4 v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4 v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	
	const int Ws_xy = ((tx >> 1) << 1 << LB) + (ty << 1) + (tx & 1);
	const int Xs_yx = ((ty >> 1) << 1 << LB) + (tx << 1) + (ty & 1);
	const int OIC = IC << 1 >> LB;
	for (int fh = 0; fh < FH; fh++)
	{
		for (int fw = 0; fw < FW; fw++)
		{
			int Wic = tx >> 1;
			Ws[buf][Ws_xy].x = W[toc0 + Wic];
			Ws[buf][Ws_xy].y = W[toc1 + Wic];
			Ws[buf][Ws_xy].z = W[toc2 + Wic];
			Ws[buf][Ws_xy].w = W[toc3 + Wic];

			int Xic = ty >> 1;
			bool unload0 = (tihs0 < -fh || tihs0 >= IH - fh || tiws0 < -fw || tiws0 >= IW - fw);
			bool unload1 = (tihs1 < -fh || tihs1 >= IH - fh || tiws1 < -fw || tiws1 >= IW - fw);

			Xs[buf][Xs_yx].x = unload0 ? 0 : X[Xoffset0 + Xic];
			Xs[buf][Xs_yx].y = unload1 ? 0 : X[Xoffset1 + Xic];
			__syncthreads();

			for (int oic = 1; oic < OIC; oic++) {
#pragma unroll
				for (int ik = 0; ik < STEP; ik++)
				{
					float4 b0 = *(float4*)(&Xs[buf][(((ik << LB) + tx) << 1)]);
					float4 a0 = Ws[buf][(((ik << LB) + ty) << 1)];
					float4 a1 = Ws[buf][(((ik << LB) + ty) << 1) + 1];

					simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
					simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
					simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
					simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
				}
				buf ^= 1;

				Wic = ((oic << LB) + tx) >> 1;
				Ws[buf][Ws_xy].x = W[toc0 + Wic];
				Ws[buf][Ws_xy].y = W[toc1 + Wic];
				Ws[buf][Ws_xy].z = W[toc2 + Wic];
				Ws[buf][Ws_xy].w = W[toc3 + Wic];

				Xic = ((oic << LB) + ty) >> 1;
				Xs[buf][Xs_yx].x = unload0 ? 0 : X[Xoffset0 + Xic];
				Xs[buf][Xs_yx].y = unload1 ? 0 : X[Xoffset1 + Xic];
				__syncthreads();
			}
#pragma unroll
			for (int ik = 0; ik < STEP; ik++) {
				float4 b0 = *(float4*)(&Xs[buf][(((ik << LB) + tx) << 1)]);
				float4 a0 = Ws[buf][(((ik << LB) + ty) << 1)];
				float4 a1 = Ws[buf][(((ik << LB) + ty) << 1) + 1];

				simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
				simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
				simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
				simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
			}
			buf ^= 1; X += IC; W += IC;
		}
		X += (IW - FW) *IC;
	}

	int oc4 = oc0 + 4; j0 *= OC;
	j1 = j0 + OC; j2 = j1 + OC; j3 = j2 + OC;
	*(float4*)(&Y[j0 + oc0]) = v0; *(float4*)(&Y[j0 + oc4]) = v1;
	*(float4*)(&Y[j1 + oc0]) = v2; *(float4*)(&Y[j1 + oc4]) = v3;
	*(float4*)(&Y[j2 + oc0]) = v4; *(float4*)(&Y[j2 + oc4]) = v5;
	*(float4*)(&Y[j3 + oc0]) = v6; *(float4*)(&Y[j3 + oc4]) = v7;
}