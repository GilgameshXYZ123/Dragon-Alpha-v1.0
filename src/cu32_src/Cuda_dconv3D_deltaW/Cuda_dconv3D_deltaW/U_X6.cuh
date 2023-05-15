



//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK >= (BLOCK_SIZE/2), IC % 4 == 0
//LB = 4: GK % 8 == 0
#ifndef XKERNEL1
#define XKERNEL1

//OHW2pow: Size = 2.25, Time = 3.196 msec, Performace = 1511.84 GFlop / s
//Size = 2.25, Time = 3.434 msec, Performace = 1407.06 GFlop / s

//GM = FH * FW * IC -> GIC = IC, gridDim.z -> GZ * FH * FW
//GN = OC

#define uxkernel1(stream, LB, GZ, oc_index, ic_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GIC) \
	Xkernel1<LB, (1<<LB>>1)>\
		<<< dim3(GIC>>LB>>3, GN>>LB>>3, GZ * FH * FW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW,\
			 IC, OC, sh, sw, ph, pw, GK, GK_slice, oc_index, ic_index)

//synchronized: 
//GK_slice = 1024, OH, OW =  4, N = 256:
//LB = 4: Size = 1, Time = 1.602 msec, Performace = 1340.5 GFlop/s
//LB = 3: Size = 1, Time = 2.072 msec, Performace = 1036.43 GFlop/s
template<int LB, int STEP>
__global__ void Xkernel1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int IC, int OC,
	int sh, int sw, int oph, int opw,
	int GK, int GK_slice,
	int oc_index, int ic_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//======================================================================
	//prepare for gridDim.z = GZ * FH * FW
	int bz = blockIdx.z;
	const int FH_FW = FH * FW;
	int pIdx = bz / FH_FW; bz %= FH_FW;
	int fh = bz / FW, fw = bz % FW;

	int GK_start = GK_slice * pIdx;
	int GK_end = IF_int((pIdx != ((gridDim.z / FH_FW) - 1)), (GK_start + GK_slice), GK);
	GK_slice = GK_end - GK_start;

	//dst[bz] = deltaW_buf[bz - 1, sizeW], bz >= 1, dst[0] = deltaW
	const int FW_IC = FW * IC, Wstride = FH * FW_IC;
	deltaW_buf += (pIdx - 1) * OC * Wstride;
	deltaW = IF_int((pIdx != 0), deltaW_buf, deltaW);//deltaW -> dst
	//======================================================================

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 3) + oc_index;
	deltaY += oc0 + ((tx >= STEP) << 2);//deltaY[0, 0, 0, toc0]

	//prepare for GM = FH * FW * IC
	int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	int W0 = ((oc0*FH + fh)*FW + fw)*IC + ic0; 
	int tic0 = ic0 + ((ty >= STEP) << 2);
	int tfh0 = fh - oph, tfw0 = fw - opw;
	X += (tfh0*IW + tfw0)*IC + tic0;//X[0, tfh0, tfw0, tic0]

	//load 4 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
	int OH_OW = OH * OW;
	int X_k = ty - ((ty >= STEP) << LB >> 1) + GK_start;
	int X_n, X_oh, X_ow; get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
	int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
	Xs[buf][ty][tx] = (LOAD_X(tfh0, tfw0) ? *(float4*)(X + xoffset) : FLOAT_ZERO4);

	//load 4 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
	int Y_k = tx - ((tx >= STEP) << LB >> 1) + GK_start;
	Ys[buf][tx][ty] = *(float4*)(deltaY + Y_k * OC);
	__syncthreads();

	//compute area-----------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK_slice << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP][ty];
			float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];

			simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
			simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
			simdMM4(v4, a0.z, b0); simdMM4(v5, a0.z, b1);
			simdMM4(v6, a0.w, b0); simdMM4(v7, a0.w, b1);
			simdMM4(v8, a1.x, b0); simdMM4(v9, a1.x, b1);
			simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
			simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
			simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
		}
		buf ^= 1;

		//load 4 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = ((ok - (ty >= STEP)) << LB >> 1) + ty + GK_start;
		int X_n, X_oh, X_ow; get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		Xs[buf][ty][tx] = (LOAD_X(tfh0, tfw0) ? *(float4*)(X + xoffset) : FLOAT_ZERO4);

		//load 4 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx + GK_start;
		Ys[buf][tx][ty] = *(float4*)(deltaY + Y_k * OC);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP][ty];
		float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];

		simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
		simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
		simdMM4(v4, a0.z, b0); simdMM4(v5, a0.z, b1);
		simdMM4(v6, a0.w, b0); simdMM4(v7, a0.w, b1);
		simdMM4(v8, a1.x, b0); simdMM4(v9, a1.x, b1);
		simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
		simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
		simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
	}

	int W1 = W0 + Wstride, W2 = W1 + Wstride, W3 = W2 + Wstride;
	int W4 = W3 + Wstride, W5 = W4 + Wstride, W6 = W5 + Wstride, W7 = W6 + Wstride;

	*(float4*)(deltaW + W0) = v0;  *(float4*)(deltaW + W0 + 4) = v1;
	*(float4*)(deltaW + W1) = v2;  *(float4*)(deltaW + W1 + 4) = v3;
	*(float4*)(deltaW + W2) = v4;  *(float4*)(deltaW + W2 + 4) = v5;
	*(float4*)(deltaW + W3) = v6;  *(float4*)(deltaW + W3 + 4) = v7;
	*(float4*)(deltaW + W4) = v8;  *(float4*)(deltaW + W4 + 4) = v9;
	*(float4*)(deltaW + W5) = v10; *(float4*)(deltaW + W5 + 4) = v11;
	*(float4*)(deltaW + W6) = v12; *(float4*)(deltaW + W6 + 4) = v13;
	*(float4*)(deltaW + W7) = v14; *(float4*)(deltaW + W7 + 4) = v15;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK >= (BLOCK_SIZE/2), IC % 4 == 0
//LB = 4: GK % 8 == 0
#ifndef XKERNEL2
#define XKERNEL2

//OHW2pow: Size = 2.25, Time = 3.196 msec, Performace = 1511.84 GFlop / s
//Size = 2.25, Time = 3.434 msec, Performace = 1407.06 GFlop / s

//GM = FH * FW * IC -> GIC = IC, gridDim.z -> GZ * FH * FW
//GN = OC

#define uxkernel2(stream, LB, GZ, oc_index, ic_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GIC) \
	Xkernel2<LB, (1<<LB>>1)>\
		<<< dim3(GIC>>LB>>3, GN>>LB>>3, GZ * FH * FW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW,\
			 IC, OC, sh, sw, ph, pw, GK, GK_slice, oc_index, ic_index)

//synchronized: 
//GK_slice = 1024, OH, OW =  4, N = 256:
//Target1: Size = 2.25, Time = 3.492 msec, Performace = 1383.69 GFlop/s
//Target2: Size = 2.25, Time = 3.242 msec, Performace = 1490.39 GFlop/s
//LB = 4: Size = 2.25, Time = 3.596 msec, Performace = 1343.67 GFlop/s
template<int LB, int STEP>
__global__ void Xkernel2(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int IC, int OC,
	int sh, int sw, int oph, int opw,
	int GK, int GK_slice,
	int oc_index, int ic_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//======================================================================
	//prepare for gridDim.z = GZ * FH * FW
	int bz = blockIdx.z;
	const int FH_FW = FH * FW;
	int pIdx = bz / FH_FW; bz -= pIdx * FH_FW;
	int fh = bz / FW, fw = bz - fh * FW;

	int GK_start = GK_slice * pIdx;
	int GK_end = IF_int((pIdx != ((gridDim.z / FH_FW) - 1)), (GK_start + GK_slice), GK);
	GK_slice = GK_end - GK_start;

	//dst[pIdx] = deltaW_buf[pIdx - 1, sizeW], pIdx >= 1, dst[0] = deltaW
	const int FW_IC = FW * IC, Wstride = FH * FW_IC;
	deltaW_buf += (pIdx - 1) * OC * Wstride;
	deltaW = IF_int((pIdx != 0), deltaW_buf, deltaW);//deltaW -> dst
	//======================================================================

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 3) + oc_index;
	deltaY += oc0 + ((tx >= STEP) << 2);//deltaY[0, 0, 0, toc0]

	//prepare for GM = FH * FW * IC
	int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	int W0 = ((oc0*FH + fh)*FW + fw)*IC + ic0;
	int tic0 = ic0 + ((ty >= STEP) << 2);
	int tfh0 = fh - oph, tfw0 = fw - opw;
	X += (tfh0*IW + tfw0)*IC + tic0;//X[0, tfh0, tfw0, tic0]

	//load 4 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
	int OH_OW = OH * OW;
	int X_k = ty - ((ty >= STEP) << LB >> 1) + GK_start;
	int X_n, X_oh, X_ow; get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
	int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
	Xs[buf][ty][tx] = (LOAD_X(tfh0, tfw0) ? *(float4*)(X + xoffset) : FLOAT_ZERO4);

	//load 4 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
	int Y_k = tx - ((tx >= STEP) << LB >> 1) + GK_start;
	Ys[buf][tx][ty] = *(float4*)(deltaY + Y_k * OC);
	__syncthreads();

	//compute area-----------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK_slice << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP][ty];
			float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];

			simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
			simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
			simdMM4(v4, a0.z, b0); simdMM4(v5, a0.z, b1);
			simdMM4(v6, a0.w, b0); simdMM4(v7, a0.w, b1);
			simdMM4(v8, a1.x, b0); simdMM4(v9, a1.x, b1);
			simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
			simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
			simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
		}
		buf ^= 1;

		//load 4 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = ((ok - (ty >= STEP)) << LB >> 1) + ty + GK_start;
		int X_n, X_oh, X_ow; get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		Xs[buf][ty][tx] = (LOAD_X(tfh0, tfw0) ? *(float4*)(X + xoffset) : FLOAT_ZERO4);

		//load 4 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx + GK_start;
		Ys[buf][tx][ty] = *(float4*)(deltaY + Y_k * OC);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP][ty];
		float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];

		simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
		simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
		simdMM4(v4, a0.z, b0); simdMM4(v5, a0.z, b1);
		simdMM4(v6, a0.w, b0); simdMM4(v7, a0.w, b1);
		simdMM4(v8, a1.x, b0); simdMM4(v9, a1.x, b1);
		simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
		simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
		simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
	}

	int W1 = W0 + Wstride, W2 = W1 + Wstride, W3 = W2 + Wstride;
	int W4 = W3 + Wstride, W5 = W4 + Wstride, W6 = W5 + Wstride, W7 = W6 + Wstride;

	*(float4*)(deltaW + W0) = v0;  *(float4*)(deltaW + W0 + 4) = v1;
	*(float4*)(deltaW + W1) = v2;  *(float4*)(deltaW + W1 + 4) = v3;
	*(float4*)(deltaW + W2) = v4;  *(float4*)(deltaW + W2 + 4) = v5;
	*(float4*)(deltaW + W3) = v6;  *(float4*)(deltaW + W3 + 4) = v7;
	*(float4*)(deltaW + W4) = v8;  *(float4*)(deltaW + W4 + 4) = v9;
	*(float4*)(deltaW + W5) = v10; *(float4*)(deltaW + W5 + 4) = v11;
	*(float4*)(deltaW + W6) = v12; *(float4*)(deltaW + W6 + 4) = v13;
	*(float4*)(deltaW + W7) = v14; *(float4*)(deltaW + W7 + 4) = v15;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK >= (BLOCK_SIZE/2), IC % 4 == 0
//LB = 4: GK % 8 == 0
#ifndef XKERNEL3
#define XKERNEL3

//OHW2pow: Size = 2.25, Time = 3.196 msec, Performace = 1511.84 GFlop / s
//Size = 2.25, Time = 3.434 msec, Performace = 1407.06 GFlop / s
//GM = FH * FW * IC -> GIC = IC, gridDim.z -> GZ * FH * FW
//GN = OC

#define uxkernel3(stream, LB, GZ, oc_index, ic_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GIC) \
	Xkernel3<LB, (1<<LB>>1)>\
		<<< dim3(GIC>>LB>>3, GN>>LB>>3, GZ * FH * FW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(GZ, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, ic_index)

//synchronized: 
//GK_slice = 1024, OH, OW =  4, N = 256:
//Target1: Size = 2.25, Time = 3.492 msec, Performace = 1383.69 GFlop/s
//Target2: Size = 2.25, Time = 3.242 msec, Performace = 1490.39 GFlop/s
//LB = 4: Size = 2.25, Time = 2.774 msec, Performace = 1741.83 GFlop/s
//LB = 3: Size = 2.25, Time = 3.32 msec, Performace = 1455.37 GFlop/s

template<int LB, int STEP>
__global__ void Xkernel3(int GZ,
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int N, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int oc_index, int ic_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for gridDim.z = GZ * FH * FW
	int bz = blockIdx.z;
	const int FH_FW = FH * FW;
	int pIdx = bz / FH_FW; bz -= pIdx * FH_FW;
	int fh = bz / FW, fw = bz - fh * FW;
	int tfh = fh - oph, tfw = fw - opw;

	//prepare for GK = N * OH * OW
	//[1] tfh + oh * sh >= 0;
	//if: tfh >= 0, oh >= 0
	//if: tfh <  0, oh >= (-tfh + sh - 1) / sh 
	int ohs = IF_int((tfh < 0), ((-tfh + sh - 1) / sh), 0);
	int ows = IF_int((tfw < 0), ((-tfw + sw - 1) / sw), 0);

	//[2] tfh + oh * sh < IH;
	//if: tfh >= 0: oh < OH
	//if: tfh <  0: oh < (IH - tfh) / sh
	int ohe = (IH - tfh + sh - 1) / sh; ohe = IF_int((ohe > OH), OH, ohe);
	int owe = (IW - tfw + sw - 1) / sw; owe = IF_int((owe > OW), OW, owe);
	int tOH = ohe - ohs;
	int tOW = owe - ows;
	int tOH_OW = tOH * tOW, GK = N * tOH_OW;
	deltaY += (ohs*OW + ows)*OC;//deltaY[0, ohs, ows, 0]
	X += ((ohs*sh)*IW + (ows*sw))*IC;//X[0, ohs*sh, ows*sw, 0]

	//prepare for GK_slice
	int GK_slice = (GK / GZ) >> 3 << 3;
	int GK_start = GK_slice * pIdx;
	int GK_end = IF_int((pIdx != (GZ - 1)), (GK_start + GK_slice), GK);
	GK_slice = GK_end - GK_start;

	//dst[pIdx] = deltaW_buf[pIdx - 1, sizeW], pIdx >= 1, dst[0] = deltaW
	const int Wstride = FH * FW * IC;
	deltaW_buf += (pIdx - 1) * OC * Wstride;
	deltaW = IF_int((pIdx != 0), deltaW_buf, deltaW);//deltaW -> dst
	//======================================================================

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 3) + oc_index;
	deltaY += oc0 + ((tx >= STEP) << 2);//deltaY[0, 0, 0, toc0]

	//prepare for GM = FH * FW * IC
	int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	int W0 = ((oc0*FH + fh)*FW + fw)*IC + ic0;
	int tic0 = ic0 + ((ty >= STEP) << 2);
	X += (tfh*IW + tfw)*IC + tic0;//X[0, tfh0, tfw0, tic0]

	//load 4 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
	int X_k = ty - ((ty >= STEP) << LB >> 1) + GK_start;
	int X_n = X_k / tOH_OW; X_k -= X_n * tOH_OW;
	int X_oh = X_k / tOW, X_ow = X_k - X_oh * tOW;
	X_oh = X_oh * sh, X_ow = X_ow * sw;
	int xoffset = ((X_n*IH + X_oh)*IW + X_ow)*IC;
	Xs[buf][ty][tx] = *(float4*)(X + xoffset);

	//load 4 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
	int Y_k = tx - ((tx >= STEP) << LB >> 1) + GK_start;
	int Y_n = Y_k / tOH_OW; Y_k -= Y_n * tOH_OW;
	int Y_oh = Y_k / tOW, Y_ow = Y_k - Y_oh * tOW;
	int yoffset = ((Y_n*OH + Y_oh)*OW + Y_ow)*OC;
	Ys[buf][tx][ty] = *(float4*)(deltaY + yoffset);
	__syncthreads();

	//compute area-----------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK_slice << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP][ty];
			float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];

			simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
			simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
			simdMM4(v4, a0.z, b0); simdMM4(v5, a0.z, b1);
			simdMM4(v6, a0.w, b0); simdMM4(v7, a0.w, b1);
			simdMM4(v8, a1.x, b0); simdMM4(v9, a1.x, b1);
			simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
			simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
			simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
		}
		buf ^= 1;

		//load 4 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = ((ok - (ty >= STEP)) << LB >> 1) + ty + GK_start;
		int X_n = X_k / tOH_OW; X_k -= X_n * tOH_OW;
		int X_oh = X_k / tOW, X_ow = X_k - X_oh * tOW;
		X_oh = X_oh * sh, X_ow = X_ow * sw;
		int xoffset = ((X_n*IH + X_oh)*IW + X_ow)*IC;
		Xs[buf][ty][tx] = *(float4*)(X + xoffset);

		//load 4 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx + GK_start;
		int Y_n = Y_k / tOH_OW; Y_k -= Y_n * tOH_OW;
		int Y_oh = Y_k / tOW, Y_ow = Y_k - Y_oh * tOW;
		int yoffset = ((Y_n*OH + Y_oh)*OW + Y_ow)*OC;
		Ys[buf][tx][ty] = *(float4*)(deltaY + yoffset);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP][ty];
		float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];

		simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
		simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
		simdMM4(v4, a0.z, b0); simdMM4(v5, a0.z, b1);
		simdMM4(v6, a0.w, b0); simdMM4(v7, a0.w, b1);
		simdMM4(v8, a1.x, b0); simdMM4(v9, a1.x, b1);
		simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
		simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
		simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
	}

	int W1 = W0 + Wstride, W2 = W1 + Wstride, W3 = W2 + Wstride;
	int W4 = W3 + Wstride, W5 = W4 + Wstride, W6 = W5 + Wstride, W7 = W6 + Wstride;

	*(float4*)(deltaW + W0) = v0;  *(float4*)(deltaW + W0 + 4) = v1;
	*(float4*)(deltaW + W1) = v2;  *(float4*)(deltaW + W1 + 4) = v3;
	*(float4*)(deltaW + W2) = v4;  *(float4*)(deltaW + W2 + 4) = v5;
	*(float4*)(deltaW + W3) = v6;  *(float4*)(deltaW + W3 + 4) = v7;
	*(float4*)(deltaW + W4) = v8;  *(float4*)(deltaW + W4 + 4) = v9;
	*(float4*)(deltaW + W5) = v10; *(float4*)(deltaW + W5 + 4) = v11;
	*(float4*)(deltaW + W6) = v12; *(float4*)(deltaW + W6 + 4) = v13;
	*(float4*)(deltaW + W7) = v14; *(float4*)(deltaW + W7 + 4) = v15;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK >= (BLOCK_SIZE/2), IC % 4 == 0
//LB = 4: GK % 8 == 0
#ifndef XKERNEL4
#define XKERNEL4

#define uxkernel4(stream, LB, GZ, oc_index, ic_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GIC) \
	Xkernel4<LB, (1<<LB>>1)>\
		<<< dim3(GIC>>LB>>3, GN>>LB>>3, GZ * FH * FW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(GZ, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, ic_index)

//synchronized: 
//LB = 4: Size = 2.25, Time = 2.774 msec, Performace = 1741.83 GFlop/s
//LB = 3: Size = 2.25, Time = 3.32 msec, Performace = 1455.37 GFlop/s

template<int LB, int STEP>
__global__ void Xkernel4(int GZ,
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int N, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int oc_index, int ic_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for gridDim.z = GZ * FH * FW
	int bz = blockIdx.z;
	const int FH_FW = FH * FW;
	int pIdx = bz / FH_FW; bz -= pIdx * FH_FW;
	int fh = bz / FW, fw = bz - fh * FW;
	int tfh = fh - oph, tfw = fw - opw;

	//prepare for GK = N * OH * OW
	int temph = (-tfh + sh - 1);
	int tempw = (-tfw + sw - 1);
	int ohs = IF_int((tfh < 0), (temph / sh), 0);
	int ows = IF_int((tfw < 0), (tempw / sw), 0);
	int ohe = (IH  + temph) / sh; ohe = IF_int((ohe > OH), OH, ohe);
	int owe = (IW  + tempw) / sw; owe = IF_int((owe > OW), OW, owe);
	int tOH = ohe - ohs;
	int tOW = owe - ows;
	int tOH_OW = tOH * tOW, GK = N * tOH_OW;
	deltaY += (ohs*OW + ows)*OC;//deltaY[0, ohs, ows, 0]
	X += ((ohs*sh)*IW + (ows*sw))*IC;//X[0, ohs*sh, ows*sw, 0]

	//prepare for GK_slice
	int GK_slice = (GK / GZ) >> 3 << 3;
	int GK_start = GK_slice * pIdx;
	int GK_end = IF_int((pIdx != (GZ - 1)), (GK_start + GK_slice), GK);
	GK_slice = GK_end - GK_start;

	//prepare for dst: deltaW_buf or deltaW 
	const int Wstride = FH * FW * IC;
	deltaW_buf += (pIdx - 1) * OC * Wstride;
	deltaW = IF_int((pIdx != 0), deltaW_buf, deltaW);//deltaW -> dst
	//======================================================================

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 3) + oc_index;
	deltaY += oc0 + ((tx >= STEP) << 2);//deltaY[0, 0, 0, toc0]

	//prepare for GM = FH * FW * IC
	int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	int W0 = ((oc0*FH + fh)*FW + fw)*IC + ic0;
	int tic0 = ic0 + ((ty >= STEP) << 2);
	X += (tfh*IW + tfw)*IC + tic0;//X[0, tfh0, tfw0, tic0]

	//load 4 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
	int X_k = ty - ((ty >= STEP) << LB >> 1) + GK_start;
	int X_n = X_k / tOH_OW; X_k -= X_n * tOH_OW;
	int X_oh = X_k / tOW, X_ow = X_k - X_oh * tOW;
	X_oh = X_oh * sh, X_ow = X_ow * sw;
	int xoffset = ((X_n*IH + X_oh)*IW + X_ow)*IC;
	Xs[buf][ty][tx] = *(float4*)(X + xoffset);

	//load 4 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
	int Y_k = tx - ((tx >= STEP) << LB >> 1) + GK_start;
	int Y_n = Y_k / tOH_OW; Y_k -= Y_n * tOH_OW;
	int Y_oh = Y_k / tOW, Y_ow = Y_k - Y_oh * tOW;
	int yoffset = ((Y_n*OH + Y_oh)*OW + Y_ow)*OC;
	Ys[buf][tx][ty] = *(float4*)(deltaY + yoffset);
	__syncthreads();

	//compute area-----------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK_slice << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP][ty];
			float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];

			simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
			simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
			simdMM4(v4, a0.z, b0); simdMM4(v5, a0.z, b1);
			simdMM4(v6, a0.w, b0); simdMM4(v7, a0.w, b1);
			simdMM4(v8, a1.x, b0); simdMM4(v9, a1.x, b1);
			simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
			simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
			simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
		}
		buf ^= 1;

		//load 4 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = ((ok - (ty >= STEP)) << LB >> 1) + ty + GK_start;
		int X_n = X_k / tOH_OW; X_k -= X_n * tOH_OW;
		int X_oh = X_k / tOW, X_ow = X_k - X_oh * tOW;
		X_oh = X_oh * sh, X_ow = X_ow * sw;
		int xoffset = ((X_n*IH + X_oh)*IW + X_ow)*IC;
		Xs[buf][ty][tx] = *(float4*)(X + xoffset);

		//load 4 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx + GK_start;
		int Y_n = Y_k / tOH_OW; Y_k -= Y_n * tOH_OW;
		int Y_oh = Y_k / tOW, Y_ow = Y_k - Y_oh * tOW;
		int yoffset = ((Y_n*OH + Y_oh)*OW + Y_ow)*OC;
		Ys[buf][tx][ty] = *(float4*)(deltaY + yoffset);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP][ty];
		float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];

		simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
		simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
		simdMM4(v4, a0.z, b0); simdMM4(v5, a0.z, b1);
		simdMM4(v6, a0.w, b0); simdMM4(v7, a0.w, b1);
		simdMM4(v8, a1.x, b0); simdMM4(v9, a1.x, b1);
		simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
		simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
		simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
	}

	int W1 = W0 + Wstride, W2 = W1 + Wstride, W3 = W2 + Wstride;
	int W4 = W3 + Wstride, W5 = W4 + Wstride, W6 = W5 + Wstride, W7 = W6 + Wstride;

	*(float4*)(deltaW + W0) = v0;  *(float4*)(deltaW + W0 + 4) = v1;
	*(float4*)(deltaW + W1) = v2;  *(float4*)(deltaW + W1 + 4) = v3;
	*(float4*)(deltaW + W2) = v4;  *(float4*)(deltaW + W2 + 4) = v5;
	*(float4*)(deltaW + W3) = v6;  *(float4*)(deltaW + W3 + 4) = v7;
	*(float4*)(deltaW + W4) = v8;  *(float4*)(deltaW + W4 + 4) = v9;
	*(float4*)(deltaW + W5) = v10; *(float4*)(deltaW + W5 + 4) = v11;
	*(float4*)(deltaW + W6) = v12; *(float4*)(deltaW + W6 + 4) = v13;
	*(float4*)(deltaW + W7) = v14; *(float4*)(deltaW + W7 + 4) = v15;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK >= (BLOCK_SIZE/2), IC % 4 == 0
//LB = 4: GK % 8 == 0
#ifndef XKERNEL5
#define XKERNEL5

#define uxkernel5(stream, LB, GZ, oc_index, ic_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GIC) \
	Xkernel5<LB, (1<<LB>>1)>\
		<<< dim3(GIC>>LB>>3, GN>>LB>>3, GZ * FH * FW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(GZ, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, ic_index)

//synchronized: 
//LB = 4: Size = 2.25, Time = 2.774 msec, Performace = 1741.83 GFlop/s
//LB = 3: Size = 2.25, Time = 3.32  msec, Performace = 1455.37 GFlop/s
template<int LB, int STEP>
__global__ void Xkernel5(int GZ,
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int N, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int oc_index, int ic_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for gridDim.z = GZ * FH * FW
	int bz = blockIdx.z;
	const int FH_FW = FH * FW;
	int pIdx = bz / FH_FW; bz -= pIdx * FH_FW;
	int fh = bz / FW, fw = bz - fh * FW;
	int tfh = fh - oph, tfw = fw - opw;

	//prepare for GK = N * OH * OW
	int temph = (-tfh + sh - 1);
	int tempw = (-tfw + sw - 1);
	int ohs = IF_int((tfh < 0), (temph / sh), 0);
	int ows = IF_int((tfw < 0), (tempw / sw), 0);
	int ohe = (IH + temph) / sh; ohe = IF_int((ohe > OH), OH, ohe);
	int owe = (IW + tempw) / sw; owe = IF_int((owe > OW), OW, owe);
	int tOH = ohe - ohs;
	int tOW = owe - ows;
	int tOW_N = tOW * N, GK = tOH * tOW_N;
	deltaY += (ohs*OW + ows)*OC;//deltaY[0, ohs, ows, 0]
	X += ((ohs*sh)*IW + (ows*sw))*IC;//X[0, ohs*sh, ows*sw, 0]

	//prepare for GK_slice
	int GK_slice = (GK / GZ) >> 3 << 3;
	int GK_start = GK_slice * pIdx;
	int GK_end = IF_int((pIdx != (GZ - 1)), (GK_start + GK_slice), GK);
	GK_slice = GK_end - GK_start;

	//prepare for dst: deltaW_buf or deltaW 
	const int Wstride = FH * FW * IC;
	deltaW_buf += (pIdx - 1) * OC * Wstride;
	deltaW = IF_int((pIdx != 0), deltaW_buf, deltaW);//deltaW -> dst
	//======================================================================

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 3) + oc_index;
	deltaY += oc0 + ((tx >= STEP) << 2);//deltaY[0, 0, 0, toc0]

	//prepare for GM = FH * FW * IC
	int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	int W0 = ((oc0*FH + fh)*FW + fw)*IC + ic0;
	int tic0 = ic0 + ((ty >= STEP) << 2);
	X += (tfh*IW + tfw)*IC + tic0;//X[0, tfh0, tfw0, tic0]

	//load 4 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
	int X_k = ty - ((ty >= STEP) << LB >> 1) + GK_start;
	int X_oh = X_k / tOW_N; X_k -= X_oh * tOW_N;
	int X_ow = X_k / N, X_n = X_k - X_ow * N;
	X_oh = X_oh * sh, X_ow = X_ow * sw;
	int xoffset = ((X_n*IH + X_oh)*IW + X_ow)*IC;
	Xs[buf][ty][tx] = *(float4*)(X + xoffset);

	//load 4 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
	int Y_k = tx - ((tx >= STEP) << LB >> 1) + GK_start;
	int Y_oh = Y_k / tOW_N; Y_k -= Y_oh * tOW_N;
	int Y_ow = Y_k / N, Y_n = Y_k - Y_ow * N;
	int yoffset = ((Y_n*OH + Y_oh)*OW + Y_ow)*OC;
	Ys[buf][tx][ty] = *(float4*)(deltaY + yoffset);
	__syncthreads();

	//compute area-----------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK_slice << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP][ty];
			float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];

			simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
			simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
			simdMM4(v4, a0.z, b0); simdMM4(v5, a0.z, b1);
			simdMM4(v6, a0.w, b0); simdMM4(v7, a0.w, b1);
			simdMM4(v8, a1.x, b0); simdMM4(v9, a1.x, b1);
			simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
			simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
			simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
		}
		buf ^= 1;

		//load 4 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = ((ok - (ty >= STEP)) << LB >> 1) + ty + GK_start;
		int X_oh = X_k / tOW_N; X_k -= X_oh * tOW_N;
		int X_ow = X_k / N, X_n = X_k - X_ow * N;
		X_oh = X_oh * sh, X_ow = X_ow * sw;
		int xoffset = ((X_n*IH + X_oh)*IW + X_ow)*IC;
		Xs[buf][ty][tx] = *(float4*)(X + xoffset);

		//load 4 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx + GK_start;
		int Y_oh = Y_k / tOW_N; Y_k -= Y_oh * tOW_N;
		int Y_ow = Y_k / N, Y_n = Y_k - Y_ow * N;
		int yoffset = ((Y_n*OH + Y_oh)*OW + Y_ow)*OC;
		Ys[buf][tx][ty] = *(float4*)(deltaY + yoffset);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP][ty];
		float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];

		simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
		simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
		simdMM4(v4, a0.z, b0); simdMM4(v5, a0.z, b1);
		simdMM4(v6, a0.w, b0); simdMM4(v7, a0.w, b1);
		simdMM4(v8, a1.x, b0); simdMM4(v9, a1.x, b1);
		simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
		simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
		simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
	}

	int W1 = W0 + Wstride, W2 = W1 + Wstride, W3 = W2 + Wstride;
	int W4 = W3 + Wstride, W5 = W4 + Wstride, W6 = W5 + Wstride, W7 = W6 + Wstride;

	*(float4*)(deltaW + W0) = v0;  *(float4*)(deltaW + W0 + 4) = v1;
	*(float4*)(deltaW + W1) = v2;  *(float4*)(deltaW + W1 + 4) = v3;
	*(float4*)(deltaW + W2) = v4;  *(float4*)(deltaW + W2 + 4) = v5;
	*(float4*)(deltaW + W3) = v6;  *(float4*)(deltaW + W3 + 4) = v7;
	*(float4*)(deltaW + W4) = v8;  *(float4*)(deltaW + W4 + 4) = v9;
	*(float4*)(deltaW + W5) = v10; *(float4*)(deltaW + W5 + 4) = v11;
	*(float4*)(deltaW + W6) = v12; *(float4*)(deltaW + W6 + 4) = v13;
	*(float4*)(deltaW + W7) = v14; *(float4*)(deltaW + W7 + 4) = v15;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK >= (BLOCK_SIZE/2), IC % 4 == 0
//LB = 4: GK % 8 == 0
#ifndef XKERNEL6
#define XKERNEL6

#define uxkernel6(stream, LB, GZ, oc_index, ic_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GIC) \
	Xkernel6<LB, (1<<LB>>1)>\
		<<< dim3(GIC>>LB>>3, GN>>LB>>3, GZ * FH * FW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(GZ, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, ic_index)

//synchronized: 
//LB = 4: Size = 2.25, Time = 2.774 msec, Performace = 1741.83 GFlop/s
//LB = 3: Size = 2.25, Time = 3.32  msec, Performace = 1455.37 GFlop/s
template<int LB, int STEP>
__global__ void Xkernel6(int GZ,
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int N, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int oc_index, int ic_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for gridDim.z = GZ * FH * FW
	int bz = blockIdx.z;
	const int FH_FW = FH * FW;
	int pIdx = bz / FH_FW; bz -= pIdx * FH_FW;
	int fh = bz / FW, fw = bz - fh * FW;
	int tfh = fh - oph, tfw = fw - opw;

	//prepare for GK = N * OH * OW
	int temph = (-tfh + sh - 1);
	int tempw = (-tfw + sw - 1);
	int ohs = IF_int((tfh < 0), (temph / sh), 0);
	int ows = IF_int((tfw < 0), (tempw / sw), 0);
	int ohe = (IH + temph) / sh; ohe = IF_int((ohe > OH), OH, ohe);
	int owe = (IW + tempw) / sw; owe = IF_int((owe > OW), OW, owe);
	int tOH = ohe - ohs;
	int tOW = owe - ows;
	int tOW_N = tOW * N, GK = tOH * tOW_N;
	deltaY += (ohs*OW + ows)*OC;//deltaY[0, ohs, ows, 0]
	X += ((ohs*sh)*IW + (ows*sw))*IC;//X[0, ohs*sh, ows*sw, 0]

	//prepare for GK_slice
	int GK_slice = (GK / GZ) >> 3 << 3;
	int GK_start = GK_slice * pIdx;
	int GK_end = IF_int((pIdx != (GZ - 1)), (GK_start + GK_slice), GK);
	GK_slice = GK_end - GK_start;

	//prepare for dst: deltaW_buf or deltaW 
	const int Wstride = FH * FW * IC;
	deltaW_buf += (pIdx - 1) * OC * Wstride;
	deltaW = IF_int((pIdx != 0), deltaW_buf, deltaW);//deltaW -> dst
	//======================================================================

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 3) + oc_index;
	deltaY += oc0 + ((tx >= STEP) << 2);//deltaY[0, 0, 0, toc0]

	//prepare for GM = FH * FW * IC
	int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	int W0 = ((oc0*FH + fh)*FW + fw)*IC + ic0;
	int tic0 = ic0 + ((ty >= STEP) << 2);
	X += (tfh*IW + tfw)*IC + tic0;//X[0, tfh0, tfw0, tic0]

	//GK_start % 8 == 0
	//[1] X_k = ok*STEP + {ty - ((ty >= STEP) << LB >> 1)} + GK_start
	//[2] Y_k = ok*STEP + {tx - ((tx >= STEP) << LB >> 1)} + GK_start
	//So:
	//[1] X_k = ok*STEP + {ty - ((ty >= STEP) << LB >> 1)} + 8z
	//[2] Y_k = ok*STEP + {tx - ((tx >= STEP) << LB >> 1)} + 8z
	//when: LB = 4, STEP = 8; N % 8 == 0, tOW_N % 8 == 0
	//[1] X_k = 8x + Ux + 8z
	//[2] Y_k = 8x + Uy + 8z
	//As: Ux, Uy belongs to [0, 7]
	//So: X_oh = X_k / tOW_N = (8*(x+z) + Ux) / 8y
	//So: Y_oh = Y_k / tOW_N = (8*(x+z) + Uy) / 8y
	//X_oh = Y_oh
	//when LB = 3, STEP = 4; N % 4 == 0, tOW_N % 4 == 0
	//As: Ux, Uy belongs to [0, 3]
	//So: X_oh = X_k / tOW_N = (4*(2x+2z) + Ux) / 4y
	//So: Y_oh = Y_k / tOW_N = (4*(2x+2z) + Uy) / 4y
	//X_oh = Y_oh

	//load 4 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
	int X_k = ty - ((ty >= STEP) << LB >> 1) + GK_start;
	int oh = X_k / tOW_N; X_k -= oh * tOW_N;
	int X_ow = X_k / N, X_n = X_k - X_ow * N;
	int xoffset = ((X_n*IH + (oh*sh))*IW + (X_ow*sw))*IC;
	Xs[buf][ty][tx] = *(float4*)(X + xoffset);

	//load 4 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
	int Y_k = tx - ((tx >= STEP) << LB >> 1) + GK_start;
	Y_k -= oh * tOW_N;
	int Y_ow = Y_k / N, Y_n = Y_k - Y_ow * N;
	int yoffset = ((Y_n*OH + oh)*OW + Y_ow)*OC;
	Ys[buf][tx][ty] = *(float4*)(deltaY + yoffset);
	__syncthreads();

	//compute area-----------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK_slice << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP][ty];
			float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];

			simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
			simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
			simdMM4(v4, a0.z, b0); simdMM4(v5, a0.z, b1);
			simdMM4(v6, a0.w, b0); simdMM4(v7, a0.w, b1);
			simdMM4(v8, a1.x, b0); simdMM4(v9, a1.x, b1);
			simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
			simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
			simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
		}
		buf ^= 1;


		//load 4 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = ((ok - (ty >= STEP)) << LB >> 1) + ty + GK_start;
		int oh = X_k / tOW_N; X_k -= oh * tOW_N;
		int X_ow = X_k / N, X_n = X_k - X_ow * N;
		int xoffset = ((X_n*IH + (oh*sh))*IW + (X_ow*sw))*IC;
		Xs[buf][ty][tx] = *(float4*)(X + xoffset);

		//load 4 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx + GK_start;
		Y_k -= oh * tOW_N;
		int Y_ow = Y_k / N, Y_n = Y_k - Y_ow * N;
		int yoffset = ((Y_n*OH + oh)*OW + Y_ow)*OC;
		Ys[buf][tx][ty] = *(float4*)(deltaY + yoffset);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP][ty];
		float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];

		simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
		simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
		simdMM4(v4, a0.z, b0); simdMM4(v5, a0.z, b1);
		simdMM4(v6, a0.w, b0); simdMM4(v7, a0.w, b1);
		simdMM4(v8, a1.x, b0); simdMM4(v9, a1.x, b1);
		simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
		simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
		simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
	}

	int W1 = W0 + Wstride, W2 = W1 + Wstride, W3 = W2 + Wstride;
	int W4 = W3 + Wstride, W5 = W4 + Wstride, W6 = W5 + Wstride, W7 = W6 + Wstride;

	*(float4*)(deltaW + W0) = v0;  *(float4*)(deltaW + W0 + 4) = v1;
	*(float4*)(deltaW + W1) = v2;  *(float4*)(deltaW + W1 + 4) = v3;
	*(float4*)(deltaW + W2) = v4;  *(float4*)(deltaW + W2 + 4) = v5;
	*(float4*)(deltaW + W3) = v6;  *(float4*)(deltaW + W3 + 4) = v7;
	*(float4*)(deltaW + W4) = v8;  *(float4*)(deltaW + W4 + 4) = v9;
	*(float4*)(deltaW + W5) = v10; *(float4*)(deltaW + W5 + 4) = v11;
	*(float4*)(deltaW + W6) = v12; *(float4*)(deltaW + W6 + 4) = v13;
	*(float4*)(deltaW + W7) = v14; *(float4*)(deltaW + W7 + 4) = v15;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK >= (BLOCK_SIZE/2), IC % 4 == 0
//LB = 4: GK % 8 == 0
#ifndef XKERNEL7
#define XKERNEL7

#define uxkernel7(stream, LB, GZ, oc_index, ic_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GIC) \
	Xkernel7<LB, (1<<LB>>1)>\
		<<< dim3(GIC>>LB>>3, GN>>LB>>3, GZ * FH * FW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(GZ, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, ic_index)

//synchronized: 
//LB = 4: Size = 2.25, Time = 2.648 msec, Performace = 1824.71 GFlop/s
//LB = 3: Size = 2.25, Time = 3.094 msec, Performace = 1561.68 GFlop/s
template<int LB, int STEP>
__global__ void Xkernel7(int GZ,
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int N, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int oc_index, int ic_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for gridDim.z = GZ * FH * FW
	int bz = blockIdx.z;
	const int FH_FW = FH * FW;
	int pIdx = bz / FH_FW; bz -= pIdx * FH_FW;
	int fh = bz / FW, fw = bz - fh * FW;
	int tfh = fh - oph, tfw = fw - opw;

	//prepare for GK = N * OH * OW
	int temph = (-tfh + sh - 1);
	int tempw = (-tfw + sw - 1);
	int ohs = IF_int((tfh < 0), (temph / sh), 0);
	int ows = IF_int((tfw < 0), (tempw / sw), 0);
	int ohe = (IH + temph) / sh; ohe = IF_int((ohe > OH), OH, ohe);
	int owe = (IW + tempw) / sw; owe = IF_int((owe > OW), OW, owe);
	int tOH = ohe - ohs;
	int tOW = owe - ows;
	int tOW_N = tOW * N, GK = tOH * tOW_N;
	deltaY += (ohs*OW + ows)*OC;//deltaY[0, ohs, ows, 0]
	X += ((ohs*sh)*IW + (ows*sw))*IC;//X[0, ohs*sh, ows*sw, 0]

	//prepare for GK_slice
	int GK_slice = (GK / GZ) >> 3 << 3;
	int GK_start = GK_slice * pIdx;
	int GK_end = IF_int((pIdx != (GZ - 1)), (GK_start + GK_slice), GK);
	GK_slice = GK_end - GK_start;

	//prepare for dst: deltaW_buf or deltaW 
	const int Wstride = FH * FW * IC;
	deltaW_buf += (pIdx - 1) * OC * Wstride;
	deltaW = IF_int((pIdx != 0), deltaW_buf, deltaW);//deltaW -> dst
	//======================================================================

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 3) + oc_index;
	deltaY += oc0 + ((tx >= STEP) << 2);//deltaY[0, 0, 0, toc0]

	//prepare for GM = FH * FW * IC
	int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	int W0 = ((oc0*FH + fh)*FW + fw)*IC + ic0;
	int tic0 = ic0 + ((ty >= STEP) << 2);
	X += (tfh*IW + tfw)*IC + tic0;//X[0, tfh0, tfw0, tic0]

	//GK_start % 8 == 0
	//[1] X_k = ok*STEP + {ty - ((ty >= STEP) << LB >> 1)} + GK_start
	//[2] Y_k = ok*STEP + {tx - ((tx >= STEP) << LB >> 1)} + GK_start
	//So:
	//[1] X_k = ok*STEP + {ty - ((ty >= STEP) << LB >> 1)} + 8z
	//[2] Y_k = ok*STEP + {tx - ((tx >= STEP) << LB >> 1)} + 8z
	//X_ow = (X_k % tOW_N) / N
	//Y_ow = (X_k % tOW_N) / N
	//(1) when LB = 4, STEP = 8; N % 8 == 0, tOW_N % 8 == 0
	//X_ow = ((8u + Ux) % 8z) / 8y
	//Y_ow = ((8u + Uy) % 8z) / 8y
	//As: Ux, Uy belongs to [0, 7]
	//X_ow = (8(u - z) + Ux) / 8y, (u - z) >= 0
	//Y_ow = (8(u - z) + 8z) / 8y
	//X_ow = Y_ow
	//(2) when LB = 3, STEP = 4; N % 4 == 0, tOW_N % 4 == 0
	//X_ow = ((4u + Ux) % 4z) / 4y
	//Y_ow = ((4u + Uy) % 4z) / 4y
	//As: Ux, Uy belongs to [0, 3]
	//X_ow = (4(u - z) + Ux) / 4y, (u - z) >= 0
	//Y_ow = (4(u - z) + 8z) / 4y
	//X_ow = Y_ow

	//load 4 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
	int X_k = ty - ((ty >= STEP) << LB >> 1) + GK_start;
	int oh = X_k / tOW_N; X_k -= oh * tOW_N;
	int ow = X_k / N, X_n = X_k - ow * N;
	int xoffset = ((X_n*IH + (oh*sh))*IW + (ow*sw))*IC;
	Xs[buf][ty][tx] = *(float4*)(X + xoffset);

	//load 4 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
	int Y_k = tx - ((tx >= STEP) << LB >> 1) + GK_start;
	Y_k -= oh * tOW_N; int Y_n = Y_k - ow * N;
	int yoffset = ((Y_n*OH + oh)*OW + ow)*OC;
	Ys[buf][tx][ty] = *(float4*)(deltaY + yoffset);
	__syncthreads();

	//compute area-----------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK_slice << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP][ty];
			float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];

			simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
			simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
			simdMM4(v4, a0.z, b0); simdMM4(v5, a0.z, b1);
			simdMM4(v6, a0.w, b0); simdMM4(v7, a0.w, b1);
			simdMM4(v8, a1.x, b0); simdMM4(v9, a1.x, b1);
			simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
			simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
			simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
		}
		buf ^= 1;

		//load 4 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = ((ok - (ty >= STEP)) << LB >> 1) + ty + GK_start;
		int oh = X_k / tOW_N; X_k -= oh * tOW_N;
		int ow = X_k / N, X_n = X_k - ow * N;
		int xoffset = ((X_n*IH + (oh*sh))*IW + (ow*sw))*IC;
		Xs[buf][ty][tx] = *(float4*)(X + xoffset);

		//load 4 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx + GK_start;
		Y_k -= oh * tOW_N; int Y_n = Y_k - ow * N;
		int yoffset = ((Y_n*OH + oh)*OW + ow)*OC;
		Ys[buf][tx][ty] = *(float4*)(deltaY + yoffset);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP][ty];
		float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];

		simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
		simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
		simdMM4(v4, a0.z, b0); simdMM4(v5, a0.z, b1);
		simdMM4(v6, a0.w, b0); simdMM4(v7, a0.w, b1);
		simdMM4(v8, a1.x, b0); simdMM4(v9, a1.x, b1);
		simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
		simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
		simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
	}

	int W1 = W0 + Wstride, W2 = W1 + Wstride, W3 = W2 + Wstride;
	int W4 = W3 + Wstride, W5 = W4 + Wstride, W6 = W5 + Wstride, W7 = W6 + Wstride;

	*(float4*)(deltaW + W0) = v0;  *(float4*)(deltaW + W0 + 4) = v1;
	*(float4*)(deltaW + W1) = v2;  *(float4*)(deltaW + W1 + 4) = v3;
	*(float4*)(deltaW + W2) = v4;  *(float4*)(deltaW + W2 + 4) = v5;
	*(float4*)(deltaW + W3) = v6;  *(float4*)(deltaW + W3 + 4) = v7;
	*(float4*)(deltaW + W4) = v8;  *(float4*)(deltaW + W4 + 4) = v9;
	*(float4*)(deltaW + W5) = v10; *(float4*)(deltaW + W5 + 4) = v11;
	*(float4*)(deltaW + W6) = v12; *(float4*)(deltaW + W6 + 4) = v13;
	*(float4*)(deltaW + W7) = v14; *(float4*)(deltaW + W7 + 4) = v15;
}

#endif


#ifndef XKERNEL8
#define XKERNEL8

#define uxkernel8(stream, LB, GZ, oc_index, ic_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GIC) \
	Xkernel8<LB, (1<<LB>>1)>\
		<<< dim3(GIC>>LB>>3, GN>>LB>>3, GZ * FH * FW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(GZ, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, ic_index)

//synchronized: 
//LB = 4: Size = 2.25, Time = 2.654 msec, Performace = 1820.59 GFlop/s
//LB = 3: Size = 2.25, Time = 3.094 msec, Performace = 1561.68 GFlop/s
template<int LB, int STEP>
__global__ void Xkernel8(int GZ,
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int N, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int oc_index, int ic_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//======================================================================
	//prepare for gridDim.z = GZ * FH * FW
	int bz = blockIdx.z;
	const int FH_FW = FH * FW;
	int pIdx = bz / FH_FW; bz -= pIdx * FH_FW;
	int fh = bz / FW, fw = bz - fh * FW;
	int tfh = fh - oph, tfw = fw - opw;

	//prepare for GK = N * OH * OW
	int temph = (-tfh + sh - 1);
	int tempw = (-tfw + sw - 1);
	int ohs = IF_int((tfh < 0), (temph / sh), 0);
	int ows = IF_int((tfw < 0), (tempw / sw), 0);
	int ohe = (IH + temph) / sh; ohe = IF_int((ohe > OH), OH, ohe);
	int owe = (IW + tempw) / sw; owe = IF_int((owe > OW), OW, owe);
	int tOH = ohe - ohs;
	int tOW = owe - ows;
	int tOW_N = tOW * N, GK = tOH * tOW_N;
	deltaY += (ohs*OW + ows)*OC;//deltaY[0, ohs, ows, 0]
	X += ((ohs*sh)*IW + (ows*sw))*IC;//X[0, ohs*sh, ows*sw, 0]

	//prepare for GK_slice
	int GK_slice = (GK / GZ) >> 3 << 3;
	int GK_start = GK_slice * pIdx;
	int GK_end = IF_int((pIdx != (GZ - 1)), (GK_start + GK_slice), GK);
	GK_slice = GK_end - GK_start;

	//prepare for dst: deltaW_buf or deltaW 
	const int Wstride = FH * FW * IC;
	deltaW_buf += (pIdx - 1) * OC * Wstride;
	deltaW = IF_int((pIdx != 0), deltaW_buf, deltaW);//deltaW -> dst
	//======================================================================

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 3) + oc_index;
	deltaY += oc0 + ((tx >= STEP) << 2);//deltaY[0, 0, 0, toc0]

	//prepare for GM = FH * FW * IC
	int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	int W0 = ((oc0*FH + fh)*FW + fw)*IC + ic0;
	int tic0 = ic0 + ((ty >= STEP) << 2);
	X += (tfh*IW + tfw)*IC + tic0;//X[0, tfh0, tfw0, tic0]

	//load 4 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
	int X_k = ty - ((ty >= STEP) << LB >> 1) + GK_start;
	int oh = X_k / tOW_N, km1 = oh * tOW_N; X_k -= km1;//X_k %= tOW_N
	int ow = X_k / N, km2 = ow * N, X_n = X_k - km2; km1 += km2;
	int xoffset = ((X_n*IH + (oh*sh))*IW + (ow*sw))*IC;
	Xs[buf][ty][tx] = *(float4*)(X + xoffset);

	//load 4 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
	int Y_k = tx - ((tx >= STEP) << LB >> 1) + GK_start;
	int Y_n = Y_k - km1;//Y_n = Y_k % N
	int yoffset = ((Y_n*OH + oh)*OW + ow)*OC;
	Ys[buf][tx][ty] = *(float4*)(deltaY + yoffset);
	__syncthreads();

	//compute area-----------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK_slice << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP][ty];
			float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];

			simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
			simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
			simdMM4(v4, a0.z, b0); simdMM4(v5, a0.z, b1);
			simdMM4(v6, a0.w, b0); simdMM4(v7, a0.w, b1);
			simdMM4(v8, a1.x, b0); simdMM4(v9, a1.x, b1);
			simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
			simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
			simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
		}
		buf ^= 1;

		//load 4 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = ((ok - (ty >= STEP)) << LB >> 1) + ty + GK_start;
		int oh = X_k / tOW_N, km1 = oh * tOW_N; X_k -= km1;
		int ow = X_k / N, km2 = ow * N, X_n = X_k - km2; km1 += km2;
		int xoffset = ((X_n*IH + (oh*sh))*IW + (ow*sw))*IC;
		Xs[buf][ty][tx] = *(float4*)(X + xoffset);

		//load 4 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx + GK_start;
		int Y_n = Y_k - km1;
		int yoffset = ((Y_n*OH + oh)*OW + ow)*OC;
		Ys[buf][tx][ty] = *(float4*)(deltaY + yoffset);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP][ty];
		float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];

		simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
		simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
		simdMM4(v4, a0.z, b0); simdMM4(v5, a0.z, b1);
		simdMM4(v6, a0.w, b0); simdMM4(v7, a0.w, b1);
		simdMM4(v8, a1.x, b0); simdMM4(v9, a1.x, b1);
		simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
		simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
		simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
	}

	int W1 = W0 + Wstride, W2 = W1 + Wstride, W3 = W2 + Wstride;
	int W4 = W3 + Wstride, W5 = W4 + Wstride, W6 = W5 + Wstride, W7 = W6 + Wstride;

	*(float4*)(deltaW + W0) = v0;  *(float4*)(deltaW + W0 + 4) = v1;
	*(float4*)(deltaW + W1) = v2;  *(float4*)(deltaW + W1 + 4) = v3;
	*(float4*)(deltaW + W2) = v4;  *(float4*)(deltaW + W2 + 4) = v5;
	*(float4*)(deltaW + W3) = v6;  *(float4*)(deltaW + W3 + 4) = v7;
	*(float4*)(deltaW + W4) = v8;  *(float4*)(deltaW + W4 + 4) = v9;
	*(float4*)(deltaW + W5) = v10; *(float4*)(deltaW + W5 + 4) = v11;
	*(float4*)(deltaW + W6) = v12; *(float4*)(deltaW + W6 + 4) = v13;
	*(float4*)(deltaW + W7) = v14; *(float4*)(deltaW + W7 + 4) = v15;
}

#endif


#ifndef XKERNEL9
#define XKERNEL9

#define uxkernel9(stream, LB, GZ, oc_index, ic_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GIC) \
	Xkernel9<LB, (1<<LB>>1)>\
		<<< dim3(GIC>>LB>>3, GN>>LB>>3, GZ * FH * FW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(GZ, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, ic_index)

//synchronized: 
//LB = 4: Size = 2.25, Time = 2.606 msec, Performace = 1854.12 GFlop/s
//LB = 3: Size = 2.25, Time = 3.094 msec, Performace = 1561.68 GFlop/s
template<int LB, int STEP>
__global__ void Xkernel9(int GZ,
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int N, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int oc_index, int ic_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//======================================================================
	//prepare for gridDim.z = GZ * FH * FW
	int bz = blockIdx.z;
	const int FH_FW = FH * FW;
	int pIdx = bz / FH_FW; bz -= pIdx * FH_FW;
	int fh = bz / FW, fw = bz - fh * FW;
	int tfh = fh - oph, tfw = fw - opw;

	//prepare for GK = N * OH * OW
	int temph = (-tfh + sh - 1);
	int tempw = (-tfw + sw - 1);
	int ohs = IF_int((tfh < 0), (temph / sh), 0);
	int ows = IF_int((tfw < 0), (tempw / sw), 0);
	int ohe = (IH + temph) / sh; ohe = IF_int((ohe > OH), OH, ohe);
	int owe = (IW + tempw) / sw; owe = IF_int((owe > OW), OW, owe);
	int tOH = ohe - ohs;
	int tOW = owe - ows;
	int tOW_N = tOW * N, GK = tOH * tOW_N;
	deltaY += (ohs*OW + ows)*OC;//deltaY[0, ohs, ows, 0]
	X += ((ohs*sh)*IW + (ows*sw))*IC;//X[0, ohs*sh, ows*sw, 0]

	//prepare for GK_slice
	int GK_slice = (GK / GZ) >> 3 << 3;
	int GK_start = GK_slice * pIdx;
	int GK_end = IF_int((pIdx != (GZ - 1)), (GK_start + GK_slice), GK);
	GK_slice = GK_end - GK_start;

	//prepare for dst: deltaW_buf or deltaW 
	const int Wstride = FH * FW * IC;
	deltaW_buf += (pIdx - 1) * OC * Wstride;
	deltaW = IF_int((pIdx != 0), deltaW_buf, deltaW);//deltaW -> dst
	//======================================================================

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 3) + oc_index;
	deltaY += oc0 + ((tx >= STEP) << 2);//deltaY[0, 0, 0, toc0]

	//prepare for GM = FH * FW * IC
	int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	int W0 = ((oc0*FH + fh)*FW + fw)*IC + ic0;
	int tic0 = ic0 + ((ty >= STEP) << 2);
	X += (tfh*IW + tfw)*IC + tic0;//X[0, tfh0, tfw0, tic0]
	IH = IH * IW;//IH -> IH * IW
	IW = IW * sh;//IW -> IW * sh

	//load 4 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
	int X_k = ty - ((ty >= STEP) << LB >> 1) + GK_start;
	int oh = X_k / tOW_N, km1 = oh * tOW_N; X_k -= km1;//X_k %= tOW_N
	int ow = X_k / N, km2 = ow * N, X_n = X_k - km2; km1 += km2;
	int xoffset = (X_n*IH + oh*IW + (ow*sw))*IC;
	Xs[buf][ty][tx] = *(float4*)(X + xoffset);

	//load 4 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
	int Y_k = tx - ((tx >= STEP) << LB >> 1) + GK_start;
	int Y_n = Y_k - km1;//Y_n = Y_k % N
	int yoffset = ((Y_n*OH + oh)*OW + ow)*OC;
	Ys[buf][tx][ty] = *(float4*)(deltaY + yoffset);
	__syncthreads();

	//compute area-----------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK_slice << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP][ty];
			float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];

			simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
			simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
			simdMM4(v4, a0.z, b0); simdMM4(v5, a0.z, b1);
			simdMM4(v6, a0.w, b0); simdMM4(v7, a0.w, b1);
			simdMM4(v8, a1.x, b0); simdMM4(v9, a1.x, b1);
			simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
			simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
			simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
		}
		buf ^= 1;

		//load 4 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = ((ok - (ty >= STEP)) << LB >> 1) + ty + GK_start;
		int oh = X_k / tOW_N, km1 = oh * tOW_N; X_k -= km1;
		int ow = X_k / N, km2 = ow * N, X_n = X_k - km2; km1 += km2;
		int xoffset = (X_n*IH + oh * IW + (ow*sw))*IC;
		Xs[buf][ty][tx] = *(float4*)(X + xoffset);

		//load 4 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx + GK_start;
		int Y_n = Y_k - km1;
		int yoffset = ((Y_n*OH + oh)*OW + ow)*OC;
		Ys[buf][tx][ty] = *(float4*)(deltaY + yoffset);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP][ty];
		float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];

		simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
		simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
		simdMM4(v4, a0.z, b0); simdMM4(v5, a0.z, b1);
		simdMM4(v6, a0.w, b0); simdMM4(v7, a0.w, b1);
		simdMM4(v8, a1.x, b0); simdMM4(v9, a1.x, b1);
		simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
		simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
		simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
	}

	int W1 = W0 + Wstride, W2 = W1 + Wstride, W3 = W2 + Wstride;
	int W4 = W3 + Wstride, W5 = W4 + Wstride, W6 = W5 + Wstride, W7 = W6 + Wstride;

	*(float4*)(deltaW + W0) = v0;  *(float4*)(deltaW + W0 + 4) = v1;
	*(float4*)(deltaW + W1) = v2;  *(float4*)(deltaW + W1 + 4) = v3;
	*(float4*)(deltaW + W2) = v4;  *(float4*)(deltaW + W2 + 4) = v5;
	*(float4*)(deltaW + W3) = v6;  *(float4*)(deltaW + W3 + 4) = v7;
	*(float4*)(deltaW + W4) = v8;  *(float4*)(deltaW + W4 + 4) = v9;
	*(float4*)(deltaW + W5) = v10; *(float4*)(deltaW + W5 + 4) = v11;
	*(float4*)(deltaW + W6) = v12; *(float4*)(deltaW + W6 + 4) = v13;
	*(float4*)(deltaW + W7) = v14; *(float4*)(deltaW + W7 + 4) = v15;
}

#endif


#ifndef XKERNEL10
#define XKERNEL10

#define uxkernel10(stream, LB, GZ, oc_index, ic_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GIC) \
	Xkernel10<LB, (1<<LB>>1)>\
		<<< dim3(GIC>>LB>>3, GN>>LB>>3, GZ * FH * FW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(GZ, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, ic_index)

//synchronized: 
//(IH, IW) = 4
//LB = 4: Size = 2.25, Time = 2.608 msec, Performace = 1852.7 GFlop/s
//LB = 3: Size = 2.25, Time = 3.094 msec, Performace = 1561.68 GFlop/s
//(IH, IW) = 8
//k88SK<4>: Size = 2.25, Time = 3.476 msec, Performace = 1390.06 GFlop/s
//k88SK_ohw2pow<4>: Size = 2.25, Time = 3.226 msec, Performace = 1497.78 GFlop/s
//k88SK_ohw2pow<3>: Size = 2.25, Time = 3.618 msec, Performace = 1335.5 GFlop/s
//LB = 4: Size = 2.25, Time = 3.042 msec, Performace = 1588.38 GFlop/s
//LB = 3: Size = 2.25, Time = 3.72  msec, Performace = 1298.88 GFlop/s
//(IH, IW) = 16
//k88SK<4>: Size = 2.25, Time = 3.492 msec, Performace = 1383.69 GFlop/s
//k88SK_ohw2pow<4>: Size = 2.25, Time = 3.254 msec, Performace = 1484.89 GFlop/s
//k88SK_ohw2pow<3>: Size = 2.25, Time = 3.618 msec, Performace = 1335.5 GFlop/s
//LB = 4: Size = 2.25, Time = 3.268 msec, Performace = 1478.53 GFlop/s
//LB = 3: Size = 2.25, Time = 4.016 msec, Performace = 1203.15 GFlop/s
//(IH, IW) = 32
//k88SK<4>: Size = 2.25, Time = 3.484 msec, Performace = 1386.86 GFlop/s
//k88SK_ohw2pow<4>: Size = 2.25, Time = 3.246 msec, Performace = 1488.55 GFlop/s
//k88SK_ohw2pow<3>: Size = 2.25, Time = 3.626 msec, Performace = 1332.55 GFlop/s
//LB = 4: Size = 2.25, Time = 3.366 msec, Performace = 1435.48 GFlop/s
//LB = 3: Size = 2.25, Time = 4.168 msec, Performace = 1159.27 GFlop/s
template<int LB, int STEP>
__global__ void Xkernel10(int GZ,
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int N, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int oc_index, int ic_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//======================================================================
	//prepare for gridDim.z = GZ * FH * FW
	int bz = blockIdx.z;
	const int FH_FW = FH * FW;
	int pIdx = bz / FH_FW; bz -= pIdx * FH_FW;
	int fh = bz / FW, fw = bz - fh * FW;
	int tfh = fh - oph, tfw = fw - opw;

	//prepare for GK = N * OH * OW
	int temph = (-tfh + sh - 1);
	int tempw = (-tfw + sw - 1);
	int ohs = IF_int((tfh < 0), (temph / sh), 0);
	int ows = IF_int((tfw < 0), (tempw / sw), 0);
	int ohe = (IH + temph) / sh; ohe = IF_int((ohe > OH), OH, ohe);
	int owe = (IW + tempw) / sw; owe = IF_int((owe > OW), OW, owe);
	int tOH = ohe - ohs;
	int tOW = owe - ows;
	int tOW_N = tOW * N, GK = tOH * tOW_N;
	deltaY += (ohs*OW + ows)*OC;//deltaY[0, ohs, ows, 0]
	X += ((ohs*sh)*IW + (ows*sw))*IC;//X[0, ohs*sh, ows*sw, 0]

	//prepare for GK_slice
	int GK_slice = (GK / GZ) >> 3 << 3;
	int GK_start = GK_slice * pIdx;
	int GK_end = IF_int((pIdx != (GZ - 1)), (GK_start + GK_slice), GK);
	GK_slice = GK_end - GK_start;

	//prepare for dst: deltaW_buf or deltaW 
	const int Wstride = FH * FW * IC;
	deltaW_buf += (pIdx - 1) * OC * Wstride;
	deltaW = IF_int((pIdx != 0), deltaW_buf, deltaW);//deltaW -> dst
	//======================================================================

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 3) + oc_index;
	deltaY += oc0 + ((tx >= STEP) << 2);//deltaY[0, 0, 0, toc0]

	//prepare for GM = FH * FW * IC
	int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	int W0 = ((oc0*FH + fh)*FW + fw)*IC + ic0;
	int tic0 = ic0 + ((ty >= STEP) << 2);
	X += (tfh*IW + tfw)*IC + tic0;//X[0, tfh0, tfw0, tic0]
	
	IH = IH * IW * IC;//IH -> IH * IW * IC
	IW = IW * sh * IC;//IW -> IW * sh
	IC = sw * IC;//IC -> sw * IC

	//load 4 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
	int X_k = ty - ((ty >= STEP) << LB >> 1) + GK_start;
	int oh = X_k / tOW_N, km1 = oh * tOW_N; X_k -= km1;//X_k %= tOW_N
	int ow = X_k / N, km2 = ow * N, X_n = X_k - km2; km1 += km2;
	int xoffset = X_n * IH + oh * IW + ow * IC;
	Xs[buf][ty][tx] = *(float4*)(X + xoffset);

	//load 4 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
	int Y_k = tx - ((tx >= STEP) << LB >> 1) + GK_start;
	int Y_n = Y_k - km1;//Y_n = Y_k % N
	int yoffset = ((Y_n*OH + oh)*OW + ow)*OC;
	Ys[buf][tx][ty] = *(float4*)(deltaY + yoffset);
	__syncthreads();

	//compute area-----------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK_slice << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];
			float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP][ty];

			simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
			simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
			simdMM4(v4, a0.z, b0); simdMM4(v5, a0.z, b1);
			simdMM4(v6, a0.w, b0); simdMM4(v7, a0.w, b1);
			simdMM4(v8, a1.x, b0); simdMM4(v9, a1.x, b1);
			simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
			simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
			simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
		}
		buf ^= 1;

		//load 4 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = ((ok - (ty >= STEP)) << LB >> 1) + ty + GK_start;
		int oh = X_k / tOW_N, km1 = oh * tOW_N; X_k -= km1;
		int ow = X_k / N, km2 = ow * N, X_n = X_k - km2; km1 += km2;
		int xoffset = X_n * IH + oh * IW + ow * IC;
		Xs[buf][ty][tx] = *(float4*)(X + xoffset);

		//load 4 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx + GK_start;
		int Y_n = Y_k - km1;
		int yoffset = ((Y_n*OH + oh)*OW + ow)*OC;
		Ys[buf][tx][ty] = *(float4*)(deltaY + yoffset);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];
		float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP][ty];

		simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
		simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
		simdMM4(v4, a0.z, b0); simdMM4(v5, a0.z, b1);
		simdMM4(v6, a0.w, b0); simdMM4(v7, a0.w, b1);
		simdMM4(v8, a1.x, b0); simdMM4(v9, a1.x, b1);
		simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
		simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
		simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
	}

	int W1 = W0 + Wstride, W2 = W1 + Wstride, W3 = W2 + Wstride;
	int W4 = W3 + Wstride, W5 = W4 + Wstride, W6 = W5 + Wstride, W7 = W6 + Wstride;

	*(float4*)(deltaW + W0) = v0;  *(float4*)(deltaW + W0 + 4) = v1;
	*(float4*)(deltaW + W1) = v2;  *(float4*)(deltaW + W1 + 4) = v3;
	*(float4*)(deltaW + W2) = v4;  *(float4*)(deltaW + W2 + 4) = v5;
	*(float4*)(deltaW + W3) = v6;  *(float4*)(deltaW + W3 + 4) = v7;
	*(float4*)(deltaW + W4) = v8;  *(float4*)(deltaW + W4 + 4) = v9;
	*(float4*)(deltaW + W5) = v10; *(float4*)(deltaW + W5 + 4) = v11;
	*(float4*)(deltaW + W6) = v12; *(float4*)(deltaW + W6 + 4) = v13;
	*(float4*)(deltaW + W7) = v14; *(float4*)(deltaW + W7 + 4) = v15;
}

#endif


//N is power of 2
//ph = pw = 1
//OH, OW = 2 -> tOH, tOW = 1, 2
//OH, OW = 4 -> tOH, tOW = 4, 3
#ifndef XKERNEL11
#define XKERNEL11

#define uxkernel11(stream, LB, GZ, oc_index, ic_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, LN, IC, OC, sh, sw, ph, pw, GN, GIC) \
	Xkernel11<LB, (1<<LB>>1)>\
		<<< dim3(GIC>>LB>>3, GN>>LB>>3, GZ * FH * FW), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(GZ, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW,\
			 LN, IC, OC, sh, sw, ph, pw, oc_index, ic_index)

//synchronized: 
//(IH, IW) = 4
//LB = 4: Size = 2.25, Time = 2.558 msec, Performace = 1888.91 GFlop/s
//LB = 3: Size = 2.25, Time = 3.094 msec, Performace = 1561.68 GFlop/s
//(IH, IW) = 8
//k88SK<4>: Size = 2.25, Time = 3.476 msec, Performace = 1390.06 GFlop/s
//k88SK_ohw2pow<4>: Size = 2.25, Time = 3.226 msec, Performace = 1497.78 GFlop/s
//k88SK_ohw2pow<3>: Size = 2.25, Time = 3.618 msec, Performace = 1335.5 GFlop/s
//LB = 4: Size = 2.25, Time = 3.042 msec, Performace = 1588.38 GFlop/s
//LB = 3: Size = 2.25, Time = 3.72  msec, Performace = 1298.88 GFlop/s
//(IH, IW) = 16
//k88SK<4>: Size = 2.25, Time = 3.492 msec, Performace = 1383.69 GFlop/s
//k88SK_ohw2pow<4>: Size = 2.25, Time = 3.254 msec, Performace = 1484.89 GFlop/s
//k88SK_ohw2pow<3>: Size = 2.25, Time = 3.618 msec, Performace = 1335.5 GFlop/s
//LB = 4: Size = 2.25, Time = 3.204 msec, Performace = 1508.06 GFlop/s
//LB = 3: Size = 2.25, Time = 4.016 msec, Performace = 1203.15 GFlop/s
//(IH, IW) = 32
//k88SK<4>: Size = 2.25, Time = 3.484 msec, Performace = 1386.86 GFlop/s
//k88SK_ohw2pow<4>: Size = 2.25, Time = 3.246 msec, Performace = 1488.55 GFlop/s
//k88SK_ohw2pow<3>: Size = 2.25, Time = 3.626 msec, Performace = 1332.55 GFlop/s
//LB = 4: Size = 2.25, Time = 3.308 msec, Performace = 1460.65 GFlop/s
//LB = 3: Size = 2.25, Time = 4.168 msec, Performace = 1159.27 GFlop/s
template<int LB, int STEP>
__global__ void Xkernel11(int GZ,
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int LN, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int oc_index, int ic_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//======================================================================
	//prepare for gridDim.z = GZ * FH * FW
	int bz = blockIdx.z;
	const int FH_FW = FH * FW;
	int pIdx = bz / FH_FW; bz -= pIdx * FH_FW;
	int fh = bz / FW, fw = bz - fh * FW;
	int tfh = fh - oph, tfw = fw - opw;

	//prepare for GK = N * OH * OW
	int temph = (-tfh + sh - 1);
	int tempw = (-tfw + sw - 1);
	int ohs = IF_int((tfh < 0), (temph / sh), 0);
	int ows = IF_int((tfw < 0), (tempw / sw), 0);
	int ohe = (IH + temph) / sh; ohe = IF_int((ohe > OH), OH, ohe);
	int owe = (IW + tempw) / sw; owe = IF_int((owe > OW), OW, owe);
	int tOH = ohe - ohs;
	int tOW = owe - ows;
	int tOW_N = tOW << LN, GK = tOH * tOW_N;
	deltaY += (ohs*OW + ows)*OC;//deltaY[0, ohs, ows, 0]
	X += ((ohs*sh)*IW + (ows*sw))*IC;//X[0, ohs*sh, ows*sw, 0]

	//prepare for GK_slice
	int GK_slice = (GK / GZ) >> 3 << 3;
	int GK_start = GK_slice * pIdx;
	int GK_end = IF_int((pIdx != (GZ - 1)), (GK_start + GK_slice), GK);
	GK_slice = GK_end - GK_start;

	//prepare for dst: deltaW_buf or deltaW 
	const int Wstride = FH * FW * IC;
	deltaW_buf += (pIdx - 1) * OC * Wstride;
	deltaW = IF_int((pIdx != 0), deltaW_buf, deltaW);//deltaW -> dst
	//======================================================================

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 3) + oc_index;
	deltaY += oc0 + ((tx >= STEP) << 2);//deltaY[0, 0, 0, toc0]

	//prepare for GM = FH * FW * IC
	int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	int W0 = ((oc0*FH + fh)*FW + fw)*IC + ic0;
	int tic0 = ic0 + ((ty >= STEP) << 2);
	X += (tfh*IW + tfw)*IC + tic0;//X[0, tfh0, tfw0, tic0]

	IH = IH * IW * IC;//IH -> IH * IW * IC
	IW = IW * sh * IC;//IW -> IW * sh
	IC = sw * IC;//IC -> sw * IC
	const int N_m1 = (1 << LN) - 1;

	//load 4 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
	int X_k = ty - ((ty >= STEP) << LB >> 1) + GK_start;
	int oh = X_k / tOW_N; X_k -= oh * tOW_N;
	int ow = X_k >> LN, X_n = X_k & N_m1;
	int xoffset = X_n * IH + oh * IW + ow * IC;
	Xs[buf][ty][tx] = *(float4*)(X + xoffset);

	//load 4 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
	int Y_k = tx - ((tx >= STEP) << LB >> 1) + GK_start;
	int Y_n = Y_k & N_m1;
	int yoffset = ((Y_n*OH + oh)*OW + ow)*OC;
	Ys[buf][tx][ty] = *(float4*)(deltaY + yoffset);
	__syncthreads();

	//compute area-----------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK_slice << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP][ty];
			float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];

			simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
			simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
			simdMM4(v4, a0.z, b0); simdMM4(v5, a0.z, b1);
			simdMM4(v6, a0.w, b0); simdMM4(v7, a0.w, b1);
			simdMM4(v8, a1.x, b0); simdMM4(v9, a1.x, b1);
			simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
			simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
			simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
		}
		buf ^= 1;

		//load 4 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = ((ok - (ty >= STEP)) << LB >> 1) + ty + GK_start;
		int oh = X_k / tOW_N; X_k -= oh * tOW_N;
		int ow = X_k >> LN, X_n = X_k & N_m1;
		int xoffset = X_n * IH + oh * IW + ow * IC;
		Xs[buf][ty][tx] = *(float4*)(X + xoffset);

		//load 4 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx + GK_start;
		int Y_n = Y_k & N_m1;
		int yoffset = ((Y_n*OH + oh)*OW + ow)*OC;
		Ys[buf][tx][ty] = *(float4*)(deltaY + yoffset);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP][ty];
		float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];

		simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
		simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
		simdMM4(v4, a0.z, b0); simdMM4(v5, a0.z, b1);
		simdMM4(v6, a0.w, b0); simdMM4(v7, a0.w, b1);
		simdMM4(v8, a1.x, b0); simdMM4(v9, a1.x, b1);
		simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
		simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
		simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
	}

	int W1 = W0 + Wstride, W2 = W1 + Wstride, W3 = W2 + Wstride;
	int W4 = W3 + Wstride, W5 = W4 + Wstride, W6 = W5 + Wstride, W7 = W6 + Wstride;

	*(float4*)(deltaW + W0) = v0;  *(float4*)(deltaW + W0 + 4) = v1;
	*(float4*)(deltaW + W1) = v2;  *(float4*)(deltaW + W1 + 4) = v3;
	*(float4*)(deltaW + W2) = v4;  *(float4*)(deltaW + W2 + 4) = v5;
	*(float4*)(deltaW + W3) = v6;  *(float4*)(deltaW + W3 + 4) = v7;
	*(float4*)(deltaW + W4) = v8;  *(float4*)(deltaW + W4 + 4) = v9;
	*(float4*)(deltaW + W5) = v10; *(float4*)(deltaW + W5 + 4) = v11;
	*(float4*)(deltaW + W6) = v12; *(float4*)(deltaW + W6 + 4) = v13;
	*(float4*)(deltaW + W7) = v14; *(float4*)(deltaW + W7 + 4) = v15;
}

#endif

