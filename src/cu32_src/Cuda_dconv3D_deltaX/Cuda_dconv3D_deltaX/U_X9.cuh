


#ifndef XKERNEL1
#define XKERNEL1

#define xkernel1(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	BXkernel1<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, ph,pw, ic_index,j_index)

//LB = 4: Size = 1.125, Time = 1.894 msec, Performace = 1275.56 GFlop/s
//LB = 4: Size = 1, Time = 1.558 msec, Performace = 1378.36 GFlop/s
//LB = 3: Size = 1, Time = 1.876 msec, Performace = 1144.71 GFlop/s
template<int LB, int STEP>
__global__ void BXkernel1(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__      W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int ph, int pw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	W += ic0 + ((ty >= STEP) << 2);//W[0, 0, 0, tic0]

	//prepare for GM = N * IH * IW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	int tj1 = tj0 + 1, tj2 = tj0 + 2, tj3 = tj0 + 3;
	const int IH_IW = IH * IW;
	get_n_ih_iw(tj0, tn0, tih0, tiw0);
	get_n_ih_iw(tj1, tn1, tih1, tiw1);
	get_n_ih_iw(tj2, tn2, tih2, tiw2);
	get_n_ih_iw(tj3, tn3, tih3, tiw3);
	const int oph = FH - ph - 1, opw = FW - pw - 1;
	tih0 = tih0 - oph, tiw0 = tiw0 - opw;
	tih1 = tih1 - oph, tiw1 = tiw1 - opw;
	tih2 = tih2 - oph, tiw2 = tiw2 - opw;
	tih3 = tih3 - oph, tiw3 = tiw3 - opw;
	const int Y0 = ((tn0*OH + tih0)*OW + tiw0)*OC;
	const int Y1 = ((tn1*OH + tih1)*OW + tiw1)*OC;
	const int Y2 = ((tn2*OH + tih2)*OW + tiw2)*OC;
	const int Y3 = ((tn3*OH + tih3)*OW + tiw3)*OC;

	//prepare for GK = FH * FW * OC
	const int FH_FW = FH * FW, GK = FH_FW * OC;

	//load 4 elements from deltaY[N, OH, OW, OC]: deltaYp[N, OH=OHp, OW=OWp, OC]
	int Y_k = tx - ((tx >= STEP) << LB >> 1);
	int dY_fh, dY_fw, dY_oc; get_dY_oc_fh_fw(Y_k, dY_oc, dY_fh, dY_fw);
	int yoffset = ((dY_fh*OW) + dY_fw)*OC + dY_oc;
	Ys[buf][tx][ty] = S1_SaveYs4(deltaY, dY_fh, dY_fw, yoffset, OH, OW,
		Y0, tih0, tiw0,
		Y1, tih1, tiw1,
		Y2, tih2, tiw2,
		Y3, tih3, tiw3);

	//load 4 elements from W[OC, FH, FW, IC]: Wr[IC, FH(r), FW(r), OC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);
	int W_oc, Wr_fh_fw; get_W_oc_fh_fw(W_k, W_oc, Wr_fh_fw);
	int woffset = (W_oc*FH_FW + Wr_fh_fw)*IC;
	Ws[buf][ty][tx] = *(float4*)(W + woffset);
	__syncthreads();

	//compute area----------------------------------------------------
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
			float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];
			float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];

			//transposed compute core: (W * dY)^T
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


		//load 4 elements from deltaY[N, OH, OW, OC]: deltaYp[N, OH=OHp, OW=OWp, OC]
		int dY_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int dY_fh, dY_fw, dY_oc; get_dY_oc_fh_fw(dY_k, dY_oc, dY_fh, dY_fw);
		int yoffset = ((dY_fh*OW) + dY_fw)*OC + dY_oc;
		Ys[buf][tx][ty] = S1_SaveYs4(deltaY, dY_fh, dY_fw, yoffset, OH, OW,
			Y0, tih0, tiw0,
			Y1, tih1, tiw1,
			Y2, tih2, tiw2,
			Y3, tih3, tiw3);

		//load 4 elements from W[OC, FH, FW, IC]: Wr[IC, FH(r), FW(r), OC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int W_oc, Wr_fh_fw; get_W_oc_fh_fw(W_k, W_oc, Wr_fh_fw);
		int woffset = (W_oc*FH_FW + Wr_fh_fw)*IC;
		Ws[buf][ty][tx] = *(float4*)(W + woffset);
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

	j0 = j0 * IC + ic0;//j0 = ((n * OH + oh) * OW + ow) * IC + ic
	int j1 = j0 + IC, j2 = j1 + IC, j3 = j2 + IC;
	int j4 = j3 + IC, j5 = j4 + IC, j6 = j5 + IC, j7 = j6 + IC;

	*(float4*)(deltaX + j0) = v0; *(float4*)(deltaX + j0 + 4) = v1;
	*(float4*)(deltaX + j1) = v2; *(float4*)(deltaX + j1 + 4) = v3;
	*(float4*)(deltaX + j2) = v4; *(float4*)(deltaX + j2 + 4) = v5;
	*(float4*)(deltaX + j3) = v6; *(float4*)(deltaX + j3 + 4) = v7;
	*(float4*)(deltaX + j4) = v8; *(float4*)(deltaX + j4 + 4) = v9;
	*(float4*)(deltaX + j5) = v10; *(float4*)(deltaX + j5 + 4) = v11;
	*(float4*)(deltaX + j6) = v12; *(float4*)(deltaX + j6 + 4) = v13;
	*(float4*)(deltaX + j7) = v14; *(float4*)(deltaX + j7 + 4) = v15;
}

#endif


#ifndef XKERNEL2
#define XKERNEL2

#define xkernel2(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	BXkernel2<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, ph,pw, ic_index,j_index)

//LB = 4: Size = 1.125, Time = 1.894 msec, Performace = 1275.56 GFlop/s
//LB = 4: Size = 1, Time = 1.558 msec, Performace = 1378.36 GFlop/s
//LB = 3: Size = 1, Time = 1.876 msec, Performace = 1144.71 GFlop/s
template<int LB, int STEP>
__global__ void BXkernel2(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__      W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int ph, int pw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	W += ic0 + ((ty >= STEP) << 2);//W[0, 0, 0, tic0]

	//prepare for GM = N * IH * IW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	int tj1 = tj0 + 1, tj2 = tj0 + 2, tj3 = tj0 + 3;
	const int IH_IW = IH * IW;
	get_n_ih_iw(tj0, tn0, tih0, tiw0);
	get_n_ih_iw(tj1, tn1, tih1, tiw1);
	get_n_ih_iw(tj2, tn2, tih2, tiw2);
	get_n_ih_iw(tj3, tn3, tih3, tiw3);
	const int oph = FH - ph - 1, opw = FW - pw - 1;
	tih0 = tih0 - oph, tiw0 = tiw0 - opw;
	tih1 = tih1 - oph, tiw1 = tiw1 - opw;
	tih2 = tih2 - oph, tiw2 = tiw2 - opw;
	tih3 = tih3 - oph, tiw3 = tiw3 - opw;
	const int Y0 = ((tn0*OH + tih0)*OW + tiw0)*OC;
	const int Y1 = ((tn1*OH + tih1)*OW + tiw1)*OC;
	const int Y2 = ((tn2*OH + tih2)*OW + tiw2)*OC;
	const int Y3 = ((tn3*OH + tih3)*OW + tiw3)*OC;

	//prepare for GK = FH * FW * OC
	const int FH_FW = FH * FW, GK = FH_FW * OC;

	//load 4 elements from deltaY[N, OH, OW, OC]: deltaYp[N, OH=OHp, OW=OWp, OC]
	int Y_k = tx - ((tx >= STEP) << LB >> 1);
	int Y_oc = Y_k / FH_FW; Y_k -= Y_oc * FH_FW; 
	int Y_fh = Y_k / FW, Y_fw = Y_k - Y_fh * FW;
	int yoffset = ((Y_fh*OW) + Y_fw)*OC + Y_oc;
	Ys[buf][tx][ty] = S1_SaveYs4(deltaY, Y_fh, Y_fw, yoffset, OH, OW,
		Y0, tih0, tiw0,
		Y1, tih1, tiw1,
		Y2, tih2, tiw2,
		Y3, tih3, tiw3);

	//load 4 elements from W[OC, FH, FW, IC]: Wr[IC, FH(r), FW(r), OC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);
	int W_oc = W_k / FH_FW; W_k -= W_oc * FH_FW;
	int W_fh = W_k / FW, W_fw = W_k - W_fh * FW;
	W_fh = FH - 1 - W_fh;
	W_fw = FW - 1 - W_fw;
	int woffset = ((W_oc*FH + W_fh)*FW + W_fw)*IC;
	Ws[buf][ty][tx] = *(float4*)(W + woffset);
	__syncthreads();

	//compute area----------------------------------------------------
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
			float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];
			float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];

			//transposed compute core: (W * dY)^T
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


		//load 4 elements from deltaY[N, OH, OW, OC]: deltaYp[N, OH=OHp, OW=OWp, OC]
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int Y_oc = Y_k / FH_FW; Y_k -= Y_oc * FH_FW;
		int Y_fh = Y_k / FW, Y_fw = Y_k - Y_fh * FW;
		int yoffset = ((Y_fh*OW) + Y_fw)*OC + Y_oc;
		Ys[buf][tx][ty] = S1_SaveYs4(deltaY, Y_fh, Y_fw, yoffset, OH, OW,
			Y0, tih0, tiw0,
			Y1, tih1, tiw1,
			Y2, tih2, tiw2,
			Y3, tih3, tiw3);

		//load 4 elements from W[OC, FH, FW, IC]: Wr[IC, FH(r), FW(r), OC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int W_oc = W_k / FH_FW; W_k -= W_oc * FH_FW;
		int W_fh = W_k / FW, W_fw = W_k - W_fh * FW;
		W_fh = FH - 1 - W_fh;
		W_fw = FW - 1 - W_fw;
		int woffset = ((W_oc*FH + W_fh)*FW + W_fw)*IC;
		Ws[buf][ty][tx] = *(float4*)(W + woffset);
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

	j0 = j0 * IC + ic0;//j0 = ((n * OH + oh) * OW + ow) * IC + ic
	int j1 = j0 + IC, j2 = j1 + IC, j3 = j2 + IC;
	int j4 = j3 + IC, j5 = j4 + IC, j6 = j5 + IC, j7 = j6 + IC;

	*(float4*)(deltaX + j0) = v0; *(float4*)(deltaX + j0 + 4) = v1;
	*(float4*)(deltaX + j1) = v2; *(float4*)(deltaX + j1 + 4) = v3;
	*(float4*)(deltaX + j2) = v4; *(float4*)(deltaX + j2 + 4) = v5;
	*(float4*)(deltaX + j3) = v6; *(float4*)(deltaX + j3 + 4) = v7;
	*(float4*)(deltaX + j4) = v8; *(float4*)(deltaX + j4 + 4) = v9;
	*(float4*)(deltaX + j5) = v10; *(float4*)(deltaX + j5 + 4) = v11;
	*(float4*)(deltaX + j6) = v12; *(float4*)(deltaX + j6 + 4) = v13;
	*(float4*)(deltaX + j7) = v14; *(float4*)(deltaX + j7 + 4) = v15;
}

#endif


#ifndef XKERNEL3
#define XKERNEL3

#define xkernel3(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	BXkernel3<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, ph,pw, ic_index,j_index)

//LB = 4: Size = 1.125, Time = 1.894 msec, Performace = 1275.56 GFlop/s
//LB = 4: Size = 1.125, Time = 1.726 msec, Performace = 1399.72 GFlop/s
//LB = 3: Size = 1, Time = 1.876 msec, Performace = 1144.71 GFlop/s
template<int LB, int STEP>
__global__ void BXkernel3(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__      W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int ph, int pw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	W += ic0 + ((ty >= STEP) << 2);//W[0, 0, 0, tic0]
	W += ((FH - 1)*FW + (FW - 1))*IC;

	//prepare for GM = N * IH * IW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	int tj1 = tj0 + 1, tj2 = tj0 + 2, tj3 = tj0 + 3;
	const int IH_IW = IH * IW;
	get_n_ih_iw(tj0, tn0, tih0, tiw0);
	get_n_ih_iw(tj1, tn1, tih1, tiw1);
	get_n_ih_iw(tj2, tn2, tih2, tiw2);
	get_n_ih_iw(tj3, tn3, tih3, tiw3);
	const int oph = FH - ph - 1, opw = FW - pw - 1;
	tih0 = tih0 - oph, tiw0 = tiw0 - opw;
	tih1 = tih1 - oph, tiw1 = tiw1 - opw;
	tih2 = tih2 - oph, tiw2 = tiw2 - opw;
	tih3 = tih3 - oph, tiw3 = tiw3 - opw;
	const int Y0 = ((tn0*OH + tih0)*OW + tiw0)*OC;
	const int Y1 = ((tn1*OH + tih1)*OW + tiw1)*OC;
	const int Y2 = ((tn2*OH + tih2)*OW + tiw2)*OC;
	const int Y3 = ((tn3*OH + tih3)*OW + tiw3)*OC;

	//prepare for GK = FH * FW * OC
	const int FH_FW = FH * FW, GK = FH_FW * OC;

	//load 4 elements from deltaY[N, OH, OW, OC]: deltaYp[N, OH=OHp, OW=OWp, OC]
	int Y_k = tx - ((tx >= STEP) << LB >> 1);
	int Y_oc = Y_k / FH_FW; Y_k -= Y_oc * FH_FW;
	int Y_fh = Y_k / FW, Y_fw = Y_k - Y_fh * FW;
	int yoffset = ((Y_fh*OW) + Y_fw)*OC + Y_oc;
	Ys[buf][tx][ty] = S1_SaveYs4(deltaY, Y_fh, Y_fw, yoffset, OH, OW,
		Y0, tih0, tiw0,
		Y1, tih1, tiw1,
		Y2, tih2, tiw2,
		Y3, tih3, tiw3);

	//load 4 elements from W[OC, FH, FW, IC]: Wr[IC, FH(r), FW(r), OC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);
	int W_oc = W_k / FH_FW; W_k -= W_oc * FH_FW;
	int W_fh = W_k / FW, W_fw = W_k - W_fh * FW;
	int woffset = ((W_oc*FH - W_fh)*FW - W_fw)*IC;
	Ws[buf][ty][tx] = *(float4*)(W + woffset);
	__syncthreads();

	//compute area----------------------------------------------------
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
			float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];
			float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];

			//transposed compute core: (W * dY)^T
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


		//load 4 elements from deltaY[N, OH, OW, OC]: deltaYp[N, OH=OHp, OW=OWp, OC]
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int Y_oc = Y_k / FH_FW; Y_k -= Y_oc * FH_FW;
		int Y_fh = Y_k / FW, Y_fw = Y_k - Y_fh * FW;
		int yoffset = ((Y_fh*OW) + Y_fw)*OC + Y_oc;
		Ys[buf][tx][ty] = S1_SaveYs4(deltaY, Y_fh, Y_fw, yoffset, OH, OW,
			Y0, tih0, tiw0,
			Y1, tih1, tiw1,
			Y2, tih2, tiw2,
			Y3, tih3, tiw3);

		//load 4 elements from W[OC, FH, FW, IC]: Wr[IC, FH(r), FW(r), OC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int W_oc = W_k / FH_FW; W_k -= W_oc * FH_FW;
		int W_fh = W_k / FW, W_fw = W_k - W_fh * FW;
		int woffset = ((W_oc*FH - W_fh)*FW - W_fw)*IC;
		Ws[buf][ty][tx] = *(float4*)(W + woffset);
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

	j0 = j0 * IC + ic0;//j0 = ((n * OH + oh) * OW + ow) * IC + ic
	int j1 = j0 + IC, j2 = j1 + IC, j3 = j2 + IC;
	int j4 = j3 + IC, j5 = j4 + IC, j6 = j5 + IC, j7 = j6 + IC;

	*(float4*)(deltaX + j0) = v0; *(float4*)(deltaX + j0 + 4) = v1;
	*(float4*)(deltaX + j1) = v2; *(float4*)(deltaX + j1 + 4) = v3;
	*(float4*)(deltaX + j2) = v4; *(float4*)(deltaX + j2 + 4) = v5;
	*(float4*)(deltaX + j3) = v6; *(float4*)(deltaX + j3 + 4) = v7;
	*(float4*)(deltaX + j4) = v8; *(float4*)(deltaX + j4 + 4) = v9;
	*(float4*)(deltaX + j5) = v10; *(float4*)(deltaX + j5 + 4) = v11;
	*(float4*)(deltaX + j6) = v12; *(float4*)(deltaX + j6 + 4) = v13;
	*(float4*)(deltaX + j7) = v14; *(float4*)(deltaX + j7 + 4) = v15;
}

#endif


#ifndef XKERNEL4
#define XKERNEL4

#define xkernel4(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	BXkernel4<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, ph,pw, ic_index,j_index)

//LB = 4: Size = 1.125, Time = 1.894 msec, Performace = 1275.56 GFlop/s
//LB = 4: Size = 1.125, Time = 1.726 msec, Performace = 1399.72 GFlop/s
//LB = 3: Size = 1, Time = 1.876 msec, Performace = 1144.71 GFlop/s
template<int LB, int STEP>
__global__ void BXkernel4(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__      W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int ph, int pw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	W += ic0 + ((ty >= STEP) << 2);//W[0, 0, 0, tic0]
	W += ((FH - 1)*FW + (FW - 1))*IC;//W[0, FH - 1, FW - 1, tic0]

	//prepare for GM = N * IH * IW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	int tj1 = tj0 + 1, tj2 = tj0 + 2, tj3 = tj0 + 3;
	const int IH_IW = IH * IW;
	get_n_ih_iw(tj0, tn0, tih0, tiw0);
	get_n_ih_iw(tj1, tn1, tih1, tiw1);
	get_n_ih_iw(tj2, tn2, tih2, tiw2);
	get_n_ih_iw(tj3, tn3, tih3, tiw3);
	const int oph = FH - ph - 1, opw = FW - pw - 1;
	tih0 = tih0 - oph, tiw0 = tiw0 - opw;
	tih1 = tih1 - oph, tiw1 = tiw1 - opw;
	tih2 = tih2 - oph, tiw2 = tiw2 - opw;
	tih3 = tih3 - oph, tiw3 = tiw3 - opw;
	const int Y0 = ((tn0*OH + tih0)*OW + tiw0)*OC;
	const int Y1 = ((tn1*OH + tih1)*OW + tiw1)*OC;
	const int Y2 = ((tn2*OH + tih2)*OW + tiw2)*OC;
	const int Y3 = ((tn3*OH + tih3)*OW + tiw3)*OC;

	//prepare for GK = FH * FW * OC
	const int FW_OC = FW * OC, GK = FH * FW_OC;

	//load 4 elements from deltaY[N, OH, OW, OC]: deltaYp[N, OH=OHp, OW=OWp, OC]
	int Y_k = tx - ((tx >= STEP) << LB >> 1);
	int Y_fh = Y_k / FW_OC; Y_k -= Y_fh * FW_OC;
	int Y_fw = Y_k / OC, Y_oc = Y_k - Y_fw * OC;
	int yoffset = ((Y_fh*OW) + Y_fw)*OC + Y_oc;
	Ys[buf][tx][ty] = S1_SaveYs4(deltaY, Y_fh, Y_fw, yoffset, OH, OW,
		Y0, tih0, tiw0,
		Y1, tih1, tiw1,
		Y2, tih2, tiw2,
		Y3, tih3, tiw3);

	//load 4 elements from W[OC, FH, FW, IC]: Wr[IC, FH(r), FW(r), OC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);
	int W_fh = W_k / FW_OC; W_k -= W_fh * FW_OC;
	int W_fw = W_k / OC, W_oc = W_k - W_fw * OC;
	int woffset = ((W_oc*FH - W_fh)*FW - W_fw)*IC;
	Ws[buf][ty][tx] = *(float4*)(W + woffset);
	__syncthreads();

	//compute area----------------------------------------------------
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
			float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];
			float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];

			//transposed compute core: (W * dY)^T
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


		//load 4 elements from deltaY[N, OH, OW, OC]: deltaYp[N, OH=OHp, OW=OWp, OC]
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int Y_fh = Y_k / FW_OC; Y_k -= Y_fh * FW_OC;
		int Y_fw = Y_k / OC, Y_oc = Y_k - Y_fw * OC;
		int yoffset = ((Y_fh*OW) + Y_fw)*OC + Y_oc;
		Ys[buf][tx][ty] = S1_SaveYs4(deltaY, Y_fh, Y_fw, yoffset, OH, OW,
			Y0, tih0, tiw0,
			Y1, tih1, tiw1,
			Y2, tih2, tiw2,
			Y3, tih3, tiw3);

		//load 4 elements from W[OC, FH, FW, IC]: Wr[IC, FH(r), FW(r), OC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int W_fh = W_k / FW_OC; W_k -= W_fh * FW_OC;
		int W_fw = W_k / OC, W_oc = W_k - W_fw * OC;
		int woffset = ((W_oc*FH - W_fh)*FW - W_fw)*IC;
		Ws[buf][ty][tx] = *(float4*)(W + woffset);
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

	j0 = j0 * IC + ic0;//j0 = ((n * OH + oh) * OW + ow) * IC + ic
	int j1 = j0 + IC, j2 = j1 + IC, j3 = j2 + IC;
	int j4 = j3 + IC, j5 = j4 + IC, j6 = j5 + IC, j7 = j6 + IC;

	*(float4*)(deltaX + j0) = v0; *(float4*)(deltaX + j0 + 4) = v1;
	*(float4*)(deltaX + j1) = v2; *(float4*)(deltaX + j1 + 4) = v3;
	*(float4*)(deltaX + j2) = v4; *(float4*)(deltaX + j2 + 4) = v5;
	*(float4*)(deltaX + j3) = v6; *(float4*)(deltaX + j3 + 4) = v7;
	*(float4*)(deltaX + j4) = v8; *(float4*)(deltaX + j4 + 4) = v9;
	*(float4*)(deltaX + j5) = v10; *(float4*)(deltaX + j5 + 4) = v11;
	*(float4*)(deltaX + j6) = v12; *(float4*)(deltaX + j6 + 4) = v13;
	*(float4*)(deltaX + j7) = v14; *(float4*)(deltaX + j7 + 4) = v15;
}

#endif


#ifndef XKERNEL5
#define XKERNEL5

#define xkernel5(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	BXkernel5<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, ph,pw, ic_index,j_index)

//LB = 4: Size = 1.125, Time = 1.894 msec, Performace = 1275.56 GFlop/s
//LB = 4: Size = 1.125, Time = 1.726 msec, Performace = 1399.72 GFlop/s
//LB = 3: Size = 1, Time = 1.876 msec, Performace = 1144.71 GFlop/s
template<int LB, int STEP>
__global__ void BXkernel5(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__      W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int ph, int pw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	W += ic0 + ((ty >= STEP) << 2);//W[0, 0, 0, tic0]
	W += ((FH - 1)*FW + (FW - 1))*IC;//W[0, FH - 1, FW - 1, tic0]

	//prepare for GM = N * IH * IW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	int tj1 = tj0 + 1, tj2 = tj0 + 2, tj3 = tj0 + 3;
	const int IH_IW = IH * IW;
	get_n_ih_iw(tj0, tn0, tih0, tiw0);
	get_n_ih_iw(tj1, tn1, tih1, tiw1);
	get_n_ih_iw(tj2, tn2, tih2, tiw2);
	get_n_ih_iw(tj3, tn3, tih3, tiw3);
	const int oph = FH - ph - 1, opw = FW - pw - 1;
	tih0 = tih0 - oph, tiw0 = tiw0 - opw;
	tih1 = tih1 - oph, tiw1 = tiw1 - opw;
	tih2 = tih2 - oph, tiw2 = tiw2 - opw;
	tih3 = tih3 - oph, tiw3 = tiw3 - opw;
	const int Y0 = ((tn0*OH + tih0)*OW + tiw0)*OC;
	const int Y1 = ((tn1*OH + tih1)*OW + tiw1)*OC;
	const int Y2 = ((tn2*OH + tih2)*OW + tiw2)*OC;
	const int Y3 = ((tn3*OH + tih3)*OW + tiw3)*OC;

	//prepare for GK = FH * FW * OC
	const int FW_OC = FW * OC, GK = FH * FW_OC;

	//load 4 elements from deltaY[N, OH, OW, OC]: deltaYp[N, OH=OHp, OW=OWp, OC]
	int Y_k = tx - ((tx >= STEP) << LB >> 1);
	int Y_fh = Y_k / FW_OC; Y_k -= Y_fh * FW_OC;
	int Y_fw = Y_k / OC;
	int OW_OC = OW * OC, yoffset = Y_fh * OW_OC + Y_k;
	Ys[buf][tx][ty] = S1_SaveYs4(deltaY, Y_fh, Y_fw, yoffset, OH, OW,
		Y0, tih0, tiw0,
		Y1, tih1, tiw1,
		Y2, tih2, tiw2,
		Y3, tih3, tiw3);

	//load 4 elements from W[OC, FH, FW, IC]: Wr[IC, FH(r), FW(r), OC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);
	int W_fh = W_k / FW_OC; W_k -= W_fh * FW_OC;
	int W_fw = W_k / OC, W_oc = W_k - W_fw * OC;
	int woffset = ((W_oc*FH - W_fh)*FW - W_fw)*IC;
	Ws[buf][ty][tx] = *(float4*)(W + woffset);
	__syncthreads();

	//compute area----------------------------------------------------
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
			float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];
			float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];

			//transposed compute core: (W * dY)^T
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


		//load 4 elements from deltaY[N, OH, OW, OC]: deltaYp[N, OH=OHp, OW=OWp, OC]
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int Y_fh = Y_k / FW_OC; Y_k -= Y_fh * FW_OC;
		int Y_fw = Y_k / OC;
		int yoffset = Y_fh * OW_OC + Y_k;
		Ys[buf][tx][ty] = S1_SaveYs4(deltaY, Y_fh, Y_fw, yoffset, OH, OW,
			Y0, tih0, tiw0,
			Y1, tih1, tiw1,
			Y2, tih2, tiw2,
			Y3, tih3, tiw3);

		//load 4 elements from W[OC, FH, FW, IC]: Wr[IC, FH(r), FW(r), OC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int W_fh = W_k / FW_OC; W_k -= W_fh * FW_OC;
		int W_fw = W_k / OC, W_oc = W_k - W_fw * OC;
		int woffset = ((W_oc*FH - W_fh)*FW - W_fw)*IC;
		Ws[buf][ty][tx] = *(float4*)(W + woffset);
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

	j0 = j0 * IC + ic0;//j0 = ((n * OH + oh) * OW + ow) * IC + ic
	int j1 = j0 + IC, j2 = j1 + IC, j3 = j2 + IC;
	int j4 = j3 + IC, j5 = j4 + IC, j6 = j5 + IC, j7 = j6 + IC;

	*(float4*)(deltaX + j0) = v0; *(float4*)(deltaX + j0 + 4) = v1;
	*(float4*)(deltaX + j1) = v2; *(float4*)(deltaX + j1 + 4) = v3;
	*(float4*)(deltaX + j2) = v4; *(float4*)(deltaX + j2 + 4) = v5;
	*(float4*)(deltaX + j3) = v6; *(float4*)(deltaX + j3 + 4) = v7;
	*(float4*)(deltaX + j4) = v8; *(float4*)(deltaX + j4 + 4) = v9;
	*(float4*)(deltaX + j5) = v10; *(float4*)(deltaX + j5 + 4) = v11;
	*(float4*)(deltaX + j6) = v12; *(float4*)(deltaX + j6 + 4) = v13;
	*(float4*)(deltaX + j7) = v14; *(float4*)(deltaX + j7 + 4) = v15;
}

#endif


#ifndef XKERNEL6
#define XKERNEL6

#define xkernel6(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	BXkernel6<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, ph,pw, ic_index,j_index)

//LB = 4: Size = 1.125, Time = 1.712 msec, Performace = 1411.17 GFlop/s
template<int LB, int STEP>
__global__ void BXkernel6(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__      W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int ph, int pw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	W += ic0 + ((ty >= STEP) << 2);//W[0, 0, 0, tic0]
	W += ((FH - 1)*FW + (FW - 1))*IC;//W[0, FH - 1, FW - 1, tic0]

	//prepare for GM = N * IH * IW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	int tj1 = tj0 + 1, tj2 = tj0 + 2, tj3 = tj0 + 3;
	const int IH_IW = IH * IW;
	get_n_ih_iw(tj0, tn0, tih0, tiw0);
	get_n_ih_iw(tj1, tn1, tih1, tiw1);
	get_n_ih_iw(tj2, tn2, tih2, tiw2);
	get_n_ih_iw(tj3, tn3, tih3, tiw3);
	const int oph = FH - ph - 1, opw = FW - pw - 1;
	tih0 = tih0 - oph, tiw0 = tiw0 - opw;
	tih1 = tih1 - oph, tiw1 = tiw1 - opw;
	tih2 = tih2 - oph, tiw2 = tiw2 - opw;
	tih3 = tih3 - oph, tiw3 = tiw3 - opw;
	const int Y0 = ((tn0*OH + tih0)*OW + tiw0)*OC;
	const int Y1 = ((tn1*OH + tih1)*OW + tiw1)*OC;
	const int Y2 = ((tn2*OH + tih2)*OW + tiw2)*OC;
	const int Y3 = ((tn3*OH + tih3)*OW + tiw3)*OC;

	//prepare for GK = FH * FW * OC
	const int FW_OC = FW * OC, GK = FH * FW_OC;

	//===========================================================================
	//Y_k = (ok * STEP) + tx - ((tx >= STEP) << LB >> 1);
	//W_k = (ok * STEP) + ty - ((ty >= STEP) << LB >> 1);
	//Y_k = ok * STEP + Ux
	//W_k = ok * STEP + Uy
	//Y_fh = Y_k / FW_OC = (ok * STEP + Uy) / (FW * OC)
	//W_fh = W_k / FW_OC = (ok * STEP + Ux) / (FW * OC)
	//when: OC % STEP = 0, FW_OC = STEP * z
	//Y_fh = Y_k / FW_OC = (ok * STEP + Ux) / (STEP * z) = (ok + (Ux/STEP)) / z
	//W_fh = W_k / FW_OC = (ok * STEP + Uy) / (STEP * z) = (ok + (Uy/STEP)) / z
	//As: Uy, Ux belongs < STEP
	//So: Y_fh = W_fh
	//when: LB = 4, OC % 8 == 0
	//when: LB = 3, OC % 4 == 0
	//===========================================================================

	//load 4 elements from deltaY[N, OH, OW, OC]: deltaYp[N, OH=OHp, OW=OWp, OC]
	int Y_k = tx - ((tx >= STEP) << LB >> 1);
	int fh = Y_k / FW_OC; Y_k -= fh * FW_OC;
	int Y_fw = Y_k / OC;
	int OW_OC = OW * OC, yoffset = fh * OW_OC + Y_k;
	Ys[buf][tx][ty] = S1_SaveYs4(deltaY, fh, Y_fw, yoffset, OH, OW,
		Y0, tih0, tiw0,
		Y1, tih1, tiw1,
		Y2, tih2, tiw2,
		Y3, tih3, tiw3);

	//load 4 elements from W[OC, FH, FW, IC]: Wr[IC, FH(r), FW(r), OC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);
	W_k -= fh * FW_OC;
	int W_fw = W_k / OC, W_oc = W_k - W_fw * OC;
	int woffset = ((W_oc*FH - fh)*FW - W_fw)*IC;
	Ws[buf][ty][tx] = *(float4*)(W + woffset);
	__syncthreads();

	//compute area----------------------------------------------------
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
			float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];
			float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];

			//transposed compute core: (W * dY)^T
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


		//load 4 elements from deltaY[N, OH, OW, OC]: deltaYp[N, OH=OHp, OW=OWp, OC]
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int fh = Y_k / FW_OC; Y_k -= fh * FW_OC;
		int Y_fw = Y_k / OC;
		int yoffset = fh * OW_OC + Y_k;
		Ys[buf][tx][ty] = S1_SaveYs4(deltaY, fh, Y_fw, yoffset, OH, OW,
			Y0, tih0, tiw0,
			Y1, tih1, tiw1,
			Y2, tih2, tiw2,
			Y3, tih3, tiw3);

		//load 4 elements from W[OC, FH, FW, IC]: Wr[IC, FH(r), FW(r), OC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		W_k -= fh * FW_OC;
		int W_fw = W_k / OC, W_oc = W_k - W_fw * OC;
		int woffset = ((W_oc*FH - fh)*FW - W_fw)*IC;
		Ws[buf][ty][tx] = *(float4*)(W + woffset);
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

	j0 = j0 * IC + ic0;//j0 = ((n * OH + oh) * OW + ow) * IC + ic
	int j1 = j0 + IC, j2 = j1 + IC, j3 = j2 + IC;
	int j4 = j3 + IC, j5 = j4 + IC, j6 = j5 + IC, j7 = j6 + IC;

	*(float4*)(deltaX + j0) = v0; *(float4*)(deltaX + j0 + 4) = v1;
	*(float4*)(deltaX + j1) = v2; *(float4*)(deltaX + j1 + 4) = v3;
	*(float4*)(deltaX + j2) = v4; *(float4*)(deltaX + j2 + 4) = v5;
	*(float4*)(deltaX + j3) = v6; *(float4*)(deltaX + j3 + 4) = v7;
	*(float4*)(deltaX + j4) = v8; *(float4*)(deltaX + j4 + 4) = v9;
	*(float4*)(deltaX + j5) = v10; *(float4*)(deltaX + j5 + 4) = v11;
	*(float4*)(deltaX + j6) = v12; *(float4*)(deltaX + j6 + 4) = v13;
	*(float4*)(deltaX + j7) = v14; *(float4*)(deltaX + j7 + 4) = v15;
}

#endif


#ifndef XKERNEL7
#define XKERNEL7

#define xkernel7(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	BXkernel7<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, ph,pw, ic_index,j_index)

//LB = 4: Size = 1.125, Time = 1.734 msec, Performace = 1393.26 GFlop/s
template<int LB, int STEP>
__global__ void BXkernel7(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__      W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int ph, int pw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	W += ic0 + ((ty >= STEP) << 2);//W[0, 0, 0, tic0]
	W += ((FH - 1)*FW + (FW - 1))*IC;//W[0, FH - 1, FW - 1, tic0]

	//prepare for GM = N * IH * IW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	int tj1 = tj0 + 1, tj2 = tj0 + 2, tj3 = tj0 + 3;
	const int IH_IW = IH * IW;
	get_n_ih_iw(tj0, tn0, tih0, tiw0);
	get_n_ih_iw(tj1, tn1, tih1, tiw1);
	get_n_ih_iw(tj2, tn2, tih2, tiw2);
	get_n_ih_iw(tj3, tn3, tih3, tiw3);
	const int oph = FH - ph - 1, opw = FW - pw - 1;
	tih0 = tih0 - oph, tiw0 = tiw0 - opw;
	tih1 = tih1 - oph, tiw1 = tiw1 - opw;
	tih2 = tih2 - oph, tiw2 = tiw2 - opw;
	tih3 = tih3 - oph, tiw3 = tiw3 - opw;
	const int Y0 = ((tn0*OH + tih0)*OW + tiw0)*OC;
	const int Y1 = ((tn1*OH + tih1)*OW + tiw1)*OC;
	const int Y2 = ((tn2*OH + tih2)*OW + tiw2)*OC;
	const int Y3 = ((tn3*OH + tih3)*OW + tiw3)*OC;

	//prepare for GK = FH * FW * OC
	const int FW_OC = FW * OC, GK = FH * FW_OC;

	//===========================================================================
	//Y_k = (ok * STEP) + tx - ((tx >= STEP) << LB >> 1);
	//W_k = (ok * STEP) + ty - ((ty >= STEP) << LB >> 1);
	//Y_fh = W_fh = fh
	//Y_k -= fh*FW_OC 
	//W_k -= fh*FW_OC
	//when: OC % STEP == 0, OC = q*STEP
	//As: fh*FW*OC = p*STEP
	//So: 
	//Y_fw = Y_k / OC = (ok * STEP + Ux - p*STEP) / (q*STEP)
	//W_fw = W_k / OC = (ok * STEP + Uy - p*STEP) / (q*STEP)
	//As: Uy, Ux belongs < STEP
	//Y_fw = Y_k / OC = ((ok - p) + Ux/STEP) / q = (ok - p) / q
	//W_fw = W_k / OC = ((ok - p) + Uy/STEP) / q = (ok - p) / q
	//So: Y_fw = W_fw = fw
	//when: LB = 4, OC % 8 == 0
	//when: LB = 3, OC % 4 == 0
	//As: fw = (ok - p) / q
	//when: ok = 0, fw = -p/q, as: fw >= 0, So: at the start fw = 0
	//more over:
	//as:
	//Y_fh = Y_k / FW_OC = (ok * STEP + Ux) / (STEP * z) = (ok + (Ux/STEP)) / z
	//W_fh = W_k / FW_OC = (ok * STEP + Uy) / (STEP * z) = (ok + (Uy/STEP)) / z
	//fh = w_fh = ok / z
	//when: ok = 0, fh = 0
	//All in all, at the start: fh = fw = 0
	//===========================================================================

	//load 4 elements from deltaY[N, OH, OW, OC]: deltaYp[N, OH=OHp, OW=OWp, OC]
	int Y_k = tx - ((tx >= STEP) << LB >> 1);
	int fh = Y_k / FW_OC, fw = (Y_k -= fh * FW_OC) / OC;
	int OW_OC = OW * OC, yoffset = fh * OW_OC + Y_k;
	Ys[buf][tx][ty] = S1_SaveYs4(deltaY, fh, fw, yoffset, OH, OW,
		Y0, tih0, tiw0,
		Y1, tih1, tiw1,
		Y2, tih2, tiw2,
		Y3, tih3, tiw3);

	//load 4 elements from W[OC, FH, FW, IC]: Wr[IC, FH(r), FW(r), OC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);
	int W_oc = W_k - fh * FW_OC - fw * OC;
	int woffset = ((W_oc*FH - fh)*FW - fw)*IC;
	Ws[buf][ty][tx] = *(float4*)(W + woffset);
	__syncthreads();

	//compute area----------------------------------------------------
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
			float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];
			float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];

			//transposed compute core: (W * dY)^T
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

		//load 4 elements from deltaY[N, OH, OW, OC]: deltaYp[N, OH=OHp, OW=OWp, OC]
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int fh = Y_k / FW_OC, fw = (Y_k -= fh * FW_OC) / OC;
		int yoffset = fh * OW_OC + Y_k;
		Ys[buf][tx][ty] = S1_SaveYs4(deltaY, fh, fw, yoffset, OH, OW,
			Y0, tih0, tiw0,
			Y1, tih1, tiw1,
			Y2, tih2, tiw2,
			Y3, tih3, tiw3);

		//load 4 elements from W[OC, FH, FW, IC]: Wr[IC, FH(r), FW(r), OC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int W_oc = W_k - fh * FW_OC - fw * OC;
		int woffset = ((W_oc*FH - fh)*FW - fw)*IC;
		Ws[buf][ty][tx] = *(float4*)(W + woffset);
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

	j0 = j0 * IC + ic0;//j0 = ((n * OH + oh) * OW + ow) * IC + ic
	int j1 = j0 + IC, j2 = j1 + IC, j3 = j2 + IC;
	int j4 = j3 + IC, j5 = j4 + IC, j6 = j5 + IC, j7 = j6 + IC;

	*(float4*)(deltaX + j0) = v0; *(float4*)(deltaX + j0 + 4) = v1;
	*(float4*)(deltaX + j1) = v2; *(float4*)(deltaX + j1 + 4) = v3;
	*(float4*)(deltaX + j2) = v4; *(float4*)(deltaX + j2 + 4) = v5;
	*(float4*)(deltaX + j3) = v6; *(float4*)(deltaX + j3 + 4) = v7;
	*(float4*)(deltaX + j4) = v8; *(float4*)(deltaX + j4 + 4) = v9;
	*(float4*)(deltaX + j5) = v10; *(float4*)(deltaX + j5 + 4) = v11;
	*(float4*)(deltaX + j6) = v12; *(float4*)(deltaX + j6 + 4) = v13;
	*(float4*)(deltaX + j7) = v14; *(float4*)(deltaX + j7 + 4) = v15;
}

#endif


#ifndef XKERNEL8
#define XKERNEL8

#define xkernel8(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	BXkernel8<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, ph,pw, ic_index,j_index)

//LB = 4: Size = 1.125, Time = 1.714 msec, Performace = 1409.52 GFlop/s
template<int LB, int STEP>
__global__ void BXkernel8(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__      W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int ph, int pw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	W += ic0 + ((ty >= STEP) << 2);//W[0, 0, 0, tic0]
	W += ((FH - 1)*FW + (FW - 1))*IC;//W[0, FH - 1, FW - 1, tic0]

	//prepare for GM = N * IH * IW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	int tj1 = tj0 + 1, tj2 = tj0 + 2, tj3 = tj0 + 3;
	const int IH_IW = IH * IW;
	get_n_ih_iw(tj0, tn0, tih0, tiw0);
	get_n_ih_iw(tj1, tn1, tih1, tiw1);
	get_n_ih_iw(tj2, tn2, tih2, tiw2);
	get_n_ih_iw(tj3, tn3, tih3, tiw3);
	const int oph = FH - ph - 1, opw = FW - pw - 1;
	tih0 = tih0 - oph, tiw0 = tiw0 - opw;
	tih1 = tih1 - oph, tiw1 = tiw1 - opw;
	tih2 = tih2 - oph, tiw2 = tiw2 - opw;
	tih3 = tih3 - oph, tiw3 = tiw3 - opw;
	const int Y0 = ((tn0*OH + tih0)*OW + tiw0)*OC;
	const int Y1 = ((tn1*OH + tih1)*OW + tiw1)*OC;
	const int Y2 = ((tn2*OH + tih2)*OW + tiw2)*OC;
	const int Y3 = ((tn3*OH + tih3)*OW + tiw3)*OC;

	//prepare for GK = FH * FW * OC
	const int FW_OC = FW * OC, GK = FH * FW_OC;

	//start point of (fh, fw) is (0, 0)
	//load 4 elements from deltaY[N, OH, OW, OC]
	int Y_k = tx - ((tx >= STEP) << LB >> 1);
	int OW_OC = OW * OC, yoffset = Y_k;
	Ys[buf][tx][ty] = S1_SaveYs4(deltaY, 0, 0, yoffset, OH, OW,
		Y0, tih0, tiw0,
		Y1, tih1, tiw1,
		Y2, tih2, tiw2,
		Y3, tih3, tiw3);

	//load 4 elements from W[OC, FH, FW, IC]
	int W_oc = ty - ((ty >= STEP) << LB >> 1);
	int woffset = W_oc * FH*FW*IC;
	Ws[buf][ty][tx] = *(float4*)(W + woffset);
	__syncthreads();

	//compute area----------------------------------------------------
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
			float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];
			float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];

			//transposed compute core: (W * dY)^T
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

		//load 4 elements from deltaY[N, OH, OW, OC]
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int fh = Y_k / FW_OC, g1 = fh * FW_OC, fw = (Y_k -= g1) / OC;
		int yoffset = fh * OW_OC + Y_k;
		Ys[buf][tx][ty] = S1_SaveYs4(deltaY, fh, fw, yoffset, OH, OW,
			Y0, tih0, tiw0,
			Y1, tih1, tiw1,
			Y2, tih2, tiw2,
			Y3, tih3, tiw3);

		//(W_oc*FH - fh)*FW - fw;
		//= W_oc*FH*FW - fh*FW - fw
		//= (W_k - fh * FW_OC - fw * OC)*FH*FW - fh*FW - fw
		//= W_k*FH*FW - fh*FW_OC*FH*FW - fw*OC*FH*FW - fh*FW - fw
		//= W_k*FH*FW - fh*FW*(FH*FW*OC) - fw*(FH*FW*OC) - fh*FW - fw
		//= W_k*FH*FW - fh*FW*GK - fw*GK - fh*FW - fw
		//= W_k*FH*FW - fh*FW*GK - fh*FW - fw*GK - fw
		//= W_k*FH*FW - fh*FW*(GK + 1) - fw*(GK + 1)
		//= FW*{ W_k*FH - fh*(GK + 1) } - fw*(GK + 1)

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int W_oc = W_k - g1 - fw * OC;
		int woffset = ((W_oc*FH - fh)*FW - fw)*IC;
		Ws[buf][ty][tx] = *(float4*)(W + woffset);
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

	j0 = j0 * IC + ic0;//j0 = ((n * OH + oh) * OW + ow) * IC + ic
	int j1 = j0 + IC, j2 = j1 + IC, j3 = j2 + IC;
	int j4 = j3 + IC, j5 = j4 + IC, j6 = j5 + IC, j7 = j6 + IC;

	*(float4*)(deltaX + j0) = v0; *(float4*)(deltaX + j0 + 4) = v1;
	*(float4*)(deltaX + j1) = v2; *(float4*)(deltaX + j1 + 4) = v3;
	*(float4*)(deltaX + j2) = v4; *(float4*)(deltaX + j2 + 4) = v5;
	*(float4*)(deltaX + j3) = v6; *(float4*)(deltaX + j3 + 4) = v7;
	*(float4*)(deltaX + j4) = v8; *(float4*)(deltaX + j4 + 4) = v9;
	*(float4*)(deltaX + j5) = v10; *(float4*)(deltaX + j5 + 4) = v11;
	*(float4*)(deltaX + j6) = v12; *(float4*)(deltaX + j6 + 4) = v13;
	*(float4*)(deltaX + j7) = v14; *(float4*)(deltaX + j7 + 4) = v15;
}

#endif


#ifndef XKERNEL9
#define XKERNEL9

#define xkernel9(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	BXkernel9<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, ph,pw, ic_index,j_index)

//LB = 4: Size = 1.125, Time = 1.714 msec, Performace = 1436.52 GFlop/s
template<int LB, int STEP>
__global__ void BXkernel9(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__      W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int ph, int pw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	W += ic0 + ((ty >= STEP) << 2);//W[0, 0, 0, tic0]
	W += ((FH - 1)*FW + (FW - 1))*IC;//W[0, FH - 1, FW - 1, tic0]

	//prepare for GM = N * IH * IW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	int tj1 = tj0 + 1, tj2 = tj0 + 2, tj3 = tj0 + 3;
	const int IH_IW = IH * IW;
	get_n_ih_iw(tj0, tn0, tih0, tiw0);
	get_n_ih_iw(tj1, tn1, tih1, tiw1);
	get_n_ih_iw(tj2, tn2, tih2, tiw2);
	get_n_ih_iw(tj3, tn3, tih3, tiw3);
	const int oph = FH - ph - 1, opw = FW - pw - 1;
	tih0 = tih0 - oph, tiw0 = tiw0 - opw;
	tih1 = tih1 - oph, tiw1 = tiw1 - opw;
	tih2 = tih2 - oph, tiw2 = tiw2 - opw;
	tih3 = tih3 - oph, tiw3 = tiw3 - opw;
	const int Y0 = ((tn0*OH + tih0)*OW + tiw0)*OC;
	const int Y1 = ((tn1*OH + tih1)*OW + tiw1)*OC;
	const int Y2 = ((tn2*OH + tih2)*OW + tiw2)*OC;
	const int Y3 = ((tn3*OH + tih3)*OW + tiw3)*OC;

	//prepare for GK = FH * FW * OC
	const int FW_OC = FW * OC, GK = FH * FW_OC;
	
	//load 4 elements from W[OC, FH, FW, IC]
	int W_oc = ty - ((ty >= STEP) << LB >> 1);
	int woffset = W_oc * FH*FW*IC;
	Ws[buf][ty][tx] = *(float4*)(W + woffset);

	//load 4 elements from deltaY[N, OH, OW, OC]
	int Y_k = tx - ((tx >= STEP) << LB >> 1);
	int OW_OC = OW * OC, yoffset = Y_k;
	Ys[buf][tx][ty] = S1_SaveYs4(deltaY, 0, 0, yoffset, OH, OW,
		Y0, tih0, tiw0,
		Y1, tih1, tiw1,
		Y2, tih2, tiw2,
		Y3, tih3, tiw3);
	__syncthreads();

	//compute area----------------------------------------------------
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
			float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];
			float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];

			//transposed compute core: (W * dY)^T
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

		//load 4 elements from deltaY[N, OH, OW, OC]
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int fh = Y_k / FW_OC, fw = (Y_k -= fh * FW_OC) / OC;
	
		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int W_oc = W_k - fh * FW_OC - fw * OC;
		int woffset = ((W_oc*FH - fh)*FW - fw)*IC;
		Ws[buf][ty][tx] = *(float4*)(W + woffset);

		int yoffset = fh * OW_OC + Y_k;
		Ys[buf][tx][ty] = S1_SaveYs4(deltaY, fh, fw, yoffset, OH, OW,
			Y0, tih0, tiw0,
			Y1, tih1, tiw1,
			Y2, tih2, tiw2,
			Y3, tih3, tiw3);

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

	j0 = j0 * IC + ic0;//j0 = ((n * OH + oh) * OW + ow) * IC + ic
	int j1 = j0 + IC, j2 = j1 + IC, j3 = j2 + IC;
	int j4 = j3 + IC, j5 = j4 + IC, j6 = j5 + IC, j7 = j6 + IC;

	*(float4*)(deltaX + j0) = v0; *(float4*)(deltaX + j0 + 4) = v1;
	*(float4*)(deltaX + j1) = v2; *(float4*)(deltaX + j1 + 4) = v3;
	*(float4*)(deltaX + j2) = v4; *(float4*)(deltaX + j2 + 4) = v5;
	*(float4*)(deltaX + j3) = v6; *(float4*)(deltaX + j3 + 4) = v7;
	*(float4*)(deltaX + j4) = v8; *(float4*)(deltaX + j4 + 4) = v9;
	*(float4*)(deltaX + j5) = v10; *(float4*)(deltaX + j5 + 4) = v11;
	*(float4*)(deltaX + j6) = v12; *(float4*)(deltaX + j6 + 4) = v13;
	*(float4*)(deltaX + j7) = v14; *(float4*)(deltaX + j7 + 4) = v15;
}

#endif


#ifndef XKERNEL10
#define XKERNEL10

#define xkernel10(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	BXkernel10<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, ph,pw, ic_index,j_index)

//LB = 4: Size = 1.125, Time = 1.686 msec, Performace = 1432.93 GFlop/s
template<int LB, int STEP>
__global__ void BXkernel10(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__      W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int ph, int pw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	W += ic0 + ((ty >= STEP) << 2);//W[0, 0, 0, tic0]
	W += ((FH - 1)*FW + (FW - 1))*IC;//W[0, FH - 1, FW - 1, tic0]

	//prepare for GM = N * IH * IW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	int tj1 = tj0 + 1, tj2 = tj0 + 2, tj3 = tj0 + 3;
	const int IH_IW = IH * IW;
	get_n_ih_iw(tj0, tn0, tih0, tiw0);
	get_n_ih_iw(tj1, tn1, tih1, tiw1);
	get_n_ih_iw(tj2, tn2, tih2, tiw2);
	get_n_ih_iw(tj3, tn3, tih3, tiw3);
	const int oph = FH - ph - 1, opw = FW - pw - 1;
	tih0 = tih0 - oph, tiw0 = tiw0 - opw;
	tih1 = tih1 - oph, tiw1 = tiw1 - opw;
	tih2 = tih2 - oph, tiw2 = tiw2 - opw;
	tih3 = tih3 - oph, tiw3 = tiw3 - opw;
	const int Y0 = ((tn0*OH + tih0)*OW + tiw0)*OC;
	const int Y1 = ((tn1*OH + tih1)*OW + tiw1)*OC;
	const int Y2 = ((tn2*OH + tih2)*OW + tiw2)*OC;
	const int Y3 = ((tn3*OH + tih3)*OW + tiw3)*OC;

	//prepare for GK = FH * FW * OC
	const int FW_OC = FW * OC, GK = FH * FW_OC;

	//load 4 elements from W[OC, FH, FW, IC]
	int W_oc = ty - ((ty >= STEP) << LB >> 1);//[fhr = fwr = 0]
	int woffset = W_oc * FH*FW*IC;
	Ws[buf][ty][tx] = *(float4*)(W + woffset);

	//load 4 elements from deltaY[N, OH, OW, OC]
	int Y_oc = tx - ((tx >= STEP) << LB >> 1);//[fhr = fwr = 0]
	const int SY = (OW - FW)*OC;
	Ys[buf][tx][ty] = S1_SaveYs4(deltaY, 0, 0, Y_oc, OH, OW,
		Y0, tih0, tiw0,
		Y1, tih1, tiw1,
		Y2, tih2, tiw2,
		Y3, tih3, tiw3);
	__syncthreads();

	//compute area----------------------------------------------------
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
			float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];
			float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];

			//transposed compute core: (W * dY)^T
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
		
		//((W_oc*FH - fh)*FW - fw)*IC;
		//= (W_oc*FH - fh)*FW - fw)
		//= W_oc*FH*FW - fh*FW - fw
		//= (W_k - fw * OC)*FH*FW - fh*FW - fw
		//= W_k*FH*FW - fw*OC*FH*FW - fh*FW - fw
		//= W_k*FH*FW - fh*FW - fw*OC*FH*FW - fw
		//= (W_k*FH*FW - fh*FW) - fw*(FH*FW*OC + 1)
		//woffset = (W_k*FH - fh)*FW*IC - fw*(GK + 1)*IC
		//(1) FW*FH*FW*OC*IC = Wsize*FW
		//(2) W_k*FH*FW*IC = FH*FW*OC*FH*FW*IC = FH*FW*Wsize
		//IC * OC <= (2^30) / (FH * FW)^2
		//IC < 404 && OC < 404

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int fh = W_k / FW_OC, fw = (W_k -= fh * FW_OC) / OC;
		int W_oc = W_k - fw * OC;
		int woffset = ((W_oc*FH - fh)*FW - fw)*IC;
		Ws[buf][ty][tx] = *(float4*)(W + woffset);

		//load 4 elements from deltaY[N, OH, OW, OC]
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx; 
		int yoffset = Y_k + fh * SY;
		Ys[buf][tx][ty] = S1_SaveYs4(deltaY, fh, fw, yoffset, OH, OW,
			Y0, tih0, tiw0,
			Y1, tih1, tiw1,
			Y2, tih2, tiw2,
			Y3, tih3, tiw3);
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

	j0 = j0 * IC + ic0;//j0 = ((n * OH + oh) * OW + ow) * IC + ic
	int j1 = j0 + IC, j2 = j1 + IC, j3 = j2 + IC;
	int j4 = j3 + IC, j5 = j4 + IC, j6 = j5 + IC, j7 = j6 + IC;

	*(float4*)(deltaX + j0) = v0;  *(float4*)(deltaX + j0 + 4) = v1;
	*(float4*)(deltaX + j1) = v2;  *(float4*)(deltaX + j1 + 4) = v3;
	*(float4*)(deltaX + j2) = v4;  *(float4*)(deltaX + j2 + 4) = v5;
	*(float4*)(deltaX + j3) = v6;  *(float4*)(deltaX + j3 + 4) = v7;
	*(float4*)(deltaX + j4) = v8;  *(float4*)(deltaX + j4 + 4) = v9;
	*(float4*)(deltaX + j5) = v10; *(float4*)(deltaX + j5 + 4) = v11;
	*(float4*)(deltaX + j6) = v12; *(float4*)(deltaX + j6 + 4) = v13;
	*(float4*)(deltaX + j7) = v14; *(float4*)(deltaX + j7 + 4) = v15;
}

#endif


#ifndef XKERNEL11
#define XKERNEL11

#define xkernel11(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM) \
	BXkernel11<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, N,IC,OC, ph,pw, ic_index,j_index)

//LB = 4: Size = 1.125, Time = 1.686 msec, Performace = 1432.93 GFlop/s
template<int LB, int STEP>
__global__ void BXkernel11(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__      W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int OC,
	int ph, int pw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	W += ic0 + ((ty >= STEP) << 2);//W[0, 0, 0, tic0]
	W += ((FH - 1)*FW + (FW - 1))*IC;//W[0, FH - 1, FW - 1, tic0]

	//j0 = 8*k + j_index
	//As: j_index % 64 == 0
	//So: j0 = 8*k
	//when: N % 8 == 0
	//(1) ihi = ji / (IW * N) = (j0 + i ) / (IW * N)
	//ihi = (8*k + i) / (8 * x)
	//As: i < 8
	//So: ih0 = ih1 = ih2 = ih3 = ... = ih7
	//(2) iwi = (ji % (IW * N)) / N
	//iwi = ((8*k + i) % (8 * x)) / (8*y)
	//iwi = (8*g + i) / 8*y
	//So: iw0 = iw1 = iw2 = iw3 = ... iw7
	//(3) ni = ji % N
	//So: ni = n0 + i

	//prepare for GM = N * IH * IW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	int tj1 = tj0 + 1, tj2 = tj0 + 2, tj3 = tj0 + 3;
	const int IW_N = IW * N;
	get_ih_iw_n(j0, ih0, iw0, n0);
	int X0 = ((n0*IH + ih0)*IW + iw0)*IC + ic0;
	int Xstride = IH * IW * IC;

	get_ih_iw_n(tj0, tih0, tiw0, tn0);
	get_ih_iw_n(tj1, tih1, tiw1, tn1);
	get_ih_iw_n(tj2, tih2, tiw2, tn2);
	get_ih_iw_n(tj3, tih3, tiw3, tn3);//ih, iw, n
	const int oph = FH - ph - 1, opw = FW - pw - 1;
	tih0 = tih0 - oph, tiw0 = tiw0 - opw;
	tih1 = tih1 - oph, tiw1 = tiw1 - opw;
	tih2 = tih2 - oph, tiw2 = tiw2 - opw;
	tih3 = tih3 - oph, tiw3 = tiw3 - opw;
	const int Y0 = ((tn0*OH + tih0)*OW + tiw0)*OC;
	const int Y1 = ((tn1*OH + tih1)*OW + tiw1)*OC;
	const int Y2 = ((tn2*OH + tih2)*OW + tiw2)*OC;
	const int Y3 = ((tn3*OH + tih3)*OW + tiw3)*OC;

	//prepare for GK = FH * FW * OC
	const int FW_OC = FW * OC, GK = FH * FW_OC;

	//load 4 elements from W[OC, FH, FW, IC]
	int W_oc = ty - ((ty >= STEP) << LB >> 1);//[fhr = fwr = 0]
	int woffset = W_oc * FH*FW*IC;
	Ws[buf][ty][tx] = *(float4*)(W + woffset);

	//load 4 elements from deltaY[N, OH, OW, OC]
	int Y_oc = tx - ((tx >= STEP) << LB >> 1);//[fhr = fwr = 0]
	const int SY = (OW - FW)*OC;
	Ys[buf][tx][ty] = S1_SaveYs4(deltaY, 0, 0, Y_oc, OH, OW,
		Y0, tih0, tiw0,
		Y1, tih1, tiw1,
		Y2, tih2, tiw2,
		Y3, tih3, tiw3);
	__syncthreads();

	//compute area----------------------------------------------------
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
			float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];
			float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];

			//transposed compute core: (W * dY)^T
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

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int fh = W_k / FW_OC, fw = (W_k -= fh * FW_OC) / OC;
		int W_oc = W_k - fw * OC;
		int woffset = ((W_oc*FH - fh)*FW - fw)*IC;
		Ws[buf][ty][tx] = *(float4*)(W + woffset);

		//load 4 elements from deltaY[N, OH, OW, OC]
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int yoffset = Y_k + fh * SY;
		Ys[buf][tx][ty] = S1_SaveYs4(deltaY, fh, fw, yoffset, OH, OW,
			Y0, tih0, tiw0,
			Y1, tih1, tiw1,
			Y2, tih2, tiw2,
			Y3, tih3, tiw3);
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

	
	int X1 = X0 + Xstride, X2 = X1 + Xstride, X3 = X2 + Xstride;
	int X4 = X3 + Xstride, X5 = X4 + Xstride, X6 = X5 + Xstride, X7 = X6 + Xstride;
	
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


#ifndef XKERNEL12
#define XKERNEL12

#define xkernel12(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, ph, pw, GN, GM) \
	BXkernel12<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, N,IC,OC, ph,pw, ic_index,j_index)

//LB = 4: Size = 1, Time = 1.498 msec, Performace = 1433.57 GFlop/s
template<int LB, int STEP>
__global__ void BXkernel12(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__      W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int OC,
	int ph, int pw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	W += ic0 + ((ty >= STEP) << 2);//W[0, 0, 0, tic0]
	W += ((FH - 1)*FW + (FW - 1))*IC;//W[0, FH - 1, FW - 1, tic0]

	//prepare for GM = N * IH * IW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	const int IW_N = IW * N;
	get_ih_iw_n(j0, ih0, iw0, n0);
	const int X0 = ((n0*IH + ih0)*IW + iw0)*IC + ic0;
	const int Xstride = IH * IW * IC;
	int tn0 = n0 + ((tx >= STEP) << 2);
	const int oph = FH - ph - 1, opw = FW - pw - 1;
	int tih = ih0 - oph, tiw = iw0 - opw;
	const int Y0 = ((tn0*OH + tih)*OW + tiw)*OC;
	const int Ystride = OH * OW * OC;
	const int Y1 = Y0 + Ystride;
	const int Y2 = Y1 + Ystride;
	const int Y3 = Y2 + Ystride;

	//prepare for GK = FH * FW * OC
	const int FW_OC = FW * OC, GK = FH * FW_OC;

	//load 4 elements from W[OC, FH, FW, IC]
	int W_oc = ty - ((ty >= STEP) << LB >> 1);//[fhr = fwr = 0]
	int woffset = W_oc * FH*FW*IC;
	Ws[buf][ty][tx] = *(float4*)(W + woffset);

	//load 4 elements from deltaY[N, OH, OW, OC]
	int Y_oc = tx - ((tx >= STEP) << LB >> 1);//[fhr = fwr = 0]
	const int SY = (OW - FW)*OC;
	float4 x; bool ly = LOAD_Y(tih, tiw, 0, 0);
	x.x = (ly ? deltaY[Y0 + Y_oc] : 0);
	x.y = (ly ? deltaY[Y1 + Y_oc] : 0);
	x.z = (ly ? deltaY[Y2 + Y_oc] : 0);
	x.w = (ly ? deltaY[Y3 + Y_oc] : 0);
	Ys[buf][tx][ty] = x;
	__syncthreads();

	//compute area----------------------------------------------------
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
			float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];
			float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];

			//transposed compute core: (W * dY)^T
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

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int fh = W_k / FW_OC, fw = (W_k -= fh * FW_OC) / OC;
		int W_oc = W_k - fw * OC;
		int woffset = ((W_oc*FH - fh)*FW - fw)*IC;
		Ws[buf][ty][tx] = *(float4*)(W + woffset);

		//load 4 elements from deltaY[N, OH, OW, OC]
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int yoffset = Y_k + fh * SY; bool ly = LOAD_Y(tih, tiw, fh, fw);
		x.x = (ly ? deltaY[Y0 + yoffset] : 0);
		x.y = (ly ? deltaY[Y1 + yoffset] : 0);
		x.z = (ly ? deltaY[Y2 + yoffset] : 0);
		x.w = (ly ? deltaY[Y3 + yoffset] : 0);
		Ys[buf][tx][ty] = x;
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


	const int X1 = X0 + Xstride, X2 = X1 + Xstride;
	const int X3 = X2 + Xstride, X4 = X3 + Xstride;
	const int X5 = X4 + Xstride, X6 = X5 + Xstride, X7 = X6 + Xstride;

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




