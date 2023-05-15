#pragma once

#ifndef DECONV3D_DX_ZERO_PADDING_KERNEL_S1_H
#define DECONV3D_DX_ZERO_PADDING_KERNEL_S1_H

//Unsparse Matrix Method:
//rot180: W[oc, fh, fw, ic] -> Wr[oc, FH - 1 - fh, FW - 1 - fw, ic]
//(1) FH*FW >= 2
//(2) GN = IC;           GN >= 4, GN%4 == 0
//(3) GM = N  * IH * IW; GM >= 4, GM%4 == 0
//(4) GK = OC * FH * FW; GK >= 8, GK%4 == 0
//(5) sh = 1, sw = 1
//
//======[improvement for woffset]=======================
//(1) woffset/IC = W_oc*FH_FW - (W_k - W_oc * FH * FW)
//= W_oc*FH_FW - (W_k - W_oc * FH * FW)
//= W_oc*FH_FW - W_k + W_oc*FH*FW
//= 2*W_oc*FH_FW - W_k
//(2) woffset = (2*W_oc*FH_FW - W_k)*IC
//As: W_oc = W_k / FH_FW;
//woffset = (((W_k / FH_FW) << 1)*FH_FW - W_k)*IC
#ifndef DECONV3D_DX_ZERO_PADDING_KERNEL_S1_CALL
#define DECONV3D_DX_ZERO_PADDING_KERNEL_S1_CALL  

//LB = log2(BLOCK_SIZE)

//======[Common]==============================================
#define k88s1(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	zeroPadding_kernel_8_8_s1<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, (W + ((FH-1)*FW + (FW-1))*IC),FH,FW, deltaX,(IH*IW),IW, IC,OC, (FH-ph-1),(FW-pw-1),\
			ic_index,j_index)

#define k44s1(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	kernel_4_4_s1<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>2, GM>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, ph,pw, ic_index,j_index)

//======[PURE: direct conv]===================================
#define k84s1_pure(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	zeroPadding_kernel_8_4_s1_pure<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,(IH*IW),IW, IC,OC, (FH-ph-1),(FW-pw-1),\
			ic_index,j_index)

#define k48s1_pure(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	zeroPadding_kernel_4_8_s1_pure<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>2, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,(IH*IW),IW, IC,OC, (FH-ph-1),(FW-pw-1),\
			ic_index,j_index)

#define k44s1_pure(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	zeroPadding_kernel_4_4_s1_pure<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>2, GM>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,(IH*IW),IW, IC,OC, (FH-ph-1),(FW-pw-1),\
			ic_index,j_index)

#define k82s1_pure(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	zeroPadding_kernel_8_2_s1_pure<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>1, GN>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,(IH*IW),IW, IC,OC, (FH-ph-1),(FW-pw-1),\
			ic_index,j_index)

#define k28s1_pure(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	zeroPadding_kernel_2_8_s1_pure<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>1, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,(IH*IW),IW, IC,OC, (FH-ph-1),(FW-pw-1),\
			ic_index,j_index)

#define k42s1_pure(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	zeroPadding_kernel_4_2_s1_pure<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>1, GN>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,(IH*IW),IW, IC,OC, (FH-ph-1),(FW-pw-1),\
			ic_index,j_index)

#define k24s1_pure(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	zeroPadding_kernel_2_4_s1_pure<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>1, GM>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,(IH*IW),IW, IC,OC, (FH-ph-1),(FW-pw-1),\
			ic_index,j_index)

//======[Small]===============================================
#define k22s1(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	kernel_2_2_s1<LB, (1<<LB)>\
		<<< dim3(GM>>LB>>1, GN>>LB>>1), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, ph,pw, ic_index,j_index)

#define k21s1(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	kernel_2_1_s1<LB, (1<<LB)>\
		<<< dim3(GM>>LB, GN>>LB>>1), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, ph,pw, ic_index,j_index)

#define k12s1(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	kernel_1_2_s1<LB, (1<<LB)>\
		<<< dim3(GM>>LB>>1, GN>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, ph,pw, ic_index,j_index)

#define k11s1(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	kernel_1_1_s1<LB, (1<<LB)>\
		<<< dim3(GM>>LB, GN>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, ph,pw, ic_index,j_index)

#endif


//======[Common]==============================================
//(Y: BLOKC_SIZE*8, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0
//LB = 4, GK % 8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_KERNEL_8_8_S1
#define DECONV3D_DX_ZERO_PADDING_KERNEL_8_8_S1

//LB = 4: Size = 1, Time = 1.558 msec, Performace = 1378.36 GFlop/s
//LB = 3: Size = 1, Time = 1.876 msec, Performace = 1144.71 GFlop/s
//LB = 4: Size = 1.125, Time = 1.746 msec, Performace = 1383.69 GFlop/s
//LB = 3: Size = 1.125, Time = 2.144 msec, Performace = 1126.83 GFlop/s
template<int LB, int STEP> 
__global__ void zeroPadding_kernel_8_8_s1(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH_IW, int IW,
	int IC, int OC,
	int oph, int opw,
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
	get_n_ih_iw(tj0, tn0, tih0, tiw0);
	get_n_ih_iw(tj1, tn1, tih1, tiw1);
	get_n_ih_iw(tj2, tn2, tih2, tiw2);
	get_n_ih_iw(tj3, tn3, tih3, tiw3);
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
	int Y_fh, Y_fw, Y_oc; get_dY_oc_fh_fw(Y_k, Y_oc, Y_fh, Y_fw);
	int yoffset = ((Y_fh*OW) + Y_fw)*OC + Y_oc;
	Ys[buf][tx][ty] = S1_SaveYs4(deltaY, Y_fh, Y_fw, yoffset, OH, OW,
		Y0, tih0, tiw0,
		Y1, tih1, tiw1,
		Y2, tih2, tiw2,
		Y3, tih3, tiw3);

	//load 4 elements from W[OC, FH, FW, IC]: Wr[IC, FH(r), FW(r), OC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);
	int woffset = (((W_k / FH_FW) << 1)*FH_FW - W_k)*IC;
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
		int Y_fh, Y_fw, Y_oc; get_dY_oc_fh_fw(Y_k, Y_oc, Y_fh, Y_fw);
		int yoffset = ((Y_fh*OW) + Y_fw)*OC + Y_oc;
		Ys[buf][tx][ty] = S1_SaveYs4(deltaY, Y_fh, Y_fw, yoffset, OH, OW,
			Y0, tih0, tiw0,
			Y1, tih1, tiw1,
			Y2, tih2, tiw2,
			Y3, tih3, tiw3);

		//load 4 elements from W[OC, FH, FW, IC]: Wr[IC, FH(r), FW(r), OC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int woffset = (((W_k / FH_FW) << 1)*FH_FW - W_k)*IC;
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
	const int j1 = j0 + IC, j2 = j1 + IC, j3 = j2 + IC;
	const int j4 = j3 + IC, j5 = j4 + IC, j6 = j5 + IC, j7 = j6 + IC;

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


//(Y: BLOKC_SIZE*4, X: BLOCK_SIZE*4), GK % (BLOCK_SIZE/2) == 0
//LB = 4, GK % 8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_KERNEL_4_4_S1
#define DECONV3D_DX_ZERO_PADDING_KERNEL_4_4_S1

//LB = 4: Size = 1.125, Time = 2.858 msec, Performace = 845.318 GFlop/s
//LB = 3: Size = 1.125, Time = 4.036 msec, Performace = 598.592 GFlop/s
//LB = 4: Size = 1, Time = 2.564 msec, Performace = 837.552 GFlop/s
//LB = 3: Size = 1, Time = 3.628 msec, Performace = 591.919 GFlop/s
template<int LB, int STEP>
__global__ void kernel_4_4_s1(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int ph, int pw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2 Ys[2][1 << LB >> 1][(2 << LB) + 2];

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 2) + ic_index;
	const int tic0 = ((ty & 1) << 1) + ic0;

	//prepare for GM = N * IH * IW
	int j0 = (((blockIdx.y << LB) + ty) << 2) + j_index;
	int tj0 = j0 + ((tx & 1) << 1), tj1 = tj0 + 1;
	const int IH_IW = IH * IW;
	get_n_ih_iw(tj0, tn0, tih0, tiw0);
	get_n_ih_iw(tj1, tn1, tih1, tiw1);
	const int oph = FH - ph - 1, opw = FW - pw - 1;
	tih0 = tih0 - oph, tiw0 = tiw0 - opw;
	tih1 = tih1 - oph, tiw1 = tiw1 - opw;
	const int Yoffset0 = ((tn0*OH + tih0)*OW + tiw0)*OC;
	const int Yoffset1 = ((tn1*OH + tih1)*OW + tiw1)*OC;

	//prepare for GK = FH * FW * OC
	const int FH_FW = FH * FW, GK = FH_FW * OC;

	//load 2 elements from deltaY[N, OH, OW, OC]: deltaYp[N, OH=OHp, OW=OWp, OC]
	int dY_k = tx >> 1;
	int dY_fh, dY_fw, dY_oc; get_dY_oc_fh_fw(dY_k, dY_oc, dY_fh, dY_fw);
	bool ly0 = (tih0 >= -dY_fh) && (tih0 < OH - dY_fh) && (tiw0 >= -dY_fw) && (tiw0 < OW - dY_fw);
	bool ly1 = (tih1 >= -dY_fh) && (tih1 < OH - dY_fh) && (tiw1 >= -dY_fw) && (tiw1 < OW - dY_fw);
	float2 y; int yoffset = ((dY_fh*OW) + dY_fw)*OC + dY_oc;
	y.x = (ly0 ? deltaY[Yoffset0 + yoffset] : 0);
	y.y = (ly1 ? deltaY[Yoffset1 + yoffset] : 0);
	const int dYs_y = (tx >> 1), dYs_x = (ty << 1) + (tx & 1);
	Ys[buf][dYs_y][dYs_x] = y;

	//load 2 elements from W[OC, FH, FW, IC]: Wr[IC, FH(r), FW(r), OC]
	int W_k = ty >> 1;
	int W_oc, Wr_fh_fw; get_W_oc_fh_fw(W_k, W_oc, Wr_fh_fw);
	const int Ws_x = (ty >> 1), Ws_y = (tx << 1) + (ty & 1);
	Ws[buf][Ws_x][Ws_y] = *(float2*)(&get3d(W, W_oc, Wr_fh_fw, tic0, FH_FW, IC));
	__syncthreads();

	//compute area----------------------------------------------------
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

			//transposed compute core: (W * dY)^T
			simdMM4(v0, y.x, w);
			simdMM4(v1, y.y, w);
			simdMM4(v2, y.z, w);
			simdMM4(v3, y.w, w);
		}
		buf ^= 1;
	
		//load 2 elements from deltaY[N, OH, OW, OC]: deltaYp[N, OH=OHp, OW=OWp, OC]
		int dY_k = ((ok << LB) + tx) >> 1;
		int dY_fh, dY_fw, dY_oc; get_dY_oc_fh_fw(dY_k, dY_oc, dY_fh, dY_fw);
		bool ly0 = (tih0 >= -dY_fh) && (tih0 < OH - dY_fh) && (tiw0 >= -dY_fw) && (tiw0 < OW - dY_fw);
		bool ly1 = (tih1 >= -dY_fh) && (tih1 < OH - dY_fh) && (tiw1 >= -dY_fw) && (tiw1 < OW - dY_fw);
		float2 y; int yoffset = ((dY_fh*OW) + dY_fw)*OC + dY_oc;
		y.x = (ly0 ? deltaY[Yoffset0 + yoffset] : 0);
		y.y = (ly1 ? deltaY[Yoffset1 + yoffset] : 0);
		Ys[buf][dYs_y][dYs_x] = y;

		//load 2 elements from W[OC, FH, FW, IC]: Wr[IC, FH(r), FW(r), OC]
		int W_k = ((ok << LB) + ty) >> 1;
		int W_oc, Wr_fh_fw; get_W_oc_fh_fw(W_k, W_oc, Wr_fh_fw);
		Ws[buf][Ws_x][Ws_y] = *(float2*)(&get3d(W, W_oc, Wr_fh_fw, tic0, FH_FW, IC));
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 dy = *(float4*)(&Ys[buf][ik][ty << 1]);
		float4 w = *(float4*)(&Ws[buf][ik][tx << 1]);

		//transposed compute core: (W * dY)^T
		simdMM4(v0, dy.x, w);
		simdMM4(v1, dy.y, w);
		simdMM4(v2, dy.z, w);
		simdMM4(v3, dy.w, w);
	}

	j0 = j0*IC + ic0; //j0 = ((n * OH + oh) * OW + ow) * IC + ic
	int j1 = j0 + IC, j2 = j1 + IC, j3 = j2 + IC;

	*(float4*)(deltaX + j0) = v0;
	*(float4*)(deltaX + j1) = v1;
	*(float4*)(deltaX + j2) = v2;
	*(float4*)(deltaX + j3) = v3;
}

#endif


//======[PURE: direct conv]===================================
//(Y: BLOKC_SIZE*8, X: BLOCK_SIZE*4), OC % (BLOCK_SIZE/2) == 0
//LB = 4: OC % 8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_KERNEL_8_4_S1_PURE
#define DECONV3D_DX_ZERO_PADDING_KERNEL_8_4_S1_PURE

//LB = 4: Size = 1, Time = 1.762 msec, Performace = 1218.78 GFlop/s
//LB = 3: Size = 1, Time = 1.948 msec, Performace = 1102.4 GFlop/s
//LB = 4: Size = 1.125, Time = 1.962 msec, Performace = 1231.36 GFlop/s
//LB = 3: Size = 1.125, Time = 2.202 msec, Performace = 1097.15 GFlop/s
template<int LB, int STEP>
__global__ void zeroPadding_kernel_8_4_s1_pure(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH_IW, int IW,
	int IC, int OC,
	int oph, int opw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];//follow k88
	__shared__ float2 Ys[2][1 << LB >> 1][(2 << LB) + 2];//follow k44

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	const int tic0 = ic0 + ((ty >= STEP) << 2);
	const int Wstride = FH * FW * IC; W += Wstride - IC + tic0;

	//prepare for GM = N * IH * IW
	int j0 = (((blockIdx.y << LB) + ty) << 2) + j_index;
	const int tj0 = j0 + ((tx & 1) << 1), tj1 = tj0 + 1;
	get_n_ih_iw(tj0, tn0, tih0, tiw0);
	get_n_ih_iw(tj1, tn1, tih1, tiw1);
	tih0 = tih0 - oph, tiw0 = tiw0 - opw;
	tih1 = tih1 - oph, tiw1 = tiw1 - opw;
	int Y0 = ((tn0*OH + tih0)*OW + tiw0)*OC;
	int Y1 = ((tn1*OH + tih1)*OW + tiw1)*OC;

	//compute area----------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);

	const int OOC = (OC << 1 >> LB), SY = (OW - FW)*OC;
	const int Ys_x = (tx >> 1), Ys_y = (ty << 1) + (tx & 1);
	for (int fh = 0; fh < FH; fh++, deltaY += SY) {
		for (int fw = 0; fw < FW; fw++, deltaY += OC, W -= IC)
		{
			//load 2 elements from deltaY
			bool ly0 = LOAD_Y(tih0, tiw0, fh, fw);
			bool ly1 = LOAD_Y(tih1, tiw1, fh, fw);
			float2 y; int Y_oc = tx >> 1;
			y.x = (ly0 ? deltaY[Y0 + Y_oc] : 0);
			y.y = (ly1 ? deltaY[Y1 + Y_oc] : 0);
			Ys[buf][Ys_x][Ys_y] = y;

			//load 4 elements from W
			int W_oc = ty - ((ty >= STEP) << LB >> 1);
			Ws[buf][ty][tx] = *(float4*)(W + W_oc * Wstride);
			__syncthreads();

			for (int ooc = 1; ooc < OOC; ooc++) {
#pragma unroll 
				for (int ik = 0; ik < STEP; ik++)
				{
					float4 y0 = *(float4*)(&Ys[buf][ik][ty << 1]);
					float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];

					//transposed compute core: (W * dY)^T
					simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
					simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
					simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
					simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
				}
				buf ^= 1;

				//load 2 elements from deltaY
				float2 y; int Y_oc = ((ooc << LB) + tx) >> 1;
				y.x = (ly0 ? deltaY[Y0 + Y_oc] : 0);
				y.y = (ly1 ? deltaY[Y1 + Y_oc] : 0);
				Ys[buf][Ys_x][Ys_y] = y;

				//load 4 elements from W
				int W_oc = ((ooc - (ty >= STEP)) << LB >> 1) + ty;
				Ws[buf][ty][tx] = *(float4*)(W + W_oc * Wstride);
				__syncthreads();
			}
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 y0 = *(float4*)(&Ys[buf][ik][ty << 1]);
				float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];

				//transposed compute core: (W * dY)^T
				simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
				simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
				simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
				simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
			}
			buf ^= 1;
		}
	}

	j0 = j0 * IC + ic0; //j0 = ((n * OH + oh) * OW + ow) * IC + ic
	int j1 = j0 + IC, j2 = j1 + IC, j3 = j2 + IC;

	*(float4*)(deltaX + j0) = v0; *(float4*)(deltaX + j0 + 4) = v1;
	*(float4*)(deltaX + j1) = v2; *(float4*)(deltaX + j1 + 4) = v3;
	*(float4*)(deltaX + j2) = v4; *(float4*)(deltaX + j2 + 4) = v5;
	*(float4*)(deltaX + j3) = v6; *(float4*)(deltaX + j3 + 4) = v7;
}

#endif


//(Y: BLOKC_SIZE*4, X: BLOCK_SIZE*8), OC % (BLOCK_SIZE/2) == 0
//LB = 4: OC % 8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_KERNEL_4_8_S1_PURE
#define DECONV3D_DX_ZERO_PADDING_KERNEL_4_8_S1_PURE

//LB = 4: Size = 1, Time = 1.768 msec, Performace = 1214.64 GFlop/s
//LB = 3: Size = 1, Time = 1.878 msec, Performace = 1143.49 GFlop/s
//LB = 4: Size = 1.125, Time = 1.978 msec, Performace = 1221.39 GFlop/s
//LB = 3: Size = 1.125, Time = 2.116 msec, Performace = 1141.74 GFlop/s
template<int LB, int STEP>
__global__ void zeroPadding_kernel_4_8_s1_pure(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH_IW, int IW,
	int IC, int OC,
	int oph, int opw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB >> 1][(2 << LB) + 2];//follow k44
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];//follow k88

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 2) + ic_index;
	const int tic0 = ((ty & 1) << 1) + ic0;
	const int Wstride = FH * FW * IC; W += Wstride - IC + tic0;

	//prepare for GM = N * IH * IW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	int tj1 = tj0 + 1, tj2 = tj0 + 2, tj3 = tj0 + 3;
	get_n_ih_iw(tj0, tn0, tih0, tiw0);
	get_n_ih_iw(tj1, tn1, tih1, tiw1);
	get_n_ih_iw(tj2, tn2, tih2, tiw2);
	get_n_ih_iw(tj3, tn3, tih3, tiw3);
	tih0 = tih0 - oph, tiw0 = tiw0 - opw;
	tih1 = tih1 - oph, tiw1 = tiw1 - opw;
	tih2 = tih2 - oph, tiw2 = tiw2 - opw;
	tih3 = tih3 - oph, tiw3 = tiw3 - opw;
	const int Y0 = ((tn0*OH + tih0)*OW + tiw0)*OC;
	const int Y1 = ((tn1*OH + tih1)*OW + tiw1)*OC;
	const int Y2 = ((tn2*OH + tih2)*OW + tiw2)*OC;
	const int Y3 = ((tn3*OH + tih3)*OW + tiw3)*OC;

	//compute area----------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0);
	float4 v1 = make_float4(0, 0, 0, 0);
	float4 v2 = make_float4(0, 0, 0, 0);
	float4 v3 = make_float4(0, 0, 0, 0);
	float4 v4 = make_float4(0, 0, 0, 0);
	float4 v5 = make_float4(0, 0, 0, 0);
	float4 v6 = make_float4(0, 0, 0, 0);
	float4 v7 = make_float4(0, 0, 0, 0);

	const int OOC = (OC << 1 >> LB), SY = (OW - FW)*OC;
	const int Ws_y = (ty >> 1), Ws_x = (tx << 1) + (ty & 1);
	for (int fh = 0; fh < FH; fh++, deltaY += SY) {
		for (int fw = 0; fw < FW; fw++, deltaY += OC, W -= IC)
		{
			//load 4 elements from deltaY
			bool ly0 = LOAD_Y(tih0, tiw0, fh, fw);
			bool ly1 = LOAD_Y(tih1, tiw1, fh, fw);
			bool ly2 = LOAD_Y(tih2, tiw2, fh, fw);
			bool ly3 = LOAD_Y(tih3, tiw3, fh, fw);
			float4 y; int Y_oc = tx - ((tx >= STEP) << LB >> 1);
			y.x = (ly0 ? deltaY[Y0 + Y_oc] : 0);
			y.y = (ly1 ? deltaY[Y1 + Y_oc] : 0);
			y.z = (ly2 ? deltaY[Y2 + Y_oc] : 0);
			y.w = (ly3 ? deltaY[Y3 + Y_oc] : 0);
			Ys[buf][tx][ty] = y;

			//load 2 elements from W
			int W_oc = ty >> 1;
			Ws[buf][Ws_y][Ws_x] = *(float2*)(W + W_oc * Wstride);
			__syncthreads();

			for (int ooc = 1; ooc < OOC; ooc++) {
#pragma unroll 
				for (int ik = 0; ik < STEP; ik++)
				{
					float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];
					float4 w0 = *(float4*)(&Ws[buf][ik][tx << 1]);

					//transposed compute core: (W * dY)^T
					simdMM4(v0, y0.x, w0);
					simdMM4(v1, y0.y, w0);
					simdMM4(v2, y0.z, w0);
					simdMM4(v3, y0.w, w0);
					simdMM4(v4, y1.x, w0);
					simdMM4(v5, y1.y, w0);
					simdMM4(v6, y1.z, w0);
					simdMM4(v7, y1.w, w0);
				}
				buf ^= 1;

				//load 4 elements from deltaY
				float4 y; int Y_oc = ((ooc - (tx >= STEP)) << LB >> 1) + tx;
				y.x = (ly0 ? deltaY[Y0 + Y_oc] : 0);
				y.y = (ly1 ? deltaY[Y1 + Y_oc] : 0);
				y.z = (ly2 ? deltaY[Y2 + Y_oc] : 0);
				y.w = (ly3 ? deltaY[Y3 + Y_oc] : 0);
				Ys[buf][tx][ty] = y;

				//load 2 elements from W
				int W_oc = ((ooc << LB) + ty) >> 1;
				Ws[buf][Ws_y][Ws_x] = *(float2*)(W + W_oc * Wstride);
				__syncthreads();
			}
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];
				float4 w0 = *(float4*)(&Ws[buf][ik][tx << 1]);

				//transposed compute core: (W * dY)^T
				simdMM4(v0, y0.x, w0);
				simdMM4(v1, y0.y, w0);
				simdMM4(v2, y0.z, w0);
				simdMM4(v3, y0.w, w0);
				simdMM4(v4, y1.x, w0);
				simdMM4(v5, y1.y, w0);
				simdMM4(v6, y1.z, w0);
				simdMM4(v7, y1.w, w0);
			}
			buf ^= 1;
		}
	}

	j0 = j0 * IC + ic0;//j0 = ((n * OH + oh) * OW + ow) * IC + ic
	int j1 = j0 + IC, j2 = j1 + IC, j3 = j2 + IC;
	int j4 = j3 + IC, j5 = j4 + IC, j6 = j5 + IC, j7 = j6 + IC;

	*(float4*)(deltaX + j0) = v0;
	*(float4*)(deltaX + j1) = v1;
	*(float4*)(deltaX + j2) = v2;
	*(float4*)(deltaX + j3) = v3;
	*(float4*)(deltaX + j4) = v4;
	*(float4*)(deltaX + j5) = v5;
	*(float4*)(deltaX + j6) = v6;
	*(float4*)(deltaX + j7) = v7;
}

#endif


//(Y: BLOKC_SIZE*4, X: BLOCK_SIZE*8), OC % (BLOCK_SIZE/2) == 0
//LB = 4: OC % 8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_KERNEL_4_4_S1_PURE
#define DECONV3D_DX_ZERO_PADDING_KERNEL_4_4_S1_PURE

//LB = 4: Size = 1.125, Time = 2.394 msec, Performace = 1009.16  GFlop/s
//LB = 3: Size = 1.125, Time = 2.638 msec, Performace =  915.815 GFlop/s
//LB = 4: Size = 1, Time = 2.1   msec, Performace = 1022.61  GFlop/s
//LB = 3: Size = 1, Time = 2.294 msec, Performace =  936.131 GFlop/s
template<int LB, int STEP>
__global__ void zeroPadding_kernel_4_4_s1_pure(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH_IW, int IW,
	int IC, int OC,
	int oph, int opw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2 Ys[2][1 << LB >> 1][(2 << LB) + 2];

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 2) + ic_index;
	const int tic0 = ((ty & 1) << 1) + ic0;
	const int Wstride = FH * FW * IC; W += Wstride - IC + tic0;

	//prepare for GM = N * IH * IW
	int j0 = (((blockIdx.y << LB) + ty) << 2) + j_index;
	const int tj0 = j0 + ((tx & 1) << 1), tj1 = tj0 + 1;
	get_n_ih_iw(tj0, tn0, tih0, tiw0);
	get_n_ih_iw(tj1, tn1, tih1, tiw1);
	tih0 = tih0 - oph, tiw0 = tiw0 - opw;
	tih1 = tih1 - oph, tiw1 = tiw1 - opw;
	int Y0 = ((tn0*OH + tih0)*OW + tiw0)*OC;
	int Y1 = ((tn1*OH + tih1)*OW + tiw1)*OC;

	//compute area----------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0);
	float4 v1 = make_float4(0, 0, 0, 0);
	float4 v2 = make_float4(0, 0, 0, 0);
	float4 v3 = make_float4(0, 0, 0, 0);

	const int OOC = (OC << 1 >> LB), SY = (OW - FW)*OC;
	const int Ws_y = (ty >> 1), Ws_x = (tx << 1) + (ty & 1);
	const int Ys_x = (tx >> 1), Ys_y = (ty << 1) + (tx & 1);
	for (int fh = 0; fh < FH; fh++, deltaY += SY) {
		for (int fw = 0; fw < FW; fw++, deltaY += OC, W -= IC)
		{
			//load 2 elements from deltaY
			bool ly0 = LOAD_Y(tih0, tiw0, fh, fw);
			bool ly1 = LOAD_Y(tih1, tiw1, fh, fw);
			float2 y; int Y_oc = tx >> 1;
			y.x = (ly0 ? deltaY[Y0 + Y_oc] : 0);
			y.y = (ly1 ? deltaY[Y1 + Y_oc] : 0);
			Ys[buf][Ys_x][Ys_y] = y;

			//load 2 elements from W
			int W_oc = ty >> 1;
			Ws[buf][Ws_y][Ws_x] = *(float2*)(W + W_oc * Wstride);
			__syncthreads();

			for (int ooc = 1; ooc < OOC; ooc++) {
#pragma unroll 
				for (int ik = 0; ik < STEP; ik++)
				{
					float4 y = *(float4*)(&Ys[buf][ik][ty << 1]);
					float4 w = *(float4*)(&Ws[buf][ik][tx << 1]);

					//transposed compute core: (W * dY)^T
					simdMM4(v0, y.x, w);
					simdMM4(v1, y.y, w);
					simdMM4(v2, y.z, w);
					simdMM4(v3, y.w, w);
				}
				buf ^= 1;

				//load 2 elements from deltaY
				float2 y; int Y_oc = ((ooc << LB) + tx) >> 1;
				y.x = (ly0 ? deltaY[Y0 + Y_oc] : 0);
				y.y = (ly1 ? deltaY[Y1 + Y_oc] : 0);
				Ys[buf][Ys_x][Ys_y] = y;

				//load 2 elements from W
				int W_oc = ((ooc << LB) + ty) >> 1;
				Ws[buf][Ws_y][Ws_x] = *(float2*)(W + W_oc * Wstride);
				__syncthreads();
			}
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 y = *(float4*)(&Ys[buf][ik][ty << 1]);
				float4 w = *(float4*)(&Ws[buf][ik][tx << 1]);

				//transposed compute core: (W * dY)^T
				simdMM4(v0, y.x, w);
				simdMM4(v1, y.y, w);
				simdMM4(v2, y.z, w);
				simdMM4(v3, y.w, w);
			}
			buf ^= 1;
		}
	}

	j0 = j0 * IC + ic0; //j0 = ((n * OH + oh) * OW + ow) * IC + ic
	int j1 = j0 + IC, j2 = j1 + IC, j3 = j2 + IC;

	*(float4*)(deltaX + j0) = v0;
	*(float4*)(deltaX + j1) = v1;
	*(float4*)(deltaX + j2) = v2;
	*(float4*)(deltaX + j3) = v3;
}

#endif


//(Y: BLOKC_SIZE*8, X: BLOCK_SIZE*2), OC % (BLOCK_SIZE/2) == 0
//LB = 4: OC % 8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_KERNEL_8_2_S1_PURE
#define DECONV3D_DX_ZERO_PADDING_KERNEL_8_2_S1_PURE

//LB = 4: Size = 1, Time = 2.222 msec, Performace = 966.464 GFlop/s
//LB = 3: Size = 1, Time = 2.388 msec, Performace = 899.281 GFlop/ss
template<int LB, int STEP>
__global__ void zeroPadding_kernel_8_2_s1_pure(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH_IW, int IW,
	int IC, int OC,
	int oph, int opw,
	int ic_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];//follow k88
	__shared__ float  Ys[2][1 << LB >> 1][(2 << LB) + 2];//follow k44

	//prepare for GN = IC
	const int ic0 = (((blockIdx.y << LB) + ty) << 3) + ic_index;
	const int tic0 = ic0 + ((tx >= STEP) << 2);
	const int Wstride = FH * FW * IC; W += Wstride - IC + tic0;

	//prepare for GM = N * IH * IW
	int j0 = (((blockIdx.x << LB) + tx) << 1) + j_index;
	const int tj0 = j0 + (ty & 1);
	get_n_ih_iw(tj0, tn0, tih0, tiw0);
	tih0 = tih0 - oph, tiw0 = tiw0 - opw;
	int Y0 = ((tn0*OH + tih0)*OW + tiw0)*OC;

	//compute area----------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);

	const int OOC = (OC << 1 >> LB), SY = (OW - FW)*OC;
	const int Ys_y = (ty >> 1), Ys_x = (tx << 1) + (ty & 1);
	for (int fh = 0; fh < FH; fh++, deltaY += SY) {
		for (int fw = 0; fw < FW; fw++, deltaY += OC, W -= IC)
		{
			//load 1 element from deltaY
			int Y_oc = ty >> 1;
			bool ly0 = LOAD_Y(tih0, tiw0, fh, fw);
			Ys[buf][Ys_y][Ys_x] = (ly0 ? deltaY[Y0 + Y_oc] : 0);

			//load 4 elements from W
			int W_oc = tx - ((tx >= STEP) << LB >> 1);
			Ws[buf][tx][ty] = *(float4*)(W + W_oc * Wstride);
			__syncthreads();

			for (int ooc = 1; ooc < OOC; ooc++) {
#pragma unroll 
				for (int ik = 0; ik < STEP; ik++)
				{
					float2 y0 = *(float2*)(&Ys[buf][ik][tx << 1]);
					float4 w0 = Ws[buf][ik][ty], w1 = Ws[buf][ik + STEP][ty];

					//transposed compute core: (W * dY)^T
					simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
					simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
				}
				buf ^= 1;

				//load 1 element from deltaY
				int Y_oc = ((ooc << LB) + ty) >> 1;
				Ys[buf][Ys_y][Ys_x] = (ly0 ? deltaY[Y0 + Y_oc] : 0);

				//load 4 elements from W
				int W_oc = ((ooc - (tx >= STEP)) << LB >> 1) + tx;
				Ws[buf][tx][ty] = *(float4*)(W + W_oc * Wstride);
				__syncthreads();
			}
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float2 y0 = *(float2*)(&Ys[buf][ik][tx << 1]);
				float4 w0 = Ws[buf][ik][ty], w1 = Ws[buf][ik + STEP][ty];

				//transposed compute core: (W * dY)^T
				simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
				simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
			}
			buf ^= 1;
		}
	}

	j0 = j0 * IC + ic0; //j0 = ((n * OH + oh) * OW + ow) * IC + ic
	int j1 = j0 + IC;

	*(float4*)(deltaX + j0) = v0; *(float4*)(deltaX + j0 + 4) = v1;
	*(float4*)(deltaX + j1) = v2; *(float4*)(deltaX + j1 + 4) = v3;
}

#endif


//(Y: BLOKC_SIZE*2, X: BLOCK_SIZE*8), OC % (BLOCK_SIZE/2) == 0
//LB = 4: OC % 8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_KERNEL_2_8_S1_PURE
#define DECONV3D_DX_ZERO_PADDING_KERNEL_2_8_S1_PURE

//LB = 4: Size = 1, Time = 2.394 msec, Performace = 897.027 GFlop/s
//LB = 3: Size = 1, Time = 2.794 msec, Performace = 768.605 GFlop/s
//LB = 4: Size = 1.125, Time = 2.654 msec, Performace = 910.293 GFlop/s
//LB = 3: Size = 1.125, Time = 2.884 msec, Performace = 837.697 GFlop/s
template<int LB, int STEP>
__global__ void zeroPadding_kernel_2_8_s1_pure(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH_IW, int IW,
	int IC, int OC,
	int oph, int opw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float  Ws[2][1 << LB >> 1][(2 << LB) + 2];//follow k44
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];//follow k88

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 1) + ic_index;
	const int tic0 = (ty & 1) + ic0;
	const int Wstride = FH * FW * IC; W += Wstride - IC + tic0;

	//prepare for GM = N * IH * IW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	int tj1 = tj0 + 1, tj2 = tj0 + 2, tj3 = tj0 + 3;
	get_n_ih_iw(tj0, tn0, tih0, tiw0);
	get_n_ih_iw(tj1, tn1, tih1, tiw1);
	get_n_ih_iw(tj2, tn2, tih2, tiw2);
	get_n_ih_iw(tj3, tn3, tih3, tiw3);
	tih0 = tih0 - oph, tiw0 = tiw0 - opw;
	tih1 = tih1 - oph, tiw1 = tiw1 - opw;
	tih2 = tih2 - oph, tiw2 = tiw2 - opw;
	tih3 = tih3 - oph, tiw3 = tiw3 - opw;
	const int Y0 = ((tn0*OH + tih0)*OW + tiw0)*OC;
	const int Y1 = ((tn1*OH + tih1)*OW + tiw1)*OC;
	const int Y2 = ((tn2*OH + tih2)*OW + tiw2)*OC;
	const int Y3 = ((tn3*OH + tih3)*OW + tiw3)*OC;

	//compute area----------------------------------------------------
	float2 v0 = make_float2(0, 0);
	float2 v1 = make_float2(0, 0);
	float2 v2 = make_float2(0, 0);
	float2 v3 = make_float2(0, 0);
	float2 v4 = make_float2(0, 0);
	float2 v5 = make_float2(0, 0);
	float2 v6 = make_float2(0, 0);
	float2 v7 = make_float2(0, 0);

	const int OOC = (OC << 1 >> LB), SY = (OW - FW)*OC;
	const int Ws_y = (ty >> 1), Ws_x = (tx << 1) + (ty & 1);
	for (int fh = 0; fh < FH; fh++, deltaY += SY) {
		for (int fw = 0; fw < FW; fw++, deltaY += OC, W -= IC)
		{
			//load 4 elements from deltaY
			bool ly0 = LOAD_Y(tih0, tiw0, fh, fw);
			bool ly1 = LOAD_Y(tih1, tiw1, fh, fw);
			bool ly2 = LOAD_Y(tih2, tiw2, fh, fw);
			bool ly3 = LOAD_Y(tih3, tiw3, fh, fw);
			float4 y; int Y_oc = tx - ((tx >= STEP) << LB >> 1);
			y.x = (ly0 ? deltaY[Y0 + Y_oc] : 0);
			y.y = (ly1 ? deltaY[Y1 + Y_oc] : 0);
			y.z = (ly2 ? deltaY[Y2 + Y_oc] : 0);
			y.w = (ly3 ? deltaY[Y3 + Y_oc] : 0);
			Ys[buf][tx][ty] = y;

			//load 1 element from W
			int W_oc = ty >> 1;
			Ws[buf][Ws_y][Ws_x] = W[W_oc * Wstride];
			__syncthreads();

			for (int ooc = 1; ooc < OOC; ooc++) {
#pragma unroll 
				for (int ik = 0; ik < STEP; ik++)
				{
					float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];
					float2 w0 = *(float2*)(&Ws[buf][ik][tx << 1]);

					//transposed compute core: (W * dY)^T
					simdMM2(v0, y0.x, w0);
					simdMM2(v1, y0.y, w0);
					simdMM2(v2, y0.z, w0);
					simdMM2(v3, y0.w, w0);
					simdMM2(v4, y1.x, w0);
					simdMM2(v5, y1.y, w0);
					simdMM2(v6, y1.z, w0);
					simdMM2(v7, y1.w, w0);
				}
				buf ^= 1;

				//load 4 elements from deltaY
				float4 y; int Y_oc = ((ooc - (tx >= STEP)) << LB >> 1) + tx;
				y.x = (ly0 ? deltaY[Y0 + Y_oc] : 0);
				y.y = (ly1 ? deltaY[Y1 + Y_oc] : 0);
				y.z = (ly2 ? deltaY[Y2 + Y_oc] : 0);
				y.w = (ly3 ? deltaY[Y3 + Y_oc] : 0);
				Ys[buf][tx][ty] = y;

				//load 1 element from W
				int W_oc = ((ooc << LB) + ty) >> 1;
				Ws[buf][Ws_y][Ws_x] = W[W_oc * Wstride];
				__syncthreads();
			}
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];
				float2 w0 = *(float2*)(&Ws[buf][ik][tx << 1]);

				//transposed compute core: (W * dY)^T
				simdMM2(v0, y0.x, w0);
				simdMM2(v1, y0.y, w0);
				simdMM2(v2, y0.z, w0);
				simdMM2(v3, y0.w, w0);
				simdMM2(v4, y1.x, w0);
				simdMM2(v5, y1.y, w0);
				simdMM2(v6, y1.z, w0);
				simdMM2(v7, y1.w, w0);
			}
			buf ^= 1;
		}
	}

	j0 = j0 * IC + ic0;//j0 = ((n * OH + oh) * OW + ow) * IC + ic
	int j1 = j0 + IC, j2 = j1 + IC, j3 = j2 + IC;
	int j4 = j3 + IC, j5 = j4 + IC, j6 = j5 + IC, j7 = j6 + IC;

	*(float2*)(deltaX + j0) = v0;
	*(float2*)(deltaX + j1) = v1;
	*(float2*)(deltaX + j2) = v2;
	*(float2*)(deltaX + j3) = v3;
	*(float2*)(deltaX + j4) = v4;
	*(float2*)(deltaX + j5) = v5;
	*(float2*)(deltaX + j6) = v6;
	*(float2*)(deltaX + j7) = v7;
}

#endif


//(Y: BLOKC_SIZE*4, X: BLOCK_SIZE*2), OC % (BLOCK_SIZE/2) == 0
//LB = 4: OC % 8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_KERNEL_4_2_S1_PURE
#define DECONV3D_DX_ZERO_PADDING_KERNEL_4_2_S1_PURE

//LB = 4: Size = 1.125, Time = 2.956 msec, Performace = 817.293 GFlop/s
//LB = 3: Size = 1.125, Time = 3.854 msec, Performace = 626.86 GFlop/s
//LB = 4: Size = 1, Time = 2.686 msec, Performace = 799.51 GFlop/s
//LB = 3: Size = 1, Time = 3.392 msec, Performace = 633.102 GFlop/s
template<int LB, int STEP>
__global__ void zeroPadding_kernel_4_2_s1_pure(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH_IW, int IW,
	int IC, int OC,
	int oph, int opw,
	int ic_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float  Ys[2][1 << LB >> 1][(2 << LB) + 2];

	//prepare for GN = IC
	const int ic0 = (((blockIdx.y << LB) + ty) << 2) + ic_index;
	const int tic0 = ((tx & 1) << 1) + ic0;
	const int Wstride = FH * FW * IC; W += Wstride - IC + tic0;

	//prepare for GM = N * IH * IW
	int j0 = (((blockIdx.x << LB) + tx) << 1) + j_index;
	int tj0 = j0 + (ty & 1);
	get_n_ih_iw(tj0, tn0, tih0, tiw0);
	tih0 = tih0 - oph, tiw0 = tiw0 - opw;
	int Y0 = ((tn0*OH + tih0)*OW + tiw0)*OC;

	//compute area----------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0);
	float4 v1 = make_float4(0, 0, 0, 0);

	const int OOC = (OC << 1 >> LB), SY = (OW - FW)*OC;
	const int Ws_x = (tx >> 1), Ws_y = (ty << 1) + (tx & 1);
	const int Ys_y = (ty >> 1), Ys_x = (tx << 1) + (ty & 1);
	for (int fh = 0; fh < FH; fh++, deltaY += SY) {
		for (int fw = 0; fw < FW; fw++, deltaY += OC, W -= IC)
		{
			//load 1 element from deltaY
			int Y_oc = ty >> 1; bool ly0 = LOAD_Y(tih0, tiw0, fh, fw);
			Ys[buf][Ys_y][Ys_x] = (ly0 ? deltaY[Y0 + Y_oc] : 0);

			//load 2 elements from W
			int W_oc = tx >> 1;
			Ws[buf][Ws_x][Ws_y] = *(float2*)(W + W_oc * Wstride);
			__syncthreads();

			for (int ooc = 1; ooc < OOC; ooc++) {
#pragma unroll 
				for (int ik = 0; ik < STEP; ik++) {
					float2 y = *(float2*)(&Ys[buf][ik][tx << 1]);
					float4 w = *(float4*)(&Ws[buf][ik][ty << 1]);
					simdMM4(v0, y.x, w);
					simdMM4(v1, y.y, w);
				}
				buf ^= 1;

				//load 1 element from deltaY
				int Y_oc = ((ooc << LB) + ty) >> 1;
				Ys[buf][Ys_y][Ys_x] = (ly0 ? deltaY[Y0 + Y_oc] : 0);

				//load 2 elements from W
				int W_oc = ((ooc << LB) + tx) >> 1;
				Ws[buf][Ws_x][Ws_y] = *(float2*)(W + W_oc * Wstride);
				__syncthreads();
			}
#pragma unroll
			for (int ik = 0; ik < STEP; ik++) {
				float2 y = *(float2*)(&Ys[buf][ik][tx << 1]);
				float4 w = *(float4*)(&Ws[buf][ik][ty << 1]);
				simdMM4(v0, y.x, w);
				simdMM4(v1, y.y, w);
			}
			buf ^= 1;
		}
	}

	j0 = j0 * IC + ic0;//j0 = ((n * OH + oh) * OW + ow) * IC + ic
	int j1 = j0 + IC;

	*(float4*)(deltaX + j0) = v0;
	*(float4*)(deltaX + j1) = v1;
}

#endif


//(Y: BLOKC_SIZE*2, X: BLOCK_SIZE*4), OC % (BLOCK_SIZE/2) == 0
//LB = 4: OC % 8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_KERNEL_2_4_S1_PURE
#define DECONV3D_DX_ZERO_PADDING_KERNEL_2_4_S1_PURE

//LB = 4: Size = 1.125, Time = 2.94  msec, Performace = 821.741 GFlop/s
//LB = 3: Size = 1.125, Time = 3.684 msec, Performace = 655.787 GFlop/s
//LB = 4: Size = 1, Time = 2.64  msec, Performace = 813.441 GFlop/s
//LB = 3: Size = 1, Time = 3.354 msec, Performace = 640.275 GFlop/ss
template<int LB, int STEP>
__global__ void zeroPadding_kernel_2_4_s1_pure(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH_IW, int IW,
	int IC, int OC,
	int oph, int opw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float  Ws[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2 Ys[2][1 << LB >> 1][(2 << LB) + 2];

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 1) + ic_index;
	const int tic0 = (ty & 1) + ic0;
	const int Wstride = FH * FW * IC; W += Wstride - IC + tic0;

	//prepare for GM = N * IH * IW
	int j0 = (((blockIdx.y << LB) + ty) << 2) + j_index;
	const int tj0 = j0 + ((tx & 1) << 1), tj1 = tj0 + 1;
	get_n_ih_iw(tj0, tn0, tih0, tiw0);
	get_n_ih_iw(tj1, tn1, tih1, tiw1);
	tih0 = tih0 - oph, tiw0 = tiw0 - opw;
	tih1 = tih1 - oph, tiw1 = tiw1 - opw;
	int Yoffset0 = ((tn0*OH + tih0)*OW + tiw0)*OC;
	int Yoffset1 = ((tn1*OH + tih1)*OW + tiw1)*OC;

	//compute area----------------------------------------------------
	float2 v0 = make_float2(0, 0);
	float2 v1 = make_float2(0, 0);
	float2 v2 = make_float2(0, 0);
	float2 v3 = make_float2(0, 0);

	const int OOC = (OC << 1 >> LB), SY = (OW - FW)*OC;
	const int Ws_y = (ty >> 1), Ws_x = (tx << 1) + (ty & 1);
	const int Ys_x = (tx >> 1), Ys_y = (ty << 1) + (tx & 1);
	for (int fh = 0; fh < FH; fh++, deltaY += SY) {
		for (int fw = 0; fw < FW; fw++, deltaY += OC, W -= IC)
		{
			//load 1 element from W
			int W_oc = ty >> 1;
			Ws[buf][Ws_y][Ws_x] = W[W_oc*Wstride];

			//load 2 elements from deltaY
			bool ly0 = LOAD_Y(tih0, tiw0, fh, fw);
			bool ly1 = LOAD_Y(tih1, tiw1, fh, fw);
			float2 y; int Y_oc = tx >> 1;
			y.x = (ly0 ? deltaY[Yoffset0 + Y_oc] : 0);
			y.y = (ly1 ? deltaY[Yoffset1 + Y_oc] : 0);
			Ys[buf][Ys_x][Ys_y] = y;
			__syncthreads();

			for (int ooc = 1; ooc < OOC; ooc++) {
#pragma unroll 
				for (int ik = 0; ik < STEP; ik++)
				{
					float4 y = *(float4*)(&Ys[buf][ik][ty << 1]);
					float2 w = *(float2*)(&Ws[buf][ik][tx << 1]);

					//transposed compute core: (W * dY)^T
					simdMM2(v0, y.x, w);
					simdMM2(v1, y.y, w);
					simdMM2(v2, y.z, w);
					simdMM2(v3, y.w, w);
				}
				buf ^= 1;

				//load 2 elements from W
				int W_oc = ((ooc << LB) + ty) >> 1;
				Ws[buf][Ws_y][Ws_x] = W[W_oc*Wstride];

				//load 2 elements from deltaY
				float2 y; int Y_oc = ((ooc << LB) + tx) >> 1;
				y.x = (ly0 ? deltaY[Yoffset0 + Y_oc] : 0);
				y.y = (ly1 ? deltaY[Yoffset1 + Y_oc] : 0);
				Ys[buf][Ys_x][Ys_y] = y;
				__syncthreads();
			}
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 y = *(float4*)(&Ys[buf][ik][ty << 1]);
				float2 w = *(float2*)(&Ws[buf][ik][tx << 1]);

				//transposed compute core: (W * dY)^T
				simdMM2(v0, y.x, w);
				simdMM2(v1, y.y, w);
				simdMM2(v2, y.z, w);
				simdMM2(v3, y.w, w);
			}
			buf ^= 1;
		}
	}

	j0 = j0 * IC + ic0; //j0 = ((n * OH + oh) * OW + ow) * IC + ic
	const int j1 = j0 + IC, j2 = j1 + IC, j3 = j2 + IC;

	*(float2*)(deltaX + j0) = v0;
	*(float2*)(deltaX + j1) = v1;
	*(float2*)(deltaX + j2) = v2;
	*(float2*)(deltaX + j3) = v3;
}

#endif


//======[Small]===============================================
//(Y: BLOKC_SIZE*2, X: BLOCK_SIZE*2) GK >= BLOCK_SIZE
//LB = 4, GK >= 16
//LB = 3, GK >=  8
#ifndef DECONV3D_DX_ZERO_PADDING_KERNEL_2_2_S1
#define DECONV3D_DX_ZERO_PADDING_KERNEL_2_2_S1

//LB = 4: Size = 0.96875 , Time = 4.634 msec, Performace = 448.937 GFlop/s
//LB = 4: Size = 0.415421, Time = 2.52  msec, Performace = 354.011 GFlop/s
//LB = 3: Size = 0.415421, Time = 3.292 msec, Performace = 270.993 GFlop/s
template<int LB, int STEP>
__global__ void kernel_2_2_s1(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int ph, int pw,
	int ic_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2  Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float2 dYs[2][1 << LB][(1 << LB) + 1];

	//prepare for GN = IC
	int ic0 = (((blockIdx.y << LB) + ty) << 1) + ic_index;

	//prepare for GM = N * IH * IW
	const int IH_IW = IH * IW;
	int j0 = (((blockIdx.x << LB) + tx) << 1) + j_index, j1 = j0 + 1;
	get_n_ih_iw(j0, n0, ih0, iw0);
	get_n_ih_iw(j1, n1, ih1, iw1);
	const int oph = FH - ph - 1, opw = FW - pw - 1;
	ih0 -= oph; iw0 -= opw;
	ih1 -= oph; iw1 -= opw;
	int Yoffset0 = ((n0*OH + ih0)*OW + iw0)*OC;
	int Yoffset1 = ((n1*OH + ih1)*OW + iw1)*OC;

	//prepare for GK = FH * FW * OC
	const int FH_FW = FH * FW, GK = FH_FW * OC;

	//load 2 elements from W[OC, FH, FW, IC] : Wr[IC, FH(r), FW(r), OC]
	int W_k = tx, W_oc, Wr_fh_fw;
	get_W_oc_fh_fw(W_k, W_oc, Wr_fh_fw);
	Ws[buf][tx][ty] = *(float2*)(&get3d(W, W_oc, Wr_fh_fw, ic0, FH_FW, IC));

	//load 2 elements from deltaY[N, OH, OW, OC] : deltaYp[N, OH=OHp, OW=OWp, OC]
	int dY_k = ty, dY_fh, dY_fw, dY_oc;
	get_dY_oc_fh_fw(dY_k, dY_oc, dY_fh, dY_fw);
	bool ly0 = (ih0 >= -dY_fh) && (ih0 < OH - dY_fh) && (iw0 >= -dY_fw) && (iw0 < OW - dY_fw);
	bool ly1 = (ih1 >= -dY_fh) && (ih1 < OH - dY_fh) && (iw1 >= -dY_fw) && (iw1 < OW - dY_fw);
	int yoffset = ((dY_fh*OW) + dY_fw)*OC + dY_oc;
	dYs[buf][ty][tx].x = (ly0 ? deltaY[Yoffset0 + yoffset] : 0);
	dYs[buf][ty][tx].y = (ly1 ? deltaY[Yoffset1 + yoffset] : 0);
	__syncthreads();

	//compute area----------------------------------------------------
	float2 v0 = make_float2(0, 0);
	float2 v1 = make_float2(0, 0);
	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float2 dy = dYs[buf][ik][tx];
			float2  w =  Ws[buf][ik][ty];
			simdMM2(v0, dy.x, w);
			simdMM2(v1, dy.y, w);
		}
		buf ^= 1;

		//load 2 elements from W[OC, FH, FW, IC] : Wr[IC, FH(r), FW(r), OC]
		int W_k = (ok << LB) + tx;
		get_W_oc_fh_fw(W_k, W_oc, Wr_fh_fw);
		Ws[buf][tx][ty] = *(float2*)(&get3d(W, W_oc, Wr_fh_fw, ic0, FH_FW, IC));

		//load 2 elements from deltaY[N, OH, OW, OC] : deltaYp[N, OH=OHp, OW=OWp, OC]
		int dY_k = (ok << LB) + ty, dY_fh, dY_fw, dY_oc;
		get_dY_oc_fh_fw(dY_k, dY_oc, dY_fh, dY_fw);
		bool ly0 = (ih0 >= -dY_fh) && (ih0 < OH - dY_fh) && (iw0 >= -dY_fw) && (iw0 < OW - dY_fw);
		bool ly1 = (ih1 >= -dY_fh) && (ih1 < OH - dY_fh) && (iw1 >= -dY_fw) && (iw1 < OW - dY_fw);
		int yoffset = ((dY_fh*OW) + dY_fw)*OC + dY_oc;
		dYs[buf][ty][tx].x = (ly0 ? deltaY[Yoffset0 + yoffset] : 0);
		dYs[buf][ty][tx].y = (ly1 ? deltaY[Yoffset1 + yoffset] : 0);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float2 dy = dYs[buf][ik][tx];
		float2  w =  Ws[buf][ik][ty];
		simdMM2(v0, dy.x, w);
		simdMM2(v1, dy.y, w);
	}

	//when GK % STEP !=0---------------------------------
	for (int k = GK - (GK&(STEP - 1)); k < GK; k++)
	{
		int dY_k = k, dY_fh, dY_fw, dY_oc;
		get_dY_oc_fh_fw(dY_k, dY_oc, dY_fh, dY_fw);
		Wr_fh_fw = FH_FW - 1 - dY_k;

		//load 2 elements from W
		float2 w = *(float2*)(&get3d(W, dY_oc, Wr_fh_fw, ic0, FH_FW, IC));

		//load 2 elements from deltaY
		int oh0 = ih0 + dY_fh, ow0 = iw0 + dY_fw;
		int oh1 = ih1 + dY_fh, ow1 = iw1 + dY_fw;
		float2 dy;
		load4d_s1(dy.x, n0, oh0, ow0, dY_oc);
		load4d_s1(dy.y, n1, oh1, ow1, dY_oc);

		simdMM2(v0, dy.x, w);
		simdMM2(v1, dy.y, w);
	}
	//when GK % STEP !=0---------------------------------

	j0 = j0 * IC + ic0; //j0 = ((n * OH + oh) * OW + ow) * IC + ic
	j1 = j0 + IC;

	*(float2*)(deltaX + j0) = v0;
	*(float2*)(deltaX + j1) = v1;
}

#endif


//(Y: BLOKC_SIZE*2, X: BLOCK_SIZE*1) GK >= BLOCK_SIZE
//LB = 4, GK >= 16
//LB = 3, GK >=  8
#ifndef DECONV3D_DX_ZERO_PADDING_KERNEL_2_1_S1
#define DECONV3D_DX_ZERO_PADDING_KERNEL_2_1_S1

//LB = 4: Size = 0.239468, Time = 1.988 msec, Performace = 258.678 GFlop/s
//LB = 4: Size = 0.415421, Time = 4.046 msec, Performace = 220.492 GFlop/s
//LB = 3: Size = 0.415421, Time = 5.33  msec, Performace = 167.375 GFlop/s
template<int LB, int STEP>
__global__ void kernel_2_1_s1(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int ph, int pw,
	int ic_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float dYs[2][1 << LB][(1 << LB) + 1];

	//prepare for GN = IC
	int ic0 = (((blockIdx.y << LB) + ty) << 1) + ic_index;

	//prepare for GM = N * IH * IW
	const int IH_IW = IH * IW;
	int j0 = ((blockIdx.x << LB) + tx) + j_index;
	get_n_ih_iw(j0, n0, ih0, iw0);
	const int oph = FH - ph - 1, opw = FW - pw - 1;
	ih0 -= oph; iw0 -= opw;
	int Yoffset0 = ((n0*OH + ih0)*OW + iw0)*OC;

	//prepare for GK = FH * FW * OC
	const int FH_FW = FH * FW, GK = FH_FW * OC;

	//load 2 elements from W[OC, FH, FW, IC] : Wr[IC, FH(r), FW(r), OC]
	int W_k = tx, W_oc, Wr_fh_fw;
	get_W_oc_fh_fw(W_k, W_oc, Wr_fh_fw);
	Ws[buf][tx][ty] = *(float2*)(&get3d(W, W_oc, Wr_fh_fw, ic0, FH_FW, IC));

	//load 1 element from deltaY[N, OH, OW, OC] : deltaYp[N, OH=OHp, OW=OWp, OC]
	int dY_k = ty, dY_fh, dY_fw, dY_oc;
	get_dY_oc_fh_fw(dY_k, dY_oc, dY_fh, dY_fw);
	bool ly0 = (ih0 >= -dY_fh) && (ih0 < OH - dY_fh) && (iw0 >= -dY_fw) && (iw0 < OW - dY_fw);
	int yoffset = ((dY_fh*OW) + dY_fw)*OC + dY_oc;
	dYs[buf][ty][tx] = (ly0 ? deltaY[Yoffset0 + yoffset] : 0);
	__syncthreads();

	//compute area----------------------------------------------------
	float2 v = make_float2(0, 0);
	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float dy = dYs[buf][ik][tx];
			float2 w =  Ws[buf][ik][ty];
			simdMM2(v, dy, w);
		}
		buf ^= 1;

		//load 2 elements from W[OC, FH, FW, IC] : Wr[IC, FH(r), FW(r), OC]
		int W_k = (ok << LB) + tx;
		get_W_oc_fh_fw(W_k, W_oc, Wr_fh_fw);
		Ws[buf][tx][ty] = *(float2*)(&get3d(W, W_oc, Wr_fh_fw, ic0, FH_FW, IC));

		//load 1 element from deltaY[N, OH, OW, OC] : deltaYp[N, OH=OHp, OW=OWp, OC]
		int dY_k = (ok << LB) + ty, dY_fh, dY_fw, dY_oc;
		get_dY_oc_fh_fw(dY_k, dY_oc, dY_fh, dY_fw);
		bool ly0 = (ih0 >= -dY_fh) && (ih0 < OH - dY_fh) && (iw0 >= -dY_fw) && (iw0 < OW - dY_fw);
		int yoffset = ((dY_fh*OW) + dY_fw)*OC + dY_oc;
		dYs[buf][ty][tx] = (ly0 ? deltaY[Yoffset0 + yoffset] : 0);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float dy = dYs[buf][ik][tx];
		float2 w =  Ws[buf][ik][ty];
		simdMM2(v, dy, w);
	}

	//when GK % STEP !=0---------------------------------
	for (int k = GK - (GK&(STEP - 1)); k < GK; k++)
	{
		int dY_k = k, dY_fh, dY_fw, dY_oc;
		get_dY_oc_fh_fw(dY_k, dY_oc, dY_fh, dY_fw);
		Wr_fh_fw = FH_FW - 1 - dY_k;

		//load 2 elements from W
		float2 w = *(float2*)(&get3d(W, dY_oc, Wr_fh_fw, ic0, FH_FW, IC));

		//load 1 element from deltaY
		int oh0 = ih0 + dY_fh, ow0 = iw0 + dY_fw;
		float dy; load4d_s1(dy, n0, oh0, ow0, dY_oc);

		simdMM2(v, dy, w);
	}
	//when GK % STEP !=0---------------------------------

	*(float2*)(&deltaX[j0*IC + ic0]) = v;
}

#endif


//(Y: BLOKC_SIZE*1, X: BLOCK_SIZE*2) GK >= BLOCK_SIZE
//LB = 4, GK >= 16
//LB = 3, GK >=  8
#ifndef DECONV3D_DX_ZERO_PADDING_KERNEL_1_2_S1
#define DECONV3D_DX_ZERO_PADDING_KERNEL_1_2_S1

//LB = 4: Size = 0.478935, Time = 4.662 msec, Performace = 220.615 GFlop/s
//LB = 4: Size = 0.415421, Time = 4.666 msec, Performace = 191.193 GFlop/s
//LB = 3: Size = 0.415421, Time = 6.078 msec, Performace = 146.777 GFlop/s
template<int LB, int STEP>
__global__ void kernel_1_2_s1(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int ph, int pw,
	int ic_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float   Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float2 dYs[2][1 << LB][(1 << LB) + 1];

	//prepare for GN = IC
	int ic0 = ((blockIdx.y << LB) + ty) + ic_index;

	//prepare for GM = N * IH * IW
	const int IH_IW = IH * IW;
	int j0 = (((blockIdx.x << LB) + tx) << 1) + j_index, j1 = j0 + 1;
	get_n_ih_iw(j0, n0, ih0, iw0);
	get_n_ih_iw(j1, n1, ih1, iw1);
	const int oph = FH - ph - 1, opw = FW - pw - 1;
	ih0 -= oph; iw0 -= opw;
	ih1 -= oph; iw1 -= opw;
	int Yoffset0 = ((n0*OH + ih0)*OW + iw0)*OC;
	int Yoffset1 = ((n1*OH + ih1)*OW + iw1)*OC;

	//prepare for GK = FH * FW * OC
	const int FH_FW = FH * FW, GK = FH_FW * OC;

	//load 1 element from W[OC, FH, FW, IC] : Wr[IC, FH(r), FW(r), OC]
	int W_k = tx, W_oc, Wr_fh_fw;
	get_W_oc_fh_fw(W_k, W_oc, Wr_fh_fw);
	Ws[buf][tx][ty] = get3d(W, W_oc, Wr_fh_fw, ic0, FH_FW, IC);

	//load 2 elements from deltaY[N, OH, OW, OC] : deltaYp[N, OH=OHp, OW=OWp, OC]
	int dY_k = ty, dY_fh, dY_fw, dY_oc;
	get_dY_oc_fh_fw(dY_k, dY_oc, dY_fh, dY_fw);
	bool ly0 = (ih0 >= -dY_fh) && (ih0 < OH - dY_fh) && (iw0 >= -dY_fw) && (iw0 < OW - dY_fw);
	bool ly1 = (ih1 >= -dY_fh) && (ih1 < OH - dY_fh) && (iw1 >= -dY_fw) && (iw1 < OW - dY_fw);
	int yoffset = ((dY_fh*OW) + dY_fw)*OC + dY_oc;
	dYs[buf][ty][tx].x = (ly0 ? deltaY[Yoffset0 + yoffset] : 0);
	dYs[buf][ty][tx].y = (ly1 ? deltaY[Yoffset1 + yoffset] : 0);
	__syncthreads();

	//compute area----------------------------------------------------
	float2 v = make_float2(0, 0);
	for (int ok = 1, OK = GK >> LB; ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float   w =  Ws[buf][ik][ty];
			float2 dy = dYs[buf][ik][tx];
			simdMM2(v, w, dy);
		}
		buf ^= 1;

		//load 1 element from W[OC, FH, FW, IC] : Wr[IC, FH(r), FW(r), OC]
		int W_k = (ok << LB) + tx;
		get_W_oc_fh_fw(W_k, W_oc, Wr_fh_fw);
		Ws[buf][tx][ty] = get3d(W, W_oc, Wr_fh_fw, ic0, FH_FW, IC);

		//load 2 elements from deltaY[N, OH, OW, OC] : deltaYp[N, OH=OHp, OW=OWp, OC]
		int dY_k = (ok << LB) + ty, dY_fh, dY_fw, dY_oc;
		get_dY_oc_fh_fw(dY_k, dY_oc, dY_fh, dY_fw);
		bool ly0 = (ih0 >= -dY_fh) && (ih0 < OH - dY_fh) && (iw0 >= -dY_fw) && (iw0 < OW - dY_fw);
		bool ly1 = (ih1 >= -dY_fh) && (ih1 < OH - dY_fh) && (iw1 >= -dY_fw) && (iw1 < OW - dY_fw);
		int yoffset = ((dY_fh*OW) + dY_fw)*OC + dY_oc;
		dYs[buf][ty][tx].x = (ly0 ? deltaY[Yoffset0 + yoffset] : 0);
		dYs[buf][ty][tx].y = (ly1 ? deltaY[Yoffset1 + yoffset] : 0);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float   w =  Ws[buf][ik][ty];
		float2 dy = dYs[buf][ik][tx];
		simdMM2(v, w, dy);
	}

	//when GK % STEP !=0---------------------------------
	for (int k = GK - (GK&(STEP - 1)); k < GK; k++)
	{
		int dY_k = k, dY_fh, dY_fw, dY_oc;
		get_dY_oc_fh_fw(dY_k, dY_oc, dY_fh, dY_fw);
		Wr_fh_fw = FH_FW - 1 - dY_k;

		//load 1 element from W
		float w = get3d(W, dY_oc, Wr_fh_fw, ic0, FH_FW, IC);

		//load 2 elements from deltaY
		int oh0 = ih0 + dY_fh, ow0 = iw0 + dY_fw;
		int oh1 = ih1 + dY_fh, ow1 = iw1 + dY_fw;
		float2 dy;
		load4d_s1(dy.x, n0, oh0, ow0, dY_oc);
		load4d_s1(dy.y, n1, oh1, ow1, dY_oc);

		simdMM2(v, w, dy);
	}
	//when GK % STEP !=0---------------------------------

	j0 *= IC;//j0 = ((n * OH + oh) * OW + ow) * IC + ic
	j1 = j0 + IC;

	deltaX[j0 + ic0] = v.x;
	deltaX[j1 + ic0] = v.y;
}

#endif


//(Y: BLOKC_SIZE*1, X: BLOCK_SIZE*1) GK >= BLOCK_SIZE
//LB = 4, GK >= 16
//LB = 3, GK >=  8
#ifndef DECONV3D_DX_ZERO_PADDING_KERNEL_1_1_S1
#define DECONV3D_DX_ZERO_PADDING_KERNEL_1_1_S1

//LB = 4: Size = 0.478935, Time = 7.168 msec, Performace = 143.486 GFlop/s
//LB = 4: Size = 0.415421, Time = 6.688 msec, Performace = 133.389  GFlop/s
//LB = 3: Size = 0.415421, Time = 9.848 msec, Performace =  90.5878 GFlop/s
template<int LB, int STEP>
__global__ void kernel_1_1_s1(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int ph, int pw,
	int ic_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float  Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float dYs[2][1 << LB][(1 << LB) + 1];

	//prepare for GN = IC
	int ic0 = ((blockIdx.y << LB) + ty) + ic_index;

	//prepare for GM = N * IH * IW
	const int IH_IW = IH * IW;
	int j0 = ((blockIdx.x << LB) + tx) + j_index;
	get_n_ih_iw(j0, n0, ih0, iw0);
	const int oph = FH - ph - 1, opw = FW - pw - 1;
	ih0 -= oph; iw0 -= opw;
	int Yoffset0 = ((n0*OH + ih0)*OW + iw0)*OC;

	//prepare for GK = FH * FW * OC
	const int FH_FW = FH * FW, GK = FH_FW * OC;

	//load 1 element from W[OC, FH, FW, IC] : Wr[IC, FH(r), FW(r), OC]
	int W_k = tx, W_oc, Wr_fh_fw;
	get_W_oc_fh_fw(W_k, W_oc, Wr_fh_fw);
	Ws[buf][tx][ty] = get3d(W, W_oc, Wr_fh_fw, ic0, FH_FW, IC);

	//load 1 element from deltaY[N, OH, OW, OC] : deltaYp[N, OH=OHp, OW=OWp, OC]
	int dY_k = ty, dY_fh, dY_fw, dY_oc;
	get_dY_oc_fh_fw(dY_k, dY_oc, dY_fh, dY_fw);
	bool ly0 = (ih0 >= -dY_fh) && (ih0 < OH - dY_fh) && (iw0 >= -dY_fw) && (iw0 < OW - dY_fw);
	int yoffset = ((dY_fh*OW) + dY_fw)*OC + dY_oc;
	dYs[buf][ty][tx] = (ly0 ? deltaY[Yoffset0 + yoffset] : 0);
	__syncthreads();

	//compute area----------------------------------------------------
	float v = 0;
	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float dy = dYs[buf][ik][tx];
			float  w =  Ws[buf][ik][ty];
			v += w * dy;
		}
		buf ^= 1;

		//load 1 element from W[OC, FH, FW, IC] : Wr[IC, FH(r), FW(r), OC]
		int W_k = (ok << LB) + tx;
		get_W_oc_fh_fw(W_k, W_oc, Wr_fh_fw);
		Ws[buf][tx][ty] = get3d(W, W_oc, Wr_fh_fw, ic0, FH_FW, IC);

		//load 2 elements from deltaY[N, OH, OW, OC] : deltaYp[N, OH=OHp, OW=OWp, OC]
		int dY_k = (ok << LB) + ty, dY_fh, dY_fw, dY_oc;
		get_dY_oc_fh_fw(dY_k, dY_oc, dY_fh, dY_fw);
		bool ly0 = (ih0 >= -dY_fh) && (ih0 < OH - dY_fh) && (iw0 >= -dY_fw) && (iw0 < OW - dY_fw);
		int yoffset = ((dY_fh*OW) + dY_fw)*OC + dY_oc;
		dYs[buf][ty][tx] = (ly0 ? deltaY[Yoffset0 + yoffset] : 0);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float dy = dYs[buf][ik][tx];
		float  w =  Ws[buf][ik][ty];
		v += w * dy;
	}

	//when GK % STEP !=0---------------------------------
	for (int k = GK - (GK&(STEP - 1)); k < GK; k++)
	{
		int dY_k = k, dY_fh, dY_fw, dY_oc;
		get_dY_oc_fh_fw(dY_k, dY_oc, dY_fh, dY_fw);
		Wr_fh_fw = FH_FW - 1 - dY_k;

		//load 1 element from W
		float w = get3d(W, dY_oc, Wr_fh_fw, ic0, FH_FW, IC);

		//load 1 element from deltaY
		int oh0 = ih0 + dY_fh, ow0 = iw0 + dY_fw;
		float dy; load4d_s1(dy, n0, oh0, ow0, dY_oc);

		v += w * dy;
	}
	//when GK % STEP !=0---------------------------------

	deltaX[j0*IC + ic0] = v;
}

#endif

#endif