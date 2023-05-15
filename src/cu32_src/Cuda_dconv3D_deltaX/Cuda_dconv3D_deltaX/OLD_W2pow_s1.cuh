
//(Y: BLOKC_SIZE*8, X: BLOCK_SIZE*4), GK % (BLOCK_SIZE/2) == 0, FH, FW is power of 2
//LB = 4, GK % 8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_KERNEL_8_4_S1_W2POW_TEXTURE
#define DECONV3D_DX_ZERO_PADDING_KERNEL_8_4_S1_W2POW_TEXTURE

#define k84s1_W2pow_tex(stream, LB, ic_index, j_index, deltaY, OH, OW, W, LFH, LFW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	zeroPadding_kernel_8_4_s1_W2pow_texture<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,LFH,LFW, deltaX,IH,IW, IC,OC, ph,pw, ic_index,j_index)

//LB = 4: Size = 1, Time = 1.744 msec, Performace = 1231.36 GFlop/s
//LB = 3: Size = 1, Time = 1.934 msec, Performace = 1110.38 GFlop/s
template<int LB, int STEP>
__global__ void zeroPadding_kernel_8_4_s1_W2pow_texture(
	cudaTextureObject_t deltaY, int OH, int OW,
	const float* __restrict__      W, int LFH, int LFW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int ph, int pw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];//follow k88
	__shared__ float2 Ys[2][1 << LB >> 1][(2 << LB) + 2];//follow k44

	//prepare for GK = FH * FW * OC
	const int LFH_FW = LFH + LFW, FH_FW_m1 = (1 << LFH_FW) - 1;
	const int FH_m1 = (1 << LFH) - 1, FW_m1 = (1 << LFW) - 1;
	const int GK = OC << LFH_FW;

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	W += ic0 + ((ty >= STEP) << 2);//W[0, 0, 0, tic0]

	//prepare for GM = N * IH * IW
	int j0 = (((blockIdx.y << LB) + ty) << 2) + j_index;
	int tj0 = j0 + ((tx & 1) << 1), tj1 = tj0 + 1;
	const int IH_IW = IH * IW;
	get_n_ih_iw(tj0, tn0, tih0, tiw0);
	get_n_ih_iw(tj1, tn1, tih1, tiw1);
	const int oph = FH_m1 - ph, opw = FW_m1 - pw;
	tih0 = tih0 - oph, tiw0 = tiw0 - opw;
	tih1 = tih1 - oph, tiw1 = tiw1 - opw;
	const int Y0 = ((tn0*OH + tih0)*OW + tiw0)*OC;
	const int Y1 = ((tn1*OH + tih1)*OW + tiw1)*OC;

	//load 2 elements from deltaY[N, OH, OW, OC]: deltaYp[N, OH=OHp, OW=OWp, OC]
	int dY_k = tx >> 1;
	int dY_fh, dY_fw, dY_oc; get_dY_oc_fh_fw_W2pow(dY_k, dY_oc, dY_fh, dY_fw);
	bool ly0 = (tih0 >= -dY_fh) && (tih0 < OH - dY_fh) && (tiw0 >= -dY_fw) && (tiw0 < OW - dY_fw);
	bool ly1 = (tih1 >= -dY_fh) && (tih1 < OH - dY_fh) && (tiw1 >= -dY_fw) && (tiw1 < OW - dY_fw);
	float2 y; int yoffset = ((dY_fh*OW) + dY_fw)*OC + dY_oc;
	zero_float(y.x, ly0, tex1Dfetch<float>(deltaY, Y0 + yoffset));
	zero_float(y.y, ly1, tex1Dfetch<float>(deltaY, Y1 + yoffset));
	const int Ys_x = (tx >> 1), Ys_y = (ty << 1) + (tx & 1);
	Ys[buf][Ys_x][Ys_y] = y;

	//load 4 elements from W[OC, FH, FW, IC]: Wr[IC, FH(r), FW(r), OC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);
	int W_oc, Wr_fh_fw; get_W_oc_fh_fw_W2pow(W_k, W_oc, Wr_fh_fw);
	Ws[buf][ty][tx] = *(float4*)(&W[((W_oc << LFH_FW) + Wr_fh_fw)*IC]);
	__syncthreads();

	//compute area----------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)
	{
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

		//load 2 elements from deltaY[N, OH, OW, OC]: deltaYp[N, OH=OHp, OW=OWp, OC]
		int dY_k = ((ok << LB) + tx) >> 1;
		int dY_fh, dY_fw, dY_oc; get_dY_oc_fh_fw_W2pow(dY_k, dY_oc, dY_fh, dY_fw);
		bool ly0 = (tih0 >= -dY_fh) && (tih0 < OH - dY_fh) && (tiw0 >= -dY_fw) && (tiw0 < OW - dY_fw);
		bool ly1 = (tih1 >= -dY_fh) && (tih1 < OH - dY_fh) && (tiw1 >= -dY_fw) && (tiw1 < OW - dY_fw);
		float2 y; int yoffset = ((dY_fh*OW) + dY_fw)*OC + dY_oc;
		zero_float(y.x, ly0, tex1Dfetch<float>(deltaY, Y0 + yoffset));
		zero_float(y.y, ly1, tex1Dfetch<float>(deltaY, Y1 + yoffset));
		Ys[buf][Ys_x][Ys_y] = y;

		//load 4 elements from W[OC, FH, FW, IC]: Wr[IC, FH(r), FW(r), OC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int W_oc, Wr_fh_fw; get_W_oc_fh_fw_W2pow(W_k, W_oc, Wr_fh_fw);
		Ws[buf][ty][tx] = *(float4*)(&W[((W_oc << LFH_FW) + Wr_fh_fw)*IC]);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];
		float4 y0 = *(float4*)(&Ys[buf][ik][ty << 1]);

		//transposed compute core: (W * dY)^T
		simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
		simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
		simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
		simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
	}

	j0 = j0 * IC + ic0;//j0 = ((n * OH + oh) * OW + ow) * IC + ic
	int j1 = j0 + IC, j2 = j1 + IC, j3 = j2 + IC;

	*(float4*)(deltaX + j0) = v0; *(float4*)(deltaX + j0 + 4) = v1;
	*(float4*)(deltaX + j1) = v2; *(float4*)(deltaX + j1 + 4) = v3;
	*(float4*)(deltaX + j2) = v4; *(float4*)(deltaX + j2 + 4) = v5;
	*(float4*)(deltaX + j3) = v6; *(float4*)(deltaX + j3 + 4) = v7;
}

#endif


//(Y: BLOKC_SIZE*8, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0, FH, FW is power of 2
//LB = 4, GK % 8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_KERNEL_4_8_S1_W2POW_TEXTURE
#define DECONV3D_DX_ZERO_PADDING_KERNEL_4_8_S1_W2POW_TEXTURE

#define k48s1_W2pow_tex(stream, LB, ic_index, j_index, deltaY, OH, OW, W, LFH, LFW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	zeroPadding_kernel_4_8_s1_W2pow_texture<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>2, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,LFH,LFW, deltaX,IH,IW, IC,OC, ph,pw, ic_index,j_index)

//LB = 4: Size = 1, Time = 1.77  msec, Performace = 1213.27 GFlop/s
//LB = 3: Size = 1, Time = 2.086 msec, Performace = 1029.47 GFlop/s
template<int LB, int STEP>
__global__ void zeroPadding_kernel_4_8_s1_W2pow_texture(
	cudaTextureObject_t deltaY, int OH, int OW,
	const float* __restrict__      W, int LFH, int LFW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int ph, int pw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float2  Ws[2][1 << LB >> 1][(2 << LB) + 2];//follow k44
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];//follow k88

	//prepare for GK = FH * FW * OC
	const int LFH_FW = LFH + LFW, FH_FW_m1 = (1 << LFH_FW) - 1;
	const int FH_m1 = (1 << LFH) - 1, FW_m1 = (1 << LFW) - 1;
	const int GK = OC << LFH_FW;

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 2) + ic_index;
	W += ((ty & 1) << 1) + ic0;//W[0, 0, 0, tic0]

	//prepare for GM = N * IH * IW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2);
	int tj1 = tj0 + 1, tj2 = tj0 + 2, tj3 = tj0 + 3;
	const int IH_IW = IH * IW;
	get_n_ih_iw(tj0, tn0, tih0, tiw0);
	get_n_ih_iw(tj1, tn1, tih1, tiw1);
	get_n_ih_iw(tj2, tn2, tih2, tiw2);
	get_n_ih_iw(tj3, tn3, tih3, tiw3);
	const int oph = FH_m1 - ph, opw = FW_m1 - pw;
	tih0 = tih0 - oph, tiw0 = tiw0 - opw;
	tih1 = tih1 - oph, tiw1 = tiw1 - opw;
	tih2 = tih2 - oph, tiw2 = tiw2 - opw;
	tih3 = tih3 - oph, tiw3 = tiw3 - opw;
	const int Y0 = ((tn0*OH + tih0)*OW + tiw0)*OC;
	const int Y1 = ((tn1*OH + tih1)*OW + tiw1)*OC;
	const int Y2 = ((tn2*OH + tih2)*OW + tiw2)*OC;
	const int Y3 = ((tn3*OH + tih3)*OW + tiw3)*OC;

	//load 2 elements from W[OC, FH, FW, IC]: Wr[IC, FH(r), FW(r), OC]
	int W_k = ty >> 1;
	int W_oc, Wr_fh_fw; get_W_oc_fh_fw_W2pow(W_k, W_oc, Wr_fh_fw);
	const int Ws_x = (ty >> 1), Ws_y = (tx << 1) + (ty & 1);
	Ws[buf][Ws_x][Ws_y] = *(float2*)(&W[((W_oc << LFH_FW) + Wr_fh_fw)*IC]);

	//load 4 elements from deltaY[N, OH, OW, OC]: deltaYp[N, OH=OHp, OW=OWp, OC]
	int dY_k = tx - ((tx >= STEP) << LB >> 1);
	int dY_fh, dY_fw, dY_oc; get_dY_oc_fh_fw_W2pow(dY_k, dY_oc, dY_fh, dY_fw);
	int yoffset = ((dY_fh*OW) + dY_fw)*OC + dY_oc;
	Ys[buf][tx][ty] = S1_SaveYs4_tex(deltaY, dY_fh, dY_fw, yoffset, OH, OW,
		Y0, tih0, tiw0,
		Y1, tih1, tiw1,
		Y2, tih2, tiw2,
		Y3, tih3, tiw3);
	__syncthreads();

	//compute area----------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 w0 = *(float4*)(&Ws[buf][ik][tx << 1]);
			float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];

			//transposed compute core: (W * dY)^T
			simdMM4(v0, y0.x, w0);
			simdMM4(v2, y0.y, w0);
			simdMM4(v4, y0.z, w0);
			simdMM4(v6, y0.w, w0);
			simdMM4(v8, y1.x, w0);
			simdMM4(v10, y1.y, w0);
			simdMM4(v12, y1.z, w0);
			simdMM4(v14, y1.w, w0);
		}
		buf ^= 1;

		//load 2 elements from W[OC, FH, FW, IC]: Wr[IC, FH(r), FW(r), OC]
		int W_k = ((ok << LB) + ty) >> 1;
		int W_oc, Wr_fh_fw; get_W_oc_fh_fw_W2pow(W_k, W_oc, Wr_fh_fw);
		Ws[buf][Ws_x][Ws_y] = *(float2*)(&W[((W_oc << LFH_FW) + Wr_fh_fw)*IC]);

		//load 4 elements from deltaY[N, OH, OW, OC]: deltaYp[N, OH=OHp, OW=OWp, OC]
		int dY_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int dY_fh, dY_fw, dY_oc; get_dY_oc_fh_fw_W2pow(dY_k, dY_oc, dY_fh, dY_fw);
		int yoffset = ((dY_fh*OW) + dY_fw)*OC + dY_oc;
		Ys[buf][tx][ty] = S1_SaveYs4_tex(deltaY, dY_fh, dY_fw, yoffset, OH, OW,
			Y0, tih0, tiw0,
			Y1, tih1, tiw1,
			Y2, tih2, tiw2,
			Y3, tih3, tiw3);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 w0 = *(float4*)(&Ws[buf][ik][tx << 1]);
		float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];

		//transposed compute core: (W * dY)^T
		simdMM4(v0, y0.x, w0);
		simdMM4(v2, y0.y, w0);
		simdMM4(v4, y0.z, w0);
		simdMM4(v6, y0.w, w0);
		simdMM4(v8, y1.x, w0);
		simdMM4(v10, y1.y, w0);
		simdMM4(v12, y1.z, w0);
		simdMM4(v14, y1.w, w0);
	}

	j0 = j0 * IC + ic0;//j0 = ((n * OH + oh) * OW + ow) * IC + ic
	int j1 = j0 + IC, j2 = j1 + IC, j3 = j2 + IC;
	int j4 = j3 + IC, j5 = j4 + IC, j6 = j5 + IC, j7 = j6 + IC;

	*(float4*)(deltaX + j0) = v0;
	*(float4*)(deltaX + j1) = v2;
	*(float4*)(deltaX + j2) = v4;
	*(float4*)(deltaX + j3) = v6;
	*(float4*)(deltaX + j4) = v8;
	*(float4*)(deltaX + j5) = v10;
	*(float4*)(deltaX + j6) = v12;
	*(float4*)(deltaX + j7) = v14;
}

#endif