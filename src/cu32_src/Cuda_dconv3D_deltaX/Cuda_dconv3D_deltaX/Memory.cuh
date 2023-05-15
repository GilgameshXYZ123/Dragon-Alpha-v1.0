



//(Y: BLOKC_SIZE*8, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0
//LB = 4, GK % 8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_KERNEL_8_8_S1_W3_X4
#define DECONV3D_DX_ZERO_PADDING_KERNEL_8_8_S1_W3_X4

#define k88s1W3x4(stream, LB, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	zeroPadding_kernel_8_8_s1_W3_x4<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W, deltaX,(IH*IW),IW, IC,OC, (2-ph),(2-pw),\
			ic_index,j_index)

//k88<4>: Size = 0.5625, Time = 0.968 msec, Performace = 1247.89 GFlop/s
//k88<3>: Size = 0.5625, Time = 1.148 msec, Performace = 1052.23 GFlop/s
//LB = 4: Size = 0.5625, Time = 0.93  msec, Performace = 1298.88 GFlop/s
//LB = 3: Size = 0.5625, Time = 1.088 msec, Performace = 1110.26 GFlop/s
//LB = 4: Size = 1.125, Time = 1.636 msec, Performace = 1476.72 GFlop/s
//LB = 3: Size = 1.125, Time = 1.986 msec, Performace = 1216.47 GFlop/s
template<int LB, int STEP>
__global__ void zeroPadding_kernel_8_8_s1_W3_x4(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W,//FH = FW = 3
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
	get_n_ih_iw(tj0, tn0, tih0, tiw0);
	tih0 = tih0 - oph, tiw0 = tiw0 - opw;
	const int tiw1 = tiw0 + 1, tiw2 = tiw0 + 2, tiw3 = tiw0 + 3;
	deltaY += ((tn0*OH + tih0)*OW + tiw1)*OC;

	//prepare for GK = FH * FW * OC
	const int GK = 9 * OC;

	//load 4 elements from deltaY[N, OH, OW, OC]: deltaYp[N, OH=OHp, OW=OWp, OC]
	int Y_k = tx - ((tx >= STEP) << LB >> 1);
	int YIdx = Y_k / OC; char Yfhw = YIDX_W33[YIdx];
	int Y_fh = Yfhw >> 2, Y_fw = Yfhw & 3;
	int yoffset = (Y_fh*OW + Y_fw - YIdx)*OC + Y_k;
	Ys[buf][tx][ty] = S1_SaveYs4x(deltaY, Y_fh, Y_fw, yoffset, OH, OW, OC,
		tih0, tiw0, tiw1, tiw2, tiw3);

	//load 4 elements from W[OC, FH, FW, IC]: Wr[IC, FH(r), FW(r), OC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);
	int WIdx = W_k / OC, W_oc = W_k - WIdx * OC;
	int woffset = (W_oc * 9 + WIDX_W33[WIdx])*IC;
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
		int YIdx = Y_k / OC; char Yfhw = YIDX_W33[YIdx];
		int Y_fh = Yfhw >> 2, Y_fw = Yfhw & 3;
		int yoffset = (Y_fh*OW + Y_fw - YIdx)*OC + Y_k;
		Ys[buf][tx][ty] = S1_SaveYs4x(deltaY, Y_fh, Y_fw, yoffset, OH, OW, OC,
			tih0, tiw0, tiw1, tiw2, tiw3);

		//load 4 elements from W[OC, FH, FW, IC]: Wr[IC, FH(r), FW(r), OC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int WIdx = W_k / OC, W_oc = W_k - WIdx * OC;
		int woffset = (W_oc * 9 + WIDX_W33[WIdx])*IC;
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


//(Y: BLOKC_SIZE*8, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0
//LB = 4, GK % 8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_KERNEL_8_8_S1_W3
#define DECONV3D_DX_ZERO_PADDING_KERNEL_8_8_S1_W3


#define k88s1W3(stream, LB, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	zeroPadding_kernel_8_8_s1_W3<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W, deltaX,(IH*IW),IW, IC,OC, (2-ph),(2-pw),\
			ic_index,j_index)

//k88<4>: Size = 0.5625, Time = 0.968 msec, Performace = 1247.89 GFlop/s
//k88<3>: Size = 0.5625, Time = 1.148 msec, Performace = 1052.23 GFlop/s
//LB = 4: Size = 0.5625, Time = 0.938 msec, Performace = 1287.8 GFlop/s
//LB = 3: Size = 0.5625, Time = 1.082 msec, Performace = 1116.41 GFlop/s
//LB = 4: Size = 1.125, Time = 1.652 msec, Performace = 1462.42 GFlop/s
template<int LB, int STEP>
__global__ void zeroPadding_kernel_8_8_s1_W3(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W,//FH = FW = 3
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
	const int GK = 9 * OC;

	//load 4 elements from deltaY[N, OH, OW, OC]: deltaYp[N, OH=OHp, OW=OWp, OC]
	int Y_k = tx - ((tx >= STEP) << LB >> 1);
	int YIdx = Y_k / OC; char Yfhw = YIDX_W33[YIdx];
	int Y_fh = Yfhw >> 2, Y_fw = Yfhw & 3;
	int yoffset = (Y_fh*OW + Y_fw - YIdx)*OC + Y_k;
	Ys[buf][tx][ty] = S1_SaveYs4(deltaY, Y_fh, Y_fw, yoffset, OH, OW,
		Y0, tih0, tiw0,
		Y1, tih1, tiw1,
		Y2, tih2, tiw2,
		Y3, tih3, tiw3);

	//load 4 elements from W[OC, FH, FW, IC]: Wr[IC, FH(r), FW(r), OC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);
	int WIdx = W_k / OC, W_oc = W_k - WIdx * OC;
	int woffset = (W_oc * 9 + WIDX_W33[WIdx])*IC;
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
		int YIdx = Y_k / OC; char Yfhw = YIDX_W33[YIdx];
		int Y_fh = Yfhw >> 2, Y_fw = Yfhw & 3;
		int yoffset = (Y_fh*OW + Y_fw - YIdx)*OC + Y_k;
		Ys[buf][tx][ty] = S1_SaveYs4(deltaY, Y_fh, Y_fw, yoffset, OH, OW,
			Y0, tih0, tiw0,
			Y1, tih1, tiw1,
			Y2, tih2, tiw2,
			Y3, tih3, tiw3);

		//load 4 elements from W[OC, FH, FW, IC]: Wr[IC, FH(r), FW(r), OC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int WIdx = W_k / OC, W_oc = W_k - WIdx * OC;
		int woffset = (W_oc * 9 + WIDX_W33[WIdx])*IC;
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


//(Y: BLOKC_SIZE*8, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0, OC is power of 2
//LB = 4, GK % 8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_KERNEL_8_8_S1_W5_X4
#define DECONV3D_DX_ZERO_PADDING_KERNEL_8_8_S1_W5_X4

#define k88s1W5x4(stream, LB, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	zeroPadding_kernel_8_8_s1_W5_x4<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W, deltaX,(IH*IW),IW, IC,OC, (4-ph),(4-pw),\
			ic_index,j_index)

//k88<4>: Size = 1.5625, Time = 2.416 msec, Performace = 1388.84 GFlop/s
//k88<3>: Size = 1.5625, Time = 2.  9 msec, Performace = 1157.05 GFlop/s
//LB = 4: Size = 1.5625, Time = 2.21  msec, Performace = 1518.3 GFlop/s
//LB = 3: Size = 1.5625, Time = 2.698 msec, Performace = 1243.68 GFlop/s
template<int LB, int STEP>
__global__ void zeroPadding_kernel_8_8_s1_W5_x4(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W,//FH = FW = 5
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
	get_n_ih_iw(tj0, tn0, tih0, tiw0);
	tih0 = tih0 - oph, tiw0 = tiw0 - opw;
	const int tiw1 = tiw0 + 1, tiw2 = tiw0 + 2, tiw3 = tiw0 + 3;
	deltaY += ((tn0*OH + tih0)*OW + tiw1)*OC;

	//prepare for GK = FH * FW * OC
	const int GK = 25 * OC;

	//load 4 elements from deltaY[N, OH, OW, OC] : deltaYp[N, OH=OHp, OW=OWp, OC]
	int Y_k = tx - ((tx >= STEP) << LB >> 1);
	int YIdx = Y_k / OC; char Yfhw = YIDX_W55[YIdx];
	int Y_fh = Yfhw >> 3, Y_fw = Yfhw & 7;
	int yoffset = (Y_fh*OW + Y_fw - YIdx)*OC + Y_k;
	Ys[buf][tx][ty] = S1_SaveYs4x(deltaY, Y_fh, Y_fw, yoffset, OH, OW, OC,
		tih0, tiw0, tiw1, tiw2, tiw3);

	//load 4 elements from W[OC, FH, FW, IC]: Wr[IC, FH(r), FW(r), OC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);
	int WIdx = W_k / OC, W_oc = W_k - WIdx * OC;
	int woffset = (W_oc * 25 + WIDX_W55[WIdx])*IC;
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
		int YIdx = Y_k / OC; char Yfhw = YIDX_W55[YIdx];
		int Y_fh = Yfhw >> 3, Y_fw = Yfhw & 7;
		int yoffset = (Y_fh*OW + Y_fw - YIdx)*OC + Y_k;
		Ys[buf][tx][ty] = S1_SaveYs4x(deltaY, Y_fh, Y_fw, yoffset, OH, OW, OC,
			tih0, tiw0, tiw1, tiw2, tiw3);

		//load 4 elements from W[OC, FH, FW, IC]: Wr[IC, FH(r), FW(r), OC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int WIdx = W_k / OC, W_oc = W_k - WIdx * OC;
		int woffset = (W_oc * 25 + WIDX_W55[WIdx])*IC;
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


//(Y: BLOKC_SIZE*8, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0
//LB = 4, GK % 8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_KERNEL_8_8_S1_W5
#define DECONV3D_DX_ZERO_PADDING_KERNEL_8_8_S1_W5

#define k88s1W5(stream, LB, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	zeroPadding_kernel_8_8_s1_W5<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W, deltaX,(IH*IW),IW, IC,OC, (4-ph),(4-pw),\
			ic_index,j_index)


//k88<4>: Size = 1.5625, Time = 2.416 msec, Performace = 1388.84 GFlop/s
//k88<3>: Size = 1.5625, Time = 2.  9 msec, Performace = 1157.05 GFlop/s
//LB = 4: Size = 1.5625, Time = 2.212 msec, Performace = 1516.93 GFlop/s
//LB = 3: Size = 1.5625, Time = 2.668 msec, Performace = 1257.66 GFlop/s
template<int LB, int STEP>
__global__ void zeroPadding_kernel_8_8_s1_W5(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W,//FH = FW = 5
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
	const int GK = 25 * OC;

	//load 4 elements from deltaY[N, OH, OW, OC]: deltaYp[N, OH=OHp, OW=OWp, OC]
	int Y_k = tx - ((tx >= STEP) << LB >> 1);
	int YIdx = Y_k / OC; char Yfhw = YIDX_W55[YIdx];
	int Y_fh = Yfhw >> 3, Y_fw = Yfhw & 7;
	int yoffset = (Y_fh*OW + Y_fw - YIdx)*OC + Y_k;
	Ys[buf][tx][ty] = S1_SaveYs4(deltaY, Y_fh, Y_fw, yoffset, OH, OW,
		Y0, tih0, tiw0,
		Y1, tih1, tiw1,
		Y2, tih2, tiw2,
		Y3, tih3, tiw3);

	//load 4 elements from W[OC, FH, FW, IC]: Wr[IC, FH(r), FW(r), OC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);
	int WIdx = W_k / OC, W_oc = W_k - WIdx * OC;
	int woffset = (W_oc * 25 + WIDX_W55[WIdx])*IC;
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
		int YIdx = Y_k / OC; char Yfhw = YIDX_W55[YIdx];
		int Y_fh = Yfhw >> 3, Y_fw = Yfhw & 7;
		int yoffset = (Y_fh*OW + Y_fw - YIdx)*OC + Y_k;
		Ys[buf][tx][ty] = S1_SaveYs4(deltaY, Y_fh, Y_fw, yoffset, OH, OW,
			Y0, tih0, tiw0,
			Y1, tih1, tiw1,
			Y2, tih2, tiw2,
			Y3, tih3, tiw3);

		//load 4 elements from W[OC, FH, FW, IC]: Wr[IC, FH(r), FW(r), OC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int WIdx = W_k / OC, W_oc = W_k - WIdx * OC;
		int woffset = (W_oc * 25 + WIDX_W55[WIdx])*IC;
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


//(Y: BLOKC_SIZE*8, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0
//LB = 4, GK % 8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_KERNEL_8_8_S1_W5_X4_TEXTURE
#define DECONV3D_DX_ZERO_PADDING_KERNEL_8_8_S1_W5_X4_TEXTURE

#define k88s1W5x4_tex(stream, LB, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	zeroPadding_kernel_8_8_s1_W5_x4_texture<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W, deltaX,IH,IW, IC,OC, ph,pw, ic_index,j_index)

//k88<4>: Size = 1.5625, Time = 2.416 msec, Performace = 1388.84 GFlop/s
//k88<3>: Size = 1.5625, Time = 2.  9 msec, Performace = 1157.05 GFlop/s
//LB = 4: Size = 1.5625, Time = 2.398 msec, Performace = 1399.27 GFlop/s
//LB = 3: Size = 1.5625, Time = 2.754 msec, Performace = 1218.39 GFlop/s
template<int LB, int STEP>
__global__ void zeroPadding_kernel_8_8_s1_W5_x4_texture(
	cudaTextureObject_t deltaY, int OH, int OW,
	const float* __restrict__     W,
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
	int IH_IW = IH * IW;
	get_n_ih_iw(tj0, tn0, tih0, tiw0);
	const int oph = 4 - ph, opw = 4 - pw;//FH - 1 - ph
	tih0 = tih0 - oph, tiw0 = tiw0 - opw;
	const int tiw1 = tiw0 + 1, tiw2 = tiw0 + 2, tiw3 = tiw0 + 3;
	int Y1 = ((tn0*OH + tih0)*OW + tiw1)*OC;

	//prepare for GK = FH * FW * OC
	const int GK = 25 * OC;

	//load 4 elements from W[OC, FH, FW, IC]: Wr[IC, FH(r), FW(r), OC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);
	int WIdx = W_k / OC, W_oc = W_k - WIdx * OC;
	int woffset = (W_oc * 25 + WIDX_W55[WIdx])*IC;
	Ws[buf][ty][tx] = *(float4*)(W + woffset);

	//load 4 elements from deltaY[N, OH, OW, OC] : deltaYp[N, OH=OHp, OW=OWp, OC]
	int dY_k = tx - ((tx >= STEP) << LB >> 1);
	int YIdx = dY_k / OC; char Yfhw = YIDX_W55[YIdx];
	int dY_fh = Yfhw >> 3, dY_fw = Yfhw & 7;
	int yoffset = Y1 + (dY_fh*OW + dY_fw - YIdx)*OC + dY_k;
	Ys[buf][tx][ty] = S1_SaveYs4x_tex(deltaY, dY_fh, dY_fw, yoffset, OH, OW, OC,
		tih0, tiw0, tiw1, tiw2, tiw3);
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

		//load 4 elements from W[OC, FH, FW, IC] : Wr[IC, FH(r), FW(r), OC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int WIdx = W_k / OC, W_oc = W_k - WIdx * OC;
		int woffset = (W_oc * 25 + WIDX_W55[WIdx])*IC;
		Ws[buf][ty][tx] = *(float4*)(W + woffset);

		//load 4 elements from deltaY[N, OH, OW, OC] : deltaYp[N, OH=OHp, OW=OWp, OC]
		int dY_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int YIdx = dY_k / OC; char Yfhw = YIDX_W55[YIdx];
		int dY_fh = Yfhw >> 3, dY_fw = Yfhw & 7;
		int yoffset = Y1 + (dY_fh*OW + dY_fw - YIdx)*OC + dY_k;
		Ys[buf][tx][ty] = S1_SaveYs4x_tex(deltaY, dY_fh, dY_fw, yoffset, OH, OW, OC,
			tih0, tiw0, tiw1, tiw2, tiw3);
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


//(Y: BLOKC_SIZE*8, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0
//LB = 4, GK % 8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_KERNEL_8_8_S1_W3_X4_TEXTURE
#define DECONV3D_DX_ZERO_PADDING_KERNEL_8_8_S1_W3_X4_TEXTURE

#define k88s1W3x4_tex(stream, LB, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	zeroPadding_kernel_8_8_s1_W3_x4_texture<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W, deltaX,(IH*IW),IW, IC,OC, (2-ph),(2-pw),\
		    ic_index,j_index)

//k88<4>: Size = 0.5625, Time = 0.968 msec, Performace = 1247.89 GFlop/s
//k88<3>: Size = 0.5625, Time = 1.148 msec, Performace = 1052.23 GFlop/s
//LB = 4: Size = 0.5625, Time = 0.94  msec, Performace = 1285.06 GFlop/s
//LB = 3: Size = 0.5625, Time = 1.09  msec, Performace = 1108.22 GFlop/s
//LB = 4: Size = 1.125, Time = 1.656 msec, Performace = 1458.89 GFlop/s
template<int LB, int STEP>
__global__ void zeroPadding_kernel_8_8_s1_W3_x4_texture(
	cudaTextureObject_t deltaY, int OH, int OW,
	const float* __restrict__ W,//FH = FW = 3
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
	get_n_ih_iw(tj0, tn0, tih0, tiw0);
	tih0 = tih0 - oph, tiw0 = tiw0 - opw;
	const int tiw1 = tiw0 + 1, tiw2 = tiw0 + 2, tiw3 = tiw0 + 3;
	int Y1 = ((tn0*OH + tih0)*OW + tiw1)*OC;

	//prepare for GK = FH * FW * OC
	const int GK = 9 * OC;

	//load 4 elements from W[OC, FH, FW, IC]: Wr[IC, FH(r), FW(r), OC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);
	int WIdx = W_k / OC, W_oc = W_k - WIdx * OC;
	int woffset = (W_oc * 9 + WIDX_W33[WIdx])*IC;
	Ws[buf][ty][tx] = *(float4*)(W + woffset);

	//load 4 elements from deltaY[N, OH, OW, OC]: deltaYp[N, OH=OHp, OW=OWp, OC]
	int Y_k = tx - ((tx >= STEP) << LB >> 1);
	int YIdx = Y_k / OC; char Yfhw = YIDX_W33[YIdx];
	int Y_fh = Yfhw >> 2, Y_fw = Yfhw & 3;
	int yoffset = Y1 + (Y_fh*OW + Y_fw - YIdx)*OC + Y_k;
	Ys[buf][tx][ty] = S1_SaveYs4x_tex(deltaY, Y_fh, Y_fw, yoffset, OH, OW, OC,
		tih0, tiw0, tiw1, tiw2, tiw3);
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

		//load 4 elements from W[OC, FH, FW, IC]: Wr[IC, FH(r), FW(r), OC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int WIdx = W_k / OC, W_oc = W_k - WIdx * OC;
		int woffset = (W_oc * 9 + WIDX_W33[WIdx])*IC;
		Ws[buf][ty][tx] = *(float4*)(W + woffset);

		//load 4 elements from deltaY[N, OH, OW, OC]: deltaYp[N, OH=OHp, OW=OWp, OC]
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int YIdx = Y_k / OC; char Yfhw = YIDX_W33[YIdx];
		int Y_fh = Yfhw >> 2, Y_fw = Yfhw & 3;
		int yoffset = Y1 + (Y_fh*OW + Y_fw - YIdx)*OC + Y_k;
		Ys[buf][tx][ty] = S1_SaveYs4x_tex(deltaY, Y_fh, Y_fw, yoffset, OH, OW, OC,
			tih0, tiw0, tiw1, tiw2, tiw3);
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


//(Y: BLOKC_SIZE*8, X: BLOCK_SIZE*4), GK % (BLOCK_SIZE/2) == 0
//LB = 4, GK % 8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_KERNEL_8_4_S1_W2POW
#define DECONV3D_DX_ZERO_PADDING_KERNEL_8_4_S1_W2POW

#define k84s1_W2pow(stream, LB, ic_index, j_index, deltaY, OH, OW, W, LFH, LFW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	zeroPadding_kernel_8_4_s1_W2pow<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,LFH,LFW, deltaX,IH,IW, IC,OC, ph,pw, ic_index,j_index)

//LB = 4: Size = 1, Time = 1.678 msec, Performace = 1279.79 GFlop/s
//LB = 3: Size = 1, Time = 1.852 msec, Performace = 1159.55 GFlop/s
template<int LB, int STEP>
__global__ void zeroPadding_kernel_8_4_s1_W2pow(
	const float* __restrict__ deltaY, int OH, int OW,
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
	const int tic0 = ic0 + ((ty >= STEP) << 2);

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
	const int Ys_x = (tx >> 1), Ys_y = (ty << 1) + (tx & 1);
	float2 y; int yoffset = ((dY_fh*OW) + dY_fw)*OC + dY_oc;
	y.x = (ly0 ? deltaY[Y0 + yoffset] : 0);;
	y.y = (ly1 ? deltaY[Y1 + yoffset] : 0);
	Ys[buf][Ys_x][Ys_y] = y;

	//load 4 elements from W[OC, FH, FW, IC]: Wr[IC, FH(r), FW(r), OC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);
	int W_oc, Wr_fh_fw; get_W_oc_fh_fw_W2pow(W_k, W_oc, Wr_fh_fw);
	Ws[buf][ty][tx] = *(float4*)(&W[((W_oc << LFH_FW) + Wr_fh_fw)*IC + tic0]);
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
		y.x = (ly0 ? deltaY[Y0 + yoffset] : 0);
		y.y = (ly1 ? deltaY[Y1 + yoffset] : 0);
		Ys[buf][Ys_x][Ys_y] = y;

		//load 4 elements from W[OC, FH, FW, IC]: Wr[IC, FH(r), FW(r), OC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int W_oc, Wr_fh_fw; get_W_oc_fh_fw_W2pow(W_k, W_oc, Wr_fh_fw);
		Ws[buf][ty][tx] = *(float4*)(&W[((W_oc << LFH_FW) + Wr_fh_fw)*IC + tic0]);
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

	j0 = j0 * IC + ic0;//j0 = ((n * OH + oh) * OW + ow) * IC + ic
	int j1 = j0 + IC, j2 = j1 + IC, j3 = j2 + IC;

	*(float4*)(deltaX + j0) = v0; *(float4*)(deltaX + j0 + 4) = v1;
	*(float4*)(deltaX + j1) = v2; *(float4*)(deltaX + j1 + 4) = v3;
	*(float4*)(deltaX + j2) = v4; *(float4*)(deltaX + j2 + 4) = v5;
	*(float4*)(deltaX + j3) = v6; *(float4*)(deltaX + j3 + 4) = v7;
}

#endif


//(Y: BLOKC_SIZE*4s, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0
//LB = 4, GK % 8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_KERNEL_4_8_S1_W2POW
#define DECONV3D_DX_ZERO_PADDING_KERNEL_4_8_S1_W2POW

#define k48s1_W2pow(stream, LB, ic_index, j_index, deltaY, OH, OW, W, LFH, LFW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	zeroPadding_kernel_4_8_s1_W2pow<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>2, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,LFH,LFW, deltaX,IH,IW, IC,OC, ph,pw, ic_index,j_index)

//LB = 4: Size = 1, Time = 1.62  msec, Performace = 1325.61 GFlop/s
//LB = 3: Size = 1, Time = 2.078 msec, Performace = 1033.44 GFlop/s
template<int LB, int STEP>
__global__ void zeroPadding_kernel_4_8_s1_W2pow(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__      W, int LFH, int LFW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int ph, int pw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB >> 1][(2 << LB) + 2];//follow k44
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

	//load 4 elements from deltaY[N, OH, OW, OC]: deltaYp[N, OH=OHp, OW=OWp, OC]
	int dY_k = tx - ((tx >= STEP) << LB >> 1);
	int dY_fh, dY_fw, dY_oc; get_dY_oc_fh_fw_W2pow(dY_k, dY_oc, dY_fh, dY_fw);
	int yoffset = ((dY_fh*OW) + dY_fw)*OC + dY_oc;
	Ys[buf][tx][ty] = S1_SaveYs4(deltaY, dY_fh, dY_fw, yoffset, OH, OW,
		Y0, tih0, tiw0,
		Y1, tih1, tiw1,
		Y2, tih2, tiw2,
		Y3, tih3, tiw3);

	//load 2 elements from W[OC, FH, FW, IC]: Wr[IC, FH(r), FW(r), OC]
	int W_k = ty >> 1;
	int W_oc, Wr_fh_fw; get_W_oc_fh_fw_W2pow(W_k, W_oc, Wr_fh_fw);
	const int Ws_x = (ty >> 1), Ws_y = (tx << 1) + (ty & 1);
	Ws[buf][Ws_x][Ws_y] = *(float2*)(&W[((W_oc << LFH_FW) + Wr_fh_fw)*IC]);
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
			float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];
			float4 w0 = *(float4*)(&Ws[buf][ik][tx << 1]);

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

		//load 4 elements from deltaY[N, OH, OW, OC]: deltaYp[N, OH=OHp, OW=OWp, OC]
		int dY_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int dY_fh, dY_fw, dY_oc; get_dY_oc_fh_fw_W2pow(dY_k, dY_oc, dY_fh, dY_fw);
		int yoffset = ((dY_fh*OW) + dY_fw)*OC + dY_oc;
		Ys[buf][tx][ty] = S1_SaveYs4(deltaY, dY_fh, dY_fw, yoffset, OH, OW,
			Y0, tih0, tiw0,
			Y1, tih1, tiw1,
			Y2, tih2, tiw2,
			Y3, tih3, tiw3);

		//load 2 elements from W[OC, FH, FW, IC]: Wr[IC, FH(r), FW(r), OC]
		int W_k = ((ok << LB) + ty) >> 1;
		int W_oc, Wr_fh_fw; get_W_oc_fh_fw_W2pow(W_k, W_oc, Wr_fh_fw);
		Ws[buf][Ws_x][Ws_y] = *(float2*)(&W[((W_oc << LFH_FW) + Wr_fh_fw)*IC]);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];
		float4 w0 = *(float4*)(&Ws[buf][ik][tx << 1]);

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


//(Y: BLOKC_SIZE*8, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0
//LB = 4, GK % 8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_KERNEL_4_2_S1
#define DECONV3D_DX_ZERO_PADDING_KERNEL_4_2_S1

#define k42s1(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	kernel_4_2_s1<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>1, GN>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, ph,pw, ic_index,j_index)

//LB = 4: Size = 1, Time = 3.692 msec, Performace = 581.659 GFlop/s
//LB = 3: Size = 1, Time = 6.02  msec, Performace = 356.725 GFlop/s
template<int LB, int STEP>
__global__ void kernel_4_2_s1(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int ph, int pw,
	int ic_index, int j_index)
{
	int by = blockIdx.y, bx = blockIdx.x;
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2  Ws[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float  dYs[2][1 << LB >> 1][(2 << LB) + 2];

	//prepare for GN = IC
	const int ic0 = (((by << LB) + ty) << 2) + ic_index;
	const int tic0 = ((tx & 1) << 1) + ic0;

	//prepare for GM = N * IH * IW
	int j0 = (((bx << LB) + tx) << 1) + j_index;
	int tj0 = j0 + (ty & 1);
	const int IH_IW = IH * IW;
	get_n_ih_iw(tj0, tn0, tih0, tiw0);
	const int oph = FH - ph - 1, opw = FW - pw - 1;
	tih0 = tih0 - oph, tiw0 = tiw0 - opw;
	const int Yoffset0 = ((tn0*OH + tih0)*OW + tiw0)*OC;

	//prepare for GK = FH * FW * OC
	const int FH_FW = FH * FW, GK = FH_FW * OC;

	//load 2 elements from W[OC, FH, FW, IC] : Wr[IC, FH(r), FW(r), OC]
	int W_k = tx >> 1, W_oc, Wr_fh_fw; //Ws: with the same ty
	get_W_oc_fh_fw(W_k, W_oc, Wr_fh_fw);
	const int Ws_x = (tx >> 1), Ws_y = (ty << 1) + (tx & 1);
	Ws[buf][Ws_x][Ws_y] = *(float2*)(&get3d(W, W_oc, Wr_fh_fw, tic0, FH_FW, IC));

	//load 1 elements from deltaY[N, OH, OW, OC] : deltaYp[N, OH=OHp, OW=OWp, OC]
	int dY_k = ty >> 1, dY_fh, dY_fw, dY_oc;
	get_dY_oc_fh_fw(dY_k, dY_oc, dY_fh, dY_fw);
	bool ly0 = (tih0 >= -dY_fh) && (tih0 < OH - dY_fh) && (tiw0 >= -dY_fw) && (tiw0 < OW - dY_fw);
	const int dYs_y = (ty >> 1), dYs_x = (tx << 1) + (ty & 1);
	int yoffset = ((dY_fh*OW) + dY_fw)*OC + dY_oc;
	dYs[buf][dYs_y][dYs_x] = (ly0 ? deltaY[Yoffset0 + yoffset] : 0);
	__syncthreads();

	//compute area----------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0);
	float4 v1 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float2 dy = *(float2*)(&dYs[buf][ik][tx << 1]);
			float4 w = *(float4*)(&Ws[buf][ik][ty << 1]);
			simdMM4(v0, dy.x, w);
			simdMM4(v1, dy.y, w);
		}
		buf ^= 1;

		//load 2 elements from W[OC, FH, FW, IC] : Wr[IC, FH(r), FW(r), OC]
		int W_k = ((ok << LB) + tx) >> 1;
		get_W_oc_fh_fw(W_k, W_oc, Wr_fh_fw);
		Ws[buf][Ws_x][Ws_y] = *(float2*)(&get3d(W, W_oc, Wr_fh_fw, tic0, FH_FW, IC));

		//load 1 element from deltaY[N, OH, OW, OC] : deltaYp[N, OH=OHp, OW=OWp, OC]
		int dY_k = ((ok << LB) + ty) >> 1, dY_fh, dY_fw, dY_oc;
		get_dY_oc_fh_fw(dY_k, dY_oc, dY_fh, dY_fw);
		bool ly0 = (tih0 >= -dY_fh) && (tih0 < OH - dY_fh) && (tiw0 >= -dY_fw) && (tiw0 < OW - dY_fw);
		int yoffset = ((dY_fh*OW) + dY_fw)*OC + dY_oc;
		dYs[buf][dYs_y][dYs_x] = (ly0 ? deltaY[Yoffset0 + yoffset] : 0);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float2 dy = *(float2*)(&dYs[buf][ik][tx << 1]);
		float4 w = *(float4*)(&Ws[buf][ik][ty << 1]);
		simdMM4(v0, dy.x, w);
		simdMM4(v1, dy.y, w);
	}

	j0 = j0 * IC + ic0;//j0 = ((n * OH + oh) * OW + ow) * IC + ic
	int j1 = j0 + IC;

	*(float4*)(deltaX + j0) = v0;
	*(float4*)(deltaX + j1) = v1;
}

#endif


//(Y: BLOKC_SIZE*8, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0
//LB = 4, GK % 8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_KERNEL_2_4_S1
#define DECONV3D_DX_ZERO_PADDING_KERNEL_2_4_S1


#define k24s1(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	kernel_2_4_s1<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>2, GN>>LB>>1), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, ph,pw, ic_index,j_index)

//LB = 4: Size = 1, Time = 4.238 msec, Performace = 506.721 GFlop/s
//LB = 3: Size = 1, Time = 6.738 msec, Performace = 318.712 GFlop/s
template<int LB, int STEP>
__global__ void kernel_2_4_s1(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int ph, int pw,
	int ic_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float   Ws[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2 dYs[2][1 << LB >> 1][(2 << LB) + 2];

	//prepare for GN = IC
	const int ic0 = (((blockIdx.y << LB) + ty) << 1) + ic_index;
	const int tic0 = (tx & 1) + ic0;

	//prepare for GM = N * IH * IW
	int j0 = (((blockIdx.x << LB) + tx) << 2) + j_index;
	int tj0 = j0 + ((ty & 1) << 1), tj1 = tj0 + 1;
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

	//load 1 element from W[OC, FH, FW, IC] : Wr[IC, FH(r), FW(r), OC]
	int W_k = tx >> 1, W_oc, Wr_fh_fw; //Ws: with the same ty
	get_W_oc_fh_fw(W_k, W_oc, Wr_fh_fw);
	const int Ws_x = (tx >> 1), Ws_y = (ty << 1) + (tx & 1);
	Ws[buf][Ws_x][Ws_y] = get3d(W, W_oc, Wr_fh_fw, tic0, FH_FW, IC);

	//load 2 elements from deltaY[N, OH, OW, OC] : deltaYp[N, OH=OHp, OW=OWp, OC]
	int dY_k = ty >> 1, dY_fh, dY_fw, dY_oc;
	get_dY_oc_fh_fw(dY_k, dY_oc, dY_fh, dY_fw);
	bool ly0 = (tih0 >= -dY_fh) && (tih0 < OH - dY_fh) && (tiw0 >= -dY_fw) && (tiw0 < OW - dY_fw);
	bool ly1 = (tih1 >= -dY_fh) && (tih1 < OH - dY_fh) && (tiw1 >= -dY_fw) && (tiw1 < OW - dY_fw);
	const int dYs_y = (ty >> 1), dYs_x = (tx << 1) + (ty & 1);
	int yoffset = ((dY_fh*OW) + dY_fw)*OC + dY_oc;
	dYs[buf][dYs_y][dYs_x].x = (ly0 ? deltaY[Yoffset0 + yoffset] : 0);
	dYs[buf][dYs_y][dYs_x].y = (ly1 ? deltaY[Yoffset1 + yoffset] : 0);
	__syncthreads();

	//compute area----------------------------------------------------
	float2 v0 = make_float2(0, 0);
	float2 v1 = make_float2(0, 0);
	float2 v2 = make_float2(0, 0);
	float2 v3 = make_float2(0, 0);
	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 dy = *(float4*)(&dYs[buf][ik][tx << 1]);
			float2 w = *(float2*)(&Ws[buf][ik][ty << 1]);

			simdMM2(v0, dy.x, w);
			simdMM2(v1, dy.y, w);
			simdMM2(v2, dy.z, w);
			simdMM2(v3, dy.w, w);
		}

		buf ^= 1;
		//load 1 element from W[OC, FH, FW, IC] : Wr[IC, FH(r), FW(r), OC]
		int W_k = ((ok << LB) + tx) >> 1;
		get_W_oc_fh_fw(W_k, W_oc, Wr_fh_fw);
		Ws[buf][Ws_x][Ws_y] = get3d(W, W_oc, Wr_fh_fw, tic0, FH_FW, IC);

		//load 2 elements from deltaY[N, OH, OW, OC] : deltaYp[N, OH=OHp, OW=OWp, OC]
		int dY_k = ((ok << LB) + ty) >> 1, dY_fh, dY_fw, dY_oc;
		get_dY_oc_fh_fw(dY_k, dY_oc, dY_fh, dY_fw);
		bool ly0 = (tih0 >= -dY_fh) && (tih0 < OH - dY_fh) && (tiw0 >= -dY_fw) && (tiw0 < OW - dY_fw);
		bool ly1 = (tih1 >= -dY_fh) && (tih1 < OH - dY_fh) && (tiw1 >= -dY_fw) && (tiw1 < OW - dY_fw);
		int yoffset = ((dY_fh*OW) + dY_fw)*OC + dY_oc;
		dYs[buf][dYs_y][dYs_x].x = (ly0 ? deltaY[Yoffset0 + yoffset] : 0);
		dYs[buf][dYs_y][dYs_x].y = (ly1 ? deltaY[Yoffset1 + yoffset] : 0);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 dy = *(float4*)(&dYs[buf][ik][tx << 1]);
		float2 w = *(float2*)(&Ws[buf][ik][ty << 1]);

		simdMM2(v0, dy.x, w);
		simdMM2(v1, dy.y, w);
		simdMM2(v2, dy.z, w);
		simdMM2(v3, dy.w, w);
	}

	j0 = j0 * IC + ic0; //j0 = ((n * OH + oh) * OW + ow) * IC + ic
	int j1 = j0 + IC, j2 = j1 + IC, j3 = j2 + IC;

	*(float2*)(deltaX + j0) = v0;
	*(float2*)(deltaX + j1) = v1;
	*(float2*)(deltaX + j2) = v2;
	*(float2*)(deltaX + j3) = v3;
}

#endif