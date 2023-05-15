#pragma once

#ifndef DECONV3D_DX_ZERO_PADDING_KERNEL_S1_EX2_H
#define DECONV3D_DX_ZERO_PADDING_KERNEL_S1_EX2_H


//Unsparse Matrix Method
//We have:
//(1) FH*FW >= 2
//(2) GN = IC;           GN >= 4, GN%4 == 0
//(3) GM = N  * IH * IW; GM >= 4, GM%4 == 0
//(4) GK = OC * FH * FW; GK >= 8, GK%4 == 0
//(5) sh = 1, sw = 1

#ifndef DECONV3D_DX_ZERO_PADDING_KERNEL_S1_EX2_CALL
#define DECONV3D_DX_ZERO_PADDING_KERNEL_S1_EX2_CALL

#define k88s1W3(stream, LB, ic_index, j_index, deltaY, OH, OW, W, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	zeroPadding_kernel_8_8_s1_W3<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W, deltaX,IH,IW, IC,OC, ph,pw, ic_index,j_index)

#endif

//=====FH = 3, FW = 3=========================================
//(Y: BLOKC_SIZE*8, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0
//LB = 4, GK % 8 == 0
#ifndef DECONV3D_DX_ZERO_PADDING_KERNEL_8_8_S1_W3
#define DECONV3D_DX_ZERO_PADDING_KERNEL_8_8_S1_W3

//LB = 4: Size = 0.5625, Time = 0.902 msec, Performace = 1339.2 GFlop/s
//LB = 3: Size = 0.5625, Time = 1.03 msec, Performace = 1172.78 GFlop/s
template<int LB, int STEP>
__global__ void zesroPadding_kernel_8_8_s1_W3(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__      W,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int ph, int pw,
	int ic_index, int j_index)#ifndef DECONV3D_DX_ZERO_PADDING_KERNEL_8_8_S1_W3
#define DECONV3D_DX_ZERO_PADDING_KERNEL_8_8_S1_W3

	//k88<4>: Size = 0.5625, Time = 0.968 msec, Performace = 1247.89 GFlop/s
	//k88<3>: Size = 0.5625, Time = 1.148 msec, Performace = 1052.23 GFlop/s
	//LB = 4: Size = 0.5625, Time = 0.932 msec, Performace = 1296.09 GFlop/s
	//LB = 3: Size = 0.5625, Time = 1.082 msec, Performace = 1116.41 GFlop/s
	template<int LB, int STEP>
__global__ void zeroPadding_kernel_8_8_s1_W3(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__      W,
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
	const int oph = 2 - ph, opw = 2 - pw;
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

	//load 4 elements from deltaY[N, OH, OW, OC] : deltaYp[N, OH=OHp, OW=OWp, OC]
	int dY_k = tx - ((tx >= STEP) << LB >> 1);
	int YIdx = dY_k / OC; char Yfhw = YIDX_W3[YIdx];
	int dY_fh = Yfhw >> 2, dY_fw = Yfhw & 3;
	int yoffset = (dY_fh*OW + dY_fw - YIdx)*OC + dY_k;
	Ys[buf][tx][ty] = S1_SaveYs4(deltaY, dY_fh, dY_fw, yoffset, OH, OW,
		Y0, tih0, tiw0,
		Y1, tih1, tiw1,
		Y2, tih2, tiw2,
		Y3, tih3, tiw3);

	//load 4 elements from W[OC, FH, FW, IC] : Wr[IC, FH(r), FW(r), OC]
	int W_k = ty - ((ty >= STEP) << LB >> 1); //Ws: with the same ty
	int WIdx = W_k / OC, W_oc = W_k - WIdx * OC;
	int woffset = (W_oc * 9 + WIDX_W3[WIdx])*IC;
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

		//yoffset = (dY_fh*OW + dY_fw)*OC + dY_oc
		//= (dY_fh*OW + dY_fw)*OC + dY_k - YIdx * OC
		//= (dY_fh*OW + dY_fw - YIdx)*OC + dY_k

		//load 4 elements from deltaY[N, OH, OW, OC] : deltaYp[N, OH=OHp, OW=OWp, OC]
		int dY_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int YIdx = dY_k / OC; char Yfhw = YIDX_W3[YIdx];
		int dY_fh = Yfhw >> 2, dY_fw = Yfhw & 3;
		int yoffset = (dY_fh*OW + dY_fw - YIdx)*OC + dY_k;
		Ys[buf][tx][ty] = S1_SaveYs4(deltaY, dY_fh, dY_fw, yoffset, OH, OW,
			Y0, tih0, tiw0,
			Y1, tih1, tiw1,
			Y2, tih2, tiw2,
			Y3, tih3, tiw3);

		//load 4 elements from W[OC, FH, FW, IC] : Wr[IC, FH(r), FW(r), OC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int WIdx = W_k / OC, W_oc = W_k - WIdx * OC;
		int woffset = (W_oc * 9 + WIDX_W3[WIdx])*IC;
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

#endif


#endif