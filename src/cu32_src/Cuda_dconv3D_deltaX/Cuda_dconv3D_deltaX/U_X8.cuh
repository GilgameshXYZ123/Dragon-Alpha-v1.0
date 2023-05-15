#ifndef CREATE_GK_INDEX
#define CREATE_GK_INDEX

//LB = log2(BLOCK_SIZE)
//lengthv = CFH * CFW * OC * IC

#define create_GK_Index4(stream, LB, LT, Index, FW, OC, lengthv)\
	create_GK_index_kernel_2\
		<<< (lengthv>>LB>>LT), 1<<LB, 0, stream >>>\
			(Index, FW, OC, lengthv)

#define create_GK_Index4_small(stream, Index, FW, OC, lengthv)\
	create_GK_index_kernel_2\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(Index, FW, OC, lengthv)

//GK = [FH, FW, OC]
//OC % 4 == 0 -> lengthv % 4 == 0
__global__ void create_GK_index_kernel_2(
	char* __restrict__ Index,
	int FW, int OC, int lengthv)
{
	int step = gridDim.x*blockDim.x, step2 = step << 1;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	int FW_OC = FW * OC;
	for (int index2 = index << 1; index2 < lengthv; index2 += step2)
	{
		int k1 = index2;
		int fh1 = k1 / FW_OC; k1 %= FW_OC;
		int fw1 = k1 / OC;

		int k2 = index2 + 1;
		int fh2 = k2 / FW_OC; k2 %= FW_OC;
		int fw2 = k2 / OC;

		char4 Idx = make_char4(fh1, fw1, fh2, fw2);
		*(char4*)(Index + (index2 << 1)) = Idx;
	}
}

void __create_GK_Index(cudaStream_t stream,
	char* Index, int FH, int FW, int OC)
{
	int lengthv = FH * FW * OC;
	if (lengthv < 256) { create_GK_Index4_small(stream, Index, FW, OC, lengthv); return; }
	create_GK_Index4(stream, 5, 2, Index, FW, OC, lengthv);
}


#endif


#ifndef XKERNEL11
#define XKERNEL11

#define xkernel11(stream, LB, ic_index, j_index, Index, deltaY,OH,OW, W,FH,FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	BXkernel11<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			((const char2*)Index, deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, ph,pw, ic_index,j_index)

//LB = 4: Size = 1.125, Time = 1.714 msec, Performace = 1409.52 GFlop/s
template<int LB, int STEP>
__global__ void BXkernel11(const char2* __restrict__ Index,
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
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

	//load 4 elements from deltaY[N, OH, OW, OC]
	int Y_oc = tx - ((tx >= STEP) << LB >> 1);
	const int SY = (OW - FW)*OC;
	Ys[buf][tx][ty] = S1_SaveYs4(deltaY, 0, 0, Y_oc, OH, OW,
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

		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		char2 Idx = *(char2*)(Index + Y_k);
		int fh = Idx.x, fw = Idx.y;

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int W_oc = W_k - fh * FW_OC - fw * OC;
		int woffset = ((W_oc*FH - fh)*FW - fw)*IC;
		Ws[buf][ty][tx] = *(float4*)(W + woffset);

		//load 4 elements from deltaY[N, OH, OW, OC]
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

#define xkernel10(stream, LB, ic_index, j_index, Index, deltaY,OH,OW, W,FH,FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	BXkernel10<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			((const char2*)Index, deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, ph,pw, ic_index,j_index)

//LB = 4: Size = 1.125, Time = 1.714 msec, Performace = 1409.52 GFlop/s
template<int LB, int STEP>
__global__ void BXkernel10(const char2* __restrict__ Index,
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
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
		char2 Idx = *(char2*)(Index + Y_k);
		int fh = Idx.x, fw = Idx.y; Y_k -= fh * FW_OC;

		//int fh = Y_k / FW_OC, fw = (Y_k -= fh * FW_OC) / OC;
		//{
		//	char2 Idx = *(char2*)(Index + oY_k);
		//	int fh2 = Idx.x, fw2 = Idx.y;
		//	if(fh2 != fh || fw2 != fw)
		//	printf("%d, %d, %d, %d\n", fh, fw, fh2, fw2);
		//}

		int yoffset = fh * OW_OC + Y_k;
		Ys[buf][tx][ty] = S1_SaveYs4(deltaY, fh, fw, yoffset, OH, OW,
			Y0, tih0, tiw0,
			Y1, tih1, tiw1,
			Y2, tih2, tiw2,
			Y3, tih3, tiw3);

		//load 4 elements from W[OC, FH, FW, IC]
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
