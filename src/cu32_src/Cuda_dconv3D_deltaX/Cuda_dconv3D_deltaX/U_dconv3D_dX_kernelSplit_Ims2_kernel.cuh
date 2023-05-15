#pragma once

//IMS2: INPUT_MOD_STEP_2
//sh = sw = 2
//(IH, IW) % (sh, sw) == 0
#ifndef DCONV3D_DX_KERNEL_SPLIT_IMS2_KERNEL_H
#define DCONV3D_DX_KERNEL_SPLIT_IMS2_KERNEL_H


#ifndef DCONV3D_DX_KERNEL_SPLIT_IMS2_CALL
#define DCONV3D_DX_KERNEL_SPLIT_IMS2_CALL

//LB = log2(BLOCK_SIZE)

#define ksIms2_k88(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	kernelSplit_Ims2_kernel_8_8<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>3), (GM>>LB>>3), 4), dim3(1<<LB, 1<<LB) >>>\
			(deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw,\
			 CFH, CFW, IH_slice, IW_slice, GK, ic_index, j_index)

#define ksIms2_k84(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	kernelSplit_Ims2_kernel_8_4<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>3), (GM>>LB>>2), 4), dim3(1<<LB, 1<<LB) >>>\
			(deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw,\
			 CFH, CFW, IH_slice, IW_slice, GK, ic_index, j_index)

#define ksIms2_k48(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	kernelSplit_Ims2_kernel_4_8<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>2), (GM>>LB>>3), 4), dim3(1<<LB, 1<<LB) >>>\
			(deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw,\
			 CFH, CFW, IH_slice, IW_slice, GK, ic_index, j_index)

#define ksIms2_k44(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	kernelSplit_Ims2_kernel_4_4<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>2), (GM>>LB>>2), 4), dim3(1<<LB, 1<<LB) >>>\
			(deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw,\
			 CFH, CFW, IH_slice, IW_slice, GK, ic_index, j_index)

#define ksIms2_k42(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	kernelSplit_Ims2_kernel_4_2<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>2), (GM>>LB>>1), 4), dim3(1<<LB, 1<<LB) >>>\
			(deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw,\
			 CFH, CFW, IH_slice, IW_slice, GK, ic_index, j_index)

#define ksIms2_k24(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	kernelSplit_Ims2_kernel_2_4<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>1), (GM>>LB>>2), 4), dim3(1<<LB, 1<<LB) >>>\
			(deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw,\
			 CFH, CFW, IH_slice, IW_slice, GK, ic_index, j_index)

#define ksIms2_k22(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	kernelSplit_Ims2_kernel_2_2<LB, (1<<LB)>\
		<<< dim3((GN>>LB>>1), (GM>>LB>>1), 4), dim3(1<<LB, 1<<LB) >>>\
			(deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw,\
			 CFH, CFW, IH_slice, IW_slice, GK, ic_index, j_index)

#define ksIms2_k21(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	kernelSplit_Ims2_kernel_2_1<LB, (1<<LB)>\
		<<< dim3((GN>>LB>>1), (GM>>LB), 4), dim3(1<<LB, 1<<LB) >>>\
			(deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw,\
			 CFH, CFW, IH_slice, IW_slice, GK, ic_index, j_index)

#define ksIms2_k12(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	kernelSplit_Ims2_kernel_1_2<LB, (1<<LB)>\
		<<< dim3((GN>>LB), (GM>>LB>>1), 4), dim3(1<<LB, 1<<LB) >>>\
			(deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw,\
			 CFH, CFW, IH_slice, IW_slice, GK, ic_index, j_index)

#define ksIms2_k11(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	kernelSplit_Ims2_kernel_1_1<LB, (1<<LB)>\
		<<< dim3((GN>>LB), (GM>>LB), 4), dim3(1<<LB, 1<<LB) >>>\
			(deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw,\
			 CFH, CFW, IH_slice, IW_slice, GK, ic_index, j_index)

#endif


//(Y: BLOCK_SIZE*8, X:BLOCK_SIZE*8), K >= (BLOCK_SIZE/2)
#ifndef DCONV3D_DX_KERNEL_SPLIT_IMS2_KERNEL_8_8
#define DCONV3D_DX_KERNEL_SPLIT_IMS2_KERNEL_8_8

//LB = 4: Size = 1.125, Time = 1.228 msec, Performace = 1967.36 GFlop/s
//LB = 3: Size = 1.125, Time = 1.29  msec, Performace = 1872.81 GFlop/s
template<int LB, int STEP>
__global__ void kernelSplit_Ims2_kernel_8_8(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int ph, int pw,
	int CFH, int CFW,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];

	//prepare for GZ = sh*sw
	int y = blockIdx.z >> 1, x = blockIdx.z & 1;
	ph = ph - y; pw = pw - x;//if(ihs < 0): ihs += (ph + 1)/sh*sh
	int ihs = -ph; ihs += -(ihs < 0) & ((ph + 1) << 1 >> 1);
	int iws = -pw; iws += -(iws < 0) & ((pw + 1) << 1 >> 1);

	//prepare for GN = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	W += ic0 + ((ty >= STEP) << 2);//W[0, 0, 0, tic0]
	deltaX += ic0; //deltaX[0, 0, 0, ic0]

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int j1 = j0 + 1, j2 = j0 + 2, j3 = j0 + 3;
	int j4 = j0 + 4, j5 = j0 + 5, j6 = j0 + 6, j7 = j0 + 7;
	int IW_slice = IW >> 1;
	int IH_IW_slice = (IH >> 1) * IW_slice;
	Ims2_n_ih_iw(j0, n0, ih0, iw0);
	Ims2_n_ih_iw(j1, n1, ih1, iw1);
	Ims2_n_ih_iw(j2, n2, ih2, iw2);
	Ims2_n_ih_iw(j3, n3, ih3, iw3);
	Ims2_n_ih_iw(j4, n4, ih4, iw4);
	Ims2_n_ih_iw(j5, n5, ih5, iw5);
	Ims2_n_ih_iw(j6, n6, ih6, iw6);
	Ims2_n_ih_iw(j7, n7, ih7, iw7);
	int Xoffset0 = ((n0*IH + ih0)*IW + iw0)*IC;
	int Xoffset1 = ((n1*IH + ih1)*IW + iw1)*IC;
	int Xoffset2 = ((n2*IH + ih2)*IW + iw2)*IC;
	int Xoffset3 = ((n3*IH + ih3)*IW + iw3)*IC;
	int Xoffset4 = ((n4*IH + ih4)*IW + iw4)*IC;
	int Xoffset5 = ((n5*IH + ih5)*IW + iw5)*IC;
	int Xoffset6 = ((n6*IH + ih6)*IW + iw6)*IC;
	int Xoffset7 = ((n7*IH + ih7)*IW + iw7)*IC;

	bool flagX = (tx >= STEP);
	int oph = CFH - 1, opw = CFW - 1;
	int tohs0 = ((choose_int(flagX, ih4, ih0) + ph) >> 1) - oph;
	int tohs1 = ((choose_int(flagX, ih5, ih1) + ph) >> 1) - oph;
	int tohs2 = ((choose_int(flagX, ih6, ih2) + ph) >> 1) - oph;
	int tohs3 = ((choose_int(flagX, ih7, ih3) + ph) >> 1) - oph;
	int tows0 = ((choose_int(flagX, iw4, iw0) + pw) >> 1) - opw;
	int tows1 = ((choose_int(flagX, iw5, iw1) + pw) >> 1) - opw;
	int tows2 = ((choose_int(flagX, iw6, iw2) + pw) >> 1) - opw;
	int tows3 = ((choose_int(flagX, iw7, iw3) + pw) >> 1) - opw;
	int Yoffset0 = ((choose_int(flagX, n4, n0)*OH + tohs0)*OW + tows0)*OC;
	int Yoffset1 = ((choose_int(flagX, n5, n1)*OH + tohs1)*OW + tows1)*OC;
	int Yoffset2 = ((choose_int(flagX, n6, n2)*OH + tohs2)*OW + tows2)*OC;
	int Yoffset3 = ((choose_int(flagX, n7, n3)*OH + tohs3)*OW + tows3)*OC;

	const int CFW_OC = CFW * OC;

	//load 4 elem from deltaY[N, FH, FW, OC]
	int dY_k = tx - ((tx >= STEP) << LB >> 1);
	Ims_fhr_fwr(dY_k, dY_fhr, dY_fwr);
	int OW_OC = OW * OC, yoffset = dY_fhr * OW_OC + dY_k;
	Ys[buf][tx][ty].x = (Ims_ldy(tohs0, tows0) ? deltaY[Yoffset0 + yoffset] : 0);
	Ys[buf][tx][ty].y = (Ims_ldy(tohs1, tows1) ? deltaY[Yoffset1 + yoffset] : 0);
	Ys[buf][tx][ty].z = (Ims_ldy(tohs2, tows2) ? deltaY[Yoffset2 + yoffset] : 0);
	Ys[buf][tx][ty].w = (Ims_ldy(tohs3, tows3) ? deltaY[Yoffset3 + yoffset] : 0);

	//load 4 elem from W[OC, FH, FW, IC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);
	Ims_fhr_fwr(W_k, W_fhr, W_fwr);
	int fh = y + ((oph - W_fhr) << 1);//fh = y + (CFH - 1 - W_fhr)*sh;
	int fw = x + ((opw - W_fwr) << 1);
	bool lw = (fh < FH) && (fw < FW);
	int Woffset = (((W_k - W_fwr * OC)*FH + fh)*FW + fw)*IC;
	Ws[buf][ty][tx] = (lw ? *(float4*)(W + Woffset) : FLOAT_ZERO4);
	__syncthreads();

	//compure_area----------------------------------------------
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
#pragma once
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
		buf ^= 1;

		//load 4 elem from deltaY[N, FH, FW, OC]
		int dY_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		Ims_fhr_fwr(dY_k, dY_fhr, dY_fwr);
		int yoffset = dY_fhr * OW_OC + dY_k;
		Ys[buf][tx][ty].x = (Ims_ldy(tohs0, tows0) ? deltaY[Yoffset0 + yoffset] : 0);
		Ys[buf][tx][ty].y = (Ims_ldy(tohs1, tows1) ? deltaY[Yoffset1 + yoffset] : 0);
		Ys[buf][tx][ty].z = (Ims_ldy(tohs2, tows2) ? deltaY[Yoffset2 + yoffset] : 0);
		Ys[buf][tx][ty].w = (Ims_ldy(tohs3, tows3) ? deltaY[Yoffset3 + yoffset] : 0);

		//load 4 elem from W[OC, FH, FW, IC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		Ims_fhr_fwr(W_k, W_fhr, W_fwr);
		int fh = y + ((oph - W_fhr) << 1);//fh = y + (oph - fhr)*sh
		int fw = x + ((opw - W_fwr) << 1);//fw = x + (opw - fwr)*sw
		bool lw = (fh < FH) && (fw < FW);//W_oc = W_k - W_fwr * OC
		int Woffset = (((W_k - W_fwr * OC)*FH + fh)*FW + fw)*IC;
		Ws[buf][ty][tx] = (lw ? *(float4*)(W + Woffset) : FLOAT_ZERO4);
		__syncthreads();
	}
#pragma once
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

	*(float4*)(deltaX + Xoffset0) = v0;  *(float4*)(deltaX + Xoffset0 + 4) = v1;
	*(float4*)(deltaX + Xoffset1) = v2;  *(float4*)(deltaX + Xoffset1 + 4) = v3;
	*(float4*)(deltaX + Xoffset2) = v4;  *(float4*)(deltaX + Xoffset2 + 4) = v5;
	*(float4*)(deltaX + Xoffset3) = v6;  *(float4*)(deltaX + Xoffset3 + 4) = v7;
	*(float4*)(deltaX + Xoffset4) = v8;  *(float4*)(deltaX + Xoffset4 + 4) = v9;
	*(float4*)(deltaX + Xoffset5) = v10; *(float4*)(deltaX + Xoffset5 + 4) = v11;
	*(float4*)(deltaX + Xoffset6) = v12; *(float4*)(deltaX + Xoffset6 + 4) = v13;
	*(float4*)(deltaX + Xoffset7) = v14; *(float4*)(deltaX + Xoffset7 + 4) = v15;
}

#endif


//(Y: BLOCK_SIZE*8, X:BLOCK_SIZE*4), K >= (BLOCK_SIZE/2)
#ifndef DCONV3D_DX_KERNEL_SPLIT_IMS2_KERNEL_8_4
#define DCONV3D_DX_KERNEL_SPLIT_IMS2_KERNEL_8_4

//LB = 4: Size = 1.125, Time = 1.25  msec, Performace = 1932.74 GFlop/s
//LB = 3: Size = 1.125, Time = 1.696 msec, Performace = 1424.48 GFlop/s
template<int LB, int STEP>
__global__ void kernelSplit_Ims2_kernel_8_4(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int ph, int pw,
	int CFH, int CFW, int IH_slice, int IW_slice, int GK,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];//follow k88
	__shared__ float2 Ys[2][1 << LB >> 1][(2 << LB) + 2];//follow k44

	//prepare for GZ = sh*sw
	int y = blockIdx.z >> 1, x = blockIdx.z & 1;
	ph = ph - y; pw = pw - x;//if(ihs < 0): ihs += (ph + 1)/sh*sh
	int ihs = -ph; ihs += -(ihs < 0) & ((ph + 1) << 1 >> 1);
	int iws = -pw; iws += -(iws < 0) & ((pw + 1) << 1 >> 1);

	//prepare for GN = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	W += ic0 + ((ty >= STEP) << 2);//W[0, 0, 0, tic0]
	deltaX += ic0; //deltaX[0, 0, 0, ic0]

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = (((blockIdx.y << LB) + ty) << 2) + j_index;
	int j1 = j0 + 1, j2 = j0 + 2, j3 = j0 + 3;
	int IH_IW_slice = IH_slice * IW_slice;
	Ims2_n_ih_iw(j0, n0, ih0, iw0);
	Ims2_n_ih_iw(j1, n1, ih1, iw1);
	Ims2_n_ih_iw(j2, n2, ih2, iw2);
	Ims2_n_ih_iw(j3, n3, ih3, iw3);
	int Xoffset0 = ((n0*IH + ih0)*IW + iw0)*IC;
	int Xoffset1 = ((n1*IH + ih1)*IW + iw1)*IC;
	int Xoffset2 = ((n2*IH + ih2)*IW + iw2)*IC;
	int Xoffset3 = ((n3*IH + ih3)*IW + iw3)*IC;

	bool flagX = (tx & 1);
	int oph = CFH - 1, opw = CFW - 1;
	int tohs0 = ((choose_int(flagX, ih2, ih0) + ph) >> 1) - oph;
	int tohs1 = ((choose_int(flagX, ih3, ih1) + ph) >> 1) - oph;
	int tows0 = ((choose_int(flagX, iw2, iw0) + pw) >> 1) - opw;
	int tows1 = ((choose_int(flagX, iw3, iw1) + pw) >> 1) - opw;
	int Yoffset0 = ((choose_int(flagX, n2, n0)*OH + tohs0)*OW + tows0)*OC;
	int Yoffset1 = ((choose_int(flagX, n3, n1)*OH + tohs1)*OW + tows1)*OC;

	const int CFW_OC = CFW * OC;

	//load 4 elem from W[OC, FH, FW, IC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);
	Ims_fhr_fwr(W_k, W_fhr, W_fwr);
	int fh = y + ((oph - W_fhr) << 1);//fh = y + (CFH - 1 - W_fhr)*sh;
	int fw = x + ((opw - W_fwr) << 1);
	bool lw = (fh < FH) && (fw < FW);
	int Woffset = (((W_k - W_fwr * OC)*FH + fh)*FW + fw)*IC;
	Ws[buf][ty][tx] = (lw ? *(float4*)(W + Woffset) : FLOAT_ZERO4);

	//load 2 elem from deltaY[N, FH, FW, OC]
	int dY_k = tx >> 1;
	Ims_fhr_fwr(dY_k, dY_fhr, dY_fwr);
	int OW_OC = OW * OC, yoffset = dY_fhr * OW_OC + dY_k;
	const int Ys_x = (tx >> 1), Ys_y = (ty << 1) + (tx & 1);
	Ys[buf][Ys_x][Ys_y].x = (Ims_ldy(tohs0, tows0) ? deltaY[Yoffset0 + yoffset] : 0);
	Ys[buf][Ys_x][Ys_y].y = (Ims_ldy(tohs1, tows1) ? deltaY[Yoffset1 + yoffset] : 0);
	__syncthreads();

	//compure_area----------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)
	{
#pragma once
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 y0 = *(float4*)(&Ys[buf][ik][ty << 1]);
			float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];

			simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
			simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
			simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
			simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
		}
		buf ^= 1;

		//load 4 elem from W[OC, FH, FW, IC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		Ims_fhr_fwr(W_k, W_fhr, W_fwr);
		int fh = y + ((oph - W_fhr) << 1);
		int fw = x + ((opw - W_fwr) << 1);
		bool lw = (fh < FH) && (fw < FW);//W_oc = W_k - W_fwr * OC
		int Woffset = (((W_k - W_fwr * OC)*FH + fh)*FW + fw)*IC;
		Ws[buf][ty][tx] = (lw ? *(float4*)(W + Woffset) : FLOAT_ZERO4);

		//load 2 elem from deltaY[N, FH, FW, OC]
		int dY_k = ((ok << LB) + tx) >> 1;
		Ims_fhr_fwr(dY_k, dY_fhr, dY_fwr);
		int yoffset = dY_fhr * OW_OC + dY_k;
		Ys[buf][Ys_x][Ys_y].x = (Ims_ldy(tohs0, tows0) ? deltaY[Yoffset0 + yoffset] : 0);
		Ys[buf][Ys_x][Ys_y].y = (Ims_ldy(tohs1, tows1) ? deltaY[Yoffset1 + yoffset] : 0);
		__syncthreads();
	}
#pragma once
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 y0 = *(float4*)(&Ys[buf][ik][ty << 1]);
		float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];

		simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
		simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
		simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
		simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
	}

	*(float4*)(deltaX + Xoffset0) = v0;  *(float4*)(deltaX + Xoffset0 + 4) = v1;
	*(float4*)(deltaX + Xoffset1) = v2;  *(float4*)(deltaX + Xoffset1 + 4) = v3;
	*(float4*)(deltaX + Xoffset2) = v4;  *(float4*)(deltaX + Xoffset2 + 4) = v5;
	*(float4*)(deltaX + Xoffset3) = v6;  *(float4*)(deltaX + Xoffset3 + 4) = v7;
}

#endif


//(Y: BLOCK_SIZE*4, X:BLOCK_SIZE*8), K % (BLOCK_SIZE/2) == 0
#ifndef DCONV3D_DX_KERNEL_SPLIT_IMS2_KERNEL_4_8
#define DCONV3D_DX_KERNEL_SPLIT_IMS2_KERNEL_4_8

//LB = 4: Size = 1.125, Time = 1.328 msec, Performace = 1819.22 GFlop/s
//LB = 3: Size = 1.125, Time = 1.58  msec, Performace = 1529.06 GFlop/s
template<int LB, int STEP>
__global__ void kernelSplit_Ims2_kernel_4_8(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int ph, int pw,
	int CFH, int CFW, int IH_slice, int IW_slice, int GK,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB >> 1][(2 << LB) + 2];//follow k44
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];//follow k88

	//prepare for GZ = sh*sw
	int y = blockIdx.z >> 1, x = blockIdx.z & 1;
	ph = ph - y; pw = pw - x;//if(ihs < 0): ihs += (ph + 1)/sh*sh
	int ihs = -ph; ihs += -(ihs < 0) & ((ph + 1) << 1 >> 1);
	int iws = -pw; iws += -(iws < 0) & ((pw + 1) << 1 >> 1);

	//prepare for GN = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 2) + ic_index;
	W += ic0 + ((ty & 1) << 1);//W[0, 0, 0, tic0]
	deltaX += ic0; //deltaX[0, 0, 0, ic0]

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int j1 = j0 + 1, j2 = j0 + 2, j3 = j0 + 3;
	int j4 = j0 + 4, j5 = j0 + 5, j6 = j0 + 6, j7 = j0 + 7;
	int IH_IW_slice = IH_slice * IW_slice;
	Ims2_n_ih_iw(j0, n0, ih0, iw0);
	Ims2_n_ih_iw(j1, n1, ih1, iw1);
	Ims2_n_ih_iw(j2, n2, ih2, iw2);
	Ims2_n_ih_iw(j3, n3, ih3, iw3);
	Ims2_n_ih_iw(j4, n4, ih4, iw4);
	Ims2_n_ih_iw(j5, n5, ih5, iw5);
	Ims2_n_ih_iw(j6, n6, ih6, iw6);
	Ims2_n_ih_iw(j7, n7, ih7, iw7);
	int Xoffset0 = ((n0*IH + ih0)*IW + iw0)*IC;
	int Xoffset1 = ((n1*IH + ih1)*IW + iw1)*IC;
	int Xoffset2 = ((n2*IH + ih2)*IW + iw2)*IC;
	int Xoffset3 = ((n3*IH + ih3)*IW + iw3)*IC;
	int Xoffset4 = ((n4*IH + ih4)*IW + iw4)*IC;
	int Xoffset5 = ((n5*IH + ih5)*IW + iw5)*IC;
	int Xoffset6 = ((n6*IH + ih6)*IW + iw6)*IC;
	int Xoffset7 = ((n7*IH + ih7)*IW + iw7)*IC;

	bool flagX = (tx >= STEP);
	int oph = CFH - 1, opw = CFW - 1;
	int tohs0 = ((choose_int(flagX, ih4, ih0) + ph) >> 1) - oph;
	int tohs1 = ((choose_int(flagX, ih5, ih1) + ph) >> 1) - oph;
	int tohs2 = ((choose_int(flagX, ih6, ih2) + ph) >> 1) - oph;
	int tohs3 = ((choose_int(flagX, ih7, ih3) + ph) >> 1) - oph;
	int tows0 = ((choose_int(flagX, iw4, iw0) + pw) >> 1) - opw;
	int tows1 = ((choose_int(flagX, iw5, iw1) + pw) >> 1) - opw;
	int tows2 = ((choose_int(flagX, iw6, iw2) + pw) >> 1) - opw;
	int tows3 = ((choose_int(flagX, iw7, iw3) + pw) >> 1) - opw;
	int Yoffset0 = ((choose_int(flagX, n4, n0)*OH + tohs0)*OW + tows0)*OC;
	int Yoffset1 = ((choose_int(flagX, n5, n1)*OH + tohs1)*OW + tows1)*OC;
	int Yoffset2 = ((choose_int(flagX, n6, n2)*OH + tohs2)*OW + tows2)*OC;
	int Yoffset3 = ((choose_int(flagX, n7, n3)*OH + tohs3)*OW + tows3)*OC;

	const int CFW_OC = CFW * OC;

	//load 2 elem from W[OC, FH, FW, IC]
	int W_k = ty >> 1;
	Ims_fhr_fwr(W_k, W_fhr, W_fwr);
	int fh = y + ((oph - W_fhr) << 1);//fh = y + (CFH - 1 - W_fhr)*sh;
	int fw = x + ((opw - W_fwr) << 1);
	bool lw = (fh < FH) && (fw < FW);
	int Woffset = (((W_k - W_fwr * OC)*FH + fh)*FW + fw)*IC;
	const int Ws_y = (ty >> 1), Ws_x = (tx << 1) + (ty & 1);
	Ws[buf][Ws_y][Ws_x] = (lw ? *(float2*)(W + Woffset) : FLOAT_ZERO2);

	//load 4 elem from deltaY[N, FH, FW, OC]
	int dY_k = tx - ((tx >= STEP) << LB >> 1);
	Ims_fhr_fwr(dY_k, dY_fhr, dY_fwr);
	int OW_OC = OW * OC, yoffset = dY_fhr * OW_OC + dY_k;
	Ys[buf][tx][ty].x = (Ims_ldy(tohs0, tows0) ? deltaY[Yoffset0 + yoffset] : 0);
	Ys[buf][tx][ty].y = (Ims_ldy(tohs1, tows1) ? deltaY[Yoffset1 + yoffset] : 0);
	Ys[buf][tx][ty].z = (Ims_ldy(tohs2, tows2) ? deltaY[Yoffset2 + yoffset] : 0);
	Ys[buf][tx][ty].w = (Ims_ldy(tohs3, tows3) ? deltaY[Yoffset3 + yoffset] : 0);
	__syncthreads();

	//compure_area----------------------------------------------
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
#pragma once
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];
			float4 w0 = *(float4*)(&Ws[buf][ik][tx << 1]);

			simdMM4( v0, y0.x, w0);
			simdMM4( v2, y0.y, w0); 
			simdMM4( v4, y0.z, w0);
			simdMM4( v6, y0.w, w0); 
			simdMM4( v8, y1.x, w0); 
			simdMM4(v10, y1.y, w0);
			simdMM4(v12, y1.z, w0);
			simdMM4(v14, y1.w, w0);
		}
		buf ^= 1;

		//load 2 elem from W[OC, FH, FW, IC]
		int W_k = ((ok << LB) + ty) >> 1;
		Ims_fhr_fwr(W_k, W_fhr, W_fwr);
		int fh = y + ((oph - W_fhr) << 1);
		int fw = x + ((opw - W_fwr) << 1);
		bool lw = (fh < FH) && (fw < FW);//W_oc = W_k - W_fwr * OC
		int Woffset = (((W_k - W_fwr * OC)*FH + fh)*FW + fw)*IC;
		Ws[buf][Ws_y][Ws_x] = (lw ? *(float2*)(W + Woffset) : FLOAT_ZERO2);

		//load 4 elem from deltaY[N, FH, FW, OC]
		int dY_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		Ims_fhr_fwr(dY_k, dY_fhr, dY_fwr);
		int yoffset = dY_fhr * OW_OC + dY_k;
		Ys[buf][tx][ty].x = (Ims_ldy(tohs0, tows0) ? deltaY[Yoffset0 + yoffset] : 0);
		Ys[buf][tx][ty].y = (Ims_ldy(tohs1, tows1) ? deltaY[Yoffset1 + yoffset] : 0);
		Ys[buf][tx][ty].z = (Ims_ldy(tohs2, tows2) ? deltaY[Yoffset2 + yoffset] : 0);
		Ys[buf][tx][ty].w = (Ims_ldy(tohs3, tows3) ? deltaY[Yoffset3 + yoffset] : 0);
		__syncthreads();
	}
#pragma once
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];
		float4 w0 = *(float4*)(&Ws[buf][ik][tx << 1]);

		simdMM4( v0, y0.x, w0);
		simdMM4( v2, y0.y, w0);
		simdMM4( v4, y0.z, w0);
		simdMM4( v6, y0.w, w0);
		simdMM4( v8, y1.x, w0);
		simdMM4(v10, y1.y, w0);
		simdMM4(v12, y1.z, w0);
		simdMM4(v14, y1.w, w0);
	}

	*(float4*)(deltaX + Xoffset0) = v0; 
	*(float4*)(deltaX + Xoffset1) = v2;
	*(float4*)(deltaX + Xoffset2) = v4;
	*(float4*)(deltaX + Xoffset3) = v6;
	*(float4*)(deltaX + Xoffset4) = v8;
	*(float4*)(deltaX + Xoffset5) = v10;
	*(float4*)(deltaX + Xoffset6) = v12; 
	*(float4*)(deltaX + Xoffset7) = v14;
}

#endif


//(Y: BLOCK_SIZE*4, X:BLOCK_SIZE*4), K % (BLOCK_SIZE/2) == 0
#ifndef DCONV3D_DX_KERNEL_SPLIT_IMS2_KERNEL_4_4
#define DCONV3D_DX_KERNEL_SPLIT_IMS2_KERNEL_4_4

//LB = 4: Size = 1.125, Time = 1.536 msec, Performace = 1572.86 GFlop/s
//LB = 3: Size = 1.125, Time = 2.23  msec, Performace = 1083.37 GFlop/s
template<int LB, int STEP>
__global__ void kernelSplit_Ims2_kernel_4_4(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int ph, int pw,
	int CFH, int CFW, int IH_slice, int IW_slice, int GK,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2 Ys[2][1 << LB >> 1][(2 << LB) + 2];

	//prepare for GZ = sh*sw
	int y = blockIdx.z >> 1, x = blockIdx.z & 1;
	ph = ph - y; pw = pw - x;//if(ihs < 0): ihs += (ph + 1)/sh*sh
	int ihs = -ph; ihs += -(ihs < 0) & ((ph + 1) << 1 >> 1);
	int iws = -pw; iws += -(iws < 0) & ((pw + 1) << 1 >> 1);

	//prepare for GN = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 2) + ic_index;
	W += ic0 + ((ty & 1) << 1);//W[0, 0, 0, tic0]
	deltaX += ic0; //deltaX[0, 0, 0, ic0]

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = (((blockIdx.y << LB) + ty) << 2) + j_index;
	int j1 = j0 + 1, j2 = j0 + 2, j3 = j0 + 3;
	int IH_IW_slice = IH_slice * IW_slice;
	Ims2_n_ih_iw(j0, n0, ih0, iw0);
	Ims2_n_ih_iw(j1, n1, ih1, iw1);
	Ims2_n_ih_iw(j2, n2, ih2, iw2);
	Ims2_n_ih_iw(j3, n3, ih3, iw3);
	int Xoffset0 = ((n0*IH + ih0)*IW + iw0)*IC;
	int Xoffset1 = ((n1*IH + ih1)*IW + iw1)*IC;
	int Xoffset2 = ((n2*IH + ih2)*IW + iw2)*IC;
	int Xoffset3 = ((n3*IH + ih3)*IW + iw3)*IC;

	bool flagX = (tx & 1);
	int oph = CFH - 1, opw = CFW - 1;
	int tohs0 = ((choose_int(flagX, ih2, ih0) + ph) >> 1) - oph;
	int tohs1 = ((choose_int(flagX, ih3, ih1) + ph) >> 1) - oph;
	int tows0 = ((choose_int(flagX, iw2, iw0) + pw) >> 1) - opw;
	int tows1 = ((choose_int(flagX, iw3, iw1) + pw) >> 1) - opw;
	int Yoffset0 = ((choose_int(flagX, n2, n0)*OH + tohs0)*OW + tows0)*OC;
	int Yoffset1 = ((choose_int(flagX, n3, n1)*OH + tohs1)*OW + tows1)*OC;

	const int CFW_OC = CFW * OC;

	//load 2 elem from W[OC, FH, FW, IC]
	int W_k = ty >> 1;
	Ims_fhr_fwr(W_k, W_fhr, W_fwr);
	int fh = y + ((oph - W_fhr) << 1);//fh = y + (CFH - 1 - W_fhr)*sh;
	int fw = x + ((opw - W_fwr) << 1);
	bool lw = (fh < FH) && (fw < FW);
	int Woffset = (((W_k - W_fwr * OC)*FH + fh)*FW + fw)*IC;
	const int Ws_y = (ty >> 1), Ws_x = (tx << 1) + (ty & 1);
	Ws[buf][Ws_y][Ws_x] = (lw ? *(float2*)(W + Woffset) : FLOAT_ZERO2);

	//load 2 elem from deltaY[N, FH, FW, OC]
	int dY_k = tx >> 1;
	Ims_fhr_fwr(dY_k, dY_fhr, dY_fwr);
	int OW_OC = OW * OC, yoffset = dY_fhr * OW_OC + dY_k;
	const int Ys_x = (tx >> 1), Ys_y = (ty << 1) + (tx & 1);
	Ys[buf][Ys_x][Ys_y].x = (Ims_ldy(tohs0, tows0)? deltaY[Yoffset0 + yoffset] : 0);
	Ys[buf][Ys_x][Ys_y].y = (Ims_ldy(tohs1, tows1)? deltaY[Yoffset1 + yoffset] : 0);
	__syncthreads();

	//compure_area----------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0);
	float4 v1 = make_float4(0, 0, 0, 0);
	float4 v2 = make_float4(0, 0, 0, 0);
	float4 v3 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)
	{
#pragma once
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

		//load 2 elem from W[OC, FH, FW, IC]
		int W_k = ((ok << LB) + ty) >> 1;
		Ims_fhr_fwr(W_k, W_fhr, W_fwr);
		int fh = y + ((oph - W_fhr) << 1);
		int fw = x + ((opw - W_fwr) << 1);
		bool lw = (fh < FH) && (fw < FW);//W_oc = W_k - W_fwr * OC
		int Woffset = (((W_k - W_fwr * OC)*FH + fh)*FW + fw)*IC;
		Ws[buf][Ws_y][Ws_x] = (lw ? *(float2*)(W + Woffset) : FLOAT_ZERO2);

		//load 2 elem from deltaY[N, FH, FW, OC]
		int dY_k = ((ok << LB) + tx) >> 1;
		Ims_fhr_fwr(dY_k, dY_fhr, dY_fwr);
		int yoffset = dY_fhr * OW_OC + dY_k;
		Ys[buf][Ys_x][Ys_y].x = (Ims_ldy(tohs0, tows0) ? deltaY[Yoffset0 + yoffset] : 0);
		Ys[buf][Ys_x][Ys_y].y = (Ims_ldy(tohs1, tows1) ? deltaY[Yoffset1 + yoffset] : 0);
		__syncthreads();
	}
#pragma once
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 y = *(float4*)(&Ys[buf][ik][ty << 1]);
		float4 w = *(float4*)(&Ws[buf][ik][tx << 1]);

		simdMM4(v0, y.x, w);
		simdMM4(v1, y.y, w);
		simdMM4(v2, y.z, w);
		simdMM4(v3, y.w, w);
	}

	*(float4*)(deltaX + Xoffset0) = v0;
	*(float4*)(deltaX + Xoffset1) = v1;
	*(float4*)(deltaX + Xoffset2) = v2;
	*(float4*)(deltaX + Xoffset3) = v3;
}

#endif


//(Y: BLOCK_SIZE*4, X:BLOCK_SIZE*2), K % (BLOCK_SIZE/2) == 0
#ifndef DCONV3D_DX_KERNEL_SPLIT_IMS2_KERNEL_4_2
#define DCONV3D_DX_KERNEL_SPLIT_IMS2_KERNEL_4_2

//LB = 4: Size = 1.125, Time = 2.266 msec, Performace = 1066.16 GFlop/s
//LB = 3: Size = 1.125, Time = 3.666 msec, Performace =  659.007 GFlop/s
template<int LB, int STEP>
__global__ void kernelSplit_Ims2_kernel_4_2(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int ph, int pw,
	int CFH, int CFW, int IH_slice, int IW_slice, int GK,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float  Ys[2][1 << LB >> 1][(2 << LB) + 2];

	//prepare for GZ = sh*sw
	int y = blockIdx.z >> 1, x = blockIdx.z & 1;
	ph = ph - y; pw = pw - x;//if(ihs < 0): ihs += (ph + 1)/sh*sh
	int ihs = -ph; ihs += -(ihs < 0) & ((ph + 1) << 1 >> 1);
	int iws = -pw; iws += -(iws < 0) & ((pw + 1) << 1 >> 1);

	//prepare for GN = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 2) + ic_index;
	W += ic0 + ((ty & 1) << 1);//W[0, 0, 0, tic0]
	deltaX += ic0; //deltaX[0, 0, 0, ic0]

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = (((blockIdx.y << LB) + ty) << 1) + j_index, j1 = j0 + 1;
	int IH_IW_slice = IH_slice * IW_slice;
	Ims2_n_ih_iw(j0, n0, ih0, iw0);
	Ims2_n_ih_iw(j1, n1, ih1, iw1);
	int Xoffset0 = ((n0*IH + ih0)*IW + iw0)*IC;
	int Xoffset1 = ((n1*IH + ih1)*IW + iw1)*IC;

	bool flagX = (tx & 1);
	int oph = CFH - 1, opw = CFW - 1;
	int tohs0 = ((choose_int(flagX, ih1, ih0) + ph) >> 1) - oph;
	int tows0 = ((choose_int(flagX, iw1, iw0) + pw) >> 1) - opw;
	int Yoffset0 = ((choose_int(flagX, n1, n0)*OH + tohs0)*OW + tows0)*OC;

	const int CFW_OC = CFW * OC;

	//load 2 elem from W[OC, FH, FW, IC]
	int W_k = ty >> 1;
	Ims_fhr_fwr(W_k, W_fhr, W_fwr);
	int fh = y + ((oph - W_fhr) << 1);//fh = y + (CFH - 1 - W_fhr)*sh;
	int fw = x + ((opw - W_fwr) << 1);
	bool lw = (fh < FH) && (fw < FW);
	int Woffset = (((W_k - W_fwr * OC)*FH + fh)*FW + fw)*IC;
	const int Ws_y = (ty >> 1), Ws_x = (tx << 1) + (ty & 1);
	Ws[buf][Ws_y][Ws_x] = (lw ? *(float2*)(W + Woffset) : FLOAT_ZERO2);

	//load 1 elem from deltaY[N, FH, FW, OC]
	int dY_k = tx >> 1;
	Ims_fhr_fwr(dY_k, dY_fhr, dY_fwr);
	int OW_OC = OW * OC, yoffset = dY_fhr * OW_OC + dY_k;
	const int Ys_x = (tx >> 1), Ys_y = (ty << 1) + (tx & 1);
	Ys[buf][Ys_x][Ys_y] = (Ims_ldy(tohs0, tows0) ? deltaY[Yoffset0 + yoffset] : 0);
	__syncthreads();

	//compure_area----------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0);
	float4 v1 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)
	{
#pragma once
		for (int ik = 0; ik < STEP; ik++){
			float2 y = *(float2*)(&Ys[buf][ik][ty << 1]);
			float4 w = *(float4*)(&Ws[buf][ik][tx << 1]);
			simdMM4(v0, y.x, w);
			simdMM4(v1, y.y, w);
		}
		buf ^= 1;

		//load 2 elem from W[OC, FH, FW, IC]
		int W_k = ((ok << LB) + ty) >> 1;
		Ims_fhr_fwr(W_k, W_fhr, W_fwr);
		int fh = y + ((oph - W_fhr) << 1);
		int fw = x + ((opw - W_fwr) << 1);
		bool lw = (fh < FH) && (fw < FW);//W_oc = W_k - W_fwr * OC
		int Woffset = (((W_k - W_fwr * OC)*FH + fh)*FW + fw)*IC;
		Ws[buf][Ws_y][Ws_x] = (lw ? *(float2*)(W + Woffset) : FLOAT_ZERO2);

		//load 1 elem from deltaY[N, FH, FW, OC]
		int dY_k = ((ok << LB) + tx) >> 1;
		Ims_fhr_fwr(dY_k, dY_fhr, dY_fwr);
		int yoffset = dY_fhr * OW_OC + dY_k;
		Ys[buf][Ys_x][Ys_y] = (Ims_ldy(tohs0, tows0) ? deltaY[Yoffset0 + yoffset] : 0);
		__syncthreads();
	}
#pragma once
	for (int ik = 0; ik < STEP; ik++) {
		float2 y = *(float2*)(&Ys[buf][ik][ty << 1]);
		float4 w = *(float4*)(&Ws[buf][ik][tx << 1]);
		simdMM4(v0, y.x, w);
		simdMM4(v1, y.y, w);
	}

	*(float4*)(deltaX + Xoffset0) = v0;
	*(float4*)(deltaX + Xoffset1) = v1;
}

#endif


//(Y: BLOCK_SIZE*2, X:BLOCK_SIZE*4), K % (BLOCK_SIZE/2) == 0
#ifndef DCONV3D_DX_KERNEL_SPLIT_IMS2_KERNEL_2_4
#define DCONV3D_DX_KERNEL_SPLIT_IMS2_KERNEL_2_4

//LB = 4: Size = 1.125, Time = 2.404 msec, Performace = 1004.96 GFlop/s
//LB = 3: Size = 1.125, Time = 3.794 msec, Performace =  636.774 GFlop/s
template<int LB, int STEP>
__global__ void kernelSplit_Ims2_kernel_2_4(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int ph, int pw,
	int CFH, int CFW, int IH_slice, int IW_slice, int GK,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float  Ws[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2 Ys[2][1 << LB >> 1][(2 << LB) + 2];

	//prepare for GZ = sh*sw
	int y = blockIdx.z >> 1, x = blockIdx.z & 1;
	ph = ph - y; pw = pw - x;//if(ihs < 0): ihs += (ph + 1)/sh*sh
	int ihs = -ph; ihs += -(ihs < 0) & ((ph + 1) << 1 >> 1);
	int iws = -pw; iws += -(iws < 0) & ((pw + 1) << 1 >> 1);

	//prepare for GN = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 1) + ic_index;
	W += ic0 + (ty & 1);//W[0, 0, 0, tic0]
	deltaX += ic0; //deltaX[0, 0, 0, ic0]

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = (((blockIdx.y << LB) + ty) << 2) + j_index;
	int j1 = j0 + 1, j2 = j0 + 2, j3 = j0 + 3;
	int IH_IW_slice = IH_slice * IW_slice;
	Ims2_n_ih_iw(j0, n0, ih0, iw0);
	Ims2_n_ih_iw(j1, n1, ih1, iw1);
	Ims2_n_ih_iw(j2, n2, ih2, iw2);
	Ims2_n_ih_iw(j3, n3, ih3, iw3);
	int Xoffset0 = ((n0*IH + ih0)*IW + iw0)*IC;
	int Xoffset1 = ((n1*IH + ih1)*IW + iw1)*IC;
	int Xoffset2 = ((n2*IH + ih2)*IW + iw2)*IC;
	int Xoffset3 = ((n3*IH + ih3)*IW + iw3)*IC;

	bool flagX = (tx & 1);
	int oph = CFH - 1, opw = CFW - 1;
	int tohs0 = ((choose_int(flagX, ih2, ih0) + ph) >> 1) - oph;
	int tohs1 = ((choose_int(flagX, ih3, ih1) + ph) >> 1) - oph;
	int tows0 = ((choose_int(flagX, iw2, iw0) + pw) >> 1) - opw;
	int tows1 = ((choose_int(flagX, iw3, iw1) + pw) >> 1) - opw;
	int Yoffset0 = ((choose_int(flagX, n2, n0)*OH + tohs0)*OW + tows0)*OC;
	int Yoffset1 = ((choose_int(flagX, n3, n1)*OH + tohs1)*OW + tows1)*OC;

	const int CFW_OC = CFW * OC;

	//load 1 elem from W[OC, FH, FW, IC]
	int W_k = ty >> 1;
	Ims_fhr_fwr(W_k, W_fhr, W_fwr);
	int fh = y + ((oph - W_fhr) << 1);//fh = y + (CFH - 1 - W_fhr)*sh;
	int fw = x + ((opw - W_fwr) << 1);
	bool lw = (fh < FH) && (fw < FW);
	int Woffset = (((W_k - W_fwr * OC)*FH + fh)*FW + fw)*IC;
	const int Ws_y = (ty >> 1), Ws_x = (tx << 1) + (ty & 1);
	Ws[buf][Ws_y][Ws_x] = (lw ? W[Woffset] : 0);

	//load 2 elem from deltaY[N, FH, FW, OC]
	int dY_k = tx >> 1;
	Ims_fhr_fwr(dY_k, dY_fhr, dY_fwr);
	int OW_OC = OW * OC, yoffset = dY_fhr * OW_OC + dY_k;
	const int Ys_x = (tx >> 1), Ys_y = (ty << 1) + (tx & 1);
	Ys[buf][Ys_x][Ys_y].x = (Ims_ldy(tohs0, tows0) ? deltaY[Yoffset0 + yoffset] : 0);
	Ys[buf][Ys_x][Ys_y].y = (Ims_ldy(tohs1, tows1) ? deltaY[Yoffset1 + yoffset] : 0);
	__syncthreads();

	//compure_area----------------------------------------------
	float2 v0 = make_float2(0, 0);
	float2 v1 = make_float2(0, 0);
	float2 v2 = make_float2(0, 0);
	float2 v3 = make_float2(0, 0);
	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)
	{
#pragma once
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 y = *(float4*)(&Ys[buf][ik][ty << 1]);
			float2 w = *(float2*)(&Ws[buf][ik][tx << 1]);

			simdMM2(v0, y.x, w);
			simdMM2(v1, y.y, w);
			simdMM2(v2, y.z, w);
			simdMM2(v3, y.w, w);
		}
		buf ^= 1;

		//load 1 elem from W[OC, FH, FW, IC]
		int W_k = ((ok << LB) + ty) >> 1;
		Ims_fhr_fwr(W_k, W_fhr, W_fwr);
		int fh = y + ((oph - W_fhr) << 1);
		int fw = x + ((opw - W_fwr) << 1);
		bool lw = (fh < FH) && (fw < FW);//W_oc = W_k - W_fwr * OC
		int Woffset = (((W_k - W_fwr * OC)*FH + fh)*FW + fw)*IC;
		Ws[buf][Ws_y][Ws_x] = (lw ? W[Woffset] : 0);

		//load 2 elem from deltaY[N, FH, FW, OC]
		int dY_k = ((ok << LB) + tx) >> 1;
		Ims_fhr_fwr(dY_k, dY_fhr, dY_fwr);
		int yoffset = dY_fhr * OW_OC + dY_k;
		Ys[buf][Ys_x][Ys_y].x = (Ims_ldy(tohs0, tows0) ? deltaY[Yoffset0 + yoffset] : 0);
		Ys[buf][Ys_x][Ys_y].y = (Ims_ldy(tohs1, tows1) ? deltaY[Yoffset1 + yoffset] : 0);
		__syncthreads();
	}
#pragma once
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 y = *(float4*)(&Ys[buf][ik][ty << 1]);
		float2 w = *(float2*)(&Ws[buf][ik][tx << 1]);

		simdMM2(v0, y.x, w);
		simdMM2(v1, y.y, w);
		simdMM2(v2, y.z, w);
		simdMM2(v3, y.w, w);
	}

	*(float2*)(deltaX + Xoffset0) = v0;
	*(float2*)(deltaX + Xoffset1) = v1;
	*(float2*)(deltaX + Xoffset2) = v2;
	*(float2*)(deltaX + Xoffset3) = v3;
}

#endif


//(Y: BLOCK_SIZE*2, X:BLOCK_SIZE*2), K >= BLOCK_SIZE
#ifndef DCONV3D_DX_KERNEL_SPLIT_IMS2_KERNEL_2_2
#define DCONV3D_DX_KERNEL_SPLIT_IMS2_KERNEL_2_2

//LB = 4: Size = 1.08984, Time = 3.602 msec, Performace = 649.756 GFlop/s
//LB = 3: Size = 1.08984, Time = 4.182 msec, Performace = 559.642 GFlop/s
template<int LB, int STEP>
__global__ void kernelSplit_Ims2_kernel_2_2(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int ph, int pw,
	int CFH, int CFW, int IH_slice, int IW_slice, int GK,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float2 Ys[2][1 << LB][(1 << LB) + 1];

	//prepare for GZ = sh*sw
	int y = blockIdx.z >> 1, x = blockIdx.z & 1;
	ph = ph - y; pw = pw - x;//if(ihs < 0): ihs += (ph + 1)/sh*sh
	int ihs = -ph; ihs += -(ihs < 0) & ((ph + 1) << 1 >> 1);
	int iws = -pw; iws += -(iws < 0) & ((pw + 1) << 1 >> 1);

	//prepare for GN = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 1) + ic_index;
	W += ic0;//W[0, 0, 0, tic0]
	deltaX += ic0; //deltaX[0, 0, 0, ic0]

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = (((blockIdx.y << LB) + ty) << 1) + j_index, j1 = j0 + 1;
	int IH_IW_slice = IH_slice * IW_slice;
	Ims2_n_ih_iw(j0, n0, ih0, iw0);
	Ims2_n_ih_iw(j1, n1, ih1, iw1);
	int Xoffset0 = ((n0*IH + ih0)*IW + iw0)*IC;
	int Xoffset1 = ((n1*IH + ih1)*IW + iw1)*IC;

	int oph = CFH - 1, opw = CFW - 1;
	int ohs0 = ((ih0 + ph) >> 1) - oph, ows0 = ((iw0 + pw) >> 1) - opw;
	int ohs1 = ((ih1 + ph) >> 1) - oph, ows1 = ((iw1 + pw) >> 1) - opw;
	int Yoffset0 = ((n0*OH + ohs0)*OW + ows0)*OC;
	int Yoffset1 = ((n1*OH + ohs1)*OW + ows1)*OC;

	const int CFW_OC = CFW * OC;

	//load 2 elem from W[OC, FH, FW, IC]
	int W_k = ty;
	Ims_fhr_fwr(W_k, W_fhr, W_fwr);
	int fh = y + ((oph - W_fhr) << 1);//fh = y + (CFH - 1 - W_fhr)*sh;
	int fw = x + ((opw - W_fwr) << 1);
	bool lw = (fh < FH) && (fw < FW);//W_oc = W_k - W_fwr * OC
	int Woffset = (((W_k - W_fwr * OC)*FH + fh)*FW + fw)*IC;
	Ws[buf][ty][tx] = (lw ? *(float2*)(W + Woffset) : FLOAT_ZERO2);

	//load 2 elem from deltaY[N, FH, FW, OC]
	int dY_k = tx;
	Ims_fhr_fwr(dY_k, dY_fhr, dY_fwr);
	int OW_OC = OW * OC, yoffset = dY_fhr * OW_OC + dY_k;
	Ys[buf][tx][ty].x = (Ims_ldy(ohs0, ows0) ? deltaY[Yoffset0 + yoffset] : 0);
	Ys[buf][tx][ty].y = (Ims_ldy(ohs1, ows1) ? deltaY[Yoffset1 + yoffset] : 0);
	__syncthreads();

	//compure_area----------------------------------------------
	float2 v0 = make_float2(0, 0);
	float2 v1 = make_float2(0, 0);
	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)
	{
#pragma once
		for (int ik = 0; ik < STEP; ik++) {
			float2 y = Ys[buf][ik][ty];
			float2 w = Ws[buf][ik][tx];
			simdMM2(v0, y.x, w);
			simdMM2(v1, y.y, w);
		}
		buf ^= 1;

		//load 2 elem from W[OC, FH, FW, IC]
		int W_k = (ok << LB) + ty;
		Ims_fhr_fwr(W_k, W_fhr, W_fwr);
		int fh = y + ((oph - W_fhr) << 1);
		int fw = x + ((opw - W_fwr) << 1);
		bool lw = (fh < FH) && (fw < FW);//W_oc = W_k - W_fwr * OC
		int Woffset = (((W_k - W_fwr * OC)*FH + fh)*FW + fw)*IC;
		Ws[buf][ty][tx] = (lw ? *(float2*)(W + Woffset) : FLOAT_ZERO2);

		//load 2 elem from deltaY[N, FH, FW, OC]
		int dY_k = (ok << LB) + tx;
		Ims_fhr_fwr(dY_k, dY_fhr, dY_fwr);
		int yoffset = dY_fhr * OW_OC + dY_k;
		Ys[buf][tx][ty].x = (Ims_ldy(ohs0, ows0) ? deltaY[Yoffset0 + yoffset] : 0);
		Ys[buf][tx][ty].y = (Ims_ldy(ohs1, ows1) ? deltaY[Yoffset1 + yoffset] : 0);
		__syncthreads();
	}
#pragma once
	for (int ik = 0; ik < STEP; ik++) {
		float2 y = Ys[buf][ik][ty];
		float2 w = Ws[buf][ik][tx];
		simdMM2(v0, y.x, w);
		simdMM2(v1, y.y, w);
	}
	
	//when GK % STEP != 0-------------------------------------------
	for (int k = GK - (GK&(STEP - 1)); k < GK; k++)
	{
		int dY_k = k; 
		Ims_fhr_fwr(dY_k, dY_fhr, dY_fwr);

		//load 2 elem from W[OC, FH, FW, IC]
		int fh = y + ((oph - dY_fhr) << 1);
		int fw = x + ((opw - dY_fwr) << 1);
		bool lw = (fh < FH) && (fw < FW);
		int Woffset = (((dY_k - dY_fwr * OC)*FH + fh)*FW + fw)*IC;
		float2 w = (lw ? *(float2*)(W + Woffset) : FLOAT_ZERO2);

		//load 2 elem from deltaY[N, FH, FW, OC]
		float2 y;
		int yoffset = dY_fhr * OW_OC + dY_k;
		y.x = (Ims_ldy(ohs0, ows0) ? deltaY[Yoffset0 + yoffset] : 0);
		y.y = (Ims_ldy(ohs1, ows1) ? deltaY[Yoffset1 + yoffset] : 0);

		simdMM2(v0, y.x, w);
		simdMM2(v1, y.y, w);
	}
	//when GK % STEP != 0-------------------------------------------

	*(float2*)(deltaX + Xoffset0) = v0;
	*(float2*)(deltaX + Xoffset1) = v1;
}

#endif


//(Y: BLOCK_SIZE*2, X:BLOCK_SIZE*1), K >= BLOCK_SIZE
#ifndef DCONV3D_DX_KERNEL_SPLIT_IMS2_KERNEL_2_1
#define DCONV3D_DX_KERNEL_SPLIT_IMS2_KERNEL_2_1

//LB = 4: Size = 0.544922, Time = 3.244 msec, Performace = 360.731 GFlop/s
//LB = 3: Size = 0.544922, Time = 3.792 msec, Performace = 308.6 GFlop/s
template<int LB, int STEP>
__global__ void kernelSplit_Ims2_kernel_2_1(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int ph, int pw,
	int CFH, int CFW, int IH_slice, int IW_slice, int GK,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float  Ys[2][1 << LB][(1 << LB) + 1];

	//prepare for GZ = sh*sw
	int y = blockIdx.z >> 1, x = blockIdx.z & 1;
	ph = ph - y; pw = pw - x;//if(ihs < 0): ihs += (ph + 1)/sh*sh
	int ihs = -ph; ihs += -(ihs < 0) & ((ph + 1) << 1 >> 1);
	int iws = -pw; iws += -(iws < 0) & ((pw + 1) << 1 >> 1);

	//prepare for GN = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 1) + ic_index;
	W += ic0;//W[0, 0, 0, tic0]
	deltaX += ic0; //deltaX[0, 0, 0, ic0]

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = ((blockIdx.y << LB) + ty) + j_index;
	int IH_IW_slice = IH_slice * IW_slice;
	Ims2_n_ih_iw(j0, n0, ih0, iw0);
	int Xoffset0 = ((n0*IH + ih0)*IW + iw0)*IC;

	int oph = CFH - 1, opw = CFW - 1;
	int ohs0 = ((ih0 + ph) >> 1) - oph, ows0 = ((iw0 + pw) >> 1) - opw;
	int Yoffset0 = ((n0*OH + ohs0)*OW + ows0)*OC;

	const int CFW_OC = CFW * OC;

	//load 2 elem from W[OC, FH, FW, IC]
	int W_k = ty;
	Ims_fhr_fwr(W_k, W_fhr, W_fwr);
	int fh = y + ((oph - W_fhr) << 1);//fh = y + (CFH - 1 - W_fhr)*sh;
	int fw = x + ((opw - W_fwr) << 1);
	bool lw = (fh < FH) && (fw < FW);//W_oc = W_k - W_fwr * OC
	int Woffset = (((W_k - W_fwr * OC)*FH + fh)*FW + fw)*IC;
	Ws[buf][ty][tx] = (lw ? *(float2*)(W + Woffset) : FLOAT_ZERO2);

	//load 2 elem from deltaY[N, FH, FW, OC]
	int dY_k = tx;
	Ims_fhr_fwr(dY_k, dY_fhr, dY_fwr);
	int OW_OC = OW * OC, yoffset = dY_fhr * OW_OC + dY_k;
	Ys[buf][tx][ty] = (Ims_ldy(ohs0, ows0) ? deltaY[Yoffset0 + yoffset] : 0);
	__syncthreads();

	//compure_area----------------------------------------------
	float2 v0 = make_float2(0, 0);
	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)
	{
#pragma once
		for (int ik = 0; ik < STEP; ik++) {
			float  y = Ys[buf][ik][ty];
			float2 w = Ws[buf][ik][tx];
			simdMM2(v0, y, w);
		}
		buf ^= 1;

		//load 2 elem from W[OC, FH, FW, IC]
		int W_k = (ok << LB) + ty;
		Ims_fhr_fwr(W_k, W_fhr, W_fwr);
		int fh = y + ((oph - W_fhr) << 1);
		int fw = x + ((opw - W_fwr) << 1);
		bool lw = (fh < FH) && (fw < FW);//W_oc = W_k - W_fwr * OC
		int Woffset = (((W_k - W_fwr * OC)*FH + fh)*FW + fw)*IC;
		Ws[buf][ty][tx] = (lw ? *(float2*)(W + Woffset) : FLOAT_ZERO2);

		//load 2 elem from deltaY[N, FH, FW, OC]
		int dY_k = (ok << LB) + tx;
		Ims_fhr_fwr(dY_k, dY_fhr, dY_fwr);
		int yoffset = dY_fhr * OW_OC + dY_k;
		Ys[buf][tx][ty] = (Ims_ldy(ohs0, ows0) ? deltaY[Yoffset0 + yoffset] : 0);
		__syncthreads();
	}
#pragma once
	for (int ik = 0; ik < STEP; ik++) {
		float  y = Ys[buf][ik][ty];
		float2 w = Ws[buf][ik][tx];
		simdMM2(v0, y, w);
	}

	//when GK % STEP != 0-------------------------------------------
	for (int k = GK - (GK&(STEP - 1)); k < GK; k++)
	{
		int dY_k = k;
		Ims_fhr_fwr(dY_k, dY_fhr, dY_fwr);

		//load 2 elem from W[OC, FH, FW, IC]
		int fh = y + ((oph - dY_fhr) << 1);
		int fw = x + ((opw - dY_fwr) << 1);
		bool lw = (fh < FH) && (fw < FW);
		int Woffset = (((dY_k - dY_fwr * OC)*FH + fh)*FW + fw)*IC;
		float2 w = (lw ? *(float2*)(W + Woffset) : FLOAT_ZERO2);

		//load 1 elem from deltaY[N, FH, FW, OC]
		int yoffset = dY_fhr * OW_OC + dY_k;
		float y = (Ims_ldy(ohs0, ows0) ? deltaY[Yoffset0 + yoffset] : 0);

		simdMM2(v0, y, w);
	}
	//when GK % STEP != 0-------------------------------------------

	*(float2*)(deltaX + Xoffset0) = v0;
}

#endif


//(Y: BLOCK_SIZE*1, X:BLOCK_SIZE*2), K >= BLOCK_SIZE
#ifndef DCONV3D_DX_KERNEL_SPLIT_IMS2_KERNEL_1_2
#define DCONV3D_DX_KERNEL_SPLIT_IMS2_KERNEL_1_2

//LB = 4: Size = 0.544922, Time = 3.314 msec, Performace = 353.111 GFlop/s
//LB = 3: Size = 1.08984, Time = 4.182 msec, Performace = 559.642 GFlop/s
template<int LB, int STEP>
__global__ void kernelSplit_Ims2_kernel_1_2(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int ph, int pw,
	int CFH, int CFW, int IH_slice, int IW_slice, int GK,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float  Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float2 Ys[2][1 << LB][(1 << LB) + 1];

	//prepare for GZ = sh*sw
	int y = blockIdx.z >> 1, x = blockIdx.z & 1;
	ph = ph - y; pw = pw - x;//if(ihs < 0): ihs += (ph + 1)/sh*sh
	int ihs = -ph; ihs += -(ihs < 0) & ((ph + 1) << 1 >> 1);
	int iws = -pw; iws += -(iws < 0) & ((pw + 1) << 1 >> 1);

	//prepare for GN = IC
	int ic0 = ((blockIdx.x << LB) + tx) + ic_index;
	W += ic0;//W[0, 0, 0, tic0]
	deltaX += ic0; //deltaX[0, 0, 0, ic0]

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = (((blockIdx.y << LB) + ty) << 1) + j_index, j1 = j0 + 1;
	int IH_IW_slice = IH_slice * IW_slice;
	Ims2_n_ih_iw(j0, n0, ih0, iw0);
	Ims2_n_ih_iw(j1, n1, ih1, iw1);
	int Xoffset0 = ((n0*IH + ih0)*IW + iw0)*IC;
	int Xoffset1 = ((n1*IH + ih1)*IW + iw1)*IC;

	int oph = CFH - 1, opw = CFW - 1;
	int ohs0 = ((ih0 + ph) >> 1) - oph, ows0 = ((iw0 + pw) >> 1) - opw;
	int ohs1 = ((ih1 + ph) >> 1) - oph, ows1 = ((iw1 + pw) >> 1) - opw;
	int Yoffset0 = ((n0*OH + ohs0)*OW + ows0)*OC;
	int Yoffset1 = ((n1*OH + ohs1)*OW + ows1)*OC;

	const int CFW_OC = CFW * OC;

	//load 2 elem from W[OC, FH, FW, IC]
	int W_k = ty;
	Ims_fhr_fwr(W_k, W_fhr, W_fwr);
	int fh = y + ((oph - W_fhr) << 1);//fh = y + (CFH - 1 - W_fhr)*sh;
	int fw = x + ((opw - W_fwr) << 1);
	bool lw = (fh < FH) && (fw < FW);//W_oc = W_k - W_fwr * OC
	int Woffset = (((W_k - W_fwr * OC)*FH + fh)*FW + fw)*IC;
	Ws[buf][ty][tx] = (lw ? W[Woffset] : 0);

	//load 2 elem from deltaY[N, FH, FW, OC]
	int dY_k = tx;
	Ims_fhr_fwr(dY_k, dY_fhr, dY_fwr);
	int OW_OC = OW * OC, yoffset = dY_fhr * OW_OC + dY_k;
	Ys[buf][tx][ty].x = (Ims_ldy(ohs0, ows0) ? deltaY[Yoffset0 + yoffset] : 0);
	Ys[buf][tx][ty].y = (Ims_ldy(ohs1, ows1) ? deltaY[Yoffset1 + yoffset] : 0);
	__syncthreads();

	//compure_area----------------------------------------------
	float2 v0 = make_float2(0, 0);
	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)
	{
#pragma once
		for (int ik = 0; ik < STEP; ik++) {
			float2 y = Ys[buf][ik][ty];
			float  w = Ws[buf][ik][tx];
			simdMM2(v0, w, y);
		}
		buf ^= 1;

		//load 2 elem from W[OC, FH, FW, IC]
		int W_k = (ok << LB) + ty;
		Ims_fhr_fwr(W_k, W_fhr, W_fwr);
		int fh = y + ((oph - W_fhr) << 1);
		int fw = x + ((opw - W_fwr) << 1);
		bool lw = (fh < FH) && (fw < FW);//W_oc = W_k - W_fwr * OC
		int Woffset = (((W_k - W_fwr * OC)*FH + fh)*FW + fw)*IC;
		Ws[buf][ty][tx] = (lw ? W[Woffset] : 0);

		//load 2 elem from deltaY[N, FH, FW, OC]
		int dY_k = (ok << LB) + tx;
		Ims_fhr_fwr(dY_k, dY_fhr, dY_fwr);
		int yoffset = dY_fhr * OW_OC + dY_k;
		Ys[buf][tx][ty].x = (Ims_ldy(ohs0, ows0) ? deltaY[Yoffset0 + yoffset] : 0);
		Ys[buf][tx][ty].y = (Ims_ldy(ohs1, ows1) ? deltaY[Yoffset1 + yoffset] : 0);
		__syncthreads();
	}
#pragma once
	for (int ik = 0; ik < STEP; ik++) {
		float2 y = Ys[buf][ik][ty];
		float  w = Ws[buf][ik][tx];
		simdMM2(v0, w, y);
	}

	//when GK % STEP != 0-------------------------------------------
	for (int k = GK - (GK&(STEP - 1)); k < GK; k++)
	{
		int dY_k = k;
		Ims_fhr_fwr(dY_k, dY_fhr, dY_fwr);

		//load 2 elem from W[OC, FH, FW, IC]
		int fh = y + ((oph - dY_fhr) << 1);
		int fw = x + ((opw - dY_fwr) << 1);
		bool lw = (fh < FH) && (fw < FW);
		int Woffset = (((dY_k - dY_fwr * OC)*FH + fh)*FW + fw)*IC;
		float w = (lw ? W[Woffset] : 0);

		//load 2 elem from deltaY[N, FH, FW, OC]
		float2 y;
		int yoffset = dY_fhr * OW_OC + dY_k;
		y.x = (Ims_ldy(ohs0, ows0) ? deltaY[Yoffset0 + yoffset] : 0);
		y.y = (Ims_ldy(ohs1, ows1) ? deltaY[Yoffset1 + yoffset] : 0);

		simdMM2(v0, w, y);
	}
	//when GK % STEP != 0-------------------------------------------

	deltaX[Xoffset0] = v0.x;
	deltaX[Xoffset1] = v0.y;
}

#endif


//(Y: BLOCK_SIZE*1, X:BLOCK_SIZE*1), K >= BLOCK_SIZE
#ifndef DCONV3D_DX_KERNEL_SPLIT_IMS2_KERNEL_1_1
#define DCONV3D_DX_KERNEL_SPLIT_IMS2_KERNEL_1_1

//LB = 4: Size = 0.544922, Time = 5.812 msec, Performace = 201.344 GFlop/s
//LB = 3: Size = 0.544922, Time = 6.9   msec, Performace = 169.596 GFlop/s
template<int LB, int STEP>
__global__ void kernelSplit_Ims2_kernel_1_1(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int ph, int pw,
	int CFH, int CFW, int IH_slice, int IW_slice, int GK,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float Ys[2][1 << LB][(1 << LB) + 1];

	//prepare for GZ = sh*sw
	int y = blockIdx.z >> 1, x = blockIdx.z & 1;
	ph = ph - y; pw = pw - x;//if(ihs < 0): ihs += (ph + 1)/sh*sh
	int ihs = -ph; ihs += -(ihs < 0) & ((ph + 1) << 1 >> 1);
	int iws = -pw; iws += -(iws < 0) & ((pw + 1) << 1 >> 1);

	//prepare for GN = IC
	int ic0 = ((blockIdx.x << LB) + tx) + ic_index;
	W += ic0;//W[0, 0, 0, tic0]
	deltaX += ic0; //deltaX[0, 0, 0, ic0]

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = ((blockIdx.y << LB) + ty) + j_index;
	int IH_IW_slice = IH_slice * IW_slice;
	Ims2_n_ih_iw(j0, n0, ih0, iw0);
	int Xoffset0 = ((n0*IH + ih0)*IW + iw0)*IC;

	int oph = CFH - 1, opw = CFW - 1;
	int ohs0 = ((ih0 + ph) >> 1) - oph, ows0 = ((iw0 + pw) >> 1) - opw;
	int Yoffset0 = ((n0*OH + ohs0)*OW + ows0)*OC;

	const int CFW_OC = CFW * OC;

	//load 1 elem from W[OC, FH, FW, IC]
	int W_k = ty;
	Ims_fhr_fwr(W_k, W_fhr, W_fwr);
	int fh = y + ((oph - W_fhr) << 1);//fh = y + (CFH - 1 - W_fhr)*sh;
	int fw = x + ((opw - W_fwr) << 1);
	bool lw = (fh < FH) && (fw < FW);//W_oc = W_k - W_fwr * OC
	int Woffset = (((W_k - W_fwr * OC)*FH + fh)*FW + fw)*IC;
	Ws[buf][ty][tx] = (lw ? W[Woffset] : 0);

	//load 1 elem from deltaY[N, FH, FW, OC]
	int dY_k = tx;
	Ims_fhr_fwr(dY_k, dY_fhr, dY_fwr);
	int OW_OC = OW * OC, yoffset = dY_fhr * OW_OC + dY_k;
	Ys[buf][tx][ty] = (Ims_ldy(ohs0, ows0) ? deltaY[Yoffset0 + yoffset] : 0);
	__syncthreads();

	//compure_area----------------------------------------------
	float v = 0;
	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)
	{
#pragma once
		for (int ik = 0; ik < STEP; ik++) {
			float y = Ys[buf][ik][ty];
			float w = Ws[buf][ik][tx];
			v += y * w;
		}
		buf ^= 1;

		//load 1 elem from W[OC, FH, FW, IC]
		int W_k = (ok << LB) + ty;
		Ims_fhr_fwr(W_k, W_fhr, W_fwr);
		int fh = y + ((oph - W_fhr) << 1);
		int fw = x + ((opw - W_fwr) << 1);
		bool lw = (fh < FH) && (fw < FW);//W_oc = W_k - W_fwr * OC
		int Woffset = (((W_k - W_fwr * OC)*FH + fh)*FW + fw)*IC;
		Ws[buf][ty][tx] = (lw ? W[Woffset] : 0);

		//load 1 elem from deltaY[N, FH, FW, OC]
		int dY_k = (ok << LB) + tx;
		Ims_fhr_fwr(dY_k, dY_fhr, dY_fwr);
		int yoffset = dY_fhr * OW_OC + dY_k;
		Ys[buf][tx][ty] = (Ims_ldy(ohs0, ows0) ? deltaY[Yoffset0 + yoffset] : 0);
		__syncthreads();
	}
#pragma once
	for (int ik = 0; ik < STEP; ik++) {
		float y = Ys[buf][ik][ty];
		float w = Ws[buf][ik][tx];
		v += y * w;
	}

	//when GK % STEP != 0-------------------------------------------
	for (int k = GK - (GK&(STEP - 1)); k < GK; k++)
	{
		int dY_k = k;
		Ims_fhr_fwr(dY_k, dY_fhr, dY_fwr);

		//load 1 elem from W[OC, FH, FW, IC]
		int fh = y + ((oph - dY_fhr) << 1);
		int fw = x + ((opw - dY_fwr) << 1);
		bool lw = (fh < FH) && (fw < FW);
		int Woffset = (((dY_k - dY_fwr * OC)*FH + fh)*FW + fw)*IC;
		float w = (lw ? W[Woffset] : 0);

		//load 1 elem from deltaY[N, FH, FW, OC]
		int yoffset = dY_fhr * OW_OC + dY_k;
		float y = (Ims_ldy(ohs0, ows0) ? deltaY[Yoffset0 + yoffset] : 0);

		v += y * w;
	}
	//when GK % STEP != 0-------------------------------------------

	deltaX[Xoffset0] = v;
}

#endif

#endif
