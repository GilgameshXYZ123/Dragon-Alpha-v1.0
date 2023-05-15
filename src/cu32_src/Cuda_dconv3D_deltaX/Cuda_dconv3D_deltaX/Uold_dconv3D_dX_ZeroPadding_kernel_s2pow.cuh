#pragma once

//Sparse Kernel: for sh*sw >= 4
#ifndef DECONV_3D_DELTAX_ZERO_PADDING_S2POW
#define DECONV_3D_DELTAX_ZERO_PADDING_S2POW

#define __dconv3D_deltaX_ZeroPadding_s2pow(streams,index,length, deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, N,IC,OC, sh,sw,ph,pw) \
	dconv3d_deltaX_s2pow(streams,index,length, deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, LOG2(sh),LOG2(sw),ph,pw,\
		GET_GN_ZeroPadding(IC), GET_GM_ZeroPadding(N,IH,IW), 0, 0)\

#define dconv3d_dX_ZeroPadding_s2pow_Branch(SIZE_Y, SIZE_X) {\
	int GNr = GN & (SIZE_Y), GMr = GM & (SIZE_X);\
	if(GNr && GMr){\
		int next_ic_index = (GN - GNr) + ic_index;\
		int next_j_index = (GM - GMr) + j_index;\
		dconv3d_deltaX_s2pow(streams, index, length, deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, lsh,lsw,ph,pw,\
			GNr, GM, next_ic_index, j_index);\
		dconv3d_deltaX_s2pow(streams, index, length, deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, lsh,lsw,ph,pw,\
			GN, GMr, ic_index, next_j_index);\
		dconv3d_deltaX_s2pow(streams, index, length, deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, lsh,lsw,ph,pw,\
			GNr, GMr, next_ic_index, next_j_index);}\
	else if(GNr){\
		int next_ic_index = (GN - GNr) + ic_index;\
		dconv3d_deltaX_s2pow(streams, index, length, deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, lsh,lsw,ph,pw,\
			GNr, GM, next_ic_index, j_index);}\
	else if(GMr){\
		int next_j_index = (GM - GMr) + j_index;\
		dconv3d_deltaX_s2pow(streams, index, length, deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, lsh,lsw,ph,pw,\
			GN, GMr, ic_index, next_j_index);}}

//(1) FH*FW >= 2
//(2) GN = IC;           GN >= 4, GN%4 == 0
//(3) GM = N  * IH * IW; GM >= 4, GM%4 == 0
//(4) GK = OC * FH * FW; GK >= 8, GK%4 == 0
//(5) sh, sw is power of 2
//performance=============================================================
//[OH, OW] = 15, [FH, FW] = 4, [N, OC] = 8, 17, [sh, sw] = 4, [ph, pw] = 1
//IC = 128: Size = 0.87262, Time = 0.648 msec, Performace = 2891.88 GFlop/s
//IC = 192: Size = 1.30893, Time = 0.906 msec, Performace = 3102.54 GFlop/s
//IC = 224: Size = 1.52708, Time = 1.066 msec, Performace = 3076.35 GFlop/s
//IC = 240: Size = 1.63616, Time = 1.208 msec, Performace = 2908.63 GFlop/s
//IC = 248: Size = 1.6907 , Time = 1.4   msec, Performace = 2593.39 GFlop/s
//IC = 252: Size = 1.71797, Time = 1.664 msec, Performace = 2217.13 GFlop/s
void dconv3d_deltaX_s2pow(jlong *streams, int &index, int length,
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int lsh, int lsw, int ph, int pw,
	int GN, int GM,
	int ic_index, int j_index)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)streams[index];
	index = (index + 1) % length;

	if ((GN > 63) && (GM > 7)) {//[64, 8] 2669.42 GFlop/s
		k81s2pow(stream, 3, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, lsh, lsw, ph, pw, GN, GM);
		dconv3d_dX_ZeroPadding_s2pow_Branch(63, 7); return;
	}
	if ((GN > 31) && (GM > 7)) {//[32, 8] 1784.7  GFlop/s
		k41s2pow(stream, 3, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, lsh, lsw, ph, pw, GN, GM);
		dconv3d_dX_ZeroPadding_s2pow_Branch(31, 7); return;
	}
	if (GN > 31) {//[32, 4] 1390.16 GFlop/s
		k81s2pow(stream, 2, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, lsh, lsw, ph, pw, GN, GM);
		dconv3d_dX_ZeroPadding_s2pow_Branch(31, 3); return;
	}
	if ((GN > 15) && (GM > 7)) {//[16, 8] 1075.74 GFlop/s
		k21s2pow(stream, 3, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, lsh, lsw, ph, pw, GN, GM);
		dconv3d_dX_ZeroPadding_s2pow_Branch(15, 7); return;
	}
	if (GN > 15) {//[16, 4] 786.047 GFlop/s
		k41s2pow(stream, 2, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, lsh, lsw, ph, pw, GN, GM);
		dconv3d_dX_ZeroPadding_s2pow_Branch(15, 3); return;
	}
	if ((GN > 7) && (GM > 7)) {//[16, 8] 564.099 GFlop/s
		k11s2pow(stream, 3, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, lsh, lsw, ph, pw, GN, GM);
		dconv3d_dX_ZeroPadding_s2pow_Branch(7, 7); return;
	}
	if (GN > 7) {//[8, 4] 
		k21s2pow(stream, 2, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, lsh, lsw, ph, pw, GN, GM);
		dconv3d_dX_ZeroPadding_s2pow_Branch(7, 3); return;
	}
	k11s2pow(stream, 2, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, lsh, lsw, ph, pw, GN, GM);
}


#endif

#ifndef DCONV3D_DX_ZERO_PADDING_KERNEL_S2POW_H
#define DCONV3D_DX_ZERO_PADDING_KERNEL_S2POW_H


//Sparse Matrix Method
//We have:
//(1) FH * FW >= 2
//(2) GM >= 4, GM % 4 == 0
//(3) GN >= 4, GN % 4 == 0
//(4) GK >= 8, GK % 4 == 0
//sh, sw is power of 2

#ifndef DCONV3D_DX_ZERO_PADDING_KERNEL_S2POW_CALL
#define DCONV3D_DX_ZERO_PADDING_KERNEL_S2POW_CALL

#define k81s2pow(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, lsh, lsw, ph, pw, GN, GM) \
	kernel_8_1_s2pow<LB, (1<<LB)>\
		<<< dim3(GM>>LB, GN>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, lsh,lsw,ph,pw, ic_index,j_index)

#define k41s2pow(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, lsh, lsw, ph, pw, GN, GM) \
	kernel_4_1_s2pow<LB, (1<<LB)>\
		<<< dim3(GM>>LB, GN>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, lsh,lsw,ph,pw, ic_index,j_index)

#define k21s2pow(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, lsh, lsw, ph, pw, GN, GM) \
	kernel_2_1_s2pow<LB, (1<<LB)>\
		<<< dim3(GM>>LB, GN>>LB>>1), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, lsh,lsw,ph,pw, ic_index,j_index)

#define k11s2pow(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, lsh, lsw, ph, pw, GN, GM) \
	kernel_1_1_s2pow<LB, (1<<LB)>\
		<<< dim3(GM>>LB, GN>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, lsh,lsw,ph,pw, ic_index,j_index)

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*1) 
#ifndef DCONV3D_DX_KERNEL_8_1_S2POW
#define DCONV3D_DX_KERNEL_8_1_S2POW

//for(sh,sw) = 5:
//LB = 4: Size = 5.64062, Time = 3.016 msec, Performace = 4016.3 GFlop/s
//for(sh,sw) = 4:
//LB = 4: Size = 0.87262, Time = 0.692 msec, Performace = 2708    GFlop/s
//LB = 3: Size = 3.98853, Time = 2.136 msec, Performace = 4009.97 GFlop/s
//LB = 3: Size = 0.87262, Time = 0.656 msec, Performace = 2856.61 GFlop/s
template<int LB, int STEP>
__global__ void kernel_8_1_s2pow(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
		  float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int lsh, int lsw, int ph, int pw,
	int ic_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float   dYs[2][1 << LB << LB];//[1 << LB][1 << LB]
	__shared__ int Woffset[2][1 << LB << LB];//[1 << LB][1 << LB]

	//preapre for GN = IC
	const int ic0 = (((blockIdx.y << LB) + ty) << 3) + ic_index, ic4 = ic0 + 4;

	//prepared for GM = N * IH * IW
	int j = ((blockIdx.x << LB) + tx) + j_index;
	const int IH_IW = IH * IW; get_n_ih_iw(j, n, ih, iw);
	int tih = ih - (FH - ph - 1);//tih = ih - oph
	int tiw = iw - (FW - pw - 1);//tiw = iw - opw

	//find (fhs, fws) to meet: (ohp % sh==0) && (ohp >= 0 && ohp < OHp), owp too
	const int sh_m1 = (1 << lsh) - 1, sw_m1 = (1 << lsw) - 1;
	const int OHp = (OH << lsh) - sh_m1, OWp = (OW << lsw) - sw_m1;
	int fhs = 0, fws = 0; FIND_FHS_FWS_s2pow(fhs, fws, tih, tiw, KW_8_1_end);

	//prepare for GK = FH * FW * OC: 
	//compress: GK -> GKr = FHr * FWr * OC, (fh, fw) -> (fhr, fwr), ignore inner-padding zeros
	const int FHr = (FH - fhs + sh_m1) >> lsh;//compress FH -> FHr
	const int FWr = (FW - fws + sw_m1) >> lsw;//compress FW -> FWr
	const int FWr_OC = FWr * OC, GKr = FHr * FWr_OC, OK = GKr >> LB;
	const int toh = (tih + fhs) >> lsh, tow = (tiw + fws) >> lsw;

	//load 1 element from deltaY[N, OH, OW, OC] : deltaYp[N, OH=OHp, OW=OWp, OC]
	int dYW_k = ty, oc, fhr, fwr;
	get_fhr_fwr_oc(dYW_k, fhr, fwr, oc);
	int oh = toh + fhr, ow = tow + fwr;//the same: tx -> j -> n, toh, tow
	const int ty_tx = (ty << LB) + tx;
	if (OK) {
		LOAD_DY_GET_WOFFSET_s2pow(
			dYs[buf][ty_tx], n, oh, ow, oc,
			Woffset[buf][ty_tx], fhs, fws, fhr, fwr);
		__syncthreads();
	}

	//compute area-----------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0);//v00 -> v40
	float4 v1 = make_float4(0, 0, 0, 0);//v40 -> v70
	for (int ok = 1; ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			int buf_index = (ik << LB) + tx;
			float dy = dYs[buf][buf_index];
			int W_oc_fh_fw = Woffset[buf][buf_index];
			float4 w0 = *(float4*)(&W[W_oc_fh_fw + ic0]);//W[oc, fh, fw, ic0 -> ic3]
			float4 w1 = *(float4*)(&W[W_oc_fh_fw + ic4]);//W[oc, fh, fw, ic4 -> ic7]
			simdMM4(v0, dy, w0);
			simdMM4(v1, dy, w1);
		}

		buf ^= 1;
		//load 1 element from deltaY[N, OH, OW, OC] : deltaYp[N, OH=OHp, OW=OWp, OC]
		dYW_k = (ok << LB) + ty;//with the same tx -> the same n, ih, iw -> the same n, toh, tow
		get_fhr_fwr_oc(dYW_k, fhr, fwr, oc);
		oh = toh + fhr, ow = tow + fwr;
		LOAD_DY_GET_WOFFSET_s2pow(
			dYs[buf][ty_tx], n, oh, ow, oc,
			Woffset[buf][ty_tx], fhs, fws, fhr, fwr);
		__syncthreads();
	}
	if (OK)
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			int buf_index = (ik << LB) + tx;
			float dy = dYs[buf][buf_index];
			int W_oc_fh_fw = Woffset[buf][buf_index];
			float4 w0 = *(float4*)(&W[W_oc_fh_fw + ic0]);//W[oc, fh, fw, ic0 -> ic3]
			float4 w1 = *(float4*)(&W[W_oc_fh_fw + ic4]);//W[oc, fh, fw, ic4 -> ic7]
			simdMM4(v0, dy, w0);
			simdMM4(v1, dy, w1);
		}

	//for GKr % STEP != 0---------------------------------------------
	for (int k = GKr - (GKr&(STEP - 1)); k < GKr; k++)
	{
		dYW_k = k;
		get_fhr_fwr_oc(dYW_k, fhr, fwr, oc);
		oh = toh + fhr, ow = tow + fwr;

		float dy; LOAD_DY(dy, n, oh, ow, oc);
		int fh = FH - 1 - ((fhr << lsh) + fhs);
		int fw = FW - 1 - ((fwr << lsw) + fws);
		int W_oc_fh_fw = (((oc * FH) + fh)*FW + fw)*IC;
		float4 w0 = *(float4*)(&W[W_oc_fh_fw + ic0]);
		float4 w1 = *(float4*)(&W[W_oc_fh_fw + ic4]);
		simdMM4(v0, dy, w0);
		simdMM4(v1, dy, w1);
	}
	//for GKr % STEP != 0---------------------------------------------

	j *= IC;//j -> [n, oh, ow]
	*(float4*)(&deltaX[j + ic0]) = v0;//v00 -> v40
	*(float4*)(&deltaX[j + ic4]) = v1;//v40 -> v70
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*1) 
#ifndef DCONV3D_DX_KERNEL_4_1_S2POW
#define DCONV3D_DX_KERNEL_4_1_S2POW

//LB = 4: Size = 0.87262, Time = 0.734 msec, Performace = 2553.05 GFlop/s
//LB = 3: Size = 0.87262, Time = 0.828 msec, Performace = 2263.21 GFlop/s
template<int LB, int STEP>
__global__ void kernel_4_1_s2pow(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
		  float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int lsh, int lsw, int ph, int pw,
	int ic_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float   dYs[2][1 << LB << LB];//[1 << LB][1 << LB]
	__shared__ int Woffset[2][1 << LB << LB];//[1 << LB][1 << LB]

	//preapre for GN = IC
	const int ic0 = (((blockIdx.y << LB) + ty) << 2) + ic_index;

	//prepared for GM = N * IH * IW
	int j = ((blockIdx.x << LB) + tx) + j_index;
	const int IH_IW = IH * IW; get_n_ih_iw(j, n, ih, iw);
	int tih = ih - (FH - ph - 1);//tih = ih - oph
	int tiw = iw - (FW - pw - 1);//tiw = iw - opw

	//find (fhs, fws) to meet: (ohp % sh==0) && (ohp >= 0 && ohp < OHp), owp too
	const int sh_m1 = (1 << lsh) - 1, sw_m1 = (1 << lsw) - 1;
	const int OHp = (OH << lsh) - sh_m1, OWp = (OW << lsw) - sw_m1;
	int fhs = 0, fws = 0; FIND_FHS_FWS_s2pow(fhs, fws, tih, tiw, KW_8_1_end);

	//prepare for GK = FH * FW * OC: 
	//compress: GK -> GKr = FHr * FWr * OC, (fh, fw) -> (fhr, fwr), ignore inner-padding zeros
	const int FHr = (FH - fhs + sh_m1) >> lsh;//compress FH -> FHr
	const int FWr = (FW - fws + sw_m1) >> lsw;//compress FW -> FWr
	const int FWr_OC = FWr * OC, GKr = FHr * FWr_OC, OK = GKr >> LB;
	const int toh = (tih + fhs) >> lsh, tow = (tiw + fws) >> lsw;

	//load 1 element from deltaY[N, OH, OW, OC] : deltaYp[N, OH=OHp, OW=OWp, OC]
	int dYW_k = ty, oc, fhr, fwr;
	get_fhr_fwr_oc(dYW_k, fhr, fwr, oc);
	int oh = toh + fhr, ow = tow + fwr;//the same: tx -> j -> n, toh, tow
	const int ty_tx = (ty << LB) + tx;

	if (OK) {
		LOAD_DY_GET_WOFFSET_s2pow(
			dYs[buf][ty_tx], n, oh, ow, oc,
			Woffset[buf][ty_tx], fhs, fws, fhr, fwr);
		__syncthreads();
	}

	//compute area-----------------------------------------
	float4 v = make_float4(0, 0, 0, 0);//v00 -> v40
	for (int ok = 1; ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			int buf_index = (ik << LB) + tx;
			float dy = dYs[buf][buf_index];
			int W_oc_fh_fw = Woffset[buf][buf_index];
			float4 w0 = *(float4*)(&W[W_oc_fh_fw + ic0]);//W[oc, fh, fw, ic0 -> ic3]
			simdMM4(v, dy, w0);
		}

		buf ^= 1;
		//load 1 element from deltaY[N, OH, OW, OC] : deltaYp[N, OH=OHp, OW=OWp, OC]
		dYW_k = (ok << LB) + ty;//with the same tx -> the same n, ih, iw -> the same n, toh, tow
		get_fhr_fwr_oc(dYW_k, fhr, fwr, oc);
		oh = toh + fhr; ow = tow + fwr;
		LOAD_DY_GET_WOFFSET_s2pow(
			dYs[buf][ty_tx], n, oh, ow, oc,
			Woffset[buf][ty_tx], fhs, fws, fhr, fwr);
		__syncthreads();
	}
	if (OK)
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			int buf_index = (ik << LB) + tx;
			float dy = dYs[buf][buf_index];
			int W_oc_fh_fw = Woffset[buf][buf_index];
			float4 w0 = *(float4*)(&W[W_oc_fh_fw + ic0]);//W[oc, fh, fw, ic0 -> ic3]
			simdMM4(v, dy, w0);
		}

	//for GKr % STEP != 0---------------------------------------------
	for (int k = GKr - (GKr&(STEP - 1)); k < GKr; k++)
	{
		dYW_k = k;
		get_fhr_fwr_oc(dYW_k, fhr, fwr, oc);
		oh = toh + fhr, ow = tow + fwr;

		float dy; LOAD_DY(dy, n, oh, ow, oc);
		int fh = FH - 1 - ((fhr << lsh) + fhs);
		int fw = FW - 1 - ((fwr << lsw) + fws);
		int W_oc_fh_fw = (((oc * FH) + fh)*FW + fw)*IC;
		float4 w0 = *(float4*)(&W[W_oc_fh_fw + ic0]);
		simdMM4(v, dy, w0);
	}
	//for GKr % STEP != 0---------------------------------------------

	*(float4*)(&deltaX[j * IC + ic0]) = v;//v00 -> v40
}

#endif


//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*1) 
#ifndef DCONV3D_DX_KERNEL_2_1_S2POW
#define DCONV3D_DX_KERNEL_2_1_S2POW

//LB = 4: Size = 0.87262, Time = 1.254 msec, Performace = 1494.37 GFlop/s
//LB = 3: Size = 0.87262, Time = 1.346 msec, Performace = 1392.23 GFlop/s
template<int LB, int STEP>
__global__ void kernel_2_1_s2pow(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
		  float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int lsh, int lsw, int ph, int pw,
	int ic_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float   dYs[2][1 << LB << LB];//[1 << LB][1 << LB]
	__shared__ int Woffset[2][1 << LB << LB];//[1 << LB][1 << LB]

	//preapre for GN = IC
	const int ic0 = (((blockIdx.y << LB) + ty) << 1) + ic_index;

	//prepared for GM = N * IH * IW
	int j = ((blockIdx.x << LB) + tx) + j_index;
	const int IH_IW = IH * IW; get_n_ih_iw(j, n, ih, iw);
	int tih = ih - (FH - ph - 1);//tih = ih - oph
	int tiw = iw - (FW - pw - 1);//tiw = iw - opw

	//find (fhs, fws) to meet: (ohp % sh==0) && (ohp >= 0 && ohp < OHp), owp too
	const int sh_m1 = (1 << lsh) - 1, sw_m1 = (1 << lsw) - 1;
	const int OHp = (OH << lsh) - sh_m1, OWp = (OW << lsw) - sw_m1;
	int fhs = 0, fws = 0; FIND_FHS_FWS_s2pow(fhs, fws, tih, tiw, KW_8_1_end);

	//prepare for GK = FH * FW * OC: 
	//compress: GK -> GKr = FHr * FWr * OC, (fh, fw) -> (fhr, fwr), ignore inner-padding zeros
	const int FHr = (FH - fhs + sh_m1) >> lsh;//compress FH -> FHr
	const int FWr = (FW - fws + sw_m1) >> lsw;//compress FW -> FWr
	const int FWr_OC = FWr * OC, GKr = FHr * FWr_OC, OK = GKr >> LB;
	const int toh = (tih + fhs) >> lsh, tow = (tiw + fws) >> lsw;

	//load 1 element from deltaY[N, OH, OW, OC] : deltaYp[N, OH=OHp, OW=OWp, OC]
	int dYW_k = ty, oc, fhr, fwr;
	get_fhr_fwr_oc(dYW_k, fhr, fwr, oc);
	int oh = toh + fhr, ow = tow + fwr;//the same: tx -> j -> n, toh, tow
	const int ty_tx = (ty << LB) + tx;

	if (OK) {
		LOAD_DY_GET_WOFFSET_s2pow(
			dYs[buf][ty_tx], n, oh, ow, oc,
			Woffset[buf][ty_tx], fhs, fws, fhr, fwr);
		__syncthreads();
	}

	//compute area-----------------------------------------
	float2 v = make_float2(0, 0);//v00 -> v20
	for (int ok = 1; ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			int buf_index = (ik << LB) + tx;
			float dy = dYs[buf][buf_index];
			int W_oc_fh_fw = Woffset[buf][buf_index];
			float2 w = *(float2*)(&W[W_oc_fh_fw + ic0]);//W[oc, fh, fw, ic0 -> ic3]
			simdMM2(v, dy, w);
		}

		buf ^= 1;
		//load 1 element from deltaY[N, OH, OW, OC] : deltaYp[N, OH=OHp, OW=OWp, OC]
		dYW_k = (ok << LB) + ty;//with the same tx -> the same n, ih, iw -> the same n, toh, tow
		get_fhr_fwr_oc(dYW_k, fhr, fwr, oc);
		oh = toh + fhr; ow = tow + fwr;
		LOAD_DY_GET_WOFFSET_s2pow(
			dYs[buf][ty_tx], n, oh, ow, oc,
			Woffset[buf][ty_tx], fhs, fws, fhr, fwr);
		__syncthreads();
	}
	if (OK)
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			int buf_index = (ik << LB) + tx;
			float dy = dYs[buf][buf_index];
			int W_oc_fh_fw = Woffset[buf][buf_index];
			float2 w = *(float2*)(&W[W_oc_fh_fw + ic0]);//W[oc, fh, fw, ic0 -> ic3]
			simdMM2(v, dy, w);
		}

	//for GKr % STEP != 0---------------------------------------------
	for (int k = GKr - (GKr&(STEP - 1)); k < GKr; k++)
	{
		dYW_k = k;
		get_fhr_fwr_oc(dYW_k, fhr, fwr, oc);
		oh = toh + fhr, ow = tow + fwr;
		float dy; LOAD_DY(dy, n, oh, ow, oc);
		int fh = FH - 1 - ((fhr << lsh) + fhs);
		int fw = FW - 1 - ((fwr << lsw) + fws);
		int W_oc_fh_fw = (((oc * FH) + fh)*FW + fw)*IC;
		float2 w0 = *(float2*)(&W[W_oc_fh_fw + ic0]);
		simdMM2(v, dy, w0);
	}
	//for GKr % STEP != 0---------------------------------------------

	*(float2*)(&deltaX[j * IC + ic0]) = v;//v00 -> v20
}

#endif


//(Y: BLOCK_SIZE*1, X: BLOCK_SIZE*1) 
#ifndef DCONV3D_DX_KERNEL_1_1_S2POW
#define DCONV3D_DX_KERNEL_1_1_S2POW

//LB = 4: Size = 0.87262, Time = 2.458 msec, Performace = 762.383 GFlop/s
//LB = 3: Size = 0.87262, Time = 2.418 msec, Performace = 774.994 GFlop/s
template<int LB, int STEP>
__global__ void kernel_1_1_s2pow(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
		  float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int lsh, int lsw, int ph, int pw,
	int ic_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float   dYs[2][1 << LB << LB];//[1 << LB][1 << LB]
	__shared__ int Woffset[2][1 << LB << LB];//[1 << LB][1 << LB]

	//preapre for GN = IC
	const int ic0 = ((blockIdx.y << LB) + ty) + ic_index;

	//prepared for GM = N * IH * IW
	int j = ((blockIdx.x << LB) + tx) + j_index;
	const int IH_IW = IH * IW; get_n_ih_iw(j, n, ih, iw);
	int tih = ih - (FH - ph - 1);//tih = ih - oph
	int tiw = iw - (FW - pw - 1);//tiw = iw - opw

	//find (fhs, fws) to meet: (ohp % sh==0) && (ohp >= 0 && ohp < OHp), owp too
	const int sh_m1 = (1 << lsh) - 1, sw_m1 = (1 << lsw) - 1;
	const int OHp = (OH << lsh) - sh_m1, OWp = (OW << lsw) - sw_m1;
	int fhs = 0, fws = 0; FIND_FHS_FWS_s2pow(fhs, fws, tih, tiw, KW_8_1_end);

	//prepare for GK = FH * FW * OC: 
	//compress: GK -> GKr = FHr * FWr * OC, (fh, fw) -> (fhr, fwr), ignore inner-padding zeros
	const int FHr = (FH - fhs + sh_m1) >> lsh;//compress FH -> FHr
	const int FWr = (FW - fws + sw_m1) >> lsw;//compress FW -> FWr
	const int FWr_OC = FWr * OC, GKr = FHr * FWr_OC, OK = GKr >> LB;
	const int toh = (tih + fhs) >> lsh, tow = (tiw + fws) >> lsw;

	//load 1 element from deltaY[N, OH, OW, OC] : deltaYp[N, OH=OHp, OW=OWp, OC]
	int dYW_k = ty, oc, fhr, fwr;
	get_fhr_fwr_oc(dYW_k, fhr, fwr, oc);
	int oh = toh + fhr, ow = tow + fwr;//the same: tx -> j -> n, toh, tow
	const int ty_tx = (ty << LB) + tx;

	if (OK) {
		LOAD_DY_GET_WOFFSET_s2pow(
			dYs[buf][ty_tx], n, oh, ow, oc,
			Woffset[buf][ty_tx], fhs, fws, fhr, fwr);
		__syncthreads();
	}

	//compute area-----------------------------------------
	float v = 0.0f;//v00 -> v20
	for (int ok = 1; ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			int buf_index = (ik << LB) + tx;
			float dy = dYs[buf][buf_index];
			int W_oc_fh_fw = Woffset[buf][buf_index];
			float w = W[W_oc_fh_fw + ic0];//W[oc, fh, fw, ic0 -> ic3]
			v += dy * w;
		}

		buf ^= 1;
		//load 1 element from deltaY[N, OH, OW, OC] : deltaYp[N, OH=OHp, OW=OWp, OC]
		dYW_k = (ok << LB) + ty;//with the same tx -> the same n, ih, iw -> the same n, toh, tow
		get_fhr_fwr_oc(dYW_k, fhr, fwr, oc);
		oh = toh + fhr; ow = tow + fwr;
		LOAD_DY_GET_WOFFSET_s2pow(
			dYs[buf][ty_tx], n, oh, ow, oc,
			Woffset[buf][ty_tx], fhs, fws, fhr, fwr);
		__syncthreads();
	}
	if (OK)
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			int buf_index = (ik << LB) + tx;
			float dy = dYs[buf][buf_index];
			int W_oc_fh_fw = Woffset[buf][buf_index];
			float w = W[W_oc_fh_fw + ic0];//W[oc, fh, fw, ic0 -> ic3]
			v += dy * w;
		}

	//for GKr % STEP != 0---------------------------------------------
	for (int k = GKr - (GKr&(STEP - 1)); k < GKr; k++)
	{
		dYW_k = k;
		get_fhr_fwr_oc(dYW_k, fhr, fwr, oc);
		oh = toh + fhr, ow = tow + fwr;

		float dy; LOAD_DY(dy, n, oh, ow, oc);
		int fh = FH - 1 - ((fhr << lsh) + fhs);
		int fw = FW - 1 - ((fwr << lsw) + fws);
		int W_oc_fh_fw = (((oc * FH) + fh)*FW + fw)*IC;
		float w = W[W_oc_fh_fw + ic0];
		v += dy * w;
	}
	//for GKr % STEP != 0---------------------------------------------

	deltaX[j * IC + ic0] = v;//v00 -> v20
}

#endif


#endif