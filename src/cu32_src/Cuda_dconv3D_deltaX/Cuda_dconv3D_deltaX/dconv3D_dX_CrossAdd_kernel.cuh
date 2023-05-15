#pragma once

#ifndef DCONV3D_DELTAX_CROSS_ADD_KERNEL_H
#define DCONV3D_DELTAX_CROSS_ADD_KERNEL_H


#ifndef DCONV3D_DELTAX_CROSS_ADD_KERNEL_CALL
#define DCONV3D_DELTAX_CROSS_ADD_KERNEL_CALL

//LB = log2(BLOCK_SIZE)

//(128, 256)
#define crossAdd_k16_2(stream, LB, oc_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, sh, sw, ph, pw, GN, GM) \
	crossAdd_kernel_16_2<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>1, GN>>LB>>4), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, sh,sw,ph,pw, oc_index, j_index)

#define crossAdd_k82(stream, LB, oc_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, sh, sw, ph, pw, GN, GM) \
	crossAdd_kernel_8_2<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>1, GN>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, sh,sw,ph,pw, oc_index, j_index)

#define crossAdd_k42(stream, LB, oc_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, sh, sw, ph, pw, GN, GM) \
	crossAdd_kernel_4_2<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>1, GN>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, sh,sw,ph,pw, oc_index, j_index)

#define crossAdd_k22(stream, LB, oc_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, sh, sw, ph, pw, GN, GM) \
	crossAdd_kernel_2_2<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>1, GN>>LB>>1), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, sh,sw,ph,pw, oc_index, j_index)

#define crossAdd_k11(stream, LB, oc_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, sh, sw, ph, pw, GN, GM) \
	crossAdd_kernel_1_1<LB, (1<<LB)>\
		<<< dim3(GM>>LB, GN>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, sh,sw,ph,pw, oc_index, j_index)

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*2)
//LB = 4: GK % 8 == 0
#ifndef DCONV3D_DX_CROSS_ADD_KERNEL_16_2
#define DCONV3D_DX_CROSS_ADD_KERNEL_16_2

//Size = 0.25, Time = 0.364 msec, Performace = 1474.92 GFlop/s
template<int LB, int STEP>
__global__ void crossAdd_kernel_16_2(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__      W, int FH, int FW,
		  float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4  Ws[2][1 << LB >> 1][(4 << LB) + 1];
	__shared__ float  dXs[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ int Xaddrs[2][1 << LB >> 1][(2 << LB) + 2];

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 4) + oc_index;
	const int FW_IC = FW * IC, GK = FH * FW_IC;
	const int toc0 = (((tx & 1) << 3) + oc0)*GK;
	const int toc1 = toc0 + GK, toc2 = toc1 + GK, toc3 = toc2 + GK;
	const int toc4 = toc3 + GK, toc5 = toc4 + GK, toc6 = toc5 + GK, toc7 = toc6 + GK;

	//prepared for GM = N * OH * OW
	int j0 = (((blockIdx.x << LB) + tx) << 1) + j_index, j1 = j0 + 1;
	const int OH_OW = OH * OW;
	get_n_oh_ow(j0, n0, oh0, ow0);
	get_n_oh_ow(j1, n1, oh1, ow1);
	bool flagY = (ty & 1);
	const int tn0 = (n1 - n0)*flagY + n0;
	const int tih0 = ((oh1 - oh0)*flagY + oh0) * sh - ph;
	const int tiw0 = ((ow1 - ow0)*flagY + ow0) * sw - pw;
	const int Xoffset0 = ((tn0*IH + tih0)*IW + tiw0)*IC;

	//load 4 elements from W[OC, FH, FW, IC]
	const int W_k = tx >> 1;
	const int Ws_x = (tx >> 1), Ws_y = ((ty << 1) + (tx & 1)) << 1;
	Ws[buf][Ws_x][Ws_y].x = W[toc0 + W_k];
	Ws[buf][Ws_x][Ws_y].y = W[toc1 + W_k];
	Ws[buf][Ws_x][Ws_y].z = W[toc2 + W_k];
	Ws[buf][Ws_x][Ws_y].w = W[toc3 + W_k];
	Ws[buf][Ws_x][Ws_y + 1].x = W[toc4 + W_k];
	Ws[buf][Ws_x][Ws_y + 1].y = W[toc5 + W_k];
	Ws[buf][Ws_x][Ws_y + 1].z = W[toc6 + W_k];
	Ws[buf][Ws_x][Ws_y + 1].w = W[toc7 + W_k];

	//compute 1 address for Xaddr[M, IH, IW, IC]
	int dX_k = ty >> 1, dX_fh, dX_fw; getX_fh_fw(dX_k, dX_fh, dX_fw);
	bool write0 = (tih0 >= -dX_fh) && (tih0 < IH - dX_fh) && (tiw0 >= -dX_fw) && (tiw0 < IW - dX_fw);
	const int IW_IC = IW * IC;
	int xoffset = Xoffset0 + (dX_fh*IW_IC) + dX_k;
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);
	Xaddrs[buf][Xs_y][Xs_x] = (write0 * xoffset) - !write0;
	dXs[buf][Xs_y][Xs_x] = 0;
	__syncthreads();

	//compute area-----------------------------------------
	j0 = j0 * OC + oc0; j1 = j0 + OC;
	float4 dy00 = *(float4*)(&deltaY[j0]);
	float4 dy01 = *(float4*)(&deltaY[j0 + 4]);
	float4 dy02 = *(float4*)(&deltaY[j0 + 8]);
	float4 dy03 = *(float4*)(&deltaY[j0 + 12]);

	float4 dy10 = *(float4*)(&deltaY[j1]);
	float4 dy11 = *(float4*)(&deltaY[j1 + 4]);
	float4 dy12 = *(float4*)(&deltaY[j1 + 8]);
	float4 dy13 = *(float4*)(&deltaY[j1 + 12]);

	for (int ok = 1, OK = GK << 1 >> LB; ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 w0 = Ws[buf][ik][(ty << 2)];
			float4 w1 = Ws[buf][ik][(ty << 2) + 1];
			float4 w2 = Ws[buf][ik][(ty << 2) + 2];
			float4 w3 = Ws[buf][ik][(ty << 2) + 3];
			int2 xoffset = *(int2*)(&Xaddrs[buf][ik][tx << 1]);

			float dx0 = CrossAdd_SUM4(w0, dy00) + 
				        CrossAdd_SUM4(w1, dy01) + 
				        CrossAdd_SUM4(w2, dy02) +
				        CrossAdd_SUM4(w3, dy03);
			float dx1 = CrossAdd_SUM4(w0, dy10) + 
						CrossAdd_SUM4(w1, dy11) +
						CrossAdd_SUM4(w2, dy12) +
						CrossAdd_SUM4(w3, dy13);

			dx0 *= (xoffset.x != -1);
			dx1 *= (xoffset.y != -1);

			atomicAdd_block(&(dXs[buf][ik][(tx << 1)]), dx0);
			atomicAdd_block(&(dXs[buf][ik][(tx << 1) + 1]), dx1);
		}
		__syncthreads();
		int Xaddr = Xaddrs[buf][Xs_y][Xs_x];
		if (Xaddr != -1) atomicAdd(deltaX + Xaddr, dXs[buf][Xs_y][Xs_x]);
		buf ^= 1;

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((ok << LB) + tx) >> 1;
		Ws[buf][Ws_x][Ws_y].x = W[toc0 + W_k];
		Ws[buf][Ws_x][Ws_y].y = W[toc1 + W_k];
		Ws[buf][Ws_x][Ws_y].z = W[toc2 + W_k];
		Ws[buf][Ws_x][Ws_y].w = W[toc3 + W_k];
		Ws[buf][Ws_x][Ws_y + 1].x = W[toc4 + W_k];
		Ws[buf][Ws_x][Ws_y + 1].y = W[toc5 + W_k];
		Ws[buf][Ws_x][Ws_y + 1].z = W[toc6 + W_k];
		Ws[buf][Ws_x][Ws_y + 1].w = W[toc7 + W_k];

		//compute 1 address for Xaddr[M, IH, IW, IC]
		int dX_k = ((ok << LB) + ty) >> 1; getX_fh_fw(dX_k, dX_fh, dX_fw);
		bool write0 = (tih0 >= -dX_fh) && (tih0 < IH - dX_fh) && (tiw0 >= -dX_fw) && (tiw0 < IW - dX_fw);
		int xoffset = Xoffset0 + (dX_fh*IW_IC) + dX_k;
		Xaddrs[buf][Xs_y][Xs_x] = (write0 * xoffset) - !write0;
		dXs[buf][Xs_y][Xs_x] = 0;
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 w0 = Ws[buf][ik][(ty << 2)];
		float4 w1 = Ws[buf][ik][(ty << 2) + 1];
		float4 w2 = Ws[buf][ik][(ty << 2) + 2];
		float4 w3 = Ws[buf][ik][(ty << 2) + 3];
		int2 xoffset = *(int2*)(&Xaddrs[buf][ik][tx << 1]);

		float dx0 = CrossAdd_SUM4(w0, dy00) +
					CrossAdd_SUM4(w1, dy01) +
					CrossAdd_SUM4(w2, dy02) +
					CrossAdd_SUM4(w3, dy03);
		float dx1 = CrossAdd_SUM4(w0, dy10) +
					CrossAdd_SUM4(w1, dy11) +
					CrossAdd_SUM4(w2, dy12) +
					CrossAdd_SUM4(w3, dy13);
		dx0 *= (xoffset.x != -1);
		dx1 *= (xoffset.y != -1);

		atomicAdd_block(&(dXs[buf][ik][(tx << 1)]), dx0);
		atomicAdd_block(&(dXs[buf][ik][(tx << 1) + 1]), dx1);
	}
	__syncthreads();
	int Xaddr = Xaddrs[buf][Xs_y][Xs_x];
	if (Xaddr != -1) atomicAdd(deltaX + Xaddr, dXs[buf][Xs_y][Xs_x]);
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*2)
//LB = 4: GK % 8 == 0
#ifndef DCONV3D_DX_CROSS_ADD_KERNEL_8_2
#define DCONV3D_DX_CROSS_ADD_KERNEL_8_2

//LB = 4: IC % 8 == 0
//LB = 3: IC % 4 == 0
//for(sh, sw) = 2: 
//LB = 4(OC >= 128): Size = 0.234619, Time = 0.382 msec, Performace = 1318.95 GFlop/s
//LB = 3(OC >=  64): Size = 0.234619, Time = 0.442 msec, Performace = 1139.91 GFlop/s
//for(sh, sw) = 1:
//LB = 4, IC = 8: Size = 0.0625, Time = 0.382 msec, Performace = 351.355 GFlop/s
//LB = 3, IC = 4, N = 4: Size = 0.03125, Time = 0.232 msec, Performace = 289.262 GFlop/s
//LB = 3, IC = 4, N = 32: Size = 0.25, Time = 1.51 msec, Performace = 355.544 GFlop/s
//LB = 4, IC = 8, N = 32: Size = 0.5, Time = 2.506 msec, Performace = 428.468 GFlop/s
template<int LB, int STEP>
__global__ void crossAdd_kernel_8_2(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4  Ws[2][1 << LB >> 1][(2 << LB) + 1];
	__shared__ float  dXs[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ int Xaddrs[2][1 << LB >> 1][(2 << LB) + 2];

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 3) + oc_index;
	const int FW_IC = FW * IC, GK = FH * FW_IC;
	const int toc0 = (((tx & 1) << 2) + oc0)*GK;
	const int toc1 = toc0 + GK, toc2 = toc1 + GK, toc3 = toc2 + GK;
	
	//prepared for GM = N * OH * OW
	int j0 = (((blockIdx.x << LB) + tx) << 1) + j_index, j1 = j0 + 1;
	const int OH_OW = OH * OW;
	get_n_oh_ow(j0, n0, oh0, ow0);
	get_n_oh_ow(j1, n1, oh1, ow1);
	bool flagY = (ty & 1);
	const int tn0 = (n1 - n0)*flagY + n0;
	const int tih0 = ((oh1 - oh0)*flagY + oh0) * sh - ph;
	const int tiw0 = ((ow1 - ow0)*flagY + ow0) * sw - pw; 
	const int Xoffset0 = ((tn0*IH + tih0)*IW + tiw0)*IC;

	//load 4 elements from W[OC, FH, FW, IC]
	const int W_k = tx >> 1;
	const int Ws_x = (tx >> 1), Ws_y = (ty << 1) + (tx & 1);
	Ws[buf][Ws_x][Ws_y].x = W[toc0 + W_k];
	Ws[buf][Ws_x][Ws_y].y = W[toc1 + W_k];
	Ws[buf][Ws_x][Ws_y].z = W[toc2 + W_k];
	Ws[buf][Ws_x][Ws_y].w = W[toc3 + W_k];
	
	//compute 1 address for Xaddr[M, IH, IW, IC]
	int dX_k = ty >> 1, dX_fh, dX_fw; getX_fh_fw(dX_k, dX_fh, dX_fw);
	bool write0 = (tih0 >= -dX_fh) && (tih0 < IH - dX_fh) && (tiw0 >= -dX_fw) && (tiw0 < IW - dX_fw);
	const int IW_IC = IW * IC;
	int xoffset = Xoffset0 + (dX_fh*IW_IC) + dX_k;
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);
	Xaddrs[buf][Xs_y][Xs_x] = (write0 * xoffset) - !write0;
	dXs[buf][Xs_y][Xs_x] = 0;
	__syncthreads();

	//compute area-----------------------------------------
	j0 = j0 * OC + oc0; j1 = j0 + OC;
	float4 dy00 = *(float4*)(&deltaY[j0]), dy01 = *(float4*)(&deltaY[j0 + 4]);
	float4 dy10 = *(float4*)(&deltaY[j1]), dy11 = *(float4*)(&deltaY[j1 + 4]);
	for (int ok = 1, OK = GK << 1 >> LB; ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 w0 = Ws[buf][ik][(ty << 1)];
			float4 w1 = Ws[buf][ik][(ty << 1) + 1];
			int2 xoffset = *(int2*)(&Xaddrs[buf][ik][tx << 1]);

			float dx0 = CrossAdd_SUM4(w0, dy00) + CrossAdd_SUM4(w1, dy01);
			float dx1 = CrossAdd_SUM4(w0, dy10) + CrossAdd_SUM4(w1, dy11);
			dx0 *= (xoffset.x != -1);
			dx1 *= (xoffset.y != -1);

			atomicAdd_block(&(dXs[buf][ik][(tx << 1)]), dx0);
			atomicAdd_block(&(dXs[buf][ik][(tx << 1) + 1]), dx1);
		}
		__syncthreads();
		int Xaddr = Xaddrs[buf][Xs_y][Xs_x];
		if(Xaddr != -1) atomicAdd(deltaX + Xaddr, dXs[buf][Xs_y][Xs_x]);
		buf ^= 1;

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((ok << LB) + tx) >> 1;
		Ws[buf][Ws_x][Ws_y].x = W[toc0 + W_k];
		Ws[buf][Ws_x][Ws_y].y = W[toc1 + W_k];
		Ws[buf][Ws_x][Ws_y].z = W[toc2 + W_k];
		Ws[buf][Ws_x][Ws_y].w = W[toc3 + W_k];

		//compute 1 address for Xaddr[M, IH, IW, IC]
		int dX_k = ((ok << LB) + ty) >> 1; getX_fh_fw(dX_k, dX_fh, dX_fw);
		bool write0 = (tih0 >= -dX_fh) && (tih0 < IH - dX_fh) && (tiw0 >= -dX_fw) && (tiw0 < IW - dX_fw);
		int xoffset = Xoffset0 + (dX_fh*IW_IC) + dX_k;
		Xaddrs[buf][Xs_y][Xs_x] = (write0 * xoffset) - !write0;
		dXs[buf][Xs_y][Xs_x] = 0;
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 w0 = Ws[buf][ik][(ty << 1)];
		float4 w1 = Ws[buf][ik][(ty << 1) + 1];
		int2 xaddr = *(int2*)(&Xaddrs[buf][ik][tx << 1]);

		float dx0 = CrossAdd_SUM4(w0, dy00) + CrossAdd_SUM4(w1, dy01);
		float dx1 = CrossAdd_SUM4(w0, dy10) + CrossAdd_SUM4(w1, dy11);
		dx0 *= (xaddr.x != -1);
		dx1 *= (xaddr.y != -1);

		atomicAdd_block(&(dXs[buf][ik][(tx << 1)]), dx0);
		atomicAdd_block(&(dXs[buf][ik][(tx << 1) + 1]), dx1);
	}
	__syncthreads();
	int Xaddr = Xaddrs[buf][Xs_y][Xs_x];
	if (Xaddr != -1) atomicAdd(deltaX + Xaddr, dXs[buf][Xs_y][Xs_x]);
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*2)
//LB = 4: GK % 8 == 0
#ifndef DCONV3D_DX_CROSS_ADD_KERNEL_4_2
#define DCONV3D_DX_CROSS_ADD_KERNEL_4_2

//for(sh, sw) = 2: 
//LB = 4(OC >= 64): Size = 0.234619, Time = 0.582 msec, Performace = 865.706 GFlop/s
//LB = 3(OC >= 32): Size = 0.234619, Time = 0.676 msec, Performace = 745.327 GFlop/s
template<int LB, int STEP>
__global__ void crossAdd_kernel_4_2(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float2  Ws[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float  dXs[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ int Xaddrs[2][1 << LB >> 1][(2 << LB) + 2];

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 2) + oc_index;
	const int FW_IC = FW * IC, GK = FH * FW_IC;
	const int toc0 = (((tx & 1) << 1) + oc0)*GK, toc1 = toc0 + GK;

	//prepared for GM = N * OH * OW
	int j0 = (((blockIdx.x << LB) + tx) << 1) + j_index, j1 = j0 + 1;
	const int OH_OW = OH * OW;
	get_n_oh_ow(j0, n0, oh0, ow0);
	get_n_oh_ow(j1, n1, oh1, ow1);
	bool flagY = (ty & 1);
	const int tn0 = (n1 - n0)*flagY + n0;
	const int tih0 = ((oh1 - oh0)*flagY + oh0) * sh - ph;
	const int tiw0 = ((ow1 - ow0)*flagY + ow0) * sw - pw;
	const int Xoffset0 = ((tn0*IH + tih0)*IW + tiw0)*IC;

	//load 2 elements from W[OC, FH, FW, IC]
	const int W_k = tx >> 1;
	const int Ws_x = (tx >> 1), Ws_y = (ty << 1) + (tx & 1);
	Ws[buf][Ws_x][Ws_y].x = W[toc0 + W_k];
	Ws[buf][Ws_x][Ws_y].y = W[toc1 + W_k];

	//compute 1 address for Xaddr[M, IH, IW, IC]
	int dX_k = ty >> 1, dX_fh, dX_fw; getX_fh_fw(dX_k, dX_fh, dX_fw);
	bool write0 = (tih0 >= -dX_fh) && (tih0 < IH - dX_fh) && (tiw0 >= -dX_fw) && (tiw0 < IW - dX_fw);
	const int IW_IC = IW * IC;
	int xoffset = Xoffset0 + (dX_fh*IW_IC) + dX_k;
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);
	Xaddrs[buf][Xs_y][Xs_x] = (write0 * xoffset) - !write0;
	dXs[buf][Xs_y][Xs_x] = 0;
	__syncthreads();

	//compute area-----------------------------------------
	j0 = j0 * OC + oc0; j1 = j0 + OC;
	float4 dy00 = *(float4*)(&deltaY[j0]);
	float4 dy10 = *(float4*)(&deltaY[j1]);
	for (int ok = 1, OK = GK << 1 >> LB; ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 w = *(float4*)(&Ws[buf][ik][(ty << 1)]);
			int2 xaddr = *(int2*)(&Xaddrs[buf][ik][tx << 1]);

			float dx0 = (xaddr.x != -1) * CrossAdd_SUM4(w, dy00);
			float dx1 = (xaddr.y != -1) * CrossAdd_SUM4(w, dy10);

			atomicAdd_block(&(dXs[buf][ik][(tx << 1)]), dx0);
			atomicAdd_block(&(dXs[buf][ik][(tx << 1) + 1]), dx1);
		}
		__syncthreads();
		int Xaddr = Xaddrs[buf][Xs_y][Xs_x];
		if (Xaddr != -1) atomicAdd(deltaX + Xaddr, dXs[buf][Xs_y][Xs_x]);
		buf ^= 1;

		//load 2 elements from W[OC, FH, FW, IC]
		int W_k = ((ok << LB) + tx) >> 1;
		Ws[buf][Ws_x][Ws_y].x = W[toc0 + W_k];
		Ws[buf][Ws_x][Ws_y].y = W[toc1 + W_k];

		//compute 1 address for Xaddr[M, IH, IW, IC]
		int dX_k = ((ok << LB) + ty) >> 1; getX_fh_fw(dX_k, dX_fh, dX_fw);
		bool write0 = (tih0 >= -dX_fh) && (tih0 < IH - dX_fh) && (tiw0 >= -dX_fw) && (tiw0 < IW - dX_fw);
		int xoffset = Xoffset0 + (dX_fh*IW_IC) + dX_k;
		Xaddrs[buf][Xs_y][Xs_x] = (write0 * xoffset) - !write0;
		dXs[buf][Xs_y][Xs_x] = 0;
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 w = *(float4*)(&Ws[buf][ik][(ty << 1)]);
		int2 xaddr = *(int2*)(&Xaddrs[buf][ik][tx << 1]);

		float dx0 = (xaddr.x != -1) * CrossAdd_SUM4(w, dy00);
		float dx1 = (xaddr.y != -1) * CrossAdd_SUM4(w, dy10);

		atomicAdd_block(&(dXs[buf][ik][(tx << 1)]), dx0);
		atomicAdd_block(&(dXs[buf][ik][(tx << 1) + 1]), dx1);
	}
	__syncthreads();
	int Xaddr = Xaddrs[buf][Xs_y][Xs_x];
	if (Xaddr != -1) atomicAdd(deltaX + Xaddr, dXs[buf][Xs_y][Xs_x]);
}

#endif


//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*2)
//LB = 4: GK % 8 == 0
#ifndef DCONV3D_DX_CROSS_ADD_KERNEL_2_2
#define DCONV3D_DX_CROSS_ADD_KERNEL_2_2

//for(sh, sw) = 2: 
//LB = 4(OC >= 64): Size = 0.234619, Time = 1.004 msec, Performace = 501.833 GFlop/s
//LB = 3(OC >= 32): Size = 0.234619, Time = 1.204 msec, Performace = 418.472 GFlop/s
template<int LB, int STEP>
__global__ void crossAdd_kernel_2_2(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float   Ws[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float  dXs[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ int Xaddrs[2][1 << LB >> 1][(2 << LB) + 2];

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 1) + oc_index;
	const int FW_IC = FW * IC, GK = FH * FW_IC;
	const int toc0 = ((tx & 1) + oc0)*GK;

	//prepared for GM = N * OH * OW
	int j0 = (((blockIdx.x << LB) + tx) << 1) + j_index, j1 = j0 + 1;
	const int OH_OW = OH * OW;
	get_n_oh_ow(j0, n0, oh0, ow0);
	get_n_oh_ow(j1, n1, oh1, ow1);
	bool flagY = (ty & 1);
	const int tn0 = (n1 - n0)*flagY + n0;
	const int tih0 = ((oh1 - oh0)*flagY + oh0) * sh - ph;
	const int tiw0 = ((ow1 - ow0)*flagY + ow0) * sw - pw;
	const int Xoffset0 = ((tn0*IH + tih0)*IW + tiw0)*IC;

	//load 1 element from W[OC, FH, FW, IC]
	const int W_k = tx >> 1;
	const int Ws_x = (tx >> 1), Ws_y = (ty << 1) + (tx & 1);
	Ws[buf][Ws_x][Ws_y] = W[toc0 + W_k];

	//compute 1 address for Xaddr[M, IH, IW, IC]
	int dX_k = ty >> 1, dX_fh, dX_fw; getX_fh_fw(dX_k, dX_fh, dX_fw);
	bool write0 = (tih0 >= -dX_fh) && (tih0 < IH - dX_fh) && (tiw0 >= -dX_fw) && (tiw0 < IW - dX_fw);
	const int IW_IC = IW * IC;
	int xoffset = Xoffset0 + (dX_fh*IW_IC) + dX_k;
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);
	Xaddrs[buf][Xs_y][Xs_x] = (write0 * xoffset) - !write0;
	dXs[buf][Xs_y][Xs_x] = 0;
	__syncthreads();

	//compute area-----------------------------------------
	j0 = j0 * OC + oc0; j1 = j0 + OC;
	float2 dy00 = *(float2*)(&deltaY[j0]);
	float2 dy10 = *(float2*)(&deltaY[j1]);
	for (int ok = 1, OK = GK << 1 >> LB; ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float2 w = *(float2*)(&Ws[buf][ik][(ty << 1)]);
			int2 xaddr = *(int2*)(&Xaddrs[buf][ik][tx << 1]);

			float dx0 = (xaddr.x != -1) * CrossAdd_SUM2(w, dy00);
			float dx1 = (xaddr.y != -1) * CrossAdd_SUM2(w, dy10);

			atomicAdd_block(&(dXs[buf][ik][(tx << 1)]), dx0);
			atomicAdd_block(&(dXs[buf][ik][(tx << 1) + 1]), dx1);
		}
		__syncthreads();
		int Xaddr = Xaddrs[buf][Xs_y][Xs_x];
		if (Xaddr != -1) atomicAdd(deltaX + Xaddr, dXs[buf][Xs_y][Xs_x]);
		buf ^= 1;

		//load 1 element from W[OC, FH, FW, IC]
		int W_k = ((ok << LB) + tx) >> 1;
		Ws[buf][Ws_x][Ws_y] = W[toc0 + W_k];

		//compute 1 address for Xaddr[M, IH, IW, IC]
		int dX_k = ((ok << LB) + ty) >> 1; getX_fh_fw(dX_k, dX_fh, dX_fw);
		bool write0 = (tih0 >= -dX_fh) && (tih0 < IH - dX_fh) && (tiw0 >= -dX_fw) && (tiw0 < IW - dX_fw);
		int xoffset = Xoffset0 + (dX_fh*IW_IC) + dX_k;
		Xaddrs[buf][Xs_y][Xs_x] = (write0 * xoffset) - !write0;
		dXs[buf][Xs_y][Xs_x] = 0;
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float2 w = *(float2*)(&Ws[buf][ik][(ty << 1)]);
		int2 xaddr = *(int2*)(&Xaddrs[buf][ik][tx << 1]);

		float dx0 = (xaddr.x != -1) * CrossAdd_SUM2(w, dy00);
		float dx1 = (xaddr.y != -1) * CrossAdd_SUM2(w, dy10);

		atomicAdd_block(&(dXs[buf][ik][(tx << 1)]), dx0);
		atomicAdd_block(&(dXs[buf][ik][(tx << 1) + 1]), dx1);
	}
	__syncthreads();
	int Xaddr = Xaddrs[buf][Xs_y][Xs_x];
	if (Xaddr != -1) atomicAdd(deltaX + Xaddr, dXs[buf][Xs_y][Xs_x]);
}

#endif


//(Y: BLOCK_SIZE*1, X: BLOCK_SIZE*1)
//LB = 4: GK % 8 == 0
#ifndef DCONV3D_DX_CROSS_ADD_KERNEL_1_1
#define DCONV3D_DX_CROSS_ADD_KERNEL_1_1

//for(sh, sw) = 2: 
//LB = 4(OC >= 64): Size = 0.234619, Time = 1.004 msec, Performace = 501.833 GFlop/s
//LB = 3(OC >= 32): Size = 0.234619, Time = 1.204 msec, Performace = 418.472 GFlop/s
template<int LB, int STEP>
__global__ void crossAdd_kernel_1_1(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float   Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float  dXs[2][1 << LB][(1 << LB) + 1];
	__shared__ int Xaddrs[2][1 << LB][(1 << LB) + 1];

	//prepare for GN = OC
	int oc0 = ((blockIdx.y << LB) + ty) + oc_index;
	const int FW_IC = FW * IC, GK = FH * FW_IC;

	//prepared for GM = N * OH * OW
	int j0  = ((blockIdx.x << LB) + tx) + j_index;;
	const int OH_OW = OH * OW;
	get_n_oh_ow(j0, n0, oh0, ow0);
	const int tih0 = oh0 * sh - ph;
	const int tiw0 = ow0 * sw - pw;
	const int Xoffset0 = ((n0*IH + tih0)*IW + tiw0)*IC;

	//load 1 element from W[OC, FH, FW, IC]
	const int W_k = tx;
	Ws[buf][tx][ty] = W[oc0*GK + W_k];

	//compute 1 address for Xaddr[M, IH, IW, IC]
	int dX_k = ty, dX_fh, dX_fw; getX_fh_fw(dX_k, dX_fh, dX_fw);
	bool write0 = (tih0 >= -dX_fh) && (tih0 < IH - dX_fh) && (tiw0 >= -dX_fw) && (tiw0 < IW - dX_fw);
	const int IW_IC = IW * IC;
	int xoffset = Xoffset0 + (dX_fh*IW_IC) + dX_k;
	Xaddrs[buf][ty][tx] = (write0 * xoffset) - !write0;
	dXs[buf][ty][tx] = 0;
	__syncthreads();

	//compute area-----------------------------------------
	float dy = deltaY[j0 * OC + oc0];
	for (int ok = 1, OK = GK >> LB; ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float w = Ws[buf][ik][ty];
			int xaddr = Xaddrs[buf][ik][tx];
			float dx = (xaddr != -1) * w * dy;
			atomicAdd_block(&(dXs[buf][ik][tx]), dx);
		}
		__syncthreads();
		int Xaddr = Xaddrs[buf][ty][tx];
		if (Xaddr != -1) atomicAdd(deltaX + Xaddr, dXs[buf][ty][tx]);
		buf ^= 1;

		//load 1 element from W[OC, FH, FW, IC]
		int W_k = (ok << LB) + tx;
		Ws[buf][tx][ty] = W[oc0*GK + W_k];

		//compute 1 address for Xaddr[M, IH, IW, IC]
		int dX_k = (ok << LB) + ty;
		getX_fh_fw(dX_k, dX_fh, dX_fw);
		bool write0 = (tih0 >= -dX_fh) && (tih0 < IH - dX_fh) && (tiw0 >= -dX_fw) && (tiw0 < IW - dX_fw);
		int xoffset = Xoffset0 + (dX_fh*IW_IC) + dX_k;
		Xaddrs[buf][ty][tx] = (write0 * xoffset) - !write0;
		dXs[buf][ty][tx] = 0;
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float w = Ws[buf][ik][ty];
		int xaddr = Xaddrs[buf][ik][tx];
		float dx = (xaddr != -1) * w * dy;
		atomicAdd_block(&(dXs[buf][ik][tx]), dx);
	}
	__syncthreads();
	int Xaddr = Xaddrs[buf][ty][tx];
	if (Xaddr != -1) atomicAdd(deltaX + Xaddr, dXs[buf][ty][tx]);
}

#endif

#endif