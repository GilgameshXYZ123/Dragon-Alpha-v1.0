#pragma once

//IMS2: INPUT_MOD_STEP_2
//sh = sw = 2
//(IH, IW) % (sh, sw) == 0
#ifndef DCONV3D_DX_KERNEL_SPLIT_IMS2R_KERNEL_TEXTURE_H
#define DCONV3D_DX_KERNEL_SPLIT_IMS2R_KERNEL_TEXTURE_H

//remode: W[OC, FH, FW, IC] -> CW[sh, sw, OC, CFH, CFW, IC]
#ifndef DCONV3D_DX_KERNEL_SPLIT_IMS2R_TEXTURE_CALL
#define DCONV3D_DX_KERNEL_SPLIT_IMS2R_TEXTURE_CALL

#define UksIms2_88R16_tex(stream, LB, ic_index, j_index, deltaY, OH, OW, CW, CFH, CFW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	UksIms2_kernel_8_8R16_texture<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>3), (GM>>LB>>3), 4), dim3(1<<LB, 1<<LB) >>>\
			(deltaY, OH, OW, CW, CFH, CFW, deltaX, IH, IW, IC, OC, ph, pw, ic_index, j_index)

#define UksIms2_88R16_CW2pow_tex(stream, LB, ic_index, j_index, deltaY, OH, OW, CW, LCFH, LCFW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	UksIms2_kernel_8_8R16_CW2pow_texture<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>3), (GM>>LB>>3), 4), dim3(1<<LB, 1<<LB) >>>\
			(deltaY, OH, OW, CW, LCFH, LCFW, deltaX, IH, IW, IC, OC, ph, pw, ic_index, j_index)

//============================================================================
#define UksIms2_88R8_tex(stream, LB, ic_index, j_index, deltaY, OH, OW, CW, CFH, CFW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	UksIms2_kernel_8_8R8_texture<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>3), (GM>>LB>>3), 4), dim3(1<<LB, 1<<LB) >>>\
			(deltaY, OH, OW, CW, CFH, CFW, deltaX, IH, IW, IC, OC, ph, pw, ic_index, j_index)

#define UksIms2_88R8_CW2pow_tex(stream, LB, ic_index, j_index, deltaY, OH, OW, CW, LCFH, LCFW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	UksIms2_kernel_8_8R8_CW2pow_texture<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>3), (GM>>LB>>3), 4), dim3(1<<LB, 1<<LB) >>>\
			(deltaY, OH, OW, CW, LCFH, LCFW, deltaX, IH, IW, IC, OC, ph, pw, ic_index, j_index)

//============================================================================
#define UksIms2_88R_tex(stream, LB, ic_index, j_index, deltaY, OH, OW, CW, CFH, CFW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	UksIms2_kernel_8_8R_texture<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>3), (GM>>LB>>3), 4), dim3(1<<LB, 1<<LB) >>>\
			(deltaY, OH, OW, CW, CFH, CFW, deltaX, IH, IW, IC, OC, ph, pw, ic_index, j_index)

#define UksIms2_88R_CW2pow_tex(stream, LB, ic_index, j_index, deltaY, OH, OW, CW, LCFH, LCFW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	UksIms2_kernel_8_8R_CW2pow_texture<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>3), (GM>>LB>>3), 4), dim3(1<<LB, 1<<LB) >>>\
			(deltaY, OH, OW, CW, LCFH, LCFW, deltaX, IH, IW, IC, OC, ph, pw, ic_index, j_index)

//=============================================================================
#define UksIms2_84R_tex(stream, LB, ic_index, j_index, deltaY, OH, OW, CW, CFH, CFW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	UksIms2_kernel_8_4R_texture<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>3), (GM>>LB>>2), 4), dim3(1<<LB, 1<<LB) >>>\
			(deltaY, OH, OW, CW, CFH, CFW, deltaX, IH, IW, IC, OC, ph, pw, ic_index, j_index)

#define UksIms2_84R_CW2pow_tex(stream, LB, ic_index, j_index, deltaY, OH, OW, CW, LCFH, LCFW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	UksIms2_kernel_8_4R_CW2pow_texture<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>3), (GM>>LB>>2), 4), dim3(1<<LB, 1<<LB) >>>\
			(deltaY, OH, OW, CW, LCFH, LCFW, deltaX, IH, IW, IC, OC, ph, pw, ic_index, j_index)

//=============================================================================
#define UksIms2_48R_tex(stream, LB, ic_index, j_index, deltaY, OH, OW, CW, CFH, CFW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	UksIms2_kernel_4_8R_texture<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>2), (GM>>LB>>3), 4), dim3(1<<LB, 1<<LB) >>>\
			(deltaY, OH, OW, CW, CFH, CFW, deltaX, IH, IW, IC, OC, ph, pw, ic_index, j_index)

#define UksIms2_48R_CW2pow_tex(stream, LB, ic_index, j_index, deltaY, OH, OW, CW, LCFH, LCFW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	UksIms2_kernel_4_8R_CW2pow_texture<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>2), (GM>>LB>>3), 4), dim3(1<<LB, 1<<LB) >>>\
			(deltaY, OH, OW, CW, LCFH, LCFW, deltaX, IH, IW, IC, OC, ph, pw, ic_index, j_index)

//=============================================================================
#define UksIms2_44R_tex(stream, LB, ic_index, j_index, deltaY, OH, OW, CW, CFH, CFW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	UksIms2_kernel_4_4R_texture<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>2), (GM>>LB>>2), 4), dim3(1<<LB, 1<<LB) >>>\
			(deltaY, OH, OW, CW, CFH, CFW, deltaX, IH, IW, IC, OC, ph, pw, ic_index, j_index)

#define UksIms2_44R_CW2pow_tex(stream, LB, ic_index, j_index, deltaY, OH, OW, CW, LCFH, LCFW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	UksIms2_kernel_4_4R_CW2pow_texture<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>2), (GM>>LB>>2), 4), dim3(1<<LB, 1<<LB) >>>\
			(deltaY, OH, OW, CW, LCFH, LCFW, deltaX, IH, IW, IC, OC, ph, pw, ic_index, j_index)

#endif


//(IH, IW) % 16 == 0 -> (IH_slice, IW_slice) % 8 == 0
//(Y: BLOCK_SIZE*8, X:BLOCK_SIZE*8), K % (BLOCK_SIZE/2) == 0, CFH, CFW is power of 2
#ifndef DCONV3D_DX_KERNEL_SPLIT_IMS2_KERNEL_8_8R16_TEXTURE
#define DCONV3D_DX_KERNEL_SPLIT_IMS2_KERNEL_8_8R16_TEXTURE

//LB = 4: Size = 1.53125, Time = 1.144 msec, Performace = 2874.42 GFlop/s
//LB = 3: Size = 1.53125, Time = 1.018 msec, Performace = 3230.19 GFlop/s
//LB = 4: Size = 1.125, Time = 1.192 msec, Performace = 2026.78 GFlop/s
//LB = 3: Size = 1.125, Time = 1.06  msec, Performace = 2279.17 GFlop/s
template<int LB, int STEP>
__global__ void UksIms2_kernel_8_8R16_texture(
	cudaTextureObject_t deltaY, int OH, int OW,
	const float* __restrict__ CW, int CFH, int CFW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int ph, int pw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];

	//prepare for GK = CFH * CFW * OC
	const int CFH_CFW = CFH * CFW, GK = OC * CFH_CFW;

	//prepare for GZ = sh*sw
	int bz = blockIdx.z;
	CW += bz * CFH_CFW * OC * IC;//CW[y, x]

	ph = ph - (bz >> 1); pw = pw - (bz & 1);//y = bz >> 1, x = bz & 1;
	int ihs = -ph + (-(ph > 0) & ((ph + 1) >> 1 << 1));
	int iws = -pw + (-(pw > 0) & ((pw + 1) >> 1 << 1));

	//prepare for GN = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	CW += ic0 + ((ty >= STEP) << 2);//CW[y, x, 0, 0, 0, tic0]
	deltaX += ic0; //deltaX[0, 0, 0, ic0]

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int IW_slice = IW >> 1, IH_IW_slice = (IH >> 1) * IW_slice;
	Ims2_n_ih_iw(j0, n0, ih0, iw0);  
	int X0 = ((n0*IH + ih0)*IW + iw0)*IC;
	
	int tohs0 = ((ih0 + ph) >> 1) - CFH + 1;
	//ows0 = ((iw0)*2 + ihs), ows4 = ((iw0 + 4)*2 + ihs)
	int tows0 = ((iw0 + ((tx >= STEP) << 3) + pw) >> 1) - CFW + 1;
	int Y1 = ((n0*OH + tohs0)*OW + tows0 + 1) * OC;
	OH -= tohs0;

	//load 4 elem from W[OC, FH, FW, IC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);//W_k ->(oc, fhr, fwr)
	Ws[buf][ty][tx] = *(float4*)(CW + W_k * IC);

	//load 4 elem from deltaY[N, FH, FW, OC]
	int dY_k = tx - ((tx >= STEP) << LB >> 1);
	Ims_oc_fhr_fwr(dY_k, dY_oc, dY_fhr, dY_fwr);
	int yoffset = (dY_fhr * OW + dY_fwr)*OC + dY_oc;
	bool ldy = (tohs0 >= -dY_fhr) && (dY_fhr < OH);//dY_fhr < OH - tohs0
	bool ldy0 = ldy && (tows0     >= -dY_fwr) && (tows0     < OW - dY_fwr);
	bool ldy1 = ldy && (tows0 + 1 >= -dY_fwr) && (tows0 + 1 < OW - dY_fwr);
	bool ldy2 = ldy && (tows0 + 2 >= -dY_fwr) && (tows0 + 2 < OW - dY_fwr);
	bool ldy3 = ldy && (tows0 + 3 >= -dY_fwr) && (tows0 + 3 < OW - dY_fwr);
	zero_float(Ys[buf][tx][ty].x, ldy0, tex1Dfetch<float>(deltaY, Y1 - OC + yoffset));
	zero_float(Ys[buf][tx][ty].y, ldy1, tex1Dfetch<float>(deltaY, Y1 + yoffset));
	zero_float(Ys[buf][tx][ty].z, ldy2, tex1Dfetch<float>(deltaY, Y1 + OC + yoffset));
	zero_float(Ys[buf][tx][ty].w, ldy3, tex1Dfetch<float>(deltaY, Y1 + (OC << 1) + yoffset));
	__syncthreads();

	//compure_area------------------------------------------------------
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

		//load 4 elem from W[OC, FH, FW, IC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;//W_k ->(oc, fhr, fwr)
		Ws[buf][ty][tx] = *(float4*)(CW + W_k * IC);

		//load 4 elem from deltaY[N, FH, FW, OC]
		int dY_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		Ims_oc_fhr_fwr(dY_k, dY_oc, dY_fhr, dY_fwr);
		int yoffset = (dY_fhr * OW + dY_fwr)*OC + dY_oc;
		bool ldy = (tohs0 >= -dY_fhr) && (dY_fhr < OH);;
		bool ldy0 = ldy && (tows0     >= -dY_fwr) && (tows0     < OW - dY_fwr);
		bool ldy1 = ldy && (tows0 + 1 >= -dY_fwr) && (tows0 + 1 < OW - dY_fwr);
		bool ldy2 = ldy && (tows0 + 2 >= -dY_fwr) && (tows0 + 2 < OW - dY_fwr);
		bool ldy3 = ldy && (tows0 + 3 >= -dY_fwr) && (tows0 + 3 < OW - dY_fwr);
		zero_float(Ys[buf][tx][ty].x, ldy0, tex1Dfetch<float>(deltaY, Y1 - OC + yoffset));
		zero_float(Ys[buf][tx][ty].y, ldy1, tex1Dfetch<float>(deltaY, Y1 + yoffset));
		zero_float(Ys[buf][tx][ty].z, ldy2, tex1Dfetch<float>(deltaY, Y1 + OC + yoffset));
		zero_float(Ys[buf][tx][ty].w, ldy3, tex1Dfetch<float>(deltaY, Y1 + (OC << 1) + yoffset));
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

	IC <<= 1;//IC * sw
	int X1 = X0 + IC, X2 = X1 + IC, X3 = X2 + IC;
	int X4 = X3 + IC, X5 = X4 + IC, X6 = X5 + IC, X7 = X6 + IC;

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


//(IH, IW) % 16 == 0 -> (IH_slice, IW_slice) % 8 == 0
//(Y: BLOCK_SIZE*8, X:BLOCK_SIZE*8), K % (BLOCK_SIZE/2) == 0, CFH, CFW is power of 2
#ifndef DCONV3D_DX_KERNEL_SPLIT_IMS2_KERNEL_8_8R16_CW_2POW_TEXTURE
#define DCONV3D_DX_KERNEL_SPLIT_IMS2_KERNEL_8_8R16_CW_2POW_TEXTURE

//LB = 4: Size = 1.53125, Time = 0.85 msec, Performace = 3868.63 GFlop/s(4*4)
//LB = 3: Size = 1.53125, Time = 0.876 msec, Performace = 3753.81 GFlop/s(4*4)
//LB = 4: Size = 2.25, Time = 1.718 msec, Performace = 2812.48 GFlop/s(2*2)
//LB = 3: Size = 2.25, Time = 1.732 msec, Performace = 2789.74 GFlop/s(2*2)
//LB = 4: Size = 1.125, Time = 0.904 msec, Performace = 2672.48 GFlop/s
//LB = 3: Size = 1.125, Time = 0.942 msec, Performace = 2564.67 GFlop/s
template<int LB, int STEP>
__global__ void UksIms2_kernel_8_8R16_CW2pow_texture(
	cudaTextureObject_t deltaY, int OH, int OW,
	const float* __restrict__ CW, int LCFH, int LCFW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int ph, int pw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];

	//prepare for GK = CFH * CFW * OC
	const int LCFH_CFW = LCFH + LCFW, CFH_CFW_m1 = (1 << LCFH_CFW) - 1;
	const int GK = OC << LCFH_CFW;

	//prepare for GZ = sh*sw
	int bz = blockIdx.z;
	CW += (bz << LCFH_CFW) * OC * IC;//CW[y, x]

	ph = ph - (bz >> 1); pw = pw - (bz & 1);//y = bz >> 1, x = bz & 1;
	int ihs = -ph + (-(ph > 0) & ((ph + 1) >> 1 << 1));
	int iws = -pw + (-(pw > 0) & ((pw + 1) >> 1 << 1));

	//prepare for GN = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	CW += ic0 + ((ty >= STEP) << 2);//CW[y, x, 0, 0, 0, tic0]

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int IW_slice = IW >> 1, IH_IW_slice = (IH >> 1) * IW_slice;
	Ims2_n_ih_iw(j0, n0, ih0, iw0);
	int X0 = ((n0*IH + ih0)*IW + iw0)*IC + ic0;

	int tohs0 = ((ih0 + ph) >> 1) - (1 << LCFH) + 1;
	int opw = (1 << LCFW) - 1;//ows0 = ((iw0)*2 + ihs), ows4 = ((iw0 + 4)*2 + ihs)
	int tows0 = ((iw0 + ((tx >= STEP) << 3) + pw) >> 1) - opw;
	int Y1 = ((n0*OH + tohs0)*OW + tows0 + 1) * OC;
	OH -= tohs0;

	//load 4 elem from W[OC, FH, FW, IC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);//W_k ->(oc, fhr, fwr)
	Ws[buf][ty][tx] = *(float4*)(CW + W_k * IC);

	//load 4 elem from deltaY[N, FH, FW, OC]
	int dY_k = tx - ((tx >= STEP) << LB >> 1);
	Ims_oc_fhr_fwr_CW2pow(dY_k, dY_oc, dY_fhr, dY_fwr);
	int yoffset = (dY_fhr * OW + dY_fwr)*OC + dY_oc;
	bool ldy = (tohs0 >= -dY_fhr) && (dY_fhr < OH);//dY_fhr < OH - tohs0
	bool ldy0 = ldy && (tows0     >= -dY_fwr) && (tows0     < OW - dY_fwr);
	bool ldy1 = ldy && (tows0 + 1 >= -dY_fwr) && (tows0 + 1 < OW - dY_fwr);
	bool ldy2 = ldy && (tows0 + 2 >= -dY_fwr) && (tows0 + 2 < OW - dY_fwr);
	bool ldy3 = ldy && (tows0 + 3 >= -dY_fwr) && (tows0 + 3 < OW - dY_fwr);
	zero_float(Ys[buf][tx][ty].x, ldy0, tex1Dfetch<float>(deltaY, Y1 - OC + yoffset));
	zero_float(Ys[buf][tx][ty].y, ldy1, tex1Dfetch<float>(deltaY, Y1 + yoffset));
	zero_float(Ys[buf][tx][ty].z, ldy2, tex1Dfetch<float>(deltaY, Y1 + OC + yoffset));
	zero_float(Ys[buf][tx][ty].w, ldy3, tex1Dfetch<float>(deltaY, Y1 + (OC << 1) + yoffset));
	__syncthreads();

	//compure_area------------------------------------------------------
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
			float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];
			float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];

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

		//load 4 elem from W[OC, FH, FW, IC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;//W_k ->(oc, fhr, fwr)
		Ws[buf][ty][tx] = *(float4*)(CW + W_k * IC);

		//load 4 elem from deltaY[N, FH, FW, OC]
		int dY_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		Ims_oc_fhr_fwr_CW2pow(dY_k, dY_oc, dY_fhr, dY_fwr);
		int yoffset = (dY_fhr * OW + dY_fwr)*OC + dY_oc;
		bool ldy = (tohs0 >= -dY_fhr) && (dY_fhr < OH);
		bool ldy0 = ldy && (tows0     >= -dY_fwr) && (tows0     < OW - dY_fwr);
		bool ldy1 = ldy && (tows0 + 1 >= -dY_fwr) && (tows0 + 1 < OW - dY_fwr);
		bool ldy2 = ldy && (tows0 + 2 >= -dY_fwr) && (tows0 + 2 < OW - dY_fwr);
		bool ldy3 = ldy && (tows0 + 3 >= -dY_fwr) && (tows0 + 3 < OW - dY_fwr);
		zero_float(Ys[buf][tx][ty].x, ldy0, tex1Dfetch<float>(deltaY, Y1 - OC + yoffset));
		zero_float(Ys[buf][tx][ty].y, ldy1, tex1Dfetch<float>(deltaY, Y1 + yoffset));
		zero_float(Ys[buf][tx][ty].z, ldy2, tex1Dfetch<float>(deltaY, Y1 + OC + yoffset));
		zero_float(Ys[buf][tx][ty].w, ldy3, tex1Dfetch<float>(deltaY, Y1 + (OC << 1) + yoffset));
		__syncthreads();
	}
#pragma once
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];
		float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];

		simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
		simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
		simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
		simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
		simdMM4(v8, y1.x, w0); simdMM4(v9, y1.x, w1);
		simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
		simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
		simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
	}

	IC <<= 1;//IC * sw
	int X1 = X0 + IC, X2 = X1 + IC, X3 = X2 + IC;
	int X4 = X3 + IC, X5 = X4 + IC, X6 = X5 + IC, X7 = X6 + IC;

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


//(IH, IW) % 8 == 0 -> (IH_slice, IW_slice) % 4 == 0
//(Y: BLOCK_SIZE*8, X:BLOCK_SIZE*8), K % (BLOCK_SIZE/2) == 0, CFH, CFW is power of 2
#ifndef DCONV3D_DX_KERNEL_SPLIT_IMS2_KERNEL_8_8R8_TEXTURE
#define DCONV3D_DX_KERNEL_SPLIT_IMS2_KERNEL_8_8R8_TEXTURE

//LB = 4: Size = 1.53125, Time = 1.146 msec, Performace = 2869.4 GFlop/s
//LB = 3: Size = 1.53125, Time = 1.018 msec, Performace = 3230.19 GFlop/s
//LB = 4: Size = 1.125, Time = 1.236 msec, Performace = 1954.63 GFlop/s
//LB = 3: Size = 1.125, Time = 1.068 msec, Performace = 2262.1 GFlop/s
template<int LB, int STEP>
__global__ void UksIms2_kernel_8_8R8_texture(
	cudaTextureObject_t deltaY, int OH, int OW,
	const float* __restrict__ CW, int CFH, int CFW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int ph, int pw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];

	//prepare for GK = CFH * CFW * OC
	const int CFH_CFW = CFH * CFW, GK = OC * CFH_CFW;

	//prepare for GZ = sh*sw
	int bz = blockIdx.z;
	CW += bz * CFH_CFW * OC * IC;//CW[y, x]

	ph = ph - (bz >> 1); pw = pw - (bz & 1);//y = bz >> 1, x = bz & 1;
	int ihs = -ph + (-(ph > 0) & ((ph + 1) >> 1 << 1));
	int iws = -pw + (-(pw > 0) & ((pw + 1) >> 1 << 1));

	//prepare for GN = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	CW += ic0 + ((ty >= STEP) << 2);//CW[y, x, 0, 0, 0, tic0]
	deltaX += ic0; //deltaX[0, 0, 0, ic0]

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index, j4 = j0 + 4;
	int IW_slice = IW >> 1, IH_IW_slice = (IH >> 1) * IW_slice;
	Ims2_n_ih_iw(j0, n0, ih0, iw0);
	Ims2_n_ih_iw(j4, n4, ih4, iw4);
	int X0 = ((n0*IH + ih0)*IW + iw0)*IC;
	int X4 = ((n0*IH + ih4)*IW + iw4)*IC;

	bool flagX = (tx >= STEP);
	int tohs0 = ((IF_int(flagX, ih4, ih0) + ph) >> 1) - CFH + 1;
	int tows0 = ((IF_int(flagX, iw4, iw0) + pw) >> 1) - CFW + 1;
	int Y1 = ((n0*OH + tohs0)*OW + tows0 + 1) * OC;
	OH -= tohs0;

	//load 4 elem from W[OC, FH, FW, IC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);//W_k ->(oc, fhr, fwr)
	Ws[buf][ty][tx] = *(float4*)(CW + W_k * IC);

	//load 4 elem from deltaY[N, FH, FW, OC]
	int dY_k = tx - ((tx >= STEP) << LB >> 1);
	Ims_oc_fhr_fwr(dY_k, dY_oc, dY_fhr, dY_fwr);
	int yoffset = (dY_fhr * OW + dY_fwr)*OC + dY_oc;
	bool ldy = (tohs0 >= -dY_fhr) && (dY_fhr < OH);//dY_fhr < OH - tohs0
	bool ldy0 = ldy && (tows0 >= -dY_fwr) && (tows0 < OW - dY_fwr);
	bool ldy1 = ldy && (tows0 + 1 >= -dY_fwr) && (tows0 + 1 < OW - dY_fwr);
	bool ldy2 = ldy && (tows0 + 2 >= -dY_fwr) && (tows0 + 2 < OW - dY_fwr);
	bool ldy3 = ldy && (tows0 + 3 >= -dY_fwr) && (tows0 + 3 < OW - dY_fwr);
	zero_float(Ys[buf][tx][ty].x, ldy0, tex1Dfetch<float>(deltaY, Y1 - OC + yoffset));
	zero_float(Ys[buf][tx][ty].y, ldy1, tex1Dfetch<float>(deltaY, Y1 + yoffset));
	zero_float(Ys[buf][tx][ty].z, ldy2, tex1Dfetch<float>(deltaY, Y1 + OC + yoffset));
	zero_float(Ys[buf][tx][ty].w, ldy3, tex1Dfetch<float>(deltaY, Y1 + (OC << 1) + yoffset));
	__syncthreads();

	//compure_area------------------------------------------------------
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

		//load 4 elem from W[OC, FH, FW, IC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;//W_k ->(oc, fhr, fwr)
		Ws[buf][ty][tx] = *(float4*)(CW + W_k * IC);

		//load 4 elem from deltaY[N, FH, FW, OC]
		int dY_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		Ims_oc_fhr_fwr(dY_k, dY_oc, dY_fhr, dY_fwr);
		int yoffset = (dY_fhr * OW + dY_fwr)*OC + dY_oc;
		bool ldy = (tohs0 >= -dY_fhr) && (dY_fhr < OH);;
		bool ldy0 = ldy && (tows0 >= -dY_fwr) && (tows0 < OW - dY_fwr);
		bool ldy1 = ldy && (tows0 + 1 >= -dY_fwr) && (tows0 + 1 < OW - dY_fwr);
		bool ldy2 = ldy && (tows0 + 2 >= -dY_fwr) && (tows0 + 2 < OW - dY_fwr);
		bool ldy3 = ldy && (tows0 + 3 >= -dY_fwr) && (tows0 + 3 < OW - dY_fwr);
		zero_float(Ys[buf][tx][ty].x, ldy0, tex1Dfetch<float>(deltaY, Y1 - OC + yoffset));
		zero_float(Ys[buf][tx][ty].y, ldy1, tex1Dfetch<float>(deltaY, Y1 + yoffset));
		zero_float(Ys[buf][tx][ty].z, ldy2, tex1Dfetch<float>(deltaY, Y1 + OC + yoffset));
		zero_float(Ys[buf][tx][ty].w, ldy3, tex1Dfetch<float>(deltaY, Y1 + (OC << 1) + yoffset));
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

	IC <<= 1;//IC * sw
	int X1 = X0 + IC, X2 = X1 + IC, X3 = X2 + IC;
	int X5 = X4 + IC, X6 = X5 + IC, X7 = X6 + IC;

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


//(IH, IW) % 8 == 0 -> (IH_slice, IW_slice) % 4 == 0
//(Y: BLOCK_SIZE*8, X:BLOCK_SIZE*8), K % (BLOCK_SIZE/2) == 0, CFH, CFW is power of 2
#ifndef DCONV3D_DX_KERNEL_SPLIT_IMS2_KERNEL_8_8R8_CW_2POW_TEXTURE
#define DCONV3D_DX_KERNEL_SPLIT_IMS2_KERNEL_8_8R8_CW_2POW_TEXTURE

//LB = 4: Size = 1.53125, Time = 0.874 msec, Performace = 3762.4 GFlop/s
//LB = 3: Size = 1.53125, Time = 0.87 msec, Performace = 3779.69 GFlop/s
//LB = 4: Size = 1.125, Time = 0.932 msec, Performace = 2592.19 GFlop/s
//LB = 3: Size = 1.125, Time = 0.94  msec, Performace = 2570.13 GFlop/s
template<int LB, int STEP>
__global__ void UksIms2_kernel_8_8R8_CW2pow_texture(
	cudaTextureObject_t deltaY, int OH, int OW,
	const float* __restrict__ CW, int LCFH, int LCFW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int ph, int pw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];

	//prepare for GK = CFH * CFW * OC
	const int LCFH_CFW = LCFH + LCFW, CFH_CFW_m1 = (1 << LCFH_CFW) - 1;
	const int GK = OC << LCFH_CFW;

	//prepare for GZ = sh*sw
	int bz = blockIdx.z;
	CW += (bz << LCFH_CFW) * OC * IC;//CW[y, x]

	ph = ph - (bz >> 1); pw = pw - (bz & 1);//y = bz >> 1, x = bz & 1;
	int ihs = -ph + (-(ph > 0) & ((ph + 1) >> 1 << 1));
	int iws = -pw + (-(pw > 0) & ((pw + 1) >> 1 << 1));

	//prepare for GN = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	CW += ic0 + ((ty >= STEP) << 2);//CW[y, x, 0, 0, 0, tic0]
	deltaX += ic0; //deltaX[0, 0, 0, ic0]

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index, j4 = j0 + 4;
	int IW_slice = IW >> 1, IH_IW_slice = (IH >> 1) * IW_slice;
	Ims2_n_ih_iw(j0, n0, ih0, iw0);
	Ims2_n_ih_iw(j4, n4, ih4, iw4);
	int X0 = ((n0*IH + ih0)*IW + iw0)*IC;
	int X4 = ((n0*IH + ih4)*IW + iw4)*IC;

	bool flagX = (tx >= STEP);
	int tohs0 = ((IF_int(flagX, ih4, ih0) + ph) >> 1) - (1 << LCFH) + 1;
	int opw = (1 << LCFW) - 1;
	int tows0 = ((IF_int(flagX, iw4, iw0) + pw) >> 1) - opw;
	int Y1 = ((n0*OH + tohs0)*OW + tows0 + 1) * OC;
	OH -= tohs0;

	//load 4 elem from W[OC, FH, FW, IC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);//W_k ->(oc, fhr, fwr)
	Ws[buf][ty][tx] = *(float4*)(CW + W_k * IC);

	//load 4 elem from deltaY[N, FH, FW, OC]
	int dY_k = tx - ((tx >= STEP) << LB >> 1);
	Ims_oc_fhr_fwr_CW2pow(dY_k, dY_oc, dY_fhr, dY_fwr);
	int yoffset = (dY_fhr * OW + dY_fwr)*OC + dY_oc;
	bool ldy = (tohs0 >= -dY_fhr) && (dY_fhr < OH);//dY_fhr < OH - tohs0
	bool ldy0 = ldy && (tows0     >= -dY_fwr) && (tows0     < OW - dY_fwr);
	bool ldy1 = ldy && (tows0 + 1 >= -dY_fwr) && (tows0 + 1 < OW - dY_fwr);
	bool ldy2 = ldy && (tows0 + 2 >= -dY_fwr) && (tows0 + 2 < OW - dY_fwr);
	bool ldy3 = ldy && (tows0 + 3 >= -dY_fwr) && (tows0 + 3 < OW - dY_fwr);
	zero_float(Ys[buf][tx][ty].x, ldy0, tex1Dfetch<float>(deltaY, Y1 - OC + yoffset));
	zero_float(Ys[buf][tx][ty].y, ldy1, tex1Dfetch<float>(deltaY, Y1 + yoffset));
	zero_float(Ys[buf][tx][ty].z, ldy2, tex1Dfetch<float>(deltaY, Y1 + OC + yoffset));
	zero_float(Ys[buf][tx][ty].w, ldy3, tex1Dfetch<float>(deltaY, Y1 + (OC << 1) + yoffset));
	__syncthreads();

	//compure_area------------------------------------------------------
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

		//load 4 elem from W[OC, FH, FW, IC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;//W_k ->(oc, fhr, fwr)
		Ws[buf][ty][tx] = *(float4*)(CW + W_k * IC);

		//load 4 elem from deltaY[N, FH, FW, OC]
		int dY_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		Ims_oc_fhr_fwr_CW2pow(dY_k, dY_oc, dY_fhr, dY_fwr);
		int yoffset = (dY_fhr * OW + dY_fwr)*OC + dY_oc;
		bool ldy = (tohs0 >= -dY_fhr) && (dY_fhr < OH);;
		bool ldy0 = ldy && (tows0     >= -dY_fwr) && (tows0     < OW - dY_fwr);
		bool ldy1 = ldy && (tows0 + 1 >= -dY_fwr) && (tows0 + 1 < OW - dY_fwr);
		bool ldy2 = ldy && (tows0 + 2 >= -dY_fwr) && (tows0 + 2 < OW - dY_fwr);
		bool ldy3 = ldy && (tows0 + 3 >= -dY_fwr) && (tows0 + 3 < OW - dY_fwr);
		zero_float(Ys[buf][tx][ty].x, ldy0, tex1Dfetch<float>(deltaY, Y1 - OC + yoffset));
		zero_float(Ys[buf][tx][ty].y, ldy1, tex1Dfetch<float>(deltaY, Y1 + yoffset));
		zero_float(Ys[buf][tx][ty].z, ldy2, tex1Dfetch<float>(deltaY, Y1 + OC + yoffset));
		zero_float(Ys[buf][tx][ty].w, ldy3, tex1Dfetch<float>(deltaY, Y1 + (OC << 1) + yoffset));
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

	IC <<= 1;//IC * sw
	int X1 = X0 + IC, X2 = X1 + IC, X3 = X2 + IC;
	int X5 = X4 + IC, X6 = X5 + IC, X7 = X6 + IC;

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


//(Y: BLOCK_SIZE*8, X:BLOCK_SIZE*8), K % (BLOCK_SIZE/2) == 0
#ifndef DCONV3D_DX_KERNEL_SPLIT_IMS2_KERNEL_8_8R_TEXTURE
#define DCONV3D_DX_KERNEL_SPLIT_IMS2_KERNEL_8_8R_TEXTURE

//LB = 3: Size = 1.53125, Time = 1.038 msec, Performace = 3167.95 GFlop/s
//LB = 4: Size = 1.53125, Time = 1.126 msec, Performace = 2920.37 GFlop/s
//LB = 4: Size = 1.125, Time = 1.226 msec, Performace = 1970.57 GFlop/s
//LB = 3: Size = 1.125, Time = 1.102 msec, Performace = 2192.3  GFlop/s
template<int LB, int STEP>
__global__ void UksIms2_kernel_8_8R_texture(
	cudaTextureObject_t deltaY, int OH, int OW,
	const float* __restrict__ CW, int CFH, int CFW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int ph, int pw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];

	//prepare for GZ = sh*sw
	int bz = blockIdx.z;
	CW += bz * CFH * CFW * OC * IC;//CW[y, x]

	ph = ph - (bz >> 1); pw = pw - (bz & 1);//if(ihs < 0): ihs += (ph + 1)/sh*sh
	int ihs = -ph + (-(ph > 0) & ((ph + 1) >> 1 << 1));
	int iws = -pw + (-(pw > 0) & ((pw + 1) >> 1 << 1));

	//prepare for GN = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	CW += ic0 + ((ty >= STEP) << 2);//CW[y, x, 0, 0, 0, tic0]
	deltaX += ic0; //deltaX[0, 0, 0, ic0]

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int j1 = j0 + 1, j2 = j0 + 2, j3 = j0 + 3;
	int j4 = j0 + 4, j5 = j0 + 5, j6 = j0 + 6, j7 = j0 + 7;
	int IW_slice = IW >> 1, IH_IW_slice = (IH >> 1) * IW_slice;
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
	int tohs0 = ((IF_int(flagX, ih4, ih0) + ph) >> 1) - oph;
	int tohs1 = ((IF_int(flagX, ih5, ih1) + ph) >> 1) - oph;
	int tohs2 = ((IF_int(flagX, ih6, ih2) + ph) >> 1) - oph;
	int tohs3 = ((IF_int(flagX, ih7, ih3) + ph) >> 1) - oph;
	int tows0 = ((IF_int(flagX, iw4, iw0) + pw) >> 1) - opw;
	int tows1 = ((IF_int(flagX, iw5, iw1) + pw) >> 1) - opw;
	int tows2 = ((IF_int(flagX, iw6, iw2) + pw) >> 1) - opw;
	int tows3 = ((IF_int(flagX, iw7, iw3) + pw) >> 1) - opw;
	int Yoffset0 = ((IF_int(flagX, n4, n0)*OH + tohs0)*OW + tows0)*OC;
	int Yoffset1 = ((IF_int(flagX, n5, n1)*OH + tohs1)*OW + tows1)*OC;
	int Yoffset2 = ((IF_int(flagX, n6, n2)*OH + tohs2)*OW + tows2)*OC;
	int Yoffset3 = ((IF_int(flagX, n7, n3)*OH + tohs3)*OW + tows3)*OC;

	//prepare for GK = CFH * CFW * OC
	const int CFH_CFW = CFW * CFW, GK = OC * CFH_CFW;

	//load 4 elem from W[OC, FH, FW, IC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);//W_k ->(oc, fhr, fwr)
	Ws[buf][ty][tx] = *(float4*)(CW + W_k * IC);

	//load 4 elem from deltaY[N, FH, FW, OC]
	int dY_k = tx - ((tx >= STEP) << LB >> 1);
	Ims_oc_fhr_fwr(dY_k, dY_oc, dY_fhr, dY_fwr);
	int yoffset = (dY_fhr * OW + dY_fwr)*OC + dY_oc;
	zero_float(Ys[buf][tx][ty].x, Ims_ldy(tohs0, tows0), tex1Dfetch<float>(deltaY, Yoffset0 + yoffset));
	zero_float(Ys[buf][tx][ty].y, Ims_ldy(tohs1, tows1), tex1Dfetch<float>(deltaY, Yoffset1 + yoffset));
	zero_float(Ys[buf][tx][ty].z, Ims_ldy(tohs2, tows2), tex1Dfetch<float>(deltaY, Yoffset2 + yoffset));
	zero_float(Ys[buf][tx][ty].w, Ims_ldy(tohs3, tows3), tex1Dfetch<float>(deltaY, Yoffset3 + yoffset));
	__syncthreads();

	//compure_area------------------------------------------------------
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
			float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];
			float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];

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
		
		//load 4 elem from W[OC, FH, FW, IC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;//W_k ->(oc, fhr, fwr)
		Ws[buf][ty][tx] = *(float4*)(CW + W_k * IC);

		//load 4 elem from deltaY[N, FH, FW, OC]
		int dY_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		Ims_oc_fhr_fwr(dY_k, dY_oc, dY_fhr, dY_fwr);
		int yoffset = (dY_fhr * OW + dY_fwr)*OC + dY_oc;
		zero_float(Ys[buf][tx][ty].x, Ims_ldy(tohs0, tows0), tex1Dfetch<float>(deltaY, Yoffset0 + yoffset));
		zero_float(Ys[buf][tx][ty].y, Ims_ldy(tohs1, tows1), tex1Dfetch<float>(deltaY, Yoffset1 + yoffset));
		zero_float(Ys[buf][tx][ty].z, Ims_ldy(tohs2, tows2), tex1Dfetch<float>(deltaY, Yoffset2 + yoffset));
		zero_float(Ys[buf][tx][ty].w, Ims_ldy(tohs3, tows3), tex1Dfetch<float>(deltaY, Yoffset3 + yoffset));
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


//(Y: BLOCK_SIZE*8, X:BLOCK_SIZE*8), K % (BLOCK_SIZE/2) == 0, CFH, CFW is power of 2
#ifndef DCONV3D_DX_KERNEL_SPLIT_IMS2_KERNEL_8_8R_CW_2POW_TEXTURE
#define DCONV3D_DX_KERNEL_SPLIT_IMS2_KERNEL_8_8R_CW_2POW_TEXTURE

//LB = 4: Size = 1.53125, Time = 1.018 msec, Performace = 3230.19 GFlop/s
//LB = 3: Size = 1.53125, Time = 0.896 msec, Performace = 3670.02 GFlop/s
//LB = 4: Size = 1.125, Time = 1.108 msec, Performace = 2180.43 GFlop/s
//LB = 3: Size = 1.125, Time = 0.956 msec, Performace = 2527.11 GFlop/s
template<int LB, int STEP>
__global__ void UksIms2_kernel_8_8R_CW2pow_texture(
	cudaTextureObject_t deltaY, int OH, int OW,
	const float* __restrict__ CW, int LCFH, int LCFW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int ph, int pw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];

	//prepare for GZ = sh*sw
	int bz = blockIdx.z;
	CW += (bz << LCFH << LCFW) * OC * IC;//CW[y, x]

	ph = ph - (bz >> 1); pw = pw - (bz & 1);//if(ihs < 0): ihs += (ph + 1)/sh*sh
	int ihs = -ph + (-(ph > 0) & ((ph + 1) >> 1 << 1));
	int iws = -pw + (-(pw > 0) & ((pw + 1) >> 1 << 1));

	//prepare for GN = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	CW += ic0 + ((ty >= STEP) << 2);//CW[y, x, 0, 0, 0, tic0]
	deltaX += ic0; //deltaX[0, 0, 0, ic0]

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int j1 = j0 + 1, j2 = j0 + 2, j3 = j0 + 3;
	int j4 = j0 + 4, j5 = j0 + 5, j6 = j0 + 6, j7 = j0 + 7;
	int IW_slice = IW >> 1, IH_IW_slice = (IH >> 1) * IW_slice;
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
	int oph = (1 << LCFH) - 1, opw = (1 << LCFW) - 1;
	int tohs0 = ((IF_int(flagX, ih4, ih0) + ph) >> 1) - oph;
	int tohs1 = ((IF_int(flagX, ih5, ih1) + ph) >> 1) - oph;
	int tohs2 = ((IF_int(flagX, ih6, ih2) + ph) >> 1) - oph;
	int tohs3 = ((IF_int(flagX, ih7, ih3) + ph) >> 1) - oph;
	int tows0 = ((IF_int(flagX, iw4, iw0) + pw) >> 1) - opw;
	int tows1 = ((IF_int(flagX, iw5, iw1) + pw) >> 1) - opw;
	int tows2 = ((IF_int(flagX, iw6, iw2) + pw) >> 1) - opw;
	int tows3 = ((IF_int(flagX, iw7, iw3) + pw) >> 1) - opw;
	int Yoffset0 = ((IF_int(flagX, n4, n0)*OH + tohs0)*OW + tows0)*OC;
	int Yoffset1 = ((IF_int(flagX, n5, n1)*OH + tohs1)*OW + tows1)*OC;
	int Yoffset2 = ((IF_int(flagX, n6, n2)*OH + tohs2)*OW + tows2)*OC;
	int Yoffset3 = ((IF_int(flagX, n7, n3)*OH + tohs3)*OW + tows3)*OC;

	//prepare for GK = CFH * CFW * OC
	const int LCFH_CFW = LCFW + LCFW, CFH_CFW_m1 = (1 << LCFH_CFW) - 1;
	const int GK = OC << LCFH_CFW;
	
	//load 4 elem from deltaY[N, FH, FW, OC]
	int dY_k = tx - ((tx >= STEP) << LB >> 1);
	Ims_oc_fhr_fwr_CW2pow(dY_k, dY_oc, dY_fhr, dY_fwr);
	int yoffset = (dY_fhr * OW + dY_fwr)*OC + dY_oc;
	zero_float(Ys[buf][tx][ty].x, Ims_ldy(tohs0, tows0), tex1Dfetch<float>(deltaY, Yoffset0 + yoffset));
	zero_float(Ys[buf][tx][ty].y, Ims_ldy(tohs1, tows1), tex1Dfetch<float>(deltaY, Yoffset1 + yoffset));
	zero_float(Ys[buf][tx][ty].z, Ims_ldy(tohs2, tows2), tex1Dfetch<float>(deltaY, Yoffset2 + yoffset));
	zero_float(Ys[buf][tx][ty].w, Ims_ldy(tohs3, tows3), tex1Dfetch<float>(deltaY, Yoffset3 + yoffset));

	//load 4 elem from W[OC, FH, FW, IC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);//W_k ->(oc, fhr, fwr)
	Ws[buf][ty][tx] = *(float4*)(CW + W_k * IC);
	__syncthreads();

	//compure_area------------------------------------------------------
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
		Ims_oc_fhr_fwr_CW2pow(dY_k, dY_oc, dY_fhr, dY_fwr);
		int yoffset = (dY_fhr * OW + dY_fwr)*OC + dY_oc;
		zero_float(Ys[buf][tx][ty].x, Ims_ldy(tohs0, tows0), tex1Dfetch<float>(deltaY, Yoffset0 + yoffset));
		zero_float(Ys[buf][tx][ty].y, Ims_ldy(tohs1, tows1), tex1Dfetch<float>(deltaY, Yoffset1 + yoffset));
		zero_float(Ys[buf][tx][ty].z, Ims_ldy(tohs2, tows2), tex1Dfetch<float>(deltaY, Yoffset2 + yoffset));
		zero_float(Ys[buf][tx][ty].w, Ims_ldy(tohs3, tows3), tex1Dfetch<float>(deltaY, Yoffset3 + yoffset));

		//load 4 elem from W[OC, FH, FW, IC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;//W_k ->(oc, fhr, fwr)
		Ws[buf][ty][tx] = *(float4*)(CW + W_k * IC);
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


//(Y: BLOCK_SIZE*8, X:BLOCK_SIZE*4), K % (BLOCK_SIZE/2) == 0
#ifndef DCONV3D_DX_KERNEL_SPLIT_IMS2_KERNEL_8_4R_TEXTURE
#define DCONV3D_DX_KERNEL_SPLIT_IMS2_KERNEL_8_4R_TEXTURE

//LB = 3: Size = 1.53125, Time = 1.234 msec, Performace = 2664.78 GFlop/s
//LB = 4: Size = 1.125, Time = 1.096 msec, Performace = 2204.31 GFlop/s
//LB = 3: Size = 1.125, Time = 1.274 msec, Performace = 1896.33 GFlop/s
template<int LB, int STEP>
__global__ void UksIms2_kernel_8_4R_texture(
	cudaTextureObject_t deltaY, int OH, int OW,
	const float* __restrict__ CW, int CFH, int CFW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int ph, int pw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];//follow k88
	__shared__ float2 Ys[2][1 << LB >> 1][(2 << LB) + 2];//follow k44

	//prepare for GZ = sh*sw
	int bz = blockIdx.z;
	CW += bz * CFH * CFW * OC * IC;//CW[y, x]

	ph = ph - (bz >> 1); pw = pw - (bz & 1);//if(ihs < 0): ihs += (ph + 1)/sh*sh
	int ihs = -ph + (-(ph > 0) & ((ph + 1) >> 1 << 1));
	int iws = -pw + (-(pw > 0) & ((pw + 1) >> 1 << 1));

	//prepare for GN = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	CW += ic0 + ((ty >= STEP) << 2);//W[0, 0, 0, tic0]
	deltaX += ic0; //deltaX[0, 0, 0, ic0]

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = (((blockIdx.y << LB) + ty) << 2) + j_index;
	int j1 = j0 + 1, j2 = j0 + 2, j3 = j0 + 3;
	int IW_slice = IW >> 1, IH_IW_slice = (IH >> 1) * IW_slice;
	Ims2_n_ih_iw(j0, n0, ih0, iw0);
	Ims2_n_ih_iw(j1, n1, ih1, iw1);
	Ims2_n_ih_iw(j2, n2, ih2, iw2);
	Ims2_n_ih_iw(j3, n3, ih3, iw3);
	int Xoffset0 = ((n0*IH + ih0)*IW + iw0) * IC;
	int Xoffset1 = ((n1*IH + ih1)*IW + iw1) * IC;
	int Xoffset2 = ((n2*IH + ih2)*IW + iw2) * IC;
	int Xoffset3 = ((n3*IH + ih3)*IW + iw3) * IC;

	bool flagX = (tx & 1);
	const int oph = CFH - 1, opw = CFW - 1;
	int tohs0 = ((IF_int(flagX, ih2, ih0) + ph) >> 1) - oph;
	int tohs1 = ((IF_int(flagX, ih3, ih1) + ph) >> 1) - oph;
	int tows0 = ((IF_int(flagX, iw2, iw0) + pw) >> 1) - opw;
	int tows1 = ((IF_int(flagX, iw3, iw1) + pw) >> 1) - opw;
	int Yoffset0 = ((IF_int(flagX, n2, n0)*OH + tohs0)*OW + tows0) * OC;
	int Yoffset1 = ((IF_int(flagX, n3, n1)*OH + tohs1)*OW + tows1) * OC;

	//prepare for GK = CFH * CFW * OC
	const int CFH_CFW = CFW * CFW, GK = OC * CFH * CFW;

	//load 2 elem from deltaY[N, FH, FW, OC]
	int dY_k = tx >> 1;
	Ims_oc_fhr_fwr(dY_k, dY_oc, dY_fhr, dY_fwr);
	int yoffset = ((dY_fhr * OW) + dY_fwr)*OC + dY_oc;
	const int Ys_x = (tx >> 1), Ys_y = (ty << 1) + (tx & 1);
	zero_float(Ys[buf][Ys_x][Ys_y].x, Ims_ldy(tohs0, tows0), tex1Dfetch<float>(deltaY, Yoffset0 + yoffset));
	zero_float(Ys[buf][Ys_x][Ys_y].y, Ims_ldy(tohs1, tows1), tex1Dfetch<float>(deltaY, Yoffset1 + yoffset));

	//load 4 elem from W[OC, FH, FW, IC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);
	Ws[buf][ty][tx] = *(float4*)(CW + W_k * IC);
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

		//load 2 elem from deltaY[N, FH, FW, OC]
		int dY_k = ((ok << LB) + tx) >> 1;
		Ims_oc_fhr_fwr(dY_k, dY_oc, dY_fhr, dY_fwr);
		int yoffset = ((dY_fhr * OW) + dY_fwr)*OC + dY_oc;
		zero_float(Ys[buf][Ys_x][Ys_y].x, Ims_ldy(tohs0, tows0), tex1Dfetch<float>(deltaY, Yoffset0 + yoffset));
		zero_float(Ys[buf][Ys_x][Ys_y].y, Ims_ldy(tohs1, tows1), tex1Dfetch<float>(deltaY, Yoffset1 + yoffset));

		//load 4 elem from W[OC, FH, FW, IC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		Ws[buf][ty][tx] = *(float4*)(CW + W_k * IC);
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


//(Y: BLOCK_SIZE*8, X:BLOCK_SIZE*4), K % (BLOCK_SIZE/2) == 0, CFH, CFW is power of 2
#ifndef DCONV3D_DX_KERNEL_SPLIT_IMS2_KERNEL_8_4R_CW_2POW_TEXTURE
#define DCONV3D_DX_KERNEL_SPLIT_IMS2_KERNEL_8_4R_CW_2POW_TEXTURE

//LB = 3: Size = 1.53125, Time = 1.028 msec, Performace = 3198.77 GFlop/s
//LB = 4: Size = 1.125, Time = 1.014 msec, Performace = 2382.56 GFlop/s
//LB = 3: Size = 1.125, Time = 1.102 msec, Performace = 2192.3 GFlop/s
template<int LB, int STEP>
__global__ void UksIms2_kernel_8_4R_CW2pow_texture(
	cudaTextureObject_t deltaY, int OH, int OW,
	const float* __restrict__ CW, int LCFH, int LCFW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int ph, int pw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];//follow k88
	__shared__ float2 Ys[2][1 << LB >> 1][(2 << LB) + 2];//follow k44

	//prepare for GZ = sh*sw
	int bz = blockIdx.z;
	CW += (bz << LCFH << LCFW) * OC * IC;//CW[y, x]

	ph = ph - (bz >> 1); pw = pw - (bz & 1);//if(ihs < 0): ihs += (ph + 1)/sh*sh
	int ihs = -ph + (-(ph > 0) & ((ph + 1) >> 1 << 1));
	int iws = -pw + (-(pw > 0) & ((pw + 1) >> 1 << 1));

	//prepare for GN = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	CW += ic0 + ((ty >= STEP) << 2);//W[0, 0, 0, tic0]
	deltaX += ic0; //deltaX[0, 0, 0, ic0]

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = (((blockIdx.y << LB) + ty) << 2) + j_index;
	int j1 = j0 + 1, j2 = j0 + 2, j3 = j0 + 3;
	int IW_slice = IW >> 1, IH_IW_slice = (IH >> 1) * IW_slice;
	Ims2_n_ih_iw(j0, n0, ih0, iw0);
	Ims2_n_ih_iw(j1, n1, ih1, iw1);
	Ims2_n_ih_iw(j2, n2, ih2, iw2);
	Ims2_n_ih_iw(j3, n3, ih3, iw3);
	int Xoffset0 = ((n0*IH + ih0)*IW + iw0) * IC;
	int Xoffset1 = ((n1*IH + ih1)*IW + iw1) * IC;
	int Xoffset2 = ((n2*IH + ih2)*IW + iw2) * IC;
	int Xoffset3 = ((n3*IH + ih3)*IW + iw3) * IC;

	bool flagX = (tx & 1);
	const int oph = (1 << LCFH) - 1, opw = (1 << LCFW) - 1;
	int tohs0 = ((IF_int(flagX, ih2, ih0) + ph) >> 1) - oph;
	int tohs1 = ((IF_int(flagX, ih3, ih1) + ph) >> 1) - oph;
	int tows0 = ((IF_int(flagX, iw2, iw0) + pw) >> 1) - opw;
	int tows1 = ((IF_int(flagX, iw3, iw1) + pw) >> 1) - opw;
	int Yoffset0 = ((IF_int(flagX, n2, n0)*OH + tohs0)*OW + tows0) * OC;
	int Yoffset1 = ((IF_int(flagX, n3, n1)*OH + tohs1)*OW + tows1) * OC;

	//prepare for GK = CFH * CFW * OC
	const int LCFH_CFW = LCFW + LCFW, CFH_CFW_m1 = (1 << LCFH_CFW) - 1;
	const int GK = OC << LCFH_CFW;

	//load 4 elem from W[OC, FH, FW, IC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);
	Ws[buf][ty][tx] = *(float4*)(CW + W_k * IC);

	//load 2 elem from deltaY[N, FH, FW, OC]
	int dY_k = tx >> 1;
	Ims_oc_fhr_fwr_CW2pow(dY_k, dY_oc, dY_fhr, dY_fwr);
	int yoffset = ((dY_fhr * OW) + dY_fwr)*OC + dY_oc;
	const int Ys_x = (tx >> 1), Ys_y = (ty << 1) + (tx & 1);
	zero_float(Ys[buf][Ys_x][Ys_y].x, Ims_ldy(tohs0, tows0), tex1Dfetch<float>(deltaY, Yoffset0 + yoffset));
	zero_float(Ys[buf][Ys_x][Ys_y].y, Ims_ldy(tohs1, tows1), tex1Dfetch<float>(deltaY, Yoffset1 + yoffset));
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
			float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];
			float4 y0 = *(float4*)(&Ys[buf][ik][ty << 1]);

			simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
			simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
			simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
			simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
		}
		buf ^= 1;

		//load 4 elem from W[OC, FH, FW, IC]
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		Ws[buf][ty][tx] = *(float4*)(CW + W_k * IC);

		//load 2 elem from deltaY[N, FH, FW, OC]
		int dY_k = ((ok << LB) + tx) >> 1;
		Ims_oc_fhr_fwr_CW2pow(dY_k, dY_oc, dY_fhr, dY_fwr);
		int yoffset = ((dY_fhr * OW) + dY_fwr)*OC + dY_oc;
		zero_float(Ys[buf][Ys_x][Ys_y].x, Ims_ldy(tohs0, tows0), tex1Dfetch<float>(deltaY, Yoffset0 + yoffset));
		zero_float(Ys[buf][Ys_x][Ys_y].y, Ims_ldy(tohs1, tows1), tex1Dfetch<float>(deltaY, Yoffset1 + yoffset));
		__syncthreads();
	}
#pragma once
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];
		float4 y0 = *(float4*)(&Ys[buf][ik][ty << 1]);

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
#ifndef DCONV3D_DX_KERNEL_SPLIT_IMS2_KERNEL_4_8R_TEXTURE
#define DCONV3D_DX_KERNEL_SPLIT_IMS2_KERNEL_4_8R_TEXTURE

//LB = 3: Size = 1.53125, Time = 1.258 msec, Performace = 2613.94 GFlop/s
//LB = 4: Size = 1.125, Time = 1.148 msec, Performace = 2104.46 GFlop/s
//LB = 3: Size = 1.125, Time = 1.326 msec, Performace = 1821.96 GFlop/s
template<int LB, int STEP>
__global__ void UksIms2_kernel_4_8R_texture(
	cudaTextureObject_t deltaY, int OH, int OW,
	const float* __restrict__ CW, int CFH, int CFW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int ph, int pw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB >> 1][(2 << LB) + 2];//follow k44
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];//follow k88

	//prepare for GZ = sh*sw
	int bz = blockIdx.z;
	CW += bz * CFH * CFW * OC * IC;//CW[y, x]

	ph = ph - (bz >> 1); pw = pw - (bz & 1);//if(ihs < 0): ihs += (ph + 1)/sh*sh
	int ihs = -ph + (-(ph > 0) & ((ph + 1) >> 1 << 1));
	int iws = -pw + (-(pw > 0) & ((pw + 1) >> 1 << 1));

	//prepare for GN = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 2) + ic_index;
	CW += ic0 + ((ty & 1) << 1);//W[0, 0, 0, tic0]
	deltaX += ic0; //deltaX[0, 0, 0, ic0]

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int j1 = j0 + 1, j2 = j0 + 2, j3 = j0 + 3;
	int j4 = j0 + 4, j5 = j0 + 5, j6 = j0 + 6, j7 = j0 + 7;
	int IW_slice = IW >> 1, IH_IW_slice = (IH >> 1) * IW_slice;
	Ims2_n_ih_iw(j0, n0, ih0, iw0);
	Ims2_n_ih_iw(j1, n1, ih1, iw1);
	Ims2_n_ih_iw(j2, n2, ih2, iw2);
	Ims2_n_ih_iw(j3, n3, ih3, iw3);
	Ims2_n_ih_iw(j4, n4, ih4, iw4);
	Ims2_n_ih_iw(j5, n5, ih5, iw5);
	Ims2_n_ih_iw(j6, n6, ih6, iw6);
	Ims2_n_ih_iw(j7, n7, ih7, iw7);
	int Xoffset0 = ((n0*IH + ih0)*IW + iw0) * IC;
	int Xoffset1 = ((n1*IH + ih1)*IW + iw1) * IC;
	int Xoffset2 = ((n2*IH + ih2)*IW + iw2) * IC;
	int Xoffset3 = ((n3*IH + ih3)*IW + iw3) * IC;
	int Xoffset4 = ((n4*IH + ih4)*IW + iw4) * IC;
	int Xoffset5 = ((n5*IH + ih5)*IW + iw5) * IC;
	int Xoffset6 = ((n6*IH + ih6)*IW + iw6) * IC;
	int Xoffset7 = ((n7*IH + ih7)*IW + iw7) * IC;

	bool flagX = (tx >= STEP);
	const int oph = CFH - 1, opw = CFW - 1;
	int tohs0 = ((IF_int(flagX, ih4, ih0) + ph) >> 1) - oph;
	int tohs1 = ((IF_int(flagX, ih5, ih1) + ph) >> 1) - oph;
	int tohs2 = ((IF_int(flagX, ih6, ih2) + ph) >> 1) - oph;
	int tohs3 = ((IF_int(flagX, ih7, ih3) + ph) >> 1) - oph;
	int tows0 = ((IF_int(flagX, iw4, iw0) + pw) >> 1) - opw;
	int tows1 = ((IF_int(flagX, iw5, iw1) + pw) >> 1) - opw;
	int tows2 = ((IF_int(flagX, iw6, iw2) + pw) >> 1) - opw;
	int tows3 = ((IF_int(flagX, iw7, iw3) + pw) >> 1) - opw;
	int Yoffset0 = ((IF_int(flagX, n4, n0)*OH + tohs0)*OW + tows0) * OC;
	int Yoffset1 = ((IF_int(flagX, n5, n1)*OH + tohs1)*OW + tows1) * OC;
	int Yoffset2 = ((IF_int(flagX, n6, n2)*OH + tohs2)*OW + tows2) * OC;
	int Yoffset3 = ((IF_int(flagX, n7, n3)*OH + tohs3)*OW + tows3) * OC;

	//prepare for GK = CFH * CFW * OC
	const int CFH_CFW = CFW * CFW, GK = OC * CFH_CFW;

	//load 4 elem from deltaY[N, FH, FW, OC]
	int dY_k = tx - ((tx >= STEP) << LB >> 1);
	Ims_oc_fhr_fwr(dY_k, dY_oc, dY_fhr, dY_fwr);
	int yoffset = (dY_fhr * OW + dY_fwr) * OC + dY_oc;
	zero_float(Ys[buf][tx][ty].x, Ims_ldy(tohs0, tows0), tex1Dfetch<float>(deltaY, Yoffset0 + yoffset));
	zero_float(Ys[buf][tx][ty].y, Ims_ldy(tohs1, tows1), tex1Dfetch<float>(deltaY, Yoffset1 + yoffset));
	zero_float(Ys[buf][tx][ty].z, Ims_ldy(tohs2, tows2), tex1Dfetch<float>(deltaY, Yoffset2 + yoffset));
	zero_float(Ys[buf][tx][ty].w, Ims_ldy(tohs3, tows3), tex1Dfetch<float>(deltaY, Yoffset3 + yoffset));

	//load 2 elem from W[OC, FH, FW, IC]
	int W_k = ty >> 1;
	const int Ws_y = (ty >> 1), Ws_x = (tx << 1) + (ty & 1);
	Ws[buf][Ws_y][Ws_x] = *(float2*)(CW + W_k * IC);
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

		//load 4 elem from deltaY[N, FH, FW, OC]
		int dY_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		Ims_oc_fhr_fwr(dY_k, dY_oc, dY_fhr, dY_fwr);
		int yoffset = (dY_fhr * OW + dY_fwr) * OC + dY_oc;
		zero_float(Ys[buf][tx][ty].x, Ims_ldy(tohs0, tows0), tex1Dfetch<float>(deltaY, Yoffset0 + yoffset));
		zero_float(Ys[buf][tx][ty].y, Ims_ldy(tohs1, tows1), tex1Dfetch<float>(deltaY, Yoffset1 + yoffset));
		zero_float(Ys[buf][tx][ty].z, Ims_ldy(tohs2, tows2), tex1Dfetch<float>(deltaY, Yoffset2 + yoffset));
		zero_float(Ys[buf][tx][ty].w, Ims_ldy(tohs3, tows3), tex1Dfetch<float>(deltaY, Yoffset3 + yoffset));

		//load 2 elem from W[OC, FH, FW, IC]
		int W_k = ((ok << LB) + ty) >> 1;
		Ws[buf][Ws_y][Ws_x] = *(float2*)(CW + W_k * IC);
		__syncthreads();
	}
#pragma once
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];
		float4 w0 = *(float4*)(&Ws[buf][ik][tx << 1]);

		simdMM4(v0, y0.x, w0);
		simdMM4(v2, y0.y, w0);
		simdMM4(v4, y0.z, w0);
		simdMM4(v6, y0.w, w0);
		simdMM4(v8, y1.x, w0);
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


//(Y: BLOCK_SIZE*4, X:BLOCK_SIZE*8), K % (BLOCK_SIZE/2) == 0, CFH, CFW is power of 2
#ifndef DCONV3D_DX_KERNEL_SPLIT_IMS2_KERNEL_4_8R_CW_2POW_TEXTURE
#define DCONV3D_DX_KERNEL_SPLIT_IMS2_KERNEL_4_8R_CW_2POW_TEXTURE

//LB = 3: Size = 1.53125, Time = 1.054 msec, Performace = 3119.86 GFlop/s
//LB = 4: Size = 1.125, Time = 1.028 msec, Performace = 2350.12 GFlop/s
//LB = 3: Size = 1.125, Time = 1.174 msec, Performace = 2057.85 GFlop/s
template<int LB, int STEP>
__global__ void UksIms2_kernel_4_8R_CW2pow_texture(
	cudaTextureObject_t deltaY, int OH, int OW,
	const float* __restrict__ CW, int LCFH, int LCFW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int ph, int pw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB >> 1][(2 << LB) + 2];//follow k44
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];//follow k88

	//prepare for GZ = sh*sw
	int bz = blockIdx.z;
	CW += (bz << LCFH << LCFW) * OC  * IC;//CW[y, x]

	ph = ph - (bz >> 1); pw = pw - (bz & 1);//if(ihs < 0): ihs += (ph + 1)/sh*sh
	int ihs = -ph + (-(ph > 0) & ((ph + 1) >> 1 << 1));
	int iws = -pw + (-(pw > 0) & ((pw + 1) >> 1 << 1));

	//prepare for GN = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 2) + ic_index;
	CW += ic0 + ((ty & 1) << 1);//W[0, 0, 0, tic0]
	deltaX += ic0; //deltaX[0, 0, 0, ic0]

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int j1 = j0 + 1, j2 = j0 + 2, j3 = j0 + 3;
	int j4 = j0 + 4, j5 = j0 + 5, j6 = j0 + 6, j7 = j0 + 7;
	int IW_slice = IW >> 1, IH_IW_slice = (IH >> 1) * IW_slice;
	Ims2_n_ih_iw(j0, n0, ih0, iw0);
	Ims2_n_ih_iw(j1, n1, ih1, iw1);
	Ims2_n_ih_iw(j2, n2, ih2, iw2);
	Ims2_n_ih_iw(j3, n3, ih3, iw3);
	Ims2_n_ih_iw(j4, n4, ih4, iw4);
	Ims2_n_ih_iw(j5, n5, ih5, iw5);
	Ims2_n_ih_iw(j6, n6, ih6, iw6);
	Ims2_n_ih_iw(j7, n7, ih7, iw7);
	int Xoffset0 = ((n0*IH + ih0)*IW + iw0) * IC;
	int Xoffset1 = ((n1*IH + ih1)*IW + iw1) * IC;
	int Xoffset2 = ((n2*IH + ih2)*IW + iw2) * IC;
	int Xoffset3 = ((n3*IH + ih3)*IW + iw3) * IC;
	int Xoffset4 = ((n4*IH + ih4)*IW + iw4) * IC;
	int Xoffset5 = ((n5*IH + ih5)*IW + iw5) * IC;
	int Xoffset6 = ((n6*IH + ih6)*IW + iw6) * IC;
	int Xoffset7 = ((n7*IH + ih7)*IW + iw7) * IC;

	bool flagX = (tx >= STEP);
	const int oph = (1 << LCFH) - 1, opw = (1 << LCFW) - 1;
	int tohs0 = ((IF_int(flagX, ih4, ih0) + ph) >> 1) - oph;
	int tohs1 = ((IF_int(flagX, ih5, ih1) + ph) >> 1) - oph;
	int tohs2 = ((IF_int(flagX, ih6, ih2) + ph) >> 1) - oph;
	int tohs3 = ((IF_int(flagX, ih7, ih3) + ph) >> 1) - oph;
	int tows0 = ((IF_int(flagX, iw4, iw0) + pw) >> 1) - opw;
	int tows1 = ((IF_int(flagX, iw5, iw1) + pw) >> 1) - opw;
	int tows2 = ((IF_int(flagX, iw6, iw2) + pw) >> 1) - opw;
	int tows3 = ((IF_int(flagX, iw7, iw3) + pw) >> 1) - opw;
	int Yoffset0 = ((IF_int(flagX, n4, n0)*OH + tohs0)*OW + tows0) * OC;
	int Yoffset1 = ((IF_int(flagX, n5, n1)*OH + tohs1)*OW + tows1) * OC;
	int Yoffset2 = ((IF_int(flagX, n6, n2)*OH + tohs2)*OW + tows2) * OC;
	int Yoffset3 = ((IF_int(flagX, n7, n3)*OH + tohs3)*OW + tows3) * OC;

	//prepare for GK = CFH * CFW * OC
	const int LCFH_CFW = LCFW + LCFW, CFH_CFW_m1 = (1 << LCFH_CFW) - 1;
	const int GK = OC << LCFH_CFW;

	//load 4 elem from deltaY[N, FH, FW, OC]
	int dY_k = tx - ((tx >= STEP) << LB >> 1);
	Ims_oc_fhr_fwr_CW2pow(dY_k, dY_oc, dY_fhr, dY_fwr);
	int yoffset = (dY_fhr * OW + dY_fwr) * OC + dY_oc;
	zero_float(Ys[buf][tx][ty].x, Ims_ldy(tohs0, tows0), tex1Dfetch<float>(deltaY, Yoffset0 + yoffset));
	zero_float(Ys[buf][tx][ty].y, Ims_ldy(tohs1, tows1), tex1Dfetch<float>(deltaY, Yoffset1 + yoffset));
	zero_float(Ys[buf][tx][ty].z, Ims_ldy(tohs2, tows2), tex1Dfetch<float>(deltaY, Yoffset2 + yoffset));
	zero_float(Ys[buf][tx][ty].w, Ims_ldy(tohs3, tows3), tex1Dfetch<float>(deltaY, Yoffset3 + yoffset));

	//load 2 elem from W[OC, FH, FW, IC]
	int W_k = ty >> 1;
	const int Ws_y = (ty >> 1), Ws_x = (tx << 1) + (ty & 1);
	Ws[buf][Ws_y][Ws_x] = *(float2*)(CW + W_k * IC);
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

		//load 4 elem from deltaY[N, FH, FW, OC]
		int dY_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		Ims_oc_fhr_fwr_CW2pow(dY_k, dY_oc, dY_fhr, dY_fwr);
		int yoffset = (dY_fhr * OW + dY_fwr) * OC + dY_oc;
		zero_float(Ys[buf][tx][ty].x, Ims_ldy(tohs0, tows0), tex1Dfetch<float>(deltaY, Yoffset0 + yoffset));
		zero_float(Ys[buf][tx][ty].y, Ims_ldy(tohs1, tows1), tex1Dfetch<float>(deltaY, Yoffset1 + yoffset));
		zero_float(Ys[buf][tx][ty].z, Ims_ldy(tohs2, tows2), tex1Dfetch<float>(deltaY, Yoffset2 + yoffset));
		zero_float(Ys[buf][tx][ty].w, Ims_ldy(tohs3, tows3), tex1Dfetch<float>(deltaY, Yoffset3 + yoffset));

		//load 2 elem from W[OC, FH, FW, IC]
		int W_k = ((ok << LB) + ty) >> 1;
		Ws[buf][Ws_y][Ws_x] = *(float2*)(CW + W_k * IC);
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
#ifndef DCONV3D_DX_KERNEL_SPLIT_IMS2_KERNEL_4_4R_TEXTURE
#define DCONV3D_DX_KERNEL_SPLIT_IMS2_KERNEL_4_4R_TEXTURE

//LB = 3: Size = 1.53125, Time = 1.61 msec, Performace = 2042.44 GFlop/s
//LB = 4: Size = 1.125, Time = 1.312 msec, Performace = 1841.4 GFlop/s
//LB = 3: Size = 1.125, Time = 1.67  msec, Performace = 1446.66 GFlop/s
template<int LB, int STEP>
__global__ void UksIms2_kernel_4_4R_texture(
	cudaTextureObject_t deltaY, int OH, int OW,
	const float* __restrict__ CW, int CFH, int CFW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int ph, int pw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2 Ys[2][1 << LB >> 1][(2 << LB) + 2];

	//prepare for GZ = sh*sw
	int bz = blockIdx.z;
	CW += bz * CFH * CFW * OC * IC;//CW[y, x]

	ph = ph - (bz >> 1); pw = pw - (bz & 1);//if(ihs < 0): ihs += (ph + 1)/sh*sh
	int ihs = -ph + (-(ph > 0) & ((ph + 1) >> 1 << 1));
	int iws = -pw + (-(pw > 0) & ((pw + 1) >> 1 << 1));

	//prepare for GN = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 2) + ic_index;
	CW += ic0 + ((ty & 1) << 1);//W[0, 0, 0, tic0]
	deltaX += ic0; //deltaX[0, 0, 0, ic0]

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = (((blockIdx.y << LB) + ty) << 2) + j_index;
	int j1 = j0 + 1, j2 = j0 + 2, j3 = j0 + 3;
	int IW_slice = IW >> 1, IH_IW_slice = (IH >> 1) * IW_slice;
	Ims2_n_ih_iw(j0, n0, ih0, iw0);
	Ims2_n_ih_iw(j1, n1, ih1, iw1);
	Ims2_n_ih_iw(j2, n2, ih2, iw2);
	Ims2_n_ih_iw(j3, n3, ih3, iw3);
	int Xoffset0 = ((n0*IH + ih0)*IW + iw0) * IC;
	int Xoffset1 = ((n1*IH + ih1)*IW + iw1) * IC;
	int Xoffset2 = ((n2*IH + ih2)*IW + iw2) * IC;
	int Xoffset3 = ((n3*IH + ih3)*IW + iw3) * IC;

	bool flagX = (tx & 1);
	const int oph = CFH - 1, opw = CFW - 1;
	int tohs0 = ((IF_int(flagX, ih2, ih0) + ph) >> 1) - oph;
	int tohs1 = ((IF_int(flagX, ih3, ih1) + ph) >> 1) - oph;
	int tows0 = ((IF_int(flagX, iw2, iw0) + pw) >> 1) - opw;
	int tows1 = ((IF_int(flagX, iw3, iw1) + pw) >> 1) - opw;
	int Yoffset0 = ((IF_int(flagX, n2, n0)*OH + tohs0)*OW + tows0) * OC;
	int Yoffset1 = ((IF_int(flagX, n3, n1)*OH + tohs1)*OW + tows1) * OC;

	//prepare for GK = CFH * CFW * OC
	const int CFH_CFW = CFW * CFW, GK = OC* CFH_CFW;

	//load 2 elem from W[OC, FH, FW, IC]
	int W_k = ty >> 1;
	const int Ws_y = (ty >> 1), Ws_x = (tx << 1) + (ty & 1);
	Ws[buf][Ws_y][Ws_x] = *(float2*)(CW + W_k * IC);

	//load 2 elem from deltaY[N, FH, FW, OC]
	int dY_k = tx >> 1;
	Ims_oc_fhr_fwr(dY_k, dY_oc, dY_fhr, dY_fwr);
	int yoffset = ((dY_fhr * OW) + dY_fwr)*OC + dY_oc;
	const int Ys_x = (tx >> 1), Ys_y = (ty << 1) + (tx & 1);
	zero_float(Ys[buf][Ys_x][Ys_y].x, Ims_ldy(tohs0, tows0), tex1Dfetch<float>(deltaY, Yoffset0 + yoffset));
	zero_float(Ys[buf][Ys_x][Ys_y].y, Ims_ldy(tohs1, tows1), tex1Dfetch<float>(deltaY, Yoffset1 + yoffset));
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
			float4 w = *(float4*)(&Ws[buf][ik][tx << 1]);
			float4 y = *(float4*)(&Ys[buf][ik][ty << 1]);

			simdMM4(v0, y.x, w);
			simdMM4(v1, y.y, w);
			simdMM4(v2, y.z, w);
			simdMM4(v3, y.w, w);
		}
		buf ^= 1;

		//load 2 elem from W[OC, FH, FW, IC]
		int W_k = ((ok << LB) + ty) >> 1;
		Ws[buf][Ws_y][Ws_x] = *(float2*)(CW + W_k * IC);

		//load 2 elem from deltaY[N, FH, FW, OC]
		int dY_k = ((ok << LB) + tx) >> 1;
		Ims_oc_fhr_fwr(dY_k, dY_oc, dY_fhr, dY_fwr);
		int yoffset = ((dY_fhr * OW) + dY_fwr)*OC + dY_oc;
		zero_float(Ys[buf][Ys_x][Ys_y].x, Ims_ldy(tohs0, tows0), tex1Dfetch<float>(deltaY, Yoffset0 + yoffset));
		zero_float(Ys[buf][Ys_x][Ys_y].y, Ims_ldy(tohs1, tows1), tex1Dfetch<float>(deltaY, Yoffset1 + yoffset));
		__syncthreads();
	}
#pragma once
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 w = *(float4*)(&Ws[buf][ik][tx << 1]);
		float4 y = *(float4*)(&Ys[buf][ik][ty << 1]);

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


//(Y: BLOCK_SIZE*4, X:BLOCK_SIZE*4), K % (BLOCK_SIZE/2) == 0, CFH, CFW is power of 2
#ifndef DCONV3D_DX_KERNEL_SPLIT_IMS2_KERNEL_4_4R_CW_2POW_TEXTURE
#define DCONV3D_DX_KERNEL_SPLIT_IMS2_KERNEL_4_4R_CW_2POW_TEXTURE

//LB = 3: Size = 1.53125, Time = 1.258 msec, Performace = 2613.94 GFlop/s
//LB = 4: Size = 1.125, Time = 1.13 msec, Performace = 2137.98 GFlop/s
//LB = 3: Size = 1.125, Time = 1.33 msec, Performace = 1816.48 GFlop/s
template<int LB, int STEP>
__global__ void UksIms2_kernel_4_4R_CW2pow_texture(
	cudaTextureObject_t deltaY, int OH, int OW,
	const float* __restrict__ CW, int LCFH, int LCFW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int ph, int pw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2 Ys[2][1 << LB >> 1][(2 << LB) + 2];

	//prepare for GZ = sh*sw
	int bz = blockIdx.z;
	CW += (bz << LCFH << LCFW) * OC * IC;//CW[y, x]

	ph = ph - (bz >> 1); pw = pw - (bz & 1);//if(ihs < 0): ihs += (ph + 1)/sh*sh
	int ihs = -ph + (-(ph > 0) & ((ph + 1) >> 1 << 1));
	int iws = -pw + (-(pw > 0) & ((pw + 1) >> 1 << 1));

	//prepare for GN = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 2) + ic_index;
	CW += ic0 + ((ty & 1) << 1);//W[0, 0, 0, tic0]
	deltaX += ic0; //deltaX[0, 0, 0, ic0]

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = (((blockIdx.y << LB) + ty) << 2) + j_index;
	int j1 = j0 + 1, j2 = j0 + 2, j3 = j0 + 3;
	int IW_slice = IW >> 1, IH_IW_slice = (IH >> 1) * IW_slice;
	Ims2_n_ih_iw(j0, n0, ih0, iw0);
	Ims2_n_ih_iw(j1, n1, ih1, iw1);
	Ims2_n_ih_iw(j2, n2, ih2, iw2);
	Ims2_n_ih_iw(j3, n3, ih3, iw3);
	int Xoffset0 = ((n0*IH + ih0)*IW + iw0) * IC;
	int Xoffset1 = ((n1*IH + ih1)*IW + iw1) * IC;
	int Xoffset2 = ((n2*IH + ih2)*IW + iw2) * IC;
	int Xoffset3 = ((n3*IH + ih3)*IW + iw3) * IC;

	bool flagX = (tx & 1);
	const int oph = (1 << LCFH) - 1, opw = (1 << LCFW) - 1;
	int tohs0 = ((IF_int(flagX, ih2, ih0) + ph) >> 1) - oph;
	int tohs1 = ((IF_int(flagX, ih3, ih1) + ph) >> 1) - oph;
	int tows0 = ((IF_int(flagX, iw2, iw0) + pw) >> 1) - opw;
	int tows1 = ((IF_int(flagX, iw3, iw1) + pw) >> 1) - opw;
	int Yoffset0 = ((IF_int(flagX, n2, n0)*OH + tohs0)*OW + tows0) * OC;
	int Yoffset1 = ((IF_int(flagX, n3, n1)*OH + tohs1)*OW + tows1) * OC;

	//prepare for GK = CFH * CFW * OC
	const int LCFH_CFW = LCFW + LCFW, CFH_CFW_m1 = (1 << LCFH_CFW) - 1;
	const int GK = OC << LCFH_CFW;

	//load 2 elem from W[OC, FH, FW, IC]
	int W_k = ty >> 1;
	const int Ws_y = (ty >> 1), Ws_x = (tx << 1) + (ty & 1);
	Ws[buf][Ws_y][Ws_x] = *(float2*)(CW + W_k * IC);

	//load 2 elem from deltaY[N, FH, FW, OC]
	int dY_k = tx >> 1;
	Ims_oc_fhr_fwr_CW2pow(dY_k, dY_oc, dY_fhr, dY_fwr);
	int yoffset = ((dY_fhr * OW) + dY_fwr)*OC + dY_oc;
	const int Ys_x = (tx >> 1), Ys_y = (ty << 1) + (tx & 1);
	zero_float(Ys[buf][Ys_x][Ys_y].x, Ims_ldy(tohs0, tows0), tex1Dfetch<float>(deltaY, Yoffset0 + yoffset));
	zero_float(Ys[buf][Ys_x][Ys_y].y, Ims_ldy(tohs1, tows1), tex1Dfetch<float>(deltaY, Yoffset1 + yoffset));
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
			float4 w = *(float4*)(&Ws[buf][ik][tx << 1]);
			float4 y = *(float4*)(&Ys[buf][ik][ty << 1]);

			simdMM4(v0, y.x, w);
			simdMM4(v1, y.y, w); 
			simdMM4(v2, y.z, w); 
			simdMM4(v3, y.w, w);
		}
		buf ^= 1;

		//load 2 elem from W[OC, FH, FW, IC]
		int W_k = ((ok << LB) + ty) >> 1;
		Ws[buf][Ws_y][Ws_x] = *(float2*)(CW + W_k * IC);

		//load 2 elem from deltaY[N, FH, FW, OC]
		int dY_k = ((ok << LB) + tx) >> 1;
		Ims_oc_fhr_fwr_CW2pow(dY_k, dY_oc, dY_fhr, dY_fwr);
		int yoffset = ((dY_fhr * OW) + dY_fwr)*OC + dY_oc;
		zero_float(Ys[buf][Ys_x][Ys_y].x, Ims_ldy(tohs0, tows0), tex1Dfetch<float>(deltaY, Yoffset0 + yoffset));
		zero_float(Ys[buf][Ys_x][Ys_y].y, Ims_ldy(tohs1, tows1), tex1Dfetch<float>(deltaY, Yoffset1 + yoffset));
		__syncthreads();
	}
#pragma once
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 w = *(float4*)(&Ws[buf][ik][tx << 1]);
		float4 y = *(float4*)(&Ys[buf][ik][ty << 1]);

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

#endif
