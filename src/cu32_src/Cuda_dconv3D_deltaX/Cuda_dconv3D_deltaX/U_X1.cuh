
//(Y: BLOCK_SIZE*8, X:BLOCK_SIZE*8), K % (BLOCK_SIZE/2) == 0, CFH, CFW is power of 2
#ifndef X_TEXTURE
#define X_TEXTURE

#define X_tex(stream, LB, ic_index, j_index, deltaY, OH, OW, CW, LCFH, LCFW, deltaX, LIH, LIW, IC, OC, ph, pw, GN, GM) \
	X_texture<LB, (1<<LB>>1), LIH, LIW>\
		<<< dim3((GN>>LB>>3), (GM>>LB>>3), 4), dim3(1<<LB, 1<<LB) >>>\
			(deltaY, OH, OW, CW, deltaX, IC, OC, ph, pw, ic_index, j_index)


//LB = 4: Size = 1.125, Time = 1.1 msec, Performace = 2196.29 GFlop/s
//LB = 3: Size = 1.125, Time = 0.952 msec, Performace = 2537.73 GFlop/s
template<int LB, int STEP, int LIH, int LIW>
__global__ void X_texture(
	cudaTextureObject_t deltaY, int OH, int OW,
	const float* __restrict__ CW, 
	float* __restrict__ deltaX, 
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
	int y = bz >> 1, x = bz & 1;
	CW += (bz << 2) * OC * IC;//CW[y, x]

	ph = ph - y; pw = pw - x;//if(ihs < 0): ihs += (ph + 1)/sh*sh
	int ihs = -ph; ihs += -(ihs < 0) & ((ph + 1) << 1 >> 1);
	int iws = -pw; iws += -(iws < 0) & ((pw + 1) << 1 >> 1);

	//prepare for GN = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	CW += ic0 + ((ty >= STEP) << 2);//CW[y, x, 0, 0, 0, tic0]
	deltaX += ic0; //deltaX[0, 0, 0, ic0]

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int j1 = j0 + 1, j2 = j0 + 2, j3 = j0 + 3;
	int j4 = j0 + 4, j5 = j0 + 5, j6 = j0 + 6, j7 = j0 + 7;
	Ims2_n_ih_iw_I2pow(j0, n0, ih0, iw0);
	Ims2_n_ih_iw_I2pow(j1, n1, ih1, iw1);
	Ims2_n_ih_iw_I2pow(j2, n2, ih2, iw2);
	Ims2_n_ih_iw_I2pow(j3, n3, ih3, iw3);
	Ims2_n_ih_iw_I2pow(j4, n4, ih4, iw4);
	Ims2_n_ih_iw_I2pow(j5, n5, ih5, iw5);
	Ims2_n_ih_iw_I2pow(j6, n6, ih6, iw6);
	Ims2_n_ih_iw_I2pow(j7, n7, ih7, iw7);
	int Xoffset0 = ((((n0 << LIH) + ih0) << LIW) + iw0)*IC;
	int Xoffset1 = ((((n1 << LIH) + ih1) << LIW) + iw1)*IC;
	int Xoffset2 = ((((n2 << LIH) + ih2) << LIW) + iw2)*IC;
	int Xoffset3 = ((((n3 << LIH) + ih3) << LIW) + iw3)*IC;
	int Xoffset4 = ((((n4 << LIH) + ih4) << LIW) + iw4)*IC;
	int Xoffset5 = ((((n5 << LIH) + ih5) << LIW) + iw5)*IC;
	int Xoffset6 = ((((n6 << LIH) + ih6) << LIW) + iw6)*IC;
	int Xoffset7 = ((((n7 << LIH) + ih7) << LIW) + iw7)*IC;

	bool flagX = (tx >= STEP);
	int tohs0 = ((IF_int(flagX, ih4, ih0) + ph) >> 1) - 1;
	int tohs1 = ((IF_int(flagX, ih5, ih1) + ph) >> 1) - 1;
	int tohs2 = ((IF_int(flagX, ih6, ih2) + ph) >> 1) - 1;
	int tohs3 = ((IF_int(flagX, ih7, ih3) + ph) >> 1) - 1;
	int tows0 = ((IF_int(flagX, iw4, iw0) + pw) >> 1) - 1;
	int tows1 = ((IF_int(flagX, iw5, iw1) + pw) >> 1) - 1;
	int tows2 = ((IF_int(flagX, iw6, iw2) + pw) >> 1) - 1;
	int tows3 = ((IF_int(flagX, iw7, iw3) + pw) >> 1) - 1;
	int Yoffset0 = ((IF_int(flagX, n4, n0)*OH + tohs0)*OW + tows0)*OC;
	int Yoffset1 = ((IF_int(flagX, n5, n1)*OH + tohs1)*OW + tows1)*OC;
	int Yoffset2 = ((IF_int(flagX, n6, n2)*OH + tohs2)*OW + tows2)*OC;
	int Yoffset3 = ((IF_int(flagX, n7, n3)*OH + tohs3)*OW + tows3)*OC;

	//prepare for GK = CFH * CFW * OC
	const int GK = OC << 2;

	//load 4 elem from deltaY[N, FH, FW, OC]
	int dY_k = tx - ((tx >= STEP) << LB >> 1);
	Ims_oc_fhr_fwr_W3(dY_k, dY_oc, dY_fhr, dY_fwr);
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
		Ims_oc_fhr_fwr_W3(dY_k, dY_oc, dY_fhr, dY_fwr);
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


//(Y: BLOCK_SIZE*8, X:BLOCK_SIZE*8), K % (BLOCK_SIZE/2) == 0, CFH, CFW is power of 2
#ifndef X2_TEXTURE
#define X2_TEXTURE

#define X2_tex(stream, LB, ic_index, j_index, deltaY, OH, OW, CW, LCFH, LCFW, deltaX, LIH, LIW, IC, OC, ph, pw, GN, GM) \
	X2_texture<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>3), (GM>>LB>>3), 4), dim3(1<<LB, 1<<LB) >>>\
			(deltaY, OH, OW, CW, LCFH, LCFW, deltaX, IH, IW, IC, OC, ph, pw, ic_index, j_index)

//LB = 4: Size = 1.125, Time = 1.108 msec, Performace = 2180.43 GFlop/s
//LB = 3: Size = 1.125, Time = 0.956 msec, Performace = 2527.11 GFlop/s
template<int LB, int STEP>
__global__ void X2_texture(
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
	const int LCFH_CFW = LCFW + LCFW, CFH_CFW_m1 = (1 << LCFH_CFW) - 1;
	const int GK = OC << LCFH_CFW;

	//prepare for GZ = sh*sw
	int bz = blockIdx.z;
	int y = bz >> 1, x = bz & 1;
	CW += (bz << LCFH_CFW) * OC * IC;//CW[y, x]

	ph = ph - y; pw = pw - x;//if(ihs < 0): ihs += (ph + 1)/sh*sh
	int ihs = -ph; ihs += -(ihs < 0) & ((ph + 1) << 1 >> 1);
	int iws = -pw; iws += -(iws < 0) & ((pw + 1) << 1 >> 1);

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

	if(bz == 0)
	printf("%d, %d, %d, %d, %d, %d, %d, %d\n", Xoffset0, Xoffset1, Xoffset2, Xoffset3, Xoffset4, Xoffset5, Xoffset6, Xoffset7);

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

	//load 4 elem from deltaY[N, FH, FW, OC]
	int dY_k = tx - ((tx >= STEP) << LB >> 1);
	int W_k = ty - ((ty >= STEP) << LB >> 1);//W_k ->(oc, fhr, fwr)
	Ims_oc_fhr_fwr_CW2pow(dY_k, dY_oc, dY_fhr, dY_fwr);
	int yoffset = (dY_fhr * OW + dY_fwr)*OC + dY_oc;

	//load 4 elem from W[OC, FH, FW, IC]
	Ws[buf][ty][tx] = *(float4*)(CW + W_k * IC);
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
		int W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;//W_k ->(oc, fhr, fwr)

		Ims_oc_fhr_fwr_CW2pow(dY_k, dY_oc, dY_fhr, dY_fwr);
		int yoffset = (dY_fhr * OW + dY_fwr)*OC + dY_oc;

		//load 4 elem from W[OC, FH, FW, IC]
		Ws[buf][ty][tx] = *(float4*)(CW + W_k * IC);
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

	/*const int Xoffset1 = Xoffset0 + 256;
	const int Xoffset2 = Xoffset1 + 256;
	const int Xoffset3 = Xoffset2 + 256;
	const int Xoffset4 = Xoffset3 + 256;
	const int Xoffset5 = Xoffset4 + 256;
	const int Xoffset6 = Xoffset5 + 256;
	const int Xoffset7 = Xoffset6 + 256;*/

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


//when: IW_slice % 8 == 0 -> IH % 16 == 0
//(Y: BLOCK_SIZE*8, X:BLOCK_SIZE*8), K % (BLOCK_SIZE/2) == 0, CFH, CFW is power of 2
#ifndef X3_TEXTURE
#define X3_TEXTURE

#define X3_tex(stream, LB, ic_index, j_index, deltaY, OH, OW, CW, LCFH, LCFW, deltaX, LIH, LIW, IC, OC, ph, pw, GN, GM) \
	X3_texture<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>3), (GM>>LB>>3), 4), dim3(1<<LB, 1<<LB) >>>\
			(deltaY, OH, OW, CW, LCFH, LCFW, deltaX, IH, IW, IC, OC, ph, pw, ic_index, j_index)

//LB = 4: Size = 1.125, Time = 0.964 msec, Performace = 2506.14 GFlop/s
//LB = 3: Size = 1.125, Time = 0.956 msec, Performace = 2527.11 GFlop/s
template<int LB, int STEP>
__global__ void X3_texture(
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
	int y = bz >> 1, x = bz & 1;
	CW += (bz << LCFH << LCFW) * OC * IC;//CW[y, x]

	ph = ph - y; pw = pw - x;//if(ihs < 0): ihs += (ph + 1)/sh*sh
	int ihs = -ph; ihs += -(ihs < 0) & ((ph + 1) << 1 >> 1);
	int iws = -pw; iws += -(iws < 0) & ((pw + 1) << 1 >> 1);

	//prepare for GN = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	CW += ic0 + ((ty >= STEP) << 2);//CW[y, x, 0, 0, 0, tic0]

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
	int X0 = ((n0*IH + ih0)*IW + iw0)*IC + ic0;

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

	
	int IC2 = IC << 1;//IC*sw
	int X1 = X0 + IC2, X2 = X1 + IC2, X3 = X2 + IC2;
	int X4 = X3 + IC2, X5 = X4 + IC2, X6 = X5 + IC2, X7 = X6 + IC2;

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

//when: IW_slice % 8 == 0
//(Y: BLOCK_SIZE*8, X:BLOCK_SIZE*8), K % (BLOCK_SIZE/2) == 0, CFH, CFW is power of 2
#ifndef X4_TEXTURE
#define X4_TEXTURE

#define X4_tex(stream, LB, ic_index, j_index, deltaY, OH, OW, CW, LCFH, LCFW, deltaX, LIH, LIW, IC, OC, ph, pw, GN, GM) \
	X4_texture<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>3), (GM>>LB>>3), 4), dim3(1<<LB, 1<<LB) >>>\
			(deltaY, OH, OW, CW, LCFH, LCFW, deltaX, IH, IW, IC, OC, ph, pw, ic_index, j_index)

//LB = 4: Size = 1.125, Time = 0.954 msec, Performace = 2532.41 GFlop/s
//LB = 3: Size = 1.125, Time = 0.956 msec, Performace = 2527.11 GFlop/s
template<int LB, int STEP>
__global__ void X4_texture(
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
	const int LCFH_CFW = LCFW + LCFW, CFH_CFW_m1 = (1 << LCFH_CFW) - 1;
	const int GK = OC << LCFH_CFW;

	//prepare for GZ = sh*sw
	int bz = blockIdx.z;
	CW += (bz << LCFH << LCFW) * OC * IC;//CW[y, x]

	//y = bz >> 1, x = bz & 1;
	ph = ph -(bz >> 1); pw = pw - (bz & 1);//if(ihs < 0): ihs += (ph + 1)/sh*sh
	int ihs = -ph; ihs += -(ihs < 0) & ((ph + 1) << 1 >> 1);
	int iws = -pw; iws += -(iws < 0) & ((pw + 1) << 1 >> 1);

	//prepare for GN = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	CW += ic0 + ((ty >= STEP) << 2);//CW[y, x, 0, 0, 0, tic0]

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int IW_slice = IW >> 1, IH_IW_slice = (IH >> 1) * IW_slice;
	Ims2_n_ih_iw(j0, n0, ih0, iw0); int X0 = ((n0*IH + ih0)*IW + iw0)*IC + ic0;
	int tj0 = j0 + ((tx >= STEP) << 2);
	int tj1 = tj0 + 1, tj2 = tj0 + 2, tj3 = tj0 + 3, tj4 = tj0 + 4;
	Ims2_n_ih_iw(tj0, tn0, tohs0, tows0);
	Ims2_n_ih_iw(tj1, tn1, tohs1, tows1);
	Ims2_n_ih_iw(tj2, tn2, tohs2, tows2);
	Ims2_n_ih_iw(tj3, tn3, tohs3, tows3);
	const int oph = (1 << LCFH) - 1, opw = (1 << LCFW) - 1;
	tohs0 = ((tohs0 + ph) >> 1) - oph, tows0 = ((tows0 + pw) >> 1) - opw;
	tohs1 = ((tohs1 + ph) >> 1) - oph, tows1 = ((tows1 + pw) >> 1) - opw;
	tohs2 = ((tohs2 + ph) >> 1) - oph, tows2 = ((tows2 + pw) >> 1) - opw;
	tohs3 = ((tohs3 + ph) >> 1) - oph, tows3 = ((tows3 + pw) >> 1) - opw;
	int Yoffset0 = ((tn0*OH + tohs0)*OW + tows0)*OC;
	int Yoffset1 = ((tn1*OH + tohs1)*OW + tows1)*OC;
	int Yoffset2 = ((tn2*OH + tohs2)*OW + tows2)*OC;
	int Yoffset3 = ((tn3*OH + tohs3)*OW + tows3)*OC;

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


	int IC2 = IC << 1;//IC*sw
	int X1 = X0 + IC2, X2 = X1 + IC2, X3 = X2 + IC2;
	int X4 = X3 + IC2, X5 = X4 + IC2, X6 = X5 + IC2, X7 = X6 + IC2;

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


//when: (IH_slice, IW_slice) % 8 == 0
//(Y: BLOCK_SIZE*8, X:BLOCK_SIZE*8), K % (BLOCK_SIZE/2) == 0, CFH, CFW is power of 2
#ifndef X5_TEXTURE
#define X5_TEXTURE

#define X5_tex(stream, LB, ic_index, j_index, deltaY, OH, OW, CW, LCFH, LCFW, deltaX, LIH, LIW, IC, OC, ph, pw, GN, GM) \
	X5_texture<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>3), (GM>>LB>>3), 4), dim3(1<<LB, 1<<LB) >>>\
			(deltaY, OH, OW, CW, LCFH, LCFW, deltaX, IH, IW, IC, OC, ph, pw, ic_index, j_index)

//LB = 4: Size = 1.125, Time = 0.952 msec, Performace = 2537.73 GFlop/s
//LB = 3: Size = 1.125, Time = 0.956 msec, Performace = 2527.11 GFlop/s
template<int LB, int STEP>
__global__ void X5_texture(
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
	const int LCFH_CFW = LCFW + LCFW, CFH_CFW_m1 = (1 << LCFH_CFW) - 1;
	const int GK = OC << LCFH_CFW;

	//prepare for GZ = sh*sw
	int bz = blockIdx.z;
	CW += (bz << LCFH << LCFW) * OC * IC;//CW[y, x]

	//y = bz >> 1, x = bz & 1;
	ph = ph - (bz >> 1); pw = pw - (bz & 1);//if(ihs < 0): ihs += (ph + 1)/sh*sh
	int ihs = -ph; ihs += -(ihs < 0) & ((ph + 1) << 1 >> 1);
	int iws = -pw; iws += -(iws < 0) & ((pw + 1) << 1 >> 1);

	//prepare for GN = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	CW += ic0 + ((ty >= STEP) << 2);//CW[y, x, 0, 0, 0, tic0]

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int IW_slice = IW >> 1, IH_IW_slice = (IH >> 1) * IW_slice;
	Ims2_n_ih_iw(j0, n0, ih0, iw0);
	int X0 = ((n0*IH + ih0)*IW + iw0)*IC + ic0;

	int tj0 = j0 + ((tx >= STEP) << 2);
	int tj1 = tj0 + 1, tj2 = tj0 + 2, tj3 = tj0 + 3, tj4 = tj0 + 4;
	Ims2_ih_iw(tj0, tohs0, tows0);
	Ims2_ih_iw(tj1, tohs1, tows1);
	Ims2_ih_iw(tj2, tohs2, tows2);
	Ims2_ih_iw(tj3, tohs3, tows3);
	const int oph = (1 << LCFH) - 1, opw = (1 << LCFW) - 1;
	tohs0 = ((tohs0 + ph) >> 1) - oph, tows0 = ((tows0 + pw) >> 1) - opw;
	tohs1 = ((tohs1 + ph) >> 1) - oph, tows1 = ((tows1 + pw) >> 1) - opw;
	tohs2 = ((tohs2 + ph) >> 1) - oph, tows2 = ((tows2 + pw) >> 1) - opw;
	tohs3 = ((tohs3 + ph) >> 1) - oph, tows3 = ((tows3 + pw) >> 1) - opw;
	int Yoffset0 = ((n0*OH + tohs0)*OW + tows0)*OC;
	int Yoffset1 = ((n0*OH + tohs1)*OW + tows1)*OC;
	int Yoffset2 = ((n0*OH + tohs2)*OW + tows2)*OC;
	int Yoffset3 = ((n0*OH + tohs3)*OW + tows3)*OC;

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


	int IC2 = IC << 1;//IC*sw
	int X1 = X0 + IC2, X2 = X1 + IC2, X3 = X2 + IC2;
	int X4 = X3 + IC2, X5 = X4 + IC2, X6 = X5 + IC2, X7 = X6 + IC2;

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


//when: (IH_slice, IW_slice) % 8 == 0
//(Y: BLOCK_SIZE*8, X:BLOCK_SIZE*8), K % (BLOCK_SIZE/2) == 0, CFH, CFW is power of 2
#ifndef X6_TEXTURE
#define X6_TEXTURE

#define X6_tex(stream, LB, ic_index, j_index, deltaY, OH, OW, CW, LCFH, LCFW, deltaX, LIH, LIW, IC, OC, ph, pw, GN, GM) \
	X6_texture<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>3), (GM>>LB>>3), 4), dim3(1<<LB, 1<<LB) >>>\
			(deltaY, OH, OW, CW, LCFH, LCFW, deltaX, IH, IW, IC, OC, ph, pw, ic_index, j_index)

//LB = 4: Size = 2.25, Time = 1.718 msec, Performace = 2812.48 GFlop/s
//LB = 3: Size = 2.25, Time = 1.732 msec, Performace = 2789.74 GFlop/s
//LB = 4: Size = 1.125, Time = 0.938 msec, Performace = 2575.61 GFlop/s
//LB = 3: Size = 1.125, Time = 0.956 msec, Performace = 2527.11 GFlop/s
template<int LB, int STEP>
__global__ void X6_texture(
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
	const int LCFH_CFW = LCFW + LCFW, CFH_CFW_m1 = (1 << LCFH_CFW) - 1;
	const int GK = OC << LCFH_CFW;

	//prepare for GZ = sh*sw
	int bz = blockIdx.z;
	CW += (bz << LCFH << LCFW) * OC * IC;//CW[y, x]

	//y = bz >> 1, x = bz & 1;
	ph = ph - (bz >> 1); pw = pw - (bz & 1);
	//if(ihs < 0): ihs += (ph + 1)/sh*sh, //(ihs < 0) <=> (ph > 0)
	int ihs = -ph + (-(ph > 0) & ((ph + 1) << 1 >> 1));
	int iws = -pw + (-(pw > 0) & ((pw + 1) << 1 >> 1));

	//prepare for GN = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	CW += ic0 + ((ty >= STEP) << 2);//CW[y, x, 0, 0, 0, tic0]

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int IW_slice = IW >> 1, IH_IW_slice = (IH >> 1) * IW_slice;
	Ims2_n_ih_iw(j0, n0, ih0, iw0);
	int X0 = ((n0*IH + ih0)*IW + iw0)*IC + ic0;

	int tohs0 = ((ih0 + ph) >> 1) - (1 << LCFH) + 1;//tows0 = (tj0 % IW_slice)*sh + iws
	int tows0 = (j0 + ((tx >= STEP) << 2)) % IW_slice;
	int tows1 = tows0 + 1, tows2 = tows0 + 2, tows3 = tows0 + 3;
	int opw = (1 << LCFW) - 1; iws += pw;
	tows0 = (((tows0 << 1) + iws) >> 1) - opw;
	tows1 = (((tows1 << 1) + iws) >> 1) - opw;
	tows2 = (((tows2 << 1) + iws) >> 1) - opw;
	tows3 = (((tows3 << 1) + iws) >> 1) - opw;
	int Y = (n0*OH + tohs0)*OW;
	int Y0 = (Y + tows0)*OC;
	int Y1 = (Y + tows1)*OC;
	int Y2 = (Y + tows2)*OC;
	int Y3 = (Y + tows3)*OC;

	//load 4 elem from W[OC, FH, FW, IC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);//W_k ->(oc, fhr, fwr)
	Ws[buf][ty][tx] = *(float4*)(CW + W_k * IC);

	//load 4 elem from deltaY[N, FH, FW, OC]
	int dY_k = tx - ((tx >= STEP) << LB >> 1);
	Ims_oc_fhr_fwr_CW2pow(dY_k, dY_oc, dY_fhr, dY_fwr);
	int yoffset = (dY_fhr * OW + dY_fwr)*OC + dY_oc;
	zero_float(Ys[buf][tx][ty].x, Ims_ldy(tohs0, tows0), tex1Dfetch<float>(deltaY, Y0 + yoffset));
	zero_float(Ys[buf][tx][ty].y, Ims_ldy(tohs0, tows1), tex1Dfetch<float>(deltaY, Y1 + yoffset));
	zero_float(Ys[buf][tx][ty].z, Ims_ldy(tohs0, tows2), tex1Dfetch<float>(deltaY, Y2 + yoffset));
	zero_float(Ys[buf][tx][ty].w, Ims_ldy(tohs0, tows3), tex1Dfetch<float>(deltaY, Y3 + yoffset));
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
		zero_float(Ys[buf][tx][ty].x, Ims_ldy(tohs0, tows0), tex1Dfetch<float>(deltaY, Y0 + yoffset));
		zero_float(Ys[buf][tx][ty].y, Ims_ldy(tohs0, tows1), tex1Dfetch<float>(deltaY, Y1 + yoffset));
		zero_float(Ys[buf][tx][ty].z, Ims_ldy(tohs0, tows2), tex1Dfetch<float>(deltaY, Y2 + yoffset));
		zero_float(Ys[buf][tx][ty].w, Ims_ldy(tohs0, tows3), tex1Dfetch<float>(deltaY, Y3 + yoffset));
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


	int IC2 = IC << 1;//IC * sw
	int X1 = X0 + IC2, X2 = X1 + IC2, X3 = X2 + IC2;
	int X4 = X3 + IC2, X5 = X4 + IC2, X6 = X5 + IC2, X7 = X6 + IC2;

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


//when: (IH_slice, IW_slice) % 8 == 0
//(Y: BLOCK_SIZE*8, X:BLOCK_SIZE*8), K % (BLOCK_SIZE/2) == 0, CFH, CFW is power of 2
#ifndef X7_TEXTURE
#define X7_TEXTURE

#define X7_tex(stream, LB, ic_index, j_index, deltaY, OH, OW, CW, LCFH, LCFW, deltaX, LIH, LIW, IC, OC, ph, pw, GN, GM) \
	X7_texture<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>3), (GM>>LB>>3), 4), dim3(1<<LB, 1<<LB) >>>\
			(deltaY, OH, OW, CW, LCFH, LCFW, deltaX, IH, IW, IC, OC, ph, pw, ic_index, j_index)

//LB = 4: Size = 2.25, Time = 1.718 msec, Performace = 2812.48 GFlop/s
//LB = 3: Size = 2.25, Time = 1.732 msec, Performace = 2789.74 GFlop/s
//LB = 4: Size = 1.125, Time = 0.922 msec, Performace = 2620.3 GFlop/s
//LB = 3: Size = 1.125, Time = 0.956 msec, Performace = 2527.11 GFlop/s
template<int LB, int STEP>
__global__ void X7_texture(
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
	const int LCFH_CFW = LCFW + LCFW, CFH_CFW_m1 = (1 << LCFH_CFW) - 1;
	const int GK = OC << LCFH_CFW;

	//prepare for GZ = sh*sw
	int bz = blockIdx.z;
	CW += (bz << LCFH << LCFW) * OC * IC;//CW[y, x]

	//y = bz >> 1, x = bz & 1;
	ph = ph - (bz >> 1); pw = pw - (bz & 1);
	//if(ihs < 0): ihs += (ph + 1)/sh*sh, //(ihs < 0) <=> (ph > 0)
	int ihs = -ph + (-(ph > 0) & ((ph + 1) << 1 >> 1));
	int iws = -pw + (-(pw > 0) & ((pw + 1) << 1 >> 1));

	//prepare for GN = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	CW += ic0 + ((ty >= STEP) << 2);//CW[y, x, 0, 0, 0, tic0]

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int IW_slice = IW >> 1, IH_IW_slice = (IH >> 1) * IW_slice;
	Ims2_n_ih_iw(j0, n0, ih0, iw0);
	int X0 = ((n0*IH + ih0)*IW + iw0)*IC + ic0;

	int tohs0 = ((ih0 + ph) >> 1) - (1 << LCFH) + 1;//tows0 = (tj0 % IW_slice)*sh + iws
	int tows0 = (j0 + ((tx >= STEP) << 2)) % IW_slice;
	int tows1 = tows0 + 1, tows2 = tows0 + 2, tows3 = tows0 + 3;
	int opw = (1 << LCFW) - 1; iws += pw;
	tows0 = (((tows0 << 1) + iws) >> 1) - opw;
	tows1 = (((tows1 << 1) + iws) >> 1) - opw;
	tows2 = (((tows2 << 1) + iws) >> 1) - opw;
	tows3 = (((tows3 << 1) + iws) >> 1) - opw;

	//tows1 = ((tows0 + 2) * 2 + iws) / 2 - opw;
	//tows1 = (2 * tows0 + 4 + iws) / 2 - opw;
	//tows1 = (2 * tows0 + iws) / 2 - opw + 2;

	int Y = (n0*OH + tohs0)*OW;
	int Y0 = (Y + tows0)*OC;
	int Y1 = (Y + tows1)*OC;
	int Y2 = (Y + tows2)*OC;
	int Y3 = (Y + tows3)*OC;

	//load 4 elem from W[OC, FH, FW, IC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);//W_k ->(oc, fhr, fwr)
	Ws[buf][ty][tx] = *(float4*)(CW + W_k * IC);

	//load 4 elem from deltaY[N, FH, FW, OC]
	int dY_k = tx - ((tx >= STEP) << LB >> 1);
	Ims_oc_fhr_fwr_CW2pow(dY_k, dY_oc, dY_fhr, dY_fwr);
	int yoffset = (dY_fhr * OW + dY_fwr)*OC + dY_oc;
	bool ldy_oh = (tohs0 >= -dY_fhr) && (tohs0 < OH - dY_fhr);
	bool ldy0 = ldy_oh && Ims_ldy_ows(tows0);
	bool ldy1 = ldy_oh && Ims_ldy_ows(tows1);
	bool ldy2 = ldy_oh && Ims_ldy_ows(tows2);
	bool ldy3 = ldy_oh && Ims_ldy_ows(tows3);
	zero_float(Ys[buf][tx][ty].x, ldy0, tex1Dfetch<float>(deltaY, Y0 + yoffset));
	zero_float(Ys[buf][tx][ty].y, ldy1, tex1Dfetch<float>(deltaY, Y1 + yoffset));
	zero_float(Ys[buf][tx][ty].z, ldy2, tex1Dfetch<float>(deltaY, Y2 + yoffset));
	zero_float(Ys[buf][tx][ty].w, ldy3, tex1Dfetch<float>(deltaY, Y3 + yoffset));
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
		bool ldy_oh = (tohs0 >= -dY_fhr) && (tohs0 < OH - dY_fhr);
		bool ldy0 = ldy_oh && Ims_ldy_ows(tows0);
		bool ldy1 = ldy_oh && Ims_ldy_ows(tows1);
		bool ldy2 = ldy_oh && Ims_ldy_ows(tows2);
		bool ldy3 = ldy_oh && Ims_ldy_ows(tows3);
		zero_float(Ys[buf][tx][ty].x, ldy0, tex1Dfetch<float>(deltaY, Y0 + yoffset));
		zero_float(Ys[buf][tx][ty].y, ldy1, tex1Dfetch<float>(deltaY, Y1 + yoffset));
		zero_float(Ys[buf][tx][ty].z, ldy2, tex1Dfetch<float>(deltaY, Y2 + yoffset));
		zero_float(Ys[buf][tx][ty].w, ldy3, tex1Dfetch<float>(deltaY, Y3 + yoffset));
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


	int IC2 = IC << 1;//IC * sw
	int X1 = X0 + IC2, X2 = X1 + IC2, X3 = X2 + IC2;
	int X4 = X3 + IC2, X5 = X4 + IC2, X6 = X5 + IC2, X7 = X6 + IC2;

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


//when: (IH_slice, IW_slice) % 8 == 0
//(Y: BLOCK_SIZE*8, X:BLOCK_SIZE*8), K % (BLOCK_SIZE/2) == 0, CFH, CFW is power of 2
#ifndef X8_TEXTURE
#define X8_TEXTURE

#define X8_tex(stream, LB, ic_index, j_index, deltaY, OH, OW, CW, LCFH, LCFW, deltaX, LIH, LIW, IC, OC, ph, pw, GN, GM) \
	X8_texture<LB, (1<<LB>>1)>\
		<<< dim3(4, (GN>>LB>>3), (GM>>LB>>3)), dim3(1<<LB, 1<<LB) >>>\
			(deltaY, OH, OW, CW, LCFH, LCFW, deltaX, IH, IW, IC, OC, ph, pw, ic_index, j_index)

//LB = 4: Size = 2.25, Time = 1.718 msec, Performace = 2812.48 GFlop/s
//LB = 3: Size = 2.25, Time = 1.732 msec, Performace = 2789.74 GFlop/s
//LB = 4: Size = 1.125, Time = 0.922 msec, Performace = 2620.3 GFlop/s
//LB = 3: Size = 1.125, Time = 0.956 msec, Performace = 2527.11 GFlop/s
template<int LB, int STEP>
__global__ void X8_texture(
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
	const int LCFH_CFW = LCFW + LCFW, CFH_CFW_m1 = (1 << LCFH_CFW) - 1;
	const int GK = OC << LCFH_CFW;

	//prepare for GZ = sh*sw
	int bz = blockIdx.x;
	CW += (bz << LCFH << LCFW) * OC * IC;//CW[y, x]

	//y = bz >> 1, x = bz & 1;
	ph = ph - (bz >> 1); pw = pw - (bz & 1);
	//if(ihs < 0): ihs += (ph + 1)/sh*sh, //(ihs < 0) <=> (ph > 0)
	int ihs = -ph + (-(ph > 0) & ((ph + 1) << 1 >> 1));
	int iws = -pw + (-(pw > 0) & ((pw + 1) << 1 >> 1));

	//prepare for GN = IC
	int ic0 = (((blockIdx.y << LB) + tx) << 3) + ic_index;
	CW += ic0 + ((ty >= STEP) << 2);//CW[y, x, 0, 0, 0, tic0]

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = (((blockIdx.z << LB) + ty) << 3) + j_index;
	int IW_slice = IW >> 1, IH_IW_slice = (IH >> 1) * IW_slice;
	Ims2_n_ih_iw(j0, n0, ih0, iw0);
	int X0 = ((n0*IH + ih0)*IW + iw0)*IC + ic0;

	int tohs0 = ((ih0 + ph) >> 1) - (1 << LCFH) + 1;//tows0 = (tj0 % IW_slice)*sh + iws
	int Y = (n0*OH + tohs0)*OW;

	int tows0 = (j0 + ((tx >= STEP) << 2)) % IW_slice;
	int tows1 = tows0 + 1, tows2 = tows0 + 2, tows3 = tows0 + 3;
	int opw = (1 << LCFW) - 1; iws += pw;
	tows0 = (((tows0 << 1) + iws) >> 1) - opw;
	tows1 = (((tows1 << 1) + iws) >> 1) - opw;
	tows2 = (((tows2 << 1) + iws) >> 1) - opw;
	tows3 = (((tows3 << 1) + iws) >> 1) - opw;
	int Y0 = (Y + tows0)*OC;
	int Y1 = (Y + tows1)*OC;
	int Y2 = (Y + tows2)*OC;
	int Y3 = (Y + tows3)*OC;

	//load 4 elem from W[OC, FH, FW, IC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);//W_k ->(oc, fhr, fwr)
	Ws[buf][ty][tx] = *(float4*)(CW + W_k * IC);


	//oc = k / CFH_CFW; k -= oc*CFH_CFW; fhr = k / CFW; fwr = k % CFW;
	//yoffset = (fhr*OW + fwr)*OC + oc
	//yoffset = fhr*OW*OC + fwr*OC + oc
	//as: k = oc*CFH*CFW + fhr*CFW + fwr 
	//load 4 elem from deltaY[N, FH, FW, OC]
	int dY_k = tx - ((tx >= STEP) << LB >> 1);
	Ims_oc_fhr_fwr_CW2pow(dY_k, dY_oc, dY_fhr, dY_fwr);
	int yoffset = (dY_fhr * OW + dY_fwr)*OC + dY_oc;
	bool ldy_oh = (tohs0 >= -dY_fhr) && (tohs0 < OH - dY_fhr);
	bool ldy0 = ldy_oh && Ims_ldy_ows(tows0);
	bool ldy1 = ldy_oh && Ims_ldy_ows(tows1);
	bool ldy2 = ldy_oh && Ims_ldy_ows(tows2);
	bool ldy3 = ldy_oh && Ims_ldy_ows(tows3);
	zero_float(Ys[buf][tx][ty].x, ldy0, tex1Dfetch<float>(deltaY, Y0 + yoffset));
	zero_float(Ys[buf][tx][ty].y, ldy1, tex1Dfetch<float>(deltaY, Y1 + yoffset));
	zero_float(Ys[buf][tx][ty].z, ldy2, tex1Dfetch<float>(deltaY, Y2 + yoffset));
	zero_float(Ys[buf][tx][ty].w, ldy3, tex1Dfetch<float>(deltaY, Y3 + yoffset));
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
		bool ldy_oh = (tohs0 >= -dY_fhr) && (tohs0 < OH - dY_fhr);
		bool ldy0 = ldy_oh && Ims_ldy_ows(tows0);
		bool ldy1 = ldy_oh && Ims_ldy_ows(tows1);
		bool ldy2 = ldy_oh && Ims_ldy_ows(tows2);
		bool ldy3 = ldy_oh && Ims_ldy_ows(tows3);
		zero_float(Ys[buf][tx][ty].x, ldy0, tex1Dfetch<float>(deltaY, Y0 + yoffset));
		zero_float(Ys[buf][tx][ty].y, ldy1, tex1Dfetch<float>(deltaY, Y1 + yoffset));
		zero_float(Ys[buf][tx][ty].z, ldy2, tex1Dfetch<float>(deltaY, Y2 + yoffset));
		zero_float(Ys[buf][tx][ty].w, ldy3, tex1Dfetch<float>(deltaY, Y3 + yoffset));
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


	int IC2 = IC << 1;//IC * sw
	int X1 = X0 + IC2, X2 = X1 + IC2, X3 = X2 + IC2;
	int X4 = X3 + IC2, X5 = X4 + IC2, X6 = X5 + IC2, X7 = X6 + IC2;

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


//when: (IH_slice, IW_slice) % 8 == 0
//(Y: BLOCK_SIZE*8, X:BLOCK_SIZE*8), K % (BLOCK_SIZE/2) == 0, CFH, CFW is power of 2
#ifndef X9_TEXTURE
#define X9_TEXTURE

#define X9_tex(stream, LB, ic_index, j_index, deltaY, OH, OW, CW, LCFH, LCFW, deltaX, LIH, LIW, LIC, LOC, ph, pw, GN, GM) \
	X9_texture<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>3), (GM>>LB>>3), 4), dim3(1<<LB, 1<<LB) >>>\
			(deltaY, OH, OW, CW, LCFH, LCFW, deltaX, IH, IW, LIC, LOC, ph, pw, ic_index, j_index)

//LB = 4: Size = 2.25, Time = 1.718 msec, Performace = 2812.48 GFlop/s
//LB = 3: Size = 2.25, Time = 1.732 msec, Performace = 2789.74 GFlop/s
//LB = 4: Size = 1.125, Time = 0.922 msec, Performace = 2620.3 GFlop/s
//LB = 3: Size = 1.125, Time = 0.956 msec, Performace = 2527.11 GFlop/s
template<int LB, int STEP>
__global__ void X9_texture(
	cudaTextureObject_t deltaY, int OH, int OW,
	const float* __restrict__ CW, int LCFH, int LCFW,
	float* __restrict__ deltaX, int IH, int IW,
	int LIC, int LOC,
	int ph, int pw,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];

	//prepare for GK = CFH * CFW * OC
	const int LCFH_CFW = LCFW + LCFW, CFH_CFW_m1 = (1 << LCFH_CFW) - 1;
	const int GK = 1 << LOC << LCFH_CFW;

	//prepare for GZ = sh*sw
	int bz = blockIdx.z;
	CW += (bz << LCFH << LCFW << LOC << LIC);//CW[y, x]

	//y = bz >> 1, x = bz & 1;
	ph = ph - (bz >> 1); pw = pw - (bz & 1);
	//if(ihs < 0): ihs += (ph + 1)/sh*sh, //(ihs < 0) <=> (ph > 0)
	int ihs = -ph + (-(ph > 0) & ((ph + 1) << 1 >> 1));
	int iws = -pw + (-(pw > 0) & ((pw + 1) << 1 >> 1));

	//prepare for GN = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	CW += ic0 + ((ty >= STEP) << 2);//CW[y, x, 0, 0, 0, tic0]

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int IW_slice = IW >> 1, IH_IW_slice = (IH >> 1) * IW_slice;
	Ims2_n_ih_iw(j0, n0, ih0, iw0);
	int X0 = (((n0*IH + ih0)*IW + iw0) << LIC) + ic0;

	int tohs0 = ((ih0 + ph) >> 1) - (1 << LCFH) + 1;//tows0 = (tj0 % IW_slice)*sh + iws
	int tows0 = (j0 + ((tx >= STEP) << 2)) % IW_slice;
	int tows1 = tows0 + 1, tows2 = tows0 + 2, tows3 = tows0 + 3;
	int opw = (1 << LCFW) - 1; iws += pw;
	tows0 = (((tows0 << 1) + iws) >> 1) - opw;
	tows1 = (((tows1 << 1) + iws) >> 1) - opw;
	tows2 = (((tows2 << 1) + iws) >> 1) - opw;
	tows3 = (((tows3 << 1) + iws) >> 1) - opw;
	int Y = (n0*OH + tohs0)*OW;
	int Y0 = (Y + tows0) << LOC;
	int Y1 = (Y + tows1) << LOC;
	int Y2 = (Y + tows2) << LOC;
	int Y3 = (Y + tows3) << LOC;

	//load 4 elem from W[OC, FH, FW, IC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);//W_k ->(oc, fhr, fwr)
	Ws[buf][ty][tx] = *(float4*)(CW + (W_k << LIC));

	//load 4 elem from deltaY[N, FH, FW, OC]
	int dY_k = tx - ((tx >= STEP) << LB >> 1);
	Ims_oc_fhr_fwr_CW2pow(dY_k, dY_oc, dY_fhr, dY_fwr);
	bool ldy_oh = (tohs0 >= -dY_fhr) && (tohs0 < OH - dY_fhr);
	bool ldy0 = ldy_oh && Ims_ldy_ows(tows0);
	bool ldy1 = ldy_oh && Ims_ldy_ows(tows1);
	bool ldy2 = ldy_oh && Ims_ldy_ows(tows2);
	bool ldy3 = ldy_oh && Ims_ldy_ows(tows3);
	int yoffset = ((dY_fhr * OW + dY_fwr) << LOC) + dY_oc;
	zero_float(Ys[buf][tx][ty].x, ldy0, tex1Dfetch<float>(deltaY, Y0 + yoffset));
	zero_float(Ys[buf][tx][ty].y, ldy1, tex1Dfetch<float>(deltaY, Y1 + yoffset));
	zero_float(Ys[buf][tx][ty].z, ldy2, tex1Dfetch<float>(deltaY, Y2 + yoffset));
	zero_float(Ys[buf][tx][ty].w, ldy3, tex1Dfetch<float>(deltaY, Y3 + yoffset));
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
		Ws[buf][ty][tx] = *(float4*)(CW + (W_k << LIC));

		//load 4 elem from deltaY[N, FH, FW, OC]
		int dY_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		Ims_oc_fhr_fwr_CW2pow(dY_k, dY_oc, dY_fhr, dY_fwr);
		bool ldy_oh = (tohs0 >= -dY_fhr) && (tohs0 < OH - dY_fhr);
		bool ldy0 = ldy_oh && Ims_ldy_ows(tows0);
		bool ldy1 = ldy_oh && Ims_ldy_ows(tows1);
		bool ldy2 = ldy_oh && Ims_ldy_ows(tows2);
		bool ldy3 = ldy_oh && Ims_ldy_ows(tows3);
		int yoffset = ((dY_fhr * OW + dY_fwr) << LOC) + dY_oc;
		zero_float(Ys[buf][tx][ty].x, ldy0, tex1Dfetch<float>(deltaY, Y0 + yoffset));
		zero_float(Ys[buf][tx][ty].y, ldy1, tex1Dfetch<float>(deltaY, Y1 + yoffset));
		zero_float(Ys[buf][tx][ty].z, ldy2, tex1Dfetch<float>(deltaY, Y2 + yoffset));
		zero_float(Ys[buf][tx][ty].w, ldy3, tex1Dfetch<float>(deltaY, Y3 + yoffset));
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

	int IC2 = 2 << LIC;//IC * sw
	int X1 = X0 + IC2, X2 = X1 + IC2, X3 = X2 + IC2;
	

	*(float4*)(deltaX + X0) = v0;  *(float4*)(deltaX + X0 + 4) = v1;
	*(float4*)(deltaX + X1) = v2;  *(float4*)(deltaX + X1 + 4) = v3;
	*(float4*)(deltaX + X2) = v4;  *(float4*)(deltaX + X2 + 4) = v5;
	*(float4*)(deltaX + X3) = v6;  *(float4*)(deltaX + X3 + 4) = v7;

	int X4 = X3 + IC2, X5 = X4 + IC2, X6 = X5 + IC2, X7 = X6 + IC2;
	*(float4*)(deltaX + X4) = v8;  *(float4*)(deltaX + X4 + 4) = v9;
	*(float4*)(deltaX + X5) = v10; *(float4*)(deltaX + X5 + 4) = v11;
	*(float4*)(deltaX + X6) = v12; *(float4*)(deltaX + X6 + 4) = v13;
	*(float4*)(deltaX + X7) = v14; *(float4*)(deltaX + X7 + 4) = v15;
}

#endif


//(Y: BLOCK_SIZE*8, X:BLOCK_SIZE*8), K % (BLOCK_SIZE/2) == 0, CFH, CFW is power of 2
#ifndef X10_TEXTURE
#define X10_TEXTURE

#define X10_tex(stream, LB, ic_index, j_index, deltaY, OH, OW, CW, LCFH, LCFW, deltaX, LIH, LIW, IC, OC, ph, pw, GN, GM) \
	X10_texture<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>3), (GM>>LB>>3), 4), dim3(1<<LB, 1<<LB) >>>\
			(deltaY, OH, OW, CW, LCFH, LCFW, deltaX, IH, IW, IC, OC, ph, pw, ic_index, j_index)

//LB = 4: Size = 2.25, Time = 1.718 msec, Performace = 2812.48 GFlop/s
//LB = 3: Size = 2.25, Time = 1.732 msec, Performace = 2789.74 GFlop/s
//LB = 4: Size = 1.125, Time = 0.922 msec, Performace = 2620.3 GFlop/s
//LB = 3: Size = 1.125, Time = 0.956 msec, Performace = 2527.11 GFlop/s
template<int LB, int STEP>
__global__ void X10_texture(
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
	const int LCFH_CFW = LCFW + LCFW, CFH_CFW_m1 = (1 << LCFH_CFW) - 1;
	const int GK = OC << LCFH_CFW;

	//prepare for GZ = sh*sw
	int bz = blockIdx.z;
	CW += (bz << LCFH << LCFW) * OC * IC;//CW[y, x]

	//y = bz >> 1, x = bz & 1;
	ph = ph - (bz >> 1); pw = pw - (bz & 1);
	//if(ihs < 0): ihs += (ph + 1)/sh*sh, //(ihs < 0) <=> (ph > 0)
	int ihs = -ph + (-(ph > 0) & ((ph + 1) << 1 >> 1));
	int iws = -pw + (-(pw > 0) & ((pw + 1) << 1 >> 1));

	//prepare for GN = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	CW += ic0 + ((ty >= STEP) << 2);//CW[y, x, 0, 0, 0, tic0]

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int IW_slice = IW >> 1, IH_IW_slice = (IH >> 1) * IW_slice;
	Ims2_n_ih_iw(j0, n0, ih0, iw0);
	int X0 = ((n0*IH + ih0)*IW + iw0)*IC + ic0;

	int tohs0 = ((ih0 + ph) >> 1) - (1 << LCFH) + 1;//tows0 = (tj0 % IW_slice)*sh + iws
	int tows0 = (j0 + ((tx >= STEP) << 2)) % IW_slice;
	int tows1 = tows0 + 1, tows2 = tows0 + 2, tows3 = tows0 + 3;
	int opw = (1 << LCFW) - 1; iws += pw;
	tows0 = (((tows0 << 1) + iws) >> 1) - opw;
	tows1 = (((tows1 << 1) + iws) >> 1) - opw;
	tows2 = (((tows2 << 1) + iws) >> 1) - opw;
	tows3 = (((tows3 << 1) + iws) >> 1) - opw;
	int Y = (n0*OH + tohs0)*OW;
	int Y0 = (Y + tows0)*OC;
	int Y1 = (Y + tows1)*OC;
	int Y2 = (Y + tows2)*OC;
	int Y3 = (Y + tows3)*OC;

	//load 4 elem from W[OC, FH, FW, IC]
	int W_k = ty - ((ty >= STEP) << LB >> 1);//W_k ->(oc, fhr, fwr)
	Ws[buf][ty][tx] = *(float4*)(&CW[W_k * IC]);

	//load 4 elem from deltaY[N, FH, FW, OC]
	int dY_k = tx - ((tx >= STEP) << LB >> 1);
	Ims_oc_fhr_fwr_CW2pow(dY_k, dY_oc, dY_fhr, dY_fwr);
	int yoffset = (dY_fhr * OW + dY_fwr)*OC + dY_oc;
	bool ldy_oh = (tohs0 >= -dY_fhr) && (tohs0 < OH - dY_fhr);
	bool ldy0 = ldy_oh && Ims_ldy_ows(tows0);
	bool ldy1 = ldy_oh && Ims_ldy_ows(tows1);
	bool ldy2 = ldy_oh && Ims_ldy_ows(tows2);
	bool ldy3 = ldy_oh && Ims_ldy_ows(tows3);
	zero_float(Ys[buf][tx][ty].x, ldy0, tex1Dfetch<float>(deltaY, Y0 + yoffset));
	zero_float(Ys[buf][tx][ty].y, ldy1, tex1Dfetch<float>(deltaY, Y1 + yoffset));
	zero_float(Ys[buf][tx][ty].z, ldy2, tex1Dfetch<float>(deltaY, Y2 + yoffset));
	zero_float(Ys[buf][tx][ty].w, ldy3, tex1Dfetch<float>(deltaY, Y3 + yoffset));
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
		Ws[buf][ty][tx] = *(float4*)(&CW[W_k * IC]);

		//load 4 elem from deltaY[N, FH, FW, OC]
		int dY_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		Ims_oc_fhr_fwr_CW2pow(dY_k, dY_oc, dY_fhr, dY_fwr);
		int yoffset = (dY_fhr * OW + dY_fwr)*OC + dY_oc;
		bool ldy_oh = (tohs0 >= -dY_fhr) && (tohs0 < OH - dY_fhr);
		bool ldy0 = ldy_oh && Ims_ldy_ows(tows0);
		bool ldy1 = ldy_oh && Ims_ldy_ows(tows1);
		bool ldy2 = ldy_oh && Ims_ldy_ows(tows2);
		bool ldy3 = ldy_oh && Ims_ldy_ows(tows3);
		zero_float(Ys[buf][tx][ty].x, ldy0, tex1Dfetch<float>(deltaY, Y0 + yoffset));
		zero_float(Ys[buf][tx][ty].y, ldy1, tex1Dfetch<float>(deltaY, Y1 + yoffset));
		zero_float(Ys[buf][tx][ty].z, ldy2, tex1Dfetch<float>(deltaY, Y2 + yoffset));
		zero_float(Ys[buf][tx][ty].w, ldy3, tex1Dfetch<float>(deltaY, Y3 + yoffset));
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

	int IC2 = IC << 1;//IC * sw
	int X1 = X0 + IC2, X2 = X1 + IC2, X3 = X2 + IC2;
	int X4 = X3 + IC2, X5 = X4 + IC2, X6 = X5 + IC2, X7 = X6 + IC2;

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


//=======================================================
//when: (IH_slice, IW_slice) % 8 == 0
//(Y: BLOCK_SIZE*8, X:BLOCK_SIZE*8), K % (BLOCK_SIZE/2) == 0, CFH, CFW is power of 2
#ifndef XA_TEXTURE
#define XA_TEXTURE

#define Xa_tex(stream, LB, ic_index, j_index, deltaY, OH, OW, CW, LCFH, LCFW, deltaX, LIH, LIW, IC, LOC, ph, pw, GN, GM) \
	Xa_texture<LB, (1<<LB>>1)>\
		<<< dim3((GN>>LB>>3), (GM>>LB>>3), 4), dim3(1<<LB, 1<<LB) >>>\
			(deltaY, OH, OW, CW, LCFH, LCFW, deltaX, IH, IW, IC, LOC, ph, pw, ic_index, j_index)

//LB = 4: Size = 2.25, Time = 1.718 msec, Performace = 2812.48 GFlop/s
//LB = 3: Size = 2.25, Time = 1.732 msec, Performace = 2789.74 GFlop/s
//LB = 4: Size = 1.125, Time = 1.186 msec, Performace = 2037.03 GFlop/s
//LB = 3: Size = 1.125, Time = 1.042 msec, Performace = 2318.54 GFlop/s
template<int LB, int STEP>
__global__ void Xa_texture(
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
	const int CFW_OC = CFW * OC, GK = CFH * CFW_OC;

	//prepare for GZ = sh*sw
	int bz = blockIdx.z;
	CW += bz * CFH * CFW_OC * IC;//CW[y, x]

	//y = bz >> 1, x = bz & 1;
	ph = ph - (bz >> 1); pw = pw - (bz & 1);
	//if(ihs < 0): ihs += (ph + 1)/sh*sh, //(ihs < 0) <=> (ph > 0)
	int ihs = -ph + (-(ph > 0) & ((ph + 1) << 1 >> 1));
	int iws = -pw + (-(pw > 0) & ((pw + 1) << 1 >> 1));

	//prepare for GN = IC
	int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	CW += ic0 + ((ty >= STEP) << 2);//CW[y, x, 0, 0, 0, tic0]

	//prepare_for_GM = N*IH_slice*IW_slice
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int IW_slice = IW >> 1, IH_IW_slice = (IH >> 1) * IW_slice;
	Ims2_n_ih_iw(j0, n0, ih0, iw0);
	int X0 = ((n0*IH + ih0)*IW + iw0)*IC + ic0;

	int tohs0 = ((ih0 + ph) >> 1) - CFH + 1;//tows0 = (tj0 % IW_slice)*sh + iws
	int tows0 = (j0 + ((tx >= STEP) << 2)) % IW_slice;
	int tows1 = tows0 + 1, tows2 = tows0 + 2, tows3 = tows0 + 3;
	int opw = CFW - 1; iws += pw;
	tows0 = (((tows0 << 1) + iws) >> 1) - opw;
	tows1 = (((tows1 << 1) + iws) >> 1) - opw;
	tows2 = (((tows2 << 1) + iws) >> 1) - opw;
	tows3 = (((tows3 << 1) + iws) >> 1) - opw;
	int Y = (n0*OH + tohs0)*OW;
	int Y0 = (Y + tows0)*OC;
	int Y1 = (Y + tows1)*OC;
	int Y2 = (Y + tows2)*OC;
	int Y3 = (Y + tows3)*OC;

	//load 4 elem from deltaY[N, FH, FW, OC]
	int dY_k = tx - ((tx >= STEP) << LB >> 1);
	Ims_fhr_fwr(dY_k, dY_fhr, dY_fwr);
	int yoffset = (dY_fhr * OW * OC) + dY_k;
	bool ldy_oh = (tohs0 >= -dY_fhr) && (tohs0 < OH - dY_fhr);
	bool ldy0 = ldy_oh && Ims_ldy_ows(tows0);
	bool ldy1 = ldy_oh && Ims_ldy_ows(tows1);
	bool ldy2 = ldy_oh && Ims_ldy_ows(tows2);
	bool ldy3 = ldy_oh && Ims_ldy_ows(tows3);
	zero_float(Ys[buf][tx][ty].x, ldy0, tex1Dfetch<float>(deltaY, Y0 + yoffset));
	zero_float(Ys[buf][tx][ty].y, ldy1, tex1Dfetch<float>(deltaY, Y1 + yoffset));
	zero_float(Ys[buf][tx][ty].z, ldy2, tex1Dfetch<float>(deltaY, Y2 + yoffset));
	zero_float(Ys[buf][tx][ty].w, ldy3, tex1Dfetch<float>(deltaY, Y3 + yoffset));
	
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

		//load 4 elem from deltaY[N, FH, FW, OC]
		int dY_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		Ims_fhr_fwr(dY_k, dY_fhr, dY_fwr);
		int yoffset = (dY_fhr * OW * OC) + dY_k;
		bool ldy_oh = (tohs0 >= -dY_fhr) && (tohs0 < OH - dY_fhr);
		bool ldy0 = ldy_oh && Ims_ldy_ows(tows0);
		bool ldy1 = ldy_oh && Ims_ldy_ows(tows1);
		bool ldy2 = ldy_oh && Ims_ldy_ows(tows2);
		bool ldy3 = ldy_oh && Ims_ldy_ows(tows3);
		zero_float(Ys[buf][tx][ty].x, ldy0, tex1Dfetch<float>(deltaY, Y0 + yoffset));
		zero_float(Ys[buf][tx][ty].y, ldy1, tex1Dfetch<float>(deltaY, Y1 + yoffset));
		zero_float(Ys[buf][tx][ty].z, ldy2, tex1Dfetch<float>(deltaY, Y2 + yoffset));
		zero_float(Ys[buf][tx][ty].w, ldy3, tex1Dfetch<float>(deltaY, Y3 + yoffset));
		
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


	int IC2 = IC << 1;//IC * sw
	int X1 = X0 + IC2, X2 = X1 + IC2, X3 = X2 + IC2;
	int X4 = X3 + IC2, X5 = X4 + IC2, X6 = X5 + IC2, X7 = X6 + IC2;

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