

//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK >= (BLOCK_SIZE/2) == 0
//LB = 4: GK % 8 == 0
#ifndef IERNEL_V1
#define IERNEL_V1

#define iernel_v1(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM)\
	Iernel_v1<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>3, GN>>LB>>3, GZ), dim3((1<<LB), (1<<LB), 2), 0, stream >>>\
			(X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW,\
			 IC, OC, sh, sw, ph, pw, GK, GK_slice, oc_index, j_index)

//synchronized: 
//GK_slice = 1024, OH = OW =  4, N = 256:
//LB = 4: Size = 1, Time = 1.602 msec, Performace = 1340.5 GFlop/s
//LB = 3: Size = 1, Time = 2.072 msec, Performace = 1036.43 GFlop/s
//GK_slice = 1024, OH = OW = 7, IH = IW = 14
//LB = 4: Size = 0.861328, Time = 1.38 msec, Performace = 1340.35 GFlop/s
//LB = 3: Size = 0.861328, Time = 1.708 msec, Performace = 1082.96 GFlop/s
//for[64, 64] -> [32, 32]: Size = 1.125, Time = 1.782 msec, Performace = 1355.73 GFlop/s
template<int LB, int STEP>
__global__ void Iernel_v1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int IC, int OC,
	int sh, int sw, int oph, int opw,
	int GK, int GK_slice,
	int oc_index, int j_index)
{
	int tz = threadIdx.z, ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];//v0, v1
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];//v2, v3

	//======================================================================
	//prepare for GK_slice
	int bz = blockIdx.z;
	int GK_start = GK_slice * bz;
	int GK_end = IF_int((bz != (gridDim.z - 1)), (GK_start + GK_slice), GK);
	GK_slice = (GK_end - GK_start);
	int GK_slice0 = (GK_slice >> 1) >> 3 << 3;//GK_slice % 8 == 0
	int GK_slice1 = GK_slice - GK_slice0;
	int GK_start1 = GK_start + GK_slice0;
	GK_slice = IF_int(bz == 0, GK_slice0, GK_slice1);
	GK_start = IF_int(bz == 0, GK_start, GK_start1);

	//dst[bz] = deltaW_buf[bz - 1, sizeW], bz >= 1, dst[0] = deltaW
	const int FW_IC = FW * IC, Wstride = FH * FW_IC;
	deltaW_buf += (bz - 1) * OC * Wstride;
	deltaW = IF_int((bz != 0), deltaW_buf, deltaW);//deltaW -> dst
	//======================================================================

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 3) + oc_index;
	deltaY += oc0 + ((tx >= STEP) << 2);//deltaY[0, 0, 0, toc0]

	//prepare for GM = FH * FW * IC
	int j0 = (((blockIdx.x << LB) + tx) << 3) + j_index;
	int tj0 = j0 + ((ty >= STEP) << 2);
	get_fh_fw_ic(tj0, tfh0, tfw0, tic0);
	tfh0 = tfh0 - oph, tfw0 = tfw0 - opw;
	X += (tfh0*IW + tfw0)*IC + tic0;

	//load 4 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
	int OH_OW = OH * OW;
	int X_k = ty - ((ty >= STEP) << LB >> 1) + GK_start;
	int X_n, X_oh, X_ow; get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
	int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
	float4 xv; xv.x = xv.y = xv.z = xv.w = 0;
	if (LOAD_X(tfh0, tfw0)) xv = *(float4*)(X + xoffset);
	Xs[buf][ty][tx] = xv;

	//load 4 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
	int Y_k = tx - ((tx >= STEP) << LB >> 1) + GK_start;
	Ys[buf][tx][ty] = *(float4*)(deltaY + Y_k * OC);
	__syncthreads();

	//compute area-----------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK_slice << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP][ty];
			float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];

			simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
			simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
			simdMM4(v4, a0.z, b0); simdMM4(v5, a0.z, b1);
			simdMM4(v6, a0.w, b0); simdMM4(v7, a0.w, b1);
			simdMM4(v8, a1.x, b0); simdMM4(v9, a1.x, b1);
			simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
			simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
			simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
		}
		buf ^= 1;

		//load 4 elements from X[N, IH, IW, IC]: Xe[IC, IH, IW, N]
		int X_k = ((ok - (ty >= STEP)) << LB >> 1) + ty + GK_start;
		int X_n, X_oh, X_ow; get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		float4 xv; xv.x = xv.y = xv.z = xv.w = 0;
		if (LOAD_X(tfh0, tfw0)) xv = *(float4*)(X + xoffset);
		Xs[buf][ty][tx] = xv;

		//load 4 elements from deltaY[N, OH, OW, OC]: deltaYpe[OC, OHp, OWp, N]
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx + GK_start;
		Ys[buf][tx][ty] = *(float4*)(deltaY + Y_k * OC);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP][ty];
		float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];

		simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
		simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
		simdMM4(v4, a0.z, b0); simdMM4(v5, a0.z, b1);
		simdMM4(v6, a0.w, b0); simdMM4(v7, a0.w, b1);
		simdMM4(v8, a1.x, b0); simdMM4(v9, a1.x, b1);
		simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
		simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
		simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
	}
	__syncthreads();

	oc0 = oc0 * Wstride + j0;//j = (fh * FW + fw)*IC + ic
	int oc1 = oc0 + Wstride, oc2 = oc1 + Wstride;
	int oc3 = oc2 + Wstride, oc4 = oc3 + Wstride;
	int oc5 = oc4 + Wstride, oc6 = oc5 + Wstride, oc7 = oc6 + Wstride;

	if(tz == 0) {
		*(float4*)(deltaW + oc0) = v0;  *(float4*)(deltaW + oc0 + 4) = v1;
		*(float4*)(deltaW + oc1) = v2;  *(float4*)(deltaW + oc1 + 4) = v3;
		*(float4*)(deltaW + oc2) = v4;  *(float4*)(deltaW + oc2 + 4) = v5;
		*(float4*)(deltaW + oc3) = v6;  *(float4*)(deltaW + oc3 + 4) = v7;
		*(float4*)(deltaW + oc4) = v8;  *(float4*)(deltaW + oc4 + 4) = v9;
		*(float4*)(deltaW + oc5) = v10; *(float4*)(deltaW + oc5 + 4) = v11;
		*(float4*)(deltaW + oc6) = v12; *(float4*)(deltaW + oc6 + 4) = v13;
		*(float4*)(deltaW + oc7) = v14; *(float4*)(deltaW + oc7 + 4) = v15;
	}
	__syncthreads();

	if (tz == 1) {
		float4  dv0 = *(float4*)(deltaW + oc0), dv1 = *(float4*)(deltaW + oc0 + 4); simdAdd4(v0, v0, dv0); simdAdd4(v1, v1, dv1);
		float4  dv2 = *(float4*)(deltaW + oc1), dv3 = *(float4*)(deltaW + oc1 + 4); simdAdd4(v2, v2, dv2); simdAdd4(v3, v3, dv3);
		float4  dv4 = *(float4*)(deltaW + oc2), dv5 = *(float4*)(deltaW + oc2 + 4); simdAdd4(v4, v4, dv4); simdAdd4(v5, v3, dv5);
		float4  dv6 = *(float4*)(deltaW + oc3), dv7 = *(float4*)(deltaW + oc3 + 4); simdAdd4(v6, v6, dv6); simdAdd4(v7, v7, dv7);
		*(float4*)(deltaW + oc0) = v0;  *(float4*)(deltaW + oc0 + 4) = v1;
		*(float4*)(deltaW + oc1) = v2;  *(float4*)(deltaW + oc1 + 4) = v3;
		*(float4*)(deltaW + oc2) = v4;  *(float4*)(deltaW + oc2 + 4) = v5;
		*(float4*)(deltaW + oc3) = v6;  *(float4*)(deltaW + oc3 + 4) = v7;
		
		float4  dv8 = *(float4*)(deltaW + oc4),  dv9 = *(float4*)(deltaW + oc4 + 4); simdAdd4(v8, v8, dv8); simdAdd4(v9, v9, dv9);
		float4 dv10 = *(float4*)(deltaW + oc5), dv11 = *(float4*)(deltaW + oc5 + 4); simdAdd4(v10, v10, dv10); simdAdd4(v11, v11, dv11);
		float4 dv12 = *(float4*)(deltaW + oc6), dv13 = *(float4*)(deltaW + oc6 + 4); simdAdd4(v12, v12, dv12); simdAdd4(v13, v13, dv13);
		float4 dv14 = *(float4*)(deltaW + oc7), dv15 = *(float4*)(deltaW + oc7 + 4); simdAdd4(v14, v14, dv14); simdAdd4(v15, v15, dv15);
		*(float4*)(deltaW + oc4) = v8;  *(float4*)(deltaW + oc4 + 4) = v9;
		*(float4*)(deltaW + oc5) = v10; *(float4*)(deltaW + oc5 + 4) = v11;
		*(float4*)(deltaW + oc6) = v12; *(float4*)(deltaW + oc6 + 4) = v13;
		*(float4*)(deltaW + oc7) = v14; *(float4*)(deltaW + oc7 + 4) = v15;
	}
}

#endif