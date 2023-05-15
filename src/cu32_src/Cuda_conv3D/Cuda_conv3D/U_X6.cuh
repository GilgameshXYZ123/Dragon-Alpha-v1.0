

//(4*BLOCK_SIZE_Y, 4*BLOCK_SIZE_X)
#ifndef XKERNEL1
#define XKERNEL1

#define xkernel1(stream, LBY, LBX, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	SXkernel1<LBY, LBX>\
		<<< dim3(GM>>LBX>>2, GN>>LBY>>2), dim3(1<<LBX, 1<<LBY), 0, stream >>>\
			(X,IH,IW, W,FH,FW, Y,(OH*OW),OW, IC,OC, sh,sw,ph,pw,\
			oc_index, j_index)

template<int LBY, int LBX>
__global__ void SXkernel1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	//prepare for GK = FH * FW * IC
	const int FW_IC = FW * IC, GK = FH * FW_IC;

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LBY) + ty) << 2) + oc_index;
	const int toc0 = oc0 * GK;
	const int toc1 = toc0 + GK, toc2 = toc1 + GK, toc3 = toc2 + GK;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.x << LBX) + tx) << 2) + j_index;
	int j1 = j0 + 1, j2 = j0 + 2, j3 = j0 + 3;
	get_n_oh_ow(j0, n0, oh0, ow0);
	get_n_oh_ow(j1, n1, oh1, ow1);
	get_n_oh_ow(j2, n2, oh2, ow2);
	get_n_oh_ow(j3, n3, oh3, ow3);
	const int toh0 = oh0 * sh - ph, tow0 = ow0 * sw - pw;
	const int toh1 = oh1 * sh - ph, tow1 = ow1 * sw - pw;
	const int toh2 = oh2 * sh - ph, tow2 = ow2 * sw - pw;
	const int toh3 = oh3 * sh - ph, tow3 = ow3 * sw - pw;
	const int X0 = ((n0*IH + toh0)*IW + tow0)*IC;
	const int X1 = ((n1*IH + toh1)*IW + tow1)*IC;
	const int X2 = ((n2*IH + toh2)*IW + tow2)*IC;
	const int X3 = ((n3*IH + toh3)*IW + tow3)*IC;
	const int IW_IC = IW * IC;

	//compute area-----------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0);
	float4 v1 = make_float4(0, 0, 0, 0);
	float4 v2 = make_float4(0, 0, 0, 0);
	float4 v3 = make_float4(0, 0, 0, 0);
	for (int k = 0; k < GK; k++) 
	{
		float4 a;//load 4 elem from W
		a.x = W[toc0 + k];
		a.y = W[toc1 + k];
		a.z = W[toc2 + k];
		a.w = W[toc3 + k];

		//load 4 elem from X
		int X_k = k, X_fh, X_fw; get_X_fh_fw(X_k, X_fh, X_fw);
		int xoffset = X_fh * IW_IC + X_k;
		float4 b = SaveX4(X, X_fh, X_fw, IH, IW, xoffset,
			X0, toh0, tow0,
			X1, toh1, tow1,
			X2, toh2, tow2,
			X3, toh3, tow3);

		simdMM4(v0, b.x, a);
		simdMM4(v1, b.y, a);
		simdMM4(v2, b.z, a);
		simdMM4(v3, b.w, a);
	}

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;

	*(float4*)(Y + j0) = v0;
	*(float4*)(Y + j1) = v1;
	*(float4*)(Y + j2) = v2;
	*(float4*)(Y + j3) = v3;
}

#endif


//(4*BLOCK_SIZE_Y, 4*BLOCK_SIZE_X)
#ifndef XKERNEL2
#define XKERNEL2

#define xkernel2(stream, LBY, LBX, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	SXkernel2<LBY, LBX>\
		<<< dim3(GN>>LBY>>2, GM>>LBX>>2), dim3(1<<LBY, 1<<LBX), 0, stream >>>\
			(X,IH,IW, W,FH,FW, Y,(OH*OW),OW, IC,OC, sh,sw,ph,pw,\
			oc_index, j_index)

//Size = 0.28125, Time = 19.5973 msec, Performace = 30.8195 GFlop/s
template<int LBY, int LBX>
__global__ void SXkernel2(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	//prepare for GK = FH * FW * IC
	const int FW_IC = FW * IC, GK = FH * FW_IC;

	//prepare for GN = OC
	int oc0 = (((blockIdx.x << LBY) + tx) << 2) + oc_index;
	const int toc0 = oc0 * GK;
	const int toc1 = toc0 + GK, toc2 = toc1 + GK, toc3 = toc2 + GK;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LBX) + ty) << 2) + j_index;
	int j1 = j0 + 1, j2 = j0 + 2, j3 = j0 + 3;
	get_n_oh_ow(j0, n0, oh0, ow0);
	get_n_oh_ow(j1, n1, oh1, ow1);
	get_n_oh_ow(j2, n2, oh2, ow2);
	get_n_oh_ow(j3, n3, oh3, ow3);
	const int toh0 = oh0 * sh - ph, tow0 = ow0 * sw - pw;
	const int toh1 = oh1 * sh - ph, tow1 = ow1 * sw - pw;
	const int toh2 = oh2 * sh - ph, tow2 = ow2 * sw - pw;
	const int toh3 = oh3 * sh - ph, tow3 = ow3 * sw - pw;
	const int X0 = ((n0*IH + toh0)*IW + tow0)*IC;
	const int X1 = ((n1*IH + toh1)*IW + tow1)*IC;
	const int X2 = ((n2*IH + toh2)*IW + tow2)*IC;
	const int X3 = ((n3*IH + toh3)*IW + tow3)*IC;
	const int IW_IC = IW * IC;

	//compute area-----------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0);
	float4 v1 = make_float4(0, 0, 0, 0);
	float4 v2 = make_float4(0, 0, 0, 0);
	float4 v3 = make_float4(0, 0, 0, 0);
	for (int k = 0; k < GK; k++)
	{
		float4 a;//load 4 elem from W
		a.x = W[toc0 + k];
		a.y = W[toc1 + k];
		a.z = W[toc2 + k];
		a.w = W[toc3 + k];

		//load 4 elem from X
		int X_k = k, X_fh, X_fw; get_X_fh_fw(X_k, X_fh, X_fw);
		int xoffset = X_fh * IW_IC + X_k;
		float4 b = SaveX4(X, X_fh, X_fw, IH, IW, xoffset,
			X0, toh0, tow0,
			X1, toh1, tow1,
			X2, toh2, tow2,
			X3, toh3, tow3);

		simdMM4(v0, b.x, a);
		simdMM4(v1, b.y, a);
		simdMM4(v2, b.z, a);
		simdMM4(v3, b.w, a);
	}

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;

	*(float4*)(Y + j0) = v0;
	*(float4*)(Y + j1) = v1;
	*(float4*)(Y + j2) = v2;
	*(float4*)(Y + j3) = v3;
}

#endif


//(4*BLOCK_SIZE_Y, 4*BLOCK_SIZE_X)
#ifndef XKERNEL3
#define XKERNEL3

#define xkernel3(stream, LBY, LBX, oc_index, j_index, X, IH, IW, W, FH, LFW, Y, OH, OW, LIC, OC, sh, sw, ph, pw, GN, GM) \
	SXkernel3<LBY, LBX>\
		<<< dim3(GM>>LBX>>2, GN>>LBY>>2), dim3(1<<LBX, 1<<LBY), 0, stream >>>\
			(X,IH,IW, W,FH,LFW, Y,(OH*OW),OW, LIC,OC, sh,sw,ph,pw,\
			oc_index, j_index)

template<int LBY, int LBX>
__global__ void SXkernel3(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W, int FH, int LFW,
	float* __restrict__ Y, int OH_OW, int OW,
	int LIC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	//prepare for GK = FH * FW * IC
	const int LFW_IC = LFW + LIC, LFW_IC_m1 = (1 << LFW_IC) - 1;
	const int GK = FH << LFW_IC;

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LBY) + ty) << 2) + oc_index;
	const int toc0 = oc0 * GK;
	const int toc1 = toc0 + GK, toc2 = toc1 + GK, toc3 = toc2 + GK;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.x << LBX) + tx) << 2) + j_index;
	int j1 = j0 + 1, j2 = j0 + 2, j3 = j0 + 3;
	get_n_oh_ow(j0, n0, oh0, ow0);
	get_n_oh_ow(j1, n1, oh1, ow1);
	get_n_oh_ow(j2, n2, oh2, ow2);
	get_n_oh_ow(j3, n3, oh3, ow3);
	const int toh0 = oh0 * sh - ph, tow0 = ow0 * sw - pw;
	const int toh1 = oh1 * sh - ph, tow1 = ow1 * sw - pw;
	const int toh2 = oh2 * sh - ph, tow2 = ow2 * sw - pw;
	const int toh3 = oh3 * sh - ph, tow3 = ow3 * sw - pw;
	const int X0 = ((n0*IH + toh0)*IW + tow0) << LIC;
	const int X1 = ((n1*IH + toh1)*IW + tow1) << LIC;
	const int X2 = ((n2*IH + toh2)*IW + tow2) << LIC;
	const int X3 = ((n3*IH + toh3)*IW + tow3) << LIC;

	//compute area-----------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0);
	float4 v1 = make_float4(0, 0, 0, 0);
	float4 v2 = make_float4(0, 0, 0, 0);
	float4 v3 = make_float4(0, 0, 0, 0);
	for (int k = 0; k < GK; k++)
	{
		float4 a;//load 4 elem from W
		a.x = W[toc0 + k];
		a.y = W[toc1 + k];
		a.z = W[toc2 + k];
		a.w = W[toc3 + k];

		//load 4 elem from X
		int X_k = k, X_fh, X_fw; get_X_fh_fw_FW_IC2pow(X_k, X_fh, X_fw);
		int xoffset = (X_fh << LIC)*IW + X_k;
		float4 b = SaveX4(X, X_fh, X_fw, IH, IW, xoffset,
			X0, toh0, tow0,
			X1, toh1, tow1,
			X2, toh2, tow2,
			X3, toh3, tow3);

		simdMM4(v0, b.x, a);
		simdMM4(v1, b.y, a);
		simdMM4(v2, b.z, a);
		simdMM4(v3, b.w, a);
	}

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;

	*(float4*)(Y + j0) = v0;
	*(float4*)(Y + j1) = v1;
	*(float4*)(Y + j2) = v2;
	*(float4*)(Y + j3) = v3;
}

#endif


