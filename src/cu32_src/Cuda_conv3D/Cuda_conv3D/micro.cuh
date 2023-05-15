#pragma once

#ifndef MICRO_H
#define MICRO_H

__device__ __constant__ float _ZERO = 0;

#ifndef CONV3D_IDX
#define CONV3D_IDX

//GK(fh, fw, ic) -> GK(index = [fh, fw], ic)
//(fh, fw) -> Idx = (fh*STEP + fw)

//for: 3 * 3 Filter, STEP = 4
//(0, 0) -> 0; (0, 1) -> 1; (0, 2) ->  2
//(1, 0) -> 4; (1, 1) -> 5; (1, 2) ->  6
//(2, 0) -> 8; (2, 1) -> 9; (2, 2) -> 10
__device__ __constant__ char XIDX_W3[] = {
	0, 1,  2,  
	4, 5,  6,  
	8, 9, 10}; 

__device__ __constant__ char XIDX_V2_W3P1[] = {//stride = 9
	0, 1, 4, 5, 0, 0, 0, 0,  0,//tFH = 2, tFW = 2, fhw_idx = (0, 0)
	0, 1, 2, 4, 5, 6, 0, 0,  0,//tFH = 2, tFW = 3, fhw_idx = (0, 1)
	0, 1, 4, 5, 8, 9, 0, 0,  0,//tFH = 3, tFW = 2, fhw_idx = (1, 0)
	0, 1, 2, 4, 5, 6, 8, 9, 10,//tFH = 3, tFW = 3, fhw_idx = (1, 1)
};

//for: 4 * 4 Filter, STEP = 4
//(0, 0) ->  0; (0, 1) ->  1; (0, 2) ->  2; (0, 3) ->  3
//(1, 0) ->  4; (1, 1) ->  5; (1, 2) ->  6; (1, 3) ->  7
//(2, 0) ->  8; (2, 1) ->  9; (2, 2) -> 10; (2, 3) -> 11
//(3, 0) -> 12; (3, 1) -> 13; (3, 2) -> 14; (3, 3) -> 15
__device__ __constant__ char XIDX_V2_W4P1[] = {//stride = 16
	0, 1, 2, 4, 5, 6, 8, 9, 10,  0,  0,  0,  0,  0,  0,  0,//tFH = 3, tFW = 3, fhw_idx = (0, 0)
	0, 1, 2, 3, 4, 5, 6, 7,  8,  9, 10, 11,  0,  0,  0,  0,//tFH = 3, tFW = 4, fhw_idx = (0, 1)
	0, 1, 2, 4, 5, 6, 8, 8, 10, 12, 13, 14,  0,  0,  0,  0,//tFH = 4, tFW = 3, fhw_idx = (1, 0)
	0, 1, 2, 3, 4, 5, 6, 7,  8,  9, 10, 11, 12, 13, 14, 15 //tFH = 3, tFW = 3, fhw_idx = (1, 1)
};

//for 5*5 Filter, STEP = 8
//(0, 0) ->  0; (0, 1) ->  1; (0, 2)->   2; (0, 3) ->  3; (0, 4) ->  4
//(1, 0) ->  8; (1, 1) ->  9; (1, 2) -> 10; (1, 3) -> 11; (1, 4) -> 12
//(2, 0) -> 16; (2, 1) -> 17; (2, 2) -> 18; (2, 3) -> 19; (2, 4) -> 20
//(3, 0) -> 24; (3, 1) -> 25; (3, 2) -> 26; (3, 3) -> 27; (3, 4) -> 28
//(4, 0) -> 32; (4, 1) -> 33; (4, 2) -> 34; (4, 3) -> 35; (4, 4) -> 36
__device__ __constant__ char XIDX_W5[] = {
	 0,  1,  2,  3,  4,
	 8,  9, 10, 11, 12,
	16, 17, 18, 19, 20,
	24, 25, 26, 27, 28,
	32, 33, 34, 35, 36 };

__device__ __constant__ char XIDX_V2_W5P2[] = {//stride = 25
	0, 1, 2, 8, 9, 10, 16, 17, 18,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,//tFH = 3, tFW = 3, fhw_idx = (0, 0)
	0, 1, 2, 3, 8,  9, 10, 11, 16, 17, 18, 19,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,//tFH = 3, tFW = 4, fhw_idx = (0, 1)
	0, 1, 2, 3, 4,  8,  9, 10, 11, 12, 16, 17, 18, 19, 20,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,//tFH = 3, tFW = 5, fhw_idx = (0, 2)
	0, 1, 2, 8, 9, 10, 16, 17, 18, 24, 25, 26,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,//tFH = 4, tFW = 3, fhw_idx = (1, 0)
	0, 1, 2, 3, 8,  9, 10, 11, 16, 17, 18, 19, 24 ,25, 26, 27,  0,  0,  0,  0,  0,  0,  0,  0,  0,//tFH = 4, tFW = 4, fhw_idx = (1, 1)
	0, 1, 2, 3, 4,  8,  9, 10, 11, 12, 16, 17, 18, 18, 20, 24, 25, 26, 27, 28,  0,  0,  0,  0,  0,//tFH = 4, tFW = 5, fhw_idx = (1, 2)
	0, 1, 2, 8, 9, 10, 16, 17, 18, 24, 25, 26, 32, 33, 34,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,//tFH = 5, tFW = 3, fhw_idx = (2, 0)
	0, 1, 2, 3, 8,  9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27, 32, 33, 34, 35,  0,  0,  0,  0,  0,//tFH = 5, tFW = 4, fhw_idx = (2, 1)
	0, 1, 2, 3, 4,  8,  9, 10, 11, 12, 16, 17, 18, 19, 20, 24, 25, 26, 27, 28, 32, 33, 34, 35, 36,//tFH = 5, tFW = 5, fhw_idx = (2, 2)
};

#endif

int LOG2(int n) {
	int result = 0;
	if (n & 0xffff0000) { result += 16; n >>= 16; }
	if (n & 0x0000ff00) { result += 8; n >>= 8; }
	if (n & 0x000000f0) { result += 4; n >>= 4; }
	if (n & 0x0000000c) { result += 2; n >>= 2; }
	if (n & 0x00000002) { result += 1; n >>= 1; }
	return result;
}

#ifndef COMMON_MICRO
#define COMMON_MICRO

#define F32_2_0 float2{ 0, 0 }
#define F32_4_0 float4{ 0, 0, 0, 0 }

#define IS_POWER2(x) ( ((x)>0) && ((x)&((x)-1))==0 )

#define get2d(A, y, x, stride)   A[(y)*(stride) + (x)]
#define lget2d(A, y, x, lstride) A[((y)<<(lstride)) + (x)] //lstride = log2(stride)

#define get3d(A, z, y, x, Sy, Sx)    A[(((z)*(Sy)) + (y))*(Sx) + (x)]
#define lget3d(A, z, y, x, Sy, LSx)  A[((((z)*(Sy)) + y)<<(LSx)) + (x)]

#define get4d(A, w, z, y, x, Sz, Sy, Sx)    A[(((w)*(Sz) + (z))*(Sy) + (y))*(Sx) + (x)]
#define get4d_s(A, w, z, y, Sz, Sy, Sx)  A[(((w)*(Sz) + (z))*(Sy) + (y))*(Sx)]
#define lget4d(A, w, z, y, x, Sz, LSy, LSx) A[(((((w)*(Sz) + (z))<<(LSy)) + (y))<<(LSx)) + (x)]

//if: flag == 1: -flag = -1 = 0xffffffff
//if: flag == 0: -flag =  0 
#define IF_int(flag, a, b) ( ((-flag) & (a - b)) + b)

//X = flag*v
#define zero_float(X, flag, v) \
	{float fv = v; int iv = -(flag) & *(int*)(&fv); X = *(float*)(&iv); }

#define zero_float4(X, flag) \
	{if (!flag) X.x = X.y = X.z = X.w = 0;}

#endif


#ifndef GEMM_MICRO
#define GEMM_MICRO

#define LOAD_X(ihs, iws, fh, fw) \
	((ihs >= -fh) && (ihs < IH - fh) && (iws >= -fw) && (iws < IW - fw))

#define load4d(V, A, w, z, y, x, Sz, Sy, Sx) \
	{(V) = ((((z)<0)||((z)>=Sz)||((y)<0)||((y)>= Sy))? 0.0f: A[(((w)*(Sz) + (z))*(Sy) + (y))*(Sx) + (x)]);}

#define load4d_tex(V, A, w, z, y, x, Sz, Sy, Sx) \
	{(V) = ((((z)<0)||((z)>=Sz)||((y)<0)||((y)>= Sy))? 0.0f: tex1Dfetch<float>(A, (((w)*(Sz) + (z))*(Sy) + (y))*(Sx) + (x)));}

#define load4d_IC2pow(V, A, w, z, y, x, Sz, Sy, LSx) \
	{(V) = ((z<0 || z>=Sz || y<0 || y>=Sy)? 0.0f: A[(((w*Sz + z)*Sy + y) << LSx) + x]);}

#define load4d_check(V, ih, iw, value) \
	{if (((ih) < 0)||((ih) >= IH) || ((iw) < 0)||((iw) >= IW)) (V) = 0.0f; else (V) = (value); }


#define simdMM4(c, av, b) {(c).x += (av) * (b).x; (c).y += (av) * (b).y; (c).z += (av) * (b).z; (c).w += (av) * (b).w;}
#define simdMM2(c, av, b) {(c).x += (av) * (b).x; (c).y += (av) * (b).y;} 
#define simdAdd4(c, av, b) {(c).x = (av) + (b).x; (c).y = (av) + (b).y; (c).z = (av) + (b).z; (c).w = (av) + (b).w;}


#define GET_GN(OC) (OC)
#define GET_GM(N, OH, OW) ((N)*(OH)*(OW))
#define GET_GK(FH, FW, IC) ((FH)*(FW)*(IC))

#define GET_OUT_DIM(inDim, kernelDim, padding, step) \
	(((inDim) + 2 * (padding) - (kernelDim)) / (step) + 1)


//prepare for GM = N*OH*OW
//=============Improvement of X[n, oh, ow, ic]=====================================
//j0 % 8 == 0
//(1) ni = ji / (OH * OW) = (j0 + i) / (16x), so: ni = nj
//(2) ihi = ((j0 + i)%(OH*OW)) / OW = (8*y + i)%(16*x) / 4*x
//So: ih0 = ih1 = ih2 = ih3, ih4 = ih5 = ih6 = ih7
//So: toh0 = toh1 = toh2 = toh3
//(3) iwi = (j0 + i)%OW = (8*x + i)%(4*y) 
//So: iw0 = iw1 - 1 = iw2 - 2 = iw3 - 3
//So: iw4 = iw5 - 1 = iw6 - 2 = iw7 - 3
//So: tow0 = tow1 - 1 = tow2 - 1 = tow3 - 1
//=============Improvement of X[n, oh, ow, ic]=====================================
#define get_n_oh_ow(j, n, oh, ow) \
	int n, oh, ow; {n = j / OH_OW; int jr = j - n * OH_OW; oh = jr / OW, ow = jr - oh * OW;}

#define get_oh_ow_n(j, oh, ow, n) \
	int oh, ow, n; {oh = j / OW_N; int jr = j - oh * OW_N; ow = jr / N, n = jr - ow * N; }



//compute for GK = FH*FW*IC
//=============Improvement of X_k\W_k[fh, fw, ic]==================================
//[1] in k88
//X_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
//W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
//X_k = ok*STEP + tx - ((tx >= STEP) << LB >> 1)
//W_k = ok*STEP + ty - ((ty >= STEP) << LB >> 1)
//Let: Ux = tx - ((tx >= STEP) << LB >> 1)
//Let: Uy = ty - ((ty >= STEP) << LB >> 1)
//X_k = ok*STEP + Ux
//W_k = ok*STEP + Uy
//[1.1] when LB = 4, IC % 8 == 0, we have: (tFH*IC) % 8 == 0, Ux, Uy belongs to [0, 7], STEP = 8
//X_fh = (ok*8*x + Ux) / tFW_IC = (ok*8*x + Ux)/8y
//W_fh = (ok*8*x + Uy) / tFW_IC = (ok*8*x + Ux)/8y
//So: X_fh = W_fh, when IC % 8 == 0
//[1.2] when LB = 3, IC % 4 == 0, we have: (tFH*IC) % 4 == 0, Ux, Uy belongs to [0, 3], STEP = 4
//X_fh = (ok*4*x + Ux) / tFW_IC = (ok*4*x + Ux)/4y
//W_fh = (ok*4*x + Uy) / tFW_IC = (ok*4*x + Ux)/4y
//So: X_fh = W_fh, when IC % 8 == 0
//
//[2] in k44
//X_k = ((ok << LB) + ty) >> 1 = ok*STEP + (ty >> 1)
//W_k = ((ok << LB) + tx) >> 1 = ok*STEP + (tx >> 1)
//So: when LB = 4\3, IC % [8\4] == 0, we have: X_fh = W_fh
//
//[3] in k84:
//W_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
//X_k = ((ok << LB) + ty) >> 1 = ok*STEP + (ty >> 1)
//So: when LB = 4\3, IC % [8\4] == 0, we have: X_fh = W_fh
//=============Improvement of X_k\W_k[fh, fw, ic]==================================
#define get_X_fh_fw_ic(k, fh, fw, ic) {\
	fh = k / FW_IC; k -= fh * FW_IC;\
	fw = k / IC; ic = k - fw * IC;}

//when IC is power of 2
#define get_X_fh_fw_ic_IC2pow(k, fh, fw, ic) {\
	fh = k / FW_IC; k -= fh * FW_IC;\
	fw = k >> LIC; ic = k & IC_m1;}

//when FW, IC is power of 2
#define get_X_fh_fw_ic_FW_IC2pow(k, fh, fw, ic) {\
	fh = k >> LFW_IC; k &= LFW_IC_m1;\
	fw = k >> LIC; ic = k & IC_m1;}


//======[for GK ordered by: <fh, fw, ic>]==================================================
#define get_X_fh_fw(k, fh, fw) {fh = k / FW_IC; k -= fh * FW_IC; fw = k / IC;}

#define get_X_fh_fw_IC2pow(k, fh, fw) {fh = k / FW_IC; k -= fh * FW_IC; fw = k >> LIC; }

#define get_X_fh_fw_FW_IC2pow(k, fh, fw) { fh = k >> LFW_IC; k &= LFW_IC_m1; fw = k >> LIC; }
//======[for GK ordered by: <fh, fw, ic>]==================================================


//======[for GK ordered by: <ic, fh, fw>]==================================================
#define get_ic_fh_fw(k, ic, fh, fw) {\
	ic = k / FH_FW; k -= ic * FH_FW;\
	fh = k / FW; fw = k - fh * FW; }

#define get_ic_fh_fw_W2pow(k, ic, fh, fw) {\
	ic = k >> LFH_FW; k &= FH_FW_m1;\
	fh = k >> LFW; fw = k & FW_m1; }
//======[for GK ordered by: <ic, fh, fw>]==================================================


#define shift_n_j(n, j) n *= OH_OW * OC; j *= OC;

#define shift_n_j_2pow(n, j) n = (n << LOH_OW) * OC; j *= OC;


#ifndef SAVE_X4
#define SAVE_X4

__device__ __forceinline__ float4 SaveX4(const float* __restrict__ X,
	int X_fh, int X_fw, int IH, int IW, int xoffset,
	int X0, int toh0, int tow0,
	int X1, int toh1, int tow1,
	int X2, int toh2, int tow2,
	int X3, int toh3, int tow3)
{
	IH -= X_fh; IW -= X_fw;
	bool lx0 = (toh0 >= -X_fh) && (toh0 < IH) && (tow0 >= -X_fw) && (tow0 < IW);
	bool lx1 = (toh1 >= -X_fh) && (toh1 < IH) && (tow1 >= -X_fw) && (tow1 < IW);
	bool lx2 = (toh2 >= -X_fh) && (toh2 < IH) && (tow2 >= -X_fw) && (tow2 < IW);
	bool lx3 = (toh3 >= -X_fh) && (toh3 < IH) && (tow3 >= -X_fw) && (tow3 < IW);
	
	float4 x;
	x.x = (lx0 ? X[X0 + xoffset] : 0);
	x.y = (lx1 ? X[X1 + xoffset] : 0);
	x.z = (lx2 ? X[X2 + xoffset] : 0);
	x.w = (lx3 ? X[X3 + xoffset] : 0);
	return x;
}

__device__ __forceinline__ float4 SaveX4x(const float* __restrict__ X,
	int X_fh, int X_fw, int IH, int IW, int xoffset, int sw_IC,
	int toh0, int tow0, int tow1, int tow2, int tow3)
{
	bool lx = (toh0 >= -X_fh) && (toh0 < IH - X_fh); IW -= X_fw;
	bool lx0 = lx && (tow0 >= -X_fw) && (tow0 < IW);
	bool lx1 = lx && (tow1 >= -X_fw) && (tow1 < IW);
	bool lx2 = lx && (tow2 >= -X_fw) && (tow2 < IW);
	bool lx3 = lx && (tow3 >= -X_fw) && (tow3 < IW);

	float4 x;
	x.x = (lx0 ? X[xoffset - sw_IC] : 0);
	x.y = (lx1 ? X[xoffset] : 0);
	x.z = (lx2 ? X[xoffset + sw_IC] : 0);
	x.w = (lx3 ? X[xoffset + (sw_IC << 1)] : 0);
	return x;
}

#endif


#ifndef SAVE_X4_TEXTURE
#define SAVE_X4_TEXTURE

__device__ __forceinline__ float4 SaveX4_tex(cudaTextureObject_t X,
	int X_fh, int X_fw, int IH, int IW, int xoffset,
	int X0, int toh0, int tow0,
	int X1, int toh1, int tow1,
	int X2, int toh2, int tow2,
	int X3, int toh3, int tow3)
{
	float4 x;
	x.x = tex1Dfetch<float>(X, X0 + xoffset);
	x.y = tex1Dfetch<float>(X, X1 + xoffset);
	x.z = tex1Dfetch<float>(X, X2 + xoffset);
	x.w = tex1Dfetch<float>(X, X3 + xoffset);

	IH -= X_fh; IW -= X_fw;
	bool lx0 = (toh0 >= -X_fh) && (toh0 < IH) && (tow0 >= -X_fw) && (tow0 < IW);
	bool lx1 = (toh1 >= -X_fh) && (toh1 < IH) && (tow1 >= -X_fw) && (tow1 < IW);
	bool lx2 = (toh2 >= -X_fh) && (toh2 < IH) && (tow2 >= -X_fw) && (tow2 < IW);
	bool lx3 = (toh3 >= -X_fh) && (toh3 < IH) && (tow3 >= -X_fw) && (tow3 < IW);
	zero_float(x.x, lx0, x.x);
	zero_float(x.y, lx1, x.y);
	zero_float(x.z, lx2, x.z);
	zero_float(x.w, lx3, x.w);
	return x;
}

__device__ __forceinline__ float4 SaveX4x_tex(cudaTextureObject_t X,
	int X_fh, int X_fw, int IH, int IW, int xoffset, int sw_IC,
	int toh0, int tow0, int tow1, int tow2, int tow3)
{
	float4 x;
	x.x = tex1Dfetch<float>(X, xoffset - sw_IC);
	x.y = tex1Dfetch<float>(X, xoffset);
	x.z = tex1Dfetch<float>(X, xoffset + sw_IC);
	x.w = tex1Dfetch<float>(X, xoffset + (sw_IC << 1));

	bool lx = (toh0 >= -X_fh) && (toh0 < IH - X_fh); IW -= X_fw;
	bool lx0 = lx && (tow0 >= -X_fw) && (tow0 < IW);
	bool lx1 = lx && (tow1 >= -X_fw) && (tow1 < IW);
	bool lx2 = lx && (tow2 >= -X_fw) && (tow2 < IW);
	bool lx3 = lx && (tow3 >= -X_fw) && (tow3 < IW);
	zero_float(x.x, lx0, x.x);
	zero_float(x.y, lx1, x.y);
	zero_float(x.z, lx2, x.z);
	zero_float(x.w, lx3, x.w);
	return x;
}

#endif


#ifndef SAVE_X2
#define SAVE_X2

__device__ __forceinline__ float2 SaveX2(const float* __restrict__ X,
	int X_fh, int X_fw, int IH, int IW, int xoffset,
	int X0, int toh0, int tow0,
	int X1, int toh1, int tow1)
{
	bool lx0 = (toh0 >= -X_fh) && (toh0 < IH - X_fh) && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
	bool lx1 = (toh1 >= -X_fh) && (toh1 < IH - X_fh) && (tow1 >= -X_fw) && (tow1 < IW - X_fw);

	float2 x;
	x.x = (lx0 ? X[X0 + xoffset] : 0);
	x.y = (lx1 ? X[X1 + xoffset] : 0);
	return x;
}

#endif


#endif


#ifndef GEMMV2_MICRO
#define GEMMV2_MICRO

#define CAN_V2_W3P1 ((FH == 3) && (FW == 3) && (ph == 1) && (pw == 1) && (IH > 1) && (IW > 1))
#define CAN_V2_W4P1 ((FH == 4) && (FW == 4) && (ph == 1) && (pw == 1) && (IH > 2) && (IW > 2))
#define CAN_V2_W5P2 ((FH == 5) && (FW == 5) && (ph == 2) && (pw == 2) && (IH > 2) && (IW > 2))

//[1] Ph = 2ph = (OH - 1)*sh - IH + FH;
//[2] Q = ((Ph + IH) * (Pw + IW)) / (IH * IW)
//[3] Ph + IH = (OH - 1)*sh + FH
//    Pw + IW = (OW - 1)*sw + FW
//	  Q = 1.0 * (((OH - 1)*sh + FH) * ((OW - 1)*sw + FW)) / (IH * IW)
#define PADDING_SCALE_UP(IH, IW, OH, OW, FH, FW, sh, sw) \
	(1.0 * (((OH - 1)*sh + FH) * ((OW - 1)*sw + FW)) / (IH * IW))

//V1.GMr = V2.Nr * OH * OW
//As: GM = V1.GMr + V1.j_index = (n_index + V2.Nr) * OH * OW = N * OH * OW
#define V2_TO_V1(FH, FW, OH, OW, N, IC, n_index)\
		const int OH_OW = OH * OW;\
		int GM = N * OH_OW, GK = FH * FW * IC;\
		int j_index = n_index * OH_OW;

#endif


#ifndef WINO_GRAD_MICRO
#define WINO_GRAD_MICRO

#define winograd_g(g0, g1, g2) float4{ g0, 0.5f*(g0 + g1 + g2), 0.5f*(g0 - g1 + g2), g2 }
#define winograd_d(d0, d1, d2, d3) float4{ d0 - d2, d1 + d2, d2 - d1, d1 - d3 }

//G.x = g0
//G.y = 0.5f*(g0 + g1 + g2)
//G.z = 0.5f*(g0 - g1 + g2)
//G.w = g2
#define WinoGrad_produce_G(G, g0, g1, g2) {\
	float v2 = 0.5f*(g0 + g1 + g2);\
	float v3 = v2 - g1;\
	G = float4{g0, v2, v3, g2}; }

#define winograd_add(v, g, d) {\
	float m1 = g.x * d.x;\
	float m2 = g.y * d.y;\
	float m3 = g.z * d.z;\
	float m4 = g.w * d.w;\
	v.x += (m1 + m2 + m3);\
	v.y += (m2 - m3 - m4); }\

#define float4_elem_mul(a, b) float4{ a.x*b.x, a.y*b.y, a.z*b.z, a.w*b.w }

#define wino_grad4_GxW(v0, v1, g0, g1, g2, g3, d) {\
	float4 m00 = float4_elem_mul(g0, d);\
	float4 m10 = float4_elem_mul(g1, d);\
	float4 m20 = float4_elem_mul(g2, d);\
	float4 m30 = float4_elem_mul(g3, d);\
	v0.x += (m00.x + m00.y + m00.z);\
	v0.y += (m10.x + m10.y + m10.z);\
	v0.z += (m20.x + m20.y + m20.z);\
	v0.w += (m30.x + m30.y + m30.z);\
	v1.x += (m00.y - m00.z - m00.w);\
	v1.y += (m10.y - m10.z - m10.w);\
	v1.z += (m20.y - m20.z - m20.w);\
	v1.w += (m30.y - m30.z - m30.w);}

#define wino_grad4_WxG(v0, v1, g0, g1, g2, g3, d) {\
	float4 m00 = float4_elem_mul(g0, d);\
	float4 m10 = float4_elem_mul(g1, d);\
	float4 m20 = float4_elem_mul(g2, d);\
	float4 m30 = float4_elem_mul(g3, d);\
	v0.x += (m00.x + m00.y + m00.z);\
	v1.x += (m00.y - m00.z - m00.w);\
	v0.y += (m10.x + m10.y + m10.z);\
	v1.y += (m10.y - m10.z - m10.w);\
	v0.z += (m20.x + m20.y + m20.z);\
	v1.z += (m20.y - m20.z - m20.w);\
	v0.w += (m30.x + m30.y + m30.z);\
	v1.w += (m30.y - m30.z - m30.w);}

#endif

#endif