#pragma once

#ifndef MICRO_H
#define MICRO_H

int LOG2(int n) {
	int result = 0;
	if (n & 0xffff0000) { result += 16; n >>= 16; }
	if (n & 0x0000ff00) { result += 8; n >>= 8; }
	if (n & 0x000000f0) { result += 4; n >>= 4; }
	if (n & 0x0000000c) { result += 2; n >>= 2; }
	if (n & 0x00000002) { result += 1; n >>= 1; }
	return result;
}

int PAD_2POW(int x) {
	unsigned int n = x - 1;
	n |= n >> 1;
	n |= n >> 2;
	n |= n >> 4;
	n |= n >> 8;
	n |= n >> 16;
	x = n;
	return ((n < 0) ? 1 : n + 1);

}

#ifndef DCONV3D_DW_IDX
#define DCONV3D_DW_IDX

//GK(oh, ow, n) -> GK(index = [oh, ow], n)
//(oh, ow) -> Idx = (oh*STEP + ow)

//for: 4 * 4 Filter, STEP = 4
//(0, 0) ->  0; (0, 1) ->  1; (0, 2) ->  2; (0, 3) ->  3
//(1, 0) ->  4; (1, 1) ->  5; (1, 2) ->  6; (1, 3) ->  7
//(2, 0) ->  8; (2, 1) ->  9; (2, 2) -> 10; (2, 3) -> 11
//(3, 0) -> 12; (3, 1) -> 13; (3, 2) -> 14; (3, 3) -> 15
__device__ __constant__ char YIDX_V2_O4P1[] = {//stride = 16
	0,  1,  2,  4,  5,  6,  8,  9, 10,  0,  0,  0,  0,  0,  0,  0,//tOH = 3, tOW = 3, ohw_idx = [0, 0]
	0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11,  0,  0,  0,  0,//tOH = 3, tOW = 4, ohw_idx = [0, 1]
	0,  1,  2,  4,  5,  6,  8,  9, 10, 12, 13, 14,  0,  0,  0,  0,//tOH = 4, tOW = 3, ohw_idx = [1, 0]
	0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 //tOH = 4, tOW = 4, ohw_idx = [1, 1]
};


//for: 8 * 8 Filter, STEP = 8
// 0,  1,  2,  3,  4,  5,  6,  7,
// 8,  9, 10, 11, 12, 13, 14, 15,
//16, 17, 18, 19, 20, 21, 22, 23,
//24, 25, 26, 27, 28, 29, 30, 31,
//32, 33, 34, 35, 36, 37, 38, 39,
//40, 41, 42, 43, 44, 45, 46, 47,
//48, 49, 50, 51, 52, 53, 54, 55,
//56, 57, 58, 59, 60, 61, 62, 63
__device__ __constant__ char YIDX_V2_O8P1X[] = {//stride = 64
	0,  1,  2,  3,  4,  5,  6,  8,  9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 53, 54,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
	0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,  0,  0,  0,  0,  0,  0,  0,  0,
	0,  1,  2,  3,  4,  5,  6,  8,  9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 53, 54, 56, 57, 58, 59, 60, 61, 62,  0,  0,  0,  0,  0,  0,  0,  0,
	0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63
};

//for: 7 * 7 Filter, STEP = 8
// 0,  1,  2,  3,  4,  5,  6
// 8,  9, 10, 11, 12, 13, 14,
//16, 17, 18, 19, 20, 21, 22,
//24, 25, 26, 27, 28, 29, 30,
//32, 33, 34, 35, 36, 37, 38,
//40, 41, 42, 43, 44, 45, 46,
//48, 49, 50, 51, 52, 53, 54
__device__ __constant__ char YIDX_V2_O7P1X[] = {//stride = 49
	0,  1,  2,  3,  4,  5,  8,  9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 28, 29, 32, 33, 34, 35, 36, 37, 40, 41, 42, 43, 44, 45,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,//tOH = 6, tOW = 6, ohw_idx = [0, 0]
	0,  1,  2,  3,  4,  5,  6,  8,  9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45, 46,  0,  0,  0,  0,  0,  0,  0,//tOH = 6, tOW = 7, ohw_idx = [0, 1]
	0,  1,  2,  3,  4,  5,  8,  9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 28, 29, 32, 33, 34, 35, 36, 37, 40, 41, 42, 43, 44, 45, 48, 49, 50, 51, 52, 53,  0,  0,  0,  0,  0,  0,  0,//tOH = 7, tOW = 6, ohw_idx = [1, 0]
	0,  1,  2,  3,  4,  5,  6,  8,  9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 53, 54,//tOH = 7, tOW = 7, ohw_idx = [1, 1]
};

#endif


#ifndef COMMON_MICRO
#define COMMON_MICRO

#define FLOAT_ZERO4 float4{0, 0, 0, 0} //make_float4(0, 0, 0, 0)
#define FLOAT_ZERO2 float2{0, 0} //make_float2(0, 0)

#define IS_POWER2(x) ( ((x)>0) && ((x)&((x)-1))==0 )

#define get2d(A, y, x, stride) A[(y)*(stride) + (x)]
#define lget2d(A, y, x, lstride) A[((y)<<(lstride)) + (x)] //lstride = log2(stride)

#define get3d(A, z, y, x, Sy, Sx) A[(((z)*(Sy)) + (y))*(Sx) + (x)]
#define lget3d(A, z, y, x, Sy, LSx) A[((((z)*(Sy)) + y)<<(LSx)) + (x)]

#define get3d_S(A, z_Sy, y, x, Sx) A[((z_Sy) + (y))*(Sx) + (x)]
#define lget3d_S(A, z_Sy, y, x, LSx) A[(((z_Sy) + y)<<(LSx)) + (x)]

#define get4d(A, w, z, y, x, Sz, Sy, Sx) A[(((w)*(Sz) + (z))*(Sy) + (y))*(Sx) + (x)]
#define get4d_S(A, w_Sz, z, y, x, Sy, Sx) A[(((w_Sz) + (z))*(Sy) + (y))*(Sx) + (x)]
#define lget4d(A, w, z, y, x, Sz, LSy, LSx) A[(((((w)*(Sz) + (z))<<(LSy)) + (y))<<(LSx)) + (x)]


#define simdMM4(c, av, b) {(c).x += (av) * (b).x; (c).y += (av) * (b).y; (c).z += (av) * (b).z; (c).w += (av) * (b).w;}
#define simdMM2(c, av, b) {(c).x += (av) * (b).x; (c).y += (av) * (b).y;} 

#define Mul4(c, av, b) {(c).x = (av) * (b).x; (c).y = (av) * (b).y; (c).z = (av) * (b).z; (c).w = (av) * (b).w;}
#define Mul2(c, av, b) {(c).x = (av) * (b).x; (c).y = (av) * (b).y; }

#define simdSadd4(c, av, b) {(c).x = (av) + (b).x; (c).y = (av) + (b).y; (c).z = (av) + (b).z; (c).w = (av) + (b).w;}

#define simdAdd4(c, a, b) {(c).x=(a).x+(b).x; (c).y=(a).y+(b).y; (c).z=(a).z+(b).z; (c).w=(a).w+(b).w;}
#define simdAdd2(c, a, b) {(c).x=(a).x+(b).x; (c).y=(a).y+(b).y;}

#define MAX_V_INT(k0, k1) (((k0) > (k1))*((k0) - (k1)) + (k1))
//#define MAX_V_INT(a, b) (b&((a-b)>>31) | a &(~(a - b) >> 31))

//if: flag == 1: -flag = -1 = 0xffffffff
//if: flag == 0: -flag =  0 
//( (( (((int)flag) << 32) - flag) & (a - b)) + b)
#define IF_int(flag, a, b) ( ((-flag) & (a - b)) + b)

//X = flag*v
#define zero_float(X, flag, v) \
	{ float fv = v; int iv = -(flag) & *(int*)(&fv); X = *(float*)(&iv); }

#define zero_float4(X, flag, v) {float fv; int iv; float4 XBUF;\
	fv = v.x; iv = -(flag) & *(int*)(&fv); XBUF.x = *(float*)(&iv);\
	fv = v.y; iv = -(flag) & *(int*)(&fv); XBUF.y = *(float*)(&iv);\
	fv = v.z; iv = -(flag) & *(int*)(&fv); XBUF.z = *(float*)(&iv);\
	fv = v.w; iv = -(flag) & *(int*)(&fv); XBUF.w = *(float*)(&iv);\
	X = XBUF; }

#endif

#define F32_4_0 float4{0, 0, 0, 0}
#define F32_2_0 float2{0, 0}


#define GET_GN(OC) (OC)// GN = OC
#define GET_GM(IC, FH, FW) ((IC)*(FH)*(FW)) // GM = IC * FH * FW
#define GET_GK(N, OH, OW) ((N)*(OH)*(OW)) // GKp = N * OHp * OWp -> GK = N * OH * OW

#define GET_OH(IH, FH, sh, ph) ( ((IH) + ((ph) << 1) - (FH))/(sh) + 1 )
#define GET_OW(IW, FW, sw, pw) ( ((IW) + ((pw) << 1) - (FW))/(sw) + 1 )

#define GET_IH(OH, FH, sh, ph) ( ((OH) - 1)*(sh) + (FH) - 2*(ph) )
#define GET_IW(OW, FW, sw, pw) ( ((OW) - 1)*(sw) + (FW) - 2*(pw) )


#ifndef GEMM_MICRO
#define GEMM_MICRO

#define load4d(V, A, w, z, y, x, Sz, Sy, Sx) \
	{if (((y)<0)||((y)>=Sy)||((z)<0)||((z)>= Sz)) (V) = 0;\
	 else (V) = get4d(A, w, z, y, x, Sz, Sy, Sx); }

#define load4d_S(V, A, w_Sz, z, y, x, Sz, Sy, Sx) \
	{if (((y)<0)||((y)>=Sy)||((z)<0)||((z)>= Sz)) (V) = 0;\
	 else (V) = get4d_S(A, w_Sz, z, y, x, Sy, Sx); }

#define load4d_check(V, ih, iw, value) \
	{if (((ih) < 0)||((ih) >= IH) || ((iw) < 0)||((iw) >= IW)) (V) = 0.0f; else (V) = (value);}

#define get_fh_fw(j, fh, fw) \
	int fh, fw; {fh = j / FW_IC; int jr = j - fh * FW_IC; fw = jr / IC;}

#define get_fh_fw_ic(j, fh, fw, ic) \
	int fh, fw, ic; {fh = j / FW_IC; int jr = j - fh * FW_IC; fw = jr / IC; ic = jr - fw * IC;}

#define get_fh_fw_ic_IC2pow(j, fh, fw, ic) \
	int fh, fw, ic; {fh = j / FW_IC; int jr = j - fh * FW_IC; fw = jr >> LIC; ic = jr & IC_m1;}

//====Improvement of j[fh, fw, ic] for k88============================
//[1] improvement of: ic
//ici = (ji % (FW * IC)) % IC = ji % IC
//As: <in k88>, j0 % 8 == 0: ji = j0 + i
//As: IC % 4 == 0
//ici = (j0 + i) % IC = (8*x + i) % (4*y)
//So: ic0 = ic1 - 1 = ic2 - 2 = ic3 - 3
//So: ic4 = ic5 - 1 = ic6 - 2 = ic7 - 3
//[2] improvement of: fh
//fhi = ji / (FW * IC)
//As: IC % 4 == 0, (FW * IC) % 4 == 0
//So: fhi = (8*x + i) / (4*y)
//So: fh0 = fh1 = fh2 = fh3, 
//So: fh4 = fh5 = fh6 = fh7
//[3] improvement of: fw
//fwi = (ji % (FW * IC)) / IC
//fwi = (8*x + i) % (4*y) / (4*u)
//let: Ui = (8*x + i) % (4*y)
//we have: U0 = U1 - 1 = U2 - 2 = U3 - 3
//we have: U4 = U5 - 1 = U6 - 2 = U7 - 3
//So: fwi = Ui / (4*u)
//for i = 0,1,2,3
//	fwi = (U0 + i) / (4*u)
//So: fw0 = fw1 = fw2 = fw3
//So: fw4 = fw5 = fw6 = fw7
//
//====Improvement of W[oc, fh, fw, ic] for k44============================
//[1] improvement of: ic
//ici = (ji % (FW * IC)) % IC = ji % IC
//As: <in k44>, j0 % 4 == 0: ji = j0 + i
//As: IC % 4 == 0
//ici = (j0 + i) % IC = (4*x + i) % (4*y)
//So: ic0 = ic1 - 1 = ic2 - 2 = ic3 - 3
//[2] improvement of: fh
//fhi = ji / (FW * IC)
//As: IC % 4 == 0, (FW * IC) % 4 == 0
//So: fhi = (4*x + i) / (4*y)
//So: fh0 = fh1 = fh2 = fh3, 
//[3] improvement of: fw
//fwi = (ji % (FW * IC)) / IC
//fwi = (4*x + i) % (4*y) / (4*u)
//let: Ui = (4*x + i) % (4*y)
//we have: U0 = U1 - 1 = U2 - 2 = U3 - 3
//So: fwi = Ui / (4*u)
//for i = 0,1,2,3:
//	fwi = (U0 + i) / (4*u)
//So: fw0 = fw1 = fw2 = fw3
//
//
//====Improvement of W[oc, fh, fw, ic] for k22============================
//[1] improvement of: ic
//ici = (ji % (FW * IC)) % IC = ji % IC
//As: <in k22>, j0 % 2 == 0: ji = j0 + i
//As: IC % 4 == 0
//ici = (j0 + i) % IC = (2*x + i) % (4*y) = (4*x1 + 2*x2 + i) % (4*y)
//ici = (2*x2 + i) % (4*y), As: i <= 1, x2 <= 1, So: 2*x2 + i <= 3
//So: ic0 = ic1 - 1
//[2] improvement of: fh
//fhi = ji / (FW * IC)
//As: IC % 4 == 0, (FW * IC) % 4 == 0
//So: fhi = (4*x1 + 2*x2 + i) / (4*y) 
//So: fhi = (x1 + (2*x2 + i)/4) / y, As: i <= 1, x2 <= 1, So: 2*x2 + i <= 3
//So: fhi = x1 / y
//So: fh0 = fh1
//[3] improvement of: fw
//fwi = (ji % (FW * IC)) / IC
//fwi = (4*x1 + 2*x2 + i) % (4*y) / (4*u)
//let: Ui = (4*x1 + 2*x2 + i) % (4*y),  As: i <= 1, x2 <= 1, So: 2*x2 + i <= 3
//we have: U0 = U1 - 1
//for i = 0, 1: fwi = (U0 + i) / (4*u)
//So: fw0 = fw1s
//========================================================================


//====[Improvement of xoffset(n, ih, ih, ic)]=============================
//X_ih = X_oh*sh
//X_iw = X_ow*sw
//xoffset = ((X_n*IH + X_ih)*IW + X_iw) * IC;
//xoffset = ((n*IH + ih)*IW + iw) * IC;
//xoffset = ((n*IH + oh*sh)*IW + ow*sw) * IC;
//(1) when: IH = OH*sh, IW = OW*sw
//xoffset = ((n*OH*sh + oh*sh)*OW*sw + ow*sw) * IC;
//xoffset = ((n*OH*sh + oh*sh)*OW + ow) * IC*sw;
//xoffset = ((n*OH + oh)*OW*sh + ow) * IC*sw;
//let: U = (n*OH + oh)*OW*sh + ow
//xoffset = U *IC*sw
//U = (n*OH + oh)*OW*sh + ow
//U = (n*OH*OW + oh*OW)*sh + ow
//As: k = n*OH*OW + oh*OW + ow, n*OH*OW + oh*OW = k - ow
//U = (k - ow)*sh + ow
//xoffset = ((k - ow)*sh + ow) *IC*sw
//xoffset = ((k*sh + (1 - sh)*ow) *IC*sw
//(2) when: sh = sw = 2;
//xoffset = (((k << 1) - ow)) << 1) * IC
//xoffset = (((k << 1) - (iw >> 1)) << 1) * IC
//xoffset = (((k << 2) - iw) * IC
//(3) when: sh = sw = 1;
//xoffset = k * IC
//====[Improvement of xoffset(n, ih, ih, ic)]=============================


//====[get_xoffset(n, oh, ow)]============================================
#define get_X_n_oh_ow(k, n, oh, ow) {\
	n = k / OH_OW; k -= n * OH_OW;\
	oh = k / OW; ow = k - oh * OW;\
	n *= IH; oh *= sh; ow *= sw;}

#define get_X_n_oh_ow_OHW2pow(k, n, oh, ow) {\
	n = k >> LOH_OW; k &= OH_OW_m1;\
	oh = k >> LOW; ow = k & OW_m1;\
	n *= IH; oh *= sh; ow *= sw;}

#define uget_X_n_oh_ow_OHW2pow(k, n, oh, ow0, ow1) int n, oh, ow0, ow1;{\
	n = k >> LOH_OW; k &= OH_OW_m1;\
	oh = k >> LOW; ow0 = k & OW_m1;\
	oh *= sh; ow0 *= sw; ow1 = ow0 + sw;}

#define get_X_n_oh_ow_s2(k, n, oh, ow) {\
	n = k / OH_OW; k -= n * OH_OW;\
	oh = k / OW; ow = k - oh * OW;\
	n *= IH; oh <<= 1; ow <<= 1;}

#define get_X_n_oh_ow_OHW2pow_s2(k, n, oh, ow) {\
	n = k >> LOH_OW; int kr = k & OH_OW_m1;\
	oh = kr >> LOW; ow = kr & OW_m1;\
	n *= IH; oh <<= 1; ow <<= 1;}

#define get_X_n_oh_ow_W1(k, n, oh, ow) {\
	n = k / OH_OW; k -= n * OH_OW;\
	oh = k / OW; ow = k - oh * OW;}
//====[get_xoffset(n, oh, ow)]============================================


//====[get_offset(oh, ow, n)]============================================
//when N % STEP == 0
//Xk = STEP*x + Ux
//Yk = STEP*x + Uy
//As: Ux, Uy < STEP
//we have: 
//<1> X_oh = X_k / (N*OW) = (STEP*x + Ux) / (STEP*v) = x/v
//<2> Y_oh = Y_k / (N*OW) = (STEP*x + Uy) / (STEP*v) = x/v
//So: X_oh = Y_oh
//<1> X_ow = (X_k % (N*OW)) / N = ((STEP*x + Ux) % (STEP*v)) / (STEP*g) = q/g
//<2> Y_ow = (Y_k % (N*OW)) / N = ((STEP*x + Uy) % (STEP*v)) / (STEP*g) = q/g
//So: X_ow = Y_ow
//As: Y_ow = X_ow = ow

#define get_oh_ow_n(k, oh, ow, n) {\
	oh = k / (OW_N); k -= oh*OW_N;\
	ow = k / N, n = k - ow*N; }
//====[get_xoffset(oh, ow, n)]============================================


#define LOAD_X(fh, fw) ((fh >= -X_oh) && (fh < IH - X_oh) && (fw >= -X_ow) && (fw < IW - X_ow))
#define LOAD_X2(fh, fw, oh, ow) ((fh >= -oh) && (fh < IH - oh) && (fw >= -ow) && (fw < IW - ow))

#endif


//GEMMSK: split K
#ifndef GEMMSK_MICRO
#define GEMMSK_MICRO

//GZ = gridDim.z
//GK = N * OH * OW, so: GK % 4 == 0
//GK_slice = (GK / gridDim.z) >> 3 << 3
//GK = GK_slice * gridDim.z + RGK
//As: GK % 8 == 0
//So: RGK % 4 == 0
//if: GK % 8 == 0, We have RGK % 8 == 0
#define GEMMSK_GK_slice(GK, GZ)  ((GK / GZ) >> 3 << 3)

#define GEMMSK_init(GZ, OH, OW, FH, FW, N, IC, OC) \
	int GN = GET_GN(OC);\
	int GM = GET_GM(IC, FH, FW);\
	int GK = GET_GK(N, OH, OW);\
	int GK_slice = GEMMSK_GK_slice(GK, GZ);

#endif


#ifndef GEMMV2_MICRO
#define GEMMV2_MICRO

#define CAN_V2_O2P1 ((OH == 2) && (OW == 2) && (ph == 1) && (pw == 1) && (IH >= 2) && (IW >= 2))
#define CAN_V2_O4P1 ((OH == 4) && (OW == 4) && (ph == 1) && (pw == 1) && (IH >= 4) && (IW >= 4))
#define CAN_V2_O7P1 ((OH == 7) && (OW == 7) && (ph == 1) && (pw == 1) && (IH >= 7) && (IW >= 7))
#define CAN_V2_O8P1 ((OH == 8) && (OW == 8) && (ph == 1) && (pw == 1) && (IH >= 8) && (IW >= 8))

//(IH + 2oph - OHp)/1 + 1 = FH
//As: OHp = OH + (OH - 1)*(sh - 1) = OH*sh - sh + 1
//IH + 2oph - (OH*sh - sh + 1) + 1 = FH
//IH + 2oph - OH*sh + 1 = FH
//Oph = 2oph = (OH - 1)*sh - IH + FH;
//Opw = 2opw = (OW - 1)*sw - IW + FW
//[3] Oph + IH = (OH - 1)*sh + FH
//    Opw + IW = (OW - 1)*sw + FW
//[4] Q = 1.0 * (((OH - 1)*sh + FH) * ((OW - 1)*sw + FW)) / (IH * IW)
//ex1: IH = IW = 32, ph = pw = 1, Q = 1.06348
//ex2: IH = IW = 16, ph = pw = 1, Q = 1.12891
//ex3: IH = IW =  8, ph = pw = 1, Q = 1.26562
#define PADDING_SCALE_UP(IH, IW, OH, OW, FH, FW, sh, sw)\
	(1.0 * (((OH - 1)*sh + FH) * ((OW - 1)*sw + FW)) / (IH * IW))

//As: GM is ordered by FH * FW * IC, we can't use SK_V2_TO_V1
#define SK_V2_TO_V1(FH, FW, N, IC, ic_index, GZ) \
		const int FH_FW = FH * FW;\
		int GM = N * FH_FW;\
		int GK = FH_FW * IC, GK_slice = GEMMSK_GK_slice(GK, GZ);\
		int j_index = ic_index * FH_FW;

#endif

#endif