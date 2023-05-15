#pragma once

#ifndef MICRO_H
#define MICRO_H

__device__ float HOLE[260];

__device__ __constant__ float _ZERO;//default value of zero is 0


#ifndef DCONV3D_DX_IX
#define DCONV3D_DX_IX

//GK(oc, fh, fw) -> GK(oc, index = [fh, fw])
//(fh, fw) -> Idx = (fh*4 + fw)

//for 3*3 filter: STEP = 4
//(0, 0) -> 0; (0, 1) -> 1; (0, 2) ->  2
//(1, 0) -> 4; (1, 1) -> 5; (1, 2) ->  6
//(2, 0) -> 8; (2, 1) -> 9; (2, 2) -> 10
__device__ __constant__ char YIDX_W33[] = {//STEP = 4
	0, 1, 2,
	4, 5, 6,
	8, 9, 10 };

//WIdx = fhr*3 + fwr
//[(0, 0), (0, 1), (0, 2)] -> [(2, 2), (2, 1), (2, 0)] -> [2*3+2, 2*3+1, 2*3 + 0] -> [8, 7, 6]
//[(1, 0), (1, 1), (1, 2)] -> [(1, 2), (1, 1), (0, 2)] -> [1*3+2, 1*3+1, 0*3 + 0] -> [5, 4, 3]
//[(2, 0), (2, 1), (2, 2)] -> [(0, 2), (0, 1), (0, 0)] -> [2, 1, 0]
__device__ __constant__ char WIDX_W33[] = {
	8, 7, 6,
	5, 4, 3,
	2, 1, 0 };

__device__ __constant__ char YIDX_V2_W3P1[]{//stride = 9, STEP = 4
	0, 1, 4, 5, 0, 0, 0, 0,  0,//tFH = 2, tFW = 3, fhw_idx = (0, 0)
	0, 1, 2, 4, 5, 6, 0, 0,  0,//tFH = 2, tFW = 3, fhw_idx = (0, 1)

	0, 1, 4, 5, 8, 9, 0, 0,  0,//tFH = 3, tFW = 2, fhw_idx = (1, 0)
	0, 1, 2, 4, 5, 6, 8, 9, 10 //tFH = 3, tFW = 3, fhw_idx = (1, 1)
};

__device__ __constant__ char YIDX_V2_W3P2[]{//stride = 9, STEP = 4
	0, 0, 0, 0, 0, 0, 0, 0,  0,//tFH = 1, tFW = 1, fhw_idx = (0, 0)
	0, 1, 0, 0, 0, 0, 0, 0,  0,//tFH = 1, tFW = 2, fhw_idx = (0, 1)
	0, 1, 2, 0, 0, 0, 0, 0,  0,//tFH = 1, tFW = 3, fhw_idx = (0, 2)

	0, 4, 0, 0, 0, 0, 0, 0,  0,//tFH = 2, tFW = 1, fhw_idx = (1, 0)
	0, 1, 4, 5, 0, 0, 0, 0,  0,//tFH = 2, tFW = 2, fhw_idx = (1, 1)
	0, 1, 2, 4, 5, 6, 0, 0,  0,//tFH = 2, tFW = 3, fhw_idx = (1, 2)

	0, 4, 9, 0, 0, 0, 0, 0,  0,//tFH = 3, tFW = 1, fhw_idx = (2, 0)
	0, 1, 4, 5, 8, 9, 0, 0,  0,//tFH = 3, tFW = 2, fhw_idx = (2, 1)
	0, 1, 2, 4, 5, 6, 8, 9, 10 //tFH = 3, tFW = 3, fhw_idx = (2, 2)
};


//for 5*5 Filter, STEP = 8
//(0, 0) ->  0; (0, 1) ->  1; (0, 2)->   2; (0, 3) ->  3; (0, 4) ->  4
//(1, 0) ->  8; (1, 1) ->  9; (1, 2) -> 10; (1, 3) -> 11; (1, 4) -> 12
//(2, 0) -> 16; (2, 1) -> 17; (2, 2) -> 18; (2, 3) -> 19; (2, 4) -> 20
//(3, 0) -> 24; (3, 1) -> 25; (3, 2) -> 26; (3, 3) -> 27; (3, 4) -> 28
//(4, 0) -> 32; (4, 1) -> 33; (4, 2) -> 34; (4, 3) -> 35; (4, 4) -> 36
__device__ __constant__ char YIDX_W55[] = {//STEP = 8
	 0,  1,  2,  3,  4,
	 8,  9, 10, 11, 12,
	16, 17, 18, 19, 20,
	24, 25, 26, 27, 28,
	32, 33, 34, 35, 36 };

//WIdx = fhr*5 + fwr
//[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)] -> [(4, 4), (4, 3), (4, 2), (4, 1), (4, 0)] -> [24, 23, 22, 21, 20]
//[(1, 0), (1, 1), (1, 2), (1, 3), (1, 4)] -> [(3, 4), (3, 3), (3, 2), (3, 1), (3, 0)] -> [19, 18, 17, 16, 15]
//[(2, 0), (2, 1), (2, 2), (2, 3), (2, 4)] -> [(2, 4), (2, 3), (2, 2), (2, 1), (2, 0)] -> [14, 13, 12, 11, 10]
//[(3, 0), (3, 1), (3, 2), (3, 3), (3, 4)] -> [(1, 4), (1, 3), (1, 2), (1, 1), (1, 0)] -> [ 9,  8,  7,  6,  5]
//[(4, 0), (4, 1), (4, 2), (4, 3), (4, 4)] -> [(0, 4), (0, 3), (0, 2), (0, 1), (0, 0)] -> [ 4,  3,  2,  1,  0]
__device__ __constant__ char WIDX_W55[] = {
	24, 23, 22, 21, 20,
	19, 18, 17, 16, 15,
	14, 13, 12, 11, 10,
	 9,  8,  7,  6,  5,
     4,  3,  2,  1,  0};

__device__ __constant__ char YIDX_V2_W5P2[] = {//stride = 25
	0, 1, 2, 8, 9, 10, 16, 17, 18,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,//tFH = 3, tFW = 3, fhw_idx = (0, 0)
	0, 1, 2, 3, 8,  9, 10, 11, 16, 17, 18, 19,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,//tFH = 3, tFW = 4, fhw_idx = (0, 1)
	0, 1, 2, 3, 4,  8,  9, 10, 11, 12, 16, 17, 18, 19, 20,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,//tFH = 3, tFW = 5, fhw_idx = (0, 2)

	0, 1, 2, 8, 9, 10, 16, 17, 18, 24, 25, 26,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,//tFH = 4, tFW = 3, fhw_idx = (1, 0)
	0, 1, 2, 3, 8,  9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27,  0,  0,  0,  0,  0,  0,  0,  0,  0,//tFH = 4, tFW = 4, fhw_idx = (1, 1)
	0, 1, 2, 3, 4,  8,  9, 10, 11, 12, 16, 17, 18, 19, 20, 24, 25, 26, 27, 28,  0,  0,  0,  0,  0,//tFH = 4, tFW = 5, fhw_idx = (1, 2)

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

#define FLOAT_ZERO4 make_float4(0, 0, 0, 0)
#define FLOAT_ZERO2 make_float2(0, 0)

#define F32_2_0 float2{ 0, 0 }

#define IS_POWER2(x) ( ((x)>0) && ((x)&((x)-1))==0 )

#define get2d(A, y, x, stride) A[(y)*(stride) + (x)]
#define get3d(A, z, y, x, Sy, Sx) A[(((z)*(Sy)) + (y))*(Sx) + (x)]
#define get4d(A, w, z, y, x, Sz, Sy, Sx) A[(((w)*(Sz) + (z))*(Sy) + (y))*(Sx) + (x)]

//#define lget2d(A, y, x, lstride) A[((y)<<(lstride)) + (x)]
//#define get3d_S(A, z_Sy, y, x, Sx) A[((z_Sy) + (y))*(Sx) + (x)]
//#define lget3d_S(A, z_Sy, y, x, LSx) A[(((z_Sy) + y)<<(LSx)) + (x)]
//#define lget3d(A, z, y, x, Sy, LSx) A[((((z)*(Sy)) + y)<<(LSx)) + (x)]
//#define lget4d(A, w, z, y, x, Sz, LSy, LSx) A[(((((w)*(Sz) + (z))<<(LSy)) + (y))<<(LSx)) + (x)]
//#define get4d_S(A, w_Sz, z, y, x, Sy, Sx) A[(((w_Sz) + (z))*(Sy) + (y))*(Sx) + (x)]


#define simdMM4(c, av, b) {(c).x += (av) * (b).x; (c).y += (av) * (b).y; (c).z += (av) * (b).z; (c).w += (av) * (b).w;}
#define simdMM2(c, av, b) {(c).x += (av) * (b).x; (c).y += (av) * (b).y;} 
#define simdAdd4(c, av, b) {(c).x = (av) + (b).x; (c).y = (av) + (b).y; (c).z = (av) + (b).z; (c).w = (av) + (b).w;}
#define Mul4(c, av, b) {(c).x = (av) * (b).x; (c).y = (av) * (b).y; (c).z = (av) * (b).z; (c).w = (av) * (b).w;}

#define MAX_V_INT(k0, k1) (((k0) > (k1))*((k0) - (k1)) + (k1))

//if: flag == 1: -flag = -1 = 0xffffffff
//if: flag == 0: -flag =  0 
#define IF_int(flag, a, b) ( ((-flag) & (a - b)) + b)

//X = flag*v
#define zero_float(X, flag, v) \
	{float fv = v; int iv = -(flag) & *(int*)(&fv); X = *(float*)(&iv); }

#define zero_float4(X, flag) \
	{ if(!flag) X.x = X.y = X.z = X.w = 0;}


#define GET_OH(IH, FH, sh, ph) ( ((IH) + ((ph) << 1) - (FH))/(sh) + 1 )
#define GET_OW(IW, FW, sw, pw) ( ((IW) + ((pw) << 1) - (FW))/(sw) + 1 )

#define GET_IH(OH, FH, sh, ph) ( ((OH) - 1)*(sh) + (FH) - 2*(ph) )
#define GET_IW(OW, FW, sw, pw) ( ((OW) - 1)*(sw) + (FW) - 2*(pw) )

#define LOAD_Y(tih, tiw, fh, fw)\
	((tih >= -fh) && (tih < OH - fh) && (tiw >= -fw) && (tiw < OW - fw))

#endif


#ifndef MICRO_ZERO_PADDING_KERNEL_DENSE
#define MICRO_ZERO_PADDING_KERNEL_DENSE

#define GET_GN_ZeroPadding(IC) (IC) // GN = IC
#define GET_GM_ZeroPadding(N, IH, IW) ((N)*(IH)*(IW)) // GM = N  * IH * IW
#define GET_GK_ZeroPadding(OC, FH, FW) ((OC)*(FH)*(FW)) // GK = OC * FH * FW;

//====Improvement of j(n, ih, iw)(4x kernel)==================
//in k88: j % 8 == 0
//when (IH, IW) % 4 == 0
//(1) iw = j % IW = (j0 + i) % IW = (8x + i) % (4y) 
//So: iw0 = iw1 - 1 = iw2 - 2 = iw3 - 3
//So: iw4 = iw5 - 1 = iw6 - 2 = iw7 - 3
//(2) n = j / (IH * IW) = (j0 + i) / (IH * IW) 
//As: i belongs to (0, 7)
//So: n = (8*x + i) / (16y)
//So: ni = nj, i,j belongs to {0, 1, 2, 3, 4, 5, 6, 7}
//(3) ih = (j % (IH * IW)) / IW
//    ih = ((j0 + i) % (IH * IW)) / IW
//    ih = ((8x + i) % (16y)) / 4z
//As: (8x + i) % (16y) = 8*y + i
//So: ih = (8*y + i) / 4z
//So: ih0 = ih1 = ih2 = ih3 
//So: ih4 = ih5 = ih6 = ih7
//============================================================
#define get_n_ih_iw(j, n, ih, iw) \
	int n, ih, iw; {n = j / IH_IW; int jr = j - n * IH_IW; ih = jr / IW, iw = jr - ih * IW;}

#define get_ih_iw_n(j, ih, iw, n) \
	int ih, iw, n; {ih = j / IW_N; int jr = j - ih * IW_N; iw = jr / N, n = jr - iw * N; }
  
#define load4d(V, A, w, z, y, x, Sz, Sy, Sx) \
	{if (((y)<0)||((y)>=Sy)||((z)<0)||((z)>= Sz)) (V) = 0;\
	 else (V) = get4d(A, w, z, y, x, Sz, Sy, Sx); }

#define load4d_S(V, A, w_Sz, z, y, x, Sz, Sy, Sx) \
	{if (((y)<0)||((y)>=Sy)||((z)<0)||((z)>= Sz)) (V) = 0;\
	 else (V) = get4d_S(A, w_Sz, z, y, x, Sy, Sx); }

#define load4d_check(V, ih, iw, value) \
	{if (((ih) < 0)||((ih) >= IH) || ((iw) < 0)||((iw) >= IW)) (V) = 0.0f; else (V) = (value);}

#define get_dY_oc_fh_fw_W3(k, oc, fh, fw) \
	oc = k / 9; k -= oc * 9;\
	fh = k / 3; fw = k - fh * 3;

//to compute the corresponding index of deltaY
//GK order(OC, FH, FW)
#define get_dY_oc_fh_fw(k, oc, fh, fw) \
	oc = k / FH_FW; k -= oc * FH_FW;\
	fh = k / FW; fw = k - fh * FW;

#define get_dY_oc_fh_fw_W2pow(k, oc, fh, fw) \
	oc = k >> LFH_FW; k &= FH_FW_m1;\
	fh = k >> LFW; fw = k & FW_m1;

//to comoute the corresponding index of W
//Wr_fh_fw -> W[oc, FH - 1 - fh, FW - 1 -fw, ic]
// = (FH - 1 - W_fh)*FW + (FW - 1 - W_fw)
// = (FH*FW - 1) - (W_fh*FW + fw) = (FH_FW - 1) - W_k
#define get_W_oc_fh_fw(k, oc, r_fh_fw) \
	oc = k / FH_FW; k -= oc * FH_FW;\
	r_fh_fw = FH_FW - 1 - k;

#define get_W_oc_fh_fw_W2pow(k, oc, r_fh_fw) \
	oc = k >> LFH_FW; k &= FH_FW_m1;\
	r_fh_fw = FH_FW_m1 - k;

#define load4d_s1(V, n, oh, ow, oc) \
	if (oh < 0 || oh >= OH || ow < 0 || ow >= OW) (V) = 0.0f;\
	else (V) = get4d(deltaY, n, oh, ow, oc, OH, OW, OC);  

#ifndef S1_LOAD_YS4
#define S1_LOAD_YS4

__device__ __forceinline__ float4 S1_SaveYs4(const float* __restrict__ deltaY,
	int Y_fh, int Y_fw, int yoffset, int OH, int OW,
	int Y0, int tih0, int tiw0,
	int Y1, int tih1, int tiw1,
	int Y2, int tih2, int tiw2,
	int Y3, int tih3, int tiw3)
{
	OH -= Y_fh; OW -= Y_fw;
	bool ly0 = (tih0 >= -Y_fh) && (tih0 < OH) && (tiw0 >= -Y_fw) && (tiw0 < OW);
	bool ly1 = (tih1 >= -Y_fh) && (tih1 < OH) && (tiw1 >= -Y_fw) && (tiw1 < OW);
	bool ly2 = (tih2 >= -Y_fh) && (tih2 < OH) && (tiw2 >= -Y_fw) && (tiw2 < OW);
	bool ly3 = (tih3 >= -Y_fh) && (tih3 < OH) && (tiw3 >= -Y_fw) && (tiw3 < OW);

	float4 x;
	x.x = (ly0 ? deltaY[Y0 + yoffset] : 0);
	x.y = (ly1 ? deltaY[Y1 + yoffset] : 0);
	x.z = (ly2 ? deltaY[Y2 + yoffset] : 0);
	x.w = (ly3 ? deltaY[Y3 + yoffset] : 0);
	return x;
}

__device__ __forceinline__ float4 S1_SaveYs4x(const float* __restrict__ deltaY,
	int Y_fh, int Y_fw, int yoffset, int OH, int OW, int OC,
	int tih0, int tiw0, int tiw1, int tiw2, int tiw3)
{
	bool ly = (tih0 >= -Y_fh) && (tih0 < OH - Y_fh);
	bool ly0 = ly && (tiw0 >= -Y_fw) && (tiw0 < OW - Y_fw);
	bool ly1 = ly && (tiw1 >= -Y_fw) && (tiw1 < OW - Y_fw);
	bool ly2 = ly && (tiw2 >= -Y_fw) && (tiw2 < OW - Y_fw);
	bool ly3 = ly && (tiw3 >= -Y_fw) && (tiw3 < OW - Y_fw);

	float4 x;
	x.x = (ly0 ? deltaY[yoffset - OC] : 0);//Y0
	x.y = (ly1 ? deltaY[yoffset] : 0);//Y1
	x.z = (ly2 ? deltaY[yoffset + OC] : 0);//Y2
	x.w = (ly3 ? deltaY[yoffset + (OC << 1)] : 0);//Y3
	return x;
}

#endif


#ifndef S1_LOAD_YS4_TEXTURE
#define S1_LOAD_YS4_TEXTURE

__device__ __forceinline__ float4 S1_SaveYs4_tex(cudaTextureObject_t deltaY,
	int Y_fh, int Y_fw, int yoffset, int OH, int OW,
	int Y0, int tih0, int tiw0,
	int Y1, int tih1, int tiw1,
	int Y2, int tih2, int tiw2,
	int Y3, int tih3, int tiw3)
{
	float4 x;
	x.x = tex1Dfetch<float>(deltaY, Y0 + yoffset);
	x.y = tex1Dfetch<float>(deltaY, Y1 + yoffset);
	x.z = tex1Dfetch<float>(deltaY, Y2 + yoffset);
	x.w = tex1Dfetch<float>(deltaY, Y3 + yoffset);

	bool ly0 = LOAD_Y(tih0, tiw0, Y_fh, Y_fw); zero_float(x.x, ly0, x.x);
	bool ly1 = LOAD_Y(tih1, tiw1, Y_fh, Y_fw); zero_float(x.y, ly1, x.y);
	bool ly2 = LOAD_Y(tih2, tiw2, Y_fh, Y_fw); zero_float(x.z, ly2, x.z);
	bool ly3 = LOAD_Y(tih3, tiw3, Y_fh, Y_fw); zero_float(x.w, ly3, x.w);
	return x;
}

__device__ __forceinline__ float4 S1_SaveYs4x_tex(cudaTextureObject_t deltaY,
	int Y_fh, int Y_fw, int yoffset, int OH, int OW, int OC,
	int tih0, int tiw0, int tiw1, int tiw2, int tiw3)
{
	float4 x;
	x.x = tex1Dfetch<float>(deltaY, yoffset - OC);
	x.y = tex1Dfetch<float>(deltaY, yoffset);
	x.z = tex1Dfetch<float>(deltaY, yoffset + OC);
	x.w = tex1Dfetch<float>(deltaY, yoffset + (OC << 1));

	bool ly = (tih0 >= -Y_fh) && (tih0 < OH - Y_fh);
	bool ly0 = ly && (tiw0 >= -Y_fw) && (tiw0 < OW - Y_fw); zero_float(x.x, ly0, x.x);//Y0
	bool ly1 = ly && (tiw1 >= -Y_fw) && (tiw1 < OW - Y_fw); zero_float(x.y, ly1, x.y);//Y1
	bool ly2 = ly && (tiw2 >= -Y_fw) && (tiw2 < OW - Y_fw); zero_float(x.z, ly2, x.z);//Y2
	bool ly3 = ly && (tiw3 >= -Y_fw) && (tiw3 < OW - Y_fw); zero_float(x.w, ly3, x.w);//Y3
	return x;
}

#endif

#endif


#ifndef MICRO_ZERO_PADDING_V2_DENSE
#define MICRO_ZERO_PADDING_V2_DENSE

#define CAN_s1_V2_W3P1 ((FH == 3) && (FW == 3) && (ph == 1) && (pw == 1) && (OH > 1) && (OW > 1))
#define CAN_s1_V2_W5P2 ((FH == 5) && (FW == 5) && (ph == 2) && (pw == 2) && (OH > 2) && (OW > 2)) 

//(OH - FH + 2oph) + 1 = IH
//[1] OH + Oph = 2oph = IH - 1 + FH
//[2] OW + Opw = 2opw = IW - 1 + FW
//[3] Q = ((OH + Oph) * (OW + Opw)) / (OH * OW)
//    Q = 1.0 * ((IH - 1 + FH) * (IW - 1 + FW)) / (IH * IW)
//(IH, IW) = 16: Qs1 = 1.26562
//(IH, IW) =  8: Qs1 = 1.5625
#define s1_PADDING_SCALE_UP(IH, IW, OH, OW, FH, FW) \
	(1.0 * ((IH - 1 + FH) * (IW - 1 + FW)) / (OH * OW))

//V1.GMr = V2.Nr * IH * IW
//As: GM = V1.GMr + V1.j_index = (n_index + V2.Nr) * IH * IW = N * IH * IW
#define s1_V2_TO_V1(FH, FW, IH, IW, N, OC, n_index) \
	const int IH_IW = IH * IW;\
	int GM = N * IH_IW, GK = FH * FW * OC;\
	int j_index = n_index * IH_IW;

#endif


#ifndef MICRO_CROSS_ADD_KERNEL
#define MICRO_CROSS_ADD_KERNEL

#define GET_GN_CrossAdd(OC) (OC)
#define GET_GM_CrossAdd(N, OH, OW) ((N)*(OH)*(OW))
#define GET_GK_CrossAdd(FH, FW, IC) ((FH)*(FW)*(IC))

#define get_n_oh_ow(j, n, oh, ow) \
	int n, oh, ow; {n = j / OH_OW; int jr = j - n * OH_OW; oh = jr / OW, ow = jr - oh * OW;}

#define getX_fh_fw(k, fh, fw) {fh = k / FW_IC; k -= fh * FW_IC; fw = k / IC;}
#define getX_fh_fw_ic2pow(k, fh, fw) {fh = k / FW_IC; k -= fh * FW_IC; fw = k / IC;}


#define CrossAdd_SUM4(a, b) ((a).x*(b).x + (a).y*(b).y + (a).z*(b).z + (a).w *(b).w)
#define CrossAdd_SUM2(a, b) ((a).x*(b).x + (a).y*(b).y)

#endif


#ifndef MICRO_KERNEL_SPLIT
#define MICRO_KERNEL_SPLIT

#define KS_IH_slice(IH, sh) ((IH + sh - 1)/(sh)) //IH_slice = (IH + sh - 1) / sh
#define KS_IW_slice(IW, sw) ((IW + sw - 1)/(sw)) //IW_slice = (IW + sw - 1) / sw
#define KS_CWstride(CFH, CFW, OC, IC) (CFH * CFW * OC * IC)

#define KS_GN(IC) (IC)
#define KS_GM(N, IH, IW, sh, sw) ((N)*(IH/sh)*(IW/sw)) //GM = N*IH_slice*IW_slice

#define KS_CFH(FH, sh) ((FH + sh - 1) / sh)
#define KS_CFW(FW, sw) ((FW + sw - 1) / sw)

//(CFH, CFW): the max (CFH, CFW)
#define KS_init(N, IH, IW, FH, FW, OC, IC, sh, sw) \
	int CFH = KS_CFH(FH, sh);\
	int CFW = KS_CFW(FW, sw);\
	int IH_slice = KS_IH_slice(IH, sh);\
	int IW_slice = KS_IW_slice(IW, sw);\
	int CWstride = KS_CWstride(CFH, CFW, OC, IC);\
	int GN = IC; \
	int GM = N * IH_slice * IW_slice;\

#define WRT_X(ih, iw) ((ih < IH) && (iw < IW))


//for k88, k88_oc2pow
#ifndef KS_SAVE_YS4
#define KS_SAVE_YS4

__device__ __forceinline__ float4 KS_SaveYs4(const float* __restrict__ deltaY,
	int tohs0, int tows0, int Y_fhr, int Y_fwr, int OH, int OW,
	int yoffset, int Y0, int Y1, int Y2, int Y3)
{
	bool ly0 = (tohs0 >= -Y_fhr) && (tohs0 < OH - Y_fhr) && 
		       (tows0 >= -Y_fwr) && (tows0 < OW - Y_fwr);
	if (!ly0) return  FLOAT_ZERO4;

	float4 y;
	y.x = deltaY[Y0 + yoffset];
	y.y = deltaY[Y1 + yoffset];
	y.z = deltaY[Y2 + yoffset];
	y.w = deltaY[Y3 + yoffset];
	return y;
}

__device__ __forceinline__ float4 KS_SaveYs4_tex(cudaTextureObject_t deltaY,
	int tohs0, int tows0, int Y_fhr, int Y_fwr, int OH, int OW,
	int yoffset, int Y0, int Y1, int Y2, int Y3)
{
	bool ly0 = (tohs0 >= -Y_fhr) && (tohs0 < OH - Y_fhr) &&
		       (tows0 >= -Y_fwr) && (tows0 < OW - Y_fwr);

	float4 y;
	zero_float(y.x, ly0, tex1Dfetch<float>(deltaY, Y0 + yoffset));
	zero_float(y.y, ly0, tex1Dfetch<float>(deltaY, Y1 + yoffset));
	zero_float(y.z, ly0, tex1Dfetch<float>(deltaY, Y2 + yoffset));
	zero_float(y.w, ly0, tex1Dfetch<float>(deltaY, Y3 + yoffset));
	return y;
}

#endif


//for k44, k44_oc2pow
#ifndef KS_SAVE_YS2
#define KS_SAVE_YS2

__device__ __forceinline__ float2 KS_SaveYs2(const float* __restrict__ deltaY,
	int tohs0, int tows0, int Y_fhr, int Y_fwr, int OH, int OW,
	int yoffset, int Y0, int Y1)
{
	bool ly0 = (tohs0 >= -Y_fhr) && (tohs0 < OH - Y_fhr) &&
	   	       (tows0 >= -Y_fwr) && (tows0 < OW - Y_fwr);
	if (!ly0) return  FLOAT_ZERO2;

	float2 y;
	y.x = deltaY[Y0 + yoffset];
	y.y = deltaY[Y1 + yoffset];
	return y;
}

__device__ __forceinline__ float2 KS_SaveYs2_tex(cudaTextureObject_t deltaY,
	int tohs0, int tows0, int Y_fhr, int Y_fwr, int OH, int OW,
	int yoffset, int Y0, int Y1)
{
	bool ly0 = (tohs0 >= -Y_fhr) && (tohs0 < OH - Y_fhr) && 
		       (tows0 >= -Y_fwr) && (tows0 < OW - Y_fwr);

	float2 y;
	zero_float(y.x, ly0, tex1Dfetch<float>(deltaY, Y0 + yoffset));
	zero_float(y.y, ly0, tex1Dfetch<float>(deltaY, Y1 + yoffset));
	return y;
}

#endif

#endif


#ifndef MICRO_KERNEL_SPLIT_V2
#define MICRO_KERNEL_SPLIT_V2

#define Ims2_CAN_W5(FH, FW, OH, OW) ((FH == 5) && (FW == 5) && (OH >= 2) && (OW >= 2))

//(OHp - FH + 2oph) + 1 = IH
//(OH + 2oph - CFH)/1 + 1 = 2oph + OH - CFH + 1 = IH_slice
//[1] Oph + OH = IH_slice + CFH - 1
//[2] Opw = OW = IW_slice + CFW - 1
//Q = avg: (Oph + OH) * (Opw + OW) / (OH * OW)
#define Ims2_PADDING_SCALE_UP(Q, IH_slice, IW_slice, OH, OW, FH, FW) {\
	int OH0 = (IH_slice) + ((FH + 1) >> 1) - 1;\
	int OW0 = (IW_slice) + ((FW + 1) >> 1) - 1;\
	int OH1 = (IH_slice) + (FH >> 1) - 1;\
	int OW1 = (IW_slice) + (FW >> 1) - 1;\
	Q = 0.25 * ((OH0 + OH1) * (OW0 + OW1)) / (OH * OW); }


//V1.GMr = V2.Nr * IH_slice * IW_slice
//As: GM = V1.GMr + V1.j_index = (n_index + V2.Nr) * IH_slice * IW_slice
//       = N * IH_slice * IW_slice
//[OH = OW = 8]: Q = 1.0625
//[OH = OW = 4]: Q = 1.125 *
//[OH = OW = 2]: Q = 1.25
#define KS_V2_TO_V1(FH, FW, IH_slice, IW_slice, N, OC, n_index) \
	const int IH_IW_slice = IH_slice * IW_slice;\
	int GM = N * IH_IW_slice;\
	int j_index = n_index * IH_IW_slice;

//(CFH, CFW): the max (CFH, CFW)
#define V2_Ims2_init(N, IH, IW, FH, FW, OC, IC) \
	int CFH = Ims2_CFH(FH);\
	int CFW = Ims2_CFW(FW);\
	int IH_slice = Ims2_IH_slice(IH);\
	int IW_slice = Ims2_IW_slice(IW);\
	int CWstride = Ims_CWstride(CFH, CFW, OC, IC);\

#endif


//IH % sh == 0, IW % sw == 0
#ifndef MICRO_KERNEL_SPLIT_INPUT_MOD_STEP
#define MICRO_KERNEL_SPLIT_INPUT_MOD_STEP

#define Ims_IH_slice(IH, sh) ((IH)/(sh)) //IH_slice = (IH + sh - 1) / sh
#define Ims_IW_slice(IW, sw) ((IW)/(sw)) //IW_slice = (IW + sw - 1) / sw
#define Ims_CWstride(CFH, CFW, OC, IC) (CFH * CFW * OC * IC)

#define Ims_GN(IC) (IC)
#define Ims_GM(N, IH, IW, sh, sw) ((N)*(IH/sh)*(IW/sw)) //GM = N*IH_slice*IW_slice

#define Ims_CFH(FH, sh) ((FH + sh - 1) / sh)
#define Ims_CFW(FW, sw) ((FW + sw - 1) / sw)

//(CFH, CFW): the max (CFH, CFW)
#define Ims_init(N, IH, IW, FH, FW, OC, IC, sh, sw) \
	int CFH = Ims_CFH(FH, sh);\
	int CFW = Ims_CFW(FW, sw);\
	int IH_slice = Ims_IH_slice(IH, sh);\
	int IW_slice = Ims_IW_slice(IW, sw);\
	int CWstride = Ims_CWstride(CFH, CFW, OC, IC);\
	int GN = IC; \
	int GM = N * IH_slice * IW_slice;


//======Impovement for Xoffset(n, oh, ow)======================================
//------[Part1]----------------------------------------------------------------
//when (IH_slice, IW_slice) > 8
//xoffset[i] = ((ni*IH + ihi*sh + ihs)*IW + iwi*sw + iws)*IC + ic
//xoffset[i] = ((ni*IH + ihi*sh)*IW + iwi*sw)*IC + ic + (ihs*IW + iws)*IC
//let: C1 = (ihs*IW + iws)*IC
//let: Ui = (ni*IH + ihi*sh)*IW + iwi*sw
//xoffset[i] = Ui*IC + ic + C1
//
//Ui = (ni*IH + ihi*sh)*IW + iwi*sw
//Ui = (ni*IH_slice*sh + ihi*sh)*IW_slice*sw + iwi*sw
//As: IH_slice = IH/sh, IW_slice = IW/sw
//Ui = ni*IH_slice*IW_slice*sh*sw + ihi*IW_slice*sh*sw + iwi*sw
//Ui = sh*sw*(ni*IH_slice*IW_slice + ihi*IW_slice) + iwi*sw
//As: ji = ni*IH_slice*IW_slice + ihi*IW_slice + iwi
//we have:  ni*IH_slice*IW_slice + ihi*IW_slice = ji - iwi
//Ui = sh*sw*(ji - iwi) + iwi*sw
//Ui = sh*sw*ji - sh*sw*iwi + iwi*sw
//Ui = sh*sw*ji + iwi*sw*(1 - sh)
//As: iwi = ji % IW_slice
//Ui = sh*sw*ji + (ji % IW_slice)*sw*(1 - sh)
//Ui = sw*{sh*ji + (1 - sh)* (ji % IW_slice) }
//
//xoffset[i] = Ui*IC + ic + C1
//xoffset[i] = sw*{sh*ji + (1 - sh)* (ji % IW_slice) }*IC + ic + (ihs*IW + iws)*IC
//
//xoffset[i] = sw*{sh*ji + (1 - sh)* (ji % IW_slice) }*IC + ic + (ihs*IW + iws)*IC
//let: deltaX += ic0 + (ihs*IW + iws)*IC
//let: IC = IC*sw
//we have: xoffset[i] = (sh*ji + (1 - sh)* (ji % IW_slice))*IC;
//
//let: alpha = sh*IC, beta = (1-sh)*IC
//xoffset[i] = alpha*ji + beta*(j0 % IW_slice)
//In conclution:
//(1) deltaX += ic0 + (ihs*IW + iws)*IC
//(2) IC = IC*sw
//(3) alpha = sh*IC, beta = (1-sh)*IC
//(4) xoffset[i] = alpha*ji + beta*(ji % IW_slice)
//
//especilla, when sh = sw = 2
//(1) deltaX += ic0 + (ihs*IW + iws)*IC
//(2) alpha = IC * 4 =  IC << 2
//(3) beta = -2 * IC = -IC << 1
//(4) xoffset[i] = alpha*ji + beta*(ji % IW_slice)
//
//------[Part2]----------------------------------------------------------------
//xoffset = ((n*IH + ih*sh + ihs)*IW + iw*sw + iws)*IC + ic
//xoffset0 = ((n0*IH + ih0*sh + ihs)*IW + iw0*sw + iws)*IC + ic0
//let: U0 = (n0*IH + (ih0*sh + ihs))*IW + (iw0*sw + iws)
//xoffseti = Ui*IC + ic0;
//U0 = n0*IH*IW + (ih0*sh + ihs)*IW + (iw0*sw + iws)
//U0 = n0*IH*IW + ih0*sh*IW + iw0*sw + (ihs*IW + iws)
//as: IH%sh == 0, IW%sw == 0
//let: C = (ihs*IW + iws)
//U0 = n0*IH_slice*IW_slice*sh*sw + ih0*IW_slice*sh*sw + iw0*sw + C
//U0 = sh*sw*(n0*IH_slice*IW_slice + ih0*IW_slice) + iw0*sw + C
//As: j0 = n0*IH_slice*IW_slice + ih0*IW_slice + iw0
//U0 = sh*sw*(j0 - iw0) + iw0*sw + C
//U0 = sh*sw*j0 - sh*sw*iw0 + sw*iw0 + C
//As: ji = j0 + i,
//Ui = sh*sw*(j0 + i) - sh*sw*iwi + sw*iwi + C
//Ui = (sh*sw*j0 + C) + sh*sw*i - sh*sw*iwi + sw*iwi
//Let: G = (sh*sw*j0 + C)
//Ui = G + sh*sw(i - iwi) + sw*iwi
//
//As: iwi = (j0 + i) % IW_slice, 
//in k88: j0 % 8 == 0
//[1]: when IW_slice % 8 == 0, we have: iwi = iw0 + i
//so: Ui = G + sh*sw*(i - iw0 - i) + sw*(iw0 + i) = 
//    Ui = (G - sh*sw*iw0 + sw*iw0) + sw*i
//so: Ui = U0 + sw*i
//so: xoffset[i] = xoffset[i] + swi*IC
//[2]: when (IW_slice, IH_slice) % 8 == 0
//(1) we have: Ui = U0 + sw*i
//(2) we have: (IH_slice * IW_slice) % 64 == 0
//As: ni = ji / (IH_slice*IW_slice) = (j0 + i)/(IH_slice*IW_slice), As: i belongs to [0, 7]
//So: ni = (8*x + i)/(8*8*y) = 8*x / (8*8*y)
//So: ni = nj, i,j belongs to [0, 7]
//(3) we have: ji = ni*IH_slice*IW_slice + ihi*IW_slice + iwi
//ihi = ((j0 + i) % (IH_slice*IW_slice)) / IW_slice
//ihi = (j0 % (IH_slice*IW_slice) + i) / IW_slice, let: (j0 % (IH_slice*IW_slice) = V
//ihi = (V + i) / IW_slic, As: V % 8 == 0
//So: ihi = ihj, i,j belongs to [0, 7]
//
//int k88: j0 % 8 == 0
//[1]: when IW_slice % 4 == 0 && IH_slice % 4 == 0
//(1) ni = ji / (IH_slice*IW_slice) = (j0 + i)/(IH_slice*IW_slice), As: i belongs to [0, 7]
//So: ni = (8*x + i) / (4*4*y) = 8*x/(16*y)
//So: ni = nj, i,j belongs to [0, 7]
//(2) ih0 = ih1 = ih2 = ih3, ih4 = ih5 = ih6 = ih7
//As: ihi = (V + i) / IW_slice, As: V % 8 == 0
//ihi = (8*x + i) / (4 * y)
//(3) iw0 = iw1 - 1 = iw2 - 2 = iw3 - 3
//    iw4 = iw5 - 1 = iw6 - 2 = iw7 - 3
//======Impovement for Xoffset(n, oh, ow)======================================
#define Ims_n_ih_iw(j, n, ih, iw) \
	int n, ih, iw; {\
	n = (j) / IH_IW_slice; int jr = (j) - n*IH_IW_slice;\
	ih = jr / IW_slice, iw = jr - ih*IW_slice;\
	ih = (ih * sh) + ihs; iw = (iw * sw) + iws; }

//======Impovement for Xoffset(oh, ow, n)======================================
//-----[Part1] in k88----------------------------------------------------------
//N % 4 == 0, j0 % 8 == 0
//(1) n = (j0 + i) % N = (8x + i) / 4y
//So: n0 = n1 - 1 = n2 - 2 = n3 - 3
//So: n4 = n5 - 1 = n6 - 2 = n7 - 3
//(2) ihi = (j0 + i) / (N * IW_slice)
//    ihi = (8x + i)/4y
//So: ih0 = ih1 = ih2 = ih3
//So: ih4 = ih5 = ih6 = ih7
//(3) iwi = ((j0 + i) % (N * IW_slice)) / N
//    iwi = ((8x + i) % 4y) / 4z, i belongs to [0, 7]
//    iwi = (4u + i) / 4z, i belongs to [0, 3]
//So: iw0 = iw1 = iw2 = iw3;
//So: iw4 = iw5 = iw6 = iw7;
//(1) n = (j0 + i) % N
//So: n0 = n1 - 1 = n2 - 2 = n3 - 3
//So: n4 = n5 - 1 = n6 - 2 = n7 - 3
//
//-----[Part2] in k44----------------------------------------------------------
//N % 4 == 0, j0 % 4 == 0
//(1) n = (j0 + i) % N = (4x + i) / 4y
//So: n0 = n1 - 1 = n2 - 2 = n3 - 3
//(2) ihi = (j0 + i) / (N * IW_slice)
//    ihi = (4x + i)/4y
//So: ih0 = ih1 = ih2 = ih3
//(3) iwi = ((j0 + i) % (N * IW_slice)) / N
//    iwi = ((4x + i) % 4y) / 4z, i belongs to [0, 7]
//    iwi = (4u + i) / 4z, i belongs to [0, 3]
//So: iw0 = iw1 = iw2 = iw3;
//======Impovement for Xoffset(n, oh, ow)======================================
#define Ims_ih_iw_n(j, ih, iw, n) \
	int n, ih, iw; {\
	ih = (j) / IW_slice_N; int jr = (j) - ih*IW_slice_N;\
	iw = jr / N; n = jr - iw * N;\
	ih = (ih * sh) + ihs; iw = (iw * sw) + iws; }

#define Ims_ldy(ohs, ows) \
	((ohs >= -dY_fhr) && (ohs < OH-dY_fhr) && (ows>=-dY_fwr) && (ows < OW-dY_fwr))

#define Ims_ldy_ows(ows) \
	((ows>=-dY_fwr) && (ows < OW-dY_fwr))


//(fhr, fwr, oc)
#define Ims_fhr_fwr(k, fhr, fwr) int fhr, fwr;\
	{fhr = k / CFW_OC; k -= fhr * CFW_OC; fwr = k / OC; }

#define Ims_fhr_fwr_oc2pow(k, fhr, fwr) int fhr, fwr;\
	{fhr = k / CFW_OC; k -= fhr * CFW_OC; fwr = (k >> LOC); }

#define Ims_fhr_fwr_oc_CFW2pow(k, fhr, fwr) int fhr, fwr;\
	{fhr = k >> LCFW_OC; k &= CFW_OC_m1; fwr = (k >> LOC); }

#define Ims_fhr_fwr_W3_oc2pow(k, fhr, fwr) int fhr, fwr;\
	{fhr = k >> (LOC + 1); k &= CFW_OC_m1; fwr = (k >> LOC); }


#ifndef IMS_SAVE_YS4
#define IMS_SAVE_YS4

__device__ __forceinline__ float4 Ims_SaveYs4(const float* __restrict__ deltaY,
	int yoffset, int Y_fhr, int Y_fwr, int OH, int OW,
	int Y0, int tohs0, int tows0,
	int Y1, int tohs1, int tows1,
	int Y2, int tohs2, int tows2,
	int Y3, int tohs3, int tows3)
{
	OH -= Y_fhr; OW -= Y_fwr;
	bool ly0 = (tohs0 >= -Y_fhr) && (tohs0 < OH) && (tows0 >= -Y_fwr) && (tows0 < OW);
	bool ly1 = (tohs1 >= -Y_fhr) && (tohs1 < OH) && (tows1 >= -Y_fwr) && (tows1 < OW);
	bool ly2 = (tohs2 >= -Y_fhr) && (tohs2 < OH) && (tows2 >= -Y_fwr) && (tows2 < OW);
	bool ly3 = (tohs3 >= -Y_fhr) && (tohs3 < OH) && (tows3 >= -Y_fwr) && (tows3 < OW);

	float4 y;
	y.x = (ly0 ? deltaY[Y0 + yoffset] : 0);
	y.y = (ly1 ? deltaY[Y1 + yoffset] : 0);
	y.z = (ly2 ? deltaY[Y2 + yoffset] : 0);
	y.w = (ly3 ? deltaY[Y3 + yoffset] : 0);
	return y;
}

__device__ __forceinline__ float4 Ims4x_SaveYs4(const float* __restrict__ deltaY,
	int yoffset, int Y_fhr, int Y_fwr, int OH_m_tohs0, int OW, int OC, 
	int tohs0, int tows0, int tows1, int tows2, int tows3)
{
	bool ly = (tohs0 >= -Y_fhr) && (Y_fhr < OH_m_tohs0); OW -= Y_fwr;
	bool ly0 = ly && (tows0 >= -Y_fwr) && (tows0 < OW);
	bool ly1 = ly && (tows1 >= -Y_fwr) && (tows1 < OW);
	bool ly2 = ly && (tows2 >= -Y_fwr) && (tows2 < OW);
	bool ly3 = ly && (tows3 >= -Y_fwr) && (tows3 < OW);

	float4 y;
	y.x = (ly0 ? deltaY[yoffset - OC] : 0);
	y.y = (ly1 ? deltaY[yoffset] : 0);
	y.z = (ly2 ? deltaY[yoffset + OC] : 0);
	y.w = (ly3 ? deltaY[yoffset + (OC << 1)] : 0);
	return y;
}

#endif


#ifndef IMS_SAVE_YS4_TEXTURE
#define IMS_SAVE_YS4_TEXTURE

__device__ __forceinline__ float4 Ims4x_SaveYs4_tex(
	cudaTextureObject_t deltaY,
	int yoffset, int Y_fhr, int Y_fwr, int OH_m_tohs0, int OW, int OC,
	int tohs0, int tows0, int tows1, int tows2, int tows3)
{
	float4 y;
	y.x = tex1Dfetch<float>(deltaY, yoffset - OC);
	y.y = tex1Dfetch<float>(deltaY, yoffset);
	y.z = tex1Dfetch<float>(deltaY, yoffset + OC);
	y.w = tex1Dfetch<float>(deltaY, yoffset + (OC << 1));

	bool ly = (tohs0 >= -Y_fhr) && (Y_fhr < OH_m_tohs0); OW -= Y_fwr;
	bool ly0 = ly && (tows0 >= -Y_fwr) && (tows0 < OW); 
	bool ly1 = ly && (tows1 >= -Y_fwr) && (tows1 < OW); 
	bool ly2 = ly && (tows2 >= -Y_fwr) && (tows2 < OW);
	bool ly3 = ly && (tows3 >= -Y_fwr) && (tows3 < OW); 
	zero_float(y.x, ly0, y.x);
	zero_float(y.y, ly1, y.y);
	zero_float(y.z, ly2, y.z);
	zero_float(y.w, ly3, y.w);
	return y;
}

__device__ __forceinline__ float4 Ims_SaveYs4_tex(
	cudaTextureObject_t deltaY, int yoffset, 
	int Y_fhr, int Y_fwr, int OH, int OW,
	int Y0, int tohs0, int tows0,
	int Y1, int tohs1, int tows1,
	int Y2, int tohs2, int tows2,
	int Y3, int tohs3, int tows3)
{
	float4 y;
	y.x = tex1Dfetch<float>(deltaY, Y0 + yoffset);
	y.y = tex1Dfetch<float>(deltaY, Y1 + yoffset);
	y.z = tex1Dfetch<float>(deltaY, Y2 + yoffset);
	y.w = tex1Dfetch<float>(deltaY, Y3 + yoffset);

	OH -= Y_fhr; OW -= Y_fwr;
	bool ly0 = (tohs0 >= -Y_fhr) && (tohs0 < OH) && (tows0 >= -Y_fwr) && (tows0 < OW);
	bool ly1 = (tohs1 >= -Y_fhr) && (tohs1 < OH) && (tows1 >= -Y_fwr) && (tows1 < OW);
	bool ly2 = (tohs2 >= -Y_fhr) && (tohs2 < OH) && (tows2 >= -Y_fwr) && (tows2 < OW);
	bool ly3 = (tohs3 >= -Y_fhr) && (tohs3 < OH) && (tows3 >= -Y_fwr) && (tows3 < OW);
	zero_float(y.x, ly0, y.x);
	zero_float(y.y, ly1, y.y);
	zero_float(y.z, ly2, y.z);
	zero_float(y.w, ly3, y.w);
	return y;
}


#endif


//for k88
#ifndef IMS_LOAD_YS4
#define IMS_LOAD_YS4

__device__ __forceinline__ float4 Ims_loadYs4(const float* __restrict__ deltaY,
	int Y_k, int OH, int OW, int OC, int CFW_OC, int OW_OC,
	int Y0, int tohs0, int tows0,
	int Y1, int tohs1, int tows1,
	int Y2, int tohs2, int tows2,
	int Y3, int tohs3, int tows3)
{
	Ims_fhr_fwr(Y_k, Y_fhr, Y_fwr);
	int yoffset = Y_fhr * OW_OC + Y_k;
	bool ly0 = (tohs0 >= -Y_fhr) && (tohs0 < OH - Y_fhr) && (tows0 >= -Y_fwr) && (tows0 < OW - Y_fwr);
	bool ly1 = (tohs1 >= -Y_fhr) && (tohs1 < OH - Y_fhr) && (tows1 >= -Y_fwr) && (tows1 < OW - Y_fwr);
	bool ly2 = (tohs2 >= -Y_fhr) && (tohs2 < OH - Y_fhr) && (tows2 >= -Y_fwr) && (tows2 < OW - Y_fwr);
	bool ly3 = (tohs3 >= -Y_fhr) && (tohs3 < OH - Y_fhr) && (tows3 >= -Y_fwr) && (tows3 < OW - Y_fwr);

	float4 y;
	y.x = (ly0 ? deltaY[Y0 + yoffset] : 0);
	y.y = (ly1 ? deltaY[Y1 + yoffset] : 0);
	y.z = (ly2 ? deltaY[Y2 + yoffset] : 0);
	y.w = (ly3 ? deltaY[Y3 + yoffset] : 0);
	return y;
}

__device__ __forceinline__ float4 Ims4x_loadYs4(const float* __restrict__ deltaY,
	int Y_k, int OH_m_tohs0, int OW, int OC, int CFW_OC, int OW_OC,
	int Y1, int tohs0, int tows0,
	int tows1, int tows2, int tows3)
{
	Ims_fhr_fwr(Y_k, Y_fhr, Y_fwr);
	int yoffset = Y_fhr * OW_OC + Y_k;
	bool ly = (tohs0 >= -Y_fhr) && (Y_fhr < OH_m_tohs0);
	bool ly0 = ly && (tows0 >= -Y_fwr) && (tows0 < OW - Y_fwr);
	bool ly1 = ly && (tows1 >= -Y_fwr) && (tows1 < OW - Y_fwr);
	bool ly2 = ly && (tows2 >= -Y_fwr) && (tows2 < OW - Y_fwr);
	bool ly3 = ly && (tows3 >= -Y_fwr) && (tows3 < OW - Y_fwr);

	float4 y;
	y.x = (ly0 ? deltaY[Y1 - OC + yoffset] : 0);
	y.y = (ly1 ? deltaY[Y1 + yoffset] : 0);
	y.z = (ly2 ? deltaY[Y1 + OC + yoffset] : 0);
	y.w = (ly3 ? deltaY[Y1 + (OC << 1) + yoffset] : 0);
	return y;
}

#endif


//for k88_oc2pow
#ifndef IMS_LOAD_YS4_OC_2POW
#define IMS_LOAD_YS4_OC_2POW

__device__ __forceinline__ float4 Ims_loadYs4_oc2pow(
	const float* __restrict__ deltaY,
	int Y_k, int OH, int OW, int LOC, int CFW_OC,
	int Y0, int tohs0, int tows0,
	int Y1, int tohs1, int tows1,
	int Y2, int tohs2, int tows2,
	int Y3, int tohs3, int tows3)
{
	Ims_fhr_fwr_oc2pow(Y_k, Y_fhr, Y_fwr);
	int yoffset = (Y_fhr << LOC) * OW + Y_k;
	bool ly0 = (tohs0 >= -Y_fhr) && (tohs0 < OH - Y_fhr) && (tows0 >= -Y_fwr) && (tows0 < OW - Y_fwr);
	bool ly1 = (tohs1 >= -Y_fhr) && (tohs1 < OH - Y_fhr) && (tows1 >= -Y_fwr) && (tows1 < OW - Y_fwr);
	bool ly2 = (tohs2 >= -Y_fhr) && (tohs2 < OH - Y_fhr) && (tows2 >= -Y_fwr) && (tows2 < OW - Y_fwr);
	bool ly3 = (tohs3 >= -Y_fhr) && (tohs3 < OH - Y_fhr) && (tows3 >= -Y_fwr) && (tows3 < OW - Y_fwr);

	float4 y;
	y.x = (ly0 ? deltaY[Y0 + yoffset] : 0);
	y.y = (ly1 ? deltaY[Y1 + yoffset] : 0);
	y.z = (ly2 ? deltaY[Y2 + yoffset] : 0);
	y.w = (ly3 ? deltaY[Y3 + yoffset] : 0);
	return y;
}

__device__ __forceinline__ float4 Ims4x_loadYs4_oc2pow(
	const float* __restrict__ deltaY,
	int Y_k, int OH_m_tohs0, int OW, int LOC, int CFW_OC,
	int Y1, int tohs0, int tows0,
	int tows1, int tows2, int tows3)
{
	Ims_fhr_fwr_oc2pow(Y_k, Y_fhr, Y_fwr);
	int yoffset = (Y_fhr << LOC) * OW + Y_k;
	bool ly = (tohs0 >= -Y_fhr) && (Y_fhr < OH_m_tohs0);
	bool ly0 = ly && (tows0 >= -Y_fwr) && (tows0 < OW - Y_fwr);
	bool ly1 = ly && (tows1 >= -Y_fwr) && (tows1 < OW - Y_fwr);
	bool ly2 = ly && (tows2 >= -Y_fwr) && (tows2 < OW - Y_fwr);
	bool ly3 = ly && (tows3 >= -Y_fwr) && (tows3 < OW - Y_fwr);

	float4 y;
	y.x = (ly0 ? deltaY[Y1 - (1 << LOC) + yoffset] : 0);
	y.y = (ly1 ? deltaY[Y1 + yoffset] : 0);
	y.z = (ly2 ? deltaY[Y1 + (1 << LOC) + yoffset] : 0);
	y.w = (ly3 ? deltaY[Y1 + (2 << LOC) + yoffset] : 0);
	return y;
}

#endif


//for k88_oc_CFW2pow
#ifndef IMS_LOAD_YS4_OC_CFW_2POW
#define IMS_LOAD_YS4_OC_CFW_2POW

__device__ __forceinline__ float4 Ims_loadYs4_oc_CFW2pow(
	const float* __restrict__ deltaY,
	int Y_k, int OH, int OW, int LOC, int LCFW_OC, int CFW_OC_m1,
	int Y0, int tohs0, int tows0,
	int Y1, int tohs1, int tows1,
	int Y2, int tohs2, int tows2,
	int Y3, int tohs3, int tows3)
{
	Ims_fhr_fwr_oc_CFW2pow(Y_k, Y_fhr, Y_fwr);
	int yoffset = (Y_fhr << LOC) * OW + Y_k;
	bool ly0 = (tohs0 >= -Y_fhr) && (tohs0 < OH - Y_fhr) && (tows0 >= -Y_fwr) && (tows0 < OW - Y_fwr);
	bool ly1 = (tohs1 >= -Y_fhr) && (tohs1 < OH - Y_fhr) && (tows1 >= -Y_fwr) && (tows1 < OW - Y_fwr);
	bool ly2 = (tohs2 >= -Y_fhr) && (tohs2 < OH - Y_fhr) && (tows2 >= -Y_fwr) && (tows2 < OW - Y_fwr);
	bool ly3 = (tohs3 >= -Y_fhr) && (tohs3 < OH - Y_fhr) && (tows3 >= -Y_fwr) && (tows3 < OW - Y_fwr);

	float4 y;
	y.x = (ly0 ? deltaY[Y0 + yoffset] : 0);
	y.y = (ly1 ? deltaY[Y1 + yoffset] : 0);
	y.z = (ly2 ? deltaY[Y2 + yoffset] : 0);
	y.w = (ly3 ? deltaY[Y3 + yoffset] : 0);
	return y;
}


__device__ __forceinline__ float4 Ims4x_loadYs4_oc_CFW2pow(
	const float* __restrict__ deltaY,
	int Y_k, int OH_m_tohs0, int OW, int LOC, int LCFW_OC, int CFW_OC_m1,
	int Y1, int tohs0, 
	int tows0, int tows1, int tows2, int tows3)
{
	Ims_fhr_fwr_oc_CFW2pow(Y_k, Y_fhr, Y_fwr);
	int yoffset = (Y_fhr << LOC) * OW + Y_k; OW -= Y_fwr;
	bool ly = (tohs0 >= -Y_fhr) && (Y_fhr < OH_m_tohs0);
	bool ly0 = ly && (tows0 >= -Y_fwr) && (tows0 < OW); 
	bool ly1 = ly && (tows1 >= -Y_fwr) && (tows1 < OW); 
	bool ly2 = ly && (tows2 >= -Y_fwr) && (tows2 < OW); 
	bool ly3 = ly && (tows3 >= -Y_fwr) && (tows3 < OW);

	float4 y;
	y.x = (ly0 ? deltaY[Y1 - (1 << LOC) + yoffset] : 0);
	y.y = (ly1 ? deltaY[Y1 + yoffset] : 0);
	y.z = (ly2 ? deltaY[Y1 + (1 << LOC) + yoffset] : 0);
	y.w = (ly3 ? deltaY[Y1 + (2 << LOC) + yoffset] : 0);
	return y;
}

#endif


//for k88_oc_CFW2pow_tex
#ifndef IMS_LOAD_YS4_OC_CFW_2POW_TEXTURE
#define IMS_LOAD_YS4_OC_CFW_2POW_TEXTURE

__device__ __forceinline__ float4 Ims_loadYs4_oc_CFW2pow_tex(
	cudaTextureObject_t deltaY,
	int Y_k, int OH, int OW, int LOC, int LCFW_OC, int CFW_OC_m1,
	int Y0, int tohs0, int tows0,
	int Y1, int tohs1, int tows1,
	int Y2, int tohs2, int tows2,
	int Y3, int tohs3, int tows3)
{
	Ims_fhr_fwr_oc_CFW2pow(Y_k, Y_fhr, Y_fwr);
	int yoffset = (Y_fhr << LOC) * OW + Y_k;
	bool ly0 = (tohs0 >= -Y_fhr) && (tohs0 < OH - Y_fhr) && (tows0 >= -Y_fwr) && (tows0 < OW - Y_fwr);
	bool ly1 = (tohs1 >= -Y_fhr) && (tohs1 < OH - Y_fhr) && (tows1 >= -Y_fwr) && (tows1 < OW - Y_fwr);
	bool ly2 = (tohs2 >= -Y_fhr) && (tohs2 < OH - Y_fhr) && (tows2 >= -Y_fwr) && (tows2 < OW - Y_fwr);
	bool ly3 = (tohs3 >= -Y_fhr) && (tohs3 < OH - Y_fhr) && (tows3 >= -Y_fwr) && (tows3 < OW - Y_fwr);

	float4 y;
	zero_float(y.x, ly0, tex1Dfetch<float>(deltaY, Y0 + yoffset));
	zero_float(y.y, ly1, tex1Dfetch<float>(deltaY, Y1 + yoffset));
	zero_float(y.z, ly2, tex1Dfetch<float>(deltaY, Y2 + yoffset));
	zero_float(y.w, ly3, tex1Dfetch<float>(deltaY, Y3 + yoffset));
	return y;
}


__device__ __forceinline__ float4 Ims4x_loadYs4_oc_CFW2pow_tex(
	cudaTextureObject_t deltaY,
	int Y_k, int OH_m_tohs0, int OW, int LOC, int LCFW_OC, int CFW_OC_m1,
	int Y1, int tohs0,
	int tows0, int tows1, int tows2, int tows3)
{
	Ims_fhr_fwr_oc_CFW2pow(Y_k, Y_fhr, Y_fwr);
	int yoffset = (Y_fhr << LOC) * OW + Y_k; OW -= Y_fwr;
	bool ly = (tohs0 >= -Y_fhr) && (Y_fhr < OH_m_tohs0);
	bool ly0 = ly && (tows0 >= -Y_fwr) && (tows0 < OW);
	bool ly1 = ly && (tows1 >= -Y_fwr) && (tows1 < OW);
	bool ly2 = ly && (tows2 >= -Y_fwr) && (tows2 < OW);
	bool ly3 = ly && (tows3 >= -Y_fwr) && (tows3 < OW);

	float4 y;
	zero_float(y.x, ly0, tex1Dfetch<float>(deltaY, Y1 - (1 << LOC) + yoffset));
	zero_float(y.y, ly1, tex1Dfetch<float>(deltaY, Y1 + yoffset));
	zero_float(y.z, ly2, tex1Dfetch<float>(deltaY, Y1 + (1 << LOC) + yoffset));
	zero_float(y.w, ly3, tex1Dfetch<float>(deltaY, Y1 + (2 << LOC) + yoffset));
	return y;

	//Ims_fhr_fwr_oc_CFW2pow(Y_k, Y_fhr, Y_fwr);
	//float4 y; int yoffset = Y1 + (Y_fhr << LOC) * OW + Y_k; LOC = (1 << LOC);
	//y.x = tex1Dfetch<float>(deltaY, yoffset - LOC);
	//y.y = tex1Dfetch<float>(deltaY, yoffset);
	//y.z = tex1Dfetch<float>(deltaY, yoffset + LOC);
	//y.w = tex1Dfetch<float>(deltaY, yoffset + (LOC << 1));

	//OW -= Y_fwr;
	//bool ly = (tohs0 >= -Y_fhr) && (Y_fhr < OH_m_tohs0);
	//bool ly0 = ly && (tows0 >= -Y_fwr) && (tows0 < OW); zero_float(y.x, ly0, y.x);
	//bool ly1 = ly && (tows1 >= -Y_fwr) && (tows1 < OW); zero_float(y.y, ly1, y.y);
	//bool ly2 = ly && (tows2 >= -Y_fwr) && (tows2 < OW); zero_float(y.z, ly2, y.z);
	//bool ly3 = ly && (tows3 >= -Y_fwr) && (tows3 < OW); zero_float(y.w, ly3, y.w);
	//return y;
}

#endif


//for k88_tex
#ifndef IMS_LOAD_YS4_TEXTURE
#define IMS_LOAD_YS4_TEXTURE

__device__ __forceinline__ float4 Ims_loadYs4_tex(
	cudaTextureObject_t deltaY,
	int Y_k, int OH, int OW, int OC, int CFW_OC, int OW_OC,
	int Y0, int tohs0, int tows0,
	int Y1, int tohs1, int tows1,
	int Y2, int tohs2, int tows2,
	int Y3, int tohs3, int tows3)
{
	Ims_fhr_fwr(Y_k, Y_fhr, Y_fwr);
	int yoffset = Y_fhr * OW_OC + Y_k;
	bool ly0 = (tohs0 >= -Y_fhr) && (tohs0 < OH - Y_fhr) && (tows0 >= -Y_fwr) && (tows0 < OW - Y_fwr);
	bool ly1 = (tohs1 >= -Y_fhr) && (tohs1 < OH - Y_fhr) && (tows1 >= -Y_fwr) && (tows1 < OW - Y_fwr);
	bool ly2 = (tohs2 >= -Y_fhr) && (tohs2 < OH - Y_fhr) && (tows2 >= -Y_fwr) && (tows2 < OW - Y_fwr);
	bool ly3 = (tohs3 >= -Y_fhr) && (tohs3 < OH - Y_fhr) && (tows3 >= -Y_fwr) && (tows3 < OW - Y_fwr);

	float4 y;
	zero_float(y.x, ly0, tex1Dfetch<float>(deltaY, Y0 + yoffset));
	zero_float(y.y, ly1, tex1Dfetch<float>(deltaY, Y1 + yoffset));
	zero_float(y.z, ly2, tex1Dfetch<float>(deltaY, Y2 + yoffset));
	zero_float(y.w, ly3, tex1Dfetch<float>(deltaY, Y3 + yoffset));
	return y;
}

__device__ __forceinline__ float4 Ims4x_loadYs4_tex(
	cudaTextureObject_t deltaY,
	int Y_k, int OH_m_tohs0, int OW, int OC, int CFW_OC, int OW_OC,
	int Y1, int tohs0, int tows0,
	int tows1, int tows2, int tows3)
{
	Ims_fhr_fwr(Y_k, Y_fhr, Y_fwr);
	int yoffset = Y_fhr * OW_OC + Y_k;
	bool ly = (tohs0 >= -Y_fhr) && (Y_fhr < OH_m_tohs0);
	bool ly0 = ly && (tows0 >= -Y_fwr) && (tows0 < OW - Y_fwr);
	bool ly1 = ly && (tows1 >= -Y_fwr) && (tows1 < OW - Y_fwr);
	bool ly2 = ly && (tows2 >= -Y_fwr) && (tows2 < OW - Y_fwr);
	bool ly3 = ly && (tows3 >= -Y_fwr) && (tows3 < OW - Y_fwr);

	float4 y;
	zero_float(y.x, ly0, tex1Dfetch<float>(deltaY, Y1 - OC + yoffset));
	zero_float(y.y, ly1, tex1Dfetch<float>(deltaY, Y1 + yoffset));
	zero_float(y.z, ly2, tex1Dfetch<float>(deltaY, Y1 + OC + yoffset));
	zero_float(y.w, ly3, tex1Dfetch<float>(deltaY, Y1 + (OC << 1) + yoffset));
	return y;
}

#endif


//for k88_oc2pow_tex
#ifndef IMS_LOAD_YS4_OC_2POW_TEXTURE
#define IMS_LOAD_YS4_OC_2POW_TEXTURE

__device__ __forceinline__ float4 Ims_loadYs4_oc2pow_tex(
	cudaTextureObject_t deltaY,
	int Y_k, int OH, int OW, int LOC, int CFW_OC,
	int Y0, int tohs0, int tows0,
	int Y1, int tohs1, int tows1,
	int Y2, int tohs2, int tows2,
	int Y3, int tohs3, int tows3)
{
	Ims_fhr_fwr_oc2pow(Y_k, Y_fhr, Y_fwr);
	int yoffset = (Y_fhr << LOC) * OW + Y_k;
	bool ly0 = (tohs0 >= -Y_fhr) && (tohs0 < OH - Y_fhr) && (tows0 >= -Y_fwr) && (tows0 < OW - Y_fwr);
	bool ly1 = (tohs1 >= -Y_fhr) && (tohs1 < OH - Y_fhr) && (tows1 >= -Y_fwr) && (tows1 < OW - Y_fwr);
	bool ly2 = (tohs2 >= -Y_fhr) && (tohs2 < OH - Y_fhr) && (tows2 >= -Y_fwr) && (tows2 < OW - Y_fwr);
	bool ly3 = (tohs3 >= -Y_fhr) && (tohs3 < OH - Y_fhr) && (tows3 >= -Y_fwr) && (tows3 < OW - Y_fwr);

	float4 y;
	zero_float(y.x, ly0, tex1Dfetch<float>(deltaY, Y0 + yoffset));
	zero_float(y.y, ly1, tex1Dfetch<float>(deltaY, Y1 + yoffset));
	zero_float(y.z, ly2, tex1Dfetch<float>(deltaY, Y2 + yoffset));
	zero_float(y.w, ly3, tex1Dfetch<float>(deltaY, Y3 + yoffset));
	return y;
}

__device__ __forceinline__ float4 Ims4x_loadYs4_oc2pow_tex(
	cudaTextureObject_t deltaY,
	int Y_k, int OH_m_tohs0, int OW, int LOC, int CFW_OC,
	int Y1, int tohs0, int tows0,
	int tows1, int tows2, int tows3)
{
	Ims_fhr_fwr_oc2pow(Y_k, Y_fhr, Y_fwr);
	int yoffset = (Y_fhr << LOC) * OW + Y_k;
	bool ly = (tohs0 >= -Y_fhr) && (Y_fhr < OH_m_tohs0);
	bool ly0 = ly && (tows0 >= -Y_fwr) && (tows0 < OW - Y_fwr);
	bool ly1 = ly && (tows1 >= -Y_fwr) && (tows1 < OW - Y_fwr);
	bool ly2 = ly && (tows2 >= -Y_fwr) && (tows2 < OW - Y_fwr);
	bool ly3 = ly && (tows3 >= -Y_fwr) && (tows3 < OW - Y_fwr);

	float4 y;
	zero_float(y.x, ly0, tex1Dfetch<float>(deltaY, Y1 - (1 << LOC) + yoffset));
	zero_float(y.y, ly1, tex1Dfetch<float>(deltaY, Y1 + yoffset));
	zero_float(y.z, ly2, tex1Dfetch<float>(deltaY, Y1 + (1 << LOC) + yoffset));
	zero_float(y.w, ly3, tex1Dfetch<float>(deltaY, Y1 + (2 << LOC) + yoffset));
	return y;
}

#endif


//for k44, k44_oc2pow
#ifndef IMS_LOAD_YS2
#define IMS_LOAD_YS2

__device__ __forceinline__ float2 Ims_loadYs2(
	const float* __restrict__ deltaY,
	int Y_k, int OH, int OW, int OC, int CFW_OC, int OW_OC,
	int Y0, int tohs0, int tows0,
	int Y1, int tohs1, int tows1)
{
	Ims_fhr_fwr(Y_k, Y_fhr, Y_fwr);
	int yoffset = Y_fhr * OW_OC + Y_k;
	bool ly0 = (tohs0 >= -Y_fhr) && (tohs0 < OH - Y_fhr) && (tows0 >= -Y_fwr) && (tows0 < OW - Y_fwr);
	bool ly1 = (tohs1 >= -Y_fhr) && (tohs1 < OH - Y_fhr) && (tows1 >= -Y_fwr) && (tows1 < OW - Y_fwr);

	float2 y;
	y.x = (ly0 ? deltaY[Y0 + yoffset] : 0);
	y.y = (ly1 ? deltaY[Y1 + yoffset] : 0);
	return y;
}

__device__ __forceinline__ float2 Ims_loadYs2_oc2pow(
	const float* __restrict__ deltaY,
	int Y_k, int OH, int OW, int LOC, int CFW_OC,
	int Y0, int tohs0, int tows0,
	int Y1, int tohs1, int tows1)
{
	Ims_fhr_fwr_oc2pow(Y_k, Y_fhr, Y_fwr);
	int yoffset = (Y_fhr << LOC) * OW + Y_k;
	bool ly0 = (tohs0 >= -Y_fhr) && (tohs0 < OH - Y_fhr) && (tows0 >= -Y_fwr) && (tows0 < OW - Y_fwr);
	bool ly1 = (tohs1 >= -Y_fhr) && (tohs1 < OH - Y_fhr) && (tows1 >= -Y_fwr) && (tows1 < OW - Y_fwr);

	float2 y;
	y.x = (ly0 ? deltaY[Y0 + yoffset] : 0);
	y.y = (ly1 ? deltaY[Y1 + yoffset] : 0);
	return y;
}

#endif


//for k44_tex, k44_oc2pow
#ifndef IMS_LOAD_YS2_TEX
#define IMS_LOAD_YS2_TEX

__device__ __forceinline__ float2 Ims_loadYs2_tex(
	cudaTextureObject_t deltaY,
	int Y_k, int OH, int OW, int OC, int CFW_OC, int OW_OC,
	int Y0, int tohs0, int tows0,
	int Y1, int tohs1, int tows1)
{
	Ims_fhr_fwr(Y_k, Y_fhr, Y_fwr);
	int yoffset = Y_fhr * OW_OC + Y_k;
	bool ly0 = (tohs0 >= -Y_fhr) && (tohs0 < OH - Y_fhr) && (tows0 >= -Y_fwr) && (tows0 < OW - Y_fwr);
	bool ly1 = (tohs1 >= -Y_fhr) && (tohs1 < OH - Y_fhr) && (tows1 >= -Y_fwr) && (tows1 < OW - Y_fwr);

	float2 y;
	zero_float(y.x, ly0, tex1Dfetch<float>(deltaY, Y0 + yoffset));
	zero_float(y.y, ly1, tex1Dfetch<float>(deltaY, Y1 + yoffset));
	return y;
}

__device__ __forceinline__ float2 Ims_loadYs2_oc2pow_tex(
	cudaTextureObject_t deltaY,
	int Y_k, int OH, int OW, int LOC, int CFW_OC,
	int Y0, int tohs0, int tows0,
	int Y1, int tohs1, int tows1)
{
	Ims_fhr_fwr_oc2pow(Y_k, Y_fhr, Y_fwr);
	int yoffset = (Y_fhr << LOC) * OW + Y_k;
	bool ly0 = (tohs0 >= -Y_fhr) && (tohs0 < OH - Y_fhr) && (tows0 >= -Y_fwr) && (tows0 < OW - Y_fwr);
	bool ly1 = (tohs1 >= -Y_fhr) && (tohs1 < OH - Y_fhr) && (tows1 >= -Y_fwr) && (tows1 < OW - Y_fwr);

	float2 y;
	zero_float(y.x, ly0, tex1Dfetch<float>(deltaY, Y0 + yoffset));
	zero_float(y.y, ly1, tex1Dfetch<float>(deltaY, Y1 + yoffset));
	return y;
}

#endif

#endif


//IH % sh == 0, IW % sw == 0, sh = sw = 2
#ifndef MICRO_KERNEL_SPLIT_INPUT_MOD_STEP2
#define MICRO_KERNEL_SPLIT_INPUT_MOD_STEP2

//As: CFW = (FW - x + 1) / 2, x belongs to {0, 1}
//LCFW = log2(CFW) = CFW >> 1
//when: FW = 2, CFW = (2 - 0 + 1)/2 = 1, CFW = (2 - 1 + 1)/2 = 1
//when: FW = 3, CFW = (3 - 0 + 1)/2 = 2, CFW = (3 - 1 + 1)/2 = 1
//when: FW = 4, CFW = (4 - 0 + 1)/2 = 2, CFW = (4 - 1 + 1)/2 = 2
#define Ims2_IS_CW_POWER2(FW) ((FW == 2) || (FW == 3) || (FW == 4))

#define Ims2_IH_slice(IH) ((IH) >> 1) //IH_slice = (IH + sh - 1) / sh
#define Ims2_IW_slice(IW) ((IW) >> 1) //IW_slice = (IW + sw - 1) / sw

#define Ims2_CFH(FH) ((FH + 1) >> 1) //((FH + sh - 1) / sh)
#define Ims2_CFW(FH) ((FW + 1) >> 1) // ((FW + sw - 1) / sw)

#define Ims2_GN(IC) (IC)
#define Ims2_GM(N, IH, IW) ((N)*(IH>>1)*(IW>>1)) //GM = N*IH_slice*IW_slice

//(CFH, CFW): the max (CFH, CFW)
#define Ims2_init(N, IH, IW, FH, FW, OC, IC) \
	int CFH = Ims2_CFH(FH);\
	int CFW = Ims2_CFW(FW);\
	int IH_slice = Ims2_IH_slice(IH);\
	int IW_slice = Ims2_IW_slice(IW);\
	int CWstride = Ims_CWstride(CFH, CFW, OC, IC);\
	int GN = IC; \
	int GM = N * IH_slice * IW_slice; \

#define Ims2_n_ih_iw(j, n, ih, iw) \
	int n, ih, iw; {\
	n = (j) / IH_IW_slice; int jr = (j) - n*IH_IW_slice;\
	ih = jr / IW_slice, iw = jr - ih*IW_slice;\
	ih = (ih << 1) + ihs; iw = (iw << 1) + iws; }

#define Ims2_ih_iw_n(j, ih, iw, n) \
	int ih, iw, n; {\
	ih = (j) / IW_slice_N; int jr = (j) - ih * IW_slice_N;\
	iw = jr / N; n = jr - iw * N;\
	ih = (ih << 1) + ihs; iw = (iw << 1) + iws; }

#endif


//IH % sh == 0, IW % sw == 0, CW(x, y) is power of 2 
#ifndef MICRO_KERNEL_SPLIT_INPUT_MOD_STEP_CW_2POW
#define MICRO_KERNEL_SPLIT_INPUT_MOD_STEP_CW_2POW

//(oc, fhr, fwr)============================================================
//<1> W_k & CFH_CFW_m1 = W_fhr*CFW + W_fwr
//<2> W_oc = W_k >> LCFH_LFW

#define Ims_oc_fhr_fwr(k, oc, fhr, fwr) int fhr, fwr, oc;\
	{ oc = k / CFH_CFW; k -= oc*CFH_CFW; fhr = k / CFW; fwr = k - fhr*CFW; }

#define Ims_oc_fhr_fwr_CW2pow(k, oc, fhr, fwr) int fhr, fwr, oc;\
	{ oc = k >> LCFH_CFW; k &= CFH_CFW_m1; fhr = k >> LCFW; fwr = k & opw; }

#define Ims_oc_CW2pow(k, oc) int fhr, fwr, oc;\
	{ oc = k >> LCFH_CFW; k &= CFH_CFW_m1;  }

//LCFH_CFW = 1 + 1 = 2, CFH_CFW_m1 = 2*2 - 1 = 3
#define Ims_oc_fhr_fwr_W3(k, oc, fhr, fwr) int fhr, fwr, oc;\
	{ oc = k >> 2; k &= 3; fhr = k >> 1; fwr = k & 1; }
//==========================================================================

#endif

#endif