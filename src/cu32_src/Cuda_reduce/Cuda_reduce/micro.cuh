#pragma once

#ifndef MICRO_H
#define MICRO_H

#define F32_2_0 float2{0, 0}
#define F32_4_0 float4{0, 0, 0, 0}
#define F64_2_0 double2{0, 0}
#define F64_4_0 double4{0, 0, 0, 0}

#define I32_2_0 int2{0, 0}
#define I32_4_0 int4{0, 0, 0, 0}

#define get(A, i, j, stride) A[(i)*(stride) + (j)]
#define lget(A, i, j, lstride) A[((i)<<(lstride)) + (j)] //lstride = log2(stride)


#ifndef NEXT_REDUCE_SIZE
#define NEXT_REDUCE_SIZE

//========[field reduce LBY >= 5]=================================================
//if (LBY >= 6) {//block reduce: 4 global result
//	if (ty < 64) {
//		int yIdx = ((ty & 31) << 1) + (ty >> 5);
//		float2 v1 = As[tx][yIdx], v2 = As[tx][yIdx + 64];//128 -> 64
//		simdAdd2(v1, v1, v2); As[tx][yIdx] = v1;
//	}
//	__syncthreads();
//}
//if (LBY >= 5) {
//	if (ty < 32) { 
//		int yIdx = ((ty & 15) << 1) + (ty >> 4);  
//		float2 v1 = As[tx][yIdx], v2 = As[tx][yIdx + 32];//64 -> 32
//		simdAdd2(v1, v1, v2); As[tx][yIdx] = v1;
//	}
//	__syncthreads();
//}
//========[field reduce LBY >= 5]=================================================

int straight_nextLengthV(int lengthv)  
{
	return (lengthv > 8192 ? (lengthv >> 13) : 1);
}

int row_nextM(int M)
{
	return (M > 255 ? (M >> 8) : 1);
}

int field_nextN(int N, int M)
{
	if (M > 15) {
		if (N > 63) return N >> 6;
		if (N > 15) return 1;
	}
	if (M > 7) {
		if (N > 127) return N >> 7;
		if (N > 31) return 1;
	}
	return (N + 63) >> 6;
}

#endif


#ifndef COMMON_MICRO
#define COMMON_MICRO

#define MAX_INT(x, y) ((x)^(((x)^(y))& -((x)<(y))))
#define MIN_INT(x, y) ((y)^(((x)^(y))& -((x)<(y))))

#define FLOAT_MAX (3.4028235E38f)
#define FLOAT_MIN (-3.4028234E38f)

#define MOVE_E(v) {(v) = (v) + 1e-9f - 2.0f*((v)<0)*1e-9f;}

#define INDEX4(index4)  make_int4((index4), (index4) + 1, (index4) + 2, (index4) + 3);


#define MIN_FLOAT4 make_float4(FLOAT_MIN, FLOAT_MIN, FLOAT_MIN, FLOAT_MIN)
#define MIN_FLOAT2 make_float2(FLOAT_MIN, FLOAT_MIN)

#define MAX_FLOAT4 make_float4(FLOAT_MAX, FLOAT_MAX, FLOAT_MAX, FLOAT_MAX)
#define MAX_FLOAT2 make_float2(FLOAT_MAX, FLOAT_MAX)


#define COPY4(a, b) {a.x = b.x; a.y=b.y; a.z=b.z; a.w=b.w; }

#define simdAdd4(c, a, b) {(c).x=(a).x+(b).x; (c).y=(a).y+(b).y; (c).z=(a).z+(b).z; (c).w=(a).w+(b).w;}
#define simdAdd2(c, a, b) {(c).x=(a).x+(b).x; (c).y=(a).y+(b).y;}

#define simdSub4(c, a, b) {(c).x=(a).x-(b).x; (c).y=(a).y-(b).y; (c).z=(a).z-(b).z; (c).w=(a).w-(b).w;}

#define SMEM_simdAdd2(c, a, b) { float2 v1 = a; float2 v2 = b; simdAdd2(v1, v1, v2); c = v1; }
#define SMEM_simdAdd4(c, a, b) { float4 v1 = a; float4 v2 = b; simdAdd4(v1, v1, v2); c = v1; }

#define SM64_simdAdd2(c, a, b) { double2 v1 = a; double2 v2 = b; simdAdd2(v1, v1, v2); c = v1; }
#define SM64_simdAdd4(c, a, b) { double4 v1 = a; double4 v2 = b; simdAdd4(v1, v1, v2); c = v1; }



#ifndef KUHAN_MICRO
#define KUHAN_MICRO

//v = v + a
#define Kahan_simdAdd4(v, a, c)	{\
	float4 dv;\
	dv.x = a.x - c.x;\
	dv.y = a.y - c.y;\
	dv.z = a.z - c.z;\
	dv.w = a.w - c.w;\
	float4 t;\
	t.x = v.x + dv.x;\
	t.y = v.y + dv.y;\
	t.z = v.z + dv.z;\
	t.w = v.w + dv.w;\
	c.x = (t.x - v.x) - dv.x;\
	c.y = (t.y - v.y) - dv.y;\
	c.z = (t.z - v.z) - dv.z;\
	c.w = (t.w - v.w) - dv.w;\
	v = t; }


//v += a.x + a.y + a.z + a.w;
#define Kahan_sum4(v, a, c) {\
	float dv = (a.x + a.y + a.z + a.w) - c;\
	float t = v + dv;\
	c = (t - v) - dv;\
	v = t; } 

//v += a
#define Kahan_sum1(v, a, c) {\
	float dv = a - c;\
	float t = v + dv;\
	c = (t - v) - dv;\
	v = t; } 

//sv1 = sv1 + sv2
//c: solve the error
//[sc_last, sc_next]: error of last_stage
#define SMEM_Kahan_simdAdd4(sv1, sv2, sc_last, sc_next) {\
	float4 v1 = sv1, v2 = sv2, c = sc_last;\
	float4 dv;\
	dv.x = v2.x - c.x;\
	dv.y = v2.y - c.y;\
	dv.z = v2.z - c.z;\
	dv.w = v2.w - c.w;\
	float4 t;\
	t.x = v1.x + dv.x;\
	t.y = v1.y + dv.y;\
	t.z = v1.z + dv.z;\
	t.w = v1.w + dv.w;\
	c.x = (t.x - v1.x) - dv.x;\
	c.y = (t.y - v1.y) - dv.y;\
	c.z = (t.z - v1.z) - dv.z;\
	c.w = (t.w - v1.w) - dv.w;\
	sv = t;\
	sc_next = c;}

#endif



#define simdQuadratic4(b, alpha, a, beta, gamma) {\
	(b).x = alpha *(a.x)*(a.x) + beta*(a.x) + gamma;\
	(b).y = alpha *(a.y)*(a.y) + beta*(a.y) + gamma;\
	(b).z = alpha *(a.z)*(a.z) + beta*(a.z) + gamma;\
	(b).w = alpha *(a.w)*(a.w) + beta*(a.w) + gamma;}

#define simdLinear4(a1, alpha, a0, beta) {\
	(a1).x = (a0).x *(alpha) + (beta);\
	(a1).y = (a0).y *(alpha) + (beta);\
	(a1).z = (a0).z *(alpha) + (beta);\
	(a1).w = (a0).w *(alpha) + (beta);}

#define simdMul4(c, a0, a1) {\
	(c).x = (a0).x * (a1).x;\
	(c).y = (a0).y * (a1).y;\
	(c).z = (a0).z * (a1).z;\
	(c).w = (a0).w * (a1).w;}

#define simdMax4(c, a, b) \
	{(c).x = fmaxf((a).x, (b).x);\
     (c).y = fmaxf((a).y, (b).y);\
     (c).z = fmaxf((a).z, (b).z);\
     (c).w = fmaxf((a).w, (b).w);}

#define simdMax2(c, a, b) \
	{(c).x = fmaxf((a).x, (b).x);\
     (c).y = fmaxf((a).y, (b).y);}\

#define simdMin4(c, a, b) \
	{(c).x=fminf((a).x, (b).x);\
     (c).y=fminf((a).y, (b).y);\
     (c).z=fminf((a).z, (b).z);\
     (c).w=fminf((a).w, (b).w);}

#define simdMin2(c, a, b) \
	{(c).x=fminf((a).x, (b).x);\
     (c).y=fminf((a).y, (b).y);}\

#define within_width4(v, index4, stride, width) {\
	v.x *= ((index4    ) % stride) < width;\
	v.y *= ((index4 + 1) % stride) < width;\
	v.z *= ((index4 + 2) % stride) < width;\
	v.w *= ((index4 + 3) % stride) < width;}

#define within_width2(v, index2, stride, width) {\
	v.x *= ((index2    ) % stride) < width;\
	v.y *= ((index2 + 1) % stride) < width;}

//use more resource, but can zero nan caused by zero: 1, if within with
#define within_width_zero_nan4(v, index4, table, stride, width) {\
	table[1] = v;\
	v.x = table[((index4    ) % stride) < width].x;\
	v.y = table[((index4 + 1) % stride) < width].y;\
	v.z = table[((index4 + 2) % stride) < width].z;\
	v.w = table[((index4 + 3) % stride) < width].w;}

#define within_width_zero_nan2(v, index2, table, stride, width) {\
	table[1] = v;\
	v.x = table[((index2    ) % stride) < width].x;\
	v.y = table[((index2 + 1) % stride) < width].y;}


#define EXCEED_width4_TO_MAX(v, index4, stride, width) {\
	int f0 = (index4    ) % stride < width;\
	int f1 = (index4 + 1) % stride < width;\
	int f2 = (index4 + 2) % stride < width;\
	int f3 = (index4 + 3) % stride < width;\
	v.x = !f0 * FLOAT_MAX + f0 * v.x;\
	v.y = !f1 * FLOAT_MAX + f1 * v.y;\
	v.z = !f2 * FLOAT_MAX + f2 * v.z;\
	v.w = !f3 * FLOAT_MAX + f3 * v.w;}

#define EXCEED_width2_TO_MAX(v, index2, stride, width) {\
	int f0 = (index4    ) % stride < width;\
	int f1 = (index4 + 1) % stride < width;\
	v.x = !f0 * FLOAT_MAX + f0 * v.x;\
	v.y = !f1 * FLOAT_MAX + f1 * v.y;}

#define EXCEED_width4_TO_MIN(v, index4, stride, width) {\
	int f0 = (index4    ) % stride < width;\
	int f1 = (index4 + 1) % stride < width;\
	int f2 = (index4 + 2) % stride < width;\
	int f3 = (index4 + 3) % stride < width;\
	v.x = !f0 * FLOAT_MIN + f0 * v.x;\
	v.y = !f1 * FLOAT_MIN + f1 * v.y;\
	v.z = !f2 * FLOAT_MIN + f2 * v.z;\
	v.w = !f3 * FLOAT_MIN + f3 * v.w;}

#define EXCEED_width2_TO_MIN(v, index2, stride, width) {\
	int f0 = (index4    ) % stride < width;\
	int f1 = (index4 + 1) % stride < width;\
	v.x = !f0 * FLOAT_MIN + f0 * v.x;\
	v.y = !f1 * FLOAT_MIN + f1 * v.y;}


#endif


#ifndef WARP_SUM
#define WARP_SUM

__device__ __forceinline__ void warp_sum_4(volatile float *sdata, int index) {
	sdata[index] += sdata[index + 4];
	sdata[index] += sdata[index + 2];
	sdata[index] += sdata[index + 1];
}

__device__ __forceinline__ void warp_sum_8(volatile float *sdata, int index) {
	sdata[index] += sdata[index + 8];
	sdata[index] += sdata[index + 4];
	sdata[index] += sdata[index + 2];
	sdata[index] += sdata[index + 1];
}

__device__ __forceinline__ void warp_sum_16(volatile float *sdata, int index) {
	sdata[index] += sdata[index + 16];
	sdata[index] += sdata[index + 8];
	sdata[index] += sdata[index + 4];
	sdata[index] += sdata[index + 2];
	sdata[index] += sdata[index + 1];
}

#endif


#ifndef WARP_SIMD_SUM2
#define WARP_SIMD_SUM2

__device__ __forceinline__ void warp_simdSum2_4(volatile float2 *sdata, int index) {
	simdAdd2(sdata[index], sdata[index], sdata[index + 4]);
	simdAdd2(sdata[index], sdata[index], sdata[index + 2]);
	simdAdd2(sdata[index], sdata[index], sdata[index + 1]);
}

__device__ __forceinline__ void warp_simdSum2_8(volatile float2 *sdata, int index) {
	simdAdd2(sdata[index], sdata[index], sdata[index + 8]);
	simdAdd2(sdata[index], sdata[index], sdata[index + 4]);
	simdAdd2(sdata[index], sdata[index], sdata[index + 2]);
	simdAdd2(sdata[index], sdata[index], sdata[index + 1]);
}

__device__ __forceinline__ void warp_simdSum2_16(volatile float2 *sdata, int index) {
	simdAdd2(sdata[index], sdata[index], sdata[index + 16]);
	simdAdd2(sdata[index], sdata[index], sdata[index + 8]);
	simdAdd2(sdata[index], sdata[index], sdata[index + 4]);
	simdAdd2(sdata[index], sdata[index], sdata[index + 2]);
	simdAdd2(sdata[index], sdata[index], sdata[index + 1]);
}

#endif


#ifndef WARP_MAX
#define WARP_MAX

__device__ __forceinline__ void warp_max_4(volatile float *sdata, int index) {
	sdata[index] = fmaxf(sdata[index], sdata[index + 4]);
	sdata[index] = fmaxf(sdata[index], sdata[index + 2]);
	sdata[index] = fmaxf(sdata[index], sdata[index + 1]);
}

__device__ __forceinline__ void warp_max_8(volatile float *sdata, int index) {
	sdata[index] = fmaxf(sdata[index], sdata[index + 8]);
	sdata[index] = fmaxf(sdata[index], sdata[index + 4]);
	sdata[index] = fmaxf(sdata[index], sdata[index + 2]);
	sdata[index] = fmaxf(sdata[index], sdata[index + 1]);
}

__device__ __forceinline__ void warp_max_16(volatile float *sdata, int index) {
	sdata[index] = fmaxf(sdata[index], sdata[index + 16]);
	sdata[index] = fmaxf(sdata[index], sdata[index + 8]);
	sdata[index] = fmaxf(sdata[index], sdata[index + 4]);
	sdata[index] = fmaxf(sdata[index], sdata[index + 2]);
	sdata[index] = fmaxf(sdata[index], sdata[index + 1]);
}

#endif 


#ifndef WARP_MIN
#define WARP_MIN

__device__ __forceinline__ void warp_min_4(volatile float *sdata, int index) {
	sdata[index] = fminf(sdata[index], sdata[index + 4]);
	sdata[index] = fminf(sdata[index], sdata[index + 2]);
	sdata[index] = fminf(sdata[index], sdata[index + 1]);
}

__device__ __forceinline__ void warp_min_8(volatile float *sdata, int index) {
	sdata[index] = fminf(sdata[index], sdata[index + 8]);
	sdata[index] = fminf(sdata[index], sdata[index + 4]);
	sdata[index] = fminf(sdata[index], sdata[index + 2]);
	sdata[index] = fminf(sdata[index], sdata[index + 1]);
}

__device__ __forceinline__ void warp_min_16(volatile float *sdata, int index) {
	sdata[index] = fminf(sdata[index], sdata[index + 16]);
	sdata[index] = fminf(sdata[index], sdata[index + 8]);
	sdata[index] = fminf(sdata[index], sdata[index + 4]);
	sdata[index] = fminf(sdata[index], sdata[index + 2]);
	sdata[index] = fminf(sdata[index], sdata[index + 1]);
}

#endif 


#ifndef WARP_MAX_INDEXED
#define WARP_MAX_INDEXED

__device__ __forceinline__ void warp_max_indexed_4(
	volatile float *sd,//data
	volatile int   *sp,//index
	int index)
{
	float a1, a2; int p1, p2;

	a1 = sd[index], a2 = sd[index + 4];
	p1 = sp[index], p2 = sp[index + 4];
	sp[index] = p1 + (a1 < a2)*(p2 - p1);
	sd[index] = fmaxf(a1, a2);

	a1 = sd[index], a2 = sd[index + 2];
	p1 = sp[index], p2 = sp[index + 2];
	sp[index] = p1 + (a1 < a2)*(p2 - p1);
	sd[index] = fmaxf(a1, a2);
	
	a1 = sd[index], a2 = sd[index + 1];
	p1 = sp[index], p2 = sp[index + 1];
	sp[index] = p1 + (a1 < a2)*(p2 - p1);
	sd[index] = fmaxf(a1, a2);
}

__device__ __forceinline__ void warp_max_indexed_8(
	volatile float *sd,//data
	volatile int   *sp,//index
	int index)
{
	float a1, a2; int p1, p2;

	a1 = sd[index], a2 = sd[index + 8];
	p1 = sp[index], p2 = sp[index + 8];
	sp[index] = p1 + (a1 < a2)*(p2 - p1);
	sd[index] = fmaxf(a1, a2);

	a1 = sd[index], a2 = sd[index + 4];
	p1 = sp[index], p2 = sp[index + 4];
	sp[index] = p1 + (a1 < a2)*(p2 - p1);
	sd[index] = fmaxf(a1, a2);

	a1 = sd[index], a2 = sd[index + 2];
	p1 = sp[index], p2 = sp[index + 2];
	sp[index] = p1 + (a1 < a2)*(p2 - p1);
	sd[index] = fmaxf(a1, a2);

	a1 = sd[index], a2 = sd[index + 1];
	p1 = sp[index], p2 = sp[index + 1];
	sp[index] = p1 + (a1 < a2)*(p2 - p1);
	sd[index] = fmaxf(a1, a2);
}

__device__ __forceinline__ void warp_max_indexed_16(
	volatile float *sd,//data
	volatile int   *sp,//index
	int index)
{
	float a1, a2; int p1, p2;

	a1 = sd[index], a2 = sd[index + 16];
	p1 = sp[index], p2 = sp[index + 16];
	sp[index] = p1 + (a1 < a2)*(p2 - p1);
	sd[index] = fmaxf(a1, a2);

	a1 = sd[index], a2 = sd[index + 8];
	p1 = sp[index], p2 = sp[index + 8];
	sp[index] = p1 + (a1 < a2)*(p2 - p1);
	sd[index] = fmaxf(a1, a2);

	a1 = sd[index], a2 = sd[index + 4];
	p1 = sp[index], p2 = sp[index + 4];
	sp[index] = p1 + (a1 < a2)*(p2 - p1);
	sd[index] = fmaxf(a1, a2);

	a1 = sd[index], a2 = sd[index + 2];
	p1 = sp[index], p2 = sp[index + 2];
	sp[index] = p1 + (a1 < a2)*(p2 - p1);
	sd[index] = fmaxf(a1, a2);

	a1 = sd[index], a2 = sd[index + 1];
	p1 = sp[index], p2 = sp[index + 1];
	sp[index] = p1 + (a1 < a2)*(p2 - p1);
	sd[index] = fmaxf(a1, a2);
}

#endif 


#ifndef WARP_MIN_INDEXED
#define WARP_MIN_INDEXED

__device__ __forceinline__ void warp_min_indexed_4(
	volatile float *sd,//data
	volatile int   *sp,//index
	int index)
{
	float a1, a2; int p1, p2;

	a1 = sd[index], a2 = sd[index + 4];
	p1 = sp[index], p2 = sp[index + 4];
	sp[index] = p1 + (a1 > a2)*(p2 - p1);
	sd[index] = fminf(a1, a2);

	a1 = sd[index], a2 = sd[index + 2];
	p1 = sp[index], p2 = sp[index + 2];
	sp[index] = p1 + (a1 > a2)*(p2 - p1);
	sd[index] = fminf(a1, a2);

	a1 = sd[index], a2 = sd[index + 1];
	p1 = sp[index], p2 = sp[index + 1];
	sp[index] = p1 + (a1 > a2)*(p2 - p1);
	sd[index] = fminf(a1, a2);
}

__device__ __forceinline__ void warp_min_indexed_8(
	volatile float *sd,//data
	volatile int   *sp,//index
	int index)
{
	float a1, a2; int p1, p2;

	a1 = sd[index], a2 = sd[index + 8];
	p1 = sp[index], p2 = sp[index + 8];
	sp[index] = p1 + (a1 > a2)*(p2 - p1);
	sd[index] = fminf(a1, a2);

	a1 = sd[index], a2 = sd[index + 4];
	p1 = sp[index], p2 = sp[index + 4];
	sp[index] = p1 + (a1 > a2)*(p2 - p1);
	sd[index] = fminf(a1, a2);

	a1 = sd[index], a2 = sd[index + 2];
	p1 = sp[index], p2 = sp[index + 2];
	sp[index] = p1 + (a1 > a2)*(p2 - p1);
	sd[index] = fminf(a1, a2);

	a1 = sd[index], a2 = sd[index + 1];
	p1 = sp[index], p2 = sp[index + 1];
	sp[index] = p1 + (a1 > a2)*(p2 - p1);
	sd[index] = fminf(a1, a2);
}

__device__ __forceinline__ void warp_min_indexed_16(
	volatile float *sd,//data
	volatile int   *sp,//index
	int index)
{
	float a1, a2; int p1, p2;

	a1 = sd[index], a2 = sd[index + 16];
	p1 = sp[index], p2 = sp[index + 16];
	sp[index] = p1 + (a1 > a2)*(p2 - p1);
	sd[index] = fminf(a1, a2);

	a1 = sd[index], a2 = sd[index + 8];
	p1 = sp[index], p2 = sp[index + 8];
	sp[index] = p1 + (a1 > a2)*(p2 - p1);
	sd[index] = fminf(a1, a2);

	a1 = sd[index], a2 = sd[index + 4];
	p1 = sp[index], p2 = sp[index + 4];
	sp[index] = p1 + (a1 > a2)*(p2 - p1);
	sd[index] = fminf(a1, a2);

	a1 = sd[index], a2 = sd[index + 2];
	p1 = sp[index], p2 = sp[index + 2];
	sp[index] = p1 + (a1 > a2)*(p2 - p1);
	sd[index] = fminf(a1, a2);

	a1 = sd[index], a2 = sd[index + 1];
	p1 = sp[index], p2 = sp[index + 1];
	sp[index] = p1 + (a1 > a2)*(p2 - p1);
	sd[index] = fminf(a1, a2);
}

#endif 

#endif