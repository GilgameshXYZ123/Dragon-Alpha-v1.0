#pragma once

#ifndef MICRO_H
#define MICRO_H

#define get2d(A, i, j, stride) A[(i)*(stride) + (j)] //get2d
#define lget2d(A, i, j, lstride) A[((i)<<(lstride)) + (j)] //lget2d, lstride = log2(stride)

#define get3d(A, z, y, x, Sy, Sx)    A[(((z)*(Sy)) + (y))*(Sx) + (x)]
#define lget3d(A, z, y, x, Sy, LSx)  A[((((z)*(Sy)) + y)<<(LSx)) + (x)]

#define simdMM4(c, av, b) {(c).x += (av) * (b).x; (c).y += (av) * (b).y; (c).z += (av) * (b).z; (c).w += (av) * (b).w;}
#define simdMM2(c, av, b) {(c).x += (av) * (b).x; (c).y += (av) * (b).y;} 

#define Mul4(c, av, b) {(c).x = (av) * (b).x; (c).y = (av) * (b).y; (c).z = (av) * (b).z; (c).w = (av) * (b).w;}
#define Mul2(c, av, b) {(c).x = (av) * (b).x; (c).y = (av) * (b).y; }

#define next_index(index, length) ((index) + 1)%(length)

#define FZERO4 make_float4(0, 0, 0, 0)
#define FZERO2 make_float2(0, 0)

#define F32_2_0 float2{0, 0}
#define F32_4_0 float4{0, 0, 0, 0}

//if: flag == 1: -flag = -1 = 0xffffffff
//if: flag == 0: -flag =  0 
#define IF_int(flag, a, b) ( ((-flag) & (a - b)) + b)

//X = flag*v
#define zero_float(X, flag, v) \
	{float fv = v; int iv = -(flag) & *(int*)(&fv); X = *(float*)(&iv); }

#define zero_float4(X, flag) \
	{if (!flag) X.x = X.y = X.z = X.w = 0;}

#define QP_128(GN, GM)\
	(1.0f * ((GN + 127) >> 7 << 7) * ((GM + 127) >> 7 << 7) / (GN * GM))

#define QP_64(GN, GM)\
	(1.0f * ((GN + 63) >> 6 << 6) * ((GM + 63) >> 6 << 6) / (GN * GM))

#define QP_32(GN, GM)\
	(1.0f * ((GN + 31) >> 5 << 5) * ((GM + 31) >> 5 << 5) / (GN * GM))

#define VP_128(GN, GM)\
	(1.0f * (GN >> 7 << 7) * (GM >> 7 << 7) / (GN * GM))

#define VP_64(GN, GM)\
	(1.0f * (GN >> 6 << 6) * (GM >> 6 << 6) / (GN * GM))

#define VP_32(GN, GM)\
	(1.0f * (GN >> 5 << 5) * (GM >> 5 << 5) / (GN * GM))


__device__ float HOLE[260] = { 0 };

__device__ __forceinline__ void IF_write4(float* __restrict__ X,
	int xoffset, bool flag, float4 v) 
{
	float* dst = IF_int(flag, (X + xoffset), HOLE);
	*(float4*)dst = v;
}

#endif