#pragma once

#ifndef MATMUL_KERNEL_K2POW_H
#define MATMUL_KERNEL_K2POW_H

//K is power of 2, LK = log2(K)
#ifndef MATMUL_KERNEL_K2POW_CALL
#define MATMUL_KERNEL_K2POW_CALL

//LB = log2(BLOCK_SIZE)

#define	k88K2(LB, stream, A, B, C, N, M, LK, SB) \
	kernel_8_8_K2pow<LB, (1<<LB>>1)>\
		<<< dim3(M>>3>>LB, N>>3>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SB, LK)

#define k84K2(LB, stream, A, B, C, N, M, LK, SB) \
	kernel_8_4_K2pow<LB, (1<<LB>>1)>\
		<<< dim3(M>>2>>LB, N>>3>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SB, LK)

#define k48K2(LB, stream, A, B, C, N, M, LK, SB) \
	kernel_4_8_K2pow<LB, (1<<LB>>1)>\
		<<< dim3(M>>3>>LB, N>>2>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SB, LK)

#define k82K2(LB, stream, A, B, C, N, M, LK, SB) \
	kernel_8_2_K2pow<LB, (1<<LB>>1)>\
		<<< dim3(M>>1>>LB, N>>3>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SB, LK)

#define k28K2(LB, stream, A, B, C, N, M, LK, SB) \
	kernel_2_8_K2pow<LB, (1<<LB>>1)>\
		<<< dim3(M>>3>>LB, N>>1>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SB, LK)

#define k81K2(LB, stream, A, B, C, N, M, LK, SB) \
	kernel_8_1_K2pow<LB, (1<<LB)>\
		<<< dim3(M>>LB, N>>3>>LB), dim3(1<<LB, 1<<LB), 0, stream >>> \
			(A, B, C, SB, LK)

#define k18K2(LB, stream, A, B, C, N, M, LK, SB) \
	kernel_1_8_K2pow<LB, (1<<LB)>\
		<<< dim3(M>>3>>LB, N>>LB), dim3(1<<LB, 1<<LB), 0, stream >>> \
			(A, B, C, SB, LK)

#define k44K2(LB, stream, A, B, C, N, M, LK, SB) \
	kernel_4_4_K2pow<LB, (1<<LB>>1)>\
		<<< dim3(M>>2>>LB, N>>2>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SB, LK)

#define k42K2(LB, stream, A, B, C, N, M, LK, SB) \
	kernel_4_2_K2pow<LB, (1<<LB>>1)>\
		<<< dim3(M>>1>>LB, N>>2>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SB, LK)

#define k24K2(LB, stream, A, B, C, N, M, LK, SB) \
	kernel_2_4_K2pow<LB, (1<<LB>>1)>\
		<<< dim3(M>>2>>LB, N>>1>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SB, LK)

#define k22K2(LB, stream, A, B, C, N, M, LK, SB) \
	kernel_2_2_K2pow<LB, (1<<LB)>\
		<<< dim3(M>>1>>LB, N>>1>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SB, LK)

#define knaiveK2(GRID_SIZE, stream, A, B, C, N, M, LK, SB) \
	kernel_naive_K2pow\
		<<< dim3(GRID_SIZE, GRID_SIZE), dim3(M/GRID_SIZE, N/GRID_SIZE), 0, stream>>>\
			(A, B, C, SB, LK)

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), K % BLOCK_SIZE/2 == 0
#ifndef MATMUL_KERNEL_8_8_K2POW
#define MATMUL_KERNEL_8_8_K2POW

//(correct)
//LB = 4: Size = 1, Time = 1.316 msec, Performace = 1631.83 GFlop/s
//for 1024*1024*1024:
//	(1) 128, LB=4, Performance= 1577.34 GFlop/s, Time= 1.361 msec
//  (2)  64, LB=3, Performance= 1339.02 GFlop/s, Time= 1.604 msec
//  (3)  32, LB=2, Performance=  689.89 GFlop/s, Time= 3.113 msec
//for 2048*64*1024: LB=4, Performance= 835.15 GFlop/s, Time= 0.321 msec
//for 1280*64*1024: LB=3, Performance= 843.45 GFlop/s, Time= 0.199 msec
//for 1024*64*1024: LB=4, Performance = 702.76 GFlop/s, Time= 0.191 msec
//for  768*64*1024: LB=3, Performance=  607.54 GFlop/s, Time= 0.166 msec,
//for 64*1024*1024: LB=3, Performance = 849.97 GFlop/s, Time= 0.158 msec
//for 64*64  *1024: LB=3, Performance =  68.36 GFlop/s, Time= 0.123 msec

template<int LB, int STEP>
__global__ void kernel_8_8_K2pow(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float*  __restrict__ C,
	int SB, int LK)
{
	bool buf = 0;
	__shared__ float4 As[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Bs[2][1 << LB][(1 << LB) + 1];

	int Y = (blockIdx.y << 3 << LB);//Y = blockIdx.y * 8 * BLOCK_SIZE
	int X = (blockIdx.x << 3 << LB);//X = blockIdx.x * 8 * BLOCK_SIZE
	A = A + (Y << LK);//A[Y, 0]
	B = B + X;//B[0, X]
	C = &get(C, Y, X, SB);//C[Y, X]

	int tx = threadIdx.x, ty = threadIdx.y;//tx - ((tx >= STEP) *STEP)
	const int Ay = (ty << 3) + ((tx >= STEP) << 2), Ax = tx - ((tx >= STEP) << LB >> 1);
	const int Bx = (tx << 3) + ((ty >= STEP) << 2), By = ty - ((ty >= STEP) << LB >> 1);
	const int Byx = By * SB + Bx;

	As[buf][tx][ty].x = lget(A, Ay    , Ax, LK);//transpose A
	As[buf][tx][ty].y = lget(A, Ay + 1, Ax, LK);
	As[buf][tx][ty].z = lget(A, Ay + 2, Ax, LK);
	As[buf][tx][ty].w = lget(A, Ay + 3, Ax, LK);
	Bs[buf][ty][tx] = *(float4*)(B + Byx);
	A += STEP; B += (SB << LB >> 1);//B += SB*STEP
	__syncthreads();

	//compute area-------------------------------------------------------
	float4  c0 = make_float4(0, 0, 0, 0),  c1 = make_float4(0, 0, 0, 0);
	float4  c2 = make_float4(0, 0, 0, 0),  c3 = make_float4(0, 0, 0, 0);
	float4  c4 = make_float4(0, 0, 0, 0),  c5 = make_float4(0, 0, 0, 0);
	float4  c6 = make_float4(0, 0, 0, 0),  c7 = make_float4(0, 0, 0, 0);
	float4  c8 = make_float4(0, 0, 0, 0),  c9 = make_float4(0, 0, 0, 0);
	float4 c10 = make_float4(0, 0, 0, 0), c11 = make_float4(0, 0, 0, 0);
	float4 c12 = make_float4(0, 0, 0, 0), c13 = make_float4(0, 0, 0, 0);
	float4 c14 = make_float4(0, 0, 0, 0), c15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (2 << LK >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a0 = As[buf][ik][ty], a1 = As[buf][ik + STEP][ty];
			float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP][tx];

			simdMM4( c0, a0.x, b0); simdMM4( c1, a0.x, b1);
			simdMM4( c2, a0.y, b0); simdMM4( c3, a0.y, b1);
			simdMM4( c4, a0.z, b0); simdMM4( c5, a0.z, b1);
			simdMM4( c6, a0.w, b0); simdMM4( c7, a0.w, b1);
			simdMM4( c8, a1.x, b0); simdMM4( c9, a1.x, b1);
			simdMM4(c10, a1.y, b0); simdMM4(c11, a1.y, b1);
			simdMM4(c12, a1.z, b0); simdMM4(c13, a1.z, b1);
			simdMM4(c14, a1.w, b0); simdMM4(c15, a1.w, b1);
		}
		buf ^= 1;

		As[buf][tx][ty].x = lget(A, Ay    , Ax, LK);//transpose A
		As[buf][tx][ty].y = lget(A, Ay + 1, Ax, LK);
		As[buf][tx][ty].z = lget(A, Ay + 2, Ax, LK);
		As[buf][tx][ty].w = lget(A, Ay + 3, Ax, LK);
		Bs[buf][ty][tx] = *(float4*)(B + Byx);
		A += STEP; B += (SB << LB >> 1);//B += SB*STEP
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a0 = As[buf][ik][ty], a1 = As[buf][ik + STEP][ty];
		float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP][tx];

		simdMM4( c0, a0.x, b0); simdMM4( c1, a0.x, b1);
		simdMM4( c2, a0.y, b0); simdMM4( c3, a0.y, b1);
		simdMM4( c4, a0.z, b0); simdMM4( c5, a0.z, b1);
		simdMM4( c6, a0.w, b0); simdMM4( c7, a0.w, b1);
		simdMM4( c8, a1.x, b0); simdMM4( c9, a1.x, b1);
		simdMM4(c10, a1.y, b0); simdMM4(c11, a1.y, b1);
		simdMM4(c12, a1.z, b0); simdMM4(c13, a1.z, b1);
		simdMM4(c14, a1.w, b0); simdMM4(c15, a1.w, b1);
	}

	ty <<= 3; tx <<= 3;
	*(float4*)(&get(C, ty    , tx, SB)) = c0;  *(float4*)(&get(C, ty    , tx + 4, SB)) = c1;
	*(float4*)(&get(C, ty + 1, tx, SB)) = c2;  *(float4*)(&get(C, ty + 1, tx + 4, SB)) = c3;
	*(float4*)(&get(C, ty + 2, tx, SB)) = c4;  *(float4*)(&get(C, ty + 2, tx + 4, SB)) = c5;
	*(float4*)(&get(C, ty + 3, tx, SB)) = c6;  *(float4*)(&get(C, ty + 3, tx + 4, SB)) = c7;
	*(float4*)(&get(C, ty + 4, tx, SB)) = c8;  *(float4*)(&get(C, ty + 4, tx + 4, SB)) = c9;
	*(float4*)(&get(C, ty + 5, tx, SB)) = c10; *(float4*)(&get(C, ty + 5, tx + 4, SB)) = c11;
	*(float4*)(&get(C, ty + 6, tx, SB)) = c12; *(float4*)(&get(C, ty + 6, tx + 4, SB)) = c13;
	*(float4*)(&get(C, ty + 7, tx, SB)) = c14; *(float4*)(&get(C, ty + 7, tx + 4, SB)) = c15;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*4), K%BLOCK_SIZE/2==0, K>0
#ifndef MATMUL_KERNEL_8_4_K2POW
#define MATMUL_KERNEL_8_4_K2POW

//(correct)
//LB = 4: Size = 1, Time = 1.62 msec, Performace = 1325.61 GFlop/s
//for 1024*1024*1024:
//	(1) 128*64, LB=4, Performance= 1188.72 GFlop/s, Time= 1.807 msec
//	(2)  64*32, LB=3, Performance=  958.52 GFlop/s, Time= 2.240 msec
//	(3)  32*16, LB=2, Performance=  475.05 GFlop/s, Time= 4.521 msec
// for 1024*32: LB=3, Performance= 369.68 GFlop/s, Time= 0.182 msec
// for   64*32: LB=3, Performance=  43.77 GFlop/s, Time= 0.096 msec

template<int LB, int STEP>
__global__ void kernel_8_4_K2pow(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float*  __restrict__ C,
	int SB, int LK)
{
	bool buf = 0;
	__shared__ float4 As[2][1 << LB][(1 << LB) + 1];//followed k88
	__shared__ float2 Bs[2][1 << LB >> 1][(2 << LB) + 2];//followed k44

	int Y = (blockIdx.y << 3 << LB);//Y = blockIdx.y * 8 * BLOCK_SIZE
	int X = (blockIdx.x << 2 << LB);//X = blockIdx.x * 4 * BLOCK_SIZE
	A = A + (Y << LK);//A[Y, 0]
	B = B + X;//B[0, X]
	C = &get(C, Y, X, SB);//C[Y, X]

	int tx = threadIdx.x, ty = threadIdx.y;
	const int Ay = (ty << 3) + ((tx >= STEP) << 2), Ax = tx - ((tx >= STEP) << LB >> 1);
	const int By = ty >> 1, Bx = (tx << 2) + ((ty & 1) << 1);
	const int Bs_y = By, Bs_x = ((tx << 1) + (ty & 1));
	const int Byx = By * SB + Bx;//[By, Bx]

	As[buf][tx][ty].x = lget(A, Ay    , Ax, LK);//transpose A
	As[buf][tx][ty].y = lget(A, Ay + 1, Ax, LK);
	As[buf][tx][ty].z = lget(A, Ay + 2, Ax, LK);
	As[buf][tx][ty].w = lget(A, Ay + 3, Ax, LK);
	Bs[buf][Bs_y][Bs_x] = *(float2*)(B + Byx);
	A += STEP; B += (SB << LB >> 1);//B += SB*STEP
	__syncthreads();

	//compute area-------------------------------------------------------
	float4  c0 = make_float4(0, 0, 0, 0);
	float4  c2 = make_float4(0, 0, 0, 0);
	float4  c4 = make_float4(0, 0, 0, 0);
	float4  c6 = make_float4(0, 0, 0, 0);
	float4  c8 = make_float4(0, 0, 0, 0);
	float4 c10 = make_float4(0, 0, 0, 0);
	float4 c12 = make_float4(0, 0, 0, 0);
	float4 c14 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (2 << LK >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a0 = As[buf][ik][ty], a1 = As[buf][ik + STEP][ty];
			float4 b0 = *(float4*)(&Bs[buf][ik][tx << 1]);

			simdMM4(c0, a0.x, b0);
			simdMM4(c2, a0.y, b0);
			simdMM4(c4, a0.z, b0);
			simdMM4(c6, a0.w, b0);
			simdMM4(c8, a1.x, b0);
			simdMM4(c10, a1.y, b0);
			simdMM4(c12, a1.z, b0);
			simdMM4(c14, a1.w, b0);
		}
		buf ^= 1;

		As[buf][tx][ty].x = lget(A, Ay    , Ax, LK);//transpose A
		As[buf][tx][ty].y = lget(A, Ay + 1, Ax, LK);
		As[buf][tx][ty].z = lget(A, Ay + 2, Ax, LK);
		As[buf][tx][ty].w = lget(A, Ay + 3, Ax, LK);
		Bs[buf][Bs_y][Bs_x] = *(float2*)(B + Byx);
		A += STEP; B += (SB << LB >> 1);//B += SB*STEP
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a0 = As[buf][ik][ty], a1 = As[buf][ik + STEP][ty];
		float4 b0 = *(float4*)(&Bs[buf][ik][tx << 1]);

		simdMM4(c0, a0.x, b0);
		simdMM4(c2, a0.y, b0);
		simdMM4(c4, a0.z, b0);
		simdMM4(c6, a0.w, b0);
		simdMM4(c8, a1.x, b0);
		simdMM4(c10, a1.y, b0);
		simdMM4(c12, a1.z, b0);
		simdMM4(c14, a1.w, b0);
	}

	ty <<= 3; tx <<= 2;
	*(float4*)(&get(C, ty    , tx, SB)) = c0;
	*(float4*)(&get(C, ty + 1, tx, SB)) = c2;
	*(float4*)(&get(C, ty + 2, tx, SB)) = c4;
	*(float4*)(&get(C, ty + 3, tx, SB)) = c6;
	*(float4*)(&get(C, ty + 4, tx, SB)) = c8;
	*(float4*)(&get(C, ty + 5, tx, SB)) = c10;
	*(float4*)(&get(C, ty + 6, tx, SB)) = c12;
	*(float4*)(&get(C, ty + 7, tx, SB)) = c14;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*8), K % BLOCK_SIZE/2 == 0
#ifndef MATMUL_KERNEL_4_8_K2POW
#define MATMUL_KERNEL_4_8_K2POW

//(correct)
//LB = 4: Size = 1, Time = 1.708 msec, Performace = 1257.31 GFlop/s
//for 1024*1024*1024:
//	(1) 64*128, LB=4, Performance= 1211.06 GFlop/s, Time= 1.773 msec
//	(2) 32*64 , LB=3, Performance= 1168.12 GFlop/s, Time= 1.838 msec
//	(3) 16*32 , LB=2, Performance=  539.34 GFlop/s, Time= 3.982 msec
// for 32*1024: Performance= 556.87 GFlop/s, Time= 0.121 msec
// for 32*64  : Performance=  40.77 GFlop/s, Time= 0.103 msec

template<int LB, int STEP>
__global__ void kernel_4_8_K2pow(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float*  __restrict__ C,
	int SB, int LK)
{
	bool buf = 0;
	__shared__ float2 As[2][1 << LB >> 1][(2 << LB) + 2];//followed k44
	__shared__ float4 Bs[2][1 << LB][(1 << LB) + 1];//followed k88

	int Y = (blockIdx.y << 2 << LB);//Y = blockIdx.y * 4 * BLOCK_SIZE
	int X = (blockIdx.x << 3 << LB);//X = blockIdx.x * 8 * BLOCK_SIZE
	A = A + (Y << LK);//A[Y, 0]
	B = B + X;//B[0, X]
	C = &get(C, Y, X, SB);//C[Y, X]

	int tx = threadIdx.x, ty = threadIdx.y;
	const int Ax = (tx >> 1), Ay = (ty << 2) + ((tx & 1) << 1);
	const int As_x = Ax, As_y = ((ty << 1) + (tx & 1));
	const int Bx = (tx << 3) + ((ty >= STEP) << 2), By = ty - ((ty >= STEP) << LB >> 1);
	const int Byx = By * SB + Bx;//[By, Bx]

	As[buf][As_x][As_y].x = lget(A, Ay    , Ax, LK);//transpose A
	As[buf][As_x][As_y].y = lget(A, Ay + 1, Ax, LK);
	Bs[buf][ty][tx] = *(float4*)(B + Byx);
	A += STEP; B += (SB << LB >> 1);//B += SB*STEP
	__syncthreads();

	//compute area-------------------------------------------------------
	float4 c0 = make_float4(0, 0, 0, 0), c1 = make_float4(0, 0, 0, 0);
	float4 c2 = make_float4(0, 0, 0, 0), c3 = make_float4(0, 0, 0, 0);
	float4 c4 = make_float4(0, 0, 0, 0), c5 = make_float4(0, 0, 0, 0);
	float4 c6 = make_float4(0, 0, 0, 0), c7 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (2 << LK >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a0 = *(float4*)(&As[buf][ik][ty << 1]);
			float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP][tx];

			simdMM4(c0, a0.x, b0); simdMM4(c1, a0.x, b1);
			simdMM4(c2, a0.y, b0); simdMM4(c3, a0.y, b1);
			simdMM4(c4, a0.z, b0); simdMM4(c5, a0.z, b1);
			simdMM4(c6, a0.w, b0); simdMM4(c7, a0.w, b1);
		}
		buf ^= 1;

		As[buf][As_x][As_y].x = lget(A, Ay    , Ax, LK);//transpose A
		As[buf][As_x][As_y].y = lget(A, Ay + 1, Ax, LK);
		Bs[buf][ty][tx] = *(float4*)(B + Byx);
		A += STEP; B += (SB << LB >> 1);//B += SB*STEP
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a0 = *(float4*)(&As[buf][ik][ty << 1]);
		float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP][tx];

		simdMM4(c0, a0.x, b0); simdMM4(c1, a0.x, b1);
		simdMM4(c2, a0.y, b0); simdMM4(c3, a0.y, b1);
		simdMM4(c4, a0.z, b0); simdMM4(c5, a0.z, b1);
		simdMM4(c6, a0.w, b0); simdMM4(c7, a0.w, b1);
	}

	ty <<= 2; tx <<= 3;
	*(float4*)(&get(C, ty    , tx, SB)) = c0; *(float4*)(&get(C, ty    , tx + 4, SB)) = c1;
	*(float4*)(&get(C, ty + 1, tx, SB)) = c2; *(float4*)(&get(C, ty + 1, tx + 4, SB)) = c3;
	*(float4*)(&get(C, ty + 2, tx, SB)) = c4; *(float4*)(&get(C, ty + 2, tx + 4, SB)) = c5;
	*(float4*)(&get(C, ty + 3, tx, SB)) = c6; *(float4*)(&get(C, ty + 3, tx + 4, SB)) = c7;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*2), K % BLOCK_SIZE/2 == 0
#ifndef MATMUL_KERNEL_8_2_K2POW
#define MATMUL_KERNEL_8_2_K2POW

//(correct)
//LB = 4: Size = 1, Time = 2.128 msec, Performace = 1009.16 GFlop/s
//for 1024*1024*1024:
//	(1) 128*32, LB=4, Performance= 789.96 GFlop/s, Time= 2.718 msec 
//	(2)  64*16, LB=3, Performance= 559.98 GFlop/s, Time= 3.835 msec
//	(3)  32*8 , LB=2, Performance= 310.97 GFlop/s, Time= 6.906 msec
//for 128*128*128:
//	(1)  16*4 , LB=1, Performance=  53.63 GFlop/s, Time= 0.078 msec,
//for   64*16: LB=3, Performance=  18.72 GFlop/s, Time= 0.112 msec
//for 1024*16: LB=3, Performance= 134.35 GFlop/s, Time= 0.250 msec
//for 1024* 8: LB=2, Performance=  61.33 GFlop/s, Time= 0.274 msec

template<int LB, int STEP>
__global__ void kernel_8_2_K2pow(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float*  __restrict__ C,
	int SB, int LK)
{
	bool buf = 0;
	__shared__ float4 As[2][1 << LB][(1 << LB) + 1];//followed k88
	__shared__ float  Bs[2][1 << LB >> 1][(2 << LB) + 2];//followed k44

	int Y = (blockIdx.y << 3 << LB);//Y = blockIdx.y * 8 * BLOCK_SIZE
	int X = (blockIdx.x << 1 << LB);//X = blockIdx.x * 2 * BLOCK_SIZE
	A = A + (Y << LK);//A[Y, 0]
	B = B + X;//B[0, X]
	C = &get(C, Y, X, SB);//C[Y, X]

	int tx = threadIdx.x, ty = threadIdx.y;
	const int Ay = (ty << 3) + ((tx >= STEP) << 2), Ax = tx - ((tx >= STEP) << LB >> 1);
	const int By = (ty >> 1), Bx = (tx << 1) + (ty & 1);
	const int Bs_y = By, Bs_x = ((tx << 1) + (ty & 1));
	const int Byx = By * SB + Bx;//[By, Bx]

	As[buf][tx][ty].x = lget(A, Ay    , Ax, LK);//transpose A
	As[buf][tx][ty].y = lget(A, Ay + 1, Ax, LK);
	As[buf][tx][ty].z = lget(A, Ay + 2, Ax, LK);
	As[buf][tx][ty].w = lget(A, Ay + 3, Ax, LK);
	Bs[buf][Bs_y][Bs_x] = B[Byx];
	A += STEP; B += (SB << LB >> 1);//B += SB*STEP
	__syncthreads();

	//compute area-------------------------------------------------------
	float2  c0 = make_float2(0, 0);
	float2  c2 = make_float2(0, 0);
	float2  c4 = make_float2(0, 0);
	float2  c6 = make_float2(0, 0);
	float2  c8 = make_float2(0, 0);
	float2 c10 = make_float2(0, 0);
	float2 c12 = make_float2(0, 0);
	float2 c14 = make_float2(0, 0);

	for (int ok = 1, OK = (2 << LK >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a0 = As[buf][ik][ty], a1 = As[buf][ik + STEP][ty];
			float2 b0 = *(float2*)(&Bs[buf][ik][tx << 1]);

			simdMM2(c0, a0.x, b0);
			simdMM2(c2, a0.y, b0);
			simdMM2(c4, a0.z, b0);
			simdMM2(c6, a0.w, b0);
			simdMM2(c8, a1.x, b0);
			simdMM2(c10, a1.y, b0);
			simdMM2(c12, a1.z, b0);
			simdMM2(c14, a1.w, b0);
		}
		buf ^= 1;

		As[buf][tx][ty].x = lget(A, Ay    , Ax, LK);//transpose A
		As[buf][tx][ty].y = lget(A, Ay + 1, Ax, LK);
		As[buf][tx][ty].z = lget(A, Ay + 2, Ax, LK);
		As[buf][tx][ty].w = lget(A, Ay + 3, Ax, LK);
		Bs[buf][Bs_y][Bs_x] = B[Byx];
		A += STEP; B += (SB << LB >> 1);//B += SB*STEP
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a0 = As[buf][ik][ty], a1 = As[buf][ik + STEP][ty];
		float2 b0 = *(float2*)(&Bs[buf][ik][tx << 1]);

		simdMM2(c0, a0.x, b0);
		simdMM2(c2, a0.y, b0);
		simdMM2(c4, a0.z, b0);
		simdMM2(c6, a0.w, b0);
		simdMM2(c8, a1.x, b0);
		simdMM2(c10, a1.y, b0);
		simdMM2(c12, a1.z, b0);
		simdMM2(c14, a1.w, b0);
	}

	ty <<= 3; tx <<= 1;
	*(float2*)(&get(C, ty    , tx, SB)) = c0;
	*(float2*)(&get(C, ty + 1, tx, SB)) = c2;
	*(float2*)(&get(C, ty + 2, tx, SB)) = c4;
	*(float2*)(&get(C, ty + 3, tx, SB)) = c6;
	*(float2*)(&get(C, ty + 4, tx, SB)) = c8;
	*(float2*)(&get(C, ty + 5, tx, SB)) = c10;
	*(float2*)(&get(C, ty + 6, tx, SB)) = c12;
	*(float2*)(&get(C, ty + 7, tx, SB)) = c14;
}

#endif 


//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*8), K % BLOCK_SIZE/2 == 0
#ifndef MATMUL_KERNEL_2_8_K2POW
#define MATMUL_KERNEL_2_8_K2POW

//(correct)
//LB = 4: Size = 1, Time = 2.712 msec, Performace = 791.845 GFlop/s
//for 1024*1024*1024:
//	(1) 32*128, LB=4, Performance= 782.57 GFlop/s, Time= 2.744 msec
//	(2) 16*64 , LB=3, Performance= 772.41 GFlop/s, Time= 2.780 msec
//	(3)  8*32 , LB=2, Performance= 343.97 GFlop/s, Time= 6.243 msec
//for 128*128*128:
//	(1)  4*16 , LB=1, Performance= 59.34 GFlop/s, Time= 0.071 msec
//for 16*64  : LB=3, Performance=  27.80 GFlop/s, Time= 0.075 msec
//for 16*1024: LB=3, Performance= 319.12 GFlop/s, Time= 0.105 msec
//for  8*1024: LB=2, Performance=  61.33 GFlop/s, Time= 0.274 msec

template<int LB, int STEP>
__global__ void kernel_2_8_K2pow(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float*  __restrict__ C,
	int SB, int LK)
{
	bool buf = 0;
	__shared__ float  As[2][1 << LB >> 1][(2 << LB) + 2];//followed k44
	__shared__ float4 Bs[2][1 << LB][(1 << LB) + 1];//followed k88

	int Y = (blockIdx.y << 1 << LB);//Y = blockIdx.y * 2 * BLOCK_SIZE
	int X = (blockIdx.x << 3 << LB);//X = blockIdx.x * 8 * BLOCK_SIZE
	A = A + (Y << LK);//A[Y, 0]
	B = B + X;//B[0, X]
	C = &get(C, Y, X, SB);//C[Y, X]

	int tx = threadIdx.x, ty = threadIdx.y;
	const int Ax = (tx >> 1), Ay = (ty << 1) + (tx & 1);
	const int As_x = Ax, As_y = ((ty << 1) + (tx & 1));
	const int Bx = (tx << 3) + ((ty >= STEP) << 2), By = ty - ((ty >= STEP) << LB >> 1);
	const int Ayx = (Ay << LK) + Ax;//[Ay, Ax]
	const int Byx = By * SB + Bx;//[By, Bx]

	As[buf][As_x][As_y] = A[Ayx]; //transpose A
	Bs[buf][ty][tx] = *(float4*)(B + Byx);
	A += STEP; B += (SB << LB >> 1);//B += SB*STEP
	__syncthreads();

	//compute area-------------------------------------------------------
	float4 c0 = make_float4(0, 0, 0, 0), c1 = make_float4(0, 0, 0, 0);
	float4 c2 = make_float4(0, 0, 0, 0), c3 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (2 << LK >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float2 a0 = *(float2*)(&As[buf][ik][ty << 1]);
			float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP][tx];
			simdMM4(c0, a0.x, b0); simdMM4(c1, a0.x, b1);
			simdMM4(c2, a0.y, b0); simdMM4(c3, a0.y, b1);
		}
		buf ^= 1;

		As[buf][As_x][As_y] = A[Ayx]; //transpose A
		Bs[buf][ty][tx] = *(float4*)(B + Byx);
		A += STEP; B += (SB << LB >> 1);//B += SB*STEP
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float2 a0 = *(float2*)(&As[buf][ik][ty << 1]);
		float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP][tx];
		simdMM4(c0, a0.x, b0); simdMM4(c1, a0.x, b1);
		simdMM4(c2, a0.y, b0); simdMM4(c3, a0.y, b1);
	}

	ty <<= 1; tx <<= 3;
	*(float4*)(&get(C, ty    , tx, SB)) = c0; *(float4*)(&get(C, ty    , tx + 4, SB)) = c1;
	*(float4*)(&get(C, ty + 1, tx, SB)) = c2; *(float4*)(&get(C, ty + 1, tx + 4, SB)) = c3;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*1), K % BLOCK_SIZE == 0
#ifndef MATMUL_KERNEL_8_1_K2POW
#define MATMUL_KENERL_8_1_K2POW

//(correct)
//LB = 4: Size = 1, Time = 3.17 msec, Performace = 677.44 GFlop/s
//for 1024*128*1014:
//	(1) 128*16, LB=4, Performance = 395.53 GFlop/s, Time= 0.679 msec
//	(2)  64*8 , LB=3, Performance = 314.75 GFlop/s, Time= 0.853 msec
//for 1024*8: LB=3, Performance = 107.05 GFlop/s, Time=  0.157 msec
//for   64*8: LB=3, Performance =  10.61 GFlop/s, Time = 0.096 msec
//for 1024*4: LB=2, Performance =  33.46 GFlop/s, Time=  0.251 msec

template<int LB, int STEP>
__global__ void kernel_8_1_K2pow(
	const float* __restrict__ A,
	const float* __restrict__ B,
	float* __restrict__ C,
	int SB, int LK)
{
	bool buf = 0;
	__shared__ float4 As[2][1 << LB][(2 << LB) + 1];
	__shared__ float  Bs[2][1 << LB][(1 << LB) + 1];

	int Y = (blockIdx.y << 3 << LB);//Y = blockIdx.y * 8 * BLOCK_SIZE
	int X = (blockIdx.x << LB);//X = blockIdx.x * BLOCK_SIZE
	A = A + (Y << LK);//A[Y, 0]
	B = B + X;//B[0, X]
	C = &get(C, Y, X, SB);//C[Y, X]

	int ty = threadIdx.y, tx = threadIdx.x;
	const int Byx = ty * SB + tx;//[ty, tx]

	As[buf][tx][(ty << 1)    ].x = lget(A, (ty << 3)    , tx, LK);
	As[buf][tx][(ty << 1)    ].y = lget(A, (ty << 3) + 1, tx, LK);
	As[buf][tx][(ty << 1)    ].z = lget(A, (ty << 3) + 2, tx, LK);
	As[buf][tx][(ty << 1)    ].w = lget(A, (ty << 3) + 3, tx, LK);
	As[buf][tx][(ty << 1) + 1].x = lget(A, (ty << 3) + 4, tx, LK);
	As[buf][tx][(ty << 1) + 1].y = lget(A, (ty << 3) + 5, tx, LK);
	As[buf][tx][(ty << 1) + 1].z = lget(A, (ty << 3) + 6, tx, LK);
	As[buf][tx][(ty << 1) + 1].w = lget(A, (ty << 3) + 7, tx, LK);
	Bs[buf][ty][tx] = B[Byx];
	A += STEP; B += (SB << LB);//B += SB * STEP
	__syncthreads();

	//compute area------------------------------------------
	float4 c0 = make_float4(0, 0, 0, 0);
	float4 c1 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (1 << LK >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a0 = As[buf][ik][(ty << 1)];
			float4 a1 = As[buf][ik][(ty << 1) + 1];
			float b = Bs[buf][ik][tx];

			simdMM4(c0, b, a0);
			simdMM4(c1, b, a1);
		}
		buf ^= 1;

		As[buf][tx][(ty << 1)    ].x = lget(A, (ty << 3)    , tx, LK);
		As[buf][tx][(ty << 1)    ].y = lget(A, (ty << 3) + 1, tx, LK);
		As[buf][tx][(ty << 1)    ].z = lget(A, (ty << 3) + 2, tx, LK);
		As[buf][tx][(ty << 1)    ].w = lget(A, (ty << 3) + 3, tx, LK);
		As[buf][tx][(ty << 1) + 1].x = lget(A, (ty << 3) + 4, tx, LK);
		As[buf][tx][(ty << 1) + 1].y = lget(A, (ty << 3) + 5, tx, LK);
		As[buf][tx][(ty << 1) + 1].z = lget(A, (ty << 3) + 6, tx, LK);
		As[buf][tx][(ty << 1) + 1].w = lget(A, (ty << 3) + 7, tx, LK);
		Bs[buf][ty][tx] = B[Byx];
		A += STEP; B += (SB << LB);//B += SB * STEP
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a0 = As[buf][ik][(ty << 1)];
		float4 a1 = As[buf][ik][(ty << 1) + 1];
		float b = Bs[buf][ik][tx];

		simdMM4(c0, b, a0);
		simdMM4(c1, b, a1);
	}

	ty <<= 3;
	get(C, ty    , tx, SB) = c0.x;
	get(C, ty + 1, tx, SB) = c0.y;
	get(C, ty + 2, tx, SB) = c0.z;
	get(C, ty + 3, tx, SB) = c0.w;
	get(C, ty + 4, tx, SB) = c1.x;
	get(C, ty + 5, tx, SB) = c1.y;
	get(C, ty + 6, tx, SB) = c1.z;
	get(C, ty + 7, tx, SB) = c1.w;
}

#endif


//(Y: BLOCK_SIZE*1, X: BLOCK_SIZE*8), K % BLOCK_SIZE == 0
#ifndef MATMUL_KERNEL_1_8_K2POW
#define MATMUL_KERNEL_1_8_K2POW

//(correct)
//LB = 4: Size = 1, Time = 9.098 msec, Performace = 236.039 GFlop/s
//for 8*1024: LB=3, Performance = 153.58 GFlop/s, Time= 0.109 msec
//for 8*64  : LB=3, Performance =  14.05 GFlop/s, Time= 0.075 msec
//for 4*1024: LB=2, Performance =  72.61 GFlop/s, Time= 0.116 msec

template<int LB, int STEP>
__global__ void kernel_1_8_K2pow(
	const float* __restrict__ A,
	const float* __restrict__ B,
	float* __restrict__ C,
	int SB, int LK)
{
	bool buf = 0;
	__shared__ float   As[2][1 << LB][(1 << LB) + 1];
	__shared__ float4  Bs[2][2 << LB][(2 << LB) + 1];

	int Y = (blockIdx.y << LB);//Y = blockIdx.y * BLOCK_SIZE
	int X = (blockIdx.x << 3 << LB);//X = blockIdx.x * 8 * BLOCK_SIZE
	A = A + (Y << LK);//A[Y, 0]
	B = B + X;//B[0, X]
	C = &get(C, Y, X, SB);//C[Y, X]

	int ty = threadIdx.y, tx = threadIdx.x;
	const int Ayx = (ty << LK) + tx;//[ty, tx]
	const int Byx = ty * SB + (tx << 3);//[ty, tx<<3]

	As[buf][tx][ty] = A[Ayx];
	Bs[buf][ty][(tx << 1)    ] = *(float4*)(B + Byx    );
	Bs[buf][ty][(tx << 1) + 1] = *(float4*)(B + Byx + 4);
	A += STEP; B += (SB << LB);//B += SB * STEP
	__syncthreads();

	//compute area------------------------------------------
	float4 c0 = make_float4(0, 0, 0, 0);
	float4 c1 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (1 << LK >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float a = As[buf][ik][ty];
			float4 b0 = Bs[buf][ik][(tx << 1)];
			float4 b1 = Bs[buf][ik][(tx << 1) + 1];
			simdMM4(c0, a, b0); simdMM4(c1, a, b1);
		}
		buf ^= 1;

		As[buf][tx][ty] = A[Ayx];
		Bs[buf][ty][(tx << 1)] = *(float4*)(B + Byx);
		Bs[buf][ty][(tx << 1) + 1] = *(float4*)(B + Byx + 4);
		A += STEP; B += (SB << LB);//B += SB * STEP
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float a = As[buf][ik][ty];
		float4 b0 = Bs[buf][ik][(tx << 1)];
		float4 b1 = Bs[buf][ik][(tx << 1) + 1];
		simdMM4(c0, a, b0); simdMM4(c1, a, b1);
	}

	tx <<= 3;
	*(float4*)(&get(C, ty, tx, SB)) = c0; *(float4*)(&get(C, ty, tx + 4, SB)) = c1;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*4), K % BLOCK_SIZE/2 == 0
#ifndef MATMUL_KERNEL_4_4_K2POW
#define MATMUL_KERNEL_4_4_K2POW

//(correct)
//LB = 4: Size = 1, Time = 2.014 msec, Performace = 1066.28 GFlop/s
//for 1024*1024*1024:
//	(1) 128*128, LB=5, Performance = 1014.73 GFlop/s, Time= 2.116 msec,
//	(2)  64*64 , LB=4, Performance =  967.09 GFlop/s, Time= 2.221 msec
//	(3)  32*32 , LB=3, Performance =  916.65 GFlop/s, Time= 2.343 msec
//	(4)  16*16 , LB=2, Performance =  346.90 GFlop/s, Time= 6.191 msec
//for 1024*32:  LB=3, Performance = 355.96 GFlop/s, Time= 0.189 msec
//for 32*1024 : LB=3, Performance = 580.88 GFlop/s, Time= 0.116 msec
//for 32*32   : LB=3, Performance =  26.79 GFlop/s, Time= 0.078 msec 

template<int LB, int STEP>
__global__ void kernel_4_4_K2pow(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float*  __restrict__ C,
	int SB, int LK)
{
	bool buf = 0;
	__shared__ float2 As[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2 Bs[2][1 << LB >> 1][(2 << LB) + 2];

	int Y = (blockIdx.y << 2 << LB);//Y = blockIdx.y * 4 * BLOCK_SIZE
	int X = (blockIdx.x << 2 << LB);//X = blockIdx.x * 4 * BLOCK_SIZE
	A = A + (Y << LK);//A[Y, 0]
	B = B + X;//B[0, X]
	C = &get(C, Y, X, SB);//C[Y, X]

	int tx = threadIdx.x, ty = threadIdx.y;
	const int Ax = (tx >> 1), Ay = (ty << 2) + ((tx & 1) << 1);
	const int By = (ty >> 1), Bx = (tx << 2) + ((ty & 1) << 1);
	const int As_x = Ax, As_y = ((ty << 1) + (tx & 1));
	const int Bs_y = By, Bs_x = ((tx << 1) + (ty & 1));
	const int Byx = By * SB + Bx;//[By, Bx]

	As[buf][As_x][As_y].x = lget(A, Ay    , Ax, LK);//transpose A
	As[buf][As_x][As_y].y = lget(A, Ay + 1, Ax, LK);
	Bs[buf][Bs_y][Bs_x] = *(float2*)(B + Byx);
	A += STEP; B += (SB << LB >> 1);//B += SB * STEP
	__syncthreads();

	//compute area------------------------------------------
	float4 c0 = make_float4(0, 0, 0, 0);
	float4 c1 = make_float4(0, 0, 0, 0);
	float4 c2 = make_float4(0, 0, 0, 0);
	float4 c3 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (2 << LK >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a = *(float4*)(&As[buf][ik][ty << 1]);
			float4 b = *(float4*)(&Bs[buf][ik][tx << 1]);

			simdMM4(c0, a.x, b);
			simdMM4(c1, a.y, b);
			simdMM4(c2, a.z, b);
			simdMM4(c3, a.w, b);
		}
		buf ^= 1;

		As[buf][As_x][As_y].x = lget(A, Ay    , Ax, LK);//transpose A
		As[buf][As_x][As_y].y = lget(A, Ay + 1, Ax, LK);
		Bs[buf][Bs_y][Bs_x] = *(float2*)(B + Byx);
		A += STEP; B += (SB << LB >> 1);//B += SB * STEP
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a = *(float4*)(&As[buf][ik][ty << 1]);
		float4 b = *(float4*)(&Bs[buf][ik][tx << 1]);

		simdMM4(c0, a.x, b);
		simdMM4(c1, a.y, b);
		simdMM4(c2, a.z, b);
		simdMM4(c3, a.w, b);
	}

	ty <<= 2; tx <<= 2;
	*(float4*)(&get(C, ty    , tx, SB)) = c0;
	*(float4*)(&get(C, ty + 1, tx, SB)) = c1;
	*(float4*)(&get(C, ty + 2, tx, SB)) = c2;
	*(float4*)(&get(C, ty + 3, tx, SB)) = c3;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*2), K % BLOCK_SIZE/2 == 0
#ifndef MATMUL_KERNEL_4_2_K2POW
#define MATMUL_KERNEL_4_2_K2POW

//(correct)
//LB = 4: Size = 1, Time = 2.53 msec, Performace = 848.808 GFlop/s
//for 1024*1024*1024:
//	(1) 128*64, LB=5, Performance = 738.51 GFlop/s, Time= 2.908 msec
//	(2)  64*32, LB=4, Performance = 680.26 GFlop/s, Time= 3.157 msec
//	(3)	 32*16, LB=3, Performance = 522.76 GFlop/s, Time= 4.108 msec
//	(4)  16*8 , LB=2, Performance = 183.64 GFlop/s, Time= 11.694 msec
//for 128*128*128:
//	(1)   8*4 , LB=1, Performance = 42.03 GFlop/s, Time= 0.100 msec
//for 1024*16: LB=3, Performance = 126.27 GFlop/s, Time= 0.266 msec
//for   64*32: LB=3, Performance = 17.69 GFlop/s, Time= 0.119 msec
//for   32*16: LB=3, Performance =  9.01 GFlop/s, Time= 0.116 msec
//for 1024* 8: LB=2, Performance = 85.87 GFlop/s, Time= 0.195 msec

template<int LB, int STEP>
__global__ void kernel_4_2_K2pow(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float*  __restrict__ C,
	int SB, int LK)
{
	bool buf = 0;
	__shared__ float2 As[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float  Bs[2][1 << LB >> 1][(2 << LB) + 2];

	int Y = (blockIdx.y << 2 << LB);//Y = blockIdx.y * 4 * BLOCK_SIZE
	int X = (blockIdx.x << 1 << LB);//X = blockIdx.x * 2 * BLOCK_SIZE
	A = A + (Y << LK);//A[Y, 0]
	B = B + X;//B[0, X]
	C = &get(C, Y, X, SB);//C[Y, X]

	int tx = threadIdx.x, ty = threadIdx.y;
	const int Ax = (tx >> 1), Ay = (ty << 2) + ((tx & 1) << 1);
	const int By = (ty >> 1), Bx = (tx << 1) + (ty & 1);
	const int As_x = Ax, As_y = ((ty << 1) + (tx & 1));
	const int Bs_y = By, Bs_x = ((tx << 1) + (ty & 1));
	const int Byx = By * SB + Bx;//[By, Bx]

	As[buf][As_x][As_y].x = lget(A, Ay    , Ax, LK);//transpose A
	As[buf][As_x][As_y].y = lget(A, Ay + 1, Ax, LK);
	Bs[buf][Bs_y][Bs_x] = B[Byx];
	A += STEP; B += (SB << LB >> 1);//B += SB * STEP
	__syncthreads();

	//compute area------------------------------------------
	float2 c0 = make_float2(0, 0);
	float2 c1 = make_float2(0, 0);
	float2 c2 = make_float2(0, 0);
	float2 c3 = make_float2(0, 0);
	for (int ok = 1, OK = (2 << LK >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a = *(float4*)(&As[buf][ik][ty << 1]);
			float2 b = *(float2*)(&Bs[buf][ik][tx << 1]);

			simdMM2(c0, a.x, b);
			simdMM2(c1, a.y, b);
			simdMM2(c2, a.z, b);
			simdMM2(c3, a.w, b);
		}
		buf ^= 1;

		As[buf][As_x][As_y].x = lget(A, Ay    , Ax, LK);//transpose A
		As[buf][As_x][As_y].y = lget(A, Ay + 1, Ax, LK);
		Bs[buf][Bs_y][Bs_x] = B[Byx];
		A += STEP; B += (SB << LB >> 1);//B += SB * STEP
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a = *(float4*)(&As[buf][ik][ty << 1]);
		float2 b = *(float2*)(&Bs[buf][ik][tx << 1]);

		simdMM2(c0, a.x, b);
		simdMM2(c1, a.y, b);
		simdMM2(c2, a.z, b);
		simdMM2(c3, a.w, b);
	}

	ty <<= 2; tx <<= 1;
	*(float2*)(&get(C, ty    , tx, SB)) = c0;
	*(float2*)(&get(C, ty + 1, tx, SB)) = c1;
	*(float2*)(&get(C, ty + 2, tx, SB)) = c2;
	*(float2*)(&get(C, ty + 3, tx, SB)) = c3;
}

#endif


//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*4), K % BLOCK_SIZE/2 == 0
#ifndef MATMUL_KERNEL_2_4_K2POW
#define MATMUL_KERNEL_2_4_K2POW

//(correct)
//LB = 4: Size = 1, Time = 3.186 msec, Performace = 674.037 GFlop/s
//for 1024*1024*1024:
//	(1) 64*128, LB=5, Performance= 613.09 GFlop/s, Time= 3.503 msec
//	(2) 32*68 , LB=4, Performance= 641.15 GFlop/s, Time= 3.349 msec
//	(3) 16*32 , LB=3, Performance= 646.47 GFlop/s, Time= 3.322 msec
//	(4)  8*16 , LB=2, Performance= 237.34 GFlop/s, Time= 9.048 msec
//for 128*128*128:
//	(1)  4*8  , LB=1, Performance= 41.45 GFlop/s, Time= 0.101 msec
//for 8*1024: LB=2, Performance= 114.54 GFlop/s, Time= 0.146 msec

template<int LB, int STEP>
__global__ void kernel_2_4_K2pow(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float*  __restrict__ C,
	int SB, int LK)
{
	bool buf = 0;
	__shared__ float  As[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2 Bs[2][1 << LB >> 1][(2 << LB) + 2];

	int Y = (blockIdx.y << 1 << LB);//Y = blockIdx.y * 2 * BLOCK_SIZE
	int X = (blockIdx.x << 2 << LB);//X = blockIdx.x * 4 * BLOCK_SIZE
	A = A + (Y << LK);//A[Y, 0]
	B = B + X;//B[0, X]
	C = &get(C, Y, X, SB);//C[Y, X]

	int tx = threadIdx.x, ty = threadIdx.y;
	const int Ax = (tx >> 1), Ay = (ty << 1) + (tx & 1);
	const int By = (ty >> 1), Bx = (tx << 2) + ((ty & 1) << 1);
	const int As_x = Ax, As_y = ((ty << 1) + (tx & 1));
	const int Bs_y = By, Bs_x = ((tx << 1) + (ty & 1));
	const int Ayx = (Ay << LK) + Ax;//[Ay, Ax]
	const int Byx = By * SB + Bx;//[By, Bx]

	As[buf][As_x][As_y] = A[Ayx]; //transpose A
	Bs[buf][Bs_y][Bs_x] = *(float2*)(B + Byx);
	A += STEP; B += (SB << LB >> 1);//B += SB * STEP
	__syncthreads();

	//compute area--------------------------------------------
	float4 c0 = make_float4(0, 0, 0, 0);
	float4 c1 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (2 << LK >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float2 a = *(float2*)(&As[buf][ik][ty << 1]);
			float4 b = *(float4*)(&Bs[buf][ik][tx << 1]);
			simdMM4(c0, a.x, b);
			simdMM4(c1, a.y, b);
		}
		buf ^= 1;

		As[buf][As_x][As_y] = A[Ayx]; //transpose A
		Bs[buf][Bs_y][Bs_x] = *(float2*)(B + Byx);
		A += STEP; B += (SB << LB >> 1);//B += SB * STEP
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float2 a = *(float2*)(&As[buf][ik][ty << 1]);
		float4 b = *(float4*)(&Bs[buf][ik][tx << 1]);
		simdMM4(c0, a.x, b);
		simdMM4(c1, a.y, b);
	}

	ty <<= 1; tx <<= 2;
	*(float4*)(&get(C, ty, tx, SB)) = c0;
	*(float4*)(&get(C, ty + 1, tx, SB)) = c1;
}

#endif


//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*2), K % BLOCK_SIZE == 0
#ifndef MATMUL_KERNEL_2_2K2POW
#define MATMUL_KERNEL_2_2K2POW

//(correct)
//LB = 4: Size = 1, Time = 3.45 msec, Performace = 622.459 GFlop/s
//for 1024*1024*1024:
//	(1) 64*64, LB=5, Performance= 501.68 GFlop/s, Time= 4.281 msec
//	(2) 32*32, LB=4, Performance= 421.54 GFlop/s, Time= 5.094 msec
//	(3) 16*16, LB=3, Performance= 480.90 GFlop/s, Time= 4.466 msec
//	(4)  8*8 , LB=2, Performance= 185.69 GFlop/s, Time= 11.565 msec
//for 128*128*128:
//	(1)  8*8 , LB=2, Performance= 149.49 GFlop/s, Time= 0.028 msec
//	(2)  4*4 , LB=1, Performance= 38.99 GFlop/s, Time= 0.108 msec
//	(3)  2*2 , LB=0, Performance= 6.87 GFlop/s, Time= 0.610 msec
//for 32*32*32:
//	(1) 32*32, LB=4, Performance= 15.54 GFlop/s, Time= 0.004 msec
//	(2) 16*16, LB=3, Performance= 13.13 GFlop/s, Time= 0.005 msec
//	(3)  8*8 , LB=2, Performance= 15.68 GFlop/s, Time= 0.004 msec
//for 1024*16: LB=3, Performance= 164.45 GFlop/s, Time= 0.204 msec
//for  8*1024: LB=2, Performance= 173.61 GFlop/s, Time= 0.097 msec
//for  1024*8: LB=2, Performance= 114.02 GFlop/s, Time= 0.147 msec

template<int LB, int STEP>
__global__ void kernel_2_2_K2pow(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float*  __restrict__ C,
	int SB, int LK)
{
	bool buf = 0;
	__shared__ float2 As[2][1 << LB][(1 << LB) + 1];
	__shared__ float2 Bs[2][1 << LB][(1 << LB) + 1];

	int Y = (blockIdx.y << 1 << LB);//Y = blockIdx.y * 2 * BLOCK_SIZE
	int X = (blockIdx.x << 1 << LB);//X = blockIdx.x * 2 * BLOCK_SIZE
	A = A + (Y << LK);//A[Y, 0]
	B = B + X;//B[0, X]
	C = &get(C, Y, X, SB);//C[Y, X]

	int tx = threadIdx.x, ty = threadIdx.y;
	const int Byx = ty * SB + (tx << 1);//[ty, tx<<1]

	As[buf][tx][ty].x = lget(A, (ty << 1)    , tx, LK);//transpose A
	As[buf][tx][ty].y = lget(A, (ty << 1) + 1, tx, LK);
	Bs[buf][ty][tx] = *(float2*)(B + Byx);
	A += STEP; B += (SB << LB);//B += SB * STEP
	__syncthreads();

	//compute area------------------------------------------
	float2 c0 = make_float2(0, 0);
	float2 c1 = make_float2(0, 0);
	for (int ok = 1, OK = (1 << LK >> LB); ok < OK; ok++)
	{
#pragma once
		for (int ik = 0; ik < STEP; ik++) {
			float2 a = As[buf][ik][ty];
			float2 b = Bs[buf][ik][tx];
			simdMM2(c0, a.x, b);
			simdMM2(c1, a.y, b);
		}
		buf ^= 1;

		As[buf][tx][ty].x = lget(A, (ty << 1)    , tx, LK);//transpose A
		As[buf][tx][ty].y = lget(A, (ty << 1) + 1, tx, LK);
		Bs[buf][ty][tx] = *(float2*)(B + Byx);
		A += STEP; B += (SB << LB);//B += SB * STEP
		__syncthreads();
	}
#pragma once
	for (int ik = 0; ik < STEP; ik++) {
		float2 a = As[buf][ik][ty];
		float2 b = Bs[buf][ik][tx];
		simdMM2(c0, a.x, b);
		simdMM2(c1, a.y, b);
	}

	ty <<= 1; tx <<= 1;
	*(float2*)(&get(C, ty    , tx, SB)) = c0;
	*(float2*)(&get(C, ty + 1, tx, SB)) = c1;
}

#endif

//(Y: N, X: M), N*M<=1024
#ifndef MATMUL_KERNEL_NAIVE_K2POW
#define MATMUL_KERNEL_NAIVE_K2POW

//(correct)
//for 32*32*32: Performance= 11.43 GFlop/s, Time= 0.006 msec,
//for 31*31*32: Performance=  9.59 GFlop/s, Time= 0.006 msec,
//for 4*256: Performance= 20.03 GFlop/s, Time= 0.105 msec
//for 256*4: Performance= 16.48 GFlop/s, Time= 0.127 msec

__global__ void kernel_naive_K2pow(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float*  __restrict__ C,
	int SB, int LK)
{
	int y = threadIdx.y + blockIdx.y*blockDim.y;
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	float c = 0;
	for (int k = 0, K = (1 << LK); k < K; k++) 
		c += lget(A, y, k, LK) * get(B, k, x, SB);
	get(C, y, x, SB) = c;
}
#endif

#endif