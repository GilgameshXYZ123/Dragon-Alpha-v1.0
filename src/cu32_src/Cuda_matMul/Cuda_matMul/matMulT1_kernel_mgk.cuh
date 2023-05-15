#pragma once

#ifndef MUTMUL_T1_KERNEL_MGK_H
#define MUTMUL_T1_KERNEL_MGK_H

//A   belongs to Mat[K, N]
//A^T belongs to Mat[N, K]
//get(A^T, i, k, K) = get(A, k, i, N)
//SB: stride of Matrix B: M
//K: stride of Matrix A
#ifndef MUTMUL_T1_KERNEL_MGK_CALL
#define MUTMUL_T1_KERNEL_MGK_CALL

#define	k88T1_mgk(LB, stream, A, B, C, N, M, K, SA, SB) \
	kernel_t1_8_8_MGK<LB, (1<<LB>>1)>\
		<<< dim3(M>>3>>LB, N>>3>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SA, SB, K)

#define k84T1_mgk(LB, stream, A, B, C, N, M, K, SA, SB) \
	kernel_t1_8_4_MGK<LB, (1<<LB>>1)>\
		<<< dim3(M>>2>>LB, N>>3>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SA, SB, K)

#define k48T1_mgk(LB, stream, A, B, C, N, M, K, SA, SB) \
	kernel_t1_4_8_MGK<LB, (1<<LB>>1)>\
		<<< dim3(M>>3>>LB, N>>2>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SA, SB, K)

#define k82T1_mgk(LB, stream, A, B, C, N, M, K, SA, SB) \
	kernel_t1_8_2_MGK<LB, (1<<LB>>1)>\
		<<< dim3(M>>1>>LB, N>>3>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SA, SB, K)

#define k28T1_mgk(LB, stream, A, B, C, N, M, K, SA, SB) \
	kernel_t1_2_8_MGK<LB, (1<<LB>>1)>\
		<<< dim3(M>>3>>LB, N>>1>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SA, SB, K)

#define k81T1_mgk(LB, stream, A, B, C, N, M, K, SA, SB) \
	kernel_t1_8_1_MGK<LB, (1<<LB)>\
		<<< dim3(M>>LB, N>>3>>LB), dim3(1<<LB, 1<<LB), 0, stream >>> \
			(A, B, C, SA, SB, K)

#define k18T1_mgk(LB, stream, A, B, C, N, M, K, SA, SB) \
	kernel_t1_1_8_MGK<LB, (1<<LB)>\
		<<< dim3(M>>3>>LB, N>>LB), dim3(1<<LB, 1<<LB), 0, stream >>> \
			(A, B, C, SA, SB, K)

#define k44T1_mgk(LB, stream, A, B, C, N, M, K, SA, SB) \
	kernel_t1_4_4_MGK<LB, (1<<LB>>1)>\
		<<< dim3(M>>2>>LB, N>>2>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SA, SB, K)

#define k42T1_mgk(LB, stream, A, B, C, N, M, K, SA, SB) \
	kernel_t1_4_2_MGK<LB, (1<<LB>>1)>\
		<<< dim3(M>>1>>LB, N>>2>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SA, SB, K)

#define k24T1_mgk(LB, stream,  A, B, C, N, M, K, SA, SB) \
	kernel_t1_2_4_MGK<LB, (1<<LB>>1)>\
		<<< dim3(M>>2>>LB, N>>1>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SA, SB, K)

#define k22T1_mgk(LB, stream, A, B, C, N, M, K, SA, SB) \
	kernel_t1_2_2_MGK<LB, (1<<LB)>\
		<<< dim3(M>>1>>LB, N>>1>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SA, SB, K)

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), K % BLOCK_SIZE/2 == 0
//LB = 4: K % 8 == 0
#ifndef MUTMUL_T1_KERNEL_8_8_MGK
#define MUTMUL_T1_KERNEL_8_8_MGK

//LB = 4: Size = 1, Time = 1.26 msec, Performace = 1704.35 GFlop/s
template<int LB, int STEP>
__global__ void kernel_t1_8_8_MGK(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float*  __restrict__ C,
	int SA, int SB, int K)
{
	bool buf = 0;
	__shared__ float4 As[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Bs[2][1 << LB][(1 << LB) + 1];

	int Y = (blockIdx.y << 3 << LB);//Y = blockIdx.y * 8 * BLOCK_SIZE
	int X = (blockIdx.x << 3 << LB);//X = blockIdx.x * 8 * BLOCK_SIZE
	A = A + Y;//A[0, Y]
	B = B + X;//B[0, X]
	C = &get(C, Y, X, SB);//C[Y, X]

	int tx = threadIdx.x, ty = threadIdx.y;//Ax = tx - ((tx >= STEP) *STEP)
	const int Ay = (ty << 3) + ((tx >= STEP) << 2), Ax = tx - ((tx >= STEP) << LB >> 1);
	const int Bx = (tx << 3) + ((ty >= STEP) << 2), By = ty - ((ty >= STEP) << LB >> 1);
	const int Axy = Ax * SA + Ay;//[Ax, Ay]
	const int Byx = By * SB + Bx;//[By, Bx]

	As[buf][tx][ty] = *(float4*)(A + Axy);
	Bs[buf][ty][tx] = *(float4*)(B + Byx);
	A += (SA << LB >> 1); B += (SB << LB >> 1);//A += SA * STEP
	__syncthreads();

	//compute area-------------------------------------------------------
	float4  c0 = make_float4(0, 0, 0, 0), c1 = make_float4(0, 0, 0, 0);
	float4  c2 = make_float4(0, 0, 0, 0), c3 = make_float4(0, 0, 0, 0);
	float4  c4 = make_float4(0, 0, 0, 0), c5 = make_float4(0, 0, 0, 0);
	float4  c6 = make_float4(0, 0, 0, 0), c7 = make_float4(0, 0, 0, 0);
	float4  c8 = make_float4(0, 0, 0, 0), c9 = make_float4(0, 0, 0, 0);
	float4 c10 = make_float4(0, 0, 0, 0), c11 = make_float4(0, 0, 0, 0);
	float4 c12 = make_float4(0, 0, 0, 0), c13 = make_float4(0, 0, 0, 0);
	float4 c14 = make_float4(0, 0, 0, 0), c15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (K << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a0 = As[buf][ik][ty], a1 = As[buf][ik + STEP][ty];
			float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP][tx];

			simdMM4(c0, a0.x, b0); simdMM4(c1, a0.x, b1);
			simdMM4(c2, a0.y, b0); simdMM4(c3, a0.y, b1);
			simdMM4(c4, a0.z, b0); simdMM4(c5, a0.z, b1);
			simdMM4(c6, a0.w, b0); simdMM4(c7, a0.w, b1);
			simdMM4(c8, a1.x, b0); simdMM4(c9, a1.x, b1);
			simdMM4(c10, a1.y, b0); simdMM4(c11, a1.y, b1);
			simdMM4(c12, a1.z, b0); simdMM4(c13, a1.z, b1);
			simdMM4(c14, a1.w, b0); simdMM4(c15, a1.w, b1);
		}
		buf ^= 1;

		As[buf][tx][ty] = *(float4*)(A + Axy);
		Bs[buf][ty][tx] = *(float4*)(B + Byx);
		A += (SA << LB >> 1); B += (SB << LB >> 1);//A += SA * STEP
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a0 = As[buf][ik][ty], a1 = As[buf][ik + STEP][ty];
		float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP][tx];

		simdMM4(c0, a0.x, b0); simdMM4(c1, a0.x, b1);
		simdMM4(c2, a0.y, b0); simdMM4(c3, a0.y, b1);
		simdMM4(c4, a0.z, b0); simdMM4(c5, a0.z, b1);
		simdMM4(c6, a0.w, b0); simdMM4(c7, a0.w, b1);
		simdMM4(c8, a1.x, b0); simdMM4(c9, a1.x, b1);
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


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*4), K % BLOCK_SIZE/2 == 0
//LB = 4: K % 8 == 0
#ifndef MUTMUL_T1_KERNEL_8_4_MGK
#define MUTMUL_T1_KERNEL_8_4_MGK

//LB = 4: Size = 1, Time = 1.508 msec, Performace = 1424.06 GFlop/s
template<int LB, int STEP>
__global__ void kernel_t1_8_4_MGK(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float*  __restrict__ C,
	int SA, int SB, int K)
{
	bool buf = 0;
	__shared__ float4 As[2][1 << LB][(1 << LB) + 1];//followed k88T1
	__shared__ float2 Bs[2][1 << LB >> 1][(2 << LB) + 2];//followed k44T1

	int Y = (blockIdx.y << 3 << LB);//Y = blockIdx.y * 8 * BLOCK_SIZE
	int X = (blockIdx.x << 2 << LB);//X = blockIdx.x * 4 * BLOCK_SIZE;
	A = A + Y;//A[0, Y]
	B = B + X;//B[0, X]
	C = &get(C, Y, X, SB);//C[Y, X]

	int tx = threadIdx.x, ty = threadIdx.y;
	const int Ay = (ty << 3) + ((tx >= STEP) << 2), Ax = tx - ((tx >= STEP) << LB >> 1);
	const int By = (ty >> 1), Bx = (tx << 2) + ((ty & 1) << 1);
	const int Bs_y = By, Bs_x = ((tx << 1) + (ty & 1));
	const int Axy = Ax * SA + Ay;//[Ax, Ay]
	const int Byx = By * SB + Bx;//[By, Bx]

	As[buf][tx][ty] = *(float4*)(A + Axy);
	Bs[buf][Bs_y][Bs_x] = *(float2*)(B + Byx);
	A += (SA << LB >> 1); B += (SB << LB >> 1);//A += SA * STEP
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

	for (int ok = 1, OK = (K << 1 >> LB); ok < OK; ok++)
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

		As[buf][tx][ty] = *(float4*)(A + Axy);
		Bs[buf][Bs_y][Bs_x] = *(float2*)(B + Byx);
		A += (SA << LB >> 1); B += (SB << LB >> 1);//A += SA * STEP
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
//LB = 4: K % 8 == 0
#ifndef MUTMUL_T1_KERNEL_4_8_MGK
#define MUTMUL_T1_KERNEL_4_8_MGK

//LB = 4: Size = 1, Time = 1.664 msec, Performace = 1290.55 GFlop/s
template<int LB, int STEP>
__global__ void kernel_t1_4_8_MGK(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float*  __restrict__ C,
	int SA, int SB, int K)
{
	bool buf = 0;
	__shared__ float2 As[2][1 << LB >> 1][(2 << LB) + 2];//followed k44T1
	__shared__ float4 Bs[2][1 << LB][(1 << LB) + 1];//followed k88T1

	int Y = (blockIdx.y << 2 << LB);//Y = blockIdx.y * 4 * BLOCK_SIZE
	int X = (blockIdx.x << 3 << LB);//X = blockIdx.x * 8 * BLOCK_SIZE
	A = A + Y;//A[0, Y]
	B = B + X;//B[0, X]
	C = &get(C, Y, X, SB);//C[Y, X]

	int tx = threadIdx.x, ty = threadIdx.y;
	const int Ax = (tx >> 1), Ay = (ty << 2) + ((tx & 1) << 1);
	const int As_x = Ax, As_y = ((ty << 1) + (tx & 1));
	const int Bx = (tx << 3) + ((ty >= STEP) << 2), By = ty - ((ty >= STEP) << LB >> 1);
	const int Axy = Ax * SA + Ay;//[Ax, Ay]
	const int Byx = By * SB + Bx;//[By, Bx]

	As[buf][As_x][As_y] = *(float2*)(A + Axy);
	Bs[buf][ty][tx] = *(float4*)(B + Byx);
	A += (SA << LB >> 1); B += (SB << LB >> 1);//A += SA * STEP
	__syncthreads();

	//compute area-------------------------------------------------------
	float4 c0 = make_float4(0, 0, 0, 0), c1 = make_float4(0, 0, 0, 0);
	float4 c2 = make_float4(0, 0, 0, 0), c3 = make_float4(0, 0, 0, 0);
	float4 c4 = make_float4(0, 0, 0, 0), c5 = make_float4(0, 0, 0, 0);
	float4 c6 = make_float4(0, 0, 0, 0), c7 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (K << 1 >> LB); ok < OK; ok++)
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

		As[buf][As_x][As_y] = *(float2*)(A + Axy);
		Bs[buf][ty][tx] = *(float4*)(B + Byx);
		A += (SA << LB >> 1); B += (SB << LB >> 1);//A += SA * STEP
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
//LB = 4: K % 8 == 0
#ifndef MUTMUL_T1_KERNEL_8_2_MGK
#define MUTMUL_T1_KERNEL_8_2_MGK

//LB = 4: Size = 1, Time = 2.022 msec, Performace = 1062.06 GFlop/s
//LB = 4: K % 8 == 0
template<int LB, int STEP>
__global__ void kernel_t1_8_2_MGK(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float*  __restrict__ C,
	int SA, int SB, int K)
{
	bool buf = 0;
	__shared__ float4 As[2][1 << LB][(1 << LB) + 1];//followed k88T1
	__shared__ float  Bs[2][1 << LB >> 1][(2 << LB) + 2];//followed k44T1

	int Y = (blockIdx.y << 3 << LB);//Y = blockIdx.y * 8 * BLOCK_SIZE
	int X = (blockIdx.x << 1 << LB);//X = blockIdx.x * 2 * BLOCK_SIZE;
	A = A + Y;//A[0, Y]
	B = B + X;//B[0, X]
	C = &get(C, Y, X, SB);//C[Y, X]

	int tx = threadIdx.x, ty = threadIdx.y;
	const int Ay = (ty << 3) + ((tx >= STEP) << 2), Ax = tx - ((tx >= STEP) << LB >> 1);
	const int By = (ty >> 1), Bx = ((tx << 1) + (ty & 1));
	const int Bs_y = By, Bs_x = ((tx << 1) + (ty & 1));
	const int Axy = Ax * SA + Ay;//[Ax, Ay]
	const int Byx = By * SB + Bx;//[By, Bx]

	As[buf][tx][ty] = *(float4*)(A + Axy);
	Bs[buf][Bs_y][Bs_x] = B[Byx];
	A += (SA << LB >> 1); B += (SB << LB >> 1);//A += SA * STEP
	__syncthreads();

	//compute area----------------------------------------------------
	float2  c0 = make_float2(0, 0);
	float2  c2 = make_float2(0, 0);
	float2  c4 = make_float2(0, 0);
	float2  c6 = make_float2(0, 0);
	float2  c8 = make_float2(0, 0);
	float2 c10 = make_float2(0, 0);
	float2 c12 = make_float2(0, 0);
	float2 c14 = make_float2(0, 0);

	for (int ok = 1, OK = (K << 1 >> LB); ok < OK; ok++)
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

		As[buf][tx][ty] = *(float4*)(A + Axy);
		Bs[buf][Bs_y][Bs_x] = B[Byx];
		A += (SA << LB >> 1); B += (SB << LB >> 1);//A += SA * STEP
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
//LB = 4: K % 8 == 0
#ifndef MUTMUL_T1_KERNEL_2_8_MGK
#define MUTMUL_T1_KERNEL_2_8_MGK

//LB = 4: Size = 1, Time = 2.688 msec, Performace = 798.915 GFlop/s
template<int LB, int STEP>
__global__ void kernel_t1_2_8_MGK(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float*  __restrict__ C,
	int SA, int SB, int K)
{
	bool buf = 0;
	__shared__ float  As[2][1 << LB >> 1][(2 << LB) + 2];//followed k44T1
	__shared__ float4 Bs[2][1 << LB][(1 << LB) + 1];//followed k88T1

	int Y = (blockIdx.y << 1 << LB);//Y = blockIdx.y * 2 * BLOCK_SIZE
	int X = (blockIdx.x << 3 << LB);//X = blockIdx.x * 8 * BLOCK_SIZE
	A = A + Y;//A[0, Y]
	B = B + X;//B[0, X]
	C = &get(C, Y, X, SB);//C[Y, X]

	int tx = threadIdx.x, ty = threadIdx.y;
	const int Ax = (tx >> 1), Ay = (ty << 1) + (tx & 1);
	const int As_x = Ax, As_y = ((ty << 1) + (tx & 1));
	const int Bx = (tx << 3) + ((ty >= STEP) << 2), By = ty - ((ty >= STEP) << LB >> 1);
	const int Axy = Ax * SA + Ay;//[Ax, Ay]
	const int Byx = By * SB + Bx;//[By, Bx]

	As[buf][As_x][As_y] = A[Axy];
	Bs[buf][ty][tx] = *(float4*)(B + Byx);
	A += (SA << LB >> 1); B += (SB << LB >> 1);//A += SA * STEP
	__syncthreads();

	//compute area----------------------------------------------------
	float4 c0 = make_float4(0, 0, 0, 0), c1 = make_float4(0, 0, 0, 0);
	float4 c2 = make_float4(0, 0, 0, 0), c3 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (K << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float2 a0 = *(float2*)(&As[buf][ik][ty << 1]);
			float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP][tx];
			simdMM4(c0, a0.x, b0); simdMM4(c1, a0.x, b1);
			simdMM4(c2, a0.y, b0); simdMM4(c3, a0.y, b1);
		}
		buf ^= 1;

		As[buf][As_x][As_y] = A[Axy];
		Bs[buf][ty][tx] = *(float4*)(B + Byx);
		A += (SA << LB >> 1); B += (SB << LB >> 1);//A += SA * STEP
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


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE  ), K % BLOCK_SIZE == 0
#ifndef MUTMUL_T1_KERsNEL_8_1_MGK
#define MUTMUL_T1_KENERL_8_1_MGK

//LB = 4: Size = 1, Time = 3.092 msec, Performace = 694.529 GFlop/s
template<int LB, int STEP>
__global__ void kernel_t1_8_1_MGK(
	const float* __restrict__ A,
	const float* __restrict__ B,
	float* __restrict__ C,
	int SA, int SB, int K)
{
	bool buf = 0;
	__shared__ float4 As[2][1 << LB][(2 << LB) + 1];
	__shared__ float  Bs[2][1 << LB][(1 << LB) + 1];

	int Y = (blockIdx.y << 3 << LB);//Y = blockIdx.y * 2 * BLOCK_SIZE
	int X = (blockIdx.x << LB);//X = blockIdx.x * BLOCK_SIZE
	A = A + Y;//A[0, Y]
	B = B + X;//B[0, X]
	C = &get(C, Y, X, SB);//C[Y, X]

	int ty = threadIdx.y, tx = threadIdx.x;
	const int Axy = tx * SA + (ty << 3);//[tx, (ty<<3)]
	const int Byx = ty * SB + tx;//[ty, tx]

	As[buf][tx][(ty << 1)] = *(float4*)(A + Axy);
	As[buf][tx][(ty << 1) + 1] = *(float4*)(A + Axy + 4);
	Bs[buf][ty][tx] = B[Byx];
	A += (SA << LB); B += (SB << LB);//A += SA * STEP
	__syncthreads();

	//compute area--------------------------------------------------
	float4 c0 = make_float4(0, 0, 0, 0);
	float4 c1 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (K >> LB); ok < OK; ok++)
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

		As[buf][tx][(ty << 1)] = *(float4*)(A + Axy);
		As[buf][tx][(ty << 1) + 1] = *(float4*)(A + Axy + 4);
		Bs[buf][ty][tx] = B[Byx];
		A += (SA << LB); B += (SB << LB);//A += SA * STEP
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


//(Y: BLOCK_SIZE  , X: BLOCK_SIZE*8), K % BLOCK_SIZE == 0
#ifndef MUTMUL_T1_KERNEL_1_8_MGK
#define MUTMUL_T1_KERNEL_1_8_MGK

//LB = 4: Size = 1, Time = 8.886 msec, Performace = 241.67 GFlop/s
template<int LB, int STEP>
__global__ void kernel_t1_1_8_MGK(
	const float* __restrict__ A,
	const float* __restrict__ B,
	float* __restrict__ C,
	int SA, int SB, int K)
{
	bool buf = 0;
	__shared__ float   As[2][1 << LB][(1 << LB) + 1];
	__shared__ float4  Bs[2][1 << LB][(2 << LB) + 2];

	int Y = (blockIdx.y << LB);//Y = blockIdx.y * BLOCK_SIZE
	int X = (blockIdx.x << 3 << LB);//X = blockIdx.x * 8 * BLOCK_SIZE
	A = A + Y;//A[0, Y]
	B = B + X;//B[0, X]
	C = &get(C, Y, X, SB);//C[Y, X]

	int ty = threadIdx.y, tx = threadIdx.x;
	const int Axy = tx * SA + ty;//[tx, ty]
	const int Byx = ty * SB + (tx << 3);//[ty, (tx<<3)]

	As[buf][tx][ty] = A[Axy];
	Bs[buf][ty][(tx << 1)] = *(float4*)(B + Byx);
	Bs[buf][ty][(tx << 1) + 1] = *(float4*)(B + Byx + 4);
	A += (SA << LB); B += (SB << LB);//A += SA * STEP
	__syncthreads();

	//compute area--------------------------------------------------
	float4 c0 = make_float4(0, 0, 0, 0);
	float4 c1 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (K >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float a = As[buf][ik][ty];
			float4 b0 = Bs[buf][ik][(tx << 1)];
			float4 b1 = Bs[buf][ik][(tx << 1) + 1];
			simdMM4(c0, a, b0); simdMM4(c1, a, b1);
		}
		buf ^= 1;

		As[buf][tx][ty] = A[Axy];
		Bs[buf][ty][(tx << 1)] = *(float4*)(B + Byx);
		Bs[buf][ty][(tx << 1) + 1] = *(float4*)(B + Byx + 4);
		A += (SA << LB); B += (SB << LB);//A += SA * STEP
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
#ifndef MUTMUL_T1_KERNEL_4_4_MGK
#define MUTMUL_T1_KERNEL_4_4_MGK

//LB = 4: Size = 1, Time = 1.958 msec, Performace = 1096.77 GFlop/s
template<int LB, int STEP>
__global__ void kernel_t1_4_4_MGK(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float*  __restrict__ C,
	int SA, int SB, int K)
{
	bool buf = 0;
	__shared__ float2 As[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2 Bs[2][1 << LB >> 1][(2 << LB) + 2];

	int Y = (blockIdx.y << 2 << LB);//Y = blockIdx.y * 4 * BLOCK_SIZE
	int X = (blockIdx.x << 2 << LB);//X = blockIdx.x * 4 * BLOCK_SIZE;
	A = A + Y;//A[0, Y]
	B = B + X;//B[0, X]
	C = &get(C, Y, X, SB);//C[Y, X]

	int tx = threadIdx.x, ty = threadIdx.y;
	const int Ax = (tx >> 1), Ay = (ty << 2) + ((tx & 1) << 1);
	const int By = (ty >> 1), Bx = (tx << 2) + ((ty & 1) << 1);
	const int As_x = Ax, As_y = ((ty << 1) + (tx & 1));
	const int Bs_y = By, Bs_x = ((tx << 1) + (ty & 1));
	const int Axy = Ax * SA + Ay;//[Ax, Ay]
	const int Byx = By * SB + Bx;//[By, Bx]

	As[buf][As_x][As_y] = *(float2*)(A + Axy);
	Bs[buf][Bs_y][Bs_x] = *(float2*)(B + Byx);
	A += (SA << LB >> 1); B += (SB << LB >> 1);//A += SA * STEP
	__syncthreads();

	//compute area----------------------------------------------------
	float4 c0 = make_float4(0, 0, 0, 0);
	float4 c1 = make_float4(0, 0, 0, 0);
	float4 c2 = make_float4(0, 0, 0, 0);
	float4 c3 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (K << 1 >> LB); ok < OK; ok++)
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

		As[buf][As_x][As_y] = *(float2*)(A + Axy);
		Bs[buf][Bs_y][Bs_x] = *(float2*)(B + Byx);
		A += (SA << LB >> 1); B += (SB << LB >> 1);//A += SA * STEP
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
#ifndef MUTMUL_T1_KERNEL_4_2_MGK
#define MUTMUL_T1_KERNEL_4_2_MGK

//LB = 4: Size = 1, Time = 2.552 msec, Performace = 841.49 GFlop/s
template<int LB, int STEP>
__global__ void kernel_t1_4_2_MGK(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float*  __restrict__ C,
	int SA, int SB, int K)
{
	bool buf = 0;
	__shared__ float2 As[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float  Bs[2][1 << LB >> 1][(2 << LB) + 2];

	int Y = (blockIdx.y << 2 << LB);//Y = blockIdx.y * 4 * BLOCK_SIZE
	int X = (blockIdx.x << 1 << LB);//X = blockIdx.x * 2 * BLOCK_SIZE;
	A = A + Y;//A[0, Y]
	B = B + X;//B[0, X]
	C = &get(C, Y, X, SB);//C[Y, X]

	int tx = threadIdx.x, ty = threadIdx.y;
	const int Ax = (tx >> 1), Ay = (ty << 2) + ((tx & 1) << 1);
	const int By = (ty >> 1), Bx = (tx << 1) + (ty & 1);
	const int Axy = Ax * SA + Ay;//[Ax, Ay]
	const int Byx = By * SB + Bx;//[By, Bx]
	const int As_x = Ax, As_y = ((ty << 1) + (tx & 1));
	const int Bs_y = By, Bs_x = ((tx << 1) + (ty & 1));

	As[buf][As_x][As_y] = *(float2*)(A + Axy);
	Bs[buf][Bs_y][Bs_x] = B[Byx];
	A += (SA << LB >> 1); B += (SB << LB >> 1);//A += SA * STEP
	__syncthreads();

	//compute area----------------------------------------------------
	float2 c0 = make_float2(0, 0);
	float2 c1 = make_float2(0, 0);
	float2 c2 = make_float2(0, 0);
	float2 c3 = make_float2(0, 0);
	for (int ok = 1, OK = (K << 1 >> LB); ok < OK; ok++)
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

		As[buf][As_x][As_y] = *(float2*)(A + Axy);
		Bs[buf][Bs_y][Bs_x] = B[Byx];
		A += (SA << LB >> 1); B += (SB << LB >> 1);//A += SA * STEP
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
#ifndef MUTMUL_T1_KERNEL_2_4_MGK
#define MUTMUL_T1_KERNEL_2_4_MGK

//LB = 4: Size = 1, Time = 3.152 msec, Performace = 681.308 GFlop/s
template<int LB, int STEP>
__global__ void kernel_t1_2_4_MGK(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float*  __restrict__ C,
	int SA, int SB, int K)
{
	bool buf = 0;
	__shared__ float  As[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2 Bs[2][1 << LB >> 1][(2 << LB) + 2];

	int Y = (blockIdx.y << 1 << LB);//Y = blockIdx.y * 2 * BLOCK_SIZE
	int X = (blockIdx.x << 2 << LB);//X = blockIdx.x * 4 * BLOCK_SIZE;
	A = A + Y;//A[0, Y]
	B = B + X;//B[0, X]
	C = &get(C, Y, X, SB);//C[Y, X]

	int tx = threadIdx.x, ty = threadIdx.y;
	const int Ax = (tx >> 1), Ay = (ty << 1) + (tx & 1);
	const int By = (ty >> 1), Bx = (tx << 2) + ((ty & 1) << 1);
	const int As_x = Ax, As_y = ((ty << 1) + (tx & 1));
	const int Bs_y = By, Bs_x = ((tx << 1) + (ty & 1));
	const int Axy = Ax * SA + Ay;//[Ax, Ay]
	const int Byx = By * SB + Bx;//[By, Bx]

	As[buf][As_x][As_y] = A[Axy];
	Bs[buf][Bs_y][Bs_x] = *(float2*)(B + Byx);
	A += (SA << LB >> 1); B += (SB << LB >> 1);//A += SA * STEP
	__syncthreads();

	//compute area----------------------------------------------------
	float4 c0 = make_float4(0, 0, 0, 0);
	float4 c1 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (K << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float2 a = *(float2*)(&As[buf][ik][ty << 1]);
			float4 b = *(float4*)(&Bs[buf][ik][tx << 1]);
			simdMM4(c0, a.x, b);
			simdMM4(c1, a.y, b);
		}
		buf ^= 1;

		As[buf][As_x][As_y] = A[Axy];
		Bs[buf][Bs_y][Bs_x] = *(float2*)(B + Byx);
		A += (SA << LB >> 1); B += (SB << LB >> 1);//A += SA * STEP
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
	*(float4*)(&get(C, ty    , tx, SB)) = c0;
	*(float4*)(&get(C, ty + 1, tx, SB)) = c1;
}

#endif


//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*2), K % BLOCK_SIZE/2 == 0
#ifndef MUTMUL_T1_KERNEL_2_2K2POW
#define MUTMUL_T1_KERNEL_2_2K2POW

//LB = 4: Size = 1, Time = 3.442 msec, Performace = 623.906 GFlop/s
template<int LB, int STEP>
__global__ void kernel_t1_2_2_MGK(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float*  __restrict__ C,
	int SA, int SB, int K)
{
	bool buf = 0;
	__shared__ float2 As[2][1 << LB][(1 << LB) + 1];
	__shared__ float2 Bs[2][1 << LB][(1 << LB) + 1];

	int Y = (blockIdx.y << 1 << LB);//Y = blockIdx.y * 2 * BLOCK_SIZE
	int X = (blockIdx.x << 1 << LB);//X = blockIdx.x * 2 * BLOCK_SIZE
	A = A + Y;//A[0, Y]
	B = B + X;//B[0, X]
	C = &get(C, Y, X, SB);//C[Y, X]

	int tx = threadIdx.x, ty = threadIdx.y;
	const int Axy = tx * SA + (ty << 1);
	const int Byx = ty * SB + (tx << 1);

	As[buf][tx][ty] = *(float2*)(A + Axy);
	Bs[buf][ty][tx] = *(float2*)(B + Byx);
	A += (SA << LB); B += (SB << LB);//A += SA * STEP
	__syncthreads();

	//compute area--------------------------------------------------
	float2 c0 = make_float2(0, 0);
	float2 c1 = make_float2(0, 0);
	for (int ok = 1, OK = (K >> LB); ok < OK; ok++)
	{
#pragma once
		for (int ik = 0; ik < STEP; ik++) {
			float2 a = As[buf][ik][ty];
			float2 b = Bs[buf][ik][tx];
			simdMM2(c0, a.x, b);
			simdMM2(c1, a.y, b);
		}
		buf ^= 1;

		As[buf][tx][ty] = *(float2*)(A + Axy);
		Bs[buf][ty][tx] = *(float2*)(B + Byx);
		A += (SA << LB); B += (SB << LB);//A += SA * STEP
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

#endif