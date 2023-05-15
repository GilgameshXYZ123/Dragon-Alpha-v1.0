#pragma once

#ifndef MUTMUL_T1_KERNEL_H
#define MUTMUL_T1_KERNEL_H

//A   belongs to Mat[K, N]
//A^T belongs to Mat[N, K]
//get(A^T, i, k, K) = get(A, k, i, N)
//SB: stride of Matrix B: M
//K: stride of Matrix A
#ifndef MUTMUL_T1_KERNEL_CALL
#define MUTMUL_T1_KERNEL_CALL

//LB = log2(BLOCK_SIZE)

#define	k88T1(LB, stream, A, B,  C, N, M, K, SA, SB) \
	kernel_t1_8_8<LB, (1<<LB>>1)>\
		<<< dim3(M>>3>>LB, N>>3>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SA, SB, K)

#define k84T1(LB, stream, A, B, C, N, M, K, SA, SB) \
	kernel_t1_8_4<LB, (1<<LB>>1)>\
		<<< dim3(M>>2>>LB, N>>3>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SA, SB, K)

#define k48T1(LB, stream, A, B, C, N, M, K, SA, SB) \
	kernel_t1_4_8<LB, (1<<LB>>1)>\
		<<< dim3(M>>3>>LB, N>>2>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SA, SB, K)

#define k82T1(LB, stream, A, B, C, N, M, K, SA, SB) \
	kernel_t1_8_2<LB, (1<<LB>>1)>\
		<<< dim3(M>>1>>LB, N>>3>>LB), dim3(1<<LB, 1<<LB), 0, stream >>> \
			(A, B, C, SA, SB, K)

#define k28T1(LB, stream, A, B, C, N, M, K, SA, SB) \
	kernel_t1_2_8<LB, (1<<LB>>1)>\
		<<< dim3(M>>3>>LB, N>>1>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SA, SB, K)

#define k81T1(LB, stream, A, B, C, N, M, K, SA, SB) \
	kernel_t1_8_1<LB, (1<<LB)>\
		<<< dim3(M>>LB, N>>3>>LB), dim3(1<<LB, 1<<LB), 0, stream >>> \
			(A, B, C, SA, SB, K)

#define k18T1(LB, stream, A, B, C, N, M, K, SA, SB) \
	kernel_t1_1_8<LB, (1<<LB)>\
		<<< dim3(M>>3>>LB, N>>LB), dim3(1<<LB, 1<<LB), 0, stream >>> \
			(A, B, C, SA, SB, K)

#define k44T1(LB, stream, A, B, C, N, M, K, SA, SB) \
	kernel_t1_4_4<LB, (1<<LB>>1)>\
		<<< dim3(M>>2>>LB, N>>2>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SA, SB, K)

#define k42T1(LB, stream, A, B, C, N, M, K, SA, SB) \
	kernel_t1_4_2<LB, (1<<LB>>1)>\
		<<< dim3(M>>1>>LB, N>>2>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SA, SB, K)

#define k24T1(LB, stream, A, B, C, N, M, K, SA, SB) \
	kernel_t1_2_4<LB, (1<<LB>>1)>\
		<<< dim3(M>>2>>LB, N>>1>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SA, SB, K)

#define k22T1(LB, stream, A, B, C, N, M, K, SA, SB) \
	kernel_t1_2_2<LB, (1<<LB)>\
		<<< dim3(M>>1>>LB, N>>1>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SA, SB, K)

//correct
#define knaiveT1(GRID_SIZE, stream, A, B, C, N, M, K, SA, SB) \
	kernel_t1_naive\
		<<< dim3(GRID_SIZE, GRID_SIZE), dim3(M/GRID_SIZE, N/GRID_SIZE), 0, stream>>>\
			(A, B, C, SA, SB, K)
#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), K >= BLOCK_SIZE/2
//LB = 4: K >= 8
#ifndef MUTMUL_T1_KERNEL_8_8
#define MUTMUL_T1_KERNEL_8_8

//LB = 4: Size = 1, Time = 1.27  msec, Performace = 1690.93 GFlop/s
//LB = 3: Size = 1, Time = 1.401 msec, Performace = 1532.82 GFlop/s
template<int LB, int STEP>
__global__ void kernel_t1_8_8(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float*  __restrict__ C,
	int SA, int SB, int K)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 As[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Bs[2][1 << LB][(1 << LB) + 1];

	const int Y = ((blockIdx.y << LB) + ty) << 3; A += Y;
	const int X = ((blockIdx.x << LB) + tx) << 3; B += X;
	C = &get(C, Y, X, SB);//C[Y, X]
	const int Ax = tx - ((tx >= STEP) << LB >> 1);
	const int By = ty - ((ty >= STEP) << LB >> 1);
	const int A0 = Ax * SA + ((tx >= STEP) << 2);
	const int B0 = By * SB + ((ty >= STEP) << 2);

	As[buf][tx][ty] = *(float4*)(A + A0);
	Bs[buf][ty][tx] = *(float4*)(B + B0);
	A += (SA << LB >> 1); B += (SB << LB >> 1);//K += STEP
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

			simdMM4(c0, a0.x, b0);  simdMM4(c1, a0.x, b1);
			simdMM4(c2, a0.y, b0);  simdMM4(c3, a0.y, b1);
			simdMM4(c4, a0.z, b0);  simdMM4(c5, a0.z, b1);
			simdMM4(c6, a0.w, b0);  simdMM4(c7, a0.w, b1);
			simdMM4(c8, a1.x, b0);  simdMM4(c9, a1.x, b1);
			simdMM4(c10, a1.y, b0); simdMM4(c11, a1.y, b1);
			simdMM4(c12, a1.z, b0); simdMM4(c13, a1.z, b1);
			simdMM4(c14, a1.w, b0); simdMM4(c15, a1.w, b1);
		}
		buf ^= 1;

		As[buf][tx][ty] = *(float4*)(A + A0);
		Bs[buf][ty][tx] = *(float4*)(B + B0);
		A += (SA << LB >> 1); B += (SB << LB >> 1);//K += STEP
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

	//when K%STEP!=0-------------------------------------------
	for (int ik = 0, RK = K & (STEP - 1); ik < RK; ik++)
	{
		int Aoffset = ik * SA, Boffset = ik * SB;
		float4 a0 = *(float4*)(A + Aoffset);
		float4 a1 = *(float4*)(A + Aoffset + 4);
		float4 b0 = *(float4*)(B + Boffset);
		float4 b1 = *(float4*)(B + Boffset + 4);

		simdMM4(c0, a0.x, b0); simdMM4(c1, a0.x, b1);
		simdMM4(c2, a0.y, b0); simdMM4(c3, a0.y, b1);
		simdMM4(c4, a0.z, b0); simdMM4(c5, a0.z, b1);
		simdMM4(c6, a0.w, b0); simdMM4(c7, a0.w, b1);
		simdMM4(c8, a1.x, b0); simdMM4(c9, a1.x, b1);
		simdMM4(c10, a1.y, b0); simdMM4(c11, a1.y, b1);
		simdMM4(c12, a1.z, b0); simdMM4(c13, a1.z, b1);
		simdMM4(c14, a1.w, b0); simdMM4(c15, a1.w, b1);
	}
	//when K%STEP!=0-------------------------------------------

	const int C0 = 0;
	const int C1 = C0 + SB, C2 = C1 + SB, C3 = C2 + SB;
	const int C4 = C3 + SB, C5 = C4 + SB, C6 = C5 + SB, C7 = C6 + SB;

	*(float4*)(C + C0) = c0;  *(float4*)(C + C0 + 4) = c1;
	*(float4*)(C + C1) = c2;  *(float4*)(C + C1 + 4) = c3;
	*(float4*)(C + C2) = c4;  *(float4*)(C + C2 + 4) = c5;
	*(float4*)(C + C3) = c6;  *(float4*)(C + C3 + 4) = c7;
	*(float4*)(C + C4) = c8;  *(float4*)(C + C4 + 4) = c9;
	*(float4*)(C + C5) = c10; *(float4*)(C + C5 + 4) = c11;
	*(float4*)(C + C6) = c12; *(float4*)(C + C6 + 4) = c13;
	*(float4*)(C + C7) = c14; *(float4*)(C + C7 + 4) = c15;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*4), K >= BLOCK_SIZE/2
//LB = 4: K >= 8
#ifndef MUTMUL_T1_KERNEL_8_4
#define MUTMUL_T1_KERNEL_8_4

//LB = 4: Size = 1, Time = 1.458 msec, Performace = 1472.9 GFlop/s
template<int LB, int STEP>
__global__ void kernel_t1_8_4(
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

	As[buf][tx][ty]     = *(float4*)(A + Axy);
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
			
			simdMM4( c0, a0.x, b0);
			simdMM4( c2, a0.y, b0);
			simdMM4( c4, a0.z, b0);
			simdMM4( c6, a0.w, b0);
			simdMM4( c8, a1.x, b0);
			simdMM4(c10, a1.y, b0);
			simdMM4(c12, a1.z, b0);
			simdMM4(c14, a1.w, b0);
		}
		buf ^= 1;

		As[buf][tx][ty]     = *(float4*)(A + Axy);
		Bs[buf][Bs_y][Bs_x] = *(float2*)(B + Byx);
		A += (SA << LB >> 1); B += (SB << LB >> 1);//A += SA * STEP
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a0 = As[buf][ik][ty], a1 = As[buf][ik + STEP][ty];
		float4 b0 = *(float4*)(&Bs[buf][ik][tx << 1]);

		simdMM4( c0, a0.x, b0);
		simdMM4( c2, a0.y, b0);
		simdMM4( c4, a0.z, b0);
		simdMM4( c6, a0.w, b0);
		simdMM4( c8, a1.x, b0);
		simdMM4(c10, a1.y, b0);
		simdMM4(c12, a1.z, b0);
		simdMM4(c14, a1.w, b0);
	}

	//when K % STEP != 0------------------------------------
	ty <<= 3; tx <<= 2;
	for (int ik = 0, RK = K & (STEP - 1); ik < RK; ik++)
	{
		float4 a0 = *(float4*)(&get(A, ik, ty    , SA));
		float4 a1 = *(float4*)(&get(A, ik, ty + 4, SA));
		float4 b0 = *(float4*)(&get(B, ik, tx, SB));
		
		simdMM4(c0, a0.x, b0);
		simdMM4(c2, a0.y, b0);
		simdMM4(c4, a0.z, b0);
		simdMM4(c6, a0.w, b0);
		simdMM4(c8, a1.x, b0);
		simdMM4(c10, a1.y, b0);
		simdMM4(c12, a1.z, b0);
		simdMM4(c14, a1.w, b0);
	}
	//when K % STEP != 0------------------------------------

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


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*8), K >= BLOCK_SIZE/2
//LB = 4: K >= 8
#ifndef MUTMUL_T1_KERNEL_4_8
#define MUTMUL_T1_KERNEL_4_8

//LB = 4: Size = 1, Time = 1.614 msec, Performace = 1330.54 GFlop/s
//LB = 4: Size = 0.99707, Time = 1.642 msec, Performace = 1304.01 GFlop/s
template<int LB, int STEP>
__global__ void kernel_t1_4_8(
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
	const int As_x = Ax,  As_y = ((ty << 1) + (tx & 1));
	const int Bx = (tx << 3) + ((ty >= STEP) << 2), By = ty - ((ty >= STEP) << LB >> 1);
	const int Axy = Ax * SA + Ay;//[Ax, Ay]
	const int Byx = By * SB + Bx;//[By, Bx]

	As[buf][As_x][As_y] = *(float2*)(A + Axy);
	Bs[buf][ty][tx]     = *(float4*)(B + Byx);
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
		Bs[buf][ty][tx]     = *(float4*)(B + Byx);
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

	//when K % STEP != 0--------------------------------------
	ty <<= 2; tx <<= 3;
	for (int ik = 0, RK = K & (STEP - 1); ik < RK; ik++)
	{
		float4 a0 = *(float4*)(&get(A, ik, ty, SA));
		float4 b0 = *(float4*)(&get(B, ik, tx    , SB));
		float4 b1 = *(float4*)(&get(B, ik, tx + 4, SB));

		simdMM4(c0, a0.x, b0); simdMM4(c1, a0.x, b1);
		simdMM4(c2, a0.y, b0); simdMM4(c3, a0.y, b1);
		simdMM4(c4, a0.z, b0); simdMM4(c5, a0.z, b1);
		simdMM4(c6, a0.w, b0); simdMM4(c7, a0.w, b1);
	}
	//when K % STEP != 0--------------------------------------

	*(float4*)(&get(C, ty    , tx, SB)) = c0; *(float4*)(&get(C, ty    , tx + 4, SB)) = c1;
	*(float4*)(&get(C, ty + 1, tx, SB)) = c2; *(float4*)(&get(C, ty + 1, tx + 4, SB)) = c3;
	*(float4*)(&get(C, ty + 2, tx, SB)) = c4; *(float4*)(&get(C, ty + 2, tx + 4, SB)) = c5;
	*(float4*)(&get(C, ty + 3, tx, SB)) = c6; *(float4*)(&get(C, ty + 3, tx + 4, SB)) = c7;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*2), K >= BLOCK_SIZE/2
//LB = 4: K >= 8
#ifndef MUTMUL_T1_KERNEL_8_2
#define MUTMUL_T1_KERNEL_8_2

//LB = 4: Size = 1, Time = 2.018 msec, Performace = 1064.16 GFlop/s
template<int LB, int STEP>
__global__ void kernel_t1_8_2(
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

			simdMM2( c0, a0.x, b0);
			simdMM2( c2, a0.y, b0);
			simdMM2( c4, a0.z, b0);
			simdMM2( c6, a0.w, b0);
			simdMM2( c8, a1.x, b0);
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
		
		simdMM2( c0, a0.x, b0);
		simdMM2( c2, a0.y, b0);
		simdMM2( c4, a0.z, b0);
		simdMM2( c6, a0.w, b0);
		simdMM2( c8, a1.x, b0);
		simdMM2(c10, a1.y, b0);
		simdMM2(c12, a1.z, b0);
		simdMM2(c14, a1.w, b0);
	}

	//when K % STEP != 0-------------------------------------
	ty <<= 3; tx <<= 1;
	for (int ik = 0, RK = K & (STEP - 1); ik < RK; ik++)
	{
		float4 a0 = *(float4*)(&get(A, ik, ty    , SA));
		float4 a1 = *(float4*)(&get(A, ik, ty + 4, SA));
		float2 b0 = *(float2*)(&get(B, ik, tx, SB));

		simdMM2( c0, a0.x, b0);
		simdMM2( c2, a0.y, b0);
		simdMM2( c4, a0.z, b0);
		simdMM2( c6, a0.w, b0);
		simdMM2( c8, a1.x, b0);
		simdMM2(c10, a1.y, b0);
		simdMM2(c12, a1.z, b0);
		simdMM2(c14, a1.w, b0);
	}
	//when K % STEP != 0-------------------------------------

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


//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*8), K >= BLOCK_SIZE/2
//LB = 4: K >= 8
#ifndef MUTMUL_T1_KERNEL_2_8
#define MUTMUL_T1_KERNEL_2_8

//LB = 4: Size = 1, Time = 2.672 msec, Performace = 803.699 GFlop/s
template<int LB, int STEP>
__global__ void kernel_t1_2_8(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float*  __restrict__ C,
	int SA, int SB, int K)
{
	bool buf = 0;
	__shared__ float  As[2][1 << LB >> 1][(2 << LB ) + 2];//followed k44T1
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
	Bs[buf][ty][tx]     = *(float4*)(B + Byx);
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
		Bs[buf][ty][tx]     = *(float4*)(B + Byx);
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

	//when K % STEP != 0-------------------------------------
	ty <<= 1; tx <<= 3;
	for (int ik = 0, RK = K & (STEP - 1); ik < RK; ik++)
	{
		float2 a0 = *(float2*)(&get(A, ik, ty, SA));
		float4 b0 = *(float4*)(&get(B, ik, tx, SB));
		float4 b1 = *(float4*)(&get(B, ik, tx + 4, SB));

		simdMM4(c0, a0.x, b0); simdMM4(c1, a0.x, b1);
		simdMM4(c2, a0.y, b0); simdMM4(c3, a0.y, b1);
	}
	//when K % STEP != 0-------------------------------------

	*(float4*)(&get(C, ty	 , tx, SB)) = c0; *(float4*)(&get(C, ty    , tx + 4, SB)) = c1;
	*(float4*)(&get(C, ty + 1, tx, SB)) = c2; *(float4*)(&get(C, ty + 1, tx + 4, SB)) = c3;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*1), K >= BLOCK_SIZE
//LB = 4: K >= 8
#ifndef MUTMUL_T1_KERNEL_8_1
#define MUTMUL_T1_KENERL_8_1

//LB = 4: Size = 1, Time = 3.382 msec, Performace = 634.974 GFlop/s
template<int LB, int STEP>
__global__ void kernel_t1_8_1(
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

	As[buf][tx][(ty << 1)    ] = *(float4*)(A + Axy    );
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
			float4 a0 = As[buf][ik][(ty << 1)    ];
			float4 a1 = As[buf][ik][(ty << 1) + 1];
			float b = Bs[buf][ik][tx];

			simdMM4(c0, b, a0);
			simdMM4(c1, b, a1);
		}
		buf ^= 1;

		As[buf][tx][(ty << 1)    ] = *(float4*)(A + Axy);
		As[buf][tx][(ty << 1) + 1] = *(float4*)(A + Axy + 4);
		Bs[buf][ty][tx] = B[Byx];
		A += (SA << LB); B += (SB << LB);//A += SA * STEP
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a0 = As[buf][ik][(ty << 1)    ];
		float4 a1 = As[buf][ik][(ty << 1) + 1];
		float b = Bs[buf][ik][tx];

		simdMM4(c0, b, a0);
		simdMM4(c1, b, a1);
	}

	//when K % STEP != 0 --------------------------------------
	ty <<= 3;
	for (int ik = 0, RK = K & (STEP - 1); ik < RK; ik++)
	{
		float4 a0 = *(float4*)(&get(A, ik, ty    , SA));
		float4 a1 = *(float4*)(&get(A, ik, ty + 4, SA));
		float b = get(B, ik, tx, SB);

		simdMM4(c0, b, a0);
		simdMM4(c1, b, a1);
	}
	//when K % STEP != 0 --------------------------------------

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


//(Y: BLOCK_SIZE  , X: BLOCK_SIZE*8), K >= BLOCK_SIZE
//LB = 4: K >= 8
#ifndef MUTMUL_T1_KERNEL_1_8
#define MUTMUL_T1_KERNEL_1_8

//LB = 4: Size = 0.99707, Time = 8.806 msec, Performace = 243.152 GFlop/s
template<int LB, int STEP>
__global__ void kernel_t1_1_8(
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
	Bs[buf][ty][(tx << 1)    ] = *(float4*)(B + Byx    );
	Bs[buf][ty][(tx << 1) + 1] = *(float4*)(B + Byx + 4);
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
			float a = As[buf][ik][ty];
			float4 b0 = Bs[buf][ik][(tx << 1)    ];
			float4 b1 = Bs[buf][ik][(tx << 1) + 1];
			simdMM4(c0, a, b0);  simdMM4(c1, a, b1);
		}
		buf ^= 1;

		As[buf][tx][ty] = A[Axy];
		Bs[buf][ty][(tx << 1)    ] = *(float4*)(B + Byx    );
		Bs[buf][ty][(tx << 1) + 1] = *(float4*)(B + Byx + 4);
		A += (SA << LB); B += (SB << LB);//A += SA * STEP
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float a = As[buf][ik][ty];
		float4 b0 = Bs[buf][ik][(tx << 1)    ];
		float4 b1 = Bs[buf][ik][(tx << 1) + 1];
		
		simdMM4(c0, a, b0); 
		simdMM4(c1, a, b1);
	}

	//when K % STEP != 0 --------------------------------------
	tx <<= 3;
	for (int ik = 0, RK = K & (STEP - 1); ik < RK; ik++) {
		float a = get(A, ik, ty, SA);
		float4 b0 = *(float4*)(&get(B, ik, tx    , SB));
		float4 b1 = *(float4*)(&get(B, ik, tx + 4, SB));
		simdMM4(c0, a, b0); simdMM4(c1, a, b1);
	}
	//when K % STEP != 0 --------------------------------------

	*(float4*)(&get(C, ty, tx, SB)) = c0; *(float4*)(&get(C, ty, tx + 4, SB)) = c1;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*4), K >= BLOCK_SIZE/2
//LB = 4: K >= 8
#ifndef MUTMUL_T1_KERNEL_4_4
#define MUTMUL_T1_KERNEL_4_4

//LB = 4: Size = 1, Time = 2.028 msec, Performace = 1058.92 GFlop/s
template<int LB, int STEP>
__global__ void kernel_t1_4_4(
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

	//when K % STEP != 0---------------------------------------------
	ty <<= 2; tx <<= 2;
	for (int ik = 0, RK = K & (STEP - 1); ik < RK; ik++)
	{
		float4 a = *(float4*)(&get(A, ik, ty, SA));
		float4 b = *(float4*)(&get(B, ik, tx, SB));

		simdMM4(c0, a.x, b);
		simdMM4(c1, a.y, b);
		simdMM4(c2, a.z, b);
		simdMM4(c3, a.w, b);
	}
	//when K % STEP != 0---------------------------------------------

	*(float4*)(&get(C, ty    , tx, SB)) = c0;
	*(float4*)(&get(C, ty + 1, tx, SB)) = c1;
	*(float4*)(&get(C, ty + 2, tx, SB)) = c2;
	*(float4*)(&get(C, ty + 3, tx, SB)) = c3;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*2), K >= BLOCK_SIZE/2
#ifndef MUTMUL_T1_KERNEL_4_2
#define MUTMUL_T1_KERNEL_4_2

//LB = 4: Size = 1, Time = 2.55 msec, Performace = 842.15 GFlop/s
//LB = 4: Size = 0.99707, Time = 2.548 msec, Performace = 840.342 GFlop/s
template<int LB, int STEP>
__global__ void kernel_t1_4_2(
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

	//when K % STEP != 0-----------------------------------------
	ty <<= 2; tx <<= 1;
	for (int ik = 0, RK = K & (STEP - 1); ik < RK; ik++)
	{
		float4 a = *(float4*)(&get(A, ik, ty, SA));
		float2 b = *(float2*)(&get(B, ik, tx, SB));

		simdMM2(c0, a.x, b);
		simdMM2(c1, a.y, b);
		simdMM2(c2, a.z, b);
		simdMM2(c3, a.w, b);
	}
	//when K % step != 0-----------------------------------------

	*(float2*)(&get(C, ty    , tx, SB)) = c0;
	*(float2*)(&get(C, ty + 1, tx, SB)) = c1;
	*(float2*)(&get(C, ty + 2, tx, SB)) = c2;
	*(float2*)(&get(C, ty + 3, tx, SB)) = c3;
}

#endif 


//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*4), K >= BLOCK_SIZE/2
#ifndef MUTMUL_T1_KERNEL_2_4
#define MUTMUL_T1_KERNEL_2_4

//LB = 4: Size = 1, Time = 3.136 msec, Performace = 684.784 GFlop/s
template<int LB, int STEP>
__global__ void kernel_t1_2_4(
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

	//when K % STEP != 0---------------------------------------
	ty <<= 1; tx <<= 2;
	for (int ik = 0, RK = K & (STEP - 1); ik < RK; ik++) {
		float2 a = *(float2*)(&get(A, ik, ty, SA));
		float4 b = *(float4*)(&get(B, ik, tx, SB));
		simdMM4(c0, a.x, b);
		simdMM4(c1, a.y, b);
	}
	//when K % STEP != 0---------------------------------------

	*(float4*)(&get(C, ty    , tx, SB)) = c0;
	*(float4*)(&get(C, ty + 1, tx, SB)) = c1;
}

#endif


//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*2), K >= BLOCK_SIZE
#ifndef MUTMUL_T1_KERNEL_2_2
#define MUTMUL_T1_KERNEL_2_2

//LB = 4: Size = 1, Time = 3.422 msec, Performace = 627.552 GFlop/s
template<int LB, int STEP>
__global__ void kernel_t1_2_2(
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

	//when K % STEP != 0 --------------------------------------
	ty <<= 1; tx <<= 1;
	for (int ik = 0, RK = K & (STEP - 1); ik < RK; ik++) {
		float2 a = *(float2*)(&get(A, ik, ty, SA));
		float2 b = *(float2*)(&get(B, ik, tx, SB));
		simdMM2(c0, a.x, b);
		simdMM2(c1, a.y, b);
	}
	//when K % STEP != 0 --------------------------------------

	*(float2*)(&get(C, ty    , tx, SB)) = c0;
	*(float2*)(&get(C, ty + 1, tx, SB)) = c1;
}

#endif


//(Y: N, X: M) X*Y<=1024
#ifndef MUTMUL_T1_KERNEL_NAIVE
#define MUTMUL_T1_KERNEL_NAIVE

//(correct)
__global__ void kernel_t1_naive(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float*  __restrict__ C,
	int SA, int SB, int K)
{
	int tx = threadIdx.x + blockIdx.x*blockDim.x;
	int ty = threadIdx.y + blockIdx.y*blockDim.y;
	
	float v = 0;
	for (int k = 0; k < K; k++)
		v += get(A, k, ty, SA) * get(B, k, tx, SB);
	get(C, ty, tx, SB) = v;
}

#endif

#endif