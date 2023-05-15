#pragma once

#ifndef MATMUL_KERNEL_H
#define MATMUL_KERNEL_H

//We have:
//(1) N % 4 == 0, N >= 4
//(2) M % 4 == 0, M >= 4
//(3) K % 4 == 0, K >= 4
#ifndef MATMUL_KERNEL_CALL
#define MATMUL_KERNEL_CALL

//LB = log2(BLOCK_SIZE)

//======[Common]========================================
#define	k88(LB, stream, A, B, C, N, M, K, SB) \
	kernel_8_8<LB, (1<<LB>>1)>\
		<<< dim3(M>>3>>LB, N>>3>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SB, K)

#define k84(LB, stream, A, B, C, N, M, K, SB) \
	kernel_8_4<LB, (1<<LB>>1)>\
		<<< dim3(M>>2>>LB, N>>3>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SB, K)

#define k48(LB, stream, A, B, C, N, M, K, SB) \
	kernel_4_8<LB, (1<<LB>>1)>\
		<<< dim3(M>>3>>LB, N>>2>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SB, K)

#define k82(LB, stream, A, B, C, N, M, K, SB) \
	kernel_8_2<LB, (1<<LB>>1)>\
		<<< dim3(M>>1>>LB, N>>3>>LB), dim3(1<<LB, 1<<LB), 0, stream >>> \
			(A, B, C, SB, K)

#define k28(LB, stream, A, B, C, N, M, K, SB) \
	kernel_2_8<LB, (1<<LB>>1)>\
		<<< dim3(M>>3>>LB, N>>1>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SB, K)
#define k44(LB, stream, A, B, C, N, M, K, SB) \
	kernel_4_4<LB, (1<<LB>>1)>\
		<<< dim3(M>>2>>LB, N>>2>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SB, K)

#define k42(LB, stream, A, B, C, N, M, K, SB) \
	kernel_4_2<LB, (1<<LB>>1)>\
		<<< dim3(M>>1>>LB, N>>2>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SB, K)

#define k24(LB, stream, A, B, C, N, M, K, SB) \
	kernel_2_4<LB, (1<<LB>>1)>\
		<<< dim3(M>>2>>LB, N>>1>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SB, K)

//======[Small]=========================================
#define k22(LB, stream, A, B, C, N, M, K, SB) \
	kernel_2_2<LB, (1<<LB)>\
		<<< dim3(M>>1>>LB, N>>1>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SB, K)

//-----------------------------------------------------
#define k81(LB, stream, A, B, C, N, M, K, SB) \
	kernel_8_1<LB, (1<<LB)>\
		<<< dim3(M>>LB, N>>3>>LB), dim3(1<<LB, 1<<LB), 0, stream >>> \
			(A, B, C, SB, K)

#define k41(LB, stream, A, B, C, N, M, K, SB) \
	kernel_4_1<LB, (1<<LB)>\
		<<< dim3(M>>LB, N>>2>>LB), dim3(1<<LB, 1<<LB), 0, stream >>> \
			(A, B, C, SB, K)

#define k21(LB, stream, A, B, C, N, M, K, SB) \
	kernel_2_1<LB, (1<<LB)>\
		<<< dim3(M>>LB, N>>1>>LB), dim3(1<<LB, 1<<LB), 0, stream >>> \
			(A, B, C, SB, K)

//-----------------------------------------------------
#define k18(LB, stream, A, B, C, N, M, K, SB) \
	kernel_1_8<LB, (1<<LB)>\
		<<< dim3(N>>LB, M>>3>>LB), dim3(1<<LB, 1<<LB), 0, stream >>> \
			(A, B, C, SB, K)

#define k14(LB, stream, A, B, C, N, M, K, SB) \
	kernel_1_4<LB, (1<<LB)>\
		<<< dim3(N>>LB, M>>2>>LB), dim3(1<<LB, 1<<LB), 0, stream >>> \
			(A, B, C, SB, K)

#define k12(LB, stream, A, B, C, N, M, K, SB) \
	kernel_1_2<LB, (1<<LB)>\
		<<< dim3(N>>LB, M>>1>>LB), dim3(1<<LB, 1<<LB), 0, stream >>> \
			(A, B, C, SB, K)

//-----------------------------------------------------
#define knaive(GRID_SIZE, stream, A, B, C, N, M, K, SB) \
	kernel_naive\
		<<< dim3(GRID_SIZE, GRID_SIZE), dim3(M/GRID_SIZE, N/GRID_SIZE), 0, stream>>>\
			(A, B, C, SB, K)

#endif


//======[Common]========================================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), K >= BLOCK_SIZE/2
//LB = 4: K >= 8
#ifndef MATMUL_KERNEL_8_8
#define MATMUL_KERNEL_8_8

//for 1024*1024*1024:
//LB = 4: Size = 1, Time = 1.308 msec, Performace = 1641.81 GFlop/s
//LB = 3: Size = 1, Time = 1.507 msec, Performace = 1425.01 GFlop/ss
//for 1024*1024*1023:
//LB = 4: Size = 0.999023, Time = 1.314 msec, Performace = 1632.71 GFlop/s
//LB = 3: Size = 0.999023, Time = 1.523 msec, Performace = 1408.66 GFlop/s
template<int LB, int STEP>
__global__ void kernel_8_8(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float*  __restrict__ C,
	int SB, int K)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 As[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Bs[2][1 << LB][(1 << LB) + 1];

	const int Y = ((blockIdx.y << LB) + ty) << 3; A += Y * K;
	const int X = ((blockIdx.x << LB) + tx) << 3; B += X;
	const int C0 = Y * SB + X;//C[Y, X]
	const int Ax = tx - ((tx >= STEP) << LB >> 1);
	const int By = ty - ((ty >= STEP) << LB >> 1);
	const int B0 = ((ty >= STEP) << 2) + By * SB;
	const int A0 = ((tx >= STEP) << 2) * K + Ax;
	const int A1 = A0 + K, A2 = A1 + K, A3 = A2 + K;

	float4 av;//transpose A
	av.x = A[A0];
	av.y = A[A1];
	av.z = A[A2];
	av.w = A[A3];
	As[buf][tx][ty] = av;

	Bs[buf][ty][tx] = *(float4*)(B + B0);
	A += STEP; B += (SB << LB >> 1);//K += STEP
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

		float4 av;//transpose A
		av.x = A[A0];
		av.y = A[A1];
		av.z = A[A2];
		av.w = A[A3];
		As[buf][tx][ty] = av;

		Bs[buf][ty][tx] = *(float4*)(B + B0);
		A += STEP; B += (SB << LB >> 1);//K += STEP
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

	//when K % STEP != 0-----------------------------------
	for (int ik = 0, RK = K & (STEP - 1); ik < RK; ik++)
	{
		float4 b0 = *(float4*)(&get(B, ik, 0, SB));
		float4 b1 = *(float4*)(&get(B, ik, 4, SB));

		float4 a0, a1;
		a0.x = get(A, 0, ik, K);
		a0.y = get(A, 1, ik, K);
		a0.z = get(A, 2, ik, K);
		a0.w = get(A, 3, ik, K);
		a1.x = get(A, 4, ik, K);
		a1.y = get(A, 5, ik, K);
		a1.z = get(A, 6, ik, K);
		a1.w = get(A, 7, ik, K);
		simdMM4(c0, a0.x, b0);  simdMM4(c1, a0.x, b1);
		simdMM4(c2, a0.y, b0);  simdMM4(c3, a0.y, b1);
		simdMM4(c4, a0.z, b0);  simdMM4(c5, a0.z, b1);
		simdMM4(c6, a0.w, b0);  simdMM4(c7, a0.w, b1);
		simdMM4(c8, a1.x, b0);  simdMM4(c9, a1.x, b1);
		simdMM4(c10, a1.y, b0); simdMM4(c11, a1.y, b1);
		simdMM4(c12, a1.z, b0); simdMM4(c13, a1.z, b1);
		simdMM4(c14, a1.w, b0); simdMM4(c15, a1.w, b1);
	}
	//when K % STEP != 0-----------------------------------

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
#ifndef MATMUL_KERNEL_8_4
#define MATMUL_KERNEL_8_4

//for [1024*1024*1024]: 
//LB = 4: Size = 1, Time = 1.57  msec, Performace = 1367.82 GFlop/s
//LB = 3: Size = 1, Time = 1.837 msec, Performace = 1169.02 GFlop/s
template<int LB, int STEP>
__global__ void kernel_8_4(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float*  __restrict__ C,
	int SB, int K)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 As[2][1 << LB][(1 << LB) + 1];//followed k88
	__shared__ float2 Bs[2][1 << LB >> 1][(2 << LB) + 2];//followed k44

	int Y = ((blockIdx.y << LB) + ty) << 3; A += Y * K;//A[Y, 0]
	int X = ((blockIdx.x << LB) + tx) << 2; B = B + X;//B[0, X]
	const int C0 = Y * SB + X;//C[Y, X]

	const int Ax = tx - ((tx >= STEP) << LB >> 1);
	const int A0 = ((tx >= STEP) << 2) * K + Ax;//[Ay, Ax]
	const int A1 = A0 + K, A2 = A1 + K, A3 = A2 + K;

	const int By = ty >> 1, Bx = ((ty & 1) << 1);
	const int B0 = By * SB + Bx;//[By, Bx]
	const int Bs_y = By, Bs_x = ((tx << 1) + (ty & 1));
	
	float4 av;//transpose A
	av.x = A[A0];
	av.y = A[A1];
	av.z = A[A2];
	av.w = A[A3];
	As[buf][tx][ty] = av;

	Bs[buf][Bs_y][Bs_x] = *(float2*)(B + B0);
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

		float4 av;//transpose A
		av.x = A[A0];
		av.y = A[A1];
		av.z = A[A2];
		av.w = A[A3];
		As[buf][tx][ty] = av;

		Bs[buf][Bs_y][Bs_x] = *(float2*)(B + B0);
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

	//when K % STEP != 0-----------------------------------
	for (int ik = 0, RK = K & (STEP - 1); ik < RK; ik++)
	{
		float4 a0, a1;
		a0.x = get(A, 0, ik, K);
		a0.y = get(A, 1, ik, K);
		a0.z = get(A, 2, ik, K);
		a0.w = get(A, 3, ik, K);
		a1.x = get(A, 4, ik, K);
		a1.y = get(A, 5, ik, K);
		a1.z = get(A, 6, ik, K);
		a1.w = get(A, 7, ik, K);

		float4 b0 = *(float4*)(B + ik * SB);

		simdMM4(c0, a0.x, b0);
		simdMM4(c2, a0.y, b0);
		simdMM4(c4, a0.z, b0);
		simdMM4(c6, a0.w, b0);
		simdMM4(c8, a1.x, b0);
		simdMM4(c10, a1.y, b0);
		simdMM4(c12, a1.z, b0);
		simdMM4(c14, a1.w, b0);
	}
	//when K % STEP != 0-----------------------------------

	const int C1 = C0 + SB, C2 = C1 + SB, C3 = C2 + SB;
	const int C4 = C3 + SB, C5 = C4 + SB, C6 = C5 + SB, C7 = C6 + SB;

	*(float4*)(C + C0) = c0;
	*(float4*)(C + C1) = c2;
	*(float4*)(C + C2) = c4;
	*(float4*)(C + C3) = c6;
	*(float4*)(C + C4) = c8;
	*(float4*)(C + C5) = c10;
	*(float4*)(C + C6) = c12;
	*(float4*)(C + C7) = c14;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*8), K >= BLOCK_SIZE/2
//LB = 4: K >= 8
#ifndef MATMUL_KERNEL_4_8
#define MATMUL_KERNEL_4_8

//for [1024*1024*1024]: 
//LB = 4: Size = 1, Time = 1.638 msec, Performace = 1311.04 GFlop/s
//LB = 3: Size = 1, Time = 1.877 msec, Performace = 1144.1 GFlop/s
template<int LB, int STEP>
__global__ void kernel_4_8(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float*  __restrict__ C,
	int SB, int K)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float2 As[2][1 << LB >> 1][(2 << LB) + 2];//followed k44
	__shared__ float4 Bs[2][1 << LB][(1 << LB) + 1];//followed k88

	int Y = ((blockIdx.y << LB) + ty) << 2; A = A + Y * K;//A[Y, 0]
	int X = ((blockIdx.x << LB) + tx) << 3; B = B + X;//B[0, X]
	const int C0 = Y * SB + X;//C[Y, X]

	const int Ax = (tx >> 1), Ay = ((tx & 1) << 1);
	const int As_x = Ax, As_y = ((ty << 1) + (tx & 1));
	const int A0 = Ay * K + Ax, A1 = A0 + K;

	const int Bx = ((ty >= STEP) << 2);
	const int By = ty - ((ty >= STEP) << LB >> 1);
	const int B0 = By * SB + Bx;
	
	float2 av;//transpose A
	av.x = A[A0];
	av.y = A[A1];
	As[buf][As_x][As_y] = av;

	Bs[buf][ty][tx] = *(float4*)(B + B0);
	A += STEP; B += (SB << LB >> 1);//K += STEP
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
			float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP][tx];
			float4 a0 = *(float4*)(&As[buf][ik][ty << 1]);

			simdMM4(c0, a0.x, b0); simdMM4(c1, a0.x, b1);
			simdMM4(c2, a0.y, b0); simdMM4(c3, a0.y, b1);
			simdMM4(c4, a0.z, b0); simdMM4(c5, a0.z, b1);
			simdMM4(c6, a0.w, b0); simdMM4(c7, a0.w, b1);
		}
		buf ^= 1;

		float2 av;//transpose A
		av.x = A[A0];
		av.y = A[A1];
		As[buf][As_x][As_y] = av;

		Bs[buf][ty][tx] = *(float4*)(B + B0);
		A += STEP; B += (SB << LB >> 1);//K += STEP
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP][tx];
		float4 a0 = *(float4*)(&As[buf][ik][ty << 1]);

		simdMM4(c0, a0.x, b0); simdMM4(c1, a0.x, b1);
		simdMM4(c2, a0.y, b0); simdMM4(c3, a0.y, b1);
		simdMM4(c4, a0.z, b0); simdMM4(c5, a0.z, b1);
		simdMM4(c6, a0.w, b0); simdMM4(c7, a0.w, b1);
	}

	//when K % STEP != 0-----------------------------------
	for (int ik = 0, RK = K & (STEP - 1); ik < RK; ik++)
	{
		float4 a0;
		a0.x = get(A, 0, ik, K);
		a0.y = get(A, 1, ik, K);
		a0.z = get(A, 2, ik, K);
		a0.w = get(A, 3, ik, K);

		float4 b0 = *(float4*)(&get(B, ik, 0, SB));
		float4 b1 = *(float4*)(&get(B, ik, 4, SB));

		simdMM4(c0, a0.x, b0); simdMM4(c1, a0.x, b1);
		simdMM4(c2, a0.y, b0); simdMM4(c3, a0.y, b1);
		simdMM4(c4, a0.z, b0); simdMM4(c5, a0.z, b1);
		simdMM4(c6, a0.w, b0); simdMM4(c7, a0.w, b1);
	}
	//when K % STEP != 0-----------------------------------

	const int C1 = C0 + SB;
	const int C2 = C1 + SB;
	const int C3 = C2 + SB;

	*(float4*)(C + C0) = c0;  *(float4*)(C + C0 + 4) = c1;
	*(float4*)(C + C1) = c2;  *(float4*)(C + C1 + 4) = c3;
	*(float4*)(C + C2) = c4;  *(float4*)(C + C2 + 4) = c5;
	*(float4*)(C + C3) = c6;  *(float4*)(C + C3 + 4) = c7;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*4), K>=BLOCK_SIZE/2
//LB = 4: K >= 8
#ifndef MATMUL_KERNEL_4_4
#define MATMUL_KERNEL_4_4

//for [1024*1024*1024]: 
//LB = 4: Size = 1, Time = 2.024 msec, Performace = 1061.01  sGFlop/s
//LB = 3: Size = 1, Time = 2.291 msec, Performace =  937.356 GFlop/s
template<int LB, int STEP>
__global__ void kernel_4_4(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float*  __restrict__ C,
	int SB, int K)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float2 As[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2 Bs[2][1 << LB >> 1][(2 << LB) + 2];

	int Y = (blockIdx.y << 2 << LB); A += Y * K;//A[Y, 0]
	int X = (blockIdx.x << 2 << LB); B += X;//B[0, X]
	C = &get(C, Y, X, SB);//C[Y, X]

	const int Ax = (tx >> 1), Ay = (ty << 2) + ((tx & 1) << 1);
	const int As_x = Ax, As_y = ((ty << 1) + (tx & 1));
	const int A0 = Ay * K + Ax;//[Ay, Ax]
	const int A1 = A0 + K;

	const int By = (ty >> 1), Bx = (tx << 2) + ((ty & 1) << 1);
	const int Bs_y = By, Bs_x = ((tx << 1) + (ty & 1));
	const int B0 = By * SB + Bx;//[By, Bx]

	float2 av;//transpose A;
	av.x = A[A0];
	av.y = A[A1];
	As[buf][As_x][As_y] = av;

	Bs[buf][Bs_y][Bs_x] = *(float2*)(B + B0);
	A += STEP; B += (SB << LB >> 1);//K += STEP
	__syncthreads();

	//compute area------------------------------------------
	float4 c0 = make_float4(0, 0, 0, 0);
	float4 c1 = make_float4(0, 0, 0, 0);
	float4 c2 = make_float4(0, 0, 0, 0);
	float4 c3 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (K << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 b = *(float4*)(&Bs[buf][ik][tx << 1]);
			float4 a = *(float4*)(&As[buf][ik][ty << 1]);

			simdMM4(c0, a.x, b);
			simdMM4(c1, a.y, b);
			simdMM4(c2, a.z, b);
			simdMM4(c3, a.w, b);
		}
		buf ^= 1;

		float2 av;//transpose A;
		av.x = A[A0];
		av.y = A[A1];
		As[buf][As_x][As_y] = av;

		Bs[buf][Bs_y][Bs_x] = *(float2*)(B + B0);
		A += STEP; B += (SB << LB >> 1);//K += STEP
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 b = *(float4*)(&Bs[buf][ik][tx << 1]);
		float4 a = *(float4*)(&As[buf][ik][ty << 1]);

		simdMM4(c0, a.x, b);
		simdMM4(c1, a.y, b);
		simdMM4(c2, a.z, b);
		simdMM4(c3, a.w, b);
	}

	//when K % STEP != 0--------------------------------------
	ty <<= 2; tx <<= 2;
	for (int ik = 0, RK = K & (STEP - 1); ik < RK; ik++)
	{
		float4 a;
		a.x = get(A, ty, ik, K);
		a.y = get(A, ty + 1, ik, K);
		a.z = get(A, ty + 2, ik, K);
		a.w = get(A, ty + 3, ik, K);

		float4 b = *(float4*)(&get(B, ik, tx, SB));

		simdMM4(c0, a.x, b);
		simdMM4(c1, a.y, b);
		simdMM4(c2, a.z, b);
		simdMM4(c3, a.w, b);
	}
	//when K % STEP != 0--------------------------------------

	const int C0 = ty * SB + tx;
	const int C1 = C0 + SB;
	const int C2 = C1 + SB;
	const int C3 = C2 + SB;

	*(float4*)(C + C0) = c0;
	*(float4*)(C + C1) = c1;
	*(float4*)(C + C2) = c2;
	*(float4*)(C + C3) = c3;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*2), K >= BLOCK_SIZE/2
//LB = 4: K >= 8
#ifndef MATMUL_KERNEL_8_2
#define MATMUL_KERNEL_8_2

//(correct)
//LB = 4: Size = 1, Time = 2.076 msec, Performace = 1034.43 GFlop/s
//for 1024*1024*1024:
//	(1) 128*64, LB=4, Performance= 808.09 GFlop/s, Time= 2.657 msec
//	(2)  64*16, LB=3, Performance= 593.14 GFlop/s, Time= 3.621 msec
//	(3)  32*8 , LB=2, Performance= 310.97 GFlop/s, Time= 6.906 msec
//for 1024*1024*1023:
//	(1) 128*64, LB=4, Performance= 806.57 GFlop/s, Time= 2.660 msec
//	(2)  64*16, LB=3, Performance= 592.81 GFlop/s, Time= 3.619 msec
//	(3)  32*8 , LB=2, Performance= 299.94 GFlop/s, Time= 7.153 msec
//for 128*128*127:
//	(1)  16*4 , LB=1, Performance= 43.72 GFlop/s, Time= 0.095 msec

template<int LB, int STEP>
__global__ void kernel_8_2(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float*  __restrict__ C,
	int SB, int K)
{
	bool buf = 0;
	__shared__ float4 As[2][1 << LB][(1 << LB) + 1];//followed k88
	__shared__ float  Bs[2][1 << LB >> 1][(2 << LB) + 2];//followed k44

	int Y = (blockIdx.y << 3 << LB);//Y = blockIdx.y * 8 * BLOCK_SIZE
	int X = (blockIdx.x << 1 << LB);//X = blockIdx.x * 2 * BLOCK_SIZE
	A = A + Y * K;//A[Y, 0]
	B = B + X;//B[0, X]
	C = &get(C, Y, X, SB);//C[Y, X]

	int tx = threadIdx.x, ty = threadIdx.y;
	const int Ay = (ty << 3) + ((tx >= STEP) << 2), Ax = tx - ((tx >= STEP) << LB >> 1);
	const int By = (ty >> 1), Bx = (tx << 1) + (ty & 1);
	const int Bs_y = By, Bs_x = ((tx << 1) + (ty & 1));
	const int Byx = By * SB + Bx;//[By, Bx]

	As[buf][tx][ty].x = get(A, Ay    , Ax, K);//transpose A
	As[buf][tx][ty].y = get(A, Ay + 1, Ax, K);
	As[buf][tx][ty].z = get(A, Ay + 2, Ax, K);
	As[buf][tx][ty].w = get(A, Ay + 3, Ax, K);
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

		As[buf][tx][ty].x = get(A, Ay    , Ax, K);//transpose A
		As[buf][tx][ty].y = get(A, Ay + 1, Ax, K);
		As[buf][tx][ty].z = get(A, Ay + 2, Ax, K);
		As[buf][tx][ty].w = get(A, Ay + 3, Ax, K);
		Bs[buf][Bs_y][Bs_x] = B[Byx];
		A += STEP; B += (SB << LB >> 1);//B += SB*STEP
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

	//when K % STEP != 0-----------------------------------
	ty <<= 3; tx <<= 1;
	for (int ik = 0, RK = K & (STEP - 1); ik < RK; ik++)
	{
		float4 a0, a1;
		a0.x = get(A, ty    , ik, K);
		a0.y = get(A, ty + 1, ik, K);
		a0.z = get(A, ty + 2, ik, K);
		a0.w = get(A, ty + 3, ik, K);
		a1.x = get(A, ty + 4, ik, K);
		a1.y = get(A, ty + 5, ik, K);
		a1.z = get(A, ty + 6, ik, K);
		a1.w = get(A, ty + 7, ik, K);

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
	//when K % STEP != 0-----------------------------------

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
#ifndef MATMUL_KERNEL_2_8
#define MATMUL_KERNEL_2_8

//(corret)
//LB = 4: Size = 1, Time = 2.748 msec, Performace = 781.471 GFlop/s
//for 1024*1024*1024:
//	(1) 128*64, LB=4, Performance= 784.58 GFlop/s, Time= 2.734 msec
//	(2)  64*16, LB=3, Performance= 775.28 GFlop/s, Time= 2.770 msec
//	(3)  32*8 , LB=2, Performance= 346.90 GFlop/s, Time= 6.190 msec
//for 1024*1024*1023:
//	(1) 128*64, LB=4, Performance= 777.18 GFlop/s, Time= 2.763 msec
//	(2)  64*16, LB=3, Performance= 778.11 GFlop/s, Time= 2.757 msec
//	(3)  32*8 , LB=2, Performance= 346.59 GFlop/s, Time= 6.190 msec
//for 128*128*127:
//	(4)  16*4 , LB=1, Performance= 63.96 GFlop/s, Time= 0.065 msec

template<int LB, int STEP>
__global__ void kernel_2_8(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float*  __restrict__ C,
	int SB, int K)
{
	bool buf = 0;
	__shared__ float  As[2][1 << LB >> 1][(2 << LB) + 2];//followed k44
	__shared__ float4 Bs[2][1 << LB][(1 << LB) + 1];//followed k88

	int Y = (blockIdx.y << 1 << LB);//Y = blockIdx.y * 2 * BLOCK_SIZE
	int X = (blockIdx.x << 3 << LB);//X = blockIdx.x * 8 * BLOCK_SIZE
	A = A + Y * K;//A[Y, 0]
	B = B + X;//B[0, X]
	C = &get(C, Y, X, SB);//C[Y, X]

	int tx = threadIdx.x, ty = threadIdx.y;
	const int Ax = (tx >> 1), Ay = (ty << 1) + (tx & 1);
	const int As_x = Ax, As_y = ((ty << 1) + (tx & 1));
	const int Bx = (tx << 3) + ((ty >= STEP) << 2), By = ty - ((ty >= STEP) << LB >> 1);
	const int Ayx = Ay * K  + Ax;//[Ay, Ax]
	const int Byx = By * SB + Bx;//[By, Bx]

	As[buf][As_x][As_y] = A[Ayx]; //transpose A
	Bs[buf][ty][tx] = *(float4*)(B + Byx);
	A += STEP; B += (SB << LB >> 1);//B += SB*STEP
	__syncthreads();

	//compute area-------------------------------------------------------
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

	//when K % STEP != 0----------------------------------------
	ty <<= 1; tx <<= 3;
	for (int ik = 0, RK = K & (STEP - 1); ik < RK; ik++)
	{
		float2 a0;
		a0.x = get(A, ty    , ik, K);
		a0.y = get(A, ty + 1, ik, K);

		float4 b0 = *(float4*)(&get(B, ik, tx    , SB));
		float4 b1 = *(float4*)(&get(B, ik, tx + 4, SB));

		simdMM4(c0, a0.x, b0); simdMM4(c1, a0.x, b1);
		simdMM4(c2, a0.y, b0); simdMM4(c3, a0.y, b1);
	}
	//when K % STEP != 0----------------------------------------

	*(float4*)(&get(C, ty    , tx, SB)) = c0; *(float4*)(&get(C, ty    , tx + 4, SB)) = c1;
	*(float4*)(&get(C, ty + 1, tx, SB)) = c2; *(float4*)(&get(C, ty + 1, tx + 4, SB)) = c3;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*2), K >= BLOCK_SIZE/2
//LB = 4: K >= 8
#ifndef MATMUL_KERNEL_4_2
#define MATMUL_KERNEL_4_2

//(correct)
//LB = 4: Size = 1, Time = 2.582 msec, Performace = 831.713 GFlop/s
//for 1024*1024*1024
//	(1) 128*64, LB=5, Performance= 749.83 GFlop/s, Time= 2.864 msec
//	(2)  64*32, LB=4, Performance= 645.84 GFlop/s, Time= 3.325 msec
//	(3)	 32*16, LB=3, Performance= 583.39 GFlop/s, Time= 3.681 msec 
//	(4)  16*8 , LB=2, Performance= 230.30 GFlop/s, Time= 9.325 msec
//for 1024*1024*1023
//	(1) 128*64, LB=5, testkernel_2_4<BLOCK_SIZE, LB>(dA, dB, dC, N, M, K);
//	(2)  64*32, LB=4, Performance= 639.59 GFlop/s, Time= 3.354 msec
//	(3)	 32*16, LB=3, Performance= 555.19 GFlop/s, Time= 3.864 msec
//	(4)  16*8 , LB=2, Performance= 229.01 GFlop/s, Time= 9.368 msec
//for 128*128*128:
//	(1)   8*4 , LB=1, Performance= 41.63 GFlop/s, Time= 0.100 msec

template<int LB, int STEP>
__global__ void kernel_4_2(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float*  __restrict__ C,
	int SB, int K)
{
	bool buf = 0;
	__shared__ float2 As[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float  Bs[2][1 << LB >> 1][(2 << LB) + 2];
	
	int Y = (blockIdx.y << 2 << LB);//Y = blockIdx.y * 4 * BLOCK_SIZE
	int X = (blockIdx.x << 1 << LB);//X = blockIdx.x * 2 * BLOCK_SIZE
	A = A + Y * K;//A[Y, 0]
	B = B + X;//B[0, X]
	C = &get(C, Y, X, SB);//C[Y, X]
	
	int tx = threadIdx.x, ty = threadIdx.y;
	const int Ax = (tx >> 1), Ay = (ty << 2) + ((tx & 1) << 1);
	const int By = (ty >> 1), Bx = (tx << 1) + (ty & 1);
	const int As_x = Ax, As_y = ((ty << 1) + (tx & 1));
	const int Bs_y = By, Bs_x = ((tx << 1) + (ty & 1));
	const int Ayx = Ay * K + Ax;//[Ay, Ax]
	const int Byx = By * SB + Bx;//[By, Bx]

	As[buf][As_x][As_y].x = A[Ayx    ];//transpose A
	As[buf][As_x][As_y].y = A[Ayx + K];//[Ay + 1, Ax]
	Bs[buf][Bs_y][Bs_x] = B[Byx];
	A += STEP; B += (SB << LB >> 1);//B += SB * STEP
	__syncthreads();

	//compute area------------------------------------------
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

		As[buf][As_x][As_y].x = A[Ayx    ];//transpose A
		As[buf][As_x][As_y].y = A[Ayx + K];//[Ay + 1, Ax]
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

	//when K % STEP != 0--------------------------------------
	ty <<= 2; tx <<= 1;
	for (int ik = 0, RK = K & (STEP - 1); ik < RK; ik++)
	{
		float4 a;
		a.x = get(A, ty    , ik, K);
		a.y = get(A, ty + 1, ik, K);
		a.z = get(A, ty + 2, ik, K);
		a.w = get(A, ty + 3, ik, K);

		float2 b = *(float2*)(&get(B, ik, tx, SB));

		simdMM2(c0, a.x, b);
		simdMM2(c1, a.y, b);
		simdMM2(c2, a.z, b);
		simdMM2(c3, a.w, b);
	}
	//when K % STEP != 0--------------------------------------

	*(float2*)(&get(C, ty    , tx, SB)) = c0;
	*(float2*)(&get(C, ty + 1, tx, SB)) = c1;
	*(float2*)(&get(C, ty + 2, tx, SB)) = c2;
	*(float2*)(&get(C, ty + 3, tx, SB)) = c3;
}

#endif 


//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*4), K >= BLOCK_SIZE/2
//LB = 4: K >= 8
#ifndef MATMUL_KERNEL_2_4
#define MATMUL_KERNEL_2_4

//(correct)
//LB = 4: Size = 1, Time = 3.1 msec, Performace = 692.737 GFlop/s
//for 1024*1024*1024
//	(1) 64*128, LB=5, Performance= 627.96 GFlop/s, Time= 3.420 msec
//	(2) 32*64 , LB=4, Performance= 638.68 GFlop/s, Time= 3.362 msec
//	(3) 16*32 , LB=3, Performance= 643.91 GFlop/s, Time= 3.335 msec
//	(4)  8*16 , LB=2, Performance= 261.45 GFlop/s, Time= 8.214 msec
//for 1024*1024*1023
//	(1) 64*128, LB=5, Performance= 626.61 GFlop/s, Time= 3.424 msec
//	(2) 32*64 , LB=4, Performance= 644.30 GFlop/s, Time= 3.330 msec
//	(3) 16*32 , LB=3, Performance= 643.82 GFlop/s, Time= 3.332 msec
//	(4)  8*16 , LB=2, Performance= 263.13 GFlop/s, Time= 8.153 msec
//for 128*128*127:
//	(1)  4*8  , LB=1, Performance=  41.34 GFlop/s, Time= 0.101 msec

template<int LB, int STEP>
__global__ void kernel_2_4(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float*  __restrict__ C,
	int SB, int K)
{
	bool buf = 0;
	__shared__ float  As[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2 Bs[2][1 << LB >> 1][(2 << LB) + 2];

	int Y = (blockIdx.y << 1 << LB);//Y = blockIdx.y * 2 * BLOCK_SIZE
	int X = (blockIdx.x << 2 << LB);//X = blockIdx.x * 4 * BLOCK_SIZE
	A = A + Y * K;//A[Y, 0]
	B = B + X;//B[0, X]
	C = &get(C, Y, X, SB);//C[Y, X]

	int tx = threadIdx.x, ty = threadIdx.y;
	const int Ax = (tx >> 1), Ay = (ty << 1) + (tx & 1);
	const int By = (ty >> 1), Bx = (tx << 2) + ((ty & 1) << 1);
	const int As_x = Ax, As_y = ((ty << 1) + (tx & 1));
	const int Bs_y = By, Bs_x = ((tx << 1) + (ty & 1)); 
	const int Ayx = Ay * K + Ax;//[Ay, Ax]
	const int Byx = By * SB + Bx;//[By, Bx]

	As[buf][As_x][As_y] = A[Ayx]; //transpose A
	Bs[buf][Bs_y][Bs_x] = *(float2*)(B + Byx);
	A += STEP; B += (SB << LB >> 1);//B += SB * STEP
	__syncthreads();

	//compute area--------------------------------------------
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

	//when K % STEP != 0--------------------------------------
	ty <<= 1; tx <<= 2;
	for (int ik = 0, RK = K & (STEP - 1); ik < RK; ik++)
	{
		float2 a;
		a.x = get(A, ty    , ik, K);
		a.y = get(A, ty + 1, ik, K);

		float4 b = *(float4*)(&get(B, ik, tx, SB));
		
		simdMM4(c0, a.x, b);
		simdMM4(c1, a.y, b);
	}
	//when K % STEP != 0--------------------------------------

	*(float4*)(&get(C, ty    , tx, SB)) = c0;
	*(float4*)(&get(C, ty + 1, tx, SB)) = c1;
}

#endif


//======[Small]=========================================
//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*2), K >= BLOCK_SIZE
//LB = 4: K >= 16
//LB = 3: K >=  8
#ifndef MATMUL_KERNEL_2_2
#define MATMUL_KERNEL_2_2

//for [1024*1024*1024]:
//LB = 4: Size = 1, Time = 3.638 msec, Performace = 590.292 GFlop/s
//LB = 3: Size = 1, Time = 4.71  msec, Performace = 455.941 GFlop/s
template<int LB, int STEP>
__global__ void kernel_2_2(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float*  __restrict__ C,
	int SB, int K)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float2 As[2][1 << LB][(1 << LB) + 1];
	__shared__ float2 Bs[2][1 << LB][(1 << LB) + 1];

	int Y = (blockIdx.y << 1 << LB); A += Y * K;//A[Y, 0]
	int X = (blockIdx.x << 1 << LB); B += X;//B[0, X]
	C = &get(C, Y, X, SB);//C[Y, X]

	const int A0 = (ty << 1) * K + tx, A1 = A0 + K;
	const int B0 = ty * SB + (tx << 1);

	As[buf][tx][ty] = float2{ A[A0], A[A1] };//transpose A
	Bs[buf][ty][tx] = *(float2*)(B + B0);
	A += STEP; B += (SB << LB);//B += SB * STEP
	__syncthreads();

	//compute area------------------------------------------
	float2 c0 = make_float2(0, 0);
	float2 c1 = make_float2(0, 0);
	for (int ok = 1, OK = (K >> LB); ok < OK; ok++)
	{
#pragma once
		for (int ik = 0; ik < STEP; ik++) {
			float2 b = Bs[buf][ik][tx];
			float2 a = As[buf][ik][ty];
			simdMM2(c0, a.x, b);
			simdMM2(c1, a.y, b);
		}
		buf ^= 1;

		As[buf][tx][ty] = float2{ A[A0], A[A1] };//transpose A
		Bs[buf][ty][tx] = *(float2*)(B + B0);
		A += STEP; B += (SB << LB);//B += SB * STEP
		__syncthreads();
	}
#pragma once
	for (int ik = 0; ik < STEP; ik++) {
		float2 b = Bs[buf][ik][tx];
		float2 a = As[buf][ik][ty];
		simdMM2(c0, a.x, b);
		simdMM2(c1, a.y, b);
	}

	//when K % STEP != 0--------------------------------------
	ty <<= 1; tx <<= 1;
	for (int ik = 0, RK = K & (STEP - 1); ik < RK; ik++)
	{
		float2 a;//transposed A
		a.x = get(A, ty, ik, K);
		a.y = get(A, ty + 1, ik, K);

		float2 b = *(float2*)(&get(B, ik, tx, SB));

		simdMM2(c0, a.x, b);
		simdMM2(c1, a.y, b);
	}
	//when K % STEP != 0--------------------------------------

	const int C0 = ty * SB + tx;
	const int C1 = C0 + SB;

	*(float2*)(C + C0) = c0;
	*(float2*)(C + C1) = c1;
}

#endif


//------------------------------------------------------
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*1), K >= BLOCK_SIZE
//LB = 4: K >= 16
//LB = 3: K >=  8
#ifndef MATMUL_KERNEL_8_1
#define MATMUL_KENERL_8_1

//[M = 16]: LB = 4: Size = 0.25, Time = 1.254 msec, Performace = 428.127 GFlop/s
//[M = 16]: LB = 3: Size = 0.25, Time = 4.602 msec, Performace = 116.66 GFlop/s
template<int LB, int STEP>
__global__ void kernel_8_1(
	const float* __restrict__ A,
	const float* __restrict__ B,
	float* __restrict__ C,
	int SB, int K)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 As[2][1 << LB][(2 << LB) + 1];
	__shared__ float  Bs[2][1 << LB][(1 << LB) + 1];

	int Y = (blockIdx.y << 3 << LB); A += Y * K;//A[Y, 0]
	int X = (blockIdx.x << LB); B += X;//B[0, X]
	C = &get(C, Y, X, SB);

	const int A0 = (ty << 3) * K + tx;
	const int A1 = A0 + K, A2 = A1 + K, A3 = A2 + K;
	const int A4 = A3 + K, A5 = A4 + K, A6 = A5 + K, A7 = A6 + K;
	const int B0 = ty * SB + tx;//[ty, tx]

	//transposed A
	As[buf][tx][(ty << 1)    ] = float4{ A[A0], A[A1], A[A2], A[A3] };
	As[buf][tx][(ty << 1) + 1] = float4{ A[A4], A[A5], A[A6], A[A7] };

	Bs[buf][ty][tx] = B[B0];
	A += STEP; B += (SB << LB);//K +=  STEP
	__syncthreads();

	//compute area------------------------------------------
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

		//transposed A
		As[buf][tx][(ty << 1)    ] = float4{ A[A0], A[A1], A[A2], A[A3] };
		As[buf][tx][(ty << 1) + 1] = float4{ A[A4], A[A5], A[A6], A[A7] };

		Bs[buf][ty][tx] = B[B0];
		A += STEP; B += (SB << LB);//K += STEP
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

	//when K % STEP != 0--------------------------------------
	ty <<= 3;
	for (int ik = 0, RK = K & (STEP - 1); ik < RK; ik++)
	{
		float4 a0, a1;
		a0.x = get(A, ty, ik, K);
		a0.y = get(A, ty + 1, ik, K);
		a0.z = get(A, ty + 2, ik, K);
		a0.w = get(A, ty + 3, ik, K);
		a1.x = get(A, ty + 4, ik, K);
		a1.y = get(A, ty + 5, ik, K);
		a1.z = get(A, ty + 6, ik, K);
		a1.w = get(A, ty + 7, ik, K);

		float b = get(B, ik, tx, SB);

		simdMM4(c0, b, a0);
		simdMM4(c1, b, a1);
	}
	//when K % STEP != 0--------------------------------------

	const int C0 = ty * SB + tx;
	const int C1 = C0 + SB, C2 = C1 + SB, C3 = C2 + SB;
	const int C4 = C3 + SB, C5 = C4 + SB, C6 = C5 + SB, C7 = C6 + SB;

	C[C0] = c0.x;
	C[C1] = c0.y;
	C[C2] = c0.z;
	C[C3] = c0.w;
	C[C4] = c1.x;
	C[C5] = c1.y;
	C[C6] = c1.z;
	C[C7] = c1.w;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*1), K >= BLOCK_SIZE
//LB = 4: K >= 16
//LB = 3: K >=  8
#ifndef MATMUL_KERNEL_4_1
#define MATMUL_KENERL_4_1

//[M = 16]: LB = 4: Size = 0.25, Time = 1.196 msec, Performace = 448.889 GFlop/s
//[M =  8]: LB = 3: Size = 0.25, Time = 4.594 msec, Performace = 116.863 GFlop/s
template<int LB, int STEP>
__global__ void kernel_4_1(
	const float* __restrict__ A,
	const float* __restrict__ B,
	float* __restrict__ C,
	int SB, int K)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 As[2][1 << LB][(1 << LB) + 1];
	__shared__ float  Bs[2][1 << LB][(1 << LB) + 1];

	int Y = (blockIdx.y << 2 << LB); A += Y * K;//A[Y, 0]
	int X = (blockIdx.x << LB); B += X;//B[0, X]
	C = &get(C, Y, X, SB);

	const int A0 = (ty << 2) * K + tx;
	const int A1 = A0 + K, A2 = A1 + K, A3 = A2 + K;
	const int B0 = ty * SB + tx;//[ty, tx]

	As[buf][tx][ty] = float4{ A[A0], A[A1], A[A2], A[A3] };//transposed A
	Bs[buf][ty][tx] = B[B0];
	A += STEP; B += (SB << LB);//K +=  STEP
	__syncthreads();

	//compute area------------------------------------------
	float4 c0 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (K >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float4 a = As[buf][ik][ty];
			float  b = Bs[buf][ik][tx];
			simdMM4(c0, b, a);
		}
		buf ^= 1;

		As[buf][tx][ty] = float4{ A[A0], A[A1], A[A2], A[A3] };//transposed A
		Bs[buf][ty][tx] = B[B0];
		A += STEP; B += (SB << LB);//K += STEP
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float4 a = As[buf][ik][ty];
		float  b = Bs[buf][ik][tx];
		simdMM4(c0, b, a);
	}

	//when K % STEP != 0--------------------------------------
	ty <<= 2;
	for (int ik = 0, RK = K & (STEP - 1); ik < RK; ik++)
	{
		float4 a;//transposed A
		a.x = get(A, ty, ik, K);
		a.y = get(A, ty + 1, ik, K);
		a.z = get(A, ty + 2, ik, K);
		a.w = get(A, ty + 3, ik, K);

		float b = get(B, ik, tx, SB);

		simdMM4(c0, b, a);
	}
	//when K % STEP != 0--------------------------------------

	const int C0 = ty * SB + tx;
	const int C1 = C0 + SB, C2 = C1 + SB, C3 = C2 + SB;

	C[C0] = c0.x;
	C[C1] = c0.y;
	C[C2] = c0.z;
	C[C3] = c0.w;
}

#endif


//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*1), K >= BLOCK_SIZE
//LB = 4: K >= 16
//LB = 3: K >=  8
#ifndef MATMUL_KERNEL_2_1
#define MATMUL_KENERL_2_1

//[M = 16]: LB = 4: Size = 0.25, Time = 1.342 msec, Performace = 400.053 GFlop/s
//[M = 16]: LB = 3: Size = 0.25, Time = 3.258 msec, Performace = 164.785 GFlop/s
template<int LB, int STEP>
__global__ void kernel_2_1(
	const float* __restrict__ A,
	const float* __restrict__ B,
	float* __restrict__ C,
	int SB, int K)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 As[2][1 << LB][(1 << LB) + 1];
	__shared__ float  Bs[2][1 << LB][(1 << LB) + 1];

	int Y = (blockIdx.y << 1 << LB); A += Y * K;//A[Y, 0]
	int X = (blockIdx.x << LB); B += X;//B[0, X]
	C = &get(C, Y, X, SB);

	const int A0 = (ty << 1) * K + tx, A1 = A0 + K;
	const int B0 = ty * SB + tx;//[ty, tx]

	As[buf][tx][ty] = float2{ A[A0], A[A1] };//transposed A
	Bs[buf][ty][tx] = B[B0];
	A += STEP; B += (SB << LB);//K +=  STEP
	__syncthreads();

	//compute area------------------------------------------
	float2 c0 = make_float2(0, 0);
	for (int ok = 1, OK = (K >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float2 a = As[buf][ik][ty];
			float  b = Bs[buf][ik][tx];
			simdMM2(c0, b, a);
		}
		buf ^= 1;

		As[buf][tx][ty] = float2{ A[A0], A[A1] };//transposed A
		Bs[buf][ty][tx] = B[B0];
		A += STEP; B += (SB << LB);//K += STEP
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float2 a = As[buf][ik][ty];
		float  b = Bs[buf][ik][tx];
		simdMM2(c0, b, a);
	}

	//when K % STEP != 0--------------------------------------
	ty <<= 1;
	for (int ik = 0, RK = K & (STEP - 1); ik < RK; ik++)
	{
		float2 a;//transposed A
		a.x = get(A, ty, ik, K);
		a.y = get(A, ty + 1, ik, K);

		float b = get(B, ik, tx, SB);

		simdMM2(c0, b, a);
	}
	//when K % STEP != 0--------------------------------------

	const int C0 = ty * SB + tx;
	const int C1 = C0 + SB;

	C[C0] = c0.x;
	C[C1] = c0.y;
}

#endif


//------------------------------------------------------
//(Y: BLOCK_SIZE*1, X: BLOCK_SIZE*8), K >= BLOCK_SIZE
//LB = 4: K >= 16
//LB = 3: K >=  8
#ifndef MATMUL_KERNEL_1_8
#define MATMUL_KERNEL_1_8

//[N = 16]: LB = 4: Size = 0.25, Time = 1.012 msec, Performace = 530.505 GFlop/s
//[N =  8]: LB = 3: Size = 0.25, Time = 1.44  msec, Performace = 372.827 GFlop/s
template<int LB, int STEP>
__global__ void kernel_1_8(
	const float* __restrict__ A,
	const float* __restrict__ B,
	float* __restrict__ C,
	int SB, int K)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float  As[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Bs[2][1 << LB][(2 << LB) + 1];

	int Y = (blockIdx.x << LB); A += Y * K;//A[Y, 0]
	int X = (blockIdx.y << 3 << LB); B += X;//B[0, X]
	C = &get(C, Y, X, SB);//C[Y, X]

	const int A0 = tx * K + ty;//[ty, tx]
	const int B0 = tx * SB + (ty << 3);//[ty, tx<<3]
	
	Bs[buf][tx][(ty << 1)] = *(float4*)(B + B0);
	Bs[buf][tx][(ty << 1) + 1] = *(float4*)(B + B0 + 4);
	As[buf][ty][tx] = A[A0];
	A += STEP; B += (SB << LB);//B += SB * STEP
	__syncthreads();

	//compute area------------------------------------------
	float4 c0 = make_float4(0, 0, 0, 0);
	float4 c1 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (K >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) 
		{
			float4 b0 = Bs[buf][ik][(ty << 1)];
			float4 b1 = Bs[buf][ik][(ty << 1) + 1];
			float a = As[buf][ik][tx];

			simdMM4(c0, a, b0); 
			simdMM4(c1, a, b1);
		}
		buf ^= 1;

		Bs[buf][tx][(ty << 1)] = *(float4*)(B + B0);
		Bs[buf][tx][(ty << 1) + 1] = *(float4*)(B + B0 + 4);
		As[buf][ty][tx] = A[A0];
		A += STEP; B += (SB << LB);//B += SB * STEP
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) 
	{
		float4 b0 = Bs[buf][ik][(ty << 1)];
		float4 b1 = Bs[buf][ik][(ty << 1) + 1];
		float a = As[buf][ik][tx];

		simdMM4(c0, a, b0);
		simdMM4(c1, a, b1);
	}

	//when K % STEP != 0--------------------------------------
	ty <<= 3;
	for (int ik = 0, RK = K & (STEP - 1); ik < RK; ik++) {
		float a = get(A, tx, ik, K);
		float4 b0 = *(float4*)(&get(B, ik, ty, SB));
		float4 b1 = *(float4*)(&get(B, ik, ty + 4, SB));
		simdMM4(c0, a, b0); simdMM4(c1, a, b1);
	}
	//when K % STEP != 0--------------------------------------

	const int C0 = tx * SB + ty;
	*(float4*)(C + C0) = c0; *(float4*)(C + C0 + 4) = c1;
}

#endif


//(Y: BLOCK_SIZE*1, X: BLOCK_SIZE*4), K >= BLOCK_SIZE
//LB = 4: K >= 16
//LB = 3: K >=  8
#ifndef MATMUL_KERNEL_1_4
#define MATMUL_KERNEL_1_4

//[N = 16]: LB = 4: Size = 0.25, Time = 1.062 msec, Performace = 505.528 GFlop/s
//[N =  8]: LB = 3: Size = 0.25, Time = 1.988 msec, Performace = 270.056 GFlop/s
template<int LB, int STEP>
__global__ void kernel_1_4(
	const float* __restrict__ A,
	const float* __restrict__ B,
	float* __restrict__ C,
	int SB, int K)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float  As[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Bs[2][1 << LB][(1 << LB) + 1];

	int Y = (blockIdx.x << LB); A += Y * K;//A[Y, 0]
	int X = (blockIdx.y << 2 << LB); B += X;//B[0, X]
	C = &get(C, Y, X, SB);//C[Y, X]

	const int A0 = tx * K + ty;//[ty, tx]
	const int B0 = tx * SB + (ty << 2);//[ty, tx<<3]

	Bs[buf][tx][ty] = *(float4*)(B + B0);
	As[buf][ty][tx] = A[A0];
	A += STEP; B += (SB << LB);//B += SB * STEP
	__syncthreads();

	//compute area------------------------------------------
	float4 c0 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (K >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float4 b = Bs[buf][ik][ty];
			float  a = As[buf][ik][tx];
			simdMM4(c0, a, b);
		}
		buf ^= 1;

		Bs[buf][tx][ty] = *(float4*)(B + B0);
		As[buf][ty][tx] = A[A0];
		A += STEP; B += (SB << LB);//B += SB * STEP
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float4 b = Bs[buf][ik][ty];
		float  a = As[buf][ik][tx];
		simdMM4(c0, a, b);
	}

	//when K % STEP != 0--------------------------------------
	ty <<= 2;
	for (int ik = 0, RK = K & (STEP - 1); ik < RK; ik++) {
		float  a = get(A, tx, ik, K);
		float4 b = *(float4*)(&get(B, ik, ty, SB));
		simdMM4(c0, a, b);
	}
	//when K % STEP != 0--------------------------------------

	const int C0 = tx * SB + ty;
	*(float4*)(C + C0) = c0; 
}

#endif


//(Y: BLOCK_SIZE*1, X: BLOCK_SIZE*2), K >= BLOCK_SIZE
//LB = 4: K >= 16
//LB = 3: K >=  8
#ifndef MATMUL_KERNEL_1_2
#define MATMUL_KERNEL_1_2

//[N = 16]: LB = 4: Size = 0.25, Time = 1.368 msec, Performace = 392.449 GFlop/s
//[N =  8]: LB = 3: Size = 0.25, Time = 2.19  msec, Performace = 245.147 GFlop/s
template<int LB, int STEP>
__global__ void kernel_1_2(
	const float* __restrict__ A,
	const float* __restrict__ B,
	float* __restrict__ C,
	int SB, int K)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float  As[2][1 << LB][(1 << LB) + 1];
	__shared__ float2 Bs[2][1 << LB][(1 << LB) + 1];

	int Y = (blockIdx.x << LB); A += Y * K;//A[Y, 0]
	int X = (blockIdx.y << 1 << LB); B += X;//B[0, X]
	C = &get(C, Y, X, SB);//C[Y, X]

	const int A0 = tx * K + ty;//[ty, tx]
	const int B0 = tx * SB + (ty << 1);//[ty, tx<<3]

	Bs[buf][tx][ty] = *(float2*)(B + B0);
	As[buf][ty][tx] = A[A0];
	A += STEP; B += (SB << LB);//B += SB * STEP
	__syncthreads();

	//compute area------------------------------------------
	float2 c0 = make_float2(0, 0);
	for (int ok = 1, OK = (K >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float2 b = Bs[buf][ik][ty];
			float  a = As[buf][ik][tx];
			simdMM2(c0, a, b);
		}
		buf ^= 1;

		Bs[buf][tx][ty] = *(float2*)(B + B0);
		As[buf][ty][tx] = A[A0];
		A += STEP; B += (SB << LB);//B += SB * STEP
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float2 b = Bs[buf][ik][ty];
		float  a = As[buf][ik][tx];
		simdMM2(c0, a, b);
	}

	//when K % STEP != 0--------------------------------------
	ty <<= 1;
	for (int ik = 0, RK = K & (STEP - 1); ik < RK; ik++) {
		float  a = get(A, tx, ik, K);
		float2 b = *(float2*)(&get(B, ik, ty, SB));
		simdMM2(c0, a, b);
	}
	//when K % STEP != 0--------------------------------------

	const int C0 = tx * SB + ty;
	*(float2*)(C + C0) = c0;
}

#endif


//-----------------------------------------------------
//(Y: N, X: M) X*Y<=1024
#ifndef MATMUL_KERNEL_NAIVE
#define MATMUL_KERNEL_NAIVE

// for 32*32*32: Performance= 11.36 GFlop/s, Time= 0.006 msec
// for 31*31*31: Performance=  9.74 GFlop/s, Time= 0.006 msec

__global__ void kernel_naive(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float*  __restrict__ C,
	int SB, int K)
{
	int tx = threadIdx.x + blockIdx.x*blockDim.x;
	int ty = threadIdx.y + blockIdx.y*blockDim.y;
	float v = 0;
	for (int k = 0; k < K; k++)
		v += get(A, ty, k, K) * get(B, k, tx, SB);
	get(C, ty, tx, SB) = v;
}
#endif

#endif