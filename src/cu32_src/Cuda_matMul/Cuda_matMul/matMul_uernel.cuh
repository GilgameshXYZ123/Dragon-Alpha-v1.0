#pragma once

#ifndef MATMUL_UERNEL_H
#define MATMUL_UERNEL_H

//We have:
//(1) N % 4 == 0, N >= 4
//(2) M % 4 == 0, M >= 4
//(3) K % 4 == 0, K >= 4
#ifndef MATMUL_UERNEL_CALL
#define MATMUL_UERNEL_CALL

//LB = log2(BLOCK_SIZE)

#define	u88_mgk(LB, stream, A, B, C, N, M, K, SB) \
	uernel_8_8_mgk<LB, (1<<LB>>1), (1<<LB)>\
		<<< dim3(M>>3>>LB, N>>3>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SB, K)

#define	u84_mgk(LB, stream, A, B, C, N, M, K, SB) \
	uernel_8_4_mgk<LB, (1<<LB>>1), (1<<LB)>\
		<<< dim3(M>>2>>LB, N>>3>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SB, K)

#define	u48_mgk(LB, stream, A, B, C, N, M, K, SB) \
	uernel_4_8_mgk<LB, (1<<LB>>1), (1<<LB)>\
		<<< dim3(M>>3>>LB, N>>2>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SB, K)

#define	u44_mgk(LB, stream, A, B, C, N, M, K, SB) \
	uernel_4_4_mgk<LB, (1<<LB>>1), (1<<LB)>\
		<<< dim3(M>>2>>LB, N>>2>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SB, K)

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), K % BLOCK_SIZE == 0
//LB = 4: K % 16 == 0
//LB = 3: K %  8 == 0
#ifndef MATMUL_UERNEL_8_8_MGK
#define MATMUL_UERNEL_8_8_MGK

//for: 1024*1024*1024
//LB = 4: Size = 1, Time = 1.281 msec, Performace = 1675.54 GFlop/s(3000)
//LB = 4: Size = 1, Time = 1.3   msec, Performace = 1651.91 GFlop/s(1000)
//LB = 3: Size = 1, Time = 1.379 msec, Performace = 1557.28 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void uernel_8_8_mgk(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float*  __restrict__ C,
	int SB, int K)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 As[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Bs[2][2 << LB][(1 << LB) + 1];

	const int Y = ((blockIdx.y << LB) + ty) << 3;
	const int X = ((blockIdx.x << LB) + tx) << 3;
	const int C0 = Y * SB + X;
	A += (Y + ((tx >= STEP) << 2)) * K;
	B += (X + ((ty >= STEP) << 2));

	const int B0 = ((ty - ((ty >= STEP) << LB >> 1)) << 1) * SB;
	const int B1 = B0 + SB;

	const int A0 = ((tx - ((tx >= STEP) << LB >> 1)) << 1);
	const int A1 = A0 + K, A2 = A1 + K, A3 = A2 + K;

	//load 4 elem from A(transposed)
	float2 a0 = *(float2*)(A + A0);
	float2 a1 = *(float2*)(A + A1);
	float2 a2 = *(float2*)(A + A2);
	float2 a3 = *(float2*)(A + A3);
	As[buf][(tx << 1)][ty] = make_float4(a0.x, a1.x, a2.x, a3.x);
	As[buf][(tx << 1) + 1][ty] = make_float4(a0.y, a1.y, a2.y, a3.y);

	//load 4 elem from B
	Bs[buf][(ty << 1)][tx] = *(float4*)(B + B0);
	Bs[buf][(ty << 1) + 1][tx] = *(float4*)(B + B1);

	A += STEP2; B += (SB << LB);//K += STEP2
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

	for (int ok = 1, OK = (K >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++)
		{
			float4 a0 = As[buf][ik][ty], a1 = As[buf][ik + STEP2][ty];
			float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP2][tx];

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

		//load 4 elem from A(transposed)
		float2 a0 = *(float2*)(A + A0);
		float2 a1 = *(float2*)(A + A1);
		float2 a2 = *(float2*)(A + A2);
		float2 a3 = *(float2*)(A + A3);
		As[buf][(tx << 1)][ty] = make_float4(a0.x, a1.x, a2.x, a3.x);
		As[buf][(tx << 1) + 1][ty] = make_float4(a0.y, a1.y, a2.y, a3.y);

		//load 4 elem from B
		Bs[buf][(ty << 1)][tx] = *(float4*)(B + B0);
		Bs[buf][(ty << 1) + 1][tx] = *(float4*)(B + B1);

		A += STEP2; B += (SB << LB);//K += STEP2
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++)
	{
		float4 a0 = As[buf][ik][ty], a1 = As[buf][ik + STEP2][ty];
		float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP2][tx];

		simdMM4(c0, a0.x, b0);  simdMM4(c1, a0.x, b1);
		simdMM4(c2, a0.y, b0);  simdMM4(c3, a0.y, b1);
		simdMM4(c4, a0.z, b0);  simdMM4(c5, a0.z, b1);
		simdMM4(c6, a0.w, b0);  simdMM4(c7, a0.w, b1);
		simdMM4(c8, a1.x, b0);  simdMM4(c9, a1.x, b1);
		simdMM4(c10, a1.y, b0); simdMM4(c11, a1.y, b1);
		simdMM4(c12, a1.z, b0); simdMM4(c13, a1.z, b1);
		simdMM4(c14, a1.w, b0); simdMM4(c15, a1.w, b1);
	}

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


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*4), K % BLOCK_SIZE == 0
//LB = 4: K % 16 == 0
//LB = 3: K %  8 == 0
#ifndef MATMUL_UERNEL_8_4_MGK
#define MATMUL_UERNEL_8_4_MGK

//for: 1024*1024*1024
//LB = 4: Size = 1, Time = 1.497 msec, Performace = 1434.52 GFlop/s
//LB = 3: Size = 1, Time = 1.688 msec, Performace = 1272.21 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void uernel_8_4_mgk(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float*  __restrict__ C,
	int SB, int K)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 As[2][2 << LB][(1 << LB) + 1];//follow k88
	__shared__ float2 Bs[2][1 << LB][(2 << LB) + 2];//follow k44

	const int Y = ((blockIdx.y << LB) + ty) << 3;
	const int X = ((blockIdx.x << LB) + tx) << 2;
	const int C0 = Y * SB + X;
	A += (Y + ((tx >= STEP) << 2)) * K;
	B += (X + ((ty & 1) << 1));

	const int A0 = ((tx - ((tx >= STEP) << LB >> 1)) << 1);
	const int A1 = A0 + K, A2 = A1 + K, A3 = A2 + K;

	const int By = (ty >> 1) << 1;
	const int Bs_y = By, Bs_x = ((tx << 1) + (ty & 1));
	const int B0 = By * SB, B1 = B0 + SB;

	//load 4 elem from A(transposed)
	float2 a0 = *(float2*)(A + A0);
	float2 a1 = *(float2*)(A + A1);
	float2 a2 = *(float2*)(A + A2);
	float2 a3 = *(float2*)(A + A3);
	As[buf][(tx << 1)][ty] = make_float4(a0.x, a1.x, a2.x, a3.x);
	As[buf][(tx << 1) + 1][ty] = make_float4(a0.y, a1.y, a2.y, a3.y);

	//load 2 elem from B
	Bs[buf][Bs_y][Bs_x] = *(float2*)(B + B0);
	Bs[buf][Bs_y + 1][Bs_x] = *(float2*)(B + B1);

	A += STEP2; B += (SB << LB);//K += STEP2
	__syncthreads();

	//compute area-------------------------------------------------------
	float4 c0 = make_float4(0, 0, 0, 0);
	float4 c1 = make_float4(0, 0, 0, 0);
	float4 c2 = make_float4(0, 0, 0, 0);
	float4 c3 = make_float4(0, 0, 0, 0);
	float4 c4 = make_float4(0, 0, 0, 0);
	float4 c5 = make_float4(0, 0, 0, 0);
	float4 c6 = make_float4(0, 0, 0, 0);
	float4 c7 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (K >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++)
		{
			float4 a0 = As[buf][ik][ty], a1 = As[buf][ik + STEP2][ty];
			float4 b0 = *(float4*)(&Bs[buf][ik][tx << 1]);

			simdMM4(c0, a0.x, b0);
			simdMM4(c1, a0.y, b0);
			simdMM4(c2, a0.z, b0); 
			simdMM4(c3, a0.w, b0);
			simdMM4(c4, a1.x, b0);
			simdMM4(c5, a1.y, b0);
			simdMM4(c6, a1.z, b0);
			simdMM4(c7, a1.w, b0);
		}
		buf ^= 1;

		//load 4 elem from A(transposed)
		float2 a0 = *(float2*)(A + A0);
		float2 a1 = *(float2*)(A + A1);
		float2 a2 = *(float2*)(A + A2);
		float2 a3 = *(float2*)(A + A3);
		As[buf][(tx << 1)][ty] = make_float4(a0.x, a1.x, a2.x, a3.x);
		As[buf][(tx << 1) + 1][ty] = make_float4(a0.y, a1.y, a2.y, a3.y);

		//load 2 elem from B
		Bs[buf][Bs_y][Bs_x] = *(float2*)(B + B0);
		Bs[buf][Bs_y + 1][Bs_x] = *(float2*)(B + B1);

		A += STEP2; B += (SB << LB);//K += STEP2
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++)
	{
		float4 a0 = As[buf][ik][ty], a1 = As[buf][ik + STEP2][ty];
		float4 b0 = *(float4*)(&Bs[buf][ik][tx << 1]);

		simdMM4(c0, a0.x, b0);
		simdMM4(c1, a0.y, b0);
		simdMM4(c2, a0.z, b0);
		simdMM4(c3, a0.w, b0);
		simdMM4(c4, a1.x, b0);
		simdMM4(c5, a1.y, b0);
		simdMM4(c6, a1.z, b0);
		simdMM4(c7, a1.w, b0);
	}

	const int C1 = C0 + SB, C2 = C1 + SB, C3 = C2 + SB;
	const int C4 = C3 + SB, C5 = C4 + SB, C6 = C5 + SB, C7 = C6 + SB;

	*(float4*)(C + C0) = c0;
	*(float4*)(C + C1) = c1;
	*(float4*)(C + C2) = c2;
	*(float4*)(C + C3) = c3; 
	*(float4*)(C + C4) = c4;
	*(float4*)(C + C5) = c5;
	*(float4*)(C + C6) = c6;
	*(float4*)(C + C7) = c7; 
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), K % BLOCK_SIZE == 0
//LB = 4: K % 16 == 0
//LB = 3: K %  8 == 0
#ifndef MATMUL_UERNEL_4_8_MGK
#define MATMUL_UERNEL_4_8_MGK

//for [1024*1024*1024]: 
//LB = 4: Size = 1, Time = 1.706 msec, Performace = 1258.78 GFlop/s
//LB = 3: Size = 1, Time = 1.82  msec, Performace = 1179.94 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void uernel_4_8_mgk(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float*  __restrict__ C,
	int SB, int K)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float2 As[2][1 << LB][(2 << LB) + 2];//follow k44
	__shared__ float4 Bs[2][2 << LB][(1 << LB) + 1];//follow k88

	const int Y = ((blockIdx.y << LB) + ty) << 2;
	const int X = ((blockIdx.x << LB) + tx) << 3;
	const int C0 = Y * SB + X;
	A += (Y + ((tx & 1) << 1))* K;
	B += (X + ((ty >= STEP) << 2));

	const int Ax = (tx >> 1) << 1;
	const int As_x = Ax, As_y = ((ty << 1) + (tx & 1));
	const int A0 = Ax, A1 = A0 + K;

	const int B0 = ((ty - ((ty >= STEP) << LB >> 1)) << 1) * SB;
	const int B1 = B0 + SB;

	//load 2 elem from A(transposed)
	float2 a0 = *(float2*)(A + A0);
	float2 a1 = *(float2*)(A + A1);
	As[buf][As_x][As_y] = float2{ a0.x, a1.x };
	As[buf][As_x + 1][As_y] = float2{ a0.y, a1.y };

	//load 4 elem from B
	Bs[buf][(ty << 1)][tx] = *(float4*)(B + B0);
	Bs[buf][(ty << 1) + 1][tx] = *(float4*)(B + B1);

	A += STEP2; B += (SB << LB);//K += STEP2
	__syncthreads();

	//compute area-------------------------------------------------------
	float4  c0 = make_float4(0, 0, 0, 0), c1 = make_float4(0, 0, 0, 0);
	float4  c2 = make_float4(0, 0, 0, 0), c3 = make_float4(0, 0, 0, 0);
	float4  c4 = make_float4(0, 0, 0, 0), c5 = make_float4(0, 0, 0, 0);
	float4  c6 = make_float4(0, 0, 0, 0), c7 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (K >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++)
		{
			float4 a0 = *(float4*)(&As[buf][ik][ty << 1]);
			float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP2][tx];

			simdMM4(c0, a0.x, b0);  simdMM4(c1, a0.x, b1);
			simdMM4(c2, a0.y, b0);  simdMM4(c3, a0.y, b1);
			simdMM4(c4, a0.z, b0);  simdMM4(c5, a0.z, b1);
			simdMM4(c6, a0.w, b0);  simdMM4(c7, a0.w, b1);
		}
		buf ^= 1;

		//load 2 elem from A(transposed)
		float2 a0 = *(float2*)(A + A0);
		float2 a1 = *(float2*)(A + A1);
		As[buf][As_x][As_y] = float2{ a0.x, a1.x };
		As[buf][As_x + 1][As_y] = float2{ a0.y, a1.y };

		//load 4 elem from B
		Bs[buf][(ty << 1)][tx] = *(float4*)(B + B0);
		Bs[buf][(ty << 1) + 1][tx] = *(float4*)(B + B1);

		A += STEP2; B += (SB << LB);//K += STEP2
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++)
	{
		float4 a0 = *(float4*)(&As[buf][ik][ty << 1]);
		float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP2][tx];

		simdMM4(c0, a0.x, b0);  simdMM4(c1, a0.x, b1);
		simdMM4(c2, a0.y, b0);  simdMM4(c3, a0.y, b1);
		simdMM4(c4, a0.z, b0);  simdMM4(c5, a0.z, b1);
		simdMM4(c6, a0.w, b0);  simdMM4(c7, a0.w, b1);
	}

	const int C1 = C0 + SB;
	const int C2 = C1 + SB;
	const int C3 = C2 + SB;

	*(float4*)(C + C0) = c0;  *(float4*)(C + C0 + 4) = c1;
	*(float4*)(C + C1) = c2;  *(float4*)(C + C1 + 4) = c3;
	*(float4*)(C + C2) = c4;  *(float4*)(C + C2 + 4) = c5;
	*(float4*)(C + C3) = c6;  *(float4*)(C + C3 + 4) = c7;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*4), K % BLOCK_SIZE == 0
//LB = 4: K % 16 == 0
//LB = 3: K %  8 == 0
#ifndef MATMUL_UERNEL_4_4_MGK
#define MATMUL_UERNEL_4_4_MGK

//for [1024*1024*1024]: 
//LB = 4: Size = 1, Time = 1.969 msec, Performace = 1090.65 GFlop/s
//LB = 3: Size = 1, Time = 2.033 msec, Performace = 1056.31 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void uernel_4_4_mgk(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float*  __restrict__ C,
	int SB, int K)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float2 As[2][1 << LB][(2 << LB) + 2];
	__shared__ float2 Bs[2][1 << LB][(2 << LB) + 2];

	int Y = ((blockIdx.y << LB) + ty) << 2; 
	int X = ((blockIdx.x << LB) + tx) << 2; 
	const int C0 = Y * SB + X;//C[Y, X]
	A += (Y + ((tx & 1) << 1))* K;
	B += (X + ((ty & 1) << 1));

	const int Ax = (tx >> 1) << 1;
	const int As_x = Ax, As_y = ((ty << 1) + (tx & 1));
	const int A0 = Ax, A1 = A0 + K;

	const int By = (ty >> 1) << 1;
	const int Bs_y = By, Bs_x = ((tx << 1) + (ty & 1));
	const int B0 = By * SB, B1 = B0 + SB;

	//load 2 elem from A(transposed)
	float2 a0 = *(float2*)(A + A0);
	float2 a1 = *(float2*)(A + A1);
	As[buf][As_x][As_y] = float2{ a0.x, a1.x };
	As[buf][As_x + 1][As_y] = float2{ a0.y, a1.y };

	//load 2 elem from B
	Bs[buf][Bs_y][Bs_x] = *(float2*)(B + B0);
	Bs[buf][Bs_y + 1][Bs_x] = *(float2*)(B + B1);

	A += STEP2; B += (SB << LB);//K += STEP2
	__syncthreads();

	//compute area------------------------------------------
	float4 c0 = make_float4(0, 0, 0, 0);
	float4 c1 = make_float4(0, 0, 0, 0);
	float4 c2 = make_float4(0, 0, 0, 0);
	float4 c3 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (K >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++)
		{
			float4 a = *(float4*)(&As[buf][ik][ty << 1]);
			float4 b = *(float4*)(&Bs[buf][ik][tx << 1]);

			simdMM4(c0, a.x, b);
			simdMM4(c1, a.y, b);
			simdMM4(c2, a.z, b);
			simdMM4(c3, a.w, b);
		}
		buf ^= 1;

		//load 2 elem from A(transposed)
		float2 a0 = *(float2*)(A + A0);
		float2 a1 = *(float2*)(A + A1);
		As[buf][As_x][As_y] = float2{ a0.x, a1.x };
		As[buf][As_x + 1][As_y] = float2{ a0.y, a1.y };

		//load 2 elem from B
		Bs[buf][Bs_y][Bs_x] = *(float2*)(B + B0);
		Bs[buf][Bs_y + 1][Bs_x] = *(float2*)(B + B1);

		A += STEP2; B += (SB << LB);//K += STEP2
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++)
	{
		float4 a = *(float4*)(&As[buf][ik][ty << 1]);
		float4 b = *(float4*)(&Bs[buf][ik][tx << 1]);

		simdMM4(c0, a.x, b);
		simdMM4(c1, a.y, b);
		simdMM4(c2, a.z, b);
		simdMM4(c3, a.w, b);
	}

	const int C1 = C0 + SB;
	const int C2 = C1 + SB;
	const int C3 = C2 + SB;

	*(float4*)(C + C0) = c0;
	*(float4*)(C + C1) = c1;
	*(float4*)(C + C2) = c2;
	*(float4*)(C + C3) = c3;
}

#endif

#endif