#pragma once

#ifndef MATMUL_SERNEL_H
#define MATMUL_SERNEL_H

//We have:
//(1) N % 4 == 0, N >= 4
//(2) M % 4 == 0, M >= 4
//(3) K % 4 == 0, K >= 4
#ifndef MATMUL_SERNEL_CALL
#define MATMUL_SERNEL_CALL

//LB = log2(BLOCK_SIZE)

//======[Small M]=======================================
#define s8x2_1(LB, stream, A, B, C, N, M, K, SB) \
	sernel_8x2_1<LB, (1<<LB), (1<<LB>>1)>\
		<<< dim3(M<<1>>LB, N>>3>>LB), dim3(1<<LB>>1, 1<<LB), 0, stream >>> \
			(A, B, C, SB, K)

#define s4x2_1(LB, stream, A, B, C, N, M, K, SB) \
	sernel_4x2_1<LB, (1<<LB), (1<<LB>>1)>\
		<<< dim3(M<<1>>LB, N>>2>>LB), dim3(1<<LB>>1, 1<<LB), 0, stream >>> \
			(A, B, C, SB, K)

#define s2x2_1(LB, stream, A, B, C, N, M, K, SB) \
	sernel_2x2_1<LB, (1<<LB), (1<<LB>>1)>\
		<<< dim3(M<<1>>LB, N>>1>>LB), dim3(1<<LB>>1, 1<<LB), 0, stream >>> \
			(A, B, C, SB, K)

#endif


//======[Small M]=======================================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*0.5), K >= BLOCK_SIZE
//LB = 4: K >= 16
//LB = 3: K >=  8
#ifndef MATMUL_SERNEL_8X2_1
#define MATMUL_SENERL_8X2_1

//[M = 8]: LB = 4: Size = 0.25, Time = 7.16  msec, Performace = 74.982  GFlop/s
//[M = 4]: LB = 3: Size = 0.25, Time = 9.208 msec, Performace = 58.3048 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void sernel_8x2_1(
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
	int X = (blockIdx.x << LB >> 1); B += X;//B[0, X]
	C = &get(C, Y, X, SB);

	const int A0 = (ty << 3) * K + tx;
	const int A1 = A0 + K, A2 = A1 + K, A3 = A2 + K;
	const int A4 = A3 + K, A5 = A4 + K, A6 = A5 + K, A7 = A6 + K;
	const int B0 = ty * SB + tx;//[ty, tx]

	float4 a0 = float4{ A[A0], A[A1], A[A2], A[A3] };//transposed A
	float4 a1 = float4{ A[A4], A[A5], A[A6], A[A7] };
	As[buf][tx][(ty << 1)] = a0;
	As[buf][tx][(ty << 1) + 1] = a1;
	A += STEP2;

	float4 a2 = float4{ A[A0], A[A1], A[A2], A[A3] };//transposed A
	float4 a3 = float4{ A[A4], A[A5], A[A6], A[A7] };
	As[buf][tx + STEP2][(ty << 1)] = a2;
	As[buf][tx + STEP2][(ty << 1) + 1] = a3;

	Bs[buf][ty][tx] = B[B0];
	A += STEP2; B += (SB << LB);//K +=  STEP
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

		float4 a0 = float4{ A[A0], A[A1], A[A2], A[A3] };//transposed A
		float4 a1 = float4{ A[A4], A[A5], A[A6], A[A7] };
		As[buf][tx][(ty << 1)] = a0;
		As[buf][tx][(ty << 1) + 1] = a1;
		A += STEP2;

		float4 a2 = float4{ A[A0], A[A1], A[A2], A[A3] };//transposed A
		float4 a3 = float4{ A[A4], A[A5], A[A6], A[A7] };
		As[buf][tx + STEP2][(ty << 1)] = a2;
		As[buf][tx + STEP2][(ty << 1) + 1] = a3;

		Bs[buf][ty][tx] = B[B0];
		A += STEP2; B += (SB << LB);//K += STEP
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


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*0.5), K >= BLOCK_SIZE
//LB = 4: K >= 16
//LB = 3: K >=  8
#ifndef MATMUL_SERNEL_4X2_1
#define MATMUL_SENERL_4X2_1

//[M = 8]: LB = 4: Size = 0.25, Time = 5.918 msec, Performace = 90.7183 GFlop/s
//[M = 4]: LB = 3: Size = 0.25, Time = 9.184 msec, Performace = 58.4572 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void sernel_4x2_1(
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
	int X = (blockIdx.x << LB >> 1); B += X;//B[0, X]
	C = &get(C, Y, X, SB);

	const int A0 = (ty << 2) * K + tx;
	const int A1 = A0 + K, A2 = A1 + K, A3 = A2 + K;
	const int B0 = ty * SB + tx;//[ty, tx]

	//transposed A
	As[buf][tx][ty] = float4{ A[A0], A[A1], A[A2], A[A3] }; A += STEP2;
	As[buf][tx + STEP2][ty] = float4{ A[A0], A[A1], A[A2], A[A3] };

	Bs[buf][ty][tx] = B[B0];
	A += STEP2; B += (SB << LB);//K +=  STEP
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

		//transposed A
		As[buf][tx][ty] = float4{ A[A0], A[A1], A[A2], A[A3] }; A += STEP2;
		As[buf][tx + STEP2][ty] = float4{ A[A0], A[A1], A[A2], A[A3] };

		Bs[buf][ty][tx] = B[B0];
		A += STEP2; B += (SB << LB);//K += STEP
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


//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*0.5), K >= BLOCK_SIZE
//LB = 4: K >= 16
//LB = 3: K >=  8
#ifndef MATMUL_SERNEL_2X2_1
#define MATMUL_SENERL_2X2_1

//[M = 8]: LB = 4: Size = 0.25, Time = 2.37  msec, Performace = 226.528  GFlop/s
//[M = 4]: LB = 3: Size = 0.25, Time = 6.594 msec, Performace =  81.4181 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void sernel_2x2_1(
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
	int X = (blockIdx.x << LB >> 1); B += X;//B[0, X]
	C = &get(C, Y, X, SB);

	const int A0 = (ty << 1) * K + tx, A1 = A0 + K;
	const int B0 = ty * SB + tx;//[ty, tx]

	//transposed A
	As[buf][tx][ty] = float2{ A[A0], A[A1] }; A += STEP2;
	As[buf][tx + STEP2][ty] = float2{ A[A0], A[A1] };

	Bs[buf][ty][tx] = B[B0];
	A += STEP2; B += (SB << LB);//K +=  STEP
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

		//transposed A
		As[buf][tx][ty] = float2{ A[A0], A[A1] }; A += STEP2;
		As[buf][tx + STEP2][ty] = float2{ A[A0], A[A1] };

		Bs[buf][ty][tx] = B[B0];
		A += STEP2; B += (SB << LB);//K += STEP
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

#endif