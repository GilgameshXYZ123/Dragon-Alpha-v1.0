#pragma once

#ifndef MATMUL_T2_KERNEL_H
#define MATMUL_T2_KERNEL_H

//B   belongs to Mat[M, K]
//B^T belongs to Mat[K, M]
//get(B^T, k, j, M) = get(B, j, k, K)
//for the first stack of function:
//SA = SB = K
//SC = M
#ifndef MATMUL_T2_KERNEL_CALL
#define MATMUL_T2_KERNEL_CALL

//LB = log2(BLOCK_SIZE)

//======[Common]==========================================
#define	k88T2(LB, stream, A, B, C, N, M, K, SC) \
	kernel_t2_8_8<LB, (1<<LB>>1)>\
		<<< dim3(M>>3>>LB, N>>3>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SC, K)

#define k84T2(LB, stream, A, B, C, N, M, K, SC) \
	kernel_t2_8_4<LB, (1<<LB>>1)>\
		<<< dim3(M>>2>>LB, N>>3>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SC, K)

#define k48T2(LB, stream, A, B, C, N, M, K, SC) \
	kernel_t2_4_8<LB, (1<<LB>>1)>\
		<<< dim3(M>>3>>LB, N>>2>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SC, K)

#define k82T2(LB, stream, A, B, C, N, M, K, SC) \
	kernel_t2_8_2<LB, (1<<LB>>1)>\
		<<< dim3(M>>1>>LB, N>>3>>LB), dim3(1<<LB, 1<<LB), 0, stream >>> \
			(A, B, C, SC, K)

#define k28T2(LB, stream, A, B, C, N, M, K, SC) \
	kernel_t2_2_8<LB, (1<<LB>>1)>\
		<<< dim3(M>>3>>LB, N>>1>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SC, K)

#define k44T2(LB, stream, A, B, C, N, M, K, SC) \
	kernel_t2_4_4<LB, (1<<LB>>1)>\
		<<< dim3(M>>2>>LB, N>>2>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SC, K)

#define k42T2(LB, stream, A, B, C, N, M, K, SC) \
	kernel_t2_4_2<LB, (1<<LB>>1)>\
		<<< dim3(M>>1>>LB, N>>2>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SC, K)

#define k24T2(LB, stream, A, B, C, N, M, K, SC) \
	kernel_t2_2_4<LB, (1<<LB>>1)>\
		<<< dim3(M>>2>>LB, N>>1>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SC, K)

//======[Small]===========================================
#define k22T2(LB, stream, A, B, C, N, M, K, SC) \
	kernel_t2_2_2<LB, (1<<LB)>\
		<<< dim3(M>>1>>LB, N>>1>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SC, K)

#define k81T2(LB, stream, A, B, C, N, M, K, SC) \
	kernel_t2_8_1<LB, (1<<LB)>\
		<<< dim3(M>>LB, N>>3>>LB), dim3(1<<LB, 1<<LB), 0, stream >>> \
			(A, B, C, SC, K)

#define k18T2(LB, stream, A, B, C, N, M, K, SC) \
	kernel_t2_1_8<LB, (1<<LB)>\
		<<< dim3(M>>3>>LB, N>>LB), dim3(1<<LB, 1<<LB), 0, stream >>> \
			(A, B, C, SC, K)

#define knaiveT2(GRID_SIZE, stream, A, B, C, N, M, K, SC) \
	kernel_t2_naive\
		<<< dim3(GRID_SIZE, GRID_SIZE), dim3(M / GRID_SIZE, N / GRID_SIZE), 0, stream>>>\
			(A, B, C, SC, K)

#endif


//======[Common]==========================================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), K >= BLOCK_SIZE/2
//LB = 4: K >= 8
#ifndef MATMUL_T2_KERNEL_8_8
#define MATMUL_T2_KERNEL_8_8

//for: 1024*1024*1024
//LB = 4: Size = 1, Time = 1.45  msec, Performace = 1481.02 GFlop/s
//LB = 3: Size = 1, Time = 1.987 msec, Performace = 1080.77 GFlop/s
//for: 1024*1024*1023
//LB = 4: Size = 0.999023, Time = 1.514 msec, Performace = 1417.03 GFlop/s
//LB = 3: Size = 0.999023, Time = 1.947 msec, Performace = 1101.89 GFlop/s
template<int LB, int STEP>
__global__ void kernel_t2_8_8(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float*  __restrict__ C,
	int SC, int K)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 As[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Bs[2][1 << LB][(1 << LB) + 1];

	const int Y = ((blockIdx.y << LB) + ty) << 3; A += Y * K;//A[Y, 0]
	const int X = ((blockIdx.x << LB) + tx) << 3; B += X * K;//B[X, 0]
	const int C0 = Y * SC + X;//C[Y, X]

	const int Ax = tx - ((tx >= STEP) << LB >> 1);
	const int A0 = ((tx >= STEP) << 2) * K + Ax;
	const int A1 = A0 + K, A2 = A1 + K, A3 = A2 + K;

	const int By = ty - ((ty >= STEP) << LB >> 1);
	const int B0 = ((ty >= STEP) << 2) * K + By;
	const int B1 = B0 + K, B2 = B1 + K, B3 = B2 + K;

	As[buf][tx][ty] = float4{ A[A0], A[A1], A[A2], A[A3] };//transposed A
	Bs[buf][ty][tx] = float4{ B[B0], B[B1], B[B2], B[B3] };
	A += STEP; B += STEP;//K += STEP
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

		As[buf][tx][ty] = float4{ A[A0], A[A1], A[A2], A[A3] };//transposed A
		Bs[buf][ty][tx] = float4{ B[B0], B[B1], B[B2], B[B3] };
		A += STEP; B += STEP;//K += STEP
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

	//when K % STEP != 0-------------------------------------
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

		float4 b0, b1;
		b0.x = get(B, 0, ik, K);
		b0.y = get(B, 1, ik, K);
		b0.z = get(B, 2, ik, K);
		b0.w = get(B, 3, ik, K);
		b1.x = get(B, 4, ik, K);
		b1.y = get(B, 5, ik, K);
		b1.z = get(B, 6, ik, K);
		b1.w = get(B, 7, ik, K);

		simdMM4(c0, a0.x, b0); simdMM4(c1, a0.x, b1);
		simdMM4(c2, a0.y, b0); simdMM4(c3, a0.y, b1);
		simdMM4(c4, a0.z, b0); simdMM4(c5, a0.z, b1);
		simdMM4(c6, a0.w, b0); simdMM4(c7, a0.w, b1);
		simdMM4(c8, a1.x, b0); simdMM4(c9, a1.x, b1);
		simdMM4(c10, a1.y, b0); simdMM4(c11, a1.y, b1);
		simdMM4(c12, a1.z, b0); simdMM4(c13, a1.z, b1);
		simdMM4(c14, a1.w, b0); simdMM4(c15, a1.w, b1);
	}
	//when K % STEP != 0-------------------------------------

	const int C1 = C0 + SC, C2 = C1 + SC, C3 = C2 + SC;
	const int C4 = C3 + SC, C5 = C4 + SC, C6 = C5 + SC, C7 = C6 + SC;

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
#ifndef MATMUL_T2_KERNEL_8_4
#define MATMUL_T2_KERNEL_8_4

//LB = 4: Size = 1, Time = 1.788 msec, Performace = 1201.05 GFlop/s
//LB = 3: Size = 1, Time = 2.778 msec, Performace =  773.032 GFlop/s
template<int LB, int STEP>
__global__ void kernel_t2_8_4(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float*  __restrict__ C,
	int SC, int K)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 As[2][1 << LB][(1 << LB) + 1];//followed k88T2
	__shared__ float2 Bs[2][1 << LB >> 1][(2 << LB) + 2];//followed k44T2

	int Y = ((blockIdx.y << LB) + ty) << 3; A += Y * K;//A[Y, 0]
	int X = ((blockIdx.x << LB) + tx) << 2; B += X * K;//B[X, 0]
	int C0 = Y * SC + X;//C[Y, X]

	const int Ay = ((tx >= STEP) << 2);
	const int A0 = Ay * K + (tx - ((tx >= STEP) << LB >> 1));
	const int A1 = A0 + K, A2 = A1 + K, A3 = A2 + K;

	const int By = (ty >> 1);
	const int Bs_y = By, Bs_x = ((tx << 1) + (ty & 1));
	const int B0 = ((ty & 1) << 1) * K + By, B1 = B0 + K;

	As[buf][tx][ty] = float4{ A[A0], A[A1], A[A2], A[A3] };//transpose A
	Bs[buf][Bs_y][Bs_x] = float2{ B[B0], B[B1] };
	A += STEP; B += STEP;//K += STEP
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

		As[buf][tx][ty] = float4{ A[A0], A[A1], A[A2], A[A3] };//transpose A
		Bs[buf][Bs_y][Bs_x] = float2{ B[B0], B[B1] };
		A += STEP; B += STEP;//K += STEP
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

	//when K % STEP != 0-------------------------------------
	for (int ik = 0, RK = K & (STEP - 1); ik < RK; ik++)
	{
		float4 b0;
		b0.x = get(B, 0, ik, K);
		b0.y = get(B, 1, ik, K);
		b0.z = get(B, 2, ik, K);
		b0.w = get(B, 3, ik, K);

		float4 a0, a1;//transposed A
		a0.x = get(A, 0, ik, K);
		a0.y = get(A, 1, ik, K);
		a0.z = get(A, 2, ik, K);
		a0.w = get(A, 3, ik, K);
		a1.x = get(A, 4, ik, K);
		a1.y = get(A, 5, ik, K);
		a1.z = get(A, 6, ik, K);
		a1.w = get(A, 7, ik, K);

		simdMM4( c0, a0.x, b0);
		simdMM4( c2, a0.y, b0);
		simdMM4( c4, a0.z, b0);
		simdMM4( c6, a0.w, b0);
		simdMM4( c8, a1.x, b0);
		simdMM4(c10, a1.y, b0);
		simdMM4(c12, a1.z, b0);
		simdMM4(c14, a1.w, b0);
	}
	//when K % STEP != 0-------------------------------------

	const int C1 = C0 + SC, C2 = C1 + SC, C3 = C2 + SC;
	const int C4 = C3 + SC, C5 = C4 + SC, C6 = C5 + SC, C7 = C6 + SC;

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
#ifndef MATMUL_T2_KERNEL_4_8
#define MATMUL_T2_KERNEL_4_8

//LB = 4: Size = 1, Time = 1.876 msec, Performace = 1144.71 GFlop/s
//LB = 3: Size = 1, Time = 2.4  msec, Performace =  894.785 GFlop/s
template<int LB, int STEP>
__global__ void kernel_t2_4_8(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float*  __restrict__ C,
	int SC, int K)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float2 As[2][1 << LB >> 1][(2 << LB) + 2];//followed k44T2
	__shared__ float4 Bs[2][1 << LB][(1 << LB) + 1];//followed k88T2

	int Y = ((blockIdx.y << LB) + ty) << 2; A += Y * K;//A[Y, 0]
	int X = ((blockIdx.x << LB) + tx) << 3; B += X * K;//B[X, 0]
	const int C0 = Y * SC + X;//C[Y, X]

	const int Ax = (tx >> 1);
	const int As_x = Ax, As_y = ((ty << 1) + (tx & 1));
	const int A0 = ((tx & 1) << 1) * K + Ax, A1 = A0 + K;

	const int Bx = ((ty >= STEP) << 2);
	const int B0 = Bx * K + (ty - ((ty >= STEP) << LB >> 1));
	const int B1 = B0 + K, B2 = B1 + K, B3 = B2 + K;

	As[buf][As_x][As_y] = float2{ A[A0], A[A1] };//transpose A
	Bs[buf][ty][tx] = float4{ B[B0], B[B1], B[B2], B[B3] };
	A += STEP; B += STEP;//K += STEP
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

		As[buf][As_x][As_y] = float2{ A[A0], A[A1] };//transpose A
		Bs[buf][ty][tx] = float4{ B[B0], B[B1], B[B2], B[B3] };
		A += STEP; B += STEP;//K += STEP
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

	//when K % STEP != 0-------------------------------------
	for (int ik = 0, RK = K & (STEP - 1); ik < RK; ik++)
	{
		float4 a0;//transposed A
		a0.x = get(A, 0, ik, K);
		a0.y = get(A, 1, ik, K);
		a0.z = get(A, 2, ik, K);
		a0.w = get(A, 3, ik, K);

		float4 b0, b1;
		b0.x = get(B, 0, ik, K);
		b0.y = get(B, 1, ik, K);
		b0.z = get(B, 2, ik, K);
		b0.w = get(B, 3, ik, K);
		b1.x = get(B, 4, ik, K);
		b1.y = get(B, 5, ik, K);
		b1.z = get(B, 6, ik, K);
		b1.w = get(B, 7, ik, K);

		simdMM4(c0, a0.x, b0); simdMM4(c1, a0.x, b1);
		simdMM4(c2, a0.y, b0); simdMM4(c3, a0.y, b1);
		simdMM4(c4, a0.z, b0); simdMM4(c5, a0.z, b1);
		simdMM4(c6, a0.w, b0); simdMM4(c7, a0.w, b1);
	}
	//when K % STEP != 0-------------------------------------

	const int C1 = C0 + SC;
	const int C2 = C1 + SC;
	const int C3 = C2 + SC;

	*(float4*)(C + C0) = c0;  *(float4*)(C + C0 + 4) = c1;
	*(float4*)(C + C1) = c2;  *(float4*)(C + C1 + 4) = c3;
	*(float4*)(C + C2) = c4;  *(float4*)(C + C2 + 4) = c5;
	*(float4*)(C + C3) = c6;  *(float4*)(C + C3 + 4) = c7;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*4), K >= BLOCK_SIZE/2
//LB = 4: K >= 8
#ifndef MATMUL_T2_KERNEL_4_4
#define MATMUL_T2_KERNEL_4_4

//LB = 4: Size = 1, Time = 2.234 msec, Performace = 961.273 GFlop/s
//LB = 3: Size = 1, Time = 3.752 msec, Performace = 572.357 GFlop/s(1000)
template<int LB, int STEP>
__global__ void kernel_t2_4_4(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float*  __restrict__ C,
	int SC, int K)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float2 As[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2 Bs[2][1 << LB >> 1][(2 << LB) + 2];

	int Y = (blockIdx.y << 2 << LB); A += Y * K;//A[Y, 0]
	int X = (blockIdx.x << 2 << LB); B += X * K;//B[X, 0]
	C = &get(C, Y, X, SC);//C[Y, X]

	const int Ax = (tx >> 1), Ay = (ty << 2) + ((tx & 1) << 1);
	const int A0 = Ay * K + Ax, A1 = A0 + K;
	const int As_x = Ax, As_y = ((ty << 1) + (tx & 1));

	const int By = (ty >> 1), Bx = (tx << 2) + ((ty & 1) << 1);
	const int B0 = Bx * K + By, B1 = B0 + K;
	const int Bs_y = By, Bs_x = ((tx << 1) + (ty & 1));

	As[buf][As_x][As_y] = float2{ A[A0], A[A1] };//transposeA
	Bs[buf][Bs_y][Bs_x] = float2{ B[B0], B[B1] };
	A += STEP; B += STEP;//K += STEP
	__syncthreads();

	//compute area-------------------------------------------------------
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

		As[buf][As_x][As_y] = float2{ A[A0], A[A1] };//transposeA
		Bs[buf][Bs_y][Bs_x] = float2{ B[B0], B[B1] };
		A += STEP; B += STEP;//K += STEP
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

	//when K % STEP != 0-------------------------------------
	ty <<= 2; tx <<= 2;
	for (int ik = 0, RK = K & (STEP - 1); ik < RK; ik++)
	{
		float4 a;
		a.x = get(A, ty, ik, K);
		a.y = get(A, ty + 1, ik, K);
		a.z = get(A, ty + 2, ik, K);
		a.w = get(A, ty + 3, ik, K);

		float4 b;
		b.x = get(B, tx, ik, K);
		b.y = get(B, tx + 1, ik, K);
		b.z = get(B, tx + 2, ik, K);
		b.w = get(B, tx + 3, ik, K);

		simdMM4(c0, a.x, b);
		simdMM4(c1, a.y, b);
		simdMM4(c2, a.z, b);
		simdMM4(c3, a.w, b);
	}
	//when K % STEP != 0-------------------------------------

	const int C0 = ty * SC + tx;
	const int C1 = C0 + SC;
	const int C2 = C1 + SC;
	const int C3 = C2 + SC;

	*(float4*)(C + C0) = c0;
	*(float4*)(C + C1) = c1;
	*(float4*)(C + C2) = c2;
	*(float4*)(C + C3) = c3;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*2), K >= BLOCK_SIZE/2
//LB = 4: K >= 8
#ifndef MATMUL_T2_KERNEL_8_2
#define MATMUL_T2_KERNEL_8_2

//LB = 4: Size = 1, Time = 2.072 msec, Performace = 1036.43 GFlop/s
template<int LB, int STEP>
__global__ void kernel_t2_8_2(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float*  __restrict__ C,
	int SC, int K)
{
	bool buf = 0;
	__shared__ float4 As[2][1 << LB][(1 << LB) + 1];//followed k88T2
	__shared__ float  Bs[2][1 << LB >> 1][(2 << LB) + 2];//followed k44T2

	int Y = (blockIdx.y << 3 << LB);//Y = blockIdx.y * 8 * BLOCK_SIZE
	int X = (blockIdx.x << 1 << LB);//X = blockIdx.x * 2 * BLOCK_SIZE
	A = A + Y * K;//A[Y, 0]
	B = B + X * K;//B[X, 0]
	C = &get(C, Y, X, SC);//C[Y, X]

	int tx = threadIdx.x, ty = threadIdx.y;
	const int Ay = (ty << 3) + ((tx >= STEP) << 2), Ax = tx - ((tx >= STEP) << LB >> 1);
	const int By = (ty >> 1), Bx = (tx << 1) + (ty & 1);
	const int Bs_y = By, Bs_x = ((tx << 1) + (ty & 1));
	const int Bxy = Bx * K + By;//[Bx, By]

	As[buf][tx][ty].x = get(A, Ay    , Ax, K);//transpose A
	As[buf][tx][ty].y = get(A, Ay + 1, Ax, K);
	As[buf][tx][ty].z = get(A, Ay + 2, Ax, K);
	As[buf][tx][ty].w = get(A, Ay + 3, Ax, K);
	Bs[buf][Bs_y][Bs_x] = B[Bxy];
	A += STEP; B += STEP;
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
		Bs[buf][Bs_y][Bs_x] = B[Bxy];
		A += STEP; B += STEP;
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
		float4 a0, a1;
		a0.x = get(A, ty    , ik, K);
		a0.y = get(A, ty + 1, ik, K);
		a0.z = get(A, ty + 2, ik, K);
		a0.w = get(A, ty + 3, ik, K);
		a1.x = get(A, ty + 4, ik, K);
		a1.y = get(A, ty + 5, ik, K);
		a1.z = get(A, ty + 6, ik, K);
		a1.w = get(A, ty + 7, ik, K);

		float2 b0;
		b0.x = get(B, tx    , ik, K);
		b0.y = get(B, tx + 1, ik, K);

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

	*(float2*)(&get(C, ty    , tx, SC)) = c0;
	*(float2*)(&get(C, ty + 1, tx, SC)) = c2;
	*(float2*)(&get(C, ty + 2, tx, SC)) = c4;
	*(float2*)(&get(C, ty + 3, tx, SC)) = c6;
	*(float2*)(&get(C, ty + 4, tx, SC)) = c8;
	*(float2*)(&get(C, ty + 5, tx, SC)) = c10;
	*(float2*)(&get(C, ty + 6, tx, SC)) = c12;
	*(float2*)(&get(C, ty + 7, tx, SC)) = c14;
}

#endif


//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*8), K >= BLOCK_SIZE/2
//LB = 4: K >= 8
#ifndef MATMUL_T2_KERNEL_2_8
#define MATMUL_T2_KERNEL_2_8

//LB = 4: Size = 1, Time = 2.68 msec, Performace = 801.3 GFlop/s
template<int LB, int STEP>
__global__ void kernel_t2_2_8(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float*  __restrict__ C,
	int SC, int K)
{
	bool buf = 0;
	__shared__ float  As[2][1 << LB >> 1][(2 << LB) + 2];//followed k44T2
	__shared__ float4 Bs[2][1 << LB][(1 << LB) + 1];//followed k88T2

	int Y = (blockIdx.y << 1 << LB);//Y = blockIdx.y * 2 * BLOCK_SIZE
	int X = (blockIdx.x << 3 << LB);//X = blockIdx.x * 8 * BLOCK_SIZE
	A = A + Y * K;//A[Y, 0]
	B = B + X * K;//B[X, 0]
	C = &get(C, Y, X, SC);//C[Y, X]

	int tx = threadIdx.x, ty = threadIdx.y;
	const int Ax = (tx >> 1), Ay = (ty << 1) + (tx & 1);
	const int As_x = Ax, As_y = ((ty << 1) + (tx & 1));
	const int Bx = (tx << 3) + ((ty >= STEP) << 2), By = ty - ((ty >= STEP) << LB >> 1);
	const int Ayx = Ay * K + Ax;//[Ay, Ax]

	As[buf][As_x][As_y] = A[Ayx];//transpose A
	Bs[buf][ty][tx].x = get(B, Bx    , By, K);
	Bs[buf][ty][tx].y = get(B, Bx + 1, By, K);
	Bs[buf][ty][tx].z = get(B, Bx + 2, By, K);
	Bs[buf][ty][tx].w = get(B, Bx + 3, By, K);
	A += STEP; B += STEP;
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

		As[buf][As_x][As_y] = A[Ayx];//transpose A
		Bs[buf][ty][tx].x = get(B, Bx    , By, K);
		Bs[buf][ty][tx].y = get(B, Bx + 1, By, K);
		Bs[buf][ty][tx].z = get(B, Bx + 2, By, K);
		Bs[buf][ty][tx].w = get(B, Bx + 3, By, K);
		A += STEP; B += STEP;
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
		float2 a0;
		a0.x = get(A, ty    , ik, K);
		a0.y = get(A, ty + 1, ik, K);

		float4 b0, b1;
		b0.x = get(B, tx    , ik, K);
		b0.y = get(B, tx + 1, ik, K);
		b0.z = get(B, tx + 2, ik, K);
		b0.w = get(B, tx + 3, ik, K);
		b1.x = get(B, tx + 4, ik, K);
		b1.y = get(B, tx + 5, ik, K);
		b1.z = get(B, tx + 6, ik, K);
		b1.w = get(B, tx + 7, ik, K);

		simdMM4(c0, a0.x, b0); simdMM4(c1, a0.x, b1);
		simdMM4(c2, a0.y, b0); simdMM4(c3, a0.y, b1);
	}
	//when K % STEP != 0-------------------------------------

	*(float4*)(&get(C, ty    , tx, SC)) = c0; *(float4*)(&get(C, ty    , tx + 4, SC)) = c1;
	*(float4*)(&get(C, ty + 1, tx, SC)) = c2; *(float4*)(&get(C, ty + 1, tx + 4, SC)) = c3;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*2), K >= BLOCK_SIZE/2
#ifndef MATMUL_T2_KERNEL_4_2
#define MATMUL_T2_KERNEL_4_2

//LB = 4: Size = 1, Time = 2.886 msec, Performace = 744.104 GFlop/s
template<int LB, int STEP>
__global__ void kernel_t2_4_2(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float*  __restrict__ C,
	int SC, int K)
{
	bool buf = 0;
	__shared__ float2 As[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float  Bs[2][1 << LB >> 1][(2 << LB) + 2];

	int Y = (blockIdx.y << 2 << LB);//Y = blockIdx.y * 4 * BLOCK_SIZE
	int X = (blockIdx.x << 1 << LB);//X = blockIdx.x * 2 * BLOCK_SIZE
	A = A + Y * K;//A[Y, 0]
	B = B + X * K;//B[X, 0]
	C = &get(C, Y, X, SC);//C[Y, X]

	int tx = threadIdx.x, ty = threadIdx.y;
	const int Ax = (tx >> 1), Ay = (ty << 2) + ((tx & 1) << 1);
	const int By = (ty >> 1), Bx = (tx << 1) + (ty & 1);
	const int As_x = Ax, As_y = ((ty << 1) + (tx & 1));
	const int Bs_y = By, Bs_x = ((tx << 1) + (ty & 1));
	const int Ayx = Ay * K + Ax;//[Ay, Ax]
	const int Bxy = Bx * K + By;//[Bx, By]

	As[buf][As_x][As_y].x = A[Ayx    ];//transposeA
	As[buf][As_x][As_y].y = A[Ayx + K];//[Ay + 1, Ax]
	Bs[buf][Bs_y][Bs_x] = B[Bxy];
	A += STEP; B += STEP;
	__syncthreads();

	//compute area-------------------------------------------------------
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

		As[buf][As_x][As_y].x = A[Ayx    ];//transposeA
		As[buf][As_x][As_y].y = A[Ayx + K];//[Ay + 1, Ax]
		Bs[buf][Bs_y][Bs_x] = B[Bxy];
		A += STEP; B += STEP;
		__syncthreads();
	}
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a = *(float4*)(&As[buf][ik][ty << 1]);
		float2 b = *(float2*)(&Bs[buf][ik][tx << 1]);

		simdMM2(c0, a.x, b);
		simdMM2(c1, a.y, b);
		simdMM2(c2, a.z, b);
		simdMM2(c3, a.w, b);
	}

	//when K % STEP != 0-------------------------------------
	ty <<= 2; tx <<= 1;
	for (int ik = 0, RK = K & (STEP - 1); ik < RK; ik++)
	{
		float4 a;
		a.x = get(A, ty    , ik, K);
		a.y = get(A, ty + 1, ik, K);
		a.z = get(A, ty + 2, ik, K);
		a.w = get(A, ty + 3, ik, K);

		float2 b;
		b.x = get(B, tx    , ik, K);
		b.y = get(B, tx + 1, ik, K);

		simdMM2(c0, a.x, b);
		simdMM2(c1, a.y, b);
		simdMM2(c2, a.z, b);
		simdMM2(c3, a.w, b);
	}
	//when K % STEP != 0-------------------------------------

	*(float2*)(&get(C, ty    , tx, SC)) = c0;
	*(float2*)(&get(C, ty + 1, tx, SC)) = c1;
	*(float2*)(&get(C, ty + 2, tx, SC)) = c2;
	*(float2*)(&get(C, ty + 3, tx, SC)) = c3;
}

#endif 


//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*4), K >= BLOCK_SIZE/2
#ifndef MATMUL_T2_KERNEL_2_4
#define MATMUL_T2_KERNEL_2_4

//LB = 4: Size = 1, Time = 3.33 msec, Performace = 644.89 GFlop/s
template<int LB, int STEP>
__global__ void kernel_t2_2_4(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float*  __restrict__ C,
	int SC, int K)
{
	bool buf = 0;
	__shared__ float  As[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2 Bs[2][1 << LB >> 1][(2 << LB) + 2];

	int Y = (blockIdx.y << 1 << LB);//Y = blockIdx.y * 2 * BLOCK_SIZE
	int X = (blockIdx.x << 2 << LB);//X = blockIdx.x * 4 * BLOCK_SIZE
	A = A + Y * K;//A[Y, 0]
	B = B + X * K;//B[X, 0]
	C = &get(C, Y, X, SC);//C[Y, X]

	int tx = threadIdx.x, ty = threadIdx.y;
	const int Ax = tx >> 1, Ay = (ty << 1) + (tx & 1);
	const int By = ty >> 1, Bx = (tx << 2) + ((ty & 1) << 1);
	const int As_x = Ax, As_y = ((ty << 1) + (tx & 1));
	const int Bs_y = By, Bs_x = ((tx << 1) + (ty & 1));
	const int Ayx = Ay * K + Ax;//[Ay, Ax]
	const int Bxy = Bx * K + By;//[Bx, By]

	As[buf][As_x][As_y] = A[Ayx];//transpose A
	Bs[buf][Bs_y][Bs_x].x = B[Bxy    ];
	Bs[buf][Bs_y][Bs_x].y = B[Bxy + K];//[Bx + 1, By]
	A += STEP; B += STEP;
	__syncthreads();

	//compute area-------------------------------------------------------
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

		As[buf][As_x][As_y] = A[Ayx];//transpose A
		Bs[buf][Bs_y][Bs_x].x = B[Bxy    ];
		Bs[buf][Bs_y][Bs_x].y = B[Bxy + K];//[Bx + 1, By]
		A += STEP; B += STEP;
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float2 a = *(float2*)(&As[buf][ik][ty << 1]);
		float4 b = *(float4*)(&Bs[buf][ik][tx << 1]);
		simdMM4(c0, a.x, b);
		simdMM4(c1, a.y, b);
	}

	//when K % STEP != 0-------------------------------------
	ty <<= 1; tx <<= 2;
	for (int ik = 0, RK = K & (STEP - 1); ik < RK; ik++)
	{
		float2 a;
		a.x = get(A, ty    , ik, K);
		a.y = get(A, ty + 1, ik, K);

		float4 b;
		b.x = get(B, tx   , ik, K);
		b.y = get(B, tx + 1, ik, K);
		b.z = get(B, tx + 2, ik, K);
		b.w = get(B, tx + 3, ik, K);

		simdMM4(c0, a.x, b);
		simdMM4(c1, a.y, b);
	}
	//when K % STEP != 0-------------------------------------

	*(float4*)(&get(C, ty    , tx, SC)) = c0;
	*(float4*)(&get(C, ty + 1, tx, SC)) = c1;
}

#endif


//======[Small]===========================================
//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*2), K >= BLOCK_SIZE
#ifndef MATMUL_T2_KERNEL_2_2
#define MATMUL_T2_KERNEL_2_2

//LB = 4: Size = 1, Time = 3.652 msec, Performace = 588.029 GFlop/s
template<int LB, int STEP>
__global__ void kernel_t2_2_2(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float*  __restrict__ C,
	int SC, int K)
{
	bool buf = 0;
	__shared__ float2 As[2][1 << LB][(1 << LB) + 2];
	__shared__ float2 Bs[2][1 << LB][(1 << LB) + 2];

	int Y = (blockIdx.y << 1 << LB);//Y = blockIdx.y * 2 * BLOCK_SIZE
	int X = (blockIdx.x << 1 << LB);//X = blockIdx.x * 2 * BLOCK_SIZE
	A = A + Y * K;//A[Y, 0]
	B = B + X * K;//B[X, 0]
	C = &get(C, Y, X, SC);//C[Y, X]

	int tx = threadIdx.x, ty = threadIdx.y;
	const int Ayx = (ty << 1) * K + tx;//[ty<<1, tx]
	const int Bxy = (tx << 1) * K + ty;//[tx<<1, ty]

	As[buf][tx][ty].x = A[Ayx    ];//transpose A
	As[buf][tx][ty].y = A[Ayx + K];//[Ay + 1, Ax]
	Bs[buf][ty][tx].x = B[Bxy    ];
	Bs[buf][ty][tx].y = B[Bxy + K];//[Bx + 1, By]
	A += STEP; B += STEP;
	__syncthreads();

	//compute area-------------------------------------------------------
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

		As[buf][tx][ty].x = A[Ayx    ];//transpose A
		As[buf][tx][ty].y = A[Ayx + K];//[Ay + 1, Ax]
		Bs[buf][ty][tx].x = B[Bxy    ];
		Bs[buf][ty][tx].y = B[Bxy + K];//[Bx + 1, By]
		A += STEP; B += STEP;
		__syncthreads();
	}
#pragma once
	for (int ik = 0; ik < STEP; ik++) {
		float2 a = As[buf][ik][ty];
		float2 b = Bs[buf][ik][tx];
		simdMM2(c0, a.x, b);
		simdMM2(c1, a.y, b);
	}

	//when K % STEP != 0-------------------------------------
	ty <<= 1; tx <<= 1;
	for (int ik = 0, RK = K & (STEP - 1); ik < RK; ik++)
	{
		float2 a;
		a.x = get(A, ty    , ik, K);
		a.y = get(A, ty + 1, ik, K);

		float2 b;
		b.x = get(B, tx    , ik, K);
		b.y = get(B, tx + 1, ik, K);

		simdMM2(c0, a.x, b);
		simdMM2(c1, a.y, b);
	}
	//when K % STEP != 0-------------------------------------

	*(float2*)(&get(C, ty    , tx, SC)) = c0;
	*(float2*)(&get(C, ty + 1, tx, SC)) = c1;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*1), K >= BLOCK_SIZE
//LB = 4: K >= 8
#ifndef MATMUL_T2_KERNEL_8_1
#define MATMUL_T2_KENERL_8_1

//LB = 4: Size = 1, Time = 2.992 msec, Performace = 717.742 GFlop/s
template<int LB, int STEP>
__global__ void kernel_t2_8_1(
	const float* __restrict__ A,
	const float* __restrict__ B,
	float* __restrict__ C,
	int SC, int K)
{
	bool buf = 0;
	__shared__ float4 As[2][1 << LB][(2 << LB) + 1];
	__shared__ float  Bs[2][1 << LB][(1 << LB) + 1];

	int Y = (blockIdx.y << 3 << LB);//blockIdx.y * 8 * BLOCK_SIZE
	int X = (blockIdx.x << LB);//blockIdx.x * BLOCK_SIZE
	A = A + Y * K;//A[Y, 0]
	B = B + X * K;//B[X, 0]
	C = &get(C, Y, X, SC);//C[Y, X]

	int ty = threadIdx.y, tx = threadIdx.x;
	const int Bxy = tx * K + ty;//[tx, ty]

	As[buf][tx][(ty << 1)].x = get(A, (ty << 3), tx, K);
	As[buf][tx][(ty << 1)].y = get(A, (ty << 3) + 1, tx, K);
	As[buf][tx][(ty << 1)].z = get(A, (ty << 3) + 2, tx, K);
	As[buf][tx][(ty << 1)].w = get(A, (ty << 3) + 3, tx, K);
	As[buf][tx][(ty << 1) + 1].x = get(A, (ty << 3) + 4, tx, K);
	As[buf][tx][(ty << 1) + 1].y = get(A, (ty << 3) + 5, tx, K);
	As[buf][tx][(ty << 1) + 1].z = get(A, (ty << 3) + 6, tx, K);
	As[buf][tx][(ty << 1) + 1].w = get(A, (ty << 3) + 7, tx, K);
	Bs[buf][ty][tx] = B[Bxy];
	A += STEP; B += STEP;
	__syncthreads();

	//compute area-------------------------------------------------------
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

		As[buf][tx][(ty << 1)].x = get(A, (ty << 3), tx, K);
		As[buf][tx][(ty << 1)].y = get(A, (ty << 3) + 1, tx, K);
		As[buf][tx][(ty << 1)].z = get(A, (ty << 3) + 2, tx, K);
		As[buf][tx][(ty << 1)].w = get(A, (ty << 3) + 3, tx, K);
		As[buf][tx][(ty << 1) + 1].x = get(A, (ty << 3) + 4, tx, K);
		As[buf][tx][(ty << 1) + 1].y = get(A, (ty << 3) + 5, tx, K);
		As[buf][tx][(ty << 1) + 1].z = get(A, (ty << 3) + 6, tx, K);
		As[buf][tx][(ty << 1) + 1].w = get(A, (ty << 3) + 7, tx, K);
		Bs[buf][ty][tx] = B[Bxy];
		A += STEP; B += STEP;
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

	//when K % STEP != 0-------------------------------------
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

		float b = get(B, tx, ik, K);

		simdMM4(c0, b, a0);
		simdMM4(c1, b, a1);
	}
	//when K % STEP != 0-------------------------------------

	get(C, ty, tx, SC) = c0.x;
	get(C, ty + 1, tx, SC) = c0.y;
	get(C, ty + 2, tx, SC) = c0.z;
	get(C, ty + 3, tx, SC) = c0.w;
	get(C, ty + 4, tx, SC) = c1.x;
	get(C, ty + 5, tx, SC) = c1.y;
	get(C, ty + 6, tx, SC) = c1.z;
	get(C, ty + 7, tx, SC) = c1.w;
}

#endif


//(Y: BLOCK_SIZE  , X: BLOCK_SIZE*8), K >= BLOCK_SIZE
//LB = 4: K >= 8
#ifndef MATMUL_T2_KERNEL_1_8
#define MATMUL_T2_KERNEL_1_8

//LB = 4: Size = 1, Time = 8.942 msec, Performace = 240.157 GFlop/s
template<int LB, int STEP>
__global__ void kernel_t2_1_8(
	const float* __restrict__ A,
	const float* __restrict__ B,
	float* __restrict__ C,
	int SC, int K)
{
	bool buf = 0;
	__shared__ float   As[2][1 << LB][(1 << LB) + 1];
	__shared__ float4  Bs[2][1 << LB][(2 << LB) + 1];

	int Y = (blockIdx.y << LB);//Y = blockIdx.y * BLOKC_SIZE
	int X = (blockIdx.x << 3 << LB);//X = blockIdx.x * 8 * BLOCK_SIZE
	A = A + Y * K;//A[Y, 0]
	B = B + X * K;//B[X, 0]
	C = &get(C, Y, X, SC);//C[Y, X]

	int ty = threadIdx.y, tx = threadIdx.x;
	const int Ayx = ty * K + tx;//[ty, tx]

	As[buf][tx][ty] = A[Ayx];
	Bs[buf][ty][(tx << 1)].x = get(B, (tx << 3), ty, K);
	Bs[buf][ty][(tx << 1)].y = get(B, (tx << 3) + 1, ty, K);
	Bs[buf][ty][(tx << 1)].z = get(B, (tx << 3) + 2, ty, K);
	Bs[buf][ty][(tx << 1)].w = get(B, (tx << 3) + 3, ty, K);
	Bs[buf][ty][(tx << 1) + 1].x = get(B, (tx << 3) + 4, ty, K);
	Bs[buf][ty][(tx << 1) + 1].y = get(B, (tx << 3) + 5, ty, K);
	Bs[buf][ty][(tx << 1) + 1].z = get(B, (tx << 3) + 6, ty, K);
	Bs[buf][ty][(tx << 1) + 1].w = get(B, (tx << 3) + 7, ty, K);
	A += STEP; B += STEP;
	__syncthreads();

	//compute area-------------------------------------------------------
	float4 c0 = make_float4(0, 0, 0, 0);
	float4 c1 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (K >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float a = As[buf][ik][ty];
			float4 b0 = Bs[buf][ik][(tx << 1)];
			float4 b1 = Bs[buf][ik][(tx << 1) + 1];
			simdMM4(c0, a, b0); simdMM4(c1, a, b1);
		}
		buf ^= 1;

		As[buf][tx][ty] = A[Ayx];
		Bs[buf][ty][(tx << 1)].x = get(B, (tx << 3), ty, K);
		Bs[buf][ty][(tx << 1)].y = get(B, (tx << 3) + 1, ty, K);
		Bs[buf][ty][(tx << 1)].z = get(B, (tx << 3) + 2, ty, K);
		Bs[buf][ty][(tx << 1)].w = get(B, (tx << 3) + 3, ty, K);
		Bs[buf][ty][(tx << 1) + 1].x = get(B, (tx << 3) + 4, ty, K);
		Bs[buf][ty][(tx << 1) + 1].y = get(B, (tx << 3) + 5, ty, K);
		Bs[buf][ty][(tx << 1) + 1].z = get(B, (tx << 3) + 6, ty, K);
		Bs[buf][ty][(tx << 1) + 1].w = get(B, (tx << 3) + 7, ty, K);
		A += STEP; B += STEP;
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float a = As[buf][ik][ty];
		float4 b0 = Bs[buf][ik][(tx << 1)];
		float4 b1 = Bs[buf][ik][(tx << 1) + 1];
		simdMM4(c0, a, b0); simdMM4(c1, a, b1);
	}

	//when K % STEP != 0-------------------------------------
	tx <<= 3;
	for (int ik = 0, RK = K & (STEP - 1); ik < RK; ik++)
	{
		float a = get(A, ty, ik, K);

		float4 b0, b1;
		b0.x = get(B, tx, ik, K);
		b0.y = get(B, tx + 1, ik, K);
		b0.z = get(B, tx + 2, ik, K);
		b0.w = get(B, tx + 3, ik, K);
		b1.x = get(B, tx + 4, ik, K);
		b1.y = get(B, tx + 5, ik, K);
		b1.z = get(B, tx + 6, ik, K);
		b1.w = get(B, tx + 7, ik, K);

		simdMM4(c1, a, b1); simdMM4(c0, a, b0);
	}
	//when K % STEP != 0-------------------------------------

	*(float4*)(&get(C, ty, tx, SC)) = c0; *(float4*)(&get(C, ty, tx + 4, SC)) = c1;
}

#endif


//(Y: N, X: M) X*Y <= 1024
#ifndef MATMUL_T2_KERNEL_NAIVE
#define MATMUL_T2_KERNEL_NAIVE

__global__ void kernel_t2_naive(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float*  __restrict__ C,
	int SC, int K)
{
	int y = threadIdx.y + blockIdx.y*blockDim.y;
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	float v = 0;
	for (int k = 0; k < K; k++)
		v += get(A, y, k, K) * get(B, x, k, K);
	get(C, y, x, SC) = v;
}

#endif

#endif