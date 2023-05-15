#pragma once

#ifndef BATCH_MATMUL_T2_UERNEL_H
#define BATCH_MATMUL_T2_UERNEL_H

//A[Batch,  N,  K] 
//B[Batch, BM,  K] logically-> B^T[Batch, K, M] 
//C[Batch,  N, CM]
//(1) K % 4 == 0
//(2) M = CM: BM % 4 != 0, CM % 4 == 0, CM >= BM, CM = (BM + 3) >> 2 << 2
//(3) N % 4 != 0
#ifndef BATCH_MATMUL_T2_UERNEL_CALL
#define BATCH_MATMUL_T2_UERNEL_CALL

//LB = log2(BLOCK_SIZE)
//MOVE_A == 0: A is a 2D tensor[N, K], logically expand A to[Batch, N, K]
//MOVE_B == 0: B is a 2D tensor[M, K], logically expand B to[Barch, M, K]

#define bmmT2_u88(stream, LB, Yindex, Xindex, A, B, C, Batch, N, CM, BM, K, GN, GM)\
	bmmT2_uernel_8_8<LB, (1<<LB>>1), (1<<LB), MOVE_A, MOVE_B>\
		<<< dim3((GM>>LB>>3), (GN>>LB>>3), Batch),\
            dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, N, CM, BM, K, Yindex, Xindex)

#define bmmT2_u44(stream, LB, Yindex, Xindex, A, B, C, Batch, N, CM, BM, K, GN, GM)\
	bmmT2_uernel_4_4<LB, (1<<LB>>1), (1<<LB), MOVE_A, MOVE_B>\
		<<< dim3((GM>>LB>>2), (GN>>LB>>2), Batch),\
            dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, N, CM, BM, K, Yindex, Xindex)

#define bmmT2_u44_p(stream, LB, Yindex, Xindex, A, B, C, Batch, N, CM, BM, K, GN, GM)\
	bmmT2_uernel_4_4_padding<LB, (1<<LB>>1), (1<<LB), MOVE_A, MOVE_B>\
		<<< dim3(((GM + (4<<LB)-1)>>LB>>2), ((GN + (4<<LB)-1)>>LB>>2), Batch),\
            dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, N, CM, BM, K, Yindex, Xindex)

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), K % BLOCK_SIZE == 0, BM % 4 == 0
//LB = 4: K % 16 == 0
//LB = 3: K %  8 == 0
#ifndef BATCH_MATMUL_T2_UERNEL_8_8
#define BATCH_MATMUL_T2_UERNEL_8_8

//LB = 4: Size = 1, Time = 1.462 msec, Performace = 1468.87 GFlop/s
//LB = 3: Size = 1, Time = 1.518 msec, Performace = 1414.68 GFlop/s
template<int LB, int STEP, int STEP2, int MOVE_A, int MOVE_B>
__global__ void bmmT2_uernel_8_8(
	const float* __restrict__ A, //A[Batch, N,  K]
	const float* __restrict__ B, //B[Batch, BM, K], BM is not memAligned
	float* __restrict__ C, //C[Batch, N, CM], CM is memAligned
	int N, int CM, int BM, int K,
	int Yindex, int Xindex)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 As[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Bs[2][2 << LB][(1 << LB) + 1];

	//prepared for A -> Y:N
	const int Y = (((blockIdx.y << LB) + ty) << 3) + Yindex;
	const int Y0 = (Y + ((tx >= STEP) << 2)) * K;
	const int Y1 = Y0 + K, Y2 = Y1 + K, Y3 = Y2 + K;

	//prepared for B -> X:M
	const int X = (((blockIdx.x << LB) + tx) << 3) + Xindex;
	const int X0 = (X + ((ty >= STEP) << 2)) * K;
	const int X1 = X0 + K, X2 = X1 + K, X3 = X2 + K;

	//compute start offset of A, B, C
	const int batch = blockIdx.z;
	A += (batch *  N * K * MOVE_A);//A[batch * MOVE_A]
	B += (batch * BM * K * MOVE_B);//B[batch * MOVE_B]
	const int C0 = (batch*N + Y)*CM + X;//C[batch, Y, X]
	const int xe = (BM - 1)*K;

	//load 4 elements from B[batch]
	int Bk = (ty - ((ty >= STEP) << LB >> 1)) << 1;
	float2 bv0 = *(float2*)(B + X0 + Bk);
	float2 bv1 = *(float2*)(B + X1 + Bk);
	float2 bv2 = *(float2*)(B + X2 + Bk);
	float2 bv3 = *(float2*)(B + X3 + Bk);
	Bs[buf][(ty << 1)][tx] = float4{ bv0.x, bv1.x, bv2.x, bv3.x };
	Bs[buf][(ty << 1) + 1][tx] = float4{ bv0.y, bv1.y, bv2.y, bv3.y };

	//load 4 elements from A[batch]
	int Ak = (tx - ((tx >= STEP) << LB >> 1)) << 1;
	float2 av0 = *(float2*)(A + Y0 + Ak);
	float2 av1 = *(float2*)(A + Y1 + Ak);
	float2 av2 = *(float2*)(A + Y2 + Ak);
	float2 av3 = *(float2*)(A + Y3 + Ak);
	As[buf][(tx << 1)][ty] = float4{ av0.x, av1.x, av2.x, av3.x };
	As[buf][(tx << 1) + 1][ty] = float4{ av0.y, av1.y, av2.y, av3.y };
	__syncthreads();

	//compute area---------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (K >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++)
		{
			float4 a0 = As[buf][ik][ty], a1 = As[buf][ik + STEP2][ty];
			float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP2][tx];

			simdMM4(v0, a0.x, b0);  simdMM4(v1, a0.x, b1);
			simdMM4(v2, a0.y, b0);  simdMM4(v3, a0.y, b1);
			simdMM4(v4, a0.z, b0);  simdMM4(v5, a0.z, b1);
			simdMM4(v6, a0.w, b0);  simdMM4(v7, a0.w, b1);
			simdMM4(v8, a1.x, b0);  simdMM4(v9, a1.x, b1);
			simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
			simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
			simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
		}
		buf ^= 1;

		//load 4 elements from B[batch]
		int Bk = (((ok - (ty >= STEP)) << LB >> 1) + ty) << 1;
		float2 bv0 = *(float2*)(B + X0 + Bk);
		float2 bv1 = *(float2*)(B + X1 + Bk);
		float2 bv2 = *(float2*)(B + X2 + Bk);
		float2 bv3 = *(float2*)(B + X3 + Bk);
		Bs[buf][(ty << 1)][tx] = float4{ bv0.x, bv1.x, bv2.x, bv3.x };
		Bs[buf][(ty << 1) + 1][tx] = float4{ bv0.y, bv1.y, bv2.y, bv3.y };

		//load 4 elements from A[batch]
		int Ak = (((ok - (tx >= STEP)) << LB >> 1) + tx) << 1;
		float2 av0 = *(float2*)(A + Y0 + Ak);
		float2 av1 = *(float2*)(A + Y1 + Ak);
		float2 av2 = *(float2*)(A + Y2 + Ak);
		float2 av3 = *(float2*)(A + Y3 + Ak);
		As[buf][(tx << 1)][ty] = float4{ av0.x, av1.x, av2.x, av3.x };
		As[buf][(tx << 1) + 1][ty] = float4{ av0.y, av1.y, av2.y, av3.y };
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++)
	{
		float4 a0 = As[buf][ik][ty], a1 = As[buf][ik + STEP2][ty];
		float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP2][tx];

		simdMM4(v0, a0.x, b0);  simdMM4(v1, a0.x, b1);
		simdMM4(v2, a0.y, b0);  simdMM4(v3, a0.y, b1);
		simdMM4(v4, a0.z, b0);  simdMM4(v5, a0.z, b1);
		simdMM4(v6, a0.w, b0);  simdMM4(v7, a0.w, b1);
		simdMM4(v8, a1.x, b0);  simdMM4(v9, a1.x, b1);
		simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
		simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
		simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
	}

	const int C1 = C0 + CM, C2 = C1 + CM;
	const int C3 = C2 + CM, C4 = C3 + CM;
	const int C5 = C4 + CM, C6 = C5 + CM, C7 = C6 + CM;

	*(float4*)(C + C0) = v0; *(float4*)(C + C0 + 4) = v1;
	*(float4*)(C + C1) = v2; *(float4*)(C + C1 + 4) = v3;
	*(float4*)(C + C2) = v4; *(float4*)(C + C2 + 4) = v5;
	*(float4*)(C + C3) = v6; *(float4*)(C + C3 + 4) = v7;
	*(float4*)(C + C4) = v8; *(float4*)(C + C4 + 4) = v9;
	*(float4*)(C + C5) = v10; *(float4*)(C + C5 + 4) = v11;
	*(float4*)(C + C6) = v12; *(float4*)(C + C6 + 4) = v13;
	*(float4*)(C + C7) = v14; *(float4*)(C + C7 + 4) = v15;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*4), K % BLOCK_SIZE == 0
//LB = 4: K % 16 == 0
//LB = 3: K %  8 == 0
#ifndef BATCH_MATMUL_T2_UERNEL_4_4
#define BATCH_MATMUL_T2_UERNEL_4_4

//LB = 4: Size = 1, Time = 2.072 msec, Performace = 1036.43 GFlop/s
//LB = 3: Size = 1, Time = 2.138 msec, Performace = 1004.44 GFlop/s
template<int LB, int STEP, int STEP2, int MOVE_A, int MOVE_B>
__global__ void bmmT2_uernel_4_4(
	const float* __restrict__ A, //A[Batch, N,  K]
	const float* __restrict__ B, //B[Batch, BM, K], BM is not memAligned
	float* __restrict__ C, //C[Batch, N, CM], CM is memAligned
	int N, int CM, int BM, int K,
	int Yindex, int Xindex)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 As[2][1 << LB][(2 << LB) + 2];
	__shared__ float2 Bs[2][1 << LB][(2 << LB) + 2];

	//prepared for A -> Y:N
	const int Y = (((blockIdx.y << LB) + ty) << 2) + Yindex;
	const int Y0 = (Y + ((tx & 1) << 1)) * K, Y1 = Y0 + K;

	//prepared for B -> X:M
	const int X = (((blockIdx.x << LB) + tx) << 2) + Xindex;
	const int X0 = (X + ((ty & 1) << 1)) * K, X1 = X0 + K;

	//compute start offset of A, B, C
	const int batch = blockIdx.z;
	A += (batch * N *  K * MOVE_A);//A[batch * MOVE_A]
	B += (batch * K * BM * MOVE_B);//B[batch * MOVE_B]
	const int C0 = (Y + batch * N)*CM + X;//C[batch, Y, X]

	const int As_x = (tx >> 1) << 1, As_y = (ty << 1) + (tx & 1);
	const int Bs_y = (ty >> 1) << 1, Bs_x = (tx << 1) + (ty & 1);
	const int xe = (BM - 1) * K;

	//load 2 elements from A[batch]
	int Ak = (tx >> 1) << 1;
	float2 a0 = *(float2*)(A + Y0 + Ak);
	float2 a1 = *(float2*)(A + Y1 + Ak);
	As[buf][As_x][As_y] = float2{ a0.x, a1.x };
	As[buf][As_x + 1][As_y] = float2{ a0.y, a1.y };

	//load 2 elements from B[batch]
	int Bk = (ty >> 1) << 1;
	float2 b0 = (X0 <= xe ? *(float2*)(B + X0 + Bk) : F32_2_0);
	float2 b1 = (X1 <= xe ? *(float2*)(B + X1 + Bk) : F32_2_0);
	Bs[buf][Bs_y][Bs_x] = float2{ b0.x, b1.x };
	Bs[buf][Bs_y + 1][Bs_x] = float2{ b0.y, b1.y };
	__syncthreads();

	//compute area---------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0);
	float4 v1 = make_float4(0, 0, 0, 0);
	float4 v2 = make_float4(0, 0, 0, 0);
	float4 v3 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (K >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++)
		{
			float4 b = *(float4*)(&Bs[buf][ik][tx << 1]);
			float4 a = *(float4*)(&As[buf][ik][ty << 1]);

			simdMM4(v0, a.x, b);
			simdMM4(v1, a.y, b);
			simdMM4(v2, a.z, b);
			simdMM4(v3, a.w, b);
		}
		buf ^= 1;

		//load 2 elements from A[batch]
		int Ak = (((ok << LB) + tx) >> 1) << 1;
		float2 a0 = *(float2*)(A + Y0 + Ak);
		float2 a1 = *(float2*)(A + Y1 + Ak);
		As[buf][As_x][As_y] = float2{ a0.x, a1.x };
		As[buf][As_x + 1][As_y] = float2{ a0.y, a1.y };

		//load 2 elements from B[batch]
		int Bk = (((ok << LB) + ty) >> 1) << 1;
		float2 b0 = (X0 <= xe ? *(float2*)(B + X0 + Bk) : F32_2_0);
		float2 b1 = (X1 <= xe ? *(float2*)(B + X1 + Bk) : F32_2_0);
		Bs[buf][Bs_y][Bs_x] = float2{ b0.x, b1.x };
		Bs[buf][Bs_y + 1][Bs_x] = float2{ b0.y, b1.y };
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++)
	{
		float4 b = *(float4*)(&Bs[buf][ik][tx << 1]);
		float4 a = *(float4*)(&As[buf][ik][ty << 1]);

		simdMM4(v0, a.x, b);
		simdMM4(v1, a.y, b);
		simdMM4(v2, a.z, b);
		simdMM4(v3, a.w, b);
	}

	const int C1 = C0 + CM;
	const int C2 = C1 + CM;
	const int C3 = C2 + CM;

	*(float4*)(C + C0) = v0;
	*(float4*)(C + C1) = v1;
	*(float4*)(C + C2) = v2;
	*(float4*)(C + C3) = v3;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*4), K % BLOCK_SIZE == 0
//LB = 4: K % 16 == 0
//LB = 3: K %  8 == 0
#ifndef BATCH_MATMUL_T2_UERNEL_4_4_PADDING
#define BATCH_MATMUL_T2_UERNEL_4_4_PADDING

//LB = 4: Size = 1, Time = 2.08 msec, Performace = 1032.44  GFlop/s
//LB = 3: Size = 1, Time = 2.16 msec, Performace =  994.205 GFlop/s
template<int LB, int STEP, int STEP2, int MOVE_A, int MOVE_B>
__global__ void bmmT2_uernel_4_4_padding(
	const float* __restrict__ A, //A[Batch,  N, K]
	const float* __restrict__ B, //B[Batch, BM, K], BM is not memAligned
	float* __restrict__ C, //C[Batch, N, CM], CM is memAligned
	int N, int CM, int BM, int K,
	int Yindex, int Xindex)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 As[2][1 << LB][(2 << LB) + 2];
	__shared__ float2 Bs[2][1 << LB][(2 << LB) + 2];

	//prepared for A -> Y:N
	const int Y = (((blockIdx.y << LB) + ty) << 2) + Yindex;
	const int Y0 = (Y + ((tx & 1) << 1)) * K, Y1 = Y0 + K;

	//prepared for B -> X:M
	const int X = (((blockIdx.x << LB) + tx) << 2) + Xindex;
	const int X0 = (X + ((ty & 1) << 1)) * K, X1 = X0 + K;

	//compute start offset of A, B, C
	const int batch = blockIdx.z;
	A += (batch * N *  K * MOVE_A);//A[batch * MOVE_A]
	B += (batch * K * BM * MOVE_B);//B[batch * MOVE_B]
	C += (batch * N * CM);//C[batch]

	const int As_x = (tx >> 1) << 1, As_y = (ty << 1) + (tx & 1);
	const int Bs_y = (ty >> 1) << 1, Bs_x = (tx << 1) + (ty & 1);
	const int ye = (N - 1) * K;
	const int xe = (BM - 1) * K;

	//load 2 elements from A[batch]
	int Ak = (tx >> 1) << 1;
	float2 a0 = (Y0 <= ye ? *(float2*)(A + Y0 + Ak) : F32_2_0);
	float2 a1 = (Y1 <= ye ? *(float2*)(A + Y1 + Ak) : F32_2_0);
	As[buf][As_x][As_y] = float2{ a0.x, a1.x };
	As[buf][As_x + 1][As_y] = float2{ a0.y, a1.y };

	//load 2 elements from B[batch]
	int Bk = (ty >> 1) << 1;
	float2 b0 = (X0 <= xe ? *(float2*)(B + X0 + Bk) : F32_2_0);
	float2 b1 = (X1 <= xe ? *(float2*)(B + X1 + Bk) : F32_2_0);
	Bs[buf][Bs_y][Bs_x] = float2{ b0.x, b1.x };
	Bs[buf][Bs_y + 1][Bs_x] = float2{ b0.y, b1.y };
	__syncthreads();

	//compute area---------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0);
	float4 v1 = make_float4(0, 0, 0, 0);
	float4 v2 = make_float4(0, 0, 0, 0);
	float4 v3 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (K >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++)
		{
			float4 b = *(float4*)(&Bs[buf][ik][tx << 1]);
			float4 a = *(float4*)(&As[buf][ik][ty << 1]);

			simdMM4(v0, a.x, b);
			simdMM4(v1, a.y, b);
			simdMM4(v2, a.z, b);
			simdMM4(v3, a.w, b);
		}
		buf ^= 1;

		//load 2 elements from A[batch]
		int Ak = (((ok << LB) + tx) >> 1) << 1;
		float2 a0 = (Y0 <= ye ? *(float2*)(A + Y0 + Ak) : F32_2_0);
		float2 a1 = (Y1 <= ye ? *(float2*)(A + Y1 + Ak) : F32_2_0);
		As[buf][As_x][As_y] = float2{ a0.x, a1.x };
		As[buf][As_x + 1][As_y] = float2{ a0.y, a1.y };

		//load 2 elements from B[batch]
		int Bk = (((ok << LB) + ty) >> 1) << 1;
		float2 b0 = (X0 <= xe ? *(float2*)(B + X0 + Bk) : F32_2_0);
		float2 b1 = (X1 <= xe ? *(float2*)(B + X1 + Bk) : F32_2_0);
		Bs[buf][Bs_y][Bs_x] = float2{ b0.x, b1.x };
		Bs[buf][Bs_y + 1][Bs_x] = float2{ b0.y, b1.y };
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++)
	{
		float4 b = *(float4*)(&Bs[buf][ik][tx << 1]);
		float4 a = *(float4*)(&As[buf][ik][ty << 1]);

		simdMM4(v0, a.x, b);
		simdMM4(v1, a.y, b);
		simdMM4(v2, a.z, b);
		simdMM4(v3, a.w, b);
	}

	const int C0 = (Y * CM) + X;
	const int C1 = C0 + CM, C2 = C1 + CM, C3 = C2 + CM;
	const int ce = (N * CM) - 4;//float4: (N - 1)*M + (M - 4) = N*M - 4

	if (X <= (CM - 4)) {//float4: X <= CM - 4
		if (C0 <= ce) { *(float4*)(C + C0) = v0; }
		if (C1 <= ce) { *(float4*)(C + C1) = v1; }
		if (C2 <= ce) { *(float4*)(C + C2) = v2; }
		if (C3 <= ce) { *(float4*)(C + C3) = v3; }
	}
}

#endif

#endif