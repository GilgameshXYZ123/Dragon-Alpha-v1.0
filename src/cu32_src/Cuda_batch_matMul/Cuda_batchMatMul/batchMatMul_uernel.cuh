#pragma once

#ifndef BATCH_MATMUL_UERNEL_H
#define BATCH_MATMUL_UERNEL_H

//A[Batch,  N, AK] 
//B[Batch, BK,  M]
//C[Batch,  N,  M]
//(1) M % 4 == 0
//(2) N % 4 != 0
//(3) K = BK: AK % 4 == 0, BK % 4 != 0, AK >= BK, AK = (BK + 3) >> 2 << 2
#ifndef BATCH_MATMUL_UERNEL_CALL
#define BATCH_MATMUL_UERNEL_CALL

//LB = log2(BLOCK_SIZE)
//MOVE_A == 0: A is a 2D tensor[N, K], logically expand A to[Batch, N, K]
//MOVE_B == 0: B is a 2D tensor[K, M], logically expand B to[Barch, K, M]

#define bmm_u88_mk(stream, LB, Yindex, Xindex, A, B, C, Batch, N, CM, BK, AK, GN, GM)\
	bmm_uernel_8_8_MK<LB, (1<<LB>>1), (1<<LB), MOVE_A, MOVE_B>\
		<<< dim3((GM>>LB>>3), (GN>>LB>>3), Batch),\
            dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, N, CM, BK, AK, Yindex, Xindex)

#define bmm_u44_mk(stream, LB, Yindex, Xindex, A, B, C, Batch, N, CM, BK, AK, GN, GM)\
	bmm_uernel_4_4_MK<LB, (1<<LB>>1), (1<<LB), MOVE_A, MOVE_B>\
		<<< dim3((GM>>LB>>2), (GN>>LB>>2), Batch),\
            dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, N, CM, BK, AK, Yindex, Xindex)

#define bmm_u44_mk_p(stream, LB, Yindex, Xindex, A, B, C, Batch, N, CM, BK, AK, GN, GM)\
	bmm_uernel_4_4_MK_padding<LB, (1<<LB>>1), (1<<LB), MOVE_A, MOVE_B>\
		<<< dim3(((GM + (4<<LB)-1)>>LB>>2), ((GN + (4<<LB)-1)>>LB>>2), Batch),\
			 dim3(1<<LB, 1<<LB), 0, stream>>>\
			(A, B, C, N, CM, BK, AK, Yindex, Xindex)

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), K % BLOCK_SIZE == 0
//LB = 4: K % 16 == 0
//LB = 3: K %  8 == 0
#ifndef BATCH_MATMUL_UERNEL_8_8_MK
#define BATCH_MATMUL_UERNEL_8_8_MK

//LB = 4: Size = 1, Time = 1.412 msec, Performace = 1520.88 GFlop/s
//LB = 3: Size = 1, Time = 1.426 msec, Performace = 1505.95 GFlop/s
template<int LB, int STEP, int STEP2, int MOVE_A, int MOVE_B>
__global__ void bmm_uernel_8_8_MK(
	const float* __restrict__ A, //A[Batch, N, AK], Ak is memAlgined
	const float* __restrict__ B, //B[Batch, BK, M], Bk is not memAligned
	float* __restrict__ C, //C[Batch, N, M]
	int N, int CM, int BK, int AK,
	int Yindex, int Xindex)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 As[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Bs[2][2 << LB][(1 << LB) + 1];

	//prepared for A -> Y:N
	const int Y = (((blockIdx.y << LB) + ty) << 3) + Yindex;
	const int Y0 = (Y + ((tx >= STEP) << 2)) * AK;
	const int Y1 = Y0 + AK, Y2 = Y1 + AK, Y3 = Y2 + AK;

	//prepared for B -> X:M
	const int X = (((blockIdx.x << LB) + tx) << 3) + Xindex;
	const int tX = X + ((ty >= STEP) << 2);

	//compute start offset of A, B, C
	int batch = blockIdx.z;
	A += ((batch * N * AK) & (-MOVE_A));//A[batch * MOVE_A]
	B += ((batch * BK * CM) & (-MOVE_B)) + tX;//B[batch * MOVE_B, 0, tX]
	const int C0 = (batch*N + Y)*CM + X;//C[batch, Y, X]

	//load 4 elements from A[batch]
	int Ak = (tx - ((tx >= STEP) << LB >> 1)) << 1;
	float2 a0 = *(float2*)(A + Ak + Y0);
	float2 a1 = *(float2*)(A + Ak + Y1);
	float2 a2 = *(float2*)(A + Ak + Y2);
	float2 a3 = *(float2*)(A + Ak + Y3);
	As[buf][(tx << 1)][ty] = float4{ a0.x, a1.x, a2.x, a3.x };
	As[buf][(tx << 1) + 1][ty] = float4{ a0.y, a1.y, a2.y, a3.y };

	//load 4 elements from B[batch]
	int Bk = (ty - ((ty >= STEP) << LB >> 1)) << 1;
	int boffset0 = Bk * CM, boffset1 = boffset0 + CM;
	Bs[buf][(ty << 1)][tx] = *(float4*)(B + boffset0);
	Bs[buf][(ty << 1) + 1][tx] = *(float4*)(B + boffset1);
	__syncthreads();

	//compute area-----------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (BK >> LB); ok < OK; ok++)
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

		//load 4 elements from A[batch]
		int Ak = (((ok - (tx >= STEP)) << LB >> 1) + tx) << 1;
		float2 a0 = *(float2*)(A + Ak + Y0);
		float2 a1 = *(float2*)(A + Ak + Y1);
		float2 a2 = *(float2*)(A + Ak + Y2);
		float2 a3 = *(float2*)(A + Ak + Y3);
		As[buf][(tx << 1)][ty] = float4{ a0.x, a1.x, a2.x, a3.x };
		As[buf][(tx << 1) + 1][ty] = float4{ a0.y, a1.y, a2.y, a3.y };

		//load 4 elements from B[batch]
		int Bk = (((ok - (ty >= STEP)) << LB >> 1) + ty) << 1;
		int boffset0 = Bk * CM, boffset1 = boffset0 + CM;
		Bs[buf][(ty << 1)][tx] = *(float4*)(B + boffset0);
		Bs[buf][(ty << 1) + 1][tx] = *(float4*)(B + boffset1);
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

	const int C1 = C0 + CM, C2 = C1 + CM, C3 = C2 + CM;
	const int C4 = C3 + CM, C5 = C4 + CM, C6 = C5 + CM, C7 = C6 + CM;

	*(float4*)(C + C0) = v0;  *(float4*)(C + C0 + 4) = v1;
	*(float4*)(C + C1) = v2;  *(float4*)(C + C1 + 4) = v3;
	*(float4*)(C + C2) = v4;  *(float4*)(C + C2 + 4) = v5;
	*(float4*)(C + C3) = v6;  *(float4*)(C + C3 + 4) = v7;
	*(float4*)(C + C4) = v8;  *(float4*)(C + C4 + 4) = v9;
	*(float4*)(C + C5) = v10; *(float4*)(C + C5 + 4) = v11;
	*(float4*)(C + C6) = v12; *(float4*)(C + C6 + 4) = v13;
	*(float4*)(C + C7) = v14; *(float4*)(C + C7 + 4) = v15;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*4), K % BLOCK_SIZE == 0
//LB = 4: K % 16 == 0
//LB = 3: K %  8 == 0
#ifndef BATCH_MATMUL_UERNEL_4_4_MK
#define BATCH_MATMUL_UERNEL_4_4_MK

//LB = 4: Size = 1, Time = 1.998 msec, Performace = 1074.82 GFlop/s
//LB = 3: Size = 1, Time = 2.046 msec, Performace = 1049.6  GFlop/s
template<int LB, int STEP, int STEP2, int MOVE_A, int MOVE_B>
__global__ void bmm_uernel_4_4_MK(
	const float* __restrict__ A, //A[Batch, N, AK], Ak is memAlgined
	const float* __restrict__ B, //B[Batch, BK, M], Bk is not memAligned
	float* __restrict__ C, //C[Batch, N, M]
	int N, int CM, int BK, int AK,
	int Yindex, int Xindex)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 As[2][1 << LB][(2 << LB) + 2];
	__shared__ float2 Bs[2][1 << LB][(2 << LB) + 2];

	//prepared for A -> Y:N
	const int Y = (((blockIdx.y << LB) + ty) << 2) + Yindex;
	const int Y0 = (Y + ((tx & 1) << 1))*AK, Y1 = Y0 + AK;

	//prepared for B -> X:M
	const int X = (((blockIdx.x << LB) + tx) << 2) + Xindex;
	const int tX = X + ((ty & 1) << 1);

	//compute start offset of A, B, C
	const int batch = blockIdx.z;
	A += (batch * N * AK * MOVE_A);//A[batch * MOVE_A]
	B += (batch * BK * CM * MOVE_B) + tX;//B[batch * MOVE_B, 0, tX]
	const int C0 = (batch*N + Y)*CM + X;//C[batch, Y, X]

	const int As_x = (tx >> 1) << 1, As_y = (ty << 1) + (tx & 1);
	const int Bs_y = (ty >> 1) << 1, Bs_x = (tx << 1) + (ty & 1);
	
	//load 2 elements from A[batch]
	int Ak = (tx >> 1) << 1;
	float2 a0 = *(float2*)(A + Ak + Y0);
	float2 a1 = *(float2*)(A + Ak + Y1);
	As[buf][As_x][As_y] = float2{ a0.x, a1.x };
	As[buf][As_x + 1][As_y] = float2{ a0.y, a1.y };

	//load 2 elements from B[btch]
	int Bk = (ty >> 1) << 1;
	int boffset0 = Bk * CM, boffset1 = boffset0 + CM;
	Bs[buf][Bs_y][Bs_x] = *(float2*)(B + boffset0);
	Bs[buf][Bs_y + 1][Bs_x] = *(float2*)(B + boffset1);
	__syncthreads();

	//compute area---------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0);
	float4 v1 = make_float4(0, 0, 0, 0);
	float4 v2 = make_float4(0, 0, 0, 0);
	float4 v3 = make_float4(0, 0, 0, 0);
	for (int ok = 1; ok < (BK >> LB); ok++)
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
		float2 a0 = *(float2*)(A + Ak + Y0);
		float2 a1 = *(float2*)(A + Ak + Y1);
		As[buf][As_x][As_y] = float2{ a0.x, a1.x };
		As[buf][As_x + 1][As_y] = float2{ a0.y, a1.y };

		//load 2 elements from B[batch]
		int Bk = (((ok << LB) + ty) >> 1) << 1;
		int boffset0 = Bk * CM, boffset1 = boffset0 + CM;
		Bs[buf][Bs_y][Bs_x] = *(float2*)(B + boffset0);
		Bs[buf][Bs_y + 1][Bs_x] = *(float2*)(B + boffset1);
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


//(Y: BLOCK_SIZE * 4, X: BLOCK_SIZE * 4),  K % BLOCK_SIZE == 0
//LB = 4: K % 16 == 0
//LB = 3: K %  8 == 20
#ifndef BATCH_MATMUL_UERNEL_4_4_MK_PADDING
#define BATCH_MATMUL_UERNEL_4_4_MK_PADDING

//LB = 4: Size = 1, Time = 2.112 msec, Performace = 1016.8 GFlop/s
//LB = 3: Size = 1, Time = 2.102 msec, Performace = 1021.64 GFlop/s
template<int LB, int STEP, int STEP2, int MOVE_A, int MOVE_B>
__global__ void bmm_uernel_4_4_MK_padding(
	const float* __restrict__ A, //A[Batch, N, K]
	const float* __restrict__ B, //B[Batch, K, M]
	      float* __restrict__ C, //C[Batch, N, M]
	int N, int CM, int BK, int AK,
	int Yindex, int Xindex)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 As[2][1 << LB][(2 << LB) + 2];
	__shared__ float2 Bs[2][1 << LB][(2 << LB) + 2];

	//prepared for A -> Y:N
	const int Y = (((blockIdx.y << LB) + ty) << 2) + Yindex;
	const int Y0 = (Y + ((tx & 1) << 1))*AK, Y1 = Y0 + AK;

	//prepared for B -> X:M
	const int X = (((blockIdx.x << LB) + tx) << 2) + Xindex;
	const int X0 = X + ((ty & 1) << 1);

	//compute start offset of A, B, C
	const int batch = blockIdx.z;
	A += (batch * N * AK * MOVE_A);//A[batch * MOVE_A]
	B += (batch * BK * CM * MOVE_B) + X0;//B[batch * MOVE_B, 0, tX]
	C += (batch * N * CM);//C[batch]
	
	const int As_x = (tx >> 1) << 1, As_y = (ty << 1) + (tx & 1);
	const int Bs_y = (ty >> 1) << 1, Bs_x = (tx << 1) + (ty & 1);
	const int ye = (N - 1) * AK;
	const int xe = (CM - 2);//float2: X <= M - 2
	
	//load 2 elements from A[batch]
	int Ak = (tx >> 1) << 1;
	float2 a0 = (Y0 <= ye ? *(float2*)(A + Y0 + Ak) : F32_2_0);
	float2 a1 = (Y1 <= ye ? *(float2*)(A + Y1 + Ak) : F32_2_0);
	As[buf][As_x][As_y] = float2{ a0.x, a1.x };
	As[buf][As_x + 1][As_y] = float2{ a0.y, a1.y };

	//load 2 elements from B[btch]
	int Bk = (ty >> 1) << 1;
	int boffset0 = Bk * CM, boffset1 = boffset0 + CM;
	Bs[buf][Bs_y][Bs_x] = (X0 <= xe ? *(float2*)(B + boffset0) : F32_2_0);
	Bs[buf][Bs_y + 1][Bs_x] = (X0 <= xe ? *(float2*)(B + boffset1) : F32_2_0);
	__syncthreads();

	//compute area---------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0);
	float4 v1 = make_float4(0, 0, 0, 0);
	float4 v2 = make_float4(0, 0, 0, 0);
	float4 v3 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (BK >> LB); ok < OK; ok++)
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
		int boffset0 = Bk * CM, boffset1 = boffset0 + CM;
		Bs[buf][Bs_y][Bs_x] = (X0 <= xe ? *(float2*)(B + boffset0) : F32_2_0);
		Bs[buf][Bs_y + 1][Bs_x] = (X0 <= xe ? *(float2*)(B + boffset1) : F32_2_0);
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

	const int C0 = Y*CM + X;//C[batch, Y, X]
	const int C1 = C0 + CM;
	const int C2 = C1 + CM;
	const int C3 = C2 + CM;

	//if X <= M - 4, (Y * M) + X <= Cend, we have: Y <= N -1
	int ce = (N * CM) - 4;//float4: (N - 1)*M + (M - 4) = N*M - 4
	if (X <= (CM - 4)) {//float4: X <= M - 4
		if (C0 <= ce) { *(float4*)(C + C0) = v0; }
		if (C1 <= ce) { *(float4*)(C + C1) = v1; }
		if (C2 <= ce) { *(float4*)(C + C2) = v2; }
		if (C3 <= ce) { *(float4*)(C + C3) = v3; }
	}
}

#endif

#endif