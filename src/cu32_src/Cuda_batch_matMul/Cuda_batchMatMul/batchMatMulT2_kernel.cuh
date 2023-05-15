#pragma once

#ifndef BATCH_MATMUL_T2_KERNEL_H
#define BATCH_MATMUL_T2_KERNEL_H

//A[Batch,  N,  K] 
//B[Batch, BM,  K] logically-> B^T[Batch, K, M] 
//C[Batch,  N, CM]
//(1) K % 4 == 0
//(2) M = CM: BM % 4 != 0, CM % 4 == 0, CM >= BM, CM = (BM + 3) >> 2 << 2
//(3) N % 4 != 0
#ifndef BATCH_MATMUL_T2_KERNEL_CALL
#define BATCH_MATMUL_T2_KERNEL_CALL

//LB = log2(BLOCK_SIZE)
//MOVE_A == 0: A is a 2D tensor[N, K], logically expand A to[Batch, N, K]
//MOVE_B == 0: B is a 2D tensor[M, K], logically expand B to[Barch, M, K]

//======[Common]==================================================
#define bmmT2_k88(stream, LB, Yindex, Xindex, A, B, C, Batch, N, CM, BM, K, GN, GM)\
	bmmT2_kernel_8_8<LB, (1<<LB>>1), MOVE_A, MOVE_B>\
		<<< dim3((GM>>LB>>3), (GN>>LB>>3), Batch),\
            dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, N, CM, BM, K, Yindex, Xindex)

#define bmmT2_k44(stream, LB, Yindex, Xindex, A, B, C, Batch, N, CM, BM, K, GN, GM)\
	bmmT2_kernel_4_4<LB, (1<<LB>>1), MOVE_A, MOVE_B>\
		<<< dim3((GM>>LB>>2), (GN>>LB>>2), Batch),\
            dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, N, CM, BM, K, Yindex, Xindex)

#define bmmT2_k82(stream, LB, Yindex, Xindex, A, B, C, Batch, N, CM, BM, K, GN, GM)\
	bmmT2_kernel_8_2<LB, (1<<LB>>1), MOVE_A, MOVE_B>\
		<<< dim3((GM>>LB>>1), (GN>>LB>>3), Batch),\
	        dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, N, CM, BM, K, Yindex, Xindex)

#define bmmT2_k28(stream, LB, Yindex, Xindex, A, B, C, Batch, N, CM, BM, K, GN, GM)\
	bmmT2_kernel_2_8<LB, (1<<LB>>1), MOVE_A, MOVE_B>\
		<<< dim3((GM>>LB>>3), (GN>>LB>>1), Batch),\
            dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, N, CM, BM, K, Yindex, Xindex)

#define bmmT2_k42(stream, LB, Yindex, Xindex, A, B, C, Batch, N, CM, BM, K, GN, GM)\
	bmmT2_kernel_4_2<LB, (1<<LB>>1), MOVE_A, MOVE_B>\
		<<< dim3((GM>>LB>>1), (GN>>LB>>2), Batch),\
	        dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, N, CM, BM, K, Yindex, Xindex)

#define bmmT2_k24(stream, LB, Yindex, Xindex, A, B, C, Batch, N, CM, BM, K, GN, GM)\
	bmmT2_kernel_2_4<LB, (1<<LB>>1), MOVE_A, MOVE_B>\
		<<< dim3((GM>>LB>>2), (GN>>LB>>1), Batch),\
            dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, N, CM, BM, K, Yindex, Xindex)

//======[Small]===================================================
#define bmmT2_k22(stream, LB, Yindex, Xindex, A, B, C, Batch, N, CM, BM, K, GN, GM)\
	bmmT2_kernel_2_2<LB, (1<<LB), MOVE_A, MOVE_B>\
		<<< dim3((GM>>LB>>1), (GN>>LB>>1), Batch),\
            dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, N, CM, BM, K, Yindex, Xindex)

#define bmmT2_k41(stream, LB, Yindex, Xindex, A, B, C, Batch, N, CM, BM, K, GN, GM)\
	bmmT2_kernel_4_1<LB, (1<<LB), MOVE_A, MOVE_B>\
		<<< dim3((GM>>LB), (GN>>LB>>2), Batch),\
            dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, N, CM, BM, K, Yindex, Xindex)

#define bmmT2_k14(stream, LB, Yindex, Xindex, A, B, C, Batch, N, CM, BM, K, GN, GM)\
	bmmT2_kernel_1_4<LB, (1<<LB), MOVE_A, MOVE_B>\
		<<< dim3((GM>>LB>>2), (GN>>LB), Batch),\
            dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, N, CM, BM, K, Yindex, Xindex)

#endif


//======[Common]==================================================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), K % (BLOCK_SIZE/2) == 0
//LB = 4: K % 8 == 0
//LB = 3: K % 4 == 0
#ifndef BATCH_MATMUL_T2_KERNEL_8_8
#define BATCH_MATMUL_T2_KERNEL_8_8

//LB = 4: Size = 1, Time = 1.582 msec, Performace = 1357.45 GFlop/s
//LB = 3: Size = 1, Time = 1.816 msec, Performace = 1182.53 GFlop/s
template<int LB, int STEP, int MOVE_A, int MOVE_B>
__global__ void bmmT2_kernel_8_8(
	const float* __restrict__ A, //A[Batch, N,  K]
	const float* __restrict__ B, //B[Batch, BM, K], BM is not memAligned
	      float* __restrict__ C, //C[Batch, N, CM], CM is memAligned
	int N, int CM, int BM, int K,
	int Yindex, int Xindex)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 As[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Bs[2][1 << LB][(1 << LB) + 1];

	//prepared for A -> Y:N
	const int Y = (((blockIdx.y << LB) + ty) << 3) + Yindex;
	const int Y0 = (Y + ((tx >= STEP) << 2)) * K;
	const int Y1 = Y0 + K, Y2 = Y1 + K, Y3 = Y2 + K;

	//prepared for B -> X:M
	const int X = (((blockIdx.x << LB) + tx) << 3) + Xindex;
	const int X0 = (X + ((ty >= STEP) << 2))*K;
	const int X1 = X0 + K, X2 = X1 + K, X3 = X2 + K;

	//compute start offset of A, B, C
	const int batch = blockIdx.z;
	A += (batch *  N * K * MOVE_A);//A[batch * MOVE_A]
	B += (batch * BM * K * MOVE_B);//B[batch * MOVE_B]
	const int C0 = (batch*N + Y)*CM + X;//C[batch, Y, X]
	const int xe = (BM - 1)*K;

	//load 4 elements from A[batch]
	float4 av; int Ak = tx - ((tx >= STEP) << LB >> 1);
	av.x = A[Y0 + Ak];
	av.y = A[Y1 + Ak];
	av.z = A[Y2 + Ak];
	av.w = A[Y3 + Ak];
	As[buf][tx][ty] = av;
	
	float4 bv; int Bk = ty - ((ty >= STEP) << LB >> 1);//load 4 elements from B[batch]
	bv.x = (X0 <= xe ? B[X0 + Bk] : 0);
	bv.y = (X1 <= xe ? B[X1 + Bk] : 0);
	bv.z = (X2 <= xe ? B[X2 + Bk] : 0);
	bv.w = (X3 <= xe ? B[X3 + Bk] : 0);
	Bs[buf][ty][tx] = bv;
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

	for (int ok = 1, OK = (K << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) 
		{
			float4 a0 = As[buf][ik][ty], a1 = As[buf][ik + STEP][ty];
			float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP][tx];

			simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
			simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
			simdMM4(v4, a0.z, b0); simdMM4(v5, a0.z, b1);
			simdMM4(v6, a0.w, b0); simdMM4(v7, a0.w, b1);
			simdMM4(v8, a1.x, b0); simdMM4(v9, a1.x, b1);
			simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
			simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
			simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
		}
		buf ^= 1;

		//load 4 elements from A[batch]
		float4 av; int Ak = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		av.x = A[Y0 + Ak];
		av.y = A[Y1 + Ak];
		av.z = A[Y2 + Ak];
		av.w = A[Y3 + Ak];
		As[buf][tx][ty] = av;

		//load 4 elements from B[batch]
		float4 bv; int Bk = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		bv.x = (X0 <= xe ? B[X0 + Bk] : 0);
		bv.y = (X1 <= xe ? B[X1 + Bk] : 0);
		bv.z = (X2 <= xe ? B[X2 + Bk] : 0);
		bv.w = (X3 <= xe ? B[X3 + Bk] : 0);
		Bs[buf][ty][tx] = bv;
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) 
	{
		float4 a0 = As[buf][ik][ty], a1 = As[buf][ik + STEP][ty];
		float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP][tx];

		simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
		simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
		simdMM4(v4, a0.z, b0); simdMM4(v5, a0.z, b1);
		simdMM4(v6, a0.w, b0); simdMM4(v7, a0.w, b1);
		simdMM4(v8, a1.x, b0); simdMM4(v9, a1.x, b1);
		simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
		simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
		simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
	}

	const int C1 = C0 + CM, C2 = C1 + CM;
	const int C3 = C2 + CM, C4 = C3 + CM;
	const int C5 = C4 + CM, C6 = C5 + CM, C7 = C6 + CM;

	*(float4*)(C + C0) =  v0; *(float4*)(C + C0 + 4) =  v1;
	*(float4*)(C + C1) =  v2; *(float4*)(C + C1 + 4) =  v3;
	*(float4*)(C + C2) =  v4; *(float4*)(C + C2 + 4) =  v5;
	*(float4*)(C + C3) =  v6; *(float4*)(C + C3 + 4) =  v7;
	*(float4*)(C + C4) =  v8; *(float4*)(C + C4 + 4) =  v9;
	*(float4*)(C + C5) = v10; *(float4*)(C + C5 + 4) = v11;
	*(float4*)(C + C6) = v12; *(float4*)(C + C6 + 4) = v13;
	*(float4*)(C + C7) = v14; *(float4*)(C + C7 + 4) = v15;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*4), K % (BLOCK_SIZE/2) == 0
//LB = 4: K % 8 == 0
#ifndef BATCH_MATMUL_T2_KERNEL_4_4
#define BATCH_MATMUL_T2_KERNEL_4_4

//LB = 4: Size = 1, Time = 2.028 msec, Performace = 1058.92  GFlop/s
//LB = 3: Size = 1, Time = 3.366 msec, Performace =  637.993 GFlop/s
template<int LB, int STEP, int MOVE_A, int MOVE_B>
__global__ void bmmT2_kernel_4_4(
	const float* __restrict__ A, //A[Batch, N,  K]
	const float* __restrict__ B, //B[Batch, BM, K], BM is not memAligned
	      float* __restrict__ C, //C[Batch, N, CM], CM is memAligned
	int N, int CM, int BM, int K,
	int Yindex, int Xindex)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 As[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2 Bs[2][1 << LB >> 1][(2 << LB) + 2];

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

	const int As_x = (tx >> 1), As_y = (ty << 1) + (tx & 1);
	const int Bs_y = (ty >> 1), Bs_x = (tx << 1) + (ty & 1);
	const int xe = (BM - 1) * K;

	//load 2 elements from A[batch]
	float2 av; int Ak = (tx >> 1);
	av.x = A[Y0 + Ak];
	av.y = A[Y1 + Ak];
	As[buf][As_x][As_y] = av;
	
	//load 2 elements from B[batch]
	float2 bv; int Bk = (ty >> 1);
	bv.x = (X0 <= xe ? B[X0 + Bk] : 0);
	bv.y = (X1 <= xe ? B[X1 + Bk] : 0);
	Bs[buf][Bs_y][Bs_x] = bv;
	__syncthreads();

	//compute area---------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0);
	float4 v1 = make_float4(0, 0, 0, 0);
	float4 v2 = make_float4(0, 0, 0, 0);
	float4 v3 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (K << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) 
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
		float2 av; int Ak = ((ok << LB) + tx) >> 1;
		av.x = A[Y0 + Ak];
		av.y = A[Y1 + Ak];
		As[buf][As_x][As_y] = av;

		//load 2 elements from B[batch]
		float2 bv; int Bk = ((ok << LB) + ty) >> 1;
		bv.x = (X0 <= xe ? B[X0 + Bk] : 0);
		bv.y = (X1 <= xe ? B[X1 + Bk] : 0);
		Bs[buf][Bs_y][Bs_x] = bv;
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) 
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


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*2), K % (BLOCK_SIZE/2) == 0
//LB = 4: K % 8 == 0
#ifndef BATCH_MATMUL_T2_KERNEL_8_2
#define BATCH_MATMUL_T2_KERNEL_8_2

//LB = 4: Size = 1, Time = 2.118 msec, Performace = 1013.92  GFlop/s
//LB = 3: Size = 1, Time = 3.642 msec, Performace =  589.644 GFlop/s
//LB = 4: Size = 0.996094, Time = 2.106 msec, Performace = 1015.71  GFlop/s
//LB = 3: Size = 0.996094, Time = 3.46  msec, Performace =  618.236 GFlop/s
template<int LB, int STEP, int MOVE_A, int MOVE_B>
__global__ void bmmT2_kernel_8_2(
	const float* __restrict__ A, //A[Batch,  N,  K]
	const float* __restrict__ B, //B[Batch, BM,  K], BM is not memAligned
	      float* __restrict__ C, //C[Batch,  N, CM], CM is memAligned
	int N, int CM, int BM, int K,
	int Yindex, int Xindex)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 As[2][1 << LB][(1 << LB) + 1];//k88
	__shared__ float  Bs[2][1 << LB >> 1][(2 << LB) + 2];//k42

	//prepared for A -> Y:N
	const int Y = (((blockIdx.y << LB) + ty) << 3) + Yindex;
	const int tY0 = (Y + ((tx >= STEP) << 2)) * K;
	const int tY1 = tY0 + K, tY2 = tY1 + K, tY3 = tY2 + K;

	//prepared for B -> X:M
	const int X = (((blockIdx.x << LB) + tx) << 1) + Xindex;
	const int tX0 = (X + (ty & 1)) * K;

	//compute start offset of A, B, C
	const int batch = blockIdx.z;
	A += (batch * N *  K * MOVE_A);//A[batch * MOVE_A]
	B += (batch * K * BM * MOVE_B);//B[batch * MOVE_B]
	C += (batch * N * CM);//C[batch]

	//load 4 elements from A[batch]
	int Ak = tx - ((tx >= STEP) << LB >> 1);
	As[buf][tx][ty].x = A[tY0 + Ak];
	As[buf][tx][ty].y = A[tY1 + Ak];
	As[buf][tx][ty].z = A[tY2 + Ak];
	As[buf][tx][ty].w = A[tY3 + Ak];

	//load 1 element from B[batch]
	const int Xend = (BM - 1) * K;
	int Bk = (ty >> 1);
	const int Bs_y = (ty >> 1), Bs_x = (tx << 1) + (ty & 1);
	Bs[buf][Bs_y][Bs_x] = (tX0 <= Xend ? B[tX0 + Bk] : 0);
	__syncthreads();

	//compute area---------------------------------------------------
	float2 v0 = make_float2(0, 0);
	float2 v1 = make_float2(0, 0);
	float2 v2 = make_float2(0, 0);
	float2 v3 = make_float2(0, 0);
	float2 v4 = make_float2(0, 0);
	float2 v5 = make_float2(0, 0);
	float2 v6 = make_float2(0, 0);
	float2 v7 = make_float2(0, 0);
	for (int ok = 1, OK = (K << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float2 b = *(float2*)(&Bs[buf][ik][tx << 1]);
			float4 a0 = As[buf][ik][ty], a1 = As[buf][ik + STEP][ty];

			simdMM2(v0, a0.x, b);
			simdMM2(v1, a0.y, b);
			simdMM2(v2, a0.z, b);
			simdMM2(v3, a0.w, b);
			simdMM2(v4, a1.x, b);
			simdMM2(v5, a1.y, b);
			simdMM2(v6, a1.z, b);
			simdMM2(v7, a1.w, b);
		}
		buf ^= 1;

		//load 4 elements from A[batch]
		int Ak = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		As[buf][tx][ty].x = A[tY0 + Ak];
		As[buf][tx][ty].y = A[tY1 + Ak];
		As[buf][tx][ty].z = A[tY2 + Ak];
		As[buf][tx][ty].w = A[tY3 + Ak];

		//load 1 element from B[batch]
		int Bk = ((ok << LB) + ty) >> 1;
		Bs[buf][Bs_y][Bs_x] = (tX0 <= Xend ? B[tX0 + Bk] : 0);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) 
	{
		float2 b = *(float2*)(&Bs[buf][ik][tx << 1]);
		float4 a0 = As[buf][ik][ty], a1 = As[buf][ik + STEP][ty];

		simdMM2(v0, a0.x, b);
		simdMM2(v1, a0.y, b);
		simdMM2(v2, a0.z, b);
		simdMM2(v3, a0.w, b);
		simdMM2(v4, a1.x, b);
		simdMM2(v5, a1.y, b);
		simdMM2(v6, a1.z, b);
		simdMM2(v7, a1.w, b);
	}

	int Y0 = (Y * CM) + X;
	int Y1 = Y0 + CM, Y2 = Y1 + CM, Y3 = Y2 + CM;
	int Y4 = Y3 + CM, Y5 = Y4 + CM, Y6 = Y5 + CM, Y7 = Y6 + CM;

	*(float2*)(C + Y0) = v0;
	*(float2*)(C + Y1) = v1;
	*(float2*)(C + Y2) = v2;
	*(float2*)(C + Y3) = v3;
	*(float2*)(C + Y4) = v4;
	*(float2*)(C + Y5) = v5;
	*(float2*)(C + Y6) = v6;
	*(float2*)(C + Y7) = v7;
}

#endif


//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*8), K % (BLOCK_SIZE/2) == 0
//LB = 4: K % 8 == 0
#ifndef BATCH_MATMUL_T2_KERNEL_2_8
#define BATCH_MATMUL_T2_KERNEL_2_8

//LB = 4: Size = 1, Time = 3.112 msec, Performace = 690.065 GFlop/s
//LB = 3: Size = 1, Time = 3.736 msec, Performace = 572.563 GFlop/s
template<int LB, int STEP, int MOVE_A, int MOVE_B>
__global__ void bmmT2_kernel_2_8(
	const float* __restrict__ A, //A[Batch,  N,  K]
	const float* __restrict__ B, //B[Batch, BM,  K], BM is not memAligned
	      float* __restrict__ C, //C[Batch,  N, CM], CM is memAligned
	int N, int CM, int BM, int K,
	int Yindex, int Xindex)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float  As[2][1 << LB >> 1][(2 << LB) + 2];//k42
	__shared__ float4 Bs[2][1 << LB][(1 << LB) + 1];//k88

	//prepared for A -> Y:N
	const int Y = (((blockIdx.y << LB) + ty) << 1) + Yindex;
	const int tY0 = (Y + (tx & 1)) * K;

	//prepared for B -> X:M
	const int X = (((blockIdx.x << LB) + tx) << 3) + Xindex;
	const int tX0 = (X + ((ty >= STEP) << 2)) * K;
	const int tX1 = tX0 + K, tX2 = tX1 + K, tX3 = tX2 + K;

	//compute start offset of A, B, C
	const int batch = blockIdx.z;
	A += (batch * N *  K * MOVE_A);//A[batch * MOVE_A]
	B += (batch * K * BM * MOVE_B);//B[batch * MOVE_B]
	C += (batch * N * CM);//C[batch]

	//load 1 element from A[batch]
	int Ak = (tx >> 1);
	const int As_x = (tx >> 1), As_y = (ty << 1) + (tx & 1);
	As[buf][As_x][As_y] = A[tY0 + Ak];

	//load 4 elements from B[batch]
	const int Xend = (BM - 1) * K;
	int Bk = ty - ((ty >= STEP) << LB >> 1);
	Bs[buf][ty][tx].x = (tX0 <= Xend ? B[tX0 + Bk] : 0);
	Bs[buf][ty][tx].y = (tX1 <= Xend ? B[tX1 + Bk] : 0);
	Bs[buf][ty][tx].z = (tX2 <= Xend ? B[tX2 + Bk] : 0);
	Bs[buf][ty][tx].w = (tX3 <= Xend ? B[tX3 + Bk] : 0);
	__syncthreads();

	//compute area---------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (K << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP][tx];
			float2 a = *(float2*)(&As[buf][ik][(ty << 1)]);
			simdMM4(v0, a.x, b0); simdMM4(v1, a.x, b1);
			simdMM4(v2, a.y, b0); simdMM4(v3, a.y, b1);
		}
		buf ^= 1;

		//load 1 element from A[batch]
		int Ak = ((ok << LB) + tx) >> 1;
		As[buf][As_x][As_y] = A[tY0 + Ak];

		//load 4 elements from B[batch]
		int Bk = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		Bs[buf][ty][tx].x = (tX0 <= Xend ? B[tX0 + Bk] : 0);
		Bs[buf][ty][tx].y = (tX1 <= Xend ? B[tX1 + Bk] : 0);
		Bs[buf][ty][tx].z = (tX2 <= Xend ? B[tX2 + Bk] : 0);
		Bs[buf][ty][tx].w = (tX3 <= Xend ? B[tX3 + Bk] : 0);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP][tx];
		float2 a = *(float2*)(&As[buf][ik][(ty << 1)]);
		simdMM4(v0, a.x, b0); simdMM4(v1, a.x, b1);
		simdMM4(v2, a.y, b0); simdMM4(v3, a.y, b1);
	}

	int Y0 = (Y * CM) + X, Y1 = Y0 + CM;
	*(float4*)(C + Y0) = v0; *(float4*)(C + Y0 + 4) = v1;
	*(float4*)(C + Y1) = v2; *(float4*)(C + Y1 + 4) = v3;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*2), K % (BLOCK_SIZE/2) == 0
//LB = 4: K % 8 == 0
#ifndef BATCH_MATMUL_T2_KERNEL_4_2
#define BATCH_MATMUL_T2_KERNEL_4_2

//LB = 4: Size = 1, Time = 2.618 msec, Performace = 820.276 GFlop/s
//LB = 3: Size = 1, Time = 4.734 msec, Performace = 453.63 GFlop/s
template<int LB, int STEP, int MOVE_A, int MOVE_B>
__global__ void bmmT2_kernel_4_2(
	const float* __restrict__ A, //A[Batch,  N,  K]
	const float* __restrict__ B, //B[Batch, BM,  K], BM is not memAligned
	      float* __restrict__ C, //C[Batch,  N, CM], CM is memAligned
	int N, int CM, int BM, int K,
	int Yindex, int Xindex)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 As[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float  Bs[2][1 << LB >> 1][(2 << LB) + 2];

	//prepared for A -> Y:N
	const int Y = (((blockIdx.y << LB) + ty) << 2) + Yindex;
	const int tY0 = (Y + ((tx & 1) << 1)) * K, tY1 = tY0 + K;

	//prepared for B -> X:M
	const int X = (((blockIdx.x << LB) + tx) << 1) + Xindex;
	const int tX0 = (X + (ty & 1)) * K;

	//compute start offset of A, B, C
	const int batch = blockIdx.z;
	A += (batch * N *  K * MOVE_A);//A[batch * MOVE_A]
	B += (batch * K * BM * MOVE_B);//B[batch * MOVE_B]
	C += (batch * N * CM);//C[batch]

	int Ak = (tx >> 1);//load 2 elements from A[batch]
	const int As_x = (tx >> 1), As_y = (ty << 1) + (tx & 1);
	As[buf][As_x][As_y].x = A[tY0 + Ak];
	As[buf][As_x][As_y].y = A[tY1 + Ak];

	const int Xend = (BM - 1) * K;
	int Bk = (ty >> 1);//load 1 element from B[batch]
	const int Bs_y = (ty >> 1), Bs_x = (tx << 1) + (ty & 1);
	Bs[buf][Bs_y][Bs_x] = (tX0 <= Xend ? B[tX0 + Bk] : 0);
	__syncthreads();

	//compute area---------------------------------------------------
	float2 v0 = make_float2(0, 0);
	float2 v1 = make_float2(0, 0);
	float2 v2 = make_float2(0, 0);
	float2 v3 = make_float2(0, 0);
	for (int ok = 1, OK = (K << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float2 b = *(float2*)(&Bs[buf][ik][tx << 1]);
			float4 a = *(float4*)(&As[buf][ik][ty << 1]);
			simdMM2(v0, a.x, b);
			simdMM2(v1, a.y, b);
			simdMM2(v2, a.z, b);
			simdMM2(v3, a.w, b);
		}
		buf ^= 1;

		//load 2 elements from A[batch]
		int Ak = ((ok << LB) + tx) >> 1;
		As[buf][As_x][As_y].x = A[tY0 + Ak];
		As[buf][As_x][As_y].y = A[tY1 + Ak];

		//load 1 element from B[batch]
		int Bk = ((ok << LB) + ty) >> 1;
		Bs[buf][Bs_y][Bs_x] = (tX0 <= Xend ? B[tX0 + Bk] : 0);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float2 b = *(float2*)(&Bs[buf][ik][tx << 1]);
		float4 a = *(float4*)(&As[buf][ik][ty << 1]);
		simdMM2(v0, a.x, b);
		simdMM2(v1, a.y, b);
		simdMM2(v2, a.z, b);
		simdMM2(v3, a.w, b);
	}

	int Y0 = (Y * CM) + X;
	int Y1 = Y0 + CM, Y2 = Y1 + CM, Y3 = Y2 + CM;

	*(float2*)(C + Y0) = v0;
	*(float2*)(C + Y1) = v1;
	*(float2*)(C + Y2) = v2;
	*(float2*)(C + Y3) = v3;
}

#endif


//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*4), K % (BLOCK_SIZE/2) == 0
//LB = 4: K % 8 == 0
#ifndef BATCH_MATMUL_T2_KERNEL_2_4
#define BATCH_MATMUL_T2_KERNEL_2_4

//LB = 4: Size = 1, Time = 3.21 msec, Performace = 668.998 GFlop/s
//LB = 3: Size = 1, Time = 5.24 msec, Performace = 409.825 GFlop/s
template<int LB, int STEP, int MOVE_A, int MOVE_B>
__global__ void bmmT2_kernel_2_4(
	const float* __restrict__ A, //A[Batch,  N,  K]
	const float* __restrict__ B, //B[Batch, BM,  K], BM is not memAligned
	      float* __restrict__ C, //C[Batch,  N, CM], CM is memAligned
	int N, int CM, int BM, int K,
	int Yindex, int Xindex)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float  As[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2 Bs[2][1 << LB >> 1][(2 << LB) + 2];

	//prepared for A -> Y:N
	int Y = (((blockIdx.y << LB) + ty) << 1) + Yindex;
	const int tY0 = (Y + (tx & 1)) * K;

	//prepared for B -> X:M
	int X = (((blockIdx.x << LB) + tx) << 2) + Xindex;
	const int tX0 = (X + ((ty & 1) << 1)) * K, tX1 = tX0 + K;

	//compute start offset of A, B, C
	int batch = blockIdx.z;
	A += (batch * N *  K * MOVE_A);//A[batch * MOVE_A]
	B += (batch * K * BM * MOVE_B);//B[batch * MOVE_B]
	C += (batch * N * CM);//C[batch]
	
	int Ak = (tx >> 1);//load 1 element from A[batch]
	const int As_x = (tx >> 1), As_y = (ty << 1) + (tx & 1);
	As[buf][As_x][As_y] = A[tY0 + Ak];

	const int Xend = (BM - 1) * K;
	int Bk = (ty >> 1);//load 2 elements from B[batch]
	const int Bs_y = (ty >> 1), Bs_x = (tx << 1) + (ty & 1);
	Bs[buf][Bs_y][Bs_x].x = (tX0 <= Xend ? B[tX0 + Bk] : 0);
	Bs[buf][Bs_y][Bs_x].y = (tX1 <= Xend ? B[tX1 + Bk] : 0);
	__syncthreads();

	//compute area---------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0);
	float4 v1 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (K << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float4 b = *(float4*)(&Bs[buf][ik][(tx << 1)]);
			float2 a = *(float2*)(&As[buf][ik][(ty << 1)]);
			simdMM4(v0, a.x, b);
			simdMM4(v1, a.y, b);
		}
		buf ^= 1;

		//load 1 element from A[batch]
		int Ak = ((ok << LB) + tx) >> 1;
		As[buf][As_x][As_y] = A[tY0 + Ak];

		//load 2 elements from B[batch]
		int Bk = ((ok << LB) + ty) >> 1;
		Bs[buf][Bs_y][Bs_x].x = (tX0 <= Xend ? B[tX0 + Bk] : 0);
		Bs[buf][Bs_y][Bs_x].y = (tX1 <= Xend ? B[tX1 + Bk] : 0);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float4 b = *(float4*)(&Bs[buf][ik][(tx << 1)]);
		float2 a = *(float2*)(&As[buf][ik][(ty << 1)]);
		simdMM4(v0, a.x, b);
		simdMM4(v1, a.y, b);
	}

	int Y0 = (Y * CM) + X, Y1 = Y0 + CM;
	*(float4*)(C + Y0) = v0;
	*(float4*)(C + Y1) = v1;
}

#endif


//======[Small]===================================================
//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*2)
#ifndef BATCH_MATMUL_T2_KERNEL_2_2
#define BATCH_MATMUL_T2_KERNEL_2_2

//LB = 4: Size = 1, Time = 3.7   msec, Performace = 580.401 GFlop/s
//LB = 3: Size = 1, Time = 4.754 msec, Performace = 451.721 GFlop/s
template<int LB, int STEP, int MOVE_A, int MOVE_B>
__global__ void bmmT2_kernel_2_2(
	const float* __restrict__ A, //A[Batch,  N,  K]
	const float* __restrict__ B, //B[Batch, BM,  K], BM is not memAligned
	      float* __restrict__ C, //C[Batch,  N, CM], CM is memAligned
	int N, int CM, int BM, int K,
	int Yindex, int Xindex)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 As[2][1 << LB][(1 << LB) + 1];
	__shared__ float2 Bs[2][1 << LB][(1 << LB) + 1];

	//prepared for Y:N
	const int Y = (((blockIdx.y << LB) + ty) << 1) + Yindex;
	const int tY0 = Y * K, tY1 = tY0 + K;

	//prepared for X
	const int X = (((blockIdx.x << LB) + tx) << 1) + Xindex;
	const int tX0 = X * K, tX1 = tX0 + K;

	//compute start offset of A, B, C
	const int batch = blockIdx.z;
	A += (batch * N *  K * MOVE_A);//A[batch * MOVE_A]
	B += (batch * K * BM * MOVE_B);//B[batch * MOVE_B]
	C += (batch * N * CM);//C[batch]

	const int Xend = (BM - 1) * K;
	const int OK = (K >> LB);
	if (OK) {
		int Ak = tx;//load 2 elements from A[batch]
		As[buf][tx][ty].x = A[tY0 + Ak];
		As[buf][tx][ty].y = A[tY1 + Ak];

		int Bk = ty;//load 2 elements from B[batch]
		Bs[buf][ty][tx].x = (tX0 <= Xend ? B[tX0 + Bk] : 0);
		Bs[buf][ty][tx].y = (tX1 <= Xend ? B[tX1 + Bk] : 0);
		__syncthreads();
	}

	//compute area---------------------------------------------------
	float2 v0 = make_float2(0, 0);
	float2 v1 = make_float2(0, 0);
	for (int ok = 1; ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float2 b = Bs[buf][ik][tx];
			float2 a = As[buf][ik][ty];
			simdMM2(v0, a.x, b);
			simdMM2(v1, a.y, b);
		}
		buf ^= 1;

		int Ak = (ok << LB) + tx;//load 2 elements from A[batch]
		As[buf][tx][ty].x = A[tY0 + Ak];
		As[buf][tx][ty].y = A[tY1 + Ak];

		int Bk = (ok << LB) + ty;//load 2 elements from B[batch]
		Bs[buf][ty][tx].x = (tX0 <= Xend ? B[tX0 + Bk] : 0);
		Bs[buf][ty][tx].y = (tX1 <= Xend ? B[tX1 + Bk] : 0);
		__syncthreads();
	}
	if (OK) {
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float2 b = Bs[buf][ik][tx];
			float2 a = As[buf][ik][ty];
			simdMM2(v0, a.x, b);
			simdMM2(v1, a.y, b);
		}
	}

	//when GK % STEP!=0----------------------------------------------
	for (int k = K - (K&(STEP - 1)); k < K; k++) {
		float2 a;//load 2 elements from A[batch]
		a.x = A[tY0 + k];
		a.y = A[tY1 + k];

		float2 b;//load 2 elements from B[batch]
		b.x = (tX0 <= Xend ? B[tX0 + k] : 0);
		b.y = (tX1 <= Xend ? B[tX1 + k] : 0);

		simdMM2(v0, a.x, b);
		simdMM2(v1, a.y, b);
	}
	//when GK % STEP!=0----------------------------------------------

	int Y0 = (Y * CM) + X, Y1 = Y0 + CM;
	*(float2*)(C + Y0) = v0;
	*(float2*)(C + Y1) = v1;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*1),
#ifndef BATCH_MATMUL_T2_KERNEL_4_1
#define BATCH_MATMUL_T2_KERNEL_4_1

//LB = 4: Size = 0.996094, Time = 4.136 msec, Performace = 517.189 GFlop/s
//LB = 3: Size = 0.996094, Time = 6.184 msec, Performace = 345.908 GFlop/s
template<int LB, int STEP, int MOVE_A, int MOVE_B>
__global__ void bmmT2_kernel_4_1(
	const float* __restrict__ A, //A[Batch,  N,  K]
	const float* __restrict__ B, //B[Batch, BM,  K], BM is not memAligned
	      float* __restrict__ C, //C[Batch,  N, CM], CM is memAligned
	int N, int CM, int BM, int K,
	int Yindex, int Xindex)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 As[2][1 << LB][(1 << LB) + 1];
	__shared__ float  Bs[2][1 << LB][(1 << LB) + 1];

	//prepared for A -> Y:N
	const int Y = (((blockIdx.y << LB) + ty) << 2) + Yindex;
	const int tY0 = Y * K;
	const int tY1 = tY0 + K, tY2 = tY1 + K, tY3 = tY2 + K;

	//prepared for B -> X:M
	const int X = ((blockIdx.x << LB) + tx) + Xindex;
	const int tX0 = X * K;

	//compute start offset of A, B, C
	const int batch = blockIdx.z;
	A += (batch * N *  K * MOVE_A);//A[batch * MOVE_A]
	B += (batch * K * BM * MOVE_B);//B[batch * MOVE_B]
	C += (batch * N * CM);//C[batch]

	const int Xend = (BM - 1) * K;
	const int OK = (K >> LB);
	if (OK) {
		int Ak = tx;//load 2 elements from A[batch]
		As[buf][tx][ty].x = A[tY0 + Ak];
		As[buf][tx][ty].y = A[tY1 + Ak];
		As[buf][tx][ty].z = A[tY2 + Ak];
		As[buf][tx][ty].w = A[tY3 + Ak];

		int Bk = ty;//load 1 element from B[batch]
		Bs[buf][ty][tx] = (tX0 <= Xend ? B[tX0 + Bk] : 0);
		__syncthreads();
	}
	
	//compute area---------------------------------------------------
	float4 v = make_float4(0, 0, 0, 0);
	for (int ok = 1; ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float  b = Bs[buf][ik][tx];
			float4 a = As[buf][ik][ty];
			simdMM4(v, b, a);
		}
		buf ^= 1;

		//load 4 elements from A[batch]
		int Ak = (ok << LB) + tx;
		As[buf][tx][ty].x = A[tY0 + Ak];
		As[buf][tx][ty].y = A[tY1 + Ak];
		As[buf][tx][ty].z = A[tY2 + Ak];
		As[buf][tx][ty].w = A[tY3 + Ak];

		//load 1 element from B[batch]
		int Bk = (ok << LB) + ty;
		Bs[buf][ty][tx] = (tX0 <= Xend ? B[tX0 + Bk] : 0);
		__syncthreads();
	}
	if (OK) {
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float  b = Bs[buf][ik][tx];
			float4 a = As[buf][ik][ty];
			simdMM4(v, b, a);
		}
	}

	//when GK % STEP!=0----------------------------------------------
	for (int k = K - (K&(STEP - 1)); k < K; k++) {
		float4 a;//load 4 elements from A
		a.x = A[tY0 + k];
		a.y = A[tY1 + k];
		a.z = A[tY2 + k];
		a.w = A[tY3 + k];

		//load 1 element from B
		float b = (tX0 <= Xend ? B[tX0 + k] : 0);

		simdMM4(v, b, a);
	}
	//when GK % STEP!=0----------------------------------------------

	int Y0 = (Y * CM) + X;
	int Y1 = Y0 + CM, Y2 = Y1 + CM, Y3 = Y2 + CM;
	C[Y0] = v.x;
	C[Y1] = v.y;
	C[Y2] = v.z;
	C[Y3] = v.w;
}

#endif


//(Y: BLOCK_SIZE*1, X: BLOCK_SIZE*4)
#ifndef BATCH_MATMUL_T2_KERNEL_1_4
#define BATCH_MATMUL_T2_KERNEL_1_4

//LB = 4: Size = 0.996094, Time = 5.918 msec, Performace = 361.456 GFlop/s
//LB = 3: Size = 0.996094, Time = 7.258 msec, Performace = 294.722 GFlop/s
template<int LB, int STEP, int MOVE_A, int MOVE_B>
__global__ void bmmT2_kernel_1_4(
	const float* __restrict__ A, //A[Batch,  N,  K]
	const float* __restrict__ B, //B[Batch, BM,  K], BM is not memAligned
	      float* __restrict__ C, //C[Batch,  N, CM], CM is memAligned
	int N, int CM, int BM, int K,
	int Yindex, int Xindex)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float  As[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Bs[2][1 << LB][(1 << LB) + 1];

	//prepared for A -> Y:N
	const int Y = ((blockIdx.y << LB) + ty) + Yindex;
	const int tY0 = Y * K;

	//prepared for B -> X:M
	const int X = (((blockIdx.x << LB) + tx) << 2) + Xindex;
	const int tX0 = X * K;
	const int tX1 = tX0 + K, tX2 = tX1 + K, tX3 = tX2 + K;

	//compute start offset of A, B, C
	const int batch = blockIdx.z;
	A += (batch *  N * K * MOVE_A);//A[batch * MOVE_A]
	B += (batch * BM * K * MOVE_B);//B[batch * MOVE_B]
	C += (batch * N * CM);//C[batch]

	const int Xend = (BM - 1) * K;
	const int OK = (K >> LB);
	if (OK) {
		int Ak = tx;//load 1 element from A[batch]
		As[buf][tx][ty] = A[tY0 + Ak];

		int Bk = ty;//load 4 elements from B[batch]
		Bs[buf][ty][tx].x = (tX0 <= Xend ? B[tX0 + Bk] : 0);
		Bs[buf][ty][tx].y = (tX1 <= Xend ? B[tX1 + Bk] : 0);
		Bs[buf][ty][tx].z = (tX2 <= Xend ? B[tX2 + Bk] : 0);
		Bs[buf][ty][tx].w = (tX3 <= Xend ? B[tX3 + Bk] : 0);
		__syncthreads();
	}
	
	//compute area---------------------------------------------------
	float4 v = make_float4(0, 0, 0, 0);
	for (int ok = 1; ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float4 b = Bs[buf][ik][tx];
			float  a = As[buf][ik][ty];
			simdMM4(v, a, b);
		}
		buf ^= 1;

		//load 1 element from A[batch]
		int Ak = (ok << LB) + tx;
		As[buf][tx][ty] = A[tY0 + Ak];

		//load 1 element from B[batch]
		int Bk = (ok << LB) + ty;
		Bs[buf][ty][tx].x = (tX0 <= Xend ? B[tX0 + Bk] : 0);
		Bs[buf][ty][tx].y = (tX1 <= Xend ? B[tX1 + Bk] : 0);
		Bs[buf][ty][tx].z = (tX2 <= Xend ? B[tX2 + Bk] : 0);
		Bs[buf][ty][tx].w = (tX3 <= Xend ? B[tX3 + Bk] : 0);
		__syncthreads();
	}
	if (OK) {
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float4 b = Bs[buf][ik][tx];
			float  a = As[buf][ik][ty];
			simdMM4(v, a, b);
		}
	}

	//when GK % STEP!=0----------------------------------------------
	for (int k = K - (K&(STEP - 1)); k < K; k++) {
		float  a = A[tY0 + k];//load 1 elements from A

		float4 b;//load 4 elements from B
		b.x = (tX0 <= Xend ? B[tX0 + k] : 0);
		b.y = (tX1 <= Xend ? B[tX1 + k] : 0);
		b.z = (tX2 <= Xend ? B[tX2 + k] : 0);
		b.w = (tX3 <= Xend ? B[tX3 + k] : 0);

		simdMM4(v, a, b);
	}
	//when GK % STEP!=0----------------------------------------------

	*(float4*)(C + (Y * CM) + X) = v;
}

#endif

#endif