#pragma once

#ifndef BATCH_MATMUL_T1_KERNEL_H
#define BATCH_MATMUL_T1_KERNEL_H

//A[Batch,  K, AN] logically-> A^T[Batch, AN, K]
//B[Batch,  K,  M]
//C[Batch, CN,  M]
//N = CN: AN % 4 == 0, CN % 4 != 0, AN >= CN, AN = (CN + 3) >> 2 << 2
//M % 4 == 0
//K % 4 != 0
#ifndef BATCH_MATMUL_T1_KERNEL_CALL
#define BATCH_MATMUL_T1_KERNEL_CALL

//LB = log2(BLOCK_SIZE)
//MOVE_A == 0: A is a 2D tensor[K, N], logically expand A to[Batch, K, N]
//MOVE_B == 0: B is a 2D tensor[K, M], logically expand B to[Barch, K, M]

#define bmmT1_k88_mk(stream, LB, Yindex, Xindex, A, B, C, Batch, CN, AN, CM, K, GN, GM)\
	bmmT1_kernel_8_8_MK<LB, (1<<LB>>1), MOVE_A, MOVE_B>\
		<<< dim3((GM>>LB>>3), (GN>>LB>>3), Batch),\
		    dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, CN, AN, CM, K, Yindex, Xindex)

#define bmmT1_k88(stream, LB, Yindex, Xindex, A, B, C, Batch, CN, AN, CM, K, GN, GM)\
	bmmT1_kernel_8_8<LB, (1<<LB>>1), MOVE_A, MOVE_B>\
		<<< dim3((GM>>LB>>3), (GN>>LB>>3), Batch),\
		    dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, CN, AN, CM, K, Yindex, Xindex)

#define bmmT1_k44(stream, LB, Yindex, Xindex, A, B, C, Batch, CN, AN, CM, K, GN, GM)\
	bmmT1_kernel_4_4<LB, (1<<LB>>1), MOVE_A, MOVE_B>\
		<<< dim3((GM>>LB>>2), (GN>>LB>>2), Batch),\
            dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, CN, AN, CM, K, Yindex, Xindex)

#define bmmT1_k82(stream, LB, Yindex, Xindex, A, B, C, Batch, CN, AN, CM, K, GN, GM)\
	bmmT1_kernel_8_2<LB, (1<<LB>>1), MOVE_A, MOVE_B>\
		<<< dim3((GM>>LB>>1), (GN>>LB>>3), Batch),\
            dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, CN, AN, CM, K, Yindex, Xindex)

#define bmmT1_k28(stream, LB, Yindex, Xindex, A, B, C, Batch, CN, AN, CM, K, GN, GM)\
	bmmT1_kernel_2_8<LB, (1<<LB>>1), MOVE_A, MOVE_B>\
		<<< dim3((GM>>LB>>3), (GN>>LB>>1), Batch),\
            dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, CN, AN, CM, K, Yindex, Xindex)

#define bmmT1_k42(stream, LB, Yindex, Xindex, A, B, C, Batch, CN, AN, CM, K, GN, GM)\
	bmmT1_kernel_4_2<LB, (1<<LB>>1), MOVE_A, MOVE_B>\
		<<< dim3((GM>>LB>>1), (GN>>LB>>2), Batch),\
            dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, CN, AN, CM, K, Yindex, Xindex)

#define bmmT1_k24(stream, LB, Yindex, Xindex, A, B, C, Batch, CN, AN, CM, K, GN, GM)\
	bmmT1_kernel_2_4<LB, (1<<LB>>1), MOVE_A, MOVE_B>\
		<<< dim3((GM>>LB>>2), (GN>>LB>>1), Batch),\
            dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, CN, AN, CM, K, Yindex, Xindex)

#define bmmT1_k22(stream, LB, Yindex, Xindex, A, B, C, Batch, CN, AN, CM, K, GN, GM)\
	bmmT1_kernel_2_2<LB, (1<<LB), MOVE_A, MOVE_B>\
		<<< dim3((GM>>LB>>1), (GN>>LB>>1), Batch),\
            dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, CN, AN, CM, K, Yindex, Xindex)

#define bmmT1_k41(stream, LB, Yindex, Xindex, A, B, C, Batch, CN, AN, CM, K, GN, GM)\
	bmmT1_kernel_4_1<LB, (1<<LB), MOVE_A, MOVE_B>\
		<<< dim3((GM>>LB), (GN>>LB>>2), Batch),\
            dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, CN, AN, CM, K, Yindex, Xindex)

#define bmmT1_k14(stream, LB, Yindex, Xindex, A, B, C, Batch, CN, AN, CM, K, GN, GM)\
	bmmT1_kernel_1_4<LB, (1<<LB), MOVE_A, MOVE_B>\
		<<< dim3((GM>>LB>>2), (GN>>LB), Batch),\
			dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, CN, AN, CM, K, Yindex, Xindex)

#endif


//(Y: BLOCK_SIZE * 8, X: BLOCK_SIZE * 8), K >= (BLOCK_SIZE / 2)
//LB = 4: K >= 8
//LB = 3: K >= 4
#ifndef BATCH_MATMUL_T1_KERNEL_8_8_MK
#define BATCH_MATMUL_T1_KERNEL_8_8_MK

//LB = 4: Size = 1, Time = 1.384 msec, Performace = 1551.65 GFlop/s
//LB = 3: Size = 1, Time = 1.448 msec, Performace = 1483.07 GFlop/s
//LB = 4: Size = 0.996094, Time = 1.414 msec, Performace = 1512.8 GFlop/s
//LB = 3: Size = 0.996094, Time = 1.45  msec, Performace = 1475.24 GFlop/s
template<int LB, int STEP, int MOVE_A, int MOVE_B>
__global__ void bmmT1_kernel_8_8_MK(
	const float* __restrict__ A, //A[Batch, K, AN], AN is memAligned
	const float* __restrict__ B, //B[Batch, K,  M]
		  float* __restrict__ C, //C[Batch, N,  M], CN is not memAlgined
	int CN, int AN, int CM, int K,
	int Yindex, int Xindex)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 As[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Bs[2][1 << LB][(1 << LB) + 1];

	//prepared for A -> Y:N
	const int Y = (((blockIdx.y << LB) + ty) << 3) + Yindex;
	const int tY = Y + ((tx >= STEP) << 2);

	//prepared for B -> X:M
	const int X = (((blockIdx.x << LB) + tx) << 3) + Xindex;
	const int tX = X + ((ty >= STEP) << 2);

	//compute start offset of A, B, C
	const int batch = blockIdx.z;
	A += (batch * AN * K * MOVE_A) + tY;//A[batch * MOVE_A, 0, tY]
	B += (batch *  K * CM * MOVE_B) + tX;//B[batch * MOVE_B, 0, tX]
	C += (batch * CN * CM);//C[batch]

	int Ak = tx - ((tx >= STEP) << LB >> 1);//load 4 elements from A[batch]
	As[buf][tx][ty] = *(float4*)(&A[Ak * AN]);//AN % 4 == 0, safe to use float4

	int Bk = ty - ((ty >= STEP) << LB >> 1);//load 4 elements from B[batch]
	Bs[buf][ty][tx] = *(float4*)(&B[Bk * CM]);
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
			float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP][tx];
			float4 a0 = As[buf][ik][ty], a1 = As[buf][ik + STEP][ty];

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

		int Ak = ((ok - (tx >= STEP)) << LB >> 1) + tx;//load 4 elements from A[batch]
		As[buf][tx][ty] = *(float4*)(&A[Ak * AN]);

		int Bk = ((ok - (ty >= STEP)) << LB >> 1) + ty;//load 4 elements from B[batch]
		Bs[buf][ty][tx] = *(float4*)(&B[Bk * CM]);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) 
	{
		float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP][tx];
		float4 a0 = As[buf][ik][ty], a1 = As[buf][ik + STEP][ty];

		simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
		simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
		simdMM4(v4, a0.z, b0); simdMM4(v5, a0.z, b1);
		simdMM4(v6, a0.w, b0); simdMM4(v7, a0.w, b1);
		simdMM4(v8, a1.x, b0); simdMM4(v9, a1.x, b1);
		simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
		simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
		simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
	}

	int Y0 = (Y * CM) + X;
	int Y1 = Y0 + CM, Y2 = Y1 + CM, Y3 = Y2 + CM;
	int Y4 = Y3 + CM, Y5 = Y4 + CM, Y6 = Y5 + CM, Y7 = Y6 + CM;

	*(float4*)(C + Y0) = v0;  *(float4*)(C + Y0 + 4) = v1;
	*(float4*)(C + Y1) = v2;  *(float4*)(C + Y1 + 4) = v3;
	*(float4*)(C + Y2) = v4;  *(float4*)(C + Y2 + 4) = v5;
	*(float4*)(C + Y3) = v6;  *(float4*)(C + Y3 + 4) = v7;
	*(float4*)(C + Y4) = v8;  *(float4*)(C + Y4 + 4) = v9;
	*(float4*)(C + Y5) = v10; *(float4*)(C + Y5 + 4) = v11;
	*(float4*)(C + Y6) = v12; *(float4*)(C + Y6 + 4) = v13;
	*(float4*)(C + Y7) = v14; *(float4*)(C + Y7 + 4) = v15;
}

#endif


//(Y: BLOCK_SIZE * 8, X: BLOCK_SIZE * 8), K >= (BLOCK_SIZE / 2)
//LB = 4: K >= 8
//LB = 3: K >= 4
#ifndef BATCH_MATMUL_T1_KERNEL_8_8
#define BATCH_MATMUL_T1_KERNEL_8_8

//LB = 4: Size = 1, Time = 1.392 msec, Performace = 1542.73 GFlop/s
//LB = 3: Size = 1, Time = 1.448 msec, Performace = 1483.07 GFlop/s
//LB = 4: Size = 0.996094, Time = 1.414 msec, Performace = 1512.8 GFlop/s
//LB = 3: Size = 0.996094, Time = 1.45  msec, Performace = 1475.24 GFlop/s
template<int LB, int STEP, int MOVE_A, int MOVE_B>
__global__ void bmmT1_kernel_8_8(
	const float* __restrict__ A, //A[Batch, K, AN], AN is memAligned
	const float* __restrict__ B, //B[Batch, K,  M]
	      float* __restrict__ C, //C[Batch, N,  M], CN is not memAlgined
	int CN, int AN, int CM, int K,
	int Yindex, int Xindex)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 As[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Bs[2][1 << LB][(1 << LB) + 1];

	//prepared for A -> Y:N
	const int Y = (((blockIdx.y << LB) + ty) << 3) + Yindex;
	const int tY = Y + ((tx >= STEP) << 2);

	//prepared for B -> X:M
	const int X = (((blockIdx.x << LB) + tx) << 3) + Xindex;
	const int tX = X + ((ty >= STEP) << 2);

	//compute start offset of A, B, C
	const int batch = blockIdx.z;
	A += (batch * AN * K * MOVE_A) + tY;//A[batch * MOVE_A, 0, tY]
	B += (batch *  K * CM * MOVE_B) + tX;//B[batch * MOVE_B, 0, tX]
	C += (batch * CN * CM);//C[batch]

	int Ak = tx - ((tx >= STEP) << LB >> 1);//load 4 elements from A[batch]
	As[buf][tx][ty] = *(float4*)(&A[Ak * AN]);//AN % 4 == 0, safe to use float4

	int Bk = ty - ((ty >= STEP) << LB >> 1);//load 4 elements from B[batch]
	Bs[buf][ty][tx] = *(float4*)(&B[Bk * CM]);
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
		for (int ik = 0; ik < STEP; ik++) {
			float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP][tx];
			float4 a0 = As[buf][ik][ty], a1 = As[buf][ik + STEP][ty];

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

		int Ak = ((ok - (tx >= STEP)) << LB >> 1) + tx;//load 4 elements from A[batch]
		As[buf][tx][ty] = *(float4*)(&A[Ak * AN]);

		int Bk = ((ok - (ty >= STEP)) << LB >> 1) + ty;//load 4 elements from B[batch]
		Bs[buf][ty][tx] = *(float4*)(&B[Bk * CM]);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP][tx];
		float4 a0 = As[buf][ik][ty], a1 = As[buf][ik + STEP][ty];

		simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
		simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
		simdMM4(v4, a0.z, b0); simdMM4(v5, a0.z, b1);
		simdMM4(v6, a0.w, b0); simdMM4(v7, a0.w, b1);
		simdMM4(v8, a1.x, b0); simdMM4(v9, a1.x, b1);
		simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
		simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
		simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
	}

	//when GK % STEP!=0----------------------------------------------
	A += (Y - tY); B += (X - tX);
	for (int k = K - (K&(STEP - 1)); k < K; k++)
	{
		//load 8 elements from A[batch]
		float4 a0 = *(float4*)(&A[(k * AN)   ]);
		float4 a1 = *(float4*)(&A[(k * AN) + 4]);

		//load 8 elements from B[batch]
		float4 b0 = *(float4*)(&B[(k * CM)    ]);
		float4 b1 = *(float4*)(&B[(k * CM) + 4]);

		simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
		simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
		simdMM4(v4, a0.z, b0); simdMM4(v5, a0.z, b1);
		simdMM4(v6, a0.w, b0); simdMM4(v7, a0.w, b1);
		simdMM4(v8, a1.x, b0); simdMM4(v9, a1.x, b1);
		simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
		simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
		simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
	}
	//when GK % STEP!=0----------------------------------------------

	int Y0 = (Y * CM) + X;
	int Y1 = Y0 + CM, Y2 = Y1 + CM, Y3 = Y2 + CM;
	int Y4 = Y3 + CM, Y5 = Y4 + CM, Y6 = Y5 + CM, Y7 = Y6 + CM;

	*(float4*)(C + Y0) = v0; *(float4*)(C + Y0 + 4) = v1;
	*(float4*)(C + Y1) = v2; *(float4*)(C + Y1 + 4) = v3;
	*(float4*)(C + Y2) = v4; *(float4*)(C + Y2 + 4) = v5;
	*(float4*)(C + Y3) = v6; *(float4*)(C + Y3 + 4) = v7;
	*(float4*)(C + Y4) = v8; *(float4*)(C + Y4 + 4) = v9;
	*(float4*)(C + Y5) = v10; *(float4*)(C + Y5 + 4) = v11;
	*(float4*)(C + Y6) = v12; *(float4*)(C + Y6 + 4) = v13;
	*(float4*)(C + Y7) = v14; *(float4*)(C + Y7 + 4) = v15;
}

#endif


//(Y: BLOCK_SIZE * 4, X: BLOCK_SIZE * 4)
#ifndef BATCH_MATMUL_T1_KERNEL_4_4
#define BATCH_MATMUL_T1_KERNEL_4_4

//LB = 4: Size = 0.996094, Time = 2.008 msec, Performace = 1065.29 GFlop/s
//LB = 3: Size = 0.996094, Time = 2.398 msec, Performace =  892.033 GFlop/s
template<int LB, int STEP, int MOVE_A, int MOVE_B>
__global__ void bmmT1_kernel_4_4(
	const float* __restrict__ A, //A[Batch,  K, AN], AN is memAligned
	const float* __restrict__ B, //B[Batch,  K,  M]
		  float* __restrict__ C, //C[Batch, CN,  M], CN is not memAlgined
	int CN, int AN, int CM, int K,
	int Yindex, int Xindex)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 As[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2 Bs[2][1 << LB >> 1][(2 << LB) + 2];

	//prepared for A -> Y:N
	const int Y = (((blockIdx.y << LB) + ty) << 2) + Yindex;
	const int tY = Y + ((tx & 1) << 1);

	//prepared for B -> X:M
	const int X = (((blockIdx.x << LB) + tx) << 2) + Xindex;
	const int tX = X + ((ty & 1) << 1);

	//compute start offset of A, B, C
	const int batch = blockIdx.z;
	A += (batch * AN * K * MOVE_A) + tY;//A[batch * MOVE_A, 0, tY]
	B += (batch *  K * CM * MOVE_B) + tX;//B[batch * MOVE_B, 0, tX]
	C += (batch * CN * CM);//C[batch]

	const int As_x = (tx >> 1), As_y = (ty << 1) + (tx & 1);
	const int Bs_y = (ty >> 1), Bs_x = (tx << 1) + (ty & 1);
	const int OK = (K << 1 >> LB);
	if (OK) {
		int Ak = (tx >> 1);//load 2 elements from A[batch]
		As[buf][As_x][As_y] = *(float2*)(&A[Ak * AN]);

		int Bk = (ty >> 1);//load 2 elements from B[btch]
		Bs[buf][Bs_y][Bs_x] = *(float2*)(&B[Bk * CM]);
		__syncthreads();
	}

	//compute area---------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0);
	float4 v1 = make_float4(0, 0, 0, 0);
	float4 v2 = make_float4(0, 0, 0, 0);
	float4 v3 = make_float4(0, 0, 0, 0);
	for (int ok = 1; ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float4 b = *(float4*)(&Bs[buf][ik][tx << 1]);
			float4 a = *(float4*)(&As[buf][ik][ty << 1]);
			simdMM4(v0, a.x, b);
			simdMM4(v1, a.y, b);
			simdMM4(v2, a.z, b);
			simdMM4(v3, a.w, b);
		}
		buf ^= 1;

		int Ak = ((ok << LB) + tx) >> 1;//load 2 elements from A[batch]
		As[buf][As_x][As_y] = *(float2*)(&A[Ak * AN]);

		int Bk = ((ok << LB) + ty) >> 1;//load 2 elements from B[batch]
		Bs[buf][Bs_y][Bs_x] = *(float2*)(&B[Bk * CM]);
		__syncthreads();
	}
	if (OK) {
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float4 b = *(float4*)(&Bs[buf][ik][tx << 1]);
			float4 a = *(float4*)(&As[buf][ik][ty << 1]);
			simdMM4(v0, a.x, b);
			simdMM4(v1, a.y, b);
			simdMM4(v2, a.z, b);
			simdMM4(v3, a.w, b);
		}
	}

	//when GK % STEP!=0----------------------------------------------
	A += (Y - tY); B += (X - tX);
	for (int k = K - (K&(STEP - 1)); k < K; k++)  {
		float4 a = *(float4*)(&A[k * AN]);//load4 elements from A[batch]
		float4 b = *(float4*)(&B[k * CM]);//load4 elements from B[batch]
		simdMM4(v0, a.x, b);
		simdMM4(v1, a.y, b);
		simdMM4(v2, a.z, b);
		simdMM4(v3, a.w, b);
	}
	//when GK % STEP!=0----------------------------------------------

	int Y0 = (Y * CM) + X;
	int Y1 = Y0 + CM, Y2 = Y1 + CM, Y3 = Y2 + CM;

	*(float4*)(C + Y0) = v0;
	*(float4*)(C + Y1) = v1;
	*(float4*)(C + Y2) = v2;
	*(float4*)(C + Y3) = v3;
}

#endif


//(Y: BLOCK_SIZE * 8, X: BLOCK_SIZE * 2)
#ifndef BATCH_MATMUL_T1_KERNEL_8_2
#define BATCH_MATMUL_T1_KERNEL_8_2

//LB = 4: Size = 0.996094, Time = 2.04  msec, Performace = 1048.58 GFlop/s
//LB = 3: Size = 0.996094, Time = 2.278 msec, Performace =  939.023 GFlop/s
template<int LB, int STEP, int MOVE_A, int MOVE_B>
__global__ void bmmT1_kernel_8_2(
	const float* __restrict__ A, //A[Batch,  K, AN], AN is memAligned
	const float* __restrict__ B, //B[Batch,  K,  M]
	      float* __restrict__ C, //C[Batch, CN,  M], CN is not memAlgined
	int CN, int AN, int CM, int K,
	int Yindex, int Xindex)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 As[2][1 << LB][(1 << LB) + 1];
	__shared__ float  Bs[2][1 << LB >> 1][(2 << LB) + 2];

	//prepared for Y:N
	const int Y = (((blockIdx.y << LB) + ty) << 3) + Yindex;
	const int tY = Y + ((tx >= STEP) << 2);

	//prepared for X:M
	const int X = (((blockIdx.x << LB) + tx) << 1) + Xindex;
	const int tX = X + (ty & 1);

	//compute start offset of A, B, C
	const int batch = blockIdx.z;
	A += (batch * AN * K * MOVE_A) + tY;//A[batch * MOVE_A, 0, tY]
	B += (batch *  K * CM * MOVE_B) + tX;//B[batch * MOVE_B, 0, tX]
	C += (batch * CN * CM);//C[batch]

	const int Bs_y = (ty >> 1), Bs_x = (tx << 1) + (ty & 1);
	const int OK = (K << 1 >> LB);
	if (OK) {
		int Ak = tx - ((tx >= STEP) << LB >> 1);//load 4 elements from A[batch]
		As[buf][tx][ty] = *(float4*)(&A[Ak * AN]);

		int Bk = (ty >> 1);//load 1 element from B[batch]
		Bs[buf][Bs_y][Bs_x] = B[Bk * CM];
		__syncthreads();
	}

	//compute area---------------------------------------------------
	float2 v0 = make_float2(0, 0);
	float2 v1 = make_float2(0, 0);
	float2 v2 = make_float2(0, 0);
	float2 v3 = make_float2(0, 0);
	float2 v4 = make_float2(0, 0);
	float2 v5 = make_float2(0, 0);
	float2 v6 = make_float2(0, 0);
	float2 v7 = make_float2(0, 0);
	for (int ok = 1; ok < OK; ok++)
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

		int Ak = ((ok - (tx >= STEP)) << LB >> 1) + tx;//load 4 elements from A[batch]
		As[buf][tx][ty] = *(float4*)(&A[Ak * AN]);

		int Bk = ((ok << LB) + ty) >> 1;//load 1 element from B[batch]
		Bs[buf][Bs_y][Bs_x] = B[Bk * CM];
		__syncthreads();
	}
	if (OK) {
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
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
	}

	//when GK % STEP!=0----------------------------------------------
	A += (Y - tY); B += (X - tX);
	for (int k = K - (K&(STEP - 1)); k < K; k++) 
	{
		//load8 elements from A[batch]
		float4 a0 = *(float4*)(&A[k * AN]);
		float4 a1 = *(float4*)(&A[k * AN + 4]);

		//load2 elements from B[batch]
		float2 b = *(float2*)(&B[k * CM]);

		simdMM2(v0, a0.x, b);
		simdMM2(v1, a0.y, b);
		simdMM2(v2, a0.z, b);
		simdMM2(v3, a0.w, b);
		simdMM2(v4, a1.x, b);
		simdMM2(v5, a1.y, b);
		simdMM2(v6, a1.z, b);
		simdMM2(v7, a1.w, b);
	}
	//when GK % STEP!=0----------------------------------------------

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


//(Y: BLOCK_SIZE * 2, X: BLOCK_SIZE * 8)
#ifndef BATCH_MATMUL_T1_KERNEL_2_8
#define BATCH_MATMUL_T1_KERNEL_2_8

//LB = 4: Size = 0.996094, Time = 2.616 msec, Performace = 817.697 GFlop/s
//LB = 3: Size = 0.996094, Time = 2.86  msec, Performace = 747.935 GFlop/s
template<int LB, int STEP, int MOVE_A, int MOVE_B>
__global__ void bmmT1_kernel_2_8(
	const float* __restrict__ A, //A[Batch,  K, AN], AN is memAligned
	const float* __restrict__ B, //B[Batch,  K,  M]
	      float* __restrict__ C, //C[Batch, CN,  M],C.N is not memAlgined
	int CN, int AN, int CM, int K,
	int Yindex, int Xindex)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float  As[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float4 Bs[2][1 << LB][(1 << LB) + 1];

	//prepared for A -> Y:N
	const int Y = (((blockIdx.y << LB) + ty) << 1) + Yindex;
	const int tY = (Y + (tx & 1));

	//prepared for B -> X:M
	const int X = (((blockIdx.x << LB) + tx) << 3) + Xindex;
	const int tX = X + ((ty >= STEP) << 2);

	//compute start offset of A, B, C
	const int batch = blockIdx.z;
	A += (batch * AN * K * MOVE_A) + tY;//A[batch * MOVE_A, 0, tY]
	B += (batch *  K * CM * MOVE_B) + tX;//B[batch * MOVE_B, 0, tX]
	C += (batch * CN * CM);//C[batch]
	
	const int As_x = (tx >> 1), As_y = (ty << 1) + (tx & 1);
	const int OK = (K << 1 >> LB);
	if (OK) {
		//load 1 element from A[batch]
		int Ak = (tx >> 1);
		As[buf][As_x][As_y] = A[Ak * AN];

		//load 4 elements from B[batch]
		int Bk = ty - ((ty >= STEP) << LB >> 1);
		Bs[buf][ty][tx] = *(float4*)(&B[Bk * CM]);
		__syncthreads();
	}

	//compute area---------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	for (int ok = 1; ok < OK; ok++)
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
		As[buf][As_x][As_y] = A[Ak * AN];

		//load 4 elements from B[batch]
		int Bk = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		Bs[buf][ty][tx] = *(float4*)(&B[Bk * CM]);
		__syncthreads();
	}
	if (OK) {
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP][tx];
			float2 a = *(float2*)(&As[buf][ik][(ty << 1)]);
			simdMM4(v0, a.x, b0); simdMM4(v1, a.x, b1);
			simdMM4(v2, a.y, b0); simdMM4(v3, a.y, b1);
		}
	}

	//when GK % STEP!=0----------------------------------------------
	A += (Y - tY); B += (X - tX);
	for (int k = K - (K&(STEP - 1)); k < K; k++)
	{
		//load 2 elements from A[batch]
		float2 a = *(float2*)(&A[k * AN]);

		//load 8 elements from B[batch]
		float4 b0 = *(float4*)(&B[(k * CM)]);
		float4 b1 = *(float4*)(&B[(k * CM) + 4]);
		
		simdMM4(v0, a.x, b0); simdMM4(v1, a.x, b1);
		simdMM4(v2, a.y, b0); simdMM4(v3, a.y, b1);
	}
	//when GK % STEP!=0----------------------------------------------

	int Y0 = (Y * CM) + X, Y1 = Y0 + CM;
	*(float4*)(C + Y0) = v0; *(float4*)(C + Y0 + 4) = v1;
	*(float4*)(C + Y1) = v2; *(float4*)(C + Y1 + 4) = v3;
}

#endif


//(Y: BLOCK_SIZE * 4, X: BLOCK_SIZE * 2)
#ifndef BATCH_MATMUL_T1_KERNEL_4_2
#define BATCH_MATMUL_T1_KERNEL_4_2

//LB = 4: Size = 1, Time = 2.57  msec, Performace = 835.597 GFlop/s
//LB = 3: Size = 1, Time = 3.254 msec, Performace = 659.952 GFlop/s
//LB = 4: Size = 0.996094, Time = 2.726 msec, Performace = 784.701 GFlop/s
//LB = 3: Size = 0.996094, Time = 3.556 msec, Performace = 601.545 GFlop/s
template<int LB, int STEP, int MOVE_A, int MOVE_B>
__global__ void bmmT1_kernel_4_2(
	const float* __restrict__ A, //A[Batch,  K, AN], AN is memAligned
	const float* __restrict__ B, //B[Batch,  K,  M]
	      float* __restrict__ C, //C[Batch, CN,  M], CN is not memAlgined
	int CN, int AN, int CM, int K,
	int Yindex, int Xindex)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 As[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float  Bs[2][1 << LB >> 1][(2 << LB) + 2];

	//prepared for A -> Y:N
	const int Y = (((blockIdx.y << LB) + ty) << 2) + Yindex;
	const int tY = Y + ((tx & 1) << 1);

	//prepared for B -> X:M
	const int X = (((blockIdx.x << LB) + tx) << 1) + Xindex;
	const int tX = X + (ty & 1);

	//compute start offset of A, B, C
	const int batch = blockIdx.z;
	A += (batch * AN * K * MOVE_A) + tY;//A[batch * MOVE_A, 0, tY]
	B += (batch *  K * CM * MOVE_B) + tX;//B[batch * MOVE_B, 0, tX]
	C += (batch * CN * CM);//C[batch]

	const int As_x = (tx >> 1), As_y = (ty << 1) + (tx & 1);
	const int Bs_y = (ty >> 1), Bs_x = (tx << 1) + (ty & 1);
	const int OK = (K << 1 >> LB);
	if (OK) {
		int Ak = (tx >> 1);//load 2 elements from A[batch]
		As[buf][As_x][As_y] = *(float2*)(&A[Ak * AN]);

		int Bk = (ty >> 1);//load 1 element from B[batch]
		Bs[buf][Bs_y][Bs_x] = B[Bk * CM];
		__syncthreads();
	}

	//compute area---------------------------------------------------
	float2 v0 = make_float2(0, 0);
	float2 v1 = make_float2(0, 0);
	float2 v2 = make_float2(0, 0);
	float2 v3 = make_float2(0, 0);
	for (int ok = 1; ok < OK; ok++)
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

		int Ak = ((ok << LB) + tx) >> 1;//load 2 elements from A[batch]
		As[buf][As_x][As_y] = *(float2*)(&A[Ak * AN]);

		int Bk = ((ok << LB) + ty) >> 1;//load 1 element from B[batch]
		Bs[buf][Bs_y][Bs_x] = B[Bk * CM];
		__syncthreads();
	}
	if (OK) {
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float2 b = *(float2*)(&Bs[buf][ik][tx << 1]);
			float4 a = *(float4*)(&As[buf][ik][ty << 1]);
			simdMM2(v0, a.x, b);
			simdMM2(v1, a.y, b);
			simdMM2(v2, a.z, b);
			simdMM2(v3, a.w, b);
		}
	}

	//when GK % STEP!=0----------------------------------------------
	A += (Y - tY); B += (X - tX);
	for (int k = K - (K&(STEP - 1)); k < K; k++) {
		float4 a = *(float4*)(&A[k * AN]);//load4 elements from A[batch]
		float2 b = *(float2*)(&B[k * CM]);//load2 elements from B[batch]
		simdMM2(v0, a.x, b);
		simdMM2(v1, a.y, b);
		simdMM2(v2, a.z, b);
		simdMM2(v3, a.w, b);
	}
	//when GK % STEP!=0----------------------------------------------

	int Y0 = (Y * CM) + X;
	int Y1 = Y0 + CM, Y2 = Y1 + CM, Y3 = Y2 + CM;

	*(float2*)(C + Y0) = v0;
	*(float2*)(C + Y1) = v1;
	*(float2*)(C + Y2) = v2;
	*(float2*)(C + Y3) = v3;
}

#endif


//(Y: BLOCK_SIZE * 2, X: BLOCK_SIZE * 4)
#ifndef BATCH_MATMUL_T1_KERNEL_2_4
#define BATCH_MATMUL_T1_KERNEL_2_4

//LB = 4: Size = 1, Time = 3.108 msec, Performace = 690.953 GFlop/s
//LB = 3: Size = 1, Time = 3.404 msec, Performace = 630.871 GFlop/s
//LB = 4: Size = 0.996094, Time = 3.168 msec, Performace = 675.219 GFlop/s
//LB = 3: Size = 0.996094, Time = 3.53  msec, Performace = 605.976 GFlop/s
template<int LB, int STEP, int MOVE_A, int MOVE_B>
__global__ void bmmT1_kernel_2_4(
	const float* __restrict__ A, //A[Batch,  K, AN], AN is memAligned
	const float* __restrict__ B, //B[Batch,  K,  M]
	      float* __restrict__ C, //C[Batch, CN,  M], CN is not memAlgined
	int CN, int AN, int CM, int K,
	int Yindex, int Xindex)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float  As[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2 Bs[2][1 << LB >> 1][(2 << LB) + 2];

	//prepared for A -> Y:N
	const int Y = (((blockIdx.y << LB) + ty) << 1) + Yindex;
	const int tY = (Y + (tx & 1));

	//prepared for B -> X:M
	const int X = (((blockIdx.x << LB) + tx) << 2) + Xindex;
	const int tX = X + ((ty & 1) << 1);

	//compute start offset of A, B, C
	const int batch = blockIdx.z;
	A += (batch * AN * K * MOVE_A) + tY;//A[batch * MOVE_A, 0, tY]
	B += (batch *  K * CM * MOVE_B) + tX;//B[batch * MOVE_B, 0, tX]
	C += (batch * CN * CM);//C[batch]

	const int As_x = (tx >> 1), As_y = (ty << 1) + (tx & 1);
	const int Bs_y = (ty >> 1), Bs_x = (tx << 1) + (ty & 1);
	const int OK = (K << 1 >> LB);
	if (OK) {
		int Ak = (tx >> 1);//load 1 element from A[batch]
		As[buf][As_x][As_y] = A[Ak * AN];

		int Bk = (ty >> 1);//load 2 elements from B[batch]
		Bs[buf][Bs_y][Bs_x] = *(float2*)(&B[Bk * CM]);
		__syncthreads();
	}

	//compute area---------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0);
	float4 v1 = make_float4(0, 0, 0, 0);
	for (int ok = 1; ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float4 b = *(float4*)(&Bs[buf][ik][(tx << 1)]);
			float2 a = *(float2*)(&As[buf][ik][(ty << 1)]);
			simdMM4(v0, a.x, b);
			simdMM4(v1, a.y, b);
		}
		buf ^= 1;

		int Ak = ((ok << LB) + tx) >> 1;//load 1 element from A[batch]
		As[buf][As_x][As_y] = A[Ak * AN];

		int Bk = ((ok << LB) + ty) >> 1;//load 2 elements from B[batch]
		Bs[buf][Bs_y][Bs_x] = *(float2*)(&B[Bk * CM]);//with the same tx
		__syncthreads();
	}
	if (OK) {
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float4 b = *(float4*)(&Bs[buf][ik][(tx << 1)]);
			float2 a = *(float2*)(&As[buf][ik][(ty << 1)]);
			simdMM4(v0, a.x, b);
			simdMM4(v1, a.y, b);
		}
	}

	//when GK % STEP!=0----------------------------------------------
	A += (Y - tY); B += (X - tX);
	for (int k = K - (K&(STEP - 1)); k < K; k++) {
		float2 a = *(float2*)(&A[k * AN]);//load2 elements from A[batch]
		float4 b = *(float4*)(&B[k *  CM]);//load4 elements from B[batch]
		simdMM4(v0, a.x, b);
		simdMM4(v1, a.y, b);
	}
	//when GK % STEP!=0----------------------------------------------

	int Y0 = (Y * CM) + X, Y1 = Y0 + CM;
	*(float4*)(C + Y0) = v0;
	*(float4*)(C + Y1) = v1;
}

#endif


//(Y: BLOCK_SIZE * 2, X: BLOCK_SIZE * 2)
#ifndef BATCH_MATMUL_T1_KERNEL_2_2
#define BATCH_MATMUL_T1_KERNEL_2_2

//LB = 4: Size = 0.996094, Time = 3.646 msec, Performace = 586.696 GFlop/s
//LB = 3: Size = 0.996094, Time = 4.45  msec, Performace = 480.696 GFlop/s
template<int LB, int STEP, int MOVE_A, int MOVE_B>
__global__ void bmmT1_kernel_2_2(
	const float* __restrict__ A, //A[Batch,  K, AN], AN is memAligned
	const float* __restrict__ B, //B[Batch,  K,  M]
	      float* __restrict__ C, //C[Batch, CN,  M], CN is not memAlgined
	int CN, int AN, int CM, int K,
	int Yindex, int Xindex)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 As[2][1 << LB][(1 << LB) + 1];
	__shared__ float2 Bs[2][1 << LB][(1 << LB) + 1];

	//prepared for A -> Y:N
	const int Y = (((blockIdx.y << LB) + ty) << 1) + Yindex;

	//prepared for B -> X:M
	const int X = (((blockIdx.x << LB) + tx) << 1) + Xindex;

	//compute start offset of A, B, C
	const int batch = blockIdx.z;
	A += (batch * AN * K * MOVE_A) + Y;//A[batch * MOVE_A, 0, Y]
	B += (batch *  K * CM * MOVE_B) + X;//B[batch * MOVE_B, 0, X]
	C += (batch * CN * CM);//C[batch]

	const int OK = (K >> LB);
	if (OK) {
		int Ak = tx;//load 2 elements from A[batch]
		As[buf][tx][ty] = *(float2*)(&A[Ak * AN]);

		int Bk = ty;//load 2 elements from B[batch]
		Bs[buf][ty][tx] = *(float2*)(&B[Bk * CM]);
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
		As[buf][tx][ty] = *(float2*)(&A[Ak * AN]);

		int Bk = (ok << LB) + ty;//load 2 elements from B[batch]
		Bs[buf][ty][tx] = *(float2*)(&B[Bk * CM]);
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
		float2 a = *(float2*)(&A[k * AN]);//load 2 elements from A[batch]
		float2 b = *(float2*)(&B[k *  CM]);//load 2 elements from B[batch]
		simdMM2(v0, a.x, b);
		simdMM2(v1, a.y, b);
	}
	//when GK % STEP!=0----------------------------------------------

	int Y0 = (Y * CM) + X, Y1 = Y0 + CM;
	*(float2*)(C + Y0) = v0;
	*(float2*)(C + Y1) = v1;
}

#endif


//(Y: BLOCK_SIZE * 4, X: BLOCK_SIZE * 1)
#ifndef BATCH_MATMUL_T1_KERNEL_4_1
#define BATCH_MATMUL_T1_KERNEL_4_1

//LB = 4: Size = 0.996094, Time = 3.632 msec, Performace = 588.958 GFlop/s
//LB = 3: Size = 0.996094, Time = 4.186 msec, Performace = 511.012 GFlop/s
template<int LB, int STEP, int MOVE_A, int MOVE_B>
__global__ void bmmT1_kernel_4_1(
    const float* __restrict__ A, //A[Batch,  K, AN], AN is memAligned
	const float* __restrict__ B, //B[Batch,  K,  M]
	      float* __restrict__ C, //C[Batch, CN,  M], CN is not memAlgined
	int CN, int AN, int CM, int K,
	int Yindex, int Xindex)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 As[2][1 << LB][(1 << LB) + 1];
	__shared__ float  Bs[2][1 << LB][(1 << LB) + 1];

	//prepared for A -> Y:N
	const int Y = (((blockIdx.y << LB) + ty) << 2) + Yindex;

	//prepared for B -> X:M
	const int X = ((blockIdx.x << LB) + tx) + Xindex;

	//compute start offset of A, B, C
	const int batch = blockIdx.z;
	A += (batch * AN * K * MOVE_A) + Y;//A[batch * MOVE_A, 0, Y]
	B += (batch *  K * CM * MOVE_B) + X;//B[batch * MOVE_B, 0, X]
	C += (batch * CN * CM);//C[batch]

	const int OK = (K >> LB);
	if (OK) {
		int Ak = tx;//load 4 elements from A[batch]
		As[buf][tx][ty] = *(float4*)(&A[Ak * AN]);

		int Bk = ty;//load 1 element from B[batch]
		Bs[buf][ty][tx] = B[Bk * CM];
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

		int Ak = (ok << LB) + tx;//load 4 elements from A[batch]
		As[buf][tx][ty] = *(float4*)(&A[Ak * AN]);

		int Bk = (ok << LB) + ty;//load 1 element from B[batch]
		Bs[buf][ty][tx] = B[Bk * CM];
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
		float4 a = *(float4*)(&A[k * AN]);//load 4 elements from A
		float  b = B[k * CM];//load 1 element from B
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


//(Y: BLOCK_SIZE * 1, X: BLOCK_SIZE * 4)
#ifndef BATCH_MATMUL_T1_KERNEL_1_4
#define BATCH_MATMUL_T1_KERNEL_1_4

//LB = 4: Size = 0.996094, Time = 5.188 msec, Performace = 412.316 GFlop/s
//LB = 3: Size = 0.996094, Time = 5.742 msec, Performace = 372.535 GFlop/s
template<int LB, int STEP, int MOVE_A, int MOVE_B>
__global__ void bmmT1_kernel_1_4(
	const float* __restrict__ A, //A[Batch,  K, AN], AN is memAligned
	const float* __restrict__ B, //B[Batch,  K,  M]
	      float* __restrict__ C, //C[Batch, CN,  M], CN is not memAlgined
	int CN, int AN, int CM, int K,
	int Yindex, int Xindex)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float  As[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Bs[2][1 << LB][(1 << LB) + 1];

	//prepared for A -> Y:N
	const int Y = ((blockIdx.y << LB) + ty) + Yindex;

	//prepared for B -> X:M
	const int X = (((blockIdx.x << LB) + tx) << 2) + Xindex;

	//compute start offset of A, B, C
	const int batch = blockIdx.z;
	A += (batch * AN * K * MOVE_A) + Y;//A[batch * MOVE_A, 0, Y]
	B += (batch * K * CM * MOVE_B) + X;//B[batch * MOVE_B, 0, X]
	C += (batch * CN * CM);//C[batch]

	const int OK = (K >> LB);
	if (OK) {
		int Ak = tx;//load 1 element from A[batch]
		As[buf][tx][ty] = A[Ak * AN];

		int Bk = ty;//load 4 elements from B[batch]
		Bs[buf][ty][tx] = *(float4*)(&B[Bk * CM]);
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

		int Ak = (ok << LB) + tx;//load 1 element from A[batch]
		As[buf][tx][ty] = A[Ak * AN];

		int Bk = (ok << LB) + ty;//load 4 elements from B[batch]
		Bs[buf][ty][tx] = *(float4*)(&B[Bk * CM]);
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
		float  a = A[k * AN];//load 1 elements from A
		float4 b = *(float4*)(&B[k * CM]);//load 4 elements from B
		simdMM4(v, a, b);
	}
	//when GK % STEP!=0----------------------------------------------

	*(float4*)(C + (Y * CM) + X) = v;
}

#endif

#endif