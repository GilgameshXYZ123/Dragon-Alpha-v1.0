#pragma once

#ifndef BATCH_MATMUL_T1_KERNEL_PADDING_H
#define BATCH_MATMUL_T1_KERNEL_PADDING_H

//A[Batch,  K, AN] logically-> A^T[Batch, AN, K]
//B[Batch,  K,  M]
//C[Batch, CN,  M]
//(1) N = CN: AN % 4 == 0, CN % 4 != 0, AN >= CN, AN = (CN + 3) >> 2 << 2
//(2) M % 4 == 0
//(3) K % 4 != 0
//M = (M + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE
//N = (N + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE
#ifndef BATCH_MATMUL_T1_KERNEL_PADDING_CALL
#define BATCH_MATMUL_T1_KERNEL_PADDING_CALL

//LB = log2(BLOCK_SIZE)
//MOVE_A == 0: A is a 2D tensor[K, N], logically expand A to[Batch, K, N]
//MOVE_B == 0: B is a 2D tensor[K, M], logically expand B to[Barch, K, M]

#define bmmT1_k88_p(stream, LB, Yindex, Xindex, A, B, C, Batch, CN, AN, CM, K, GN, GM)\
	bmmT1_kernel_8_8_padding<LB, (1<<LB>>1), MOVE_A, MOVE_B>\
		<<< dim3(((GM + (8<<LB)-1)>>LB>>3), ((GN + (8<<LB)-1)>>LB>>3), Batch),\
            dim3(1<<LB, 1<<LB), 0, stream>>>\
			(A, B, C, CN, AN, CM, K, Yindex, Xindex)

#define bmmT1_k44_p(stream, LB, Yindex, Xindex, A, B, C, Batch, CN, AN, CM, K, GN, GM)\
	bmmT1_kernel_4_4_padding<LB, (1<<LB>>1), MOVE_A, MOVE_B>\
		<<< dim3(((GM + (4<<LB)-1)>>LB>>2), ((GN + (4<<LB)-1)>>LB>>2), Batch),\
		    dim3(1<<LB, 1<<LB), 0, stream>>>\
			(A, B, C, CN, AN, CM, K, Yindex, Xindex)

#define bmmT1_k22_p(stream, LB, Yindex, Xindex, A, B, C, Batch, CN, AN, CM, K, GN, GM)\
	bmmT1_kernel_2_2_padding<LB, (1<<LB), MOVE_A, MOVE_B>\
		<<< dim3(((GM + (2<<LB)-1)>>LB>>1), ((GN + (2<<LB)-1)>>LB>>1), Batch),\
		    dim3(1<<LB, 1<<LB), 0, stream>>>\
			(A, B, C, CN, AN, CM, K, Yindex, Xindex)

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), K >= (BLOCK_SIZE/2)
//LB = 4: K >= 8
//LB = 3: K >= 4
#ifndef BATCH_MATMUL_T1_KERNEL_8_8_PADDING
#define BATCH_MATMUL_T1_KERNEL_8_8_PADDING

//LB = 4: Size = 0.98053, Time = 1.632 msec, Performace = 1290.24 GFlop/s
//LB = 3: Size = 0.98053, Time = 1.74  msec, Performace = 1210.16 GFlop/s
template<int LB, int STEP, int MOVE_A, int MOVE_B>
__global__ void bmmT1_kernel_8_8_padding(
	const float* __restrict__ A, //A[Batch,  K, AN], AN is memAligned
	const float* __restrict__ B, //B[Batch,  K,  M],
	      float* __restrict__ C, //C[Batch, CN,  M], CN is not memAlgined
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
	
	//load 4 elements from A[batch]
	const int Yend = (AN - 4);//float4: Y <= AN - 4
	int Ak = tx - ((tx >= STEP) << LB >> 1);
	As[buf][tx][ty] = (tY <= Yend ? *(float4*)(&A[Ak * AN]) : FZERO4);

	//load 4 elements from B[batch]
	const int Xend = (CM - 4);//float4: X <= M - 4 
	int Bk = ty - ((ty >= STEP) << LB >> 1);
	Bs[buf][ty][tx] = (tX <= Xend ? *(float4*)(&B[Bk * CM]) : FZERO4);
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

		int Ak = ((ok - (tx >= STEP)) << LB >> 1) + tx;//load 4 elements from A[batch]
		As[buf][tx][ty] = (tY <= Yend ? *(float4*)(&A[Ak * AN]) : FZERO4);

		int Bk = ((ok - (ty >= STEP)) << LB >> 1) + ty;//load 4 elements from B[batch]
		Bs[buf][ty][tx] = (tX <= Xend ? *(float4*)(&B[Bk * CM]) : FZERO4);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
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

	//when GK % STEP!=0----------------------------------------------
	A += (Y - tY); B += (X - tX);
	for (int k = K - (K&(STEP - 1)); k < K; k++) 
	{
		float4 a0, a1;//load 8 elements from A[batch]
		a0 = (Y     <= Yend ? *(float4*)(&A[k * AN    ]) : make_float4(0, 0, 0, 0));
		a1 = (Y + 4 <= Yend ? *(float4*)(&A[k * AN + 4]) : make_float4(0, 0, 0, 0));

		float4 b0, b1;//load 8 elements from B[batch]
		b0 = (X     <= Xend ? *(float4*)(&B[k * CM    ]) : make_float4(0, 0, 0, 0));
		b1 = (X + 4 <= Xend ? *(float4*)(&B[k * CM + 4]) : make_float4(0, 0, 0, 0));

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

	//As: Y:N -> N % 4 ==0, From the point view of floa4: 
	int Cend = (CN * CM) - 4;//float4: (CN - 1)*M + (M - 4) = CN*M - 4
	if (X <= Xend) {//float4: X <= M - 4
		if (Y0 <= Cend) *(float4*)(C + Y0) = v0;
		if (Y1 <= Cend) *(float4*)(C + Y1) = v2;
		if (Y2 <= Cend) *(float4*)(C + Y2) = v4;
		if (Y3 <= Cend) *(float4*)(C + Y3) = v6;
		if (Y4 <= Cend) *(float4*)(C + Y4) = v8;
		if (Y5 <= Cend) *(float4*)(C + Y5) = v10;
		if (Y6 <= Cend) *(float4*)(C + Y6) = v12;
		if (Y7 <= Cend) *(float4*)(C + Y7) = v14;
	}

	Cend = Cend - 4;//float8: (CN - 1)*M + (M - 8) = CN*M - 8
	if (X + 4 <= Xend) {//float8: X <= M - 8
		if (Y0 <= Cend) *(float4*)(C + Y0 + 4) = v1;
		if (Y1 <= Cend) *(float4*)(C + Y1 + 4) = v3;
		if (Y2 <= Cend) *(float4*)(C + Y2 + 4) = v5;
		if (Y3 <= Cend) *(float4*)(C + Y3 + 4) = v7;
		if (Y4 <= Cend) *(float4*)(C + Y4 + 4) = v9;
		if (Y5 <= Cend) *(float4*)(C + Y5 + 4) = v11;
		if (Y6 <= Cend) *(float4*)(C + Y6 + 4) = v13;
		if (Y7 <= Cend) *(float4*)(C + Y7 + 4) = v15;
	}
}

#endif


//(Y: BLOCK_SIZE * 4, X: BLOCK_SIZE * 4)
#ifndef BATCH_MATMUL_T1_KERNEL_4_4_PADDING
#define BATCH_MATMUL_T1_KERNEL_4_4_PADDING

//LB = 4: Size = 0.9767, Time = 2.058 msec, Performace = 1019.17 GFlop/s
//LB = 3: Size = 0.9767, Time = 2.468 msec, Performace = 849.857 GFlop/s
template<int LB, int STEP, int MOVE_A, int MOVE_B>
__global__ void bmmT1_kernel_4_4_padding(
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
	
	int Yend = (AN - 2), Xend = (CM - 2);//float2: Y <= AN - 2, X <= M - 2 
	const int As_x = (tx >> 1), As_y = (ty << 1) + (tx & 1);
	const int Bs_y = (ty >> 1), Bs_x = (tx << 1) + (ty & 1);
	const int OK = (K << 1 >> LB);
	if (OK) {
		int Ak = (tx >> 1);//load 2 elements from A[batch]
		As[buf][As_x][As_y] = (tY <= Yend ? *(float2*)(&A[Ak * AN]) : FZERO2);

		int Bk = (ty >> 1);//load 2 elements from B[btch]
		Bs[buf][Bs_y][Bs_x] = (tX <= Xend ? *(float2*)(&B[Bk * CM]) : FZERO2);
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

		//load 2 elements from A[batch]
		int Ak = ((ok << LB) + tx) >> 1;
		As[buf][As_x][As_y] = (tY <= Yend ? *(float2*)(&A[Ak * AN]) : FZERO2);

		//load 2 elements from B[batch]
		int Bk = ((ok << LB) + ty) >> 1;
		Bs[buf][Bs_y][Bs_x] = (tX <= Xend ? *(float2*)(&B[Bk * CM]) : FZERO2);
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
	Yend = (AN - 4); Xend = (CM - 4); //float4: Y <= AN - 4, X <= M - 4
	for (int k = K - (K&(STEP - 1)); k < K; k++) {
		float4 a = (Y <= Yend ? *(float4*)(&A[k * AN]) : FZERO4);//load4 elements from A[batch]
		float4 b = (X <= Xend ? *(float4*)(&B[k *  CM]) : FZERO4);//load4 elements from B[batch]
		simdMM4(v0, a.x, b);
		simdMM4(v1, a.y, b);
		simdMM4(v2, a.z, b);
		simdMM4(v3, a.w, b);
	}
	//when GK % STEP!=0----------------------------------------------

	int Y0 = (Y * CM) + X;
	int Y1 = Y0 + CM, Y2 = Y1 + CM, Y3 = Y2 + CM;
	int Cend = (CN * CM) - 4;//use float4, (CN - 1)*M + (M - 4)

	if (X <= Xend) {//float4: X <= M - 4
		if (Y0 <= Cend) *(float4*)(C + Y0) = v0;
		if (Y1 <= Cend) *(float4*)(C + Y1) = v1;
		if (Y2 <= Cend) *(float4*)(C + Y2) = v2;
		if (Y3 <= Cend) *(float4*)(C + Y3) = v3;
	}
}

#endif


//(Y: BLOCK_SIZE * 2, X: BLOCK_SIZE * 2)
#ifndef BATCH_MATMUL_T1_KERNEL_2_2_PADDING
#define BATCH_MATMUL_T1_KERNEL_2_2_PADDING

//LB = 4: Size = 0.9767, Time = 3.542 msec, Performace = 592.164 GFlop/s
//LB = 3: Size = 0.9767, Time = 4.464 msec, Performace = 469.858 GFlop/s
template<int LB, int STEP, int MOVE_A, int MOVE_B>
__global__ void bmmT1_kernel_2_2_padding(
	const float* __restrict__ A, //A[Batch,  K, AN], AN is memAligned
	const float* __restrict__ B, //B[Batch,  K,  M],
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
	int batch = blockIdx.z;
	A += (batch * AN * K * MOVE_A) + Y;//A[batch * MOVE_A, 0, Y]
	B += (batch * K * CM * MOVE_B) + X;//B[batch * MOVE_B, 0, X]
	C += (batch * CN * CM);//C[batch]
	
	int Yend = (AN - 2), Xend = (CM - 2);//float2: Y <= AN - 2, X <= M - 2 
	const int OK = (K >> LB);
	if (OK) {
		int Ak = tx;//load 2 elements from A[batch]
		As[buf][tx][ty] = (Y <= Yend ? *(float2*)(&A[Ak * AN]) : FZERO2);

		int Bk = ty;//load 2 elements from B[batch]
		Bs[buf][ty][tx] = (X <= Xend ? *(float2*)(&B[Bk * CM]) : FZERO2);
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

		//load 2 elements from A[batch]
		int Ak = (ok << LB) + tx;
		As[buf][tx][ty] = (Y <= Yend ? *(float2*)(&A[Ak * AN]) : FZERO2);

		//load 2 elements from B[batch]
		int Bk = (ok << LB) + ty;
		Bs[buf][ty][tx] = (X <= Xend ? *(float2*)(&B[Bk * CM]) : FZERO2);
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
		float2 a = (Y <= Yend ? *(float2*)(&A[k * AN]) : make_float2(0, 0));//load 2 elements from A[batch]
		float2 b = (X <= Xend ? *(float2*)(&B[k * CM]) : make_float2(0, 0));//load 2 elements from B[batch]
		simdMM2(v0, a.x, b);
		simdMM2(v1, a.y, b);
	}
	//when GK % STEP!=0----------------------------------------------

	int Y0 = (Y * CM) + X, Y1 = Y0 + CM;
	int Cend = CN * CM - 2;//float2: (CN - 1)*M + (M - 2) = CN*M - 2

	if (X <= Xend) {//float2: X <= M - 2
		if (Y0 <= Cend) *(float2*)(C + Y0) = v0;
		if (Y1 <= Cend) *(float2*)(C + Y1) = v1;
	}
}

#endif

#endif