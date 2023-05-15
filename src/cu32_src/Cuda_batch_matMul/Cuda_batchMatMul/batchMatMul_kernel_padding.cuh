#pragma once

#ifndef BATCH_MATMUL_KERNEL_PADDING_H
#define BATCH_MATMUL_KERNEL_PADDING_H

//A[Batch,  N, AK] 
//B[Batch, BK,  M]
//C[Batch,  N,  M]
//(1) M % 4 == 0
//(2) N % 4 != 0
//(3) K = BK: AK % 4 == 0, BK % 4 != 0, AK >= BK, AK = (BK + 3) >> 2 << 2
//(4) M = (M + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE
//(5) N = (N + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE
#ifndef BATCH_MATMUL_KERNEL_PADDING_CALL
#define BATCH_MATMUL_KERNEL_PADDING_CALL

//MOVE_A == 0: A is a 2D tensor[N, K], logically expand A to[Batch, N, K]
//MOVE_B == 0: B is a 2D tensor[K, M], logically expand B to[Barch, K, M]

#define bmm_k88_ptex(stream, LB, Yindex, Xindex, A, B, C, Batch, N, CM, BK, AK, GN, GM)\
	bmm_kernel_8_8_padding_texture<LB, (1<<LB>>1), MOVE_A, MOVE_B>\
		<<< dim3(((GM + (8<<LB)-1)>>LB>>3), ((GN + (8<<LB)-1)>>LB>>3), Batch),\
			 dim3(1<<LB, 1<<LB), 0, stream>>>\
			(A, B, C, N, CM, BK, AK, Yindex, Xindex)

#define bmm_k88_p(stream, LB, Yindex, Xindex, A, B, C, Batch, N, CM, BK, AK, GN, GM)\
	bmm_kernel_8_8_padding<LB, (1<<LB>>1), MOVE_A, MOVE_B>\
		<<< dim3(((GM + (8<<LB)-1)>>LB>>3), ((GN + (8<<LB)-1)>>LB>>3), Batch),\
			 dim3(1<<LB, 1<<LB), 0, stream>>>\
			(A, B, C, N, CM, BK, AK, Yindex, Xindex)

#define bmm_k44_p(stream, LB, Yindex, Xindex, A, B, C, Batch, N, CM, BK, AK, GN, GM)\
	bmm_kernel_4_4_padding<LB, (1<<LB>>1), MOVE_A, MOVE_B>\
		<<< dim3(((GM + (4<<LB)-1)>>LB>>2), ((GN + (4<<LB)-1)>>LB>>2), Batch),\
			 dim3(1<<LB, 1<<LB), 0, stream>>>\
			(A, B, C, N, CM, BK, AK, Yindex, Xindex)

#define bmm_k22_p(stream, LB, Yindex, Xindex, A, B, C, Batch, N, CM, BK, AK, GN, GM)\
	bmm_kernel_2_2_padding<LB, (1<<LB), MOVE_A, MOVE_B>\
		<<< dim3(((GM + (2<<LB)-1)>>LB>>1), ((GN + (2<<LB)-1)>>LB>>1), Batch),\
			 dim3(1<<LB, 1<<LB), 0, stream>>>\
			(A, B, C, N, CM, BK, AK, Yindex, Xindex)

#endif


//(Y: BLOCK_SIZE * 8, X: BLOCK_SIZE * 8), K >= (BLOCK_SIZE/2)
#ifndef BATCH_MATMUL_KERNEL_8_8_PADDING_TEXTURE
#define BATCH_MATMUL_KERNEL_8_8_PADDING_TEXTURE

//LB = 4: Size = 0.9767, Time = 1.508 msec, Performace = 1390.88 GFlop/s
//LB = 3: Size = 0.9767, Time = 1.614 msec, Performace = 1299.53 GFlop/s
template<int LB, int STEP, int MOVE_A, int MOVE_B>
__global__ void bmm_kernel_8_8_padding_texture(
	cudaTextureObject_t       A, //A[Batch,  N, AK], Ak is memAlgined
	const float* __restrict__ B, //B[Batch, BK,  M], Bk is not memAligned
		  float* __restrict__ C, //C[Batch,  N,  M]
	int N, int CM, int BK, int AK,
	int Yindex, int Xindex)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 As[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Bs[2][1 << LB][(1 << LB) + 1];

	//prepared for X
	const int X = (((blockIdx.x << LB) + tx) << 3) + Xindex;
	const int tX = X + ((ty >= STEP) << 2);

	//compute start offset of A, B, C
	const int batch = blockIdx.z;
	const int ABatch = (batch * N * AK * MOVE_A);//A[batch * MOVE_A]
	B += (batch * BK * CM * MOVE_B) + tX;//B[batch * MOVE_B, 0, tX]
	C += (batch * N * CM);//C[batch]

	//prepared for A -> Y:N
	const int Y = (((blockIdx.y << LB) + ty) << 3) + Yindex;
	const int tY0 = (Y + ((tx >= STEP) << 2)) * AK + ABatch;
	const int tY1 = tY0 + AK, tY2 = tY1 + AK, tY3 = tY2 + AK;

	//load 4 elements from B[batch]
	const int Xend = (CM - 4);//float4: X <= M - 4
	int Bk = ty - ((ty >= STEP) << LB >> 1);
	Bs[buf][ty][tx] = (tX <= Xend ? *(float4*)(&B[Bk * CM]) : FZERO4);

	//load 4 elements from A[batch]
	const int Yend = (N - 1) * AK + ABatch;
	int Ak = tx - ((tx >= STEP) << LB >> 1);
	As[buf][tx][ty].x = (tY0 <= Yend) * tex1Dfetch<float>(A, tY0 + Ak);
	As[buf][tx][ty].y = (tY1 <= Yend) * tex1Dfetch<float>(A, tY1 + Ak);
	As[buf][tx][ty].z = (tY2 <= Yend) * tex1Dfetch<float>(A, tY2 + Ak);
	As[buf][tx][ty].w = (tY3 <= Yend) * tex1Dfetch<float>(A, tY3 + Ak);
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
	for (int ok = 1, OK = (BK << 1 >> LB); ok < OK; ok++)
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

		//load 4 elements from B[batch]
		int Bk = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		Bs[buf][ty][tx] = (tX <= Xend ? *(float4*)(&B[Bk * CM]) : FZERO4);

		//load 4 elements from A[batch]
		int Ak = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		As[buf][tx][ty].x = (tY0 <= Yend) * tex1Dfetch<float>(A, tY0 + Ak);
		As[buf][tx][ty].y = (tY1 <= Yend) * tex1Dfetch<float>(A, tY1 + Ak);
		As[buf][tx][ty].z = (tY2 <= Yend) * tex1Dfetch<float>(A, tY2 + Ak);
		As[buf][tx][ty].w = (tY3 <= Yend) * tex1Dfetch<float>(A, tY3 + Ak);
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
	int Y0 = (Y * AK) + ABatch;
	int Y1 = Y0 + AK, Y2 = Y1 + AK, Y3 = Y2 + AK;
	int Y4 = Y3 + AK, Y5 = Y4 + AK, Y6 = Y5 + AK, Y7 = Y6 + AK;
	B += (X - tX);
	for (int k = BK - (BK&(STEP - 1)); k < BK; k++)
	{
		//load 8 elements from A[batch]
		float4 a0, a1;
		a0.x = (Y0 <= Yend) * tex1Dfetch<float>(A, Y0 + k);
		a0.y = (Y1 <= Yend) * tex1Dfetch<float>(A, Y1 + k);
		a0.z = (Y2 <= Yend) * tex1Dfetch<float>(A, Y2 + k);
		a0.w = (Y3 <= Yend) * tex1Dfetch<float>(A, Y3 + k);
		a1.x = (Y4 <= Yend) * tex1Dfetch<float>(A, Y4 + k);
		a1.y = (Y5 <= Yend) * tex1Dfetch<float>(A, Y5 + k);
		a1.z = (Y6 <= Yend) * tex1Dfetch<float>(A, Y6 + k);
		a1.w = (Y7 <= Yend) * tex1Dfetch<float>(A, Y7 + k);

		//load 8 elements from B[batch]
		float4 b0 = (X     <= Xend ? *(float4*)(&B[(k * CM)    ]) : FZERO4);
		float4 b1 = (X + 4 <= Xend ? *(float4*)(&B[(k * CM) + 4]) : FZERO4);

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

	Y0 = (Y * CM) + X;
	Y1 = Y0 + CM, Y2 = Y1 + CM, Y3 = Y2 + CM;
	Y4 = Y3 + CM, Y5 = Y4 + CM, Y6 = Y5 + CM, Y7 = Y6 + CM;

	int Cend = (N * CM) - 4;//float4: (N - 1)*M + (M - 4) = N*M - 4
	if (X <= Xend) {//float4: X <= M - 4
		if (Y0 <= Cend) { *(float4*)(C + Y0) = v0; }
		if (Y1 <= Cend) { *(float4*)(C + Y1) = v2; }
		if (Y2 <= Cend) { *(float4*)(C + Y2) = v4; }
		if (Y3 <= Cend) { *(float4*)(C + Y3) = v6; }
		if (Y4 <= Cend) { *(float4*)(C + Y4) = v8; }
		if (Y5 <= Cend) { *(float4*)(C + Y5) = v10; }
		if (Y6 <= Cend) { *(float4*)(C + Y6) = v12; }
		if (Y7 <= Cend) { *(float4*)(C + Y7) = v14; }
	}

	Cend = Cend - 4;//float8: (N - 1)*M + (M - 8) = N*M - 8
	if (X + 4 <= Xend) {//float8: X <= M - 8
		if (Y0 <= Cend) { *(float4*)(C + Y0 + 4) = v1; }
		if (Y1 <= Cend) { *(float4*)(C + Y1 + 4) = v3; }
		if (Y2 <= Cend) { *(float4*)(C + Y2 + 4) = v5; }
		if (Y3 <= Cend) { *(float4*)(C + Y3 + 4) = v7; }
		if (Y4 <= Cend) { *(float4*)(C + Y4 + 4) = v9; }
		if (Y5 <= Cend) { *(float4*)(C + Y5 + 4) = v11; }
		if (Y6 <= Cend) { *(float4*)(C + Y6 + 4) = v13; }
		if (Y7 <= Cend) { *(float4*)(C + Y7 + 4) = v15; }
	}
}

#endif


//(Y: BLOCK_SIZE * 8, X: BLOCK_SIZE * 8), K >= (BLOCK_SIZE/2)
#ifndef BATCH_MATMUL_KERNEL_8_8_PADDING
#define BATCH_MATMUL_KERNEL_8_8_PADDING

//LB = 4: Size = 1, Time = 1.49 msec, Performace = 1441.26 GFlop/s
//LB = 4: Size = 0.9767, Time = 1.526 msec, Performace = 1374.47 GFlop/s
//LB = 3: Size = 0.9767, Time = 1.726 msec, Performace = 1215.21 GFlop/s
template<int LB, int STEP, int MOVE_A, int MOVE_B>
__global__ void bmm_kernel_8_8_padding(
	const float* __restrict__ A, //A[Batch, N, AK], Ak is memAlgined
	const float* __restrict__ B, //B[Batch, BK, M], Bk is not memAligned
		  float* __restrict__ C, //C[Batch, N,  M]
	int N, int CM, int BK, int AK,
	int Yindex, int Xindex)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 As[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Bs[2][1 << LB][(1 << LB) + 1];

	//prepared for A -> Y:N
	const int Y = (((blockIdx.y << LB) + ty) << 3) + Yindex;
	const int tY0 = (Y + ((tx >= STEP) << 2)) * AK;
	const int tY1 = tY0 + AK, tY2 = tY1 + AK, tY3 = tY2 + AK;

	//prepared for B -> X:M
	const int X = (((blockIdx.x << LB) + tx) << 3) + Xindex;
	const int tX = X + ((ty >= STEP) << 2);

	//compute start offset of A, B, C
	const int batch = blockIdx.z;
	A += (batch * N * AK * MOVE_A);//A[batch * MOVE_A]
	B += (batch * BK * CM * MOVE_B) + tX;//B[batch * MOVE_B, 0, tX]
	C += (batch * N * CM);//C[batch]
	const int ye = (N - 1) * AK;
	const int xe = (CM - 4);//float4: X <= M - 4 

	//load 4 elements from A[batch]
	float4 av; int Ak = tx - ((tx >= STEP) << LB >> 1);
	av.x = (tY0 <= ye ? A[tY0 + Ak] : 0);
	av.y = (tY1 <= ye ? A[tY1 + Ak] : 0);
	av.z = (tY2 <= ye ? A[tY2 + Ak] : 0);
	av.w = (tY3 <= ye ? A[tY3 + Ak] : 0);
	As[buf][tx][ty] = av;

	//load 4 elements from B[batch]
	int Bk = ty - ((ty >= STEP) << LB >> 1);
	Bs[buf][ty][tx] = (tX <= xe ? *(float4*)(B + Bk * CM) : F32_4_0);
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

	for (int ok = 1, OK = (BK << 1 >> LB); ok < OK; ok++)
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

		//load 4 elements from A[batch]
		float4 av; int Ak = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		av.x = (tY0 <= ye ? A[tY0 + Ak] : 0);
		av.y = (tY1 <= ye ? A[tY1 + Ak] : 0);
		av.z = (tY2 <= ye ? A[tY2 + Ak] : 0);
		av.w = (tY3 <= ye ? A[tY3 + Ak] : 0);
		As[buf][tx][ty] = av;

		//load 4 elements from B[batch]
		int Bk = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		Bs[buf][ty][tx] = (tX <= xe ? *(float4*)(B + Bk * CM) : F32_4_0);
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

	//when GK % STEP!=0----------------------------------------------
	int Y0 = (Y * AK);
	int Y1 = Y0 + AK, Y2 = Y1 + AK, Y3 = Y2 + AK;
	int Y4 = Y3 + AK, Y5 = Y4 + AK, Y6 = Y5 + AK, Y7 = Y6 + AK;
	B += (X - tX);
	for (int k = BK - (BK&(STEP - 1)); k < BK; k++)
	{
		//load 8 elements from A[batch]
		float4 a0, a1;
		a0.x = (Y0 <= ye ? A[Y0 + k] : 0);
		a0.y = (Y1 <= ye ? A[Y1 + k] : 0);
		a0.z = (Y2 <= ye ? A[Y2 + k] : 0);
		a0.w = (Y3 <= ye ? A[Y3 + k] : 0);
		a1.x = (Y4 <= ye ? A[Y4 + k] : 0);
		a1.y = (Y5 <= ye ? A[Y5 + k] : 0);
		a1.z = (Y6 <= ye ? A[Y6 + k] : 0);
		a1.w = (Y7 <= ye ? A[Y7 + k] : 0);

		//load 8 elements from B[batch]
		float4 b0 = (X <= xe ? *(float4*)(&B[(k * CM)]) : FZERO4);
		float4 b1 = (X + 4 <= xe ? *(float4*)(&B[(k * CM) + 4]) : FZERO4);

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

	const int C0 = (Y * CM) + X;
	const int C1 = C0 + CM, C2 = C1 + CM, C3 = C2 + CM;
	const int C4 = C3 + CM, C5 = C4 + CM, C6 = C5 + CM, C7 = C6 + CM;

	int ce = (N * CM) - 4;//float4: (N - 1)*M + (M - 4) = N*M - 4
	if (X <= xe) {//float4: X <= M - 4
		if (C0 <= ce) { *(float4*)(C + C0) = v0; }
		if (C1 <= ce) { *(float4*)(C + C1) = v2; }
		if (C2 <= ce) { *(float4*)(C + C2) = v4; }
		if (C3 <= ce) { *(float4*)(C + C3) = v6; }
		if (C4 <= ce) { *(float4*)(C + C4) = v8; }
		if (C5 <= ce) { *(float4*)(C + C5) = v10; }
		if (C6 <= ce) { *(float4*)(C + C6) = v12; }
		if (C7 <= ce) { *(float4*)(C + C7) = v14; }
	}

	ce = ce - 4;//float8: (N - 1)*M + (M - 8) = N*M - 8
	if (X + 4 <= xe) {//float8: X <= M - 8
		if (C0 <= ce) { *(float4*)(C + C0 + 4) = v1; }
		if (C1 <= ce) { *(float4*)(C + C1 + 4) = v3; }
		if (C2 <= ce) { *(float4*)(C + C2 + 4) = v5; }
		if (C3 <= ce) { *(float4*)(C + C3 + 4) = v7; }
		if (C4 <= ce) { *(float4*)(C + C4 + 4) = v9; }
		if (C5 <= ce) { *(float4*)(C + C5 + 4) = v11; }
		if (C6 <= ce) { *(float4*)(C + C6 + 4) = v13; }
		if (C7 <= ce) { *(float4*)(C + C7 + 4) = v15; }
	}
}

#endif


//(Y: BLOCK_SIZE * 4, X: BLOCK_SIZE * 4)
#ifndef BATCH_MATMUL_KERNEL_4_4_PADDING
#define BATCH_MATMUL_KERNEL_4_4_PADDING

//LB = 4: Size = 0.9767, Time = 2.118 msec, Performace = 990.296 GFlop/s
//LB = 3: Size = 0.9767, Time = 2.53  msec, Performace = 829.03 GFlop/s
template<int LB, int STEP, int MOVE_A, int MOVE_B>
__global__ void bmm_kernel_4_4_padding(
	const float* __restrict__ A, //A[Batch, N, K]
	const float* __restrict__ B, //B[Batch, K, M]
	      float* __restrict__ C, //C[Batch, N, M]
	int N, int CM, int BK, int AK,
	int Yindex, int Xindex)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 As[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2 Bs[2][1 << LB >> 1][(2 << LB) + 2];

	//prepared for A -> Y:N
	const int Y = (((blockIdx.y << LB) + ty) << 2) + Yindex;
	const int tY0 = (Y + ((tx & 1) << 1)) * AK, tY1 = tY0 + AK;

	//prepared for B -> X:M
	const int X = (((blockIdx.x << LB) + tx) << 2) + Xindex;
	const int tX = X + ((ty & 1) << 1);

	//compute start offset of A, B, C
	const int batch = blockIdx.z;
	A += (batch * N * AK * MOVE_A);//A[batch * MOVE_A]
	B += (batch * BK * CM * MOVE_B) + tX;//B[batch * MOVE_B, 0, tX]
	C += (batch * N * CM);//C[batch]
	
	const int OK = (BK << 1 >> LB);
	const int Yend = (N - 1) * AK;
	const int Xend = (CM - 2);//float2: X <= M - 2
	const int As_x = (tx >> 1), As_y = (ty << 1) + (tx & 1);
	const int Bs_y = (ty >> 1), Bs_x = (tx << 1) + (ty & 1);
	if (OK) {
		int Ak = (tx >> 1);//load 2 elements from A[batch]
		As[buf][As_x][As_y].x = (tY0 <= Yend ? A[tY0 + Ak] : 0);
		As[buf][As_x][As_y].y = (tY1 <= Yend ? A[tY1 + Ak] : 0);

		int Bk = (ty >> 1);//load 2 elements from B[batch]
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

		int Ak = ((ok << LB) + tx) >> 1;//load 2 elements from A[batch]
		As[buf][As_x][As_y].x = (tY0 <= Yend ? A[tY0 + Ak] : 0);
		As[buf][As_x][As_y].y = (tY1 <= Yend ? A[tY1 + Ak] : 0);

		int Bk = ((ok << LB) + ty) >> 1;//load 2 elements from B[batch]
		Bs[buf][Bs_y][Bs_x] = (tX <= Xend ? *(float2*)(&B[Bk * CM]) : FZERO2);
		__syncthreads();
	}
	if (OK) {
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float4 b = *(float4*)(&Bs[buf][ik][(tx << 1)]);
			float4 a = *(float4*)(&As[buf][ik][(ty << 1)]);
			simdMM4(v0, a.x, b);
			simdMM4(v1, a.y, b);
			simdMM4(v2, a.z, b);
			simdMM4(v3, a.w, b);
		}
	}

	//when GK % STEP!=0----------------------------------------------
	int Y0 = (Y * AK);
	int Y1 = Y0 + AK, Y2 = Y1 + AK, Y3 = Y2 + AK;
	B += (X - tX);
	for (int k = BK - (BK&(STEP - 1)); k < BK; k++)
	{
		float4 a;//load 4 elements from A[batch]
		a.x = (Y0 <= Yend ? A[Y0 + k] : 0);
		a.y = (Y1 <= Yend ? A[Y1 + k] : 0);
		a.z = (Y2 <= Yend ? A[Y2 + k] : 0);
		a.w = (Y3 <= Yend ? A[Y3 + k] : 0);

		//load 4 elements from B[batch]
		float4 b = (X <= Xend ? *(float4*)(&B[k * CM]) : FZERO4);

		simdMM4(v0, a.x, b);
		simdMM4(v1, a.y, b);
		simdMM4(v2, a.z, b);
		simdMM4(v3, a.w, b);
	}
	//when GK % STEP!=0----------------------------------------------

	Y0 = (Y * CM) + X;
	Y1 = Y0 + CM, Y2 = Y1 + CM, Y3 = Y2 + CM;

	//if X <= M - 4, (Y * M) + X <= Cend, we have: Y <= N -1
	int Cend = (N * CM) - 4;//float4: (N - 1)*M + (M - 4) = N*M - 4
	if (X <= (CM - 4)) {//float4: X <= M - 4
		if (Y0 <= Cend) { *(float4*)(C + Y0) = v0; }
		if (Y1 <= Cend) { *(float4*)(C + Y1) = v1; }
		if (Y2 <= Cend) { *(float4*)(C + Y2) = v2; }
		if (Y3 <= Cend) { *(float4*)(C + Y3) = v3; }
	}
}

#endif


//(Y: BLOCK_SIZE * 2, X: BLOCK_SIZE * 2)
#ifndef BATCH_MATMUL_KERNEL_2_2_PADDING
#define BATCH_MATMUL_KERNEL_2_2_PADDING

//LB = 4: Size = 0.9767, Time = 3.696 msec, Performace = 567.491 GFlop/s
//LB = 3: Size = 0.9767, Time = 4.666 msec, Performace = 449.517 GFlop/s
template<int LB, int STEP, int MOVE_A, int MOVE_B>
__global__ void bmm_kernel_2_2_padding(
	const float* __restrict__ A, //A[Batch,  N, AK], Ak is memAlgined
	const float* __restrict__ B, //B[Batch, BK,  M], Bk is not memAligned
		  float* __restrict__ C, //C[Batch,  N,  M]
	int N, int CM, int BK, int AK,
	int Yindex, int Xindex)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 As[2][1 << LB][(1 << LB) + 1];
	__shared__ float2 Bs[2][1 << LB][(1 << LB) + 1];

	//prepared for A -> Y:N
	const int Y = (((blockIdx.y << LB) + ty) << 1) + Yindex;
	const int tY0 = Y * AK, tY1 = tY0 + AK;

	//prepared for B -> X:M
	const int X = (((blockIdx.x << LB) + tx) << 1) + Xindex;

	//compute start offset of A, B, C
	const int batch = blockIdx.z;
	A += (batch * N * AK * MOVE_A);//A[batch * MOVE_A]
	B += (batch * BK * CM * MOVE_B) + X;//B[batch * MOVE_B, 0, tX]
	C += (batch * N * CM);//C[batch]

	const int Yend = (N - 1) * AK;
	const int Xend = (CM - 2);//use float2: M - 2
	const int OK = (BK >> LB);
	if (OK) {
		int Ak = tx;//load 2 elements from A[batch]
		As[buf][tx][ty].x = (tY0 <= Yend ? A[tY0 + Ak] : 0);
		As[buf][tx][ty].y = (tY1 <= Yend ? A[tY1 + Ak] : 0);

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

		int Ak = (ok << LB) + tx;//load 2 elements from A[batch]
		As[buf][tx][ty].x = (tY0 <= Yend ? A[tY0 + Ak] : 0);
		As[buf][tx][ty].y = (tY1 <= Yend ? A[tY1 + Ak] : 0);

		int Bk = (ok << LB) + ty;//load 2 elements from B[batch]
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
	for (int k = BK - (BK&(STEP - 1)); k < BK; k++) {
		float2 a;//load 2 elements from A[batch]
		a.x = (tY0 <= Yend ? A[tY0 + k] : 0);
		a.y = (tY1 <= Yend ? A[tY1 + k] : 0);

		//load 2 elements from B[batch]
		float2 b = (X <= Xend ? *(float2*)(&B[k * CM]) : FZERO2);

		simdMM2(v0, a.x, b);
		simdMM2(v1, a.y, b);
	}
	//when GK % STEP!=0----------------------------------------------

	int Y0 = (Y * CM) + X, Y1 = Y0 + CM;
	int Cend = (N * CM) - 2;//float2: (N - 1)*M + (M - 2) = N*M - 2

	if (X <= Xend) {
		if (Y0 <= Cend) { *(float2*)(C + Y0) = v0; }
		if (Y1 <= Cend) { *(float2*)(C + Y1) = v1; }
	}
}

#endif

#endif