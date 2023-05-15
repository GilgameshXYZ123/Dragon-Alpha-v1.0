#pragma once

#ifndef BATCH_MATMUL_T2_KERNEL_PADDING_H
#define BATCH_MATMUL_T2_KERNEL_PADDING_H

//A[Batch,  N,  K] 
//B[Batch, BM,  K] logically-> B^T[Batch, K, M] 
//C[Batch,  N, CM]
//(1) K % 4 == 0
//(2) M = CM: BM % 4 != 0, CM % 4 == 0, CM >= BM, CM = (BM + 3) >> 2 << 2
//(3) N % 4 != 0
//(4) M = (M + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE
//(5) N = (N + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE
//N % 4 != 0: for Y axis: we can't use float4
//M = CM % 4 == 0: for X axis: we can   use float4
#ifndef BATCH_MATMUL_T2_KERNEL_PADDING_CALL
#define BATCH_MATMUL_T2_KERNEL_PADDING_CALL

//LB = log2(BLOCK_SIZE)
//MOVE_A == 0: A is a 2D tensor[N, K], logically expand A to[Batch, N, K]
//MOVE_B == 0: B is a 2D tensor[M, K], logically expand B to[Barch, M, K]

#define bmmT2_k88_ptex(stream, LB, Yindex, Xindex, A, B, C, Batch, N, CM, BM, K, GN, GM)\
	bmmT2_kernel_8_8_padding_texture<LB, (1<<LB>>1), MOVE_A, MOVE_B>\
		<<< dim3(((GM + (8<<LB)-1)>>LB>>3), ((GN + (8<<LB)-1)>>LB>>3), Batch),\
            dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, N, CM, BM, K, Yindex, Xindex)

#define bmmT2_k88_p(stream, LB, Yindex, Xindex, A, B, C, Batch, N, CM, BM, K, GN, GM)\
	bmmT2_kernel_8_8_padding<LB, (1<<LB>>1), MOVE_A, MOVE_B>\
		<<< dim3(((GM + (8<<LB)-1)>>LB>>3), ((GN + (8<<LB)-1)>>LB>>3), Batch),\
            dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, N, CM, BM, K, Yindex, Xindex)

#define bmmT2_k44_p(stream, LB, Yindex, Xindex, A, B, C, Batch, N, CM, BM, K, GN, GM)\
	bmmT2_kernel_4_4_padding<LB, (1<<LB>>1), MOVE_A, MOVE_B>\
		<<< dim3(((GM + (4<<LB)-1)>>LB>>2), ((GN + (4<<LB)-1)>>LB>>2), Batch),\
            dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, N, CM, BM, K, Yindex, Xindex)

#define bmmT2_k22_p(stream, LB, Yindex, Xindex, A, B, C, Batch, N, CM, BM, K, GN, GM)\
	bmmT2_kernel_2_2_padding<LB, (1<<LB), MOVE_A, MOVE_B>\
		<<< dim3(((GM + (2<<LB)-1)>>LB>>1), ((GN + (2<<LB)-1)>>LB>>1), Batch),\
            dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, N, CM, BM, K, Yindex, Xindex)

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), K % (BLOCK_SIZE/2) == 0
//LB = 4: K % 8 == 0
//LB = 3: K % 4 == 0
#ifndef BATCH_MATMUL_T2_KERNEL_8_8_PADDING_TEXTURE
#define BATCH_MATMUL_T2_KERNEL_8_8_PADDING_TEXTURE

//LB = 4: Size = 0.968994, Time = 1.946 msec, Performace = 1069.32 GFlop/s
//LB = 3: Size = 0.968994, Time = 2.02  msec, Performace = 1030.15 GFlop/s
template<int LB, int STEP, int MOVE_A, int MOVE_B>
__global__ void bmmT2_kernel_8_8_padding_texture(
	cudaTextureObject_t A, //A[Batch,  N,  K]
	cudaTextureObject_t B, //B[Batch, BM,  K], BM is not memAligned
	float* __restrict__ C, //C[Batch,  N, CM], CM is memAligned
	int N, int CM, int BM, int K,
	int Yindex, int Xindex)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 As[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Bs[2][1 << LB][(1 << LB) + 1];

	//prepared for A -> Y:N
	int Y = (((blockIdx.y << LB) + ty) << 3) + Yindex;
	const int Y0 = (Y + ((tx >= STEP) << 2)) * K;
	const int Y1 = Y0 + K, Y2 = Y1 + K, Y3 = Y2 + K;

	//prepared for B -> X:M
	const int X = (((blockIdx.x << LB) + tx) << 3) + Xindex;
	const int X0 = (X + ((ty >= STEP) << 2)) * K;
	const int X1 = X0 + K, X2 = X1 + K, X3 = X2 + K;

	//compute start offset of A, B, C
	const int batch = blockIdx.z;
	const int Aoffset = (batch *  N * K * MOVE_A);//A[batch * MOVE_A]
	const int Boffset = (batch * BM * K * MOVE_B);//B[batch * MOVE_B]
	C += (batch * N * CM);//C[batch]
	const int ye = (N - 1) * K;
	const int xe = (BM - 1) * K;

	//load 4 elements from A[batch]
	float4 av; int Ak = tx - ((tx >= STEP) << LB >> 1) + Aoffset;
	av.x = tex1Dfetch<float>(A, Y0 + Ak);
	av.y = tex1Dfetch<float>(A, Y1 + Ak);
	av.z = tex1Dfetch<float>(A, Y2 + Ak);
	av.w = tex1Dfetch<float>(A, Y3 + Ak);
	zero_float(av.x, (Y0 <= ye), av.x);
	zero_float(av.y, (Y1 <= ye), av.y);
	zero_float(av.z, (Y2 <= ye), av.z);
	zero_float(av.w, (Y3 <= ye), av.w);
	As[buf][tx][ty] = av;

	//load 4 elements from B[batch]
	float4 bv; int Bk = ty - ((ty >= STEP) << LB >> 1) + Boffset;
	bv.x = tex1Dfetch<float>(B, X0 + Bk);
	bv.y = tex1Dfetch<float>(B, X1 + Bk);
	bv.z = tex1Dfetch<float>(B, X2 + Bk);
	bv.w = tex1Dfetch<float>(B, X3 + Bk);
	zero_float(bv.x, (X0 <= xe), bv.x);
	zero_float(bv.y, (X1 <= xe), bv.y);
	zero_float(bv.z, (X2 <= xe), bv.z);
	zero_float(bv.w, (X3 <= xe), bv.w);
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
		float4 av; int Ak = ((ok - (tx >= STEP)) << LB >> 1) + tx + Aoffset;
		av.x = tex1Dfetch<float>(A, Y0 + Ak);
		av.y = tex1Dfetch<float>(A, Y1 + Ak);
		av.z = tex1Dfetch<float>(A, Y2 + Ak);
		av.w = tex1Dfetch<float>(A, Y3 + Ak);
		zero_float(av.x, (Y0 <= ye), av.x);
		zero_float(av.y, (Y1 <= ye), av.y);
		zero_float(av.z, (Y2 <= ye), av.z);
		zero_float(av.w, (Y3 <= ye), av.w);
		As[buf][tx][ty] = av;

		//load 4 elements from B[batch]
		float4 bv; int Bk = ((ok - (ty >= STEP)) << LB >> 1) + ty + Boffset;
		bv.x = tex1Dfetch<float>(B, X0 + Bk);
		bv.y = tex1Dfetch<float>(B, X1 + Bk);
		bv.z = tex1Dfetch<float>(B, X2 + Bk);
		bv.w = tex1Dfetch<float>(B, X3 + Bk);
		zero_float(bv.x, (X0 <= xe), bv.x);
		zero_float(bv.y, (X1 <= xe), bv.y);
		zero_float(bv.z, (X2 <= xe), bv.z);
		zero_float(bv.w, (X3 <= xe), bv.w);
		Bs[buf][ty][tx] = bv;
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

	int C0 = (Y * CM) + X;
	int C1 = C0 + CM, C2 = C1 + CM, C3 = C2 + CM;
	int C4 = C3 + CM, C5 = C4 + CM, C6 = C5 + CM, C7 = C6 + CM;

	int ce = (N * CM) - 4;//float4: (N - 1)*M + (M - 4) = N*M - 4
	if (X <= (CM - 4)) {//float4: X <= M - 4
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
	if (X <= (CM - 8)) {//float8: X <= M - 8
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


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), K % (BLOCK_SIZE/2) == 0
//LB = 4: K % 8 == 0
//LB = 3: K % 4 == 0
#ifndef BATCH_MATMUL_T2_KERNEL_8_8_PADDING
#define BATCH_MATMUL_T2_KERNEL_8_8_PADDING

//LB = 4: Size = 0.968994, Time = 2.346 msec, Performace =  886.999 GFlop/s
//LB = 3: Size = 0.968994, Time = 1.916 msec, Performace = 1086.06 GFlop/s
template<int LB, int STEP, int MOVE_A, int MOVE_B>
__global__ void bmmT2_kernel_8_8_padding(
	const float* __restrict__ A, //A[Batch,  N,  K]
	const float* __restrict__ B, //B[Batch, BM,  K], BM is not memAligned
	      float* __restrict__ C, //C[Batch,  N, CM], CM is memAligned
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
	const int X0 = (X + ((ty >= STEP) << 2)) * K;
	const int X1 = X0 + K, X2 = X1 + K, X3 = X2 + K;

	//compute start offset of A, B, C
	const int batch = blockIdx.z;
	A += (batch *  N * K * MOVE_A);//A[batch * MOVE_A]
	B += (batch * BM * K * MOVE_B);//B[batch * MOVE_B]
	C += (batch * N * CM);//C[batch]

	const int ye = (N - 1) * K;
	const int xe = (BM - 1) * K;

	//load 4 elements from A[batch]
	float4 av; int Ak = tx - ((tx >= STEP) << LB >> 1);
	av.x = (Y0 <= ye ? A[Y0 + Ak] : 0);
	av.y = (Y1 <= ye ? A[Y1 + Ak] : 0);
	av.z = (Y2 <= ye ? A[Y2 + Ak] : 0);
	av.w = (Y3 <= ye ? A[Y3 + Ak] : 0);
	As[buf][tx][ty] = av;

	//load 4 elements from B[batch]
	float4 bv; int Bk = ty - ((ty >= STEP) << LB >> 1);
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
		av.x = (Y0 <= ye ? A[Y0 + Ak] : 0);
		av.y = (Y1 <= ye ? A[Y1 + Ak] : 0);
		av.z = (Y2 <= ye ? A[Y2 + Ak] : 0);
		av.w = (Y3 <= ye ? A[Y3 + Ak] : 0);
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

	const int C0 = (Y * CM) + X;
	const int C1 = C0 + CM, C2 = C1 + CM, C3 = C2 + CM;
	const int C4 = C3 + CM, C5 = C4 + CM, C6 = C5 + CM, C7 = C6 + CM;

	int ce = (N * CM) - 4;//float4: (N - 1)*M + (M - 4) = N*M - 4
	if (X <= (CM - 4)) {//float4: X <= M - 4
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
	if (X <= (CM - 8)) {//float8: X <= M - 8
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


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*4), K % (BLOCK_SIZE/2) == 0
//LB = 4: K % 8 == 0
//LB = 3: K % 4 == 0
#ifndef BATCH_MATMUL_T2_KERNEL_4_4_PADDING
#define BATCH_MATMUL_T2_KERNEL_4_4_PADDING

//LB = 4: Size = 1, Time = 2.076 msec, Performace = 1034.43  GFlop/s
//LB = 3: Size = 1, Time = 3.678 msec, Performace =  583.873 GFlop/s
template<int LB, int STEP, int MOVE_A, int MOVE_B>
__global__ void bmmT2_kernel_4_4_padding(
	const float* __restrict__ A, //A[Batch,  N,  K]
	const float* __restrict__ B, //B[Batch, BM,  K], BM is not memAligned
	      float* __restrict__ C, //C[Batch,  N, CM], CM is memAligned
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
	C += (batch * N * CM);//C[batch]

	const int As_x = (tx >> 1), As_y = (ty << 1) + (tx & 1);
	const int Bs_y = (ty >> 1), Bs_x = (tx << 1) + (ty & 1);
	const int ye = (N - 1) * K;
	const int xe = (BM - 1) * K;

	//load 2 elements from A[batch]
	float2 av; int Ak = (tx >> 1);
	av.x = (Y0 <= ye ? A[Y0 + Ak] : 0);
	av.y = (Y1 <= ye ? A[Y1 + Ak] : 0);
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
		av.x = (Y0 <= ye ? A[Y0 + Ak] : 0);
		av.y = (Y1 <= ye ? A[Y1 + Ak] : 0);
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


//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*2)
#ifndef BATCH_MATMUL_T2_KERNEL_2_2_PADDING
#define BATCH_MATMUL_T2_KERNEL_2_2_PADDING

//LB = 4: Size = 0.965209, Time = 3.868 msec, Performace = 535.877 GFlop/s
//LB = 3: Size = 0.965209, Time = 5.2   msec, Performace = 398.61 GFlop/s
template<int LB, int STEP, int MOVE_A, int MOVE_B>
__global__ void bmmT2_kernel_2_2_padding(
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

	//prepared for A -> Y:N
	const int Y = (((blockIdx.y << LB) + ty) << 1) + Yindex;
	const int tY0 = Y * K, tY1 = tY0 + K;

	//prepared for B -> X:M
	const int X = (((blockIdx.x << LB) + tx) << 1) + Xindex;
	const int tX0 = X * K, tX1 = tX0 + K;

	//compute start offset of A, B, C
	const int batch = blockIdx.z;
	A += (batch * N *  K * MOVE_A);//A[batch * MOVE_A]
	B += (batch * K * BM * MOVE_B);//B[batch * MOVE_B]
	C += (batch * N * CM);//C[batch]

	const int Yend = (N  - 1) * K;
	const int Xend = (BM - 1) * K;
	const int OK = (K >> LB);
	if (OK) {
		int Ak = tx;//load 2 elements from A[batch]
		As[buf][tx][ty].x = (tY0 <= Yend ? A[tY0 + Ak] : 0);
		As[buf][tx][ty].y = (tY1 <= Yend ? A[tY1 + Ak] : 0);

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

		//load 2 elements from A[batch]
		int Ak = (ok << LB) + tx;
		As[buf][tx][ty].x = (tY0 <= Yend ? A[tY0 + Ak] : 0);
		As[buf][tx][ty].y = (tY1 <= Yend ? A[tY1 + Ak] : 0);

		//load 2 elements from B[batch]
		int Bk = (ok << LB) + ty;
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
		a.x = (tY0 <= Yend ? A[tY0 + k] : 0);
		a.y = (tY1 <= Yend ? A[tY1 + k] : 0);

		float2 b;//load 2 elements from B[batch]
		b.x = (tX0 <= Xend ? B[tX0 + k] : 0);
		b.y = (tX1 <= Xend ? B[tX1 + k] : 0);

		simdMM2(v0, a.x, b);
		simdMM2(v1, a.y, b);
	}
	//when GK % STEP!=0----------------------------------------------

	int Y0 = (Y * CM) + X, Y1 = Y0 + CM;
	int Cend = (N * CM) - 2;//float2: (N - 1)*M + (M - 2) = N*M - 2

	if (X <= (CM - 2)) {//float2: X <= CM - 2
		if (Y0 <= Cend) { *(float2*)(C + Y0) = v0; }
		if (Y1 <= Cend) { *(float2*)(C + Y1) = v1; }
	}
}

#endif

#endif

