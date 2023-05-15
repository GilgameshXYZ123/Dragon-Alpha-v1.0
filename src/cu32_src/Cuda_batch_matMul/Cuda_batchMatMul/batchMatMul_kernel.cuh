#pragma once

#ifndef BATCH_MATMUL_KERNEL_H
#define BATCH_MATMUL_KERNEL_H

//A[Batch,  N, AK] 
//B[Batch, BK,  M]
//C[Batch,  N,  M]
//(1) M % 4 == 0
//(2) N % 4 != 0
//(3) K = BK: AK % 4 == 0, BK % 4 != 0, AK >= BK, AK = (BK + 3) >> 2 << 2
#ifndef BATCH_MATMUL_KERNEL_CALL
#define BATCH_MATMUL_KERNEL_CALL

//LB = log2(BLOCK_SIZE)
//MOVE_A == 0: A is a 2D tensor[N, K], logically expand A to[Batch, N, K]
//MOVE_B == 0: B is a 2D tensor[K, M], logically expand B to[Barch, K, M]

//======[Common]==============================================
#define bmm_k88_mk(stream, LB, Yindex, Xindex, A, B, C, Batch, N, CM, BK, AK, GN, GM)\
	bmm_kernel_8_8_MK<LB, (1<<LB>>1), MOVE_A, MOVE_B>\
		<<< dim3((GM>>LB>>3), (GN>>LB>>3), Batch),\
            dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, N, CM, BK, AK, Yindex, Xindex)

#define bmm_k88(stream, LB, Yindex, Xindex, A, B, C, Batch, N, CM, BK, AK, GN, GM)\
	bmm_kernel_8_8<LB, (1<<LB>>1), MOVE_A, MOVE_B>\
		<<< dim3((GM>>LB>>3), (GN>>LB>>3), Batch),\
            dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, N, CM, BK, AK, Yindex, Xindex)

#define bmm_k44(stream, LB, Yindex, Xindex, A, B, C, Batch, N, CM, BK, AK, GN, GM)\
	bmm_kernel_4_4<LB, (1<<LB>>1), MOVE_A, MOVE_B>\
		<<< dim3((GM>>LB>>2), (GN>>LB>>2), Batch),\
            dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, N, CM, BK, AK, Yindex, Xindex)

#define bmm_k82(stream, LB, Yindex, Xindex, A, B, C, Batch, N, CM, BK, AK, GN, GM)\
	bmm_kernel_8_2<LB, (1<<LB>>1), MOVE_A, MOVE_B>\
		<<< dim3((GM>>LB>>1), (GN>>LB>>3), Batch),\
            dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, N, CM, BK, AK, Yindex, Xindex)

#define bmm_k28(stream, LB, Yindex, Xindex, A, B, C, Batch, N, CM, BK, AK, GN, GM)\
	bmm_kernel_2_8<LB, (1<<LB>>1), MOVE_A, MOVE_B>\
		<<< dim3((GM>>LB>>3), (GN>>LB>>1), Batch),\
            dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, N, CM, BK, AK, Yindex, Xindex)

#define bmm_k42(stream, LB, Yindex, Xindex, A, B, C, Batch, N, CM, BK, AK, GN, GM)\
	bmm_kernel_4_2<LB, (1<<LB>>1), MOVE_A, MOVE_B>\
		<<< dim3((GM>>LB>>1), (GN>>LB>>2), Batch),\
	        dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, N, CM, BK, AK, Yindex, Xindex)

#define bmm_k24(stream, LB, Yindex, Xindex, A, B, C, Batch, N, CM, BK, AK, GN, GM)\
	bmm_kernel_2_4<LB, (1<<LB>>1), MOVE_A, MOVE_B>\
		<<< dim3((GM>>LB>>2), (GN>>LB>>1), Batch),\
            dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, N, CM, BK, AK, Yindex, Xindex)

//======[Small]================================================
#define bmm_k22(stream, LB, Yindex, Xindex, A, B, C, Batch, N, CM, BK, AK, GN, GM)\
	bmm_kernel_2_2<LB, (1<<LB), MOVE_A, MOVE_B>\
		<<< dim3((GM>>LB>>1), (GN>>LB>>1), Batch),\
            dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, N, CM, BK, AK, Yindex, Xindex)

#define bmm_k41(stream, LB, Yindex, Xindex, A, B, C, Batch, N, CM, BK, AK, GN, GM)\
	bmm_kernel_4_1<LB, (1<<LB), MOVE_A, MOVE_B>\
		<<< dim3((GM>>LB), (GN>>LB>>2), Batch),\
            dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, N, CM, BK, AK, Yindex, Xindex)

#define bmm_k14(stream, LB, Yindex, Xindex, A, B, C, Batch, N, CM, BK, AK, GN, GM)\
	bmm_kernel_1_4<LB, (1<<LB), MOVE_A, MOVE_B>\
		<<< dim3((GM>>LB>>2), (GN>>LB), Batch),\
            dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, N, CM, BK, AK, Yindex, Xindex)

#endif


//======[Common]==============================================
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), K % (BLOCK_SIZE/2) == 0
//LB = 4: K % 8 == 0
#ifndef BATCH_MATMUL_KERNEL_8_8_MK
#define BATCH_MATMUL_KERNEL_8_8_MK

//LB = 4: Size = 1, Time = 1.432 msec, Performace = 1499.64 GFlop/s
//LB = 3: Size = 1, Time = 1.544 msec, Performace = 1390.86 GFlop/s
template<int LB, int STEP, int MOVE_A, int MOVE_B>
__global__ void bmm_kernel_8_8_MK(
	const float* __restrict__ A, //A[Batch,  N, AK], Ak is memAlgined
	const float* __restrict__ B, //B[Batch, BK,  M], Bk is not memAligned
	float* __restrict__ C, //C[Batch,  N,  M]
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
	int batch = blockIdx.z;
	A += ((batch * N * AK) & (-MOVE_A));//A[batch * MOVE_A]
	B += ((batch * BK * CM) & (-MOVE_B)) + tX;//B[batch * MOVE_B, 0, tX]
	C += (batch * N * CM);//C[batch]

	//load 4 elements from A[batch]
	float4 av; int Ak = tx - ((tx >= STEP) << LB >> 1);
	av.x = A[tY0 + Ak];
	av.y = A[tY1 + Ak];
	av.z = A[tY2 + Ak];
	av.w = A[tY3 + Ak];
	As[buf][tx][ty] = av;

	//load 4 elements from B[batch]
	int Bk = ty - ((ty >= STEP) << LB >> 1);
	Bs[buf][ty][tx] = *(float4*)(B + Bk * CM);
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

	for (int ok = 1, OK = (BK << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) 
		{
			float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP][tx];
			float4 a0 = As[buf][ik][ty], a1 = As[buf][ik + STEP][ty];

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
		float4 av; int Ak = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		av.x = A[tY0 + Ak];
		av.y = A[tY1 + Ak];
		av.z = A[tY2 + Ak];
		av.w = A[tY3 + Ak];
		As[buf][tx][ty] = av;

		//load 4 elements from B[batch]
		int Bk = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		Bs[buf][ty][tx] = *(float4*)(B + Bk * CM);
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

	const int Y0 = (Y * CM) + X;
	const int Y1 = Y0 + CM, Y2 = Y1 + CM, Y3 = Y2 + CM;
	const int Y4 = Y3 + CM, Y5 = Y4 + CM, Y6 = Y5 + CM, Y7 = Y6 + CM;

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


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), K >= (BLOCK_SIZE/2)
//LB = 4: K >= 8
#ifndef BATCH_MATMUL_KERNEL_8_8
#define BATCH_MATMUL_KERNEL_8_8

//LB = 4: Size = 1, Time = 1.458 msec, Performace = 1472.9 GFlop/s
//LB = 3: Size = 1, Time = 1.64  msec, Performace = 1309.44 GFlop/s
//LB = 4: Size = 0.996094, Time = 1.484 msec, Performace = 1441.44 GFlop/s
//LB = 3: Size = 0.996094, Time = 1.646 msec, Performace = 1299.57 GFlop/s
template<int LB, int STEP, int MOVE_A, int MOVE_B>
__global__ void bmm_kernel_8_8(
	const float* __restrict__ A, //A[Batch,  N, AK], Ak is memAlgined
	const float* __restrict__ B, //B[Batch, BK,  M], Bk is not memAligned
		  float* __restrict__ C, //C[Batch,  N,  M]
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
	int batch = blockIdx.z;
	A += (batch * N * AK * MOVE_A);//A[batch * MOVE_A]
	B += (batch * BK * CM * MOVE_B) + tX;//B[batch * MOVE_B, 0, tX]
	C += (batch * N * CM);//C[batch]

	//load 4 elements from A[batch]
	float4 av; int Ak = tx - ((tx >= STEP) << LB >> 1);
	av.x = A[tY0 + Ak];
	av.y = A[tY1 + Ak];
	av.z = A[tY2 + Ak];
	av.w = A[tY3 + Ak];
	As[buf][tx][ty] = av;

	//load 4 elements from B[batch]
	int Bk = ty - ((ty >= STEP) << LB >> 1);
	Bs[buf][ty][tx] = *(float4*)(B + Bk * CM);
	__syncthreads();

	//compute area----------------------------------------------------
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
		float4 av; int Ak = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		av.x = A[tY0 + Ak];
		av.y = A[tY1 + Ak];
		av.z = A[tY2 + Ak];
		av.w = A[tY3 + Ak];
		As[buf][tx][ty] = av;

		//load 4 elements from B[batch]
		int Bk = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		Bs[buf][ty][tx] = *(float4*)(B + Bk * CM);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float4 a0 = As[buf][ik][ty], a1 = As[buf][ik + STEP][ty];
		float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP][tx];

		simdMM4(v0, a0.x, b0);  simdMM4(v1, a0.x, b1);
		simdMM4(v2, a0.y, b0);  simdMM4(v3, a0.y, b1);
		simdMM4(v4, a0.z, b0);  simdMM4(v5, a0.z, b1);
		simdMM4(v6, a0.w, b0);  simdMM4(v7, a0.w, b1);
		simdMM4(v8, a1.x, b0);  simdMM4(v9, a1.x, b1);
		simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
		simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
		simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
	}

	//when GK % STEP!=0----------------------------------------------
	A += Y * AK; B += (X - tX);
	for (int k = BK - (BK & (STEP - 1)); k < BK; k++)
	{
		//load 8 elements from A[batch]
		float4 a0 = { A[k], A[k + AK], A[k + (AK << 1)], A[k + (AK * 3)] };
		float4 a1 = { A[k + (AK << 2)], A[k + (AK * 5)], A[k + (AK * 6)], A[k + (AK * 7)] };
			
		//load 8 elements from B[batch]
		int boffset = k * CM;
		float4 b0 = *(float4*)(B + boffset);
		float4 b1 = *(float4*)(B + boffset + 4);

		simdMM4(v0, a0.x, b0);  simdMM4(v1, a0.x, b1);
		simdMM4(v2, a0.y, b0);  simdMM4(v3, a0.y, b1);
		simdMM4(v4, a0.z, b0);  simdMM4(v5, a0.z, b1);
		simdMM4(v6, a0.w, b0);  simdMM4(v7, a0.w, b1);
		simdMM4(v8, a1.x, b0);  simdMM4(v9, a1.x, b1);
		simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
		simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
		simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
	}
	//when GK % STEP!=0----------------------------------------------

	const int Y0 = (Y * CM) + X;
	const int Y1 = Y0 + CM, Y2 = Y1 + CM, Y3 = Y2 + CM;
	const int Y4 = Y3 + CM, Y5 = Y4 + CM, Y6 = Y5 + CM, Y7 = Y6 + CM;

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


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*4)
#ifndef BATCH_MATMUL_KERNEL_4_4
#define BATCH_MATMUL_KERNEL_4_4

//LB = 4: Size = 1, Time = 2.012 msec, Performace = 1067.34  GFlop/s
//LB = 3: Size = 1, Time = 2.272 msec, Performace =  945.195 GFlop/s
//LB = 4: Size = 0.996094, Time = 2.008 msec, Performace = 1065.29  GFlop/s
//LB = 3: Size = 0.996094, Time = 2.3   msec, Performace =  930.041 GFlop/s
template<int LB, int STEP, int MOVE_A, int MOVE_B>
__global__ void bmm_kernel_4_4(
	const float* __restrict__ A, //A[Batch, N, AK], Ak is memAlgined
	const float* __restrict__ B, //B[Batch, BK, M], Bk is not memAligned
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
	const int C0 = (batch*N + Y)*CM + X;//C[batch, Y, X]

	const int As_x = (tx >> 1), As_y = (ty << 1) + (tx & 1);
	const int Bs_y = (ty >> 1), Bs_x = (tx << 1) + (ty & 1);
	const int OK = (BK << 1 >> LB);
	if (OK) {
		//load 2 elements from A[batch]
		float2 av; int Ak = (tx >> 1);
		av.x = A[tY0 + Ak];
		av.y = A[tY1 + Ak];
		As[buf][As_x][As_y] = av;

		//load 2 elements from B[btch]
		int Bk = (ty >> 1);
		Bs[buf][Bs_y][Bs_x] = *(float2*)(B + Bk * CM);
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
		av.x = A[tY0 + Ak];
		av.y = A[tY1 + Ak];
		As[buf][As_x][As_y] = av;

		//load 2 elements from B[batch]
		int Bk = ((ok << LB) + ty) >> 1;
		Bs[buf][Bs_y][Bs_x] = *(float2*)(B + Bk * CM);
		__syncthreads();
	}
	if (OK) {
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
	}

	//when GK % STEP!=0----------------------------------------------
	A += Y * AK; B += (X - tX);
	for (int k = BK - (BK & (STEP - 1)); k < BK; k++)
	{
		//load 4 elements from A[batch]
		float4 a = { A[k], A[k + AK], A[k + (AK << 1)], A[k + (AK * 3)] };

		//load 4 elements from B[batch]
		float4 b = *(float4*)(B + k * CM);

		simdMM4(v0, a.x, b);
		simdMM4(v1, a.y, b);
		simdMM4(v2, a.z, b);
		simdMM4(v3, a.w, b);
	}
	//when GK % STEP!=0----------------------------------------------

	const int C1 = C0 + CM;
	const int C2 = C1 + CM;
	const int C3 = C2 + CM;

	*(float4*)(C + C0) = v0; 
	*(float4*)(C + C1) = v1; 
	*(float4*)(C + C2) = v2; 
	*(float4*)(C + C3) = v3;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE * 2)
#ifndef BATCH_MATMUL_KERNEL_8_2
#define BATCH_MATMUL_KERNEL_8_2

//LB = 4: Size = 1, Time = 2.132 msec, Performace = 1007.26 GFlop/s
//LB = 3: Size = 1, Time = 3.106 msec, Performace =  691.398 GFlop/s
//LB = 4: Size = 0.996094, Time = 2.176 msec, Performace = 983.04 GFlop/s
//LB = 3: Size = 0.996094, Time = 3.168 msec, Performace = 675.219 GFlop/s
template<int LB, int STEP, int MOVE_A, int MOVE_B>
__global__ void bmm_kernel_8_2(
	const float* __restrict__ A, //A[Batch,  N, AK], Ak is memAlgined
	const float* __restrict__ B, //B[Batch, BK,  M], Bk is not memAligned
		  float* __restrict__ C, //C[Batch,  N,  M]
	int N, int CM, int BK, int AK,
	int Yindex, int Xindex)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 As[2][1 << LB][(1 << LB) + 1];//k88
	__shared__ float  Bs[2][1 << LB >> 1][(2 << LB) + 2];//k42

	//prepared for A -> Y:N
	const int Y = (((blockIdx.y << LB) + ty) << 3) + Yindex;
	const int tY0 = (Y + ((tx >= STEP) << 2)) * AK;
	const int tY1 = tY0 + AK, tY2 = tY1 + AK, tY3 = tY2 + AK;

	//prepared for B -> X:M
	const int X = (((blockIdx.x << LB) + tx) << 1) + Xindex;
	const int tX = X + (ty & 1);

	//compute start offset of A, B, C
	const int batch = blockIdx.z;
	A += (batch * N * AK * MOVE_A);//A[batch * MOVE_A]
	B += (batch * BK * CM * MOVE_B) + tX;//B[batch * MOVE_B, 0, tX]
	C += (batch * N * CM);//C[batch]

	const int OK = (BK << 1 >> LB);
	const int Bs_y = (ty >> 1), Bs_x = (tx << 1) + (ty & 1);
	if (OK) {
		int Ak = tx - ((tx >= STEP) << LB >> 1);//load 4 elements from A[batch]
		As[buf][tx][ty].x = A[tY0 + Ak];
		As[buf][tx][ty].y = A[tY1 + Ak];
		As[buf][tx][ty].z = A[tY2 + Ak];
		As[buf][tx][ty].w = A[tY3 + Ak];

		int Bk = (ty >> 1);//load 1 element from B[batch]
		Bs[buf][Bs_y][Bs_x] = B[Bk * CM];//with the same tx
		__syncthreads();
	}

	//compute area---------------------------------------------------
	float2  v0 = make_float2(0, 0);
	float2  v2 = make_float2(0, 0);
	float2  v4 = make_float2(0, 0);
	float2  v6 = make_float2(0, 0);
	float2  v8 = make_float2(0, 0);
	float2 v10 = make_float2(0, 0);
	float2 v12 = make_float2(0, 0);
	float2 v14 = make_float2(0, 0);
	for (int ok = 1; ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float2 b0 = *(float2*)(&Bs[buf][ik][tx << 1]);
			float4 a0 = As[buf][ik][ty], a1 = As[buf][ik + STEP][ty];

			simdMM2( v0, a0.x, b0);
			simdMM2( v2, a0.y, b0);
			simdMM2( v4, a0.z, b0);
			simdMM2( v6, a0.w, b0);
			simdMM2( v8, a1.x, b0);
			simdMM2(v10, a1.y, b0);
			simdMM2(v12, a1.z, b0);
			simdMM2(v14, a1.w, b0);
		}
		buf ^= 1;

		int Ak = ((ok - (tx >= STEP)) << LB >> 1) + tx;//load 4 elements from A[batch]
		As[buf][tx][ty].x = A[tY0 + Ak];
		As[buf][tx][ty].y = A[tY1 + Ak];
		As[buf][tx][ty].z = A[tY2 + Ak];
		As[buf][tx][ty].w = A[tY3 + Ak];

		int Bk = ((ok << LB) + ty) >> 1;//load 1 element from B[batch]
		Bs[buf][Bs_y][Bs_x] = B[Bk * CM];
		__syncthreads();
	}
	if (OK) {
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float2 b0 = *(float2*)(&Bs[buf][ik][tx << 1]);
			float4 a0 = As[buf][ik][ty], a1 = As[buf][ik + STEP][ty];

			simdMM2(v0, a0.x, b0);
			simdMM2(v2, a0.y, b0);
			simdMM2(v4, a0.z, b0);
			simdMM2(v6, a0.w, b0);
			simdMM2(v8, a1.x, b0);
			simdMM2(v10, a1.y, b0);
			simdMM2(v12, a1.z, b0);
			simdMM2(v14, a1.w, b0);
		}
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
		a0.x = A[Y0 + k];
		a0.y = A[Y1 + k];
		a0.z = A[Y2 + k];
		a0.w = A[Y3 + k];
		a1.x = A[Y4 + k];
		a1.y = A[Y5 + k];
		a1.z = A[Y6 + k];
		a1.w = A[Y7 + k];

		//load 2 elements from B[batch]
		float2 b0 = *(float2*)(&B[k * CM]);

		simdMM2(v0, a0.x, b0);
		simdMM2(v2, a0.y, b0);
		simdMM2(v4, a0.z, b0);
		simdMM2(v6, a0.w, b0);
		simdMM2(v8, a1.x, b0);
		simdMM2(v10, a1.y, b0);
		simdMM2(v12, a1.z, b0);
		simdMM2(v14, a1.w, b0);
	}
	//when GK % STEP!=0----------------------------------------------

	Y0 = (Y * CM) + X;
	Y1 = Y0 + CM, Y2 = Y1 + CM, Y3 = Y2 + CM;
	Y4 = Y3 + CM, Y5 = Y4 + CM, Y6 = Y5 + CM, Y7 = Y6 + CM;

	*(float2*)(C + Y0) = v0;
	*(float2*)(C + Y1) = v2;
	*(float2*)(C + Y2) = v4;
	*(float2*)(C + Y3) = v6;
	*(float2*)(C + Y4) = v8;
	*(float2*)(C + Y5) = v10;
	*(float2*)(C + Y6) = v12;
	*(float2*)(C + Y7) = v14;
}

#endif


//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*8)
#ifndef BATCH_MATMUL_KERNEL_2_8
#define BATCH_MATMUL_KERNEL_2_8

//LB = 4: Size = 1, Time = 2.652 msec, Performace = 809.76 GFlop/s
//LB = 3: Size = 1, Time = 2.914 msec, Performace = 736.954 GFlop/s
//LB = 4: Size = 0.996094, Time = 2.626 msec, Performace = 814.583 GFlop/s
//LB = 3: Size = 0.996094, Time = 2.848 msec, Performace = 751.087 GFlop/s
template<int LB, int STEP, int MOVE_A, int MOVE_B>
__global__ void bmm_kernel_2_8(
	const float* __restrict__ A, //A[Batch,  N, AK], Ak is memAlgined
	const float* __restrict__ B, //B[Batch, BK,  M], Bk is not memAligned
		  float* __restrict__ C, //C[Batch,  N,  M]
	int N, int CM, int BK, int AK,
	int Yindex, int Xindex)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float  As[2][1 << LB >> 1][(2 << LB) + 2];//k24
	__shared__ float4 Bs[2][1 << LB][(1 << LB) + 1];//k88

	//prepared for Y:N
	const int Y = (((blockIdx.y << LB) + ty) << 1) + Yindex;//2
	const int tY0 = (Y + (tx & 1)) * AK;

	//prepared for X:M
	const int X = (((blockIdx.x << LB) + tx) << 3) + Xindex;//8
	const int tX = X + ((ty >= STEP) << 2);

	//compute start offset of A, B, C
	const int batch = blockIdx.z;
	A += (batch * N * AK * MOVE_A);//A[batch * MOVE_A]
	B += (batch * BK * CM * MOVE_B) + tX;//B[batch * MOVE_B, 0, tX]
	C += (batch * N * CM);//C[batch]

	const int  OK = (BK << 1 >> LB);
	const int As_x = (tx >> 1), As_y = (ty << 1) + (tx & 1);
	if (OK) {
		int Ak = (tx >> 1);//load 1 element from A[batch]
		As[buf][As_x][As_y] = A[tY0 + Ak];

		int Bk = ty - ((ty >= STEP) << LB >> 1);//load 4 elements from B[batch]
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
			float2 a0 = *(float2*)(&As[buf][ik][(ty << 1)]);
			simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
			simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
		}
		buf ^= 1;

		int Ak = ((ok << LB) + tx) >> 1;//load 1 element from A[batch]
		As[buf][As_x][As_y] = A[tY0 + Ak];

		int Bk = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		Bs[buf][ty][tx] = *(float4*)(&B[Bk * CM]);
		__syncthreads();
	}
	if (OK) {
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP][tx];
			float2 a0 = *(float2*)(&As[buf][ik][(ty << 1)]);
			simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
			simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
		}
	}

	//when GK % STEP!=0----------------------------------------------
	int Y0 = (Y * AK), Y1 = Y0 + AK;
	B += (X - tX);
	for (int k = BK - (BK&(STEP - 1)); k < BK; k++)
	{
		//load 2 elements from A[batch]
		float4 a0;
		a0.x = A[Y0 + k];
		a0.y = A[Y1 + k];

		//load 8 elements from B[batch]
		float4 b0 = *(float4*)(&B[k * CM]);
		float4 b1 = *(float4*)(&B[k * CM + 4]);

		simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
		simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
	}
	//when GK % STEP!=0----------------------------------------------

	Y0 = (Y * CM) + X, Y1 = Y0 + CM;
	*(float4*)(C + Y0) = v0; *(float4*)(C + Y0 + 4) = v1;
	*(float4*)(C + Y1) = v2; *(float4*)(C + Y1 + 4) = v3;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*2)
#ifndef BATCH_MATMUL_KERNEL_4_2
#define BATCH_MATMUL_KERNEL_4_2

//LB = 4: Size = 1, Time = 2.57  msec, Performace = 835.597 GFlop/s
//LB = 3: Size = 1, Time = 3.666 msec, Performace = 585.784 GFlop/s
//LB = 4: Size = 0.996094, Time = 2.632 msec, Performace = 812.726 GFlop/s
//LB = 3: Size = 0.996094, Time = 3.618 msec, Performace = 591.237 GFlop/s
template<int LB, int STEP, int MOVE_A, int MOVE_B>
__global__ void bmm_kernel_4_2(
	const float* __restrict__ A, //A[Batch,  N, AK], Ak is memAlgined
	const float* __restrict__ B, //B[Batch, BK,  M], Bk is not memAligned
		  float* __restrict__ C, //C[Batch,  N,  M]
	int N, int CM, int BK, int AK,
	int Yindex, int Xindex)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 As[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float  Bs[2][1 << LB >> 1][(2 << LB) + 2];

	//prepared for A -> Y:N
	const int Y = (((blockIdx.y << LB) + ty) << 2) + Yindex;
	const int tY0 = (Y + ((tx & 1) << 1)) * AK, tY1 = tY0 + AK;

	//prepared for B -> X:M
	const int X = (((blockIdx.x << LB) + tx) << 1) + Xindex;
	const int tX = X + (ty & 1);

	//compute start offset of A, B, C
	const int batch = blockIdx.z;
	A += (batch * N * AK * MOVE_A);//A[batch * MOVE_A]
	B += (batch * BK * CM * MOVE_B) + tX;//B[batch * MOVE_B, 0, tX]
	C += (batch * N * CM);//C[batch]

	const int OK = (BK << 1 >> LB);
	const int As_x = (tx >> 1), As_y = (ty << 1) + (tx & 1);
	const int Bs_y = (ty >> 1), Bs_x = (tx << 1) + (ty & 1);
	if (OK) {
		int Ak = (tx >> 1);//load 2 elements from A[batch]
		As[buf][As_x][As_y].x = A[tY0 + Ak];//with the same ty
		As[buf][As_x][As_y].y = A[tY1 + Ak];

		int Bk = (ty >> 1);//load 1 element from B[batch]
		Bs[buf][Bs_y][Bs_x] = B[Bk * CM];//with the same tx
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
		As[buf][As_x][As_y].x = A[tY0 + Ak];
		As[buf][As_x][As_y].y = A[tY1 + Ak];

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
	int Y0 = (Y * AK);
	int Y1 = Y0 + AK, Y2 = Y1 + AK, Y3 = Y2 + AK;
	B += (X - tX);
	for (int k = BK - (BK&(STEP - 1)); k < BK; k++)
	{
		//load 4 elements from A[batch]
		float4 a;
		a.x = A[Y0 + k];
		a.y = A[Y1 + k];
		a.z = A[Y2 + k];
		a.w = A[Y3 + k];

		//load 2 elements from B[batch]
		float2 b = *(float2*)(&B[k * CM]);

		simdMM2(v0, a.x, b);
		simdMM2(v1, a.y, b);
		simdMM2(v2, a.z, b);
		simdMM2(v3, a.w, b);
	}
	//when GK % STEP!=0----------------------------------------------

	Y0 = (Y * CM) + X;
	Y1 = Y0 + CM, Y2 = Y1 + CM, Y3 = Y2 + CM;
	*(float2*)(C + Y0) = v0;
	*(float2*)(C + Y1) = v1;
	*(float2*)(C + Y2) = v2;
	*(float2*)(C + Y3) = v3;
}

#endif


//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE * 4)
#ifndef BATCH_MATMUL_KERNEL_2_4
#define BATCH_MATMUL_KERNEL_2_4

//LB = 4: Size = 1, Time = 3.106 msec, Performace = 691.398 GFlop/s
//LB = 3: Size = 1, Time = 3.47  msec, Performace = 618.871 GFlop/s
//LB = 4: Size = 0.996094, Time = 3.068 msec, Performace = 697.228 GFlop/s
//LB = 3: Size = 0.996094, Time = 3.472 msec, Performace = 616.099 GFlop/s
template<int LB, int STEP, int MOVE_A, int MOVE_B>
__global__ void bmm_kernel_2_4(
	const float* __restrict__ A, //A[Batch, N, K]
	const float* __restrict__ B, //B[Batch, K, M]
	      float* __restrict__ C, //C[Batch, N, M]
	int N, int CM, int BK, int AK, 
	int Yindex, int Xindex)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float  As[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2 Bs[2][1 << LB >> 1][(2 << LB) + 2];

	//prepared for A -> Y:N
	const int Y = (((blockIdx.y << LB) + ty) << 1) + Yindex;
	const int tY0 = (Y + (tx & 1)) * AK;

	//prepared for B -> X:M
	const int X = (((blockIdx.x << LB) + tx) << 2) + Xindex;
	const int tX = X + ((ty & 1) << 1);

	//compute start offset of A, B, C
	const int batch = blockIdx.z;
	A += (batch * N * AK * MOVE_A);//A[batch * MOVE_A]
	B += (batch * BK * CM * MOVE_B) + tX;//B[batch * MOVE_B, 0, tX]
	C += (batch * N * CM);//C[batch]

	const int As_x = (tx >> 1), As_y = (ty << 1) + (tx & 1);
	const int Bs_y = (ty >> 1), Bs_x = (tx << 1) + (ty & 1);
	const int OK = (BK << 1 >> LB);
	if (OK) {
		int Ak = (tx >> 1);//load 1 element from A[batch]
		As[buf][As_x][As_y] = A[tY0 + Ak];

		int Bk = (ty >> 1);//load 2 elements from B[batch]
		Bs[buf][Bs_y][Bs_x] = *(float2*)(&B[Bk * CM]);//with the same tx
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
		As[buf][As_x][As_y] = A[tY0 + Ak];

		int Bk = ((ok << LB) + ty) >> 1;//load 2 elements from B[batch]
		Bs[buf][Bs_y][Bs_x] = *(float2*)(&B[Bk * CM]);
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
	int Y0 = (Y * AK), Y1 = Y0 + AK;
	B += (X - tX);
	for (int k = BK - (BK&(STEP - 1)); k < BK; k++)
	{
		//load 2 elements from A[batch]
		float2 a;
		a.x = A[Y0 + k];
		a.y = A[Y1 + k];

		//load 4 elements from B[batch]
		float4 b = *(float4*)(&B[k * CM]);

		simdMM4(v0, a.x, b);
		simdMM4(v1, a.y, b);
	}
	//when GK % STEP!=0----------------------------------------------

	Y0 = (Y * CM) + X, Y1 = Y0 + CM;
	*(float4*)(C + Y0) = v0;
	*(float4*)(C + Y1) = v1;
}

#endif


//======[Small]================================================
//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*2)
#ifndef BATCH_MATMUL_KERNEL_2_2
#define BATCH_MATMUL_KERNEL_2_2

//LB = 4: Size = 1, Time = 3.532 msec, Performace = 608.008 GFlop/s
//LB = 3: Size = 1, Time = 4.414 msec, Performace = 486.516 GFlop/s
//LB = 4: Size = 0.996094, Time = 3.552 msec, Performace = 602.223 GFlop/s
//LB = 3: Size = 0.996094, Time = 4.504 msec, Performace = 474.932 GFlop/s
template<int LB, int STEP, int MOVE_A, int MOVE_B>
__global__ void bmm_kernel_2_2(
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

	const int OK = (BK >> LB);
	if (OK) {
		int Ak = tx;//load 2 elements from A[batch]
		As[buf][tx][ty].x = A[tY0 + Ak];
		As[buf][tx][ty].y = A[tY1 + Ak];

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
		As[buf][tx][ty].x = A[tY0 + Ak];
		As[buf][tx][ty].y = A[tY1 + Ak];

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
	for (int k = BK - (BK&(STEP - 1)); k < BK; k++)  {
		float2 a;//load 2 elements from A[batch]
		a.x = A[tY0 + k];
		a.y = A[tY1 + k];

		float2 b = *(float2*)(&B[k*CM]);//load 2 elements from B[batch]

		simdMM2(v0, a.x, b);
		simdMM2(v1, a.y, b);
	}
	//when GK % STEP!=0----------------------------------------------

	int Y0 = (Y * CM) + X, Y1 = Y0 + CM;
	*(float2*)(C + Y0) = v0;
	*(float2*)(C + Y1) = v1;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*1)
#ifndef BATCH_MATMUL_KERNEL_4_1
#define BATCH_MATMUL_KERNEL_4_1

//LB = 4: Size = 0.996094, Time = 4.018 msec, Performace = 532.378 GFlop/s
//LB = 3: Size = 0.996094, Time = 5.16  msec, Performace = 414.553 GFlop/s
template<int LB, int STEP, int MOVE_A, int MOVE_B>
__global__ void bmm_kernel_4_1(
	const float* __restrict__ A, //A[Batch,  N, AK], Ak is memAlgined
	const float* __restrict__ B, //B[Batch, BK,  M], Bk is not memAligned
		  float* __restrict__ C, //C[Batch,  N,  M]
	int N, int CM, int BK, int AK,
	int Yindex, int Xindex)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 As[2][1 << LB][(1 << LB) + 1];
	__shared__ float  Bs[2][1 << LB][(1 << LB) + 1];

	//prepared for A -> Y:N
	const int Y = (((blockIdx.y << LB) + ty) << 2) + Yindex;
	const int tY0 = Y * AK;
	const int tY1 = tY0 + AK, tY2 = tY1 + AK, tY3 = tY2 + AK;

	//prepared for B -> X:M
	const int X = ((blockIdx.x << LB) + tx) + Xindex;

	//compute start offset of A, B, C
	const int batch = blockIdx.z;
	A += (batch * N * AK * MOVE_A);//A[batch * MOVE_A]
	B += (batch * BK * CM * MOVE_B) + X;//B[batch * MOVE_B, 0, tX]
	C += (batch * N * CM);//C[batch]

	const int OK = (BK >> LB);
	if (OK) {
		int Ak = tx;//load 4 elements from A[batch]
		As[buf][tx][ty].x = A[tY0 + Ak];
		As[buf][tx][ty].y = A[tY1 + Ak];
		As[buf][tx][ty].z = A[tY2 + Ak];
		As[buf][tx][ty].w = A[tY3 + Ak];

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

		int Ak = (ok << LB) + tx;//load 2 elements from A[batch]
		As[buf][tx][ty].x = A[tY0 + Ak];
		As[buf][tx][ty].y = A[tY1 + Ak];
		As[buf][tx][ty].z = A[tY2 + Ak];
		As[buf][tx][ty].w = A[tY3 + Ak];

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
	for (int k = BK - (BK&(STEP - 1)); k < BK; k++) {
		float4 a;//load 4 elements from A
		a.x = A[tY0 + k];
		a.y = A[tY1 + k];
		a.z = A[tY2 + k];
		a.w = A[tY3 + k];

		float b = B[k * CM];//load 1 element from B

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
#ifndef BATCH_MATMUL_KERNEL_1_4
#define BATCH_MATMUL_KERNEL_1_4

//LB = 4: Size = 0.996094, Time = 5.164 msec, Performace = 414.232 GFlop/s
//LB = 3: Size = 0.996094, Time = 5.76  msec, Performace = 371.371 GFlop/s
template<int LB, int STEP, int MOVE_A, int MOVE_B>
__global__ void bmm_kernel_1_4(
	const float* __restrict__ A, //A[Batch,  N, AK], Ak is memAlgined
	const float* __restrict__ B, //B[Batch, BK,  M], Bk is not memAligned
		  float* __restrict__ C, //C[Batch,  N,  M]
	int N, int CM, int BK, int AK,
	int Yindex, int Xindex)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float  As[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Bs[2][1 << LB][(1 << LB) + 1];

	//prepared for A -> Y:N
	const int Y = ((blockIdx.y << LB) + ty) + Yindex;
	const int tY0 = Y * AK;

	//prepared for B -> X:M
	const int X = (((blockIdx.x << LB) + tx) << 2) + Xindex;

	//compute start offset of A, B, C
	const int batch = blockIdx.z;
	A += (batch * N * AK * MOVE_A);//A[batch * MOVE_A]
	B += (batch * BK * CM * MOVE_B) + X;//B[batch * MOVE_B, 0, tX]
	C += (batch * N * CM);//C[batch]

	const int OK = (BK >> LB);
	if (OK) {
		int Ak = tx;//load 1 element from A[batch]
		As[buf][tx][ty] = A[tY0 + Ak];

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
		As[buf][tx][ty] = A[tY0 + Ak];

		int Bk = (ok << LB) + ty;//load 1 element from B[batch]
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
	for (int k = BK - (BK&(STEP - 1)); k < BK; k++) {
		float  a = A[tY0 + k];//load 1 elements from A
		float4 b = *(float4*)(&B[k * CM]);//load 2 elements from B
		simdMM4(v, a, b);
	}
	//when GK % STEP!=0----------------------------------------------

	*(float4*)(C + (Y * CM) + X) = v;
}

#endif

#endif