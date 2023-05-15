


//(Y: BLOCK_SIZE * 8, X: BLOCK_SIZE * 8), K % BLOCK_SIZE == 0
//LB = 4: K % 16 == 0
//LB = 3: K %  8 == 0
#ifndef BATCH_MATMUL_UERNEL_8_8_PADDING_MK
#define BATCH_MATMUL_UERNEL_8_8_PADDING_MK

#define bmm_u88p_mk(stream, LB, Yindex, Xindex, A, B, C, Batch, N, CM, BK, AK, GN, GM)\
	bmm_uernel_8_8_padding_MK<LB, (1<<LB>>1), (1<<LB), MOVE_A, MOVE_B>\
		<<< dim3((GM>>LB>>3), (GN>>LB>>3), Batch),\
            dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, N, CM, BK, AK, Yindex, Xindex)

//LB = 4: Size = 1, Time = 1.49 msec, Performace = 1441.26 GFlop/s
//LB = 4: Size = 0.9767, Time = 1.526 msec, Performace = 1374.47 GFlop/s
//LB = 3: Size = 0.9767, Time = 1.726 msec, Performace = 1215.21 GFlop/s
template<int LB, int STEP, int STEP2, int MOVE_A, int MOVE_B>
__global__ void bmm_uernel_8_8_padding_MK(
	const float* __restrict__ A, //A[Batch, N, AK], Ak is memAlgined
	const float* __restrict__ B, //B[Batch, BK, M], Bk is not memAligned
	float* __restrict__ C, //C[Batch,  N,  M]
	int N, int CM, int BK, int AK,
	int Yindex, int Xindex)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 As[2][2 << LB][(1 << LB) + 1];
	__shared__ float4 Bs[2][2 << LB][(1 << LB) + 1];

	//prepared for A -> Y:N
	const int Y = (((blockIdx.y << LB) + ty) << 3) + Yindex;
	const int Y0 = (Y + ((tx >= STEP) << 2))*AK;
	const int Y1 = Y0 + AK, Y2 = Y1 + AK, Y3 = Y2 + AK;

	//prepared for B -> X:M
	const int X = (((blockIdx.x << LB) + tx) << 3) + Xindex;
	const int X0 = X + ((ty >= STEP) << 2);

	//compute start offset of A, B, C
	const int batch = blockIdx.z;
	A += (batch * N * AK * MOVE_A);//A[batch * MOVE_A]
	B += (batch * BK * CM * MOVE_B) + X0;//B[batch * MOVE_B, 0, tX]
	const int C0 = (Y * CM) + X + (batch * N * CM);//C[batch]

	const int ye = (N - 1)*AK;
	const int xe = (CM - 4);//float4: X <= M - 4 

	//load 4 elements from A[batch]
	int Ak = (tx - ((tx >= STEP) << LB >> 1)) << 1;
	float2 a0 = (Y0 <= ye ? *(float2*)(A + Y0 + Ak) : F32_2_0);
	float2 a1 = (Y1 <= ye ? *(float2*)(A + Y1 + Ak) : F32_2_0);
	float2 a2 = (Y2 <= ye ? *(float2*)(A + Y2 + Ak) : F32_2_0);
	float2 a3 = (Y3 <= ye ? *(float2*)(A + Y3 + Ak) : F32_2_0);
	As[buf][(tx << 1)][ty] = float4{ a0.x, a1.x, a2.x, a3.x };
	As[buf][(tx << 1) + 1][ty] = float4{ a0.y, a1.y, a2.y, a3.y };

	//load 4 elements from B[batch]
	int Bk = (ty - ((ty >= STEP) << LB >> 1)) << 1;
	int boffset0 = Bk * CM, boffset1 = boffset0 + CM;
	Bs[buf][(ty << 1)][tx] = (X0 <= xe ? *(float4*)(B + boffset0) : F32_4_0);
	Bs[buf][(ty << 1) + 1][tx] = (X0 <= xe ? *(float4*)(B + boffset1) : F32_4_0);
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

	for (int ok = 1, OK = (BK >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++)
		{
			float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP2][tx];
			float4 a0 = As[buf][ik][ty], a1 = As[buf][ik + STEP2][ty];

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
		float2 a0 = (Y0 <= ye ? *(float2*)(A + Y0 + Ak) : F32_2_0);
		float2 a1 = (Y1 <= ye ? *(float2*)(A + Y1 + Ak) : F32_2_0);
		float2 a2 = (Y2 <= ye ? *(float2*)(A + Y2 + Ak) : F32_2_0);
		float2 a3 = (Y3 <= ye ? *(float2*)(A + Y3 + Ak) : F32_2_0);
		As[buf][(tx << 1)][ty] = float4{ a0.x, a1.x, a2.x, a3.x };
		As[buf][(tx << 1) + 1][ty] = float4{ a0.y, a1.y, a2.y, a3.y };

		//load 4 elements from B[batch]
		int Bk = (((ok - (ty >= STEP)) << LB >> 1) + ty) << 1;
		int boffset0 = Bk * CM, boffset1 = boffset0 + CM;
		Bs[buf][(ty << 1)][tx] = (X0 <= xe ? *(float4*)(B + boffset0) : F32_4_0);
		Bs[buf][(ty << 1) + 1][tx] = (X0 <= xe ? *(float4*)(B + boffset1) : F32_4_0);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++)
	{
		float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP2][tx];
		float4 a0 = As[buf][ik][ty], a1 = As[buf][ik + STEP2][ty];

		simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
		simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
		simdMM4(v4, a0.z, b0); simdMM4(v5, a0.z, b1);
		simdMM4(v6, a0.w, b0); simdMM4(v7, a0.w, b1);
		simdMM4(v8, a1.x, b0); simdMM4(v9, a1.x, b1);
		simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
		simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
		simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
	}

	const int C1 = C0 + CM, C2 = C1 + CM, C3 = C2 + CM;
	const int C4 = C3 + CM, C5 = C4 + CM, C6 = C5 + CM, C7 = C6 + CM;

	int ce0 = (N * CM) - 4, ce1 = ce0 - 4;
	bool wx0 = (X <= xe), wx1 = (X + 4 <= xe);

	*(float4*)IF_int((wx0 && (C0 <= ce0)), (C + C0), (HOLE + (C0 & 255))) = v0;
	*(float4*)IF_int((wx1 && (C0 <= ce1)), (C + C0 + 4), (HOLE + (C0 & 255))) = v1;
	*(float4*)IF_int((wx0 && (C1 <= ce0)), (C + C1), (HOLE + (C1 & 255))) = v2;
	*(float4*)IF_int((wx1 && (C1 <= ce1)), (C + C1 + 4), (HOLE + (C1 & 255))) = v3;
	*(float4*)IF_int((wx0 && (C2 <= ce0)), (C + C2), (HOLE + (C2 & 255))) = v4;
	*(float4*)IF_int((wx1 && (C2 <= ce1)), (C + C2 + 4), (HOLE + (C2 & 255))) = v5;

	*(float4*)IF_int((wx0 && (C3 <= ce0)), (C + C3), (HOLE + (C3 & 255))) = v6;
	*(float4*)IF_int((wx0 && (C4 <= ce0)), (C + C4), (HOLE + (C4 & 255))) = v8;
	*(float4*)IF_int((wx0 && (C5 <= ce0)), (C + C5), (HOLE + (C5 & 255))) = v10;
	*(float4*)IF_int((wx0 && (C6 <= ce0)), (C + C6), (HOLE + (C6 & 255))) = v12;
	*(float4*)IF_int((wx0 && (C7 <= ce0)), (C + C7), (HOLE + (C7 & 255))) = v14;
	*(float4*)IF_int((wx1 && (C3 <= ce1)), (C + C3 + 4), (HOLE + (C3 & 255))) = v7;
	*(float4*)IF_int((wx1 && (C4 <= ce1)), (C + C4 + 4), (HOLE + (C4 & 255))) = v9;
	*(float4*)IF_int((wx1 && (C5 <= ce1)), (C + C5 + 4), (HOLE + (C5 & 255))) = v11;
	*(float4*)IF_int((wx1 && (C6 <= ce1)), (C + C6 + 4), (HOLE + (C6 & 255))) = v13;
	*(float4*)IF_int((wx1 && (C7 <= ce1)), (C + C7 + 4), (HOLE + (C7 & 255))) = v15;
}

#endif
