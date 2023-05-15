

#ifndef MAT_MUL4X_K2POW
#define MAT_MUL4X_K2POW

#define mm4x_k2_Branch(SIZE_Y, SIZE_X) {\
	int N1 = (N & SIZE_Y), M1 = (M & SIZE_X);\
	if (N1 && M1) {\
		int N0 = N - N1, M0 = M - M1;\
		const float *A1 = &get(A, N0, 0, K), *B1 = &get(B, 0, M0, SB);\
		float *C01 = &get(C, 0, M0, SB);\
		float *C10 = &get(C, N0, 0, SB), *C11 = &get(C, N0, M0, SB);\
		matMul4x_K2pow(streams, index, length, A , B1, C01, N0, M1, K, SB);\
		matMul4x_K2pow(streams, index, length, A1, B , C10, N1, M0, K, SB);\
		matMul4x_K2pow(streams, index, length, A1, B1, C11, N1, M1, K, SB);}\
	else if (N1){\
		int N0 = N - N1;\
		const float *A1 = &get(A, N0, 0, K);\
		float *C1 = &get(C, N0, 0, SB);\
		matMul4x_K2pow(streams, index, length, A1, B, C1, N1, M, K, SB);}\
	else if (M1){\
		int M0 = M - M1;\
		const float *B1 = &get(B, 0, M0, SB);\
		float *C1 = &get(C, 0, M0, SB);\
		matMul4x_K2pow(streams, index, length, A, B1, C1, N, M1, K, SB);}}

//N % 4 == 0
//M % 4 == 0
//K % 4 == 0
void matMul4x_K2pow(jlong *streams, int &index, int length,
	const float* A,
	const float* B,
	float* C,
	int N, int M, int K, int SB)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)streams[index];
	index = (index + 1) % length;

	int size = N * M;
	if (size <= 256) { knaiveK2(1, stream, A, B, C, N, M, LOG2(K), SB); return; }

	bool k7 = !(K & 7);
	if ((N > 127) && (M > 127) && k7) { k88K2(4, stream, A, B, C, N, M, LOG2(K), SB); mm4x_k2_Branch(127, 127); return; } //1576.14 GFlop/s
	if ((N > 63) && (M > 63)) { k88K2(3, stream, A, B, C, N, M, LOG2(K), SB);  mm4x_k2_Branch(63, 63); return; }//1339.02 GFlop/s,

	if ((N > 63) && (M > 31)) { k84K2(3, stream, A, B, C, N, M, LOG2(K), SB); mm4x_k2_Branch(63, 31); return; }//958.52 GFlop/s
	if ((N > 31) && (M > 63)) { k48K2(3, stream, A, B, C, N, M, LOG2(K), SB); mm4x_k2_Branch(31, 63); return; }//1211.06 GFlop/s

	if ((N > 31) && (M > 31)) { k44K2(3, stream, A, B, C, N, M, LOG2(K), SB); mm4x_k2_Branch(31, 31); return; }//916.65 GFlop/s

	if ((N > 63) && (M > 15)) { k82K2(3, stream, A, B, C, N, M, LOG2(K), SB); mm4x_k2_Branch(63, 15); return; }//559.98 GFlop/s
	if ((N > 15) && (M > 63)) { k28K2(3, stream, A, B, C, N, M, LOG2(K), SB); mm4x_k2_Branch(15, 63); return; }//772.41 GFlop/s, 

	if ((N > 31) && (M > 15)) { k42K2(3, stream, A, B, C, N, M, LOG2(K), SB); mm4x_k2_Branch(31, 15); return; }//522.76 GFlop/s
	if ((N > 15) && (M > 31)) { k24K2(3, stream, A, B, C, N, M, LOG2(K), SB); mm4x_k2_Branch(15, 31); return; }//646.47 GFlop/s

	if (k7) {
		if ((N > 63) && (M > 7)) { k81K2(3, stream, A, B, C, N, M, LOG2(K), SB); mm4x_k2_Branch(63, 7); return; }
		if ((N > 15) && (M > 15)) { k22K2(3, stream, A, B, C, N, M, LOG2(K), SB); mm4x_k2_Branch(15, 15); return; }//480.90 GFlop/s
		if ((N > 7) && (M > 63)) { k18K2(3, stream, A, B, C, N, M, LOG2(K), SB); mm4x_k2_Branch(7, 63); return; }
	}

	if ((N > 31) && (M > 15)) { k84K2(2, stream, A, B, C, N, M, LOG2(K), SB); mm4x_k2_Branch(31, 15); return; }
	if ((N > 15) && (M > 31)) { k48K2(2, stream, A, B, C, N, M, LOG2(K), SB); mm4x_k2_Branch(15, 31); return; }

	if ((N > 15) && (M > 7)) { k42K2(2, stream, A, B, C, N, M, LOG2(K), SB); mm4x_k2_Branch(15, 7); return; }
	if ((N > 7) && (M > 15)) { k24K2(2, stream, A, B, C, N, M, LOG2(K), SB); mm4x_k2_Branch(7, 15); return; }

	if ((N > 31)) { k81K2(2, stream, A, B, C, N, M, LOG2(K), SB); mm4x_k2_Branch(31, 3); return; }
	if ((N > 7) && (M > 7)) { k22K2(2, stream, A, B, C, N, M, LOG2(K), SB); mm4x_k2_Branch(7, 7); return; }
	if ((M > 31)) { k18K2(2, stream, A, B, C, N, M, LOG2(K), SB); mm4x_k2_Branch(3, 31); return; }

	k22K2(1, stream, A, B, C, N, M, LOG2(K), SB);
}

#endif


#ifndef MAT_MUL4X_T2_K2POW
#define MAT_MUL4X_T2_K2POW

#define mm4xT2_k2_Branch(SIZE_Y, SIZE_X) {\
	int N1 = (N & SIZE_Y), M1 = (M & SIZE_X);\
	if (N1 && M1) {\
		int N0 = N - N1, M0 = M - M1;\
		const float *A1 = &get(A, N0, 0, K), *B1 = &get(B, M0, 0, K);\
		float *C01 = &get(C, 0, M0, SC);\
		float *C10 = &get(C, N0, 0, SC), *C11 = &get(C, N0, M0, SC);\
		matMul4x_T2_K2pow(streams, index, length, A , B1, C01, N0, M1, K, SC);\
		matMul4x_T2_K2pow(streams, index, length, A1, B , C10, N1, M0, K, SC);\
		matMul4x_T2_K2pow(streams, index, length, A1, B1, C11, N1, M1, K, SC);}\
	else if (N1){\
		int N0 = N - N1;\
		const float *A1 = &get(A, N0, 0, K);\
		float *C1 = &get(C, N0, 0, SC);\
		matMul4x_T2_K2pow(streams, index, length, A1, B, C1, N1, M, K, SC);}\
	else if (M1){\
		int M0 = M - M1;\
		const float *B1 = &get(B, M0, 0, K);\
		float *C1 = &get(C, 0, M0, SC);\
		matMul4x_T2_K2pow(streams, index, length, A, B1, C1, N, M1, K, SC);}}

//for the first stack of this function: SC = M
void matMul4x_T2_K2pow(jlong *streams, int &index, int length,
	const float* A,
	const float* B,
	float* C,
	int N, int M, int K, int SC)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)streams[index];
	index = (index + 1) % length;

	int size = N * M;
	if (size <= 256) { knaiveT2K2(2, stream, A, B, C, N, M, LOG2(K), SC); return; }

	bool k7 = !(K & 7);
	if ((N > 127) && (M > 127) && k7) { k88T2K2(4, stream, A, B, C, N, M, LOG2(K), SC); mm4xT2_k2_Branch(127, 127); return; } //1576.14 GFlop/s
	if ((N > 63) && (M > 63)) { k88T2K2(3, stream, A, B, C, N, M, LOG2(K), SC); mm4xT2_k2_Branch(63, 63); return; }//1339.02 GFlop/s,

	if ((N > 63) && (M > 31)) { k84T2K2(3, stream, A, B, C, N, M, LOG2(K), SC); mm4xT2_k2_Branch(63, 31); return; }//958.52 GFlop/s
	if ((N > 31) && (M > 63)) { k48T2K2(3, stream, A, B, C, N, M, LOG2(K), SC); mm4xT2_k2_Branch(31, 63); return; }//1211.06 GFlop/s

	if ((N > 31) && (M > 31)) { k44T2K2(3, stream, A, B, C, N, M, LOG2(K), SC); mm4xT2_k2_Branch(31, 31); return; }//916.65 GFlop/s

	if ((N > 63) && (M > 15)) { k82T2K2(3, stream, A, B, C, N, M, LOG2(K), SC); mm4xT2_k2_Branch(63, 15); return; }//559.98 GFlop/s
	if ((N > 15) && (M > 63)) { k28T2K2(3, stream, A, B, C, N, M, LOG2(K), SC); mm4xT2_k2_Branch(15, 63); return; }//772.41 GFlop/s, 

	if ((N > 31) && (M > 15)) { k42T2K2(3, stream, A, B, C, N, M, LOG2(K), SC); mm4xT2_k2_Branch(31, 15); return; }//522.76 GFlop/s
	if ((N > 15) && (M > 31)) { k24T2K2(3, stream, A, B, C, N, M, LOG2(K), SC); mm4xT2_k2_Branch(15, 31); return; }//646.47 GFlop/s

	if (k7) {
		if ((N > 63) && (M > 7)) { k81T2K2(3, stream, A, B, C, N, M, LOG2(K), SC); mm4xT2_k2_Branch(63, 7); return; }
		if ((N > 15) && (M > 15)) { k22T2K2(3, stream, A, B, C, N, M, LOG2(K), SC); mm4xT2_k2_Branch(15, 15); return; }//480.90 GFlop/s
		if ((N > 7) && (M > 63)) { k18T2K2(3, stream, A, B, C, N, M, LOG2(K), SC); mm4xT2_k2_Branch(7, 63); return; }
	}

	if ((N > 31) && (M > 15)) { k84T2K2(2, stream, A, B, C, N, M, LOG2(K), SC); mm4xT2_k2_Branch(31, 15); return; }//475.05 GFlop/s
	if ((N > 15) && (M > 31)) { k48T2K2(2, stream, A, B, C, N, M, LOG2(K), SC); mm4xT2_k2_Branch(15, 31); return; }//539.34 GFlop/s

	if ((N > 15) && (M > 7)) { k42T2K2(2, stream, A, B, C, N, M, LOG2(K), SC); mm4xT2_k2_Branch(15, 7); return; }
	if ((N > 7) && (M > 15)) { k24T2K2(2, stream, A, B, C, N, M, LOG2(K), SC); mm4xT2_k2_Branch(7, 15); return; }

	if (N > 31) { k81T2K2(2, stream, A, B, C, N, M, LOG2(K), SC); mm4xT2_k2_Branch(31, 3); return; }
	if ((N > 7) && (M > 7)) { k22T2K2(2, stream, A, B, C, N, M, LOG2(K), SC); mm4xT2_k2_Branch(7, 7); return; }
	if (M > 31) { k18T2K2(2, stream, A, B, C, N, M, LOG2(K), SC); mm4xT2_k2_Branch(3, 31); return; }

	k22T2K2(1, stream, A, B, C, N, M, LOG2(K), SC); return;
}

#endif




//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*4), K % BLOCK_SIZE == 0
//LB = 4: K % 16 == 0 
//LB = 3: K %  8 == 0
#ifndef MATMUL_T2_UERNEL_4_4_MGK
#define MATMUL_T2_UERNEL_4_4_MGK

#define	u44T2_mgk(LB, stream, A, B, C, N, M, K, SC) \
	uernel_t2_4_4_mgk<LB, (1<<LB>>1), (1<<LB)>\
		<<< dim3(M>>2>>LB, N>>2>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SC, K)

//LB = 4: Size = 1, Time = 2.018 msec, Performace = 1064.16 GFlop/s(1000)
//LB = 3: Size = 1, Time = 3.606 msec, Performace = 595.531 GFlop/s(1000)
template<int LB, int STEP, int STEP2>
__global__ void uernel_t2_4_4_mgk(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float*  __restrict__ C,
	int SC, int K)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float2 As[2][1 << LB][(2 << LB) + 2];
	__shared__ float2 Bs[2][1 << LB][(2 << LB) + 2];

	int Y = ((blockIdx.y << LB) + ty) << 2;
	int X = ((blockIdx.x << LB) + tx) << 2;
	int C0 = Y * SC + X;//C[Y, X]

	const int Ax = (tx >> 1) << 1;
	const int A0 = (Y + ((tx & 1) << 1)) * K + Ax, A1 = A0 + K;
	const int As_x = Ax, As_y = ((ty << 1) + (tx & 1));

	const int By = (ty >> 1) << 1;
	const int B0 = (X + ((ty & 1) << 1)) * K + By, B1 = B0 + K;
	const int Bs_y = By, Bs_x = ((tx << 1) + (ty & 1));

	//load 2 elem from A(transposed)
	float2 a0 = *(float2*)(A + A0);
	float2 a1 = *(float2*)(A + A1);
	As[buf][As_x][As_y] = float2{ a0.x, a1.x };
	As[buf][As_x + 1][As_y] = float2{ a0.y, a1.y };

	//load 2 elem from B
	float2 b0 = *(float2*)(B + B0);
	float2 b1 = *(float2*)(B + B1);
	Bs[buf][Bs_y][Bs_x] = float2{ b0.x, b1.x };
	Bs[buf][Bs_y + 1][Bs_x] = float2{ b0.y, b1.y };

	A += STEP2; B += STEP2;//K += STEP2
	__syncthreads();

	//compute area-------------------------------------------------------
	float4 c0 = make_float4(0, 0, 0, 0);
	float4 c1 = make_float4(0, 0, 0, 0);
	float4 c2 = make_float4(0, 0, 0, 0);
	float4 c3 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (K >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP2; ik++)
		{
			float4 b = *(float4*)(&Bs[buf][ik][tx << 1]);
			float4 a = *(float4*)(&As[buf][ik][ty << 1]);

			simdMM4(c0, a.x, b);
			simdMM4(c1, a.y, b);
			simdMM4(c2, a.z, b);
			simdMM4(c3, a.w, b);
		}
		buf ^= 1;

		//load 2 elem from A(transposed)
		float2 a0 = *(float2*)(A + A0);
		float2 a1 = *(float2*)(A + A1);
		As[buf][As_x][As_y] = float2{ a0.x, a1.x };
		As[buf][As_x + 1][As_y] = float2{ a0.y, a1.y };

		//load 2 elem from B
		float2 b0 = *(float2*)(B + B0);
		float2 b1 = *(float2*)(B + B1);
		Bs[buf][Bs_y][Bs_x] = float2{ b0.x, b1.x };
		Bs[buf][Bs_y + 1][Bs_x] = float2{ b0.y, b1.y };

		A += STEP2; B += STEP2;//K += STEP2
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP2; ik++)
	{
		float4 b = *(float4*)(&Bs[buf][ik][tx << 1]);
		float4 a = *(float4*)(&As[buf][ik][ty << 1]);

		simdMM4(c0, a.x, b);
		simdMM4(c1, a.y, b);
		simdMM4(c2, a.z, b);
		simdMM4(c3, a.w, b);
	}

	const int C1 = C0 + SC;
	const int C2 = C1 + SC;
	const int C3 = C2 + SC;

	*(float4*)(C + C0) = c0;
	*(float4*)(C + C1) = c1;
	*(float4*)(C + C2) = c2;
	*(float4*)(C + C3) = c3;
}

#endif