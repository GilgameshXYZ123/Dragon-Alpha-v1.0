


//we have: Batch % 4 ==0
//we have: K % 4 == 0
//we have: M % 4 == 0

#ifndef V1
#define V1

#define bmm_kv1(stream, LB, A, B, C, Batch, N, CM, K)\
	bmm_kernel_v1<LB>\
		<<< dim3(CM>>LB, N>>LB, Batch), dim3(1<<LB, 1<<LB)>>>\
			(A, B, C, N, CM, K)

//Size = 1, Time = 17.84 msec, Performace = 120.375 GFlop/s
//Batch -> batch
//N -> Y, M -> X

template<int LB>
__global__ void bmm_kernel_v1(
	const float* __restrict__ A,//A[Batch, N, K]
	const float* __restrict__ B,//B[Batch, K, M]
		  float* __restrict__ C,//C[Batch, N, M]
	int N, int CM, int K)
{
	int ty = threadIdx.y, tx = threadIdx.x;
	int bz = blockIdx.z, by = blockIdx.y, bx = blockIdx.x;
	
	int X = (bx << LB) + tx;
	int Y = (by << LB) + ty;
	int batch = bz;

	float v = 0;
	for (int k = 0; k < K; k++) {
		float a = get3d(A, batch, Y, k, N, K);//K % 4 == 0, safe for float4
		float b = get3d(B, batch, k, X, K, CM);//M % 4 == 0
		v += a * b;
	}
	get3d(C, batch, Y, X, N, CM) = v;
}

#endif



#ifndef V2
#define V2

#define bmm_kv2(stream, LB, A, B, C, Batch, N, CM, K)\
	bmm_kernel_v2<LB, (1<<LB)>\
		<<< dim3(CM>>LB, N>>LB, Batch), dim3(1<<LB, 1<<LB)>>>\
			(A, B, C, N, CM, K)

//Size = 1, Time = 11.42 msec, Performace = 188.046 GFlop/s
//Batch -> batch
//N -> Y, M -> X

template<int LB, int STEP>
__global__ void bmm_kernel_v2(
	const float* __restrict__ A,//A[Batch, N, K]
	const float* __restrict__ B,//B[Batch, K, M]
	float* __restrict__ C,//C[Batch, N, M]
	int N, int CM, int K)
{
	int ty = threadIdx.y, tx = threadIdx.x;
	int by = blockIdx.y, bx = blockIdx.x;

	int X = (bx << LB) + tx;
	int Y = (by << LB) + ty;
	int batch = blockIdx.z;

	__shared__ float As[1 << LB][(1 << LB) + 1];
	__shared__ float Bs[1 << LB][(1 << LB) + 1];//with the same tx

	float v = 0;
	for (int ok = 0, OK = (K >> LB); ok < OK; ok++) 
	{
		int Ak = (ok << LB) + tx;
		As[tx][ty] = get3d(A, batch, Y, Ak, N, K);//with the same ty

		int Bk = (ok << LB) + ty;
		Bs[ty][tx] = get3d(B, batch, Bk, X, K, CM);//with the same tx
		__syncthreads();

#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float a = As[ik][ty];
			float b = Bs[ik][tx];
			v += a * b;
		}
		__syncthreads();
	}

	get3d(C, batch, Y, X, N, CM) = v;
}

#endif


#ifndef V3
#define V3

#define bmm_kv3(stream, LB, A, B, C, Batch, N, CM, K)\
	bmm_kernel_v3<LB, (1<<LB)>\
		<<< dim3(CM>>LB, N>>LB, Batch), dim3(1<<LB, 1<<LB)>>>\
			(A, B, C, N, CM, K)

//Size = 1, Time = 10.88 msec, Performace = 197.379 GFlop/s
//Batch -> batch
//N -> Y, M -> X

template<int LB, int STEP>
__global__ void bmm_kernel_v3(
	const float* __restrict__ A,//A[Batch, N, K]
	const float* __restrict__ B,//B[Batch, K, M]
	float* __restrict__ C,//C[Batch, N, M]
	int N, int CM, int K)
{
	int ty = threadIdx.y, tx = threadIdx.x;
	int by = blockIdx.y, bx = blockIdx.x;

	int X = (bx << LB) + tx;
	int Y = (by << LB) + ty;
	int batch = blockIdx.z;

	bool buf = 0;
	__shared__ float As[2][1 << LB][(1 << LB) + 1];
	__shared__ float Bs[2][1 << LB][(1 << LB) + 1];//with the same tx

	//preload--------------------------------------------------------
	int Ak = tx;
	As[buf][tx][ty] = get3d(A, batch, Y, Ak, N, K);//with the same ty

	int Bk = ty;
	Bs[buf][ty][tx] = get3d(B, batch, Bk, X, K, CM);//with the same tx
	__syncthreads();

	//compute area---------------------------------------------------
	float v = 0;
	for (int ok = 1, OK = (K >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float a = As[buf][ik][ty];
			float b = Bs[buf][ik][tx];
			v += a * b;
		}
		buf ^= 1;

		int Ak = (ok << LB) + tx;
		As[buf][tx][ty] = get3d(A, batch, Y, Ak, N, K);//with the same ty

		int Bk = (ok << LB) + ty;
		Bs[buf][ty][tx] = get3d(B, batch, Bk, X, K, CM);//with the same tx
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float a = As[buf][ik][ty];
		float b = Bs[buf][ik][tx];
		v += a * b;
	}

	get3d(C, batch, Y, X, N, CM) = v;
}

#endif


#ifndef V4
#define V4

#define bmm_kv4(stream, LB, A, B, C, Batch, N, CM, K)\
	bmm_kernel_v4<LB, (1<<LB)>\
		<<< dim3(CM>>LB>>1, N>>LB>>1, Batch), dim3(1<<LB, 1<<LB)>>>\
			(A, B, C, N, CM, K)

//Size = 1, Time = 4.32 msec, Performace = 497.103 GFlop/s
//Batch -> batch
//N -> Y, M -> X

template<int LB, int STEP>
__global__ void bmm_kernel_v4(
	const float* __restrict__ A,//A[Batch, N, K]
	const float* __restrict__ B,//B[Batch, K, M]
	float* __restrict__ C,//C[Batch, N, M]
	int N, int CM, int K)
{
	int ty = threadIdx.y, tx = threadIdx.x;
	int by = blockIdx.y, bx = blockIdx.x;

	int X = ((bx << LB) + tx) << 1;
	int Y = ((by << LB) + ty) << 1;
	int batch = blockIdx.z;

	bool buf = 0;
	__shared__ float2 As[2][1 << LB][(1 << LB) + 1];
	__shared__ float2 Bs[2][1 << LB][(1 << LB) + 1];//with the same tx

	//preload--------------------------------------------------------
	int Ak = tx;
	As[buf][tx][ty].x = get3d(A, batch, Y    , Ak, N, K);//with the same ty
	As[buf][tx][ty].y = get3d(A, batch, Y + 1, Ak, N, K);

	int Bk = ty;
	Bs[buf][ty][tx] = *(float2*)(&get3d(B, batch, Bk, X, K, CM));//with the same tx
	__syncthreads();

	//compute area---------------------------------------------------
	float2 v0 = make_float2(0, 0);
	float2 v1 = make_float2(0, 0);
	for (int ok = 1, OK = (K >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float2 a = As[buf][ik][ty];
			float2 b = Bs[buf][ik][tx];
			simdMM2(v0, a.x, b);
			simdMM2(v1, a.y, b);
		}
		buf ^= 1;

		int Ak = (ok << LB) + tx;
		As[buf][tx][ty].x = get3d(A, batch, Y   , Ak, N, K);//with the same ty
		As[buf][tx][ty].y = get3d(A, batch, Y + 1, Ak, N, K);

		int Bk = (ok << LB) + ty;
		Bs[buf][ty][tx] = *(float2*)(&get3d(B, batch, Bk, X, K, CM));//with the same tx
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float2 a = As[buf][ik][ty];
		float2 b = Bs[buf][ik][tx];
		simdMM2(v0, a.x, b);
		simdMM2(v1, a.y, b);
	}

	*(float2*)(&get3d(C, batch, Y    , X, N, CM)) = v0;
	*(float2*)(&get3d(C, batch, Y + 1, X, N, CM)) = v1;
}

#endif


#ifndef V5
#define V5

#define bmm_kv5(stream, LB, A, B, C, Batch, N, CM, K)\
	bmm_kernel_v5<LB, (1<<LB>>1)>\
		<<< dim3(CM>>LB>>2, N>>LB>>2, Batch), dim3(1<<LB, 1<<LB)>>>\
			(A, B, C, N, CM, K)

//Size = 1, Time = 2.048 msec, Performace = 1048.58 GFlop/s
//Batch -> batch
//N -> Y, M -> X

template<int LB, int STEP>
__global__ void bmm_kernel_v5(
	const float* __restrict__ A,//A[Batch, N, K]
	const float* __restrict__ B,//B[Batch, K, M]
	float* __restrict__ C,//C[Batch, N, M]
	int N, int CM, int K)
{
	int ty = threadIdx.y, tx = threadIdx.x;
	int by = blockIdx.y, bx = blockIdx.x;

	int Y = ((by << LB) + ty) << 2;
	int X = ((bx << LB) + tx) << 2;
	int batch = blockIdx.z;
	
	int tY = Y + ((tx & 1) << 1);
	int tX = X + ((ty & 1) << 1);

	bool buf = 0;
	__shared__ float2 As[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2 Bs[2][1 << LB >> 1][(2 << LB) + 2];//with the same tx

	//preload--------------------------------------------------------
	int Ak = (tx >> 1);
	const int As_x = (tx >> 1), As_y = (ty << 1) + (tx & 1);
	As[buf][As_x][As_y].x = get3d(A, batch, tY    , Ak, N, K);//with the same ty
	As[buf][As_x][As_y].y = get3d(A, batch, tY + 1, Ak, N, K);

	int Bk = (ty >> 1);
	const int Bs_y = (ty >> 1), Bs_x = (tx << 1) + (ty & 1);
	Bs[buf][Bs_y][Bs_x] = *(float2*)(&get3d(B, batch, Bk, tX, K, CM));//with the same tx
	__syncthreads();

	//compute area---------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0);
	float4 v1 = make_float4(0, 0, 0, 0);
	float4 v2 = make_float4(0, 0, 0, 0);
	float4 v3 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (K << 1 >> LB); ok < OK; ok++)
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

		int Ak = ((ok << LB) + tx) >> 1;
		As[buf][As_x][As_y].x = get3d(A, batch, tY    , Ak, N, K);//with the same ty
		As[buf][As_x][As_y].y = get3d(A, batch, tY + 1, Ak, N, K);

		int Bk = ((ok << LB) + ty) >> 1;
		Bs[buf][Bs_y][Bs_x] = *(float2*)(&get3d(B, batch, Bk, tX, K, CM));//with the same tx
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float4 b = *(float4*)(&Bs[buf][ik][tx << 1]);
		float4 a = *(float4*)(&As[buf][ik][ty << 1]);
		simdMM4(v0, a.x, b);
		simdMM4(v1, a.y, b);
		simdMM4(v2, a.z, b);
		simdMM4(v3, a.w, b);
	}

	*(float4*)(&get3d(C, batch, Y    , X, N, CM)) = v0;
	*(float4*)(&get3d(C, batch, Y + 1, X, N, CM)) = v1;
	*(float4*)(&get3d(C, batch, Y + 2, X, N, CM)) = v2;
	*(float4*)(&get3d(C, batch, Y + 3, X, N, CM)) = v3;
}

#endif


#ifndef V6
#define V6

#define bmm_kv6(stream, LB, A, B, C, Batch, N, CM, K)\
	bmm_kernel_v6<LB, (1<<LB>>1)>\
		<<< dim3((CM>>LB>>3), (N>>LB>>3), Batch), dim3(1<<LB, 1<<LB)>>>\
			(A, B, C, N, CM, K)

//Size = 1, Time = 1.688 msec, Performace = 1272.21 GFlop/s
//Batch -> batch
//N -> Y, M -> X

template<int LB, int STEP>
__global__ void bmm_kernel_v6(
	const float* __restrict__ A,//A[Batch, N, K]
	const float* __restrict__ B,//B[Batch, K, M]
	float* __restrict__ C,//C[Batch, N, M]
	int N, int CM, int K)
{
	int ty = threadIdx.y, tx = threadIdx.x;
	int by = blockIdx.y, bx = blockIdx.x;

	int Y = ((by << LB) + ty) << 3, tY = Y + ((tx & 1) << 2);
	int X = ((bx << LB) + tx) << 3, tX = X + ((ty & 1) << 2);
	int batch = blockIdx.z;

	bool buf = 0;
	__shared__ float4 As[2][1 << LB >> 1][(2 << LB) + 1];
	__shared__ float4 Bs[2][1 << LB >> 1][(2 << LB) + 1];//with the same tx

	//preload--------------------------------------------------------
	int Ak = (tx >> 1);
	const int As_x = (tx >> 1), As_y = (ty << 1) + (tx & 1);
	As[buf][As_x][As_y].x = get3d(A, batch, tY    , Ak, N, K);//with the same ty
	As[buf][As_x][As_y].y = get3d(A, batch, tY + 1, Ak, N, K);
	As[buf][As_x][As_y].z = get3d(A, batch, tY + 2, Ak, N, K);//with the same ty
	As[buf][As_x][As_y].w = get3d(A, batch, tY + 3, Ak, N, K);

	int Bk = (ty >> 1);
	const int Bs_y = (ty >> 1), Bs_x = (tx << 1) + (ty & 1);
	Bs[buf][Bs_y][Bs_x] = *(float4*)(&get3d(B, batch, Bk, tX, K, CM));//with the same tx
	__syncthreads();

	//compute area---------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0),  v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0),  v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0),  v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0),  v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0),  v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (K << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float4 a0 = As[buf][ik][(ty << 1)], a1 = As[buf][ik][(ty << 1) + 1];
			float4 b0 = Bs[buf][ik][(tx << 1)], b1 = Bs[buf][ik][(tx << 1) + 1];
			
			simdMM4( v0, a0.x, b0); simdMM4(v1, a0.x, b1);
			simdMM4( v2, a0.y, b0); simdMM4(v3, a0.y, b1);
			simdMM4( v4, a0.z, b0); simdMM4(v5, a0.z, b1);
			simdMM4( v6, a0.w, b0); simdMM4(v7, a0.w, b1);
			simdMM4( v8, a1.x, b0); simdMM4(v9, a1.x, b1);
			simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
			simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
			simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
		}
		buf ^= 1;

		int Ak = ((ok << LB) + tx) >> 1;
		As[buf][As_x][As_y].x = get3d(A, batch, tY, Ak, N, K);//with the same ty
		As[buf][As_x][As_y].y = get3d(A, batch, tY + 1, Ak, N, K);
		As[buf][As_x][As_y].z = get3d(A, batch, tY + 2, Ak, N, K);//with the same ty
		As[buf][As_x][As_y].w = get3d(A, batch, tY + 3, Ak, N, K);

		int Bk = ((ok << LB) + ty) >> 1;
		Bs[buf][Bs_y][Bs_x] = *(float4*)(&get3d(B, batch, Bk, tX, K, CM));//with the same tx
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float4 a0 = As[buf][ik][(ty << 1)], a1 = As[buf][ik][(ty << 1) + 1];
		float4 b0 = Bs[buf][ik][(tx << 1)], b1 = Bs[buf][ik][(tx << 1) + 1];

		simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
		simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
		simdMM4(v4, a0.z, b0); simdMM4(v5, a0.z, b1);
		simdMM4(v6, a0.w, b0); simdMM4(v7, a0.w, b1);
		simdMM4(v8, a1.x, b0); simdMM4(v9, a1.x, b1);
		simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
		simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
		simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
	}

	*(float4*)(&get3d(C, batch, Y    , X, N, CM)) =  v0; *(float4*)(&get3d(C, batch, Y    , X + 4, N, CM)) =  v1;
	*(float4*)(&get3d(C, batch, Y + 1, X, N, CM)) =  v2; *(float4*)(&get3d(C, batch, Y + 1, X + 4, N, CM)) =  v3;
	*(float4*)(&get3d(C, batch, Y + 2, X, N, CM)) =  v4; *(float4*)(&get3d(C, batch, Y + 2, X + 4, N, CM)) =  v5;
	*(float4*)(&get3d(C, batch, Y + 3, X, N, CM)) =  v6; *(float4*)(&get3d(C, batch, Y + 3, X + 4, N, CM)) =  v7;
	*(float4*)(&get3d(C, batch, Y + 4, X, N, CM)) =  v8; *(float4*)(&get3d(C, batch, Y + 4, X + 4, N, CM)) =  v9;
	*(float4*)(&get3d(C, batch, Y + 5, X, N, CM)) = v10; *(float4*)(&get3d(C, batch, Y + 5, X + 4, N, CM)) = v11;
	*(float4*)(&get3d(C, batch, Y + 6, X, N, CM)) = v12; *(float4*)(&get3d(C, batch, Y + 6, X + 4, N, CM)) = v13;
	*(float4*)(&get3d(C, batch, Y + 7, X, N, CM)) = v14; *(float4*)(&get3d(C, batch, Y + 7, X + 4, N, CM)) = v15;
}

#endif


#ifndef V7
#define V7

#define bmm_kv7(stream, LB, A, B, C, Batch, N, CM, K)\
	bmm_kernel_v7<LB, (1<<LB>>1)>\
		<<< dim3((CM>>LB>>3), (N>>LB>>3), Batch), dim3(1<<LB, 1<<LB)>>>\
			(A, B, C, N, CM, K)

//Size = 1, Time = 1.602 msec, Performace = 1340.5 GFlop/s
//Batch -> batch
//N -> Y, M -> X
//K % 4 == 0
//Batch % 4 == 0
//M % 4 == 0

template<int LB, int STEP>
__global__ void bmm_kernel_v7(
	const float* __restrict__ A,//A[Batch, N, K]
	const float* __restrict__ B,//B[Batch, K, M]
	float* __restrict__ C,//C[Batch, N, M]
	int N, int CM, int K)
{
	int ty = threadIdx.y, tx = threadIdx.x;
	
	//prepared for Y:N
	int Y = ((blockIdx.y << LB) + ty) << 3;
	const int tY0 = (Y + ((tx & 1) << 2)) * K;
	const int tY1 = tY0 + K, tY2 = tY1 + K, tY3 = tY2 + K;

	//prepared for X
	int X = ((blockIdx.x << LB) + tx) << 3, tX = X + ((ty & 1) << 2);

	//compute start offset of A, B, C
	int batch = blockIdx.z; 
	A += (batch * N * K);//A[batch]
	B += (batch * K * CM) + tX;//B[batch, 0, tX]
	C += (batch * N * CM);//C[batch]
	
	bool buf = 0;
	__shared__ float4 As[2][1 << LB >> 1][(2 << LB) + 1];
	__shared__ float4 Bs[2][1 << LB >> 1][(2 << LB) + 1];

	//preload--------------------------------------------------------
	int Ak = (tx >> 1);
	const int As_x = (tx >> 1), As_y = (ty << 1) + (tx & 1);
	As[buf][As_x][As_y].x = A[tY0 + Ak];//with the same ty
	As[buf][As_x][As_y].y = A[tY1 + Ak];
	As[buf][As_x][As_y].z = A[tY2 + Ak];
	As[buf][As_x][As_y].w = A[tY3 + Ak];

	int Bk = (ty >> 1);
	const int Bs_y = (ty >> 1), Bs_x = (tx << 1) + (ty & 1);
	Bs[buf][Bs_y][Bs_x] = *(float4*)(&B[Bk * CM]);//with the same tx
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
			float4 b0 = Bs[buf][ik][(tx << 1)], b1 = Bs[buf][ik][(tx << 1) + 1];
			float4 a0 = As[buf][ik][(ty << 1)], a1 = As[buf][ik][(ty << 1) + 1];

			simdMM4( v0, a0.x, b0); simdMM4( v1, a0.x, b1);
			simdMM4( v2, a0.y, b0); simdMM4( v3, a0.y, b1);
			simdMM4( v4, a0.z, b0); simdMM4( v5, a0.z, b1);
			simdMM4( v6, a0.w, b0); simdMM4( v7, a0.w, b1);
			simdMM4( v8, a1.x, b0); simdMM4( v9, a1.x, b1);
			simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
			simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
			simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
		}
		buf ^= 1;

		int Ak = ((ok << LB) + tx) >> 1;
		As[buf][As_x][As_y].x = A[tY0 + Ak];
		As[buf][As_x][As_y].y = A[tY1 + Ak];
		As[buf][As_x][As_y].z = A[tY2 + Ak];
		As[buf][As_x][As_y].w = A[tY3 + Ak];

		int Bk = ((ok << LB) + ty) >> 1;
		Bs[buf][Bs_y][Bs_x] = *(float4*)(&B[Bk * CM]);//with the same tx
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float4 b0 = Bs[buf][ik][(tx << 1)], b1 = Bs[buf][ik][(tx << 1) + 1];
		float4 a0 = As[buf][ik][(ty << 1)], a1 = As[buf][ik][(ty << 1) + 1];
		
		simdMM4( v0, a0.x, b0); simdMM4( v1, a0.x, b1);
		simdMM4( v2, a0.y, b0); simdMM4( v3, a0.y, b1);
		simdMM4( v4, a0.z, b0); simdMM4( v5, a0.z, b1);
		simdMM4( v6, a0.w, b0); simdMM4( v7, a0.w, b1);
		simdMM4( v8, a1.x, b0); simdMM4( v9, a1.x, b1);
		simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
		simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
		simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
	}

	int Y0 = (Y * CM) + X; 
	int Y1 = Y0 + CM, Y2 = Y1 + CM, Y3 = Y2 + CM;
	int Y4 = Y3 + CM, Y5 = Y4 + CM, Y6 = Y5 + CM, Y7 = Y6 + CM;

	*(float4*)(C + Y0) =  v0; *(float4*)(C + Y0 + 4) =  v1;
	*(float4*)(C + Y1) =  v2; *(float4*)(C + Y1 + 4) =  v3;
	*(float4*)(C + Y2) =  v4; *(float4*)(C + Y2 + 4) =  v5;
	*(float4*)(C + Y3) =  v6; *(float4*)(C + Y3 + 4) =  v7;
	*(float4*)(C + Y4) =  v8; *(float4*)(C + Y4 + 4) =  v9;
	*(float4*)(C + Y5) = v10; *(float4*)(C + Y5 + 4) = v11;
	*(float4*)(C + Y6) = v12; *(float4*)(C + Y6 + 4) = v13;
	*(float4*)(C + Y7) = v14; *(float4*)(C + Y7 + 4) = v15;
}

#endif