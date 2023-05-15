#pragma once

#ifndef MATMUL_T2_KERNEL_K2POW_H
#define MATMUL_T2_KERNEL_K2POW_H

//K is power of 2
//B   belongs to Mat[M, K]
//B^T belongs to Mat[K, M]
//get(B^T, k, j, M) = get(B, j, k, K)
#ifndef MATMUL_T2_KERNEL_K2POW_CALL
#define MATMUL_T2_KERNEL_K2POW_CALL

//LB = log2(BLOCK_SIZE)

#define	k88T2K2(LB, stream, A, B, C, N, M, LK, SC) \
	kernel_t2_8_8_K2pow<LB, (1<<LB>>1)>\
		<<< dim3(M>>3>>LB, N>>3>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SC, LK)

#define k84T2K2(LB, stream, A, B, C, N, M, LK, SC) \
	kernel_t2_8_4_K2pow<LB, (1<<LB>>1)>\
		<<< dim3(M>>2>>LB, N>>3>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SC, LK)

#define k48T2K2(LB, stream, A, B, C, N, M, LK, SC) \
	kernel_t2_4_8_K2pow<LB, (1<<LB>>1)>\
		<<< dim3(M>>3>>LB, N>>2>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SC, LK)

#define k82T2K2(LB, stream, A, B, C, N, M, LK, SC) \
	kernel_t2_8_2_K2pow<LB, (1<<LB>>1)>\
		<<< dim3(M>>1>>LB, N>>3>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SC, LK)

#define k28T2K2(LB, stream, A, B, C, N, M, LK, SC) \
	kernel_t2_2_8_K2pow<LB, (1<<LB>>1)>\
		<<< dim3(M>>3>>LB, N>>1>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SC, LK)

#define k81T2K2(LB, stream, A, B, C, N, M, LK, SC) \
	kernel_t2_8_1_K2pow<LB, (1<<LB)>\
		<<< dim3(M>>LB, N>>3>>LB), dim3(1<<LB, 1<<LB), 0, stream >>> \
			(A, B, C, SC, LK)

#define k18T2K2(LB, stream, A, B, C, N, M, LK, SC) \
	kernel_t2_1_8_K2pow<LB, (1<<LB)>\
		<<< dim3(M>>3>>LB, N>>LB), dim3(1<<LB, 1<<LB), 0, stream >>> \
			(A, B, C, SC, LK)

#define k44T2K2(LB, stream, A, B, C, N, M, LK, SC) \
	kernel_t2_4_4_K2pow<LB, (1<<LB>>1)>\
		<<< dim3(M>>2>>LB, N>>2>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SC, LK)

#define k42T2K2(LB, stream, A, B, C, N, M, LK, SC) \
	kernel_t2_4_2_K2pow<LB, (1<<LB>>1)>\
		<<< dim3(M>>1>>LB, N>>2>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SC, LK)

#define k24T2K2(LB, stream, A, B, C, N, M, LK, SC) \
	kernel_t2_2_4_K2pow<LB, (1<<LB>>1)>\
		<<< dim3(M>>2>>LB, N>>1>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SC, LK)

#define k22T2K2(LB, stream, A, B, C, N, M, LK, SC) \
	kernel_t2_2_2_K2pow<LB, (1<<LB)>\
		<<< dim3(M>>1>>LB, N>>1>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(A, B, C, SC, LK)

#define knaiveT2K2(GRID_SIZE, stream, A, B, C, N, M, LK, SC) \
	kernel_t2_naive_K2pow\
		<<< dim3(GRID_SIZE, GRID_SIZE), dim3(M/GRID_SIZE, N/GRID_SIZE), 0, stream>>>\
			(A, B, C, SC, LK)

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), K % BLOCK_SIZE/2 == 0
#ifndef MATMUL_T2_KERNEL_8_8_K2POW
#define MATMUL_T2_KERNEL_8_8_K2POW

//LB = 4: Size = 1, Time = 1.512 msec, Performace = 1420.29 GFlop/s
template<int LB, int STEP>
__global__ void kernel_t2_8_8_K2pow(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float*  __restrict__ C,
	int SC, int LK)
{
	bool buf = 0;
	__shared__ float4 As[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Bs[2][1 << LB][(1 << LB) + 1];

	int Y = (blockIdx.y << 3 << LB);//Y = blockIdx.y * 8 * BLOCK_SIZE
	int X = (blockIdx.x << 3 << LB);//X = blockIdx.x * 8 * BLOCK_SIZE
	A = A + (Y << LK);//A[Y, 0]
	B = B + (X << LK);//B[X, 0]
	C = &get(C, Y, X, SC);//C[Y, X]

	int tx = threadIdx.x, ty = threadIdx.y;
	const int Ay = (ty << 3) + ((tx >= STEP) << 2), Ax = tx - ((tx >= STEP) << LB >> 1);
	const int Bx = (tx << 3) + ((ty >= STEP) << 2), By = ty - ((ty >= STEP) << LB >> 1);

	As[buf][tx][ty].x = lget(A, Ay    , Ax, LK);//transpose A
	As[buf][tx][ty].y = lget(A, Ay + 1, Ax, LK);
	As[buf][tx][ty].z = lget(A, Ay + 2, Ax, LK);
	As[buf][tx][ty].w = lget(A, Ay + 3, Ax, LK);
	Bs[buf][ty][tx].x = lget(B, Bx    , By, LK);
	Bs[buf][ty][tx].y = lget(B, Bx + 1, By, LK);
	Bs[buf][ty][tx].z = lget(B, Bx + 2, By, LK);
	Bs[buf][ty][tx].w = lget(B, Bx + 3, By, LK);
	A += STEP; B += STEP;
	__syncthreads();

	//compute area-------------------------------------------------------
	float4  c0 = make_float4(0, 0, 0, 0),  c1 = make_float4(0, 0, 0, 0);
	float4  c2 = make_float4(0, 0, 0, 0),  c3 = make_float4(0, 0, 0, 0);
	float4  c4 = make_float4(0, 0, 0, 0),  c5 = make_float4(0, 0, 0, 0);
	float4  c6 = make_float4(0, 0, 0, 0),  c7 = make_float4(0, 0, 0, 0);
	float4  c8 = make_float4(0, 0, 0, 0),  c9 = make_float4(0, 0, 0, 0);
	float4 c10 = make_float4(0, 0, 0, 0), c11 = make_float4(0, 0, 0, 0);
	float4 c12 = make_float4(0, 0, 0, 0), c13 = make_float4(0, 0, 0, 0);
	float4 c14 = make_float4(0, 0, 0, 0), c15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (2 << LK >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a0 = As[buf][ik][ty], a1 = As[buf][ik + STEP][ty];
			float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP][tx];
			
			simdMM4( c0, a0.x, b0); simdMM4( c1, a0.x, b1);
			simdMM4( c2, a0.y, b0); simdMM4( c3, a0.y, b1);
			simdMM4( c4, a0.z, b0); simdMM4( c5, a0.z, b1);
			simdMM4( c6, a0.w, b0); simdMM4( c7, a0.w, b1);
			simdMM4( c8, a1.x, b0); simdMM4( c9, a1.x, b1);
			simdMM4(c10, a1.y, b0); simdMM4(c11, a1.y, b1);
			simdMM4(c12, a1.z, b0); simdMM4(c13, a1.z, b1);
			simdMM4(c14, a1.w, b0); simdMM4(c15, a1.w, b1);
		}
		buf ^= 1;

		As[buf][tx][ty].x = lget(A, Ay    , Ax, LK);//transpose A
		As[buf][tx][ty].y = lget(A, Ay + 1, Ax, LK);
		As[buf][tx][ty].z = lget(A, Ay + 2, Ax, LK);
		As[buf][tx][ty].w = lget(A, Ay + 3, Ax, LK);
		Bs[buf][ty][tx].x = lget(B, Bx    , By, LK);
		Bs[buf][ty][tx].y = lget(B, Bx + 1, By, LK);
		Bs[buf][ty][tx].z = lget(B, Bx + 2, By, LK);
		Bs[buf][ty][tx].w = lget(B, Bx + 3, By, LK);
		A += STEP; B += STEP;
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a0 = As[buf][ik][ty], a1 = As[buf][ik + STEP][ty];
		float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP][tx];

		simdMM4( c0, a0.x, b0); simdMM4( c1, a0.x, b1);
		simdMM4( c2, a0.y, b0); simdMM4( c3, a0.y, b1);
		simdMM4( c4, a0.z, b0); simdMM4( c5, a0.z, b1);
		simdMM4( c6, a0.w, b0); simdMM4( c7, a0.w, b1);
		simdMM4( c8, a1.x, b0); simdMM4( c9, a1.x, b1);
		simdMM4(c10, a1.y, b0); simdMM4(c11, a1.y, b1);
		simdMM4(c12, a1.z, b0); simdMM4(c13, a1.z, b1);
		simdMM4(c14, a1.w, b0); simdMM4(c15, a1.w, b1);
	}

	ty <<= 3; tx <<= 3;
	*(float4*)(&get(C, ty    , tx, SC)) = c0;  *(float4*)(&get(C, ty    , tx + 4, SC)) = c1;
	*(float4*)(&get(C, ty + 1, tx, SC)) = c2;  *(float4*)(&get(C, ty + 1, tx + 4, SC)) = c3;
	*(float4*)(&get(C, ty + 2, tx, SC)) = c4;  *(float4*)(&get(C, ty + 2, tx + 4, SC)) = c5;
	*(float4*)(&get(C, ty + 3, tx, SC)) = c6;  *(float4*)(&get(C, ty + 3, tx + 4, SC)) = c7;
	*(float4*)(&get(C, ty + 4, tx, SC)) = c8;  *(float4*)(&get(C, ty + 4, tx + 4, SC)) = c9;
	*(float4*)(&get(C, ty + 5, tx, SC)) = c10; *(float4*)(&get(C, ty + 5, tx + 4, SC)) = c11;
	*(float4*)(&get(C, ty + 6, tx, SC)) = c12; *(float4*)(&get(C, ty + 6, tx + 4, SC)) = c13;
	*(float4*)(&get(C, ty + 7, tx, SC)) = c14; *(float4*)(&get(C, ty + 7, tx + 4, SC)) = c15;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*4), K % BLOCK_SIZE/2 == 0
#ifndef MATMUL_T2_KERNEL_8_4_K2POW
#define MATMUL_T2_KERNEL_8_4_K2POW

//LB = 4: Size = 1, Time = 1.624 msec, Performace = 1322.34 GFlop/s
template<int LB, int STEP>
__global__ void kernel_t2_8_4_K2pow(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float*  __restrict__ C,
	int SC, int LK)
{
	bool buf = 0;
	__shared__ float4 As[2][1 << LB][(1 << LB) + 1];//followed k88T2
	__shared__ float2 Bs[2][1 << LB >> 1][(2 << LB) + 2];//followed k44T2

	int Y = (blockIdx.y << 3 << LB);//Y = blockIdx.y * 8 * BLOCK_SIZE
	int X = (blockIdx.x << 2 << LB);//X = blockIdx.x * 4 * BLOCK_SIZE
	A = A + (Y << LK);//A[Y, 0]
	B = B + (X << LK);//B[X, 0]
	C = &get(C, Y, X, SC);//C[Y, X]

	int tx = threadIdx.x, ty = threadIdx.y;
	const int Ay = (ty << 3) + ((tx >= STEP) << 2), Ax = tx - (tx >= STEP)*STEP;
	const int By = (ty >> 1), Bx = (tx << 2) + ((ty & 1) << 1);
	const int Bs_y = By, Bs_x = ((tx << 1) + (ty & 1));

	As[buf][tx][ty].x = lget(A, Ay    , Ax, LK);//transpose A
	As[buf][tx][ty].y = lget(A, Ay + 1, Ax, LK);
	As[buf][tx][ty].z = lget(A, Ay + 2, Ax, LK);
	As[buf][tx][ty].w = lget(A, Ay + 3, Ax, LK);
	Bs[buf][Bs_y][Bs_x].x = lget(B, Bx    , By, LK); 
	Bs[buf][Bs_y][Bs_x].y = lget(B, Bx + 1, By, LK);
	A += STEP; B += STEP;
	__syncthreads();

	//compute area-------------------------------------------------------
	float4  c0 = make_float4(0, 0, 0, 0);
	float4  c2 = make_float4(0, 0, 0, 0);
	float4  c4 = make_float4(0, 0, 0, 0);
	float4  c6 = make_float4(0, 0, 0, 0);
	float4  c8 = make_float4(0, 0, 0, 0);
	float4 c10 = make_float4(0, 0, 0, 0);
	float4 c12 = make_float4(0, 0, 0, 0);
	float4 c14 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (2 << LK >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a0 = As[buf][ik][ty], a1 = As[buf][ik + STEP][ty];
			float4 b0 = *(float4*)(&Bs[buf][ik][tx << 1]);

			simdMM4( c0, a0.x, b0);
			simdMM4( c2, a0.y, b0);
			simdMM4( c4, a0.z, b0);
			simdMM4( c6, a0.w, b0);
			simdMM4( c8, a1.x, b0);
			simdMM4(c10, a1.y, b0);
			simdMM4(c12, a1.z, b0);
			simdMM4(c14, a1.w, b0);
		}
		buf ^= 1;

		As[buf][tx][ty].x = lget(A, Ay    , Ax, LK);//transpose A
		As[buf][tx][ty].y = lget(A, Ay + 1, Ax, LK);
		As[buf][tx][ty].z = lget(A, Ay + 2, Ax, LK);
		As[buf][tx][ty].w = lget(A, Ay + 3, Ax, LK);
		Bs[buf][Bs_y][Bs_x].x = lget(B, Bx    , By, LK);
		Bs[buf][Bs_y][Bs_x].y = lget(B, Bx + 1, By, LK);
		A += STEP; B += STEP;
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a0 = As[buf][ik][ty], a1 = As[buf][ik + STEP][ty];
		float4 b0 = *(float4*)(&Bs[buf][ik][tx << 1]);

		simdMM4( c0, a0.x, b0);
		simdMM4( c2, a0.y, b0);
		simdMM4( c4, a0.z, b0);
		simdMM4( c6, a0.w, b0);
		simdMM4( c8, a1.x, b0);
		simdMM4(c10, a1.y, b0);
		simdMM4(c12, a1.z, b0);
		simdMM4(c14, a1.w, b0);
	}

	ty <<= 3; tx <<= 2;
	*(float4*)(&get(C, ty    , tx, SC)) = c0;
	*(float4*)(&get(C, ty + 1, tx, SC)) = c2;
	*(float4*)(&get(C, ty + 2, tx, SC)) = c4;
	*(float4*)(&get(C, ty + 3, tx, SC)) = c6;
	*(float4*)(&get(C, ty + 4, tx, SC)) = c8;
	*(float4*)(&get(C, ty + 5, tx, SC)) = c10;
	*(float4*)(&get(C, ty + 6, tx, SC)) = c12;
	*(float4*)(&get(C, ty + 7, tx, SC)) = c14;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*8), K % BLOCK_SIZE/2 == 0
#ifndef MATMUL_T2_KERNEL_4_8_K2POW
#define MATMUL_T2_KERNEL_4_8_K2POW

//LB = 4: Size = 1, Time = 1.79 msec, Performace = 1199.71 GFlop/s
template<int LB, int STEP>
__global__ void kernel_t2_4_8_K2pow(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float*  __restrict__ C,
	int SC, int LK)
{
	bool buf = 0;
	__shared__ float2 As[2][1 << LB >> 1][(2 << LB) + 2];//followed k44T2
	__shared__ float4 Bs[2][1 << LB][(1 << LB) + 1];//followed k88T2

	int Y = (blockIdx.y << 2 << LB);//Y = blockIdx.y * 4 * BLOCK_SIZE
	int X = (blockIdx.x << 3 << LB);//X = blockIdx.x * 8 * BLOCK_SIZE
	A = A + (Y << LK);//A[Y, 0]
	B = B + (X << LK);//B[X, 0]
	C = &get(C, Y, X, SC);//C[Y, X]

	int tx = threadIdx.x, ty = threadIdx.y;
	const int Ax = (tx >> 1), Ay = (ty << 2) + ((tx & 1) << 1);
	const int As_x = Ax, As_y = ((ty << 1) + (tx & 1));
	const int Bx = (tx << 3) + ((ty >= STEP) << 2), By = ty - ((ty >= STEP) << LB >> 1);

	As[buf][As_x][As_y].x = lget(A, Ay    , Ax, LK);//transpose A
	As[buf][As_x][As_y].y = lget(A, Ay + 1, Ax, LK);
	Bs[buf][ty][tx].x = lget(B, Bx    , By, LK);
	Bs[buf][ty][tx].y = lget(B, Bx + 1, By, LK);
	Bs[buf][ty][tx].z = lget(B, Bx + 2, By, LK);
	Bs[buf][ty][tx].w = lget(B, Bx + 3, By, LK);
	A += STEP; B += STEP;
	__syncthreads();

	//compute area-------------------------------------------------------
	float4 c0 = make_float4(0, 0, 0, 0), c1 = make_float4(0, 0, 0, 0);
	float4 c2 = make_float4(0, 0, 0, 0), c3 = make_float4(0, 0, 0, 0);
	float4 c4 = make_float4(0, 0, 0, 0), c5 = make_float4(0, 0, 0, 0);
	float4 c6 = make_float4(0, 0, 0, 0), c7 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (2 << LK >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a0 = *(float4*)(&As[buf][ik][ty << 1]);
			float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP][tx];

			simdMM4(c0, a0.x, b0); simdMM4(c1, a0.x, b1);
			simdMM4(c2, a0.y, b0); simdMM4(c3, a0.y, b1);
			simdMM4(c4, a0.z, b0); simdMM4(c5, a0.z, b1);
			simdMM4(c6, a0.w, b0); simdMM4(c7, a0.w, b1);
		}
		buf ^= 1;

		As[buf][As_x][As_y].x = lget(A, Ay    , Ax, LK);//transpose A
		As[buf][As_x][As_y].y = lget(A, Ay + 1, Ax, LK);
		Bs[buf][ty][tx].x = lget(B, Bx    , By, LK);
		Bs[buf][ty][tx].y = lget(B, Bx + 1, By, LK);
		Bs[buf][ty][tx].z = lget(B, Bx + 2, By, LK);
		Bs[buf][ty][tx].w = lget(B, Bx + 3, By, LK);
		A += STEP; B += STEP;
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a0 = *(float4*)(&As[buf][ik][ty << 1]);
		float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP][tx];

		simdMM4(c0, a0.x, b0); simdMM4(c1, a0.x, b1);
		simdMM4(c2, a0.y, b0); simdMM4(c3, a0.y, b1);
		simdMM4(c4, a0.z, b0); simdMM4(c5, a0.z, b1);
		simdMM4(c6, a0.w, b0); simdMM4(c7, a0.w, b1);
	}

	ty <<= 2; tx <<= 3;
	*(float4*)(&get(C, ty    , tx, SC)) = c0; *(float4*)(&get(C, ty    , tx + 4, SC)) = c1;
	*(float4*)(&get(C, ty + 1, tx, SC)) = c2; *(float4*)(&get(C, ty + 1, tx + 4, SC)) = c3;
	*(float4*)(&get(C, ty + 2, tx, SC)) = c4; *(float4*)(&get(C, ty + 2, tx + 4, SC)) = c5;
	*(float4*)(&get(C, ty + 3, tx, SC)) = c6; *(float4*)(&get(C, ty + 3, tx + 4, SC)) = c7;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*2), K % BLOCK_SIZE/2 == 0
#ifndef MATMUL_T2_KERNEL_8_2_K2POW
#define MATMUL_T2_KERNEL_8_2_K2POW

//LB = 4: Size = 1, Time = 2.104 msec, Performace = 1020.67 GFlop/s
template<int LB, int STEP>
__global__ void kernel_t2_8_2_K2pow(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float*  __restrict__ C,
	int SC, int LK)
{
	bool buf = 0;
	__shared__ float4 As[2][1 << LB][(1 << LB) + 1];//followed k88T2
	__shared__ float  Bs[2][1 << LB >> 1][(2 << LB) + 2];//followed k44T2

	int Y = (blockIdx.y << 3 << LB);//Y = blockIdx.y * 8 * BLOCK_SIZE
	int X = (blockIdx.x << 1 << LB);//X = blockIdx.x * 2 * BLOCK_SIZE
	A = A + (Y << LK);//A[Y, 0]
	B = B + (X << LK);//B[X, 0]
	C = &get(C, Y, X, SC);//C[Y, X]

	int tx = threadIdx.x, ty = threadIdx.y;
	const int Ay = (ty << 3) + ((tx >= STEP) << 2), Ax = tx - ((tx >= STEP) << LB >> 1);
	const int By = (ty >> 1), Bx = (tx << 1) + (ty & 1);
	const int Bs_y = By, Bs_x = ((tx << 1) + (ty & 1));
	const int Bxy = (Bx << LK) + By;//[Bx, By]

	As[buf][tx][ty].x = lget(A, Ay    , Ax, LK);//transpose A
	As[buf][tx][ty].y = lget(A, Ay + 1, Ax, LK);
	As[buf][tx][ty].z = lget(A, Ay + 2, Ax, LK);
	As[buf][tx][ty].w = lget(A, Ay + 3, Ax, LK);
	Bs[buf][Bs_y][Bs_x] = B[Bxy];
	A += STEP; B += STEP;
	__syncthreads();

	//compute area-------------------------------------------------------
	float2  c0 = make_float2(0, 0);
	float2  c2 = make_float2(0, 0);
	float2  c4 = make_float2(0, 0);
	float2  c6 = make_float2(0, 0);
	float2  c8 = make_float2(0, 0);
	float2 c10 = make_float2(0, 0);
	float2 c12 = make_float2(0, 0);
	float2 c14 = make_float2(0, 0);

	for (int ok = 1, OK = (2 << LK >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a0 = As[buf][ik][ty], a1 = As[buf][ik + STEP][ty];
			float2 b0 = *(float2*)(&Bs[buf][ik][tx << 1]);
			
			simdMM2( c0, a0.x, b0);
			simdMM2( c2, a0.y, b0);
			simdMM2( c4, a0.z, b0);
			simdMM2( c6, a0.w, b0);
			simdMM2( c8, a1.x, b0);
			simdMM2(c10, a1.y, b0);
			simdMM2(c12, a1.z, b0);
			simdMM2(c14, a1.w, b0);
		}
		buf ^= 1;

		As[buf][tx][ty].x = lget(A, Ay    , Ax, LK);//transpose A
		As[buf][tx][ty].y = lget(A, Ay + 1, Ax, LK);
		As[buf][tx][ty].z = lget(A, Ay + 2, Ax, LK);
		As[buf][tx][ty].w = lget(A, Ay + 3, Ax, LK);
		Bs[buf][Bs_y][Bs_x] = B[Bxy];
		A += STEP; B += STEP;
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a0 = As[buf][ik][ty], a1 = As[buf][ik + STEP][ty];
		float2 b0 = *(float2*)(&Bs[buf][ik][tx << 1]);

		simdMM2( c0, a0.x, b0);
		simdMM2( c2, a0.y, b0);
		simdMM2( c4, a0.z, b0);
		simdMM2( c6, a0.w, b0);
		simdMM2( c8, a1.x, b0);
		simdMM2(c10, a1.y, b0);
		simdMM2(c12, a1.z, b0);
		simdMM2(c14, a1.w, b0);
	}

	ty <<= 3; tx <<= 1;
	*(float2*)(&get(C, ty    , tx, SC)) = c0;
	*(float2*)(&get(C, ty + 1, tx, SC)) = c2;
	*(float2*)(&get(C, ty + 2, tx, SC)) = c4;
	*(float2*)(&get(C, ty + 3, tx, SC)) = c6;
	*(float2*)(&get(C, ty + 4, tx, SC)) = c8;
	*(float2*)(&get(C, ty + 5, tx, SC)) = c10;
	*(float2*)(&get(C, ty + 6, tx, SC)) = c12;
	*(float2*)(&get(C, ty + 7, tx, SC)) = c14;
}

#endif 


//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*8), K % BLOCK_SIZE/2 == 0
#ifndef MATMUL_T2_KERNEL_2_8_K2POW
#define MATMUL_T2_KERNEL_2_8_K2POW

//LB = 4: Size = 1, Time = 2.742 msec, Performace = 783.181 GFlop/s
template<int LB, int STEP>
__global__ void kernel_t2_2_8_K2pow(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float*  __restrict__ C,
	int SC, int LK)
{
	bool buf = 0;
	__shared__ float  As[2][1 << LB >> 1][(2 << LB) + 2];//followed k44T2
	__shared__ float4 Bs[2][1 << LB][(1 << LB) + 1];//followed k88T2

	int Y = (blockIdx.y << 1 << LB);//Y = blockIdx.y * 2 * BLOCK_SIZE
	int X = (blockIdx.x << 3 << LB);//X = blockIdx.x * 8 * BLOCK_SIZE
	A = A + (Y << LK);//A[Y, 0]
	B = B + (X << LK);//B[X, 0]
	C = &get(C, Y, X, SC);//C[Y, X]

	int tx = threadIdx.x, ty = threadIdx.y;
	const int Ax = (tx >> 1), Ay = (ty << 1) + (tx & 1);
	const int As_x = Ax, As_y = ((ty << 1) + (tx & 1));
	const int Bx = (tx << 3) + ((ty >= STEP) << 2), By = ty - ((ty >= STEP) << LB >> 1);
	const int Ayx = (Ay << LK) + Ax;//[Ay, Ax]

	As[buf][As_x][As_y] = A[Ayx];//transpose A
	Bs[buf][ty][tx].x = lget(B, Bx    , By, LK);
	Bs[buf][ty][tx].y = lget(B, Bx + 1, By, LK);
	Bs[buf][ty][tx].z = lget(B, Bx + 2, By, LK);
	Bs[buf][ty][tx].w = lget(B, Bx + 3, By, LK);
	A += STEP; B += STEP;
	__syncthreads();

	//compute area-------------------------------------------------------
	float4 c0 = make_float4(0, 0, 0, 0), c1 = make_float4(0, 0, 0, 0);
	float4 c2 = make_float4(0, 0, 0, 0), c3 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (2 << LK >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float2 a0 = *(float2*)(&As[buf][ik][ty << 1]);
			float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP][tx];
			simdMM4(c0, a0.x, b0); simdMM4(c1, a0.x, b1);
			simdMM4(c2, a0.y, b0); simdMM4(c3, a0.y, b1);
		}
		buf ^= 1;

		As[buf][As_x][As_y] = A[Ayx];//transpose A
		Bs[buf][ty][tx].x = lget(B, Bx    , By, LK);
		Bs[buf][ty][tx].y = lget(B, Bx + 1, By, LK);
		Bs[buf][ty][tx].z = lget(B, Bx + 2, By, LK);
		Bs[buf][ty][tx].w = lget(B, Bx + 3, By, LK);
		A += STEP; B += STEP;
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float2 a0 = *(float2*)(&As[buf][ik][ty << 1]);
		float4 b0 = Bs[buf][ik][tx], b1 = Bs[buf][ik + STEP][tx];
		simdMM4(c0, a0.x, b0); simdMM4(c1, a0.x, b1);
		simdMM4(c2, a0.y, b0); simdMM4(c3, a0.y, b1);
	}

	ty <<= 1; tx <<= 3;
	*(float4*)(&get(C, ty    , tx, SC)) = c0; *(float4*)(&get(C, ty    , tx + 4, SC)) = c1;
	*(float4*)(&get(C, ty + 1, tx, SC)) = c2; *(float4*)(&get(C, ty + 1, tx + 4, SC)) = c3;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE  ), K % BLOCK_SIZE == 0
#ifndef MATMUL_T2_KERNEL_8_1_K2POW
#define MATMUL_T2_KENERL_8_1_K2POW

//LB = 4: Size = 1, Time = 3.052 msec, Performace = 703.632 GFlop/s
template<int LB, int STEP>
__global__ void kernel_t2_8_1_K2pow(
	const float* __restrict__ A,
	const float* __restrict__ B,
	float* __restrict__ C,
	int SC, int LK)
{
	bool buf = 0;
	__shared__ float4 As[2][1 << LB][(2 << LB) + 1];
	__shared__ float  Bs[2][1 << LB][(1 << LB) + 1];

	int Y = (blockIdx.y << 3 << LB);//blockIdx.y * 8 * BLOCK_SIZE
	int X = (blockIdx.x << LB);//blockIdx.x * BLOCK_SIZE
	A = A + (Y << LK);//A[Y, 0]
	B = B + (X << LK);//B[X, 0]
	C = &get(C, Y, X, SC);//C[Y, X]

	int ty = threadIdx.y, tx = threadIdx.x;
	const int Bxy = (tx << LK) + ty;//[tx, ty]

	As[buf][tx][(ty << 1)    ].x = lget(A, (ty << 3)    , tx, LK);
	As[buf][tx][(ty << 1)    ].y = lget(A, (ty << 3) + 1, tx, LK);
	As[buf][tx][(ty << 1)    ].z = lget(A, (ty << 3) + 2, tx, LK);
	As[buf][tx][(ty << 1)    ].w = lget(A, (ty << 3) + 3, tx, LK);
	As[buf][tx][(ty << 1) + 1].x = lget(A, (ty << 3) + 4, tx, LK);
	As[buf][tx][(ty << 1) + 1].y = lget(A, (ty << 3) + 5, tx, LK);
	As[buf][tx][(ty << 1) + 1].z = lget(A, (ty << 3) + 6, tx, LK);
	As[buf][tx][(ty << 1) + 1].w = lget(A, (ty << 3) + 7, tx, LK);
	Bs[buf][ty][tx] = B[Bxy]; 
	A += STEP; B += STEP;
	__syncthreads();

	//compute area-------------------------------------------------------
	float4 c0 = make_float4(0, 0, 0, 0);
	float4 c1 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (1 << LK >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a0 = As[buf][ik][(ty << 1)    ];
			float4 a1 = As[buf][ik][(ty << 1) + 1];
			float b = Bs[buf][ik][tx];

			simdMM4(c0, b, a0);
			simdMM4(c1, b, a1);
		}
		buf ^= 1;

		As[buf][tx][(ty << 1)    ].x = lget(A, (ty << 3)    , tx, LK);
		As[buf][tx][(ty << 1)    ].y = lget(A, (ty << 3) + 1, tx, LK);
		As[buf][tx][(ty << 1)    ].z = lget(A, (ty << 3) + 2, tx, LK);
		As[buf][tx][(ty << 1)    ].w = lget(A, (ty << 3) + 3, tx, LK);
		As[buf][tx][(ty << 1) + 1].x = lget(A, (ty << 3) + 4, tx, LK);
		As[buf][tx][(ty << 1) + 1].y = lget(A, (ty << 3) + 5, tx, LK);
		As[buf][tx][(ty << 1) + 1].z = lget(A, (ty << 3) + 6, tx, LK);
		As[buf][tx][(ty << 1) + 1].w = lget(A, (ty << 3) + 7, tx, LK);
		Bs[buf][ty][tx] = B[Bxy];
		A += STEP; B += STEP;
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a0 = As[buf][ik][(ty << 1)    ];
		float4 a1 = As[buf][ik][(ty << 1) + 1];
		float b = Bs[buf][ik][tx];

		simdMM4(c0, b, a0);
		simdMM4(c1, b, a1);
	}

	ty <<= 3;
	get(C, ty    , tx, SC) = c0.x;
	get(C, ty + 1, tx, SC) = c0.y;
	get(C, ty + 2, tx, SC) = c0.z;
	get(C, ty + 3, tx, SC) = c0.w;
	get(C, ty + 4, tx, SC) = c1.x;
	get(C, ty + 5, tx, SC) = c1.y;
	get(C, ty + 6, tx, SC) = c1.z;
	get(C, ty + 7, tx, SC) = c1.w;
}

#endif


//(Y: BLOCK_SIZE  , X: BLOCK_SIZE*8), K % BLOCK_SIZE == 0
#ifndef MATMUL_T2_KERNEL_1_8_K2POW
#define MATMUL_T2_KERNEL_1_8_K2POW

//LB = 4: Size = 1, Time = 8.9 msec, Performace = 241.29 GFlop/s
template<int LB, int STEP>
__global__ void kernel_t2_1_8_K2pow(
	const float* __restrict__ A,
	const float* __restrict__ B,
	float* __restrict__ C,
	int SC, int LK)
{
	bool buf = 0;
	__shared__ float   As[2][1 << LB][(1 << LB) + 1];
	__shared__ float4  Bs[2][1 << LB][(2 << LB) + 1];

	int Y = (blockIdx.y << LB);//Y = blockIdx.y * BLOKC_SIZE
	int X = (blockIdx.x << 3 << LB);//X = blockIdx.x * 8 * BLOCK_SIZE
	A = A + (Y << LK);//A[Y, 0]
	B = B + (X << LK);//B[X, 0]
	C = &get(C, Y, X, SC);//C[Y, X]
	
	int ty = threadIdx.y, tx = threadIdx.x;
	const int Ayx = (ty << LK) + tx;//[ty, tx]

	As[buf][tx][ty] = A[Ayx];
	Bs[buf][ty][(tx << 1)    ].x = lget(B, (tx << 3)    , ty, LK);
	Bs[buf][ty][(tx << 1)    ].y = lget(B, (tx << 3) + 1, ty, LK);
	Bs[buf][ty][(tx << 1)    ].z = lget(B, (tx << 3) + 2, ty, LK);
	Bs[buf][ty][(tx << 1)    ].w = lget(B, (tx << 3) + 3, ty, LK);
	Bs[buf][ty][(tx << 1) + 1].x = lget(B, (tx << 3) + 4, ty, LK);
	Bs[buf][ty][(tx << 1) + 1].y = lget(B, (tx << 3) + 5, ty, LK);
	Bs[buf][ty][(tx << 1) + 1].z = lget(B, (tx << 3) + 6, ty, LK);
	Bs[buf][ty][(tx << 1) + 1].w = lget(B, (tx << 3) + 7, ty, LK);
	A += STEP; B += STEP;
	__syncthreads();

	//compute area-------------------------------------------------------
	float4 c0 = make_float4(0, 0, 0, 0);
	float4 c1 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (1 << LK >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float a = As[buf][ik][ty];
			float4 b0 = Bs[buf][ik][(tx << 1)   ];
			float4 b1 = Bs[buf][ik][(tx << 1) + 1];
			simdMM4(c0, a, b0); simdMM4(c1, a, b1);
		}
		buf ^= 1;

		As[buf][tx][ty] = A[Ayx];
		Bs[buf][ty][(tx << 1)    ].x = lget(B, (tx << 3)    , ty, LK);
		Bs[buf][ty][(tx << 1)    ].y = lget(B, (tx << 3) + 1, ty, LK);
		Bs[buf][ty][(tx << 1)    ].z = lget(B, (tx << 3) + 2, ty, LK);
		Bs[buf][ty][(tx << 1)    ].w = lget(B, (tx << 3) + 3, ty, LK);
		Bs[buf][ty][(tx << 1) + 1].x = lget(B, (tx << 3) + 4, ty, LK);
		Bs[buf][ty][(tx << 1) + 1].y = lget(B, (tx << 3) + 5, ty, LK);
		Bs[buf][ty][(tx << 1) + 1].z = lget(B, (tx << 3) + 6, ty, LK);
		Bs[buf][ty][(tx << 1) + 1].w = lget(B, (tx << 3) + 7, ty, LK);
		A += STEP; B += STEP;
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float a = As[buf][ik][ty];
		float4 b0 = Bs[buf][ik][(tx << 1)    ];
		float4 b1 = Bs[buf][ik][(tx << 1) + 1];
		simdMM4(c0, a, b0); simdMM4(c1, a, b1);
	}

	tx <<= 3;
	*(float4*)(&get(C, ty, tx, SC)) = c0; *(float4*)(&get(C, ty, tx + 4, SC)) = c1;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*4), K % BLOCK_SIZE/2 == 0
#ifndef MATMUL_T2_KERNEL_4_4_K2POW
#define MATMUL_T2_KERNEL_4_4_K2POW

//LB = 4: Size = 1, Time = 2.436 msec, Performace = 881.561 GFlop/s
template<int LB, int STEP>
__global__ void kernel_t2_4_4_K2pow(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float*  __restrict__ C,
	int SC, int LK)
{
	bool buf = 0;
	__shared__ float2 As[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2 Bs[2][1 << LB >> 1][(2 << LB) + 2];

	int Y = (blockIdx.y << 2 << LB);//Y = blockIdx.y * 4 * BLOCK_SIZE
	int X = (blockIdx.x << 2 << LB);//X = blockIdx.x * 4 * BLOCK_SIZE
	A = A + (Y << LK);//A[Y, 0]
	B = B + (X << LK);//B[X, 0]
	C = &get(C, Y, X, SC);//C[Y, X]
	
	int tx = threadIdx.x, ty = threadIdx.y;
	const int Ax = (tx >> 1), Ay = (ty << 2) + ((tx & 1) << 1);
	const int By = (ty >> 1), Bx = (tx << 2) + ((ty & 1) << 1);
	const int As_x = Ax, As_y = ((ty << 1) + (tx & 1));
	const int Bs_y = By, Bs_x = ((tx << 1) + (ty & 1));
	
	As[buf][As_x][As_y].x = lget(A, Ay    , Ax, LK);//transposeA
	As[buf][As_x][As_y].y = lget(A, Ay + 1, Ax, LK);
	Bs[buf][Bs_y][Bs_x].x = lget(B, Bx    , By, LK);
	Bs[buf][Bs_y][Bs_x].y = lget(B, Bx + 1, By, LK);
	A += STEP; B += STEP;
	__syncthreads();

	//compute area-------------------------------------------------------
	float4 c0 = make_float4(0, 0, 0, 0);
	float4 c1 = make_float4(0, 0, 0, 0);
	float4 c2 = make_float4(0, 0, 0, 0);
	float4 c3 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (2 << LK >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a = *(float4*)(&As[buf][ik][ty << 1]);
			float4 b = *(float4*)(&Bs[buf][ik][tx << 1]);

			simdMM4(c0, a.x, b);
			simdMM4(c1, a.y, b);
			simdMM4(c2, a.z, b);
			simdMM4(c3, a.w, b);
		}
		buf ^= 1;

		As[buf][As_x][As_y].x = lget(A, Ay    , Ax, LK);//transposeA
		As[buf][As_x][As_y].y = lget(A, Ay + 1, Ax, LK);
		Bs[buf][Bs_y][Bs_x].x = lget(B, Bx    , By, LK);
		Bs[buf][Bs_y][Bs_x].y = lget(B, Bx + 1, By, LK);
		A += STEP; B += STEP;
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a = *(float4*)(&As[buf][ik][ty << 1]);
		float4 b = *(float4*)(&Bs[buf][ik][tx << 1]);

		simdMM4(c0, a.x, b);
		simdMM4(c1, a.y, b);
		simdMM4(c2, a.z, b);
		simdMM4(c3, a.w, b);
	}

	ty <<= 2; tx <<= 2;
	*(float4*)(&get(C, ty    , tx, SC)) = c0;
	*(float4*)(&get(C, ty + 1, tx, SC)) = c1;
	*(float4*)(&get(C, ty + 2, tx, SC)) = c2;
	*(float4*)(&get(C, ty + 3, tx, SC)) = c3;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*2), K % BLOCK_SIZE/2 == 0
#ifndef MATMUL_T2_KERNEL_4_2_K2POW
#define MATMUL_T2_KERNEL_4_2_K2POW

//LB = 4: Size = 1, Time = 2.798 msec, Performace = 767.507 GFlop/s
template<int LB, int STEP>
__global__ void kernel_t2_4_2_K2pow(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float*  __restrict__ C,
	int SC, int LK)
{
	bool buf = 0;
	__shared__ float2 As[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float  Bs[2][1 << LB >> 1][(2 << LB) + 2];

	int Y = (blockIdx.y << 2 << LB);//Y = blockIdx.y * 4 * BLOCK_SIZE
	int X = (blockIdx.x << 1 << LB);//X = blockIdx.x * 2 * BLOCK_SIZE
	A = A + (Y << LK);//A[Y, 0]
	B = B + (X << LK);//B[X, 0]
	C = &get(C, Y, X, SC);//C[Y, X]

	int tx = threadIdx.x, ty = threadIdx.y;
	const int Ax = (tx >> 1), Ay = (ty << 2) + ((tx & 1) << 1);
	const int By = (ty >> 1), Bx = (tx << 1) + (ty & 1);
	const int As_x = Ax, As_y = ((ty << 1) + (tx & 1));
	const int Bs_y = By, Bs_x = ((tx << 1) + (ty & 1));
	const int Bxy = (Bx << LK) + By;//[Bx, By]

	As[buf][As_x][As_y].x = lget(A, Ay    , Ax, LK);//transposeA
	As[buf][As_x][As_y].y = lget(A, Ay + 1, Ax, LK);
	Bs[buf][Bs_y][Bs_x] = B[Bxy];
	A += STEP; B += STEP;
	__syncthreads();

	//compute area-------------------------------------------------------
	float2 c0 = make_float2(0, 0);
	float2 c1 = make_float2(0, 0);
	float2 c2 = make_float2(0, 0);
	float2 c3 = make_float2(0, 0);
	for (int ok = 1, OK = (2 << LK >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a = *(float4*)(&As[buf][ik][ty << 1]);
			float2 b = *(float2*)(&Bs[buf][ik][tx << 1]);

			simdMM2(c0, a.x, b);
			simdMM2(c1, a.y, b);
			simdMM2(c2, a.z, b);
			simdMM2(c3, a.w, b);
		}
		buf ^= 1;

		As[buf][As_x][As_y].x = lget(A, Ay    , Ax, LK);//transposeA
		As[buf][As_x][As_y].y = lget(A, Ay + 1, Ax, LK);
		Bs[buf][Bs_y][Bs_x] = B[Bxy];
		A += STEP; B += STEP;
		__syncthreads();
	}
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a = *(float4*)(&As[buf][ik][ty << 1]);
		float2 b = *(float2*)(&Bs[buf][ik][tx << 1]);

		simdMM2(c0, a.x, b);
		simdMM2(c1, a.y, b);
		simdMM2(c2, a.z, b);
		simdMM2(c3, a.w, b);
	}

	ty <<= 2; tx <<= 1;
	*(float2*)(&get(C, ty    , tx, SC)) = c0;
	*(float2*)(&get(C, ty + 1, tx, SC)) = c1;
	*(float2*)(&get(C, ty + 2, tx, SC)) = c2;
	*(float2*)(&get(C, ty + 3, tx, SC)) = c3;
}

#endif


//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*4), K % BLOCK_SIZE/2 == 0
#ifndef MATMUL_T2_KERNEL_2_4_K2POW
#define MATMUL_T2_KERNEL_2_4_K2POW

//LB = 4: Size = 1, Time = 3.398 msec, Performace = 631.984 GFlop/s
template<int LB, int STEP>
__global__ void kernel_t2_2_4_K2pow(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float*  __restrict__ C,
	int SC, int LK)
{
	bool buf = 0;
	__shared__ float  As[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2 Bs[2][1 << LB >> 1][(2 << LB) + 2];

	int Y = (blockIdx.y << 1 << LB);//Y = blockIdx.y * 2 * BLOCK_SIZE
	int X = (blockIdx.x << 2 << LB);//X = blockIdx.x * 4 * BLOCK_SIZE
	A = A + (Y << LK);//A[Y, 0]
	B = B + (X << LK);//B[X, 0]
	C = &get(C, Y, X, SC);//C[Y, X]

	int tx = threadIdx.x, ty = threadIdx.y;
	const int Ax = tx >> 1, Ay = (ty << 1) + (tx & 1);
	const int By = ty >> 1, Bx = (tx << 2) + ((ty & 1) << 1);
	const int As_x = Ax, As_y = ((ty << 1) + (tx & 1));
	const int Bs_y = By, Bs_x = ((tx << 1) + (ty & 1));
	const int Ayx = (Ay << LK) + Ax;//[Ay, Ax]

	As[buf][As_x][As_y] = A[Ayx];//transpose A
	Bs[buf][Bs_y][Bs_x].x = lget(B, Bx    , By, LK);
	Bs[buf][Bs_y][Bs_x].y = lget(B, Bx + 1, By, LK);
	A += STEP; B += STEP;
	__syncthreads();

	//compute area-------------------------------------------------------
	float4 c0 = make_float4(0, 0, 0, 0);
	float4 c1 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (2 << LK >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float2 a = *(float2*)(&As[buf][ik][ty << 1]);
			float4 b = *(float4*)(&Bs[buf][ik][tx << 1]);
			simdMM4(c0, a.x, b);
			simdMM4(c1, a.y, b);
		}
		buf ^= 1;

		As[buf][As_x][As_y] = A[Ayx];//transpose A
		Bs[buf][Bs_y][Bs_x].x = lget(B, Bx    , By, LK);
		Bs[buf][Bs_y][Bs_x].y = lget(B, Bx + 1, By, LK);
		A += STEP; B += STEP;
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float2 a = *(float2*)(&As[buf][ik][ty << 1]);
		float4 b = *(float4*)(&Bs[buf][ik][tx << 1]);
		simdMM4(c0, a.x, b);
		simdMM4(c1, a.y, b);
	}

	ty <<= 1; tx <<= 2;
	*(float4*)(&get(C, ty    , tx, SC)) = c0;
	*(float4*)(&get(C, ty + 1, tx, SC)) = c1;
}

#endif


//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*2), K % BLOCK_SIZE/2 == 0
#ifndef MATMUL_T2_KERNEL_2_2_K2POW
#define MATMUL_T2_KERNEL_2_2_K2POW

//LB = 4: Size = 1, Time = 3.61 msec, Performace = 594.871 GFlop/s
template<int LB, int STEP>
__global__ void kernel_t2_2_2_K2pow(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float*  __restrict__ C,
	int SC, int LK)
{
	bool buf = 0;
	__shared__ float2 As[2][1 << LB][(1 << LB) + 2];
	__shared__ float2 Bs[2][1 << LB][(1 << LB) + 2];

	int Y = (blockIdx.y << 1 << LB);//Y = blockIdx.y * 2 * BLOCK_SIZE
	int X = (blockIdx.x << 1 << LB);//X = blockIdx.x * 2 * BLOCK_SIZE
	A = A + (Y << LK);//A[Y, 0]
	B = B + (X << LK);//B[X, 0]
	C = &get(C, Y, X, SC);//C[Y, X]

	int tx = threadIdx.x, ty = threadIdx.y;

	As[buf][tx][ty].x = lget(A, (ty << 1)    , tx, LK);//transpose A
	As[buf][tx][ty].y = lget(A, (ty << 1) + 1, tx, LK);//[Ay + 1, Ax]
	Bs[buf][ty][tx].x = lget(B, (tx << 1)    , ty, LK);
	Bs[buf][ty][tx].y = lget(B, (tx << 1) + 1, ty, LK);
	A += STEP; B += STEP;
	__syncthreads();

	//compute area-------------------------------------------------------
	float2 c0 = make_float2(0, 0);
	float2 c1 = make_float2(0, 0);
	for (int ok = 1, OK = (1 << LK >> LB); ok < OK; ok++)
	{
#pragma once
		for (int ik = 0; ik < STEP; ik++) {
			float2 a = As[buf][ik][ty]; 
			float2 b = Bs[buf][ik][tx];
			simdMM2(c0, a.x, b);
			simdMM2(c1, a.y, b);
		}
		buf ^= 1;

		As[buf][tx][ty].x = lget(A, (ty << 1)    , tx, LK);//transpose A
		As[buf][tx][ty].y = lget(A, (ty << 1) + 1, tx, LK);//[Ay + 1, Ax]
		Bs[buf][ty][tx].x = lget(B, (tx << 1)    , ty, LK);
		Bs[buf][ty][tx].y = lget(B, (tx << 1) + 1, ty, LK);
		A += STEP; B += STEP;
		__syncthreads();
	}
#pragma once
	for (int ik = 0; ik < STEP; ik++) {
		float2 a = As[buf][ik][ty];
		float2 b = Bs[buf][ik][tx];
		simdMM2(c0, a.x, b);
		simdMM2(c1, a.y, b);
	}

	ty <<= 1; tx <<= 1;
	*(float2*)(&get(C, ty    , tx, SC)) = c0;
	*(float2*)(&get(C, ty + 1, tx, SC)) = c1;
}

#endif


//(Y: N, X: M), N*M <= 1024
#ifndef MATMUL_T2_KERNEL_NAIVE_K2POW
#define MATMUL_T2_KERNEL_NAIVE_K2POW

//(correct)
__global__ void kernel_t2_naive_K2pow(
	const float*  __restrict__ A,
	const float*  __restrict__ B,
	float*  __restrict__ C,
	int SC, int LK)
{
	int y = threadIdx.y + blockIdx.y*blockDim.y;
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	float c = 0;
	for (int k = 0, K = (1 << LK); k < K; k++)
		c += lget(A, y, k, LK) * lget(B, x, k, LK);
	get(C, y, x, SC) = c;
}

#endif

#endif