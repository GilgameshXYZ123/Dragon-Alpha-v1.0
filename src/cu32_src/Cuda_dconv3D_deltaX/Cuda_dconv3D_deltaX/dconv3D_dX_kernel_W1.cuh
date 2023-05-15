#pragma once

#ifndef DECONV3D_DW_KERNEL_W1_H
#define DECONV3D_DW_KERNEL_W1_H

//Pre test:
//for[OH, OW] = 64, [FH, FW] = 1, N = 8, [IC, OC] = 128, 64
//k88s1<4>: Size = 0.25, Time = 0.82  msec, Performace = 654.721 GFlop/s
//k88s1<3>: Size = 0.25, Time = 0.84  msec, Performace = 639.132 GFlop/s
//k44s1<4>: Size = 0.25, Time = 1.006 msec, Performace = 533.669 GFlop/s
//k44s1<3>: Size = 0.25, Time = 1.236 msec, Performace = 434.362 GFlop/s
//
//Unsparse Matrix Method:
//(1) FH ==1, FW == 1
//(2) GN = IC;           GN >= 4, GN%4 == 0
//(3) GM = N  * IH * IW; GM >= 4, GM%4 == 0
//(4) GK = OC;           GK >= 4, GK%4 == 0
//(5) sh = 1, sw = 1
//(6) ph = 0, pw = 0
#ifndef DECONV3D_KERNEL_DX_W1_CALL
#define DECONV3D_KERNEL_DX_W1_CALL

//LB = log2(BLOCK_SIZE)

//======[Common]=================================================
//LB = 4, STEP = 8, BLOCK_SIZE = 16, log2(128, 128) = (7, 7)
#define k88W1_LB4(stream, ic_index, j_index, deltaY, W, deltaX, IC, OC, GN, GM) \
	kernel_8_8_W1_LB4<4, 8>\
		<<< dim3(GN>>7, GM>>7), dim3(16, 16), 0, stream >>>\
			(deltaY, W, deltaX, IC,OC, ic_index,j_index)

//LB = 3, STEP = 4, BLOCK_SIZE = 8, log2(64, 64) = (6, 6)
#define k88W1_LB3(stream, ic_index, j_index, deltaY, W, deltaX, IC, OC, GN, GM) \
	kernel_8_8_W1_LB3<3, 4>\
		<<< dim3(GN>>6, GM>>6), dim3(8, 8), 0, stream >>>\
			(deltaY, W, deltaX, IC,OC, ic_index,j_index)

//---------------------------------------------------------------
#define k84W1(stream, LB, ic_index, j_index, deltaY, W, deltaX, IC, OC, GN, GM) \
	kernel_8_4_W1<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY, W, deltaX, IC,OC, ic_index,j_index)

#define k48W1(stream, LB, ic_index, j_index, deltaY, W, deltaX, IC, OC, GN, GM) \
	kernel_4_8_W1<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>2, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY, W, deltaX, IC,OC, ic_index,j_index)

#define k44W1(stream, LB, ic_index, j_index, deltaY, W, deltaX, IC, OC, GN, GM) \
	kernel_4_4_W1<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>2, GM>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY, W, deltaX, IC,OC, ic_index,j_index)

#define k82W1(stream, LB, ic_index, j_index, deltaY, W, deltaX, IC, OC, GN, GM) \
	kernel_8_2_W1<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>1, GN>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY, W, deltaX, IC,OC, ic_index,j_index)

#define k28W1(stream, LB, ic_index, j_index, deltaY, W, deltaX, IC, OC, GN, GM) \
	kernel_2_8_W1<LB, (1<<LB>>1)>\
		<<< dim3(GN>>LB>>1, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY, W, deltaX, IC,OC, ic_index,j_index)

#define k42W1(stream, LB, ic_index, j_index, deltaY, W, deltaX, IC, OC, GN, GM) \
	kernel_4_2_W1<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>1, GN>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY, W, deltaX, IC,OC, ic_index,j_index)

#define k24W1(stream, LB, ic_index, j_index, deltaY, W, deltaX, IC, OC, GN, GM) \
	kernel_2_4_W1<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>2, GN>>LB>>1), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY, W, deltaX, IC,OC, ic_index,j_index)

//======[Small]=================================================
#define k22W1(stream, LB, ic_index, j_index, deltaY, W, deltaX, IC, OC, GN, GM) \
	kernel_2_2_W1<LB, (1<<LB)>\
		<<< dim3(GM>>LB>>1, GN>>LB>>1), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY, W, deltaX, IC, OC, ic_index,j_index)

#define k12W1(stream, LB, ic_index, j_index, deltaY, W, deltaX, IC, OC, GN, GM) \
	kernel_1_2_W1<LB, (1<<LB)>\
		<<< dim3(GM>>LB>>1, GN>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY, W, deltaX, IC,OC, ic_index,j_index)

#define k21W1(stream, LB, ic_index, j_index, deltaY, W, deltaX, IC, OC, GN, GM) \
	kernel_2_1_W1<LB, (1<<LB)>\
		<<< dim3(GM>>LB, GN>>LB>>1), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY, W, deltaX, IC,OC, ic_index,j_index)

#define k11W1(stream, LB, ic_index, j_index, deltaY, W, deltaX, IC, OC, GN, GM) \
	kernel_1_1_W1<LB, (1<<LB)>\
		<<< dim3(GM>>LB, GN>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY, W, deltaX, IC,OC, ic_index,j_index)

#endif


//======[Common]==============================================
//(Y: BLOKC_SIZE*8, X: BLOCK_SIZE*8), GK % 8 == 0, BLOCK_SIZE = 16
//LB = 4, GK % 8 == 0
#ifndef DECONV3D_DX_KERNEL_8_8_W1_LB4
#define DECONV3D_DX_KERNEL_8_8_W1_LB4

//GK = OC = 256: Size = 1, Time = 1.382 msec, Performace = 1553.9  GFlop/s
//GK = OC = 512: Size = 1, Time = 1.322 msec, Performace = 1624.42 GFlop/s
template<int LB, int STEP>
__global__ void kernel_8_8_W1_LB4(
	const float* __restrict__ deltaY,
	const float* __restrict__ W,
	float* __restrict__ deltaX,
	int IC, int OC,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	W += ic0 + ((ty >= STEP) << 2);//W[0, 0, 0, tic0]

	//prepare for GM = N * IH * IW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	const int tj0 = (j0 + ((tx >= STEP) << 2)) * OC;
	const int tj1 = tj0 + OC, tj2 = tj1 + OC, tj3 = tj2 + OC;

	//load 4 elements from W[OC, FH, FW, IC]: Wr[IC, FH(r), FW(r), OC]
	int W_oc = ty - ((ty >= STEP) << LB >> 1);
	Ws[buf][ty][tx] = *(float4*)(W + W_oc * IC);

	//load 4 elements from deltaY[N, OH, OW, OC]: deltaYp[N, OH=OHp, OW=OWp, OC]
	float4 x; int Y_oc = tx - ((tx >= STEP) << LB >> 1);
	x.x = deltaY[tj0 + Y_oc];
	x.y = deltaY[tj1 + Y_oc];
	x.z = deltaY[tj2 + Y_oc];
	x.w = deltaY[tj3 + Y_oc];
	Ys[buf][tx][ty] = x;
	__syncthreads();

	//compute area----------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4 v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4 v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4 v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4 v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (OC << 1 >> LB); ok < OK; ok++)//GK = OC
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];
			float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];

			//transposed compute core: (W * dY)^T
			simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
			simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
			simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
			simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
			simdMM4(v8, y1.x, w0); simdMM4(v9, y1.x, w1);
			simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
			simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
			simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
		}
		buf ^= 1;

		//load 4 elements from W[OC, FH, FW, IC]: Wr[IC, FH(r), FW(r), OC]
		int W_oc = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		Ws[buf][ty][tx] = *(float4*)(W + W_oc * IC);

		//load 4 elements from deltaY[N, OH, OW, OC]: deltaYp[N, OH=OHp, OW=OWp, OC]
		float4 x; int Y_oc = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		x.x = deltaY[tj0 + Y_oc];
		x.y = deltaY[tj1 + Y_oc];
		x.z = deltaY[tj2 + Y_oc];
		x.w = deltaY[tj3 + Y_oc];
		Ys[buf][tx][ty] = x;
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];
		float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];

		//transposed compute core: (W * dY)^T
		simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
		simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
		simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
		simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
		simdMM4(v8, y1.x, w0); simdMM4(v9, y1.x, w1);
		simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
		simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
		simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
	}

	j0 = j0 * IC + ic0;//j0 = ((n * OH + oh) * OW + ow) * IC + ic
	const int j1 = j0 + IC, j2 = j1 + IC, j3 = j2 + IC;
	const int j4 = j3 + IC, j5 = j4 + IC, j6 = j5 + IC, j7 = j6 + IC;

	*(float4*)(deltaX + j0) = v0;  *(float4*)(deltaX + j0 + 4) = v1;
	*(float4*)(deltaX + j1) = v2;  *(float4*)(deltaX + j1 + 4) = v3;
	*(float4*)(deltaX + j2) = v4;  *(float4*)(deltaX + j2 + 4) = v5;
	*(float4*)(deltaX + j3) = v6;  *(float4*)(deltaX + j3 + 4) = v7;
	*(float4*)(deltaX + j4) = v8;  *(float4*)(deltaX + j4 + 4) = v9;
	*(float4*)(deltaX + j5) = v10; *(float4*)(deltaX + j5 + 4) = v11;
	*(float4*)(deltaX + j6) = v12; *(float4*)(deltaX + j6 + 4) = v13;
	*(float4*)(deltaX + j7) = v14; *(float4*)(deltaX + j7 + 4) = v15;
}

#endif


//(Y: BLOKC_SIZE*8, X: BLOCK_SIZE*8), GK % 4 == 0, BLOCK_SIZE = 8
//LB = 3, GK % 4 == 0
#ifndef DECONV3D_DX_KERNEL_8_8_W1_LB3
#define DECONV3D_DX_KERNEL_8_8_W1_LB3

//GK = OC = 256: Size = 1, Time = 1.78 msec, Performace = 1206.45 GFlop/s
//GK = OC = 512: Size = 1, Time = 1.69 msec, Performace = 1270.7  GFlop/s
template<int LB, int STEP>
__global__ void kernel_8_8_W1_LB3(
	const float* __restrict__ deltaY,
	const float* __restrict__ W, 
		  float* __restrict__ deltaX,
	int IC, int OC,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	W += ic0 + ((ty >= STEP) << 2);//W[0, 0, 0, tic0]

	//prepare for GM = N * IH * IW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	const int tj0 = (j0 + ((tx >= STEP) << 2)) * OC;
	const int tj1 = tj0 + OC, tj2 = tj1 + OC, tj3 = tj2 + OC;

	//load 4 elements from deltaY[N, OH, OW, OC]: deltaYp[N, OH=OHp, OW=OWp, OC]
	float4 x; int Y_oc = tx - ((tx >= STEP) << LB >> 1);
	x.x = deltaY[tj0 + Y_oc];
	x.y = deltaY[tj1 + Y_oc];
	x.z = deltaY[tj2 + Y_oc];
	x.w = deltaY[tj3 + Y_oc];
	Ys[buf][tx][ty] = x;

	//load 4 elements from W[OC, FH, FW, IC]: Wr[IC, FH(r), FW(r), OC]
	int W_oc = ty - ((ty >= STEP) << LB >> 1);
	Ws[buf][ty][tx] = *(float4*)(W + W_oc * IC);
	__syncthreads();

	//compute area----------------------------------------------------
	float4 v0  = make_float4(0, 0, 0, 0),  v1 = make_float4(0, 0, 0, 0);
	float4 v2  = make_float4(0, 0, 0, 0),  v3 = make_float4(0, 0, 0, 0);
	float4 v4  = make_float4(0, 0, 0, 0),  v5 = make_float4(0, 0, 0, 0);
	float4 v6  = make_float4(0, 0, 0, 0),  v7 = make_float4(0, 0, 0, 0);
	float4 v8  = make_float4(0, 0, 0, 0),  v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (OC << 1 >> LB); ok < OK; ok++)//GK = OC
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];
			float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];
			
			//transposed compute core: (W * dY)^T
			simdMM4( v0, y0.x, w0); simdMM4( v1, y0.x, w1);
			simdMM4( v2, y0.y, w0); simdMM4( v3, y0.y, w1);
			simdMM4( v4, y0.z, w0); simdMM4( v5, y0.z, w1);
			simdMM4( v6, y0.w, w0); simdMM4( v7, y0.w, w1);
			simdMM4( v8, y1.x, w0); simdMM4( v9, y1.x, w1);
			simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
			simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
			simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
		}
		buf ^= 1;

		//load 4 elements from deltaY[N, OH, OW, OC]: deltaYp[N, OH=OHp, OW=OWp, OC]
		float4 x; int Y_oc = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		x.x = deltaY[tj0 + Y_oc];
		x.y = deltaY[tj1 + Y_oc];
		x.z = deltaY[tj2 + Y_oc];
		x.w = deltaY[tj3 + Y_oc];
		Ys[buf][tx][ty] = x;

		//load 4 elements from W[OC, FH, FW, IC]: Wr[IC, FH(r), FW(r), OC]
		int W_oc = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		Ws[buf][ty][tx] = *(float4*)(W + W_oc * IC);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];
		float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];

		//transposed compute core: (W * dY)^T
		simdMM4( v0, y0.x, w0); simdMM4( v1, y0.x, w1);
		simdMM4( v2, y0.y, w0); simdMM4( v3, y0.y, w1);
		simdMM4( v4, y0.z, w0); simdMM4( v5, y0.z, w1);
		simdMM4( v6, y0.w, w0); simdMM4( v7, y0.w, w1);
		simdMM4( v8, y1.x, w0); simdMM4( v9, y1.x, w1);
		simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
		simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
		simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
	}

	j0 = j0 * IC + ic0;//j0 = ((n * OH + oh) * OW + ow) * IC + ic
	const int j1 = j0 + IC, j2 = j1 + IC, j3 = j2 + IC;
	const int j4 = j3 + IC, j5 = j4 + IC, j6 = j5 + IC, j7 = j6 + IC;

	*(float4*)(deltaX + j0) = v0;  *(float4*)(deltaX + j0 + 4) = v1;
	*(float4*)(deltaX + j1) = v2;  *(float4*)(deltaX + j1 + 4) = v3;
	*(float4*)(deltaX + j2) = v4;  *(float4*)(deltaX + j2 + 4) = v5;
	*(float4*)(deltaX + j3) = v6;  *(float4*)(deltaX + j3 + 4) = v7;
	*(float4*)(deltaX + j4) = v8;  *(float4*)(deltaX + j4 + 4) = v9;
	*(float4*)(deltaX + j5) = v10; *(float4*)(deltaX + j5 + 4) = v11;
	*(float4*)(deltaX + j6) = v12; *(float4*)(deltaX + j6 + 4) = v13;
	*(float4*)(deltaX + j7) = v14; *(float4*)(deltaX + j7 + 4) = v15;
}

#endif

//------------------------------------------------------------
//(Y: BLOKC_SIZE*8, X: BLOCK_SIZE*4), GK % (BLOCK_SIZE/2) == 0
//LB = 4, GK % 8 == 0
#ifndef DECONV3D_DX_KERNEL_8_4_W1
#define DECONV3D_DX_KERNEL_8_4_W1

//LB = 4: Size = 1, Time = 1.718 msec, Performace = 1249.99 GFlop/s
//LB = 3: Size = 1, Time = 1.84  msec, Performace = 1167.11 GFlop/s
template<int LB, int STEP>
__global__ void kernel_8_4_W1(
	const float* __restrict__ deltaY,
	const float* __restrict__ W,
	float* __restrict__ deltaX,
	int IC, int OC,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];//follow k88
	__shared__ float2 Ys[2][1 << LB >> 1][(2 << LB) + 2];//follow k44

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 3) + ic_index;
	W += ic0 + ((ty >= STEP) << 2);//W[0, 0, 0, tic0]

	//prepare for GM = N * IH * IW
	int j0 = (((blockIdx.y << LB) + ty) << 2) + j_index;
	const int tj0 = (((tx & 1) << 1) + j0) * OC, tj1 = tj0 + OC;
	
	//load 2 elements from deltaY[N, OH, OW, OC]: deltaYp[N, OH=OHp, OW=OWp, OC]
	float2 x; int Y_oc = tx >> 1;
	x.x = deltaY[tj0 + Y_oc];
	x.y = deltaY[tj1 + Y_oc];
	const int Ys_x = (tx >> 1), Ys_y = (ty << 1) + (tx & 1);
	Ys[buf][Ys_x][Ys_y] = x;

	//load 4 elements from W[OC, FH, FW, IC]: Wr[IC, FH(r), FW(r), OC]
	int W_oc = ty - ((ty >= STEP) << LB >> 1);
	Ws[buf][ty][tx] = *(float4*)(W + W_oc * IC);
	__syncthreads();

	//compute area----------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4 v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4 v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4 v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (OC << 1 >> LB); ok < OK; ok++)//GK = OC
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 y0 = *(float4*)(&Ys[buf][ik][ty << 1]);
			float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];

			//transposed compute core: (W * dY)^T
			simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
			simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
			simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
			simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
		}
		buf ^= 1;

		//load 2 elements from deltaY[N, OH, OW, OC]: deltaYp[N, OH=OHp, OW=OWp, OC]
		float2 x; int Y_oc = ((ok << LB) + tx) >> 1;
		x.x = deltaY[tj0 + Y_oc];
		x.y = deltaY[tj1 + Y_oc];
		Ys[buf][Ys_x][Ys_y] = x;

		//load 4 elements from W[OC, FH, FW, IC]: Wr[IC, FH(r), FW(r), OC]
		int W_oc = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		Ws[buf][ty][tx] = *(float4*)(W + W_oc * IC);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 y0 = *(float4*)(&Ys[buf][ik][ty << 1]);
		float4 w0 = Ws[buf][ik][tx], w1 = Ws[buf][ik + STEP][tx];

		//transposed compute core: (W * dY)^T
		simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
		simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
		simdMM4(v4, y0.z, w0); simdMM4(v5, y0.z, w1);
		simdMM4(v6, y0.w, w0); simdMM4(v7, y0.w, w1);
	}

	j0 = j0 * IC + ic0;//j0 = ((n * OH + oh) * OW + ow) * IC + ic
	const int j1 = j0 + IC, j2 = j1 + IC, j3 = j2 + IC;

	*(float4*)(deltaX + j0) = v0;  *(float4*)(deltaX + j0 + 4) = v1;
	*(float4*)(deltaX + j1) = v2;  *(float4*)(deltaX + j1 + 4) = v3;
	*(float4*)(deltaX + j2) = v4;  *(float4*)(deltaX + j2 + 4) = v5;
	*(float4*)(deltaX + j3) = v6;  *(float4*)(deltaX + j3 + 4) = v7;
}

#endif


//(Y: BLOKC_SIZE*4, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0
//LB = 4, GK % 8 == 0
#ifndef DECONV3D_DX_KERNEL_4_8_W1
#define DECONV3D_DX_KERNEL_4_8_W1

//GK = OC = 256:
//LB = 4: Size = 1, Time = 1.492 msec, Performace = 1439.33 GFlop/s
//LB = 3: Size = 1, Time = 1.722 msec, Performace = 1247.09 GFlop/s
//GK = OC = 512:
//LB = 4: Size = 1, Time = 1.494 msec, Performace = 1437.41 GFlop/s
//LB = 3: Size = 1, Time = 1.766 msec, Performace = 1216.02 GFlop/s
template<int LB, int STEP>
__global__ void kernel_4_8_W1(
	const float* __restrict__ deltaY,
	const float* __restrict__ W,
	float* __restrict__ deltaX,
	int IC, int OC,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB >> 1][(2 << LB) + 2];//follow k44
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];//follow k88

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 2) + ic_index;
	W += ((ty & 1) << 1) + ic0;//W[0, 0, 0, tic0]

	//prepare for GM = N * IH * IW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	const int tj0 = (j0 + ((tx >= STEP) << 2)) * OC;
	const int tj1 = tj0 + OC, tj2 = tj1 + OC, tj3 = tj2 + OC;

	//load 4 elements from deltaY[N, OH, OW, OC]: deltaYp[N, OH=OHp, OW=OWp, OC]
	float4 x; int Y_oc = tx - ((tx >= STEP) << LB >> 1);
	x.x = deltaY[tj0 + Y_oc];
	x.y = deltaY[tj1 + Y_oc];
	x.z = deltaY[tj2 + Y_oc];
	x.w = deltaY[tj3 + Y_oc];
	Ys[buf][tx][ty] = x;

	//load 2 elements from W[OC, FH, FW, IC]: Wr[IC, FH(r), FW(r), OC]
	int W_oc = (ty >> 1);
	const int Ws_y = (ty >> 1), Ws_x = (tx << 1) + (ty & 1);
	Ws[buf][Ws_y][Ws_x] = *(float2*)(W + W_oc * IC);
	__syncthreads();

	//compute area----------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0);
	float4 v1 = make_float4(0, 0, 0, 0);
	float4 v2 = make_float4(0, 0, 0, 0);
	float4 v3 = make_float4(0, 0, 0, 0);
	float4 v4 = make_float4(0, 0, 0, 0);
	float4 v5 = make_float4(0, 0, 0, 0);
	float4 v6 = make_float4(0, 0, 0, 0);
	float4 v7 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (OC << 1 >> LB); ok < OK; ok++)//GK = OC
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];
			float4 w0 = *(float4*)(&Ws[buf][ik][tx << 1]);

			//transposed compute core: (W * dY)^T
			simdMM4(v0, y0.x, w0);
			simdMM4(v1, y0.y, w0);
			simdMM4(v2, y0.z, w0);
			simdMM4(v3, y0.w, w0);
			simdMM4(v4, y1.x, w0);
			simdMM4(v5, y1.y, w0);
			simdMM4(v6, y1.z, w0);
			simdMM4(v7, y1.w, w0); 
		}
		buf ^= 1;

		//load 4 elements from deltaY[N, OH, OW, OC]: deltaYp[N, OH=OHp, OW=OWp, OC]
		float4 x; int Y_oc = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		x.x = deltaY[tj0 + Y_oc];
		x.y = deltaY[tj1 + Y_oc];
		x.z = deltaY[tj2 + Y_oc];
		x.w = deltaY[tj3 + Y_oc];
		Ys[buf][tx][ty] = x;

		//load 2 elements from W[OC, FH, FW, IC]: Wr[IC, FH(r), FW(r), OC]
		int W_oc = ((ok << LB) + ty) >> 1;
		Ws[buf][Ws_y][Ws_x] = *(float2*)(W + W_oc * IC);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];
		float4 w0 = *(float4*)(&Ws[buf][ik][tx << 1]);

		//transposed compute core: (W * dY)^T
		simdMM4(v0, y0.x, w0);
		simdMM4(v1, y0.y, w0);
		simdMM4(v2, y0.z, w0);
		simdMM4(v3, y0.w, w0);
		simdMM4(v4, y1.x, w0);
		simdMM4(v5, y1.y, w0);
		simdMM4(v6, y1.z, w0);
		simdMM4(v7, y1.w, w0);
	}

	j0 = j0 * IC + ic0;//j0 = ((n * OH + oh) * OW + ow) * IC + ic
	const int j1 = j0 + IC, j2 = j1 + IC, j3 = j2 + IC;
	const int j4 = j3 + IC, j5 = j4 + IC, j6 = j5 + IC, j7 = j6 + IC;

	*(float4*)(deltaX + j0) = v0; 
	*(float4*)(deltaX + j1) = v1;
	*(float4*)(deltaX + j2) = v2;
	*(float4*)(deltaX + j3) = v3; 
	*(float4*)(deltaX + j4) = v4;
	*(float4*)(deltaX + j5) = v5;
	*(float4*)(deltaX + j6) = v6;
	*(float4*)(deltaX + j7) = v7; 
}

#endif


//(Y: BLOKC_SIZE*4, X: BLOCK_SIZE*4), GK % (BLOCK_SIZE/2) == 0
//LB = 4, GK % 8 == 0
#ifndef DECONV3D_DX_KERNEL_4_4_W1
#define DECONV3D_DX_KERNEL_4_4_W1

//LB = 4: Size = 1, Time = 1.974 msec, Performace = 1087.88  GFlop/s
//LB = 3: Size = 1, Time = 2.21  msec, Performace =  971.712 GFlop/s
template<int LB, int STEP>
__global__ void kernel_4_4_W1(
	const float* __restrict__ deltaY,
	const float* __restrict__ W,
	float* __restrict__ deltaX,
	int IC, int OC,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2 Ys[2][1 << LB >> 1][(2 << LB) + 2];

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 2) + ic_index;
	W += ((ty & 1) << 1) + ic0;//W[0, 0, 0, tic0]

	//prepare for GM = N * IH * IW
	int j0 = (((blockIdx.y << LB) + ty) << 2) + j_index;
	const int tj0 = (((tx & 1) << 1) + j0) * OC, tj1 = tj0 + OC;

	//load 2 elements from deltaY[N, OH, OW, OC]: deltaYp[N, OH=OHp, OW=OWp, OC]
	float2 x; int Y_oc = tx >> 1;
	x.x = deltaY[tj0 + Y_oc];
	x.y = deltaY[tj1 + Y_oc];
	const int Ys_x = (tx >> 1), Ys_y = (ty << 1) + (tx & 1);
	Ys[buf][Ys_x][Ys_y] = x;

	//load 2 elements from W[OC, FH, FW, IC]: Wr[IC, FH(r), FW(r), OC]
	int W_oc = (ty >> 1);
	const int Ws_y = (ty >> 1), Ws_x = (tx << 1) + (ty & 1);
	Ws[buf][Ws_y][Ws_x] = *(float2*)(W + W_oc * IC);
	__syncthreads();

	//compute area----------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0);
	float4 v1 = make_float4(0, 0, 0, 0);
	float4 v2 = make_float4(0, 0, 0, 0);
	float4 v3 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (OC << 1 >> LB); ok < OK; ok++)//GK = OC
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) 
		{
			float4 w = *(float4*)(&Ws[buf][ik][tx << 1]);
			float4 y = *(float4*)(&Ys[buf][ik][ty << 1]);

			//transposed compute core: (W * dY)^T
			simdMM4(v0, y.x, w);
			simdMM4(v1, y.y, w);
			simdMM4(v2, y.z, w);
			simdMM4(v3, y.w, w);
		}
		buf ^= 1;

		//load 2 elements from deltaY[N, OH, OW, OC]: deltaYp[N, OH=OHp, OW=OWp, OC]
		float2 x; int Y_oc = ((ok << LB) + tx) >> 1;
		x.x = deltaY[tj0 + Y_oc];
		x.y = deltaY[tj1 + Y_oc];
		Ys[buf][Ys_x][Ys_y] = x;

		//load 2 elements from W[OC, FH, FW, IC]: Wr[IC, FH(r), FW(r), OC]
		int W_oc = ((ok << LB) + ty) >> 1;
		Ws[buf][Ws_y][Ws_x] = *(float2*)(W + W_oc * IC);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) 
	{
		float4 w = *(float4*)(&Ws[buf][ik][tx << 1]);
		float4 y = *(float4*)(&Ys[buf][ik][ty << 1]);

		//transposed compute core: (W * dY)^T
		simdMM4(v0, y.x, w);
		simdMM4(v1, y.y, w);
		simdMM4(v2, y.z, w);
		simdMM4(v3, y.w, w);
	}

	j0 = j0 * IC + ic0;//j0 = ((n * OH + oh) * OW + ow) * IC + ic
	const int j1 = j0 + IC, j2 = j1 + IC, j3 = j2 + IC;

	*(float4*)(deltaX + j0) = v0;
	*(float4*)(deltaX + j1) = v1;
	*(float4*)(deltaX + j2) = v2;
	*(float4*)(deltaX + j3) = v3;
}

#endif


//(Y: BLOKC_SIZE*8, X: BLOCK_SIZE*2), GK % (BLOCK_SIZE/2) == 0
//LB = 4, GK % 8 == 0
#ifndef DECONV3D_DX_KERNEL_8_2_W1
#define DECONV3D_DX_KERNEL_8_2_W1

//LB = 4: Size = 1, Time = 2.092 msec, Performace = 1026.52  GFlop/s
//LB = 3: Size = 1, Time = 2.534 msec, Performace =  847.468 GFlop/s
template<int LB, int STEP>
__global__ void kernel_8_2_W1(
	const float* __restrict__ deltaY,
	const float* __restrict__ W,
	float* __restrict__ deltaX,
	int IC, int OC,
	int ic_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];//follow k88
	__shared__ float  Ys[2][1 << LB >> 1][(2 << LB) + 2];//follow k44

	//prepare for GN = IC
	const int ic0 = (((blockIdx.y << LB) + ty) << 3) + ic_index;
	W += ic0 + ((tx >= STEP) << 2);//W[0, 0, 0, tic0]

	//prepare for GM = N * IH * IW
	int j0 = (((blockIdx.x << LB) + tx) << 1) + j_index;
	const int tj0 = ((ty & 1) + j0) * OC;

	//load 1 element from deltaY[N, OH, OW, OC]: deltaYp[N, OH=OHp, OW=OWp, OC]
	int Y_oc = ty >> 1;
	const int Ys_y = (ty >> 1), Ys_x = (tx << 1) + (ty & 1);
	Ys[buf][Ys_y][Ys_x] = deltaY[tj0 + Y_oc];

	//load 4 elements from W[OC, FH, FW, IC]: Wr[IC, FH(r), FW(r), OC]
	int W_oc = tx - ((tx >= STEP) << LB >> 1);
	Ws[buf][tx][ty] = *(float4*)(W + W_oc * IC);
	__syncthreads();

	//compute area----------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4 v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (OC << 1 >> LB); ok < OK; ok++)//GK = OC
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float2 y0 = *(float2*)(&Ys[buf][ik][tx << 1]);
			float4 w0 = Ws[buf][ik][ty], w1 = Ws[buf][ik + STEP][ty];

			//transposed compute core: (W * dY)^T
			simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
			simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
		}
		buf ^= 1;

		//load 1 element from deltaY[N, OH, OW, OC]: deltaYp[N, OH=OHp, OW=OWp, OC]
		int Y_oc = ((ok << LB) + ty) >> 1;
		Ys[buf][Ys_y][Ys_x] = deltaY[tj0 + Y_oc];

		//load 4 elements from W[OC, FH, FW, IC]: Wr[IC, FH(r), FW(r), OC]
		int W_oc = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		Ws[buf][tx][ty] = *(float4*)(W + W_oc * IC);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float2 y0 = *(float2*)(&Ys[buf][ik][tx << 1]);
		float4 w0 = Ws[buf][ik][ty], w1 = Ws[buf][ik + STEP][ty];

		//transposed compute core: (W * dY)^T
		simdMM4(v0, y0.x, w0); simdMM4(v1, y0.x, w1);
		simdMM4(v2, y0.y, w0); simdMM4(v3, y0.y, w1);
	}

	j0 = j0 * IC + ic0;//j0 = ((n * OH + oh) * OW + ow) * IC + ic
	const int j1 = j0 + IC;

	*(float4*)(deltaX + j0) = v0;  *(float4*)(deltaX + j0 + 4) = v1;
	*(float4*)(deltaX + j1) = v2;  *(float4*)(deltaX + j1 + 4) = v3;
}

#endif


//(Y: BLOKC_SIZE*2, X: BLOCK_SIZE*8), GK % (BLOCK_SIZE/2) == 0
//LB = 4, GK % 8 == 0
#ifndef DECONV3D_DX_KERNEL_2_8_W1
#define DECONV3D_DX_KERNEL_2_8_W1

//LB = 4: Size = 1, Time = 4.642 msec, Performace = 462.62 GFlop/s
//LB = 3: Size = 1, Time = 6.63  msec, Performace = 323.904 GFlop/s
template<int LB, int STEP>
__global__ void kernel_2_8_W1(
	const float* __restrict__ deltaY,
	const float* __restrict__ W,
	float* __restrict__ deltaX,
	int IC, int OC,
	int ic_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float  Ws[2][1 << LB >> 1][(2 << LB) + 2];//follow k44
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];//follow k88

	//prepare for GN = IC
	const int ic0 = (((blockIdx.x << LB) + tx) << 1) + ic_index;
	W += (ty & 1) + ic0;//W[0, 0, 0, tic0]

	//prepare for GM = N * IH * IW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	const int tj0 = (j0 + ((tx >= STEP) << 2)) * OC;
	const int tj1 = tj0 + OC, tj2 = tj1 + OC, tj3 = tj2 + OC;

	//load 4 elements from deltaY[N, OH, OW, OC]: deltaYp[N, OH=OHp, OW=OWp, OC]
	float4 x; int Y_oc = tx - ((tx >= STEP) << LB >> 1);
	x.x = deltaY[tj0 + Y_oc];
	x.y = deltaY[tj1 + Y_oc];
	x.z = deltaY[tj2 + Y_oc];
	x.w = deltaY[tj3 + Y_oc];
	Ys[buf][tx][ty] = x;

	//load 1 element from W[OC, FH, FW, IC]: Wr[IC, FH(r), FW(r), OC]
	int W_oc = (ty >> 1);
	const int Ws_y = (ty >> 1), Ws_x = (tx << 1) + (ty & 1);
	Ws[buf][Ws_y][Ws_x] = W[W_oc * IC];
	__syncthreads();

	//compute area----------------------------------------------------
	float2 v0 = make_float2(0, 0);
	float2 v1 = make_float2(0, 0);
	float2 v2 = make_float2(0, 0);
	float2 v3 = make_float2(0, 0);
	float2 v4 = make_float2(0, 0);
	float2 v5 = make_float2(0, 0);
	float2 v6 = make_float2(0, 0);
	float2 v7 = make_float2(0, 0);

	for (int ok = 1, OK = (OC << 1 >> LB); ok < OK; ok++)//GK = OC
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];
			float2 w0 = *(float2*)(&Ws[buf][ik][tx << 1]);

			//transposed compute core: (W * dY)^T
			simdMM2(v0, y0.x, w0);
			simdMM2(v1, y0.y, w0);
			simdMM2(v2, y0.z, w0);
			simdMM2(v3, y0.w, w0);
			simdMM2(v4, y1.x, w0);
			simdMM2(v5, y1.y, w0);
			simdMM2(v6, y1.z, w0);
			simdMM2(v7, y1.w, w0);
		}
		buf ^= 1;
		
		//load 4 elements from deltaY[N, OH, OW, OC]: deltaYp[N, OH=OHp, OW=OWp, OC]
		float4 x; int Y_oc = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		x.x = deltaY[tj0 + Y_oc];
		x.y = deltaY[tj1 + Y_oc];
		x.z = deltaY[tj2 + Y_oc];
		x.w = deltaY[tj3 + Y_oc];
		Ys[buf][tx][ty] = x;

		//load 1 element from W[OC, FH, FW, IC]: Wr[IC, FH(r), FW(r), OC]
		int W_oc = ((ok << LB) + ty) >> 1;
		Ws[buf][Ws_y][Ws_x] = W[W_oc * IC];
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 y0 = Ys[buf][ik][ty], y1 = Ys[buf][ik + STEP][ty];
		float2 w0 = *(float2*)(&Ws[buf][ik][tx << 1]);

		//transposed compute core: (W * dY)^T
		simdMM2(v0, y0.x, w0);
		simdMM2(v1, y0.y, w0);
		simdMM2(v2, y0.z, w0);
		simdMM2(v3, y0.w, w0);
		simdMM2(v4, y1.x, w0);
		simdMM2(v5, y1.y, w0);
		simdMM2(v6, y1.z, w0);
		simdMM2(v7, y1.w, w0);
	}

	j0 = j0 * IC + ic0;//j0 = ((n * OH + oh) * OW + ow) * IC + ic
	const int j1 = j0 + IC, j2 = j1 + IC, j3 = j2 + IC;
	const int j4 = j3 + IC, j5 = j4 + IC, j6 = j5 + IC, j7 = j6 + IC;

	*(float2*)(deltaX + j0) = v0;
	*(float2*)(deltaX + j1) = v1;
	*(float2*)(deltaX + j2) = v2;
	*(float2*)(deltaX + j3) = v3;
	*(float2*)(deltaX + j4) = v4;
	*(float2*)(deltaX + j5) = v5;
	*(float2*)(deltaX + j6) = v6;
	*(float2*)(deltaX + j7) = v7;
}

#endif


//(Y: BLOKC_SIZE*4, X: BLOCK_SIZE*2), GK % (BLOCK_SIZE/2) == 0
//LB = 4, GK % 8 == 0
#ifndef DECONV3D_DX_KERNEL_4_2_W1
#define DECONV3D_DX_KERNEL_4_2_W1

//LB = 4: Size = 1, Time = 2.614 msec, Performace = 821.532 GFlop/s
//LB = 3: Size = 1, Time = 4.886 msec, Performace = 439.518 GFlop/s
template<int LB, int STEP>
__global__ void kernel_4_2_W1(
	const float* __restrict__ deltaY,
	const float* __restrict__ W,
		  float* __restrict__ deltaX,
	int IC, int OC,
	int ic_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float  Ys[2][1 << LB >> 1][(2 << LB) + 2];

	//prepare for GN = IC
	const int ic0 = (((blockIdx.y << LB) + ty) << 2) + ic_index;
	W += ((tx & 1) << 1) + ic0;//W[0, 0, 0, tic0]

	//prepare for GM = N * IH * IW
	int j0 = (((blockIdx.x << LB) + tx) << 1) + j_index;
	const int tj0 = ((ty & 1) + j0) * OC;

	//load 2 element from W[OC, FH, FW, IC]: Wr[IC, FH(r), FW(r), OC]
	int W_oc = (tx >> 1);
	const int Ws_x = (tx >> 1), Ws_y = (ty << 1) + (tx & 1);
	Ws[buf][Ws_x][Ws_y] = *(float2*)(W + W_oc * IC);

	//load 1 element from deltaY[N, OH, OW, OC]: deltaYp[N, OH=OHp, OW=OWp, OC]
	int Y_oc = ty >> 1;
	const int Ys_y = (ty >> 1), Ys_x = (tx << 1) + (ty & 1);
	Ys[buf][Ys_y][Ys_x] = deltaY[tj0 + Y_oc];
	__syncthreads();

	//compute area----------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0);
	float4 v1 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (OC << 1 >> LB); ok < OK; ok++)//GK = OC
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float4 w = *(float4*)(&Ws[buf][ik][ty << 1]);
			float2 y = *(float2*)(&Ys[buf][ik][tx << 1]);
			simdMM4(v0, y.x, w);
			simdMM4(v1, y.y, w);
		}
		buf ^= 1;

		//load 2 elements from W[OC, FH, FW, IC]: Wr[IC, FH(r), FW(r), OC]
		int W_oc = ((ok << LB) + tx) >> 1;
		Ws[buf][Ws_x][Ws_y] = *(float2*)(W + W_oc * IC);

		//load 1 element from deltaY[N, OH, OW, OC]: deltaYp[N, OH=OHp, OW=OWp, OC]
		int Y_oc = ((ok << LB) + ty) >> 1;
		Ys[buf][Ys_y][Ys_x] = deltaY[tj0 + Y_oc];
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float4 w = *(float4*)(&Ws[buf][ik][ty << 1]);
		float2 y = *(float2*)(&Ys[buf][ik][tx << 1]);
		simdMM4(v0, y.x, w);
		simdMM4(v1, y.y, w);
	}

	j0 = j0 * IC + ic0;//j0 = ((n * OH + oh) * OW + ow) * IC + ic
	const int j1 = j0 + IC; 

	*(float4*)(deltaX + j0) = v0;
	*(float4*)(deltaX + j1) = v1;
}

#endif


//(Y: BLOKC_SIZE*2, X: BLOCK_SIZE*4), GK % (BLOCK_SIZE/2) == 0
//LB = 4, GK % 8 == 0
#ifndef DECONV3D_DX_KERNEL_2_4_W1
#define DECONV3D_DX_KERNEL_2_4_W1

//LB = 4: Size = 1, Time = 4.72 msec, Performace = 454.975 GFlop/s
//LB = 4: Size = 1, Time = 11.476 msec, Performace = 187.128 GFlop/s
template<int LB, int STEP>
__global__ void kernel_2_4_W1(
	const float* __restrict__ deltaY,
	const float* __restrict__ W,
	float* __restrict__ deltaX,
	int IC, int OC,
	int ic_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float  Ws[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2 Ys[2][1 << LB >> 1][(2 << LB) + 2];

	//prepare for GN = IC
	const int ic0 = (((blockIdx.y << LB) + ty) << 1) + ic_index;
	W += (tx & 1) + ic0;//W[0, 0, 0, tic0]

	//prepare for GM = N * IH * IW
	int j0 = (((blockIdx.x << LB) + tx) << 2) + j_index;
	const int tj0 = (((ty & 1) << 1) + j0) * OC, tj1 = tj0 + OC;
	
	//load 1 element from W[OC, FH, FW, IC]: Wr[IC, FH(r), FW(r), OC]
	int W_oc = (tx >> 1); //Ws: with the same ty
	const int Ws_x = (tx >> 1), Ws_y = (ty << 1) + (tx & 1);
	Ws[buf][Ws_x][Ws_y] = W[W_oc*IC];

	//load 2 elements from deltaY[N, OH, OW, OC]: deltaYp[N, OH=OHp, OW=OWp, OC]
	int Y_oc = ty >> 1;
	const int Ys_y = (ty >> 1), Ys_x = (tx << 1) + (ty & 1);
	Ys[buf][Ys_y][Ys_x].x = deltaY[tj0 + Y_oc];
	Ys[buf][Ys_y][Ys_x].y = deltaY[tj1 + Y_oc];
	__syncthreads();

	//compute area----------------------------------------------------
	float2 v0 = make_float2(0, 0);
	float2 v1 = make_float2(0, 0);
	float2 v2 = make_float2(0, 0);
	float2 v3 = make_float2(0, 0);
	for (int ok = 1, OK = (OC << 1 >> LB); ok < OK; ok++)//GK = OC
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) 
		{
			float2 w = *(float2*)(&Ws[buf][ik][ty << 1]);
			float4 y = *(float4*)(&Ys[buf][ik][tx << 1]);

			simdMM2(v0, y.x, w);
			simdMM2(v1, y.y, w);
			simdMM2(v2, y.z, w);
			simdMM2(v3, y.w, w);
		}
		buf ^= 1;

		//load 1 element from W[OC, FH, FW, IC]: Wr[IC, FH(r), FW(r), OC]
		int W_oc = ((ok << LB) + tx) >> 1;
		Ws[buf][Ws_x][Ws_y] = W[W_oc*IC];

		//load 2 elements from deltaY[N, OH, OW, OC]: deltaYp[N, OH=OHp, OW=OWp, OC]
		int Y_oc = ((ok << LB) + ty) >> 1;
		Ys[buf][Ys_y][Ys_x].x = deltaY[tj0 + Y_oc];
		Ys[buf][Ys_y][Ys_x].y = deltaY[tj1 + Y_oc];
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) 
	{
		float2 w = *(float2*)(&Ws[buf][ik][ty << 1]);
		float4 y = *(float4*)(&Ys[buf][ik][tx << 1]);

		simdMM2(v0, y.x, w);
		simdMM2(v1, y.y, w);
		simdMM2(v2, y.z, w);
		simdMM2(v3, y.w, w);
	}

	j0 = j0 * IC + ic0;//j0 = ((n * OH + oh) * OW + ow) * IC + ic
	const int j1 = j0 + IC, j2 = j1 + IC, j3 = j2 + IC;

	*(float2*)(deltaX + j0) = v0;
	*(float2*)(deltaX + j1) = v1;
	*(float2*)(deltaX + j2) = v2;
	*(float2*)(deltaX + j3) = v3;
}

#endif


//======[Small]==============================================
//(Y: BLOKC_SIZE*2, X: BLOCK_SIZE*2)
#ifndef DECONV3D_DX_KERNEL_2_2_W1
#define DECONV3D_DX_KERNEL_2_2_W1

//LB = 4: Size = 0.5, Time = 2.77 msec, Performace = 387.632 GFlop/s
//LB = 4: Size = 0.246094, Time = 1.684 msec, Performace = 313.826 GFlop/s
//LB = 3: Size = 0.246094, Time = 2.018 msec, Performace = 261.884 GFlop/s
template<int LB, int STEP>
__global__ void kernel_2_2_W1(
	const float* __restrict__ deltaY,
	const float* __restrict__ W,
		  float* __restrict__ deltaX,
	int IC, int OC,
	int ic_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2  Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float2 dYs[2][1 << LB][(1 << LB) + 1];

	//prepare for GN = IC
	int ic0 = (((blockIdx.y << LB) + ty) << 1) + ic_index;

	//prepare for GM = N * IH * IW
	int j0 = (((blockIdx.x << LB) + tx) << 1) + j_index;
	int tj0 = j0 * OC, tj1 = tj0 + OC;

	const int OK = (OC >> LB);
	if (OK) {
		//load 2 elements from W[OC, FH, FW, IC] : Wr[IC, FH(r), FW(r), OC]
		int W_oc = tx;
		Ws[buf][tx][ty] = *(float2*)(&W[W_oc*IC + ic0]);

		//load 2 elements from deltaY[N, OH, OW, OC] : deltaYp[N, OH=OHp, OW=OWp, OC]
		int dY_oc = ty;
		dYs[buf][ty][tx].x = deltaY[tj0 + dY_oc];
		dYs[buf][ty][tx].y = deltaY[tj1 + dY_oc];
		__syncthreads();
	}

	//compute area----------------------------------------------------
	float2 v0 = make_float2(0, 0);
	float2 v1 = make_float2(0, 0);
	for (int ok = 1; ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float2  w = Ws[buf][ik][ty];
			float2 dy = dYs[buf][ik][tx];
			simdMM2(v0, dy.x, w);
			simdMM2(v1, dy.y, w);
		}
		buf ^= 1;

		//load 2 elements from W[OC, FH, FW, IC] : Wr[IC, FH(r), FW(r), OC]
		int W_oc = (ok << LB) + tx;
		Ws[buf][tx][ty] = *(float2*)(&W[W_oc*IC + ic0]);

		//load 2 elements from deltaY[N, OH, OW, OC] : deltaYp[N, OH=OHp, OW=OWp, OC]
		int dY_oc = (ok << LB) + ty;
		dYs[buf][ty][tx].x = deltaY[tj0 + dY_oc];
		dYs[buf][ty][tx].y = deltaY[tj1 + dY_oc];
		__syncthreads();
	}
	if (OK)
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float2  w = Ws[buf][ik][ty];
			float2 dy = dYs[buf][ik][tx];
			simdMM2(v0, dy.x, w);
			simdMM2(v1, dy.y, w);
		}

	//when GK % STEP !=0---------------------------------
	for (int oc = OC - (OC&(STEP - 1)); oc < OC; oc++)
	{
		int dY_oc = oc;
		//load 2 elements from W
		float2 w = *(float2*)(&W[dY_oc*IC + ic0]);

		//load 2 elements from deltaY
		float2 dy;
		dy.x = deltaY[tj0 + dY_oc];
		dy.y = deltaY[tj1 + dY_oc];

		simdMM2(v0, dy.x, w);
		simdMM2(v1, dy.y, w);
	}
	//when GK % STEP !=0---------------------------------

	j0 = j0 * IC + ic0;
	int j1 = j0 + IC;//j0 = ((n * OH + oh) * OW + ow) * IC + ic

	*(float2*)(deltaX + j0) = v0;
	*(float2*)(deltaX + j1) = v1;
}

#endif


//(Y: BLOKC_SIZE*2, X: BLOCK_SIZE*1)
#ifndef DECONV3D_DX_KERNEL_2_1_W1
#define DECONV3D_DX_KERNEL_2_1_W1

//LB = 4: Size = 0.5, Time = 2.614 msec, Performace = 410.766 GFlop/s
//LB = 4: Size = 0.136719, Time = 1.358 msec, Performace = 216.201 GFlop/s
//LB = 3: Size = 0.136719, Time = 1.236 msec, Performace = 237.541 GFlop/s
template<int LB, int STEP>
__global__ void kernel_2_1_W1(
	const float* __restrict__ deltaY,
	const float* __restrict__ W,
		  float* __restrict__ deltaX,
	int IC, int OC,
	int ic_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float dYs[2][1 << LB][(1 << LB) + 1];

	//prepare for GN = IC
	int ic0 = (((blockIdx.y << LB) + ty) << 1) + ic_index;

	//prepare for GM = N * IH * IW
	int j0 = ((blockIdx.x << LB) + tx) + j_index;
	int tj0 = j0 * OC;

	const int OK = (OC >> LB);
	if (OK) {
		//load 2 elements from W[OC, FH, FW, IC] : Wr[IC, FH(r), FW(r), OC]
		int W_oc = tx;
		Ws[buf][tx][ty] = *(float2*)(&W[W_oc*IC + ic0]);

		//load 1 element from deltaY[N, OH, OW, OC] : deltaYp[N, OH=OHp, OW=OWp, OC]
		int dY_oc = ty;
		dYs[buf][ty][tx] = deltaY[tj0 + dY_oc];
		__syncthreads();
	}

	//compute area----------------------------------------------------
	float2 v = make_float2(0, 0);
	for (int ok = 1; ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float2  w =  Ws[buf][ik][ty];
			float  dy = dYs[buf][ik][tx];
			simdMM2(v, dy, w);
		}
		buf ^= 1;

		//load 2 elements from W[OC, FH, FW, IC] : Wr[IC, FH(r), FW(r), OC]
		int W_oc = (ok << LB) + tx;
		Ws[buf][tx][ty] = *(float2*)(&W[W_oc*IC + ic0]);

		//load 2 elements from deltaY[N, OH, OW, OC] : deltaYp[N, OH=OHp, OW=OWp, OC]
		int dY_oc = (ok << LB) + ty;
		dYs[buf][ty][tx] = deltaY[tj0 + dY_oc];
		__syncthreads();
	}
	if (OK)
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float2  w =  Ws[buf][ik][ty];
			float  dy = dYs[buf][ik][tx];
			simdMM2(v, dy, w);
		}

	//when GK % STEP !=0---------------------------------
	for (int oc = OC - (OC&(STEP - 1)); oc < OC; oc++)
	{
		//load 2 elements from W
		float2 w = *(float2*)(&W[oc*IC + ic0]);

		//load 1 element1 from deltaY
		float dy = deltaY[tj0 + oc];

		simdMM2(v, dy, w);
	}
	//when GK % STEP !=0---------------------------------

	*(float2*)(&deltaX[j0 * IC + ic0]) = v;
}

#endif


//(Y: BLOKC_SIZE*2, X: BLOCK_SIZE*2)
#ifndef DECONV3D_DX_KERNEL_1_2_W1
#define DECONV3D_DX_KERNEL_1_2_W1

//LB = 4: Size = 0.5, Time = 4.926 msec, Performace = 217.974 GFlop/s
//LB = 4: Size = 0.136719, Time = 1.998 msec, Performace = 146.948 GFlop/s
//LB = 3: Size = 0.136719, Time = 1.808 msec, Performace = 162.39  GFlop/s
template<int LB, int STEP>
__global__ void kernel_1_2_W1(
	const float* __restrict__ deltaY,
	const float* __restrict__ W,
		  float* __restrict__ deltaX,
	int IC, int OC,
	int ic_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float   Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float2 dYs[2][1 << LB][(1 << LB) + 1];

	//prepare for GN = IC
	int ic0 = ((blockIdx.y << LB) + ty) + ic_index;

	//prepare for GM = N * IH * IW
	int j0 = (((blockIdx.x << LB) + tx) << 1) + j_index;
	int tj0 = j0 * OC, tj1 = tj0 + OC;

	int OK = (OC >> LB);
	if (OK) {
		//load 1 element from W[OC, FH, FW, IC] : Wr[IC, FH(r), FW(r), OC]
		int W_oc = tx;
		Ws[buf][tx][ty] = W[W_oc*IC + ic0];

		//load 2 elements from deltaY[N, OH, OW, OC] : deltaYp[N, OH=OHp, OW=OWp, OC]
		int dY_oc = ty;
		dYs[buf][ty][tx].x = deltaY[tj0 + dY_oc];
		dYs[buf][ty][tx].y = deltaY[tj1 + dY_oc];
		__syncthreads();
	}

	//compute area----------------------------------------------------
	float2 v = make_float2(0, 0);
	for (int ok = 1; ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float   w =  Ws[buf][ik][ty];
			float2 dy = dYs[buf][ik][tx];
			simdMM2(v, w, dy);
		}
		buf ^= 1;

		//load 1 element from W[OC, FH, FW, IC] : Wr[IC, FH(r), FW(r), OC]
		int W_oc = (ok << LB) + tx;
		Ws[buf][tx][ty] = W[W_oc*IC + ic0];

		//load 2 elements from deltaY[N, OH, OW, OC] : deltaYp[N, OH=OHp, OW=OWp, OC]
		int dY_oc = (ok << LB) + ty;
		dYs[buf][ty][tx].x = deltaY[tj0 + dY_oc];
		dYs[buf][ty][tx].y = deltaY[tj1 + dY_oc];
		__syncthreads();
	}
	if (OK)
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float   w =  Ws[buf][ik][ty];
			float2 dy = dYs[buf][ik][tx];
			simdMM2(v, w, dy);
		}

	//when GK % STEP !=0---------------------------------
	for (int oc = OC - (OC&(STEP - 1)); oc < OC; oc++)
	{
		//load 1 element from W
		float w = W[oc*IC + ic0];

		//load 2 elements from deltaY
		float2 dy;
		dy.x = deltaY[tj0 + oc];
		dy.y = deltaY[tj1 + oc];

		simdMM2(v, w, dy);
	}
	//when GK % STEP !=0---------------------------------

	j0 = j0*IC + ic0; //j0 = ((n * OH + oh) * OW + ow) * IC + ic
	int j1 = j0 + IC;

	deltaX[j0] = v.x;
	deltaX[j1] = v.y;
}

#endif


//(Y: BLOKC_SIZE*2, X: BLOCK_SIZE*2)
#ifndef DECONV3D_KERNEL_DX_1_1_W1
#define DECONV3D_KERNEL_DX_1_1_W1

//LB = 4: Size = 0.5, Time = 4.946 msec, Performace = 217.093 GFlop/s
//LB = 4: Size = 0.136719, Time = 2.2 msec, Performace = 133.455 GFlop/s
//LB = 3: Size = 0.136719, Time = 1.9 msec, Performace = 154.527 GFlop/s
template<int LB, int STEP>
__global__ void kernel_1_1_W1(
	const float* __restrict__ deltaY,
	const float* __restrict__ W,
		  float* __restrict__ deltaX,
	int IC, int OC,
	int ic_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float  Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float dYs[2][1 << LB][(1 << LB) + 1];

	//prepare for GN = IC
	int ic0 = ((blockIdx.y << LB) + ty) + ic_index;

	//prepare for GM = N * IH * IW
	int j0 = ((blockIdx.x << LB) + tx) + j_index;
	int tj0 = j0 * OC;

	int OK = (OC >> LB);
	if (OK) {
		//load 1 element from W[OC, FH, FW, IC] : Wr[IC, FH(r), FW(r), OC]
		int W_oc = tx;
		Ws[buf][tx][ty] = W[W_oc*IC + ic0];

		//load 1 element from deltaY[N, OH, OW, OC] : deltaYp[N, OH=OHp, OW=OWp, OC]
		int dY_oc = ty;
		dYs[buf][ty][tx] = deltaY[tj0 + dY_oc];
		__syncthreads();
	}

	//compute area----------------------------------------------------
	float v = 0;
	for (int ok = 1; ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float  w =  Ws[buf][ik][ty];
			float dy = dYs[buf][ik][tx];
			v += dy * w;
		}
		buf ^= 1;

		//load 1 element from W[OC, FH, FW, IC] : Wr[IC, FH(r), FW(r), OC]
		int W_oc = (ok << LB) + tx;
		Ws[buf][tx][ty] = W[W_oc*IC + ic0];

		//load 1 element from deltaY[N, OH, OW, OC] : deltaYp[N, OH=OHp, OW=OWp, OC]
		int dY_oc = (ok << LB) + ty;
		dYs[buf][ty][tx] = deltaY[tj0 + dY_oc];
		__syncthreads();
	}
	if (OK)
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float  w =  Ws[buf][ik][ty];
			float dy = dYs[buf][ik][tx];
			v += dy * w;
		}

	//when GK % STEP !=0---------------------------------
	for (int oc = OC - (OC&(STEP - 1)); oc < OC; oc++)
	{
		float w = W[oc*IC + ic0];//load 1 element from W
		float dy = deltaY[tj0 + oc];//load 1 element from deltaY

		v += dy * w;
	}
	//when GK % STEP !=0---------------------------------

	deltaX[j0*IC + ic0] = v;
}

#endif

#endif