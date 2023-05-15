

//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*4)
#ifndef CONV_3D_WINOGRAD_KERNEL_W3_C1
#define CONV_3D_WINOGRAD_KERNEL_W3_C1

#define wingrad_c1(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	conv3dWinoGrad_kernel_c1<LB,(1<<LB), (1<<LB>>2),(2<<LB>>2),(3<<LB>>2)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, IC, OC, ph, pw, oc_index, j_index)

//128 register
//LB = 4: Size = 1.125, Time = 2.09723 msec, Performace = 1151.96 GFlop/s
template<int LB, int BSIZE, int STEP, int STEP2, int STEP3>
__global__ void conv3dWinoGrad_kernel_c1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	__shared__ float4 Gs[1 << LB][(2 << LB) + 1];
	__shared__ float4 Ds[1 << LB][(1 << LB) + 1];

	//prepare for GN = OC
	int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty / STEP) << 1);
	const int Wstride = IC * OC;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx / STEP) << 1);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	const int Y0 = j0 * OC + oc0;

	//compute area-------------------------------------------------
	float4  v0 = float4{ 0, 0, 0, 0 }, v1 = float4{ 0, 0, 0, 0 };
	float4  v2 = float4{ 0, 0, 0, 0 }, v3 = float4{ 0, 0, 0, 0 };
	float4  v4 = float4{ 0, 0, 0, 0 }, v5 = float4{ 0, 0, 0, 0 };
	float4  v6 = float4{ 0, 0, 0, 0 }, v7 = float4{ 0, 0, 0, 0 };
	float4  v8 = float4{ 0, 0, 0, 0 }, v9 = float4{ 0, 0, 0, 0 };
	float4 v10 = float4{ 0, 0, 0, 0 }, v11 = float4{ 0, 0, 0, 0 };
	float4 v12 = float4{ 0, 0, 0, 0 }, v13 = float4{ 0, 0, 0, 0 };
	float4 v14 = float4{ 0, 0, 0, 0 }, v15 = float4{ 0, 0, 0, 0 };

	for (int fh = 0; fh < 3; fh++)
	{
		//4 thread work together
		int tih0 = toh0 - ph + fh, tiw0 = tow0 - pw;
		bool lh0 = (tih0 >= 0) && (tih0 < IH);
		bool ly0 = lh0 && (tiw0 >=  0) && (tiw0     < IW);
		bool ly1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);
		bool ly2 = lh0 && (tiw0 >= -2) && (tiw0 + 2 < IW);
		bool ly3 = lh0 && (tiw0 >= -3) && (tiw0 + 3 < IW);
		char LY = ly0 + (ly1 << 1) + (ly2 << 2) + (ly3 << 3);
	
		for (int oic = 0, OIC = (IC << 2 >> LB); oic < OIC; oic++)
		{
			//load 4 group from CW: 4 threads work together
			int wic = (ty & (STEP - 1)) + (oic << LB >> 2);
			int W0 = ((fh * 3)*IC + wic)*OC;
			int W1 = W0 + Wstride, W2 = W1 + Wstride;
			float2 g0 = *(float2*)(CW + W0);
			float2 g1 = *(float2*)(CW + W1);
			float2 g2 = *(float2*)(CW + W2);
			Gs[ty][tx] = winograd_g(g0.x, g1.x, g2.x);
			Gs[ty][tx + BSIZE] = winograd_g(g0.y, g1.y, g2.y);

			//load 2 group from X: 4 threads work together
			int xic = (tx & (STEP - 1)) + (oic << LB >> 2);
			int X1 = ((tn0*IH + tih0)*IW + tiw0 + 1)*IC + xic;
			float d0 = ((LY & 1) ? X[X1 - IC] : 0);
			float d1 = ((LY & 2) ? X[X1] : 0);
			float d2 = ((LY & 4) ? X[X1 + IC] : 0);
			float d3 = ((LY & 8) ? X[X1 + (IC << 1)] : 0);
			Ds[tx][ty] = winograd_d(d0, d1, d2, d3);
			__syncthreads();

//#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 d0 = Ds[ik        ][ty];
				float4 d1 = Ds[ik + STEP ][ty];
				float4 d2 = Ds[ik + STEP2][ty];
				float4 d3 = Ds[ik + STEP3][ty];

				float4 g0 = Gs[ik        ][tx], g1 = Gs[ik        ][tx + BSIZE];
				float4 g2 = Gs[ik + STEP ][tx], g3 = Gs[ik + STEP ][tx + BSIZE];
				float4 g4 = Gs[ik + STEP2][tx], g5 = Gs[ik + STEP2][tx + BSIZE];
				float4 g6 = Gs[ik + STEP3][tx], g7 = Gs[ik + STEP3][tx + BSIZE];

				wino_grad4_GxW( v0,  v2, g0, g1, g2, g3, d0);
				wino_grad4_GxW( v1,  v3, g4, g5, g6, g7, d0);

				wino_grad4_GxW( v4,  v6, g0, g1, g2, g3, d1);
				wino_grad4_GxW( v5,  v7, g4, g5, g6, g7, d1);

				wino_grad4_GxW( v8, v10, g0, g1, g2, g3, d2);
				wino_grad4_GxW( v9, v11, g4, g5, g6, g7, d2);

				wino_grad4_GxW(v12, v14, g0, g1, g2, g3, d3);
				wino_grad4_GxW(v13, v15, g4, g5, g6, g7, d3);
			}
			__syncthreads();
		}
	}

	const int Y1 = Y0 + OC, Y2 = Y1 + OC, Y3 = Y2 + OC;
	const int Y4 = Y3 + OC, Y5 = Y4 + OC, Y6 = Y5 + OC, Y7 = Y6 + OC;

	*(float4*)(Y + Y0) = v0; *(float4*)(Y + Y0 + 4) = v1;
	*(float4*)(Y + Y1) = v2; *(float4*)(Y + Y1 + 4) = v3;
	*(float4*)(Y + Y2) = v4; *(float4*)(Y + Y2 + 4) = v5;
	*(float4*)(Y + Y3) = v6; *(float4*)(Y + Y3 + 4) = v7;
	*(float4*)(Y + Y4) = v8; *(float4*)(Y + Y4 + 4) = v9;
	*(float4*)(Y + Y5) = v10; *(float4*)(Y + Y5 + 4) = v11;
	*(float4*)(Y + Y6) = v12; *(float4*)(Y + Y6 + 4) = v13;
	*(float4*)(Y + Y7) = v14; *(float4*)(Y + Y7 + 4) = v15;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*4)
#ifndef CONV_3D_WINOGRAD_KERNEL_W3_C2
#define CONV_3D_WINOGRAD_KERNEL_W3_C2

#define wingrad_c2(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	conv3dWinoGrad_kernel_c2<LB,(1<<LB), (1<<LB>>2),(2<<LB>>2),(3<<LB>>2)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, IC, OC, ph, pw, oc_index, j_index)

//128 register
//LB = 4: Size = 1.125, Time = 2.09723 msec, Performace = 1151.96 GFlop/s
template<int LB, int BSIZE, int STEP, int STEP2, int STEP3>
__global__ void conv3dWinoGrad_kernel_c2(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	__shared__ float4 Gs[1 << LB][(2 << LB) + 1];
	__shared__ float4 Ds[1 << LB][(1 << LB) + 1];

	//prepare for GN = OC
	int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty << 2 >> LB) << 1);
	const int Wstride = IC * OC;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx << 2 >> LB) << 1);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	const int Y0 = j0 * OC + oc0;

	//compute area-------------------------------------------------
	float4  v0 = float4{ 0, 0, 0, 0 }, v1 = float4{ 0, 0, 0, 0 };
	float4  v2 = float4{ 0, 0, 0, 0 }, v3 = float4{ 0, 0, 0, 0 };
	float4  v4 = float4{ 0, 0, 0, 0 }, v5 = float4{ 0, 0, 0, 0 };
	float4  v6 = float4{ 0, 0, 0, 0 }, v7 = float4{ 0, 0, 0, 0 };
	float4  v8 = float4{ 0, 0, 0, 0 }, v9 = float4{ 0, 0, 0, 0 };
	float4 v10 = float4{ 0, 0, 0, 0 }, v11 = float4{ 0, 0, 0, 0 };
	float4 v12 = float4{ 0, 0, 0, 0 }, v13 = float4{ 0, 0, 0, 0 };
	float4 v14 = float4{ 0, 0, 0, 0 }, v15 = float4{ 0, 0, 0, 0 };

#pragma unroll
	for (int fh = 0; fh < 3; fh++)
	{
		int tih0 = toh0 - ph + fh, tiw0 = tow0 - pw;
		bool lh0 = (tih0 >= 0) && (tih0 < IH);
		bool ly0 = lh0 && (tiw0 >= 0) && (tiw0 < IW);
		bool ly1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);
		bool ly2 = lh0 && (tiw0 >= -2) && (tiw0 + 2 < IW);
		bool ly3 = lh0 && (tiw0 >= -3) && (tiw0 + 3 < IW);
		char LY = ly0 + (ly1 << 1) + (ly2 << 2) + (ly3 << 3);

		for (int oic = 0, OIC = (IC << 2 >> LB); oic < OIC; oic++)
		{
			//load 4 group from CW: 4 threads work together
			int wic = (ty & (STEP - 1)) + (oic << LB >> 2);
			int W0 = ((fh * 3)*IC + wic)*OC;
			int W1 = W0 + Wstride, W2 = W1 + Wstride;
			float2 g0 = *(float2*)(CW + W0);
			float2 g1 = *(float2*)(CW + W1);
			float2 g2 = *(float2*)(CW + W2);
			Gs[ty][tx] = winograd_g(g0.x, g1.x, g2.x);
			Gs[ty][tx + BSIZE] = winograd_g(g0.y, g1.y, g2.y);

			//load 2 group from X: 4 threads work together
			int xic = (tx & (STEP - 1)) + (oic << LB >> 2);
			int X1 = ((tn0*IH + tih0)*IW + tiw0 + 1)*IC + xic;
			float d0 = ((LY & 1) ? X[X1 - IC] : 0);
			float d1 = ((LY & 2) ? X[X1] : 0);
			float d2 = ((LY & 4) ? X[X1 + IC] : 0);
			float d3 = ((LY & 8) ? X[X1 + (IC << 1)] : 0);
			Ds[tx][ty] = winograd_d(d0, d1, d2, d3);
			__syncthreads();

//#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 g0 = Gs[ik][tx], g1 = Gs[ik][tx + BSIZE];
				float4 g2 = Gs[ik + STEP][tx], g3 = Gs[ik + STEP][tx + BSIZE];
				float4 g4 = Gs[ik + STEP2][tx], g5 = Gs[ik + STEP2][tx + BSIZE];
				float4 g6 = Gs[ik + STEP3][tx], g7 = Gs[ik + STEP3][tx + BSIZE];

				float4 d0 = Ds[ik][ty];
				float4 d1 = Ds[ik + STEP][ty];
				float4 d2 = Ds[ik + STEP2][ty];
				float4 d3 = Ds[ik + STEP3][ty];

				wino_grad4_GxW(v0, v2, g0, g1, g2, g3, d0);
				wino_grad4_GxW(v1, v3, g4, g5, g6, g7, d0);

				wino_grad4_GxW(v4, v6, g0, g1, g2, g3, d1);
				wino_grad4_GxW(v5, v7, g4, g5, g6, g7, d1);

				wino_grad4_GxW(v8, v10, g0, g1, g2, g3, d2);
				wino_grad4_GxW(v9, v11, g4, g5, g6, g7, d2);

				wino_grad4_GxW(v12, v14, g0, g1, g2, g3, d3);
				wino_grad4_GxW(v13, v15, g4, g5, g6, g7, d3);
			}
			__syncthreads();
		}
	}

	const int Y1 = Y0 + OC, Y2 = Y1 + OC, Y3 = Y2 + OC;
	const int Y4 = Y3 + OC, Y5 = Y4 + OC, Y6 = Y5 + OC, Y7 = Y6 + OC;

	*(float4*)(Y + Y0) = v0; *(float4*)(Y + Y0 + 4) = v1;
	*(float4*)(Y + Y1) = v2; *(float4*)(Y + Y1 + 4) = v3;
	*(float4*)(Y + Y2) = v4; *(float4*)(Y + Y2 + 4) = v5;
	*(float4*)(Y + Y3) = v6; *(float4*)(Y + Y3 + 4) = v7;
	*(float4*)(Y + Y4) = v8; *(float4*)(Y + Y4 + 4) = v9;
	*(float4*)(Y + Y5) = v10; *(float4*)(Y + Y5 + 4) = v11;
	*(float4*)(Y + Y6) = v12; *(float4*)(Y + Y6 + 4) = v13;
	*(float4*)(Y + Y7) = v14; *(float4*)(Y + Y7 + 4) = v15;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8)
#ifndef CONV_3D_WINOGRAD_KERNEL_W3_C3
#define CONV_3D_WINOGRAD_KERNEL_W3_C3

#define wingrad_c3(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	conv3dWinoGrad_kernel_c3<LB,(1<<LB), (1<<LB>>2),(2<<LB>>2),(3<<LB>>2)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, IC, OC, ph, pw, oc_index, j_index)

//128 register
//LB = 4: Size = 1.125, Time = 2.09723 msec, Performace = 1151.96 GFlop/s
template<int LB, int BSIZE, int STEP, int STEP2, int STEP3>
__global__ void conv3dWinoGrad_kernel_c3(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	__shared__ float4 Gs[1 << LB][(2 << LB) + 1];
	__shared__ float4 Ds[1 << LB][(1 << LB) + 1];

	//prepare for GN = OC
	int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty << 2 >> LB) << 1);
	const int Wstride = IC * OC;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx << 2 >> LB) << 1);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	const int Y0 = j0 * OC + oc0;

	//compute area-------------------------------------------------
	float4  v0 = float4{ 0, 0, 0, 0 }, v1 = float4{ 0, 0, 0, 0 };
	float4  v2 = float4{ 0, 0, 0, 0 }, v3 = float4{ 0, 0, 0, 0 };
	float4  v4 = float4{ 0, 0, 0, 0 }, v5 = float4{ 0, 0, 0, 0 };
	float4  v6 = float4{ 0, 0, 0, 0 }, v7 = float4{ 0, 0, 0, 0 };
	float4  v8 = float4{ 0, 0, 0, 0 }, v9 = float4{ 0, 0, 0, 0 };
	float4 v10 = float4{ 0, 0, 0, 0 }, v11 = float4{ 0, 0, 0, 0 };
	float4 v12 = float4{ 0, 0, 0, 0 }, v13 = float4{ 0, 0, 0, 0 };
	float4 v14 = float4{ 0, 0, 0, 0 }, v15 = float4{ 0, 0, 0, 0 };

#pragma unroll
	for (int fh = 0; fh < 3; fh++)
	{
		int tih0 = toh0 - ph + fh, tiw0 = tow0 - pw;
		bool lh0 = (tih0 >= 0) && (tih0 < IH);
		bool ly0 = lh0 && (tiw0 >= 0) && (tiw0 < IW);
		bool ly1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);
		bool ly2 = lh0 && (tiw0 >= -2) && (tiw0 + 2 < IW);
		bool ly3 = lh0 && (tiw0 >= -3) && (tiw0 + 3 < IW);
		char LY = ly0 + (ly1 << 1) + (ly2 << 2) + (ly3 << 3);

		for (int oic = 0, OIC = (IC << 2 >> LB); oic < OIC; oic++)
		{
			//load 4 group from CW: 4 threads work together
			int wic = (ty & (STEP - 1)) + (oic << LB >> 2);
			int W0 = ((fh * 3)*IC + wic)*OC;
			int W1 = W0 + Wstride, W2 = W1 + Wstride;
			float2 g0 = *(float2*)(CW + W0);
			float2 g1 = *(float2*)(CW + W1);
			float2 g2 = *(float2*)(CW + W2);
			Gs[ty][tx] = winograd_g(g0.x, g1.x, g2.x);
			Gs[ty][tx + BSIZE] = winograd_g(g0.y, g1.y, g2.y);

			//load 2 group from X: 4 threads work together
			int xic = (tx & (STEP - 1)) + (oic << LB >> 2);
			int X1 = ((tn0*IH + tih0)*IW + tiw0 + 1)*IC + xic;
			float d0 = ((LY & 1) ? X[X1 - IC] : 0);
			float d1 = ((LY & 2) ? X[X1] : 0);
			float d2 = ((LY & 4) ? X[X1 + IC] : 0);
			float d3 = ((LY & 8) ? X[X1 + (IC << 1)] : 0);
			Ds[tx][ty] = winograd_d(d0, d1, d2, d3);
			__syncthreads();

			//#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 g0 = Gs[ik][tx], g1 = Gs[ik][tx + BSIZE];
				float4 g2 = Gs[ik + STEP][tx], g3 = Gs[ik + STEP][tx + BSIZE];
				float4 g4 = Gs[ik + STEP2][tx], g5 = Gs[ik + STEP2][tx + BSIZE];
				float4 g6 = Gs[ik + STEP3][tx], g7 = Gs[ik + STEP3][tx + BSIZE];

				float4 d0 = Ds[ik][ty];
				float4 d1 = Ds[ik + STEP][ty];
				float4 d2 = Ds[ik + STEP2][ty];
				float4 d3 = Ds[ik + STEP3][ty];

				wino_grad4_GxW(v0, v2, g0, g1, g2, g3, d0);
				wino_grad4_GxW(v1, v3, g4, g5, g6, g7, d0);

				wino_grad4_GxW(v4, v6, g0, g1, g2, g3, d1);
				wino_grad4_GxW(v5, v7, g4, g5, g6, g7, d1);

				wino_grad4_GxW(v8, v10, g0, g1, g2, g3, d2);
				wino_grad4_GxW(v9, v11, g4, g5, g6, g7, d2);

				wino_grad4_GxW(v12, v14, g0, g1, g2, g3, d3);
				wino_grad4_GxW(v13, v15, g4, g5, g6, g7, d3);
			}
			__syncthreads();
		}
	}

	const int Y1 = Y0 + OC, Y2 = Y1 + OC, Y3 = Y2 + OC;
	const int Y4 = Y3 + OC, Y5 = Y4 + OC, Y6 = Y5 + OC, Y7 = Y6 + OC;

	*(float4*)(Y + Y0) = v0; *(float4*)(Y + Y0 + 4) = v1;
	*(float4*)(Y + Y1) = v2; *(float4*)(Y + Y1 + 4) = v3;
	*(float4*)(Y + Y2) = v4; *(float4*)(Y + Y2 + 4) = v5;
	*(float4*)(Y + Y3) = v6; *(float4*)(Y + Y3 + 4) = v7;
	*(float4*)(Y + Y4) = v8; *(float4*)(Y + Y4 + 4) = v9;
	*(float4*)(Y + Y5) = v10; *(float4*)(Y + Y5 + 4) = v11;
	*(float4*)(Y + Y6) = v12; *(float4*)(Y + Y6 + 4) = v13;
	*(float4*)(Y + Y7) = v14; *(float4*)(Y + Y7 + 4) = v15;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*4)
#ifndef CONV_3D_WINOGRAD_KERNEL_W3_C4
#define CONV_3D_WINOGRAD_KERNEL_W3_C4

#define wingrad_c4(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	conv3dWinoGrad_kernel_c4<LB,(1<<LB), (1<<LB>>2),(2<<LB>>2),(3<<LB>>2)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, IC, OC, ph, pw, oc_index, j_index)

//128 register
//LB = 4: Size = 1.125, Time = 1.98027 msec, Performace = 1219.99 GFlop/s
template<int LB, int BSIZE, int STEP, int STEP2, int STEP3>
__global__ void conv3dWinoGrad_kernel_c4(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Gs[2][1 << LB][(2 << LB) + 1];
	__shared__ float4 Ds[2][1 << LB][(1 << LB) + 1];

	//prepare for GN = OC
	int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty << 2 >> LB) << 1);
	const int Wstride = IC * OC;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx << 2 >> LB) << 1);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	const int Y0 = j0 * OC + oc0;

	//compute area-------------------------------------------------
	float4  v0 = float4{ 0, 0, 0, 0 }, v1 = float4{ 0, 0, 0, 0 };
	float4  v2 = float4{ 0, 0, 0, 0 }, v3 = float4{ 0, 0, 0, 0 };
	float4  v4 = float4{ 0, 0, 0, 0 }, v5 = float4{ 0, 0, 0, 0 };
	float4  v6 = float4{ 0, 0, 0, 0 }, v7 = float4{ 0, 0, 0, 0 };
	float4  v8 = float4{ 0, 0, 0, 0 }, v9 = float4{ 0, 0, 0, 0 };
	float4 v10 = float4{ 0, 0, 0, 0 }, v11 = float4{ 0, 0, 0, 0 };
	float4 v12 = float4{ 0, 0, 0, 0 }, v13 = float4{ 0, 0, 0, 0 };
	float4 v14 = float4{ 0, 0, 0, 0 }, v15 = float4{ 0, 0, 0, 0 };

#pragma unroll
	for (int fh = 0; fh < 3; fh++)
	{
		int tih0 = toh0 - ph + fh, tiw0 = tow0 - pw;
		bool lh0 = (tih0 >= 0) && (tih0 < IH);
		bool ly0 = lh0 && (tiw0 >= 0) && (tiw0 < IW);
		bool ly1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);
		bool ly2 = lh0 && (tiw0 >= -2) && (tiw0 + 2 < IW);
		bool ly3 = lh0 && (tiw0 >= -3) && (tiw0 + 3 < IW);
		char LY = ly0 + (ly1 << 1) + (ly2 << 2) + (ly3 << 3);

		//load 4 group from CW: 4 threads work together
		int wic = (ty & (STEP - 1));
		int W0 = ((fh * 3)*IC + wic)*OC;
		int W1 = W0 + Wstride, W2 = W1 + Wstride;
		float2 g0 = *(float2*)(CW + W0);
		float2 g1 = *(float2*)(CW + W1);
		float2 g2 = *(float2*)(CW + W2);
		Gs[buf][ty][tx] = winograd_g(g0.x, g1.x, g2.x);
		Gs[buf][ty][tx + BSIZE] = winograd_g(g0.y, g1.y, g2.y);

		//load 2 group from X: 4 threads work together
		int xic = (tx & (STEP - 1));
		int X1 = ((tn0*IH + tih0)*IW + tiw0 + 1)*IC + xic;
		float d0 = ((LY & 1) ? X[X1 - IC] : 0);
		float d1 = ((LY & 2) ? X[X1] : 0);
		float d2 = ((LY & 4) ? X[X1 + IC] : 0);
		float d3 = ((LY & 8) ? X[X1 + (IC << 1)] : 0);
		Ds[buf][tx][ty] = winograd_d(d0, d1, d2, d3);
		__syncthreads();

		for (int oic = 1, OIC = (IC << 2 >> LB); oic < OIC; oic++)
		{
			//#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 g0 = Gs[buf][ik][tx], g1 = Gs[buf][ik][tx + BSIZE];
				float4 g2 = Gs[buf][ik + STEP][tx], g3 = Gs[buf][ik + STEP][tx + BSIZE];
				float4 g4 = Gs[buf][ik + STEP2][tx], g5 = Gs[buf][ik + STEP2][tx + BSIZE];
				float4 g6 = Gs[buf][ik + STEP3][tx], g7 = Gs[buf][ik + STEP3][tx + BSIZE];

				float4 d0 = Ds[buf][ik][ty];
				float4 d1 = Ds[buf][ik + STEP][ty];
				float4 d2 = Ds[buf][ik + STEP2][ty];
				float4 d3 = Ds[buf][ik + STEP3][ty];

				wino_grad4_GxW(v0, v2, g0, g1, g2, g3, d0);
				wino_grad4_GxW(v1, v3, g4, g5, g6, g7, d0);

				wino_grad4_GxW(v4, v6, g0, g1, g2, g3, d1);
				wino_grad4_GxW(v5, v7, g4, g5, g6, g7, d1);

				wino_grad4_GxW(v8, v10, g0, g1, g2, g3, d2);
				wino_grad4_GxW(v9, v11, g4, g5, g6, g7, d2);

				wino_grad4_GxW(v12, v14, g0, g1, g2, g3, d3);
				wino_grad4_GxW(v13, v15, g4, g5, g6, g7, d3);
			}
			buf ^= 1;

			//load 4 group from CW: 4 threads work together
			int wic = (ty & (STEP - 1)) + (oic << LB >> 2);
			int W0 = ((fh * 3)*IC + wic)*OC;
			int W1 = W0 + Wstride, W2 = W1 + Wstride;
			float2 g0 = *(float2*)(CW + W0);
			float2 g1 = *(float2*)(CW + W1);
			float2 g2 = *(float2*)(CW + W2);
			Gs[buf][ty][tx] = winograd_g(g0.x, g1.x, g2.x);
			Gs[buf][ty][tx + BSIZE] = winograd_g(g0.y, g1.y, g2.y);

			//load 2 group from X: 4 threads work together
			int xic = (tx & (STEP - 1)) + (oic << LB >> 2);
			int X1 = ((tn0*IH + tih0)*IW + tiw0 + 1)*IC + xic;
			float d0 = ((LY & 1) ? X[X1 - IC] : 0);
			float d1 = ((LY & 2) ? X[X1] : 0);
			float d2 = ((LY & 4) ? X[X1 + IC] : 0);
			float d3 = ((LY & 8) ? X[X1 + (IC << 1)] : 0);
			Ds[buf][tx][ty] = winograd_d(d0, d1, d2, d3);
			__syncthreads();
		}
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 g0 = Gs[buf][ik][tx], g1 = Gs[buf][ik][tx + BSIZE];
			float4 g2 = Gs[buf][ik + STEP][tx], g3 = Gs[buf][ik + STEP][tx + BSIZE];
			float4 g4 = Gs[buf][ik + STEP2][tx], g5 = Gs[buf][ik + STEP2][tx + BSIZE];
			float4 g6 = Gs[buf][ik + STEP3][tx], g7 = Gs[buf][ik + STEP3][tx + BSIZE];

			float4 d0 = Ds[buf][ik][ty];
			float4 d1 = Ds[buf][ik + STEP][ty];
			float4 d2 = Ds[buf][ik + STEP2][ty];
			float4 d3 = Ds[buf][ik + STEP3][ty];

			wino_grad4_GxW(v0, v2, g0, g1, g2, g3, d0);
			wino_grad4_GxW(v1, v3, g4, g5, g6, g7, d0);

			wino_grad4_GxW(v4, v6, g0, g1, g2, g3, d1);
			wino_grad4_GxW(v5, v7, g4, g5, g6, g7, d1);

			wino_grad4_GxW(v8, v10, g0, g1, g2, g3, d2);
			wino_grad4_GxW(v9, v11, g4, g5, g6, g7, d2);

			wino_grad4_GxW(v12, v14, g0, g1, g2, g3, d3);
			wino_grad4_GxW(v13, v15, g4, g5, g6, g7, d3);
		}
	}

	const int Y1 = Y0 + OC, Y2 = Y1 + OC, Y3 = Y2 + OC;
	const int Y4 = Y3 + OC, Y5 = Y4 + OC, Y6 = Y5 + OC, Y7 = Y6 + OC;

	*(float4*)(Y + Y0) = v0; *(float4*)(Y + Y0 + 4) = v1;
	*(float4*)(Y + Y1) = v2; *(float4*)(Y + Y1 + 4) = v3;
	*(float4*)(Y + Y2) = v4; *(float4*)(Y + Y2 + 4) = v5;
	*(float4*)(Y + Y3) = v6; *(float4*)(Y + Y3 + 4) = v7;
	*(float4*)(Y + Y4) = v8; *(float4*)(Y + Y4 + 4) = v9;
	*(float4*)(Y + Y5) = v10; *(float4*)(Y + Y5 + 4) = v11;
	*(float4*)(Y + Y6) = v12; *(float4*)(Y + Y6 + 4) = v13;
	*(float4*)(Y + Y7) = v14; *(float4*)(Y + Y7 + 4) = v15;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*4)
#ifndef CONV_3D_WINOGRAD_KERNEL_W3_C5
#define CONV_3D_WINOGRAD_KERNEL_W3_C5

#define wingrad_c5(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	conv3dWinoGrad_kernel_c5<LB,(1<<LB), (1<<LB>>2),(2<<LB>>2),(3<<LB>>2)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, IC, OC, ph, pw, oc_index, j_index)

//128 register
//LB = 4: Size = 1.125, Time = 1.98027 msec, Performace = 1219.99 GFlop/s
//LB = 3: Size = 1.125, Time = 2.23152 msec, Performace = 1082.63 GFlop/s
template<int LB, int BSIZE, int STEP, int STEP2, int STEP3>
__global__ void conv3dWinoGrad_kernel_c5(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Gs[2][1 << LB][(2 << LB) + 1];
	__shared__ float4 Ds[2][1 << LB][(1 << LB) + 1];

	//prepare for GN = OC
	int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty << 2 >> LB) << 1);
	const int Wstride = IC * OC;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx << 2 >> LB) << 1);
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	const int Y0 = j0 * OC + oc0;

	//compute area-------------------------------------------------
	float4  v0 = float4{ 0, 0, 0, 0 }, v1 = float4{ 0, 0, 0, 0 };
	float4  v2 = float4{ 0, 0, 0, 0 }, v3 = float4{ 0, 0, 0, 0 };
	float4  v4 = float4{ 0, 0, 0, 0 }, v5 = float4{ 0, 0, 0, 0 };
	float4  v6 = float4{ 0, 0, 0, 0 }, v7 = float4{ 0, 0, 0, 0 };
	float4  v8 = float4{ 0, 0, 0, 0 }, v9 = float4{ 0, 0, 0, 0 };
	float4 v10 = float4{ 0, 0, 0, 0 }, v11 = float4{ 0, 0, 0, 0 };
	float4 v12 = float4{ 0, 0, 0, 0 }, v13 = float4{ 0, 0, 0, 0 };
	float4 v14 = float4{ 0, 0, 0, 0 }, v15 = float4{ 0, 0, 0, 0 };

#pragma unroll
	for (int fh = 0; fh < 3; fh++)
	{
		int tih0 = toh0 - ph + fh, tiw0 = tow0 - pw;
		bool lh0 = (tih0 >= 0) && (tih0 < IH);
		bool ly0 = lh0 && (tiw0 >= 0) && (tiw0 < IW);
		bool ly1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);
		bool ly2 = lh0 && (tiw0 >= -2) && (tiw0 + 2 < IW);
		bool ly3 = lh0 && (tiw0 >= -3) && (tiw0 + 3 < IW);
		char LY = ly0 + (ly1 << 1) + (ly2 << 2) + (ly3 << 3);

		//load 4 group from CW: 4 threads work together
		int wic = (ty & (STEP - 1));
		int W0 = ((fh * 3)*IC + wic)*OC;
		int W1 = W0 + Wstride, W2 = W1 + Wstride;
		float2 g0 = *(float2*)(CW + W0);
		float2 g1 = *(float2*)(CW + W1);
		float2 g2 = *(float2*)(CW + W2);
		Gs[buf][ty][tx] = winograd_g(g0.x, g1.x, g2.x);
		Gs[buf][ty][tx + BSIZE] = winograd_g(g0.y, g1.y, g2.y);

		//load 2 group from X: 4 threads work together
		int xic = (tx & (STEP - 1));
		int X1 = ((tn0*IH + tih0)*IW + tiw0 + 1)*IC + xic;
		float d0 = ((LY & 1) ? X[X1 - IC] : 0);
		float d1 = ((LY & 2) ? X[X1] : 0);
		float d2 = ((LY & 4) ? X[X1 + IC] : 0);
		float d3 = ((LY & 8) ? X[X1 + (IC << 1)] : 0);
		Ds[buf][tx][ty] = winograd_d(d0, d1, d2, d3);
		__syncthreads();

		for (int oic = 1, OIC = (IC << 2 >> LB); oic < OIC; oic++)
		{
			//#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 g0 = Gs[buf][ik][tx], g1 = Gs[buf][ik][tx + BSIZE];
				float4 g2 = Gs[buf][ik + STEP][tx], g3 = Gs[buf][ik + STEP][tx + BSIZE];
				float4 g4 = Gs[buf][ik + STEP2][tx], g5 = Gs[buf][ik + STEP2][tx + BSIZE];
				float4 g6 = Gs[buf][ik + STEP3][tx], g7 = Gs[buf][ik + STEP3][tx + BSIZE];

				float4 d0 = Ds[buf][ik][ty];
				float4 d1 = Ds[buf][ik + STEP][ty];
				float4 d2 = Ds[buf][ik + STEP2][ty];
				float4 d3 = Ds[buf][ik + STEP3][ty];

				wino_grad4_GxW(v0, v2, g0, g1, g2, g3, d0);
				wino_grad4_GxW(v1, v3, g4, g5, g6, g7, d0);

				wino_grad4_GxW(v4, v6, g0, g1, g2, g3, d1);
				wino_grad4_GxW(v5, v7, g4, g5, g6, g7, d1);

				wino_grad4_GxW(v8, v10, g0, g1, g2, g3, d2);
				wino_grad4_GxW(v9, v11, g4, g5, g6, g7, d2);

				wino_grad4_GxW(v12, v14, g0, g1, g2, g3, d3);
				wino_grad4_GxW(v13, v15, g4, g5, g6, g7, d3);
			}
			buf ^= 1;

			//load 4 group from CW: 4 threads work together
			int wic = (ty & (STEP - 1)) + (oic << LB >> 2);
			int W0 = ((fh * 3)*IC + wic)*OC;
			int W1 = W0 + Wstride, W2 = W1 + Wstride;
			float2 g0 = *(float2*)(CW + W0);
			float2 g1 = *(float2*)(CW + W1);
			float2 g2 = *(float2*)(CW + W2);
			Gs[buf][ty][tx] = winograd_g(g0.x, g1.x, g2.x);
			Gs[buf][ty][tx + BSIZE] = winograd_g(g0.y, g1.y, g2.y);

			//load 2 group from X: 4 threads work together
			int xic = (tx & (STEP - 1)) + (oic << LB >> 2);
			int X1 = ((tn0*IH + tih0)*IW + tiw0 + 1)*IC + xic;
			float d0 = ((LY & 1) ? X[X1 - IC] : 0);
			float d1 = ((LY & 2) ? X[X1] : 0);
			float d2 = ((LY & 4) ? X[X1 + IC] : 0);
			float d3 = ((LY & 8) ? X[X1 + (IC << 1)] : 0);
			Ds[buf][tx][ty] = winograd_d(d0, d1, d2, d3);
			__syncthreads();
		}
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 g0 = Gs[buf][ik][tx], g1 = Gs[buf][ik][tx + BSIZE];
			float4 g2 = Gs[buf][ik + STEP][tx], g3 = Gs[buf][ik + STEP][tx + BSIZE];
			float4 g4 = Gs[buf][ik + STEP2][tx], g5 = Gs[buf][ik + STEP2][tx + BSIZE];
			float4 g6 = Gs[buf][ik + STEP3][tx], g7 = Gs[buf][ik + STEP3][tx + BSIZE];

			float4 d0 = Ds[buf][ik][ty];
			float4 d1 = Ds[buf][ik + STEP][ty];
			float4 d2 = Ds[buf][ik + STEP2][ty];
			float4 d3 = Ds[buf][ik + STEP3][ty];

			wino_grad4_GxW(v0, v2, g0, g1, g2, g3, d0);
			wino_grad4_GxW(v1, v3, g4, g5, g6, g7, d0);

			wino_grad4_GxW(v4, v6, g0, g1, g2, g3, d1);
			wino_grad4_GxW(v5, v7, g4, g5, g6, g7, d1);

			wino_grad4_GxW(v8, v10, g0, g1, g2, g3, d2);
			wino_grad4_GxW(v9, v11, g4, g5, g6, g7, d2);

			wino_grad4_GxW(v12, v14, g0, g1, g2, g3, d3);
			wino_grad4_GxW(v13, v15, g4, g5, g6, g7, d3);
		}
	}

	const int Y1 = Y0 + OC, Y2 = Y1 + OC, Y3 = Y2 + OC;
	const int Y4 = Y3 + OC, Y5 = Y4 + OC, Y6 = Y5 + OC, Y7 = Y6 + OC;

	*(float4*)(Y + Y0) = v0; *(float4*)(Y + Y0 + 4) = v1;
	*(float4*)(Y + Y1) = v2; *(float4*)(Y + Y1 + 4) = v3;
	*(float4*)(Y + Y2) = v4; *(float4*)(Y + Y2 + 4) = v5;
	*(float4*)(Y + Y3) = v6; *(float4*)(Y + Y3 + 4) = v7;
	*(float4*)(Y + Y4) = v8; *(float4*)(Y + Y4 + 4) = v9;
	*(float4*)(Y + Y5) = v10; *(float4*)(Y + Y5 + 4) = v11;
	*(float4*)(Y + Y6) = v12; *(float4*)(Y + Y6 + 4) = v13;
	*(float4*)(Y + Y7) = v14; *(float4*)(Y + Y7 + 4) = v15;
}

#endif