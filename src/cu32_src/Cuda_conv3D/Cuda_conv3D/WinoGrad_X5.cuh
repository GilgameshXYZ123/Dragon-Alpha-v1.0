

//OH % 4 == 0, OW % 4 == 0
//(1)
//ni = (j0 + i) / (OH * OW) = (8*x + i) / (16z) 
//n0 = n1 = ...n7
//(2)
//ihi = ((j0 + i) / (OH * OW)) % OW
//ihi = (8*u + i) / (4*x)
//ih0 = ih1 = ih2 = ih3
//ih4 = ih5 = ih6 = ih7
//(3) 
//iwi = (j0 + i) % OW
//iwi = (8*x + i) / (4 * x)
//iw0 = iw1 - 1 = iw2 - 2 = iw3 - 3
//iw4 = iw5 - 1 = iw6 - 2 = iw7 - 3
//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*4)
#ifndef CONV_3D_WINOGRAD_KERNEL_W3_D1
#define CONV_3D_WINOGRAD_KERNEL_W3_D1

#define wingrad_d1(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	conv3dWinoGrad_kernel_d1<LB, (1<<LB>>1), (1<<LB), (2<<LB), (3<<LB)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, IC, OC, ph, pw, oc_index, j_index)

//LB = 4: Size = 1.125, Time = 1.91113 msec, Performace = 1264.13 GFlop/s
template<int LB, int STEP, int STEP2, int STEP4, int STEP6>
__global__ void conv3dWinoGrad_kernel_d1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	__shared__ float4 Gs[1 << LB][(4 << LB) + 1];//{toc0, toc1}
	__shared__ float4 Ds[1 << LB][(2 << LB) + 1];//{tgroup0}

	//prepare for GN = OC
	int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]
	const int Wstride = IC * OC;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2), tj2 = tj0 + 2;
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	get_n_oh_ow(tj2, tn2, toh2, tow2);
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

		int tih2 = toh2 - ph + fh, tiw2 = tow2 - pw;
		bool lh2 = (tih2 >= 0) && (tih2 < IH);
		bool ly4 = lh2 && (tiw2 >= 0) && (tiw2 < IW);
		bool ly5 = lh2 && (tiw2 >= -1) && (tiw2 + 1 < IW);
		bool ly6 = lh2 && (tiw2 >= -2) && (tiw2 + 2 < IW);
		bool ly7 = lh2 && (tiw2 >= -3) && (tiw2 + 3 < IW);

		for (int oic = 0, OIC = (IC << 1 >> LB); oic < OIC; oic++)
		{
			//load 2 group from X
			int xic = ((oic - (tx >= STEP)) << LB >> 1) + tx;
			int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + xic;
			int X1 = X0 + IC, X2 = X1 + IC, X3 = X2 + IC;
			int X4 = ((tn2*IH + tih2)*IW + tiw2)*IC + xic;
			int X6 = X4 + (IC << 1), X7 = X6 + IC;
			float d0 = (ly0 ? X[X0] : 0);
			float d1 = (ly1 ? X[X1] : 0);
			float d2 = (ly2 ? X[X2] : 0);
			float d3 = (ly3 ? X[X3] : 0);
			float d6 = (ly6 ? X[X6] : 0);
			float d7 = (ly7 ? X[X7] : 0);
			Ds[tx][ty] = winograd_d(d0, d1, d2, d3);
			Ds[tx][ty + STEP2] = winograd_d(d2, d3, d6, d7);

			//load 4 group from CW
			int wic = ((oic - (ty >= STEP)) << LB >> 1) + ty;
			int W0 = ((fh * 3)*IC + wic)*OC;
			int W1 = W0 + Wstride, W2 = W1 + Wstride;
			float4 g0 = *(float4*)(CW + W0);
			float4 g1 = *(float4*)(CW + W1);
			float4 g2 = *(float4*)(CW + W2);
			Gs[ty][tx] = winograd_g(g0.x, g1.x, g2.x);
			Gs[ty][tx + STEP2] = winograd_g(g0.y, g1.y, g2.y);
			Gs[ty][tx + STEP4] = winograd_g(g0.z, g1.z, g2.z);
			Gs[ty][tx + STEP6] = winograd_g(g0.w, g1.w, g2.w);
			__syncthreads();

			for (int ik = 0; ik < STEP; ik++)
			{
				float4 d0 = Ds[ik][ty];
				float4 d1 = Ds[ik][ty + STEP2];
				float4 d2 = Ds[ik + STEP][ty];
				float4 d3 = Ds[ik + STEP][ty + STEP2];

				float4 g0 = Gs[ik][tx];
				float4 g1 = Gs[ik][tx + STEP2];
				float4 g2 = Gs[ik][tx + STEP4];
				float4 g3 = Gs[ik][tx + STEP6];
				float4 g4 = Gs[ik + STEP][tx];
				float4 g5 = Gs[ik + STEP][tx + STEP2];
				float4 g6 = Gs[ik + STEP][tx + STEP4];
				float4 g7 = Gs[ik + STEP][tx + STEP6];

				wino_grad4_GxW(v0, v2, g0, g1, g2, g3, d0);
				wino_grad4_GxW(v4, v6, g0, g1, g2, g3, d1);
				wino_grad4_GxW(v8, v10, g0, g1, g2, g3, d2);
				wino_grad4_GxW(v12, v14, g0, g1, g2, g3, d3);
				wino_grad4_GxW(v1, v3, g4, g5, g6, g7, d0);
				wino_grad4_GxW(v5, v7, g4, g5, g6, g7, d1);
				wino_grad4_GxW(v9, v11, g4, g5, g6, g7, d2);
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


//OH % 4 == 0, OW % 4 == 0
//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*4)
#ifndef CONV_3D_WINOGRAD_KERNEL_W3_D2
#define CONV_3D_WINOGRAD_KERNEL_W3_D2

#define wingrad_d2(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	conv3dWinoGrad_kernel_d2<LB, (1<<LB>>1), (1<<LB), (2<<LB), (3<<LB)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, IC, OC, ph, pw, oc_index, j_index)

//LB = 4: Size = 1.125, Time = 1.9005 msec, Performace = 1271.2 GFlop/s
template<int LB, int STEP, int STEP2, int STEP4, int STEP6>
__global__ void conv3dWinoGrad_kernel_d2(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	__shared__ float4 Gs[1 << LB][(4 << LB) + 1];//{toc0, toc1}
	__shared__ float4 Ds[1 << LB][(2 << LB) + 1];//{tgroup0}

	//prepare for GN = OC
	int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]
	const int Wstride = IC * OC;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2), tj2 = tj0 + 2;
	get_n_oh_ow(tj0, tn0, toh0, tow0);

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
		int tih0 = toh0 - ph + fh, tiw0 = tow0 - pw;//fw = {0, 1, 2}

		bool lh0 = (tih0 >= 0) && (tih0 < IH);
		bool ly0 = lh0 && (tiw0 >=  0) && (tiw0     < IW);
		bool ly1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);
		bool ly2 = lh0 && (tiw0 >= -2) && (tiw0 + 2 < IW);
		bool ly3 = lh0 && (tiw0 >= -3) && (tiw0 + 3 < IW);
		bool ly4 = lh0 && (tiw0 >= -4) && (tiw0 + 4 < IW);
		bool ly5 = lh0 && (tiw0 >= -5) && (tiw0 + 5 < IW);

		for (int oic = 0, OIC = (IC << 1 >> LB); oic < OIC; oic++)
		{
			//load 2 group from X
			int xic = ((oic - (tx >= STEP)) << LB >> 1) + tx;
			int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + xic;
			int X1 = X0 + IC, X2 = X1 + IC, X3 = X2 + IC;
			int X4 = X3 + IC, X5 = X4 + IC;
			float d0 = (ly0 ? X[X0] : 0);
			float d1 = (ly1 ? X[X1] : 0);
			float d2 = (ly2 ? X[X2] : 0);
			float d3 = (ly3 ? X[X3] : 0);
			float d4 = (ly4 ? X[X4] : 0);
			float d5 = (ly5 ? X[X5] : 0);
			Ds[tx][ty        ] = winograd_d(d0, d1, d2, d3);
			Ds[tx][ty + STEP2] = winograd_d(d2, d3, d4, d5);

			//load 4 group from CW
			int wic = ((oic - (ty >= STEP)) << LB >> 1) + ty;
			int W0 = ((fh * 3)*IC + wic)*OC;
			int W1 = W0 + Wstride, W2 = W1 + Wstride;
			float4 g0 = *(float4*)(CW + W0);
			float4 g1 = *(float4*)(CW + W1);
			float4 g2 = *(float4*)(CW + W2);
			Gs[ty][tx        ] = winograd_g(g0.x, g1.x, g2.x);
			Gs[ty][tx + STEP2] = winograd_g(g0.y, g1.y, g2.y);
			Gs[ty][tx + STEP4] = winograd_g(g0.z, g1.z, g2.z);
			Gs[ty][tx + STEP6] = winograd_g(g0.w, g1.w, g2.w);
			__syncthreads();

			for (int ik = 0; ik < STEP; ik++)
			{
				float4 d0 = Ds[ik][ty];
				float4 d1 = Ds[ik][ty + STEP2];
				float4 d2 = Ds[ik + STEP][ty];
				float4 d3 = Ds[ik + STEP][ty + STEP2];

				float4 g0 = Gs[ik][tx];
				float4 g1 = Gs[ik][tx + STEP2];
				float4 g2 = Gs[ik][tx + STEP4];
				float4 g3 = Gs[ik][tx + STEP6];
				float4 g4 = Gs[ik + STEP][tx];
				float4 g5 = Gs[ik + STEP][tx + STEP2];
				float4 g6 = Gs[ik + STEP][tx + STEP4];
				float4 g7 = Gs[ik + STEP][tx + STEP6];

				wino_grad4_GxW(v0, v2, g0, g1, g2, g3, d0);
				wino_grad4_GxW(v4, v6, g0, g1, g2, g3, d1);
				wino_grad4_GxW(v8, v10, g0, g1, g2, g3, d2);
				wino_grad4_GxW(v12, v14, g0, g1, g2, g3, d3);
				wino_grad4_GxW(v1, v3, g4, g5, g6, g7, d0);
				wino_grad4_GxW(v5, v7, g4, g5, g6, g7, d1);
				wino_grad4_GxW(v9, v11, g4, g5, g6, g7, d2);
				wino_grad4_GxW(v13, v15, g4, g5, g6, g7, d3);
			}
			__syncthreads();
		}
	}

	const int Y0 = j0 * OC + oc0;
	const int Y1 = Y0 + OC, Y2 = Y1 + OC, Y3 = Y2 + OC;
	const int Y4 = Y3 + OC, Y5 = Y4 + OC, Y6 = Y5 + OC, Y7 = Y6 + OC;

	*(float4*)(Y + Y0) = v0;  *(float4*)(Y + Y0 + 4) = v1;
	*(float4*)(Y + Y1) = v2;  *(float4*)(Y + Y1 + 4) = v3;
	*(float4*)(Y + Y2) = v4;  *(float4*)(Y + Y2 + 4) = v5;
	*(float4*)(Y + Y3) = v6;  *(float4*)(Y + Y3 + 4) = v7;
	*(float4*)(Y + Y4) = v8;  *(float4*)(Y + Y4 + 4) = v9;
	*(float4*)(Y + Y5) = v10; *(float4*)(Y + Y5 + 4) = v11;
	*(float4*)(Y + Y6) = v12; *(float4*)(Y + Y6 + 4) = v13;
	*(float4*)(Y + Y7) = v14; *(float4*)(Y + Y7 + 4) = v15;
}

#endif


//OH % 4 == 0, OW % 4 == 0
//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*4)
#ifndef CONV_3D_WINOGRAD_KERNEL_W3_D3
#define CONV_3D_WINOGRAD_KERNEL_W3_D3

#define wingrad_d3(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	conv3dWinoGrad_kernel_d3<LB, (1<<LB>>1), (1<<LB), (2<<LB), (3<<LB)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, IC, OC, ph, pw, oc_index, j_index)

//LB = 4: Size = 1.125, Time = 1.89169 msec, Performace = 1277.12 GFlop/s
template<int LB, int STEP, int STEP2, int STEP4, int STEP6>
__global__ void conv3dWinoGrad_kernel_d3(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	__shared__ float4 Gs[1 << LB][(4 << LB) + 1];//{toc0, toc1}
	__shared__ float4 Ds0[1 << LB][(1 << LB) + 1];//{d0 -> d3}
	__shared__ float2 Ds1[1 << LB][(1 << LB) + 1];//{d4 -> d5}

	//prepare for GN = OC
	int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]
	const int Wstride = IC * OC;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2), tj2 = tj0 + 2;
	get_n_oh_ow(tj0, tn0, toh0, tow0);

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
		int tih0 = toh0 - ph + fh, tiw0 = tow0 - pw;//fw = {0, 1, 2}

		bool lh0 = (tih0 >= 0) && (tih0 < IH);
		bool ly0 = lh0 && (tiw0 >= 0) && (tiw0 < IW);
		bool ly1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);
		bool ly2 = lh0 && (tiw0 >= -2) && (tiw0 + 2 < IW);
		bool ly3 = lh0 && (tiw0 >= -3) && (tiw0 + 3 < IW);
		bool ly4 = lh0 && (tiw0 >= -4) && (tiw0 + 4 < IW);
		bool ly5 = lh0 && (tiw0 >= -5) && (tiw0 + 5 < IW);

		for (int oic = 0, OIC = (IC << 1 >> LB); oic < OIC; oic++)
		{
			//load 2 group from X
			int xic = ((oic - (tx >= STEP)) << LB >> 1) + tx;
			int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + xic;
			int X1 = X0 + IC, X2 = X1 + IC, X3 = X2 + IC;
			int X4 = X3 + IC, X5 = X4 + IC;
			float d0 = (ly0 ? X[X0] : 0);
			float d1 = (ly1 ? X[X1] : 0);
			float d2 = (ly2 ? X[X2] : 0);
			float d3 = (ly3 ? X[X3] : 0);
			float d4 = (ly4 ? X[X4] : 0);
			float d5 = (ly5 ? X[X5] : 0);
			Ds0[tx][ty] = float4{ d0, d1, d2, d3 };
			Ds1[tx][ty] = float2{ d4, d5 };

			//load 4 group from CW
			int wic = ((oic - (ty >= STEP)) << LB >> 1) + ty;
			int W0 = ((fh * 3)*IC + wic)*OC;
			int W1 = W0 + Wstride, W2 = W1 + Wstride;
			float4 g0 = *(float4*)(CW + W0);
			float4 g1 = *(float4*)(CW + W1);
			float4 g2 = *(float4*)(CW + W2);
			Gs[ty][tx] = winograd_g(g0.x, g1.x, g2.x);
			Gs[ty][tx + STEP2] = winograd_g(g0.y, g1.y, g2.y);
			Gs[ty][tx + STEP4] = winograd_g(g0.z, g1.z, g2.z);
			Gs[ty][tx + STEP6] = winograd_g(g0.w, g1.w, g2.w);
			__syncthreads();

			for (int ik = 0; ik < STEP; ik++)
			{
				float4 b0 = Ds0[ik][ty], b2 = Ds0[ik + STEP][ty];
				float2 b1 = Ds1[ik][ty], b3 = Ds1[ik + STEP][ty];

				float4 d0 = winograd_d(b0.x, b0.y, b0.z, b0.w);//d0, d1, d2, d3
				float4 d1 = winograd_d(b0.z, b0.w, b1.x, b1.y);//d2, d3, d4, d5
				float4 d2 = winograd_d(b2.x, b2.y, b2.z, b2.w);
				float4 d3 = winograd_d(b2.z, b2.w, b3.x, b3.y);

				float4 g0 = Gs[ik][tx];
				float4 g1 = Gs[ik][tx + STEP2];
				float4 g2 = Gs[ik][tx + STEP4];
				float4 g3 = Gs[ik][tx + STEP6];
				float4 g4 = Gs[ik + STEP][tx];
				float4 g5 = Gs[ik + STEP][tx + STEP2];
				float4 g6 = Gs[ik + STEP][tx + STEP4];
				float4 g7 = Gs[ik + STEP][tx + STEP6];

				wino_grad4_GxW(v0, v2, g0, g1, g2, g3, d0);
				wino_grad4_GxW(v4, v6, g0, g1, g2, g3, d1);
				wino_grad4_GxW(v8, v10, g0, g1, g2, g3, d2);
				wino_grad4_GxW(v12, v14, g0, g1, g2, g3, d3);
				wino_grad4_GxW(v1, v3, g4, g5, g6, g7, d0);
				wino_grad4_GxW(v5, v7, g4, g5, g6, g7, d1);
				wino_grad4_GxW(v9, v11, g4, g5, g6, g7, d2);
				wino_grad4_GxW(v13, v15, g4, g5, g6, g7, d3);
			}
			__syncthreads();
		}
	}

	const int Y0 = j0 * OC + oc0;
	const int Y1 = Y0 + OC, Y2 = Y1 + OC, Y3 = Y2 + OC;
	const int Y4 = Y3 + OC, Y5 = Y4 + OC, Y6 = Y5 + OC, Y7 = Y6 + OC;

	*(float4*)(Y + Y0) = v0;  *(float4*)(Y + Y0 + 4) = v1;
	*(float4*)(Y + Y1) = v2;  *(float4*)(Y + Y1 + 4) = v3;
	*(float4*)(Y + Y2) = v4;  *(float4*)(Y + Y2 + 4) = v5;
	*(float4*)(Y + Y3) = v6;  *(float4*)(Y + Y3 + 4) = v7;
	*(float4*)(Y + Y4) = v8;  *(float4*)(Y + Y4 + 4) = v9;
	*(float4*)(Y + Y5) = v10; *(float4*)(Y + Y5 + 4) = v11;
	*(float4*)(Y + Y6) = v12; *(float4*)(Y + Y6 + 4) = v13;
	*(float4*)(Y + Y7) = v14; *(float4*)(Y + Y7 + 4) = v15;
}

#endif


//OH % 4 == 0, OW % 4 == 0
//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*4)
#ifndef CONV_3D_WINOGRAD_KERNEL_W3_D4
#define CONV_3D_WINOGRAD_KERNEL_W3_D4

#define wingrad_d4(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	conv3dWinoGrad_kernel_d4<LB, (1<<LB>>1), (1<<LB), (2<<LB), (3<<LB)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, IC, OC, ph, pw, oc_index, j_index)

//LB = 4: Size = 1.125, Time = 1.80971 msec, Performace = 1334.98 GFlop/s
template<int LB, int STEP, int STEP2, int STEP4, int STEP6>
__global__ void conv3dWinoGrad_kernel_d4(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	__shared__ float4 Gs[1 << LB][(4 << LB) + 1];
	__shared__ float4 Ds0[1 << LB][(1 << LB) + 1];//{d0 -> d3}
	__shared__ float2 Ds1[1 << LB][(1 << LB) + 1];//{d4 -> d5}

	//prepare for GN = OC
	int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]
	const int Wstride = IC * OC;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2), tj2 = tj0 + 2;
	get_n_oh_ow(tj0, tn0, toh0, tow0);

	//compute area-------------------------------------------------
	float4  v0 = float4{ 0, 0, 0, 0 }, v1 = float4{ 0, 0, 0, 0 };
	float4  v2 = float4{ 0, 0, 0, 0 }, v3 = float4{ 0, 0, 0, 0 };
	float4  v4 = float4{ 0, 0, 0, 0 }, v5 = float4{ 0, 0, 0, 0 };
	float4  v6 = float4{ 0, 0, 0, 0 }, v7 = float4{ 0, 0, 0, 0 };
	float4  v8 = float4{ 0, 0, 0, 0 }, v9 = float4{ 0, 0, 0, 0 };
	float4 v10 = float4{ 0, 0, 0, 0 }, v11 = float4{ 0, 0, 0, 0 };
	float4 v12 = float4{ 0, 0, 0, 0 }, v13 = float4{ 0, 0, 0, 0 };
	float4 v14 = float4{ 0, 0, 0, 0 }, v15 = float4{ 0, 0, 0, 0 };

//#pragma unroll
	for (int fh = 0; fh < 3; fh++)
	{
		int tih0 = toh0 - ph + fh, tiw0 = tow0 - pw;//fw = {0, 1, 2}

		bool lh0 = (tih0 >= 0) && (tih0 < IH);
		bool ly0 = lh0 && (tiw0 >= 0) && (tiw0 < IW);
		bool ly1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);
		bool ly2 = lh0 && (tiw0 >= -2) && (tiw0 + 2 < IW);
		bool ly3 = lh0 && (tiw0 >= -3) && (tiw0 + 3 < IW);
		bool ly4 = lh0 && (tiw0 >= -4) && (tiw0 + 4 < IW);
		bool ly5 = lh0 && (tiw0 >= -5) && (tiw0 + 5 < IW);

		for (int oic = 0, OIC = (IC << 1 >> LB); oic < OIC; oic++)
		{
			//load 2 group from X
			int xic = ((oic - (tx >= STEP)) << LB >> 1) + tx;
			int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + xic;
			int X1 = X0 + IC, X2 = X1 + IC, X3 = X2 + IC;
			int X4 = X3 + IC, X5 = X4 + IC;
			float d0 = (ly0 ? X[X0] : 0);
			float d1 = (ly1 ? X[X1] : 0);
			float d2 = (ly2 ? X[X2] : 0);
			float d3 = (ly3 ? X[X3] : 0);
			float d4 = (ly4 ? X[X4] : 0);
			float d5 = (ly5 ? X[X5] : 0);
			Ds0[tx][ty] = float4{ d0, d1, d2, d3 };
			Ds1[tx][ty] = float2{ d4, d5 };

			//load 4 group from CW
			int wic = ((oic - (ty >= STEP)) << LB >> 1) + ty;
			int W0 = ((fh * 3)*IC + wic)*OC;
			int W1 = W0 + Wstride, W2 = W1 + Wstride;
			float4 g0 = *(float4*)(CW + W0);
			float4 g1 = *(float4*)(CW + W1);
			float4 g2 = *(float4*)(CW + W2);
			Gs[ty][tx        ] = winograd_g(g0.x, g1.x, g2.x);
			Gs[ty][tx + STEP2] = winograd_g(g0.y, g1.y, g2.y);
			Gs[ty][tx + STEP4] = winograd_g(g0.z, g1.z, g2.z);
			Gs[ty][tx + STEP6] = winograd_g(g0.w, g1.w, g2.w);
			__syncthreads();

#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 b0 = Ds0[ik][ty], b2 = Ds0[ik + STEP][ty];
				float2 b1 = Ds1[ik][ty], b3 = Ds1[ik + STEP][ty];

				float4 d0 = winograd_d(b0.x, b0.y, b0.z, b0.w);//d0, d1, d2, d3
				float4 d1 = winograd_d(b0.z, b0.w, b1.x, b1.y);//d2, d3, d4, d5
				float4 d2 = winograd_d(b2.x, b2.y, b2.z, b2.w);
				float4 d3 = winograd_d(b2.z, b2.w, b3.x, b3.y);

				float4 g0 = Gs[ik][tx];
				float4 g1 = Gs[ik][tx + STEP2];
				float4 g2 = Gs[ik][tx + STEP4];
				float4 g3 = Gs[ik][tx + STEP6];
				float4 g4 = Gs[ik + STEP][tx];
				float4 g5 = Gs[ik + STEP][tx + STEP2];
				float4 g6 = Gs[ik + STEP][tx + STEP4];
				float4 g7 = Gs[ik + STEP][tx + STEP6];

				wino_grad4_GxW(v0, v2, g0, g1, g2, g3, d0);
				wino_grad4_GxW(v4, v6, g0, g1, g2, g3, d1);
				wino_grad4_GxW(v8, v10, g0, g1, g2, g3, d2);
				wino_grad4_GxW(v12, v14, g0, g1, g2, g3, d3);
				wino_grad4_GxW(v1, v3, g4, g5, g6, g7, d0);
				wino_grad4_GxW(v5, v7, g4, g5, g6, g7, d1);
				wino_grad4_GxW(v9, v11, g4, g5, g6, g7, d2);
				wino_grad4_GxW(v13, v15, g4, g5, g6, g7, d3);
			}
			__syncthreads();
		}
	}

	const int Y0 = j0 * OC + oc0;
	const int Y1 = Y0 + OC, Y2 = Y1 + OC, Y3 = Y2 + OC;
	const int Y4 = Y3 + OC, Y5 = Y4 + OC, Y6 = Y5 + OC, Y7 = Y6 + OC;

	*(float4*)(Y + Y0) = v0;  *(float4*)(Y + Y0 + 4) = v1;
	*(float4*)(Y + Y1) = v2;  *(float4*)(Y + Y1 + 4) = v3;
	*(float4*)(Y + Y2) = v4;  *(float4*)(Y + Y2 + 4) = v5;
	*(float4*)(Y + Y3) = v6;  *(float4*)(Y + Y3 + 4) = v7;
	*(float4*)(Y + Y4) = v8;  *(float4*)(Y + Y4 + 4) = v9;
	*(float4*)(Y + Y5) = v10; *(float4*)(Y + Y5 + 4) = v11;
	*(float4*)(Y + Y6) = v12; *(float4*)(Y + Y6 + 4) = v13;
	*(float4*)(Y + Y7) = v14; *(float4*)(Y + Y7 + 4) = v15;
}

#endif


//OH % 4 == 0, OW % 4 == 0
//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*4)
#ifndef CONV_3D_WINOGRAD_KERNEL_W3_D5
#define CONV_3D_WINOGRAD_KERNEL_W3_D5

#define wingrad_d5(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	conv3dWinoGrad_kernel_d5<LB, (1<<LB>>1), (1<<LB), (2<<LB), (3<<LB)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, IC, OC, ph, pw, oc_index, j_index)

//LB = 4: Size = 1.125, Time = 1.81274 msec, Performace = 1332.74 GFlop/s
template<int LB, int STEP, int STEP2, int STEP4, int STEP6>
__global__ void conv3dWinoGrad_kernel_d5(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	__shared__ float4  Gs[1 << LB][(4 << LB) + 1];
	__shared__ float4 Ds0[1 << LB][(1 << LB) + 1];//{d0 -> d3}
	__shared__ float2 Ds1[1 << LB][(1 << LB) + 1];//{d4 -> d5}

	//prepare for GN = OC
	int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]
	const int Wstride = IC * OC;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2), tj2 = tj0 + 2;
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	int tih0 = toh0 - ph, tiw0 = tow0 - pw;//fw = {0, 1, 2}

	//compute area-------------------------------------------------
	float4  v0 = float4{ 0, 0, 0, 0 }, v1 = float4{ 0, 0, 0, 0 };
	float4  v2 = float4{ 0, 0, 0, 0 }, v3 = float4{ 0, 0, 0, 0 };
	float4  v4 = float4{ 0, 0, 0, 0 }, v5 = float4{ 0, 0, 0, 0 };
	float4  v6 = float4{ 0, 0, 0, 0 }, v7 = float4{ 0, 0, 0, 0 };
	float4  v8 = float4{ 0, 0, 0, 0 }, v9 = float4{ 0, 0, 0, 0 };
	float4 v10 = float4{ 0, 0, 0, 0 }, v11 = float4{ 0, 0, 0, 0 };
	float4 v12 = float4{ 0, 0, 0, 0 }, v13 = float4{ 0, 0, 0, 0 };
	float4 v14 = float4{ 0, 0, 0, 0 }, v15 = float4{ 0, 0, 0, 0 };

#pragma unnroll
	for (int fh = 0; fh < 3; fh++, tih0++)
	{
		for (int oic = 0, OIC = (IC << 1 >> LB); oic < OIC; oic++)
		{
			//load 2 group from X
			int xic = ((oic - (tx >= STEP)) << LB >> 1) + tx;
			int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + xic;
			int X1 = X0 + IC, X2 = X1 + IC, X3 = X2 + IC;
			int X4 = X3 + IC, X5 = X4 + IC;
			
			bool lh0 = (tih0 >= 0) && (tih0 < IH);
			bool ly0 = lh0 && (tiw0 >= 0) && (tiw0 < IW);
			bool ly1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);
			bool ly2 = lh0 && (tiw0 >= -2) && (tiw0 + 2 < IW);
			bool ly3 = lh0 && (tiw0 >= -3) && (tiw0 + 3 < IW);
			bool ly4 = lh0 && (tiw0 >= -4) && (tiw0 + 4 < IW);
			bool ly5 = lh0 && (tiw0 >= -5) && (tiw0 + 5 < IW);

			float d0 = (ly0 ? X[X0] : 0);
			float d1 = (ly1 ? X[X1] : 0);
			float d2 = (ly2 ? X[X2] : 0);
			float d3 = (ly3 ? X[X3] : 0);
			float d4 = (ly4 ? X[X4] : 0);
			float d5 = (ly5 ? X[X5] : 0);
			Ds0[tx][ty] = float4{ d0, d1, d2, d3 };
			Ds1[tx][ty] = float2{ d4, d5 };

			//load 4 group from CW
			int wic = ((oic - (ty >= STEP)) << LB >> 1) + ty;
			int W0 = ((fh * 3)*IC + wic)*OC;
			int W1 = W0 + Wstride, W2 = W1 + Wstride;
			float4 g0 = *(float4*)(CW + W0);
			float4 g1 = *(float4*)(CW + W1);
			float4 g2 = *(float4*)(CW + W2);
			WinoGrad_produce_G(Gs[ty][tx], g0.x, g1.x, g2.x);
			WinoGrad_produce_G(Gs[ty][tx + STEP2], g0.y, g1.y, g2.y);
			WinoGrad_produce_G(Gs[ty][tx + STEP4], g0.z, g1.z, g2.z);
			WinoGrad_produce_G(Gs[ty][tx + STEP6], g0.w, g1.w, g2.w);
			__syncthreads();

#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 b0 = Ds0[ik][ty], b2 = Ds0[ik + STEP][ty];
				float2 b1 = Ds1[ik][ty], b3 = Ds1[ik + STEP][ty];

				float4 d0 = winograd_d(b0.x, b0.y, b0.z, b0.w);//d0, d1, d2, d3
				float4 d1 = winograd_d(b0.z, b0.w, b1.x, b1.y);//d2, d3, d4, d5
				float4 d2 = winograd_d(b2.x, b2.y, b2.z, b2.w);
				float4 d3 = winograd_d(b2.z, b2.w, b3.x, b3.y);

				float4 g0 = Gs[ik][tx];
				float4 g1 = Gs[ik][tx + STEP2];
				float4 g2 = Gs[ik][tx + STEP4];
				float4 g3 = Gs[ik][tx + STEP6];
				float4 g4 = Gs[ik + STEP][tx];
				float4 g5 = Gs[ik + STEP][tx + STEP2];
				float4 g6 = Gs[ik + STEP][tx + STEP4];
				float4 g7 = Gs[ik + STEP][tx + STEP6];

				wino_grad4_GxW(v0, v2, g0, g1, g2, g3, d0);
				wino_grad4_GxW(v4, v6, g0, g1, g2, g3, d1);
				wino_grad4_GxW(v8, v10, g0, g1, g2, g3, d2);
				wino_grad4_GxW(v12, v14, g0, g1, g2, g3, d3);
				wino_grad4_GxW(v1, v3, g4, g5, g6, g7, d0);
				wino_grad4_GxW(v5, v7, g4, g5, g6, g7, d1);
				wino_grad4_GxW(v9, v11, g4, g5, g6, g7, d2);
				wino_grad4_GxW(v13, v15, g4, g5, g6, g7, d3);
			}
			__syncthreads();
		}
	}

	const int Y0 = j0 * OC + oc0;
	const int Y1 = Y0 + OC, Y2 = Y1 + OC, Y3 = Y2 + OC;
	const int Y4 = Y3 + OC, Y5 = Y4 + OC, Y6 = Y5 + OC, Y7 = Y6 + OC;

	*(float4*)(Y + Y0) = v0;  *(float4*)(Y + Y0 + 4) = v1;
	*(float4*)(Y + Y1) = v2;  *(float4*)(Y + Y1 + 4) = v3;
	*(float4*)(Y + Y2) = v4;  *(float4*)(Y + Y2 + 4) = v5;
	*(float4*)(Y + Y3) = v6;  *(float4*)(Y + Y3 + 4) = v7;
	*(float4*)(Y + Y4) = v8;  *(float4*)(Y + Y4 + 4) = v9;
	*(float4*)(Y + Y5) = v10; *(float4*)(Y + Y5 + 4) = v11;
	*(float4*)(Y + Y6) = v12; *(float4*)(Y + Y6 + 4) = v13;
	*(float4*)(Y + Y7) = v14; *(float4*)(Y + Y7 + 4) = v15;
}

#endif


//OH % 4 == 0, OW % 4 == 0
//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*4)
#ifndef CONV_3D_WINOGRAD_KERNEL_W3_D6
#define CONV_3D_WINOGRAD_KERNEL_W3_D6

#define wingrad_d6(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	conv3dWinoGrad_kernel_d6<LB, (1<<LB>>1), (1<<LB), (2<<LB), (3<<LB)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, IC, OC, ph, pw, oc_index, j_index)

//LB = 4: Size = 1.125, Time = 1.81274 msec, Performace = 1332.74 GFlop/s
template<int LB, int STEP, int STEP2, int STEP4, int STEP6>
__global__ void conv3dWinoGrad_kernel_d6(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	__shared__ float4  Gs[1 << LB][(4 << LB) + 1];
	__shared__ float4 Ds0[1 << LB][(1 << LB) + 1];//{d0 -> d3}
	__shared__ float2 Ds1[1 << LB][(1 << LB) + 1];//{d4 -> d5}

	//prepare for GN = OC
	int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]
	const int Wstride = IC * OC;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2), tj2 = tj0 + 2;
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	int tih0 = toh0 - ph, tiw0 = tow0 - pw;//fw = {0, 1, 2}

	//compute area-------------------------------------------------
	float4  v0 = float4{ 0, 0, 0, 0 }, v1 = float4{ 0, 0, 0, 0 };
	float4  v2 = float4{ 0, 0, 0, 0 }, v3 = float4{ 0, 0, 0, 0 };
	float4  v4 = float4{ 0, 0, 0, 0 }, v5 = float4{ 0, 0, 0, 0 };
	float4  v6 = float4{ 0, 0, 0, 0 }, v7 = float4{ 0, 0, 0, 0 };
	float4  v8 = float4{ 0, 0, 0, 0 }, v9 = float4{ 0, 0, 0, 0 };
	float4 v10 = float4{ 0, 0, 0, 0 }, v11 = float4{ 0, 0, 0, 0 };
	float4 v12 = float4{ 0, 0, 0, 0 }, v13 = float4{ 0, 0, 0, 0 };
	float4 v14 = float4{ 0, 0, 0, 0 }, v15 = float4{ 0, 0, 0, 0 };

#pragma unnroll
	for (int fh = 0; fh < 3; fh++, tih0++)
	{
		for (int oic = 0, OIC = (IC << 1 >> LB); oic < OIC; oic++)
		{
			//load 2 group from X
			int xic = ((oic - (tx >= STEP)) << LB >> 1) + tx;
			int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + xic;
			int X1 = X0 + IC, X2 = X1 + IC;
			int X3 = X2 + IC, X4 = X3 + IC, X5 = X4 + IC;
			bool lh0 = (tih0 >= 0) && (tih0 < IH);
			bool ly0 = lh0 && (tiw0 >= 0) && (tiw0 < IW);
			bool ly1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);
			bool ly2 = lh0 && (tiw0 >= -2) && (tiw0 + 2 < IW);
			bool ly3 = lh0 && (tiw0 >= -3) && (tiw0 + 3 < IW);
			bool ly4 = lh0 && (tiw0 >= -4) && (tiw0 + 4 < IW);
			bool ly5 = lh0 && (tiw0 >= -5) && (tiw0 + 5 < IW);
			float d0 = (ly0 ? X[X0] : 0);
			float d1 = (ly1 ? X[X1] : 0);
			float d2 = (ly2 ? X[X2] : 0);
			float d3 = (ly3 ? X[X3] : 0);
			float d4 = (ly4 ? X[X4] : 0);
			float d5 = (ly5 ? X[X5] : 0);
			Ds0[tx][ty] = float4{ d0, d1, d2, d3 };
			Ds1[tx][ty] = float2{ d4, d5 };

			//load 4 group from CW
			int wic = ((oic - (ty >= STEP)) << LB >> 1) + ty;
			int W0 = ((fh * 3)*IC + wic)*OC;
			int W1 = W0 + Wstride, W2 = W1 + Wstride;
			float4 g0 = *(float4*)(CW + W0);
			float4 g1 = *(float4*)(CW + W1);
			float4 g2 = *(float4*)(CW + W2);
			WinoGrad_produce_G(Gs[ty][tx], g0.x, g1.x, g2.x);
			WinoGrad_produce_G(Gs[ty][tx + STEP2], g0.y, g1.y, g2.y);
			WinoGrad_produce_G(Gs[ty][tx + STEP4], g0.z, g1.z, g2.z);
			WinoGrad_produce_G(Gs[ty][tx + STEP6], g0.w, g1.w, g2.w);
			__syncthreads();

#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 b0 = Ds0[ik][ty], b2 = Ds0[ik + STEP][ty];
				float2 b1 = Ds1[ik][ty], b3 = Ds1[ik + STEP][ty];

				float4 d0 = winograd_d(b0.x, b0.y, b0.z, b0.w);//d0, d1, d2, d3
				float4 d1 = winograd_d(b0.z, b0.w, b1.x, b1.y);//d2, d3, d4, d5
				float4 d2 = winograd_d(b2.x, b2.y, b2.z, b2.w);
				float4 d3 = winograd_d(b2.z, b2.w, b3.x, b3.y);

				float4 g0 = Gs[ik][tx];
				float4 g1 = Gs[ik][tx + STEP2];
				float4 g2 = Gs[ik][tx + STEP4];
				float4 g3 = Gs[ik][tx + STEP6];
				float4 g4 = Gs[ik + STEP][tx];
				float4 g5 = Gs[ik + STEP][tx + STEP2];
				float4 g6 = Gs[ik + STEP][tx + STEP4];
				float4 g7 = Gs[ik + STEP][tx + STEP6];

				wino_grad4_GxW(v0, v2, g0, g1, g2, g3, d0);
				wino_grad4_GxW(v4, v6, g0, g1, g2, g3, d1);
				wino_grad4_GxW(v8, v10, g0, g1, g2, g3, d2);
				wino_grad4_GxW(v12, v14, g0, g1, g2, g3, d3);
				wino_grad4_GxW(v1, v3, g4, g5, g6, g7, d0);
				wino_grad4_GxW(v5, v7, g4, g5, g6, g7, d1);
				wino_grad4_GxW(v9, v11, g4, g5, g6, g7, d2);
				wino_grad4_GxW(v13, v15, g4, g5, g6, g7, d3);
			}
			__syncthreads();
		}
	}

	const int Y0 = j0 * OC + oc0;
	const int Y1 = Y0 + OC, Y2 = Y1 + OC, Y3 = Y2 + OC;
	const int Y4 = Y3 + OC, Y5 = Y4 + OC, Y6 = Y5 + OC, Y7 = Y6 + OC;

	*(float4*)(Y + Y0) = v0;  *(float4*)(Y + Y0 + 4) = v1;
	*(float4*)(Y + Y1) = v2;  *(float4*)(Y + Y1 + 4) = v3;
	*(float4*)(Y + Y2) = v4;  *(float4*)(Y + Y2 + 4) = v5;
	*(float4*)(Y + Y3) = v6;  *(float4*)(Y + Y3 + 4) = v7;
	*(float4*)(Y + Y4) = v8;  *(float4*)(Y + Y4 + 4) = v9;
	*(float4*)(Y + Y5) = v10; *(float4*)(Y + Y5 + 4) = v11;
	*(float4*)(Y + Y6) = v12; *(float4*)(Y + Y6 + 4) = v13;
	*(float4*)(Y + Y7) = v14; *(float4*)(Y + Y7 + 4) = v15;
}

#endif



//OH % 4 == 0, OW % 4 == 0
//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*4)
#ifndef CONV_3D_WINOGRAD_KERNEL_W3_D7
#define CONV_3D_WINOGRAD_KERNEL_W3_D7

#define wingrad_d7(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	conv3dWinoGrad_kernel_d7<LB, (1<<LB>>1), (1<<LB), (2<<LB), (3<<LB)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, IC, OC, ph, pw, oc_index, j_index)

//LB = 4: Size = 1.125, Time = 1.81274 msec, Performace = 1332.74 GFlop/s
template<int LB, int STEP, int STEP2, int STEP4, int STEP6>
__global__ void conv3dWinoGrad_kernel_d7(
	cudaTextureObject_t X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	__shared__ float4  Gs[1 << LB][(4 << LB) + 1];
	__shared__ float4 Ds0[1 << LB][(1 << LB) + 1];//{d0 -> d3}
	__shared__ float2 Ds1[1 << LB][(1 << LB) + 1];//{d4 -> d5}

	//prepare for GN = OC
	int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]
	const int Wstride = IC * OC;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 2), tj2 = tj0 + 2;
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	int tih0 = toh0 - ph, tiw0 = tow0 - pw;//fw = {0, 1, 2}

	//compute area-------------------------------------------------
	float4  v0 = float4{ 0, 0, 0, 0 }, v1 = float4{ 0, 0, 0, 0 };
	float4  v2 = float4{ 0, 0, 0, 0 }, v3 = float4{ 0, 0, 0, 0 };
	float4  v4 = float4{ 0, 0, 0, 0 }, v5 = float4{ 0, 0, 0, 0 };
	float4  v6 = float4{ 0, 0, 0, 0 }, v7 = float4{ 0, 0, 0, 0 };
	float4  v8 = float4{ 0, 0, 0, 0 }, v9 = float4{ 0, 0, 0, 0 };
	float4 v10 = float4{ 0, 0, 0, 0 }, v11 = float4{ 0, 0, 0, 0 };
	float4 v12 = float4{ 0, 0, 0, 0 }, v13 = float4{ 0, 0, 0, 0 };
	float4 v14 = float4{ 0, 0, 0, 0 }, v15 = float4{ 0, 0, 0, 0 };

#pragma unnroll
	for (int fh = 0; fh < 3; fh++, tih0++)
	{
		for (int oic = 0, OIC = (IC << 1 >> LB); oic < OIC; oic++)
		{
			//load 2 group from X
			int xic = ((oic - (tx >= STEP)) << LB >> 1) + tx;
			int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + xic;
			int X1 = X0 + IC, X2 = X1 + IC;
			int X3 = X2 + IC, X4 = X3 + IC, X5 = X4 + IC;
			bool lh0 = (tih0 >= 0) && (tih0 < IH);
			bool ly0 = lh0 && (tiw0 >= 0) && (tiw0 < IW);
			bool ly1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);
			bool ly2 = lh0 && (tiw0 >= -2) && (tiw0 + 2 < IW);
			bool ly3 = lh0 && (tiw0 >= -3) && (tiw0 + 3 < IW);
			bool ly4 = lh0 && (tiw0 >= -4) && (tiw0 + 4 < IW);
			bool ly5 = lh0 && (tiw0 >= -5) && (tiw0 + 5 < IW);
			float4 dsv0; float2 dsv1;
			zero_float(dsv0.x, ly0, tex1Dfetch<float>(X, X0));
			zero_float(dsv0.y, ly1, tex1Dfetch<float>(X, X1));
			zero_float(dsv0.z, ly2, tex1Dfetch<float>(X, X2));
			zero_float(dsv0.w, ly3, tex1Dfetch<float>(X, X3));
			zero_float(dsv1.x, ly4, tex1Dfetch<float>(X, X4));
			zero_float(dsv1.y, ly5, tex1Dfetch<float>(X, X5));
			Ds0[tx][ty] = dsv0;
			Ds1[tx][ty] = dsv1;

			//load 4 group from CW
			int wic = ((oic - (ty >= STEP)) << LB >> 1) + ty;
			int W0 = ((fh * 3)*IC + wic)*OC;
			int W1 = W0 + Wstride, W2 = W1 + Wstride;
			float4 g0 = *(float4*)(CW + W0);
			float4 g1 = *(float4*)(CW + W1);
			float4 g2 = *(float4*)(CW + W2);
			WinoGrad_produce_G(Gs[ty][tx], g0.x, g1.x, g2.x);
			WinoGrad_produce_G(Gs[ty][tx + STEP2], g0.y, g1.y, g2.y);
			WinoGrad_produce_G(Gs[ty][tx + STEP4], g0.z, g1.z, g2.z);
			WinoGrad_produce_G(Gs[ty][tx + STEP6], g0.w, g1.w, g2.w);
			__syncthreads();

#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 b0 = Ds0[ik][ty], b2 = Ds0[ik + STEP][ty];
				float2 b1 = Ds1[ik][ty], b3 = Ds1[ik + STEP][ty];

				float4 d0 = winograd_d(b0.x, b0.y, b0.z, b0.w);//d0, d1, d2, d3
				float4 d1 = winograd_d(b0.z, b0.w, b1.x, b1.y);//d2, d3, d4, d5
				float4 d2 = winograd_d(b2.x, b2.y, b2.z, b2.w);
				float4 d3 = winograd_d(b2.z, b2.w, b3.x, b3.y);

				float4 g0 = Gs[ik][tx];
				float4 g1 = Gs[ik][tx + STEP2];
				float4 g2 = Gs[ik][tx + STEP4];
				float4 g3 = Gs[ik][tx + STEP6];
				float4 g4 = Gs[ik + STEP][tx];
				float4 g5 = Gs[ik + STEP][tx + STEP2];
				float4 g6 = Gs[ik + STEP][tx + STEP4];
				float4 g7 = Gs[ik + STEP][tx + STEP6];

				wino_grad4_GxW(v0, v2, g0, g1, g2, g3, d0);
				wino_grad4_GxW(v4, v6, g0, g1, g2, g3, d1);
				wino_grad4_GxW(v8, v10, g0, g1, g2, g3, d2);
				wino_grad4_GxW(v12, v14, g0, g1, g2, g3, d3);
				wino_grad4_GxW(v1, v3, g4, g5, g6, g7, d0);
				wino_grad4_GxW(v5, v7, g4, g5, g6, g7, d1);
				wino_grad4_GxW(v9, v11, g4, g5, g6, g7, d2);
				wino_grad4_GxW(v13, v15, g4, g5, g6, g7, d3);
			}
			__syncthreads();
		}
	}

	const int Y0 = j0 * OC + oc0;
	const int Y1 = Y0 + OC, Y2 = Y1 + OC, Y3 = Y2 + OC;
	const int Y4 = Y3 + OC, Y5 = Y4 + OC, Y6 = Y5 + OC, Y7 = Y6 + OC;

	*(float4*)(Y + Y0) = v0;  *(float4*)(Y + Y0 + 4) = v1;
	*(float4*)(Y + Y1) = v2;  *(float4*)(Y + Y1 + 4) = v3;
	*(float4*)(Y + Y2) = v4;  *(float4*)(Y + Y2 + 4) = v5;
	*(float4*)(Y + Y3) = v6;  *(float4*)(Y + Y3 + 4) = v7;
	*(float4*)(Y + Y4) = v8;  *(float4*)(Y + Y4 + 4) = v9;
	*(float4*)(Y + Y5) = v10; *(float4*)(Y + Y5 + 4) = v11;
	*(float4*)(Y + Y6) = v12; *(float4*)(Y + Y6 + 4) = v13;
	*(float4*)(Y + Y7) = v14; *(float4*)(Y + Y7 + 4) = v15;
}

#endif

