
//OH % 2 == 0, OW % 2 == 0
//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*4)
#ifndef CONV_3D_WINOGRAD_KERNEL_W3_B1
#define CONV_3D_WINOGRAD_KERNEL_W3_B1

#define wingrad_b1(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	conv3dWinoGrad_kernel_b1<LB, (1<<LB>>1), (1<<LB)>\
		<<< dim3(GN>>LB>>2, GM>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, IC, OC, ph, pw, oc_index, j_index)

//LB = 4: Size = 1.125, Time = 2.36042 msec, Performace = 1023.51 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void conv3dWinoGrad_kernel_b1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	__shared__ float4 Gs[1 << LB][(2 << LB) + 1];//{toc0, toc1}
	__shared__ float4 Ds[1 << LB][(1 << LB) + 1];//{tgroup0}

	//prepare for GN = OC
	int oc0 = (((blockIdx.x << LB) + tx) << 2) + oc_index;
	CW += oc0 + ((ty >= STEP) << 1);//CW[0, 0, 0, oc0]
	const int Wstride = IC * OC;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 2) + j_index, j2 = j0 + 2;
	int tj0 = j0 + ((tx >= STEP) << 1);
	get_n_oh_ow(tj0, tn0, toh0, tow0);

	//compute area---------------------------------------
	float4 v0 = float4{ 0, 0, 0, 0 };
	float4 v1 = float4{ 0, 0, 0, 0 };
	float4 v2 = float4{ 0, 0, 0, 0 };
	float4 v3 = float4{ 0, 0, 0, 0 };

#pragma unroll
	for (int fh = 0; fh < 3; fh++)
	{
		int tih0 = toh0 - ph + fh, tiw0 = tow0 - pw;
		bool lh0 = (tih0 >= 0) && (tih0 < IH);
		bool ly0 = lh0 && (tiw0 >= 0) && (tiw0 < IW);
		bool ly1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);
		bool ly2 = lh0 && (tiw0 >= -2) && (tiw0 + 2 < IW);
		bool ly3 = lh0 && (tiw0 >= -3) && (tiw0 + 3 < IW);

		for (int oic = 0, OIC = (IC << 1 >> LB); oic < OIC; oic++)
		{
			//load 2 group from CW
			int wic = ((oic - (ty >= STEP)) << LB >> 1) + ty;
			int W0 = ((fh * 3)*IC + wic)*OC;
			int W1 = W0 + Wstride, W2 = W1 + Wstride;
			float2 g0 = *(float2*)(CW + W0);
			float2 g1 = *(float2*)(CW + W1);
			float2 g2 = *(float2*)(CW + W2);
			Gs[ty][tx] = winograd_g(g0.x, g1.x, g2.x);
			Gs[ty][tx + STEP2] = winograd_g(g0.y, g1.y, g2.y);

			//load 1 group from X
			int xic = ((oic - (tx >= STEP)) << LB >> 1) + tx;
			int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + xic;
			int X1 = X0 + IC, X2 = X1 + IC, X3 = X2 + IC;
			float d0 = (ly0 ? X[X0] : 0);
			float d1 = (ly1 ? X[X1] : 0);
			float d2 = (ly2 ? X[X2] : 0);
			float d3 = (ly3 ? X[X3] : 0);
			Ds[tx][ty] = winograd_d(d0, d1, d2, d3);
			__syncthreads();

#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 d0 = Ds[ik][ty];
				float4 d1 = Ds[ik + STEP][ty];

				float4 g0 = Gs[ik][tx];
				float4 g1 = Gs[ik][tx + STEP2];
				float4 g2 = Gs[ik + STEP][tx];
				float4 g3 = Gs[ik + STEP][tx + STEP2];

				wino_grad4_GxW(v0, v1, g0, g1, g2, g3, d0);//d0 * {g0, g1, g2, g3}
				wino_grad4_GxW(v2, v3, g0, g1, g2, g3, d1);//d1 * {g0, g1, g2, g3}
			}
			__syncthreads();
		}
	}

	const int Y0 = j0 * OC + oc0;
	const int Y1 = Y0 + OC;
	const int Y2 = Y1 + OC;
	const int Y3 = Y2 + OC;

	*(float4*)(Y + Y0) = v0;
	*(float4*)(Y + Y1) = v1;
	*(float4*)(Y + Y2) = v2;
	*(float4*)(Y + Y3) = v3;
}

#endif


//OH % 2 == 0, OW % 2 == 0
//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*4)
#ifndef CONV_3D_WINOGRAD_KERNEL_W3_B2
#define CONV_3D_WINOGRAD_KERNEL_W3_B2

#define wingrad_b2(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	conv3dWinoGrad_kernel_b2<LB, (1<<LB>>1), (1<<LB)>\
		<<< dim3(GN>>LB>>2, GM>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, IC, OC, ph, pw, oc_index, j_index)

//LB = 4: Size = 1.125, Time = 2.36042 msec, Performace = 1023.51 GFlop/s
//LB = 4: Size = 1.125, Time = 4.0287 msec, Performace = 599.677 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void conv3dWinoGrad_kernel_b2(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Gs[2][1 << LB][(2 << LB) + 1];//{toc0, toc1}
	__shared__ float4 Ds[2][1 << LB][(1 << LB) + 1];//{tgroup0}

	//prepare for GN = OC
	int oc0 = (((blockIdx.x << LB) + tx) << 2) + oc_index;
	CW += oc0 + ((ty >= STEP) << 1);//CW[0, 0, 0, oc0]
	const int Wstride = IC * OC;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 2) + j_index, j2 = j0 + 2;
	int tj0 = j0 + ((tx >= STEP) << 1);
	get_n_oh_ow(tj0, tn0, toh0, tow0);

	//compute area---------------------------------------
	float4 v0 = float4{ 0, 0, 0, 0 };
	float4 v1 = float4{ 0, 0, 0, 0 };
	float4 v2 = float4{ 0, 0, 0, 0 };
	float4 v3 = float4{ 0, 0, 0, 0 };

	for (int fh = 0; fh < 3; fh++)
	{
		//load 2 group from CW
		int wic = ty - ((ty >= STEP) << LB >> 1);
		int W0 = ((fh * 3)*IC + wic)*OC;
		int W1 = W0 + Wstride, W2 = W1 + Wstride;
		float2 g0 = *(float2*)(CW + W0);
		float2 g1 = *(float2*)(CW + W1);
		float2 g2 = *(float2*)(CW + W2);
		Gs[buf][ty][tx] = winograd_g(g0.x, g1.x, g2.x);
		Gs[buf][ty][tx + STEP2] = winograd_g(g0.y, g1.y, g2.y);

		//load 1 group from X
		int tih0 = toh0 - ph + fh, tiw0 = tow0 - pw;
		bool lh0 = (tih0 >= 0) && (tih0 < IH);
		bool ly0 = lh0 && (tiw0 >= 0) && (tiw0 < IW);
		bool ly1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);
		bool ly2 = lh0 && (tiw0 >= -2) && (tiw0 + 2 < IW);
		bool ly3 = lh0 && (tiw0 >= -3) && (tiw0 + 3 < IW);

		int xic = tx - ((tx >= STEP) << LB >> 1);
		int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + xic;
		int X1 = X0 + IC, X2 = X1 + IC, X3 = X2 + IC;
		float d0 = (ly0 ? X[X0] : 0);
		float d1 = (ly1 ? X[X1] : 0);
		float d2 = (ly2 ? X[X2] : 0);
		float d3 = (ly3 ? X[X3] : 0);
		Ds[buf][tx][ty] = winograd_d(d0, d1, d2, d3);
		__syncthreads();

		for (int oic = 1, OIC = (IC << 1 >> LB); oic < OIC; oic++)
		{
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 g0 = Gs[buf][ik][tx];
				float4 g1 = Gs[buf][ik][tx + STEP2];
				float4 g2 = Gs[buf][ik + STEP][tx];
				float4 g3 = Gs[buf][ik + STEP][tx + STEP2];

				float4 d0 = Ds[buf][ik][ty];
				float4 d1 = Ds[buf][ik + STEP][ty];

				wino_grad4_GxW(v0, v1, g0, g1, g2, g3, d0);//d0 * {g0, g1, g2, g3}
				wino_grad4_GxW(v2, v3, g0, g1, g2, g3, d1);//d1 * {g0, g1, g2, g3}
			}
			buf ^= 1;

			//load 1 group from X
			int xic = ((oic - (tx >= STEP)) << LB >> 1) + tx;
			int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + xic;
			int X1 = X0 + IC, X2 = X1 + IC, X3 = X2 + IC;
			float d0 = (ly0 ? X[X0] : 0);
			float d1 = (ly1 ? X[X1] : 0);
			float d2 = (ly2 ? X[X2] : 0);
			float d3 = (ly3 ? X[X3] : 0);
			Ds[buf][tx][ty] = winograd_d(d0, d1, d2, d3);

			//load 2 group from CW
			int wic = ((oic - (ty >= STEP)) << LB >> 1) + ty;
			int W0 = ((fh * 3)*IC + wic)*OC;
			int W1 = W0 + Wstride, W2 = W1 + Wstride;
			float2 g0 = *(float2*)(CW + W0);
			float2 g1 = *(float2*)(CW + W1);
			float2 g2 = *(float2*)(CW + W2);
			Gs[buf][ty][tx] = winograd_g(g0.x, g1.x, g2.x);
			Gs[buf][ty][tx + STEP2] = winograd_g(g0.y, g1.y, g2.y);
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 d0 = Ds[buf][ik][ty];
			float4 d1 = Ds[buf][ik + STEP][ty];

			float4 g0 = Gs[buf][ik][tx];
			float4 g1 = Gs[buf][ik][tx + STEP2];
			float4 g2 = Gs[buf][ik + STEP][tx];
			float4 g3 = Gs[buf][ik + STEP][tx + STEP2];

			wino_grad4_GxW(v0, v1, g0, g1, g2, g3, d0);//d0 * {g0, g1, g2, g3}
			wino_grad4_GxW(v2, v3, g0, g1, g2, g3, d1);//d1 * {g0, g1, g2, g3}
		}
	}

	const int Y0 = j0 * OC + oc0;
	const int Y1 = Y0 + OC;
	const int Y2 = Y1 + OC;
	const int Y3 = Y2 + OC;

	*(float4*)(Y + Y0) = v0;
	*(float4*)(Y + Y1) = v1;
	*(float4*)(Y + Y2) = v2;
	*(float4*)(Y + Y3) = v3;
}

#endif


//OH % 2 == 0, OW % 2 == 0
//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*4)
#ifndef CONV_3D_WINOGRAD_KERNEL_W3_B3
#define CONV_3D_WINOGRAD_KERNEL_W3_B3

#define wingrad_b3(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	conv3dWinoGrad_kernel_b3<LB, (1<<LB>>1), (1<<LB), (2<<LB), (3<<LB)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, IC, OC, ph, pw, oc_index, j_index)

//139 register
//LB = 4: Size = 1.125, Time = 3.66619 msec, Performace = 658.973 GFlop/s
template<int LB, int STEP, int STEP2, int STEP4, int STEP6>
__global__ void conv3dWinoGrad_kernel_b3(
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

			//load 2 group from X
			int xic = ((oic - (tx >= STEP)) << LB >> 1) + tx;
			int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + xic;
			int X4 = ((tn2*IH + tih2)*IW + tiw2)*IC + xic;
			int X1 = X0 + IC, X2 = X1 + IC, X3 = X2 + IC;
			int X5 = X4 + IC, X6 = X5 + IC, X7 = X6 + IC;
			float d0 = (ly0 ? X[X0] : 0);
			float d1 = (ly1 ? X[X1] : 0);
			float d2 = (ly2 ? X[X2] : 0);
			float d3 = (ly3 ? X[X3] : 0);
			float d4 = (ly4 ? X[X4] : 0);
			float d5 = (ly5 ? X[X5] : 0);
			float d6 = (ly6 ? X[X6] : 0);
			float d7 = (ly7 ? X[X7] : 0);
			Ds[tx][ty] = winograd_d(d0, d1, d2, d3);
			Ds[tx][ty + STEP2] = winograd_d(d4, d5, d6, d7);
			__syncthreads();

#pragma unroll
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

				wino_grad4_GxW(v0, v2, g0, g1, g2, g3, d0);
				wino_grad4_GxW(v4, v6, g0, g1, g2, g3, d1);
				wino_grad4_GxW(v8, v10, g0, g1, g2, g3, d2);
				wino_grad4_GxW(v12, v14, g0, g1, g2, g3, d3);

				float4 g4 = Gs[ik + STEP][tx];
				float4 g5 = Gs[ik + STEP][tx + STEP2];
				float4 g6 = Gs[ik + STEP][tx + STEP4];
				float4 g7 = Gs[ik + STEP][tx + STEP6];
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


//OH % 2 == 0, OW % 2 == 0
//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*4)
#ifndef CONV_3D_WINOGRAD_KERNEL_W3_B4
#define CONV_3D_WINOGRAD_KERNEL_W3_B4

#define wingrad_b4(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	conv3dWinoGrad_kernel_b4<LB, (1<<LB>>1), (1<<LB), (2<<LB), (3<<LB)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, IC, OC, ph, pw, oc_index, j_index)

//139 register
//LB = 4: Size = 1.125, Time = 2.60397 msec, Performace = 927.784 GFlop/s
template<int LB, int STEP, int STEP2, int STEP4, int STEP6>
__global__ void conv3dWinoGrad_kernel_b4(
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

	for (int fh = 0; fh < 3; fh++)
	{
		int tih0 = toh0 - ph + fh, tiw0 = tow0 - pw;
		bool lh0 = (tih0 >= 0) && (tih0 < IH);
		bool ly0 = lh0 && (tiw0 >= 0) && (tiw0 < IW);
		bool ly1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);
		bool ly2 = lh0 && (tiw0 >= -2) && (tiw0 + 2 < IW);
		bool ly3 = lh0 && (tiw0 >= -3) && (tiw0 + 3 < IW);
		char LY0 = ly0 + (ly1 << 1) + (ly2 << 2) + (ly3 << 3);

		/*int tih2 = toh2 - ph + fh, tiw2 = tow2 - pw;
		bool lh2 = (tih2 >= 0) && (tih2 < IH);
		bool ly4 = lh2 && (tiw2 >= 0) && (tiw2 < IW);
		bool ly5 = lh2 && (tiw2 >= -1) && (tiw2 + 1 < IW);
		bool ly6 = lh2 && (tiw2 >= -2) && (tiw2 + 2 < IW);
		bool ly7 = lh2 && (tiw2 >= -3) && (tiw2 + 3 < IW);
		char LY1 = ly4 + (ly5 << 1) + (ly6 << 2) + (ly7 << 3);*/

		for (int oic = 0, OIC = (IC << 1 >> LB); oic < OIC; oic++)
		{
			//load 2 group from X
			int xic = ((oic - (tx >= STEP)) << LB >> 1) + tx;
			int X1 = ((tn0*IH + tih0)*IW + tiw0 + 1)*IC + xic;
			float d0 = (LY0 & 1 ? X[X1 - IC] : 0);
			float d1 = (LY0 & 2 ? X[X1] : 0);
			float d2 = (LY0 & 4 ? X[X1 + IC] : 0);
			float d3 = (LY0 & 8 ? X[X1 + (IC << 1)] : 0);
			Ds[tx][ty] = winograd_d(d0, d1, d2, d3);

			/*int X5 = ((tn2*IH + tih2)*IW + tiw2 + 1)*IC + xic;
			float d4 = (LY1 & 1 ? X[X5 - IC] : 0);
			float d5 = (LY1 & 2 ? X[X5] : 0);
			float d6 = (LY1 & 4 ? X[X5 + IC] : 0);
			float d7 = (LY1 & 8 ? X[X5 + (IC << 1)] : 0);
			Ds[tx][ty + STEP2] = winograd_d(d4, d5, d6, d7);*/

			//load 4 group from CW
			int wic = ((oic - (ty >= STEP)) << LB >> 1) + ty;
			int W0 = ((fh * 3)*IC + wic)*OC;
			int W1 = W0 + Wstride, W2 = W1 + Wstride;
			float2 g0 = *(float2*)(CW + W0);
			float2 g1 = *(float2*)(CW + W1);
			float2 g2 = *(float2*)(CW + W2);
			Gs[ty][tx] = winograd_g(g0.x, g1.x, g2.x);
			Gs[ty][tx + STEP2] = winograd_g(g0.y, g1.y, g2.y);
			//Gs[ty][tx + STEP4] = winograd_g(g0.z, g1.z, g2.z);
			//Gs[ty][tx + STEP6] = winograd_g(g0.w, g1.w, g2.w);
			__syncthreads();

#pragma unroll
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


//OH % 2 == 0, OW % 2 == 0
//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*4)
#ifndef CONV_3D_WINOGRAD_KERNEL_W3_B5
#define CONV_3D_WINOGRAD_KERNEL_W3_B5
#define wingrad_b5(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	conv3dWinoGrad_kernel_b5<LB, (1<<LB>>1), (1<<LB), (2<<LB), (3<<LB)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, IC, OC, ph, pw, oc_index, j_index)

//LB = 4: Size = 1.125, Time = 2.23904 msec, Performace = 1079 GFlop/s
template<int LB, int STEP, int STEP2, int STEP4, int STEP6>
__global__ void conv3dWinoGrad_kernel_b5(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	bool buf = 0;
	__shared__ float4 Gs[2][1 << LB][(4 << LB) + 1];//{toc0, toc1}
	__shared__ float4 Ds[2][1 << LB][(1 << LB) + 1];//{tgroup0}

	//prepare for GN = OC
	int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
	CW += oc0 + ((ty >= STEP) << 2);//CW[0, 0, 0, toc0]
	const int Wstride = IC * OC;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 2) + j_index;
	int tj0 = j0 + ((tx >= STEP) << 1);
	get_n_oh_ow(tj0, tn0, toh0, tow0);

	//compute area-----------------------------------------------
	float4 v0 = float4{ 0, 0, 0, 0 }, v1 = float4{ 0, 0, 0, 0 };
	float4 v2 = float4{ 0, 0, 0, 0 }, v3 = float4{ 0, 0, 0, 0 };
	float4 v4 = float4{ 0, 0, 0, 0 }, v5 = float4{ 0, 0, 0, 0 };
	float4 v6 = float4{ 0, 0, 0, 0 }, v7 = float4{ 0, 0, 0, 0 };

//#pragma unroll
	for (int fh = 0; fh < 3; fh++)
	{
		int tih0 = toh0 - ph + fh, tiw0 = tow0 - pw;
		bool lh0 = (tih0 >= 0) && (tih0 < IH);
		bool ly0 = lh0 && (tiw0 >= 0) && (tiw0 < IW);
		bool ly1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);
		bool ly2 = lh0 && (tiw0 >= -2) && (tiw0 + 2 < IW);
		bool ly3 = lh0 && (tiw0 >= -3) && (tiw0 + 3 < IW);

		//load 4 group from CW
		int wic = ty - ((ty >= STEP) << LB >> 1);
		int W0 = ((fh * 3)*IC + wic)*OC;
		int W1 = W0 + Wstride, W2 = W1 + Wstride;
		float4 g0 = *(float4*)(CW + W0);
		float4 g1 = *(float4*)(CW + W1);
		float4 g2 = *(float4*)(CW + W2);
		Gs[buf][ty][tx] = winograd_g(g0.x, g1.x, g2.x);
		Gs[buf][ty][tx + STEP2] = winograd_g(g0.y, g1.y, g2.y);
		Gs[buf][ty][tx + STEP4] = winograd_g(g0.z, g1.z, g2.z);
		Gs[buf][ty][tx + STEP6] = winograd_g(g0.w, g1.w, g2.w);

		//load 1 group from X
		int xic = tx - ((tx >= STEP) << LB >> 1);
		int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + xic;
		int X1 = X0 + IC, X2 = X1 + IC, X3 = X2 + IC;
		float d0 = (ly0 ? X[X0] : 0);
		float d1 = (ly1 ? X[X1] : 0);
		float d2 = (ly2 ? X[X2] : 0);
		float d3 = (ly3 ? X[X3] : 0);
		Ds[buf][tx][ty] = winograd_d(d0, d1, d2, d3);
		__syncthreads();

		for (int oic = 1, OIC = (IC << 1 >> LB); oic < OIC; oic++)
		{
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 g0 = Gs[buf][ik][tx];
				float4 g1 = Gs[buf][ik][tx + STEP2];
				float4 g2 = Gs[buf][ik][tx + STEP4];
				float4 g3 = Gs[buf][ik][tx + STEP6];

				float4 g4 = Gs[buf][ik + STEP][tx];
				float4 g5 = Gs[buf][ik + STEP][tx + STEP2];
				float4 g6 = Gs[buf][ik + STEP][tx + STEP4];
				float4 g7 = Gs[buf][ik + STEP][tx + STEP6];

				float4 d0 = Ds[buf][ik][ty];
				float4 d1 = Ds[buf][ik + STEP][ty];

				wino_grad4_GxW(v0, v2, g0, g1, g2, g3, d0);
				wino_grad4_GxW(v4, v6, g0, g1, g2, g3, d1);
				wino_grad4_GxW(v1, v3, g4, g5, g6, g7, d0);
				wino_grad4_GxW(v5, v7, g4, g5, g6, g7, d1);
			}
			buf ^= 1;

			//load 4 group from CW
			int wic = ((oic - (ty >= STEP)) << LB >> 1) + ty;
			int W0 = ((fh * 3)*IC + wic)*OC;
			int W1 = W0 + Wstride, W2 = W1 + Wstride;
			float4 g0 = *(float4*)(CW + W0);
			float4 g1 = *(float4*)(CW + W1);
			float4 g2 = *(float4*)(CW + W2);
			Gs[buf][ty][tx] = winograd_g(g0.x, g1.x, g2.x);
			Gs[buf][ty][tx + STEP2] = winograd_g(g0.y, g1.y, g2.y);
			Gs[buf][ty][tx + STEP4] = winograd_g(g0.z, g1.z, g2.z);
			Gs[buf][ty][tx + STEP6] = winograd_g(g0.w, g1.w, g2.w);

			//load 1 group from X
			int xic = ((oic - (tx >= STEP)) << LB >> 1) + tx;
			int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + xic;
			int X1 = X0 + IC, X2 = X1 + IC, X3 = X2 + IC;
			float d0 = (ly0 ? X[X0] : 0);
			float d1 = (ly1 ? X[X1] : 0);
			float d2 = (ly2 ? X[X2] : 0);
			float d3 = (ly3 ? X[X3] : 0);
			Ds[buf][tx][ty] = winograd_d(d0, d1, d2, d3);
			__syncthreads();
		}
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 g0 = Gs[buf][ik][tx];
			float4 g1 = Gs[buf][ik][tx + STEP2];
			float4 g2 = Gs[buf][ik][tx + STEP4];
			float4 g3 = Gs[buf][ik][tx + STEP6];

			float4 g4 = Gs[buf][ik + STEP][tx];
			float4 g5 = Gs[buf][ik + STEP][tx + STEP2];
			float4 g6 = Gs[buf][ik + STEP][tx + STEP4];
			float4 g7 = Gs[buf][ik + STEP][tx + STEP6];

			float4 d0 = Ds[buf][ik][ty];
			float4 d1 = Ds[buf][ik + STEP][ty];

			wino_grad4_GxW(v0, v2, g0, g1, g2, g3, d0);
			wino_grad4_GxW(v4, v6, g0, g1, g2, g3, d1);
			wino_grad4_GxW(v1, v3, g4, g5, g6, g7, d0);
			wino_grad4_GxW(v5, v7, g4, g5, g6, g7, d1);
		}
	}

	const int Y0 = j0 * OC + oc0;
	const int Y1 = Y0 + OC;
	const int Y2 = Y1 + OC;
	const int Y3 = Y2 + OC;

	*(float4*)(Y + Y0) = v0; *(float4*)(Y + Y0 + 4) = v1;
	*(float4*)(Y + Y1) = v2; *(float4*)(Y + Y1 + 4) = v3;
	*(float4*)(Y + Y2) = v4; *(float4*)(Y + Y2 + 4) = v5;
	*(float4*)(Y + Y3) = v6; *(float4*)(Y + Y3 + 4) = v7;
}

#endif
