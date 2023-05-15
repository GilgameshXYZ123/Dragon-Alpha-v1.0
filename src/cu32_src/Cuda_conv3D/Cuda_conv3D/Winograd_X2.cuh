

//OH % 2 == 0, OW % 2 == 0
//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*4)
#ifndef CONV_3D_WINOGRAD_KERNEL_W3_A1
#define CONV_3D_WINOGRAD_KERNEL_W3_A1

#define wingrad_a1(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	conv3dWinoGrad_kernel_a1<LB, (1<<LB), (2<<LB), (3<<LB)>\
		<<< dim3(GM>>LB>>1, GN>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, IC, OC, ph, pw, oc_index, j_index)

//LB = 4: Size = 1.125, Time = 4.3863 msec, Performace = 550.787 GFlop/s
template<int LB, int STEP, int STEP2, int STEP3>
__global__ void conv3dWinoGrad_kernel_a1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int ph, int pw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	__shared__ float4 Gs[1 << LB][4 << LB];//{oc0, oc1, oc2, oc3}
	__shared__ float4 Ds[1 << LB][1 << LB];

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 2) + oc_index;
	CW += oc0;//CW[0, 0, 0, oc0]
	const int Wstride = IC * OC;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.x << LB) + tx) << 1) + j_index;
	get_n_oh_ow(j0, n0, oh0, ow0);

	//compute area---------------------------------------
	float4 v0 = float4{ 0, 0, 0, 0 };
	float4 v1 = float4{ 0, 0, 0, 0 };

#pragma unroll
	for (int fh = 0; fh < 3; fh++)
	{
		int ih = oh0 - ph + fh; bool ly = (ih >= 0) && (ih < IH);
		int iw0 = ow0 - pw; bool ly0 = ly && (iw0 >= 0) && (iw0 < IW);
		int iw1 = iw0 + 1; bool ly1 = ly && (iw1 >= 0) && (iw1 < IW);
		int iw2 = iw0 + 2; bool ly2 = ly && (iw2 >= 0) && (iw2 < IW);
		int iw3 = iw0 + 3; bool ly3 = ly && (iw3 >= 0) && (iw3 < IW);

		for (int oic = 0, OIC = (IC >> LB); oic < OIC; oic++)
		{
			//load 1 group from X
			int xic = (oic << LB) + ty;
			int X0 = ((n0*IH + ih)*IW + iw0)*IC + xic;
			int X1 = X0 + IC, X2 = X1 + IC, X3 = X2 + IC;
			float d0 = (ly0 ? X[X0] : 0);
			float d1 = (ly1 ? X[X1] : 0);
			float d2 = (ly2 ? X[X2] : 0);
			float d3 = (ly3 ? X[X3] : 0);
			Ds[ty][tx] = winograd_d(d0, d1, d2, d3);

			//load 4 group from CW
			int wic = (oic << LB) + tx;
			int W0 = ((fh * 3)*IC + wic)*OC;
			int W1 = W0 + Wstride, W2 = W1 + Wstride;
			float4 g0 = *(float4*)(CW + W0);
			float4 g1 = *(float4*)(CW + W1);
			float4 g2 = *(float4*)(CW + W2);
			Gs[tx][ty        ] = winograd_g(g0.x, g1.x, g2.x);
			Gs[tx][ty + STEP ] = winograd_g(g0.y, g1.y, g2.y);
			Gs[tx][ty + STEP2] = winograd_g(g0.z, g1.z, g2.z);
			Gs[tx][ty + STEP3] = winograd_g(g0.w, g1.w, g2.w);
			__syncthreads();

#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 g0 = Gs[ik][ty        ];
				float4 g1 = Gs[ik][ty + STEP ];
				float4 g2 = Gs[ik][ty + STEP2];
				float4 g3 = Gs[ik][ty + STEP3];
				float4 d0 = Ds[ik][tx];

				float4 m00 = float4_elem_mul(g0, d0);
				float4 m10 = float4_elem_mul(g1, d0);
				float4 m20 = float4_elem_mul(g2, d0);
				float4 m30 = float4_elem_mul(g3, d0);

				v0.x += (m00.x + m00.y + m00.z);
				v0.y += (m10.x + m10.y + m10.z);
				v0.z += (m20.x + m20.y + m20.z);
				v0.w += (m30.x + m30.y + m30.z);

				v1.x += (m00.y - m00.z - m00.w);
				v1.y += (m10.y - m10.z - m10.w);
				v1.z += (m20.y - m20.z - m20.w);
				v1.w += (m30.y - m30.z - m30.w);
			}
			__syncthreads();
		}
	}

	const int Y0 = j0 * OC + oc0;
	const int Y1 = Y0 + OC;

	*(float4*)(Y + Y0) = v0;
	*(float4*)(Y + Y1) = v1;
}

#endif



//OH % 2 == 0, OW % 2 == 0
//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*4)
#ifndef CONV_3D_WINOGRAD_KERNEL_W3_A2
#define CONV_3D_WINOGRAD_KERNEL_W3_A2

#define wingrad_a2(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	conv3dWinoGrad_kernel_a2<LB, (1<<LB), (2<<LB), (3<<LB)>\
		<<< dim3(GM>>LB>>2, GN>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, IC, OC, ph, pw, oc_index, j_index)

//LB = 4: Size = 1.125, Time = 4.59284 msec, Performace = 526.019 GFlop/s
template<int LB, int STEP, int STEP2, int STEP3>
__global__ void conv3dWinoGrad_kernel_a2(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int ph, int pw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	__shared__ float4 Gs[1 << LB][4 << LB];//{oc0, oc1, oc2, oc3}
	__shared__ float4 Ds[1 << LB][2 << LB];//{group0, group1}

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 2) + oc_index;
	CW += oc0;//CW[0, 0, 0, oc0]
	const int Wstride = IC * OC;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.x << LB) + tx) << 2) + j_index, j2 = j0 + 2;
	get_n_oh_ow(j0, n0, oh0, ow0);//j0, j1
	get_n_oh_ow(j2, n2, oh2, ow2);//j2, j3

	//compute area---------------------------------------
	float4 v0 = float4{ 0, 0, 0, 0 };
	float4 v1 = float4{ 0, 0, 0, 0 };
	float4 v2 = float4{ 0, 0, 0, 0 };
	float4 v3 = float4{ 0, 0, 0, 0 };

#pragma unroll
	for (int fh = 0; fh < 3; fh++)
	{
		int ih0 = oh0 - ph + fh; bool lh0 = (ih0 >= 0) && (ih0 < IH);
		int ih2 = oh2 - ph + fh; bool lh2 = (ih2 >= 0) && (ih2 < IH);

		int iw0 = ow0 - pw; bool ly0 = lh0 && (iw0 >= 0) && (iw0 < IW);
		int iw1 = iw0 + 1;  bool ly1 = lh0 && (iw1 >= 0) && (iw1 < IW);
		int iw2 = iw0 + 2;  bool ly2 = lh0 && (iw2 >= 0) && (iw2 < IW);
		int iw3 = iw0 + 3;  bool ly3 = lh0 && (iw3 >= 0) && (iw3 < IW);
		
		int iw4 = ow2 - pw; bool ly4 = lh2 && (iw4 >= 0) && (iw4 < IW);
		int iw5 = iw4 + 1;  bool ly5 = lh2 && (iw5 >= 0) && (iw5 < IW);
		int iw6 = iw4 + 2;  bool ly6 = lh2 && (iw6 >= 0) && (iw6 < IW);
		int iw7 = iw4 + 3;  bool ly7 = lh2 && (iw7 >= 0) && (iw7 < IW);

		for (int oic = 0, OIC = (IC >> LB); oic < OIC; oic++)
		{
			//load 2 group from X
			int xic = (oic << LB) + ty;
			int X0 = ((n0*IH + ih0)*IW + iw0)*IC + xic;
			int X4 = ((n2*IH + ih2)*IW + iw4)*IC + xic;
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

			Ds[ty][tx       ] = winograd_d(d0, d1, d2, d3);
			Ds[ty][tx + STEP] = winograd_d(d4, d5, d6, d7);

			//load 4 group from CW
			int wic = (oic << LB) + tx;
			int W0 = ((fh * 3)*IC + wic)*OC;
			int W1 = W0 + Wstride, W2 = W1 + Wstride;
			float4 g0 = *(float4*)(CW + W0);
			float4 g1 = *(float4*)(CW + W1);
			float4 g2 = *(float4*)(CW + W2);

			Gs[tx][ty        ] = winograd_g(g0.x, g1.x, g2.x);
			Gs[tx][ty + STEP ] = winograd_g(g0.y, g1.y, g2.y);

			Gs[tx][ty + STEP2] = winograd_g(g0.z, g1.z, g2.z);
			Gs[tx][ty + STEP3] = winograd_g(g0.w, g1.w, g2.w);
			__syncthreads();

#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 g0 = Gs[ik][ty        ];
				float4 g1 = Gs[ik][ty + STEP ];
				float4 g2 = Gs[ik][ty + STEP2];
				float4 g3 = Gs[ik][ty + STEP3];

				float4 d0 = Ds[ik][tx];
				float4 d1 = Ds[ik][tx + STEP];

				//compute for group0 * {oc0, oc1, oc2, oc3}
				float4 m00 = float4_elem_mul(g0, d0);
				float4 m10 = float4_elem_mul(g1, d0);
				float4 m20 = float4_elem_mul(g2, d0);
				float4 m30 = float4_elem_mul(g3, d0);

				v0.x += (m00.x + m00.y + m00.z);
				v0.y += (m10.x + m10.y + m10.z);
				v0.z += (m20.x + m20.y + m20.z);
				v0.w += (m30.x + m30.y + m30.z);

				v1.x += (m00.y - m00.z - m00.w);
				v1.y += (m10.y - m10.z - m10.w);
				v1.z += (m20.y - m20.z - m20.w);
				v1.w += (m30.y - m30.z - m30.w);

				//compute for group1 * {oc0, oc1, oc2, oc3}
				float4 m01 = float4_elem_mul(g0, d1);
				float4 m11 = float4_elem_mul(g1, d1);
				float4 m21 = float4_elem_mul(g2, d1);
				float4 m31 = float4_elem_mul(g3, d1);

				v2.x += (m01.x + m01.y + m01.z);
				v2.y += (m11.x + m11.y + m11.z);
				v2.z += (m21.x + m21.y + m21.z);
				v2.w += (m31.x + m31.y + m31.z);

				v3.x += (m01.y - m01.z - m01.w);
				v3.y += (m11.y - m11.z - m11.w);
				v3.z += (m21.y - m21.z - m21.w);
				v3.w += (m31.y - m31.z - m31.w);
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
#ifndef CONV_3D_WINOGRAD_KERNEL_W3_A3
#define CONV_3D_WINOGRAD_KERNEL_W3_A3

#define wingrad_a3(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	conv3dWinoGrad_kernel_a3<LB, (1<<LB), (2<<LB), (3<<LB)>\
		<<< dim3(GN>>LB>>2, GM>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, IC, OC, ph, pw, oc_index, j_index)

//LB = 4: Size = 1.125, Time = 4.26008 msec, Performace = 567.107 GFlop/s
template<int LB, int STEP, int STEP2, int STEP3>
__global__ void conv3dWinoGrad_kernel_a3(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	__shared__ float4 Gs[1 << LB][4 << LB];//{oc0, oc1, oc2, oc3}
	__shared__ float4 Ds[1 << LB][2 << LB];//{group0, group1}

	//prepare for GN = OC
	int oc0 = (((blockIdx.x << LB) + tx) << 2) + oc_index;
	CW += oc0;//CW[0, 0, 0, oc0]
	const int Wstride = IC * OC;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 2) + j_index, j2 = j0 + 2;
	get_n_oh_ow(j0, n0, oh0, ow0);//j0, j1
	get_n_oh_ow(j2, n2, oh2, ow2);//j2, j3

	//compute area---------------------------------------
	float4 v0 = float4{ 0, 0, 0, 0 };
	float4 v1 = float4{ 0, 0, 0, 0 };
	float4 v2 = float4{ 0, 0, 0, 0 };
	float4 v3 = float4{ 0, 0, 0, 0 };

#pragma unroll
	for (int fh = 0; fh < 3; fh++)
	{
		int ih0 = oh0 - ph + fh; bool lh0 = (ih0 >= 0) && (ih0 < IH);
		int ih2 = oh2 - ph + fh; bool lh2 = (ih2 >= 0) && (ih2 < IH);

		int iw0 = ow0 - pw; bool ly0 = lh0 && (iw0 >= 0) && (iw0 < IW);
		int iw1 = iw0 + 1;  bool ly1 = lh0 && (iw1 >= 0) && (iw1 < IW);
		int iw2 = iw0 + 2;  bool ly2 = lh0 && (iw2 >= 0) && (iw2 < IW);
		int iw3 = iw0 + 3;  bool ly3 = lh0 && (iw3 >= 0) && (iw3 < IW);

		int iw4 = ow2 - pw; bool ly4 = lh2 && (iw4 >= 0) && (iw4 < IW);
		int iw5 = iw4 + 1;  bool ly5 = lh2 && (iw5 >= 0) && (iw5 < IW);
		int iw6 = iw4 + 2;  bool ly6 = lh2 && (iw6 >= 0) && (iw6 < IW);
		int iw7 = iw4 + 3;  bool ly7 = lh2 && (iw7 >= 0) && (iw7 < IW);

		for (int oic = 0, OIC = (IC >> LB); oic < OIC; oic++)
		{
			//load 2 group from X
			int xic = (oic << LB) + tx;
			int X0 = ((n0*IH + ih0)*IW + iw0)*IC + xic;
			int X4 = ((n2*IH + ih2)*IW + iw4)*IC + xic;
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
			Ds[tx][ty + STEP] = winograd_d(d4, d5, d6, d7);

			//load 4 group from CW
			int wic = (oic << LB) + ty;
			int W0 = ((fh * 3)*IC + wic)*OC;
			int W1 = W0 + Wstride, W2 = W1 + Wstride;
			float4 g0 = *(float4*)(CW + W0);
			float4 g1 = *(float4*)(CW + W1);
			float4 g2 = *(float4*)(CW + W2);

			Gs[ty][tx] = winograd_g(g0.x, g1.x, g2.x);
			Gs[ty][tx + STEP] = winograd_g(g0.y, g1.y, g2.y);

			Gs[ty][tx + STEP2] = winograd_g(g0.z, g1.z, g2.z);
			Gs[ty][tx + STEP3] = winograd_g(g0.w, g1.w, g2.w);
			__syncthreads();

#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 g0 = Gs[ik][tx];
				float4 g1 = Gs[ik][tx + STEP];
				float4 g2 = Gs[ik][tx + STEP2];
				float4 g3 = Gs[ik][tx + STEP3];

				float4 d0 = Ds[ik][ty];
				float4 d1 = Ds[ik][ty + STEP];

				//compute for group0 * {oc0, oc1, oc2, oc3}
				float4 m00 = float4_elem_mul(g0, d0);
				float4 m10 = float4_elem_mul(g1, d0);
				float4 m20 = float4_elem_mul(g2, d0);
				float4 m30 = float4_elem_mul(g3, d0);

				v0.x += (m00.x + m00.y + m00.z);
				v0.y += (m10.x + m10.y + m10.z);
				v0.z += (m20.x + m20.y + m20.z);
				v0.w += (m30.x + m30.y + m30.z);

				v1.x += (m00.y - m00.z - m00.w);
				v1.y += (m10.y - m10.z - m10.w);
				v1.z += (m20.y - m20.z - m20.w);
				v1.w += (m30.y - m30.z - m30.w);

				//compute for group1 * {oc0, oc1, oc2, oc3}
				float4 m01 = float4_elem_mul(g0, d1);
				float4 m11 = float4_elem_mul(g1, d1);
				float4 m21 = float4_elem_mul(g2, d1);
				float4 m31 = float4_elem_mul(g3, d1);

				v2.x += (m01.x + m01.y + m01.z);
				v2.y += (m11.x + m11.y + m11.z);
				v2.z += (m21.x + m21.y + m21.z);
				v2.w += (m31.x + m31.y + m31.z);

				v3.x += (m01.y - m01.z - m01.w);
				v3.y += (m11.y - m11.z - m11.w);
				v3.z += (m21.y - m21.z - m21.w);
				v3.w += (m31.y - m31.z - m31.w);
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
#ifndef CONV_3D_WINOGRAD_KERNEL_W3_A4
#define CONV_3D_WINOGRAD_KERNEL_W3_A4

#define wingrad_a4(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	conv3dWinoGrad_kernel_a4<LB, (1<<LB>>1), (1<<LB)>\
		<<< dim3(GN>>LB>>2, GM>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, IC, OC, ph, pw, oc_index, j_index)

//LB = 4: Size = 1.125, Time = 4.04058 msec, Performace = 597.913 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void conv3dWinoGrad_kernel_a4(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	__shared__ float4 Gs[1 << LB][2 << LB];//{toc0, toc1}
	__shared__ float4 Ds[1 << LB][1 << LB];//{tgroup0}

	//prepare for GN = OC
	int oc0 = (((blockIdx.x << LB) + tx) << 2) + oc_index;
	CW += oc0 + ((ty >= STEP) << 1);//CW[0, 0, 0, oc0]
	const int Wstride = IC * OC;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.y << LB) + ty) << 2) + j_index, j2 = j0 + 2;
	int tj0 = j0 + ((tx >= STEP) << 1);
	get_n_oh_ow(tj0, tn0, toh0, tow0);

	const int Gs_x = tx + ((ty >= STEP) << LB);
	const int Ds_y = ty + ((tx >= STEP) << LB);

	//compute area---------------------------------------
	float4 v0 = float4{ 0, 0, 0, 0 };
	float4 v1 = float4{ 0, 0, 0, 0 };
	float4 v2 = float4{ 0, 0, 0, 0 };
	float4 v3 = float4{ 0, 0, 0, 0 };

#pragma unroll
	for (int fh = 0; fh < 3; fh++)
	{
		int tih0 = toh0 - ph + fh; bool lh0 = (tih0 >= 0) && (tih0 < IH);
		int tiw0 = tow0 - pw; bool ly0 = lh0 && (tiw0 >= 0) && (tiw0 < IW);
		int tiw1 = tiw0 + 1;  bool ly1 = lh0 && (tiw1 >= 0) && (tiw1 < IW);
		int tiw2 = tiw0 + 2;  bool ly2 = lh0 && (tiw2 >= 0) && (tiw2 < IW);
		int tiw3 = tiw0 + 3;  bool ly3 = lh0 && (tiw3 >= 0) && (tiw3 < IW);

		for (int oic = 0, OIC = (IC << 1 >> LB); oic < OIC; oic++)
		{
			//load 1 group from X
			int xic = ((oic - (tx >= STEP)) << LB >> 1) + tx;
			int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + xic;
			int X1 = X0 + IC, X2 = X1 + IC, X3 = X2 + IC;
			float d0 = (ly0 ? X[X0] : 0);
			float d1 = (ly1 ? X[X1] : 0);
			float d2 = (ly2 ? X[X2] : 0);
			float d3 = (ly3 ? X[X3] : 0);
			Ds[tx][ty] = winograd_d(d0, d1, d2, d3);

			//load 2 group from CW
			int wic = ((oic - (ty >= STEP)) << LB >> 1) + ty;
			int W0 = ((fh * 3)*IC + wic)*OC;
			int W1 = W0 + Wstride, W2 = W1 + Wstride;
			float2 g0 = *(float2*)(CW + W0);
			float2 g1 = *(float2*)(CW + W1);
			float2 g2 = *(float2*)(CW + W2);
			Gs[ty][(tx << 1)] = winograd_g(g0.x, g1.x, g2.x);
			Gs[ty][(tx << 1) + 1] = winograd_g(g0.y, g1.y, g2.y);
			__syncthreads();

#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 g0 = Gs[ik][(tx << 1)];
				float4 g1 = Gs[ik][(tx << 1) + 1];
				float4 g2 = Gs[ik + STEP][(tx << 1)];
				float4 g3 = Gs[ik + STEP][(tx << 1) + 1];

				float4 d0 = Ds[ik][ty];
				float4 d1 = Ds[ik + STEP][ty];

				//compute for group0 * {oc0, oc1, oc2, oc3}
				float4 m00 = float4_elem_mul(g0, d0);
				float4 m10 = float4_elem_mul(g1, d0);
				float4 m20 = float4_elem_mul(g2, d0);
				float4 m30 = float4_elem_mul(g3, d0);

				v0.x += (m00.x + m00.y + m00.z);
				v0.y += (m10.x + m10.y + m10.z);
				v0.z += (m20.x + m20.y + m20.z);
				v0.w += (m30.x + m30.y + m30.z);

				v1.x += (m00.y - m00.z - m00.w);
				v1.y += (m10.y - m10.z - m10.w);
				v1.z += (m20.y - m20.z - m20.w);
				v1.w += (m30.y - m30.z - m30.w);

				//compute for group1 * {oc0, oc1, oc2, oc3}
				float4 m01 = float4_elem_mul(g0, d1);
				float4 m11 = float4_elem_mul(g1, d1);
				float4 m21 = float4_elem_mul(g2, d1);
				float4 m31 = float4_elem_mul(g3, d1);

				v2.x += (m01.x + m01.y + m01.z);
				v2.y += (m11.x + m11.y + m11.z);
				v2.z += (m21.x + m21.y + m21.z);
				v2.w += (m31.x + m31.y + m31.z);

				v3.x += (m01.y - m01.z - m01.w);
				v3.y += (m11.y - m11.z - m11.w);
				v3.z += (m21.y - m21.z - m21.w);
				v3.w += (m31.y - m31.z - m31.w);
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
#ifndef CONV_3D_WINOGRAD_KERNEL_W3_A5
#define CONV_3D_WINOGRAD_KERNEL_W3_A5

#define wingrad_a5(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	conv3dWinoGrad_kernel_a5<LB, (1<<LB>>1), (1<<LB)>\
		<<< dim3(GN>>LB>>2, GM>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, IC, OC, ph, pw, oc_index, j_index)

//LB = 4: Size = 1.125, Time = 2.42332 msec, Performace = 996.945 GFlop/
template<int LB, int STEP, int STEP2>
__global__ void conv3dWinoGrad_kernel_a5(
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
		bool ly0 = lh0 && (tiw0 >=  0) && (tiw0     < IW);
		bool ly1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);
		bool ly2 = lh0 && (tiw0 >= -2) && (tiw0 + 2 < IW);
		bool ly3 = lh0 && (tiw0 >= -3) && (tiw0 + 3 < IW);

		for (int oic = 0, OIC = (IC << 1 >> LB); oic < OIC; oic++)
		{
			//load 1 group from X
			int xic = ((oic - (tx >= STEP)) << LB >> 1) + tx;
			int X0 = ((tn0*IH + tih0)*IW + tiw0)*IC + xic;
			int X1 = X0 + IC, X2 = X1 + IC, X3 = X2 + IC;
			float d0 = (ly0 ? X[X0] : 0);
			float d1 = (ly1 ? X[X1] : 0);
			float d2 = (ly2 ? X[X2] : 0);
			float d3 = (ly3 ? X[X3] : 0);
			Ds[tx][ty] = winograd_d(d0, d1, d2, d3);

			//load 2 group from CW
			int wic = ((oic - (ty >= STEP)) << LB >> 1) + ty;
			int W0 = ((fh * 3)*IC + wic)*OC;
			int W1 = W0 + Wstride, W2 = W1 + Wstride;
			float2 g0 = *(float2*)(CW + W0);
			float2 g1 = *(float2*)(CW + W1);
			float2 g2 = *(float2*)(CW + W2);
			Gs[ty][tx        ] = winograd_g(g0.x, g1.x, g2.x);
			Gs[ty][tx + STEP2] = winograd_g(g0.y, g1.y, g2.y);
			__syncthreads();

#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 g0 = Gs[ik][tx];
				float4 g1 = Gs[ik][tx + STEP2];
				float4 g2 = Gs[ik + STEP][tx];
				float4 g3 = Gs[ik + STEP][tx + STEP2];

				float4 d0 = Ds[ik][ty];
				float4 d1 = Ds[ik + STEP][ty];

				//compute for group0 * {oc0, oc1, oc2, oc3}
				float4 m00 = float4_elem_mul(g0, d0);
				float4 m10 = float4_elem_mul(g1, d0);
				float4 m20 = float4_elem_mul(g2, d0);
				float4 m30 = float4_elem_mul(g3, d0);

				v0.x += (m00.x + m00.y + m00.z);
				v0.y += (m10.x + m10.y + m10.z);
				v0.z += (m20.x + m20.y + m20.z);
				v0.w += (m30.x + m30.y + m30.z);

				v1.x += (m00.y - m00.z - m00.w);
				v1.y += (m10.y - m10.z - m10.w);
				v1.z += (m20.y - m20.z - m20.w);
				v1.w += (m30.y - m30.z - m30.w);

				//compute for group1 * {oc0, oc1, oc2, oc3}
				float4 m01 = float4_elem_mul(g0, d1);
				float4 m11 = float4_elem_mul(g1, d1);
				float4 m21 = float4_elem_mul(g2, d1);
				float4 m31 = float4_elem_mul(g3, d1);

				v2.x += (m01.x + m01.y + m01.z);
				v2.y += (m11.x + m11.y + m11.z);
				v2.z += (m21.x + m21.y + m21.z);
				v2.w += (m31.x + m31.y + m31.z);

				v3.x += (m01.y - m01.z - m01.w);
				v3.y += (m11.y - m11.z - m11.w);
				v3.z += (m21.y - m21.z - m21.w);
				v3.w += (m31.y - m31.z - m31.w);
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
#ifndef CONV_3D_WINOGRAD_KERNEL_W3_A6
#define CONV_3D_WINOGRAD_KERNEL_W3_A6

#define wingrad_a6(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	conv3dWinoGrad_kernel_a6<LB, (1<<LB>>1), (1<<LB)>\
		<<< dim3(GN>>LB>>2, GM>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, IC, OC, ph, pw, oc_index, j_index)

//LB = 4: Size = 1.125, Time = 2.36042 msec, Performace = 1023.51 GFlop/s
template<int LB, int STEP, int STEP2>
__global__ void conv3dWinoGrad_kernel_a6(
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
				float4 g0 = Gs[ik][tx];
				float4 g1 = Gs[ik][tx + STEP2];

				float4 g2 = Gs[ik + STEP][tx];
				float4 g3 = Gs[ik + STEP][tx + STEP2];

				float4 d0 = Ds[ik][ty];
				float4 d1 = Ds[ik + STEP][ty];

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
#ifndef CONV_3D_WINOGRAD_KERNEL_W3_A7
#define CONV_3D_WINOGRAD_KERNEL_W3_A7

#define wingrad_a7(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	conv3dWinoGrad_kernel_a7<LB, (1<<LB>>1), (1<<LB), (2<<LB), (3<<LB)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, IC, OC, ph, pw, oc_index, j_index)

//LB = 4: Size = 1.125, Time = 2.93215 msec, Performace = 823.942 GFlop/s
template<int LB, int STEP, int STEP2, int STEP4, int STEP6>
__global__ void conv3dWinoGrad_kernel_a7(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int ph, int pw,
	int oc_index, int j_index)
{
	int tx = threadIdx.x, ty = threadIdx.y;

	__shared__ float4 Gs[1 << LB][(4 << LB) + 1];//{toc0, toc1}
	__shared__ float4 Ds[1 << LB][(1 << LB) + 1];//{tgroup0}

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
				float4 g0 = Gs[ik][tx];
				float4 g1 = Gs[ik][tx + STEP2];
				float4 g2 = Gs[ik][tx + STEP4];
				float4 g3 = Gs[ik][tx + STEP6];

				float4 g4 = Gs[ik + STEP][tx];
				float4 g5 = Gs[ik + STEP][tx + STEP2];
				float4 g6 = Gs[ik + STEP][tx + STEP4];
				float4 g7 = Gs[ik + STEP][tx + STEP6];

				float4 d0 = Ds[ik][ty];
				float4 d1 = Ds[ik + STEP][ty];

				wino_grad4_GxW(v0, v2, g0, g1, g2, g3, d0);
				wino_grad4_GxW(v1, v3, g4, g5, g6, g7, d0);
				wino_grad4_GxW(v4, v6, g0, g1, g2, g3, d1);
				wino_grad4_GxW(v5, v7, g4, g5, g6, g7, d1);
			}
			__syncthreads();
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


//OH % 2 == 0, OW % 2 == 0
//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*4)
#ifndef CONV_3D_WINOGRAD_KERNEL_W3_A8
#define CONV_3D_WINOGRAD_KERNEL_W3_A8

#define wingrad_a8(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	conv3dWinoGrad_kernel_a8<LB, (1<<LB>>1), (1<<LB), (2<<LB), (3<<LB)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, IC, OC, ph, pw, oc_index, j_index)

//LB = 4: Size = 1.125, Time = 2.93215 msec, Performace = 823.942 GFlop/s
template<int LB, int STEP, int STEP2, int STEP4, int STEP6>
__global__ void conv3dWinoGrad_kernel_a8(
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

	//compute area-------------------------------------------------
	float4  v0 = float4{ 0, 0, 0, 0 },  v1 = float4{ 0, 0, 0, 0 };
	float4  v2 = float4{ 0, 0, 0, 0 },  v3 = float4{ 0, 0, 0, 0 };
	float4  v4 = float4{ 0, 0, 0, 0 },  v5 = float4{ 0, 0, 0, 0 };
	float4  v6 = float4{ 0, 0, 0, 0 },  v7 = float4{ 0, 0, 0, 0 };
	float4  v8 = float4{ 0, 0, 0, 0 },  v9 = float4{ 0, 0, 0, 0 };
	float4 v10 = float4{ 0, 0, 0, 0 }, v11 = float4{ 0, 0, 0, 0 };
	float4 v12 = float4{ 0, 0, 0, 0 }, v13 = float4{ 0, 0, 0, 0 };
	float4 v14 = float4{ 0, 0, 0, 0 }, v15 = float4{ 0, 0, 0, 0 };

#pragma unroll
	for (int fh = 0; fh < 3; fh++)
	{
		int tih0 = toh0 - ph + fh, tiw0 = tow0 - pw;
		bool lh0 = (tih0 >= 0) && (tih0 < IH);
		bool ly0 = lh0 && (tiw0 >=  0) && (tiw0     < IW);
		bool ly1 = lh0 && (tiw0 >= -1) && (tiw0 + 1 < IW);
		bool ly2 = lh0 && (tiw0 >= -2) && (tiw0 + 2 < IW);
		bool ly3 = lh0 && (tiw0 >= -3) && (tiw0 + 3 < IW);

		int tih2 = toh2 - ph + fh, tiw2 = tow2 - pw;
		bool lh2 = (tih2 >= 0) && (tih2 < IH);
		bool ly4 = lh2 && (tiw2 >=  0) && (tiw2     < IW);
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
			Gs[ty][tx        ] = winograd_g(g0.x, g1.x, g2.x);
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
			Ds[tx][ty        ] = winograd_d(d0, d1, d2, d3);
			Ds[tx][ty + STEP2] = winograd_d(d4, d5, d6, d7);
			__syncthreads();

#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 g0 = Gs[ik][tx];
				float4 g1 = Gs[ik][tx + STEP2];
				float4 g2 = Gs[ik][tx + STEP4];
				float4 g3 = Gs[ik][tx + STEP6];
				float4 g4 = Gs[ik + STEP][tx];
				float4 g5 = Gs[ik + STEP][tx + STEP2];
				float4 g6 = Gs[ik + STEP][tx + STEP4];
				float4 g7 = Gs[ik + STEP][tx + STEP6];

				float4 d0 = Ds[ik][ty];
				float4 d1 = Ds[ik][ty + STEP2];

				float4 d2 = Ds[ik + STEP][ty];
				float4 d3 = Ds[ik + STEP][ty + STEP2];

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

	const int Y0 = j0 * OC + oc0;
	const int Y1 = Y0 + OC, Y2 = Y1 + OC, Y3 = Y2 + OC;
	const int Y4 = Y3 + OC, Y5 = Y4 + OC, Y6 = Y5 + OC, Y7 = Y6 + OC;

	*(float4*)(Y + Y0) =  v0; *(float4*)(Y + Y0 + 4) =  v1;
	*(float4*)(Y + Y1) =  v2; *(float4*)(Y + Y1 + 4) =  v3;
	*(float4*)(Y + Y2) =  v4; *(float4*)(Y + Y2 + 4) =  v5;
	*(float4*)(Y + Y3) =  v6; *(float4*)(Y + Y3 + 4) =  v7;
	*(float4*)(Y + Y4) =  v8; *(float4*)(Y + Y4 + 4) =  v9;
	*(float4*)(Y + Y5) = v10; *(float4*)(Y + Y5 + 4) = v11;
	*(float4*)(Y + Y6) = v12; *(float4*)(Y + Y6 + 4) = v13;
	*(float4*)(Y + Y7) = v14; *(float4*)(Y + Y7 + 4) = v15;
}

#endif


//OH % 2 == 0, OW % 2 == 0
//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*4)
#ifndef CONV_3D_WINOGRAD_KERNEL_W3_A9
#define CONV_3D_WINOGRAD_KERNEL_W3_A9

#define wingrad_a9(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	conv3dWinoGrad_kernel_a9<LB, (1<<LB>>1), (1<<LB), (2<<LB), (3<<LB)>\
		<<< dim3(GN>>LB>>3, GM>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, IC, OC, ph, pw, oc_index, j_index)

//LB = 4: Size = 1.125, Time = 2.93215 msec, Performace = 823.942 GFlop/s
template<int LB, int STEP, int STEP2, int STEP4, int STEP6>
__global__ void conv3dWinoGrad_kernel_a9(
	cudaTextureObject_t X, int IH, int IW,
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
			float d0 = ly0 * tex1Dfetch<float>(X, X0);
			float d1 = ly1 * tex1Dfetch<float>(X, X1);
			float d2 = ly2 * tex1Dfetch<float>(X, X2);
			float d3 = ly3 * tex1Dfetch<float>(X, X3);
			float d4 = ly4 * tex1Dfetch<float>(X, X4);
			float d5 = ly5 * tex1Dfetch<float>(X, X5);
			float d6 = ly6 * tex1Dfetch<float>(X, X6);
			float d7 = ly7 * tex1Dfetch<float>(X, X7);
			Ds[tx][ty] = winograd_d(d0, d1, d2, d3);
			Ds[tx][ty + STEP2] = winograd_d(d4, d5, d6, d7);
			__syncthreads();

#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 g0 = Gs[ik][tx];
				float4 g1 = Gs[ik][tx + STEP2];
				float4 g2 = Gs[ik][tx + STEP4];
				float4 g3 = Gs[ik][tx + STEP6];
				float4 g4 = Gs[ik + STEP][tx];
				float4 g5 = Gs[ik + STEP][tx + STEP2];
				float4 g6 = Gs[ik + STEP][tx + STEP4];
				float4 g7 = Gs[ik + STEP][tx + STEP6];

				float4 d0 = Ds[ik][ty];
				wino_grad4_GxW(v0, v2, g0, g1, g2, g3, d0);
				wino_grad4_GxW(v1, v3, g4, g5, g6, g7, d0);
				
				float4 d1 = Ds[ik][ty + STEP2];
				wino_grad4_GxW(v4, v6, g0, g1, g2, g3, d1);
				wino_grad4_GxW(v5, v7, g4, g5, g6, g7, d1);


				float4 d2 = Ds[ik + STEP][ty];
				wino_grad4_GxW(v8, v10, g0, g1, g2, g3, d2);
				wino_grad4_GxW(v9, v11, g4, g5, g6, g7, d2);

				float4 d3 = Ds[ik + STEP][ty + STEP2];
				wino_grad4_GxW(v12, v14, g0, g1, g2, g3, d3);
				wino_grad4_GxW(v13, v15, g4, g5, g6, g7, d3);
			}
			__syncthreads();
		}
	}

	const int Y0 = j0 * OC + oc0;
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



