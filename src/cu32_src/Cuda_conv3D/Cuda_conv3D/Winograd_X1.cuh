



//FH = FW = 3
//sh = sw = 1
//OH % 2 == 0, OW % 2 == 0
//(Y: BLOCK_SIZE*1, X: BLOCK_SIZE*2)
#ifndef CONV_3D_WINOGRAD_KERNEL_W3_V1
#define CONV_3D_WINOGRAD_KERNEL_W3_V1

#define wingrad_v1(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	conv3dWinoGrad_kernel_v1<LB, (1<<LB)>\
		<<< dim3(GM>>LB>>1, GN>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, 3, 3, Y, OH, OW, IC, OC, ph, pw, oc_index, j_index)

template<int LB, int STEP>
__global__ void conv3dWinoGrad_kernel_v1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ Y, int OH, int OW, 
	int IC, int OC,
	int ph, int pw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	//prepare for GN = OC
	int oc0 = ((blockIdx.y << LB) + ty) + oc_index;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.x << LB) + tx) << 1) + j_index;
	int OH_OW = OH * OW;
	get_n_oh_ow(j0, n0, oh0, ow0);
	int ow1 = ow0 + 1;

	//prepare for GK = FH * FW
	int GK = FH * FW;

	//compute area---------------------------------------
	float v0 = 0, v1 = 0;
	for (int k = 0; k < GK; k += 3)
	{
		int fh = k / FW, fw0 = k % FW;
		int fw1 = fw0 + 1;
		int fw2 = fw0 + 2;
		
		int ih = oh0 - ph + fh;
		int iw0 = ow0 - pw + fw0;
		int iw1 = iw0 + 1;
		int iw2 = iw0 + 2;
		int iw3 = iw0 + 3;

		for (int ic = 0; ic < IC; ic++) 
		{
			bool ly = (ih >= 0) && (ih < IH);
			bool ly0 = ly && (iw0 >= 0) && (iw0 < IW);
			bool ly1 = ly && (iw1 >= 0) && (iw1 < IW);
			bool ly2 = ly && (iw2 >= 0) && (iw2 < IW);
			bool ly3 = ly && (iw3 >= 0) && (iw3 < IW);

			float d0 = (ly0 ? get4d(X, n0, ih, iw0, ic, IH, IW, IC) : 0);
			float d1 = (ly1 ? get4d(X, n0, ih, iw1, ic, IH, IW, IC) : 0);
			float d2 = (ly2 ? get4d(X, n0, ih, iw2, ic, IH, IW, IC) : 0);
			float d3 = (ly3 ? get4d(X, n0, ih, iw3, ic, IH, IW, IC) : 0);

			float g0 = get4d(W, oc0, fh, fw0, ic, FH, FW, IC);
			float g1 = get4d(W, oc0, fh, fw1, ic, FH, FW, IC);
			float g2 = get4d(W, oc0, fh, fw2, ic, FH, FW, IC);

			float m1 = g0 * (d0 - d2);
			float m2 = 0.5f * (d1 + d2) * (g0 + g1 + g2);
			float m3 = 0.5f * (d2 - d1) * (g0 - g1 + g2);
			float m4 = g2 * (d1 - d3);

			v0 += (m1 + m2 + m3);
			v1 += (m2 - m3 - m4);
		}
	}

	get4d(Y, n0, oh0, ow0, oc0, OH, OW, OC) = v0;
	get4d(Y, n0, oh0, ow1, oc0, OH, OW, OC) = v1;
}

#endif


//FH = FW = 3
//sh = sw = 1
//OH % 2 == 0, OW % 2 == 0
//(Y: BLOCK_SIZE*1, X: BLOCK_SIZE*2)
#ifndef CONV_3D_WINOGRAD_KERNEL_W3_V2
#define CONV_3D_WINOGRAD_KERNEL_W3_V2

#define wingrad_v2(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	conv3dWinoGrad_kernel_v2<LB, (1<<LB)>\
		<<< dim3(GM>>LB>>1, GN>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, oc_index, j_index)


template<int LB, int STEP>
__global__ void conv3dWinoGrad_kernel_v2(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W,
	float* __restrict__ Y, int OH, int OW,
	int IC, int OC,
	int ph, int pw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	//prepare for GN = OC
	int oc0 = ((blockIdx.y << LB) + ty) + oc_index;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.x << LB) + tx) << 1) + j_index;
	int OH_OW = OH * OW;
	get_n_oh_ow(j0, n0, oh0, ow0);
	int ow1 = ow0 + 1;

	//compute area---------------------------------------
	float v0 = 0, v1 = 0;
	for (int k = 0; k < 9; k += 3)
	{
		int fh = k / 3, fw0 = k % 3;
		int fw1 = fw0 + 1;
		int fw2 = fw0 + 2;

		int ih = oh0 - ph + fh;
		int iw0 = ow0 - pw + fw0;
		int iw1 = iw0 + 1;
		int iw2 = iw0 + 2;
		int iw3 = iw0 + 3;

		for (int ic = 0; ic < IC; ic++)
		{
			bool ly = (ih >= 0) && (ih < IH);
			bool ly0 = ly && (iw0 >= 0) && (iw0 < IW);
			bool ly1 = ly && (iw1 >= 0) && (iw1 < IW);
			bool ly2 = ly && (iw2 >= 0) && (iw2 < IW);
			bool ly3 = ly && (iw3 >= 0) && (iw3 < IW);

			float d0 = (ly0 ? get4d(X, n0, ih, iw0, ic, IH, IW, IC) : 0);
			float d1 = (ly1 ? get4d(X, n0, ih, iw1, ic, IH, IW, IC) : 0);
			float d2 = (ly2 ? get4d(X, n0, ih, iw2, ic, IH, IW, IC) : 0);
			float d3 = (ly3 ? get4d(X, n0, ih, iw3, ic, IH, IW, IC) : 0);

			float g0 = get4d(W, oc0, fh, fw0, ic, 3, 3, IC);
			float g1 = get4d(W, oc0, fh, fw1, ic, 3, 3, IC);
			float g2 = get4d(W, oc0, fh, fw2, ic, 3, 3, IC);

			float m1 = g0 * (d0 - d2);
			float m2 = 0.5f * (d1 + d2) * (g0 + g1 + g2);
			float m3 = 0.5f * (d2 - d1) * (g0 - g1 + g2);
			float m4 = g2 * (d1 - d3);

			v0 += (m1 + m2 + m3);
			v1 += (m2 - m3 - m4);
		}
	}

	get4d(Y, n0, oh0, ow0, oc0, OH, OW, OC) = v0;
	get4d(Y, n0, oh0, ow1, oc0, OH, OW, OC) = v1;
}

#endif


//FH = FW = 3
//sh = sw = 1
//OH % 2 == 0, OW % 2 == 0
//(Y: BLOCK_SIZE*1, X: BLOCK_SIZE*2)
#ifndef CONV_3D_WINOGRAD_KERNEL_W3_V3
#define CONV_3D_WINOGRAD_KERNEL_W3_V3

#define wingrad_v3(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	conv3dWinoGrad_kernel_v3<LB, (1<<LB)>\
		<<< dim3(GM>>LB>>1, GN>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, oc_index, j_index)

template<int LB, int STEP>
__global__ void conv3dWinoGrad_kernel_v3(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W,
	float* __restrict__ Y, int OH, int OW,
	int IC, int OC,
	int ph, int pw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	//prepare for GN = OC
	int oc0 = ((blockIdx.y << LB) + ty) + oc_index;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.x << LB) + tx) << 1) + j_index;
	int OH_OW = OH * OW;
	get_n_oh_ow(j0, n0, oh0, ow0);
	int ow1 = ow0 + 1;

	//compute area---------------------------------------
	float v0 = 0, v1 = 0;
	for (int k = 0; k < 9; k += 3)
	{
		int fh = k / 3, fw0 = k % 3;
		int fw1 = fw0 + 1;
		int fw2 = fw0 + 2;

		int ih = oh0 - ph + fh;
		int iw0 = ow0 - pw + fw0;
		int iw1 = iw0 + 1;
		int iw2 = iw0 + 2;
		int iw3 = iw0 + 3;

		bool ly = (ih >= 0) && (ih < IH);
		bool ly0 = ly && (iw0 >= 0) && (iw0 < IW);
		bool ly1 = ly && (iw1 >= 0) && (iw1 < IW);
		bool ly2 = ly && (iw2 >= 0) && (iw2 < IW);
		bool ly3 = ly && (iw3 >= 0) && (iw3 < IW);

		for (int oic = 0, OIC = (IC >> LB); oic < OIC; oic++)
		{
#pragma unroll
			for (int iic = 0; iic < STEP; iic++) 
			{
				int ic = (oic << LB) + iic;

				float d0 = (ly0 ? get4d(X, n0, ih, iw0, ic, IH, IW, IC) : 0);
				float d1 = (ly1 ? get4d(X, n0, ih, iw1, ic, IH, IW, IC) : 0);
				float d2 = (ly2 ? get4d(X, n0, ih, iw2, ic, IH, IW, IC) : 0);
				float d3 = (ly3 ? get4d(X, n0, ih, iw3, ic, IH, IW, IC) : 0);

				float g0 = get4d(W, oc0, fh, fw0, ic, 3, 3, IC);
				float g1 = get4d(W, oc0, fh, fw1, ic, 3, 3, IC);
				float g2 = get4d(W, oc0, fh, fw2, ic, 3, 3, IC);

				float m1 = g0 * (d0 - d2);
				float m2 = 0.5f * (d1 + d2) * (g0 + g1 + g2);
				float m3 = 0.5f * (d2 - d1) * (g0 - g1 + g2);
				float m4 = g2 * (d1 - d3);

				v0 += (m1 + m2 + m3);
				v1 += (m2 - m3 - m4);
			}
		}
	}

	get4d(Y, n0, oh0, ow0, oc0, OH, OW, OC) = v0;
	get4d(Y, n0, oh0, ow1, oc0, OH, OW, OC) = v1;
}

#endif


//FH = FW = 3
//sh = sw = 1
//OH % 2 == 0, OW % 2 == 0
//(Y: BLOCK_SIZE*1, X: BLOCK_SIZE*2)
#ifndef CONV_3D_WINOGRAD_KERNEL_W3_V4
#define CONV_3D_WINOGRAD_KERNEL_W3_V4

#define wingrad_v4(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	conv3dWinoGrad_kernel_v4<LB, (1<<LB)>\
		<<< dim3(GM>>LB>>1, GN>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, oc_index, j_index)


//LB = 4: Size = 1.125, Time = 7.56881 msec, Performace = 319.194 GFlop/s
template<int LB, int STEP>
__global__ void conv3dWinoGrad_kernel_v4(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W,
	float* __restrict__ Y, int OH, int OW,
	int IC, int OC,
	int ph, int pw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	__shared__ float4 Ds[1 << LB][1 << LB];
	__shared__ float4 Gs[1 << LB][1 << LB];

	//prepare for GN = OC
	int oc0 = ((blockIdx.y << LB) + ty) + oc_index;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.x << LB) + tx) << 1) + j_index;
	int OH_OW = OH * OW;
	get_n_oh_ow(j0, n0, oh0, ow0);
	int ow1 = ow0 + 1;

	//compute area---------------------------------------
	float v0 = 0, v1 = 0;
	for (int fhw = 0; fhw < 9; fhw += 3)
	{
		int fh = fhw / 3, fw0 = fhw % 3;
		int fw1 = fw0 + 1;
		int fw2 = fw0 + 2;

		int ih = oh0 - ph + fh;
		int iw0 = ow0 - pw + fw0;
		int iw1 = iw0 + 1;
		int iw2 = iw0 + 2;
		int iw3 = iw0 + 3;

		bool ly = (ih >= 0) && (ih < IH);
		bool ly0 = ly && (iw0 >= 0) && (iw0 < IW);
		bool ly1 = ly && (iw1 >= 0) && (iw1 < IW);
		bool ly2 = ly && (iw2 >= 0) && (iw2 < IW);
		bool ly3 = ly && (iw3 >= 0) && (iw3 < IW);

		for (int oic = 0, OIC = (IC >> LB); oic < OIC; oic++)
		{
			int xic = (oic << LB) + ty;//with the same tx -> d0
			float d0 = (ly0 ? get4d(X, n0, ih, iw0, xic, IH, IW, IC) : 0);
			float d1 = (ly1 ? get4d(X, n0, ih, iw1, xic, IH, IW, IC) : 0);
			float d2 = (ly2 ? get4d(X, n0, ih, iw2, xic, IH, IW, IC) : 0);
			float d3 = (ly3 ? get4d(X, n0, ih, iw3, xic, IH, IW, IC) : 0);
			Ds[ty][tx] = float4{ d0, d1, d2, d3 };

			int wic = (oic << LB) + tx;//with the same ty -> oc
			float g0 = get4d(W, oc0, fh, fw0, wic, 3, 3, IC);
			float g1 = get4d(W, oc0, fh, fw1, wic, 3, 3, IC);
			float g2 = get4d(W, oc0, fh, fw2, wic, 3, 3, IC);
			Gs[tx][ty] = float4{ g0, g1, g2, 0 };
			__syncthreads();

#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 d = Ds[ik][tx];
				float d0 = d.x, d1 = d.y, d2 = d.z, d3 = d.w;

				float4 g = Gs[ik][ty];
				float g0 = g.x, g1 = g.y, g2 = g.z;

				float m1 = g0 * (d0 - d2);
				float m2 = 0.5f * (d1 + d2) * (g0 + g1 + g2);
				float m3 = 0.5f * (d2 - d1) * (g0 - g1 + g2);
				float m4 = g2 * (d1 - d3);

				v0 += (m1 + m2 + m3);
				v1 += (m2 - m3 - m4);
			}
			__syncthreads();
		}
	}

	get4d(Y, n0, oh0, ow0, oc0, OH, OW, OC) = v0;
	get4d(Y, n0, oh0, ow1, oc0, OH, OW, OC) = v1;
}

#endif


//OH % 2 == 0, OW % 2 == 0
//(Y: BLOCK_SIZE*1, X: BLOCK_SIZE*2)
#ifndef CONV_3D_WINOGRAD_KERNEL_W3_V5
#define CONV_3D_WINOGRAD_KERNEL_W3_V5

#define wingrad_v5(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	conv3dWinoGrad_kernel_v5<LB, (1<<LB)>\
		<<< dim3(GM>>LB>>1, GN>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, oc_index, j_index)

//LB = 4: Size = 1.125, Time = 7.56881 msec, Performace = 319.194 GFlop/s
template<int LB, int STEP>
__global__ void conv3dWinoGrad_kernel_v5(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W,
	float* __restrict__ Y, int OH, int OW,
	int IC, int OC,
	int ph, int pw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	__shared__ float4 Ds[1 << LB][1 << LB];
	__shared__ float4 Gs[1 << LB][1 << LB];

	//prepare for GN = OC
	int oc0 = ((blockIdx.y << LB) + ty) + oc_index;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.x << LB) + tx) << 1) + j_index;
	int OH_OW = OH * OW;
	get_n_oh_ow(j0, n0, oh0, ow0);
	int ow1 = ow0 + 1;

	//compute area---------------------------------------
	float v0 = 0, v1 = 0;

#pragma unroll
	for (int fh = 0; fh < 3; fh++)
	{
		int ih = oh0 - ph + fh;
		int iw0 = ow0 - pw;
		int iw1 = iw0 + 1;
		int iw2 = iw0 + 2;
		int iw3 = iw0 + 3;

		bool ly = (ih >= 0) && (ih < IH);
		bool ly0 = ly && (iw0 >= 0) && (iw0 < IW);
		bool ly1 = ly && (iw1 >= 0) && (iw1 < IW);
		bool ly2 = ly && (iw2 >= 0) && (iw2 < IW);
		bool ly3 = ly && (iw3 >= 0) && (iw3 < IW);

		for (int oic = 0, OIC = (IC >> LB); oic < OIC; oic++)
		{
			int xic = (oic << LB) + ty;//with the same tx -> d0
			int X0 = ((n0*IH + ih)*IW + iw0)*IC + xic;
			int X1 = X0 + IC, X2 = X1 + IC, X3 = X2 + IC;
			float d0 = (ly0 ? X[X0] : 0);
			float d1 = (ly1 ? X[X1] : 0);
			float d2 = (ly2 ? X[X2] : 0);
			float d3 = (ly3 ? X[X3] : 0);
			Ds[ty][tx] = float4{ d0, d1, d2, d3 };

			int wic = (oic << LB) + tx;//with the same ty -> oc
			int W0 = (oc0 * 3 + fh) * 3 * IC + wic;
			int W1 = W0 + IC, W2 = W1 + IC;
			float g0 = W[W0];
			float g1 = W[W1];
			float g2 = W[W2];
			Gs[tx][ty] = float4{ g0, g1, g2, 0 };
			__syncthreads();

#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 d = Ds[ik][tx];
				float d0 = d.x, d1 = d.y, d2 = d.z, d3 = d.w;

				float4 g = Gs[ik][ty];
				float g0 = g.x, g1 = g.y, g2 = g.z;

				float m1 = g0 * (d0 - d2);
				float m2 = 0.5f * (d1 + d2) * (g0 + g1 + g2);
				float m3 = 0.5f * (d2 - d1) * (g0 - g1 + g2);
				float m4 = g2 * (d1 - d3);

				v0 += (m1 + m2 + m3);
				v1 += (m2 - m3 - m4);
			}
			__syncthreads();
		}
	}

	get4d(Y, n0, oh0, ow0, oc0, OH, OW, OC) = v0;
	get4d(Y, n0, oh0, ow1, oc0, OH, OW, OC) = v1;
}

#endif


//OH % 2 == 0, OW % 2 == 0
//(Y: BLOCK_SIZE*1, X: BLOCK_SIZE*2)
#ifndef CONV_3D_WINOGRAD_KERNEL_W3_V6
#define CONV_3D_WINOGRAD_KERNEL_W3_V6

#define wingrad_v6(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	conv3dWinoGrad_kernel_v6<LB, (1<<LB)>\
		<<< dim3(GM>>LB>>1, GN>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, oc_index, j_index)

//Size = 1.125, Time = 9.58315 msec, Performace = 252.101 GFlop/s
//LB = 4: Size = 1.125, Time = 7.56881 msec, Performace = 319.194 GFlop/s
template<int LB, int STEP>
__global__ void conv3dWinoGrad_kernel_v6(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W,
	float* __restrict__ Y, int OH, int OW,
	int IC, int OC,
	int ph, int pw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	__shared__ float4 Ds[1 << LB][1 << LB];
	__shared__ float4 Gs[1 << LB][1 << LB];

	//prepare for GN = OC
	int oc0 = ((blockIdx.y << LB) + ty) + oc_index;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.x << LB) + tx) << 1) + j_index;
	int OH_OW = OH * OW;
	get_n_oh_ow(j0, n0, oh0, ow0);
	int ow1 = ow0 + 1;

	//compute area---------------------------------------
	float v0 = 0, v1 = 0;

#pragma unroll
	for (int fh = 0; fh < 3; fh++)
	{
		int ih = oh0 - ph + fh;
		int iw0 = ow0 - pw;
		int iw1 = iw0 + 1;
		int iw2 = iw0 + 2;
		int iw3 = iw0 + 3;

		bool ly = (ih >= 0) && (ih < IH);
		bool ly0 = ly && (iw0 >= 0) && (iw0 < IW);
		bool ly1 = ly && (iw1 >= 0) && (iw1 < IW);
		bool ly2 = ly && (iw2 >= 0) && (iw2 < IW);
		bool ly3 = ly && (iw3 >= 0) && (iw3 < IW);

		for (int oic = 0, OIC = (IC >> LB); oic < OIC; oic++)
		{
			int xic = (oic << LB) + ty;//with the same tx -> d0
			int X0 = ((n0*IH + ih)*IW + iw0)*IC + xic;
			int X1 = X0 + IC, X2 = X1 + IC, X3 = X2 + IC;
			float d0 = (ly0 ? X[X0] : 0);
			float d1 = (ly1 ? X[X1] : 0);
			float d2 = (ly2 ? X[X2] : 0);
			float d3 = (ly3 ? X[X3] : 0);
			Ds[ty][tx] = float4{ d0 - d2, d1 + d2, d2 - d1, d1 - d3 };

			int wic = (oic << LB) + tx;//with the same ty -> oc
			int W0 = (oc0 * 3 + fh) * 3 * IC + wic;
			int W1 = W0 + IC, W2 = W1 + IC;
			float g0 = W[W0];
			float g1 = W[W1];
			float g2 = W[W2];
			Gs[tx][ty] = float4{ g0, 0.5f*(g0 + g1 + g2), 0.5f*(g0 - g1 + g2), g2 };
			__syncthreads();

#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 d = Ds[ik][tx], g = Gs[ik][ty];

				float m1 = g.x * d.x;
				float m2 = g.y * d.y;
				float m3 = g.z * d.z;
				float m4 = g.w * d.w;

				v0 += (m1 + m2 + m3);
				v1 += (m2 - m3 - m4);
			}
			__syncthreads();
		}
	}

	get4d(Y, n0, oh0, ow0, oc0, OH, OW, OC) = v0;
	get4d(Y, n0, oh0, ow1, oc0, OH, OW, OC) = v1;
}

#endif


//OH % 2 == 0, OW % 2 == 0
//(Y: BLOCK_SIZE*1, X: BLOCK_SIZE*2)
#ifndef CONV_3D_WINOGRAD_KERNEL_W3_V7
#define CONV_3D_WINOGRAD_KERNEL_W3_V7

#define wingrad_v7(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	conv3dWinoGrad_kernel_v7<LB, (1<<LB)>\
		<<< dim3(GM>>LB>>1, GN>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, IC, OC, ph, pw, oc_index, j_index)

//Size = 1.125, Time = 9.58315 msec, Performace = 252.101 GFlop/s
//LB = 4: Size = 1.125, Time = 7.56881 msec, Performace = 319.194 GFlop/s
template<int LB, int STEP>
__global__ void conv3dWinoGrad_kernel_v7(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W,
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int ph, int pw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	__shared__ float4 Ds[1 << LB][1 << LB];
	__shared__ float4 Gs[1 << LB][1 << LB];

	//prepare for GN = OC
	int oc0 = ((blockIdx.y << LB) + ty) + oc_index;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.x << LB) + tx) << 1) + j_index;
	get_n_oh_ow(j0, n0, oh0, ow0);

	//compute area---------------------------------------
	float2 v0 = float2{ 0, 0 };

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
			int xic = (oic << LB) + ty;//with the same tx -> d0
			int X0 = ((n0*IH + ih)*IW + iw0)*IC + xic;
			int X1 = X0 + IC, X2 = X1 + IC, X3 = X2 + IC;
			float d0 = (ly0 ? X[X0] : 0);
			float d1 = (ly1 ? X[X1] : 0);
			float d2 = (ly2 ? X[X2] : 0);
			float d3 = (ly3 ? X[X3] : 0);
			Ds[ty][tx] = winograd_d(d0, d1, d2, d3);

			int wic = (oic << LB) + tx;//with the same ty -> oc
			int W0 = (oc0 * 9 + fh * 3)*IC + wic;
			int W1 = W0 + IC, W2 = W1 + IC;
			float g0 = W[W0];
			float g1 = W[W1];
			float g2 = W[W2];
			Gs[tx][ty] = winograd_g(g0, g1, g2);
			__syncthreads();

#pragma unroll
			for (int ik = 0; ik < STEP; ik++) {
				float4 d = Ds[ik][tx];
				float4 g = Gs[ik][ty];
				winograd_add(v0, g, d);
			}
			__syncthreads();
		}
	}

	const int Y0 = j0 * OC + oc0;
	const int Y1 = Y0 + OC;

	Y[Y0] = v0.x;
	Y[Y1] = v0.y;
}

#endif


//OH % 2 == 0, OW % 2 == 0
//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*2)
#ifndef CONV_3D_WINOGRAD_KERNEL_W3_V8
#define CONV_3D_WINOGRAD_KERNEL_W3_V8

#define wingrad_v8(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	conv3dWinoGrad_kernel_v8<LB, (1<<LB)>\
		<<< dim3(GM>>LB>>1, GN>>LB>>1), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, IC, OC, ph, pw, oc_index, j_index)

//LB = 4: Size = 1.125, Time = 5.09251 msec, Performace = 474.406 GFlop/s
template<int LB, int STEP>
__global__ void conv3dWinoGrad_kernel_v8(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int ph, int pw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	__shared__ float4 Gs[1 << LB][2 << LB];//{oc0, oc1}
	__shared__ float4 Ds[1 << LB][1 << LB];

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 1) + oc_index;
	CW += oc0;//CW[0, 0, 0, oc0]
	const int Wstride = IC * OC;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.x << LB) + tx) << 1) + j_index;
	get_n_oh_ow(j0, n0, oh0, ow0);

	//compute area---------------------------------------
	float2 v0 = float2{ 0, 0 };
	float2 v1 = float2{ 0, 0 };

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
			int xic = (oic << LB) + ty;//with the same tx -> d0
			int X0 = ((n0*IH + ih)*IW + iw0)*IC + xic;
			int X1 = X0 + IC, X2 = X1 + IC, X3 = X2 + IC;
			float d0 = (ly0 ? X[X0] : 0);
			float d1 = (ly1 ? X[X1] : 0);
			float d2 = (ly2 ? X[X2] : 0);
			float d3 = (ly3 ? X[X3] : 0);
			Ds[ty][tx] = winograd_d(d0, d1, d2, d3);

			int wic = (oic << LB) + tx;//with the same ty -> oc
			int W0 = ((fh * 3)*IC + wic)*OC;
			int W1 = W0 + Wstride, W2 = W1 + Wstride;
			float2 g0 = *(float2*)(CW + W0);
			float2 g1 = *(float2*)(CW + W1);
			float2 g2 = *(float2*)(CW + W2);
			Gs[tx][(ty << 1)    ] = winograd_g(g0.x, g1.x, g2.x);
			Gs[tx][(ty << 1) + 1] = winograd_g(g0.y, g1.y, g2.y);
			__syncthreads();

#pragma unroll
			for (int ik = 0; ik < STEP; ik++) 
			{
				float4 g0 = Gs[ik][(ty << 1)], g1 = Gs[ik][(ty << 1) + 1];
				float4 d0 = Ds[ik][tx];

				//transposed 
				winograd_add(v0, g0, d0);
				winograd_add(v1, g1, d0);
			}
			__syncthreads();
		}
	}

	const int Y0 = j0 * OC + oc0;
	const int Y1 = Y0 + OC;

	*(float2*)(Y + Y0) = float2{ v0.x, v1.x };
	*(float2*)(Y + Y1) = float2{ v0.y, v1.y };
}

#endif


//OH % 2 == 0, OW % 2 == 0
//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*2)
#ifndef CONV_3D_WINOGRAD_KERNEL_W3_V9
#define CONV_3D_WINOGRAD_KERNEL_W3_V9

#define wingrad_v9(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	conv3dWinoGrad_kernel_v9<LB, (1<<LB)>\
		<<< dim3(GM>>LB>>1, GN>>LB>>1), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, IC, OC, ph, pw, oc_index, j_index)

//Size = 1.125, Time = 9.58315 msec, Performace = 252.101 GFlop/s
//LB = 4: Size = 1.125, Time = 5.09251 msec, Performace = 474.406 GFlop/s
template<int LB, int STEP>
__global__ void conv3dWinoGrad_kernel_v9(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int ph, int pw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	__shared__ float4 Gs[1 << LB][2 << LB];//{oc0, oc1}
	__shared__ float4 Ds[1 << LB][1 << LB];

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 1) + oc_index;
	CW += oc0;//CW[0, 0, 0, oc0]
	const int Wstride = IC * OC;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.x << LB) + tx) << 1) + j_index;
	get_n_oh_ow(j0, n0, oh0, ow0);

	//compute area---------------------------------------
	float2 v0 = float2{ 0, 0 };
	float2 v1 = float2{ 0, 0 };

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
			int xic = (oic << LB) + ty;//with the same tx -> d0
			int X0 = ((n0*IH + ih)*IW + iw0)*IC + xic;
			int X1 = X0 + IC, X2 = X1 + IC, X3 = X2 + IC;
			float d0 = (ly0 ? X[X0] : 0);
			float d1 = (ly1 ? X[X1] : 0);
			float d2 = (ly2 ? X[X2] : 0);
			float d3 = (ly3 ? X[X3] : 0);
			Ds[ty][tx] = winograd_d(d0, d1, d2, d3);

			int wic = (oic << LB) + tx;//with the same ty -> oc
			int W0 = ((fh * 3)*IC + wic)*OC;
			int W1 = W0 + Wstride, W2 = W1 + Wstride;
			float2 g0 = *(float2*)(CW + W0);
			float2 g1 = *(float2*)(CW + W1);
			float2 g2 = *(float2*)(CW + W2);
			Gs[tx][(ty << 1)] = winograd_g(g0.x, g1.x, g2.x);
			Gs[tx][(ty << 1) + 1] = winograd_g(g0.y, g1.y, g2.y);
			__syncthreads();

#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 g0 = Gs[ik][(ty << 1)], g1 = Gs[ik][(ty << 1) + 1];
				float4 d0 = Ds[ik][tx];

				float m00_1 = g0.x * d0.x;
				float m00_2 = g0.y * d0.y;
				float m00_3 = g0.z * d0.z;
				float m00_4 = g0.w * d0.w;

				float m10_2 = g1.y * d0.y;
				float m10_3 = g1.z * d0.z;
				float m10_4 = g1.w * d0.w;
				float m10_1 = g1.x * d0.x;

				v0.x += (m00_1 + m00_2 + m00_3);
				v0.y += (m10_1 + m10_2 + m10_3);

				v1.x += (m00_2 - m00_3 - m00_4);
				v1.y += (m10_2 - m10_3 - m10_4);
			}
			__syncthreads();
		}
	}

	const int Y0 = j0 * OC + oc0;
	const int Y1 = Y0 + OC;

	*(float2*)(Y + Y0) = v0;
	*(float2*)(Y + Y1) = v1;
}

#endif


//OH % 2 == 0, OW % 2 == 0
//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*4)
#ifndef CONV_3D_WINOGRAD_KERNEL_W3_V10
#define CONV_3D_WINOGRAD_KERNEL_W3_V10

#define wingrad_v10(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	conv3dWinoGrad_kernel_v10<LB, (1<<LB)>\
		<<< dim3(GM>>LB>>2, GN>>LB>>1), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, IC, OC, ph, pw, oc_index, j_index)

//Size = 1.125, Time = 9.58315 msec, Performace = 252.101 GFlop/s
//LB = 4: Size = 1.125, Time = 7.56881 msec, Performace = 319.194 GFlop/s
template<int LB, int STEP>
__global__ void conv3dWinoGrad_kernel_v10(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int ph, int pw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	__shared__ float4 Gs[1 << LB][2 << LB];//{oc0, oc1}
	__shared__ float4 Ds[1 << LB][2 << LB];//{ j0,  j1}

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 2) + oc_index;
	CW += oc0;//CW[0, 0, 0, oc0]
	const int Wstride = IC * OC;

	//OH_OW = 4*x
	//(1) for ni:
	//ni = (j0 + i) / (OH_OW), j0 = 4*y
	//ni = (4*y + i) / (4*x) = y/x
	//n0 = n1 = n2 = n3
	//(2) for iwi:
	//iwi = (j0 + i) % (OW)
	//iwi = (4*y + i) % (2*x)
	//iw0 = iw1 - 1
	//iw2 = iw3 - 1
	//(3) for ihi
	//ihi = ((j0 + i) % (OH_OW)) / OW
	//ih0 = ih1
	//ih2 = ih3

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.x << LB) + tx) << 2) + j_index, j2 = j0 + 2;
	get_n_oh_ow(j0, n0, oh0, ow0);
	get_n_oh_ow(j2, n2, oh2, ow2);

	//compute area---------------------------------------
	float4 v0 = float4{ 0, 0 };
	float4 v1 = float4{ 0, 0 };

#pragma unroll
	for (int fh = 0; fh < 3; fh++)
	{
		int ih0 = oh0 - ph + fh; bool ly0 = (ih0 >= 0) && (ih0 < IH);
		int iw0 = ow0 - pw; bool ly00 = ly0 && (iw0 >= 0) && (iw0 < IW);
		int iw01 = iw0 + 1; bool ly01 = ly0 && (iw01 >= 0) && (iw01 < IW);
		int iw02 = iw0 + 2; bool ly02 = ly0 && (iw02 >= 0) && (iw02 < IW);
		int iw03 = iw0 + 3; bool ly03 = ly0 && (iw03 >= 0) && (iw03 < IW);

		int ih2 = oh2 - ph + fh; bool ly2 = (ih2 >= 0) && (ih2 < IH);
		int iw2 = ow2 - pw; bool ly20 = ly2 && (iw2 >= 0) && (iw2 < IW);
		int iw21 = iw2 + 1; bool ly21 = ly2 && (iw21 >= 0) && (iw21 < IW);
		int iw22 = iw2 + 2; bool ly22 = ly2 && (iw22 >= 0) && (iw22 < IW);
		int iw23 = iw2 + 3; bool ly23 = ly2 && (iw23 >= 0) && (iw23 < IW);
		
		for (int oic = 0, OIC = (IC >> LB); oic < OIC; oic++)
		{
			int xic = (oic << LB) + ty;

			int X00 = ((n0*IH + ih0)*IW + iw0)*IC + xic;
			int X01 = X00 + IC;
			int X02 = X01 + IC;
			int X03 = X02 + IC;
			float d00 = (ly00 ? X[X00] : 0);
			float d01 = (ly01 ? X[X01] : 0);
			float d02 = (ly02 ? X[X02] : 0);
			float d03 = (ly03 ? X[X03] : 0);
			Ds[ty][(tx << 1)] = winograd_d(d00, d01, d02, d03);

			int X20 = ((n0*IH + ih2)*IW + iw2)*IC + xic;
			int X21 = X20 + IC;
			int X22 = X21 + IC;
			int X23 = X22 + IC;
			float d20 = (ly20 ? X[X20] : 0);
			float d21 = (ly21 ? X[X21] : 0);
			float d22 = (ly22 ? X[X22] : 0);
			float d23 = (ly23 ? X[X23] : 0);
			Ds[ty][(tx << 1) + 1] = winograd_d(d20, d21, d22, d23);

			int wic = (oic << LB) + tx;
			int W0 = ((fh * 3)*IC + wic)*OC;
			int W1 = W0 + Wstride, W2 = W1 + Wstride;
			float2 g0 = *(float2*)(CW + W0);
			float2 g1 = *(float2*)(CW + W1);
			float2 g2 = *(float2*)(CW + W2);
			Gs[tx][(ty << 1)] = winograd_g(g0.x, g1.x, g2.x);
			Gs[tx][(ty << 1) + 1] = winograd_g(g0.y, g1.y, g2.y);
			__syncthreads();

#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 g0 = Gs[ik][(ty << 1)], g1 = Gs[ik][(ty << 1) + 1];
				float4 d0 = Ds[ik][(tx << 1)], d1 = Ds[ik][(tx << 1) + 1];

				float m00_1 = g0.x * d0.x; float m01_1 = g0.x * d1.x;
				float m00_2 = g0.y * d0.y; float m01_2 = g0.y * d1.y;
				float m00_3 = g0.z * d0.z; float m01_3 = g0.z * d1.z;
				float m00_4 = g0.w * d0.w; float m01_4 = g0.w * d1.w;

				float m10_2 = g1.y * d0.y; float m11_1 = g1.x * d1.x;
				float m10_3 = g1.z * d0.z; float m11_2 = g1.y * d1.y;
				float m10_4 = g1.w * d0.w; float m11_3 = g1.z * d1.z;
				float m10_1 = g1.x * d0.x; float m11_4 = g1.w * d1.w;

				v0.x += (m00_1 + m00_2 + m00_3);
				v1.x += (m00_2 - m00_3 - m00_4);

				v0.y += (m10_1 + m10_2 + m10_3);
				v1.y += (m10_2 - m10_3 - m10_4);
			}
			__syncthreads();
		}
	}

	const int Y0 = j0 * OC + oc0;
	const int Y1 = Y0 + OC;

	*(float2*)(Y + Y0) = v0;
	*(float2*)(Y + Y1) = v1;
}

#endif


//OH % 2 == 0, OW % 2 == 0
//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*2)
#ifndef CONV_3D_WINOGRAD_KERNEL_W3_V11
#define CONV_3D_WINOGRAD_KERNEL_W3_V11

#define wingrad_v11(stream, LB, oc_index, j_index, X, IH, IW, CW, Y, OH, OW, IC, OC, ph, pw, GN, GM) \
	conv3dWinoGrad_kernel_v11<LB, (1<<LB)>\
		<<< dim3(GM>>LB>>1, GN>>LB>>1), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, CW, Y, (OH*OW), OW, IC, OC, ph, pw, oc_index, j_index)

//LB = 4: Size = 1.125, Time = 5.0659 msec, Performace = 476.898 GFlop/s
template<int LB, int STEP>
__global__ void conv3dWinoGrad_kernel_v11(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ CW,//[FH, FW, IC, OC]
	float* __restrict__ Y, int OH_OW, int OW,
	int IC, int OC,
	int ph, int pw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	__shared__ float4 Gs[1 << LB][2 << LB];
	__shared__ float4 Ds[1 << LB][1 << LB];

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 1) + oc_index;
	CW += oc0;//CW[0, 0, 0, oc0]
	const int Wstride = IC * OC;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.x << LB) + tx) << 1) + j_index;
	get_n_oh_ow(j0, n0, oh0, ow0);

	//compute area---------------------------------------
	float2 v0 = float2{ 0, 0 };
	float2 v1 = float2{ 0, 0 };

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
			int xic = (oic << LB) + ty;//with the same tx -> d0
			int X0 = ((n0*IH + ih)*IW + iw0)*IC + xic;
			int X1 = X0 + IC, X2 = X1 + IC, X3 = X2 + IC;
			float d0 = (ly0 ? X[X0] : 0);
			float d1 = (ly1 ? X[X1] : 0);
			float d2 = (ly2 ? X[X2] : 0);
			float d3 = (ly3 ? X[X3] : 0);
			Ds[ty][tx] = winograd_d(d0, d1, d2, d3);

			int wic = (oic << LB) + tx;//with the same ty -> oc
			int W0 = ((fh * 3)*IC + wic)*OC;
			int W1 = W0 + Wstride, W2 = W1 + Wstride;
			float2 g0 = *(float2*)(CW + W0);
			float2 g1 = *(float2*)(CW + W1);
			float2 g2 = *(float2*)(CW + W2);
			Gs[tx][ty] = winograd_g(g0.x, g1.x, g2.x);
			Gs[tx][ty + STEP] = winograd_g(g0.y, g1.y, g2.y);
			__syncthreads();

#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 g0 = Gs[ik][ty], g1 = Gs[ik][ty + STEP];
				float4 d0 = Ds[ik][tx];

				float m00_1 = g0.x * d0.x;
				float m00_2 = g0.y * d0.y;
				float m00_3 = g0.z * d0.z;
				float m00_4 = g0.w * d0.w;

				float m10_1 = g1.x * d0.x;
				float m10_2 = g1.y * d0.y;
				float m10_3 = g1.z * d0.z;
				float m10_4 = g1.w * d0.w;

				v0.x += (m00_1 + m00_2 + m00_3);
				v0.y += (m10_1 + m10_2 + m10_3);

				v1.x += (m00_2 - m00_3 - m00_4);
				v1.y += (m10_2 - m10_3 - m10_4);
			}
			__syncthreads();
		}
	}

	const int Y0 = j0 * OC + oc0;
	const int Y1 = Y0 + OC;

	*(float2*)(Y + Y0) = v0;
	*(float2*)(Y + Y1) = v1;
}

#endif

