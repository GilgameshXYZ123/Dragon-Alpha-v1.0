#pragma once

#ifndef CONV_3D_GEMM_KERNEL_TEXTURE_H
#define CONV_3D_GEMM_KERNEL_TEXTURE_H


#ifndef CONV_3D_GEMM_KERNEL_TEXTURE_CALL
#define CONV_3D_GEMM_KERNEL_TEXTURE_CALL

#define conv3dGemm_k88_tex(stream, LB, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_kernel_8_8_texture<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>3, GN>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, oc_index, j_index)

#define conv3dGemm_k48_tex(stream, LB, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_kernel_4_8_texture<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>3, GN>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, oc_index, j_index)

#define conv3dPure_k44_tex(stream, LB, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dPure_kernel_4_4_texture<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>2, GN>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, oc_index, j_index)

#define conv3dPure_k42_tex(stream, LB, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dPure_kernel_4_2_texture<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>1, GN>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, oc_index, j_index)

#define conv3dPure_k24_tex(stream, LB, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dPure_kernel_2_4_texture<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>2, GN>>LB>>1), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, oc_index, j_index)

#define conv3dGemm_k22_tex(stream, LB, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_kernel_2_2_texture<LB, (1<<LB)>\
		<<< dim3(GM>>LB>>1, GN>>LB>>1), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, oc_index, j_index)

#define conv3dGemm_k41_tex(stream, LB, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_kernel_4_1_texture<LB, (1<<LB)>\
		<<< dim3(GM>>LB, GN>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, oc_index, j_index)

#define conv3dGemm_k14_tex(stream, LB, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_kernel_1_4_texture<LB, (1<<LB)>\
		<<< dim3(GM>>LB>>2, GN>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, oc_index, j_index)

#define conv3dGemm_k21_tex(stream, LB, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_kernel_2_1_texture<LB, (1<<LB)>\
		<<< dim3(GM>>LB, GN>>LB>>1), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, oc_index, j_index)

#define conv3dGemm_k12_tex(stream, LB, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_kernel_1_2_texture<LB, (1<<LB)>\
		<<< dim3(GM>>LB>>1, GN>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, oc_index, j_index)

#define conv3dGemm_k11_tex(stream, LB, oc_index, j_index, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, GN, GM) \
	conv3dGemm_kernel_1_1_texture<LB, (1<<LB)>\
		<<< dim3(GM>>LB, GN>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw, oc_index, j_index)

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*4), IC % (BLOCK_SIZE/2)
//LB = 4: IC % 8 == 0
#ifndef CONV_3D_GEMM_KERNEL_8_8_TEXTURE
#define CONV_3D_GEMM_KERNEL_8_8_TEXTURE

//LB = 4: Size = 1, Time = 1.647 msec, Performace = 1303.88 GFlop/s
//LB = 3: Size = 1, Time = 2.082 msec, Performace = 1031.45 GFlop/s
template<int LB, int STEP>
__global__ void Vconv3dGemm_kernel_8_8_texture(
	cudaTextureObject_t X, int IH, int IW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ Y, int OH, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GK = FH * FW * IC
	const int FW_IC = FW * IC, GK = FH * FW_IC;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.y << LB) + ty) << 3) + oc_index;
	int toc0 = (oc0 + ((tx >= STEP) << 2)) * GK;
	int toc1 = toc0 + GK, toc2 = toc1 + GK, toc3 = toc2 + GK;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.x << LB) + tx) << 3) + j_index;
	int tj0 = j0 + ((ty >= STEP) << 2);
	int tj1 = tj0 + 1, tj2 = tj0 + 2, tj3 = tj0 + 3;
	int OH_OW = OH * OW;
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	get_n_oh_ow(tj1, tn1, toh1, tow1);
	get_n_oh_ow(tj2, tn2, toh2, tow2);
	get_n_oh_ow(tj3, tn3, toh3, tow3);
	toh0 = toh0 * sh - ph, tow0 = tow0 * sw - pw;
	toh1 = toh1 * sh - ph, tow1 = tow1 * sw - pw;
	toh2 = toh2 * sh - ph, tow2 = tow2 * sw - pw;
	toh3 = toh3 * sh - ph, tow3 = tow3 * sw - pw;
	const int Xoffset0 = ((tn0*IH + toh0)*IW + tow0)*IC;
	const int Xoffset1 = ((tn1*IH + toh1)*IW + tow1)*IC;
	const int Xoffset2 = ((tn2*IH + toh2)*IW + tow2)*IC;
	const int Xoffset3 = ((tn3*IH + toh3)*IW + tow3)*IC;

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = tx - ((tx >= STEP) << LB >> 1);//k = ((fh*FW) + fw)*IC + ic
	Ws[buf][tx][ty].x = W[toc0 + W_k];
	Ws[buf][tx][ty].y = W[toc1 + W_k];
	Ws[buf][tx][ty].z = W[toc2 + W_k];
	Ws[buf][tx][ty].w = W[toc3 + W_k];

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = ty - ((ty >= STEP) << LB >> 1), X_fh, X_fw;
	get_X_fh_fw(X_k, X_fh, X_fw);
	bool lx0 = (toh0 >= -X_fh) && (toh0 < IH - X_fh) && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
	bool lx1 = (toh1 >= -X_fh) && (toh1 < IH - X_fh) && (tow1 >= -X_fw) && (tow1 < IW - X_fw);
	bool lx2 = (toh2 >= -X_fh) && (toh2 < IH - X_fh) && (tow2 >= -X_fw) && (tow2 < IW - X_fw);
	bool lx3 = (toh3 >= -X_fh) && (toh3 < IH - X_fh) && (tow3 >= -X_fw) && (tow3 < IW - X_fw);
	int IW_IC = IW * IC, xoffset = (X_fh * IW_IC) + X_k;//X[n, ih, iw, ic]
	Xs[buf][ty][tx].x = lx0 * tex1Dfetch<float>(X, Xoffset0 + xoffset);
	Xs[buf][ty][tx].y = lx1 * tex1Dfetch<float>(X, Xoffset1 + xoffset);
	Xs[buf][ty][tx].z = lx2 * tex1Dfetch<float>(X, Xoffset2 + xoffset);
	Xs[buf][ty][tx].w = lx3 * tex1Dfetch<float>(X, Xoffset3 + xoffset);
	__syncthreads();

	//compute area----------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK << 1 >> LB); ok < OK; ok++)//OK = GK/STEP
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];
			float4 a0 = Ws[buf][ik][ty], a1 = Ws[buf][ik + STEP][ty];

			simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
			simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
			simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
			simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
			simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);
			simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
			simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
			simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
		}
		buf ^= 1;

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		Ws[buf][tx][ty].x = W[toc0 + W_k];
		Ws[buf][tx][ty].y = W[toc1 + W_k];
		Ws[buf][tx][ty].z = W[toc2 + W_k];
		Ws[buf][tx][ty].w = W[toc3 + W_k];

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		get_X_fh_fw(X_k, X_fh, X_fw);
		bool lx0 = (toh0 >= -X_fh) && (toh0 < IH - X_fh) && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
		bool lx1 = (toh1 >= -X_fh) && (toh1 < IH - X_fh) && (tow1 >= -X_fw) && (tow1 < IW - X_fw);
		bool lx2 = (toh2 >= -X_fh) && (toh2 < IH - X_fh) && (tow2 >= -X_fw) && (tow2 < IW - X_fw);
		bool lx3 = (toh3 >= -X_fh) && (toh3 < IH - X_fh) && (tow3 >= -X_fw) && (tow3 < IW - X_fw);
		int xoffset = (X_fh * IW_IC) + X_k;
		Xs[buf][ty][tx].x = lx0 * tex1Dfetch<float>(X, Xoffset0 + xoffset);
		Xs[buf][ty][tx].y = lx1 * tex1Dfetch<float>(X, Xoffset1 + xoffset);
		Xs[buf][ty][tx].z = lx2 * tex1Dfetch<float>(X, Xoffset2 + xoffset);
		Xs[buf][ty][tx].w = lx3 * tex1Dfetch<float>(X, Xoffset3 + xoffset);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];
		float4 a0 = Ws[buf][ik][ty], a1 = Ws[buf][ik + STEP][ty];

		simdMM4(v0, b0.x, a0); simdMM4(v1, b0.x, a1);
		simdMM4(v2, b0.y, a0); simdMM4(v3, b0.y, a1);
		simdMM4(v4, b0.z, a0); simdMM4(v5, b0.z, a1);
		simdMM4(v6, b0.w, a0); simdMM4(v7, b0.w, a1);
		simdMM4(v8, b1.x, a0); simdMM4(v9, b1.x, a1);
		simdMM4(v10, b1.y, a0); simdMM4(v11, b1.y, a1);
		simdMM4(v12, b1.z, a0); simdMM4(v13, b1.z, a1);
		simdMM4(v14, b1.w, a0); simdMM4(v15, b1.w, a1);
	}

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;
	int j4 = j3 + OC, j5 = j4 + OC, j6 = j5 + OC, j7 = j6 + OC;

	*(float4*)(Y + j0) = v0;  *(float4*)(Y + j0 + 4) = v1;
	*(float4*)(Y + j1) = v2;  *(float4*)(Y + j1 + 4) = v3;
	*(float4*)(Y + j2) = v4;  *(float4*)(Y + j2 + 4) = v5;
	*(float4*)(Y + j3) = v6;  *(float4*)(Y + j3 + 4) = v7;
	*(float4*)(Y + j4) = v8;  *(float4*)(Y + j4 + 4) = v9;
	*(float4*)(Y + j5) = v10; *(float4*)(Y + j5 + 4) = v11;
	*(float4*)(Y + j6) = v12; *(float4*)(Y + j6 + 4) = v13;
	*(float4*)(Y + j7) = v14; *(float4*)(Y + j7 + 4) = v15;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*4), IC % (BLOCK_SIZE/2)
//LB = 4: IC % 8 == 0
#ifndef CONV_3D_PURE_KERNEL_4_4_TEXTURE
#define CONV_3D_PURE_KERNEL_4_4_TEXTURE

//LB = 4: Size = 1, Time = 2.358 msec, Performace = 910.722 GFlop/s
//LB = 3: Size = 1, Time = 3.666 msec, Performace = 585.784 GFlop/s
template<int LB, int STEP>
__global__ void conv3dPure_kernel_4_4_texture(
	cudaTextureObject_t X, int IH, int IW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ Y, int OH, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2 Xs[2][1 << LB >> 1][(2 << LB) + 2];

	const int oc0 = (((blockIdx.y << LB) + ty) << 2) + oc_index;
	const int GK = FH * FW * IC;
	const int toc0 = (((tx & 1) << 1) + oc0) * GK, toc1 = toc0 + GK;

	int j0 = (((blockIdx.x << LB) + tx) << 2) + j_index;
	int tj0 = j0 + ((ty & 1) << 1), tj1 = tj0 + 1;
	const int OH_OW = OH * OW;
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	get_n_oh_ow(tj1, tn1, toh1, tow1);
	const int tihs0 = toh0 * sh - ph, tiws0 = tow0 * sw - pw;
	const int tihs1 = toh1 * sh - ph, tiws1 = tow1 * sw - pw;
	const int Xoffset0 = (((tn0 *IH) + tihs0) * IW + tiws0) * IC;
	const int Xoffset1 = (((tn1 *IH) + tihs1) * IW + tiws1) * IC;

	//compute area----------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0);
	float4 v1 = make_float4(0, 0, 0, 0);
	float4 v2 = make_float4(0, 0, 0, 0);
	float4 v3 = make_float4(0, 0, 0, 0);

	const int Ws_x = (tx >> 1), Ws_y = (ty << 1) + (tx & 1);
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);
	const int OIC = (IC << 1 >> LB);
	for (int fh = 0; fh < FH; fh++) {
		for (int fw = 0; fw < FW; fw++, W += IC)
		{
			int Wic = tx >> 1;
			Ws[buf][Ws_x][Ws_y].x = W[toc0 + Wic];
			Ws[buf][Ws_x][Ws_y].y = W[toc1 + Wic];

			int Xic = ty >> 1;
			bool lx0 = (tihs0 >= -fh) && (tihs0 < IH - fh) && (tiws0 >= -fw) && (tiws0 < IW - fw);
			bool lx1 = (tihs1 >= -fh) && (tihs1 < IH - fh) && (tiws1 >= -fw) && (tiws1 < IW - fw);
			int xoffset = (fh*IW + fw)*IC;
			Xs[buf][Xs_y][Xs_x].x = lx0 * tex1Dfetch<float>(X, Xoffset0 + xoffset + Xic);
			Xs[buf][Xs_y][Xs_x].y = lx1 * tex1Dfetch<float>(X, Xoffset1 + xoffset + Xic);
			__syncthreads();

			for (int oic = 1; oic < OIC; oic++) {
#pragma unroll
				for (int ik = 0; ik < STEP; ik++)
				{
					float4 b = *(float4*)(&Xs[buf][ik][tx << 1]);
					float4 a = *(float4*)(&Ws[buf][ik][ty << 1]);

					simdMM4(v0, b.x, a);
					simdMM4(v1, b.y, a);
					simdMM4(v2, b.z, a);
					simdMM4(v3, b.w, a);
				}
				buf ^= 1;

				int Wic = ((oic << LB) + tx) >> 1;
				Ws[buf][Ws_x][Ws_y].x = W[toc0 + Wic];
				Ws[buf][Ws_x][Ws_y].y = W[toc1 + Wic];

				int Xic = ((oic << LB) + ty) >> 1;
				Xs[buf][Xs_y][Xs_x].x = lx0 * tex1Dfetch<float>(X, Xoffset0 + xoffset + Xic);
				Xs[buf][Xs_y][Xs_x].y = lx1 * tex1Dfetch<float>(X, Xoffset1 + xoffset + Xic);
				__syncthreads();
			}
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 b = *(float4*)(&Xs[buf][ik][tx << 1]);
				float4 a = *(float4*)(&Ws[buf][ik][ty << 1]);

				simdMM4(v0, b.x, a);
				simdMM4(v1, b.y, a);
				simdMM4(v2, b.z, a);
				simdMM4(v3, b.w, a);
			}
			buf ^= 1;
		}
	}

	j0 = j0 * OC + oc0; //oc = f(by), j = f(bx) -> (n, oh, ow)
	int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;

	*(float4*)(Y + j0) = v0;
	*(float4*)(Y + j1) = v1;
	*(float4*)(Y + j2) = v2;
	*(float4*)(Y + j3) = v3;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*2), IC % (BLOCK_SIZE/2)
//LB = 4: IC % 8 == 0
#ifndef CONV_3D_PURE_KERNEL_4_2_TEXTURE
#define CONV_3D_PURE_KERNEL_4_2_TEXTURE

//LB = 4: Size = 1, Time = 2.824 msec, Performace = 760.44 GFlop/s
//LB = 3: Size = 1, Time = 4.264 msec, Performace = 503.631 GFlop/s
template<int LB, int STEP>
__global__ void conv3dPure_kernel_4_2_texture(
	cudaTextureObject_t X, int IH, int IW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ Y, int OH, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float  Xs[2][1 << LB >> 1][(2 << LB) + 2];

	const int oc0 = (((blockIdx.y << LB) + ty) << 2) + oc_index;
	const int GK = FH * FW * IC;
	const int toc0 = (((tx & 1) << 1) + oc0) * GK, toc1 = toc0 + GK;

	int j0 = (((blockIdx.x << LB) + tx) << 1) + j_index;
	int tj0 = j0 + (ty & 1);
	const int OH_OW = OH * OW;
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	const int tihs0 = toh0 * sh - ph;
	const int tiws0 = tow0 * sw - pw;
	const int Xoffset0 = (((tn0 *IH) + tihs0) * IW + tiws0) * IC;

	//compute area----------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0);
	float4 v1 = make_float4(0, 0, 0, 0);

	const int Ws_x = (tx >> 1), Ws_y = (ty << 1) + (tx & 1);
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);
	const int OIC = (IC << 1 >> LB);
	for (int fh = 0; fh < FH; fh++) {
		for (int fw = 0; fw < FW; fw++, W += IC)
		{
			int Wic = tx >> 1;
			Ws[buf][Ws_x][Ws_y].x = W[toc0 + Wic];
			Ws[buf][Ws_x][Ws_y].y = W[toc1 + Wic];

			int Xic = ty >> 1;
			bool lx0 = (tihs0 >= -fh) && (tihs0 < IH - fh) && (tiws0 >= -fw) && (tiws0 < IW - fw);
			int xoffset = (fh*IW + fw)*IC;
			Xs[buf][Xs_y][Xs_x] = lx0 * tex1Dfetch<float>(X, Xoffset0 + xoffset + Xic);
			__syncthreads();

			for (int oic = 1; oic < OIC; oic++) {
#pragma unroll
				for (int ik = 0; ik < STEP; ik++) {
					float2 b = *(float2*)(&Xs[buf][ik][tx << 1]);
					float4 a = *(float4*)(&Ws[buf][ik][ty << 1]);
					simdMM4(v0, b.x, a);
					simdMM4(v1, b.y, a);
				}
				buf ^= 1;

				int Wic = ((oic << LB) + tx) >> 1;
				Ws[buf][Ws_x][Ws_y].x = W[toc0 + Wic];
				Ws[buf][Ws_x][Ws_y].y = W[toc1 + Wic];

				int Xic = ((oic << LB) + ty) >> 1;
				Xs[buf][Xs_y][Xs_x] = lx0 * tex1Dfetch<float>(X, Xoffset0 + xoffset + Xic);
				__syncthreads();
			}
#pragma unroll
			for (int ik = 0; ik < STEP; ik++) {
				float2 b = *(float2*)(&Xs[buf][ik][tx << 1]);
				float4 a = *(float4*)(&Ws[buf][ik][ty << 1]);
				simdMM4(v0, b.x, a);
				simdMM4(v1, b.y, a);
			}
			buf ^= 1;
		}
	}

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	int j1 = j0 + OC;

	*(float4*)(Y + j0) = v0;
	*(float4*)(Y + j1) = v1;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*4), IC % (BLOCK_SIZE/2)
//LB = 4: IC % 8 == 0
#ifndef CONV_3D_PURE_KERNEL_2_4_TEXTURE
#define CONV_3D_PURE_KERNEL_2_4_TEXTURE

//LB = 4: Size = 1, Time = 3.508 msec, Performace = 612.167 GFlop/s
//LB = 3: Size = 1, Time = 6.916 msec, Performace = 310.509 GFlop/s
template<int LB, int STEP>
__global__ void conv3dPure_kernel_2_4_texture(
	cudaTextureObject_t X, int IH, int IW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ Y, int OH, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float  Ws[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2 Xs[2][1 << LB >> 1][(2 << LB) + 2];

	const int oc0 = (((blockIdx.y << LB) + ty) << 1) + oc_index;
	const int GK = FH * FW * IC;
	const int toc0 = ((tx & 1) + oc0) * GK;

	int j0 = (((blockIdx.x << LB) + tx) << 2) + j_index;
	int tj0 = j0 + ((ty & 1) << 1), tj1 = tj0 + 1;
	const int OH_OW = OH * OW;
	get_n_oh_ow(tj0, tn0, toh0, tow0);
	get_n_oh_ow(tj1, tn1, toh1, tow1);
	const int tihs0 = toh0 * sh - ph, tiws0 = tow0 * sw - pw;
	const int tihs1 = toh1 * sh - ph, tiws1 = tow1 * sw - pw;
	const int Xoffset0 = (((tn0 *IH) + tihs0) * IW + tiws0) * IC;
	const int Xoffset1 = (((tn1 *IH) + tihs1) * IW + tiws1) * IC;

	//compute area----------------------------------------------------
	float2 v0 = make_float2(0, 0);
	float2 v1 = make_float2(0, 0);
	float2 v2 = make_float2(0, 0);
	float2 v3 = make_float2(0, 0);

	const int Ws_x = (tx >> 1), Ws_y = (ty << 1) + (tx & 1);
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);
	const int OIC = (IC << 1 >> LB);
	for (int fh = 0; fh < FH; fh++) {
		for (int fw = 0; fw < FW; fw++, W += IC)
		{
			int Wic = tx >> 1;
			Ws[buf][Ws_x][Ws_y] = W[toc0 + Wic];

			int Xic = ty >> 1;
			bool lx0 = (tihs0 >= -fh) && (tihs0 < IH - fh) && (tiws0 >= -fw) && (tiws0 < IW - fw);
			bool lx1 = (tihs1 >= -fh) && (tihs1 < IH - fh) && (tiws1 >= -fw) && (tiws1 < IW - fw);
			int xoffset = (fh*IW + fw)*IC;
			Xs[buf][Xs_y][Xs_x].x = lx0 * tex1Dfetch<float>(X, Xoffset0 + xoffset + Xic);
			Xs[buf][Xs_y][Xs_x].y = lx1 * tex1Dfetch<float>(X, Xoffset1 + xoffset + Xic);
			__syncthreads();

			for (int oic = 1; oic < OIC; oic++) {
#pragma unroll
				for (int ik = 0; ik < STEP; ik++)
				{
					float4 b = *(float4*)(&Xs[buf][ik][tx << 1]);
					float2 a = *(float2*)(&Ws[buf][ik][ty << 1]);

					simdMM2(v0, b.x, a);
					simdMM2(v1, b.y, a);
					simdMM2(v2, b.z, a);
					simdMM2(v3, b.w, a);
				}
				buf ^= 1;

				int Wic = ((oic << LB) + tx) >> 1;
				Ws[buf][Ws_x][Ws_y] = W[toc0 + Wic];

				int Xic = ((oic << LB) + ty) >> 1;
				Xs[buf][Xs_y][Xs_x].x = lx0 * tex1Dfetch<float>(X, Xoffset0 + xoffset + Xic);
				Xs[buf][Xs_y][Xs_x].y = lx1 * tex1Dfetch<float>(X, Xoffset1 + xoffset + Xic);
				__syncthreads();
			}
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 b = *(float4*)(&Xs[buf][ik][tx << 1]);
				float2 a = *(float2*)(&Ws[buf][ik][ty << 1]);

				simdMM2(v0, b.x, a);
				simdMM2(v1, b.y, a);
				simdMM2(v2, b.z, a);
				simdMM2(v3, b.w, a);
			}
			buf ^= 1;
		}
	}

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	int j1 = j0 + OC, j2 = j1 + OC, j3 = j2 + OC;

	*(float2*)(Y + j0) = v0;
	*(float2*)(Y + j1) = v1;
	*(float2*)(Y + j2) = v2;
	*(float2*)(Y + j3) = v3;
}

#endif


//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*2), GK >= BLOCK_SIZE
//LB = 4, GK >= 16
//LB = 3, GK >=  8
#ifndef CONV_3D_GEMM_KERNEL_2_2_TEXTURE
#define CONV_3D_GEMM_KERNEL_2_2_TEXTURE

//LB = 4: Size = 0.710273, Time = 3.086 msec, Performace = 494.264 GFlop/
//LB = 3: Size = 0.710273, Time = 4.198 msec, Performace = 363.339 GFlop/s
template<int LB, int STEP>
__global__ void conv3dGemm_kernel_2_2_texture(
	cudaTextureObject_t X, int IH, int IW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ Y, int OH, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float2 Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GK = FH * FW * IC
	const int FW_IC = FW * IC, GK = FH * FW_IC;

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 1) + oc_index;
	const int toc0 = oc0 * GK, toc1 = toc0 + GK;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.x << LB) + tx) << 1) + j_index, j1 = j0 + 1;
	const int OH_OW = OH * OW;
	get_n_oh_ow(j0, n0, oh0, ow0);
	get_n_oh_ow(j1, n1, oh1, ow1);
	const int toh0 = oh0 * sh - ph;
	const int toh1 = oh1 * sh - ph;
	const int tow0 = ow0 * sw - pw;
	const int tow1 = ow1 * sw - pw;
	const int Xoffset0 = ((n0*IH + toh0)*IW + tow0)*IC;
	const int Xoffset1 = ((n1*IH + toh1)*IW + tow1)*IC;

	//load 2 elements from W[OC, FH, FW, IC]
	int W_k = tx;
	Ws[buf][tx][ty].x = W[toc0 + W_k];
	Ws[buf][tx][ty].y = W[toc1 + W_k];

	//load 2 elements from X[N, IH, IW, IC]
	int X_k = ty, X_fh, X_fw;
	get_X_fh_fw(X_k, X_fh, X_fw);
	bool lx0 = (toh0 >= -X_fh) && (toh0 < IH - X_fh) && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
	bool lx1 = (toh1 >= -X_fh) && (toh1 < IH - X_fh) && (tow1 >= -X_fw) && (tow1 < IW - X_fw);
	int IW_IC = IW * IC, xoffset = X_fh * IW_IC + X_k;//X[n, ih, iw, ic]
	Xs[buf][ty][tx].x = lx0 * tex1Dfetch<float>(X, Xoffset0 + xoffset);
	Xs[buf][ty][tx].y = lx1 * tex1Dfetch<float>(X, Xoffset1 + xoffset);
	__syncthreads();

	//compute area-----------------------------------------
	float2 v0 = make_float2(0, 0);
	float2 v1 = make_float2(0, 0);
	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float2 b = Xs[buf][ik][tx];
			float2 a = Ws[buf][ik][ty];
			simdMM2(v0, b.x, a);
			simdMM2(v1, b.y, a);
		}
		buf ^= 1;

		//load 2 elements from W[OC, FH, FW, IC]
		int W_k = (ok << LB) + tx;
		Ws[buf][tx][ty].x = W[toc0 + W_k];
		Ws[buf][tx][ty].y = W[toc1 + W_k];

		//load 2 elements from X[N, IH, IW, IC]
		int X_k = (ok << LB) + ty, X_fh, X_fw;
		get_X_fh_fw(X_k, X_fh, X_fw);
		bool lx0 = (toh0 >= -X_fh) && (toh0 < IH - X_fh) && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
		bool lx1 = (toh1 >= -X_fh) && (toh1 < IH - X_fh) && (tow1 >= -X_fw) && (tow1 < IW - X_fw);
		int xoffset = X_fh * IW_IC + X_k;//X[n, ih, iw, ic]
		Xs[buf][ty][tx].x = lx0 * tex1Dfetch<float>(X, Xoffset0 + xoffset);
		Xs[buf][ty][tx].y = lx1 * tex1Dfetch<float>(X, Xoffset1 + xoffset);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float2 b = Xs[buf][ik][tx];
		float2 a = Ws[buf][ik][ty];
		simdMM2(v0, b.x, a);
		simdMM2(v1, b.y, a);
	}

	//when GK%STEP != 0 --------------------------------------
	for (int k = GK - (GK & (STEP - 1)); k < GK; k++)
	{
		float2 a;//load 2 elements from W
		a.x = W[toc0 + k];
		a.y = W[toc1 + k];

		float2 b;//load 2 elements from X
		int X_k = k, X_fh, X_fw;
		get_X_fh_fw(X_k, X_fh, X_fw);
		bool lx0 = (toh0 >= -X_fh) && (toh0 < IH - X_fh) && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
		bool lx1 = (toh1 >= -X_fh) && (toh1 < IH - X_fh) && (tow1 >= -X_fw) && (tow1 < IW - X_fw);
		int xoffset = X_fh * IW_IC + X_k;//X[n, ih, iw, ic]
		b.x = lx0 * tex1Dfetch<float>(X, Xoffset0 + xoffset);
		b.y = lx1 * tex1Dfetch<float>(X, Xoffset1 + xoffset);

		simdMM2(v0, b.x, a);
		simdMM2(v1, b.y, a);
	}
	//when GK%STEP != 0 --------------------------------------

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	j1 = j0 + OC;

	*(float2*)(Y + j0) = v0;
	*(float2*)(Y + j1) = v1;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*1), GK >= BLOCK_SIZE
//LB = 4, GK >= 16
//LB = 3, GK >=  8
#ifndef CONV_3D_GEMM_KERNEL_4_1_TEXTURE
#define CONV_3D_GEMM_KERNEL_4_1_TEXTURE

//LB = 4: Size = 1, Time = 4.16  msec, Performace = 516.222 GFlop/s
//LB = 3: Size = 1, Time = 5.664 msec, Performace = 379.146 GFlop/s
template<int LB, int STEP>
__global__ void conv3dGemm_kernel_4_1_texture(
	cudaTextureObject_t X, int IH, int IW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ Y, int OH, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4  Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float   Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GK = FH * FW * IC
	const int FW_IC = FW * IC, GK = FH * FW_IC;

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 2) + oc_index;
	int toc0 = oc0 * GK, toc1 = toc0 + GK, toc2 = toc1 + GK, toc3 = toc2 + GK;

	//prepare for GM = N * OH * OW
	int j0 = ((blockIdx.x << LB) + tx) + j_index;
	int OH_OW = OH * OW;
	get_n_oh_ow(j0, n0, oh0, ow0);
	const int toh0 = oh0 * sh - ph;
	const int tow0 = ow0 * sw - pw;
	const int Xoffset0 = ((n0*IH + toh0)*IW + tow0)*IC;

	//load 4 elements from W[OC, FH, FW, IC]
	int W_k = tx;
	Ws[buf][tx][ty].x = W[toc0 + W_k];
	Ws[buf][tx][ty].y = W[toc1 + W_k];
	Ws[buf][tx][ty].z = W[toc2 + W_k];
	Ws[buf][tx][ty].w = W[toc3 + W_k];

	//load 1 element from X[N, IH, IW, IC]
	int X_k = ty, X_fh, X_fw;
	get_X_fh_fw(X_k, X_fh, X_fw);
	bool lx0 = (toh0 >= -X_fh) && (toh0 < IH - X_fh) && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
	int IW_IC = IW * IC, xoffset = X_fh * IW_IC + X_k;//X[n, ih, iw, ic]
	Xs[buf][ty][tx] = lx0 * tex1Dfetch<float>(X, Xoffset0 + xoffset);
	__syncthreads();

	//compute area----------------------------------
	float4 v = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float  b = Xs[buf][ik][tx];
			float4 a = Ws[buf][ik][ty];
			simdMM4(v, b, a);
		}
		buf ^= 1;

		//load 4 elements from W[OC, FH, FW, IC]
		int W_k = (ok << LB) + tx;
		Ws[buf][tx][ty].x = W[toc0 + W_k];
		Ws[buf][tx][ty].y = W[toc1 + W_k];
		Ws[buf][tx][ty].z = W[toc2 + W_k];
		Ws[buf][tx][ty].w = W[toc3 + W_k];

		//load 1 element from X[N, IH, IW, IC]
		int X_k = (ok << LB) + ty, X_fh, X_fw;
		get_X_fh_fw(X_k, X_fh, X_fw);
		bool lx0 = (toh0 >= -X_fh) && (toh0 < IH - X_fh) && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
		int xoffset = X_fh * IW_IC + X_k;//X[n, ih, iw, ic]
		Xs[buf][ty][tx] = lx0 * tex1Dfetch<float>(X, Xoffset0 + xoffset);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float  b = Xs[buf][ik][tx];
		float4 a = Ws[buf][ik][ty];
		simdMM4(v, b, a);
	}

	//when GK % STEP != 0---------------------------------------
	for (int k = GK - (GK & (STEP - 1)); k < GK; k++)
	{
		float4 a;//load 4 elements from W
		a.x = W[toc0 + k];
		a.y = W[toc1 + k];
		a.z = W[toc2 + k];
		a.w = W[toc3 + k];

		float b;//load 1 element from X
		int X_k = k, X_fh, X_fw; get_X_fh_fw(X_k, X_fh, X_fw);
		bool lx0 = (toh0 >= -X_fh) && (toh0 < IH - X_fh) && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
		int xoffset = X_fh * IW_IC + X_k;//X[n, ih, iw, ic]
		b = lx0 * tex1Dfetch<float>(X, Xoffset0 + xoffset);

		simdMM4(v, b, a);
	}
	//when GK % STEP != 0---------------------------------------

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	*(float4*)(Y + j0) = v;
}

#endif


//(Y: BLOCK_SIZE*1£¬X: BLOCK_SIZE*4), GK >= BLOCK_SIZE, 
//LB = 4, GK >= 16
//LB = 3, GK >=  8
#ifndef CONV_3D_GEMM_KERNEL_1_4_TEXTURE
#define CONV_3D_GEMM_KERNEL_1_4_TEXTURE

//(GN, GM, GK) = (128, 8192, 512)
//LB = 4: Size = 1, Time = 6.72 msec , Performace = 319.566 GFlop/s
//LB = 3: Size = 1, Time = 7.79  msec, Performace = 275.672 GFlop/s
template<int LB, int STEP>
__global__ void conv3dGemm_kernel_1_4_texture(
	cudaTextureObject_t X, int IH, int IW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ Y, int OH, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float  Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GK = FH * FW * IC
	const int FW_IC = FW * IC, GK = FH * FW_IC;

	//prepare for GN = OC
	int oc0 = ((blockIdx.y << LB) + ty) + oc_index;
	int toc0 = oc0 * GK;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.x << LB) + tx) << 2) + j_index;
	int j1 = j0 + 1, j2 = j0 + 2, j3 = j0 + 3;
	const int OH_OW = OH * OW;
	get_n_oh_ow(j0, n0, oh0, ow0);
	get_n_oh_ow(j1, n1, oh1, ow1);
	get_n_oh_ow(j2, n2, oh2, ow2);
	get_n_oh_ow(j3, n3, oh3, ow3);
	const int toh0 = oh0 * sh - ph;
	const int toh1 = oh1 * sh - ph;
	const int toh2 = oh2 * sh - ph;
	const int toh3 = oh3 * sh - ph;
	const int tow0 = ow0 * sw - pw;
	const int tow1 = ow1 * sw - pw;
	const int tow2 = ow2 * sw - pw;
	const int tow3 = ow3 * sw - pw;
	const int Xoffset0 = ((n0*IH + toh0)*IW + tow0)*IC;
	const int Xoffset1 = ((n1*IH + toh1)*IW + tow1)*IC;
	const int Xoffset2 = ((n2*IH + toh2)*IW + tow2)*IC;
	const int Xoffset3 = ((n3*IH + toh3)*IW + tow3)*IC;

	//load 1 element from W[OC, FH, FW, IC]
	int W_k = tx;
	Ws[buf][tx][ty] = W[toc0 + W_k];

	//load 4 elements from X[N, IH, IW, IC]
	int X_k = ty, X_fh, X_fw;
	get_X_fh_fw(X_k, X_fh, X_fw);
	bool lx0 = (toh0 >= -X_fh) && (toh0 < IH - X_fh) && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
	bool lx1 = (toh1 >= -X_fh) && (toh1 < IH - X_fh) && (tow1 >= -X_fw) && (tow1 < IW - X_fw);
	bool lx2 = (toh2 >= -X_fh) && (toh2 < IH - X_fh) && (tow2 >= -X_fw) && (tow2 < IW - X_fw);
	bool lx3 = (toh3 >= -X_fh) && (toh3 < IH - X_fh) && (tow3 >= -X_fw) && (tow3 < IW - X_fw);
	int IW_IC = IW * IC, xoffset = X_fh * IW_IC + X_k;//X[n, ih, iw, ic]
	Xs[buf][ty][tx].x = lx0 * tex1Dfetch<float>(X, Xoffset0 + xoffset);
	Xs[buf][ty][tx].y = lx1 * tex1Dfetch<float>(X, Xoffset1 + xoffset);
	Xs[buf][ty][tx].z = lx2 * tex1Dfetch<float>(X, Xoffset2 + xoffset);
	Xs[buf][ty][tx].w = lx3 * tex1Dfetch<float>(X, Xoffset3 + xoffset);
	__syncthreads();

	//compute area------------------------------------
	float4 v = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float4 b = Xs[buf][ik][tx];
			float  a = Ws[buf][ik][ty];
			simdMM4(v, a, b);
		}
		buf ^= 1;

		//load 1 element from W[OC, FH, FW, IC]
		int W_k = (ok << LB) + tx;
		Ws[buf][tx][ty] = W[toc0 + W_k];

		//load 4 elements from X[N, IH, IW, IC]
		int X_k = (ok << LB) + ty, X_fh, X_fw;
		get_X_fh_fw(X_k, X_fh, X_fw);
		bool lx0 = (toh0 >= -X_fh) && (toh0 < IH - X_fh) && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
		bool lx1 = (toh1 >= -X_fh) && (toh1 < IH - X_fh) && (tow1 >= -X_fw) && (tow1 < IW - X_fw);
		bool lx2 = (toh2 >= -X_fh) && (toh2 < IH - X_fh) && (tow2 >= -X_fw) && (tow2 < IW - X_fw);
		bool lx3 = (toh3 >= -X_fh) && (toh3 < IH - X_fh) && (tow3 >= -X_fw) && (tow3 < IW - X_fw);
		int xoffset = X_fh * IW_IC + X_k;//X[n, ih, iw, ic]
		Xs[buf][ty][tx].x = lx0 * tex1Dfetch<float>(X, Xoffset0 + xoffset);
		Xs[buf][ty][tx].y = lx1 * tex1Dfetch<float>(X, Xoffset1 + xoffset);
		Xs[buf][ty][tx].z = lx2 * tex1Dfetch<float>(X, Xoffset2 + xoffset);
		Xs[buf][ty][tx].w = lx3 * tex1Dfetch<float>(X, Xoffset3 + xoffset);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float4 b = Xs[buf][ik][tx];
		float  a = Ws[buf][ik][ty];
		simdMM4(v, a, b);
	}

	//when GK % STEP != 0---------------------------------------
	for (int k = GK - (GK & (STEP - 1)); k < GK; k++)
	{
		//load 1 element from W
		float a = W[toc0 + k];

		float4 b;//load 4 elements from X
		int X_k = k, X_fh, X_fw;
		get_X_fh_fw(X_k, X_fh, X_fw);
		bool lx0 = (toh0 >= -X_fh) && (toh0 < IH - X_fh) && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
		bool lx1 = (toh1 >= -X_fh) && (toh1 < IH - X_fh) && (tow1 >= -X_fw) && (tow1 < IW - X_fw);
		bool lx2 = (toh2 >= -X_fh) && (toh2 < IH - X_fh) && (tow2 >= -X_fw) && (tow2 < IW - X_fw);
		bool lx3 = (toh3 >= -X_fh) && (toh3 < IH - X_fh) && (tow3 >= -X_fw) && (tow3 < IW - X_fw);
		int xoffset = X_fh * IW_IC + X_k;//X[n, ih, iw, ic]
		b.x = lx0 * tex1Dfetch<float>(X, Xoffset0 + xoffset);
		b.y = lx1 * tex1Dfetch<float>(X, Xoffset1 + xoffset);
		b.z = lx2 * tex1Dfetch<float>(X, Xoffset2 + xoffset);
		b.w = lx3 * tex1Dfetch<float>(X, Xoffset3 + xoffset);

		simdMM4(v, a, b);
	}
	//when GK % STEP != 0---------------------------------------

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	j1 = j0 + OC; j2 = j1 + OC; j3 = j2 + OC;

	Y[j0] = v.x;
	Y[j1] = v.y;
	Y[j2] = v.z;
	Y[j3] = v.w;
}

#endif


//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*1), GK >= BLOCK_SIZE
//LB = 4, GK >= 16
#ifndef CONV_3D_GEMM_KERNEL_2_1_TEXTURE
#define CONV_3D_GEMM_KERNEL_2_1_TEXTURE

//LB = 4: Size = 0.710273, Time = 4.674 msec, Performace = 326.337 GFlop/s
//LB = 3: Size = 1, Time = 9.032 msec, Performace = 237.764 GFlop/s
template<int LB, int STEP>
__global__ void conv3dGemm_kernel_2_1_texture(
	cudaTextureObject_t X, int IH, int IW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ Y, int OH, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float  Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GK = FH * FW * IC
	const int FW_IC = FW * IC, GK = FH * FW_IC;

	//prepare for GN = OC
	const int oc0 = (((blockIdx.y << LB) + ty) << 1) + oc_index;
	const int toc0 = oc0 * GK, toc1 = toc0 + GK;

	//prepare for GM = N * OH * OW
	int j0 = ((blockIdx.x << LB) + tx) + j_index;
	int OH_OW = OH * OW;
	get_n_oh_ow(j0, n0, oh0, ow0);
	const int toh0 = oh0 * sh - ph;
	const int tow0 = ow0 * sw - pw;
	const int Xoffset0 = ((n0*IH + toh0)*IW + tow0)*IC;

	//load 2 elements from W[OC, FH, FW, IC]
	int W_k = tx;
	Ws[buf][tx][ty].x = W[toc0 + W_k];
	Ws[buf][tx][ty].y = W[toc1 + W_k];

	//load 1 element from X[N, IH, IW, IC]
	int X_k = ty, X_fh, X_fw;
	get_X_fh_fw(X_k, X_fh, X_fw);
	bool lx0 = (toh0 >= -X_fh) && (toh0 < IH - X_fh) && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
	int IW_IC = IW * IC, xoffset = X_fh * IW_IC + X_k;//X[n, ih, iw, ic]
	Xs[buf][ty][tx] = lx0 * tex1Dfetch<float>(X, Xoffset0 + xoffset);
	__syncthreads();

	//compute area-------------------------------
	float2 v = make_float2(0, 0);
	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float  b = Xs[buf][ik][tx];
			float2 a = Ws[buf][ik][ty];
			simdMM2(v, b, a)
		}
		buf ^= 1;

		//load 2 elements from W[OC, FH, FW, IC]
		int W_k = (ok << LB) + tx;
		Ws[buf][tx][ty].x = W[toc0 + W_k];
		Ws[buf][tx][ty].y = W[toc1 + W_k];

		//load 1 element from X[N, IH, IW, IC]
		int X_k = (ok << LB) + ty, X_fh, X_fw;
		get_X_fh_fw(X_k, X_fh, X_fw);
		bool lx0 = (toh0 >= -X_fh) && (toh0 < IH - X_fh) && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
		int xoffset = X_fh * IW_IC + X_k;
		Xs[buf][ty][tx] = lx0 * tex1Dfetch<float>(X, Xoffset0 + xoffset);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float  b = Xs[buf][ik][tx];
		float2 a = Ws[buf][ik][ty];
		simdMM2(v, b, a)
	}

	//when GK % STEP != 0---------------------------------------
	for (int k = GK - (GK & (STEP - 1)); k < GK; k++)
	{
		float2 a;//load 2 elements from W
		a.x = W[toc0 + k];
		a.y = W[toc1 + k];

		float b;//load 1 element from X
		int X_k = k, X_fh, X_fw;
		get_X_fh_fw(X_k, X_fh, X_fw);
		bool lx0 = (toh0 >= -X_fh) && (toh0 < IH - X_fh) && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
		int xoffset = X_fh * IW_IC + X_k;//X[n, ih, iw, ic]
		b = lx0 * tex1Dfetch<float>(X, Xoffset0 + xoffset);

		simdMM2(v, b, a);
	}
	//when GK % STEP != 0---------------------------------------

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	*(float2*)(Y + j0) = v;
}

#endif


//(Y: BLOCK_SIZE*1, X: BLOCK_SIZE*2), GK >= BLOCK_SIZE, 
//LB = 4, GK >= 16
#ifndef CONV_3D_GEMM_KERNEL_1_2_TEXTURE
#define CONV_3D_GEMM_KERNEL_1_2_TEXTURE

//LB = 4: Size = 0.710273, Time = 5.34  msec, Performace = 285.637 GFlop/s
//LB = 3: Size = 0.710273, Time = 7.136 msec, Performace = 213.747 GFlop/s
template<int LB, int STEP>
__global__ void conv3dGemm_kernel_1_2_texture(
	cudaTextureObject_t X, int IH, int IW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ Y, int OH, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float  Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float2 Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GK = FH * FW * IC
	const int FW_IC = FW * IC, GK = FH * FW_IC;

	//prepare for GN = OC
	const int oc0 = ((blockIdx.y << LB) + ty) + oc_index;
	const int toc0 = oc0 * GK;

	//prepare for GM = N * OH * OW
	int j0 = (((blockIdx.x << LB) + tx) << 1) + j_index, j1 = j0 + 1;
	int OH_OW = OH * OW;
	get_n_oh_ow(j0, n0, oh0, ow0);
	get_n_oh_ow(j1, n1, oh1, ow1);
	const int toh0 = oh0 * sh - ph;
	const int toh1 = oh1 * sh - ph;
	const int tow0 = ow0 * sw - pw;
	const int tow1 = ow1 * sw - pw;
	const int Xoffset0 = ((n0*IH + toh0)*IW + tow0)*IC;
	const int Xoffset1 = ((n1*IH + toh1)*IW + tow1)*IC;

	//load 1 element from W[OC, FH, FW, IC]
	int W_k = tx;
	Ws[buf][tx][ty] = W[toc0 + W_k];

	//load 2 elements from X[N, IH, IW, IC]
	int X_k = ty, X_fh, X_fw;
	get_X_fh_fw(X_k, X_fh, X_fw);
	bool lx0 = (toh0 >= -X_fh) && (toh0 < IH - X_fh) && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
	bool lx1 = (toh1 >= -X_fh) && (toh1 < IH - X_fh) && (tow1 >= -X_fw) && (tow1 < IW - X_fw);
	int IW_IC = IW * IC, xoffset = X_fh * IW_IC + X_k;//X[n, ih, iw, ic]
	Xs[buf][ty][tx].x = lx0 * tex1Dfetch<float>(X, Xoffset0 + xoffset);
	Xs[buf][ty][tx].y = lx1 * tex1Dfetch<float>(X, Xoffset1 + xoffset);
	__syncthreads();

	//compute area-------------------------------------------
	float2 v = make_float2(0, 0);
	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float2 b = Xs[buf][ik][tx];
			float  a = Ws[buf][ik][ty];
			simdMM2(v, a, b);
		}
		buf ^= 1;

		//load 1 element from W[OC, FH, FW, IC]
		int W_k = (ok << LB) + tx;
		Ws[buf][tx][ty] = W[toc0 + W_k];

		//load 2 elements from X[N, IH, IW, IC]
		int X_k = (ok << LB) + ty, X_fh, X_fw;
		get_X_fh_fw(X_k, X_fh, X_fw);
		bool lx0 = (toh0 >= -X_fh) && (toh0 < IH - X_fh) && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
		bool lx1 = (toh1 >= -X_fh) && (toh1 < IH - X_fh) && (tow1 >= -X_fw) && (tow1 < IW - X_fw);
		int xoffset = X_fh * IW_IC + X_k;//X[n, ih, iw, ic]
		Xs[buf][ty][tx].x = lx0 * tex1Dfetch<float>(X, Xoffset0 + xoffset);
		Xs[buf][ty][tx].y = lx1 * tex1Dfetch<float>(X, Xoffset1 + xoffset);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float2 b = Xs[buf][ik][tx];
		float  a = Ws[buf][ik][ty];
		simdMM2(v, a, b);
	}

	//when GK % STEP != 0---------------------------------------
	for (int k = GK - (GK & (STEP - 1)); k < GK; k++)
	{
		float a = W[toc0 + k];//load 1 element from W

		float2 b;//load 2 elements from X
		int X_k = k, X_fh, X_fw;
		get_X_fh_fw(X_k, X_fh, X_fw);
		bool lx0 = (toh0 >= -X_fh) && (toh0 < IH - X_fh) && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
		bool lx1 = (toh1 >= -X_fh) && (toh1 < IH - X_fh) && (tow1 >= -X_fw) && (tow1 < IW - X_fw);
		int xoffset = X_fh * IW_IC + X_k;//X[n, ih, iw, ic]
		b.x = lx0 * tex1Dfetch<float>(X, Xoffset0 + xoffset);
		b.y = lx1 * tex1Dfetch<float>(X, Xoffset1 + xoffset);

		simdMM2(v, a, b);
	}
	//when GK % STEP != 0---------------------------------------

	j0 = j0 * OC + oc0;//oc = f(by), j = f(bx) -> (n, oh, ow)
	j1 = j0 + OC;

	Y[j0] = v.x;
	Y[j1] = v.y;
}

#endif


//(Y: BLOCK_SIZE*1, X: BLOCK_SIZE*1), GK >= BLOCK_SIZE
//LB = 4, GK >= 16
#ifndef CONV_3D_GEMM_KERNEL_1_1_TEXTURE
#define CONV_3D_GEMM_KERNEL_1_1_TEXTURE

//LB = 4: Size = 0.710273, Time = 8.012  msec, Performace = 190.377 GFlop/s
//LB = 3: Size = 0.710273, Time = 10.726 msec, Performace = 142.206 GFlop/s
template<int LB, int STEP>
__global__ void conv3dGemm_kernel_1_1_texture(
	cudaTextureObject_t X, int IH, int IW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ Y, int OH, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GK = FH * FW * IC
	const int FW_IC = FW * IC, GK = FH * FW_IC;

	//prepare for GN = OC
	const int oc0 = ((blockIdx.y << LB) + ty) + oc_index;
	const int toc0 = oc0 * GK;

	//prepare for GM = N * OH * OW
	int j0 = ((blockIdx.x << LB) + tx) + j_index;
	int OH_OW = OH * OW;
	get_n_oh_ow(j0, n0, oh0, ow0);
	const int toh0 = oh0 * sh - ph, tow0 = ow0 * sw - pw;
	const int Xoffset0 = ((n0*IH + toh0)*IW + tow0)*IC;

	//load 1 element from W[OC, FH, FW, IC]
	int W_k = tx;
	Ws[buf][tx][ty] = W[toc0 + W_k];

	//load 1 element from X[N, IH, IW, IC]
	int X_k = ty, X_fh, X_fw;
	get_X_fh_fw(X_k, X_fh, X_fw);
	bool lx0 = (toh0 >= -X_fh) && (toh0 < IH - X_fh) && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
	int IW_IC = IW * IC, xoffset = X_fh * IW_IC + X_k;//X[n, ih, iw, ic]
	Xs[buf][ty][tx] = lx0 * tex1Dfetch<float>(X, Xoffset0 + xoffset);
	__syncthreads();

	//compute area----------------------------------------------------
	float v = 0;
	for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float b = Xs[buf][ik][tx];
			float a = Ws[buf][ik][ty];
			v += a * b;
		}
		buf ^= 1;

		//load 1 element from W[OC, FH, FW, IC]
		int W_k = (ok << LB) + tx;
		Ws[buf][tx][ty] = W[toc0 + W_k];

		//load 1 element from X[N, IH, IW, IC]
		int X_k = (ok << LB) + ty, X_fh, X_fw;
		get_X_fh_fw(X_k, X_fh, X_fw);
		bool lx0 = (toh0 >= -X_fh) && (toh0 < IH - X_fh) && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
		int xoffset = X_fh * IW_IC + X_k;//X[n, ih, iw, ic]
		Xs[buf][ty][tx] = lx0 * tex1Dfetch<float>(X, Xoffset0 + xoffset);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float b = Xs[buf][ik][tx];
		float a = Ws[buf][ik][ty];
		v += a * b;
	}

	//when GK % STEP != 0---------------------------------------
	for (int k = GK - (GK & (STEP - 1)); k < GK; k++)
	{
		//load 1 element from W
		float a = W[toc0 + k];

		float b;//load 1 element from X
		int X_k = k, X_fh, X_fw; get_X_fh_fw(X_k, X_fh, X_fw);
		bool lx0 = (toh0 >= -X_fh) && (toh0 < IH - X_fh) && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
		int xoffset = X_fh * IW_IC + X_k;//X[n, ih, iw, ic]
		b = lx0 * tex1Dfetch<float>(X, Xoffset0 + xoffset);

		v += a * b;
	}
	//when GK % STEP != 0---------------------------------------

	Y[j0*OC + oc0] = v;
}

#endif

#endif