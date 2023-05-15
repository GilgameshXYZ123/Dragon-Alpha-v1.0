#pragma once


#ifndef DCONV3D_DX_KERNEL_SPLIT_KERNEL_TEXTURE_H
#define DCONV3D_DX_KERNEL_SPLIT_KERNEL_TEXTURE_H

//Sparse Matrix Method
//We have:
//(1) FH * FW >= 2
//(2) GM >= 4, GM % 4 == 0
//(3) GN >= 4, GN % 4 == 0
//(4) GK >= 8, GK % 4 == 0
#ifndef DCONV3D_DX_KERNEL_SPLIT_KERNEL_TEXTURE_CALL
#define DCONV3D_DX_KERNEL_SPLIT_KERNEL_TEXTURE_CALL

#define kernelSplit_k88_tex(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, CFH, CFW, GN, GM) \
	kernelSplit_kernel_8_8_texture<LB, (1<<LB>>1)>\
		<<< dim3((GM + (8<<LB) - 1)>>LB>>3, GN>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, N,IC,OC, sh,sw,ph,pw, CFH, CFW, ic_index, j_index)

#define kernelSplit_k88LB_tex(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, CFH, CFW, GN, GM) \
	kernelSplit_kernel_8_8_texture<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>3, GN>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, N,IC,OC, sh,sw,ph,pw, CFH, CFW, ic_index, j_index)

#define kernelSplit_k44_tex(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, CFH, CFW, GN, GM) \
	kernelSplit_kernel_4_4_texture<LB, (1<<LB>>1)>\
		<<< dim3((GM + (4<<LB) - 1)>>LB>>2, GN>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, N,IC,OC, sh,sw,ph,pw, CFH, CFW, ic_index, j_index)

#define kernelSplit_k44LB_tex(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, CFH, CFW, GN, GM) \
	kernelSplit_kernel_4_4_texture<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>2, GN>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, N,IC,OC, sh,sw,ph,pw, CFH, CFW, ic_index, j_index)

#define kernelSplit_k24_tex(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, CFH, CFW, GN, GM) \
	kernelSplit_kernel_2_4_texture<LB, (1<<LB>>1)>\
		<<< dim3((GM + (4<<LB) - 1)>>LB>>2, GN>>LB>>1), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, N,IC,OC, sh,sw,ph,pw, CFH, CFW, ic_index, j_index)

#define kernelSplit_k22_tex(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, CFH, CFW, GN, GM) \
	kernelSplit_kernel_2_2_texture<LB, (1<<LB)>\
		<<< dim3((GM + (2<<LB) - 1)>>LB>>1, GN>>LB>>1), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, N,IC,OC, sh,sw,ph,pw, CFH, CFW, ic_index, j_index)

#define kernelSplit_k12_tex(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, CFH, CFW, GN, GM) \
	kernelSplit_kernel_1_2_texture<LB, (1<<LB)>\
		<<< dim3((GM + (2<<LB) - 1)>>LB>>1, GN>>LB), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, N,IC,OC, sh,sw,ph,pw, CFH, CFW, ic_index, j_index)

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8) 
//LB = 4: OC % 8 == 0
#ifndef DCONV3D_DX_KERNEL_SPLIT_KERNEL_8_8_TEXTURE
#define DCONV3D_DX_KERNEL_SPLIT_KERNEL_8_8_TEXTURE

//for(sh, sw) = 5:
//LB = 4: Size = 5.64062, Time = 1.658 msec, Performace = 7305.88 GFlop/s
//for(sh, sw) = 3:
//LB = 4: Size = 2.06641, Time = 1.54 msec, Performace = 2881.54 GFlop/s
//for(sh, sw) = 4:
//LB = 4: Size = 3.63379, Time = 2.6 msec  , Performace = 3001.35 GFlop/s
//LB = 3: Size = 3.63379, Time = 2.622 msec, Performace = 2976.16 GFlop/s
//for(sh, sw) = 2:
//LB = 4: Size = 0.938477, Time = 1.396 msec, Performace = 1443.67 GFlop/s
//LB = 3: Size = 0.938477, Time = 1.488 msec, Performace = 1354.41 GFlop/s
//LB = 4: Size = 1, Time = 1.546 msec, Performace = 1389.06 GFlop/s
//LB = 3: Size = 1, Time = 1.52  msec, Performace = 1412.82 GFlop/s
template<int LB, int STEP>
__global__ void kernelSplit_kernel_8_8_texture(
	cudaTextureObject_t deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw,
	int CFH, int CFW,
	int ic_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4  Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 dYs[2][1 << LB][(1 << LB) + 1];

	//preapre for GN = IC
	const int ic0 = (((blockIdx.y << LB) + ty) << 3) + ic_index;
	const int tic0 = ic0 + ((tx >= STEP) << 2);

	//prepared for GM = N * OHS * OWS
	const int j0 = (((blockIdx.x << LB) + tx) << 3) + j_index;//[n, ohs, ows]
	const int j1 = j0 + 1, j2 = j0 + 2, j3 = j0 + 3;
	const int j4 = j0 + 4, j5 = j0 + 5, j6 = j0 + 6, j7 = j0 + 7;
	const int oph = --CFH, opw = --CFW;
	const int OHS = OH + CFH;//OHS = OH - CFH + (oph << 1) + 1
	const int OWS = OW + CFW;//OWS = OW - CFW + (opw << 1) + 1;
	const int OHS_OWS = OHS * OWS;
	get_n_ohs_ows(j0, n0, ohs0, ows0);
	get_n_ohs_ows(j1, n1, ohs1, ows1);
	get_n_ohs_ows(j2, n2, ohs2, ows2);
	get_n_ohs_ows(j3, n3, ohs3, ows3);
	get_n_ohs_ows(j4, n4, ohs4, ows4);
	get_n_ohs_ows(j5, n5, ohs5, ows5);
	get_n_ohs_ows(j6, n6, ohs6, ows6);
	get_n_ohs_ows(j7, n7, ohs7, ows7);
	const int tih0 = ohs0 * sh - ph; ohs0 -= oph;
	const int tih1 = ohs1 * sh - ph; ohs1 -= oph;
	const int tih2 = ohs2 * sh - ph; ohs2 -= oph;
	const int tih3 = ohs3 * sh - ph; ohs3 -= oph;
	const int tih4 = ohs4 * sh - ph; ohs4 -= oph;
	const int tih5 = ohs5 * sh - ph; ohs5 -= oph;
	const int tih6 = ohs6 * sh - ph; ohs6 -= oph;
	const int tih7 = ohs7 * sh - ph; ohs7 -= oph;
	const int tiw0 = ows0 * sw - pw; ows0 -= opw;
	const int tiw1 = ows1 * sw - pw; ows1 -= opw;
	const int tiw2 = ows2 * sw - pw; ows2 -= opw;
	const int tiw3 = ows3 * sw - pw; ows3 -= opw;
	const int tiw4 = ows4 * sw - pw; ows4 -= opw;
	const int tiw5 = ows5 * sw - pw; ows5 -= opw;
	const int tiw6 = ows6 * sw - pw; ows6 -= opw;
	const int tiw7 = ows7 * sw - pw; ows7 -= opw;
	int X_offset0 = ((n0*IH + tih0)*IW + tiw0)*IC + ic0;
	int X_offset1 = ((n1*IH + tih1)*IW + tiw1)*IC + ic0;
	int X_offset2 = ((n2*IH + tih2)*IW + tiw2)*IC + ic0;
	int X_offset3 = ((n3*IH + tih3)*IW + tiw3)*IC + ic0;
	int X_offset4 = ((n4*IH + tih4)*IW + tiw4)*IC + ic0;
	int X_offset5 = ((n5*IH + tih5)*IW + tiw5)*IC + ic0;
	int X_offset6 = ((n6*IH + tih6)*IW + tiw6)*IC + ic0;
	int X_offset7 = ((n7*IH + tih7)*IW + tiw7)*IC + ic0;

	bool flagY = (ty >= STEP);
	const int tn0 = (n4 - n0)*flagY + n0;
	const int tn1 = (n5 - n1)*flagY + n1;
	const int tn2 = (n6 - n2)*flagY + n2;
	const int tn3 = (n7 - n3)*flagY + n3;
	const int tohs0 = (ohs4 - ohs0)*flagY + ohs0;
	const int tohs1 = (ohs5 - ohs1)*flagY + ohs1;
	const int tohs2 = (ohs6 - ohs2)*flagY + ohs2;
	const int tohs3 = (ohs7 - ohs3)*flagY + ohs3;
	const int tows0 = (ows4 - ows0)*flagY + ows0;
	const int tows1 = (ows5 - ows1)*flagY + ows1;
	const int tows2 = (ows6 - ows2)*flagY + ows2;
	const int tows3 = (ows7 - ows3)*flagY + ows3;
	int Y_offset0 = ((tn0*OH + tohs0)*OW + tows0)*OC;
	int Y_offset1 = ((tn1*OH + tohs1)*OW + tows1)*OC;
	int Y_offset2 = ((tn2*OH + tohs2)*OW + tows2)*OC;
	int Y_offset3 = ((tn3*OH + tohs3)*OW + tows3)*OC;

	const int GK = FH * FW * IC, strideY = (IW - sw)*IC;
	for (int y = 0; y < sh; y++, deltaX += strideY) {
		for (int x = 0; x < sw; x++, deltaX += IC)
		{
			//compute area-----------------------------------------------------
			float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
			float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
			float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
			float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
			float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
			float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
			float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
			float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);
			for (int fhr = 0; fhr <= CFH; fhr++) {
				for (int fwr = 0; fwr <= CFW; fwr++)
				{
					//load 4 elements from W[OC, FH, FW, IC]
					int fh = y + (CFH - fhr)*sh, fw = x + (CFW - fwr)*sw;
					bool lw =  (fh < FH) && (fw < FW);
					int W_offset = (fh*FW + fw)*IC + tic0;
					int W_oc = tx - ((tx >= STEP) << LB >> 1);
					Ws[buf][tx][ty] = lw ? *(float4*)(&W[W_oc*GK + W_offset]) : make_float4(0, 0, 0, 0);

					//load 4 elements from deltaY[N, OH, OW, OC]
					bool ldy0 = (tohs0 >= -fhr) && (tohs0 < OH - fhr) && (tows0 >= -fwr) && (tows0 < OW - fwr);
					bool ldy1 = (tohs1 >= -fhr) && (tohs1 < OH - fhr) && (tows1 >= -fwr) && (tows1 < OW - fwr);
					bool ldy2 = (tohs2 >= -fhr) && (tohs2 < OH - fhr) && (tows2 >= -fwr) && (tows2 < OW - fwr);
					bool ldy3 = (tohs3 >= -fhr) && (tohs3 < OH - fhr) && (tows3 >= -fwr) && (tows3 < OW - fwr);
					int Y_offset = (fhr*OW + fwr)*OC;
					int Y_oc = ty - ((ty >= STEP) << LB >> 1);
					dYs[buf][ty][tx].x = ldy0 * tex1Dfetch<float>(deltaY, Y_offset0 + Y_offset + Y_oc);
					dYs[buf][ty][tx].y = ldy1 * tex1Dfetch<float>(deltaY, Y_offset1 + Y_offset + Y_oc);
					dYs[buf][ty][tx].z = ldy2 * tex1Dfetch<float>(deltaY, Y_offset2 + Y_offset + Y_oc);
					dYs[buf][ty][tx].w = ldy3 * tex1Dfetch<float>(deltaY, Y_offset3 + Y_offset + Y_oc);
					__syncthreads();

					for (int ooc = 1, OOC = OC << 1 >> LB; ooc < OOC; ooc++) {
#pragma unroll
						for (int ik = 0; ik < STEP; ik++)
						{
							float4 dy0 = dYs[buf][ik][tx], dy1 = dYs[buf][ik + STEP][tx];
							float4  w0 = Ws[buf][ik][ty], w1 = Ws[buf][ik + STEP][ty];

							simdMM4(v0, dy0.x, w0); simdMM4(v1, dy0.x, w1);
							simdMM4(v2, dy0.y, w0); simdMM4(v3, dy0.y, w1);
							simdMM4(v4, dy0.z, w0); simdMM4(v5, dy0.z, w1);
							simdMM4(v6, dy0.w, w0); simdMM4(v7, dy0.w, w1);
							simdMM4(v8, dy1.x, w0); simdMM4(v9, dy1.x, w1);
							simdMM4(v10, dy1.y, w0); simdMM4(v11, dy1.y, w1);
							simdMM4(v12, dy1.z, w0); simdMM4(v13, dy1.z, w1);
							simdMM4(v14, dy1.w, w0); simdMM4(v15, dy1.w, w1);
						}
						buf ^= 1;

						//load 4 elements from W[OC, FH, FW, IC]
						int W_oc = ((ooc - (tx >= STEP)) << LB >> 1) + tx;
						Ws[buf][tx][ty] = lw ? *(float4*)(&W[W_oc*GK + W_offset]) : make_float4(0, 0, 0, 0);

						//load 4 elements from deltaY[N, OH, OW, OC]
						int Y_oc = ((ooc - (ty >= STEP)) << LB >> 1) + ty;
						dYs[buf][ty][tx].x = ldy0 * tex1Dfetch<float>(deltaY, Y_offset0 + Y_offset + Y_oc);
						dYs[buf][ty][tx].y = ldy1 * tex1Dfetch<float>(deltaY, Y_offset1 + Y_offset + Y_oc);
						dYs[buf][ty][tx].z = ldy2 * tex1Dfetch<float>(deltaY, Y_offset2 + Y_offset + Y_oc);
						dYs[buf][ty][tx].w = ldy3 * tex1Dfetch<float>(deltaY, Y_offset3 + Y_offset + Y_oc);
						__syncthreads();
					}
#pragma unroll
					for (int ik = 0; ik < STEP; ik++)
					{
						float4 dy0 = dYs[buf][ik][tx], dy1 = dYs[buf][ik + STEP][tx];
						float4  w0 = Ws[buf][ik][ty], w1 = Ws[buf][ik + STEP][ty];

						simdMM4(v0, dy0.x, w0); simdMM4(v1, dy0.x, w1);
						simdMM4(v2, dy0.y, w0); simdMM4(v3, dy0.y, w1);
						simdMM4(v4, dy0.z, w0); simdMM4(v5, dy0.z, w1);
						simdMM4(v6, dy0.w, w0); simdMM4(v7, dy0.w, w1);
						simdMM4(v8, dy1.x, w0); simdMM4(v9, dy1.x, w1);
						simdMM4(v10, dy1.y, w0); simdMM4(v11, dy1.y, w1);
						simdMM4(v12, dy1.z, w0); simdMM4(v13, dy1.z, w1);
						simdMM4(v14, dy1.w, w0); simdMM4(v15, dy1.w, w1);
					}
					buf ^= 1;
				}
			}

			bool wrt0 = (tih0 >= -y) && (tih0 < IH - y) && (tiw0 >= -x) && (tiw0 < IW - x) && (n0 < N);
			bool wrt1 = (tih1 >= -y) && (tih1 < IH - y) && (tiw1 >= -x) && (tiw1 < IW - x) && (n1 < N);
			bool wrt2 = (tih2 >= -y) && (tih2 < IH - y) && (tiw2 >= -x) && (tiw2 < IW - x) && (n2 < N);
			bool wrt3 = (tih3 >= -y) && (tih3 < IH - y) && (tiw3 >= -x) && (tiw3 < IW - x) && (n3 < N);
			if (wrt0) { *(float4*)(&deltaX[X_offset0]) = v0;  *(float4*)(&deltaX[X_offset0 + 4]) = v1; }
			if (wrt1) { *(float4*)(&deltaX[X_offset1]) = v2;  *(float4*)(&deltaX[X_offset1 + 4]) = v3; }
			if (wrt2) { *(float4*)(&deltaX[X_offset2]) = v4;  *(float4*)(&deltaX[X_offset2 + 4]) = v5; }
			if (wrt3) { *(float4*)(&deltaX[X_offset3]) = v6;  *(float4*)(&deltaX[X_offset3 + 4]) = v7; }

			bool wrt4 = (tih4 >= -y) && (tih4 < IH - y) && (tiw4 >= -x) && (tiw4 < IW - x) && (n4 < N);
			bool wrt5 = (tih5 >= -y) && (tih5 < IH - y) && (tiw5 >= -x) && (tiw5 < IW - x) && (n5 < N);
			bool wrt6 = (tih6 >= -y) && (tih6 < IH - y) && (tiw6 >= -x) && (tiw6 < IW - x) && (n6 < N);
			bool wrt7 = (tih7 >= -y) && (tih7 < IH - y) && (tiw7 >= -x) && (tiw7 < IW - x) && (n7 < N);
			if (wrt4) { *(float4*)(&deltaX[X_offset4]) = v8;  *(float4*)(&deltaX[X_offset4 + 4]) = v9; }
			if (wrt5) { *(float4*)(&deltaX[X_offset5]) = v10; *(float4*)(&deltaX[X_offset5 + 4]) = v11; }
			if (wrt6) { *(float4*)(&deltaX[X_offset6]) = v12; *(float4*)(&deltaX[X_offset6 + 4]) = v13; }
			if (wrt7) { *(float4*)(&deltaX[X_offset7]) = v14; *(float4*)(&deltaX[X_offset7 + 4]) = v15; }
		}
	}
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*1) 
//LB = 4: OC % 8 == 0
#ifndef DCONV3D_DX_KERNEL_SPLIT_KERNEL_4_4_TEXTURE
#define DCONV3D_DX_KERNEL_SPLIT_KERNEL_4_4_TEXTURE

//for(sh, sw) = 4:
//LB=4: Size = 3.63379, Time = 2.77 msec, Performace = 2817.15 GFlop/s
//LB=3: Size = 3.63379, Time = 3.49 msec, Performace = 2235.96 GFlop/s
//for(sh, sw) = 2:
//LB=4: Size = 0.938477, Time = 1.61  msec, Performace = 1251.78 GFlop/s
//LB=3: Size = 0.938477, Time = 1.966 msec, Performace = 1025.11 GFlop/s
template<int LB, int STEP>
__global__ void kernelSplit_kernel_4_4_texture(
	cudaTextureObject_t deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw,
	int CFH, int CFW,
	int ic_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2  Ws[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2 dYs[2][1 << LB >> 1][(2 << LB) + 2];

	//preapre for GN = IC
	const int ic0 = (((blockIdx.y << LB) + ty) << 2) + ic_index;
	const int tic0 = ((tx & 1) << 1) + ic0;

	//prepared for GM = N * OHS * OWS
	const int j0 = (((blockIdx.x << LB) + tx) << 2) + j_index;//[n, ohs, ows]
	const int j1 = j0 + 1, j2 = j0 + 2, j3 = j0 + 3;
	const int oph = --CFH, opw = --CFW;
	const int OHS = OH + CFH;//OHS = OH - CFH + (oph << 1) + 1
	const int OWS = OW + CFW;//OWS = OW - CFW + (opw << 1) + 1;
	const int OHS_OWS = OHS * OWS;
	get_n_ohs_ows(j0, n0, ohs0, ows0);
	get_n_ohs_ows(j1, n1, ohs1, ows1);
	get_n_ohs_ows(j2, n2, ohs2, ows2);
	get_n_ohs_ows(j3, n3, ohs3, ows3);
	const int tih0 = ohs0 * sh - ph; ohs0 -= oph;
	const int tih1 = ohs1 * sh - ph; ohs1 -= oph;
	const int tih2 = ohs2 * sh - ph; ohs2 -= oph;
	const int tih3 = ohs3 * sh - ph; ohs3 -= oph;
	const int tiw0 = ows0 * sw - pw; ows0 -= opw;
	const int tiw1 = ows1 * sw - pw; ows1 -= opw;
	const int tiw2 = ows2 * sw - pw; ows2 -= opw;
	const int tiw3 = ows3 * sw - pw; ows3 -= opw;
	int X_offset0 = ((n0*IH + tih0)*IW + tiw0)*IC + ic0;
	int X_offset1 = ((n1*IH + tih1)*IW + tiw1)*IC + ic0;
	int X_offset2 = ((n2*IH + tih2)*IW + tiw2)*IC + ic0;
	int X_offset3 = ((n3*IH + tih3)*IW + tiw3)*IC + ic0;

	bool flagY = (ty & 1);
	const int tn0 = (n2 - n0)*flagY + n0;
	const int tn1 = (n3 - n1)*flagY + n1;
	const int tohs0 = (ohs2 - ohs0)*flagY + ohs0;
	const int tohs1 = (ohs3 - ohs1)*flagY + ohs1;
	const int tows0 = (ows2 - ows0)*flagY + ows0;
	const int tows1 = (ows3 - ows1)*flagY + ows1;
	int Y_offset0 = ((tn0*OH + tohs0)*OW + tows0)*OC;
	int Y_offset1 = ((tn1*OH + tohs1)*OW + tows1)*OC;

	const int GK = FH * FW * IC, strideY = (IW - sw)*IC;
	const int Ws_x = (tx >> 1), Ws_y = (ty << 1) + (tx & 1);
	const int dYs_y = (ty >> 1), dYs_x = (tx << 1) + (ty & 1);
	for (int y = 0; y < sh; y++, deltaX += strideY) {
		for (int x = 0; x < sw; x++, deltaX += IC)
		{
			//compute area-----------------------------------------------------
			float4 v0 = make_float4(0, 0, 0, 0);
			float4 v1 = make_float4(0, 0, 0, 0);
			float4 v2 = make_float4(0, 0, 0, 0);
			float4 v3 = make_float4(0, 0, 0, 0);
			for (int fhr = 0; fhr <= CFH; fhr++) {
				for (int fwr = 0; fwr <= CFW; ++fwr) 
				{
					//load 2 elements from W[OC, FH, FW, IC]
					int fh = y + (CFH - fhr)*sh, fw = x + (CFW - fwr)*sw;
					bool lw =  (fh < FH) && (fw < FW);
					int W_oc = tx >> 1, W_offset = (fh*FW + fw)*IC + tic0;
					Ws[buf][Ws_x][Ws_y] = lw ? *(float2*)(&W[W_oc*GK + W_offset]) : make_float2(0, 0);

					//load 2 elements from deltaY[N, OH, OW, OC]
					bool ldy0 = (tohs0 >= -fhr) && (tohs0 < OH - fhr) && (tows0 >= -fwr) && (tows0 < OW - fwr);
					bool ldy1 = (tohs1 >= -fhr) && (tohs1 < OH - fhr) && (tows1 >= -fwr) && (tows1 < OW - fwr);
					int Y_oc = ty >> 1, Y_offset = (fhr*OW + fwr)*OC;
					dYs[buf][dYs_y][dYs_x].x = ldy0 * tex1Dfetch<float>(deltaY, Y_offset0 + Y_offset + Y_oc);
					dYs[buf][dYs_y][dYs_x].y = ldy1 * tex1Dfetch<float>(deltaY, Y_offset1 + Y_offset + Y_oc);
					__syncthreads();

					for (int ooc = 1, OOC = OC << 1 >> LB; ooc < OOC; ++ooc) {
#pragma unroll
						for (int ik = 0; ik < STEP; ik++) {
							float4 dy = *(float4*)(&dYs[buf][ik][tx << 1]);
							float4 w = *(float4*)(&Ws[buf][ik][ty << 1]);
							simdMM4(v0, dy.x, w);
							simdMM4(v1, dy.y, w);
							simdMM4(v2, dy.z, w);
							simdMM4(v3, dy.w, w);
						}
						buf ^= 1;

						//load 2 elements from W[OC, FH, FW, IC]
						int W_oc = ((ooc << LB) + tx) >> 1;
						Ws[buf][Ws_x][Ws_y] = lw ? *(float2*)(&W[W_oc*GK + W_offset]) : make_float2(0, 0);

						//load 2 elements from deltaY[N, OH, OW, OC]
						int Y_oc = ((ooc << LB) + ty) >> 1;
						dYs[buf][dYs_y][dYs_x].x = ldy0 * tex1Dfetch<float>(deltaY, Y_offset0 + Y_offset + Y_oc);
						dYs[buf][dYs_y][dYs_x].y = ldy1 * tex1Dfetch<float>(deltaY, Y_offset1 + Y_offset + Y_oc);
						__syncthreads();
					}
#pragma unroll
					for (int ik = 0; ik < STEP; ik++) {
						float4 dy = *(float4*)(&dYs[buf][ik][tx << 1]);
						float4 w = *(float4*)(&Ws[buf][ik][ty << 1]);
						simdMM4(v0, dy.x, w);
						simdMM4(v1, dy.y, w);
						simdMM4(v2, dy.z, w);
						simdMM4(v3, dy.w, w);
					}
					buf ^= 1;
				}
			}

			bool wrt0 = (tih0 >= -y) && (tih0 < IH - y) && (tiw0 >= -x) && (tiw0 < IW - x) && (n0 < N);
			bool wrt1 = (tih1 >= -y) && (tih1 < IH - y) && (tiw1 >= -x) && (tiw1 < IW - x) && (n1 < N);
			bool wrt2 = (tih2 >= -y) && (tih2 < IH - y) && (tiw2 >= -x) && (tiw2 < IW - x) && (n2 < N);
			bool wrt3 = (tih3 >= -y) && (tih3 < IH - y) && (tiw3 >= -x) && (tiw3 < IW - x) && (n3 < N);
			if (wrt0) *(float4*)(&deltaX[X_offset0]) = v0; 
			if (wrt1) *(float4*)(&deltaX[X_offset1]) = v1;
			if (wrt2) *(float4*)(&deltaX[X_offset2]) = v2;
			if (wrt3) *(float4*)(&deltaX[X_offset3]) = v3;
		}
	}
}

#endif


//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*4) 
//LB = 4: OC % 8 == 0
#ifndef DCONV3D_DX_KERNEL_SPLIT_KERNEL_2_4_TEXTURE
#define DCONV3D_DX_KERNEL_SPLIT_KERNEL_2_4_TEXTURE

//for(sh, sw) = 4:
//LB=4: Size = 3.63379, Time = 4.432 msec, Performace = 1760.72 GFlop/
//LB=3: Size = 3.63379, Time = 6.11  msec, Performace = 1277.17 GFlop/s
//for(sh, sw) = 2:
//LB=4: Size = 0.938477, Time = 2.468 msec, Performace = 816.598 GFlop/s
//LB=3: Size = 0.938477, Time = 3.088 msec, Performace = 652.643 GFlop/s
template<int LB, int STEP>
__global__ void kernelSplit_kernel_2_4_texture(
	cudaTextureObject_t deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw,
	int CFH, int CFW,
	int ic_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float   Ws[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2 dYs[2][1 << LB >> 1][(2 << LB) + 2];

	//preapre for GN = IC
	const int ic0 = (((blockIdx.y << LB) + ty) << 1) + ic_index;
	const int tic0 = (tx & 1) + ic0;

	//prepared for GM = N * OHS * OWS
	const int j0 = (((blockIdx.x << LB) + tx) << 2) + j_index;//[n, ohs, ows]
	const int j1 = j0 + 1, j2 = j0 + 2, j3 = j0 + 3;
	const int oph = --CFH, opw = --CFW;
	const int OHS = OH + CFH;//OHS = OH - CFH + (oph << 1) + 1
	const int OWS = OW + CFW;//OWS = OW - CFW + (opw << 1) + 1;
	const int OHS_OWS = OHS * OWS;
	get_n_ohs_ows(j0, n0, ohs0, ows0);
	get_n_ohs_ows(j1, n1, ohs1, ows1);
	get_n_ohs_ows(j2, n2, ohs2, ows2);
	get_n_ohs_ows(j3, n3, ohs3, ows3);
	const int tih0 = ohs0 * sh - ph; ohs0 -= oph;
	const int tih1 = ohs1 * sh - ph; ohs1 -= oph;
	const int tih2 = ohs2 * sh - ph; ohs2 -= oph;
	const int tih3 = ohs3 * sh - ph; ohs3 -= oph;
	const int tiw0 = ows0 * sw - pw; ows0 -= opw;
	const int tiw1 = ows1 * sw - pw; ows1 -= opw;
	const int tiw2 = ows2 * sw - pw; ows2 -= opw;
	const int tiw3 = ows3 * sw - pw; ows3 -= opw;
	int X_offset0 = ((n0*IH + tih0)*IW + tiw0)*IC + ic0;
	int X_offset1 = ((n1*IH + tih1)*IW + tiw1)*IC + ic0;
	int X_offset2 = ((n2*IH + tih2)*IW + tiw2)*IC + ic0;
	int X_offset3 = ((n3*IH + tih3)*IW + tiw3)*IC + ic0;

	bool flagY = (ty & 1);
	const int tn0 = (n2 - n0)*flagY + n0;
	const int tn1 = (n3 - n1)*flagY + n1;
	const int tohs0 = (ohs2 - ohs0)*flagY + ohs0;
	const int tohs1 = (ohs3 - ohs1)*flagY + ohs1;
	const int tows0 = (ows2 - ows0)*flagY + ows0;
	const int tows1 = (ows3 - ows1)*flagY + ows1;
	int Y_offset0 = ((tn0*OH + tohs0)*OW + tows0)*OC;
	int Y_offset1 = ((tn1*OH + tohs1)*OW + tows1)*OC;

	const int GK = FH * FW * IC, strideY = (IW - sw)*IC;
	const int Ws_x = (tx >> 1), Ws_y = (ty << 1) + (tx & 1);
	const int dYs_y = (ty >> 1), dYs_x = (tx << 1) + (ty & 1);
	for (int y = 0; y < sh; y++, deltaX += strideY) {
		for (int x = 0; x < sw; x++, deltaX += IC)
		{
			//compute area-----------------------------------------------------
			float2 v0 = make_float2(0, 0);
			float2 v1 = make_float2(0, 0);
			float2 v2 = make_float2(0, 0);
			float2 v3 = make_float2(0, 0);
			for (int fhr = 0; fhr <= CFH; fhr++) {
				for (int fwr = 0; fwr <= CFW; ++fwr)
				{
					//load 1 element from W[OC, FH, FW, IC]
					int fh = y + (CFH - fhr)*sh, fw = x + (CFW - fwr)*sw;
					bool lw =  (fh < FH) && (fw < FW);
					int W_oc = tx >> 1, W_offset = (fh*FW + fw)*IC + tic0;
					Ws[buf][Ws_x][Ws_y] = lw ? W[W_oc*GK + W_offset] : 0;

					//load 2 elements from deltaY[N, OH, OW, OC]
					bool ldy0 = (tohs0 >= -fhr) && (tohs0 < OH - fhr) && (tows0 >= -fwr) && (tows0 < OW - fwr);
					bool ldy1 = (tohs1 >= -fhr) && (tohs1 < OH - fhr) && (tows1 >= -fwr) && (tows1 < OW - fwr);
					int Y_oc = ty >> 1, Y_offset = (fhr*OW + fwr)*OC;
					dYs[buf][dYs_y][dYs_x].x = ldy0 * tex1Dfetch<float>(deltaY, Y_offset0 + Y_offset + Y_oc);
					dYs[buf][dYs_y][dYs_x].y = ldy1 * tex1Dfetch<float>(deltaY, Y_offset1 + Y_offset + Y_oc);
					__syncthreads();

					for (int ooc = 1, OOC = OC << 1 >> LB; ooc < OOC; ++ooc) {
#pragma unroll
						for (int ik = 0; ik < STEP; ik++) {
							float4 dy = *(float4*)(&dYs[buf][ik][tx << 1]);
							float2 w = *(float2*)(&Ws[buf][ik][ty << 1]);
							simdMM2(v0, dy.x, w);
							simdMM2(v1, dy.y, w);
							simdMM2(v2, dy.z, w);
							simdMM2(v3, dy.w, w);
						}
						buf ^= 1;

						//load 1 element from W[OC, FH, FW, IC]
						int W_oc = ((ooc << LB) + tx) >> 1;
						Ws[buf][Ws_x][Ws_y] = lw ? W[W_oc*GK + W_offset] : 0;

						//load 2 elements from deltaY[N, OH, OW, OC]
						int Y_oc = ((ooc << LB) + ty) >> 1;
						dYs[buf][dYs_y][dYs_x].x = ldy0 * tex1Dfetch<float>(deltaY, Y_offset0 + Y_offset + Y_oc);
						dYs[buf][dYs_y][dYs_x].y = ldy1 * tex1Dfetch<float>(deltaY, Y_offset1 + Y_offset + Y_oc);
						__syncthreads();
					}
#pragma unroll
					for (int ik = 0; ik < STEP; ik++) {
						float4 dy = *(float4*)(&dYs[buf][ik][tx << 1]);
						float2 w = *(float2*)(&Ws[buf][ik][ty << 1]);
						simdMM2(v0, dy.x, w);
						simdMM2(v1, dy.y, w);
						simdMM2(v2, dy.z, w);
						simdMM2(v3, dy.w, w);
					}
					buf ^= 1;
				}
			}

			bool wrt0 = (tih0 >= -y) && (tih0 < IH - y) && (tiw0 >= -x) && (tiw0 < IW - x) && (n0 < N);
			bool wrt1 = (tih1 >= -y) && (tih1 < IH - y) && (tiw1 >= -x) && (tiw1 < IW - x) && (n1 < N);
			bool wrt2 = (tih2 >= -y) && (tih2 < IH - y) && (tiw2 >= -x) && (tiw2 < IW - x) && (n2 < N);
			bool wrt3 = (tih3 >= -y) && (tih3 < IH - y) && (tiw3 >= -x) && (tiw3 < IW - x) && (n3 < N);
			if (wrt0) *(float2*)(&deltaX[X_offset0]) = v0;
			if (wrt1) *(float2*)(&deltaX[X_offset1]) = v1;
			if (wrt2) *(float2*)(&deltaX[X_offset2]) = v2;
			if (wrt3) *(float2*)(&deltaX[X_offset3]) = v3;
		}
	}
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*2) 
//LB = 4: OC % 16 == 0
//LB = 3: OC %  8 == 0
#ifndef DCONV3D_DX_KERNEL_SPLIT_KERNEL_2_2_TEXTURE
#define DCONV3D_DX_KERNEL_SPLIT_KERNEL_2_2_TEXTURE

//for(sh, sw) = 4:
//LB=3: Size = 0.908447, Time = 1.636 msec, Performace = 1192.47 GFlop/s
//for(sh, sw) = 2:
//LB=4: Size = 0.938477, Time = 2.542 msec, Performace = 792.826 GFlop/s
//LB=3: Size = 0.938477, Time = 2.814 msec, Performace = 716.192 GFlop/s
//for(sh=2, sw=2, IC=16, N=4):
//LB=3: Size = 0.234619, Time = 0.908 msec, Performace = 554.891 GFlop/s
template<int LB, int STEP>
__global__ void kernelSplit_kernel_2_2_texture(
	cudaTextureObject_t deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw,
	int CFH, int CFW,
	int ic_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2  Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float2 dYs[2][1 << LB][(1 << LB) + 1];

	//preapre for GN = IC
	const int ic0 = (((blockIdx.y << LB) + ty) << 1) + ic_index;

	//prepared for GM = N * OHS * OWS
	const int j0 = (((blockIdx.x << LB) + tx) << 1) + j_index, j1 = j0 + 1;
	const int oph = --CFH, opw = --CFW;
	const int OHS = OH + CFH;//OHS = OH - CFH + (oph << 1) + 1
	const int OWS = OW + CFW;//OWS = OW - CFW + (opw << 1) + 1;
	const int OHS_OWS = OHS * OWS;
	get_n_ohs_ows(j0, n0, ohs0, ows0);
	get_n_ohs_ows(j1, n1, ohs1, ows1);
	const int tih0 = ohs0 * sh - ph; ohs0 -= oph;
	const int tih1 = ohs1 * sh - ph; ohs1 -= oph;
	const int tiw0 = ows0 * sw - pw; ows0 -= opw;
	const int tiw1 = ows1 * sw - pw; ows1 -= opw;
	int X_offset0 = ((n0*IH + tih0)*IW + tiw0)*IC + ic0;
	int X_offset1 = ((n1*IH + tih1)*IW + tiw1)*IC + ic0;
	int Y_offset0 = ((n0*OH + ohs0)*OW + ows0)*OC;
	int Y_offset1 = ((n1*OH + ohs1)*OW + ows1)*OC;

	const int GK = FH * FW * IC, strideY = (IW - sw)*IC;
	for (int y = 0; y < sh; y++, deltaX += strideY) {
		for (int x = 0; x < sw; x++, deltaX += IC)
		{
			//compute area-----------------------------------------------------
			float2 v0 = make_float2(0, 0);
			float2 v1 = make_float2(0, 0);
			for (int fhr = 0; fhr <= CFH; fhr++) {
				for (int fwr = 0; fwr <= CFW; ++fwr)
				{
					//load 2 element from W[OC, FH, FW, IC]
					int fh = y + (CFH - fhr)*sh, fw = x + (CFW - fwr)*sw;
					bool lw =  (fh < FH) && (fw < FW);
					int W_oc = tx, W_offset = (fh*FW + fw)*IC + ic0;
					Ws[buf][tx][ty] = lw ? *(float2*)(&W[W_oc*GK + W_offset]) : make_float2(0, 0);

					//load 2 elements from deltaY[N, OH, OW, OC]
					bool ldy0 = (ohs0 >= -fhr) && (ohs0 < OH - fhr) && (ows0 >= -fwr) && (ows0 < OW - fwr);
					bool ldy1 = (ohs1 >= -fhr) && (ohs1 < OH - fhr) && (ows1 >= -fwr) && (ows1 < OW - fwr);
					int Y_oc = ty, Y_offset = (fhr*OW + fwr)*OC;
					dYs[buf][ty][tx].x = ldy0 * tex1Dfetch<float>(deltaY, Y_offset0 + Y_offset + Y_oc);
					dYs[buf][ty][tx].y = ldy1 * tex1Dfetch<float>(deltaY, Y_offset1 + Y_offset + Y_oc);
					__syncthreads();

					for (int ooc = 1, OOC = OC >> LB; ooc < OOC; ++ooc) {
#pragma unroll
						for (int ik = 0; ik < STEP; ik++) {
							float2 dy = dYs[buf][ik][tx];
							float2 w = Ws[buf][ik][ty];
							simdMM2(v0, dy.x, w);
							simdMM2(v1, dy.y, w);
						}
						buf ^= 1;

						//load 2 elements from W[OC, FH, FW, IC]
						int W_oc = (ooc << LB) + tx;
						Ws[buf][tx][ty] = lw ? *(float2*)(&W[W_oc*GK + W_offset]) : make_float2(0, 0);

						//load 2 elements from deltaY[N, OH, OW, OC]
						int Y_oc = (ooc << LB) + ty;
						dYs[buf][ty][tx].x = ldy0 * tex1Dfetch<float>(deltaY, Y_offset0 + Y_offset + Y_oc);
						dYs[buf][ty][tx].y = ldy1 * tex1Dfetch<float>(deltaY, Y_offset1 + Y_offset + Y_oc);
						__syncthreads();
					}
#pragma unroll
					for (int ik = 0; ik < STEP; ik++) {
						float2 dy = dYs[buf][ik][tx];
						float2 w = Ws[buf][ik][ty];
						simdMM2(v0, dy.x, w);
						simdMM2(v1, dy.y, w);
					}
					buf ^= 1;
				}
			}

			bool wrt0 = (tih0 >= -y) && (tih0 < IH - y) && (tiw0 >= -x) && (tiw0 < IW - x) && (n0 < N);
			bool wrt1 = (tih1 >= -y) && (tih1 < IH - y) && (tiw1 >= -x) && (tiw1 < IW - x) && (n1 < N);
			if (wrt0) *(float2*)(&deltaX[X_offset0]) = v0;
			if (wrt1) *(float2*)(&deltaX[X_offset1]) = v1;
		}
	}
}

#endif


//(Y: BLOCK_SIZE*1, X: BLOCK_SIZE*2) 
//LB = 4: OC % 16 == 0
//LB = 3: OC %  8 == 0
#ifndef DCONV3D_DX_KERNEL_SPLIT_KERNEL_1_2_TEXTURE
#define DCONV3D_DX_KERNEL_SPLIT_KERNEL_1_2_TEXTURE

//for(sh, sw) = 2:
//LB=4: Size = 0.938477, Time = 4.706 msec, Performace = 428.254 GFlop/s(OC = 16)
//LB=3: Size = 0.938477, Time = 5.218 msec, Performace = 386.233 GFlop/s(OC = 8)
template<int LB, int STEP>
__global__ void kernelSplit_kernel_1_2_texture(
	cudaTextureObject_t deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw,
	int CFH, int CFW,
	int ic_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float   Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float2 dYs[2][1 << LB][(1 << LB) + 1];

	//preapre for GN = IC
	const int ic0 = ((blockIdx.y << LB) + ty) + ic_index;

	//prepared for GM = N * OHS * OWS
	const int j0 = (((blockIdx.x << LB) + tx) << 1) + j_index, j1 = j0 + 1;
	const int oph = --CFH, opw = --CFW;
	const int OHS = OH + CFH;//OHS = OH - CFH + (oph << 1) + 1
	const int OWS = OW + CFW;//OWS = OW - CFW + (opw << 1) + 1;
	const int OHS_OWS = OHS * OWS;
	get_n_ohs_ows(j0, n0, ohs0, ows0);
	get_n_ohs_ows(j1, n1, ohs1, ows1);
	const int tih0 = ohs0 * sh - ph; ohs0 -= oph;
	const int tih1 = ohs1 * sh - ph; ohs1 -= oph;
	const int tiw0 = ows0 * sw - pw; ows0 -= opw;
	const int tiw1 = ows1 * sw - pw; ows1 -= opw;
	int X_offset0 = ((n0*IH + tih0)*IW + tiw0)*IC + ic0;
	int X_offset1 = ((n1*IH + tih1)*IW + tiw1)*IC + ic0;
	int Y_offset0 = ((n0*OH + ohs0)*OW + ows0)*OC;
	int Y_offset1 = ((n1*OH + ohs1)*OW + ows1)*OC;

	const int GK = FH * FW * IC, strideY = (IW - sw)*IC;
	for (int y = 0; y < sh; y++, deltaX += strideY) {
		for (int x = 0; x < sw; x++, deltaX += IC)
		{
			//compute area-----------------------------------------------------
			float2 v0 = make_float2(0, 0);
			for (int fhr = 0; fhr <= CFH; fhr++) {
				for (int fwr = 0; fwr <= CFW; ++fwr)
				{
					//load 1 element from W[OC, FH, FW, IC]
					int fh = y + (CFH - fhr)*sh, fw = x + (CFW - fwr)*sw;
					bool lw =  (fh < FH) && (fw < FW);
					int W_oc = tx, W_offset = (fh*FW + fw)*IC + ic0;
					Ws[buf][tx][ty] = lw ? W[W_oc*GK + W_offset] : 0;

					//load 2 elements from deltaY[N, OH, OW, OC]
					bool ldy0 = (ohs0 >= -fhr) && (ohs0 < OH - fhr) && (ows0 >= -fwr) && (ows0 < OW - fwr);
					bool ldy1 = (ohs1 >= -fhr) && (ohs1 < OH - fhr) && (ows1 >= -fwr) && (ows1 < OW - fwr);
					int Y_oc = ty, Y_offset = (fhr*OW + fwr)*OC;
					dYs[buf][ty][tx].x = ldy0 * tex1Dfetch<float>(deltaY, Y_offset0 + Y_offset + Y_oc);
					dYs[buf][ty][tx].y = ldy1 * tex1Dfetch<float>(deltaY, Y_offset1 + Y_offset + Y_oc);
					__syncthreads();

					for (int ooc = 1, OOC = OC >> LB; ooc < OOC; ++ooc) {
#pragma unroll
						for (int ik = 0; ik < STEP; ik++) {
							float2 dy = dYs[buf][ik][tx];
							float w = Ws[buf][ik][ty];
							simdMM2(v0, w, dy);
						}
						buf ^= 1;

						//load 1 element from W[OC, FH, FW, IC]
						int W_oc = (ooc << LB) + tx;
						Ws[buf][tx][ty] = lw ? W[W_oc*GK + W_offset] : 0;

						//load 2 elements from deltaY[N, OH, OW, OC]
						int Y_oc = (ooc << LB) + ty;
						dYs[buf][ty][tx].x = ldy0 * tex1Dfetch<float>(deltaY, Y_offset0 + Y_offset + Y_oc);
						dYs[buf][ty][tx].y = ldy1 * tex1Dfetch<float>(deltaY, Y_offset1 + Y_offset + Y_oc);
						__syncthreads();
					}
#pragma unroll
					for (int ik = 0; ik < STEP; ik++) {
						float2 dy = dYs[buf][ik][tx];
						float w = Ws[buf][ik][ty];
						simdMM2(v0, w, dy);
					}
					buf ^= 1;
				}
			}

			bool wrt0 = (tih0 >= -y) && (tih0 < IH - y) && (tiw0 >= -x) && (tiw0 < IW - x) && (n0 < N);
			bool wrt1 = (tih1 >= -y) && (tih1 < IH - y) && (tiw1 >= -x) && (tiw1 < IW - x) && (n1 < N);
			if (wrt0) deltaX[X_offset0] = v0.x;
			if (wrt1) deltaX[X_offset1] = v0.y;
		}
	}
}

#endif


#endif