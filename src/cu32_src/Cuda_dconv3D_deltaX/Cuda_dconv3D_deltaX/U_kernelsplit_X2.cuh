

#define X_k88(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, CFH, CFW, GN, GM) \
	X_kernel_8_8<LB, (1<<LB>>1)>\
		<<< dim3((GM + (8<<LB) - 1)>>LB>>3, GN>>LB>>3, (sh*sw)), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, N,IC,OC, sh,sw,ph,pw, CFH, CFW, ic_index, j_index)


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8) 
//LB = 4: OC % 8 == 0
#ifndef X_KERNEL_8_8
#define X_KERNEL_8_8

//LB = 4: Size = 1, Time = 1.724 msec, Performace = 1245.64 GFlop/s
template<int LB, int STEP>
__global__ void X_kernel_8_8(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw,
	int CFH, int CFW,
	int ic_index, int j_index)
{
	int bz = blockIdx.z;
	int y = bz / sw, x = bz % sw;
	deltaX += (y*IW + x) * IC;

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

	bool flagY = (ty >= STEP);
	const int tn0 = (n4 - n0)*flagY + n0;
	const int tn1 = (n5 - n1)*flagY + n1;
	const int tn2 = (n6 - n2)*flagY + n2;
	const int tn3 = (n7 - n3)*flagY + n3;
	const int tohs0 = (ohs4 - ohs0)*flagY + ohs0 - oph;
	const int tohs1 = (ohs5 - ohs1)*flagY + ohs1 - oph;
	const int tohs2 = (ohs6 - ohs2)*flagY + ohs2 - oph;
	const int tohs3 = (ohs7 - ohs3)*flagY + ohs3 - oph;
	const int tows0 = (ows4 - ows0)*flagY + ows0 - opw;
	const int tows1 = (ows5 - ows1)*flagY + ows1 - opw;
	const int tows2 = (ows6 - ows2)*flagY + ows2 - opw;
	const int tows3 = (ows7 - ows3)*flagY + ows3 - opw;
	int Y_offset0 = ((tn0*OH + tohs0)*OW + tows0)*OC;
	int Y_offset1 = ((tn1*OH + tohs1)*OW + tows1)*OC;
	int Y_offset2 = ((tn2*OH + tohs2)*OW + tows2)*OC;
	int Y_offset3 = ((tn3*OH + tohs3)*OW + tows3)*OC;

	const int GK = FH * FW * IC;
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
		for (int fwr = 0; fwr <= CFW; ++fwr)
		{
			//load 4 elements from W[OC, FH, FW, IC]
			int fh = y + (CFH - fhr)*sh, fw = x + (CFW - fwr)*sw;
			bool lw = (fh < FH) && (fw < FW);
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
			dYs[buf][ty][tx].x = ldy0 ? deltaY[Y_offset0 + Y_offset + Y_oc] : 0;
			dYs[buf][ty][tx].y = ldy1 ? deltaY[Y_offset1 + Y_offset + Y_oc] : 0;
			dYs[buf][ty][tx].z = ldy2 ? deltaY[Y_offset2 + Y_offset + Y_oc] : 0;
			dYs[buf][ty][tx].w = ldy3 ? deltaY[Y_offset3 + Y_offset + Y_oc] : 0;
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
				dYs[buf][ty][tx].x = ldy0 ? deltaY[Y_offset0 + Y_offset + Y_oc] : 0;
				dYs[buf][ty][tx].y = ldy1 ? deltaY[Y_offset1 + Y_offset + Y_oc] : 0;
				dYs[buf][ty][tx].z = ldy2 ? deltaY[Y_offset2 + Y_offset + Y_oc] : 0;
				dYs[buf][ty][tx].w = ldy3 ? deltaY[Y_offset3 + Y_offset + Y_oc] : 0;
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

	const int tih0 = ohs0 * sh - ph, tiw0 = ows0 * sw - pw;
	const int tih1 = ohs1 * sh - ph, tiw1 = ows1 * sw - pw;
	const int tih2 = ohs2 * sh - ph, tiw2 = ows2 * sw - pw;
	const int tih3 = ohs3 * sh - ph, tiw3 = ows3 * sw - pw;
	const int tih4 = ohs4 * sh - ph, tiw4 = ows4 * sw - pw;
	const int tih5 = ohs5 * sh - ph, tiw5 = ows5 * sw - pw;
	const int tih6 = ohs6 * sh - ph, tiw6 = ows6 * sw - pw;
	const int tih7 = ohs7 * sh - ph, tiw7 = ows7 * sw - pw;
	int X_offset0 = ((n0*IH + tih0)*IW + tiw0)*IC + ic0;
	int X_offset1 = ((n1*IH + tih1)*IW + tiw1)*IC + ic0;
	int X_offset2 = ((n2*IH + tih2)*IW + tiw2)*IC + ic0;
	int X_offset3 = ((n3*IH + tih3)*IW + tiw3)*IC + ic0;
	int X_offset4 = ((n4*IH + tih4)*IW + tiw4)*IC + ic0;
	int X_offset5 = ((n5*IH + tih5)*IW + tiw5)*IC + ic0;
	int X_offset6 = ((n6*IH + tih6)*IW + tiw6)*IC + ic0;
	int X_offset7 = ((n7*IH + tih7)*IW + tiw7)*IC + ic0;
	bool wrt0 = (tih0 >= -y) && (tih0 < IH - y) && (tiw0 >= -x) && (tiw0 < IW - x) && (n0 < N);
	if (wrt0) { *(float4*)(&deltaX[X_offset0]) = v0;  *(float4*)(&deltaX[X_offset0 + 4]) = v1; }

	bool wrt1 = (tih1 >= -y) && (tih1 < IH - y) && (tiw1 >= -x) && (tiw1 < IW - x) && (n1 < N);
	if (wrt1) { *(float4*)(&deltaX[X_offset1]) = v2;  *(float4*)(&deltaX[X_offset1 + 4]) = v3; }

	bool wrt2 = (tih2 >= -y) && (tih2 < IH - y) && (tiw2 >= -x) && (tiw2 < IW - x) && (n2 < N);
	if (wrt2) { *(float4*)(&deltaX[X_offset2]) = v4;  *(float4*)(&deltaX[X_offset2 + 4]) = v5; }


	bool wrt3 = (tih3 >= -y) && (tih3 < IH - y) && (tiw3 >= -x) && (tiw3 < IW - x) && (n3 < N);
	if (wrt3) { *(float4*)(&deltaX[X_offset3]) = v6;  *(float4*)(&deltaX[X_offset3 + 4]) = v7; }

	bool wrt4 = (tih4 >= -y) && (tih4 < IH - y) && (tiw4 >= -x) && (tiw4 < IW - x) && (n4 < N);
	if (wrt4) { *(float4*)(&deltaX[X_offset4]) = v8;  *(float4*)(&deltaX[X_offset4 + 4]) = v9; }

	bool wrt5 = (tih5 >= -y) && (tih5 < IH - y) && (tiw5 >= -x) && (tiw5 < IW - x) && (n5 < N);
	if (wrt5) { *(float4*)(&deltaX[X_offset5]) = v10; *(float4*)(&deltaX[X_offset5 + 4]) = v11; }

	bool wrt6 = (tih6 >= -y) && (tih6 < IH - y) && (tiw6 >= -x) && (tiw6 < IW - x) && (n6 < N);
	if (wrt6) { *(float4*)(&deltaX[X_offset6]) = v12; *(float4*)(&deltaX[X_offset6 + 4]) = v13; }

	bool wrt7 = (tih7 >= -y) && (tih7 < IH - y) && (tiw7 >= -x) && (tiw7 < IW - x) && (n7 < N);
	if (wrt7) { *(float4*)(&deltaX[X_offset7]) = v14; *(float4*)(&deltaX[X_offset7 + 4]) = v15; }
}

#endif
