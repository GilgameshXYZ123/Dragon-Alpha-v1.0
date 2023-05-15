


#define InitParam_X(N, OH, OW, FH, FW, sh, sw, ph, pw) \
	float Nh0 = KernelSplit_Nh0(FH, sh);\
	float Nw0 = KernelSplit_Nw0(FW, sw);\
	bool lh5 = KernelSplit_lh5(Nh0);\
	bool lw5 = KernelSplit_lw5(Nw0);\
	int CFH = KernelSplit_CFH(Nh0, lh5);\
	int CFW = KernelSplit_CFW(Nw0, lw5);\
	int oph = KernelSplit_oph(CFH);\
	int opw = KernelSplit_opw(CFW);\
	int GN = sh*sw*KernelSplit_GN(IC);\
	int GM = KernelSplit_GM(sh, sw, N, OH, OW, CFH, CFW, oph, opw);

#define get_y_x_ic(i, y, x, ic) \
	int y, x, ic; {y = i / sw_IC; int ir = i - y*sw_IC; x = ir / IC; ic = ir % IC; }


#define X_k22_tex(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, CFH, CFW, GN, GM) \
	X_2_2_texture<LB, (1<<LB)>\
		<<< dim3((GM + (2<<LB) - 1)>>LB>>1, GN>>LB>>1), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, N,IC,OC, sh,sw,ph,pw, CFH, CFW, ic_index, j_index)

#define X_k88_tex(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, CFH, CFW, GN, GM) \
	X_8_8_texture<LB, (1<<LB>>1)>\
		<<< dim3((GM + (8<<LB) - 1)>>LB>>3, GN>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, N,IC,OC, sh,sw,ph,pw, CFH, CFW, ic_index, j_index)

#define X_k88_tex_s2(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, CFH, CFW, GN, GM) \
	X_8_8_texture_s2<LB, (1<<LB>>1)>\
		<<< dim3((GM + (8<<LB) - 1)>>LB>>3, GN>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, N,IC,OC, ph,pw, CFH, CFW, ic_index, j_index)

#define X_k88_tex_s2_P(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, N, IC, OC, sh, sw, ph, pw, CFH, CFW, GN, GM) \
	X_8_8_texture_s2_P<LB, (1<<LB>>1)>\
		<<< dim3((GM + (8<<LB) - 1)>>LB>>3, GN>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, N,IC,OC, ph,pw, CFH, CFW, ic_index, j_index)


#ifndef X_2_2_TEXTURE
#define X_2_2_TEXTURE

template<int LB, int STEP>
__global__ void X_2_2_texture(
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
	const int i0 = (((blockIdx.y << LB) + ty) << 1) + ic_index, i1 = i0 + 1;
	const int sw_IC = sw * IC;
	get_y_x_ic(i0, y0, x0, ic0); 
	get_y_x_ic(i1, y1, x1, ic1);

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
	int X_offset0 = ((n0*IH + tih0 + y0)*IW + tiw0 + x0)*IC;
	int X_offset1 = ((n1*IH + tih1 + y1)*IW + tiw1 + x1)*IC;
	bool wrt0 = (tih0 >= -y0) && (tih0 < IH - y0) && (tiw0 >= -x0) && (tiw0 < IW - x0) && (n0 < N);
	bool wrt1 = (tih1 >= -y1) && (tih1 < IH - y1) && (tiw1 >= -x1) && (tiw1 < IW - x1) && (n1 < N);
	int Y_offset0 = ((n0*OH + ohs0)*OW + ows0)*OC;
	int Y_offset1 = ((n1*OH + ohs1)*OW + ows1)*OC;
	
	//compute area-----------------------------------------------------
	float2 v0 = make_float2(0, 0);
	float2 v1 = make_float2(0, 0);

	const int GK = FH * FW* IC;
	for (int fhr = 0; fhr <= CFH; fhr++) {
		for (int fwr = 0; fwr <= CFW; fwr++)
		{
			//load 2 element from W[OC, FH, FW, IC]
			int fh_offset = (CFH - fhr)*sh, fw_offset = (CFW - fwr)*sw;
			int fh0 = y0 + fh_offset, fw0 = x0 + fw_offset;
			int fh1 = y1 + fh_offset, fw1 = x1 + fw_offset;
			int W_offset0 = (fh0*FW + fw0)*IC + ic0;
			int W_offset1 = (fh1*FW + fw1)*IC + ic1;
			bool lw0 = (fh0 >= 0) && (fw0 >= 0) && (fh0 < FH) && (fw0 < FW);
			bool lw1 = (fh1 >= 0) && (fw1 >= 0) && (fh1 < FH) && (fw1 < FW);
			int W_oc = tx * GK;
			Ws[buf][tx][ty].x = lw0 ? W[W_oc + W_offset0] : 0;
			Ws[buf][tx][ty].y = lw1 ? W[W_oc + W_offset1] : 0;

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
				int W_oc = ((ooc << LB) + tx) * GK;
				Ws[buf][tx][ty].x = lw0 ? W[W_oc + W_offset0] : 0;
				Ws[buf][tx][ty].y = lw1 ? W[W_oc + W_offset1] : 0;

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

	if (wrt0) *(float2*)(deltaX + X_offset0 + ic0) = v0;
	if (wrt1) *(float2*)(deltaX + X_offset1 + ic0) = v1;
	//if (wrt0 && ic0 < IC) deltaX[X_offset0 + ic0] = v0.x;
	//if (wrt0 && ic1 < IC) deltaX[X_offset0 + ic1] = v0.y;
	//if (wrt1 && ic0 < IC) deltaX[X_offset1 + ic0] = v1.x;
	//if (wrt1 && ic1 < IC) deltaX[X_offset1 + ic1] = v1.y;
}

#endif


#ifndef X_8_8_TEXTURE
#define X_8_8_TEXTURE

//Size = 1, Time = 1.626 msec, Performace = 1320.72 GFlop/s
template<int LB, int STEP>
__global__ void X_8_8_texture(
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
	const int i0 = (((blockIdx.y << LB) + ty) << 3) + ic_index;
	const int i1 = i0 + 1, i2 = i0 + 2, i3 = i0 + 3;
	const int i4 = i0 + 4, i5 = i0 + 5, i6 = i0 + 6, i7 = i0 + 7;
	const int sw_IC = sw * IC;
	get_y_x_ic(i0, y0, x0, ic0);
	get_y_x_ic(i1, y1, x1, ic1);
	get_y_x_ic(i2, y2, x2, ic2);
	get_y_x_ic(i3, y3, x3, ic3);
	get_y_x_ic(i4, y4, x4, ic4);
	get_y_x_ic(i5, y5, x5, ic5);
	get_y_x_ic(i6, y6, x6, ic6); 
	get_y_x_ic(i7, y7, x7, ic7);
	bool flagX = (tx >= STEP);
	const int ty0 = (y4 - y0)*flagX + y0;
	const int ty1 = (y5 - y1)*flagX + y1;
	const int ty2 = (y6 - y2)*flagX + y2;
	const int ty3 = (y7 - y3)*flagX + y3;
	const int tx0 = (x4 - x0)*flagX + x0;
	const int tx1 = (x5 - x1)*flagX + x1;
	const int tx2 = (x6 - x2)*flagX + x2;
	const int tx3 = (x7 - x3)*flagX + x3;
	const int tic0 = (ic4 - ic0)*flagX + ic0;
	const int tic1 = (ic5 - ic1)*flagX + ic1;
	const int tic2 = (ic6 - ic2)*flagX + ic2;
	const int tic3 = (ic7 - ic3)*flagX + ic3;

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
	int X_offset0 = ((n0*IH + tih0 + y0)*IW + tiw0 + x0)*IC + ic0;
	int X_offset1 = ((n1*IH + tih1 + y1)*IW + tiw1 + x1)*IC + ic0;
	int X_offset2 = ((n2*IH + tih2 + y2)*IW + tiw2 + x2)*IC + ic0;
	int X_offset3 = ((n3*IH + tih3 + y3)*IW + tiw3 + x3)*IC + ic0;
	int X_offset4 = ((n4*IH + tih4 + y4)*IW + tiw4 + x4)*IC + ic0;
	int X_offset5 = ((n5*IH + tih5 + y5)*IW + tiw5 + x5)*IC + ic0;
	int X_offset6 = ((n6*IH + tih6 + y6)*IW + tiw6 + x6)*IC + ic0;
	int X_offset7 = ((n7*IH + tih7 + y7)*IW + tiw7 + x7)*IC + ic0;
	bool wrt0 = (tih0 >= -y0) && (tih0 < IH - y0) && (tiw0 >= -x0) && (tiw0 < IW - x0) && (n0 < N);
	bool wrt1 = (tih1 >= -y1) && (tih1 < IH - y1) && (tiw1 >= -x1) && (tiw1 < IW - x1) && (n1 < N);
	bool wrt2 = (tih2 >= -y2) && (tih2 < IH - y2) && (tiw2 >= -x2) && (tiw2 < IW - x2) && (n2 < N);
	bool wrt3 = (tih3 >= -y3) && (tih3 < IH - y3) && (tiw3 >= -x3) && (tiw3 < IW - x3) && (n3 < N);
	bool wrt4 = (tih4 >= -y4) && (tih4 < IH - y4) && (tiw4 >= -x4) && (tiw4 < IW - x4) && (n4 < N);
	bool wrt5 = (tih5 >= -y5) && (tih5 < IH - y5) && (tiw5 >= -x5) && (tiw5 < IW - x5) && (n5 < N);
	bool wrt6 = (tih6 >= -y6) && (tih6 < IH - y6) && (tiw6 >= -x6) && (tiw6 < IW - x6) && (n6 < N);
	bool wrt7 = (tih7 >= -y7) && (tih7 < IH - y7) && (tiw7 >= -x7) && (tiw7 < IW - x7) && (n7 < N);

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
	
	//compute area-----------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);
	const int GK = FH * FW * IC;
	for (int fhr = 0; fhr <= CFH; fhr++) {
		for (int fwr = 0; fwr <= CFW; fwr++)
		{
			//load 4 elements from W[OC, FH, FW, IC]
			int fh = (CFH - fhr)*sh, fw = (CFW - fwr)*sw;
			int fh0 = ty0 + fh, fw0 = tx0 + fw;
			int fh1 = ty1 + fh, fw1 = tx1 + fw;
			int fh2 = ty2 + fh, fw2 = tx2 + fw;
			int fh3 = ty3 + fh, fw3 = tx3 + fw;
			bool lw0 = (fh0 >= 0) && (fw0 >= 0) && (fh0 < FH) && (fw0 < FW);
			bool lw1 = (fh1 >= 0) && (fw1 >= 0) && (fh1 < FH) && (fw1 < FW);
			bool lw2 = (fh2 >= 0) && (fw2 >= 0) && (fh2 < FH) && (fw2 < FW);
			bool lw3 = (fh3 >= 0) && (fw3 >= 0) && (fh3 < FH) && (fw3 < FW);
			int W_offset0 = (fh0*FW + fw0)*IC + tic0;
			int W_offset1 = (fh1*FW + fw1)*IC + tic1;
			int W_offset2 = (fh2*FW + fw2)*IC + tic2;
			int W_offset3 = (fh3*FW + fw3)*IC + tic3;
			int W_oc = (tx - ((tx >= STEP) << LB >> 1)) * GK;
			Ws[buf][tx][ty].x = lw0 ? W[W_oc + W_offset0] : 0;
			Ws[buf][tx][ty].y = lw1 ? W[W_oc + W_offset1] : 0;
			Ws[buf][tx][ty].z = lw2 ? W[W_oc + W_offset2] : 0;
			Ws[buf][tx][ty].w = lw3 ? W[W_oc + W_offset3] : 0;

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
				int W_oc = (((ooc - (tx >= STEP)) << LB >> 1) + tx) * GK;
				Ws[buf][tx][ty].x = lw0 ? W[W_oc + W_offset0] : 0;
				Ws[buf][tx][ty].y = lw1 ? W[W_oc + W_offset1] : 0;
				Ws[buf][tx][ty].z = lw2 ? W[W_oc + W_offset2] : 0;
				Ws[buf][tx][ty].w = lw3 ? W[W_oc + W_offset3] : 0;

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

	if (wrt0) { *(float4*)(&deltaX[X_offset0]) = v0;  *(float4*)(&deltaX[X_offset0 + 4]) = v1; }
	if (wrt1) { *(float4*)(&deltaX[X_offset1]) = v2;  *(float4*)(&deltaX[X_offset1 + 4]) = v3; }
	if (wrt2) { *(float4*)(&deltaX[X_offset2]) = v4;  *(float4*)(&deltaX[X_offset2 + 4]) = v5; }
	if (wrt3) { *(float4*)(&deltaX[X_offset3]) = v6;  *(float4*)(&deltaX[X_offset3 + 4]) = v7; }
	if (wrt4) { *(float4*)(&deltaX[X_offset4]) = v8;  *(float4*)(&deltaX[X_offset4 + 4]) = v9; }
	if (wrt5) { *(float4*)(&deltaX[X_offset5]) = v10; *(float4*)(&deltaX[X_offset5 + 4]) = v11; }
	if (wrt6) { *(float4*)(&deltaX[X_offset6]) = v12; *(float4*)(&deltaX[X_offset6 + 4]) = v13; }
	if (wrt7) { *(float4*)(&deltaX[X_offset7]) = v14; *(float4*)(&deltaX[X_offset7 + 4]) = v15; }
}

#endif


#ifndef X_8_8_TEXTURE_S2
#define X_8_8_TEXTURE_S2

//IC % 8 ==0
//int y, x, ic; {y = i / sw_IC; int ir = i - y*sw_IC; x = ir / IC; ic = ir%IC; }
	//sw * IC % 8 == 0
	//so: yx = y0 + x
	//y = i / sw_IC，相邻的8个i得出相同的y，因此所有的y都是相同的
	//ir = i % sw_IC, x = ir / IC， 相邻的4个i得出相同的x， 如果IC是8的倍数

//Size = 1, Time = 1.626 msec, Performace = 1320.72 GFlop/s
//Size = 1, Time = 1.9 msec, Performace = 1130.25 GFlop / s
template<int LB, int STEP>
__global__ void X_8_8_texture_s2(
	cudaTextureObject_t deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int OC, 
	int ph, int pw,
	int CFH, int CFW,
	int ic_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];

	//preapre for GN = IC, 
	const int i0 = (((blockIdx.y << LB) + ty) << 3) + ic_index;
	const int sw_IC = (IC << 1);
	get_y_x_ic(i0, y, x, ic0);
	const int tic0 = ((tx >= STEP) << 2) + ic0;

	//prepared for GM = N * OHS * OWS
	const int j0 = (((blockIdx.x << LB) + tx) << 3) + j_index;//[n, ohs, ows]
	const int j1 = j0 + 1, j2 = j0 + 2, j3 = j0 + 3;
	const int j4 = j0 + 4, j5 = j0 + 5, j6 = j0 + 6, j7 = j0 + 7;
	const int OHS = OH + (--CFH), OWS = OW + (--CFW);
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
	const int tohs0 = (ohs4 - ohs0)*flagY + ohs0 - CFH;
	const int tohs1 = (ohs5 - ohs1)*flagY + ohs1 - CFH;
	const int tohs2 = (ohs6 - ohs2)*flagY + ohs2 - CFH;
	const int tohs3 = (ohs7 - ohs3)*flagY + ohs3 - CFH;
	const int tows0 = (ows4 - ows0)*flagY + ows0 - CFW;
	const int tows1 = (ows5 - ows1)*flagY + ows1 - CFW;
	const int tows2 = (ows6 - ows2)*flagY + ows2 - CFW;
	const int tows3 = (ows7 - ows3)*flagY + ows3 - CFW;
	int Y_offset0 = ((tn0*OH + tohs0)*OW + tows0)*OC;
	int Y_offset1 = ((tn1*OH + tohs1)*OW + tows1)*OC;
	int Y_offset2 = ((tn2*OH + tohs2)*OW + tows2)*OC;
	int Y_offset3 = ((tn3*OH + tohs3)*OW + tows3)*OC;

	//compute area-----------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);
	const int GK = FH * FW * IC;
	for (int fhr = 0; fhr <= CFH; fhr++) {
		for (int fwr = 0; fwr <= CFW; fwr++)
		{
			//load 4 elements from W[OC, FH, FW, IC]
			int fh = y + ((CFH - fhr) << 1), fw = x + ((CFW - fwr) << 1);
			int W_offset = (fh*FW + fw)*IC + tic0;
			int W_oc = (tx - ((tx >= STEP) << LB >> 1)) * GK;
			bool lw = (fh < FH) && (fw < FW);
			Ws[buf][tx][ty] = lw ? *(float4*)(W + W_oc + W_offset) : make_float4(0, 0, 0, 0);

			//load 4 elements from deltaY[N, OH, OW, OC]
			int Y_offset = (fhr*OW + fwr)*OC;
			int Y_oc = (ty - ((ty >= STEP) << LB >> 1)) + Y_offset;
			bool ldy0 = (tohs0 >= -fhr) && (tohs0 < OH - fhr) && (tows0 >= -fwr) && (tows0 < OW - fwr);
			bool ldy1 = (tohs1 >= -fhr) && (tohs1 < OH - fhr) && (tows1 >= -fwr) && (tows1 < OW - fwr);
			bool ldy2 = (tohs2 >= -fhr) && (tohs2 < OH - fhr) && (tows2 >= -fwr) && (tows2 < OW - fwr);
			bool ldy3 = (tohs3 >= -fhr) && (tohs3 < OH - fhr) && (tows3 >= -fwr) && (tows3 < OW - fwr);
			Ys[buf][ty][tx].x = ldy0 * tex1Dfetch<float>(deltaY, Y_offset0 + Y_oc);
			Ys[buf][ty][tx].y = ldy1 * tex1Dfetch<float>(deltaY, Y_offset1 + Y_oc);
			Ys[buf][ty][tx].z = ldy2 * tex1Dfetch<float>(deltaY, Y_offset2 + Y_oc);
			Ys[buf][ty][tx].w = ldy3 * tex1Dfetch<float>(deltaY, Y_offset3 + Y_oc);
			__syncthreads();

			for (int ooc = 1, OOC = OC << 1 >> LB; ooc < OOC; ooc++) {
#pragma unroll
				for (int ik = 0; ik < STEP; ik++)
				{
					float4 y0 = Ys[buf][ik][tx], y1 = Ys[buf][ik + STEP][tx];
					float4 w0 = Ws[buf][ik][ty], w1 = Ws[buf][ik + STEP][ty];

					simdMM4(v0,  y0.x, w0); simdMM4(v1,  y0.x, w1);
					simdMM4(v2,  y0.y, w0); simdMM4(v3,  y0.y, w1);
					simdMM4(v4,  y0.z, w0); simdMM4(v5,  y0.z, w1);
					simdMM4(v6,  y0.w, w0); simdMM4(v7,  y0.w, w1);
					simdMM4(v8,  y1.x, w0); simdMM4(v9,  y1.x, w1);
					simdMM4(v10, y1.y, w0); simdMM4(v11, y1.y, w1);
					simdMM4(v12, y1.z, w0); simdMM4(v13, y1.z, w1);
					simdMM4(v14, y1.w, w0); simdMM4(v15, y1.w, w1);
				}
				buf ^= 1;

				//load 4 elements from W[OC, FH, FW, IC]
				int W_oc = (((ooc - (tx >= STEP)) << LB >> 1) + tx) * GK;
				Ws[buf][tx][ty] = lw ? *(float4*)(W + W_oc + W_offset) : make_float4(0, 0, 0, 0);

				//load 4 elements from deltaY[N, OH, OW, OC]
				int Y_oc = ((ooc - (ty >= STEP)) << LB >> 1) + ty + Y_offset;
				Ys[buf][ty][tx].x = ldy0 * tex1Dfetch<float>(deltaY, Y_offset0 + Y_oc);
				Ys[buf][ty][tx].y = ldy1 * tex1Dfetch<float>(deltaY, Y_offset1 + Y_oc);
				Ys[buf][ty][tx].z = ldy2 * tex1Dfetch<float>(deltaY, Y_offset2 + Y_oc);
				Ys[buf][ty][tx].w = ldy3 * tex1Dfetch<float>(deltaY, Y_offset3 + Y_oc);
				__syncthreads();
			}
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 y0 = Ys[buf][ik][tx], y1 = Ys[buf][ik + STEP][tx];
				float4 w0 =  Ws[buf][ik][ty], w1 = Ws[buf][ik + STEP][ty];

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
		}
	}

	ph = y - ph; pw = x - pw; deltaX += ic0;
	const int tih0 = (ohs0 << 1) + ph, tiw0 = (ows0 << 1) + pw;
	const int tih1 = (ohs1 << 1) + ph, tiw1 = (ows1 << 1) + pw;
	const int tih2 = (ohs2 << 1) + ph, tiw2 = (ows2 << 1) + pw;
	const int tih3 = (ohs3 << 1) + ph, tiw3 = (ows3 << 1) + pw;
	const int tih4 = (ohs4 << 1) + ph, tiw4 = (ows4 << 1) + pw;
	const int tih5 = (ohs5 << 1) + ph, tiw5 = (ows5 << 1) + pw;
	const int tih6 = (ohs6 << 1) + ph, tiw6 = (ows6 << 1) + pw;
	const int tih7 = (ohs7 << 1) + ph, tiw7 = (ows7 << 1) + pw;
	bool wrt0 = (tih0 >= 0) && (tih0 < IH) && (tiw0 >= 0) && (tiw0 < IW) && (n0 < N);
	bool wrt1 = (tih1 >= 0) && (tih1 < IH) && (tiw1 >= 0) && (tiw1 < IW) && (n1 < N);
	bool wrt2 = (tih2 >= 0) && (tih2 < IH) && (tiw2 >= 0) && (tiw2 < IW) && (n2 < N);
	bool wrt3 = (tih3 >= 0) && (tih3 < IH) && (tiw3 >= 0) && (tiw3 < IW) && (n3 < N);
	bool wrt4 = (tih4 >= 0) && (tih4 < IH) && (tiw4 >= 0) && (tiw4 < IW) && (n4 < N);
	bool wrt5 = (tih5 >= 0) && (tih5 < IH) && (tiw5 >= 0) && (tiw5 < IW) && (n5 < N);
	bool wrt6 = (tih6 >= 0) && (tih6 < IH) && (tiw6 >= 0) && (tiw6 < IW) && (n6 < N);
	bool wrt7 = (tih7 >= 0) && (tih7 < IH) && (tiw7 >= 0) && (tiw7 < IW) && (n7 < N);
	int X_offset0 = ((n0*IH + tih0)*IW + tiw0)*IC;
	int X_offset1 = ((n1*IH + tih1)*IW + tiw1)*IC;
	int X_offset2 = ((n2*IH + tih2)*IW + tiw2)*IC;
	int X_offset3 = ((n3*IH + tih3)*IW + tiw3)*IC;
	int X_offset4 = ((n4*IH + tih4)*IW + tiw4)*IC;
	int X_offset5 = ((n5*IH + tih5)*IW + tiw5)*IC;
	int X_offset6 = ((n6*IH + tih6)*IW + tiw6)*IC;
	int X_offset7 = ((n7*IH + tih7)*IW + tiw7)*IC;
	if (wrt0) { *(float4*)(deltaX + X_offset0) = v0;  *(float4*)(deltaX + X_offset0 + 4) = v1; }
	if (wrt1) { *(float4*)(deltaX + X_offset1) = v2;  *(float4*)(deltaX + X_offset1 + 4) = v3; }
	if (wrt2) { *(float4*)(deltaX + X_offset2) = v4;  *(float4*)(deltaX + X_offset2 + 4) = v5; }
	if (wrt3) { *(float4*)(deltaX + X_offset3) = v6;  *(float4*)(deltaX + X_offset3 + 4) = v7; }
	if (wrt4) { *(float4*)(deltaX + X_offset4) = v8;  *(float4*)(deltaX + X_offset4 + 4) = v9; }
	if (wrt5) { *(float4*)(deltaX + X_offset5) = v10; *(float4*)(deltaX + X_offset5 + 4) = v11; }
	if (wrt6) { *(float4*)(deltaX + X_offset6) = v12; *(float4*)(deltaX + X_offset6 + 4) = v13; }
	if (wrt7) { *(float4*)(deltaX + X_offset7) = v14; *(float4*)(deltaX + X_offset7 + 4) = v15; }
}

#endif


#ifndef X_8_8_TEXTURE_S2_P
#define X_8_8_TEXTURE_S2_P

//IC % 8 ==0
//int y, x, ic; {y = i / sw_IC; int ir = i - y*sw_IC; x = ir / IC; ic = ir%IC; }
	//sw * IC % 8 == 0
	//so: yx = y0 + x
	//y = i / sw_IC，相邻的8个i得出相同的y，因此所有的y都是相同的
	//ir = i % sw_IC, x = ir / IC， 相邻的4个i得出相同的x， 如果IC是8的倍数

//Size = 1, Time = 1.626 msec, Performace = 1320.72 GFlop/s
//Size = 1, Time = 1.9 msec, Performace = 1130.25 GFlop / s
template<int LB, int STEP>
__global__ void X_8_8_texture_s2_P(
	cudaTextureObject_t deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int OC,
	int ph, int pw,
	int CFH, int CFW,
	int ic_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];

	//preapre for GN = IC, 
	const int i0 = (((blockIdx.y << LB) + ty) << 3) + ic_index;
	const int sw_IC = (IC << 1);
	get_y_x_ic(i0, y, x, ic0);
	const int tic0 = ((tx >= STEP) << 2) + ic0;

	//prepared for GM = N * OHS * OWS
	const int j0 = (((blockIdx.x << LB) + tx) << 3) + j_index;//[n, ohs, ows]
	const int j1 = j0 + 1, j2 = j0 + 2, j3 = j0 + 3;
	const int j4 = j0 + 4, j5 = j0 + 5, j6 = j0 + 6, j7 = j0 + 7;
	const int OHS = OH + (--CFH), OWS = OW + (--CFW);
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
	const int tohs0 = (ohs4 - ohs0)*flagY + ohs0 - CFH;
	const int tohs1 = (ohs5 - ohs1)*flagY + ohs1 - CFH;
	const int tohs2 = (ohs6 - ohs2)*flagY + ohs2 - CFH;
	const int tohs3 = (ohs7 - ohs3)*flagY + ohs3 - CFH;
	const int tows0 = (ows4 - ows0)*flagY + ows0 - CFW;
	const int tows1 = (ows5 - ows1)*flagY + ows1 - CFW;
	const int tows2 = (ows6 - ows2)*flagY + ows2 - CFW;
	const int tows3 = (ows7 - ows3)*flagY + ows3 - CFW;
	int Y_offset0 = ((tn0*OH + tohs0)*OW + tows0)*OC;
	int Y_offset1 = ((tn1*OH + tohs1)*OW + tows1)*OC;
	int Y_offset2 = ((tn2*OH + tohs2)*OW + tows2)*OC;
	int Y_offset3 = ((tn3*OH + tohs3)*OW + tows3)*OC;

	//compute area-----------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);
	const int GK = FH * FW * IC;
	for (int fhr = 0; fhr <= CFH; fhr++) {
		for (int fwr = 0; fwr <= CFW; fwr++)
		{
			//load 4 elements from W[OC, FH, FW, IC]
			int fh = y + ((CFH - fhr) << 1), fw = x + ((CFW - fwr) << 1);
			int W_offset = (fh*FW + fw)*IC + tic0;
			int W_oc = (tx - ((tx >= STEP) << LB >> 1)) * GK;
			bool lw = (fh < FH) && (fw < FW);
			Ws[buf][tx][ty] = lw ? *(float4*)(W + W_oc + W_offset) : make_float4(0, 0, 0, 0);

			//load 4 elements from deltaY[N, OH, OW, OC]
			int Y_offset = (fhr*OW + fwr)*OC;
			int Y_oc = (ty - ((ty >= STEP) << LB >> 1)) + Y_offset;
			bool ldy0 = (tohs0 >= -fhr) && (tohs0 < OH - fhr) && (tows0 >= -fwr) && (tows0 < OW - fwr);
			bool ldy1 = (tohs1 >= -fhr) && (tohs1 < OH - fhr) && (tows1 >= -fwr) && (tows1 < OW - fwr);
			bool ldy2 = (tohs2 >= -fhr) && (tohs2 < OH - fhr) && (tows2 >= -fwr) && (tows2 < OW - fwr);
			bool ldy3 = (tohs3 >= -fhr) && (tohs3 < OH - fhr) && (tows3 >= -fwr) && (tows3 < OW - fwr);
			Ys[buf][ty][tx].x = ldy0 * tex1Dfetch<float>(deltaY, Y_offset0 + Y_oc);
			Ys[buf][ty][tx].y = ldy1 * tex1Dfetch<float>(deltaY, Y_offset1 + Y_oc);
			Ys[buf][ty][tx].z = ldy2 * tex1Dfetch<float>(deltaY, Y_offset2 + Y_oc);
			Ys[buf][ty][tx].w = ldy3 * tex1Dfetch<float>(deltaY, Y_offset3 + Y_oc);
			__syncthreads();

			for (int ooc = 1, OOC = OC << 1 >> LB; ooc < OOC; ooc++) {
#pragma unroll
				for (int ik = 0; ik < STEP; ik++)
				{
					float4 y0 = Ys[buf][ik][tx], y1 = Ys[buf][ik + STEP][tx];
					float4 w0 = Ws[buf][ik][ty], w1 = Ws[buf][ik + STEP][ty];

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

				//load 4 elements from W[OC, FH, FW, IC]
				int W_oc = (((ooc - (tx >= STEP)) << LB >> 1) + tx) * GK;
				Ws[buf][tx][ty] = lw ? *(float4*)(W + W_oc + W_offset) : make_float4(0, 0, 0, 0);

				//load 4 elements from deltaY[N, OH, OW, OC]
				int Y_oc = ((ooc - (ty >= STEP)) << LB >> 1) + ty + Y_offset;
				Ys[buf][ty][tx].x = ldy0 * tex1Dfetch<float>(deltaY, Y_offset0 + Y_oc);
				Ys[buf][ty][tx].y = ldy1 * tex1Dfetch<float>(deltaY, Y_offset1 + Y_oc);
				Ys[buf][ty][tx].z = ldy2 * tex1Dfetch<float>(deltaY, Y_offset2 + Y_oc);
				Ys[buf][ty][tx].w = ldy3 * tex1Dfetch<float>(deltaY, Y_offset3 + Y_oc);
				__syncthreads();
			}
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 y0 = Ys[buf][ik][tx], y1 = Ys[buf][ik + STEP][tx];
				float4 w0 = Ws[buf][ik][ty], w1 = Ws[buf][ik + STEP][ty];

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
		}
	}

	ph = y - ph; pw = x - pw; deltaX += ic0;
	const int tih0 = (ohs0 << 1) + ph, tiw0 = (ows0 << 1) + pw;
	const int tih1 = (ohs1 << 1) + ph, tiw1 = (ows1 << 1) + pw;
	const int tih2 = (ohs2 << 1) + ph, tiw2 = (ows2 << 1) + pw;
	const int tih3 = (ohs3 << 1) + ph, tiw3 = (ows3 << 1) + pw;
	const int tih4 = (ohs4 << 1) + ph, tiw4 = (ows4 << 1) + pw;
	const int tih5 = (ohs5 << 1) + ph, tiw5 = (ows5 << 1) + pw;
	const int tih6 = (ohs6 << 1) + ph, tiw6 = (ows6 << 1) + pw;
	const int tih7 = (ohs7 << 1) + ph, tiw7 = (ows7 << 1) + pw;
	bool wrt0 = (tih0 >= 0) && (tih0 < IH) && (tiw0 >= 0) && (tiw0 < IW) && (n0 < N);
	bool wrt1 = (tih1 >= 0) && (tih1 < IH) && (tiw1 >= 0) && (tiw1 < IW) && (n1 < N);
	bool wrt2 = (tih2 >= 0) && (tih2 < IH) && (tiw2 >= 0) && (tiw2 < IW) && (n2 < N);
	bool wrt3 = (tih3 >= 0) && (tih3 < IH) && (tiw3 >= 0) && (tiw3 < IW) && (n3 < N);
	bool wrt4 = (tih4 >= 0) && (tih4 < IH) && (tiw4 >= 0) && (tiw4 < IW) && (n4 < N);
	bool wrt5 = (tih5 >= 0) && (tih5 < IH) && (tiw5 >= 0) && (tiw5 < IW) && (n5 < N);
	bool wrt6 = (tih6 >= 0) && (tih6 < IH) && (tiw6 >= 0) && (tiw6 < IW) && (n6 < N);
	bool wrt7 = (tih7 >= 0) && (tih7 < IH) && (tiw7 >= 0) && (tiw7 < IW) && (n7 < N);
	int X_offset0 = ((n0*IH + tih0)*IW + tiw0)*IC;
	int X_offset1 = ((n1*IH + tih1)*IW + tiw1)*IC;
	int X_offset2 = ((n2*IH + tih2)*IW + tiw2)*IC;
	int X_offset3 = ((n3*IH + tih3)*IW + tiw3)*IC;
	int X_offset4 = ((n4*IH + tih4)*IW + tiw4)*IC;
	int X_offset5 = ((n5*IH + tih5)*IW + tiw5)*IC;
	int X_offset6 = ((n6*IH + tih6)*IW + tiw6)*IC;
	int X_offset7 = ((n7*IH + tih7)*IW + tiw7)*IC;
	if (wrt0) { *(float4*)(deltaX + X_offset0) = v0;  *(float4*)(deltaX + X_offset0 + 4) = v1; }
	if (wrt1) { *(float4*)(deltaX + X_offset1) = v2;  *(float4*)(deltaX + X_offset1 + 4) = v3; }
	if (wrt2) { *(float4*)(deltaX + X_offset2) = v4;  *(float4*)(deltaX + X_offset2 + 4) = v5; }
	if (wrt3) { *(float4*)(deltaX + X_offset3) = v6;  *(float4*)(deltaX + X_offset3 + 4) = v7; }
	if (wrt4) { *(float4*)(deltaX + X_offset4) = v8;  *(float4*)(deltaX + X_offset4 + 4) = v9; }
	if (wrt5) { *(float4*)(deltaX + X_offset5) = v10; *(float4*)(deltaX + X_offset5 + 4) = v11; }
	if (wrt6) { *(float4*)(deltaX + X_offset6) = v12; *(float4*)(deltaX + X_offset6 + 4) = v13; }
	if (wrt7) { *(float4*)(deltaX + X_offset7) = v14; *(float4*)(deltaX + X_offset7 + 4) = v15; }
}

#endif



#define X_8_8_TEXTURE_S2_U
#ifndef X_8_8_TEXTURE_S2_U
#define X_8_8_TEXTURE_S2_U

//Size = 1, Time = 1.626 msec, Performace = 1320.72 GFlop/s
//Size = 1, Time = 1.9 msec, Performace = 1130.25 GFlop / s
template<int LB, int STEP>
__global__ void X_8_8_texture_s2_U(
	cudaTextureObject_t deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int N, int IC, int OC,
	int ph, int pw,
	int CFH, int CFW,
	int ic_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4  Ws[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 dYs[2][1 << LB][(1 << LB) + 1];

	//preapre for GN = IC, 
	const int i0 = (((blockIdx.y << LB) + ty) << 3) + ic_index;
	const int sw_IC = (IC << 1);
	get_y_x_ic(i0, y, x, ic0);
	const int tic0 = ((tx >= STEP) << 2) + ic0;

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

	//load 4 elements from W[OC, FH, FW, IC]
	int fh = y + ((CFH - fhr) << 1), fw = x + ((CFW - fwr) << 1);
	int W_offset = (fh*FW + fw)*IC + tic0;
	bool lw = (fh >= 0) && (fw >= 0) && (fh < FH) && (fw < FW);
	int W_oc = (tx - ((tx >= STEP) << LB >> 1)) * GK;
	Ws[buf][tx][ty] = lw ? *(float4*)(W + W_oc + W_offset) : make_float4(0, 0, 0, 0);

	//load 4 elements from deltaY[N, OH, OW, OC]
	bool ldy0 = (tohs0 >= -fhr) && (tohs0 < OH - fhr) && (tows0 >= -fwr) && (tows0 < OW - fwr);
	bool ldy1 = (tohs1 >= -fhr) && (tohs1 < OH - fhr) && (tows1 >= -fwr) && (tows1 < OW - fwr);
	bool ldy2 = (tohs2 >= -fhr) && (tohs2 < OH - fhr) && (tows2 >= -fwr) && (tows2 < OW - fwr);
	bool ldy3 = (tohs3 >= -fhr) && (tohs3 < OH - fhr) && (tows3 >= -fwr) && (tows3 < OW - fwr);
	int Y_offset = (fhr*OW + fwr)*OC;
	int Y_oc = ty - ((ty >= STEP) << LB >> 1) + Y_offset;
	dYs[buf][ty][tx].x = ldy0 * tex1Dfetch<float>(deltaY, Y_offset0 + Y_oc);
	dYs[buf][ty][tx].y = ldy1 * tex1Dfetch<float>(deltaY, Y_offset1 + Y_oc);
	dYs[buf][ty][tx].z = ldy2 * tex1Dfetch<float>(deltaY, Y_offset2 + Y_oc);
	dYs[buf][ty][tx].w = ldy3 * tex1Dfetch<float>(deltaY, Y_offset3 + Y_oc);
	__syncthreads();

	//compute area-----------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);
	const int GK = FH * FW * IC;
	const int CFW_OC = CFH * CFW;
	const int GKs = CFH * CFW_OC;
	for (int ok = 0, OK = GKs << 1 >> LB; ok < OK; ok++) 
	{
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

		int W_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		int W_fhr = W_k / CFW_OC; W_k %= CFW_OC;
		int W_fwr = W_k / OC, W_oc = W_k % OC;
		
		//load 4 elements from W[OC, FH, FW, IC]
		int fh = y + ((CFH - fhr) << 1), fw = x + ((CFW - fwr) << 1);
		int W_offset = (fh*FW + fw)*IC + tic0;
		bool lw = (fh >= 0) && (fw >= 0) && (fh < FH) && (fw < FW);
		int W_oc = (tx - ((tx >= STEP) << LB >> 1)) * GK;
		Ws[buf][tx][ty] = lw ? *(float4*)(W + W_oc + W_offset) : make_float4(0, 0, 0, 0);


		int Y_k = ((ok - (ty >= STEP)) << LB >> 1) + ty;
		int Y_fhr = W_k / 
		//load 4 elements from deltaY[N, OH, OW, OC]
		bool ldy0 = (tohs0 >= -fhr) && (tohs0 < OH - fhr) && (tows0 >= -fwr) && (tows0 < OW - fwr);
		bool ldy1 = (tohs1 >= -fhr) && (tohs1 < OH - fhr) && (tows1 >= -fwr) && (tows1 < OW - fwr);
		bool ldy2 = (tohs2 >= -fhr) && (tohs2 < OH - fhr) && (tows2 >= -fwr) && (tows2 < OW - fwr);
		bool ldy3 = (tohs3 >= -fhr) && (tohs3 < OH - fhr) && (tows3 >= -fwr) && (tows3 < OW - fwr);
		int Y_offset = (fhr*OW + fwr)*OC;
		int Y_oc = ty - ((ty >= STEP) << LB >> 1) + Y_offset;
		dYs[buf][ty][tx].x = ldy0 * tex1Dfetch<float>(deltaY, Y_offset0 + Y_oc);
		dYs[buf][ty][tx].y = ldy1 * tex1Dfetch<float>(deltaY, Y_offset1 + Y_oc);
		dYs[buf][ty][tx].z = ldy2 * tex1Dfetch<float>(deltaY, Y_offset2 + Y_oc);
		dYs[buf][ty][tx].w = ldy3 * tex1Dfetch<float>(deltaY, Y_offset3 + Y_oc);
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


	ph = y - ph; pw = x - pw;
	const int tih0 = (ohs0 << 1) + ph; const int tiw0 = (ows0 << 1) + pw;
	const int tih1 = (ohs1 << 1) + ph; const int tiw1 = (ows1 << 1) + pw;
	const int tih2 = (ohs2 << 1) + ph; const int tiw2 = (ows2 << 1) + pw;
	const int tih3 = (ohs3 << 1) + ph; const int tiw3 = (ows3 << 1) + pw;
	const int tih4 = (ohs4 << 1) + ph; const int tiw4 = (ows4 << 1) + pw;
	const int tih5 = (ohs5 << 1) + ph; const int tiw5 = (ows5 << 1) + pw;
	const int tih6 = (ohs6 << 1) + ph; const int tiw6 = (ows6 << 1) + pw;
	const int tih7 = (ohs7 << 1) + ph; const int tiw7 = (ows7 << 1) + pw;
	int X_offset0 = ((n0*IH + tih0)*IW + tiw0)*IC + ic0;
	int X_offset1 = ((n1*IH + tih1)*IW + tiw1)*IC + ic0;
	int X_offset2 = ((n2*IH + tih2)*IW + tiw2)*IC + ic0;
	int X_offset3 = ((n3*IH + tih3)*IW + tiw3)*IC + ic0;
	int X_offset4 = ((n4*IH + tih4)*IW + tiw4)*IC + ic0;
	int X_offset5 = ((n5*IH + tih5)*IW + tiw5)*IC + ic0;
	int X_offset6 = ((n6*IH + tih6)*IW + tiw6)*IC + ic0;
	int X_offset7 = ((n7*IH + tih7)*IW + tiw7)*IC + ic0;
	bool wrt0 = (tih0 >= 0) && (tih0 < IH) && (tiw0 >= 0) && (tiw0 < IW) && (n0 < N);
	bool wrt1 = (tih1 >= 0) && (tih1 < IH) && (tiw1 >= 0) && (tiw1 < IW) && (n1 < N);
	bool wrt2 = (tih2 >= 0) && (tih2 < IH) && (tiw2 >= 0) && (tiw2 < IW) && (n2 < N);
	bool wrt3 = (tih3 >= 0) && (tih3 < IH) && (tiw3 >= 0) && (tiw3 < IW) && (n3 < N);
	bool wrt4 = (tih4 >= 0) && (tih4 < IH) && (tiw4 >= 0) && (tiw4 < IW) && (n4 < N);
	bool wrt5 = (tih5 >= 0) && (tih5 < IH) && (tiw5 >= 0) && (tiw5 < IW) && (n5 < N);
	bool wrt6 = (tih6 >= 0) && (tih6 < IH) && (tiw6 >= 0) && (tiw6 < IW) && (n6 < N);
	bool wrt7 = (tih7 >= 0) && (tih7 < IH) && (tiw7 >= 0) && (tiw7 < IW) && (n7 < N);

	if (wrt0) { *(float4*)(deltaX + X_offset0) = v0;  *(float4*)(deltaX + X_offset0 + 4) = v1; }
	if (wrt1) { *(float4*)(deltaX + X_offset1) = v2;  *(float4*)(deltaX + X_offset1 + 4) = v3; }
	if (wrt2) { *(float4*)(deltaX + X_offset2) = v4;  *(float4*)(deltaX + X_offset2 + 4) = v5; }
	if (wrt3) { *(float4*)(deltaX + X_offset3) = v6;  *(float4*)(deltaX + X_offset3 + 4) = v7; }
	if (wrt4) { *(float4*)(deltaX + X_offset4) = v8;  *(float4*)(deltaX + X_offset4 + 4) = v9; }
	if (wrt5) { *(float4*)(deltaX + X_offset5) = v10; *(float4*)(deltaX + X_offset5 + 4) = v11; }
	if (wrt6) { *(float4*)(deltaX + X_offset6) = v12; *(float4*)(deltaX + X_offset6 + 4) = v13; }
	if (wrt7) { *(float4*)(deltaX + X_offset7) = v14; *(float4*)(deltaX + X_offset7 + 4) = v15; }
}

#endif
