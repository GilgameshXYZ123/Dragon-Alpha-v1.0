//LB = log2(BLOCK_SIZE)
#define k88s1_pure(stream, LB, ic_index, j_index, deltaY, OH, OW, W, FH, FW, deltaX, IH, IW, IC, OC, ph, pw, GN, GM) \
	kernel_8_8_s1_pure<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>3, GN>>LB>>3), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(deltaY,OH,OW, W,FH,FW, deltaX,IH,IW, IC,OC, ph,pw, ic_index,j_index)


//Size = 1, Time = 2.176 msec, Performace = 986.895 GFlop/s
//((FH - 1 - fh)*FW + (FW - 1 - fw))*IC 
//(FH*FW - FW - fh*FW + FW - 1 - fw)*IC 
//(FH*FW - fh*FW - 1 - fw)*IC`
//(FH * FW - 1)*IC - fh*FW*IC - fw*IC
template<int LB, int STEP>
__global__ void kernel_8_8_s1_pure(
	const float* __restrict__ deltaY, int OH, int OW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC, int OC,
	int ph, int pw,
	int ic_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4  Ws[2][1 << LB >> 1][(2 << LB) + 1];
	__shared__ float4 dYs[2][1 << LB >> 1][(2 << LB) + 1];

	//prepare for GN = IC
	const int ic0 = (((blockIdx.y << LB) + ty) << 3) + ic_index;
	const int tic0 = ((tx & 1) << 2) + ic0;//tic0 = (ic4 - ic0)*(tx&1) + ic0
	const int FH_FW_IC = FH * FW * IC; W += FH_FW_IC - IC + tic0;

	//prepare for GM = N * IH * IW
	int j0 = (((blockIdx.x << LB) + tx) << 3) + j_index;
	int j1 = j0 + 1, j2 = j0 + 2, j3 = j0 + 3;
	int j4 = j0 + 4, j5 = j0 + 5, j6 = j0 + 6, j7 = j0 + 7;
	const int IH_IW = IH * IW;
	get_n_ih_iw(j0, n0, ih0, iw0);
	get_n_ih_iw(j1, n1, ih1, iw1);
	get_n_ih_iw(j2, n2, ih2, iw2);
	get_n_ih_iw(j3, n3, ih3, iw3);
	get_n_ih_iw(j4, n4, ih4, iw4);
	get_n_ih_iw(j5, n5, ih5, iw5);
	get_n_ih_iw(j6, n6, ih6, iw6);
	get_n_ih_iw(j7, n7, ih7, iw7);
	bool flagY = (ty & 1);
	int tn0 = (n4 - n0)*flagY + n0;
	int tn1 = (n5 - n1)*flagY + n1;
	int tn2 = (n6 - n2)*flagY + n2;
	int tn3 = (n7 - n3)*flagY + n3;
	const int oph = FH - ph - 1, opw = FW - pw - 1;
	int tih0 = ((ih4 - ih0)*flagY + ih0) - oph;
	int tih1 = ((ih5 - ih1)*flagY + ih1) - oph;
	int tih2 = ((ih6 - ih2)*flagY + ih2) - oph;
	int tih3 = ((ih7 - ih3)*flagY + ih3) - oph;
	int tiw0 = ((iw4 - iw0)*flagY + iw0) - opw;
	int tiw1 = ((iw5 - iw1)*flagY + iw1) - opw;
	int tiw2 = ((iw6 - iw2)*flagY + iw2) - opw;
	int tiw3 = ((iw7 - iw3)*flagY + iw3) - opw;
	int dY_offset0 = ((tn0*OH + tih0)*OW + tiw0)*OC;
	int dY_offset1 = ((tn1*OH + tih1)*OW + tiw1)*OC;
	int dY_offset2 = ((tn2*OH + tih2)*OW + tiw2)*OC;
	int dY_offset3 = ((tn3*OH + tih3)*OW + tiw3)*OC;

	//compute area----------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4 v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4 v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4 v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	float4 v8 = make_float4(0, 0, 0, 0), v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	const int OOC = OC << 1 >> LB, strideY = (OW - FW)*OC;
	for (int fh = 0; fh < FH; fh++, deltaY += strideY) {
		for (int fw = 0; fw < FW; fw++, deltaY += OC, W -= IC)
		{

			int W_oc = tx >> 1;
			int Ws_x = (tx >> 1), Ws_y = (ty << 1) + (tx & 1);
			Ws[buf][Ws_x][Ws_y] = *(float4*)(&W[W_oc*FH_FW_IC]);

			bool ldy0 = (tih0 >= -fh) && (tih0 < OH - fh) && (tiw0 >= -fw) && (tiw0 < OW - fw);
			bool ldy1 = (tih1 >= -fh) && (tih1 < OH - fh) && (tiw1 >= -fw) && (tiw1 < OW - fw);
			bool ldy2 = (tih2 >= -fh) && (tih2 < OH - fh) && (tiw2 >= -fw) && (tiw2 < OW - fw);
			bool ldy3 = (tih3 >= -fh) && (tih3 < OH - fh) && (tiw3 >= -fw) && (tiw3 < OW - fw);
			int dY_oc = ty >> 1;
			int dYs_y = (ty >> 1), dYs_x = (tx << 1) + (ty & 1);
			dYs[buf][dYs_y][dYs_x].x = ldy0 ? deltaY[dY_offset0 + dY_oc] : 0;
			dYs[buf][dYs_y][dYs_x].y = ldy1 ? deltaY[dY_offset1 + dY_oc] : 0;
			dYs[buf][dYs_y][dYs_x].z = ldy2 ? deltaY[dY_offset2 + dY_oc] : 0;
			dYs[buf][dYs_y][dYs_x].w = ldy3 ? deltaY[dY_offset3 + dY_oc] : 0;
			__syncthreads();

			for (int ooc = 1; ooc < OOC; ooc++) {
#pragma unroll
				for (int ik = 0; ik < STEP; ik++)
				{
					float4 dy0 = dYs[buf][ik][(tx << 1)];
					float4 dy1 = dYs[buf][ik][(tx << 1) + 1];
					float4  w0 = Ws[buf][ik][(ty << 1)];
					float4  w1 = Ws[buf][ik][(ty << 1) + 1];

					//transposed compute core: (W * dY)^T
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

				W_oc = ((ooc << LB) + tx) >> 1;
				Ws[buf][Ws_x][Ws_y] = *(float4*)(&W[W_oc*FH_FW_IC]);

				dY_oc = ((ooc << LB) + ty) >> 1;
				dYs[buf][dYs_y][dYs_x].x = ldy0 ? deltaY[dY_offset0 + dY_oc] : 0;
				dYs[buf][dYs_y][dYs_x].y = ldy1 ? deltaY[dY_offset1 + dY_oc] : 0;
				dYs[buf][dYs_y][dYs_x].z = ldy2 ? deltaY[dY_offset2 + dY_oc] : 0;
				dYs[buf][dYs_y][dYs_x].w = ldy3 ? deltaY[dY_offset3 + dY_oc] : 0;
				__syncthreads();
			}
#pragma unroll
			for (int ik = 0; ik < STEP; ik++)
			{
				float4 dy0 = dYs[buf][ik][(tx << 1)];
				float4 dy1 = dYs[buf][ik][(tx << 1) + 1];
				float4  w0 = Ws[buf][ik][(ty << 1)];
				float4  w1 = Ws[buf][ik][(ty << 1) + 1];

				//transposed compute core: (W * dY)^T
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

	int ic4 = ic0 + 4; j0 *= IC;//j0 = ((n * OH + oh) * OW + ow) * IC + ic
	j1 = j0 + IC; j2 = j1 + IC; j3 = j2 + IC;
	j4 = j3 + IC; j5 = j4 + IC; j6 = j5 + IC; j7 = j6 + IC;

	*(float4*)(&deltaX[j0 + ic0]) = v0; *(float4*)(&deltaX[j0 + ic4]) = v1;
	*(float4*)(&deltaX[j1 + ic0]) = v2; *(float4*)(&deltaX[j1 + ic4]) = v3;
	*(float4*)(&deltaX[j2 + ic0]) = v4; *(float4*)(&deltaX[j2 + ic4]) = v5;
	*(float4*)(&deltaX[j3 + ic0]) = v6; *(float4*)(&deltaX[j3 + ic4]) = v7;
	*(float4*)(&deltaX[j4 + ic0]) = v8; *(float4*)(&deltaX[j4 + ic4]) = v9;
	*(float4*)(&deltaX[j5 + ic0]) = v10; *(float4*)(&deltaX[j5 + ic4]) = v11;
	*(float4*)(&deltaX[j6 + ic0]) = v12; *(float4*)(&deltaX[j6 + ic4]) = v13;
	*(float4*)(&deltaX[j7 + ic0]) = v14; *(float4*)(&deltaX[j7 + ic4]) = v15;
}