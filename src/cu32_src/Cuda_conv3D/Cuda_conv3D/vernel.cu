//#ifndef V0
//#define V0
//
//__shared__ float2 Ws[2][2 << LB][(1 << LB) + 2];
//__shared__ float2 Xs[2][2 << LB][(1 << LB) + 2];
//
////STEP = 1 << LB >> 2
//
////prepare for GN = OC
//const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
//int ty_up = ((txy >= STEP * 1) + (ty >= STEP * 2) + (ty >= STEP * 3)) << 1;
//CW += oc0 + ty_up;
//
////prepare for GM = N * OH * OW
//int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
//int tx_up = ((tx >= STEP * 1) + (tx >= STEP * 2) + (tx >= STEP * 3)) << 1;
//int tj0 = j0 + tx_up, tj0 + 1;
//get_n_oh_ow(tj0, tn0, toh0, tow0);
//get_n_oh_ow(tj1, tn1, toh1, tow1);
//toh0 = toh0 * sh - ph, tow0 = tow0 * sw - pw;
//toh1 = toh1 * sh - ph, tow1 = tow1 * sw - pw;
//const int X0 = ((tn0*IH + toh0)*IW + tow0)*IC;
//const int X1 = ((tn1*IH + toh1)*IW + tow1)*IC;
//
////load 2 elements from W[OC, FH, FW, IC]
//int W_k = (ty - ((ty_up << LB >> 2) << 1)) + +(ok*STEP2);
//int woffset0 = W_k * OC, woffset1 = woffset0 + OC;
//Ws[buf][(ty << 1) + 0][tx] = *(float2*)(CW + woffset0);
//Ws[buf][(ty << 1) + 1][tx] = *(float2*)(CW + woffset1);
//
////load 4 elements from X[N, IH, IW, IC]
//int X_k = ((tx - (tx_up << LB >> 2)) << 1) + (ok*STEP2);
//int X_fh, X_fw; get_X_fh_fw(X_k, X_fh, X_fw);
//int xoffset = X_fh * IW_IC + X_k;
//bool lx0 = LOAD_X(toh0, tow0, X_fh, X_fw);
//bool lx1 = LOAD_X(toh1, tow1, X_fh, X_fw);
//float2 x0 = (lx0 ? *(float2*)(X + X0 + xoffset) : F32_2_0);
//float2 x1 = (lx1 ? *(float2*)(X + X1 + xoffset) : F32_2_0);
//Xs[buf][(tx << 1)][ty] = make_float2(x0.x, x1.x);
//Xs[buf][(tx << 1) + 1][ty] = make_float2(x0.y, x1.y);
//__syncthreads();
//
//for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)//OK = GK / STEP
//{
//#pragma unroll
//	for (int ik = 0; ik < STEP2; ik++)
//	{
//		float2 b0_p1 = Xs[buf][ik + STEP * 1][ty];
//		float2 b0_p2 = Xs[buf][ik + STEP * 2][ty];
//		float4 b0 = { b0_p1, b0_p2 };
//
//		float2 b1_p2 = Xs[buf][ik + STEP * 3][ty];
//		float2 b1_p4 = Xs[buf][ik + STEP * 2][ty];
//		float4 b1 = { b1_p1, b1_p2 };
//		
//		float4 a0_p1 = Ws[buf][ik + STEP * 0][tx];
//		float4 a0_p2 = Ws[buf][ik + STEP * 1][tx];
//		float4 a1 = { b1_p1, b1_p2 };
//
//		float4 a1_p1 = Ws[buf][ik + STEP * 2][tx];
//		float4 a1_p2 = Ws[buf][ik + STEP * 3][tx];
//		float4 b0 = { b1_p1, b1_p2 };
//	}
//}
//
//
//#endif
//
//
//#ifndef V1
//#define V1
//
//__shared__ float2 Ws[2][2 << LB][(1 << LB) + 2];
//__shared__ float2 Xs[2][2 << LB][(1 << LB) + 2];
//
////STEP = 1 << LB >> 2
////STEP2 = 2 << LB
//
////prepare for GN = OC
//const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
//int ty_up = ((txy >= STEP * 1) + (ty >= STEP * 2) + (ty >= STEP * 3)) << 1;
//CW += oc0 + ty_up;
//
////prepare for GM = N * OH * OW
//int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
//int tx_up = ((tx >= STEP * 1) + (tx >= STEP * 2) + (tx >= STEP * 3)) << 1;
//int tj0 = j0 + tx_up, tj0 + 1;
//get_n_oh_ow(tj0, tn0, toh0, tow0);
//get_n_oh_ow(tj1, tn1, toh1, tow1);
//toh0 = toh0 * sh - ph, tow0 = tow0 * sw - pw;
//toh1 = toh1 * sh - ph, tow1 = tow1 * sw - pw;
//const int X0 = ((tn0*IH + toh0)*IW + tow0)*IC;
//const int X1 = ((tn1*IH + toh1)*IW + tow1)*IC;
//
////load 2 elements from W[OC, FH, FW, IC]
//int W_k = (ty - ((ty_up << LB >> 2) << 1)) + (ok*STEP2);
//int woffset0 = W_k * OC;
//int woffset1 = woffset0 + OC;
//int woffset2 = woffset1 + OC;
//int woffset3 = woffset2 + OC;
//Ws[buf][(ty << 1) + 0][tx] = *(float2*)(CW + woffset0);
//Ws[buf][(ty << 1) + 1][tx] = *(float2*)(CW + woffset1);
//Ws[buf][(ty << 1) + 2][tx] = *(float2*)(CW + woffset2);
//Ws[buf][(ty << 1) + 3][tx] = *(float2*)(CW + woffset3);
//
////load 4 elements from X[N, IH, IW, IC]
//int X_k = ((tx - (tx_up << LB >> 2)) << 1) + (ok*STEP2);
//int X_fh, X_fw; get_X_fh_fw(X_k, X_fh, X_fw);
//int xoffset = X_fh * IW_IC + X_k;
//bool lx0 = LOAD_X(toh0, tow0, X_fh, X_fw);
//bool lx1 = LOAD_X(toh1, tow1, X_fh, X_fw);
//float4 x0 = (lx0 ? *(float4*)(X + X0 + xoffset) : F32_4_0);
//float4 x1 = (lx1 ? *(float4*)(X + X1 + xoffset) : F32_4_0);
//Xs[buf][(tx << 2) + 0][ty] = make_float2(x0.x, x1.x);
//Xs[buf][(tx << 2) + 1][ty] = make_float2(x0.y, x1.y);
//Xs[buf][(tx << 2) + 2][ty] = make_float2(x0.z, x1.z);
//Xs[buf][(tx << 2) + 3][ty] = make_float2(x0.w, x1.w);
//__syncthreads();
//
//
//#endif
//
//
//#ifndef V2
//#define V2
//
//__shared__ float2 Ws[2][1 << LB][(2 << LB) + 2];
//__shared__ float2 Xs[2][1 << LB][(2 << LB) + 2];
//
////STEP = 1 << LB >> 2
//
////prepare for GN = OC
//const int oc0 = (((blockIdx.x << LB) + tx) << 3) + oc_index;
//int ty_up = ((ty >= STEP) << 2) + ((ty & 1) << 1);
//CW += oc0 + ty_up;
//
////prepare for GM = N * OH * OW
//int j0 = (((blockIdx.y << LB) + ty) << 3) + j_index;
//int tx_up = ((tx >= STEP) << 2) + ((tx & 1) << 1);
//int tj0 = j0 + tx_up, tj0 + 1;
//get_n_oh_ow(tj0, tn0, toh0, tow0);
//get_n_oh_ow(tj1, tn1, toh1, tow1);
//toh0 = toh0 * sh - ph, tow0 = tow0 * sw - pw;
//toh1 = toh1 * sh - ph, tow1 = tow1 * sw - pw;
//const int X0 = ((tn0*IH + toh0)*IW + tow0)*IC;
//const int X1 = ((tn1*IH + toh1)*IW + tow1)*IC;
//
//
//const int Ws_y = (ty >> 1), Ws_x = (tx << 1) + (ty & 1);
//
//
////load 2 elements from W[OC, FH, FW, IC]
//int W_k = (ty - ((ty_up << LB >> 2) << 1)) + +(ok*STEP2);
//int woffset0 = W_k * OC, woffset1 = woffset0 + OC;
//Ws[buf][(ty << 1) + 0][tx] = *(float2*)(CW + woffset0);
//Ws[buf][(ty << 1) + 1][tx] = *(float2*)(CW + woffset1);
//
////load 4 elements from X[N, IH, IW, IC]
//int X_k = ((tx - (tx_up << LB >> 2)) << 1) + (ok*STEP2);
//int X_fh, X_fw; get_X_fh_fw(X_k, X_fh, X_fw);
//int xoffset = X_fh * IW_IC + X_k;
//bool lx0 = LOAD_X(toh0, tow0, X_fh, X_fw);
//bool lx1 = LOAD_X(toh1, tow1, X_fh, X_fw);
//float2 x0 = (lx0 ? *(float2*)(X + X0 + xoffset) : F32_2_0);
//float2 x1 = (lx1 ? *(float2*)(X + X1 + xoffset) : F32_2_0);
//Xs[buf][(tx << 1)][ty] = make_float2(x0.x, x1.x);
//Xs[buf][(tx << 1) + 1][ty] = make_float2(x0.y, x1.y);
//__syncthreads();
//
//for (int ok = 1, OK = (GK >> LB); ok < OK; ok++)//OK = GK / STEP
//{
//#pragma unroll
//	for (int ik = 0; ik < STEP2; ik++)
//	{
//		float2 b0_p1 = Xs[buf][ik + STEP * 1][ty];
//		float2 b0_p2 = Xs[buf][ik + STEP * 2][ty];
//		float4 b0 = { b0_p1, b0_p2 };
//
//		float2 b1_p2 = Xs[buf][ik + STEP * 3][ty];
//		float2 b1_p4 = Xs[buf][ik + STEP * 2][ty];
//		float4 b1 = { b1_p1, b1_p2 };
//
//		float4 a0_p1 = Ws[buf][ik + STEP * 0][tx];
//		float4 a0_p2 = Ws[buf][ik + STEP * 1][tx];
//		float4 a1 = { b1_p1, b1_p2 };
//
//		float4 a1_p1 = Ws[buf][ik + STEP * 2][tx];
//		float4 a1_p2 = Ws[buf][ik + STEP * 3][tx];
//		float4 b0 = { b1_p1, b1_p2 };
//	}
//}
//
//
//#endif
