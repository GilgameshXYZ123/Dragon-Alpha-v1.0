#pragma once

#ifndef DECONV_3D_DELTAW_BGEMM
#define DECONV_3D_DELTAW_BGEMM

#define __dconv3D_deltaW_BGemm(streams, index, length, GZ, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw)\
	deconv3d_deltaW_BGemm(streams, index, length, GZ, X,IH,IW, deltaY,OH,OW, deltaW,deltaW_buf,FH,FW, N,IC,OC, sh,sw,ph,pw,\
		GET_GN(OC), GET_GM(IC, FH, FW), GET_GK(N, OH, OW), 0, 0)

#define dconv3d_dW_BGemm_Branch(SIZE_Y, SIZE_X) {\
	int GNr = GN & (SIZE_Y), GMr = GM & (SIZE_X);\
	if(GNr && GMr){\
		int next_oc_index = (GN - GNr) + oc_index;\
		int next_j_index = (GM - GMr) + j_index;\
		deconv3d_deltaW_BGemm(streams, index, length, GZ, X,IH,IW, deltaY,OH,OW, deltaW,deltaW_buf,FH,FW, N,IC,OC, sh,sw,ph,pw,\
			GNr, GM, GK, next_oc_index, j_index);\
		deconv3d_deltaW_BGemm(streams, index, length, GZ, X,IH,IW, deltaY,OH,OW, deltaW,deltaW_buf,FH,FW, N,IC,OC, sh,sw,ph,pw,\
            GN, GMr, GK, oc_index, next_j_index);\
		deconv3d_deltaW_BGemm(streams, index, length, GZ, X,IH,IW, deltaY,OH,OW, deltaW,deltaW_buf,FH,FW, N,IC,OC, sh,sw,ph,pw,\
            GNr, GMr, GK, next_oc_index, next_j_index);}\
	else if(GNr){\
		int next_oc_index = (GN - GNr) + oc_index;\
		deconv3d_deltaW_BGemm(streams, index, length, GZ, X,IH,IW, deltaY,OH,OW, deltaW,deltaW_buf,FH,FW, N,IC,OC, sh,sw,ph,pw,\
			 GNr, GM, GK, next_oc_index, j_index);}\
	else if(GMr){\
		int next_j_index = (GM - GMr) + j_index;\
		deconv3d_deltaW_BGemm(streams, index, length, GZ, X,IH,IW, deltaY,OH,OW, deltaW,deltaW_buf,FH,FW, N,IC,OC, sh,sw,ph,pw,\
			 GN, GMr, GK, oc_index, next_j_index);}}

//We have: 
//	(1) GN >= 4, GN % 4==0
//	(2) GM >= 4, GM % 4==0
//	(3) GK >= 4, GK % 4==0, As N >= 16, so GK >= 16
void deconv3d_deltaW_BGemm(jlong *streams, int &index, int length, int GZ,
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw,
	int GN, int GM, int GK,
	int oc_index, int j_index)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)streams[index];
	index = (index + 1) % length;

	if ((GN > 127) && (GM > 127) && !(GK & 7)) {//GK % 8 ==0
		if (IS_POWER2(OH) && IS_POWER2(OW)) {
			if (IS_POWER2(IC) && IS_POWER2(OC))
				kBGemm88_ohw_oic2pow(stream, 4, GZ, oc_index, j_index, X, IH, IW, deltaY, LOG2(OH), LOG2(OW), deltaW, deltaW_buf, FH, FW, N, LOG2(IC), LOG2(OC), sh, sw, ph, pw, GN, GM);
			else kBGemm88_ohw2pow(stream, 4, GZ, oc_index, j_index, X, IH, IW, deltaY, LOG2(OH), LOG2(OW), deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
		}
		else kBGemm88(stream, 4, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dW_BGemm_Branch(127, 127); return;
	}

	if ((GN > 63) && (GM > 63)) {
		if (IS_POWER2(OH) && IS_POWER2(OW)) {
			if (IS_POWER2(IC) && IS_POWER2(OC))
				kBGemm88_ohw_oic2pow(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, LOG2(OH), LOG2(OW), deltaW, deltaW_buf, FH, FW, N, LOG2(IC), LOG2(OC), sh, sw, ph, pw, GN, GM);
			else kBGemm88_ohw2pow(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, LOG2(OH), LOG2(OW), deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
		}
		else kBGemm88(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dW_BGemm_Branch(63, 63); return;
	}

	if ((GN > 63) && (GM > 31)) {
		if (IS_POWER2(OH) && IS_POWER2(OW)) {
			if (IS_POWER2(IC) && IS_POWER2(OC))
				kBGemm84_ohw_oic2pow(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, LOG2(OH), LOG2(OW), deltaW, deltaW_buf, FH, FW, N, LOG2(IC), LOG2(OC), sh, sw, ph, pw, GN, GM);
			else kBGemm84_ohw2pow(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, LOG2(OH), LOG2(OW), deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
		}
		else kBGemm84(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dW_BGemm_Branch(63, 31); return;
	}

	if ((GN > 31) && (GM > 63)) {
		if (IS_POWER2(OH) && IS_POWER2(OW)) {
			if (IS_POWER2(IC) && IS_POWER2(OC))
				kBGemm48_ohw_oic2pow(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, LOG2(OH), LOG2(OW), deltaW, deltaW_buf, FH, FW, N, LOG2(IC), LOG2(OC), sh, sw, ph, pw, GN, GM);
			else kBGemm48_ohw2pow(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, LOG2(OH), LOG2(OW), deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
		}
		else kBGemm48(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dW_BGemm_Branch(31, 63); return;
	}

	if ((GN > 31) && (GM > 31)) {
		if (IS_POWER2(OH) && IS_POWER2(OW)) {
			if (IS_POWER2(IC) && IS_POWER2(OC))
				kBGemm44_ohw_oic2pow(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, LOG2(OH), LOG2(OW), deltaW, deltaW_buf, FH, FW, N, LOG2(IC), LOG2(OC), sh, sw, ph, pw, GN, GM);
			else kBGemm44_ohw2pow(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, LOG2(OH), LOG2(OW), deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
		}
		else kBGemm44(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dW_BGemm_Branch(31, 31); return;
	}

	if ((GN > 31) && (GM > 15)) {//GK % 4 ==0
		kBGemm42(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dW_BGemm_Branch(31, 15); return;
	}
	if ((GN > 15) && (GM > 31)) {//GK % 4 ==0
		kBGemm24(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dW_BGemm_Branch(15, 31); return;
	}

	if ((GN > 15) && (GM > 15)) {//GK >= 8;
		kBGemm22(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dW_BGemm_Branch(15, 15); return;
	}
	if ((GN > 31) && (GM > 7)) {//GK >= 8;
		kBGemm41(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dW_BGemm_Branch(31, 7); return;
	}
	if ((GN > 7) && (GM > 31)) {//GK >= 8;
		kBGemm14(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dW_BGemm_Branch(7, 31); return;
	}
	if ((GN > 15) && (GM > 7)) {//GK >= 8;
		kBGemm21(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dW_BGemm_Branch(15, 7); return;
	}
	if ((GN > 7) && (GM > 15)) {//GK >= 8;
		kBGemm12(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dW_BGemm_Branch(7, 15); return;
	}
	if ((GN > 7) && (GM > 7)) {//GK >= 8;
		kBGemm11(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dW_BGemm_Branch(7, 7); return;
	}

	if (GN > 15) {//GK >= 4;
		kBGemm41(stream, 2, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dW_BGemm_Branch(15, 3); return;
	}
	if (GM > 15) {//GK >= 4;
		kBGemm14(stream, 2, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dW_BGemm_Branch(3, 15); return;
	}
	if (GN > 7) {//GK >= 4;
		kBGemm21(stream, 2, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dW_BGemm_Branch(7, 3); return;
	}
	if (GM > 7) {//GK >= 4;
		kBGemm12(stream, 2, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dW_BGemm_Branch(3, 7); return;
	}
	kBGemm11(stream, 2, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
}

#endif


#ifndef DECONV3D_DW_BGEMM_KERNEL_H
#define DECONV3D_DW_BGEMM_KERNEL_H

//when batchsize >= 16
//oph = ph, opw = pw

//===================Split K to improve parallism=======================================
//oph = ph, opw = pw
//deltaY_pe is the 4D convolution kernel: deltaW_e = conv(X, deltaY_pe)| step=1, padding=(oph, opw)
//logically:
//deltaY[N, OH, OW, OC]  -> deltaYpe[OC, OHp, OWp, N]
//     X[N, IH, IW, IC]  ->       Xe[IC, IH, IW, N]
//deltaW[OC, FH, FW, IC] ->  deltaWe[IC, FH, FW, OC]
//we have:
//	(1) FH * FW >= 2
//	(2) GN = OC:           GN%4 == 0, GN >= 4
//	(3) GM = IC * FH * FW: GM%4 == 0, GM >= 8
//	(4) GK = N  * OH * OW: GK%4 == 0, GK >= 4
//	(5) GK0 = N * OHp * OWp(compress GK -> GKp)
//
//GZ = N / 8
//As: N % 4 == 0, so Nslice at least % 4 == 0
//Nslice == 8 for [0, GZ - 2],
//NSlice >= 8 && %4 == 0 for[GZ]
//
//int bz = blockIdx.z;
//int N_slice = (N / gridDim.z) >> 3 << 3;
//int N_start = bz * N_slice;
//when: div = 8
//GZ = N >> 3 = (8x + 4) / 8 = x
//N_slice = ((8x + 4) / GZ) >> 3 << 3
//N_slice = ((8x + 4) / x) >> 3 << 3
//N_slice = (8 + 4/x) >> 3 << 3
//as (4 / x) from 1 to 4
//so (4/x) >> 3 << 3  == 0
//when: div = 16
//GZ = N >> 3 = (16x + 4y) / 16 = x
//N_slice = ((16x + 4y) / GZ) >> 3 << 3
//N_slice = ((16x + 4y) / x) >> 3 << 3
//N_slice = (16 + 4y/x) >> 3 << 3
//N_slice = 16 + [(4y_x) >> 3 << 3]
//as (4y / x) from 1 to 12
//so (4 / x) >> 3 from 0 to 1
//so (4 / x) >> 3 << 3 = [0, 8]
//N_slice = 16 + 8 * ((4/x) >> 3)
//===================Split K to improve parallism=======================================

#ifndef DECONV3D_DW_BGEMM_KERNEL_CALL
#define DECONV3D_DW_BGEMM_KERNEL_CALL

//LB = log2(BLOCK_SIZE)
//GZ = gridDim.z

#define kBGemm88(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM)\
	kernel_BGemm_8_8<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>3, GN>>LB>>3, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, j_index)

#define kBGemm84(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM)\
	kernel_BGemm_8_4<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>2, GN>>LB>>3, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, j_index)

#define kBGemm48(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM)\
	kernel_BGemm_4_8<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>3, GN>>LB>>2, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, j_index)

#define kBGemm44(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM)\
	kernel_BGemm_4_4<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>2, GN>>LB>>2, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, j_index)

#define kBGemm42(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM)\
	kernel_BGemm_4_2<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>1, GN>>LB>>2, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, j_index)

#define kBGemm24(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM)\
	kernel_BGemm_2_4<LB, (1<<LB>>1)>\
		<<< dim3(GM>>LB>>2, GN>>LB>>1, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, j_index)

#define kBGemm41(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM)\
	kernel_BGemm_4_1<LB, (1<<LB)>\
		<<< dim3(GM>>LB, GN>>LB>>2, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, j_index)

#define kBGemm14(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM)\
	kernel_BGemm_1_4<LB, (1<<LB)>\
		<<< dim3(GM>>LB>>2, GN>>LB, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, j_index)

#define kBGemm22(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM)\
	kernel_BGemm_2_2<LB, (1<<LB)>\
		<<< dim3(GM>>LB>>1, GN>>LB>>1, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, j_index)

#define kBGemm21(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM)\
	kernel_BGemm_2_1<LB, (1<<LB)>\
		<<< dim3(GM>>LB, GN>>LB>>1, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, j_index)

#define kBGemm12(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM)\
	kernel_BGemm_1_2<LB, (1<<LB)>\
		<<< dim3(GM>>LB>>1, GN>>LB, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, j_index)

#define kBGemm11(stream, LB, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM)\
	kernel_BGemm_1_1<LB, (1<<LB)>\
		<<< dim3(GM>>LB, GN>>LB, GZ), dim3(1<<LB, 1<<LB), 0, stream >>>\
			(X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW,\
			 N, IC, OC, sh, sw, ph, pw, oc_index, j_index)

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*8), GK >= (BLOCK_SIZE/2), IC % 4 == 0
//LB = 4: GK % 8 == 0
#ifndef DECONV3D_DW_BGEMM_KERNEL_8_8
#define DECONV3D_DW_BGEMM_KERNEL_8_8

//for GZ = N >> 3
//LB = 4: Size = 1, Time = 1.624 msec, Performace = 1322.34 GFlop/s
//LB = 3: Size = 1, Time = 1.986 msec, Performace = 1081.31 GFlop/s
template<int LB, int STEP>
__global__ void kernel_BGemm_8_8(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int N, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];

	//======================================================================
	//prepare for GK_slice = (N_end - N_start) * OH_OW, N_end = nextZ.Nstart
	int bz = blockIdx.z;
	int N_slice = (N / gridDim.z) >> 3 << 3;
	int N_start = bz * N_slice;
	int N_end = IF_int((bz != (gridDim.z - 1)), N_start + N_slice, N);//min(N_start + Nslice, N)
	const int OH_OW = OH * OW, GK_slice = (N_end - N_start) * OH_OW;
	X += N_start * IH*IW*IC; //X[N_start,...]
	deltaY += N_start * OH_OW*OC; //deltaY[N_start,...]

	//dst[bz] = deltaW_buf[bz - 1, sizeW], bz >= 1, dst[0] = deltaW
	const int FW_IC = FW * IC, FH_FW_IC = FH * FW_IC;
	float* buf_addr = deltaW_buf + (bz - 1)*OC*FH_FW_IC;
	float *dst = (bz != 0) * (buf_addr - deltaW) + deltaW;
	//======================================================================

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 3) + oc_index;
	deltaY += oc0 + ((tx >= STEP) << 2);//deltaY[0, 0, 0, toc0]

	//prepare for GM = FH * FW * IC
	int j0 = (((blockIdx.x << LB) + tx) << 3) + j_index;
	int tj0 = j0 + ((ty >= STEP) << 2);
	get_fh_fw_ic(tj0, tfh0, tfw0, tic0);
	tfh0 = tfh0 - oph, tfw0 = tfw0 - opw;
	X += (tfh0*IW + tfw0)*IC + tic0;

	//load 4 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
	int Y_k = tx - ((tx >= STEP) << LB >> 1);//k = (n*OH + oh)*OW + ow
	Ys[buf][tx][ty] = *(float4*)(deltaY + Y_k * OC);

	//load 4 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
	int X_k = ty - ((ty >= STEP) << LB >> 1), X_n, X_oh, X_ow;
	get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
	int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
	Xs[buf][ty][tx] = (LOAD_X(tfh0, tfw0) ? *(float4*)(X + xoffset) : FLOAT_ZERO4);
	__syncthreads();

	//compute area-----------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0),  v1 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0),  v3 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0),  v5 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0),  v7 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0),  v9 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0), v11 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0), v13 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0), v15 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK_slice << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP][ty];
			float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];

			simdMM4( v0, a0.x, b0); simdMM4( v1, a0.x, b1);
			simdMM4( v2, a0.y, b0); simdMM4( v3, a0.y, b1);
			simdMM4( v4, a0.z, b0); simdMM4( v5, a0.z, b1);
			simdMM4( v6, a0.w, b0); simdMM4( v7, a0.w, b1);
			simdMM4( v8, a1.x, b0); simdMM4( v9, a1.x, b1);
			simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
			simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
			simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
		}
		buf ^= 1;

		//load 4 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		Ys[buf][tx][ty] = *(float4*)(deltaY + Y_k * OC);

		//load 4 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = ((ok - (ty >= STEP)) << LB >> 1) + ty, X_n, X_oh, X_ow;
		get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		Xs[buf][ty][tx] = (LOAD_X(tfh0, tfw0) ? *(float4*)(X + xoffset) : FLOAT_ZERO4);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP][ty];
		float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];

		simdMM4(v0,  a0.x, b0); simdMM4(v1,  a0.x, b1);
		simdMM4(v2,  a0.y, b0); simdMM4(v3,  a0.y, b1);
		simdMM4(v4,  a0.z, b0); simdMM4(v5,  a0.z, b1);
		simdMM4(v6,  a0.w, b0); simdMM4(v7,  a0.w, b1);
		simdMM4(v8,  a1.x, b0); simdMM4(v9,  a1.x, b1);
		simdMM4(v10, a1.y, b0); simdMM4(v11, a1.y, b1);
		simdMM4(v12, a1.z, b0); simdMM4(v13, a1.z, b1);
		simdMM4(v14, a1.w, b0); simdMM4(v15, a1.w, b1);
	}

	oc0 = oc0 * FH_FW_IC + j0;//j = (fh * FW + fw)*IC + ic
	int oc1 = oc0 + FH_FW_IC, oc2 = oc1 + FH_FW_IC;
	int oc3 = oc2 + FH_FW_IC, oc4 = oc3 + FH_FW_IC;
	int oc5 = oc4 + FH_FW_IC, oc6 = oc5 + FH_FW_IC, oc7 = oc6 + FH_FW_IC;

	*(float4*)(dst + oc0) = v0;  *(float4*)(dst + oc0 + 4) = v1;
	*(float4*)(dst + oc1) = v2;  *(float4*)(dst + oc1 + 4) = v3;
	*(float4*)(dst + oc2) = v4;  *(float4*)(dst + oc2 + 4) = v5;
	*(float4*)(dst + oc3) = v6;  *(float4*)(dst + oc3 + 4) = v7;
	*(float4*)(dst + oc4) = v8;  *(float4*)(dst + oc4 + 4) = v9;
	*(float4*)(dst + oc5) = v10; *(float4*)(dst + oc5 + 4) = v11;
	*(float4*)(dst + oc6) = v12; *(float4*)(dst + oc6 + 4) = v13;
	*(float4*)(dst + oc7) = v14; *(float4*)(dst + oc7 + 4) = v15;
}

#endif


//(Y: BLOCK_SIZE*8, X: BLOCK_SIZE*4), GK >= (BLOCK_SIZE/2), IC % 4 == 0
//LB = 4: GK % 8 == 0
#ifndef DECONV3D_DW_BGEMM_KERNEL_8_4
#define DECONV3D_DW_BGEMM_KERNEL_8_4

//for GZ = N >> 3
//LB = 4: Size = 1, Time = 2.016 msec, Performace = 1065.22 GFlop/s
//LB = 3: Size = 1, Time = 2.53  msec, Performace =  848.808 GFlop/s
template<int LB, int STEP>
__global__ void kernel_BGemm_8_4(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int N, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 Ys[2][1 << LB][(1 << LB) + 1];//followed k88
	__shared__ float2 Xs[2][1 << LB >> 1][(2 << LB) + 2];//followed k44

	//======================================================================
	//prepare for GK_slice = (N_end - N_start) * OH_OW, N_end = nextZ.Nstart
	int bz = blockIdx.z;
	int N_slice = (N / gridDim.z) >> 3 << 3;
	int N_start = bz * N_slice;
	int N_end = (N_start + N_slice - N) * (bz != (gridDim.z - 1)) + N;//min(N_start + Nslice, N)
	const int OH_OW = OH * OW, GK_slice = (N_end - N_start) * OH_OW;
	X += N_start * IH*IW*IC; //X[N_start,...]
	deltaY += N_start * OH_OW*OC; //deltaY[N_start,...]

	//dst[bz] = deltaW_buf[bz - 1, sizeW], bz >= 1, dst[0] = deltaW
	const int FW_IC = FW * IC, FH_FW_IC = FH * FW_IC;
	float* buf_addr = deltaW_buf + (bz - 1)*OC*FH_FW_IC;
	float *dst = (bz != 0) * (buf_addr - deltaW) + deltaW;
	//======================================================================

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 3) + oc_index;
	deltaY += oc0 + ((tx >= STEP) << 2);//deltaY[0, 0, 0, toc0]

	//prepare for GM = FH * FW * IC
	int j0 = (((blockIdx.x << LB) + tx) << 2) + j_index;
	const int tj0 = j0 + ((ty & 1) << 1);
	get_fh_fw_ic(tj0, tfh0, tfw0, tic0);
	tfh0 = tfh0 - oph, tfw0 = tfw0 - opw;
	X += (tfh0*IW + tfw0)*IC + tic0;

	//load 4 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
	int Y_k = tx - ((tx >= STEP) << LB >> 1);//k = (n*OH + oh)*OW + ow
	Ys[buf][tx][ty] = *(float4*)(deltaY + Y_k * OC);

	//load 2 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
	int X_k = (ty >> 1), X_n, X_oh, X_ow;
	get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
	int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);
	Xs[buf][Xs_y][Xs_x] = (LOAD_X(tfh0, tfw0) ? *(float2*)(X + xoffset) : FLOAT_ZERO2);
	__syncthreads();

	//compute area-----------------------------------------------------
	float4  v0 = make_float4(0, 0, 0, 0);
	float4  v2 = make_float4(0, 0, 0, 0);
	float4  v4 = make_float4(0, 0, 0, 0);
	float4  v6 = make_float4(0, 0, 0, 0);
	float4  v8 = make_float4(0, 0, 0, 0);
	float4 v10 = make_float4(0, 0, 0, 0);
	float4 v12 = make_float4(0, 0, 0, 0);
	float4 v14 = make_float4(0, 0, 0, 0);

	for (int ok = 1, OK = (GK_slice << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP][ty];
			float4 b0 = *(float4*)(&Xs[buf][ik][tx << 1]);

			simdMM4(v0, a0.x, b0);
			simdMM4(v2, a0.y, b0);
			simdMM4(v4, a0.z, b0);
			simdMM4(v6, a0.w, b0);
			simdMM4(v8, a1.x, b0);
			simdMM4(v10, a1.y, b0);
			simdMM4(v12, a1.z, b0);
			simdMM4(v14, a1.w, b0); 
		}
		buf ^= 1;

		//load 4 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int Y_k = ((ok - (tx >= STEP)) << LB >> 1) + tx;
		Ys[buf][tx][ty] = *(float4*)(deltaY + Y_k * OC);

		//load 2 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = ((ok << LB) + ty) >> 1, X_n, X_oh, X_ow;
		get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		Xs[buf][Xs_y][Xs_x] = (LOAD_X(tfh0, tfw0) ? *(float2*)(X + xoffset) : FLOAT_ZERO2);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a0 = Ys[buf][ik][ty], a1 = Ys[buf][ik + STEP][ty];
		float4 b0 = *(float4*)(&Xs[buf][ik][tx << 1]);

		simdMM4( v0, a0.x, b0);
		simdMM4( v2, a0.y, b0);
		simdMM4( v4, a0.z, b0);
		simdMM4( v6, a0.w, b0);
		simdMM4( v8, a1.x, b0);
		simdMM4(v10, a1.y, b0);
		simdMM4(v12, a1.z, b0);
		simdMM4(v14, a1.w, b0);
	}

	oc0 = oc0 * FH_FW_IC + j0;//j = (fh * FW + fw)*IC + ic
	int oc1 = oc0 + FH_FW_IC, oc2 = oc1 + FH_FW_IC;
	int oc3 = oc2 + FH_FW_IC, oc4 = oc3 + FH_FW_IC;
	int oc5 = oc4 + FH_FW_IC, oc6 = oc5 + FH_FW_IC, oc7 = oc6 + FH_FW_IC;

	*(float4*)(dst + oc0) = v0;  
	*(float4*)(dst + oc1) = v2;  
	*(float4*)(dst + oc2) = v4;  
	*(float4*)(dst + oc3) = v6;  
	*(float4*)(dst + oc4) = v8; 
	*(float4*)(dst + oc5) = v10; 
	*(float4*)(dst + oc6) = v12;
	*(float4*)(dst + oc7) = v14;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*8), GK >= (BLOCK_SIZE/2), IC % 4 == 0
//LB = 4: GK % 8 == 0
#ifndef DECONV3D_DW_BGEMM_KERNEL_4_8
#define DECONV3D_DW_BGEMM_KERNEL_4_8

//for GZ = N >> 3
//LB = 4: Size = 1, Time = 2.184 msec, Performace = 983.28  GFlop/s
//LB = 3: Size = 1, Time = 2.488 msec, Performace = 863.136 GFlop/s
template<int LB, int STEP>
__global__ void kernel_BGemm_4_8(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int N, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 Ys[2][1 << LB >> 1][(2 << LB) + 2];//followed k44
	__shared__ float4 Xs[2][1 << LB][(1 << LB) + 1];//followed k88

	//======================================================================
	//prepare for GK_slice = (N_end - N_start) * OH_OW, N_end = nextZ.Nstart
	int bz = blockIdx.z;
	int N_slice = (N / gridDim.z) >> 3 << 3;
	int N_start = bz * N_slice;
	int N_end = IF_int((bz != (gridDim.z - 1)), N_start + N_slice, N);//min(N_start + Nslice, N)
	const int OH_OW = OH * OW, GK_slice = (N_end - N_start) * OH_OW;
	X += N_start * IH*IW*IC; //X[N_start,...]
	deltaY += N_start * OH_OW*OC; //deltaY[N_start,...]

	//dst[bz] = deltaW_buf[bz - 1, sizeW], bz >= 1, dst[0] = deltaW
	const int FW_IC = FW * IC, FH_FW_IC = FH * FW_IC;
	float* buf_addr = deltaW_buf + (bz - 1)*OC*FH_FW_IC;
	float *dst = (bz != 0) * (buf_addr - deltaW) + deltaW;
	//======================================================================

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 2) + oc_index;
	deltaY += ((tx & 1) << 1) + oc0;//deltaY[0, 0, 0, toc0]

	//prepare for GM = FH * FW * IC
	int j0 = (((blockIdx.x << LB) + tx) << 3) + j_index;
	int tj0 = j0 + ((ty >= STEP) << 2);
	get_fh_fw_ic(tj0, tfh0, tfw0, tic0);
	tfh0 = tfh0 - oph, tfw0 = tfw0 - opw;
	X += (tfh0*IW + tfw0)*IC + tic0;

	//load 2 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
	int Y_k = (tx >> 1);//k = (n*OH + oh)*OW + ow
	const int dYs_x = (tx >> 1), dYs_y = (ty << 1) + (tx & 1);
	Ys[buf][dYs_x][dYs_y] = *(float2*)(deltaY + Y_k*OC);

	//load 4 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
	int X_k = ty - ((ty >= STEP) << LB >> 1), X_n, X_oh, X_ow;
	get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
	int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
	Xs[buf][ty][tx] = (LOAD_X(tfh0, tfw0) ? *(float4*)(X + xoffset) : FLOAT_ZERO4);
	__syncthreads();

	//compute area-----------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0), v1 = make_float4(0, 0, 0, 0);
	float4 v2 = make_float4(0, 0, 0, 0), v3 = make_float4(0, 0, 0, 0);
	float4 v4 = make_float4(0, 0, 0, 0), v5 = make_float4(0, 0, 0, 0);
	float4 v6 = make_float4(0, 0, 0, 0), v7 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (GK_slice << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++)
		{
			float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];
			float4 a0 = *(float4*)(&Ys[buf][ik][ty << 1]);

			simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
			simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
			simdMM4(v4, a0.z, b0); simdMM4(v5, a0.z, b1);
			simdMM4(v6, a0.w, b0); simdMM4(v7, a0.w, b1);
		}
		buf ^= 1;

		///load 2 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int Y_k = ((ok << LB) + tx) >> 1;
		Ys[buf][dYs_x][dYs_y] = *(float2*)(deltaY + Y_k * OC);

		//load 4 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = ((ok - (ty >= STEP)) << LB >> 1) + ty, X_n, X_oh, X_ow;
		get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		Xs[buf][ty][tx] = (LOAD_X(tfh0, tfw0) ? *(float4*)(X + xoffset) : FLOAT_ZERO4);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 b0 = Xs[buf][ik][tx], b1 = Xs[buf][ik + STEP][tx];
		float4 a0 = *(float4*)(&Ys[buf][ik][ty << 1]);

		simdMM4(v0, a0.x, b0); simdMM4(v1, a0.x, b1);
		simdMM4(v2, a0.y, b0); simdMM4(v3, a0.y, b1);
		simdMM4(v4, a0.z, b0); simdMM4(v5, a0.z, b1);
		simdMM4(v6, a0.w, b0); simdMM4(v7, a0.w, b1);
	}

	oc0 = oc0 * FH_FW_IC + j0;//j = (fh * FW + fw)*IC + ic
	int oc1 = oc0 + FH_FW_IC, oc2 = oc1 + FH_FW_IC, oc3 = oc2 + FH_FW_IC;

	*(float4*)(dst + oc0) = v0;  *(float4*)(dst + oc0 + 4) = v1;
	*(float4*)(dst + oc1) = v2;  *(float4*)(dst + oc1 + 4) = v3;
	*(float4*)(dst + oc2) = v4;  *(float4*)(dst + oc2 + 4) = v5;
	*(float4*)(dst + oc3) = v6;  *(float4*)(dst + oc3 + 4) = v7;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*4), GK >= (BLOCK_SIZE/2), IC % 4 == 0
//LB = 4: GK % 8 == 0
#ifndef DECONV3D_DW_BGEMM_KERNEL_4_4
#define DECONV3D_DW_BGEMM_KERNEL_4_4

//for GZ = N >> 3
//LB = 4: Size = 1, Time = 2.552 msec, Performace = 841.49 GFlop/s
//LB = 3: Size = 1, Time = 3.526 msec, Performace = 609.042 GFlop/s
template<int LB, int STEP>
__global__ void kernel_BGemm_4_4(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int N, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 Ys[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2 Xs[2][1 << LB >> 1][(2 << LB) + 2];

	//======================================================================
	//prepare for GK_slice = (N_end - N_start) * OH_OW, N_end = nextZ.Nstart
	int bz = blockIdx.z;
	int N_slice = (N / gridDim.z) >> 3 << 3;
	int N_start = bz * N_slice;
	int N_end = IF_int((bz != (gridDim.z - 1)), N_start + N_slice, N);//min(N_start + Nslice, N)
	const int OH_OW = OH * OW, GK_slice = (N_end - N_start) * OH_OW;
	X += N_start * IH*IW*IC; //X[N_start,...]
	deltaY += N_start * OH_OW*OC; //deltaY[N_start,...]

	//dst[bz] = deltaW_buf[bz - 1, sizeW], bz >= 1, dst[0] = deltaW
	const int FW_IC = FW * IC, FH_FW_IC = FH * FW_IC;
	float* buf_addr = deltaW_buf + (bz - 1)*OC*FH_FW_IC;
	float *dst = (bz != 0) * (buf_addr - deltaW) + deltaW;
	//======================================================================

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 2) + oc_index;
	deltaY += ((tx & 1) << 1) + oc0;

	//prepare for GM = FH * FW * IC
	int j0 = (((blockIdx.x << LB) + tx) << 2) + j_index;
	const int tj0 = j0 + ((ty & 1) << 1);
	get_fh_fw_ic(tj0, tfh0, tfw0, tic0);
	tfh0 = tfh0 - oph, tfw0 = tfw0 - opw;
	X += (tfh0*IW + tfw0)*IC + tic0;

	//load 2 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
	int Y_k = (tx >> 1);//k = (n*OH + oh)*OW + ow
	const int dYs_x = (tx >> 1), dYs_y = (ty << 1) + (tx & 1);
	Ys[buf][dYs_x][dYs_y] = *(float2*)(deltaY + Y_k * OC);

	//load 2 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
	int X_k = (ty >> 1), X_n, X_oh, X_ow;
	get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
	int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);
	Xs[buf][Xs_y][Xs_x] = (LOAD_X(tfh0, tfw0) ? *(float2*)(X + xoffset) : FLOAT_ZERO2);
	__syncthreads();

	//compute area-----------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0);
	float4 v1 = make_float4(0, 0, 0, 0);
	float4 v2 = make_float4(0, 0, 0, 0);
	float4 v3 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (GK_slice << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) 
		{
			float4 a = *(float4*)(&Ys[buf][ik][ty << 1]);
			float4 b = *(float4*)(&Xs[buf][ik][tx << 1]);

			simdMM4(v0, a.x, b);
			simdMM4(v1, a.y, b);
			simdMM4(v2, a.z, b);
			simdMM4(v3, a.w, b);
		}
		buf ^= 1;

		//load 2 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int Y_k = ((ok << LB) + tx) >> 1;
		Ys[buf][dYs_x][dYs_y] = *(float2*)(deltaY + Y_k * OC);

		//load 2 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = ((ok << LB) + ty) >> 1, X_n, X_oh, X_ow;
		get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		Xs[buf][Xs_y][Xs_x] = (LOAD_X(tfh0, tfw0) ? *(float2*)(X + xoffset) : FLOAT_ZERO2);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++)
	{
		float4 a = *(float4*)(&Ys[buf][ik][ty << 1]);
		float4 b = *(float4*)(&Xs[buf][ik][tx << 1]);

		simdMM4(v0, a.x, b);
		simdMM4(v1, a.y, b);
		simdMM4(v2, a.z, b);
		simdMM4(v3, a.w, b);
	}

	oc0 = oc0 * FH_FW_IC + j0; //j = *fh * FW + fw)*IC + ic
	int oc1 = oc0 + FH_FW_IC, oc2 = oc1 + FH_FW_IC, oc3 = oc2 + FH_FW_IC;

	*(float4*)(dst + oc0) = v0;
	*(float4*)(dst + oc1) = v1;
	*(float4*)(dst + oc2) = v2;
	*(float4*)(dst + oc3) = v3;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*2),  GK >= (BLOCK_SIZE/2), IC % 4 == 0
//LB = 4: GK % 8 == 0
#ifndef DECONV3D_DW_BGEMM_KERNEL_4_2
#define DECONV3D_DW_BGEMM_KERNEL_4_2

//LB = 4: Size = 1, Time = 3.67  msec, Performace = 585.145 GFlop/s
//LB = 3: Size = 1, Time = 5.636 msec, Performace = 381.03  GFlop/s
template<int LB, int STEP>
__global__ void kernel_BGemm_4_2(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int N, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 dYs[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float   Xs[2][1 << LB >> 1][(2 << LB) + 2];

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 2) + oc_index;
	const int toc0 = ((tx & 1) << 1) + oc0;

	//prepare for GM = FH * FW * IC
	int j0 = (((blockIdx.x << LB) + tx) << 1) + j_index;
	int tj0 = j0 + (ty & 1);
	const int FW_IC = FW * IC;
	get_fh_fw_ic(tj0, tfh0, tfw0, tic0);
	tfh0 = tfh0 - oph, tfw0 = tfw0 - opw;
	const int Xoffset0 = (tfh0*IW + tfw0)*IC + tic0;

	//======================================================================
	//prepare for GK_slice = (N_end - N_start) * OH_OW, N_end = nextZ.Nstart
	int bz = blockIdx.z;
	int N_slice = (N / gridDim.z) >> 3 << 3;
	int N_start = bz * N_slice;
	int N_end = (N_start + N_slice - N) * (bz != (gridDim.z - 1)) + N;//min(N_start + Nslice, N)
	const int OH_OW = OH * OW, GK_slice = (N_end - N_start) * OH_OW;
	X += N_start * IH*IW*IC; //X[N_start,...]
	deltaY += N_start * OH_OW*OC; //deltaY[N_start,...]
	//======================================================================

	//load 2 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
	int dY_k = (tx >> 1);//k = (n*OH + oh)*OW + ow
	const int dYs_x = (tx >> 1), dYs_y = (ty << 1) + (tx & 1);
	dYs[buf][dYs_x][dYs_y] = *(float2*)(&deltaY[dY_k*OC + toc0]);

	//load 2 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
	int X_k = (ty >> 1), X_n, X_oh, X_ow;
	get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
	bool lx0 = (tfh0 >= -X_oh) && (tfh0 < IH - X_oh) && (tfw0 >= -X_ow) && (tfw0 < IW - X_ow);
	int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);
	Xs[buf][Xs_y][Xs_x] = (lx0 ? X[Xoffset0 + xoffset] : 0);
	__syncthreads();

	//compute area-----------------------------------------------------
	float2 v0 = make_float2(0, 0);
	float2 v1 = make_float2(0, 0);
	float2 v2 = make_float2(0, 0);
	float2 v3 = make_float2(0, 0);
	for (int ok = 1, OK = (GK_slice << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) 
		{
			float2 b = *(float2*)(&Xs[buf][ik][tx << 1]);
			float4 a = *(float4*)(&dYs[buf][ik][ty << 1]);

			simdMM2(v0, a.x, b);
			simdMM2(v1, a.y, b);
			simdMM2(v2, a.z, b);
			simdMM2(v3, a.w, b);
		}
		buf ^= 1;

		//load 2 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int dY_k = ((ok << LB) + tx) >> 1;
		dYs[buf][dYs_x][dYs_y] = *(float2*)(&deltaY[dY_k*OC + toc0]);

		//load 1 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = ((ok << LB) + ty) >> 1, X_n, X_oh, X_ow;
		get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
		bool lx0 = (tfh0 >= -X_oh) && (tfh0 < IH - X_oh) && (tfw0 >= -X_ow) && (tfw0 < IW - X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		Xs[buf][Xs_y][Xs_x] = (lx0 ? X[Xoffset0 + xoffset] : 0);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float2 b = *(float2*)(&Xs[buf][ik][tx << 1]);
		float4 a = *(float4*)(&dYs[buf][ik][ty << 1]);

		simdMM2(v0, a.x, b);
		simdMM2(v1, a.y, b);
		simdMM2(v2, a.z, b);
		simdMM2(v3, a.w, b);
	}

	const int FH_FW_IC = FH * FW_IC;
	//dst[bz] = deltaW_buf[bz - 1, sizeW], bz >= 1; dst[0] = deltaW
	float* buf_addr = deltaW_buf + (bz - 1)*OC*FH_FW_IC;
	float *dst = (bz != 0) * (buf_addr - deltaW) + deltaW;

	oc0 = oc0 * FH_FW_IC + j0;//j = (fh * FW + fw)*IC + ic
	int oc1 = oc0 + FH_FW_IC, oc2 = oc1 + FH_FW_IC, oc3 = oc2 + FH_FW_IC;

	*(float2*)(dst + oc0) = v0;
	*(float2*)(dst + oc1) = v1;
	*(float2*)(dst + oc2) = v2;
	*(float2*)(dst + oc3) = v3;
}

#endif


//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*4), GK >= (BLOCK_SIZE/2), IC % 4 == 0
//LB = 4: GK % 8 == 0
#ifndef DECONV3D_DW_BGEMM_KERNEL_2_4
#define DECONV3D_DW_BGEMM_KERNEL_2_4

//LB = 4: Size = 1, Time = 4.006 msec, Performace = 536.067 GFlop/s
//LB = 3: Size = 1, Time = 5.962 msec, Performace = 360.195 GFlop/s
template<int LB, int STEP>
__global__ void kernel_BGemm_2_4(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int N, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float  Ys[2][1 << LB >> 1][(2 << LB) + 2];
	__shared__ float2 Xs[2][1 << LB >> 1][(2 << LB) + 2];

	//======================================================================
	//prepare for GK_slice = (N_end - N_start) * OH_OW, N_end = nextZ.Nstart
	int bz = blockIdx.z;
	int N_slice = (N / gridDim.z) >> 3 << 3;
	int N_start = bz * N_slice;
	int N_end = IF_int((bz != (gridDim.z - 1)), N_start + N_slice, N);//min(N_start + Nslice, N)
	const int OH_OW = OH * OW, GK_slice = (N_end - N_start) * OH_OW;
	X += N_start * IH*IW*IC; //X[N_start,...]
	deltaY += N_start * OH_OW*OC; //deltaY[N_start,...]\

	//dst[bz] = deltaW_buf[bz - 1, sizeW], bz >= 1, dst[0] = deltaW
	const int FW_IC = FW * IC, FH_FW_IC = FH * FW_IC;
	float* buf_addr = deltaW_buf + (bz - 1)*OC*FH_FW_IC;
	float *dst = (bz != 0) * (buf_addr - deltaW) + deltaW;
	//======================================================================

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 1) + oc_index;
	deltaY += (tx & 1) + oc0;

	//prepare for GM = FH * FW * IC
	int j0 = (((blockIdx.x << LB) + tx) << 2) + j_index;
	const int tj0 = j0 + ((ty & 1) << 1);
	get_fh_fw_ic(tj0, tfh0, tfw0, tic0);
	tfh0 = tfh0 - oph, tfw0 = tfw0 - opw;
	X += (tfh0*IW + tfw0)*IC + tic0;

	//load 1 element from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
	int Y_k = (tx >> 1);//k = (n*OH + oh)*OW + ow
	const int dYs_x = (tx >> 1), dYs_y = (ty << 1) + (tx & 1);
	Ys[buf][dYs_x][dYs_y] = deltaY[Y_k*OC];

	//load 2 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
	int X_k = (ty >> 1), X_n, X_oh, X_ow;
	get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
	int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
	const int Xs_y = (ty >> 1), Xs_x = (tx << 1) + (ty & 1);
	Xs[buf][Xs_y][Xs_x] = (LOAD_X(tfh0, tfw0) ? *(float2*)(X + xoffset) : FLOAT_ZERO2);
	__syncthreads();

	//compute area-----------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0);
	float4 v1 = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (GK_slice << 1 >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float2 a = *(float2*)(&Ys[buf][ik][ty << 1]);
			float4 b = *(float4*)(&Xs[buf][ik][tx << 1]);
			simdMM4(v0, a.x, b);
			simdMM4(v1, a.y, b);
		}
		buf ^= 1;

		//load 1 element from deltaY[N, OH, OW, OC] :
		int Y_k = ((ok << LB) + tx) >> 1;
		Ys[buf][dYs_x][dYs_y] = deltaY[Y_k*OC];

		//load 2 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = ((ok << LB) + ty) >> 1, X_n, X_oh, X_ow;
		get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		Xs[buf][Xs_y][Xs_x] = (LOAD_X(tfh0, tfw0) ? *(float2*)(X + xoffset) : FLOAT_ZERO2);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float2 a = *(float2*)(&Ys[buf][ik][ty << 1]);
		float4 b = *(float4*)(&Xs[buf][ik][tx << 1]);
		simdMM4(v0, a.x, b);
		simdMM4(v1, a.y, b);
	}

	oc0 = oc0 * FH_FW_IC + j0;//j = *fh * FW + fw)*IC + ic
	int oc1 = oc0 + FH_FW_IC;

	*(float4*)(dst + oc0) = v0;
	*(float4*)(dst + oc1) = v1;
}

#endif


//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*2), GK >= BLOCK_SIZE
//LB = 4: GK >= 16
//LB = 3: GK >=  8
#ifndef DECONV3D_DW_BGEMM_KERNEL_2_2
#define DECONV3D_DW_BGEMM_KERNEL_2_2

//LB = 4: Size = 1, Time = 4.272 msec, Performace = 502.688 GFlop/s
//LB = 4: Size = 0.85144, Time = 3.898 msec, Performace = 469.075 GFlop/s
//LB = 3: Size = 0.85144, Time = 5.306 msec, Performace = 344.601 GFlop/s
template<int LB, int STEP>
__global__ void kernel_BGemm_2_2(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int N, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 Ys[2][1 << LB][(1 << LB) + 1];
	__shared__ float2  Xs[2][1 << LB][(1 << LB) + 1];

	//======================================================================
	//prepare for GK_slice = (N_end - N_start) * OH_OW, N_end = nextZ.Nstart
	int bz = blockIdx.z;
	int N_slice = (N / gridDim.z) >> 3 << 3;
	int N_start = bz * N_slice;
	int N_end = (N_start + N_slice - N) * (bz != (gridDim.z - 1)) + N;//min(N_start + Nslice, N)
	const int OH_OW = OH * OW, GK_slice = (N_end - N_start) * OH_OW;
	X += N_start * IH*IW*IC; //X[N_start,...]
	deltaY += N_start * OH_OW*OC; //deltaY[N_start,...]

	//dst[bz] = deltaW_buf[bz - 1, sizeW], bz >= 1, dst[0] = deltaW
	const int FW_IC = FW * IC, FH_FW_IC = FH * FW_IC;
	float* buf_addr = deltaW_buf + (bz - 1)*OC*FH_FW_IC;
	float *dst = (bz != 0) * (buf_addr - deltaW) + deltaW;
	//======================================================================

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 1) + oc_index;
	deltaY += oc0;

	//prepare for GM = IC * FH * FW
	int j0 = (((blockIdx.x << LB) + tx) << 1) + j_index, j1 = j0 + 1;
	get_fh_fw_ic(j0, fh0, fw0, ic0); fh0 -= oph; fw0 -= opw;
	get_fh_fw_ic(j1, fh1, fw1, ic1); fh1 -= oph; fw1 -= opw;
	const int Xoffset0 = (fh0*IW + fw0)*IC + ic0;
	const int Xoffset1 = (fh1*IW + fw1)*IC + ic1;

	//load 2 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
	int Y_k = tx;
	Ys[buf][tx][ty] = *(float2*)(deltaY + Y_k*OC);

	//load 2 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N] //X_n *= IC;
	int X_k = ty, X_n, X_oh, X_ow;
	get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
	bool lx0 = (fh0 >= -X_oh) && (fh0 < IH - X_oh) && (fw0 >= -X_ow) && (fw0 < IW - X_ow);
	bool lx1 = (fh1 >= -X_oh) && (fh1 < IH - X_oh) && (fw1 >= -X_ow) && (fw1 < IW - X_ow);
	int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
	Xs[buf][ty][tx].x = (lx0 ? X[Xoffset0 + xoffset] : 0);
	Xs[buf][ty][tx].y = (lx1 ? X[Xoffset1 + xoffset] : 0);
	__syncthreads();

	//compute area-----------------------------------------------------
	float2 v0 = make_float2(0, 0);
	float2 v1 = make_float2(0, 0);
	for (int ok = 1, OK = GK_slice >> LB; ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float2 b = Xs[buf][ik][tx];
			float2 a = Ys[buf][ik][ty];
			simdMM2(v0, a.x, b);
			simdMM2(v1, a.y, b);
		}
		buf ^= 1;

		//load 2 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int Y_k = ((ok << LB) + tx);
		Ys[buf][tx][ty] = *(float2*)(deltaY + Y_k*OC);

		//load 2 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N] //X_n *= IC;
		int X_k = (ok << LB) + ty, X_n, X_oh, X_ow;
		get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
		bool lx0 = (fh0 >= -X_oh) && (fh0 < IH - X_oh) && (fw0 >= -X_ow) && (fw0 < IW - X_ow);
		bool lx1 = (fh1 >= -X_oh) && (fh1 < IH - X_oh) && (fw1 >= -X_ow) && (fw1 < IW - X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		Xs[buf][ty][tx].x = (lx0 ? X[Xoffset0 + xoffset] : 0);
		Xs[buf][ty][tx].y = (lx1 ? X[Xoffset1 + xoffset] : 0);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float2 b =  Xs[buf][ik][tx];
		float2 a = Ys[buf][ik][ty];
		simdMM2(v0, a.x, b);
		simdMM2(v1, a.y, b);
	}

	//when GK % STEP != 0--------------------------------------------
	for (int k = GK_slice - (GK_slice&(STEP - 1)); k < GK_slice; k++)
	{
		//load 2 elements from deltaY
		float2 a = *(float2*)(deltaY + k * OC);

		float2 b;//load 2 elements from X
		int X_k = k, X_n, X_oh, X_ow;
		get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
		bool lx0 = (fh0 >= -X_oh) && (fh0 < IH - X_oh) && (fw0 >= -X_ow) && (fw0 < IW - X_ow);
		bool lx1 = (fh1 >= -X_oh) && (fh1 < IH - X_oh) && (fw1 >= -X_ow) && (fw1 < IW - X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		b.x = (lx0 ? X[Xoffset0 + xoffset] : 0);
		b.y = (lx1 ? X[Xoffset1 + xoffset] : 0);

		simdMM2(v0, a.x, b);
		simdMM2(v1, a.y, b);
	}
	//when GK % STEP != 0--------------------------------------------

	oc0 = oc0 * FH_FW_IC + j0; //j = *fh * FW + fw)*IC + ic
	int oc1 = oc0 + FH_FW_IC;

	*(float2*)(dst + oc0) = v0;
	*(float2*)(dst + oc1) = v1;
}

#endif


//(Y: BLOCK_SIZE*4, X: BLOCK_SIZE*1), GK >= BLOCK_SIZE
//LB = 4: GK >= 16
//LB = 3: GK >=  8
#ifndef DECONV3D_DW_BGEMM_KERNEL_4_1
#define DECONV3D_DW_BGEMM_KERNEL_4_1

//LB = 4: Size = 1, Time = 4.286 msec, Performace = 501.046 GFlop/s
//LB = 4: Size = 0.85144, Time = 3.806 msec, Performace = 480.414 GFlop/s
//LB = 3: Size = 0.85144, Time = 5.032 msec, Performace = 363.365 GFlop/s
template<int LB, int STEP>
__global__ void kernel_BGemm_4_1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int N, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float4 dYs[2][1 << LB][(1 << LB) + 1];
	__shared__ float   Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 2) + oc_index;

	//prepare for GM = IC * FH * FW
	int j0 = ((blockIdx.x << LB) + tx) + j_index;
	const int FW_IC = FW * IC;
	get_fh_fw_ic(j0, fh0, fw0, ic0); fh0 -= oph; fw0 -= opw;
	const int Xoffset0 = (fh0*IW + fw0)*IC + ic0;

	//======================================================================
	//prepare for GK_slice = (N_end - N_start) * OH_OW, N_end = nextZ.Nstart
	int bz = blockIdx.z;
	int N_slice = (N / gridDim.z) >> 3 << 3;
	int N_start = bz * N_slice;
	int N_end = (N_start + N_slice - N) * (bz != (gridDim.z - 1)) + N;//min(N_start + Nslice, N)
	const int OH_OW = OH * OW, GK_slice = (N_end - N_start) * OH_OW;
	X += N_start * IH*IW*IC; //X[N_start,...]
	deltaY += N_start * OH_OW*OC; //deltaY[N_start,...]
	//======================================================================

	//load 4 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
	int dY_k = tx;
	dYs[buf][tx][ty] = *(float4*)(&deltaY[dY_k*OC + oc0]);

	//load 1 element from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
	int X_k = ty, X_n, X_oh, X_ow;
	get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
	bool lx0 = (fh0 >= -X_oh) && (fh0 < IH - X_oh) && (fw0 >= -X_ow) && (fw0 < IW - X_ow);
	int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
	Xs[buf][ty][tx] = (lx0 ? X[Xoffset0 + xoffset] : 0);
	__syncthreads();

	//compute area-----------------------------------------------------
	float4 v = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (GK_slice >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float  b = Xs[buf][ik][tx];
			float4 a = dYs[buf][ik][ty];
			simdMM4(v, b, a);
		}
		buf ^= 1;

		//load 4 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int dY_k = (ok << LB) + tx;
		dYs[buf][tx][ty] = *(float4*)(&deltaY[dY_k*OC + oc0]);

		//load 1 element from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = (ok << LB) + ty, X_n, X_oh, X_ow;
		get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
		bool lx0 = (fh0 >= -X_oh) && (fh0 < IH - X_oh) && (fw0 >= -X_ow) && (fw0 < IW - X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		Xs[buf][ty][tx] = (lx0 ? X[Xoffset0 + xoffset] : 0);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float  b =  Xs[buf][ik][tx];
		float4 a = dYs[buf][ik][ty];
		simdMM4(v, b, a);
	}

	//when GK %S TEP != 0--------------------------------------------
	for (int k = GK_slice - (GK_slice&(STEP - 1)); k < GK_slice; k++)
	{
		float4 a = *(float4*)(&deltaY[k*OC + oc0]);//load 4 elements from deltaY

		float b;//load 1 element from X
		int X_k = k;
		get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
		bool lx0 = (fh0 >= -X_oh) && (fh0 < IH - X_oh) && (fw0 >= -X_ow) && (fw0 < IW - X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		b = (lx0 ? X[Xoffset0 + xoffset] : 0);

		simdMM4(v, b, a);
	}
	//when GK %S TEP != 0--------------------------------------------

	const int FH_FW_IC = FH * FW_IC;
	//dst[bz] = deltaW_buf[bz - 1, sizeW], bz >= 1, dst[0] = deltaW
	float* buf_addr = deltaW_buf + (bz - 1)*OC*FH_FW_IC;
	float *dst = (bz != 0) * (buf_addr - deltaW) + deltaW;

	oc0 = oc0 * FH_FW_IC + j0;//j = *fh * FW + fw)*IC + ic
	int oc1 = oc0 + FH_FW_IC, oc2 = oc1 + FH_FW_IC, oc3 = oc2 + FH_FW_IC;

	dst[oc0] = v.x;
	dst[oc1] = v.y;
	dst[oc2] = v.z;
	dst[oc3] = v.w;
}

#endif


//(Y: BLOCK_SIZE*1, X: BLOCK_SIZE*4), GK >= BLOCK_SIZE
//LB = 4: GK >= 16
//LB = 3: GK >=  8
#ifndef DECONV3D_DW_BGEMM_KERNEL_1_4
#define DECONV3D_DW_BGEMM_KERNEL_1_4

//(correct)
//LB = 4: Size = 1, Time = 6.316 msec, Performace = 340.007 GFlop/s
//LB = 4: Size = 0.85144, Time = 5.686 msec, Performace = 321.571 GFlop/s
//LB = 3: Size = 0.85144, Time = 6.582 msec, Performace = 277.796 GFlop/s
template<int LB, int STEP>
__global__ void kernel_BGemm_1_4(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int N, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float  dYs[2][1 << LB][(1 << LB) + 1];
	__shared__ float4  Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GN = OC
	const int oc0 = ((blockIdx.y << LB) + ty) + oc_index;

	//prepare for GM = IC * FH * FW
	int j0 = (((blockIdx.x << LB) + tx) << 2) + j_index;
	int j1 = j0 + 1, j2 = j0 + 2, j3 = j0 + 3;
	const int FW_IC = FW * IC;
	get_fh_fw_ic(j0, fh0, fw0, ic0); fh0 -= oph; fw0 -= opw;
	get_fh_fw_ic(j1, fh1, fw1, ic1); fh1 -= oph; fw1 -= opw;
	get_fh_fw_ic(j2, fh2, fw2, ic2); fh2 -= oph; fw2 -= opw;
	get_fh_fw_ic(j3, fh3, fw3, ic3); fh3 -= oph; fw3 -= opw;
	const int Xoffset0 = (fh0*IW + fw0)*IC + ic0;
	const int Xoffset1 = (fh1*IW + fw1)*IC + ic1;
	const int Xoffset2 = (fh2*IW + fw2)*IC + ic2;
	const int Xoffset3 = (fh3*IW + fw3)*IC + ic3;

	//======================================================================
	//prepare for GK_slice = (N_end - N_start) * OH_OW, N_end = nextZ.Nstart
	int bz = blockIdx.z;
	int N_slice = (N / gridDim.z) >> 3 << 3;
	int N_start = bz * N_slice;
	int N_end = (N_start + N_slice - N) * (bz != (gridDim.z - 1)) + N;//min(N_start + Nslice, N)
	const int OH_OW = OH * OW, GK_slice = (N_end - N_start) * OH_OW;
	X += N_start * IH*IW*IC; //X[N_start,...]
	deltaY += N_start * OH_OW*OC; //deltaY[N_start,...]
	//======================================================================

	//load 1 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
	int dY_k = tx;
	dYs[buf][tx][ty] = deltaY[dY_k*OC + oc0];

	//load 4 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
	int X_k = ty, X_n, X_oh, X_ow;
	get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
	bool lx0 = (fh0 >= -X_oh) && (fh0 < IH - X_oh) && (fw0 >= -X_ow) && (fw0 < IW - X_ow);
	bool lx1 = (fh1 >= -X_oh) && (fh1 < IH - X_oh) && (fw1 >= -X_ow) && (fw1 < IW - X_ow);
	bool lx2 = (fh2 >= -X_oh) && (fh2 < IH - X_oh) && (fw2 >= -X_ow) && (fw2 < IW - X_ow);
	bool lx3 = (fh3 >= -X_oh) && (fh3 < IH - X_oh) && (fw3 >= -X_ow) && (fw3 < IW - X_ow);
	int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
	Xs[buf][ty][tx].x = (lx0 ? X[Xoffset0 + xoffset] : 0);
	Xs[buf][ty][tx].y = (lx1 ? X[Xoffset1 + xoffset] : 0);
	Xs[buf][ty][tx].z = (lx2 ? X[Xoffset2 + xoffset] : 0);
	Xs[buf][ty][tx].w = (lx3 ? X[Xoffset3 + xoffset] : 0);
	__syncthreads();

	//compute area-----------------------------------------------------
	float4 v = make_float4(0, 0, 0, 0);
	for (int ok = 1, OK = (GK_slice >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float4 b =  Xs[buf][ik][tx];
			float  a = dYs[buf][ik][ty];
			simdMM4(v, a, b);
		}
		buf ^= 1;

		//load 1 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int dY_k = (ok << LB) + tx;
		dYs[buf][tx][ty] = deltaY[dY_k*OC + oc0];

		//load 4 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = (ok << LB) + ty, X_n, X_oh, X_ow;
		get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
		bool lx0 = (fh0 >= -X_oh) && (fh0 < IH - X_oh) && (fw0 >= -X_ow) && (fw0 < IW - X_ow);
		bool lx1 = (fh1 >= -X_oh) && (fh1 < IH - X_oh) && (fw1 >= -X_ow) && (fw1 < IW - X_ow);
		bool lx2 = (fh2 >= -X_oh) && (fh2 < IH - X_oh) && (fw2 >= -X_ow) && (fw2 < IW - X_ow);
		bool lx3 = (fh3 >= -X_oh) && (fh3 < IH - X_oh) && (fw3 >= -X_ow) && (fw3 < IW - X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		Xs[buf][ty][tx].x = (lx0 ? X[Xoffset0 + xoffset] : 0);
		Xs[buf][ty][tx].y = (lx1 ? X[Xoffset1 + xoffset] : 0);
		Xs[buf][ty][tx].z = (lx2 ? X[Xoffset2 + xoffset] : 0);
		Xs[buf][ty][tx].w = (lx3 ? X[Xoffset3 + xoffset] : 0);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float4 b =  Xs[buf][ik][tx];
		float  a = dYs[buf][ik][ty];
		simdMM4(v, a, b);
	}

	//when GK%STEP != 0 -------------------------------------------
	for (int k = GK_slice - (GK_slice&(STEP - 1)); k < GK_slice; k++)
	{
		float a = deltaY[k*OC + oc0];//load 1 element from deltaY

		float4 b;//load 4 elements from X
		int X_k = k;
		get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
		bool lx0 = (fh0 >= -X_oh) && (fh0 < IH - X_oh) && (fw0 >= -X_ow) && (fw0 < IW - X_ow);
		bool lx1 = (fh1 >= -X_oh) && (fh1 < IH - X_oh) && (fw1 >= -X_ow) && (fw1 < IW - X_ow);
		bool lx2 = (fh2 >= -X_oh) && (fh2 < IH - X_oh) && (fw2 >= -X_ow) && (fw2 < IW - X_ow);
		bool lx3 = (fh3 >= -X_oh) && (fh3 < IH - X_oh) && (fw3 >= -X_ow) && (fw3 < IW - X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		b.x = lx0 ? X[Xoffset0 + xoffset] : 0;
		b.y = lx1 ? X[Xoffset1 + xoffset] : 0;
		b.z = lx2 ? X[Xoffset2 + xoffset] : 0;
		b.w = lx3 ? X[Xoffset3 + xoffset] : 0;

		simdMM4(v, a, b);
	}
	//when GK%STEP != 0 -------------------------------------------

	const int FH_FW_IC = FH * FW_IC;
	//dst[bz] = deltaW_buf[bz - 1, sizeW], bz >= 1, dst[0] = deltaW
	float* buf_addr = deltaW_buf + (bz - 1)*OC*FH_FW_IC;
	float *dst = (blockIdx.z != 0) * (buf_addr - deltaW) + deltaW;

	*(float4*)(&dst[oc0*FH_FW_IC + j0]) = v;
}

#endif


//(Y: BLOCK_SIZE*2, X: BLOCK_SIZE*1), GK >= BLOCK_SIZE
//LB = 4: GK >= 16
//LB = 3: GK >=  8
#ifndef DECONV3D_DW_BGEMM_KERNEL_2_1
#define DECONV3D_DW_BGEMM_KERNEL_2_1

//LB=4: Size = 0.85144, Time = 5.506 msec, Performace = 332.084 GFlop/s
//LB=3: Size = 0.85144, Time = 8.438 msec, Performace = 216.693 GFlop/s
template<int LB, int STEP>
__global__ void kernel_BGemm_2_1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int N, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 dYs[2][1 << LB][(1 << LB) + 1];
	__shared__ float   Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GN = OC
	int oc0 = (((blockIdx.y << LB) + ty) << 1) + oc_index;

	//prepare for GM = IC * FH * FW
	int j0 = ((blockIdx.x << LB) + tx) + j_index;
	const int FW_IC = FW * IC;
	get_fh_fw_ic(j0, fh0, fw0, ic0); fh0 -= oph; fw0 -= opw;
	const int Xoffset0 = (fh0*IW + fw0)*IC + ic0;

	//======================================================================
	//prepare for GK_slice = (N_end - N_start) * OH_OW, N_end = nextZ.Nstart
	int bz = blockIdx.z;
	int N_slice = (N / gridDim.z) >> 3 << 3;
	int N_start = bz * N_slice;
	int N_end = (N_start + N_slice - N) * (bz != (gridDim.z - 1)) + N;//min(N_start + Nslice, N)
	const int OH_OW = OH * OW, GK_slice = (N_end - N_start) * OH_OW;
	X += N_start * IH*IW*IC; //X[N_start,...]
	deltaY += N_start * OH_OW*OC; //deltaY[N_start,...]
	//======================================================================

	//load 2 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
	int dY_k = tx;
	dYs[buf][tx][ty] = *(float2*)(&deltaY[dY_k*OC + oc0]);

	//load 1 element from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
	int X_k = ty, X_n, X_oh, X_ow;
	get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
	bool lx0 = (fh0 >= -X_oh) && (fh0 < IH - X_oh) && (fw0 >= -X_ow) && (fw0 < IW - X_ow);
	int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
	Xs[buf][ty][tx] = lx0 ? X[Xoffset0 + xoffset] : 0;
	__syncthreads();

	//compute area-----------------------------------------------------
	float2 v = make_float2(0, 0);
	for (int ok = 1, OK = (GK_slice >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float  b =  Xs[buf][ik][tx];
			float2 a = dYs[buf][ik][ty];
			simdMM2(v, b, a);
		}
		buf ^= 1;

		//load 2 elements from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int dY_k = (ok << LB) + tx;
		dYs[buf][tx][ty] = *(float2*)(&deltaY[dY_k*OC + oc0]);

		//load 1 element from X[N, IH, IW, IC] : Xe[IC, IH, IW, N] //X_n *= IC;
		int X_k = (ok << LB) + ty, X_n, X_oh, X_ow;
		get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
		bool lx0 = (fh0 >= -X_oh) && (fh0 < IH - X_oh) && (fw0 >= -X_ow) && (fw0 < IW - X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		Xs[buf][ty][tx] = (lx0 ? X[Xoffset0 + xoffset] : 0);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float  b =  Xs[buf][ik][tx];
		float2 a = dYs[buf][ik][ty];
		simdMM2(v, b, a);
	}

	//when GK % STEP != 0--------------------------------------------
	for (int k = GK_slice - (GK_slice&(STEP - 1)); k < GK_slice; k++)
	{
		float2 a = *(float2*)(&deltaY[k * OC + oc0]);//load 2 elements from deltaY

		float b;//load 1 element from X
		int X_k = k, X_n, X_oh, X_ow;
		get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
		bool lx0 = (fh0 >= -X_oh) && (fh0 < IH - X_oh) && (fw0 >= -X_ow) && (fw0 < IW - X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		b = (lx0 ? X[Xoffset0 + xoffset] : 0);

		simdMM2(v, b, a);
	}
	//when GK % STEP != 0--------------------------------------------

	const int FH_FW_IC = FH * FW_IC;
	//dst[bz] = deltaW_buf[bz - 1, sizeW], bz >= 1, dst[0] = deltaW
	float* buf_addr = deltaW_buf + (bz - 1)*OC*FH_FW_IC;
	float *dst = (bz != 0) * (buf_addr - deltaW) + deltaW;

	oc0 = oc0 * FH_FW_IC + j0; int oc1 = oc0 + FH_FW_IC;
	dst[oc0] = v.x;
	dst[oc1] = v.y;
}

#endif


//(Y: BLOCK_SIZE*1, X: BLOCK_SIZE*2), GK >= BLOCK_SIZE
//LB = 4: GK >= 16
//LB = 3: GK >=  8
#ifndef DECONV3D_DW_BGEMM_KERNEL_1_2
#define DECONV3D_DW_BGEMM_KERNEL_1_2

//LB = 4: Size = 0.85144, Time = 6.5   msec, Performace = 281.301 GFlop/s
//LB = 3: Size = 0.85144, Time = 9.426 msec, Performace = 193.98  GFlop/s
template<int LB, int STEP>
__global__ void kernel_BGemm_1_2(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int N, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float  dYs[2][1 << LB][(1 << LB) + 1];
	__shared__ float2  Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GN = OC
	int oc0 = ((blockIdx.y << LB) + ty) + oc_index;

	//prepare for GM = IC * FH * FW
	int j0 = (((blockIdx.x << LB) + tx) << 1) + j_index, j1 = j0 + 1;
	const int FW_IC = FW * IC;
	get_fh_fw_ic(j0, fh0, fw0, ic0); fh0 -= oph; fw0 -= opw;
	get_fh_fw_ic(j1, fh1, fw1, ic1); fh1 -= oph; fw1 -= opw;
	const int Xoffset0 = (fh0*IW + fw0)*IC + ic0;
	const int Xoffset1 = (fh1*IW + fw1)*IC + ic1;

	//======================================================================
	//prepare for GK_slice = (N_end - N_start) * OH_OW, N_end = nextZ.Nstart
	int bz = blockIdx.z;
	int N_slice = (N / gridDim.z) >> 3 << 3;
	int N_start = bz * N_slice;
	int N_end = (N_start + N_slice - N) * (bz != (gridDim.z - 1)) + N;//min(N_start + Nslice, N)
	const int OH_OW = OH * OW, GK_slice = (N_end - N_start) * OH_OW;
	X += N_start * IH*IW*IC; //X[N_start,...]
	deltaY += N_start * OH_OW*OC; //deltaY[N_start,...]
	//======================================================================

	//load 1 element from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
	int dY_k = tx;
	dYs[buf][tx][ty] = deltaY[dY_k *OC + oc0];

	//load 2 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N] //X_n *= IC;
	int X_k = ty, X_n, X_oh, X_ow;
	get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
	bool lx0 = (fh0 >= -X_oh) && (fh0 < IH - X_oh) && (fw0 >= -X_ow) && (fw0 < IW - X_ow);
	bool lx1 = (fh1 >= -X_oh) && (fh1 < IH - X_oh) && (fw1 >= -X_ow) && (fw1 < IW - X_ow);
	int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
	Xs[buf][ty][tx].x = (lx0 ? X[Xoffset0 + xoffset] : 0);
	Xs[buf][ty][tx].y = (lx1 ? X[Xoffset1 + xoffset] : 0);
	__syncthreads();

	//compute area-----------------------------------------------------
	float2 v = make_float2(0, 0);
	for (int ok = 1, OK = (GK_slice >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float2 b =  Xs[buf][ik][tx];
			float  a = dYs[buf][ik][ty];
			simdMM2(v, a, b);
		}
		buf ^= 1;

		//load 1 element from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int dY_k = (ok << LB) + tx;
		dYs[buf][tx][ty] = deltaY[dY_k*OC + oc0];

		//load 2 elements from X[N, IH, IW, IC] : Xe[IC, IH, IW, N] //X_n *= IC;
		int X_k = (ok << LB) + ty, X_n, X_oh, X_ow;
		get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
		bool lx0 = (fh0 >= -X_oh) && (fh0 < IH - X_oh) && (fw0 >= -X_ow) && (fw0 < IW - X_ow);
		bool lx1 = (fh1 >= -X_oh) && (fh1 < IH - X_oh) && (fw1 >= -X_ow) && (fw1 < IW - X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		Xs[buf][ty][tx].x = (lx0 ? X[Xoffset0 + xoffset] : 0);
		Xs[buf][ty][tx].y = (lx1 ? X[Xoffset1 + xoffset] : 0);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float2 b =  Xs[buf][ik][tx];
		float  a = dYs[buf][ik][ty];
		simdMM2(v, a, b);
	}

	//when GK % STEP != 0--------------------------------------------
	for (int k = GK_slice - (GK_slice&(STEP - 1)); k < GK_slice; k++)
	{
		float a = deltaY[k * OC + oc0];//load 1 element from deltaY

		float2 b;//load 2 elements from X
		int X_k = k, X_n, X_oh, X_ow;
		get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
		bool lx0 = (fh0 >= -X_oh) && (fh0 < IH - X_oh) && (fw0 >= -X_ow) && (fw0 < IW - X_ow);
		bool lx1 = (fh1 >= -X_oh) && (fh1 < IH - X_oh) && (fw1 >= -X_ow) && (fw1 < IW - X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		b.x = (lx0 ? X[Xoffset0 + xoffset] : 0);
		b.y = (lx1 ? X[Xoffset1 + xoffset] : 0);

		simdMM2(v, a, b);
	}
	//when GK % STEP != 0--------------------------------------------

	const int FH_FW_IC = FH * FW_IC;
	//dst[bz] = deltaW_buf[bz - 1, sizeW], bz >= 1, dst[0] = deltaW
	float* buf_addr = deltaW_buf + (bz - 1)*OC*FH_FW_IC;
	float *dst = (bz != 0) * (buf_addr - deltaW) + deltaW;

	*(float2*)(&dst[oc0*FH_FW_IC + j0]) = v;
}


#endif


//(Y: BLOCK_SIZE*1, X: BLOCK_SIZE*1), GK >= BLOCK_SIZE
//LB = 4: GK >= 16
//LB = 3: GK >=  8
#ifndef DECONV3D_DW_BGEMM_KERNEL_1_1
#define DECONV3D_DW_BGEMM_KERNEL_1_1

//LB = 4: Size = 0.85144, Time =  9.718 msec, Performace = 188.151 GFlop/s
//LB = 3: Size = 0.85144, Time = 15.894 msec, Performace = 115.041 GFlop/s
template<int LB, int STEP>
__global__ void kernel_BGemm_1_1(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW,
	int N, int IC, int OC,
	int sh, int sw, int oph, int opw,
	int oc_index, int j_index)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float dYs[2][1 << LB][(1 << LB) + 1];
	__shared__ float  Xs[2][1 << LB][(1 << LB) + 1];

	//prepare for GN = OC
	const int oc0 = ((blockIdx.y << LB) + ty) + oc_index;

	//prepare for GM = IC * FH * FW
	int j0 = ((blockIdx.x << LB) + tx) + j_index;
	const int FW_IC = FW * IC;
	get_fh_fw_ic(j0, fh0, fw0, ic0); fh0 -= oph; fw0 -= opw;
	const int Xoffset0 = (fh0*IW + fw0)*IC + ic0;

	//======================================================================
	//prepare for GK_slice = (N_end - N_start) * OH_OW, N_end = nextZ.Nstart
	int bz = blockIdx.z;
	int N_slice = (N / gridDim.z) >> 3 << 3;
	int N_start = bz * N_slice;
	int N_end = (N_start + N_slice - N) * (bz != (gridDim.z - 1)) + N;//min(N_start + Nslice, N)
	const int OH_OW = OH * OW, GK_slice = (N_end - N_start) * OH_OW;
	X += N_start * IH*IW*IC; //X[N_start,...]
	deltaY += N_start * OH_OW*OC; //deltaY[N_start,...]
	//======================================================================
	//load 1 element from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
	int dY_k = tx;
	dYs[buf][tx][ty] = deltaY[dY_k*OC + oc0];

	//load 1 element from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
	int X_k = ty, X_n, X_oh, X_ow;
	get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
	bool lx0 = (fh0 >= -X_oh) && (fh0 < IH - X_oh) && (fw0 >= -X_ow) && (fw0 < IW - X_ow);
	int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
	Xs[buf][ty][tx] = (lx0 ? X[Xoffset0 + xoffset] : 0);
	__syncthreads();

	//compute area-----------------------------------------------------
	float v = 0;
	for (int ok = 1, OK = (GK_slice >> LB); ok < OK; ok++)
	{
#pragma unroll
		for (int ik = 0; ik < STEP; ik++) {
			float b =  Xs[buf][ik][tx];
			float a = dYs[buf][ik][ty];
			v += a * b;
		}
		buf ^= 1;

		//load 1 element from deltaY[N, OH, OW, OC] : deltaYpe[OC, OHp, OWp, N]
		int dY_k = (ok << LB) + tx;
		dYs[buf][tx][ty] = deltaY[dY_k*OC + oc0];

		//load 1 element from X[N, IH, IW, IC] : Xe[IC, IH, IW, N]
		int X_k = (ok << LB) + ty, X_n, X_oh, X_ow;
		get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		bool lx0 = (fh0 >= -X_oh) && (fh0 < IH - X_oh) && (fw0 >= -X_ow) && (fw0 < IW - X_ow);
		Xs[buf][ty][tx] = (lx0 ? X[Xoffset0 + xoffset] : 0);
		__syncthreads();
	}
#pragma unroll
	for (int ik = 0; ik < STEP; ik++) {
		float b =  Xs[buf][ik][tx];
		float a = dYs[buf][ik][ty];
		v += a * b;
	}

	//when GK % STEP != 0--------------------------------------------
	for (int k = GK_slice - (GK_slice&(STEP - 1)); k < GK_slice; k++)
	{
		float a = deltaY[k * OC + oc0];//load 1 element from deltaY

		float b;//load 1 element from X
		int X_k = k;
		get_X_n_oh_ow(X_k, X_n, X_oh, X_ow);
		bool lx0 = (fh0 >= -X_oh) && (fh0 < IH - X_oh) && (fw0 >= -X_ow) && (fw0 < IW - X_ow);
		int xoffset = ((X_n + X_oh)*IW + X_ow)*IC;
		b = (lx0 ? X[Xoffset0 + xoffset] : 0);

		v += a * b;
	}
	//when GK % STEP != 0--------------------------------------------

	const int FH_FW_IC = FH * FW_IC;
	//dst[bz] = deltaW_buf[bz - 1, sizeW], bz >= 1, dst[0] = deltaW
	float* buf_addr = deltaW_buf + (bz - 1)*OC*FH_FW_IC;
	float *dst = (bz != 0) * (buf_addr - deltaW) + deltaW;

	dst[oc0 *FH_FW_IC + j0] = v;
}

#endif

#endif