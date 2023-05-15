#pragma once

#ifndef DECONV_3D_DELTAW_H
#define DECONV_3D_DELTAW_H

//Gemm---------------------------------------
#include "dconv3d_dW_Gemm_kernel.cuh"
#include "dconv3d_dW_Gemm_kernel_EX.cuh"
#include "dconv3d_dW_Gemm_kernel_EX2.cuh"
#include "dconv3d_dW_Gemm_kernel_W1.cuh"

//#include "dconv3d_dW_GemmV2_kernel.cuh"
//#include "dconv3d_dW_GemmV2_kernel_EX.cuh"
//#include "dconv3d_dW_GemmV2_kernel_EX2.cuh"

//GemmSK-------------------------------------
#include "dconv3d_dW_GemmSK_bufSum.cuh"
#include "dconv3d_dW_GemmSK_kernel.cuh"
#include "dconv3d_dW_GemmSK_kernel_EX.cuh"
#include "dconv3d_dW_GemmSK_kernel_EX2.cuh"
#include "dconv3d_dW_GemmSK_uernel.cuh"
#include "dconv3d_dW_GemmSK_sernel.cuh"

#include "dconv3d_dW_GemmSK_kernel_W1.cuh"
#include "dconv3d_dW_GemmSK_sernel_W1.cuh"

//GemmSK V2----------------------------------
#include "dconv3d_dW_GemmV2SK_kernel.cuh"
#include "dconv3d_dW_GemmV2SK_kernel_EX.cuh"
#include "dconv3d_dW_GemmV2SK_kernel_EX2.cuh"
#include "dconv3d_dW_GemmV2SK_sernel.cuh"

//FFT[for large conv kernel]-----------------
//#include "dconv3D_dW_FFT.cuh"
//#include "dconv3D_dW_FFT_s1.cuh"


#ifndef GEMM_AREA
#define GEMM_AREA

#ifndef DECONV_3D_DELTAW_GEMM
#define DECONV_3D_DELTAW_GEMM

#define __dconv3D_deltaW_Gemm(streams, index, length, X, IH, IW, deltaY, OH, OW, deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw)\
	deconv3d_deltaW_Gemm(streams, index, length, X,IH,IW, deltaY,OH,OW, deltaW,FH,FW, N,IC,OC, sh,sw,ph,pw,\
		GET_GN(OC), GET_GM(IC, FH, FW), GET_GK(N, OH, OW), 0, 0)

#define dconv3d_dW_Gemm_Branch(SIZE_Y, SIZE_X) {\
	int GNr = GN & (SIZE_Y), GMr = GM & (SIZE_X);\
	if(GNr && GMr){\
		int next_oc_index = (GN - GNr) + oc_index;\
		int next_j_index = (GM - GMr) + j_index;\
		deconv3d_deltaW_Gemm(streams, index, length, X,IH,IW, deltaY,OH,OW, deltaW,FH,FW, N,IC,OC, sh,sw,ph,pw,\
			GNr, GM, GK, next_oc_index, j_index);\
		deconv3d_deltaW_Gemm(streams, index, length, X,IH,IW, deltaY,OH,OW, deltaW,FH,FW, N,IC,OC, sh,sw,ph,pw,\
            GN, GMr, GK, oc_index, next_j_index);\
		deconv3d_deltaW_Gemm(streams, index, length, X,IH,IW, deltaY,OH,OW, deltaW,FH,FW, N,IC,OC, sh,sw,ph,pw,\
            GNr, GMr, GK, next_oc_index, next_j_index);}\
	else if(GNr){\
		int next_oc_index = (GN - GNr) + oc_index;\
		deconv3d_deltaW_Gemm(streams, index, length, X,IH,IW, deltaY,OH,OW, deltaW,FH,FW, N,IC,OC, sh,sw,ph,pw,\
			 GNr, GM, GK, next_oc_index, j_index);}\
	else if(GMr){\
		int next_j_index = (GM - GMr) + j_index;\
		deconv3d_deltaW_Gemm(streams, index, length, X,IH,IW, deltaY,OH,OW, deltaW,FH,FW, N,IC,OC, sh,sw,ph,pw,\
			 GN, GMr, GK, oc_index, next_j_index);}}

//We have: 
//	(1) GN>=4, GN%4==0
//	(2) GM>=4, GM%4==0
//	(3) GK>=4, GK%4==0
void deconv3d_deltaW_Gemm(jlong *streams, int &index, int length,
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
		  float* __restrict__ deltaW, int FH, int FW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw,
	int GN, int GM, int GK,
	int oc_index, int j_index)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)streams[index];
	index = (index + 1) % length;

	if ((GN > 127) && (GM > 127) && !(GK & 7)) {//[128, 128], GK % 8 ==0
		if (IS_POWER2(OH) && IS_POWER2(OW)) kGemm88_ohw2pow(stream, 4, oc_index, j_index, X, IH, IW, deltaY, LOG2(OH), LOG2(OW), deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
		else if (OH == 3 && OW == 3) fGemm88(stream, 4, oc_index, j_index, X, IH, IW, deltaY, 3, 3, deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
		else if (OH == 5 && OW == 5) fGemm88(stream, 4, oc_index, j_index, X, IH, IW, deltaY, 5, 5, deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
		else if (OH == 7 && OW == 7) fGemm88(stream, 4, oc_index, j_index, X, IH, IW, deltaY, 7, 7, deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
		else kGemm88(stream, 4, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dW_Gemm_Branch(127, 127); return;
	}

	if ((GN > 63) && (GM > 63)) {//[64, 64]
		if (IS_POWER2(OH) && IS_POWER2(OW)) kGemm88_ohw2pow(stream, 3, oc_index, j_index, X, IH, IW, deltaY, LOG2(OH), LOG2(OW), deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
		else if (OH == 3 && OW == 3) fGemm88(stream, 3, oc_index, j_index, X, IH, IW, deltaY, 3, 3, deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
		else if (OH == 5 && OW == 5) fGemm88(stream, 3, oc_index, j_index, X, IH, IW, deltaY, 5, 5, deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
		else if (OH == 7 && OW == 7) fGemm88(stream, 3, oc_index, j_index, X, IH, IW, deltaY, 7, 7, deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
		else kGemm88(stream, 3, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dW_Gemm_Branch(63, 63); return;
	}

	if ((GN > 63) && (GM > 31)) {//[64, 32]
		if (IS_POWER2(OH) && IS_POWER2(OW)) kGemm84_ohw2pow(stream, 3, oc_index, j_index, X, IH, IW, deltaY, LOG2(OH), LOG2(OW), deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
		else kGemm84(stream, 3, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dW_Gemm_Branch(63, 31); return;
	}

	if ((GN > 31) && (GM > 63)) {//[32, 64]
		if (IS_POWER2(OH) && IS_POWER2(OW)) kGemm48_ohw2pow(stream, 3, oc_index, j_index, X, IH, IW, deltaY, LOG2(OH), LOG2(OW), deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
		else kGemm48(stream, 3, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dW_Gemm_Branch(31, 63); return;
	}

	if ((GN > 31) && (GM > 31)) {//[32, 32]
		if (IS_POWER2(OH) && IS_POWER2(OW)) kGemm44_ohw2pow(stream, 3, oc_index, j_index, X, IH, IW, deltaY, LOG2(OH), LOG2(OW), deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
		else kGemm44(stream, 3, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dW_Gemm_Branch(31, 31); return;
	}

	if ((GN > 63) && (GM > 15)) {//[64, 16]
		if (IS_POWER2(OH) && IS_POWER2(OW)) kGemm82_ohw2pow(stream, 3, oc_index, j_index, X, IH, IW, deltaY, LOG2(OH), LOG2(OW), deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
		else kGemm82(stream, 3, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dW_Gemm_Branch(63, 15); return;
	}
	if ((GN > 15) && (GM > 63)) {//[16, 64]
		if (IS_POWER2(OH) && IS_POWER2(OW)) kGemm28_ohw2pow(stream, 3, oc_index, j_index, X, IH, IW, deltaY, LOG2(OH), LOG2(OW), deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
		else kGemm28(stream, 3, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dW_Gemm_Branch(15, 63); return;
	}

	if ((GN > 31) && (GM > 15)) {//[32, 16]
		kGemm42(stream, 3, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dW_Gemm_Branch(31, 15); return;
	}
	if ((GN > 15) && (GM > 31)) {//[16, 32]
		kGemm24(stream, 3, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dW_Gemm_Branch(15, 31); return;
	}

	if (GK > 7) {//GK >= 8
		if ((GN > 15) && (GM > 15)) {
			kGemm22(stream, 3, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
			dconv3d_dW_Gemm_Branch(15, 15); return;
		}
		if ((GN > 31) && (GM > 7)) {
			kGemm41(stream, 3, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
			dconv3d_dW_Gemm_Branch(31, 7); return;
		}
		if ((GN > 7) && (GM > 31)) {
			kGemm14(stream, 3, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
			dconv3d_dW_Gemm_Branch(7, 31); return;
		}
		if ((GN > 15) && (GM > 7)) {
			kGemm21(stream, 3, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
			dconv3d_dW_Gemm_Branch(15, 7); return;
		}
		if ((GN > 7) && (GM > 15)) {
			kGemm12(stream, 3, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
			dconv3d_dW_Gemm_Branch(7, 15); return;
		}
		if ((GN > 7) && (GM > 7)) {
			kGemm11(stream, 3, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
			dconv3d_dW_Gemm_Branch(7, 7); return;
		}
	}

	if (GN > 15) {
		kGemm41(stream, 2, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dW_Gemm_Branch(15, 3); return;
	}
	if (GM > 15) {
		kGemm14(stream, 2, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dW_Gemm_Branch(3, 15); return;
	}
	if (GN > 7) {
		kGemm21(stream, 2, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dW_Gemm_Branch(7, 3); return;
	}
	if (GM>7) {
		kGemm12(stream, 2, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dW_Gemm_Branch(3, 7); return;
	}
	kGemm11(stream, 2, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GM);
}

#endif


#ifndef DECONV_3D_DELTAW_GEMM_W1
#define DECONV_3D_DELTAW_GEMM_W1

#define __dconv3D_deltaW_W1(streams, index, length, X,IH,IW, deltaY, deltaW,  N,IC,OC)\
	deconv3d_deltaW_W1(streams, index, length, X,IH,IW, deltaY, deltaW, N,IC,OC,\
		GET_GN(OC), IC, GET_GK(N, IH, IW), 0, 0)

#define dconv3d_dW_W1_Branch(SIZE_Y, SIZE_X) {\
	int GNr = GN & (SIZE_Y), GMr = GM & (SIZE_X);\
	if(GNr && GMr){\
		int next_oc_index = (GN - GNr) + oc_index;\
		int next_j_index = (GM - GMr) + j_index;\
		deconv3d_deltaW_W1(streams, index, length, X,IH,IW, deltaY, deltaW, N,IC,OC,\
			GNr, GM, GK, next_oc_index, j_index);\
		deconv3d_deltaW_W1(streams, index, length, X,IH,IW, deltaY, deltaW, N,IC,OC,\
            GN, GMr, GK, oc_index, next_j_index);\
		deconv3d_deltaW_W1(streams, index, length, X,IH,IW, deltaY, deltaW, N,IC,OC,\
            GNr, GMr, GK, next_oc_index, next_j_index);}\
	else if(GNr){\
		int next_oc_index = (GN - GNr) + oc_index;\
		deconv3d_deltaW_W1(streams, index, length, X,IH,IW, deltaY, deltaW, N,IC,OC,\
			 GNr, GM, GK, next_oc_index, j_index);}\
	else if(GMr){\
		int next_j_index = (GM - GMr) + j_index;\
		deconv3d_deltaW_W1(streams, index, length, X,IH,IW, deltaY, deltaW, N,IC,OC,\
			 GN, GMr, GK, oc_index, next_j_index);}}

//GN%4 == 0, GN >= 4
//GM%4 == 0, GM >= 4
//GK%4 == 0, GK >= 4
void deconv3d_deltaW_W1(jlong *streams, int &index, int length,
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY,
		  float* __restrict__ deltaW, 
	int N, int IC, int OC,
	int GN, int GM, int GK,
	int oc_index, int j_index)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)streams[index];
	index = (index + 1) % length;

	if(!(GK & 7)){//GK % 8 == 0
		if ((GN > 63) && (GM > 63)) {
			k44W1(stream, 4, oc_index, j_index, X, IH, IW, deltaY, deltaW, N, IC, OC, GN, GM);
			dconv3d_dW_W1_Branch(63, 63); return;
		}
		if ((GN > 31) && (GM > 63)) {
			k24W1(stream, 4, oc_index, j_index, X, IH, IW, deltaY, deltaW, N, IC, OC, GN, GM);
			dconv3d_dW_W1_Branch(31, 63); return;
		}
		if ((GN > 63) && (GM > 31)) {
			k42W1(stream, 4, oc_index, j_index, X, IH, IW, deltaY, deltaW, N, IC, OC, GN, GM);
			dconv3d_dW_W1_Branch(63, 31); return;
		}
	}

	if ((GN > 31) && (GM > 31)) {//GK % 4 == 0
		k44W1(stream, 3, oc_index, j_index, X, IH, IW, deltaY, deltaW, N, IC, OC, GN, GM);
		dconv3d_dW_W1_Branch(31, 31); return;
	}
	if ((GN > 15) && (GM > 31)) {//GK % 4 == 0
		k24W1(stream, 3, oc_index, j_index, X, IH, IW, deltaY, deltaW, N, IC, OC, GN, GM);
		dconv3d_dW_W1_Branch(15, 31); return;
	}
	if ((GN > 31) && (GM > 15)) {//GK % 4 == 0
		k42W1(stream, 3, oc_index, j_index, X, IH, IW, deltaY, deltaW, N, IC, OC, GN, GM);
		dconv3d_dW_W1_Branch(31, 15); return;
	}
	if ((GN > 15) && (GM > 15)) {
		k22W1(stream, 3, oc_index, j_index, X, IH, IW, deltaY, deltaW, N, IC, OC, GN, GM);
		dconv3d_dW_W1_Branch(15, 15); return;
	}
	if ((GN > 7) && (GM > 15)) {
		k12W1(stream, 3, oc_index, j_index, X, IH, IW, deltaY, deltaW, N, IC, OC, GN, GM);
		dconv3d_dW_W1_Branch(7, 15); return;
	}
	if ((GN > 15) && (GM > 7)) {
		k21W1(stream, 3, oc_index, j_index, X, IH, IW, deltaY, deltaW, N, IC, OC, GN, GM);
		dconv3d_dW_W1_Branch(15, 7); return;
	}
	if ((GN > 7) && (GM > 7)) {
		k11W1(stream, 3, oc_index, j_index, X, IH, IW, deltaY, deltaW, N, IC, OC, GN, GM);
		dconv3d_dW_W1_Branch(7, 7); return;
	}

	if (GN > 7) {
		k21W1(stream, 2, oc_index, j_index, X, IH, IW, deltaY, deltaW, N, IC, OC, GN, GM);
		dconv3d_dW_W1_Branch(7, 3); return;
	}
	if (GM > 7) {
		k12W1(stream, 2, oc_index, j_index, X, IH, IW, deltaY, deltaW, N, IC, OC, GN, GM);
		dconv3d_dW_W1_Branch(3, 7); return;
	}
	k11W1(stream, 2, oc_index, j_index, X, IH, IW, deltaY, deltaW, N, IC, OC, GN, GM);
}


#endif

#endif


//Split K to improve parallism
#ifndef GEMMSK_AREA
#define GEMMSK_AREA

#ifndef DECONV_3D_DELTAW_GEMMSK
#define DECONV_3D_DELTAW_GEMMSK

#define __dconv3D_deltaW_GemmSK(streams, index, length, GZ, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N,IC, OC, sh, sw, ph, pw)\
	deconv3d_deltaW_GemmSK(streams,index,length,GZ,\
		X,IH,IW, deltaY,OH,OW, deltaW,deltaW_buf,FH,FW, N,IC,OC, sh,sw,ph,pw,\
		GN, GM, GK, GK_slice, 0, 0)

#define dconv3d_dW_GemmSK_Branch(SIZE_Y, SIZE_X) {\
	int GNr = GN & (SIZE_Y), GMr = GM & (SIZE_X);\
	if(GNr && GMr){\
		int next_oc_index = (GN - GNr) + oc_index;\
		int next_j_index = (GM - GMr) + j_index;\
		deconv3d_deltaW_GemmSK(streams, index, length, GZ, X,IH,IW, deltaY,OH,OW, deltaW,deltaW_buf,FH,FW, N,IC,OC, sh,sw,ph,pw,\
			GNr, GM, GK, GK_slice, next_oc_index, j_index);\
		deconv3d_deltaW_GemmSK(streams, index, length, GZ, X,IH,IW, deltaY,OH,OW, deltaW,deltaW_buf,FH,FW, N,IC,OC, sh,sw,ph,pw,\
            GN, GMr, GK, GK_slice, oc_index, next_j_index);\
		deconv3d_deltaW_GemmSK(streams, index, length, GZ, X,IH,IW, deltaY,OH,OW, deltaW,deltaW_buf,FH,FW, N,IC,OC, sh,sw,ph,pw,\
            GNr, GMr, GK, GK_slice, next_oc_index, next_j_index);}\
	else if(GNr){\
		int next_oc_index = (GN - GNr) + oc_index;\
		deconv3d_deltaW_GemmSK(streams, index, length, GZ, X,IH,IW, deltaY,OH,OW, deltaW,deltaW_buf,FH,FW, N,IC,OC, sh,sw,ph,pw,\
			 GNr, GM, GK, GK_slice, next_oc_index, j_index);}\
	else if(GMr){\
		int next_j_index = (GM - GMr) + j_index;\
		deconv3d_deltaW_GemmSK(streams, index, length, GZ, X,IH,IW, deltaY,OH,OW, deltaW,deltaW_buf,FH,FW, N,IC,OC, sh,sw,ph,pw,\
			 GN, GMr, GK, GK_slice, oc_index, next_j_index);}}

//We have: 
//	(1) GN >= 4, GN % 4==0
//	(2) GM >= 4, GM % 4==0
//	(3) GK >= 4, GK % 4==0, As N >= 16, so GK >= 16
//	(4) GK_slice % 8 == 0
void deconv3d_deltaW_GemmSK(jlong *streams, int &index, int length, int GZ,
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW, 
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw,
	int GN, int GM, int GK, int GK_slice,
	int oc_index, int j_index)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)streams[index];
	index = (index + 1) % length;
	
	if ((GN > 127) && (GM > 127) && !(GK & 7)) {//[128, 128], GK % 8 ==0
		if (IS_POWER2(OH) && IS_POWER2(OW)) kGemmSK88_ohw2pow(stream, 4, GZ, oc_index, j_index, X, IH, IW, deltaY, LOG2(OH), LOG2(OW), deltaW, deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
		//======[Fixed Kernel]========================================================================
		else if ((OH ==   7) && (OW) ==   7) fGemmSK88(stream, 4, GZ, oc_index, j_index, X, IH, IW, deltaY,   7,   7, deltaW, deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
		else if ((OH ==  14) && (OW) ==  14) fGemmSK88(stream, 4, GZ, oc_index, j_index, X, IH, IW, deltaY,  14,  14, deltaW, deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
		else if ((OH ==  28) && (OW) ==  28) fGemmSK88(stream, 4, GZ, oc_index, j_index, X, IH, IW, deltaY,  28,  28, deltaW, deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
		else if ((OH ==  56) && (OW) ==  56) fGemmSK88(stream, 4, GZ, oc_index, j_index, X, IH, IW, deltaY,  56,  56, deltaW, deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
		else if ((OH == 112) && (OW) == 112) fGemmSK88(stream, 4, GZ, oc_index, j_index, X, IH, IW, deltaY, 112, 112, deltaW, deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
		else if ((OH == 224) && (OW) == 224) fGemmSK88(stream, 4, GZ, oc_index, j_index, X, IH, IW, deltaY, 224, 224, deltaW, deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
		//======[Common]==============================================================================
		//else if (!(N & 7) && IS_POWER2(N)) kGemmSK88_n2pow(stream, 4, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, LOG2(N), IC, OC, sh, sw, ph, pw, GN, GM);
		else kGemmSK88(stream, 4, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dW_GemmSK_Branch(127, 127); return;
	}

	if ((GN > 63) && (GM > 63)) {//[64, 64]
		if (IS_POWER2(OH) && IS_POWER2(OW)) kGemmSK88_ohw2pow(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, LOG2(OH), LOG2(OW), deltaW, deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
		//======[Fixed Kernel]========================================================================
		else if ((OH ==   7) && (OW) ==   7) fGemmSK88(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY,   7,   7, deltaW, deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
		else if ((OH ==  14) && (OW) ==  14) fGemmSK88(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY,  14,  14, deltaW, deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
		else if ((OH ==  28) && (OW) ==  28) fGemmSK88(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY,  28,  28, deltaW, deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
		else if ((OH ==  56) && (OW) ==  56) fGemmSK88(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY,  56,  56, deltaW, deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
		else if ((OH == 112) && (OW) == 112) fGemmSK88(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, 112, 112, deltaW, deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
		else if ((OH == 224) && (OW) == 224) fGemmSK88(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, 224, 224, deltaW, deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
		//======[Common]==============================================================================
		//else if (!(N & 3) && IS_POWER2(N)) kGemmSK88_n2pow(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, LOG2(N), IC, OC, sh, sw, ph, pw, GN, GM);
		else kGemmSK88(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dW_GemmSK_Branch(63, 63); return;
	}

	if ((GN > 63) && (GM > 31)) {//[64, 32]
		if (IS_POWER2(OH) && IS_POWER2(OW)) kGemmSK84_ohw2pow(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, LOG2(OH), LOG2(OW), deltaW, deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
		else kGemmSK84(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dW_GemmSK_Branch(63, 31); return;
	}
	if ((GN > 31) && (GM > 63)) {//[32, 64]
		if (IS_POWER2(OH) && IS_POWER2(OW)) kGemmSK48_ohw2pow(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, LOG2(OH), LOG2(OW), deltaW, deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
		else kGemmSK48(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dW_GemmSK_Branch(31, 63); return;
	}

	if ((GN > 31) && (GM > 31)) {//[32, 32]
		if (IS_POWER2(OH) && IS_POWER2(OW)) kGemmSK44_ohw2pow(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, LOG2(OH), LOG2(OW), deltaW, deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
		else kGemmSK44(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dW_GemmSK_Branch(31, 31); return;
	}

	if ((GN > 63) && (GM > 15)) {//[64, 156]
		if (IS_POWER2(OH) && IS_POWER2(OW)) kGemmSK82_ohw2pow(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, LOG2(OH), LOG2(OW), deltaW, deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
		else kGemmSK82(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dW_GemmSK_Branch(63, 15); return;
	}
	if ((GN > 15) && (GM > 63)) {//[16, 64]
		if (IS_POWER2(OH) && IS_POWER2(OW)) kGemmSK28_ohw2pow(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, LOG2(OH), LOG2(OW), deltaW, deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
		else kGemmSK28(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dW_GemmSK_Branch(15, 63); return;
	}

	if ((GN > 31) && (GM > 15)) {//[32, 16]
		if (IS_POWER2(OH) && IS_POWER2(OW)) kGemmSK42_ohw2pow(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, LOG2(OH), LOG2(OW), deltaW, deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
		else kGemmSK42(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dW_GemmSK_Branch(31, 15); return;
	}
	if ((GN > 15) && (GM > 31)) {//[16, 32]
		if (IS_POWER2(OH) && IS_POWER2(OW)) kGemmSK24_ohw2pow(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, LOG2(OH), LOG2(OW), deltaW, deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
		else kGemmSK24(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dW_GemmSK_Branch(15, 31); return;
	}
	
	//======[Small: GK_slice >= 8]=====================================================
	if ((GN > 15) && (GM > 15)) {//[16, 16]
		kGemmSK22(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dW_GemmSK_Branch(15, 15); return;
	}
	
	if (GM > 7) {//GM >= 8
		if (GN > 63) {//[63, 8]
			kGemmSK81(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
			dconv3d_dW_GemmSK_Branch(63, 7); return;
		}
		if (GN > 31) {//[32, 8]
			kGemmSK41(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
			dconv3d_dW_GemmSK_Branch(31, 7); return;
		}
		if (GN > 15) {//[16, 8]
			kGemmSK21(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
			dconv3d_dW_GemmSK_Branch(15, 7); return;
		}
	}

	if (GN > 7) {//GN >= 7
		if (GM > 31) {//[8, 32]
			kGemmSK14(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
			dconv3d_dW_GemmSK_Branch(7, 31); return;
		}
		if (GM > 15) {//[8, 16]
			kGemmSK12(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
			dconv3d_dW_GemmSK_Branch(7, 15); return;
		}
		if (GM >  7) {//[8, 8]
			kGemmSK11(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
			dconv3d_dW_GemmSK_Branch(7, 7); return;
		}
	}

	if (GN > 63) {//[64, 4]
		sGemmSK_8x2_1(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dW_GemmSK_Branch(63, 3); return;
	}
	if (GN > 31) {//[32, 4]
		sGemmSK_4x2_1(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dW_GemmSK_Branch(31, 3); return;
	}
	if (GN > 15) {//[16, 4]
		sGemmSK_2x2_1(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dW_GemmSK_Branch(15, 3); return;
	}
	if (GN >  7) {//[8, 4]
		kGemmSK21(stream, 2, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dW_GemmSK_Branch(7, 3); return;
	}
	
	if (GM > 31) {//[4, 32]
		sGemmSK_1_4x2(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dW_GemmSK_Branch(3, 31); return;
	}
	if (GM > 15) {//[4, 16]
		sGemmSK_1_2x2(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dW_GemmSK_Branch(3, 15); return;
	}
	if (GM >  7) {//[4, 8]
		kGemmSK12(stream, 2, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
		dconv3d_dW_GemmSK_Branch(3, 7); return;
	}
	kGemmSK11(stream, 2, GZ, oc_index, j_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, IC, OC, sh, sw, ph, pw, GN, GM);
}

#endif

	
#ifndef DECONV_3D_DELTAW_GEMM_V2_SK
#define DECONV_3D_DELTAW_GEMM_V2_SK

#define __dconv3D_deltaW_GemmV2SK(streams, index, length, GZ, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw)\
	deconv3d_deltaW_GemmV2SK(streams,index,length,GZ,\
		X,IH,IW, deltaY,OH,OW, deltaW,deltaW_buf,FH,FW, N,IC,OC, sh,sw,ph,pw,\
		OC, IC, 0, 0)

#define dconv3d_dW_GemmV2SK_Branch(SIZE_Y, SIZE_X) {\
	int GNr = GN & (SIZE_Y), GICr = GIC & (SIZE_X);\
	if(GNr && GICr){\
		int next_oc_index = (GN - GNr) + oc_index;\
		int next_ic_index = (GIC - GICr) + ic_index;\
		deconv3d_deltaW_GemmV2SK(streams,index,length,GZ, X,IH,IW, deltaY,OH,OW, deltaW,deltaW_buf,FH,FW, N,IC,OC, sh,sw,ph,pw,\
			GNr, GIC, next_oc_index, ic_index);\
		deconv3d_deltaW_GemmV2SK(streams,index,length,GZ, X,IH,IW, deltaY,OH,OW, deltaW,deltaW_buf,FH,FW, N,IC,OC, sh,sw,ph,pw,\
            GN, GICr, oc_index, next_ic_index);\
		deconv3d_deltaW_GemmV2SK(streams,index,length,GZ, X,IH,IW, deltaY,OH,OW, deltaW,deltaW_buf,FH,FW, N,IC,OC, sh,sw,ph,pw,\
            GNr, GICr, next_oc_index, next_ic_index);}\
	else if(GNr){\
		int next_oc_index = (GN - GNr) + oc_index;\
		deconv3d_deltaW_GemmV2SK(streams,index,length,GZ, X,IH,IW, deltaY,OH,OW, deltaW,deltaW_buf,FH,FW, N,IC,OC, sh,sw,ph,pw,\
			 GNr, GIC, next_oc_index, ic_index);}\
	else if(GICr){\
		int next_ic_index = (GIC - GICr) + ic_index;\
		deconv3d_deltaW_GemmV2SK(streams,index,length,GZ, X,IH,IW, deltaY,OH,OW, deltaW,deltaW_buf,FH,FW, N,IC,OC, sh,sw,ph,pw,\
			 GN, GICr, oc_index, next_ic_index);}}

//We have: 
//	(1) GN = OC  >= 64, GN % 4==0
//	(2) GIC >= 64, GM % 4==0
//	(3) GK >= 4, GK % 4==0, As N >= 16, so GK >= 16
//	(4) Q = padding_scaleUp > 1.1 (IH = IW = 16, ph = pw = 1)
//	(5)
void deconv3d_deltaW_GemmV2SK(jlong *streams, int &index, int length, int GZ,
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY, int OH, int OW,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf, int FH, int FW, 
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw,
	int GN, int GIC, 
	int oc_index, int ic_index)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)streams[index];
	index = (index + 1) % length;
	
	if ((GN > 127) && (GIC > 127) && !(N & 7)) {//[128, 128], N % 8 == 0
		if (CAN_V2_O2P1) kGemmV2SK88O2P1_LB4(stream, 4, GZ, oc_index, ic_index, X, IH, IW, deltaY, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, GN, GIC);
		//======[N is power of 2]=======================================================
		else if (IS_POWER2(N)) {
			if (CAN_V2_O4P1) kGemmV2SK88O4P1_n2pow(stream, 4, GZ, oc_index, ic_index, X, IH, IW, deltaY, deltaW, deltaW_buf, FH, FW, LOG2(N), IC, OC, sh, sw, GN, GIC);
			else if (CAN_V2_O7P1) kGemmV2SK88O7P1_n2pow(stream, 4, GZ, oc_index, ic_index, X, IH, IW, deltaY, deltaW, deltaW_buf, FH, FW, LOG2(N), IC, OC, sh, sw, GN, GIC);
			else if (CAN_V2_O8P1) kGemmV2SK88O8P1_n2pow(stream, 4, GZ, oc_index, ic_index, X, IH, IW, deltaY, deltaW, deltaW_buf, FH, FW, LOG2(N), IC, OC, sh, sw, GN, GIC);
			else kGemmV2SK88_n2pow_LB4(stream, 4, GZ, oc_index, ic_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, LOG2(N), IC, OC, sh, sw, ph, pw, GN, GIC);
		}
		//======[Common]================================================================
		else if (CAN_V2_O4P1) kGemmV2SK88O4P1(stream, 4, GZ, oc_index, ic_index, X, IH, IW, deltaY, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, GN, GIC);
		else if (CAN_V2_O7P1) kGemmV2SK88O7P1(stream, 4, GZ, oc_index, ic_index, X, IH, IW, deltaY, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, GN, GIC);
		else if (CAN_V2_O8P1) kGemmV2SK88O8P1(stream, 4, GZ, oc_index, ic_index, X, IH, IW, deltaY, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, GN, GIC);
		else kGemmV2SK88(stream, 4, GZ, oc_index, ic_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GIC);
		dconv3d_dW_GemmV2SK_Branch(127, 127); return;
	}
	
	if ((GN > 63) && (GIC > 63)) {//[64, 64], N % 4 == 0
		if (CAN_V2_O2P1) kGemmV2SK88O2P1_LB3(stream, 3, GZ, oc_index, ic_index, X, IH, IW, deltaY, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, GN, GIC);
		//======[N is power of 2]=======================================================
		else if (IS_POWER2(N)) {
			if (CAN_V2_O4P1) kGemmV2SK88O4P1_n2pow(stream, 3, GZ, oc_index, ic_index, X, IH, IW, deltaY, deltaW, deltaW_buf, FH, FW, LOG2(N), IC, OC, sh, sw, GN, GIC);
			else if (CAN_V2_O7P1) kGemmV2SK88O7P1_n2pow(stream, 3, GZ, oc_index, ic_index, X, IH, IW, deltaY, deltaW, deltaW_buf, FH, FW, LOG2(N), IC, OC, sh, sw, GN, GIC);
			else if (CAN_V2_O8P1) kGemmV2SK88O8P1_n2pow(stream, 3, GZ, oc_index, ic_index, X, IH, IW, deltaY, deltaW, deltaW_buf, FH, FW, LOG2(N), IC, OC, sh, sw, GN, GIC);
			else kGemmV2SK88_n2pow_LB3(stream, 3, GZ, oc_index, ic_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, LOG2(N), IC, OC, sh, sw, ph, pw, GN, GIC);
		}
		//======[Common]================================================================
		else if (CAN_V2_O4P1) kGemmV2SK88O4P1(stream, 3, GZ, oc_index, ic_index, X, IH, IW, deltaY, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, GN, GIC);
		else if (CAN_V2_O7P1) kGemmV2SK88O7P1(stream, 3, GZ, oc_index, ic_index, X, IH, IW, deltaY, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, GN, GIC);
		else if (CAN_V2_O8P1) kGemmV2SK88O8P1(stream, 3, GZ, oc_index, ic_index, X, IH, IW, deltaY, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, GN, GIC);
		else kGemmV2SK88(stream, 3, GZ, oc_index, ic_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GIC);
		dconv3d_dW_GemmV2SK_Branch(63, 63); return;
	}

	if ((GN > 63) && (GIC > 31)) {//[64, 32]
		if (IS_POWER2(N)) kGemmV2SK84_n2pow(stream, 3, GZ, oc_index, ic_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, LOG2(N), IC, OC, sh, sw, ph, pw, GN, GIC);
		else kGemmV2SK84(stream, 3, GZ, oc_index, ic_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GIC);
		dconv3d_dW_GemmV2SK_Branch(63, 31); return;
	}
	if ((GN > 31) && (GIC > 63)) {//[32, 64]
		if (IS_POWER2(N)) kGemmV2SK48_n2pow(stream, 3, GZ, oc_index, ic_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, LOG2(N), IC, OC, sh, sw, ph, pw, GN, GIC);
		else kGemmV2SK48(stream, 3, GZ, oc_index, ic_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GIC);
		dconv3d_dW_GemmV2SK_Branch(31, 63); return;
	}

	if ((GN > 31) && (GIC > 31)) {//[32, 32]
		if (IS_POWER2(N)) kGemmV2SK44_n2pow(stream, 3, GZ, oc_index, ic_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, LOG2(N), IC, OC, sh, sw, ph, pw, GN, GIC);
		else kGemmV2SK44(stream, 3, GZ, oc_index, ic_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GIC);
		dconv3d_dW_GemmV2SK_Branch(31, 31); return;
	}

	if ((GN > 63) && (GIC > 15)) {//[64, 16]
		if (IS_POWER2(N)) kGemmV2SK82_n2pow(stream, 3, GZ, oc_index, ic_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, LOG2(N), IC, OC, sh, sw, ph, pw, GN, GIC);
		else kGemmV2SK82(stream, 3, GZ, oc_index, ic_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GIC);
		dconv3d_dW_GemmV2SK_Branch(63, 15); return;
	}
	if ((GN > 15) && (GIC > 63)) {//[16, 64]
		if (IS_POWER2(N)) kGemmV2SK28_n2pow(stream, 3, GZ, oc_index, ic_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, LOG2(N), IC, OC, sh, sw, ph, pw, GN, GIC);
		else kGemmV2SK28(stream, 3, GZ, oc_index, ic_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GIC);
		dconv3d_dW_GemmV2SK_Branch(15, 63); return;
	}

	if ((GN > 31) && (GIC > 15)) {//[32, 16]
		if (IS_POWER2(N)) kGemmV2SK42_n2pow(stream, 3, GZ, oc_index, ic_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, LOG2(N), IC, OC, sh, sw, ph, pw, GN, GIC);
		else kGemmV2SK42(stream, 3, GZ, oc_index, ic_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GIC);
		dconv3d_dW_GemmV2SK_Branch(31, 15); return;
	}
	if ((GN > 15) && (GIC > 31)) {//[16, 32]
		if (IS_POWER2(N)) kGemmV2SK24_n2pow(stream, 3, GZ, oc_index, ic_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, LOG2(N), IC, OC, sh, sw, ph, pw, GN, GIC);
		else kGemmV2SK24(stream, 3, GZ, oc_index, ic_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GIC);
		dconv3d_dW_GemmV2SK_Branch(15, 31); return;
	}

	if (N > 7) {//N >= 8
		if ((GN > 15) && (GIC > 15)) {//[16, 16]
			kGemmV2SK22(stream, 3, GZ, oc_index, ic_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GIC);
			dconv3d_dW_GemmV2SK_Branch(15, 15); return;
		}

		if ((GN > 63) && (GIC > 7)) {//[64, 8]
			kGemmV2SK81(stream, 3, GZ, oc_index, ic_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GIC);
			dconv3d_dW_GemmV2SK_Branch(63, 7); return;
		}
		if ((GN > 31) && (GIC > 7)) {//[32, 8]
			kGemmV2SK41(stream, 3, GZ, oc_index, ic_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GIC);
			dconv3d_dW_GemmV2SK_Branch(31, 7); return;
		}
		if ((GN > 15) && (GIC > 7)) {//[16, 8]
			kGemmV2SK21(stream, 3, GZ, oc_index, ic_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GIC);
			dconv3d_dW_GemmV2SK_Branch(15, 7); return;
		}

		if ((GN > 7) && (GIC > 31)) {//[8, 32]
			kGemmV2SK14(stream, 3, GZ, oc_index, ic_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GIC);
			dconv3d_dW_GemmV2SK_Branch(7, 31); return;
		}
		if ((GN > 7) && (GIC > 15)) {//[8, 16]
			kGemmV2SK12(stream, 3, GZ, oc_index, ic_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GIC);
			dconv3d_dW_GemmV2SK_Branch(7, 15); return;
		}
		if ((GN > 7) && (GIC > 7)) {//[8, 8]
			kGemmV2SK11(stream, 3, GZ, oc_index, ic_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GIC);
			dconv3d_dW_GemmV2SK_Branch(7, 7); return;
		}
	}

	if (GN > 63) {//[64, 4]
		sGemmV2SK_8x2_1(stream, 3, GZ, oc_index, ic_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GIC);
		dconv3d_dW_GemmV2SK_Branch(63, 3); return;
	}
	if (GN > 31) {//[32, 4]
		sGemmV2SK_4x2_1(stream, 3, GZ, oc_index, ic_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GIC);
		dconv3d_dW_GemmV2SK_Branch(31, 3); return;
	}
	if (GN > 15) {//[16, 4]
		sGemmV2SK_2x2_1(stream, 3, GZ, oc_index, ic_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GIC);
		dconv3d_dW_GemmV2SK_Branch(15, 3); return;
	}
	if (GN > 7) {//[8, 4]
		kGemmV2SK21(stream, 2, GZ, oc_index, ic_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GIC);
		dconv3d_dW_GemmV2SK_Branch(7, 3); return;
	}
	
	if (GIC > 31) {//[4, 32]
		sGemmV2SK_1_4x2(stream, 3, GZ, oc_index, ic_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GIC);
		dconv3d_dW_GemmV2SK_Branch(3, 31); return;
	}
	if (GIC > 15) {//[4, 16]
		sGemmV2SK_1_2x2(stream, 3, GZ, oc_index, ic_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GIC);
		dconv3d_dW_GemmV2SK_Branch(3, 15); return;
	}
	if (GIC > 7) {//[4, 8]
		kGemmV2SK12(stream, 2, GZ, oc_index, ic_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GIC);
		dconv3d_dW_GemmV2SK_Branch(3, 7); return;
	}
	kGemmV2SK11(stream, 2, GZ, oc_index, ic_index, X, IH, IW, deltaY, OH, OW, deltaW, deltaW_buf, FH, FW, N, IC, OC, sh, sw, ph, pw, GN, GIC);

}

#endif


#ifndef DECONV_3D_DELTAW_GEMMSK_W1
#define DECONV_3D_DELTAW_GEMMSK_W1

#define __dconv3D_deltaW_GemmSK_W1(streams, index, length, GZ, X,IH,IW, deltaY, deltaW, deltaW_buf, IC,OC)\
	deconv3d_deltaW_GemmSK_W1(streams, index, length, GZ, X,IH,IW, deltaY, deltaW, deltaW_buf, IC,OC,\
		GN, GM, GK, GK_slice, 0, 0)

#define dconv3d_dW_GemmSK_W1_Branch(SIZE_Y, SIZE_X) {\
	int GNr = GN & (SIZE_Y), GMr = GM & (SIZE_X);\
	if(GNr && GMr){\
		int next_oc_index = (GN - GNr) + oc_index;\
		int next_j_index = (GM - GMr) + j_index;\
		deconv3d_deltaW_GemmSK_W1(streams, index, length, GZ, X,IH,IW, deltaY, deltaW, deltaW_buf, IC,OC,\
			GNr, GM, GK, GK_slice, next_oc_index, j_index);\
		deconv3d_deltaW_GemmSK_W1(streams, index, length, GZ, X,IH,IW, deltaY, deltaW, deltaW_buf, IC,OC,\
            GN, GMr, GK, GK_slice, oc_index, next_j_index);\
		deconv3d_deltaW_GemmSK_W1(streams, index, length, GZ, X,IH,IW, deltaY, deltaW, deltaW_buf, IC,OC,\
            GNr, GMr, GK, GK_slice, next_oc_index, next_j_index);}\
	else if(GNr){\
		int next_oc_index = (GN - GNr) + oc_index;\
		deconv3d_deltaW_GemmSK_W1(streams, index, length, GZ, X,IH,IW, deltaY, deltaW, deltaW_buf, IC,OC,\
			 GNr, GM, GK, GK_slice, next_oc_index, j_index);}\
	else if(GMr){\
		int next_j_index = (GM - GMr) + j_index;\
		deconv3d_deltaW_GemmSK_W1(streams, index, length, GZ, X,IH,IW, deltaY, deltaW, deltaW_buf, IC,OC,\
			 GN, GMr, GK, GK_slice, oc_index, next_j_index);}}

//GN%4 == 0, GN >= 4
//GM%4 == 0, GM >= 4
//GK%4 == 0, As: GN >= 16, We have: GK >= 16
void deconv3d_deltaW_GemmSK_W1(jlong *streams, int &index, int length, int GZ,
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ deltaY,
	float* __restrict__ deltaW,//for blockIdx.z == 0
	float* __restrict__ deltaW_buf,
	int IC, int OC,
	int GN, int GM, int GK, int GK_slice,
	int oc_index, int j_index)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)streams[index];
	index = (index + 1) % length;

	if ((GN > 127) && (GM > 127) && !(GK & 7)) {//[128, 128], GK % 8 == 0
		kGemmSK88W1(stream, 4, GZ, oc_index, j_index, X, IH, IW, deltaY, deltaW, deltaW_buf, IC, OC, GN, GM);
		dconv3d_dW_GemmSK_W1_Branch(127, 127); return;
	}
	if ((GN > 63) && (GM > 63)) {//[64, 64]
		kGemmSK88W1(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, deltaW, deltaW_buf, IC, OC, GN, GM);
		dconv3d_dW_GemmSK_W1_Branch(63, 63); return;
	}

	if ((GN > 63) && (GM > 31)) {//[64, 32]
		kGemmSK84W1(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, deltaW, deltaW_buf, IC, OC, GN, GM);
		dconv3d_dW_GemmSK_W1_Branch(63, 31); return;
	}
	if ((GN > 31) && (GM > 63)) {//[32, 64]
		kGemmSK48W1(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, deltaW, deltaW_buf, IC, OC, GN, GM);
		dconv3d_dW_GemmSK_W1_Branch(31, 63); return;
	}

	if ((GN > 31) && (GM > 31)) {//[32, 32]
		kGemmSK44W1(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, deltaW, deltaW_buf, IC, OC, GN, GM);
		dconv3d_dW_GemmSK_W1_Branch(31, 31); return;
	}

	if ((GN > 63) && (GM > 15)) {//[64, 16]
		kGemmSK82W1(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, deltaW, deltaW_buf, IC, OC, GN, GM);
		dconv3d_dW_GemmSK_W1_Branch(63, 15); return;
	}
	if ((GN > 15) && (GM > 63)) {//[16, 64]
		kGemmSK28W1(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, deltaW, deltaW_buf, IC, OC, GN, GM);
		dconv3d_dW_GemmSK_W1_Branch(15, 63); return;
	}

	if ((GN > 31) && (GM > 15)) {//[32, 16]
		kGemmSK42W1(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, deltaW, deltaW_buf, IC, OC, GN, GM);
		dconv3d_dW_GemmSK_W1_Branch(31, 15); return;
	}
	if ((GN > 15) && (GM > 31)) {//[16, 32]
		kGemmSK24W1(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, deltaW, deltaW_buf, IC, OC, GN, GM);
		dconv3d_dW_GemmSK_W1_Branch(15, 31); return;
	}

	if ((GN > 15) && (GM > 15)) {//[16, 16]
		kGemmSK22W1(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, deltaW, deltaW_buf, IC, OC, GN, GM);
		dconv3d_dW_GemmSK_W1_Branch(15, 15); return;
	}

	if (GM > 7) {//GM >= 8
		if (GN > 63) {//[64, 8]
			kGemmSK81W1(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, deltaW, deltaW_buf, IC, OC, GN, GM);
			dconv3d_dW_GemmSK_W1_Branch(63, 7); return;
		}
		if (GN > 31) {//[32, 8]
			kGemmSK41W1(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, deltaW, deltaW_buf, IC, OC, GN, GM);
			dconv3d_dW_GemmSK_W1_Branch(31, 7); return;
		}
		if (GN > 15) {//[16, 8]
			kGemmSK21W1(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, deltaW, deltaW_buf, IC, OC, GN, GM);
			dconv3d_dW_GemmSK_W1_Branch(15, 7); return;
		}
	}

	if (GN > 7) {//GN >= 8
		if (GM > 31) {//[8, 32]
			kGemmSK14W1(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, deltaW, deltaW_buf, IC, OC, GN, GM);
			dconv3d_dW_GemmSK_W1_Branch(7, 31); return;
		}
		if (GM > 15) {//[8, 16]
			kGemmSK12W1(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, deltaW, deltaW_buf, IC, OC, GN, GM);
			dconv3d_dW_GemmSK_W1_Branch(7, 15); return;
		}
		if (GM >  7) {//[8, 8]
			kGemmSK11W1(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, deltaW, deltaW_buf, IC, OC, GN, GM);
			dconv3d_dW_GemmSK_W1_Branch(7, 7); return;
		}
	}

	if (GN > 63) {//[63, 4]
		sGemmSK_8x2_1_W1(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, deltaW, deltaW_buf, IC, OC, GN, GM);
		dconv3d_dW_GemmSK_W1_Branch(63, 3); return;
	}
	if (GN > 31) {//[32, 4]
		sGemmSK_4x2_1_W1(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, deltaW, deltaW_buf, IC, OC, GN, GM);
		dconv3d_dW_GemmSK_W1_Branch(31, 3); return;
	}
	if (GN > 15) {//[16, 4]
		sGemmSK_2x2_1_W1(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, deltaW, deltaW_buf, IC, OC, GN, GM);
		dconv3d_dW_GemmSK_W1_Branch(15, 3); return;
	}
	if (GN >  7) {//[8, 4]
		kGemmSK21W1(stream, 2, GZ, oc_index, j_index, X, IH, IW, deltaY, deltaW, deltaW_buf, IC, OC, GN, GM);
		dconv3d_dW_GemmSK_W1_Branch(7, 3); return;
	}

	if (GM > 31) {//[4, 32]
		sGemmSK_1_4x2_W1(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, deltaW, deltaW_buf, IC, OC, GN, GM);
		dconv3d_dW_GemmSK_W1_Branch(3, 31); return;
	}
	if (GM > 15) {//[4, 16]
		sGemmSK_1_2x2_W1(stream, 3, GZ, oc_index, j_index, X, IH, IW, deltaY, deltaW, deltaW_buf, IC, OC, GN, GM);
		dconv3d_dW_GemmSK_W1_Branch(3, 15); return;
	}
	if (GM >  7) {//[4, 8]
		kGemmSK12W1(stream, 2, GZ, oc_index, j_index, X, IH, IW, deltaY, deltaW, deltaW_buf, IC, OC, GN, GM);
		dconv3d_dW_GemmSK_W1_Branch(3, 7); return;
	}
	kGemmSK11W1(stream, 2, GZ, oc_index, j_index, X, IH, IW, deltaY, deltaW, deltaW_buf, IC, OC, GN, GM);
}

#endif

#endif

#endif