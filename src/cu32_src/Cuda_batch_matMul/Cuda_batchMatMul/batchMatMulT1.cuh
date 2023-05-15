#pragma once

#ifndef BATCH_MATMUL_T1_H
#define BATCH_MATMUL_T1_H

#include "batchMatMulT1_kernel.cuh"
#include "batchMatMulT1_kernel_padding.cuh"

//A[Batch,  K, AN] logically-> A^T[Batch, AN, K]
//B[Batch,  K,  M]
//C[Batch, CN,  M]
//(1) N = CN: AN % 4 == 0, CN % 4 != 0, AN >= CN, AN = (CN + 3) >> 2 << 2
//(2) M % 4 == 0
//(3) K % 4 != 0

#ifndef BATCH_MATMUL_T1
#define BATCH_MATMUL_T1

#define __batch_matMulT1(MOVE_A, MOVE_B, streams, index, length, A, B, C, Batch, CN, AN, CM, K)\
	batch_matMulT1<MOVE_A, MOVE_B>\
		(streams, index, length, A, B, C, Batch, CN, AN, CM, K, CN, CM, 0, 0)

#define bmmT1Branch(SIZE_Y, SIZE_X)  {\
	int GNr = GN & (SIZE_Y), GMr = GM & (SIZE_X);\
	if(GNr && GMr){\
		int next_Yindex = (GN - GNr) + Yindex;\
		int next_Xindex = (GM - GMr) + Xindex;\
		batch_matMulT1<MOVE_A, MOVE_B>(streams,index,length, A, B, C, Batch, CN,AN,CM,K, GNr, GM , next_Yindex,      Xindex);\
		batch_matMulT1<MOVE_A, MOVE_B>(streams,index,length, A, B, C, Batch, CN,AN,CM,K, GN , GMr,      Yindex, next_Xindex);\
		batch_matMulT1<MOVE_A, MOVE_B>(streams,index,length, A, B, C, Batch, CN,AN,CM,K, GNr, GMr, next_Yindex, next_Xindex);}\
	else if(GNr){\
		int next_Yindex = (GN - GNr) + Yindex;\
		batch_matMulT1<MOVE_A, MOVE_B>(streams,index,length, A, B, C, Batch, CN,AN,CM,K, GNr, GM , next_Yindex,      Xindex);}\
	else if(GMr){\
		int next_Xindex = (GM - GMr) + Xindex;\
		batch_matMulT1<MOVE_A, MOVE_B>(streams,index,length, A, B, C, Batch, CN,AN,CM,K, GN , GMr,      Yindex, next_Xindex);}}


template<int MOVE_A, int MOVE_B>
void batch_matMulT1(jlong *streams, int &index, int length,
	float* __restrict__ A,
	float* __restrict__ B,
	float* __restrict__ C,
	int Batch, int CN, int AN, int CM, int K,
	int GN, int GM,
	int Yindex, int Xindex)
{
	cudaStream_t stream = (cudaStream_t)streams[index];
	index = (index + 1) % length;

	int GN_GM = GN * GM;
	if (K > 7) {//STEP = 8
		int GNp = ((GN + 127) >> 7) << 7;
		int GMp = ((GM + 127) >> 7) << 7;
		int GNp_GMp = GNp * GMp;

		if ((GN > 127) && (GM > 127)) {
			if (!(GN & 63) && !(GM & 63)) {
				if (GN_GM >= 0.7f * GNp_GMp) {
					if (!(K & 7)) bmmT1_k88_mk(stream, 4, Yindex, Xindex, A, B, C, Batch, CN, AN, CM, K, GN, GM);
					else bmmT1_k88(stream, 4, Yindex, Xindex, A, B, C, Batch, CN, AN, CM, K, GN, GM);
					bmmT1Branch(127, 127); return;
				}
				if ((GN & 127) && (GM & 127)) {
					if (!(K & 3)) bmmT1_k88_mk(stream, 3, Yindex, Xindex, A, B, C, Batch, CN, AN, CM, K, GN, GM);
					else bmmT1_k88(stream, 3, Yindex, Xindex, A, B, C, Batch, CN, AN, CM, K, GN, GM);
					return;
				}
			}
		}
		if ((GN > 95) && (GM > 95) && (GN_GM >= 0.6f * GNp_GMp)) {//128 * 0.75 = 96
			bmmT1_k88_p(stream, 4, Yindex, Xindex, A, B, C, Batch, CN, AN, CM, K, GN, GM);
			return;
		}
	}

	if (K > 3) {//STEP = 4
		if ((GN > 63) && (GM > 63)) {//8*8 = 64
			int GNp = ((GN + 63) >> 6) << 6;
			int GMp = ((GM + 63) >> 6) << 6;
			int GNp_GMp = GNp * GMp;

			if ((!(GN & 31) && !(GM & 31)) && (GN_GM >= 0.7f * GNp_GMp)) {
				if (!(K & 3)) bmmT1_k88_mk(stream, 3, Yindex, Xindex, A, B, C, Batch, CN, AN, CM, K, GN, GM);
				else bmmT1_k88(stream, 3, Yindex, Xindex, A, B, C, Batch, CN, AN, CM, K, GN, GM);
				bmmT1Branch(63, 63); return;
			}
			if (GN_GM >= 0.6f * GNp_GMp) {
				bmmT1_k88_p(stream, 3, Yindex, Xindex, A, B, C, Batch, CN, AN, CM, K, GN, GM);
				return;//padding GN and GM both
			}
		}
		if ((GN > 47) && (GM > 47)) {//64 * 0.75 = 48
			bmmT1_k88_p(stream, 3, Yindex, Xindex, A, B, C, Batch, CN, AN, CM, K, GN, GM);
			return;
		}
	}

	if ((GN > 31) && (GM > 31)) {//4*8 = 32
		int GNp = ((GN + 31) >> 5) << 5;
		int GMp = ((GM + 31) >> 5) << 5;
		int GNp_GMp = GNp * GMp;

		if ((!(GN & 15) && !(GM & 15)) && (GN_GM >= 0.7f * GNp_GMp)) {
			bmmT1_k44(stream, 3, Yindex, Xindex, A, B, C, Batch, CN, AN, CM, K, GN, GM);
			bmmT1Branch(31, 31); return;
		}
		if (GN_GM >= 0.6f * GNp_GMp) {
			bmmT1_k44_p(stream, 3, Yindex, Xindex, A, B, C, Batch, CN, AN, CM, K, GN, GM);
			return;//padding GN and GM both
		}
	}
	if ((GN > 23) && (GM > 23)) {//32 * 0.75 = 24
		bmmT1_k44_p(stream, 3, Yindex, Xindex, A, B, C, Batch, CN, AN, CM, K, GN, GM);
		return;//padding GN and GM both
	}

	if ((GN > 15) && (GM > 63)) {//2*8 = 16, 8*8 = 64
		bmmT1_k28(stream, 3, Yindex, Xindex, A, B, C, Batch, CN, AN, CM, K, GN, GM);
		bmmT1Branch(15, 63); return;
	}
	if ((GN > 63) && (GM > 15)) {//8*8 = 64, 2*8 = 16
		bmmT1_k82(stream, 3, Yindex, Xindex, A, B, C, Batch, CN, AN, CM, K, GN, GM);
		bmmT1Branch(63, 15); return;
	}
	if ((GN > 15) && (GM > 31)) {//2*8 = 16, 4*8 = 32
		bmmT1_k24(stream, 3, Yindex, Xindex, A, B, C, Batch, CN, AN, CM, K, GN, GM);
		bmmT1Branch(15, 31); return;
	}
	if ((GN > 31) && (GM > 15)) {//4*8 = 32, 2*8 = 16
		bmmT1_k42(stream, 3, Yindex, Xindex, A, B, C, Batch, CN, AN, CM, K, GN, GM);
		bmmT1Branch(31, 15); return;
	}
	
	if ((GN > 15) && (GM > 15)) {//2*8 = 16
		bmmT1_k22(stream, 3, Yindex, Xindex, A, B, C, Batch, CN, AN, CM, K, GN, GM);
		bmmT1Branch(15, 15); return;
	}
	if ((GN > 31) && (GM > 7)) {//4*8 = 32, 1*8 = 8
		bmmT1_k41(stream, 3, Yindex, Xindex, A, B, C, Batch, CN, AN, CM, K, GN, GM);
		bmmT1Branch(31, 7); return;
	}
	if ((GN > 7) && (GM > 31)) {//1*8 = 8, 4*8 = 32
		bmmT1_k14(stream, 3, Yindex, Xindex, A, B, C, Batch, CN, AN, CM, K, GN, GM);
		bmmT1Branch(7, 31); return;
	}

	bmmT1_k22_p(stream, 3, Yindex, Xindex, A, B, C, Batch, CN, AN, CM, K, GN, GM);
}

#endif

#endif