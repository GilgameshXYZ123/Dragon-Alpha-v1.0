#pragma once

#ifndef BATCH_MATMUL_T2_H
#define BATCH_MATMUL_T2_H

#include "batchMatMulT2_kernel.cuh"
#include "batchMatMulT2_kernel_padding.cuh"
#include "batchMatMulT2_uernel.cuh"

//A[Batch,  N,  K] 
//B[Batch, BM,  K] logically-> B^T[Batch, K, M] 
//C[Batch,  N, CM]
//(1) K % 4 == 0
//(2) M = CM: BM % 4 != 0, CM % 4 == 0, CM >= BM, CM = (BM + 3) >> 2 << 2
//(3) N % 4 != 0

#ifndef BATCH_T2_MATMUL
#define BATCH_T2_MATMUL

#define __batch_matMulT2(MOVE_A, MOVE_B, streams, index, length, A, B, C, Batch, N, CM, BM, K)\
	batch_matMulT2<MOVE_A, MOVE_B>\
		(streams, index, length, A, B, C, Batch, N, CM, BM, K, N, CM, 0, 0)

#define bmmT2Branch(SIZE_Y, SIZE_X)  {\
	int GNr = GN & (SIZE_Y), GMr = GM & (SIZE_X);\
	if(GNr && GMr){\
		int next_Yindex = (GN - GNr) + Yindex;\
		int next_Xindex = (GM - GMr) + Xindex;\
		batch_matMulT2<MOVE_A, MOVE_B>(streams,index,length, A, B, C, Batch,N,CM,BM,K, GNr, GM , next_Yindex,      Xindex);\
		batch_matMulT2<MOVE_A, MOVE_B>(streams,index,length, A, B, C, Batch,N,CM,BM,K, GN , GMr,      Yindex, next_Xindex);\
		batch_matMulT2<MOVE_A, MOVE_B>(streams,index,length, A, B, C, Batch,N,CM,BM,K, GNr, GMr, next_Yindex, next_Xindex);}\
	else if(GNr){\
		int next_Yindex = (GN - GNr) + Yindex;\
		batch_matMulT2<MOVE_A, MOVE_B>(streams,index,length, A, B, C, Batch,N,CM,BM,K, GNr, GM , next_Yindex,      Xindex);}\
	else if(GMr){\
		int next_Xindex = (GM - GMr) + Xindex;\
		batch_matMulT2<MOVE_A, MOVE_B>(streams,index,length, A, B, C, Batch,N,CM,BM,K, GN , GMr,      Yindex, next_Xindex);}}

//[1] ((GN & 127) > 119) && ((GM & 127) > 119) => (GN & 127) && (GM & 127);
//[2] ((GN & 63) > 55) && ((GM & 63) > 55) => (GN & 63) && (GM & 63)
template<int MOVE_A, int MOVE_B>
void batch_matMulT2(jlong *streams, int &index, int length,
	float* __restrict__ A,
	float* __restrict__ B,
	float* __restrict__ C,
	int Batch, int N, int CM, int BM, int K,
	int GN, int GM,
	int Yindex, int Xindex)
{
	cudaStream_t stream = (cudaStream_t)streams[index];
	index = (index + 1) % length;

	if ((GN > 127) && (GM > 127) && !(K & 7)) {//[128, 128], K % 8 == 0
		bool flag0 = ((GN & 127) > 111) && ((GM & 127) > 111) && (GN < 1024) && (GM < 1024);
		if (flag0) { bmmT2_k88_p(stream, 3, Yindex, Xindex, A, B, C, Batch, N, CM, BM, K, GN, GM); return; }

		float Q = QP_128(GN, GM), V = VP_128(GN, GM);
		bool flag1 = (!(GN & 63) && !(GM & 63)) || (Q < 1.25f);
		bool flag2 = (V > 0.85f);

		if (flag1 || flag2) {
			if (!(K & 15) && !(BM & 3)) bmmT2_u88(stream, 4, Yindex, Xindex, A, B, C, Batch, N, CM, BM, K, GN, GM);
			else bmmT2_k88(stream, 4, Yindex, Xindex, A, B, C, Batch, N, CM, BM, K, GN, GM);
			bmmT2Branch(127, 127); return;
		}
		if (Q < 1.45f) { bmmT2_k88_p(stream, 3, Yindex, Xindex, A, B, C, Batch, N, CM, BM, K, GN, GM); return; }
	}
	if ((GN > 95) && (GM > 95)) {//(96, 96)
		if (QP_128(GN, GM) < 1.5f) { bmmT2_k88_p(stream, 3, Yindex, Xindex, A, B, C, Batch, N, CM, BM, K, GN, GM); return; }
	}

	if ((GN > 63) && (GM > 63)) {//[64, 64]
		bool flag0 = ((GN & 63) > 55) && ((GM & 63) > 55);
		if (flag0) { bmmT2_k88_p(stream, 3, Yindex, Xindex, A, B, C, Batch, N, CM, BM, K, GN, GM); return; }

		float Q = QP_64(GN, GM), V = VP_64(GN, GM);
		bool flag1 = (!(GN & 31) && !(GM & 31)) || (Q < 1.25f);
		bool flag2 = (V > 0.85f);

		if (flag1 || flag2) {
			if (!(K & 7) && !(BM & 3)) bmmT2_u88(stream, 3, Yindex, Xindex, A, B, C, Batch, N, CM, BM, K, GN, GM);
			else bmmT2_k88(stream, 3, Yindex, Xindex, A, B, C, Batch, N, CM, BM, K, GN, GM);
			bmmT2Branch(63, 63); return;
		}
		if (Q < 1.45f) { bmmT2_k88_p(stream, 3, Yindex, Xindex, A, B, C, Batch, N, CM, BM, K, GN, GM); return; }
	}
	if ((GN > 47) && (GM > 47)) {//(48, 48)
		if (QP_64(GN, GM) < 1.55f) { bmmT2_k88_p(stream, 3, Yindex, Xindex, A, B, C, Batch, N, CM, BM, K, GN, GM); return; }
	}

	if ((GN > 31) && (GM > 31)) {//[32, 32]
		if ((!(GN & 31) && !(GM & 31))) {
			if (!(K & 7)) bmmT2_u44(stream, 3, Yindex, Xindex, A, B, C, Batch, N, CM, BM, K, GN, GM);
			else bmmT2_k44(stream, 3, Yindex, Xindex, A, B, C, Batch, N, CM, BM, K, GN, GM);
			bmmT2Branch(31, 31); return;
		}
	}
	if ((GN > 23) && (GM > 23)) {//[24, 24]
		if (!(K & 7)) bmmT2_u44_p(stream, 3, Yindex, Xindex, A, B, C, Batch, N, CM, BM, K, GN, GM);
		else bmmT2_k44_p(stream, 3, Yindex, Xindex, A, B, C, Batch, N, CM, BM, K, GN, GM);
		return;
	}

	if ((GN > 15) && (GM > 63)) {//[16, 64]
		bmmT2_k28(stream, 3, Yindex, Xindex, A, B, C, Batch, N, CM, BM, K, GN, GM);
		bmmT2Branch(15, 63); return;
	}
	if ((GN > 63) && (GM > 15)) {//[64, 16]
		bmmT2_k82(stream, 3, Yindex, Xindex, A, B, C, Batch, N, CM, BM, K, GN, GM);
		bmmT2Branch(63, 15); return;
	}
	if ((GN > 15) && (GM > 31)) {//[16, 32]
		bmmT2_k24(stream, 3, Yindex, Xindex, A, B, C, Batch, N, CM, BM, K, GN, GM);
		bmmT2Branch(15, 31); return;
	}
	if ((GN > 31) && (GM > 15)) {//[32, 16]
		bmmT2_k42(stream, 3, Yindex, Xindex, A, B, C, Batch, N, CM, BM, K, GN, GM);
		bmmT2Branch(31, 15); return;
	}

	//======[Small]======================================================
	if ((GN > 15) && (GM > 15)) {//[16, 16]
		bmmT2_k22(stream, 3, Yindex, Xindex, A, B, C, Batch, N, CM, BM, K, GN, GM);
		bmmT2Branch(15, 15); return;
	}
	if ((GN > 31) && (GM > 7)) {//[32, 8]
		bmmT2_k41(stream, 3, Yindex, Xindex, A, B, C, Batch, N, CM, BM, K, GN, GM);
		bmmT2Branch(31, 7); return;
	}
	if ((GN > 7) && (GM > 31)) {//[8, 32]
		bmmT2_k14(stream, 3, Yindex, Xindex, A, B, C, Batch, N, CM, BM, K, GN, GM);
		bmmT2Branch(7, 31); return;
	}
	bmmT2_k22_p(stream, 3, Yindex, Xindex, A, B, C, Batch, N, CM, BM, K, GN, GM);
}

#endif


#ifndef BATCH_T2_MATMUL_TEXTURE
#define BATCH_T2_MATMUL_TEXTURE

#define __batch_matMulT2_tex(MOVE_A, MOVE_B, streams, index, length, A, texA, B, texB, C, Batch, N, CM, BM, K)\
	batch_matMulT2_texture<MOVE_A, MOVE_B>\
		(streams, index, length, A,texA, B,texB, C, Batch, N, CM, BM, K, N, CM, 0, 0)

template<int MOVE_A, int MOVE_B>
void batch_matMulT2_texture(jlong *streams, int &index, int length,
	float* __restrict__ A, cudaTextureObject_t texA,
	float* __restrict__ B, cudaTextureObject_t texB,
	float* __restrict__ C,
	int Batch, int N, int CM, int BM, int K,
	int GN, int GM,
	int Yindex, int Xindex)
{
	cudaStream_t stream = (cudaStream_t)streams[index];
	index = (index + 1) % length;

	if ((GN > 127) && (GM > 127) && !(K & 7)) {//[128, 128], K % 8 == 0
		bool flag0 = ((GN & 127) > 111) && ((GM & 127) > 111) && (GN < 1024) && (GM < 1024);
		if (flag0) { bmmT2_k88_ptex(stream, 3, Yindex, Xindex, texA, texB, C, Batch, N, CM, BM, K, GN, GM); return; }

		float Q = QP_128(GN, GM), V = VP_128(GN, GM);
		bool flag1 = (!(GN & 63) && !(GM & 63)) || (Q < 1.25f);
		bool flag2 = (V > 0.85f);

		if (flag1 || flag2) {
			if (!(K & 15) && !(BM & 3)) bmmT2_u88(stream, 4, Yindex, Xindex, A, B, C, Batch, N, CM, BM, K, GN, GM);
			else bmmT2_k88(stream, 4, Yindex, Xindex, A, B, C, Batch, N, CM, BM, K, GN, GM);
			bmmT2Branch(127, 127); return;
		}
		if (Q < 1.45f) { bmmT2_k88_ptex(stream, 3, Yindex, Xindex, texA, texB, C, Batch, N, CM, BM, K, GN, GM); return; }
	}
	if ((GN > 95) && (GM > 95)) {//(96, 96)
		if (QP_128(GN, GM) < 1.5f) { bmmT2_k88_ptex(stream, 3, Yindex, Xindex, texA, texB, C, Batch, N, CM, BM, K, GN, GM); return; }
	}

	if ((GN > 63) && (GM > 63)) {//[64, 64]
		bool flag0 = ((GN & 63) > 55) && ((GM & 63) > 55);
		if (flag0) { bmmT2_k88_ptex(stream, 3, Yindex, Xindex, texA, texB, C, Batch, N, CM, BM, K, GN, GM); return; }

		float Q = QP_64(GN, GM), V = VP_64(GN, GM);
		bool flag1 = (!(GN & 31) && !(GM & 31)) || (Q < 1.25f);
		bool flag2 = (V > 0.85f);

		if (flag1 || flag2) {
			if (!(K & 7) && !(BM & 3)) bmmT2_u88(stream, 3, Yindex, Xindex, A, B, C, Batch, N, CM, BM, K, GN, GM);
			else bmmT2_k88(stream, 3, Yindex, Xindex, A, B, C, Batch, N, CM, BM, K, GN, GM);
			bmmT2Branch(63, 63); return;
		}
		if (Q < 1.45f) { bmmT2_k88_ptex(stream, 3, Yindex, Xindex, texA, texB, C, Batch, N, CM, BM, K, GN, GM); return; }
	}
	if ((GN > 47) && (GM > 47)) {//(48, 48)
		if (QP_64(GN, GM) < 1.55f) { bmmT2_k88_ptex(stream, 3, Yindex, Xindex, texA, texB, C, Batch, N, CM, BM, K, GN, GM); return; }
	}

	if ((GN > 31) && (GM > 31)) {//[32, 32]
		if ((!(GN & 31) && !(GM & 31))) {
			if (!(K & 7)) bmmT2_u44(stream, 3, Yindex, Xindex, A, B, C, Batch, N, CM, BM, K, GN, GM);
			else bmmT2_k44(stream, 3, Yindex, Xindex, A, B, C, Batch, N, CM, BM, K, GN, GM);
			bmmT2Branch(31, 31); return;
		}
	}
	if ((GN > 23) && (GM > 23)) {//[24, 24]
		if (!(K & 7)) bmmT2_u44_p(stream, 3, Yindex, Xindex, A, B, C, Batch, N, CM, BM, K, GN, GM);
		else bmmT2_k44_p(stream, 3, Yindex, Xindex, A, B, C, Batch, N, CM, BM, K, GN, GM);
		return;
	}

	if ((GN > 15) && (GM > 63)) {//[16, 64]
		bmmT2_k28(stream, 3, Yindex, Xindex, A, B, C, Batch, N, CM, BM, K, GN, GM);
		bmmT2Branch(15, 63); return;
	}
	if ((GN > 63) && (GM > 15)) {//[64, 16]
		bmmT2_k82(stream, 3, Yindex, Xindex, A, B, C, Batch, N, CM, BM, K, GN, GM);
		bmmT2Branch(63, 15); return;
	}
	if ((GN > 15) && (GM > 31)) {//[16, 32]
		bmmT2_k24(stream, 3, Yindex, Xindex, A, B, C, Batch, N, CM, BM, K, GN, GM);
		bmmT2Branch(15, 31); return;
	}
	if ((GN > 31) && (GM > 15)) {//[32, 16]
		bmmT2_k42(stream, 3, Yindex, Xindex, A, B, C, Batch, N, CM, BM, K, GN, GM);
		bmmT2Branch(31, 15); return;
	}

	//======[Small]======================================================
	if ((GN > 15) && (GM > 15)) {//[16, 16]
		bmmT2_k22(stream, 3, Yindex, Xindex, A, B, C, Batch, N, CM, BM, K, GN, GM);
		bmmT2Branch(15, 15); return;
	}
	if ((GN > 31) && (GM > 7)) {//[32, 8]
		bmmT2_k41(stream, 3, Yindex, Xindex, A, B, C, Batch, N, CM, BM, K, GN, GM);
		bmmT2Branch(31, 7); return;
	}
	if ((GN > 7) && (GM > 31)) {//[8, 32]
		bmmT2_k14(stream, 3, Yindex, Xindex, A, B, C, Batch, N, CM, BM, K, GN, GM);
		bmmT2Branch(7, 31); return;
	}
	bmmT2_k22_p(stream, 3, Yindex, Xindex, A, B, C, Batch, N, CM, BM, K, GN, GM);
}

#endif

#endif