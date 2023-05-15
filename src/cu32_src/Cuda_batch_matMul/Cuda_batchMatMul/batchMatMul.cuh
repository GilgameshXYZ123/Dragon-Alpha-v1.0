#pragma once

#ifndef BATCH_MATMUL_H
#define BATCH_MATMUL_H

#include "batchMatMul_kernel.cuh"
#include "batchMatMul_kernel_padding.cuh"
#include "batchMatMul_uernel.cuh"

//A[Batch,  N, AK] 
//B[Batch, BK,  M]
//C[Batch,  N,  M]
//(1) M % 4 == 0
//(2) N % 4 != 0
//(3) K = BK: AK % 4 == 0, BK % 4 != 0, AK >= BK, AK = (BK + 3) >> 2 << 2

#ifndef BATCH_MATMUL
#define BATCH_MATMUL

#define __batch_matMul(MOVE_A, MOVE_B, streams, index, length, A, B, C, Batch, N, CM, BK, AK)\
	batch_matMul<MOVE_A, MOVE_B>\
		(streams, index, length, A, B, C, Batch, N, CM, BK, AK, N, CM, 0, 0)

#define bmmBranch(SIZE_Y, SIZE_X)  {\
	int GNr = GN & (SIZE_Y), GMr = GM & (SIZE_X);\
	if(GNr && GMr){\
		int next_Yindex = (GN - GNr) + Yindex;\
		int next_Xindex = (GM - GMr) + Xindex;\
		batch_matMul<MOVE_A, MOVE_B>(streams,index,length, A, B, C, Batch,N,CM,BK,AK, GNr, GM , next_Yindex,      Xindex);\
		batch_matMul<MOVE_A, MOVE_B>(streams,index,length, A, B, C, Batch,N,CM,BK,AK, GN , GMr,      Yindex, next_Xindex);\
		batch_matMul<MOVE_A, MOVE_B>(streams,index,length, A, B, C, Batch,N,CM,BK,AK, GNr, GMr, next_Yindex, next_Xindex);}\
	else if(GNr){\
		int next_Yindex = (GN - GNr) + Yindex;\
		batch_matMul<MOVE_A, MOVE_B>(streams,index,length, A, B, C, Batch,N,CM,BK,AK, GNr, GM , next_Yindex,      Xindex);}\
	else if(GMr){\
		int next_Xindex = (GM - GMr) + Xindex;\
		batch_matMul<MOVE_A, MOVE_B>(streams,index,length, A, B, C, Batch,N,CM,BK,AK, GN , GMr,      Yindex, next_Xindex);}}


//[1] ((GN & 127) > 119) && ((GM & 127) > 119) => (GN & 127) && (GM & 127);
//[2] ((GN & 63) > 55) && ((GM & 63) > 55) => (GN & 63) && (GM & 63)
template<int MOVE_A, int MOVE_B>
void batch_matMul(jlong *streams, int &index, int length,
	float* __restrict__ A,
	float* __restrict__ B,
	float* __restrict__ C,
	int Batch, int N, int CM, int BK, int AK,
	int GN, int GM, 
	int Yindex, int Xindex)
{
	cudaStream_t stream = (cudaStream_t)streams[index];
	index = (index + 1) % length;

	if ((GN > 127) && (GM > 127) && (BK > 7)) {//[128, 128], BK >= 8
		bool flag0 = ((GN & 127) > 111) && ((GM & 127) > 111) && (GN < 1024) && (GM < 1024);
		if(flag0) { bmm_k88_p(stream, 4, Yindex, Xindex, A, B, C, Batch, N, CM, BK, AK, GN, GM); return; }

		float Q = QP_128(GN, GM), V = VP_128(GN, GM); 
		bool flag1 = (!(GN & 63) && !(GM & 63)) || (Q < 1.25f);
		bool flag2 = (V > 0.85f);

		if (flag1 || flag2) {
			if (!(BK & 15)) { bmm_u88_mk(stream, 4, Yindex, Xindex, A, B, C, Batch, N, CM, BK, AK, GN, GM); }
			else if (!(BK & 7)) bmm_k88_mk(stream, 4, Yindex, Xindex, A, B, C, Batch, N, CM, BK, AK, GN, GM);
			else bmm_k88(stream, 4, Yindex, Xindex, A, B, C, Batch, N, CM, BK, AK, GN, GM);
			bmmBranch(127, 127); return;
		}
		if (Q < 1.45f) { bmm_k88_p(stream, 4, Yindex, Xindex, A, B, C, Batch, N, CM, BK, AK, GN, GM); return; }
	}
	if ((GN > 95) && (GM > 95) && (BK > 7)) {//(96, 96)
		if (QP_128(GN, GM) < 1.55f) { bmm_k88_p(stream, 4, Yindex, Xindex, A, B, C, Batch, N, CM, BK, AK, GN, GM); return; }
	}
	
	if ((GN > 63) && (GM > 63) && (BK > 3)) {//[64, 64]
		bool flag0 = ((GN & 63) > 55) && ((GM & 63) > 55);
		if(flag0) { bmm_k88_p(stream, 3, Yindex, Xindex, A, B, C, Batch, N, CM, BK, AK, GN, GM); return; }

		float Q = QP_64(GN, GM), V = VP_64(GN, GM);
		bool flag1 = (!(GN & 31) && !(GM & 31)) || (Q < 1.25f);
		bool flag2 = (V > 0.85f);
		
		if (flag1 || flag2) { 
			if (!(BK & 7)) bmm_u88_mk(stream, 3, Yindex, Xindex, A, B, C, Batch, N, CM, BK, AK, GN, GM);
			else if (!(BK & 3)) bmm_k88_mk(stream, 3, Yindex, Xindex, A, B, C, Batch, N, CM, BK, AK, GN, GM);
			else bmm_k88(stream, 3, Yindex, Xindex, A, B, C, Batch, N, CM, BK, AK, GN, GM);
			bmmBranch(63, 63); return;
		}
		if(Q < 1.45f) { bmm_k88_p(stream, 3, Yindex, Xindex, A, B, C, Batch, N, CM, BK, AK, GN, GM); return; }
	}
	if ((GN > 47) && (GM > 47) && (BK > 3)) {//(48, 48)
		if (QP_64(GN, GM) < 1.55f) { bmm_k88_p(stream, 3, Yindex, Xindex, A, B, C, Batch, N, CM, BK, AK, GN, GM); return; }
	}

	if ((GN > 31) && (GM > 31)) {//[32, 32]
		if ((!(GN & 31) && !(GM & 31))) {
			if (!(BK & 7)) bmm_u44_mk(stream, 3, Yindex, Xindex, A, B, C, Batch, N, CM, BK, AK, GN, GM);
			else bmm_k44(stream, 3, Yindex, Xindex, A, B, C, Batch, N, CM, BK, AK, GN, GM);
			bmmBranch(31, 31); return;
		}
	}
	if ((GN > 23) && (GM > 23)) {//(24, 24)
		if (!(BK & 7)) bmm_u44_mk_p(stream, 3, Yindex, Xindex, A, B, C, Batch, N, CM, BK, AK, GN, GM);
		else bmm_k44_p(stream, 3, Yindex, Xindex, A, B, C, Batch, N, CM, BK, AK, GN, GM); 
		return;
	}

	if ((GN > 15) && (GM > 63)) {//[16, 64]
		bmm_k28(stream, 3, Yindex, Xindex, A, B, C, Batch, N, CM, BK, AK, GN, GM);
		bmmBranch(15, 63); return;
	}
	if ((GN > 63) && (GM > 15)) {//[64, 16]
		bmm_k82(stream, 3, Yindex, Xindex, A, B, C, Batch, N, CM, BK, AK, GN, GM);
		bmmBranch(63, 15); return;
	}
	if ((GN > 15) && (GM > 31)) {//[16, 32]
		bmm_k24(stream, 3, Yindex, Xindex, A, B, C, Batch, N, CM, BK, AK, GN, GM);
		bmmBranch(15, 31); return;
	}
	if ((GN > 31) && (GM > 15)) {//[32, 15]
		bmm_k42(stream, 3, Yindex, Xindex, A, B, C, Batch, N, CM, BK, AK, GN, GM);
		bmmBranch(31, 15); return;
	}

	//======[Small]==============================================================
	if ((GN > 15) && (GM > 15)) {//[8, 8]
		bmm_k22(stream, 3, Yindex, Xindex, A, B, C, Batch, N, CM, BK, AK, GN, GM);
		bmmBranch(15, 15); return;
	}
	if ((GN > 31) && (GM > 7)) {//[32, 8]
		bmm_k41(stream, 3, Yindex, Xindex, A, B, C, Batch, N, CM, BK, AK, GN, GM);
		bmmBranch(31, 7); return;
	}
	if ((GN > 7) && (GM > 31)) {//[8, 32]
		bmm_k14(stream, 3, Yindex, Xindex, A, B, C, Batch, N, CM, BK, AK, GN, GM);
		bmmBranch(7, 31); return;
	}
	bmm_k22_p(stream, 3, Yindex, Xindex, A, B, C, Batch, N, CM, BK, AK, GN, GM);
}

#endif


#ifndef BATCH_MATMUL_TEXTURE
#define BATCH_MATMUL_TEXTURE

#define __batch_matMul_tex(MOVE_A, MOVE_B, streams, index, length, A, texA, B, C, Batch, N, CM, BK, AK)\
	batch_matMul_texture<MOVE_A, MOVE_B>\
		(streams, index, length, A, texA, B, C, Batch, N, CM, BK, AK, N, CM, 0, 0)

template<int MOVE_A, int MOVE_B>
void batch_matMul_texture(jlong *streams, int &index, int length,
	float* __restrict__ A, cudaTextureObject_t texA,
	float* __restrict__ B,
	float* __restrict__ C,
	int Batch, int N, int CM, int BK, int AK,
	int GN, int GM,
	int Yindex, int Xindex)
{
	cudaStream_t stream = (cudaStream_t)streams[index];
	index = (index + 1) % length;

	if ((GN > 127) && (GM > 127) && (BK > 7)) {//[128, 128], BK >= 8
		bool flag0 = ((GN & 127) > 111) && ((GM & 127) > 111) && (GN < 1024) && (GM < 1024);
		if (flag0) { bmm_k88_ptex(stream, 4, Yindex, Xindex, texA, B, C, Batch, N, CM, BK, AK, GN, GM); return; }

		float Q = QP_128(GN, GM), V = VP_128(GN, GM);
		bool flag1 = (!(GN & 63) && !(GM & 63)) || (Q < 1.25f);
		bool flag2 = (V > 0.85f);

		if (flag1 || flag2) {
			if (!(BK & 15)) { bmm_u88_mk(stream, 4, Yindex, Xindex, A, B, C, Batch, N, CM, BK, AK, GN, GM); }
			else if (!(BK & 7)) bmm_k88_mk(stream, 4, Yindex, Xindex, A, B, C, Batch, N, CM, BK, AK, GN, GM);
			else bmm_k88(stream, 4, Yindex, Xindex, A, B, C, Batch, N, CM, BK, AK, GN, GM);
			bmmBranch(127, 127); return;
		}
		if (Q < 1.45f) { bmm_k88_ptex(stream, 4, Yindex, Xindex, texA, B, C, Batch, N, CM, BK, AK, GN, GM); return; }
	}
	if ((GN > 95) && (GM > 95) && (BK > 7)) {//(96, 96)
		if (QP_128(GN, GM) < 1.5f) { bmm_k88_ptex(stream, 4, Yindex, Xindex, texA, B, C, Batch, N, CM, BK, AK, GN, GM); return; }
	}

	if ((GN > 63) && (GM > 63) && (BK > 3)) {//[64, 64]
		bool flag0 = ((GN & 63) > 55) && ((GM & 63) > 55);
		if (flag0) { bmm_k88_ptex(stream, 3, Yindex, Xindex, texA, B, C, Batch, N, CM, BK, AK, GN, GM); return; }

		float Q = QP_64(GN, GM), V = VP_64(GN, GM);
		bool flag1 = (!(GN & 31) && !(GM & 31)) || (Q < 1.25f);
		bool flag2 = (V > 0.85f);

		if (flag1 || flag2) {
			if (!(BK & 7)) bmm_u88_mk(stream, 3, Yindex, Xindex, A, B, C, Batch, N, CM, BK, AK, GN, GM);
			else if (!(BK & 3)) bmm_k88_mk(stream, 3, Yindex, Xindex, A, B, C, Batch, N, CM, BK, AK, GN, GM);
			else bmm_k88(stream, 3, Yindex, Xindex, A, B, C, Batch, N, CM, BK, AK, GN, GM);
			bmmBranch(63, 63); return;
		}
		if (Q < 1.4f) { bmm_k88_ptex(stream, 3, Yindex, Xindex, texA, B, C, Batch, N, CM, BK, AK, GN, GM); return; }
	}
	if ((GN > 47) && (GM > 47) && (BK > 3)) {//(48, 48)
		if (QP_64(GN, GM) < 1.55f) { bmm_k88_ptex(stream, 3, Yindex, Xindex, texA, B, C, Batch, N, CM, BK, AK, GN, GM); return; }
	}

	if ((GN > 31) && (GM > 31)) {//[32, 32]
		if ((!(GN & 31) && !(GM & 31))) {
			if (!(BK & 7)) bmm_u44_mk(stream, 3, Yindex, Xindex, A, B, C, Batch, N, CM, BK, AK, GN, GM);
			else bmm_k44(stream, 3, Yindex, Xindex, A, B, C, Batch, N, CM, BK, AK, GN, GM);
			bmmBranch(31, 31); return;
		}
	}
	if ((GN > 23) && (GM > 23)) {//(24, 24)
		if (!(BK & 7)) bmm_u44_mk_p(stream, 3, Yindex, Xindex, A, B, C, Batch, N, CM, BK, AK, GN, GM);
		else bmm_k44_p(stream, 3, Yindex, Xindex, A, B, C, Batch, N, CM, BK, AK, GN, GM);
		return;
	}

	if ((GN > 15) && (GM > 63)) {//[16, 64]
		bmm_k28(stream, 3, Yindex, Xindex, A, B, C, Batch, N, CM, BK, AK, GN, GM);
		bmmBranch(15, 63); return;
	}
	if ((GN > 63) && (GM > 15)) {//[64, 16]
		bmm_k82(stream, 3, Yindex, Xindex, A, B, C, Batch, N, CM, BK, AK, GN, GM);
		bmmBranch(63, 15); return;
	}
	if ((GN > 15) && (GM > 31)) {//[16, 32]
		bmm_k24(stream, 3, Yindex, Xindex, A, B, C, Batch, N, CM, BK, AK, GN, GM);
		bmmBranch(15, 31); return;
	}
	if ((GN > 31) && (GM > 15)) {//[32, 15]
		bmm_k42(stream, 3, Yindex, Xindex, A, B, C, Batch, N, CM, BK, AK, GN, GM);
		bmmBranch(31, 15); return;
	}

	//======[Small]==============================================================
	if ((GN > 15) && (GM > 15)) {//[8, 8]
		bmm_k22(stream, 3, Yindex, Xindex, A, B, C, Batch, N, CM, BK, AK, GN, GM);
		bmmBranch(15, 15); return;
	}
	if ((GN > 31) && (GM > 7)) {//[32, 8]
		bmm_k41(stream, 3, Yindex, Xindex, A, B, C, Batch, N, CM, BK, AK, GN, GM);
		bmmBranch(31, 7); return;
	}
	if ((GN > 7) && (GM > 31)) {//[8, 32]
		bmm_k14(stream, 3, Yindex, Xindex, A, B, C, Batch, N, CM, BK, AK, GN, GM);
		bmmBranch(7, 31); return;
	}

	bmm_k22_p(stream, 3, Yindex, Xindex, A, B, C, Batch, N, CM, BK, AK, GN, GM);
	return;
}

#endif

#endif