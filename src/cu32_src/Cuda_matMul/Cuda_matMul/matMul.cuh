#pragma once 

#ifndef MAT_MUL_H
#define MAT_MUL_H

#include "matMul_kernel.cuh"
#include "matMul_uernel.cuh"
#include "matMul_sernel.cuh"


#ifndef MAT_MUL4X
#define MAT_MUL4X

#define mm4xBranch(SIZE_Y, SIZE_X) {\
	int N1 = (N & SIZE_Y), M1 = (M & SIZE_X);\
	if (N1 && M1) {\
		int N0 = N - N1, M0 = M - M1;\
		const float *A1 = &get(A, N0, 0, K), *B1 = &get(B, 0, M0, SB);\
		float *C01 = &get(C, 0, M0, SB);\
		float *C10 = &get(C, N0, 0, SB), *C11 = &get(C, N0, M0, SB);\
		matMul4x(streams, index, length, A , B1, C01, N0, M1, K, SB);\
		matMul4x(streams, index, length, A1, B , C10, N1, M0, K, SB);\
		matMul4x(streams, index, length, A1, B1, C11, N1, M1, K, SB);}\
	else if (N1){\
		int N0 = N - N1;\
		const float *A1 = &get(A, N0, 0, K);\
		float *C1 = &get(C, N0, 0, SB);\
		matMul4x(streams, index, length, A1, B, C1, N1, M, K, SB);}\
	else if (M1){\
		int M0 = M - M1;\
		const float *B1 = &get(B, 0, M0, SB);\
		float *C1 = &get(C, 0, M0, SB);\
		matMul4x(streams, index, length, A, B1, C1, N, M1, K, SB);}}

void matMul4x(jlong* streams, int index, int length,
	const float* A,
	const float* B,
	      float* C,
	int N, int M, int K, int SB)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)streams[index];
	index = (index + 1) % length;

	int size = N * M; 
	if (size <= 64) { knaive(1, stream, A, B, C, N, M, K, SB); return; }

	if ((N > 127) && (M > 127) & (K > 7)) {//[128, 128]
		if (!(K & 15)) u88_mgk(4, stream, A, B, C, N, M, K, SB);
		else k88(4, stream, A, B, C, N, M, K, SB); 
		mm4xBranch(127, 127); return; 
	} 
	if ((N > 63) && (M > 63)) {//[64, 64]
		if (!(K & 7)) u88_mgk(3, stream, A, B, C, N, M, K, SB);
		else k88(3, stream, A, B, C, N, M, K, SB);
		mm4xBranch(63, 63); return; 
	}

	if ((N > 63) && (M > 31)) {//[64, 32]
		if (!(K & 7)) u84_mgk(3, stream, A, B, C, N, M, K, SB);
		else k84(3, stream, A, B, C, N, M, K, SB); 
		mm4xBranch(63, 31); return; 
	}
	if ((N > 31) && (M > 63)) {//[32, 64]
		if (!(K & 7)) u48_mgk(3, stream, A, B, C, N, M, K, SB);
		else k48(3, stream, A, B, C, N, M, K, SB);
		mm4xBranch(31, 63); return; 
	}

	if ((N > 31) && (M > 31)) {//[32, 32]
		if (!(K & 7)) u44_mgk(3, stream, A, B, C, N, M, K, SB);
		else k44(3, stream, A, B, C, N, M, K, SB); 
		mm4xBranch(31, 31); return; 
	}

	if ((N > 63) && (M > 15)) { k82(3, stream, A, B, C, N, M, K, SB); mm4xBranch(63, 15); return; }//[64, 16]
	if ((N > 15) && (M > 63)) { k28(3, stream, A, B, C, N, M, K, SB); mm4xBranch(15, 63); return; }//[16, 64]

	if ((N > 31) && (M > 15)) { k42(3, stream, A, B, C, N, M, K, SB); mm4xBranch(31, 15); return; }//[32, 16]
	if ((N > 15) && (M > 31)) { k24(3, stream, A, B, C, N, M, K, SB); mm4xBranch(15, 31); return; }//[16, 32]

	if (K > 7) {//K > = 8
		if ((N > 15) && (M > 15)) { k22(3, stream, A, B, C, N, M, K, SB); mm4xBranch(15, 15); return; }//[16, 16]

		if ((N > 63) && (M > 7)) { k81(3, stream, A, B, C, N, M, K, SB); mm4xBranch(63, 7); return; }//[64, 8]
		if ((N > 31) && (M > 7)) { k41(3, stream, A, B, C, N, M, K, SB); mm4xBranch(31, 7); return; }//[32, 8]
		if ((N > 15) && (M > 7)) { k21(3, stream, A, B, C, N, M, K, SB); mm4xBranch(15, 7); return; }//[16, 8]
		if (N > 63) { s8x2_1(3, stream, A, B, C, N, M, K, SB); mm4xBranch(63, 3); return; }//[64, 4]
		if (N > 31) { s4x2_1(3, stream, A, B, C, N, M, K, SB); mm4xBranch(31, 3); return; }//[32, 4]
		if (N > 15) { s2x2_1(3, stream, A, B, C, N, M, K, SB); mm4xBranch(15, 3); return; }//[16, 4]

		if ((N > 7) && (M > 63)) { k18(3, stream, A, B, C, N, M, K, SB); mm4xBranch(7, 63); return; }//[8, 64]
		if ((N > 7) && (M > 31)) { k14(3, stream, A, B, C, N, M, K, SB); mm4xBranch(7, 31); return; }//[8, 32]
		if ((N > 7) && (M > 15)) { k12(3, stream, A, B, C, N, M, K, SB); mm4xBranch(7, 15); return; }//[8, 16]
	}
	
	if ((N > 7) && (M > 7)) { k22(2, stream, A, B, C, N, M, K, SB); mm4xBranch(7, 7); return; }//[8, 8]
	
	if (N > 31) { k81(2, stream, A, B, C, N, M, K, SB); mm4xBranch(31, 3); return; }//[32, 4]
	if (N > 15) { k41(2, stream, A, B, C, N, M, K, SB); mm4xBranch(15, 3); return; }//[16, 4]
	if (N >  7) { k21(2, stream, A, B, C, N, M, K, SB); mm4xBranch( 7, 3); return; }//[ 8, 4]

	if (M > 31) { k18(2, stream, A, B, C, N, M, K, SB); mm4xBranch(3, 31); return; }//[4, 32]
	if (M > 15) { k14(2, stream, A, B, C, N, M, K, SB); mm4xBranch(3, 15); return; }//[4, 16]
	if (M >  7) { k12(2, stream, A, B, C, N, M, K, SB); mm4xBranch(3,  7); return; }//[4, 8]

	k22(1, stream, A, B, C, N, M, K, SB);
}

#endif

#endif