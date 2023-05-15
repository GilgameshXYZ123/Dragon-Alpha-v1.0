#pragma once

#ifndef MATMUL_T2_H
#define MATMUL_T2_H

#include "matMulT2_kernel.cuh"
#include "matMulT2_uernel.cuh"

#ifndef MAT_MUL4X_T2
#define MAT_MUL4X_T2

#define mm4xT2_Branch(SIZE_Y, SIZE_X) {\
	int N1 = (N & SIZE_Y), M1 = (M & SIZE_X);\
	if (N1 && M1) {\
		int N0 = N - N1, M0 = M - M1;\
		const float *A1 = &get(A, N0, 0, K), *B1 = &get(B, M0, 0, K);\
		float *C01 = &get(C, 0, M0, SC);\
		float *C10 = &get(C, N0, 0, SC), *C11 = &get(C, N0, M0, SC);\
		matMul4x_T2(streams, index, length, A , B1, C01, N0, M1, K, SC);\
		matMul4x_T2(streams, index, length, A1, B , C10, N1, M0, K, SC);\
		matMul4x_T2(streams, index, length, A1, B1, C11, N1, M1, K, SC);}\
	else if (N1){\
		int N0 = N - N1;\
		const float *A1 = &get(A, N0, 0, K);\
		float *C1 = &get(C, N0, 0, SC);\
		matMul4x_T2(streams, index, length, A1, B, C1, N1, M, K, SC);}\
	else if (M1){\
		int M0 = M - M1;\
		const float *B1 = &get(B, M0, 0, K);\
		float *C1 = &get(C, 0, M0, SC);\
		matMul4x_T2(streams, index, length, A, B1, C1, N, M1, K, SC);}}

//for the first stack of this function: SC = M
void matMul4x_T2(jlong *streams, int &index, int length,
	const float* A,
	const float* B,
	      float* C,
	int N, int M, int K, int SC)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)streams[index];
	index = (index + 1) % length;

	int size = N * M; if (size <= 256) { knaiveT2(2, stream, A, B, C, N, M, K, SC); return; }

	if ((N > 127) && (M > 127) && (K > 7)) {//[128, 128]
		if (!(K & 15)) u88T2_mgk(4, stream, A, B, C, N, M, K, SC);
		else k88T2(4, stream, A, B, C, N, M, K, SC); 
		mm4xT2_Branch(127, 127); return; 
	}
	if ((N > 63) && (M > 63)) {//[64, 64]
		if (!(K & 7)) u88T2_mgk(3, stream, A, B, C, N, M, K, SC);
		else k88T2(3, stream, A, B, C, N, M, K, SC);
		mm4xT2_Branch(63, 63); return; 
	}

	if ((N > 31) && (M > 63)) {//[32, 64]
		if (!(K & 7)) u48T2_mgk(3, stream, A, B, C, N, M, K, SC);
		else k48T2(3, stream, A, B, C, N, M, K, SC);
		mm4xT2_Branch(31, 63); return; 
	}
	if ((N > 63) && (M > 31)) {//[64, 32]
		if (!(K & 7)) u84T2_mgk(3, stream, A, B, C, N, M, K, SC);
		else k84T2(3, stream, A, B, C, N, M, K, SC); 
		mm4xT2_Branch(63, 31); return; 
	}
	
	if ((N > 31) && (M > 31)) { k44T2(3, stream, A, B, C, N, M, K, SC); mm4xT2_Branch(31, 31); return; }//[32, 32]

	if ((N > 63) && (M > 15)) { k82T2(3, stream, A, B, C, N, M, K, SC); mm4xT2_Branch(63, 15); return; }//[64, 16]
	if ((N > 15) && (M > 63)) { k28T2(3, stream, A, B, C, N, M, K, SC); mm4xT2_Branch(15, 63); return; }//[16, 64]

	if ((N > 31) && (M > 15)) { k42T2(3, stream, A, B, C, N, M, K, SC); mm4xT2_Branch(31, 15); return; }//[32, 16]
	if ((N > 15) && (M > 31)) { k24T2(3, stream, A, B, C, N, M, K, SC); mm4xT2_Branch(15, 31); return; }//[16, 32]

	if (K > 7) {
		if ((N > 15) && (M > 15)) { k22T2(3, stream, A, B, C, N, M, K, SC); mm4xT2_Branch(15, 15); return; }//[16, 16]
		if ((N > 63) && (M >  7)) { k81T2(3, stream, A, B, C, N, M, K, SC); mm4xT2_Branch(63, 7); return; }//[64, 8]
		if ((N >  7) && (M > 63)) { k18T2(3, stream, A, B, C, N, M, K, SC); mm4xT2_Branch(7, 63); return; }//[8, 64]
	}

	if ((N > 31) && (M > 15)) { k84T2(2, stream, A, B, C, N, M, K, SC); mm4xT2_Branch(31, 15); return; }
	if ((N > 15) && (M > 31)) { k48T2(2, stream, A, B, C, N, M, K, SC); mm4xT2_Branch(15, 31); return; }

	if ((N > 15) && (M >  7)) { k42T2(2, stream, A, B, C, N, M, K, SC); mm4xT2_Branch(15, 7); return; }
	if ((N >  7) && (M > 15)) { k24T2(2, stream, A, B, C, N, M, K, SC); mm4xT2_Branch(7, 15); return; }

	if ((N > 7) && (M > 7)) { k22T2(2, stream, A, B, C, N, M, K, SC); mm4xT2_Branch(7, 7); return; }
	if ((N > 31)) { k81T2(2, stream, A, B, C, N, M, K, SC); mm4xT2_Branch(31, 3); return; }
	if ((M > 31)) { k18T2(2, stream, A, B, C, N, M, K, SC); mm4xT2_Branch(3, 31); return; }

	k22T2(1, stream, A, B, C, N, M, K, SC);
}

#endif

#endif 
