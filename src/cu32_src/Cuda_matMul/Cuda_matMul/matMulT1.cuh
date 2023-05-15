#pragma once

#ifndef MATMUL_T1_H
#define MATMUL_T1_H

#include "matMulT1_kernel.cuh"
#include "matMulT1SK_kernel.cuh"


#ifndef MAT_MUL4X_T1
#define MAT_MUL4X_T1

#define mm4xT1_Branch(SIZE_Y, SIZE_X) {\
	int N1 = (N & SIZE_Y), M1 = (M & SIZE_X);\
	if (N1 && M1) {\
		int N0 = N - N1, M0 = M - M1;\
		const float *A1 = &get(A, 0, N0, SA), *B1 = &get(B, 0, M0, SB);\
		float *C01 = &get(C, 0, M0, SB);\
		float *C10 = &get(C, N0, 0, SB), *C11 = &get(C, N0, M0, SB);\
		matMul4x_T1(streams, index, length, A , B1, C01, N0, M1, K, SA, SB);\
		matMul4x_T1(streams, index, length, A1, B , C10, N1, M0, K, SA, SB);\
		matMul4x_T1(streams, index, length, A1, B1, C11, N1, M1, K, SA, SB);}\
	else if (N1){\
		int N0 = N - N1;\
		const float *A1 = &get(A, 0, N0, SA);\
		float *C1 = &get(C, N0, 0, SB);\
		matMul4x_T1(streams, index, length, A1, B, C1, N1, M, K, SA, SB);}\
	else if (M1){\
		int M0 = M - M1;\
		const float *B1 = &get(B, 0, M0, SB);\
		float *C1 = &get(C, 0, M0, SB);\
		matMul4x_T1(streams, index, length, A, B1, C1, N, M1, K, SA, SB);}}

//for the first stack of this function: SA = N, SB = M
void matMul4x_T1(jlong *streams, int &index, int length,
	const float* A,
	const float* B,
		  float* C,
	int N, int M, int K, int SA, int SB)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)streams[index];
	index = (index + 1) % length;

	int size = N * M; if (size <= 256) { knaiveT1(2, stream, A, B, C, N, M, K, SA, SB); return; }

	if ((N > 127) && (M > 127) && (K > 7)) { k88T1(4, stream, A, B, C, N, M, K, SA, SB); mm4xT1_Branch(127, 127); return; } //1540.60 GFlop/s
	if ((N > 63) && (M > 63)) { k88T1(3, stream, A, B, C, N, M, K, SA, SB); mm4xT1_Branch(63, 63); return; }//1303.04 GFlop/s

	if ((N > 63) && (M > 31)) { k84T1(3, stream, A, B, C, N, M, K, SA, SB); mm4xT1_Branch(63, 31); return; }//958.52 GFlop/s
	if ((N > 31) && (M > 63)) { k48T1(3, stream, A, B, C, N, M, K, SA, SB); mm4xT1_Branch(31, 63); return; }//1211.06 GFlop/s

	if ((N > 31) && (M > 31)) { k44T1(3, stream, A, B, C, N, M, K, SA, SB); mm4xT1_Branch(31, 31); return; }//916.65 GFlop/s

	if ((N > 63) && (M > 15)) { k82T1(3, stream, A, B, C, N, M, K, SA, SB); mm4xT1_Branch(63, 15); return; }//559.98 GFlop/s
	if ((N > 15) && (M > 63)) { k28T1(3, stream, A, B, C, N, M, K, SA, SB); mm4xT1_Branch(15, 63); return; }//772.41 GFlop/s, 

	if ((N > 31) && (M > 15)) { k42T1(3, stream, A, B, C, N, M, K, SA, SB); mm4xT1_Branch(31, 15); return; }//522.76 GFlop/s
	if ((N > 15) && (M > 31)) { k24T1(3, stream, A, B, C, N, M, K, SA, SB); mm4xT1_Branch(15, 31); return; }//646.47 GFlop/s
		
	if (K > 7) {
		if ((N > 63) && (M >  7)) { k81T1(3, stream, A, B, C, N, M, K, SA, SB); mm4xT1_Branch(63, 7); return; }
		if ((N > 15) && (M > 15)) { k22T1(3, stream, A, B, C, N, M, K, SA, SB); mm4xT1_Branch(15, 15); return; }//480.90 GFlop/s
		if ((N >  7) && (M > 63)) { k18T1(3, stream, A, B, C, N, M, K, SA, SB); mm4xT1_Branch(7, 63); return; }
	}

	if ((N > 31) && (M > 15)) { k84T1(2, stream, A, B, C, N, M, K, SA, SB); mm4xT1_Branch(31, 15); return; }//475.05 GFlop/s
	if ((N > 15) && (M > 31)) { k48T1(2, stream, A, B, C, N, M, K, SA, SB); mm4xT1_Branch(15, 31); return; }//539.34 GFlop/s

	if ((N > 15) && (M >  7)) { k42T1(2, stream, A, B, C, N, M, K, SA, SB); mm4xT1_Branch(15, 7); return; }
	if ((N >  7) && (M > 15)) { k24T1(2, stream, A, B, C, N, M, K, SA, SB); mm4xT1_Branch(7, 15); return; }
	
	if (N > 31) { k81T1(2, stream, A, B, C, N, M, K, SA, SB); mm4xT1_Branch(31, 3); return; }
	if ((N > 7) && (M > 7)) { k22T1(1, stream, A, B, C, N, M, K, SA, SB); mm4xT1_Branch(7, 7); return; }
	if (M > 31) { k18T1(2, stream, A, B, C, N, M, K, SA, SB); mm4xT1_Branch(3, 31); return; }

	k22T1(1, stream, A, B, C, N, M, K, SA, SB);//4 * 4 = 16
}

#endif


//Split K to improve parallism
#ifndef MAT_MUL4X_T1_SK
#define MAT_MUL4X_T1_SK

#define mm4xT1SK_Branch(SIZE_Y, SIZE_X) {\
	int N1 = (N & SIZE_Y), M1 = (M & SIZE_X);\
	if (N1 && M1) {\
		int N0 = N - N1, M0 = M - M1;\
		const float *A1 = &get(A, 0, N0, SA), *B1 = &get(B, 0, M0, SB);\
		float *C01    = &get(C   , 0, M0, SB);\
		float *Cbuf01 = &get(Cbuf, 0, M0, SB);\
		float *C10    = &get(C   , N0, 0, SB), *C11    = &get(C   , N0, M0, SB);\
		float *Cbuf10 = &get(Cbuf, N0, 0, SB), *Cbuf11 = &get(Cbuf, N0, M0, SB);\
		matMul4x_T1_SK(streams, index, length, GZ, A , B1, C01, Cbuf01, N0, M1, K, K_slice, SA, SB);\
		matMul4x_T1_SK(streams, index, length, GZ, A1, B , C10, Cbuf10, N1, M0, K, K_slice, SA, SB);\
		matMul4x_T1_SK(streams, index, length, GZ, A1, B1, C11, Cbuf11, N1, M1, K, K_slice, SA, SB);}\
	else if (N1){\
		int N0 = N - N1;\
		const float *A1 = &get(A, 0, N0, SA);\
		float *C1    = &get(C   , N0, 0, SB);\
		float *Cbuf1 = &get(Cbuf, N0, 0, SB);\
		matMul4x_T1_SK(streams, index, length, GZ, A1, B, C1, Cbuf1, N1, M, K, K_slice, SA, SB);}\
	else if (M1){\
		int M0 = M - M1;\
		const float *B1 = &get(B, 0, M0, SB);\
		float *C1    = &get(C   , 0, M0, SB);\
		float *Cbuf1 = &get(Cbuf, 0, M0, SB);\
		matMul4x_T1_SK(streams, index, length, GZ, A, B1, C1, Cbuf1, N, M1, K, K_slice, SA, SB);}}

//K >= 512
//for the first stack of this function: SA = N, SB = M
void matMul4x_T1_SK(jlong *streams, int &index, int length, int GZ,
	const float* A,
	const float* B,
	float* C, float * Cbuf,
	int N, int M, int K, int K_slice,
	int SA, int SB)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)streams[index];
	index = (index + 1) % length;

	if ((N > 127) && (M > 127)) { 
		if (!(K & 7)) k88T1SK_mgk(4, GZ, stream, A, B, C, Cbuf, N, M, K, SA, SB);//K % 8 == 0
		else k88T1SK(4, GZ, stream, A, B, C, Cbuf, N, M, K, SA, SB);
		mm4xT1SK_Branch(127, 127); return; 
	}
	if ((N > 63) && (M > 63)) { 
		if (!(K & 3)) k88T1SK_mgk(3, GZ, stream, A, B, C, Cbuf, N, M, K, SA, SB);//K % 4 == 0
		else k88T1SK(3, GZ, stream, A, B, C, Cbuf, N, M, K, SA, SB);
		mm4xT1SK_Branch(63, 63); return; 
	}
	
	if ((N > 63) && (M > 31)) { k84T1SK(3, GZ, stream, A, B, C, Cbuf, N, M, K, SA, SB); mm4xT1SK_Branch(63, 31); return; }
	if ((N > 31) && (M > 63)) { k48T1SK(3, GZ, stream, A, B, C, Cbuf, N, M, K, SA, SB); mm4xT1SK_Branch(31, 63); return; }

	if ((N > 31) && (M > 31)) { k44T1SK(3, GZ, stream, A, B, C, Cbuf, N, M, K, SA, SB); mm4xT1SK_Branch(31, 31); return; }

	if ((N > 63) && (M > 15)) { k82T1SK(3, GZ, stream, A, B, C, Cbuf, N, M, K, SA, SB); mm4xT1SK_Branch(63, 15); return; }
	if ((N > 15) && (M > 63)) { k28T1SK(3, GZ, stream, A, B, C, Cbuf, N, M, K, SA, SB); mm4xT1SK_Branch(15, 63); return; }

	if ((N > 31) && (M > 15)) { k42T1SK(3, GZ, stream, A, B, C, Cbuf, N, M, K, SA, SB); mm4xT1SK_Branch(31, 15); return; }
	if ((N > 15) && (M > 31)) { k24T1SK(3, GZ, stream, A, B, C, Cbuf, N, M, K, SA, SB); mm4xT1SK_Branch(15, 31); return; }

	if ((N > 15) && (M > 15)) { k22T1SK(3, GZ, stream, A, B, C, Cbuf, N, M, K, SA, SB); mm4xT1SK_Branch(15, 15); return; }
	if ((N > 15) && (M >  7)) { k21T1SK(3, GZ, stream, A, B, C, Cbuf, N, M, K, SA, SB); mm4xT1SK_Branch(15, 7); return; }
	if ((N >  7) && (M > 15)) { k12T1SK(3, GZ, stream, A, B, C, Cbuf, N, M, K, SA, SB); mm4xT1SK_Branch(7, 15); return; }
	if ((N >  7) && (M >  7)) { k11T1SK(3, GZ, stream, A, B, C, Cbuf, N, M, K, SA, SB); mm4xT1SK_Branch(7, 7); return; }

	if (N > 7) { k21T1SK(2, GZ, stream, A, B, C, Cbuf, N, M, K, SA, SB); mm4xT1SK_Branch(7, 3); return; }
	if (M > 7) { k12T1SK(2, GZ, stream, A, B, C, Cbuf, N, M, K, SA, SB); mm4xT1SK_Branch(3, 7); return; }
	k11T1SK(2, GZ, stream, A, B, C, Cbuf, N, M, K, SA, SB);
}

#endif

#endif