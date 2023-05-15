#pragma once

#ifndef POOL2D_H
#define POOL2D_H

#include "pool2D_max_kernel.cuh"
#include "pool2D_max_indexed_kernel.cuh"
#include "pool2D_avg_kernel.cuh"
#include "pool2D_avg_ip_kernel.cuh"


#ifndef POOL2D_MAX
#define POOL2D_MAX

#define __pool2D_max(streams,index,length, X,IH,IW, FH,FW, Y,OH,OW, N,IC, sh,sw,ph,pw)\
	pool2D_max(streams,index,length, X,IH,IW, FH,FW, Y,OH,OW, IC, sh, sw, ph, pw,\
		GET_GN(IC), GET_GM(N,OH,OW), 0,0)

#define poo2dMaxBranch(SIZE_Y, SIZE_X) {\
	int GNr = GN & (SIZE_Y), GMr = GM & (SIZE_X);\
	if(GNr && GMr){\
		int next_ic_index = (GN - GNr) + ic_index;\
		int next_j_index = (GM - GMr) + j_index;\
		pool2D_max(streams, index, length, X,IH,IW, FH,FW, Y,OH,OW, IC, sh,sw,ph,pw,\
			GNr, GM, next_ic_index, j_index);\
		pool2D_max(streams, index, length, X,IH,IW, FH,FW, Y,OH,OW, IC, sh,sw,ph,pw,\
            GN, GMr, ic_index, next_j_index);\
		pool2D_max(streams, index, length, X,IH,IW, FH,FW, Y,OH,OW, IC, sh,sw,ph,pw,\
            GNr, GMr, next_ic_index, next_j_index);}\
	else if(GNr){\
		int next_ic_index = (GN - GNr) + ic_index;\
		pool2D_max(streams, index, length, X,IH,IW, FH,FW, Y,OH,OW, IC, sh,sw,ph,pw,\
			 GNr, GM, next_ic_index, j_index);}\
	else if(GMr){\
		int next_j_index = (GM - GMr) + j_index;\
		pool2D_max(streams, index, length, X,IH,IW, FH,FW, Y,OH,OW, IC, sh,sw,ph,pw,\
			 GN, GMr, ic_index, next_j_index);}}



//IC = 128: Size = 0.125000, Time = 2.062000 msec, Performance = 65.091034 GFlop/s
//IC = 192: Size = 0.187500, Time = 3.146000 msec, Performance = 63.994465 GFlop/s
//IC = 224: Size = 0.218750, Time = 3.776000 msec, Performance = 62.203655 GFlop/s
//IC = 240: Size = 0.234375, Time = 4.322000 msec, Performance = 58.227261 GFlop/s
//IC = 244: Size = 0.238281, Time = 5.316000 msec, Performance = 48.128769 GFlop/s
//GN = IC;          GN >= 4, GN % 4 == 0
//GM = N * OH * OW; GM >= 4, GM % 4 == 0
void pool2D_max(jlong *streams, int &index, int length,
	const float* X, int IH, int IW,
	int FH, int FW,
	float*  Y, int OH, int OW,
	int IC,
	int sh, int sw, int ph, int pw,
	int GN, int GM,
	int ic_index, int j_index)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)(streams[index]);
	index = (index + 1) % length;

	if (GN > 127) {//[128, 1]
		kmax4(stream, 5, 0, X, IH, IW, FH, FW, Y, OH, OW, sh, sw, ph, pw, GN, GM, ic_index, j_index);
		poo2dMaxBranch(127, 0); return;
	}
	if (GN > 63) {//[64, 2]
		kmax4(stream, 4, 1, X, IH, IW, FH, FW, Y, OH, OW, sh, sw, ph, pw, GN, GM, ic_index, j_index);
		poo2dMaxBranch(63, 1); return;
	}
	if (GN > 31) {//[32, 4]
		kmax4(stream, 3, 2, X, IH, IW, FH, FW, Y, OH, OW, sh, sw, ph, pw, GN, GM, ic_index, j_index);
		poo2dMaxBranch(31, 3); return;
	}
	if (GN > 15 && GM > 7) {//[16, 8]
		kmax4(stream, 2, 3, X, IH, IW, FH, FW, Y, OH, OW, sh, sw, ph, pw, GN, GM, ic_index, j_index);
		poo2dMaxBranch(15, 7); return;
	}
	if (GN > 7 && GM > 7) {//[8, 8]
		kmax2(stream, 2, 3, X, IH, IW, FH, FW, Y, OH, OW, sh, sw, ph, pw, GN, GM, ic_index, j_index);
		poo2dMaxBranch(7, 7); return;
	}
	if (GN > 7) {//[8, 4]
		kmax1(stream, 3, 2, X, IH, IW, FH, FW, Y, OH, OW, sh, sw, ph, pw, GN, GM, ic_index, j_index);
		poo2dMaxBranch(7, 3); return;
	}
	if (GM > 7) {//[4, 8]
		kmax1(stream, 2, 3, X, IH, IW, FH, FW, Y, OH, OW, sh, sw, ph, pw, GN, GM, ic_index, j_index);
		poo2dMaxBranch(3, 7); return;
	}
	kmax1(stream, 2, 2, X, IH, IW, FH, FW, Y, OH, OW, sh, sw, ph, pw, GN, GM, ic_index, j_index);//[4, 4]
}

#endif


#ifndef POOL2D_MAX_INDEXED
#define POOL2D_MAX_INDEXED

#define __pool2D_max_indexed(streams,index,length, X,IH,IW, FH,FW, Y, Index, OH,OW, N,IC, sh,sw,ph,pw)\
	pool2D_max_indexed(streams,index,length, X,IH,IW, FH,FW, Y,Index,OH,OW, IC, sh, sw, ph, pw,\
		GET_GN(IC), GET_GM(N,OH,OW), 0,0)

#define poo2dMaxIndexedBranch(SIZE_Y, SIZE_X) {\
	int GNr = GN & (SIZE_Y), GMr = GM & (SIZE_X);\
	if(GNr && GMr){\
		int next_ic_index = (GN - GNr) + ic_index;\
		int next_j_index = (GM - GMr) + j_index;\
		pool2D_max_indexed(streams, index, length, X,IH,IW, FH,FW, Y,Index,OH,OW, IC, sh,sw,ph,pw,\
			GNr, GM, next_ic_index, j_index);\
		pool2D_max_indexed(streams, index, length, X,IH,IW, FH,FW, Y,Index,OH,OW, IC, sh,sw,ph,pw,\
            GN, GMr, ic_index, next_j_index);\
		pool2D_max_indexed(streams, index, length, X,IH,IW, FH,FW, Y,Index,OH,OW, IC, sh,sw,ph,pw,\
            GNr, GMr, next_ic_index, next_j_index);}\
	else if(GNr){\
		int next_ic_index = (GN - GNr) + ic_index;\
		pool2D_max_indexed(streams, index, length, X,IH,IW, FH,FW, Y,Index,OH,OW, IC, sh,sw,ph,pw,\
			 GNr, GM, next_ic_index, j_index);}\
	else if(GMr){\
		int next_j_index = (GM - GMr) + j_index;\
		pool2D_max_indexed(streams, index, length, X,IH,IW, FH,FW, Y,Index,OH,OW, IC, sh,sw,ph,pw,\
			 GN, GMr, ic_index, next_j_index);}}



//IC = 128: Size = 0.125000, Time = 2.062000 msec, Performance = 65.091034 GFlop/s
//IC = 192: Size = 0.187500, Time = 3.146000 msec, Performance = 63.994465 GFlop/s
//IC = 224: Size = 0.218750, Time = 3.776000 msec, Performance = 62.203655 GFlop/s
//IC = 240: Size = 0.234375, Time = 4.322000 msec, Performance = 58.227261 GFlop/s
//IC = 244: Size = 0.238281, Time = 5.316000 msec, Performance = 48.128769 GFlop/s
//GN = IC;          GN >= 4, GN % 4 == 0
//GM = N * OH * OW; GM >= 4, GM % 4 == 0
void pool2D_max_indexed(jlong *streams, int &index, int length,
	const float* X, int IH, int IW,
	int FH, int FW,
	float*  Y, int* Index, int OH, int OW,
	int IC,
	int sh, int sw, int ph, int pw,
	int GN, int GM,
	int ic_index, int j_index)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)(streams[index]);
	index = (index + 1) % length;

	if (GN > 127) {//[128, 1]
		kmaxIdx4(stream, 5, 0, X, IH, IW, FH, FW, Y, Index, OH, OW, sh, sw, ph, pw, GN, GM, ic_index, j_index);
		poo2dMaxIndexedBranch(127, 0); return;
	}
	if (GN > 63) {//[64, 2]
		kmaxIdx4(stream, 4, 1, X, IH, IW, FH, FW, Y, Index, OH, OW, sh, sw, ph, pw, GN, GM, ic_index, j_index);
		poo2dMaxIndexedBranch(63, 1); return;
	}
	if (GN > 31) {//[32, 4]
		kmaxIdx4(stream, 3, 2, X, IH, IW, FH, FW, Y, Index, OH, OW, sh, sw, ph, pw, GN, GM, ic_index, j_index);
		poo2dMaxIndexedBranch(31, 3); return;
	}
	if (GN > 15 && GM > 7) {//[16, 8]
		kmaxIdx4(stream, 2, 3, X, IH, IW, FH, FW, Y, Index, OH, OW, sh, sw, ph, pw, GN, GM, ic_index, j_index);
		poo2dMaxIndexedBranch(15, 7); return;
	}
	if (GN > 7 && GM > 7) {//[8, 8]
		kmaxIdx2(stream, 2, 3, X, IH, IW, FH, FW, Y, Index, OH, OW, sh, sw, ph, pw, GN, GM, ic_index, j_index);
		poo2dMaxIndexedBranch(7, 7); return;
	}
	if (GN > 7) {//[8, 4]
		kmaxIdx1(stream, 3, 2, X, IH, IW, FH, FW, Y, Index, OH, OW, sh, sw, ph, pw, GN, GM, ic_index, j_index);
		poo2dMaxIndexedBranch(7, 3); return;
	}
	if (GM > 7) {//[4, 8]
		kmaxIdx1(stream, 2, 3, X, IH, IW, FH, FW, Y, Index, OH, OW, sh, sw, ph, pw, GN, GM, ic_index, j_index);
		poo2dMaxIndexedBranch(3, 7); return;
	}
	kmaxIdx1(stream, 2, 2, X, IH, IW, FH, FW, Y, Index, OH, OW, sh, sw, ph, pw, GN, GM, ic_index, j_index);//[4, 4]
}

#endif


#ifndef POOL2D_AVG
#define POOL2D_AVG

#define __pool2D_avg(streams,index,length, X,IH,IW, FH,FW, Y,OH,OW, N,IC, sh,sw,ph,pw)\
	pool2D_avg(streams,index,length, X,IH,IW, FH,FW, Y,OH,OW, IC, sh, sw, ph, pw,\
		GET_GN(IC), GET_GM(N,OH,OW), 0,0)

#define poo2dAvgBranch(SIZE_Y, SIZE_X) {\
	int GNr = GN & (SIZE_Y), GMr = GM & (SIZE_X);\
	if(GNr && GMr){\
		int next_ic_index = (GN - GNr) + ic_index;\
		int next_j_index = (GM - GMr) + j_index;\
		pool2D_avg(streams, index, length, X,IH,IW, FH,FW, Y,OH,OW, IC, sh,sw,ph,pw,\
			GNr, GM, next_ic_index, j_index);\
		pool2D_avg(streams, index, length, X,IH,IW, FH,FW, Y,OH,OW, IC, sh,sw,ph,pw,\
            GN, GMr, ic_index, next_j_index);\
		pool2D_avg(streams, index, length, X,IH,IW, FH,FW, Y,OH,OW, IC, sh,sw,ph,pw,\
            GNr, GMr, next_ic_index, next_j_index);}\
	else if(GNr){\
		int next_ic_index = (GN - GNr) + ic_index;\
		pool2D_avg(streams, index, length, X,IH,IW, FH,FW, Y,OH,OW, IC, sh,sw,ph,pw,\
			 GNr, GM, next_ic_index, j_index);}\
	else if(GMr){\
		int next_j_index = (GM - GMr) + j_index;\
		pool2D_avg(streams, index, length, X,IH,IW, FH,FW, Y,OH,OW, IC, sh,sw,ph,pw,\
			 GN, GMr, ic_index, next_j_index);}}


//IC = 128: Size = 0.125000, Time = 2.062000 msec, Performance = 65.091034 GFlop/s
//IC = 192: Size = 0.187500, Time = 3.146000 msec, Performance = 63.994465 GFlop/s
//IC = 224: Size = 0.218750, Time = 3.776000 msec, Performance = 62.203655 GFlop/s
//IC = 240: Size = 0.234375, Time = 4.322000 msec, Performance = 58.227261 GFlop/s
//IC = 248: Size = 0.014205, Time = 0.306000 msec, Performance = 49.846378 GFlop/s
//IC = 252: Size = 0.014435, Time = 0.358000 msec, Performance = 43.293316 GFlop/s
//GN = IC;          GN >= 4, GN % 4 == 0
//GM = N * OH * OW; GM >= 4, GM % 4 == 0
void pool2D_avg(jlong *streams, int &index, int length,
	const float* X, int IH, int IW,
	int FH, int FW,
	float* Y, int OH, int OW,
	int IC,
	int sh, int sw, int ph, int pw,
	int GN, int GM,
	int ic_index, int j_index)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)(streams[index]);
	index = (index + 1) % length;

	if (GN > 127) {//[128, 1]
		kavg4(stream, 5, 0, X, IH, IW, FH, FW, Y, OH, OW, sh, sw, ph, pw, GN, GM, ic_index, j_index);
		poo2dAvgBranch(127, 0); return;
	}
	if (GN > 63) {//[64, 2]
		kavg4(stream, 4, 1, X, IH, IW, FH, FW, Y, OH, OW, sh, sw, ph, pw, GN, GM, ic_index, j_index);
		poo2dAvgBranch(63, 1); return;
	}
	if (GN > 31) {//[32, 4]
		kavg4(stream, 3, 2, X, IH, IW, FH, FW, Y, OH, OW, sh, sw, ph, pw, GN, GM, ic_index, j_index);
		poo2dAvgBranch(31, 3); return;
	}
	if (GN > 15 && GM > 7) {//[16, 8]
		kavg4(stream, 2, 3, X, IH, IW, FH, FW, Y, OH, OW, sh, sw, ph, pw, GN, GM, ic_index, j_index);
		poo2dAvgBranch(15, 7); return;
	}
	if (GN > 7 && GM > 7) {//[8, 8]
		kavg2(stream, 2, 3, X, IH, IW, FH, FW, Y, OH, OW, sh, sw, ph, pw, GN, GM, ic_index, j_index);
		poo2dAvgBranch(7, 7); return;
	}
	if (GN > 7) {//[8, 4]
		kavg1(stream, 3, 2, X, IH, IW, FH, FW, Y, OH, OW, sh, sw, ph, pw, GN, GM, ic_index, j_index);
		poo2dAvgBranch(7, 3); return;
	}
	if (GM > 7) {//[4, 8]
		kavg1(stream, 2, 3, X, IH, IW, FH, FW, Y, OH, OW, sh, sw, ph, pw, GN, GM, ic_index, j_index);
		poo2dAvgBranch(3, 7); return;
	}
	kavg1(stream, 2, 2, X, IH, IW, FH, FW, Y, OH, OW, sh, sw, ph, pw, GN, GM, ic_index, j_index);//[4, 4]
}

#endif


#ifndef POOL2D_AVG_IGNORE_PADDING
#define POOL2D_AVG_IGNORE_PADDING

#define __pool2D_avg_ip(streams,index,length, X,IH,IW, FH,FW, Y,OH,OW, N,IC, sh,sw,ph,pw)\
	pool2D_avg_ignore_padding(streams,index,length, X,IH,IW, FH,FW, Y,OH,OW, IC, sh, sw, ph, pw,\
		GET_GN(IC), GET_GM(N,OH,OW), 0,0)

#define poo2dAvgIPBranch(SIZE_Y, SIZE_X) {\
	int GNr = GN & (SIZE_Y), GMr = GM & (SIZE_X);\
	if(GNr && GMr){\
		int next_ic_index = (GN - GNr) + ic_index;\
		int next_j_index = (GM - GMr) + j_index;\
		pool2D_avg_ignore_padding(streams, index, length, X,IH,IW, FH,FW, Y,OH,OW, IC, sh,sw,ph,pw,\
			GNr, GM, next_ic_index, j_index);\
		pool2D_avg_ignore_padding(streams, index, length, X,IH,IW, FH,FW, Y,OH,OW, IC, sh,sw,ph,pw,\
            GN, GMr, ic_index, next_j_index);\
		pool2D_avg_ignore_padding(streams, index, length, X,IH,IW, FH,FW, Y,OH,OW, IC, sh,sw,ph,pw,\
            GNr, GMr, next_ic_index, next_j_index);}\
	else if(GNr){\
		int next_ic_index = (GN - GNr) + ic_index;\
		pool2D_avg_ignore_padding(streams, index, length, X,IH,IW, FH,FW, Y,OH,OW, IC, sh,sw,ph,pw,\
			 GNr, GM, next_ic_index, j_index);}\
	else if(GMr){\
		int next_j_index = (GM - GMr) + j_index;\
		pool2D_avg_ignore_padding(streams, index, length, X,IH,IW, FH,FW, Y,OH,OW, IC, sh,sw,ph,pw,\
			 GN, GMr, ic_index, next_j_index);}}


//IC = 128: Size = 0.125000, Time = 2.062000 msec, Performance = 65.091034 GFlop/s
//IC = 192: Size = 0.187500, Time = 3.146000 msec, Performance = 63.994465 GFlop/s
//IC = 224: Size = 0.218750, Time = 3.776000 msec, Performance = 62.203655 GFlop/s
//IC = 240: Size = 0.234375, Time = 4.322000 msec, Performance = 58.227261 GFlop/s
//IC = 248: Size = 0.014205, Time = 0.306000 msec, Performance = 49.846378 GFlop/s
//IC = 252: Size = 0.014435, Time = 0.358000 msec, Performance = 43.293316 GFlop/s
//GN = IC;          GN >= 4, GN % 4 == 0
//GM = N * OH * OW; GM >= 4, GM % 4 == 0
void pool2D_avg_ignore_padding(jlong *streams, int &index, int length,
	const float* X, int IH, int IW,
	int FH, int FW,
	float* Y, int OH, int OW,
	int IC,
	int sh, int sw, int ph, int pw,
	int GN, int GM,
	int ic_index, int j_index)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)(streams[index]);
	index = (index + 1) % length;

	if (GN > 127) {//[128, 1]
		kavg4_ip(stream, 5, 0, X, IH, IW, FH, FW, Y, OH, OW, sh, sw, ph, pw, GN, GM, ic_index, j_index);
		poo2dAvgIPBranch(127, 0); return;
	}
	if (GN > 63) {//[64, 2]
		kavg4_ip(stream, 4, 1, X, IH, IW, FH, FW, Y, OH, OW, sh, sw, ph, pw, GN, GM, ic_index, j_index);
		poo2dAvgIPBranch(63, 1); return;
	}
	if (GN > 31) {//[32, 4]
		kavg4_ip(stream, 3, 2, X, IH, IW, FH, FW, Y, OH, OW, sh, sw, ph, pw, GN, GM, ic_index, j_index);
		poo2dAvgIPBranch(31, 3); return;
	}
	if (GN > 15 && GM > 7) {//[16, 8]
		kavg4_ip(stream, 2, 3, X, IH, IW, FH, FW, Y, OH, OW, sh, sw, ph, pw, GN, GM, ic_index, j_index);
		poo2dAvgIPBranch(15, 7); return;
	}
	if (GN > 7 && GM > 7) {//[8, 8]
		kavg2_ip(stream, 2, 3, X, IH, IW, FH, FW, Y, OH, OW, sh, sw, ph, pw, GN, GM, ic_index, j_index);
		poo2dAvgIPBranch(7, 7); return;
	}
	if (GN > 7) {//[8, 4]
		kavg1_ip(stream, 3, 2, X, IH, IW, FH, FW, Y, OH, OW, sh, sw, ph, pw, GN, GM, ic_index, j_index);
		poo2dAvgIPBranch(7, 3); return;
	}
	if (GM > 7) {//[4, 8]
		kavg1_ip(stream, 2, 3, X, IH, IW, FH, FW, Y, OH, OW, sh, sw, ph, pw, GN, GM, ic_index, j_index);
		poo2dAvgIPBranch(3, 7); return;
	}
	kavg1_ip(stream, 2, 2, X, IH, IW, FH, FW, Y, OH, OW, sh, sw, ph, pw, GN, GM, ic_index, j_index);//[4, 4]
}

#endif

#endif 

