#pragma once

#ifndef UPOOL2D_H
#define UPOOL2D_H

#include "upool2D_max_kernel.cuh"
#include "upool2D_max_indexed_kernel.cuh"

#include "upool2D_avg_kernel.cuh"
#include "upool2D_avg_kernel_tiled.cuh"
#include "upool2D_avg_ip_kernel.cuh"
#include "upool2D_avg_ip_kernel_tiled.cuh"

#ifndef UNPOOL2D_MAX
#define UNPOOL2D_MAX

#define __upool2D_max(streams, index, length, deltaY, Y, OH, OW, FH, FW, deltaX, X, IH, IW, N, IC, sh, sw, ph, pw)\
	upool2d_max(streams,index,length, deltaY,Y,OH,OW, FH,FW, deltaX,X,IH,IW, IC, sh,sw,ph,pw,\
		GET_GN(IC), GET_GM(N, IH, IW), 0, 0)

#define upool2DMaxBranch(SIZE_Y, SIZE_X) {\
	int GNr = GN & SIZE_Y, GMr = GM & SIZE_X;\
	if(GNr && GMr){\
		int next_ic_index = (GN - GNr) + ic_index;\
		int next_j_index = (GM - GMr) + j_index;\
		upool2d_max(streams,index,length, deltaY,Y,OH,OW, FH,FW, deltaX,X,IH,IW, IC, sh,sw,ph,pw,\
			GNr, GM, next_ic_index, j_index);\
		upool2d_max(streams,index,length, deltaY,Y,OH,OW, FH,FW, deltaX,X,IH,IW, IC, sh,sw,ph,pw,\
            GN, GMr, ic_index, next_j_index);\
		upool2d_max(streams,index,length, deltaY,Y,OH,OW, FH,FW, deltaX,X,IH,IW, IC, sh,sw,ph,pw,\
            GNr, GMr, next_ic_index, next_j_index);}\
	else if(GNr){\
		int next_ic_index = (GN - GNr) + ic_index;\
		upool2d_max(streams,index,length, deltaY,Y,OH,OW, FH,FW, deltaX,X,IH,IW, IC, sh,sw,ph,pw,\
			 GNr, GM, next_ic_index, j_index);}\
	else if(GMr){\
		int next_j_index = (GM - GMr) + j_index;\
		upool2d_max(streams,index,length, deltaY,Y,OH,OW, FH,FW, deltaX,X,IH,IW, IC, sh,sw,ph,pw,\
			 GN, GMr, ic_index, next_j_index);}}

//(1) FH * FW >=2
//(2) GN % 4==0, GN >= 4
//(3) GM % 4==0, GM >= 4
//(4) GK = FH * FW >= 2
//performance=======================================================
//[OH, OW] = 32, [FH, FW] = 8, [[N] = 8, [sh, sw] = 4, [ph, pw] = 2
//IC = 128: Size = 1.000000, Time = 1.968000 msec, Peformance = 545.600464 GFlop/s
//IC = 192: Size = 1.500000, Time = 2.932000 msec, Peformance = 549.322205 GFlop/s
//IC = 224: Size = 1.750000, Time = 3.456000 msec, Peformance = 543.705994 GFlop/s
//IC = 240: Size = 1.875000, Time = 3.774000 msec, Peformance = 533.456726 GFlop/s
//IC = 248: Size = 1.937500, Time = 4.068000 msec, Peformance = 511.399841 GFlop/s
//IC = 252: Size = 1.937500, Time = 4.054000 msec, Peformance = 513.165894 GFlop/s
void upool2d_max(jlong *streams, int &index, int length,
	const float*  __restrict__ deltaY,
	const float*  __restrict__ Y, int OH, int OW,
	int FH, int FW,
	float * __restrict__ deltaX,
	const float* __restrict__ X, int IH, int IW,
	int IC,
	int sh, int sw, int ph, int pw,
	int GN, int GM,
	int ic_index, int j_index)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)streams[index];
	index = (index + 1) % length;

	if (GN > 127) {//[128, 2]
		kmax81(stream, 4, 1, ic_index, j_index, deltaY, Y, OH, OW, FH, FW, deltaX, X, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		upool2DMaxBranch(127, 1); return;
	}
	if ((GN > 63) && (GM > 7)) {//[64, 8]
		kmax81(stream, 3, 3, ic_index, j_index, deltaY, Y, OH, OW, FH, FW, deltaX, X, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		upool2DMaxBranch(63, 7); return;
	}
	if ((GN > 31) && (GM > 15)) {//[32, 16]
		kmax81(stream, 2, 4, ic_index, j_index, deltaY, Y, OH, OW, FH, FW, deltaX, X, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		upool2DMaxBranch(31, 15); return;
	}
	if ((GN > 15) && (GM > 31)) {//[16, 32]
		kmax81(stream, 1, 5, ic_index, j_index, deltaY, Y, OH, OW, FH, FW, deltaX, X, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		upool2DMaxBranch(15, 31); return;
	}

	if (GN > 63) {//GM >= 4, [64, 4]
		kmax81(stream, 3, 2, ic_index, j_index, deltaY, Y, OH, OW, FH, FW, deltaX, X, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		upool2DMaxBranch(63, 3); return;
	}
	if ((GN > 31) && (GM > 7)) {//[32, 8]
		kmax81(stream, 2, 3, ic_index, j_index, deltaY, Y, OH, OW, FH, FW, deltaX, X, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		upool2DMaxBranch(31, 7); return;
	}
	if ((GN > 15) && (GM > 15)) {//[16, 16]
		kmax81(stream, 1, 4, ic_index, j_index, deltaY, Y, OH, OW, FH, FW, deltaX, X, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		upool2DMaxBranch(15, 15); return;
	}
	if ((GN > 7) && (GM > 31)) {//[8, 32]
		kmax81(stream, 0, 5, ic_index, j_index, deltaY, Y, OH, OW, FH, FW, deltaX, X, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		upool2DMaxBranch(7, 31); return;
	}

	if (GN > 31) {//[32, 4]
		kmax41(stream, 3, 2, ic_index, j_index, deltaY, Y, OH, OW, FH, FW, deltaX, X, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		upool2DMaxBranch(31, 3); return;
	}
	if ((GN > 15) && (GM > 7)) {//[16, 8]
		kmax41(stream, 2, 3, ic_index, j_index, deltaY, Y, OH, OW, FH, FW, deltaX, X, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		upool2DMaxBranch(15, 7); return;
	}
	if ((GN > 7) && (GM > 15)) {//[15, 7]
		kmax41(stream, 1, 4, ic_index, j_index, deltaY, Y, OH, OW, FH, FW, deltaX, X, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		upool2DMaxBranch(7, 15); return;
	}
	if (GM > 31) {//[4, 32]
		kmax41(stream, 0, 5, ic_index, j_index, deltaY, Y, OH, OW, FH, FW, deltaX, X, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		upool2DMaxBranch(3, 31); return;
	}

	if (GN > 15) {//[16, 4]
		kmax21(stream, 3, 2, ic_index, j_index, deltaY, Y, OH, OW, FH, FW, deltaX, X, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		upool2DMaxBranch(15, 3); return;
	}
	if ((GN > 7) && (GM > 7)) {//[8, 8]
		kmax21(stream, 2, 3, ic_index, j_index, deltaY, Y, OH, OW, FH, FW, deltaX, X, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		upool2DMaxBranch(7, 7); return;
	}
	if (GM > 15) {//[4, 16]
		kmax21(stream, 1, 4, ic_index, j_index, deltaY, Y, OH, OW, FH, FW, deltaX, X, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		upool2DMaxBranch(3, 15); return;
	}

	if (GN > 7) {//[8, 4]
		kmax11(stream, 3, 2, ic_index, j_index, deltaY, Y, OH, OW, FH, FW, deltaX, X, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		upool2DMaxBranch(7, 3); return;
	}
	if (GM > 7) {//[4, 8]
		kmax11(stream, 2, 3, ic_index, j_index, deltaY, Y, OH, OW, FH, FW, deltaX, X, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		upool2DMaxBranch(3, 7); return;
	}
	kmax11(stream, 2, 2, ic_index, j_index, deltaY, Y, OH, OW, FH, FW, deltaX, X, IH, IW, IC, sh, sw, ph, pw, GN, GM);
}

#endif


#ifndef UNPOOL2D_MAX_INDEXED
#define UNPOOL2D_MAX_INDEXED

#define __upool2D_max_indexed(streams, index, length, deltaY, Index, OH, OW, FH, FW, deltaX, IH, IW, N, IC, sh, sw, ph, pw)\
	upool2d_max_indexed(streams,index,length, deltaY,Index,OH,OW, FH,FW, deltaX,IH,IW, IC, sh,sw,ph,pw,\
		GET_GN(IC), GET_GM(N, IH, IW), 0, 0)

#define upool2DMaxIndexedBranch(SIZE_Y, SIZE_X) {\
	int GNr = GN & SIZE_Y, GMr = GM & SIZE_X;\
	if(GNr && GMr){\
		int next_ic_index = (GN - GNr) + ic_index;\
		int next_j_index = (GM - GMr) + j_index;\
		upool2d_max_indexed(streams,index,length, deltaY,Index,OH,OW, FH,FW, deltaX,IH,IW, IC, sh,sw,ph,pw,\
			GNr, GM, next_ic_index, j_index);\
		upool2d_max_indexed(streams,index,length, deltaY,Index,OH,OW, FH,FW, deltaX,IH,IW, IC, sh,sw,ph,pw,\
            GN, GMr, ic_index, next_j_index);\
		upool2d_max_indexed(streams,index,length, deltaY,Index,OH,OW, FH,FW, deltaX,IH,IW, IC, sh,sw,ph,pw,\
            GNr, GMr, next_ic_index, next_j_index);}\
	else if(GNr){\
		int next_ic_index = (GN - GNr) + ic_index;\
		upool2d_max_indexed(streams,index,length, deltaY,Index,OH,OW, FH,FW, deltaX,IH,IW, IC, sh,sw,ph,pw,\
			 GNr, GM, next_ic_index, j_index);}\
	else if(GMr){\
		int next_j_index = (GM - GMr) + j_index;\
		upool2d_max_indexed(streams,index,length, deltaY,Index,OH,OW, FH,FW, deltaX,IH,IW, IC, sh,sw,ph,pw,\
			 GN, GMr, ic_index, next_j_index);}}

void upool2d_max_indexed(jlong *streams, int &index, int length,
	const float*  __restrict__ deltaY,
	const int*  __restrict__ Index, int OH, int OW,
	int FH, int FW,
	float * __restrict__ deltaX, int IH, int IW,
	int IC,
	int sh, int sw, int ph, int pw,
	int GN, int GM,
	int ic_index, int j_index)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)streams[index];
	index = (index + 1) % length;

	if (GN > 127) {//[128, 2]
		kmaxIdx81(stream, 4, 1, ic_index, j_index, deltaY, Index, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		upool2DMaxIndexedBranch(127, 1); return;
	}
	if ((GN > 63) && (GM > 7)) {//[64, 8]
		kmaxIdx81(stream, 3, 3, ic_index, j_index, deltaY, Index, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		upool2DMaxIndexedBranch(63, 7); return;
	}
	if ((GN > 31) && (GM > 15)) {//[32, 16]
		kmaxIdx81(stream, 2, 4, ic_index, j_index, deltaY, Index, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		upool2DMaxIndexedBranch(31, 15); return;
	}
	if ((GN > 15) && (GM > 31)) {//[16, 32]
		kmaxIdx81(stream, 1, 5, ic_index, j_index, deltaY, Index, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		upool2DMaxIndexedBranch(15, 31); return;
	}

	if (GN > 63) {//GM >= 4, [64, 4]
		kmaxIdx81(stream, 3, 2, ic_index, j_index, deltaY, Index, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		upool2DMaxIndexedBranch(63, 3); return;
	}
	if ((GN > 31) && (GM > 7)) {//[32, 8]
		kmaxIdx81(stream, 2, 3, ic_index, j_index, deltaY, Index, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		upool2DMaxIndexedBranch(31, 7); return;
	}
	if ((GN > 15) && (GM > 15)) {//[16, 16]
		kmaxIdx81(stream, 1, 4, ic_index, j_index, deltaY, Index, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		upool2DMaxIndexedBranch(15, 15); return;
	}
	if ((GN > 7) && (GM > 31)) {//[8, 32]
		kmaxIdx81(stream, 0, 5, ic_index, j_index, deltaY, Index, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		upool2DMaxIndexedBranch(7, 31); return;
	}

	if (GN > 31) {//[32, 4]
		kmaxIdx41(stream, 3, 2, ic_index, j_index, deltaY, Index, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		upool2DMaxIndexedBranch(31, 3); return;
	}
	if ((GN > 15) && (GM > 7)) {//[16, 8]
		kmaxIdx41(stream, 2, 3, ic_index, j_index, deltaY, Index, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		upool2DMaxIndexedBranch(15, 7); return;
	}
	if ((GN > 7) && (GM > 15)) {//[15, 7]
		kmaxIdx41(stream, 1, 4, ic_index, j_index, deltaY, Index, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		upool2DMaxIndexedBranch(7, 15); return;
	}
	if (GM > 31) {//[4, 32]
		kmaxIdx41(stream, 0, 5, ic_index, j_index, deltaY, Index, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		upool2DMaxIndexedBranch(3, 31); return;
	}

	if (GN > 15) {//[16, 4]
		kmaxIdx21(stream, 3, 2, ic_index, j_index, deltaY, Index, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		upool2DMaxIndexedBranch(15, 3); return;
	}
	if ((GN > 7) && (GM > 7)) {//[8, 8]
		kmaxIdx21(stream, 2, 3, ic_index, j_index, deltaY, Index, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		upool2DMaxIndexedBranch(7, 7); return;
	}
	if (GM > 15) {//[4, 16]
		kmaxIdx21(stream, 1, 4, ic_index, j_index, deltaY, Index, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		upool2DMaxIndexedBranch(3, 15); return;
	}

	if (GN > 7) {//[8, 4]
		kmaxIdx11(stream, 3, 2, ic_index, j_index, deltaY, Index, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		upool2DMaxIndexedBranch(7, 3); return;
	}
	if (GM > 7) {//[4, 8]
		kmaxIdx11(stream, 2, 3, ic_index, j_index, deltaY, Index, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		upool2DMaxIndexedBranch(3, 7); return;
	}
	kmaxIdx11(stream, 2, 2, ic_index, j_index, deltaY, Index, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
}

#endif


#ifndef UNPOOL2D_AVG
#define UNPOOL2D_AVG

#define __upool2D_avg(streams, index, length, deltaY, OH, OW, FH, FW, deltaX,IH, IW, N, IC, sh, sw, ph, pw)\
	upool2d_avg(streams,index,length, deltaY,OH,OW, FH,FW, deltaX,IH,IW, IC, sh,sw,ph,pw,\
		GET_GN(IC), GET_GM(N, IH, IW), 0, 0)

#define upool2DAvgBranch(SIZE_Y, SIZE_X) {\
	int GNr = GN & SIZE_Y, GMr = GM & SIZE_X;\
	if(GNr && GMr){\
		int next_ic_index = (GN - GNr) + ic_index;\
		int next_j_index = (GM - GMr) + j_index;\
		upool2d_avg(streams,index,length, deltaY,OH,OW, FH,FW, deltaX,IH,IW, IC, sh,sw,ph,pw,\
			GNr, GM, next_ic_index, j_index);\
		upool2d_avg(streams,index,length, deltaY,OH,OW, FH,FW, deltaX,IH,IW, IC, sh,sw,ph,pw,\
            GN, GMr, ic_index, next_j_index);\
		upool2d_avg(streams,index,length, deltaY,OH,OW, FH,FW, deltaX,IH,IW, IC, sh,sw,ph,pw,\
            GNr, GMr, next_ic_index, next_j_index);}\
	else if(GNr){\
		int next_ic_index = (GN - GNr) + ic_index;\
		upool2d_avg(streams,index,length, deltaY,OH,OW, FH,FW, deltaX,IH,IW, IC, sh,sw,ph,pw,\
			 GNr, GM, next_ic_index, j_index);}\
	else if(GMr){\
		int next_j_index = (GM - GMr) + j_index;\
		upool2d_avg(streams,index,length, deltaY,OH,OW, FH,FW, deltaX,IH,IW, IC, sh,sw,ph,pw,\
			 GN, GMr, ic_index, next_j_index);}}

//(1) FH * FW >=2
//(2) GN % 4==0, GN >= 4
//(3) GM % 4==0, GM >= 4
//(4) GK = FH * FW >= 2
//performance=======================================================
//[OH, OW] = 32, [FH, FW] = 8, [[N] = 8, [sh, sw] = 4, [ph, pw] = 2
//IC = 128: Size = 1.000000, Time = 1.948000 msec, Peformance = 551.202148 GFlop/s
//IC = 192: Size = 1.500000, Time = 2.918000 msec, Peformance = 551.957703 GFlop/s
//IC = 224: Size = 1.750000, Time = 3.454000 msec, Peformance = 544.020813 GFlop/s
//IC = 240: Size = 1.875000, Time = 3.706000 msec, Peformance = 543.244934 GFlop/s
//IC = 248: Size = 1.937500, Time = 3.862000 msec, Peformance = 538.678040 GFlop/s
//IC = 252: Size = 1.968750, Time = 4.000000 msec, Peformance = 528.482300 GFlop/s
void upool2d_avg(jlong *streams, int &index, int length,
	const float*  __restrict__ deltaY, int OH, int OW,
	int FH, int FW,
	float * __restrict__ deltaX, int IH, int IW,
	int IC,
	int sh, int sw, int ph, int pw,
	int GN, int GM,
	int ic_index, int j_index)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)streams[index];
	index = (index + 1) % length;

	if (GN > 127) {//[128, 2]
		kavg81(stream, 4, 1, ic_index, j_index, deltaY, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		upool2DAvgBranch(127, 1); return;
	}
	if ((GN > 63) && (GM > 7)) {//[64, 8]
		kavg81(stream, 3, 3, ic_index, j_index, deltaY, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		upool2DAvgBranch(63, 7); return;
	}
	if ((GN > 31) && (GM > 15)) {//[32, 16]
		kavg81(stream, 2, 4, ic_index, j_index, deltaY, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		upool2DAvgBranch(31, 15); return;
	}
	if ((GN > 15) && (GM > 31)) {//[16, 32]
		kavg81(stream, 1, 5, ic_index, j_index, deltaY, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		upool2DAvgBranch(15, 31); return;
	}

	if (GN > 63) {//GM >= 4, [64, 4]
		kavg81(stream, 3, 2, ic_index, j_index, deltaY, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		upool2DAvgBranch(63, 3); return;
	}
	if ((GN > 31) && (GM > 7)) {//[32, 8]
		kavg81(stream, 2, 3, ic_index, j_index, deltaY, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		upool2DAvgBranch(31, 7); return;
	}
	if ((GN > 15) && (GM > 15)) {//[16, 16]
		kavg81(stream, 1, 4, ic_index, j_index, deltaY, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		upool2DAvgBranch(15, 15); return;
	}
	if ((GN > 7) && (GM > 31)) {//[8, 32]
		kavg81(stream, 0, 5, ic_index, j_index, deltaY, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		upool2DAvgBranch(7, 31); return;
	}

	if (GN > 31) {//[32, 4]
		kavg41(stream, 3, 2, ic_index, j_index, deltaY, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		upool2DAvgBranch(31, 3); return;
	}
	if ((GN > 15) && (GM > 7)) {//[16, 8]
		kavg41(stream, 2, 3, ic_index, j_index, deltaY, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		upool2DAvgBranch(15, 7); return;
	}
	if ((GN > 7) && (GM > 15)) {//[15, 7]
		kavg41(stream, 1, 4, ic_index, j_index, deltaY, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		upool2DAvgBranch(7, 15); return;
	}
	if (GM > 31) {//[4, 32]
		kavg41(stream, 0, 5, ic_index, j_index, deltaY, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		upool2DAvgBranch(3, 31); return;
	}

	if (GN > 15) {//[16, 4]
		kavg21(stream, 3, 2, ic_index, j_index, deltaY, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		upool2DAvgBranch(15, 3); return;
	}
	if ((GN > 7) && (GM > 7)) {//[8, 8]
		kavg21(stream, 2, 3, ic_index, j_index, deltaY, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		upool2DAvgBranch(7, 7); return;
	}
	if (GM > 15) {//[4, 16]
		kavg21(stream, 1, 4, ic_index, j_index, deltaY, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		upool2DAvgBranch(3, 15); return;
	}

	if (GN > 7) {//[8, 4]
		kavg11(stream, 3, 2, ic_index, j_index, deltaY, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		upool2DAvgBranch(7, 3); return;
	}
	if (GM > 7) {//[4, 8]
		kavg11(stream, 2, 3, ic_index, j_index, deltaY, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		upool2DAvgBranch(3, 7); return;
	}
	kavg11(stream, 2, 2, ic_index, j_index, deltaY, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
}

#endif


#ifndef UNPOOL2D_AVG_TIELD
#define UNPOOL2D_AVG_TIELD

#define __unpool2D_avg_tiled(streams,index,length, deltaY, OH, OW, FH, FW, deltaX, IH, IW, N, IC, sh,sw,ph,pw)\
	unpool2D_avg_tiled(streams,index,length, deltaY,OH,OW, FH,FW, deltaX,IH,IW, IC, sh,sw,ph,pw,\
		GET_GN(IC), GET_GM_TILED(N,OH,OW), 0,0)

#define unpoo2dAvgTiledBranch(SIZE_Y, SIZE_X) {\
	int GNr = GN & (SIZE_Y), GMr = GM & (SIZE_X);\
	if(GNr && GMr){\
		int next_ic_index = (GN - GNr) + ic_index;\
		int next_j_index = (GM - GMr) + j_index;\
		unpool2D_avg_tiled(streams, index, length, deltaY,OH,OW, FH,FW, deltaX,IH,IW, IC, sh,sw,ph,pw,\
			GNr, GM, next_ic_index, j_index);\
		unpool2D_avg_tiled(streams, index, length, deltaY,OH,OW, FH,FW, deltaX,IH,IW, IC, sh,sw,ph,pw,\
            GN, GMr, ic_index, next_j_index);\
		unpool2D_avg_tiled(streams, index, length, deltaY,OH,OW, FH,FW, deltaX,IH,IW, IC, sh,sw,ph,pw,\
            GNr, GMr, next_ic_index, next_j_index);}\
	else if(GNr){\
		int next_ic_index = (GN - GNr) + ic_index;\
		unpool2D_avg_tiled(streams, index, length, deltaY,OH,OW, FH,FW, deltaX,IH,IW, IC, sh,sw,ph,pw,\
			 GNr, GM, next_ic_index, j_index);}\
	else if(GMr){\
		int next_j_index = (GM - GMr) + j_index;\
		unpool2D_avg_tiled(streams, index, length, deltaY,OH,OW, FH,FW, deltaX,IH,IW, IC, sh,sw,ph,pw,\
			 GN, GMr, ic_index, next_j_index);}}

//GN = IC;          GN >= 4, GN % 4 == 0
//GM = N * OH * OW; GM >= 4, GM % 4 == 0
void unpool2D_avg_tiled(jlong *streams, int &index, int length,
	const float* __restrict__ deltaY, int OH, int OW,
	int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC,
	int sh, int sw, int ph, int pw,
	int GN, int GM,
	int ic_index, int j_index)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)(streams[index]);
	index = (index + 1) % length;

	if (GN > 127) {//[128, 1]
		kavg_tiled4(stream, 5, 0, ic_index, j_index, deltaY, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		unpoo2dAvgTiledBranch(127, 0); return;
	}
	if (GN > 63) {//[64, 2]
		kavg_tiled4(stream, 4, 1, ic_index, j_index, deltaY, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		unpoo2dAvgTiledBranch(63, 1); return;
	}
	if (GN > 31) {//[32, 4]
		kavg_tiled4(stream, 3, 2, ic_index, j_index, deltaY, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		unpoo2dAvgTiledBranch(31, 3); return;
	}
	if (GN > 15 && GM > 7) {//[16, 8]
		kavg_tiled4(stream, 2, 3, ic_index, j_index, deltaY, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		unpoo2dAvgTiledBranch(15, 7); return;
	}
	if (GN > 7 && GM > 7) {//[8, 8]
		kavg_tiled2(stream, 2, 3, ic_index, j_index, deltaY, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		unpoo2dAvgTiledBranch(7, 7); return;
	}
	if (GN > 7) {//[8, 4]
		kavg_tiled1(stream, 3, 2, ic_index, j_index, deltaY, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		unpoo2dAvgTiledBranch(7, 3); return;
	}
	if (GM > 7) {//[4, 8]
		kavg_tiled1(stream, 2, 3, ic_index, j_index, deltaY, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		unpoo2dAvgTiledBranch(3, 7); return;
	}
	kavg_tiled1(stream, 2, 2, ic_index, j_index, deltaY, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);//[4, 4]
}

#endif


#ifndef UNPOOL2D_AVG_IGNORE_PADDING
#define UNPOOL2D_AVG_IGNORE_PADDING

#define __upool2D_avg_ip(streams, index, length, deltaY, OH, OW, FH, FW, deltaX,IH, IW, N, IC, sh, sw, ph, pw)\
	upool2d_avg_ignore_padding(streams,index,length, deltaY,OH,OW, FH,FW, deltaX,IH,IW, IC, sh,sw,ph,pw,\
		GET_GN(IC), GET_GM(N, IH, IW), 0, 0)

#define upool2DAvgIPBranch(SIZE_Y, SIZE_X) {\
	int GNr = GN & SIZE_Y, GMr = GM & SIZE_X;\
	if(GNr && GMr){\
		int next_ic_index = (GN - GNr) + ic_index;\
		int next_j_index = (GM - GMr) + j_index;\
		upool2d_avg_ignore_padding(streams,index,length, deltaY,OH,OW, FH,FW, deltaX,IH,IW, IC, sh,sw,ph,pw,\
			GNr, GM, next_ic_index, j_index);\
		upool2d_avg_ignore_padding(streams,index,length, deltaY,OH,OW, FH,FW, deltaX,IH,IW, IC, sh,sw,ph,pw,\
            GN, GMr, ic_index, next_j_index);\
		upool2d_avg_ignore_padding(streams,index,length, deltaY,OH,OW, FH,FW, deltaX,IH,IW, IC, sh,sw,ph,pw,\
            GNr, GMr, next_ic_index, next_j_index);}\
	else if(GNr){\
		int next_ic_index = (GN - GNr) + ic_index;\
		upool2d_avg_ignore_padding(streams,index,length, deltaY,OH,OW, FH,FW, deltaX,IH,IW, IC, sh,sw,ph,pw,\
			 GNr, GM, next_ic_index, j_index);}\
	else if(GMr){\
		int next_j_index = (GM - GMr) + j_index;\
		upool2d_avg_ignore_padding(streams,index,length, deltaY,OH,OW, FH,FW, deltaX,IH,IW, IC, sh,sw,ph,pw,\
			 GN, GMr, ic_index, next_j_index);}}

//(1) FH * FW >=2
//(2) GN % 4==0, GN >= 4
//(3) GM % 4==0, GM >= 4
//(4) GK = FH * FW >= 2
//performance=======================================================
//[OH, OW] = 32, [FH, FW] = 8, [[N] = 8, [sh, sw] = 4, [ph, pw] = 2
//IC = 128: Size = 1.000000, Time = 1.948000 msec, Peformance = 551.202148 GFlop/s
//IC = 192: Size = 1.500000, Time = 2.918000 msec, Peformance = 551.957703 GFlop/s
//IC = 224: Size = 1.750000, Time = 3.454000 msec, Peformance = 544.020813 GFlop/s
//IC = 240: Size = 1.875000, Time = 3.706000 msec, Peformance = 543.244934 GFlop/s
//IC = 248: Size = 1.937500, Time = 3.862000 msec, Peformance = 538.678040 GFlop/s
//IC = 252: Size = 1.968750, Time = 4.000000 msec, Peformance = 528.482300 GFlop/s
void upool2d_avg_ignore_padding(jlong *streams, int &index, int length,
	const float*  __restrict__ deltaY, int OH, int OW,
	int FH, int FW,
	float * __restrict__ deltaX, int IH, int IW,
	int IC,
	int sh, int sw, int ph, int pw,
	int GN, int GM,
	int ic_index, int j_index)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)streams[index];
	index = (index + 1) % length;

	if (GN > 127) {//[128, 2]
		kavg81_ip(stream, 4, 1, ic_index, j_index, deltaY, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		upool2DAvgIPBranch(127, 1); return;
	}
	if ((GN > 63) && (GM > 7)) {//[64, 8]
		kavg81_ip(stream, 3, 3, ic_index, j_index, deltaY, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		upool2DAvgIPBranch(63, 7); return;
	}
	if ((GN > 31) && (GM > 15)) {//[32, 16]
		kavg81_ip(stream, 2, 4, ic_index, j_index, deltaY, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		upool2DAvgIPBranch(31, 15); return;
	}
	if ((GN > 15) && (GM > 31)) {//[16, 32]
		kavg81_ip(stream, 1, 5, ic_index, j_index, deltaY, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		upool2DAvgIPBranch(15, 31); return;
	}

	if (GN > 63) {//GM >= 4, [64, 4]
		kavg81_ip(stream, 3, 2, ic_index, j_index, deltaY, OH, OW, FH, FW, deltaX,  IH, IW, IC, sh, sw, ph, pw, GN, GM);
		upool2DAvgIPBranch(63, 3); return;
	}
	if ((GN > 31) && (GM > 7)) {//[32, 8]
		kavg81_ip(stream, 2, 3, ic_index, j_index, deltaY, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		upool2DAvgIPBranch(31, 7); return;
	}
	if ((GN > 15) && (GM > 15)) {//[16, 16]
		kavg81_ip(stream, 1, 4, ic_index, j_index, deltaY, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		upool2DAvgIPBranch(15, 15); return;
	}
	if ((GN > 7) && (GM > 31)) {//[8, 32]
		kavg81_ip(stream, 0, 5, ic_index, j_index, deltaY, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		upool2DAvgIPBranch(7, 31); return;
	}

	if (GN > 31) {//[32, 4]
		kavg41_ip(stream, 3, 2, ic_index, j_index, deltaY, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		upool2DAvgIPBranch(31, 3); return;
	}
	if ((GN > 15) && (GM > 7)) {//[16, 8]
		kavg41_ip(stream, 2, 3, ic_index, j_index, deltaY, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		upool2DAvgIPBranch(15, 7); return;
	}
	if ((GN > 7) && (GM > 15)) {//[15, 7]
		kavg41_ip(stream, 1, 4, ic_index, j_index, deltaY, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		upool2DAvgIPBranch(7, 15); return;
	}
	if (GM > 31) {//[4, 32]
		kavg41_ip(stream, 0, 5, ic_index, j_index, deltaY, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		upool2DAvgIPBranch(3, 31); return;
	}

	if (GN > 15) {//[16, 4]
		kavg21_ip(stream, 3, 2, ic_index, j_index, deltaY, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		upool2DAvgIPBranch(15, 3); return;
	}
	if ((GN > 7) && (GM > 7)) {//[8, 8]
		kavg21_ip(stream, 2, 3, ic_index, j_index, deltaY, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		upool2DAvgIPBranch(7, 7); return;
	}
	if (GM > 15) {//[4, 16]
		kavg21_ip(stream, 1, 4, ic_index, j_index, deltaY, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		upool2DAvgIPBranch(3, 15); return;
	}

	if (GN > 7) {//[8, 4]
		kavg11_ip(stream, 3, 2, ic_index, j_index, deltaY, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		upool2DAvgIPBranch(7, 3); return;
	}
	if (GM > 7) {//[4, 8]
		kavg11_ip(stream, 2, 3, ic_index, j_index, deltaY, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		upool2DAvgIPBranch(3, 7); return;
	}
	kavg11_ip(stream, 2, 2, ic_index, j_index, deltaY, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
}

#endif


#ifndef UNPOOL2D_AVG_IGNORE_PADDING_TIELD
#define UNPOOL2D_AVG_IGNORE_PADDING_TIELD

#define __unpool2D_avg_ip_tiled(streams,index,length, deltaY, OH, OW, FH, FW, deltaX, IH, IW, N, IC, sh,sw,ph,pw)\
	unpool2D_avg_ignore_padding_tiled(streams,index,length, deltaY,OH,OW, FH,FW, deltaX,IH,IW, IC, sh,sw,ph,pw,\
		GET_GN(IC), GET_GM_TILED(N,OH,OW), 0,0)

#define unpoo2dAvgIPTiledBranch(SIZE_Y, SIZE_X) {\
	int GNr = GN & (SIZE_Y), GMr = GM & (SIZE_X);\
	if(GNr && GMr){\
		int next_ic_index = (GN - GNr) + ic_index;\
		int next_j_index = (GM - GMr) + j_index;\
		unpool2D_avg_ignore_padding_tiled(streams, index, length, deltaY,OH,OW, FH,FW, deltaX,IH,IW, IC, sh,sw,ph,pw,\
			GNr, GM, next_ic_index, j_index);\
		unpool2D_avg_ignore_padding_tiled(streams, index, length, deltaY,OH,OW, FH,FW, deltaX,IH,IW, IC, sh,sw,ph,pw,\
            GN, GMr, ic_index, next_j_index);\
		unpool2D_avg_ignore_padding_tiled(streams, index, length, deltaY,OH,OW, FH,FW, deltaX,IH,IW, IC, sh,sw,ph,pw,\
            GNr, GMr, next_ic_index, next_j_index);}\
	else if(GNr){\
		int next_ic_index = (GN - GNr) + ic_index;\
		unpool2D_avg_ignore_padding_tiled(streams, index, length, deltaY,OH,OW, FH,FW, deltaX,IH,IW, IC, sh,sw,ph,pw,\
			 GNr, GM, next_ic_index, j_index);}\
	else if(GMr){\
		int next_j_index = (GM - GMr) + j_index;\
		unpool2D_avg_ignore_padding_tiled(streams, index, length, deltaY,OH,OW, FH,FW, deltaX,IH,IW, IC, sh,sw,ph,pw,\
			 GN, GMr, ic_index, next_j_index);}}

//GN = IC;          GN >= 4, GN % 4 == 0
//GM = N * OH * OW; GM >= 4, GM % 4 == 0
void unpool2D_avg_ignore_padding_tiled(jlong *streams, int &index, int length,
	const float* __restrict__ deltaY, int OH, int OW,
	int FH, int FW,
	float* __restrict__ deltaX, int IH, int IW,
	int IC,
	int sh, int sw, int ph, int pw,
	int GN, int GM,
	int ic_index, int j_index)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)(streams[index]);
	index = (index + 1) % length;

	if (GN > 127) {//[128, 1]
		kavg_ip_tiled4(stream, 5, 0, ic_index, j_index, deltaY, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		unpoo2dAvgIPTiledBranch(127, 0); return;
	}
	if (GN > 63) {//[64, 2]
		kavg_ip_tiled4(stream, 4, 1, ic_index, j_index, deltaY, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		unpoo2dAvgIPTiledBranch(63, 1); return;
	}
	if (GN > 31) {//[32, 4]
		kavg_ip_tiled4(stream, 3, 2, ic_index, j_index, deltaY, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		unpoo2dAvgIPTiledBranch(31, 3); return;
	}
	if (GN > 15 && GM > 7) {//[16, 8]
		kavg_ip_tiled4(stream, 2, 3, ic_index, j_index, deltaY, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		unpoo2dAvgIPTiledBranch(15, 7); return;
	}
	if (GN > 7 && GM > 7) {//[8, 8]
		kavg_ip_tiled2(stream, 2, 3, ic_index, j_index, deltaY, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		unpoo2dAvgIPTiledBranch(7, 7); return;
	}
	if (GN > 7) {//[8, 4]
		kavg_ip_tiled1(stream, 3, 2, ic_index, j_index, deltaY, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		unpoo2dAvgIPTiledBranch(7, 3); return;
	}
	if (GM > 7) {//[4, 8]
		kavg_ip_tiled1(stream, 2, 3, ic_index, j_index, deltaY, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);
		unpoo2dAvgIPTiledBranch(3, 7); return;
	}
	kavg_ip_tiled1(stream, 2, 2, ic_index, j_index, deltaY, OH, OW, FH, FW, deltaX, IH, IW, IC, sh, sw, ph, pw, GN, GM);//[4, 4]
}

#endif

#endif
