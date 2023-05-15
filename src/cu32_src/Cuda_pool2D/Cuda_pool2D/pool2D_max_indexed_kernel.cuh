#pragma once

#ifndef POOL2D_MAX_INDEXED_KERNEL_H
#define POOL2D_MAX_INDEXED_KERNEL_H

//(1) FH * FW >= 2;
//(2) GN = IC; GN % 4 == 0, GN >= 4
//(3) GM = N * OH * OW;
//(4) GK = FH * FW >= 2
//(5) the memory padding alignment only effects N, IC, OC
#ifndef POOL2D_MAX_INDEXED_KERNEL_CALL
#define POOL2D_MAX_INDEXED_KERNEL_CALL

#define kmaxIdx4(stream, LBY, LBX, X, IH, IW, FH, FW, Y, Index, OH, OW, sh, sw, ph, pw, GN, GM, ic_index, j_index)\
	max_indexed_kernel_4\
		<<< dim3(GM>>LBX , GN>>LBY>>2), dim3(1<<LBX, 1<<LBY), 0, stream>>>\
			(X, IH, IW, FH, FW, Y, Index, OH, OW, IC, sh, sw, ph, pw, ic_index, j_index)

#define kmaxIdx2(stream, LBY, LBX, X, IH, IW, FH, FW, Y, Index, OH, OW, sh, sw, ph, pw, GN, GM, ic_index, j_index)\
	max_indexed_kernel_2\
		<<< dim3(GM>>LBX , GN>>LBY>>1), dim3(1<<LBX, 1<<LBY), 0, stream>>>\
			(X, IH, IW, FH, FW, Y, Index, OH, OW, IC, sh, sw, ph, pw, ic_index, j_index)

#define kmaxIdx1(stream, LBY, LBX, X, IH, IW, FH, FW, Y, Index, OH, OW, sh, sw, ph, pw, GN, GM, ic_index, j_index)\
	max_indexed_kernel_1\
		<<< dim3(GM>>LBX , GN>>LBY), dim3(1<<LBX, 1<<LBY), 0, stream>>>\
			(X, IH, IW, FH, FW, Y, Index, OH, OW, IC, sh, sw, ph, pw, ic_index, j_index)

#endif


//(Y: BLOCK_SIZE * 4)
#ifndef POOL2D_MAX_INDEXED_KERNEL_4
#define POOL2D_MAX_INDEXED_KERNEL_4

//<2, 2>: Size = 0.062500, Time = 1.538000 msec, Performance = 43.633850 GFlop/s [ 16, 4]
//<2, 3>: Size = 0.062500, Time = 1.402000 msec, Performance = 47.866524 GFlop/s [ 16, 8]
//<3, 2>: Size = 0.125000, Time = 2.312000 msec, Performance = 58.052647 GFlop/s [ 32, 4]
//<4, 1>: Size = 0.062500, Time = 1.110000 msec, Performance = 60.458431 GFlop/s [ 64, 2]
//<5, 0>: Size = 0.125000, Time = 2.024000 msec, Performance = 66.313103 GFlop/s [128, 1]
__global__ void max_indexed_kernel_4(
	const float* __restrict__ X, int IH, int IW,
	int FH, int FW,
	float* __restrict__ Y, 
	int* __restrict__ Index, int OH, int OW,
	int IC,
	int sh, int sw, int ph, int pw,
	int ic_index, int j_index)
{
	//prepare for GN = IC
	int ic = (((blockIdx.y*blockDim.y) + threadIdx.y) << 2) + ic_index;

	//prepare for GM = N * OH * OW
	int j = (blockIdx.x * blockDim.x) + threadIdx.x + j_index;
	const int OH_OW = OH * OW; get_n_oh_ow(j, n, oh, ow);
	const int toh = oh * sh - ph, tow = ow * sw - pw;
	const int shiftX = n * IH * IW * IC;
	X += shiftX;

	float4 v = FLOAT_MIN4;
	int4 idx = make_int4(0, 0, 0, 0);
	for (int fh = 0; fh < FH; fh++)
	{
		int ih = toh + fh;
		bool lflag = (ih >= 0) && (ih < IH);
		for (int fw = 0; fw < FW; fw++)
		{
			int iw = tow + fw;
			if (lflag && iw >= 0 && iw < IW) 
			{
				int xoffset = (ih*IW + iw)*IC + ic;
				float4 x = *(float4*)(X + xoffset);
				
				//compute index: which X -> Y
				//if: x  > v: idx = (idx - xoffset)*0 + xoffset = xoffset
				//if: x <= v: idx = (idx - xoffset)*1 + xoffset = idx
				idx.x = (idx.x - (xoffset    )) * (x.x < v.x) + (xoffset   );
				idx.y = (idx.y - (xoffset + 1)) * (x.y < v.y) + (xoffset + 1);
				idx.z = (idx.z - (xoffset + 2)) * (x.z < v.z) + (xoffset + 2);
				idx.w = (idx.w - (xoffset + 3)) * (x.w < v.w) + (xoffset + 3);

				simdMAX4(v, v, x);
			}
		}
	}

	idx.x += shiftX;
	idx.y += shiftX;
	idx.z += shiftX;
	idx.w += shiftX;

	int yoffset = j * IC + ic;
	*(float4*)(Y + yoffset) = v;
	*(int4*)(Index + yoffset) = idx;
}

#endif


//(Y: BLOCK_SIZE * 2)
#ifndef POOL2D_MAX_INDEXED_KERNEL_2
#define POOL2D_MAX_INDEXED_KERNEL_2

//<2, 2>: Size = 0.062500, Time = 2.796000 msec, Performance = 24.001738 GFlop/s [ 4, 8]
//<2, 3>: Size = 0.062500, Time = 1.862000 msec, Performance = 36.041279 GFlop/s [ 8, 8]
//<3, 2>: Size = 0.062500, Time = 1.740000 msec, Performance = 38.568310 GFlop/s [16, 4]
//<4, 1>: Size = 0.062500, Time = 1.626000 msec, Performance = 41.272358 GFlop/s [32, 2]
//<5, 0>: Size = 0.062500, Time = 1.516000 msec, Performance = 44.267056 GFlop/s [64, 1]
__global__ void max_indexed_kernel_2(
	const float* __restrict__ X, int IH, int IW,
	int FH, int FW,
	float* __restrict__ Y, 
	int* __restrict__ Index, int OH, int OW,
	int IC,
	int sh, int sw, int ph, int pw,
	int ic_index, int j_index)
{
	//prepare for GN = IC
	int ic = (((blockIdx.y*blockDim.y) + threadIdx.y) << 1) + ic_index;

	//prepare for GM = N * OH * OW
	int j = (blockIdx.x * blockDim.x) + threadIdx.x + j_index;
	const int OH_OW = OH * OW; get_n_oh_ow(j, n, oh, ow);
	const int toh = oh * sh - ph, tow = ow * sw - pw;
	const int shiftX = n * IH * IW * IC;
	X += shiftX;

	float2 v = FLOAT_MIN2;
	int2 idx = make_int2(0, 0);
	for (int fh = 0; fh < FH; fh++)
	{
		int ih = toh + fh;
		bool lflag = (ih >= 0) && (ih < IH);
		for (int fw = 0; fw < FW; fw++)
		{
			int iw = tow + fw;
			if (lflag && iw >= 0 && iw < IW) 
			{
				int xoffset = (ih*IW + iw)*IC + ic;
				float2 x = *(float2*)(X + xoffset);

				//compute index: which X -> Y
				//if: x  > v: idx = (idx - xoffset)*0 + xoffset = xoffset
				//if: x <= v: idx = (idx - xoffset)*1 + xoffset = idx
				idx.x = (idx.x - (xoffset    )) * (x.x < v.x) + (xoffset    );
				idx.y = (idx.y - (xoffset + 1)) * (x.y < v.y) + (xoffset + 1);

				simdMAX2(v, v, x);
			}
		}
	}

	idx.x += shiftX;
	idx.y += shiftX;

	int yoffset = j * IC + ic;
	*(float2*)(Y + yoffset) = v;
	*(int2*)(Index + yoffset) = idx;
}

#endif


//(Y: BLOCK_SIZE * 1)
#ifndef POOL2D_MAX_INDEXED_KERNEL_1
#define POOL2D_MAX_INDEXED_KERNEL_1

//<2, 2>: Size = 0.062500, Time = 5.292000 msec, Performance = 12.681190 GFlop/s [ 4, 4]
//<2, 3>: Size = 0.062500, Time = 3.634000 msec, Performance = 18.466940 GFlop/s [ 4, 8]
//<3, 2>: Size = 0.062500, Time = 2.842000 msec, Performance = 23.613251 GFlop/s [ 8, 4]
//<4, 1>: Size = 0.062500, Time = 2.562000 msec, Performance = 26.193933 GFlop/s [16, 2]
//<5, 0>: Size = 0.062500, Time = 2.664000 msec, Performance = 25.191011 GFlop/s [32, 1]
__global__ void max_indexed_kernel_1(
	const float* __restrict__ X, int IH, int IW,
	int FH, int FW,
	float* __restrict__ Y, 
	int* __restrict__ Index, int OH, int OW,
	int IC,
	int sh, int sw, int ph, int pw,
	int ic_index, int j_index)
{
	//prepare for GN = IC
	int ic = (blockIdx.y*blockDim.y) + threadIdx.y + ic_index;

	//prepare for GM = N * OH * OW
	int j = (blockIdx.x * blockDim.x) + threadIdx.x + j_index;
	const int OH_OW = OH * OW; get_n_oh_ow(j, n, oh, ow);
	const int toh = oh * sh - ph, tow = ow * sw - pw;
	const int shiftX = n * IH * IW * IC;
	X += shiftX;

	float v = FLOAT_MIN;
	int idx = 0;
	for (int fh = 0; fh < FH; fh++)
	{
		int ih = toh + fh;
		bool lflag = (ih >= 0) && (ih < IH);
		for (int fw = 0; fw < FW; fw++)
		{
			int iw = tow + fw;
			if (lflag && iw >= 0 && iw < IW) 
			{
				int xoffset = (ih*IW + iw)*IC + ic;
				float x = X[xoffset];

				//compute index: which X -> Y
				//if: x  > v: idx = (idx - xoffset)*0 + xoffset = xoffset
				//if: x <= v: idx = (idx - xoffset)*1 + xoffset = idx
				idx = (idx - (xoffset)) * (x < v) + (xoffset);
				
				v = fmaxf(v, x);
			}
		}
	}

	idx += shiftX;

	int yoffset = j * IC + ic;
	Y[yoffset] = v;
	Index[yoffset] = idx;
}

#endif

#endif