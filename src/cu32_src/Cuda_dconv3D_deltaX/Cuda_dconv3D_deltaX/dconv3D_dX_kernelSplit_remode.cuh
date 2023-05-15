#pragma once

#ifndef DCONV3D_DX_KERNEL_SPLIT_REMODE_H
#define DCONV3D_DX_KERNEL_SPLIT_REMODE_H


#ifndef DCONV3D_DX_KERNEL_SPLIT_REMODE_CALL
#define DCONV3D_DX_KERNEL_SPLIT_REMODE_CALL

//LB = log2(BLOCK_SIZE)
//lengthv = CFH * CFW * OC * IC

#define ks_remode_k4(stream, LB, LT, W, FH, FW, CW, CFH, CFW, OC, IC, sh, sw, lengthv)\
	ks_remode_kernel_4\
		<<< dim3((lengthv>>LB>>LT), (sh*sw)), 1<<LB, 0, stream >>>\
			(W, FH, FW, CW, CFH, CFW, OC, IC, sh, sw)

#define ks_remode_k4_small(stream, W, FH, FW, CW, CFH, CFW, OC, IC, sh, sw, lengthv)\
	ks_remode_kernel_4\
		<<< dim3(1, (sh*sw)), ((lengthv + 3) >> 2), 0, stream >>>\
			(W, FH, FW, CW, CFH, CFW, OC, IC, sh, sw)

#endif


#ifndef DCONV3D_DX_KERNEL_SPLIT_REMODE_KERNEL_4
#define DCONV3D_DX_KERNEL_SPLIT_REMODE_KERNEL_4

//(CFH, CFW) = max(CFH, CFW)
//lengthv = sh*sw * OC*CFH*CFW*IC
//W[oc, fh, fw, ic] -> CFH[y, x, oc:OC, fhr:CFH, fwr:CFW, ic:IC]
//fh = y + (oph - fhr)*sh;
//fw = x + (opw - fwr)*sw
__global__ void ks_remode_kernel_4(
	const float* __restrict__ W, int FH, int FW,
    float* __restrict__ CW, int mCFH, int mCFW,
	int OC, int IC, int sh, int sw)
{
	int by = blockIdx.y;
	CW += (by * OC * mCFH * mCFW * IC);//Wks[y, x]

	int y = by / sw, x = by - y * sw;
	int CFH = (FH - y + sh - 1) / sh, oph = CFH - 1;
	int CFW = (FW - x + sw - 1) / sw, opw = CFW - 1;

	int CFW_IC = CFW * IC;
	int CFH_CFW_IC = CFH * CFW_IC;
	
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	int lengthv = OC * CFH * CFW * IC;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		int oc = index4 / CFH_CFW_IC, ir = index4 - oc * CFH_CFW_IC;
		int fhr = ir / CFW_IC; ir -= fhr * CFW_IC;
		int fwr = ir / IC, ic = ir - fwr * IC;

		int fh = y + (oph - fhr)*sh;
		int fw = x + (opw - fwr)*sw;
		
		int woffset = ((oc*FH + fh)*FW + fw)*IC + ic;
		bool lw = (fh < FH) && (fw < FW);
		float4 w = (lw ? *(float4*)(W + woffset) : FLOAT_ZERO4);

		*(float4*)(CW + index4) = w;
	}
}

#endif


void __ks_remode(cudaStream_t stream,
	const float* W, int FH, int FW,
	float * CW, int CFH, int CFW,
	int OC, int IC, int sh, int sw)
{
	int lengthv = OC * CFH * CFW * IC;
	if (lengthv < 256) { ks_remode_k4_small(stream, W, FH, FW, CW, CFH, CFW, OC, IC, sh, sw, lengthv); return;  }
	ks_remode_k4(stream, 5, 2, W, FH, FW, CW, CFH, CFW, OC, IC, sh, sw, lengthv);
}

#endif