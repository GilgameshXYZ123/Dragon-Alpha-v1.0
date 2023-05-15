#pragma once

#ifndef CONV3D_KERNEL_REMODE_H
#define CONV3D_KERNEL_REMODE_H

//Remode the kernel:
//W[OC, FH, FW, IC] -> CW[FH, FW, IC, OC]
#ifndef CONV3D_KERNEL_REMODE_CALL
#define CONV3D_KERNEL_REMODE_CALL

//LB = log2(BLOCK_SIZE)
//lengthv = FH * FW * OC * IC

#define kremode_k4(stream, LB, LT, W, CW, FH, FW, OC, IC, lengthv)\
	kremode_kernel_4\
		<<< (lengthv>>LB>>LT), 1<<LB, 0, stream >>>\
			(W, CW, FH, FW, OC, IC, lengthv)

#define kremode_k4_small(stream, W, CW, FH, FW, OC, IC, lengthv)\
	kremode_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(W, CW, FH, FW, OC, IC, lengthv)

//lengthv: OC*FH*FW*IC -> (OC>>2)*FH*FW*IC
#define kremode_k4_4x(stream, LB, LT, W, CW, FH, FW, OC, IC, lengthv)\
	kremode_kernel_4X\
		<<< ((lengthv>>2)>>LB>>LT), (1<<LB), 0, stream >>>\
			(W, CW, FH, FW, OC, IC, (lengthv>>2))

#endif


#ifndef CONV3D_KERNEL_REMODE_KERNEL_4
#define CONV3D_KERNEL_REMODE_KERNEL_4

//W[OC, FH, FW, IC] -> CW[FH, FW, IC, OC]
//lengthv = FH * FW * OC * IC

__global__ void kremode_kernel_4(
	const float* __restrict__ W, 
	float* __restrict__ CW,
	int FH, int FW, int OC, int IC, int lengthv)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	int FW_IC = FW * IC;
	int FH_FW_IC = FH * FW_IC;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		//[oc, fh, fw, ic] -> [fh, fw, ic, oc]
		int oc = index4 / FH_FW_IC, ir = index4 - oc * FH_FW_IC;
		int fh = ir / FW_IC; ir -= fh * FW_IC;
		int fw = ir / IC, ic = ir - fw * IC;

		//Woffset[oc, fh, fw, ic0 -> ic3]
		float4 w = *(float4*)(W + index4);

		//CW1 = (((fh*FW + fw)*IC + ic)*OC + oc);
		//= fh*FW*IC*OC + fw*IC*OC + ic*OC + oc
		//= fh*FW*IC*OC + (fw*IC + ic)*OC + oc
		//As: ir = fw*IC + ic
		//= (fh*FW*IC + (fw*IC + ic))*OC + oc
		//= (fh*FW*IC + ir)*OC + oc

		int CW1 = (((fh*FW + fw)*IC + ic + 1)*OC + oc);
		CW[CW1 - OC]        = w.x;//[fh, fw, ic0, oc]
		CW[CW1]             = w.y;//[fh, fw, ic1, oc]
		CW[CW1 + OC]        = w.z;//[fh, fw, ic2, oc]
		CW[CW1 + (OC << 1)] = w.w;//[fh, fw, ic3, oc]
	}
}

#endif


#ifndef CONV3D_KERNEL_REMODE_KERNEL_4X
#define CONV3D_KERNEL_REMODE_KERNEL_4X

//lengthv = FH * FW * (OC >> 2) * IC
//W[OC, FH, FW, IC] -> CW[FH, FW, OC]
//lengthv = FH * FW * OC * IC
//As: IC % 4 == 0, So: lengthv % 4 == 0

__global__ void kremode_kernel_4X(
	const float* __restrict__ W,
	float* __restrict__ CW,
	int FH, int FW, int OC, int IC, int lengthv)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	int FW_IC = FW * IC;
	int FH_FW_IC = FH * FW_IC;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		//[oc, fh, fw, ic] -> [fh, fw, ic, oc]
		int oc = index4 / FH_FW_IC, ir = index4 - oc * FH_FW_IC;
		int fh = ir / FW_IC; ir -= fh * FW_IC;
		int fw = ir / IC, ic = ir - fw * IC;

		int oc0 = oc << 2;
		int woffset0 = ((oc0*FH + fh)*FW + fw)*IC + ic;
		int woffset1 = woffset0 + FH_FW_IC;
		int woffset2 = woffset1 + FH_FW_IC;
		int woffset3 = woffset2 + FH_FW_IC;

		float4 w0 = *(float4*)(W + woffset0);//W[oc0, fh, fw, ic0-ic3]
		float4 w1 = *(float4*)(W + woffset1);//W[oc1, fh, fw, ic0-ic3]
		float4 w2 = *(float4*)(W + woffset2);//W[oc2, fh, fw, ic0-ic3]
		float4 w3 = *(float4*)(W + woffset3);//W[oc3, fh, fw, ic0-ic3]

		int cwoffset0 = (((fh*FW + fw)*IC + ic)*OC + oc0);//CW[FH, FW, IC, OC]
		int cwoffset1 = cwoffset0 + OC;
		int cwoffset2 = cwoffset1 + OC;
		int cwoffset3 = cwoffset2 + OC;

		float4 cw0 = make_float4(w0.x, w1.x, w2.x, w3.x);//W[oc0-oc3, fh, fw, ic0]
		float4 cw1 = make_float4(w0.y, w1.y, w2.y, w3.y);//W[oc0-oc3, fh, fw, ic1]
		float4 cw2 = make_float4(w0.z, w1.z, w2.z, w3.z);//W[oc0-oc3, fh, fw, ic2]
		float4 cw3 = make_float4(w0.w, w1.w, w2.w, w3.w);//W[oc0-oc3, fh, fw, ic3]

		*(float4*)(CW + cwoffset0) = cw0;//CW[ic0, fh, fw, oc0-oc3]
		*(float4*)(CW + cwoffset1) = cw1;//CW[ic1, fh, fw, oc0-oc3]
		*(float4*)(CW + cwoffset2) = cw2;//CW[ic2, fh, fw, oc0-oc3]
		*(float4*)(CW + cwoffset3) = cw3;//CW[ic3, fh, fw, oc0-oc3]
	}
}

#endif


//OC >= 4, OC % 4 == 0
void __kernel_remode(cudaStream_t stream,
	const float* W, float * CW,
	int FH, int FW, int OC, int IC)
{
	int lengthv = OC * FH * FW * IC;
	if (lengthv < 256) { kremode_k4_small(stream, W, CW, FH, FW, OC, IC, lengthv); return; }
	else if(lengthv > 1024) { kremode_k4_4x(stream, 5, 2, W, CW, FH, FW, OC, IC, lengthv); return; }
	else kremode_k4(stream, 5, 2, W, CW, FH, FW, OC, IC, lengthv);
}

#endif