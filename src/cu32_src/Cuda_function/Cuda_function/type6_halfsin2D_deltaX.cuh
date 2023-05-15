#pragma once

#ifndef HALF_SIN_2D_DELTAX_H
#define HALF_SIN_2D_DELTAX_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef HALF_SIN_2D_DELTAX_CALL
#define HALF_SIN_2D_DELTAX_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define halfsin2d_deltaX_k4(stream, LB, LT, deltaX, deltaY, Y, Amp, alpha, lengthv, width, stride)\
	halfsin2D_deltaX_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(deltaX, deltaY, Y, Amp, alpha, lengthv, width, stride)

#define halfsin2d_deltaX_k4_small(stream,  deltaX, deltaY, Y, Amp, alpha, lengthv, width, stride)\
	halfsin2D_deltaX_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(deltaX, deltaY, Y, Amp, alpha, lengthv, width, stride)

#endif


#ifndef HALF_SIN_2D_DELTAX_KERNEL
#define HALF_SIN_2D_DELTAX_KERNEL

//(1) Y = Amp * {2 * |sin(alpha*X + beta)| - 1}
//    Y = 2*Amp *{|sin(alpha*X + beta)|- 0.5}
//(2) Y' = 2*Amp * |alpha * cos(alpha*X + beta)|
//    Y' = 2*Amp*|alpha| * |cos(alpha*X + beta)|
//As: Y = 2*Amp *{|sin(alpha*X + beta)| -0.5 }
//So: |sin(alpha*X + beta)| = Y/(2*Amp) + 0.5
//As: |cos(alpha*X + beta)| = sqrt(1 - |sin(alpha*X + beta)|^2)
//So: |cos(alpha*X + beta)| = sqrt(1 - (Y/(2*Amp) + 0.5)^2)
//Y' = 2*Amp*|alpha| * sqrt{1 - (Y/(2*Amp) + 0.5)^2}
//Let:
//[1] amp2 = 2*Amp*|alpha|
//[2] ramp2 = 1/(2*Amp) = 0.5 * Amp
//Y' = amp2 * sqrt{1 - (Y*ramp2 + 0.5)^2}
// (3) deltaX = deltaY <*> Y'.

__global__ void halfsin2D_deltaX_kernel_4(
	float* __restrict__ deltaX,
	const float* __restrict__ deltaY,
	const float* __restrict__ Y,
	float Amp, float alpha, 
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	//[1] amp2 = 2*Amp*|alpha|
	//[2] ramp2 = 1/(2*Amp) = 0.5 * Amp
	float amp2 = 2.0f * Amp * fabsf(alpha);
	float ramp2 = 0.5f * Amp;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 y  = *(float4*)(Y      + index4);
		float4 dy = *(float4*)(deltaY + index4);

		float4 dx;//Y' = amp2 * sqrt{1 - (Y*ramp2 + 0.5)^2}
		y.x = y.x * ramp2 + 0.5f; dx.x = amp2 * sqrtf(1.0f - y.x*y.x);
		y.y = y.y * ramp2 + 0.5f; dx.y = amp2 * sqrtf(1.0f - y.y*y.y);
		y.z = y.z * ramp2 + 0.5f; dx.z = amp2 * sqrtf(1.0f - y.z*y.z);
		y.w = y.w * ramp2 + 0.5f; dx.w = amp2 * sqrtf(1.0f - y.w*y.w);

		simdMul4(dx, dx, dy);//deltaX = deltaY * driY

		within_width(dx, index4, stride, width);
		*(float4*)(deltaX + index4) = dx;
	}
}

#endif


void __halfsin2D_deltaX(cudaStream_t stream,
	float* deltaX,
	const float* deltaY,
	const float *Y,
	float Amp, float alpha,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { halfsin2d_deltaX_k4_small(stream, deltaX, deltaY, Y, Amp, alpha, lengthv, width, stride); return; }
	halfsin2d_deltaX_k4(stream, 5, 2, deltaX, deltaY, Y, Amp, alpha, lengthv, width, stride);
}

#endif