#pragma once

#ifndef MICRO_H
#define MICRO_H

#ifndef TEXTURE_FUNCTION
#define TEXTURE_FUNCTION

cudaTextureObject_t createFloat4Texture(float *X, long sizeX)
{
	cudaResourceDesc rdesc;
	memset(&rdesc, 0, sizeof(rdesc));
	rdesc.resType = cudaResourceTypeLinear;
	rdesc.res.linear.devPtr = X;
	rdesc.res.linear.desc.f = cudaChannelFormatKindFloat;
	rdesc.res.linear.desc.x = 32;
	rdesc.res.linear.desc.y = 32;
	rdesc.res.linear.desc.z = 32;
	rdesc.res.linear.desc.w = 32;
	rdesc.res.linear.sizeInBytes = sizeX * sizeof(float);

	cudaTextureDesc tdesc;
	memset(&tdesc, 0, sizeof(tdesc));
	tdesc.readMode = cudaReadModeElementType;

	cudaTextureObject_t texW = NULL;
	cudaCreateTextureObject(&texW, &rdesc, &tdesc, NULL);
	return texW;
}

#endif 

#define COPY4(a, b) {(a).x = (b).x; (a).y = (b).y; (a).z = (b).z; (a).w = (b).w;}


#define MAX_INT(x, y) ((x)^(((x)^(y))& -((x)<(y))))
#define MIN_INT(x, y) ((y)^(((x)^(y))& -((x)<(y))))

#define PI  (3.141592f)
#define RPI (0.3183099f)

#define FLOAT_MAX (3.4028235E38)
#define FLOAT_MIN (-3.4028234E38)

#define F32_4_0 float4{0, 0, 0, 0}


#define MOVE_E(v) {(v) = (v) + 1e-9f - 2.0f*((v)<0)*1e-9f;}


#define simdLinear4(b, alpha, a, beta) {\
	(b).x = (a).x *(alpha) + (beta);\
	(b).y = (a).y *(alpha) + (beta);\
	(b).z = (a).z *(alpha) + (beta);\
	(b).w = (a).w *(alpha) + (beta);}

//pay attention to nan caused by 0
#define within_width(v, index4, stride, width) {\
	v.x *= ((index4    ) % stride) < width;\
	v.y *= ((index4 + 1) % stride) < width;\
	v.z *= ((index4 + 2) % stride) < width;\
	v.w *= ((index4 + 3) % stride) < width;}

//use more resource, but can zero nan caused by zero: 1, if within with
#define within_width_zero_nan(v, index4, table, stride, width) {\
	table[1] = v;\
	v.x = table[((index4    ) % stride) < width].x;\
	v.y = table[((index4 + 1) % stride) < width].y;\
	v.z = table[((index4 + 2) % stride) < width].z;\
	v.w = table[((index4 + 3) % stride) < width].w;}

#define simdMul4(c, a, b) {\
	(c).x = (a).x * (b).x;\
	(c).y = (a).y * (b).y;\
	(c).z = (a).z * (b).z;\
	(c).w = (a).w * (b).w; }

#endif