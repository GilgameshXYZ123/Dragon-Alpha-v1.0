#pragma once

#ifndef MICRO_H
#define MICRO_H

#define within_width(v, index4, stride, width) {\
	v.x *= ((index4    ) % stride) < width;\
	v.y *= ((index4 + 1) % stride) < width;\
	v.z *= ((index4 + 2) % stride) < width;\
	v.w *= ((index4 + 3) % stride) < width;}

#define COPY4(a, b) {(a).x = (b).x; (a).y = (b).y; (a).z = (b).z; (a).w = (b).w;}



#endif