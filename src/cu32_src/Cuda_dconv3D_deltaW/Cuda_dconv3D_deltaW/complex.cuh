#pragma once

#ifndef COMPLEX_H
#define COMPLEX_H

#define PI 3.1415926f

//float2{real, imag}
#define cpx_real(a) (a.x)
#define cpx_imag(a) (a.y)
#define cpx_F32_0 float2{0, 0}

#define cpx_add(c, a, b) { c.x = a.x + b.x; c.y = a.y + b.y; }
#define cpx_sub(c, a, b) { c.x = a.x - b.x; c.y = a.y - b.y; }
#define cpx_mul(c, a, b) { c.x = a.x*b.x - a.y*b.y; c.y = a.x*b.y + a.y*b.x; }
#define cpx_div(c, a, b) { float s2 = 1.0f / (b.x*b.x + b.y*b.y);\
	c.x = s2 * (a.x*b.x + a.y*b.y);\
	c.y = s2 * (a.y*b.x - a.x*b.y);}

#define cpx_conj(b, a) { b.x = a.x; b.y = -a.y; }
#define cpx_scale(b, a, k) { b.x = a.x*k; b.y = a.y*k; }



#endif
