#pragma once

#ifndef MICRO_H
#define MICRO_H


#define MAX_INT(x, y) ((x)^(((x)^(y))& -((x)<(y))))
#define MIN_INT(x, y) ((y)^(((x)^(y))& -((x)<(y))))

#define TWO_PI 6.28325f

#define FLOAT_MAX (3.4028235E38)
#define FLOAT_MIN (-3.4028234E38)


//original: 4194303u = (1 << 23) - 1
#define THREAD_MOD 8388607u

//origianl: step
#define THREAD_MUL 4148271


//generate a float belongs to (0, 1) depending on the seed
//(((seed) = (mul*(seed) + inc) & mod) / (mod + 1) )
#define NEXT_FLOAT(seed) \
	(((seed) = (632229u*(seed) + 2100473u) & 4194303u) / 4194304.0f)
	
//original: (((seed) = (632229u*(seed) + 2100473u) & 4194303u) / 4194304.0f)
//optim1:   (((seed) = (632229u*(seed) + 21473u) & 4194303u) / 4194304.0f)
//optim2:   (((seed) = (32083u*(seed) + 2100473) & 4194303u) / 4194304.0f)
//optim3:   (((seed) = (4148271u*(seed) + 2100473u) & 33554431u) / 33554432.0f)


//[v, p] belong to (0, 1)
//if v > p : v = v1
//else: v = v2
#define BERNOULI(v, p, v1, v2) \
	(((v)<=(p))*((v1)-(v2)) + (v2))


#define simdLinear4(b, alpha, a, beta) {\
	(b).x = (a).x *(alpha) + (beta);\
	(b).y = (a).y *(alpha) + (beta);\
	(b).z = (a).z *(alpha) + (beta);\
	(b).w = (a).w *(alpha) + (beta);}

#define simdMul4(c, a, b) {\
	(c).x = (a).x * (b).x;\
	(c).y = (a).y * (b).y;\
	(c).z = (a).z * (b).z;\
	(c).w = (a).w * (b).w; }

#define simdNextFloat4(v, seed) {\
	(v).x = NEXT_FLOAT(seed);\
	(v).y = NEXT_FLOAT(seed);\
	(v).z = NEXT_FLOAT(seed);\
	(v).w = NEXT_FLOAT(seed);}

#define simdNextFloat2(v, seed) {\
	(v).x = NEXT_FLOAT(seed);\
	(v).y = NEXT_FLOAT(seed);}

#define within_width(v, index4, stride, width) {\
	v.x *= ((index4    ) % stride) < width;\
	v.y *= ((index4 + 1) % stride) < width;\
	v.z *= ((index4 + 2) % stride) < width;\
	v.w *= ((index4 + 3) % stride) < width;}

#endif