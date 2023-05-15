#pragma once

#ifndef MICRO_H
#define MICRO_H

#define FLOAT_MIN4 make_float4(FLOAT_MIN, FLOAT_MIN, FLOAT_MIN, FLOAT_MIN)
#define FLOAT_MIN2 make_float2(FLOAT_MIN, FLOAT_MIN)


#define get2d(A, y, x, stride) A[(y)*(stride) + (x)]
#define lget2d(A, y, x, lstride) A[((y)<<(lstride)) + (x)] //lstride = log2(stride)

#define get3d(A, z, y, x, Sy, Sx) A[(((z)*(Sy)) + (y))*(Sx) + (x)]
#define lget3d(A, z, y, x, Sy, LSx) A[((((z)*(Sy)) + y)<<(LSx)) + (x)]

#define get4d(A, w, z, y, x, Sz, Sy, Sx)  A[(((w)*(Sz) + (z))*(Sy) + (y))*(Sx) + (x)]
#define lget4d(A, w, z, y, x, Sz, LSy, LSx) A[(((((w)*(Sz) + (z))<<(LSy)) + (y))<<(LSx)) + (x)]


#define simdMM4(c, av, b) {(c).x += (av) * (b).x; (c).y += (av) * (b).y; (c).z += (av) * (b).z; (c).w += (av) * (b).w;}
#define simdMM2(c, av, b) {(c).x += (av) * (b).x; (c).y += (av) * (b).y;} 

#define simdAdd4(c, a, b) {(c).x = (a).x + (b).x; (c).y = (a).y + (b).y; (c).z = (a).z + (b).z; (c).w = (a).w + (b).w;}
#define simdAdd2(c, a, b) {(c).x = (a).x + (b).x; (c).y = (a).y + (b).y; }

#define simdSDiv4(c, a, v) {(c).x = (a).x / (v); (c).y = (a).y / (v); (c).z = (a).z / (v); (c).w = (a).w / (v);}
#define simdSDiv2(c, a, v) {(c).x = (a).x / (v); (c).y = (a).y / (v); }


#define simdMAX4(c, a, b) {c.x = fmaxf(a.x, b.x); c.y = fmaxf(a.y, b.y);c.z = fmaxf(a.z, b.z); c.w = fmaxf(a.w, b.w);}
#define simdMAX2(c, a, b) {c.x = fmaxf(a.x, b.x); c.y = fmaxf(a.y, b.y);}


#define MAX_V_INT(x, y) ((x)^(((x)^(y))& -((x)<(y))))
#define MIN_V_INT(x, y) ((y)^(((x)^(y))& -((x)<(y))))

#define choose(flag, a, b) ((flag)*((a)-(b)) + (b))

#define FLOAT_MAX (3.4028235E38f)
#define FLOAT_MIN (-3.4028234E38f)


#define GET_GN(IC) (IC) //GN = IC;
#define GET_GM(N,OH,OW) ((N)*(OH)*(OW))//GM = N * OH * OW;
#define GET_GK(FH, FW) ((FH)*(FW)) //GK = FH * FW


#define get_n_oh_ow(j, n, oh, ow)\
	int n, oh, ow;{n = j/OH_OW; int jr = j - n*OH_OW; oh = jr/OW; ow = jr - oh*OW;}

#define loadX(v, n, ic, ih, iw, ih_iw) \
	if((ih<0)||(iw<0)||(iw>=IW)||(ih>=IH)) (v) = 0;\
	else (v) = get3d(X, n, ic, ih_iw, IC, IH_IW);

#endif