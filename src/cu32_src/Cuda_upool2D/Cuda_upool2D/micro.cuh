#pragma once 

#ifndef MICRO_H
#define MICRO_H

#define get2d(A, y, x, stride) A[(y)*(stride) + (x)]
#define lget2d(A, y, x, lstride) A[((y)<<(lstride)) + (x)] //lstride = log2(stride)

#define get3d(A, z, y, x, Sy, Sx) A[(((z)*(Sy)) + (y))*(Sx) + (x)]
#define lget3d(A, z, y, x, Sy, LSx)  A[((((z)*(Sy)) + y)<<(LSx)) + (x)]

#define get4d(A, w, z, y, x, Sz, Sy, Sx) A[(((w)*(Sz) + (z))*(Sy) + (y))*(Sx) + (x)]
#define lget4d(A, w, z, y, x, Sz, LSy, LSx) A[(((((w)*(Sz) + (z))<<(LSy)) + (y))<<(LSx)) + (x)]

#define simdMM4(c, av, b) {(c).x += (av) * (b).x; (c).y += (av) * (b).y; (c).z += (av) * (b).z; (c).w += (av) * (b).w;}
#define simdMM2(c, av, b) {(c).x += (av) * (b).x; (c).y += (av) * (b).y;} 

#define simdAdd4(c, av, b) {(c).x = (av) + (b).x; (c).y = (av) + (b).y; (c).z = (av) + (b).z; (c).w = (av) + (b).w;}

#define MAX_INT(x, y) ((x)^(((x)^(y))& -((x)<(y))))
#define MIN_INT(x, y) ((y)^(((x)^(y))& -((x)<(y))))


#define choose(flag, a, b) ((flag)*((a)-(b)) + (b))


#define FLOAT_MAX (3.4028235E38)
#define FLOAT_MIN (-3.4028234E38)


#define GET_GN(IC) (IC) //GN = IC
#define GET_GM(N, IH, IW) ((N)*(IH)*(IW)) //GM = N * IH * IW
#define GET_GK(FH, FW) ((FH)*(FW)) //GK = FH * FW

#define GET_GM_TILED(N,OH,OW)  ((N)*(OH)*(OW))


#define get_n_oh_ow(j, n, oh, ow)\
	int n, oh, ow;{n = j/OH_OW; int jr = j - n*OH_OW; oh = jr/OW; ow = jr - oh*OW;}

#define get_n_ih_iw(j, n, ih, iw) \
	int n, ih, iw; {n = j / IH_IW; int jr = j - n*IH_IW; ih = jr / IW, iw = jr - ih * IW; };

#define FIND_FHS_FWS(fhs, fws, tih, tiw, END) {\
	for (; fhs < FH; fhs++){\
		int oh = tih + fhs;\
		if (oh < 0 || oh >= OHp || oh % sh) continue;\
		for (fws = 0; fws < FW; fws++) {\
			int ow = tiw + fws;\
			if (ow >= 0 && ow < OWp && ow % sw == 0) goto END;\
		}}} END: 

#define get_alpha(oh, ow) {\
	int ih_min = oh * sh - ph, ih_max = ih_min + FH;\
	ih_min = MAX_INT(ih_min, 0); ih_max = MIN_INT(ih_max, IH);\
	int iw_min = ow * sw - pw, iw_max = iw_min + FW;\
	iw_min = MAX_INT(iw_min, 0); iw_max = MIN_INT(iw_max, IW);\
	alpha = 1.0f / ((ih_max - ih_min)*(iw_max - iw_min));}

#endif
