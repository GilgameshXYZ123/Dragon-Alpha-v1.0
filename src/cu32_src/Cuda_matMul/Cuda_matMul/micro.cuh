#pragma once

#ifndef MICRO_H
#define MICRO_H

int LOG2(int n) {
	int result = 0;
	if (n & 0xffff0000) { result += 16; n >>= 16; }
	if (n & 0x0000ff00) { result += 8; n >>= 8; }
	if (n & 0x000000f0) { result += 4; n >>= 4; }
	if (n & 0x0000000c) { result += 2; n >>= 2; }
	if (n & 0x00000002) { result += 1; n >>= 1; }
	return result;
}

#ifndef COMMON_MICRO
#define COMMON_MICRO

#define IF_int(flag, a, b) ( ((-flag) & (a - b)) + b)

#define get(A, i, j, stride) A[(i)*(stride) + (j)] //get2d
#define lget(A, i, j, lstride) A[((i)<<(lstride)) + (j)] //lget2d, lstride = log2(stride)

#define simdMM4(c, av, b) {(c).x += av * b.x; (c).y += av * b.y; (c).z += av * b.z; (c).w += av * b.w;}
#define simdMM2(c, av, b) {(c).x += av * b.x; (c).y += av * b.y;}

#define Mul4(c, av, b) {(c).x = (av) * (b).x; (c).y = (av) * (b).y; (c).z = (av) * (b).z; (c).w = (av) * (b).w;}
#define Mul2(c, av, b) {(c).x = (av) * (b).x; (c).y = (av) * (b).y; }

#define next_index(index, length) ((index) + 1)%(length)

#endif

//GZ = gridDim.z
//GK = N * OH * OW, so: GK % 4 == 0
//GK_slice = (GK / gridDim.z) >> 3 << 3
//GK = GK_slice * gridDim.z + RGK
//As: GK % 8 == 0
//So: RGK % 4 == 0
//if: GK % 8 == 0, We have RGK % 8 == 0s
#define SK_K_slice(K, GZ)  ((K / GZ) >> 3 << 3)

#endif