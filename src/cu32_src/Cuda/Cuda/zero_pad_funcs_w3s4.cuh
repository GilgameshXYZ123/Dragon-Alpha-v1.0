#pragma once

#ifndef ZERO_PAD_FUNCS_W3S4_H
#define ZERO_PAD_FUNCS_W3S4_H

//[1] W3: width  = 3
//[2] S4: stride = 4
//designed for pictures(JPG) with 3 channels but padding to 4

//height % 64 == 0
#ifndef ZERO_PAD_W3S4_Hx64
#define ZERO_PAD_W3S4_Hx64

void __zeroPad_W3S4_Hx64(const char *A, char *B, int height) 
{
	int_16* A_512bit = (int_16*)A;//512 bits
	int_16* B_512bit = (int_16*)B;

	char  rbuf[192];//3*64 = 192 elems
	char_4 wbuf[64];//4*64 = 256 elems
	memset(wbuf, 0, sizeof(char_4) * 64);

	for (int i = 0; i < height; i += 64)//3*64 = 192 element -> 4*64 = 256 element
	{
		*(int_16*)(rbuf      ) = *(A_512bit++);//  c0 ->  c63, offset += 512bit
		*(int_16*)(rbuf +  64) = *(A_512bit++);// c64 -> c127, offset += 512bit
		*(int_16*)(rbuf + 128) = *(A_512bit++);//c128 -> c255, offset += 512bit

		*(char_3*)(wbuf +  0) = *(char_3*)(rbuf     );//0-47
		*(char_3*)(wbuf +  1) = *(char_3*)(rbuf +  3);
		*(char_3*)(wbuf +  2) = *(char_3*)(rbuf +  6);
		*(char_3*)(wbuf +  3) = *(char_3*)(rbuf +  9);
		*(char_3*)(wbuf +  4) = *(char_3*)(rbuf + 12);
		*(char_3*)(wbuf +  5) = *(char_3*)(rbuf + 15);
		*(char_3*)(wbuf +  6) = *(char_3*)(rbuf + 18);
		*(char_3*)(wbuf +  7) = *(char_3*)(rbuf + 21);
		*(char_3*)(wbuf +  8) = *(char_3*)(rbuf + 24);
		*(char_3*)(wbuf +  9) = *(char_3*)(rbuf + 27);
		*(char_3*)(wbuf + 10) = *(char_3*)(rbuf + 30);
		*(char_3*)(wbuf + 11) = *(char_3*)(rbuf + 33);
		*(char_3*)(wbuf + 12) = *(char_3*)(rbuf + 36);
		*(char_3*)(wbuf + 13) = *(char_3*)(rbuf + 39);
		*(char_3*)(wbuf + 14) = *(char_3*)(rbuf + 42);
		*(char_3*)(wbuf + 15) = *(char_3*)(rbuf + 45);

		*(char_3*)(wbuf + 16) = *(char_3*)(rbuf + 48);//48-95
		*(char_3*)(wbuf + 17) = *(char_3*)(rbuf + 51);
		*(char_3*)(wbuf + 18) = *(char_3*)(rbuf + 54);
		*(char_3*)(wbuf + 19) = *(char_3*)(rbuf + 57);
		*(char_3*)(wbuf + 20) = *(char_3*)(rbuf + 60);
		*(char_3*)(wbuf + 21) = *(char_3*)(rbuf + 63);
		*(char_3*)(wbuf + 22) = *(char_3*)(rbuf + 66);
		*(char_3*)(wbuf + 23) = *(char_3*)(rbuf + 69);
		*(char_3*)(wbuf + 24) = *(char_3*)(rbuf + 72);
		*(char_3*)(wbuf + 25) = *(char_3*)(rbuf + 75);
		*(char_3*)(wbuf + 26) = *(char_3*)(rbuf + 78);
		*(char_3*)(wbuf + 27) = *(char_3*)(rbuf + 81);
		*(char_3*)(wbuf + 28) = *(char_3*)(rbuf + 84);
		*(char_3*)(wbuf + 29) = *(char_3*)(rbuf + 87);
		*(char_3*)(wbuf + 30) = *(char_3*)(rbuf + 90);
		*(char_3*)(wbuf + 31) = *(char_3*)(rbuf + 93);

		*(char_3*)(wbuf + 32) = *(char_3*)(rbuf + 96);//96-143
		*(char_3*)(wbuf + 33) = *(char_3*)(rbuf + 99);
		*(char_3*)(wbuf + 34) = *(char_3*)(rbuf + 102);
		*(char_3*)(wbuf + 35) = *(char_3*)(rbuf + 105);
		*(char_3*)(wbuf + 36) = *(char_3*)(rbuf + 108);
		*(char_3*)(wbuf + 37) = *(char_3*)(rbuf + 111);
		*(char_3*)(wbuf + 38) = *(char_3*)(rbuf + 114);
		*(char_3*)(wbuf + 39) = *(char_3*)(rbuf + 117);
		*(char_3*)(wbuf + 40) = *(char_3*)(rbuf + 120);
		*(char_3*)(wbuf + 41) = *(char_3*)(rbuf + 123);
		*(char_3*)(wbuf + 42) = *(char_3*)(rbuf + 126);
		*(char_3*)(wbuf + 43) = *(char_3*)(rbuf + 129);
		*(char_3*)(wbuf + 44) = *(char_3*)(rbuf + 132);
		*(char_3*)(wbuf + 45) = *(char_3*)(rbuf + 135);
		*(char_3*)(wbuf + 46) = *(char_3*)(rbuf + 138);
		*(char_3*)(wbuf + 47) = *(char_3*)(rbuf + 141);

		*(char_3*)(wbuf + 48) = *(char_3*)(rbuf + 144);//144-191
		*(char_3*)(wbuf + 49) = *(char_3*)(rbuf + 147);
		*(char_3*)(wbuf + 50) = *(char_3*)(rbuf + 150);
		*(char_3*)(wbuf + 51) = *(char_3*)(rbuf + 153);
		*(char_3*)(wbuf + 52) = *(char_3*)(rbuf + 156);
		*(char_3*)(wbuf + 53) = *(char_3*)(rbuf + 159);
		*(char_3*)(wbuf + 54) = *(char_3*)(rbuf + 162);
		*(char_3*)(wbuf + 55) = *(char_3*)(rbuf + 165);
		*(char_3*)(wbuf + 56) = *(char_3*)(rbuf + 168);
		*(char_3*)(wbuf + 57) = *(char_3*)(rbuf + 171);
		*(char_3*)(wbuf + 58) = *(char_3*)(rbuf + 174);
		*(char_3*)(wbuf + 59) = *(char_3*)(rbuf + 177);
		*(char_3*)(wbuf + 60) = *(char_3*)(rbuf + 180);
		*(char_3*)(wbuf + 61) = *(char_3*)(rbuf + 183);
		*(char_3*)(wbuf + 62) = *(char_3*)(rbuf + 186);
		*(char_3*)(wbuf + 63) = *(char_3*)(rbuf + 189);

		*(B_512bit++) = *(int_16*)(wbuf);//offset += 512bit
		*(B_512bit++) = *(int_16*)(wbuf + 16);//offset += 512bit
		*(B_512bit++) = *(int_16*)(wbuf + 32);//offset += 512bit
		*(B_512bit++) = *(int_16*)(wbuf + 48);//offset += 512bit
	}
}

#endif


//height % 32 == 0
#ifndef ZERO_PAD_W3S4_Hx32
#define ZERO_PAD_W3S4_Hx32

void __zeroPad_W3S4_Hx32(const char *A, char *B, int height)
{
	int_8* A_256bit = (int_8*)A;//256 bits
	int_8* B_256bit = (int_8*)B;

	char   rbuf[96];//3*32 =  96 elems
	char_4 wbuf[32];//4*32 = 128 elems
	memset(wbuf, 0, sizeof(char_4) * 32);

	for (int i = 0; i < height; i += 32)//3*32 = 96 element -> 4*32 = 128 element
	{
		*(int_8*)(rbuf     ) = *(A_256bit++);// c0 -> c31, offset += 256bit
		*(int_8*)(rbuf + 32) = *(A_256bit++);//c32 -> c63, offset += 256bit
		*(int_8*)(rbuf + 64) = *(A_256bit++);//c64 -> c95, offset += 256bit

		*(char_3*)(wbuf +  0) = *(char_3*)(rbuf     );//0 -> 23
		*(char_3*)(wbuf +  1) = *(char_3*)(rbuf +  3);
		*(char_3*)(wbuf +  2) = *(char_3*)(rbuf +  6);
		*(char_3*)(wbuf +  3) = *(char_3*)(rbuf +  9);
		*(char_3*)(wbuf +  4) = *(char_3*)(rbuf + 12);
		*(char_3*)(wbuf +  5) = *(char_3*)(rbuf + 15);
		*(char_3*)(wbuf +  6) = *(char_3*)(rbuf + 18);
		*(char_3*)(wbuf +  7) = *(char_3*)(rbuf + 21);

		*(char_3*)(wbuf +  8) = *(char_3*)(rbuf + 24);//24 -> 47
		*(char_3*)(wbuf +  9) = *(char_3*)(rbuf + 27);
		*(char_3*)(wbuf + 10) = *(char_3*)(rbuf + 30);
		*(char_3*)(wbuf + 11) = *(char_3*)(rbuf + 33);
		*(char_3*)(wbuf + 12) = *(char_3*)(rbuf + 36);
		*(char_3*)(wbuf + 13) = *(char_3*)(rbuf + 39);
		*(char_3*)(wbuf + 14) = *(char_3*)(rbuf + 42);
		*(char_3*)(wbuf + 15) = *(char_3*)(rbuf + 45);

		*(char_3*)(wbuf + 16) = *(char_3*)(rbuf + 48);//48 -> 71
		*(char_3*)(wbuf + 17) = *(char_3*)(rbuf + 51);
		*(char_3*)(wbuf + 18) = *(char_3*)(rbuf + 54);
		*(char_3*)(wbuf + 19) = *(char_3*)(rbuf + 57);
		*(char_3*)(wbuf + 20) = *(char_3*)(rbuf + 60);
		*(char_3*)(wbuf + 21) = *(char_3*)(rbuf + 63);
		*(char_3*)(wbuf + 22) = *(char_3*)(rbuf + 66);
		*(char_3*)(wbuf + 23) = *(char_3*)(rbuf + 69);

		*(char_3*)(wbuf + 24) = *(char_3*)(rbuf + 72);//72 -> 95
		*(char_3*)(wbuf + 25) = *(char_3*)(rbuf + 75);
		*(char_3*)(wbuf + 26) = *(char_3*)(rbuf + 78);
		*(char_3*)(wbuf + 27) = *(char_3*)(rbuf + 81);
		*(char_3*)(wbuf + 28) = *(char_3*)(rbuf + 84);
		*(char_3*)(wbuf + 29) = *(char_3*)(rbuf + 87);
		*(char_3*)(wbuf + 30) = *(char_3*)(rbuf + 90);
		*(char_3*)(wbuf + 31) = *(char_3*)(rbuf + 93);

		*(B_256bit++) = *(int_8*)(wbuf     );//offset += 256bit
		*(B_256bit++) = *(int_8*)(wbuf +  8);//offset += 256bit
		*(B_256bit++) = *(int_8*)(wbuf + 16);//offset += 256bit
		*(B_256bit++) = *(int_8*)(wbuf + 24);//offset += 256bit
	}
}

#endif


//height % 8 == 0
#ifndef ZERO_PAD_W3S4_Hx8
#define ZERO_PAD_W3S4_Hx8

void __zeroPad_W3S4_Hx8(const char *A, char *B, int height)
{
	int64_t* A_64bit = (int64_t*)A;//64 bits
	int64_t* B_64bit = (int64_t*)B;

	char  rbuf[24];//3*8 = 24 elems
	char_4 wbuf[8];//4*8 = 32 elems
	memset(wbuf, 0, sizeof(char_4) * 8);

	for (int i = 0; i < height; i += 8)//4*4 = 16
	{
		*(int64_t*)(rbuf     ) = *(A_64bit++);//c0  -> c7 , offset += 64bit
		*(int64_t*)(rbuf +  8) = *(A_64bit++);//c8  -> c15, offset += 64bit
		*(int64_t*)(rbuf + 16) = *(A_64bit++);//c16 -> c23, offset += 64bit

		*(char_3*)(wbuf + 0) = *(char_3*)(rbuf     );//0 -> 5
		*(char_3*)(wbuf + 1) = *(char_3*)(rbuf +  3);
		*(char_3*)(wbuf + 2) = *(char_3*)(rbuf +  6);//6 -> 11
		*(char_3*)(wbuf + 3) = *(char_3*)(rbuf +  9);
		*(char_3*)(wbuf + 4) = *(char_3*)(rbuf + 12);//12 -> 17
		*(char_3*)(wbuf + 5) = *(char_3*)(rbuf + 15);
		*(char_3*)(wbuf + 6) = *(char_3*)(rbuf + 18);//18 -> 25
		*(char_3*)(wbuf + 7) = *(char_3*)(rbuf + 21);

		*(B_64bit++) = *(int64_t*)(wbuf    );//offset += 64bit
		*(B_64bit++) = *(int64_t*)(wbuf + 2);//offset += 64bit
		*(B_64bit++) = *(int64_t*)(wbuf + 4);//offset += 64bit
		*(B_64bit++) = *(int64_t*)(wbuf + 6);//offset += 64bit
	}
}

#endif


//height % 4 == 0
#ifndef ZERO_PAD_W3S4_Hx4
#define ZERO_PAD_W3S4_Hx4

void __zeroPad_W3S4_Hx4(const char *A, char *B, int height)
{
	int* A_32bit = (int*)A;//32 bits
	int* B_32bit = (int*)B;

	char  rbuf[12];//3 * 4 = 12 elems
	char_4 wbuf[4];//4 * 4 = 16 elems
	memset(wbuf, 0, sizeof(char_4) * 4);

	for (int i = 0; i < height; i += 4)//4*4 = 16
	{
		*(int*)(rbuf    ) = *(A_32bit++);//c0 ->  c3, offset += 32bit
		*(int*)(rbuf + 4) = *(A_32bit++);//c4 ->  c7, offset += 32bit
		*(int*)(rbuf + 8) = *(A_32bit++);//c8 -> c11, offset += 32bit

		*(char_3*)(wbuf + 0) = *(char_3*)(rbuf    );//c0 ->  c2
		*(char_3*)(wbuf + 1) = *(char_3*)(rbuf + 3);//c3 ->  c5
		*(char_3*)(wbuf + 2) = *(char_3*)(rbuf + 6);//c6 ->  c8
		*(char_3*)(wbuf + 3) = *(char_3*)(rbuf + 9);//c9 -> c10

		*(B_32bit++) = *(int*)(wbuf + 0);//{c0,  c1,  c2, 0}, offset += 32bit
		*(B_32bit++) = *(int*)(wbuf + 1);//{c3,  c4,  c5, 0}, offset += 32bit
		*(B_32bit++) = *(int*)(wbuf + 2);//{c6,  c7,  c8, 0}, offset += 32bit
		*(B_32bit++) = *(int*)(wbuf + 3);//{c9, c10, c11, 0}, offset += 32bit
	}
}

#endif


#ifndef ZERO_PAD_W3S4_FUNCTION
#define ZERO_PAD_W3S4_FUNCTION

//height % 4 == 0
//B = zeroPad(A, width = 3, stride = 4)
void __zeroPad_W3S4(const char *A, char *B, int height)
{
	int height64 = (height >> 6) << 6;
	if (height64 > 0) {
		__zeroPad_W3S4_Hx64(A, B, height64);
		height &= 63; if (height == 0) return;
		A += (height64 * 3); B += (height64 << 2);//[3, 4]
	}

	int height32 = (height >> 5) << 5; 
	if (height32 > 0) {
		__zeroPad_W3S4_Hx32(A, B, height32);
		height &= 31; if (height == 0) return;
		A += (height32 * 3); B += (height32 << 2);//[3, 4]
	}

	int height8 = (height >> 3) << 3;
	if (height8 > 0) {
		__zeroPad_W3S4_Hx8(A, B, height);
		height &= 7; if (height == 0) return;
		A += (height8 * 3); B += (height8 << 2);//[3, 4]
	}

	int height4 = (height >> 2) << 2;
	if (height4 > 0) {
		__zeroPad_W3S4_Hx4(A, B, height);
		height &= 3; if (height == 0) return;
		A += (height4 * 3); B += (height4 << 2);//[3, 4]
	}

	//process the remainder
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < 3; j++) *(B++) = *(A++);
		*(B++) = 0;
	}
}

#endif

#endif