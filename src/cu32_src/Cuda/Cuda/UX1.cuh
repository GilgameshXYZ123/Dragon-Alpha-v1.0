


#ifndef W3P4_V0
#define W3P4_V0

void padd_to_stride_int8_4x(const char*  A, char *B,
	int height, int width, int stride)
{
	int width4 = width >> 2 << 2;
	for (int i = 0; i < height; i++)
	{
		int *a64 = (int*)A;
		int *b64 = (int*)B;

		int j = 0;
		for (; j < width4; j += 4) *(b64++) = *(a64++);

		A = (char*)a64;
		B = (char*)b64;

		for (; j < width; j++) *(B++) = *(A++);
		for (j = width; j < stride; j++) *(B++) = 0;
	}
}

#endif


//width = 3, stride = 4, padding = 1
#ifndef W3P4_V
#define W3P4_V

void pad2stride_int8_w3_p1(const char*  A, char *B, int height)
{
	char buf[4] = { 0, 0, 0, 0 };//{c0, c1, c2, 0}
	for (int i = 0; i < height; i++)
	{
		*(int*)(buf) = *(int*)(A);//read c0, c1, c2, x
		buf[3] = 0;
		*(int*)(B) = *(int*)(buf);//write: {c0, c1, c2, 0}
		A += 3; B += 4;
	}
}

#endif


//read  4 line: 4*3 = 12
//write 4 line: 4*4 = 16
//widthx = 3, stride = 4, padding = 1, height % 4 == 0
#ifndef W3P4_V1
#define W3P4_V1

void pad2stride_int8_w3_p1_h4x(const char*  A, char *B, int height)
{
	char v0[4], v1[4], v2[4];
	char buf0[4] = { 0, 0, 0, 0 };//{c0, c1, c2, 0}
	char buf1[4] = { 0, 0, 0, 0 };//{c0, c1, c2, 0}
	char buf2[4] = { 0, 0, 0, 0 };//{c0, c1, c2, 0}
	char buf3[4] = { 0, 0, 0, 0 };//{c0, c1, c2, 0}
	for (int i = 0; i < height; i += 4)
	{
		*(int*)(v0) = *(int*)(A);//c0, c1,  c2,  c3
		*(int*)(v1) = *(int*)(A + 4);//c4, c5,  c6,  c7
		*(int*)(v2) = *(int*)(A + 8);//c8, c9, c10, c11

		buf0[0] = v0[0]; buf0[1] = v0[1]; buf0[2] = v0[2];
		buf1[0] = v0[3]; buf1[1] = v1[0]; buf1[2] = v1[1];
		buf2[0] = v1[2]; buf2[1] = v1[3]; buf2[2] = v2[0];
		buf3[0] = v2[1]; buf3[1] = v2[2]; buf3[2] = v2[3];

		*(int*)(B) = *(int*)(buf0);
		*(int*)(B + 4) = *(int*)(buf1);
		*(int*)(B + 8) = *(int*)(buf2);
		*(int*)(B + 12) = *(int*)(buf3);
		A += 12; B += 16;
	}
}

#endif


//read  4 line: 4*3 = 12
//write 4 line: 4*4 = 16
//widthx = 3, stride = 4, padding = 1, height % 4 == 0
#ifndef W3P4_V2
#define W3P4_V2

void pad2stride_int8_w3s4_v2(const char* A, char *B, int height)
{
	int* A_int32 = (int*)A;
	int* B_int32 = (int*)B;

	char v0[4], v1[4], v2[4];
	char buf0[4] = { 0, 0, 0, 0 };//{c0,  c1,  c2, 0}
	char buf1[4] = { 0, 0, 0, 0 };//{c3,  c4,  c5, 0}
	char buf2[4] = { 0, 0, 0, 0 };//{c6,  c7,  c8, 0}
	char buf3[4] = { 0, 0, 0, 0 };//{c9, c10, c11, 0}

	for (int i = 0; i < height; i += 4)
	{
		*(int*)(v0) = *(A_int32++);//c0, c1,  c2,  c3, offset += 32bit
		*(int*)(v1) = *(A_int32++);//c4, c5,  c6,  c7, offset += 32bit
		*(int*)(v2) = *(A_int32++);//c8, c9, c10, c11, offset += 32bit

		buf0[0] = v0[0]; buf0[1] = v0[1]; buf0[2] = v0[2];
		buf1[0] = v0[3]; buf1[1] = v1[0]; buf1[2] = v1[1];
		buf2[0] = v1[2]; buf2[1] = v1[3]; buf2[2] = v2[0];
		buf3[0] = v2[1]; buf3[1] = v2[2]; buf3[2] = v2[3];

		*(B_int32++) = *(int*)(buf0);//{c0,  c1,  c2, 0}, offset += 32bit
		*(B_int32++) = *(int*)(buf1);//{c3,  c4,  c5, 0}, offset += 32bit
		*(B_int32++) = *(int*)(buf2);//{c6,  c7,  c8, 0}, offset += 32bit
		*(B_int32++) = *(int*)(buf3);//{c9, c10, c11, 0}, offset += 32bit
	}
}

#endif


//read  4 line: 4*3 = 12
//write 4 line: 4*4 = 16
//widthx = 3, stride = 4, padding = 1, height % 4 == 0
#ifndef W3P4_V3
#define W3P4_V3

void pad2stride_int8_w3s4_v3(const char* A, char *B, int height)
{
	int* A_int32 = (int*)A;
	int* B_int32 = (int*)B;

	char rbuf[12], wbuf[16] = { 0 };
	for (int i = 0; i < height; i += 4)//4*4 = 16
	{
		*(int*)(rbuf    ) = *(A_int32++);//c0, c1,  c2,  c3, offset += 32bit
		*(int*)(rbuf + 4) = *(A_int32++);//c4, c5,  c6,  c7, offset += 32bit
		*(int*)(rbuf + 8) = *(A_int32++);//c8, c9, c10, c11, offset += 32bit

		wbuf[0] = rbuf[0];//0
		wbuf[1] = rbuf[1];
		wbuf[2] = rbuf[2];

		wbuf[4] = rbuf[3];//4
		wbuf[5] = rbuf[4];
		wbuf[6] = rbuf[5];

		wbuf[8]  = rbuf[6];//8
		wbuf[9]  = rbuf[7];
		wbuf[10] = rbuf[8];

		wbuf[12] =  rbuf[9];//12
		wbuf[13] = rbuf[10];
		wbuf[14] = rbuf[11];

		*(B_int32++) = *(int*)(wbuf    );//{c0,  c1,  c2, 0}, offset += 32bit
		*(B_int32++) = *(int*)(wbuf + 4);//{c3,  c4,  c5, 0}, offset += 32bit
		*(B_int32++) = *(int*)(wbuf + 8);//{c6,  c7,  c8, 0}, offset += 32bit
		*(B_int32++) = *(int*)(wbuf + 12);//{c9, c10, c11, 0}, offset += 32bit
	}
}

#endif


//read  4 line: 4*3 = 12
//write 4 line: 4*4 = 16
//widthx = 3, stride = 4, padding = 1, height % 4 == 0
#ifndef W3P4_V4
#define W3P4_V4

void pad2stride_int8_w3s4_v4(const char* A, char *B, int height)
{
	int* A_int32 = (int*)A;
	int* B_int32 = (int*)B;

	char rbuf[12], wbuf[16] = { 0 };
	for (int i = 0; i < height; i += 4)//4*4 = 16
	{
		*(int*)(rbuf) = *(A_int32++);//c0, c1,  c2,  c3, offset += 32bit
		*(int*)(rbuf + 4) = *(A_int32++);//c4, c5,  c6,  c7, offset += 32bit
		*(int*)(rbuf + 8) = *(A_int32++);//c8, c9, c10, c11, offset += 32bit

		*(char3*)(wbuf     ) = *(char3*)(rbuf    );
		*(char3*)(wbuf +  4) = *(char3*)(rbuf + 3);
		*(char3*)(wbuf +  8) = *(char3*)(rbuf + 6);
		*(char3*)(wbuf + 12) = *(char3*)(rbuf + 9);

		*(B_int32++) = *(int*)(wbuf);//{c0,  c1,  c2, 0}, offset += 32bit
		*(B_int32++) = *(int*)(wbuf + 4);//{c3,  c4,  c5, 0}, offset += 32bit
		*(B_int32++) = *(int*)(wbuf + 8);//{c6,  c7,  c8, 0}, offset += 32bit
		*(B_int32++) = *(int*)(wbuf + 12);//{c9, c10, c11, 0}, offset += 32bit
	}
}

#endif



//read  4 line: 4*3 = 12
//write 4 line: 4*4 = 16
//widthx = 3, stride = 4, padding = 1, height % 4 == 0
#ifndef W3P4_V5
#define W3P4_V5

void pad2stride_int8_w3s4_v5(const char* A, char *B, int height)
{
	int* A_int32 = (int*)A;
	int* B_int32 = (int*)B;

	char rbuf[12], wbuf[16] = { 0 };
	for (int i = 0; i < height; i += 4)//4*4 = 16
	{
		*(int*)(rbuf) = *(A_int32++);//c0, c1,  c2,  c3, offset += 32bit
		*(int*)(rbuf + 4) = *(A_int32++);//c4, c5,  c6,  c7, offset += 32bit
		*(int*)(rbuf + 8) = *(A_int32++);//c8, c9, c10, c11, offset += 32bit

		*(char_3*)(wbuf     ) = *(char_3*)(rbuf    );
		*(char_3*)(wbuf +  4) = *(char_3*)(rbuf + 3);
		*(char_3*)(wbuf +  8) = *(char_3*)(rbuf + 6);
		*(char_3*)(wbuf + 12) = *(char_3*)(rbuf + 9);

		*(B_int32++) = *(int*)(wbuf);//{c0,  c1,  c2, 0}, offset += 32bit
		*(B_int32++) = *(int*)(wbuf + 4);//{c3,  c4,  c5, 0}, offset += 32bit
		*(B_int32++) = *(int*)(wbuf + 8);//{c6,  c7,  c8, 0}, offset += 32bit
		*(B_int32++) = *(int*)(wbuf + 12);//{c9, c10, c11, 0}, offset += 32bit
	}
}

#endif


//read  4 line: 4*3 = 12
//write 4 line: 4*4 = 16
//widthx = 3, stride = 4, padding = 1, height % 4 == 0
#ifndef W3P4_V6
#define W3P4_V6

void pad2stride_int8_w3s4_v6(const char* A, char *B, int height)
{
	int* A_int32 = (int*)A;
	int* B_int32 = (int*)B;

	char rbuf[12];
	char_4 wbuf[4]; memset(wbuf, 0, sizeof(char_4) * 4);
	for (int i = 0; i < height; i += 4)//4*4 = 16
	{
		*(int*)(rbuf) = *(A_int32++);//c0, c1,  c2,  c3, offset += 32bit
		*(int*)(rbuf + 4) = *(A_int32++);//c4, c5,  c6,  c7, offset += 32bit
		*(int*)(rbuf + 8) = *(A_int32++);//c8, c9, c10, c11, offset += 32bit

		*(char_3*)(wbuf + 0) = *(char_3*)(rbuf    );
		*(char_3*)(wbuf + 1) = *(char_3*)(rbuf + 3);
		*(char_3*)(wbuf + 2) = *(char_3*)(rbuf + 6);
		*(char_3*)(wbuf + 3) = *(char_3*)(rbuf + 9);

		*(B_int32++) = *(int*)(wbuf + 0);//{c0,  c1,  c2, 0}, offset += 32bit
		*(B_int32++) = *(int*)(wbuf + 1);//{c3,  c4,  c5, 0}, offset += 32bit
		*(B_int32++) = *(int*)(wbuf + 2);//{c6,  c7,  c8, 0}, offset += 32bit
		*(B_int32++) = *(int*)(wbuf + 3);//{c9, c10, c11, 0}, offset += 32bit
	}
}

#endif


//read  4 line: 4*3 = 12
//write 4 line: 4*4 = 16
//widthx = 3, stride = 4, padding = 1, height % 8 == 0
#ifndef W3P4_V7
#define W3P4_V7

void pad2stride_int8_w3s4_v7(const char* A, char *B, int height)
{
	int64_t* A_int64 = (int64_t*)A;
	int64_t* B_int64 = (int64_t*)B;

	char   rbuf[24];//3*8 = 24 elements
	char_4 wbuf[8];//4*8 = 32 elements
	
	memset(wbuf, 0, sizeof(char_4) * 8);
	for (int i = 0; i < height; i += 8)//4*4 = 16
	{
		*(int64_t*)(rbuf     ) = *(A_int64++);//c0  -> c7 , offset += 64bit
		*(int64_t*)(rbuf +  8) = *(A_int64++);//c8  -> c15, offset += 64bit
		*(int64_t*)(rbuf + 16) = *(A_int64++);//c16 -> c23, offset += 64bit

		*(char_3*)(wbuf + 0) = *(char_3*)(rbuf     );
		*(char_3*)(wbuf + 1) = *(char_3*)(rbuf +  3);
		*(char_3*)(wbuf + 2) = *(char_3*)(rbuf +  6);
		*(char_3*)(wbuf + 3) = *(char_3*)(rbuf +  9);
		*(char_3*)(wbuf + 4) = *(char_3*)(rbuf + 12);
		*(char_3*)(wbuf + 5) = *(char_3*)(rbuf + 15);
		*(char_3*)(wbuf + 6) = *(char_3*)(rbuf + 18);
		*(char_3*)(wbuf + 7) = *(char_3*)(rbuf + 21);

		*(B_int64++) = *(int64_t*)(wbuf    );//offset += 64bit
		*(B_int64++) = *(int64_t*)(wbuf + 2);//offset += 64bit
		*(B_int64++) = *(int64_t*)(wbuf + 4);//offset += 64bit
		*(B_int64++) = *(int64_t*)(wbuf + 6);//offset += 64bit
	}
}

#endif



//read  4 line: 4*3 = 12
//write 4 line: 4*4 = 16
//widthx = 3, stride = 4, padding = 1, height % 16 == 0
#ifndef W3P4_V8
#define W3P4_V8

void pad2stride_int8_w3s4_v8(const char* A, char *B, int height)
{
	int_4* A_128bit = (int_4*)A;
	int_4* B_128bit = (int_4*)B;

	char    rbuf[48];//3*16 = 48 elements
	char_4 wbuf[16];//4*16 = 64 elements
	
	memset(wbuf, 0, sizeof(char_4) * 16);
	for (int i = 0; i < height; i += 16)//4*4 = 16
	{
		*(int_4*)(rbuf     ) = *(A_128bit++);//c0  -> c15, offset += 128bit
		*(int_4*)(rbuf + 16) = *(A_128bit++);//c16 -> c31, offset += 128bit
		*(int_4*)(rbuf + 32) = *(A_128bit++);//c32 -> c47, offset += 128bit

		*(char_3*)(wbuf +  0) = *(char_3*)(rbuf     );
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

		*(B_128bit++) = *(int_4*)(wbuf     );//offset += 128bit
		*(B_128bit++) = *(int_4*)(wbuf +  4);//offset += 128bit
		*(B_128bit++) = *(int_4*)(wbuf +  8);//offset += 128bit
		*(B_128bit++) = *(int_4*)(wbuf + 12);//offset += 128bit
	}
}

#endif


//read  4 line: 4*3 = 12
//write 4 line: 4*4 = 16
//widthx = 3, stride = 4, padding = 1, height % 32 == 0
#ifndef W3P4_V9
#define W3P4_V9

void pad2stride_int8_w3s4_v9(const char* A, char *B, int height)
{
	int_8* A_256bit = (int_8*)A;
	int_8* B_256bit = (int_8*)B;

	char    rbuf[96];//3*32 =  96 elements
	char_4 wbuf[32];//4*32 = 128 elements
	
	memset(wbuf, 0, sizeof(char_4)*32);
	for (int i = 0; i < height; i += 32)//3*32 = 96 element -> 4*32 = 128 element
	{
		*(int_8*)(rbuf     ) = *(A_256bit++);//c0  -> c31, offset += 256bit
		*(int_8*)(rbuf + 32) = *(A_256bit++);//c32 -> c63, offset += 256bit
		*(int_8*)(rbuf + 64) = *(A_256bit++);//c64 -> c95, offset += 256bit
		
		*(char_3*)(wbuf +  0) = *(char_3*)(rbuf     );
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

		*(char_3*)(wbuf + 16) = *(char_3*)(rbuf + 48);
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

		*(B_256bit++) = *(int_8*)(wbuf     );//offset += 256bit
		*(B_256bit++) = *(int_8*)(wbuf +  8);//offset += 256bit
		*(B_256bit++) = *(int_8*)(wbuf + 16);//offset += 256bit
		*(B_256bit++) = *(int_8*)(wbuf + 24);//offset += 256bit
	}
}

#endif


//read  4 line: 4*3 = 12
//write 4 line: 4*4 = 16
//widthx = 3, stride = 4, padding = 1, height % 32 == 0
#ifndef W3P4_V10
#define W3P4_V10

void pad2stride_int8_w3s4_v10(const char* A, char *B, int height)
{
	int_8* A_256bit = (int_8*)A;
	int_8* B_256bit = (int_8*)B;

	char    rbuf[96];//3*32 =  96 elements
	char_4 wbuf[ 8];//4*32 = 128 elements
	
	memset(wbuf, 0, sizeof(char_4) * 8);
	for (int i = 0; i < height; i += 32)//3*32 = 96 element -> 4*32 = 128 element
	{
		*(int_8*)(rbuf     ) = *(A_256bit++);//c0  -> c31, offset += 256bit
		*(int_8*)(rbuf + 32) = *(A_256bit++);//c32 -> c63, offset += 256bit
		*(int_8*)(rbuf + 64) = *(A_256bit++);//c64 -> c95, offset += 256bit

		*(char_3*)(wbuf + 0) = *(char_3*)(rbuf     );
		*(char_3*)(wbuf + 1) = *(char_3*)(rbuf +  3);
		*(char_3*)(wbuf + 2) = *(char_3*)(rbuf +  6);
		*(char_3*)(wbuf + 3) = *(char_3*)(rbuf +  9);
		*(char_3*)(wbuf + 4) = *(char_3*)(rbuf + 12);
		*(char_3*)(wbuf + 5) = *(char_3*)(rbuf + 15);
		*(char_3*)(wbuf + 6) = *(char_3*)(rbuf + 18);
		*(char_3*)(wbuf + 7) = *(char_3*)(rbuf + 21);
		*(B_256bit++) = *(int_8*)(wbuf);//offset += 256bit

		*(char_3*)(wbuf + 0) = *(char_3*)(rbuf + 24);
		*(char_3*)(wbuf + 1) = *(char_3*)(rbuf + 27);
		*(char_3*)(wbuf + 2) = *(char_3*)(rbuf + 30);
		*(char_3*)(wbuf + 3) = *(char_3*)(rbuf + 33);
		*(char_3*)(wbuf + 4) = *(char_3*)(rbuf + 36);
		*(char_3*)(wbuf + 5) = *(char_3*)(rbuf + 39);
		*(char_3*)(wbuf + 6) = *(char_3*)(rbuf + 42);
		*(char_3*)(wbuf + 7) = *(char_3*)(rbuf + 45);
		*(B_256bit++) = *(int_8*)(wbuf);//offset += 256bit

		*(char_3*)(wbuf + 0) = *(char_3*)(rbuf + 48);
		*(char_3*)(wbuf + 1) = *(char_3*)(rbuf + 51);
		*(char_3*)(wbuf + 2) = *(char_3*)(rbuf + 54);
		*(char_3*)(wbuf + 3) = *(char_3*)(rbuf + 57);
		*(char_3*)(wbuf + 4) = *(char_3*)(rbuf + 60);
		*(char_3*)(wbuf + 5) = *(char_3*)(rbuf + 63);
		*(char_3*)(wbuf + 6) = *(char_3*)(rbuf + 66);
		*(char_3*)(wbuf + 7) = *(char_3*)(rbuf + 69);
		*(B_256bit++) = *(int_8*)(wbuf);//offset += 256bit

		*(char_3*)(wbuf + 0) = *(char_3*)(rbuf + 72);
		*(char_3*)(wbuf + 1) = *(char_3*)(rbuf + 75);
		*(char_3*)(wbuf + 2) = *(char_3*)(rbuf + 78);
		*(char_3*)(wbuf + 3) = *(char_3*)(rbuf + 81);
		*(char_3*)(wbuf + 4) = *(char_3*)(rbuf + 84);
		*(char_3*)(wbuf + 5) = *(char_3*)(rbuf + 87);
		*(char_3*)(wbuf + 6) = *(char_3*)(rbuf + 90);
		*(char_3*)(wbuf + 7) = *(char_3*)(rbuf + 93);
		*(B_256bit++) = *(int_8*)(wbuf);//offset += 256bit
	}
}

#endif


