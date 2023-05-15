#pragma once


#define TEST_H
#ifndef TEST_H
#define TEST_H


#ifndef UTIL
#define UTIL

float* newRandomFloatVec(int length)//0-256
{
	float *p = new float[length];
	for (int i = 0; i < length; i++) p[i] = (float)(rand() % 1000) / 1000 + 1;
	return p;
}

float *newRandomFloatVec(int height, int width, int stride) 
{
	int lengthv = height * stride;
	float *p = new float[lengthv];
	int index = 0;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++)
			p[index++] = (float)(rand() % 1000) / 1000 + 1;
		for (int j = width; j < stride; j++)
			p[index++] = 0;
	}
	return p;

}

float *newDevFloatVec(int length)
{
	float *dp = NULL;
	size_t size = sizeof(float)*length;
	cudaError_t error = cudaMalloc((void**)&dp, size); printError(error);
	error = cudaMemset(dp, 0, size); printError(error);
	return dp;
}
float* newDevFloatVec(float *p, int length)
{
	float *dp = NULL;
	size_t size = sizeof(float)*length;
	cudaError_t error = cudaMalloc((void**)&dp, size); printError(error);
	error = cudaMemcpy(dp, p, size, cudaMemcpyHostToDevice); printError(error);
	return dp;
}
void println(float *p, int length)
{
	for (int i = 0; i < length; i++) cout << p[i] << ' ';
	cout << endl;
}
float samePercent(float *a, float *b, int length)
{
	int sum = 0;
	for (int i = 0; i < length; i++)
	{
		//if (b[i] == 0) cout << "zero: " << i << endl;
		if (a[i] == b[i]) { sum++; continue; }
		float dif = fabs(a[i] - b[i]);
		//float dif = fabs((a[i] - b[i]) / (a[i] + b[i]));
		if (dif < 1e-2) sum++;
		else {
			//if (a[i] < b[i]) cout << i << ":" << dif << " " << a[i] << ", " << b[i] << endl;
			//cout << i << ":" << dif << " " << a[i] << ", " << b[i] << endl;
		}

	}
	return 1.0f*sum / length;
}

float zeroPercent(float *a, int length)
{
	int sum = 0;
	for (int i = 0; i < length; i++)
		if (a[i] == 0) sum++;
	return 1.0f*sum / length;
}

float sum(float* a, int length)
{
	float r = 0;
	for (int i = 0; i < length; i++) r += a[i];
	return r;
}

#endif

#ifndef TRANSPOSE_TEST
#define TRANSPOSE_TEST

#ifndef TRANSPOSE4D_CPU
#define TRANSPOSE4D_CPU

void trans4D_12_CPU(
	const float *X,
	float *Y,
	int Xdim0, int Xdim1, int Xdim2, int Xdim3,
	int strideX, int strideY) 
{
	for (int d0 = 0; d0 < Xdim0; d0++)
		for (int d1 = 0; d1 < Xdim1; d1++)
			for (int d2 = 0; d2 < Xdim2; d2++)
				for (int d3 = 0; d3 < Xdim3; d3++)
					get4d(Y, d0, d2, d1, d3, Xdim2, Xdim1, strideY) =
					get4d(X, d0, d1, d2, d3, Xdim1, Xdim2, strideX);
}

void trans4D_13_CPU(
	const float *X,
	float *Y,
	int Xdim0, int Xdim1, int Xdim2, int Xdim3,
	int strideX, int strideY)
{
	for (int d0 = 0; d0 < Xdim0; d0++)
		for (int d1 = 0; d1 < Xdim1; d1++)
			for (int d2 = 0; d2 < Xdim2; d2++)
				for (int d3 = 0; d3 < Xdim3; d3++)
					get4d(Y, d0, d3, d2, d1, Xdim3, Xdim2, strideY) =
					get4d(X, d0, d1, d2, d3, Xdim1, Xdim2, strideX);
}

void trans4D_02_CPU(
	const float *X,
	float *Y,
	int Xdim0, int Xdim1, int Xdim2, int Xdim3,
	int strideX, int strideY)
{
	//Xdim2, Xdim1, Xdim0, Xdim3
	for (int d0 = 0; d0 < Xdim0; d0++)
		for (int d1 = 0; d1 < Xdim1; d1++)
			for (int d2 = 0; d2 < Xdim2; d2++)
				for (int d3 = 0; d3 < Xdim3; d3++)
					get4d(Y, d2, d1, d0, d3, Xdim1, Xdim0, strideY) =
					get4d(X, d0, d1, d2, d3, Xdim1, Xdim2, strideX);
}

void trans4D_03_CPU(
	const float *X,
	float *Y,
	int Xdim0, int Xdim1, int Xdim2, int Xdim3,
	int strideX, int strideY)
{
	//Xdim3, Xdim1, Xdim2, Xdim0
	for (int d0 = 0; d0 < Xdim0; d0++)
		for (int d1 = 0; d1 < Xdim1; d1++)
			for (int d2 = 0; d2 < Xdim2; d2++)
				for (int d3 = 0; d3 < Xdim3; d3++)
					get4d(Y, d3, d1, d2, d0, Xdim1, Xdim2, strideY) = 
					get4d(X, d0, d1, d2, d3, Xdim1, Xdim2, strideX);
}

void tranpose4d_CPU(
	const float *X,
	float *Y,
	int Xdim1, int Xdim2, int Xdim3,
	int Ydim1, int Ydim2, int Ydim3,
	int dimIndex2, int dimIndex1,
	int strideX, int strideY, int length)
{
	int Xdim23  =         Xdim2 * Xdim3;
	int Xdim123 = Xdim1 * Xdim2 * Xdim3;

	int x[4];
	for (int i = 0; i < length; i++) {
		int xoffset = i;
		x[0] = xoffset / Xdim123; int xoffset_res = xoffset % Xdim123;
		x[1] = xoffset_res / Xdim23; xoffset_res %= Xdim23;
		x[2] = xoffset_res / Xdim3;
		x[3] = xoffset_res % Xdim3;

		int t = x[dimIndex1]; x[dimIndex1] = x[dimIndex2]; x[dimIndex2] = t;
		int yoffset = ((x[0] * Ydim1 + x[1])*Ydim2 + x[2])*Ydim3 + x[3];

		//consider the mem alignment
		xoffset = (xoffset / Xdim3)*strideX + (xoffset % Xdim3);
		yoffset = (yoffset / Ydim3)*strideY + (yoffset % Ydim3);

		Y[yoffset] = X[xoffset];
	}
}

#endif

void prove(
	int dim0, int dim1, int dim2, int dim3,
	int dimIndex1, int dimIndex2)
{
	cout << "Xdim: " << dim0 << ',' << dim1 << ',' << dim2 << ',' << dim3 << endl;

	int length = dim0 * dim1 * dim2 * dim3;
	int height = dim0 * dim1 * dim2;
	int strideX = (dim3 + 3) >> 2 << 2;

	float *X = newRandomFloatVec(height, dim3, strideX);
	cout << "X = "; println(X, 10);

	//exchange dims-------------------------------------------------------------
	int dim[4]{ dim0, dim1, dim2, dim3 };
	int t = dim[dimIndex1]; dim[dimIndex1] = dim[dimIndex2]; dim[dimIndex2] = t;
	int Ydim0 = dim[0], Ydim1 = dim[1], Ydim2 = dim[2], Ydim3 = dim[3];
	cout << "Ydim: " << Ydim0 << ',' << Ydim1 << ',' << Ydim2 << ',' << Ydim3 << endl;

	int strideY = (Ydim3 + 3) >> 2 << 2;
	int lengthY = Ydim0 * Ydim1 * Ydim2 * strideY;
	//exchange dims-------------------------------------------------------------

	//general transpose---------------------------------------------------------
	float *Y1 = new float[lengthY]; memset(Y1, 0, lengthY * sizeof(float));
	float *Y2 = new float[lengthY]; memset(Y2, 0, lengthY * sizeof(float));

	tranpose4d_CPU(X, Y1,
		dim1, dim2, dim3,
		Ydim1, Ydim2, Ydim3,
		dimIndex1, dimIndex2,
		strideX, strideY,
		length);
	
	cout << "Y1 = "; println(Y1, 10);

	//trans4D_12_CPU(X, Y2, dim0, dim1, dim2, dim3, strideX, strideY);//trans(1, 2)
	//trans4D_13_CPU(X, Y2, dim0, dim1, dim2, dim3, strideX, strideY);
	trans4D_02_CPU(X, Y2, dim0, dim1, dim2, dim3, strideX, strideY);
	//trans4D_03_CPU(X, Y2, dim0, dim1, dim2, dim3, strideX, strideY);
	cout << "Y2 = "; println(Y2, 10);

	//compare
	float sp = samePercent(Y1, Y2, lengthY);
	cout << "sp = " << sp << endl;
}


void testCorrect_transpose(
	int dim0, int dim1, int dim2, int dim3,
	int dimIndex1, int dimIndex2)
{
	cout << "testCorrect:" << endl;
	cout << "Xdim: " << dim0 << ',' << dim1 << ',' << dim2 << ',' << dim3 << endl;

	int length = dim0 * dim1 * dim2 * dim3;
	int height = dim0 * dim1 * dim2;
	int strideX = (dim3 + 3) >> 2 << 2;
	int lengthXv = dim0 * dim1 * dim2 * strideX;

	float *X = newRandomFloatVec(height, dim3, strideX);

	//exchange dims-------------------------------------------------------------
	int dim[4]{ dim0, dim1, dim2, dim3 };
	int t = dim[dimIndex1]; dim[dimIndex1] = dim[dimIndex2]; dim[dimIndex2] = t;
	int Ydim0 = dim[0], Ydim1 = dim[1], Ydim2 = dim[2], Ydim3 = dim[3];
	cout << "Ydim: " << Ydim0 << ',' << Ydim1 << ',' << Ydim2 << ',' << Ydim3 << endl;
	int strideY = (Ydim3 + 3) >> 2 << 2;
	int lengthYv = Ydim0 * Ydim1 * Ydim2 * strideY;
	
	//CPU---------------------------------------------------------
	float *Y1 = new float[lengthYv]; memset(Y1, 0, lengthYv * sizeof(float));

	tranpose4d_CPU(X, Y1,
		dim1, dim2, dim3,
		Ydim1, Ydim2, Ydim3,
		dimIndex1, dimIndex2,
		strideX, strideY,
		length);

	cout << "Y1 = "; println(Y1, 15);
	

	//GPU-----------------------------------------------------
	float *dX = newDevFloatVec(X, lengthXv);
	float *dY = newDevFloatVec(lengthYv);

	__transpose4d(NULL,
		dX, dY, 
		dim1, dim2, dim3,
		Ydim1, Ydim2, Ydim3,
		dimIndex2, dimIndex1,
		strideX, strideY,
		length);

	cudaError_t error = cudaDeviceSynchronize(); printError(error);
	float *Y2 = new float[lengthYv]; memset(Y2, 0, lengthYv * sizeof(float));
	error = cudaMemcpy(Y2, dY, sizeof(float)*lengthYv, cudaMemcpyDeviceToHost); printError(error);
	cout << "Y2 = "; println(Y2, 30);
	

	//compare------------------------------------------------------
	float zpY1 = zeroPercent(Y1, lengthYv); cout << "zpY1 = " << zpY1 << endl;
	float zpY2 = zeroPercent(Y2, lengthYv); cout << "zpY2 = " << zpY2 << endl;
	float sp = samePercent(Y1, Y2, lengthYv); cout << "sp = " << sp << endl;
	
	delete X;
	delete Y1;
	delete Y2;

	error = cudaFree(dX); printError(error);
	error = cudaFree(dY); printError(error);

	if (sp != 1) exit(2);
	cout << endl;
}


void testSpeed_transpose(int nIter,
	int dim0, int dim1, int dim2, int dim3,
	int dimIndex1, int dimIndex2)
{
	cout << "testSpeed:" << endl;
	cout << "Xdim: " << dim0 << ',' << dim1 << ',' << dim2 << ',' << dim3 << endl;

	int length = dim0 * dim1 * dim2 * dim3;
	int height = dim0 * dim1 * dim2;
	int strideX = (dim3 + 3) >> 2 << 2;
	int lengthXv = dim0 * dim1 * dim2 * strideX;

	float *X = newRandomFloatVec(height, dim3, strideX);

	//exchange dims-------------------------------------------------------------
	int dim[4]{ dim0, dim1, dim2, dim3 };
	int t = dim[dimIndex1]; dim[dimIndex1] = dim[dimIndex2]; dim[dimIndex2] = t;
	int Ydim0 = dim[0], Ydim1 = dim[1], Ydim2 = dim[2], Ydim3 = dim[3];
	cout << "Ydim: " << Ydim0 << ',' << Ydim1 << ',' << Ydim2 << ',' << Ydim3 << endl;
	int strideY = (Ydim3 + 3) >> 2 << 2;
	int lengthYv = Ydim0 * Ydim1 * Ydim2 * strideY;
	
	//---------------------------------------------------
	float *dX = newDevFloatVec(X, lengthXv);
	float *dY = newDevFloatVec(lengthYv);

	clock_t start = clock();
	for (int i = 0; i < nIter; i++)
	{
		__transpose4d(NULL,
			dX, dY,
			dim1, dim2, dim3,
			Ydim1, Ydim2, Ydim3,
			dimIndex1, dimIndex2,
			strideX, strideY,
			length);
	}
	cudaError_t error = cudaDeviceSynchronize(); printError(error);
	clock_t end = clock();

	int div = (end - start);
	float time = 1.0f * div / nIter;
	int data_size = length * 2 * sizeof(float);
	float speed = (1.0f *(data_size) / (1 << 30)) / (time*1e-3);
	cout << "Size = " << (1.0 * data_size/1024/1024) << ", "
		<< "Time = " << time << " mesc, "
		<< "Speed = " << speed << "GB/s" << endl;

	error = cudaFree(dX); printError(error);
	error = cudaFreeHost(dX);

	delete X;
}

#endif

#ifndef ROT180_TEST
#define ROT180_TEST

void rot180_CPU(const float* X, 
	int N, int IH, int IW, int IC, float* Y)
{
	for(int n=0; n<N; n++)
	for(int ih=0; ih<IH; ih++)
	for(int iw=0; iw<IW; iw++)
	for (int ic = 0; ic < IC; ic++)
	{
		int rih = IH - 1 - ih;
		int riw = IW - 1 - iw;
		get4d(Y, n, rih, riw, ic, IH, IW, IC) =
			get4d(X, n, ih, iw, ic, IH, IW, IC);
	}
}

void testCorrect_rot180(int N, int IH, int IW, int IC)
{
	cout << "testCorrect:" << endl;
	cout << "(N, IH, IW, IC) = (" << N << ", " << IH << ", " << IW << ", " << IC << ")" << endl;

	int length = N* IH*IW*IC;
	float *X = newRandomFloatVec(length);

	//CPU---------------------------------------------------------
	float *Y1 = new float[length]; memset(Y1, 0, length * sizeof(float));
	rot180_CPU(X, N, IH, IW, IC, Y1);
	cout << "Y1 = "; println(Y1, 15);


	//GPU-----------------------------------------------------
	float *dX = newDevFloatVec(X, length);
	float *dY = newDevFloatVec(length);

	__rot180(NULL, dX, dY, N, IH, IW, IC, length);

	cudaError_t error = cudaDeviceSynchronize(); printError(error);
	float *Y2 = new float[length]; memset(Y2, 0, length * sizeof(float));
	error = cudaMemcpy(Y2, dY, sizeof(float)*length, cudaMemcpyDeviceToHost); printError(error);
	cout << "Y2 = "; println(Y2, 10);

	//compare------------------------------------------------------
	float zpY1 = zeroPercent(Y1, length); cout << "zpY1 = " << zpY1 << endl;
	float zpY2 = zeroPercent(Y2, length); cout << "zpY2 = " << zpY2 << endl;
	float sp = samePercent(Y1, Y2, length); cout << "sp = " << sp << endl;

	delete X;
	delete Y1;
	delete Y2;

	error = cudaFree(dX); printError(error);
	error = cudaFree(dY); printError(error);

	if (sp != 1) exit(2);
	cout << endl;
}

#endif

void test()
{
	//3, 4, 2, 3
	//12 * 8 = 96
	//for (int dim1 = 1; dim1 <= 10; dim1++)
	//		for (int dim2 = 1; dim2 <= 10; dim2++)
	//			for (int dim3 = 1; dim3 <= 10; dim3++)
	//				testCorrect_transpose(10, dim1, dim2, dim3, 2, 3);

	//testCorrect(10, 1, 8, 3, 2, 3);

	//[n, OH*OW, head, d_k]
	//int dim0 = 64, dim1 = 16 * 16, dim2 = 8, dim3 = 32;
	//testCorrect(dim0, dim1, dim2, dim3, 2, 3);
	//testSpeed(500, dim0, dim1, dim2, dim3, 2, 3);

	testCorrect_rot180(32, 32, 32, 32);
}

main() { test(); }

#endif

