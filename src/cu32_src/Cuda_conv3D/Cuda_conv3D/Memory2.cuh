

//for k88R_ic2pow, k88R4_ic2pow
#ifndef LOADX_4_IC_2POW
#define LOADX_4_IC_2POW

__device__ __forceinline__ float4 LoadX4_ic2pow(const float* __restrict__ X,
	int X_k, int IH, int IW, int LIC, int FW_IC,
	int X0, int toh0, int tow0,
	int X1, int toh1, int tow1,
	int X2, int toh2, int tow2,
	int X3, int toh3, int tow3)
{
	int X_fh, X_fw; get_X_fh_fw_IC2pow(X_k, X_fh, X_fw);
	bool lx0 = (toh0 >= -X_fh) && (toh0 < IH - X_fh) && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
	bool lx1 = (toh1 >= -X_fh) && (toh1 < IH - X_fh) && (tow1 >= -X_fw) && (tow1 < IW - X_fw);
	bool lx2 = (toh2 >= -X_fh) && (toh2 < IH - X_fh) && (tow2 >= -X_fw) && (tow2 < IW - X_fw);
	bool lx3 = (toh3 >= -X_fh) && (toh3 < IH - X_fh) && (tow3 >= -X_fw) && (tow3 < IW - X_fw);
	int xoffset = ((X_fh * IW) << LIC) + X_k;//X[n, ih, iw, ic]

	float4 x;
	x.x = (lx0 ? X[X0 + xoffset] : 0);
	x.y = (lx1 ? X[X1 + xoffset] : 0);
	x.z = (lx2 ? X[X2 + xoffset] : 0);
	x.w = (lx3 ? X[X3 + xoffset] : 0);
	return x;
}

__device__ __forceinline__ float4 LoadX4x_ic2pow(const float* __restrict__ X,
	int X_k, int IH, int IW, int LIC, int FW_IC, int sw_IC,
	int toh0, int tow0, int tow1, int tow2, int tow3)
{
	int X_fh, int X_fw; get_X_fh_fw_IC2pow(X_k, X_fh, X_fw);
	bool lx = (toh0 >= -X_fh) && (toh0 < IH - X_fh);
	bool lx0 = lx && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
	bool lx1 = lx && (tow1 >= -X_fw) && (tow1 < IW - X_fw);
	bool lx2 = lx && (tow2 >= -X_fw) && (tow2 < IW - X_fw);
	bool lx3 = lx && (tow3 >= -X_fw) && (tow3 < IW - X_fw);
	int xoffset = (X_fh << LIC)*IW + X_k;//X[n, ih, iw, ic]

	float4 x;
	x.x = (lx0 ? X[xoffset - sw_IC] : 0);
	x.y = (lx1 ? X[xoffset] : 0);
	x.z = (lx2 ? X[xoffset + sw_IC] : 0);
	x.w = (lx3 ? X[xoffset + (sw_IC << 1)] : 0);
	return x;
}

#endif


//for k88R_FW_ic2pow, k88R4_FW_ic2pow
#ifndef LOADX_4_FW_IC_2POW
#define LOADX_4_FW_IC_2POW

__device__ __forceinline__ float4 LoadX4_FW_ic2pow(const float* __restrict__ X,
	int X_k, int IH, int IW, int LFW_IC, int LFW_IC_m1, int LIC,
	int X0, int toh0, int tow0,
	int X1, int toh1, int tow1,
	int X2, int toh2, int tow2,
	int X3, int toh3, int tow3)
{
	int X_fh, X_fw; get_X_fh_fw_FW_IC2pow(X_k, X_fh, X_fw);
	bool lx0 = (toh0 >= -X_fh) && (toh0 < IH - X_fh) && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
	bool lx1 = (toh1 >= -X_fh) && (toh1 < IH - X_fh) && (tow1 >= -X_fw) && (tow1 < IW - X_fw);
	bool lx2 = (toh2 >= -X_fh) && (toh2 < IH - X_fh) && (tow2 >= -X_fw) && (tow2 < IW - X_fw);
	bool lx3 = (toh3 >= -X_fh) && (toh3 < IH - X_fh) && (tow3 >= -X_fw) && (tow3 < IW - X_fw);
	int xoffset = ((X_fh * IW) << LIC) + X_k;//X[n, ih, iw, ic]

	float4 x;
	x.x = (lx0 ? X[X0 + xoffset] : 0);
	x.y = (lx1 ? X[X1 + xoffset] : 0);
	x.z = (lx2 ? X[X2 + xoffset] : 0);
	x.w = (lx3 ? X[X3 + xoffset] : 0);
	return x;
}

__device__ __forceinline__ float4 LoadX4x_FW_ic2pow(const float* __restrict__ X,
	int X_k, int IH, int IW, int LFW_IC, int LFW_IC_m1, int LIC, int sw_IC,
	int toh0, int tow0, int tow1, int tow2, int tow3)
{
	int X_fh, X_fw; get_X_fh_fw_FW_IC2pow(X_k, X_fh, X_fw);
	bool lx = (toh0 >= -X_fh) && (toh0 < IH - X_fh);
	bool lx0 = lx && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
	bool lx1 = lx && (tow1 >= -X_fw) && (tow1 < IW - X_fw);
	bool lx2 = lx && (tow2 >= -X_fw) && (tow2 < IW - X_fw);
	bool lx3 = lx && (tow3 >= -X_fw) && (tow3 < IW - X_fw);
	int xoffset = (X_fh << LIC)*IW + X_k;//X[n, ih, iw, ic]

	float4 x;
	x.x = (lx0 ? X[xoffset - sw_IC] : 0);
	x.y = (lx1 ? X[xoffset] : 0);
	x.z = (lx2 ? X[xoffset + sw_IC] : 0);
	x.w = (lx3 ? X[xoffset + (sw_IC << 1)] : 0);
	return x;
}

#endif


#ifndef LOADX_4
#define LOADX_4

__device__ __forceinline__ float4 LoadX4(const float* __restrict__ X,
	int X_k, int IH, int IW, int IC, int FW_IC, int IW_IC,
	int X0, int toh0, int tow0,
	int X1, int toh1, int tow1,
	int X2, int toh2, int tow2,
	int X3, int toh3, int tow3)
{
	int X_fh, X_fw; get_X_fh_fw(X_k, X_fh, X_fw);
	bool lx0 = (toh0 >= -X_fh) && (toh0 < IH - X_fh) && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
	bool lx1 = (toh1 >= -X_fh) && (toh1 < IH - X_fh) && (tow1 >= -X_fw) && (tow1 < IW - X_fw);
	bool lx2 = (toh2 >= -X_fh) && (toh2 < IH - X_fh) && (tow2 >= -X_fw) && (tow2 < IW - X_fw);
	bool lx3 = (toh3 >= -X_fh) && (toh3 < IH - X_fh) && (tow3 >= -X_fw) && (tow3 < IW - X_fw);
	int xoffset = X_fh * IW_IC + X_k;

	float4 x;
	x.x = (lx0 ? X[X0 + xoffset] : 0);
	x.y = (lx1 ? X[X1 + xoffset] : 0);
	x.z = (lx2 ? X[X2 + xoffset] : 0);
	x.w = (lx3 ? X[X3 + xoffset] : 0);
	return x;
}

__device__ __forceinline__ float4 LoadX4x(const float* __restrict__ X,
	int X_k, int IH, int IW, int IC, int FW_IC, int IW_IC, int sw_IC,
	int toh0, int tow0, int tow1, int tow2, int tow3)
{
	int X_fh, X_fw; get_X_fh_fw(X_k, X_fh, X_fw);
	bool lx = (toh0 >= -X_fh) && (toh0 < IH - X_fh);
	bool lx0 = lx && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
	bool lx1 = lx && (tow1 >= -X_fw) && (tow1 < IW - X_fw);
	bool lx2 = lx && (tow2 >= -X_fw) && (tow2 < IW - X_fw);
	bool lx3 = lx && (tow3 >= -X_fw) && (tow3 < IW - X_fw);
	int xoffset = X_fh * IW_IC + X_k;//X[n, ih, iw, ic]

	float4 x;
	x.x = (lx0 ? X[xoffset - sw_IC] : 0);
	x.y = (lx1 ? X[xoffset] : 0);
	x.z = (lx2 ? X[xoffset + sw_IC] : 0);
	x.w = (lx3 ? X[xoffset + (sw_IC << 1)] : 0);
	return x;
}

#endif


#ifndef LOADX_4_IC_TEXTURE
#define LOADX_4_IC_TEXTURE

__device__ __forceinline__ float4 LoadX4_tex(cudaTextureObject_t X,
	int X_k, int IH, int IW, int IC, int FW_IC, int IW_IC,
	int X0, int toh0, int tow0,
	int X1, int toh1, int tow1,
	int X2, int toh2, int tow2,
	int X3, int toh3, int tow3)
{
	int X_fh, X_fw; get_X_fh_fw(X_k, X_fh, X_fw);
	bool lx0 = (toh0 >= -X_fh) && (toh0 < IH - X_fh) && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
	bool lx1 = (toh1 >= -X_fh) && (toh1 < IH - X_fh) && (tow1 >= -X_fw) && (tow1 < IW - X_fw);
	bool lx2 = (toh2 >= -X_fh) && (toh2 < IH - X_fh) && (tow2 >= -X_fw) && (tow2 < IW - X_fw);
	bool lx3 = (toh3 >= -X_fh) && (toh3 < IH - X_fh) && (tow3 >= -X_fw) && (tow3 < IW - X_fw);
	int xoffset = X_fh * IW_IC + X_k;

	float4 x;
	zero_float(x.x, lx0, tex1Dfetch<float>(X, X0 + xoffset));
	zero_float(x.y, lx1, tex1Dfetch<float>(X, X1 + xoffset));
	zero_float(x.z, lx2, tex1Dfetch<float>(X, X2 + xoffset));
	zero_float(x.w, lx3, tex1Dfetch<float>(X, X3 + xoffset));
	return x;
}

__device__ __forceinline__ float4 LoadX4x_tex(cudaTextureObject_t X,
	int X_k, int IH, int IW, int IC, int FW_IC, int IW_IC, int sw_IC,
	int X1, int toh0, int tow0, int tow1, int tow2, int tow3)
{
	int X_fh, X_fw; get_X_fh_fw(X_k, X_fh, X_fw);
	bool lx = (toh0 >= -X_fh) && (toh0 < IH - X_fh);
	bool lx0 = lx && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
	bool lx1 = lx && (tow1 >= -X_fw) && (tow1 < IW - X_fw);
	bool lx2 = lx && (tow2 >= -X_fw) && (tow2 < IW - X_fw);
	bool lx3 = lx && (tow3 >= -X_fw) && (tow3 < IW - X_fw);
	int xoffset = X1 + X_fh * IW_IC + X_k;//X[n, ih, iw, ic]

	float4 x;
	zero_float(x.x, lx0, tex1Dfetch<float>(X, xoffset - sw_IC));
	zero_float(x.y, lx1, tex1Dfetch<float>(X, xoffset));
	zero_float(x.z, lx2, tex1Dfetch<float>(X, xoffset + sw_IC));
	zero_float(x.w, lx3, tex1Dfetch<float>(X, xoffset + (sw_IC << 1)));
	return x;
}

#endif


//for k88R_ic2pow_tex, k88R4_ic2pow_tex
#ifndef LOADX_4_IC_2POW_TEXTURE
#define LOADX_4_IC_2POW_TEXTURE

__device__ __forceinline__ float4 LoadX4_ic2pow_tex(cudaTextureObject_t X,
	int X_k, int IH, int IW, int LIC, int FW_IC,
	int X0, int toh0, int tow0,
	int X1, int toh1, int tow1,
	int X2, int toh2, int tow2,
	int X3, int toh3, int tow3)
{
	int X_fh, X_fw; get_X_fh_fw_IC2pow(X_k, X_fh, X_fw);
	bool lx0 = (toh0 >= -X_fh) && (toh0 < IH - X_fh) && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
	bool lx1 = (toh1 >= -X_fh) && (toh1 < IH - X_fh) && (tow1 >= -X_fw) && (tow1 < IW - X_fw);
	bool lx2 = (toh2 >= -X_fh) && (toh2 < IH - X_fh) && (tow2 >= -X_fw) && (tow2 < IW - X_fw);
	bool lx3 = (toh3 >= -X_fh) && (toh3 < IH - X_fh) && (tow3 >= -X_fw) && (tow3 < IW - X_fw);
	int xoffset = ((X_fh * IW) << LIC) + X_k;//X[n, ih, iw, ic]

	float4 x;
	zero_float(x.x, lx0, tex1Dfetch<float>(X, X0 + xoffset));
	zero_float(x.y, lx1, tex1Dfetch<float>(X, X1 + xoffset));
	zero_float(x.z, lx2, tex1Dfetch<float>(X, X2 + xoffset));
	zero_float(x.w, lx3, tex1Dfetch<float>(X, X3 + xoffset));
	return x;
}

__device__ __forceinline__ float4 LoadX4x_ic2pow_tex(cudaTextureObject_t X,
	int X_k, int IH, int IW, int LIC, int FW_IC, int sw_IC,
	int X1, int toh0, int tow0, int tow1, int tow2, int tow3)
{
	int X_fh, int X_fw; get_X_fh_fw_IC2pow(X_k, X_fh, X_fw);
	bool lx = (toh0 >= -X_fh) && (toh0 < IH - X_fh);
	bool lx0 = lx && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
	bool lx1 = lx && (tow1 >= -X_fw) && (tow1 < IW - X_fw);
	bool lx2 = lx && (tow2 >= -X_fw) && (tow2 < IW - X_fw);
	bool lx3 = lx && (tow3 >= -X_fw) && (tow3 < IW - X_fw);
	int xoffset = X1 + (X_fh << LIC)*IW + X_k;//X[n, ih, iw, ic]

	float4 x;
	zero_float(x.x, lx0, tex1Dfetch<float>(X, xoffset - sw_IC));
	zero_float(x.y, lx1, tex1Dfetch<float>(X, xoffset));
	zero_float(x.z, lx2, tex1Dfetch<float>(X, xoffset + sw_IC));
	zero_float(x.w, lx3, tex1Dfetch<float>(X, xoffset + (sw_IC << 1)));
	return x;
}

#endif


//for k88R_FW_ic2pow_tex, k88R4_FW_ic2pow_tex
#ifndef LOADX_4_FW_IC_2POW_TEXTURE
#define LOADX_4_FW_IC_2POW_TEXTURE

__device__ __forceinline__ float4 LoadX4_FW_ic2pow_tex(cudaTextureObject_t X,
	int X_k, int IH, int IW, int LFW_IC, int LFW_IC_m1, int LIC,
	int X0, int toh0, int tow0,
	int X1, int toh1, int tow1,
	int X2, int toh2, int tow2,
	int X3, int toh3, int tow3)
{
	int X_fh, X_fw; get_X_fh_fw_FW_IC2pow(X_k, X_fh, X_fw);
	bool lx0 = (toh0 >= -X_fh) && (toh0 < IH - X_fh) && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
	bool lx1 = (toh1 >= -X_fh) && (toh1 < IH - X_fh) && (tow1 >= -X_fw) && (tow1 < IW - X_fw);
	bool lx2 = (toh2 >= -X_fh) && (toh2 < IH - X_fh) && (tow2 >= -X_fw) && (tow2 < IW - X_fw);
	bool lx3 = (toh3 >= -X_fh) && (toh3 < IH - X_fh) && (tow3 >= -X_fw) && (tow3 < IW - X_fw);
	int xoffset = ((X_fh * IW) << LIC) + X_k;//X[n, ih, iw, ic]

	float4 x;
	zero_float(x.x, lx0, tex1Dfetch<float>(X, X0 + xoffset));
	zero_float(x.y, lx1, tex1Dfetch<float>(X, X1 + xoffset));
	zero_float(x.z, lx2, tex1Dfetch<float>(X, X2 + xoffset));
	zero_float(x.w, lx3, tex1Dfetch<float>(X, X3 + xoffset));
	return x;
}

__device__ __forceinline__ float4 LoadX4x_FW_ic2pow_tex(cudaTextureObject_t X,
	int X_k, int IH, int IW, int LFW_IC, int LFW_IC_m1, int LIC, int sw_IC,
	int X1, int toh0, int tow0, int tow1, int tow2, int tow3)
{
	int X_fh, X_fw; get_X_fh_fw_FW_IC2pow(X_k, X_fh, X_fw);
	bool lx = (toh0 >= -X_fh) && (toh0 < IH - X_fh);
	bool lx0 = lx && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
	bool lx1 = lx && (tow1 >= -X_fw) && (tow1 < IW - X_fw);
	bool lx2 = lx && (tow2 >= -X_fw) && (tow2 < IW - X_fw);
	bool lx3 = lx && (tow3 >= -X_fw) && (tow3 < IW - X_fw);
	int xoffset = X1 + (X_fh << LIC)*IW + X_k;//X[n, ih, iw, ic]

	float4 x;
	zero_float(x.x, lx0, tex1Dfetch<float>(X, xoffset - sw_IC));
	zero_float(x.y, lx1, tex1Dfetch<float>(X, xoffset));
	zero_float(x.z, lx2, tex1Dfetch<float>(X, xoffset + sw_IC));
	zero_float(x.w, lx3, tex1Dfetch<float>(X, xoffset + (sw_IC << 1)));
	return x;
}

#endif


#ifndef LOADX_2
#define LOADX_2

__device__ __forceinline__ float2 LoadX2_ic2pow(const float* __restrict__ X,
	int X_k, int IH, int IW, int LIC, int FW_IC,
	int X0, int toh0, int tow0,
	int X1, int toh1, int tow1)
{
	int X_fh, X_fw; get_X_fh_fw_IC2pow(X_k, X_fh, X_fw);
	bool lx0 = (toh0 >= -X_fh) && (toh0 < IH - X_fh) && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
	bool lx1 = (toh1 >= -X_fh) && (toh1 < IH - X_fh) && (tow1 >= -X_fw) && (tow1 < IW - X_fw);
	int xoffset = ((X_fh * IW) << LIC) + X_k;//X[n, ih, iw, ic]

	float2 x;
	x.x = (lx0 ? X[X0 + xoffset] : 0);
	x.y = (lx1 ? X[X1 + xoffset] : 0);
	return x;
}

__device__ __forceinline__ float2 LoadX2_FW_ic2pow(const float* __restrict__ X,
	int X_k, int IH, int IW, int LFW_IC, int LFW_IC_m1, int LIC,
	int X0, int toh0, int tow0,
	int X1, int toh1, int tow1)
{
	int X_fh, X_fw; get_X_fh_fw_FW_IC2pow(X_k, X_fh, X_fw);
	bool lx0 = (toh0 >= -X_fh) && (toh0 < IH - X_fh) && (tow0 >= -X_fw) && (tow0 < IW - X_fw);
	bool lx1 = (toh1 >= -X_fh) && (toh1 < IH - X_fh) && (tow1 >= -X_fw) && (tow1 < IW - X_fw);
	int xoffset = ((X_fh * IW) << LIC) + X_k;//X[n, ih, iw, ic]

	float2 x;
	x.x = (lx0 ? X[X0 + xoffset] : 0);
	x.y = (lx1 ? X[X1 + xoffset] : 0);
	return x;
}
#endif