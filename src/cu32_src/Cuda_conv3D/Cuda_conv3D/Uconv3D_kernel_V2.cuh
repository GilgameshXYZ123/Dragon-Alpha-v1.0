#define conv3dv2_2_2(stream, LB, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw)\
	conv3d_direct_kernelv8<LB, 1<<LB>\
		<<< dim3((N*OH*OW)>>LB>>1, (OC)>>LB>>1), dim3(1<<LB, 1<<LB), 0, stream>>>\
			(X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw)

//Size = 1, Time = 4.726 msec, Performace = 454.398 GFlop/s
template<int LB, int STEP>
__global__ void conv3d_v2_kernel_2_2(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ Y, int OH, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 Xs[2][1 << LB][2 << LB];
	__shared__ float2 Ws[2][1 << LB][2 << LB];

	int oc0 = ((blockIdx.y << LB) + ty) << 1, oc1 = oc0 + 1;
	int j0 = ((blockIdx.x << LB) + tx) << 1, j1 = j0 + 1;
	int OH_OW = OH * OW;
	get_n_oh_ow(j0, n0, oh0, ow0);
	get_n_oh_ow(j1, n1, oh1, ow1);

	float2 v0 = make_float2(0, 0);
	float2 v1 = make_float2(0, 0);
	const int OIC = IC >> LB >> 1, STEP = 1 << LB;
	const int ihs0 = oh0 * sh - ph, iws0 = ow0 * sw - pw;
	const int ihs1 = oh1 * sh - ph, iws1 = ow1 * sw - pw;
	for (int fh = 0; fh < FH; fh++)
	{
		int ih0 = ihs0 + fh, ih1 = ihs1 + fh;
		for (int fw = 0; fw < FW; fw++)
		{
			int iw0 = iws0 + fw, iw1 = iws1 + fw;
			bool unload0 = ih0 < 0 || iw0 < 0 || ih0 >= IH || iw0 >= IW;
			bool unload1 = ih1 < 0 || iw1 < 0 || ih1 >= IH || iw1 >= IW;

			int Wic = tx << 1;
			Ws[buf][tx][(ty << 1)] = *(float2*)(&get4d(W, oc0, fh, fw, Wic, FH, FW, IC));
			Ws[buf][tx][(ty << 1) + 1] = *(float2*)(&get4d(W, oc1, fh, fw, Wic, FH, FW, IC));

			int Xic = ty << 1;
			Xs[buf][ty][(tx << 1)] = unload0 ? make_float2(0, 0) : *(float2*)(&get4d(X, n0, ih0, iw0, Xic, IH, IW, IC));
			Xs[buf][ty][(tx << 1) + 1] = unload1 ? make_float2(0, 0) : *(float2*)(&get4d(X, n1, ih1, iw1, Xic, IH, IW, IC));
			__syncthreads();

			for (int oic = 1; oic < OIC; oic++)
			{
#pragma unroll
				for (int ik = 0; ik < STEP; ik++) {
					float4 w = *(float4*)(&Ws[buf][ik][ty << 1]);
					float4 x = *(float4*)(&Xs[buf][ik][tx << 1]);
					v0.x += w.x * x.x + w.y * x.y; v0.y += w.z * x.x + w.w * x.y;
					v1.x += w.x * x.z + w.y * x.w; v1.y += w.z * x.z + w.w * x.w;
				}

				buf ^= 1;
				int Wic = ((oic << LB) + tx) << 1;
				Ws[buf][tx][(ty << 1)] = *(float2*)(&get4d(W, oc0, fh, fw, Wic, FH, FW, IC));
				Ws[buf][tx][(ty << 1) + 1] = *(float2*)(&get4d(W, oc1, fh, fw, Wic, FH, FW, IC));

				int Xic = ((oic << LB) + ty) << 1;
				if (unload0) Xs[buf][ty][(tx << 1)] = make_float2(0, 0);
				Xs[buf][ty][(tx << 1)] = unload0 ? make_float2(0, 0) : *(float2*)(&get4d(X, n0, ih0, iw0, Xic, IH, IW, IC));
				Xs[buf][ty][(tx << 1) + 1] = unload1 ? make_float2(0, 0) : *(float2*)(&get4d(X, n1, ih1, iw1, Xic, IH, IW, IC));
				__syncthreads();
			}
#pragma unroll
			for (int ik = 0; ik < STEP; ik++) {
				float4 w = *(float4*)(&Ws[buf][ik][ty << 1]);
				float4 x = *(float4*)(&Xs[buf][ik][tx << 1]);
				v0.x += w.x * x.x + w.y * x.y; v0.y += w.z * x.x + w.w * x.y;
				v1.x += w.x * x.z + w.y * x.w; v1.y += w.z * x.z + w.w * x.w;
			}
			__syncthreads();
		}
	}

	j0 *= OC; j1 = j0 + OC;
	*(float2*)(&Y[j0 + oc0]) = v0;
	*(float2*)(&Y[j1 + oc0]) = v1;
}


#define conv3dv2_2_2(stream, LB, X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw)\
	conv3d_direct_kernelv9<LB, (1<<LB>>1)>\
		<<< dim3((N*OH*OW)>>LB>>2, (OC)>>LB>>2), dim3(1<<LB, 1<<LB), 0, stream>>>\
			(X, IH, IW, W, FH, FW, Y, OH, OW, IC, OC, sh, sw, ph, pw)

//Size = 1, Time = 2.486 msec, Performace = 863.831 GFlop/s
template<int LB, int STEP>
__global__ void conv3d_v2_kernel_4_4(
	const float* __restrict__ X, int IH, int IW,
	const float* __restrict__ W, int FH, int FW,
	float* __restrict__ Y, int OH, int OW,
	int IC, int OC,
	int sh, int sw, int ph, int pw)
{
	int ty = threadIdx.y, tx = threadIdx.x;

	bool buf = 0;
	__shared__ float2 Xs[2][1 << LB << LB];//[1<<LB][2<<LB]
	__shared__ float2 Ws[2][1 << LB << LB];//[1<<LB][2<<LB]

	const int oc0 = ((blockIdx.y << LB) + ty) << 2;
	const int GK = FH * FW * IC;
	const int toc0 = (((tx & 1) << 1) + oc0) * GK, toc1 = toc0 + GK;

	int j0 = ((blockIdx.x << LB) + tx) << 2;
	int j1 = j0 + 1, j2 = j0 + 2, j3 = j0 + 3;
	const int OH_OW = OH * OW;
	get_n_oh_ow(j0, n0, oh0, ow0);
	get_n_oh_ow(j1, n1, oh1, ow1);
	get_n_oh_ow(j2, n2, oh2, ow2);
	get_n_oh_ow(j3, n3, oh3, ow3);
	bool flagY = (ty & 1);
	const int tn0 = (n2 - n0)*flagY + n0;
	const int tn1 = (n3 - n1)*flagY + n1;
	const int tihs0 = ((oh2 - oh0)*flagY + oh0) * sh - ph;
	const int tihs1 = ((oh3 - oh1)*flagY + oh1) * sh - ph;
	const int tiws0 = ((ow2 - ow0)*flagY + ow0) * sw - pw;
	const int tiws1 = ((ow3 - ow1)*flagY + ow1) * sw - pw;
	const int Xoffset0 = (((tn0 *IH) + tihs0) * IW + tiws0) * IC;
	const int Xoffset1 = (((tn1 *IH) + tihs1) * IW + tiws1) * IC;

	//compute area----------------------------------------------------
	float4 v0 = make_float4(0, 0, 0, 0);
	float4 v1 = make_float4(0, 0, 0, 0);
	float4 v2 = make_float4(0, 0, 0, 0);
	float4 v3 = make_float4(0, 0, 0, 0);

	const int Ws_xy = ((tx >> 1) << 1 << LB) + (ty << 1) + (tx & 1);
	const int Xs_yx = ((ty >> 1) << 1 << LB) + (tx << 1) + (ty & 1);
	for (int fh = 0; fh < FH; fh++, X += (IW - FW) *IC)
		for (int fw = 0; fw < FW; fw++, X += IC, W += IC)
		{
			int Wic = tx >> 1;
			Ws[buf][Ws_xy].x = W[toc0 + Wic];
			Ws[buf][Ws_xy].y = W[toc1 + Wic];

			int Xic = ty >> 1;
			bool load0 = (tihs0 >= -fh) && (tihs0 < IH - fh) && (tiws0 >= -fw) && (tiws0 < IW - fw);
			bool load1 = (tihs1 >= -fh) && (tihs1 < IH - fh) && (tiws1 >= -fw) && (tiws1 < IW - fw);
			Xs[buf][Xs_yx].x = load0 ? X[Xoffset0 + Xic] : 0;
			Xs[buf][Xs_yx].y = load1 ? X[Xoffset1 + Xic] : 0;
			__syncthreads();

			for (int oic = 1, OIC = IC << 1 >> LB; oic < OIC; oic++) {
#pragma unroll
				for (int ik = 0; ik < STEP; ik++)
				{
					float4 b = *(float4*)(&Xs[buf][((ik << LB) + tx) << 1]);
					float4 a = *(float4*)(&Ws[buf][((ik << LB) + ty) << 1]);
					simdMM4(v0, b.x, a);
					simdMM4(v1, b.y, a);
					simdMM4(v2, b.z, a);
					simdMM4(v3, b.w, a);
				}
				buf ^= 1;

				Wic = ((oic << LB) + tx) >> 1;
				Ws[buf][Ws_xy].x = W[toc0 + Wic];
				Ws[buf][Ws_xy].y = W[toc1 + Wic];

				Xic = ((oic << LB) + ty) >> 1;
				Xs[buf][Xs_yx].x = load0 ? X[Xoffset0 + Xic] : 0;
				Xs[buf][Xs_yx].y = load1 ? X[Xoffset1 + Xic] : 0;
				__syncthreads();
			}
#pragma unroll
			for (int ik = 0; ik < STEP; ik++) {
				float4 b = *(float4*)(&Xs[buf][((ik << LB) + tx) << 1]);
				float4 a = *(float4*)(&Ws[buf][((ik << LB) + ty) << 1]);

				simdMM4(v0, b.x, a);
				simdMM4(v1, b.y, a);
				simdMM4(v2, b.z, a);
				simdMM4(v3, b.w, a);
			}
			buf ^= 1;
		}

	j0 *= OC; j1 = j0 + OC; j2 = j1 + OC; j3 = j2 + OC;
	*(float4*)(&Y[j0 + oc0]) = v0;
	*(float4*)(&Y[j1 + oc0]) = v1;
	*(float4*)(&Y[j2 + oc0]) = v2;
	*(float4*)(&Y[j3 + oc0]) = v3;
}