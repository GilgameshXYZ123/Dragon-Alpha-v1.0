/*
 * Copyright 2014 Google Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "../../nvmatrix/include/nvmatrix.cuh"
#include "../include/cudaconv2.cuh"


__device__ __forceinline__ void filterActs_YxX_color_preload_ty_4_tx_32_f_16_cc_3_setImgCoords(
	int fPidx,
	int imgLoadModPosY,
	int imgLoadModPosX,
	int imgSizeX, int filterSize, int& iPidx)
{
	int x = imgLoadModPosX + (fPidx) % filterSize;
	int y = imgLoadModPosY + (fPidx) / filterSize;
	iPidx = y >= 0 && y < imgSizeX && x >= 0 && x < imgSizeX ? y * imgSizeX + x : -1;
}

#define FA_COLOR3_IMPRELOAD(c,i) imPreload[c][i] = iPidxNext < 0 || (checkImgBounds && myImgIdx + i * BLOCK_X >= N) ? 0 : mm[c * imgPixels * imgStride + i * BLOCK_X];
#define FA_COLOR3_IMPRELOAD_TX(c,i) imPreload[c][i] = iPidxNext < 0 || (checkImgBounds && myImgIdx + i * BLOCK_X >= N) ? 0 : tex1Dfetch<float>(images, imagesOffset2 + c * imgPixels * imgStride + i * BLOCK_X);


/*
 * images:      (numImgColors, imgSizeY, imgSizeX, numImages) with stride given
 * filters:     (numFilterColors, filterPixels, numFilters) if conv
 *              (numModules, numFilterColors, filterPixels, numFilters) otherwise
 *
 * targets:     (numFilters, numModulesY, numModulesX, numImages)
 *
 */
template <int BLOCK_Y, int BLOCK_X, int imgsPerThread, int filtersPerThread, int numColors, int pixelCache, bool scale, bool checkImgBounds>
	//__launch_bounds__(128,3)
__global__ void filterActs_YxX_color_preload_ty_4_tx_32_i_4_f_16_px_4_cc_3_tex(cudaTextureObject_t images, cudaTextureObject_t filters, float* targets,
	const int numImages, const int numFilters,
	const int IH,//IH
	const int IW,//IW
	const int filterSize, const int paddingStart, const int moduleStride,
	const int numModulesY, const int numModulesX, const int imgStride,
	const float scaleTargets, const float scaleOutputs,
	const bool conv/*, const bool noloads*/) 
{
	__shared__ float shFilters[numColors][pixelCache][BLOCK_Y * filtersPerThread]; // pre-load 1 pixel from B_Y*filtersPerThread filters
	__shared__ float shImages[numColors][pixelCache][BLOCK_X * imgsPerThread]; // pre-load 1 pixel from B_X*imgsPerThread images

	const int imgPixels = IH * IW;
	const int filterPixels = filterSize * filterSize;
	const int blocksPerModule = numFilters / (BLOCK_Y*filtersPerThread);
	const int moduleIdx = blockIdx.y / blocksPerModule;
	const int blockFilterIdx = filtersPerThread * BLOCK_Y * (blockIdx.y % blocksPerModule);

	const int numModules = numModulesX * numModulesY;
	// Another fun insanity: the % B_X makes things faster, even thought threadIdx.x is
	// in the range 0..31. It appears that this allows the compiler to optimize?
	const int tx = threadIdx.x % BLOCK_X;
	const int ty = threadIdx.y % BLOCK_Y;
	const int tidx = ty * BLOCK_X + threadIdx.x;

	const int imgLoadModPosY = paddingStart + (moduleIdx / numModulesX) * moduleStride;
	const int imgLoadModPosX = paddingStart + (moduleIdx % numModulesX) * moduleStride;

	const int shFilterLoadY = tidx / (BLOCK_Y * filtersPerThread);
	const int shFilterLoadX = tidx % (BLOCK_Y * filtersPerThread);
	const int myImgIdx = blockIdx.x * BLOCK_X * imgsPerThread + threadIdx.x;

	//    images += myImgIdx;
	//    filters += blockFilterIdx
	//            + shFilterLoadY * numFilters + shFilterLoadX;
	//    if (!conv) { // NOTE: UNTESTED!
	//        filters += moduleIdx * numColors * filterPixels * numFilters;
	//    }

	const int imagesOffset = myImgIdx;
	const int filtersOffset = blockFilterIdx + shFilterLoadY * numFilters + shFilterLoadX
		+ (conv ? 0 : moduleIdx * numColors * filterPixels * numFilters);

	targets += moduleIdx * numImages
		+ (blockFilterIdx + threadIdx.y * filtersPerThread) * numImages * numModules
		+ myImgIdx;

	float prod[imgsPerThread][filtersPerThread];
#pragma unroll
	for (int i = 0; i < imgsPerThread; i++) {
#pragma unroll
		for (int f = 0; f < filtersPerThread; f++) {
			prod[i][f] = 0;
		}
	}

	int iPidxNext;
	float imPreload[numColors][imgsPerThread];
	float fPreload[numColors][pixelCache*filtersPerThread / BLOCK_X];

#pragma unroll
	for (int c = 0; c < numColors; ++c) {
#pragma unroll
		for (int p = 0; p < pixelCache; p += BLOCK_X / filtersPerThread) {
			if (p + shFilterLoadY < filterPixels) {
				fPreload[c][p*filtersPerThread / BLOCK_X] = tex1Dfetch<float>(filters, filtersOffset + p * numFilters + c * numFilters * filterPixels);
			}
			else {
				fPreload[c][p*filtersPerThread / BLOCK_X] = 0;
			}
		}
	}

	filterActs_YxX_color_preload_ty_4_tx_32_f_16_cc_3_setImgCoords(ty, imgLoadModPosY, imgLoadModPosX, IW, filterSize, iPidxNext);

#pragma unroll
	for (int c = 0; c < numColors; ++c) {
#pragma unroll
		for (int i = 0; i < imgsPerThread; i++) {
			if (iPidxNext >= 0 && (!checkImgBounds || myImgIdx + i * BLOCK_X < numImages)) {
				imPreload[c][i] = tex1Dfetch<float>(images, imagesOffset + (c * imgPixels + iPidxNext) * imgStride + i * BLOCK_X);
			}
			else {
				imPreload[c][i] = 0;
			}
		}
	}

	for (int p = 0; p < filterPixels; p += pixelCache) {
#pragma unroll
		for (int i = 0; i < imgsPerThread; i++) {
#pragma unroll
			for (int c = 0; c < numColors; ++c) {
				// NOTE: bank conflicts here!
				shImages[c][ty][tx * imgsPerThread + i] = imPreload[c][i];
			}
		}

		const int fPidxNext = p + pixelCache >= filterPixels ? 0 : p + pixelCache;
		filterActs_YxX_color_preload_ty_4_tx_32_f_16_cc_3_setImgCoords(fPidxNext + ty, imgLoadModPosY, imgLoadModPosX, IW, filterSize, iPidxNext);

		//        const float* ff = &filters[numFilters * fPidxNext];
		//        const float* mm = &images[imgStride * iPidxNext];
		const int filtersOffset2 = filtersOffset + numFilters * fPidxNext;
		const int imagesOffset2 = imagesOffset + imgStride * iPidxNext;

		FA_COLOR3_IMPRELOAD_TX(0, 0);
		FA_COLOR3_IMPRELOAD_TX(0, 1);
		FA_COLOR3_IMPRELOAD_TX(0, 2);
		FA_COLOR3_IMPRELOAD_TX(0, 3);

#pragma unroll
		for (int c = 0; c < numColors; ++c) {
#pragma unroll
			for (int pp = 0; pp < pixelCache; pp += BLOCK_X / filtersPerThread) {
				shFilters[c][pp + shFilterLoadY][shFilterLoadX] = fPreload[c][pp*filtersPerThread / BLOCK_X];
			}
		}

		__syncthreads();
		FA_COLOR3_IMPRELOAD_TX(1, 0);
		FA_COLOR3_IMPRELOAD_TX(1, 1);
		FA_COLOR3_IMPRELOAD_TX(1, 2);
		FA_COLOR3_IMPRELOAD_TX(1, 3);
		FA_COLOR3_IMPRELOAD_TX(2, 0);
		FA_COLOR3_IMPRELOAD_TX(2, 1);
		FA_COLOR3_IMPRELOAD_TX(2, 2);
		FA_COLOR3_IMPRELOAD_TX(2, 3);
#pragma unroll
		for (int c = 0; c < numColors; c++) {
#pragma unroll
			for (int pp = 0; pp < pixelCache*filtersPerThread / BLOCK_X; pp++) {
				fPreload[c][pp] = fPidxNext + pp * (BLOCK_X / filtersPerThread) + shFilterLoadY >= filterPixels ? 0 : tex1Dfetch<float>(filters, filtersOffset2 + c * numFilters* filterPixels + pp * (BLOCK_X / filtersPerThread) * numFilters);
			}
		}
#pragma unroll
		for (int pp = 0; pp < pixelCache; pp++) {
#pragma unroll
			for (int c = 0; c < numColors; c++) {
#pragma unroll
				for (int f = 0; f < filtersPerThread; f++) {
#pragma unroll
					for (int i = 0; i < imgsPerThread; i++) {
						prod[i][f] += shImages[c][pp][tx * imgsPerThread + i] * shFilters[c][pp][ty * filtersPerThread + f];
					}
				}
			}
		}

		__syncthreads();
	}

	if (scale) {
#pragma unroll
		for (int f = 0; f < filtersPerThread; f++) {
#pragma unroll
			for (int i = 0; i < imgsPerThread; i++) {
				if (!checkImgBounds || myImgIdx + i * BLOCK_X < numImages) {
					targets[i * BLOCK_X + f * numImages * numModules] = scaleTargets * targets[i * BLOCK_X + f * numImages * numModules] + scaleOutputs * prod[i][f];
				}
			}
		}
	}
	else {
		// Note: reversing order of these loops saves 2 registers, but costs time
#pragma unroll
		for (int i = 0; i < imgsPerThread; i++) {
#pragma unroll
			for (int f = 0; f < filtersPerThread; f++) {
				if (!checkImgBounds || myImgIdx + i * BLOCK_X < numImages) {
					targets[i * BLOCK_X + f * numImages * numModules] = scaleOutputs * prod[i][f];
				}
			}
		}
	}
}

/*
 * images:      (numImgColors, imgSizeY, imgSizeX, numImages) with stride given
 * filters:     (numFilterColors, filterPixels, numFilters) if conv
 *              (numModules, numFilterColors, filterPixels, numFilters) otherwise
 *
 * targets:     (numFilters, numModulesY, numModulesX, numImages)
 *
 * This won't be pretty.
 */
template <int BLOCK_Y, int BLOCK_X, int imgsPerThread, int filtersPerThread, int numColors, int pixelCache,
	bool scale, bool checkImgBounds>
	__global__ void filterActs_YxX_color_preload_ty_4_tx_32_i_4_f_12_px_4_cc_3_tex(cudaTextureObject_t images, cudaTextureObject_t filters, float* targets,
		const int N, const int OC,
		const int IH, const int IW, const int FS, const int paddingStart,
		const int moduleStride,
		const int OH, const int OW, const int imgStride,
		const float scaleTargets, const float scaleOutputs,
		const bool conv/*, const bool noloads*/)
{
	__shared__ float shFilters[numColors][pixelCache][BLOCK_Y * filtersPerThread]; // pre-load 1 pixel from B_Y*filtersPerThread filters
	__shared__ float shImages[numColors][pixelCache][BLOCK_X * imgsPerThread]; // pre-load 1 pixel from B_X*imgsPerThread images
	const int imgPixels = IH * IW;
	const int filterPixels = FS * FS;
	const int blocksPerModule = OC / (BLOCK_Y*filtersPerThread);
	const int moduleIdx = blockIdx.y / blocksPerModule;
	const int blockFilterIdx = filtersPerThread * BLOCK_Y * (blockIdx.y % blocksPerModule);

	const int numModules = OW * OH;
	// Another fun insanity: the % B_X makes things faster, even though threadIdx.x is
	// in the range 0..31. It appears that this allows the compiler to optimize?
	const int tx = threadIdx.x % BLOCK_X;
	const int ty = threadIdx.y % BLOCK_Y;
	const int tidx = ty * BLOCK_X + threadIdx.x;
	const int warp = tidx / 32;

	const int imgLoadModPosY = paddingStart + (moduleIdx / OW) * moduleStride;
	const int imgLoadModPosX = paddingStart + (moduleIdx % OW) * moduleStride;

	const int shFilterLoadY = tidx / (BLOCK_Y * filtersPerThread);
	const int shFilterLoadX = tidx % (BLOCK_Y * filtersPerThread);
	const int myImgIdx = blockIdx.x * BLOCK_X * imgsPerThread + threadIdx.x;

	//    images += myImgIdx;
	//    filters += blockFilterIdx
	//            + shFilterLoadY * numFilters + shFilterLoadX;
	//    if (!conv) { // NOTE: UNTESTED!
	//        filters += moduleIdx * numColors * filterPixels * numFilters;
	//    }

	const int imagesOffset = myImgIdx;
	const int filtersOffset = blockFilterIdx + shFilterLoadY * OC + shFilterLoadX
		+ (conv ? 0 : moduleIdx * numColors * filterPixels * OC);

	targets += moduleIdx * N
		+ (blockFilterIdx + threadIdx.y * filtersPerThread) * N * numModules
		+ myImgIdx;

	float prod[imgsPerThread][filtersPerThread];
#pragma unroll
	for (int i = 0; i < imgsPerThread; i++) {
#pragma unroll
		for (int f = 0; f < filtersPerThread; f++) {
			prod[i][f] = 0;
		}
	}

	int iPidxNext;
	float imPreload[numColors][imgsPerThread];
	float fPreload[numColors][DIVUP(pixelCache*filtersPerThread, BLOCK_X)];

	if (warp < 3) {
#pragma unroll
		for (int c = 0; c < numColors; ++c) {
#pragma unroll
			for (int p = 0; p < pixelCache; p += 2) {
				if (p + shFilterLoadY < filterPixels) {
					fPreload[c][p / 2] = tex1Dfetch<float>(filters, filtersOffset + p * OC + c * OC * filterPixels);
				}
				else {
					fPreload[c][p / 2] = 0;
				}
			}
		}
	}

	filterActs_YxX_color_preload_ty_4_tx_32_f_16_cc_3_setImgCoords(ty, imgLoadModPosY, imgLoadModPosX, IW, FS, iPidxNext);

#pragma unroll
	for (int c = 0; c < numColors; ++c) {
#pragma unroll
		for (int i = 0; i < imgsPerThread; i++) {
			if (iPidxNext >= 0 && (!checkImgBounds || myImgIdx + i * BLOCK_X < N)) {
				imPreload[c][i] = tex1Dfetch<float>(images, imagesOffset + (c * imgPixels + iPidxNext) * imgStride + i * BLOCK_X);
			}
			else {
				imPreload[c][i] = 0;
			}
		}
	}

	for (int p = 0; p < filterPixels; p += pixelCache) {
		const int fPidxNext = p + pixelCache >= filterPixels ? 0 : p + pixelCache;
		filterActs_YxX_color_preload_ty_4_tx_32_f_16_cc_3_setImgCoords(fPidxNext + ty, imgLoadModPosY, imgLoadModPosX, IW, FS, iPidxNext);

#pragma unroll
		for (int c = 0; c < numColors; ++c) {
#pragma unroll
			for (int i = 0; i < imgsPerThread; i++) {
				// NOTE: bank conflicts here!
				shImages[c][ty][tx * imgsPerThread + i] = imPreload[c][i];
			}
		}

		if (warp < 3) {
#pragma unroll
			for (int c = 0; c < numColors; ++c) {
#pragma unroll
				for (int pp = 0; pp < pixelCache; pp += 2) {
					shFilters[c][pp + shFilterLoadY][shFilterLoadX] = fPreload[c][pp / 2];
				}
			}
		}

		__syncthreads();
		//        const float* ff = &filters[numFilters * fPidxNext];
		//        const float* mm = &images[imgStride * iPidxNext];
		const int filtersOffset2 = filtersOffset + OC * fPidxNext;
		const int imagesOffset2 = imagesOffset + imgStride * iPidxNext;

#pragma unroll
		for (int i = 0; i < imgsPerThread; ++i) {
#pragma unroll
			for (int c = 0; c < numColors; c++) {
				FA_COLOR3_IMPRELOAD_TX(c, i);
			}
		}

#pragma unroll
		for (int c = 0; c < numColors; c++) {
#pragma unroll
			for (int pp = 0; pp < 2; pp++) {
				fPreload[c][pp] = warp >= 3 || fPidxNext + pp * 2 + shFilterLoadY >= filterPixels ? 0 : tex1Dfetch<float>(filters, filtersOffset2 + c * OC* filterPixels + pp * 2 * OC);
			}
#pragma unroll
			for (int pp = 0; pp < pixelCache; pp++) {
#pragma unroll
				for (int i = 0; i < imgsPerThread; i++) {
#pragma unroll
					for (int f = 0; f < filtersPerThread; f++) {
						prod[i][f] += shImages[c][pp][tx * imgsPerThread + i] * shFilters[c][pp][ty * filtersPerThread + f];
					}
				}
			}

		}
		__syncthreads();
	}

	if (scale) {
#pragma unroll
		for (int i = 0; i < imgsPerThread; i++) {
#pragma unroll
			for (int f = 0; f < filtersPerThread; f++) {
				if (!checkImgBounds || myImgIdx + i * BLOCK_X < N) {
					targets[i * BLOCK_X + f * N * numModules] = scaleTargets * targets[i * BLOCK_X + f * N * numModules] + scaleOutputs * prod[i][f];
				}
			}
		}
	}
	else {
		// Note: reversing order of these loops costs 2 registers, but saves time
#pragma unroll
		for (int i = 0; i < imgsPerThread; i++) {
#pragma unroll
			for (int f = 0; f < filtersPerThread; f++) {
				if (!checkImgBounds || myImgIdx + i * BLOCK_X < N) {
					targets[i * BLOCK_X + f * N * numModules] = scaleOutputs * prod[i][f];
				}
			}
		}
	}
}

__device__ inline void filterActs_YxX_sparse2_preload_ty_4_tx_32_f_16_c_4_setPixelCoords(int filterSize, int imgSizeX, int imgLoadModPosY, int imgLoadModPosX, int imgY, int imgX, int& fPidx, int& iPidx) {
	int filterPxY = imgY - imgLoadModPosY;
	int filterPxX = imgX - imgLoadModPosX;
	fPidx = filterPxY * filterSize + filterPxX;
	iPidx = imgY * imgSizeX + imgX; // Pixel index in img
}

/*
 * images:      (numImgColors, imgSizeY, imgSizeX, numImages) with stride given
 * filters:     (numFilterColors, filterPixels, numFilters) if conv
 *              (numModules, numFilterColors, filterPixels, numFilters) otherwise
 *
 * targets:     (numFilters, numModulesY, numModulesX, numImages)
 *
 * Note: in git there's a 1.5% faster version of this which sues 167 registers instead of 154...
 * it's basically the same thing, but it doesn't do the next-pixel computation. It just avoids
 * pre-loading when it rolls over to the next pixel.
 */
template <int BLOCK_Y, int BLOCK_X, int imgsPerThread, int filtersPerThread, int colorCache,
	bool scale, bool checkImgBounds>
	__global__ void filterActs_YxX_sparse2_preload_ty_4_tx_32_i_4_f_16_c_4(float* images, float* filters, float* targets,
		const int N, const int OC,
		const int imgSizeY, const int imgSizeX, const int filterSize, const int paddingStart,
		const int moduleStride,
		const int numModulesY, const int numModulesX, const int imgStride, const int numImgColors,
		const int numGroups,
		const float scaleTargets, const float scaleOutputs,
		const bool conv/*, const bool noloads*/) 
{
	__shared__ float shFilters[colorCache][BLOCK_Y * filtersPerThread]; // pre-load 1 pixel from B_Y*filtersPerThread filters
	__shared__ float shImages[colorCache][BLOCK_X * imgsPerThread]; // pre-load 1 pixel from B_X*imgsPerThread images
	const int imgPixels = imgSizeY * imgSizeX;
	const int filterPixels = filterSize * filterSize;
	const int numFilterColors = numImgColors / numGroups;
	const int blocksPerModule = OC / (BLOCK_Y*filtersPerThread);
	const int moduleIdx = blockIdx.y / blocksPerModule;
	const int blockFilterIdx = filtersPerThread * BLOCK_Y * (blockIdx.y % blocksPerModule);
	const int numFiltersPerGroup = OC / numGroups;
	const int blockGroupIdx = blockFilterIdx / numFiltersPerGroup;

	const int numModules = numModulesX * numModulesY;
	const int blockColorIdx = numFilterColors * blockGroupIdx;
	// Another fun insanity: the % B_X makes things faster, even thought threadIdx.x is
	// in the range 0..31. It appears that this allows the compiler to optimize?
	const int tx = threadIdx.x % BLOCK_X;
	const int ty = threadIdx.y % BLOCK_Y;
	const int tidx = ty * BLOCK_X + threadIdx.x;

	const int imgLoadModPosY = paddingStart + (moduleIdx / numModulesX) * moduleStride;
	const int imgLoadModPosX = paddingStart + (moduleIdx % numModulesX) * moduleStride;

	const int shFilterLoadY = tidx / (BLOCK_Y * filtersPerThread);
	const int shFilterLoadX = tidx % (BLOCK_Y * filtersPerThread);
	const int myImgIdx = blockIdx.x * BLOCK_X * imgsPerThread + threadIdx.x;

	images += (blockColorIdx + threadIdx.y) * imgPixels * imgStride + myImgIdx;
	filters += blockFilterIdx
		+ shFilterLoadY * OC * filterPixels + shFilterLoadX;
	if (!conv) {
		filters += moduleIdx * numFilterColors * filterPixels * OC;
	}

	targets += moduleIdx * N
		+ (blockFilterIdx + threadIdx.y * filtersPerThread) * N * numModules
		+ myImgIdx;

	float prod[imgsPerThread][filtersPerThread];
	//    float fCache[filtersPerThread];
#pragma unroll
	for (int i = 0; i < imgsPerThread; i++) {
#pragma unroll
		for (int f = 0; f < filtersPerThread; f++) {
			prod[i][f] = 0;
		}
	}
	// NOTE: these max/min functions increase register usage as compared to my macros
	const int imgStartX = max(0, imgLoadModPosX);
	const int imgStartY = max(0, imgLoadModPosY);
	const int imgEndX = min(imgLoadModPosX + filterSize, imgSizeX);
	const int imgEndY = min(imgLoadModPosY + filterSize, imgSizeY);
	//    __shared__ int imgPos[]

	int fPidx, iPidx;
	float imPreload[imgsPerThread];
	float fPreload[colorCache*filtersPerThread / BLOCK_X];
	//    float fCache[filtersPerThread];

	filterActs_YxX_sparse2_preload_ty_4_tx_32_f_16_c_4_setPixelCoords(filterSize, imgSizeX, imgLoadModPosY, imgLoadModPosX, imgStartY, imgStartX, fPidx, iPidx);

#pragma unroll
	for (int i = 0; i < imgsPerThread; i++) {
		if (!checkImgBounds || myImgIdx + i * BLOCK_X < N) {
			imPreload[i] = images[imgStride * iPidx + i * BLOCK_X];
		}
		else {
			imPreload[i] = 0;
		}
	}
	if (/*B_X % filtersPerThread == 0 ||*/ shFilterLoadY < BLOCK_X / filtersPerThread) { // This if statement reduces reg usage..
#pragma unroll
		for (int c = 0; c < colorCache; c += BLOCK_X / filtersPerThread) {
			fPreload[c*filtersPerThread / BLOCK_X] = filters[(c * filterPixels + fPidx) * OC];
		}
	}
	for (int imgY = imgStartY; imgY < imgEndY; ++imgY) {
		//        const int filterPxY = imgY - imgLoadModPosY;
		for (int imgX = imgStartX; imgX < imgEndX; ++imgX) {
			//            const int filterPxX = imgX - imgLoadModPosX;
			//            const int p = filterPxY * filterSize + filterPxX;
			//            const int pixIdx = imgY * imgSizeX + imgX;// Pixel index in img
			//            setPixelCoords(filterSize, imgSizeX, imgLoadModPosY, imgLoadModPosX, imgY, imgX, &p, &pixIdx);
			//            float* m = &images[imgStride * pixIdx];
			const bool lastPixel = imgY == imgEndY - 1 && imgX == imgEndX - 1;
			int imgYNext = imgY;
			int imgXNext = imgX;
			int fPidxNext, iPidxNext;
			if (!lastPixel) {
				imgYNext = imgY + (imgX + 1 == imgEndX);
				imgXNext = imgX + 1 == imgEndX ? imgStartX : imgX + 1;
			}
			filterActs_YxX_sparse2_preload_ty_4_tx_32_f_16_c_4_setPixelCoords(filterSize, imgSizeX, imgLoadModPosY, imgLoadModPosX, imgYNext, imgXNext, fPidxNext, iPidxNext);
			for (int oc = 0; oc < numFilterColors; oc += colorCache) { // oc stands for outer color (loop)
				const float* ff = &filters[OC * ((oc + colorCache) * filterPixels + fPidx)];
				const float* mm = &images[imgStride * ((oc + colorCache) * imgPixels + iPidx)];
				if (oc == numFilterColors - colorCache) {
					ff = &filters[fPidxNext * OC];
					mm = &images[iPidxNext * imgStride];
					fPidx = fPidxNext;
					iPidx = iPidxNext;
				}

#pragma unroll
				for (int c = 0; c < colorCache; c += BLOCK_X / filtersPerThread) {
					shFilters[c + shFilterLoadY][shFilterLoadX] = fPreload[c*filtersPerThread / BLOCK_X];
				}

#pragma unroll
				for (int i = 0; i < imgsPerThread; i++) {
					// NOTE: bank conflicts here!
					shImages[ty][tx * imgsPerThread + i] = imPreload[i];
				}
				imPreload[0] = (checkImgBounds && myImgIdx + 0 * BLOCK_X >= N) ? 0 : mm[0 * BLOCK_X];
				imPreload[1] = (checkImgBounds && myImgIdx + 1 * BLOCK_X >= N) ? 0 : mm[1 * BLOCK_X];
				imPreload[2] = (checkImgBounds && myImgIdx + 2 * BLOCK_X >= N) ? 0 : mm[2 * BLOCK_X];

				__syncthreads();

#pragma unroll
				for (int i = 0; i < imgsPerThread; i++) {
#pragma unroll
					for (int f = 0; f < filtersPerThread; f++) {
						prod[i][f] += shImages[0][threadIdx.x * imgsPerThread + i] * shFilters[0][threadIdx.y * filtersPerThread + f];
					}
				}

				fPreload[0] = ff[0];

#pragma unroll
				for (int i = 0; i < imgsPerThread; i++) {
#pragma unroll
					for (int f = 0; f < filtersPerThread; f++) {
						prod[i][f] += shImages[1][threadIdx.x * imgsPerThread + i] * shFilters[1][threadIdx.y * filtersPerThread + f];
					}
				}

				fPreload[1] = ff[(BLOCK_X / filtersPerThread * filterPixels) * OC];

#pragma unroll
				for (int i = 0; i < imgsPerThread; i++) {
#pragma unroll
					for (int f = 0; f < filtersPerThread; f++) {
						prod[i][f] += shImages[2][threadIdx.x * imgsPerThread + i] * shFilters[2][threadIdx.y * filtersPerThread + f];
					}
				}

				imPreload[3] = (checkImgBounds && myImgIdx + 3 * BLOCK_X >= N) ? 0 : mm[3 * BLOCK_X];

#pragma unroll
				for (int i = 0; i < imgsPerThread; i++) {
#pragma unroll
					for (int f = 0; f < filtersPerThread; f++) {
						prod[i][f] += shImages[3][threadIdx.x * imgsPerThread + i] * shFilters[3][threadIdx.y * filtersPerThread + f];
					}
				}
				__syncthreads();
			}
		}
	}

	if (scale) {
#pragma unroll
		for (int f = 0; f < filtersPerThread; f++) {
#pragma unroll
			for (int i = 0; i < imgsPerThread; i++) {
				if (!checkImgBounds || myImgIdx + i * BLOCK_X < N) {
					targets[i * BLOCK_X + f * N * numModules] = scaleTargets * targets[i * BLOCK_X + f * N * numModules] + scaleOutputs * prod[i][f];
				}
			}
		}
	}
	else {
		// Note: reversing order of these loops saves 2 registers, but costs time
#pragma unroll
		for (int i = 0; i < imgsPerThread; i++) {
#pragma unroll
			for (int f = 0; f < filtersPerThread; f++) {
				if (!checkImgBounds || myImgIdx + i * BLOCK_X < N) {
					targets[i * BLOCK_X + f * N * numModules] = scaleOutputs * prod[i][f];
				}
			}
		}
	}
}

/*****************************Function Revision Record*****************************
 * Author: Tencent BestImage Team(ankerguo@tencent.com)                           *
 * Date:   2015-05-18                                                             *
 * Reason: Optimizing kernel to get faster speed according to GPU features        *
 * Method:                                                                        *
 *         1. reorganizing data structure to avoid bank conflict;                 *
 *         2. using vectorized data type;                                         *
 *         3. improving instruction-level parallelism;                            *
 *         4. removing redundant 'if' branches;                                   *
 *         5. removing local variables to save registers.                         *
 *********************************************************************************/

 /*
  * images:      (numImgColors, imgSizeY, imgSizeX, numImages) with stride given
  * filters:     (numFilterColors, filterPixels, numFilters) if conv
  *              (numModules, numFilterColors, filterPixels, numFilters) otherwise
  *
  * targets:     (numFilters, numModulesY, numModulesX, numImages)
  *
  */
template <int BLOCK_Y, int BLOCK_X, int imgsPerThread, int filtersPerThread, int colorCache,
	bool scale, bool checkImgBounds>
	__global__ void
	__launch_bounds__(128, 4)
	filterActs_YxX_sparse2_preload_ty_4_tx_32_i_4_f_16_c_4_tex(cudaTextureObject_t images, cudaTextureObject_t filters, float* targets,
		const int N, const int OC,
		const int IH, const int IW, const int FS, const int paddingStart,
		const int moduleStride,
		const int OH, const int OW, const int imgStride, const int numImgColors,
		const int numGroups,
		const float scaleTargets, const float scaleOutputs,
		const bool conv/*, const bool noloads*/) {
	// avoid bank conflict by reorganizing the data structure and improve the band width by using 'float2' instead of 'float'
	__shared__ float2 shFilters[colorCache / 2][BLOCK_Y * filtersPerThread]; // pre-load 1 pixel from B_Y*filtersPerThread filters
	__shared__ float2 shImages[colorCache][BLOCK_X * imgsPerThread / 2]; // pre-load 1 pixel from B_X*imgsPerThread images
	const int imgPixels = IH * IW;
	const int filterPixels = filterSize * filterSize;
	const int numFilterColors = numImgColors / numGroups;
	const int blocksPerModule = OC / (BLOCK_Y*filtersPerThread);
	const int moduleIdx = blockIdx.y / blocksPerModule;
	const int blockFilterIdx = filtersPerThread * BLOCK_Y * (blockIdx.y % blocksPerModule);
	const int numFiltersPerGroup = OC / numGroups;
	const int blockGroupIdx = blockFilterIdx / numFiltersPerGroup;

	const int numModules = OW * OH;
	const int blockColorIdx = numFilterColors * blockGroupIdx;
	// Another fun insanity: the % B_X makes things faster, even thought threadIdx.x is
	// in the range 0..31. It appears that this allows the compiler to optimize?
	const int tx = threadIdx.x % BLOCK_X;
	const int ty = threadIdx.y % BLOCK_Y;
	//const int tidx = ty * B_X + threadIdx.x; // reduce one register

	const int imgLoadModPosY = paddingStart + (moduleIdx / OW) * moduleStride;
	const int imgLoadModPosX = paddingStart + (moduleIdx % OW) * moduleStride;

	// reduce two registers
	//const int shFilterLoadY = tidx / (B_Y * filtersPerThread);
	//const int shFilterLoadX = tidx % (B_Y * filtersPerThread);
	const int myImgIdx = blockIdx.x * BLOCK_X * imgsPerThread + tx;
	const int imgOffset = (blockColorIdx + ty) * imgPixels * imgStride + myImgIdx;

	//    images += (blockColorIdx + threadIdx.y) * imgPixels * imgStride + myImgIdx;
	const int filterOffset = blockFilterIdx
		+ ((ty * BLOCK_X + tx) / (BLOCK_Y * filtersPerThread)) * OC * filterPixels + ((ty * BLOCK_X + tx) % (BLOCK_Y * filtersPerThread)) + (conv ? 0 : moduleIdx * numFilterColors * filterPixels * OC);
	//    filters +=blockFilterIdx
	//            + shFilterLoadY * numFilters * filterPixels + shFilterLoadX;
	//    if (!conv) {
	//        filters += moduleIdx * numFilterColors * filterPixels * numFilters;
	//    }

	targets += moduleIdx * N
		+ (blockFilterIdx + threadIdx.y * filtersPerThread) * N * numModules
		+ myImgIdx;

	// combine two registers into one
	const int numModImages = numModules * N;
	float prod[imgsPerThread][filtersPerThread];
	//    float fCache[filtersPerThread];
#pragma unroll
	for (int i = 0; i < imgsPerThread; i++) {
#pragma unroll
		for (int f = 0; f < filtersPerThread; f++) {
			prod[i][f] = 0;
		}
	}
	// NOTE: these max/min functions increase register usage as compared to my macros
	const int imgStartX = max(0, imgLoadModPosX);
	const int imgStartY = max(0, imgLoadModPosY);
	const int imgEndX = min(imgLoadModPosX + filterSize, IW);
	const int imgEndY = min(imgLoadModPosY + filterSize, IH);
	//    __shared__ int imgPos[]

	int fPidx, iPidx;
	float imPreload[imgsPerThread]; // [4]
	float fPreload[colorCache*filtersPerThread / BLOCK_X]; // [2]
//    float fCache[filtersPerThread];

	filterActs_YxX_sparse2_preload_ty_4_tx_32_f_16_c_4_setPixelCoords(filterSize, IW, imgLoadModPosY, imgLoadModPosX, imgStartY, imgStartX, fPidx, iPidx);

	// remove redundant conditions
#pragma unroll
	for (int i = 0; i < imgsPerThread; i++) {
		imPreload[i] = tex1Dfetch<float>(images, imgOffset + imgStride * iPidx + i * BLOCK_X);
	}

#pragma unroll
	for (int c = 0; c < colorCache; c += BLOCK_X / filtersPerThread) {
		fPreload[c*filtersPerThread / BLOCK_X] = tex1Dfetch<float>(filters, filterOffset + (c * filterPixels + fPidx) * OC);
	}
	for (int imgY = imgStartY; imgY < imgEndY; ++imgY) {
		//        const int filterPxY = imgY - imgLoadModPosY;
		for (int imgX = imgStartX; imgX < imgEndX; ++imgX) {
			//            const int filterPxX = imgX - imgLoadModPosX;
			//            const int p = filterPxY * filterSize + filterPxX;
			//            const int pixIdx = imgY * imgSizeX + imgX;// Pixel index in img
			//            setPixelCoords(filterSize, imgSizeX, imgLoadModPosY, imgLoadModPosX, imgY, imgX, &p, &pixIdx);
			//            float* m = &images[imgStride * pixIdx];
			const bool lastPixel = imgY == imgEndY - 1 && imgX == imgEndX - 1;
			int imgYNext = imgY;
			int imgXNext = imgX;
			int fPidxNext, iPidxNext;
			if (!lastPixel) {
				imgYNext = imgY + (imgX + 1 == imgEndX);
				imgXNext = imgX + 1 == imgEndX ? imgStartX : imgX + 1;
			}
			filterActs_YxX_sparse2_preload_ty_4_tx_32_f_16_c_4_setPixelCoords(FS, IW, imgLoadModPosY, imgLoadModPosX, imgYNext, imgXNext, fPidxNext, iPidxNext);
			for (int oc = 0; oc < numFilterColors; oc += colorCache) { // oc stands for outer color (loop)
				// store the preloaded pixel of filter and image into shared memory
				shFilters[(ty * BLOCK_X + tx) / (BLOCK_Y * filtersPerThread)][(ty * BLOCK_X + tx) % (BLOCK_Y * filtersPerThread)].x = fPreload[0];
				shFilters[(ty * BLOCK_X + tx) / (BLOCK_Y * filtersPerThread)][(ty * BLOCK_X + tx) % (BLOCK_Y * filtersPerThread)].y = fPreload[1];
				shImages[ty][tx].x = imPreload[0];
				shImages[ty][tx].y = imPreload[1];
				shImages[ty][tx + BLOCK_X].x = imPreload[2];
				shImages[ty][tx + BLOCK_X].y = imPreload[3];

				int imgOffset2 = imgOffset + imgStride * ((oc + colorCache) * imgPixels + iPidx);
				int filterOffset2 = filterOffset + OC * ((oc + colorCache) * filterPixels + fPidx);
				if (oc == numFilterColors - colorCache) {
					filterOffset2 = filterOffset + fPidxNext * OC;
					imgOffset2 = imgOffset + iPidxNext * imgStride;
					fPidx = fPidxNext;
					iPidx = iPidxNext;
				}

				// preload one pixel of filter and image from texture, and no need to check 'checkImgBounds' with all callers setting it as false
				imPreload[0] = tex1Dfetch<float>(images, imgOffset2);
				imPreload[1] = tex1Dfetch<float>(images, imgOffset2 + BLOCK_X);
				imPreload[2] = tex1Dfetch<float>(images, imgOffset2 + 2 * BLOCK_X);
				imPreload[3] = tex1Dfetch<float>(images, imgOffset2 + 3 * BLOCK_X);
				fPreload[0] = tex1Dfetch<float>(filters, filterOffset2);
				fPreload[1] = tex1Dfetch<float>(filters, filterOffset2 + 2 * filterPixels * OC);

				__syncthreads();

				// put together the instructions with same type to improve instruction-level parallelism 
				// calculate the convolution between images and filters
#pragma unroll 
				for (int f = 0; f < filtersPerThread; f++) {
#pragma unroll
					for (int r = 0; r < colorCache / 2; r++) {
						prod[0][f] += shImages[r][tx].x      * shFilters[r][ty*filtersPerThread + f].x;
						prod[1][f] += shImages[r][tx].y      * shFilters[r][ty*filtersPerThread + f].x;
						prod[2][f] += shImages[r][tx + BLOCK_X].x   * shFilters[r][ty*filtersPerThread + f].x;
						prod[3][f] += shImages[r][tx + BLOCK_X].y   * shFilters[r][ty*filtersPerThread + f].x;
						prod[0][f] += shImages[r + 2][tx].x    * shFilters[r][ty*filtersPerThread + f].y;
						prod[1][f] += shImages[r + 2][tx].y    * shFilters[r][ty*filtersPerThread + f].y;
						prod[2][f] += shImages[r + 2][tx + BLOCK_X].x * shFilters[r][ty*filtersPerThread + f].y;
						prod[3][f] += shImages[r + 2][tx + BLOCK_X].y * shFilters[r][ty*filtersPerThread + f].y;
					}
				}
				__syncthreads();
			}
		}
	}

	if (scale) {
#pragma unroll
		for (int f = 0; f < filtersPerThread; f++) {
#pragma unroll
			for (int i = 0; i < imgsPerThread; i++) {
				// remove the redundant condition for less registers
				targets[i * BLOCK_X + f * numModImages] = scaleTargets * targets[i * BLOCK_X + f * numModImages] + scaleOutputs * prod[i][f];
			}
		}
	}
	else {
		// Note: reversing order of these loops saves 2 registers, but costs time
#pragma unroll
		for (int i = 0; i < imgsPerThread; i++) {
#pragma unroll
			for (int f = 0; f < filtersPerThread; f++) {
				// remove the redundant condition for less registers
				targets[i * BLOCK_X + f * numModImages] = scaleOutputs * prod[i][f];
			}
		}
	}
}

/*
 * Block size B_YxB_X. Each block applies B_Y * filtersPerThread filters to B_X * imgsPerThread images.
 * threadIdx.x determines image
 * threadIdx.y determines filter
 *
 * blockIdx.x determines image batch of B_X * imgsPerThread
 * blockIdx.y determines filter batch of module and B_Y * filtersPerThread
 *
 * images:      (numColors, imgSizeY, imgSizeX, numImages) with stride given
 * filters:     (numColors, filterPixels, numFilters) if conv
 *              (numModules, numColors, filterPixels, numFilters) otherwise
 *
 * targets:     (numFilters, numModulesY, numModulesX, numImages)
 *
 *
 * Number of filters per module should be divisible by B_Y * filtersPerThread
 * checkImgBounds indicates whether number of images is divisible by B_X * imgsPerThread
 *
 * The imgSize here is the size of the actual image without the padding.
 *
 */
template <int BLOCK_Y, int BLOCK_X, int imgsPerThread, int filtersPerThread, int numColors, int pixelCache,
	bool scale, bool checkImgBounds>
	__global__ void filterActs_YxX_color(float* images, float* filters, float* targets,
		const int numImages, const int numFilters,
		const int imgSizeY, const int imgSizeX, const int filterSize, const int paddingStart,
		const int moduleStride,
		const int numModulesY, const int numModulesX, const int imgStride,
		const float scaleTargets, const float scaleOutputs,
		const bool conv) {
	__shared__ float shFilters[pixelCache*numColors][BLOCK_Y * filtersPerThread]; // pre-load pixelCache pixels from B_Y*filtersPerThread filters
	__shared__ float shImages[pixelCache*numColors][BLOCK_X * imgsPerThread]; // pre-load pixelCache pixels from B_X*imgsPerThread images
	const int imgPixels = IH * IW;
	const int filterPixels = filterSize * filterSize;

	const int blocksPerModule = OC / (BLOCK_Y*filtersPerThread);
	const int moduleIdx = blockIdx.y / blocksPerModule;
	const int blockFilterIdx = blockIdx.y % blocksPerModule;

	const int tidx = threadIdx.y * BLOCK_X + threadIdx.x;

	const int imgLoadModPosY = paddingStart + (moduleIdx / OW) * moduleStride;
	const int imgLoadModPosX = paddingStart + (moduleIdx % OW) * moduleStride;
	const int numModules = OH * OW;
	const int shFilterLoadY = tidx / (BLOCK_Y * filtersPerThread);
	const int shFilterLoadX = tidx % (BLOCK_Y * filtersPerThread);
	const int myImgIdx = blockIdx.x * BLOCK_X * imgsPerThread + threadIdx.x;
	images += myImgIdx;
	filters += filtersPerThread * BLOCK_Y * blockFilterIdx
		+ shFilterLoadY * OC + shFilterLoadX;
	if (!conv) {
		filters += moduleIdx * numColors * filterPixels * OC;
	}

	targets += moduleIdx * N
		+ (blockFilterIdx * BLOCK_Y * filtersPerThread + threadIdx.y*filtersPerThread) * N * OH * OW
		+ myImgIdx;


	float prod[filtersPerThread][imgsPerThread];
#pragma unroll
	for (int f = 0; f < filtersPerThread; f++) {
#pragma unroll
		for (int g = 0; g < imgsPerThread; g++) {
			prod[f][g] = 0;
		}
	}
	//float* shImgLoad = &shImages[0][threadIdx.x];
	for (int p = 0; p < filterPixels; p += pixelCache) {
		/*
		 * Load pixelCache pixels from B_Y*filtersPerThread filters
		 * This condition covers the case when B_X is not divisible by filtersPerThread.
		 * In this case, not all of the threads will participate in the loading operation.
		 * This ensures that in each loop iteration, an integer number of rows of shFilters
		 * are filled, which makes indexing simple.
		 */
		if (BLOCK_X % filtersPerThread == 0 || shFilterLoadY < BLOCK_X / filtersPerThread) {
#pragma unroll
			for (int p2 = 0; p2 < pixelCache; p2 += BLOCK_X / filtersPerThread) {
				const bool omit = pixelCache % (BLOCK_X / filtersPerThread) == 0;
				const int preloadPx = shFilterLoadY + p2;
				if (omit || preloadPx < pixelCache) {
					if (p + preloadPx < filterPixels) {
#pragma unroll
						for (int c = 0; c < numColors; c++) {
							shFilters[shFilterLoadY + p2 + c * pixelCache][shFilterLoadX] = filters[(c * filterPixels + p + p2) * OC];
						}
					}
					else {
#pragma unroll
						for (int c = 0; c < numColors; c++) {
							shFilters[shFilterLoadY + p2 + c * pixelCache][shFilterLoadX] = 0;
						}
					}
				}
			}
		}

		/*
		 * Load pixelCache pixels from B_X*imgsPerThread images.
		 */
#pragma unroll
		for (int ly = 0; ly < pixelCache; ly += BLOCK_Y) {
			const int preloadPx = ly + threadIdx.y;
			const int pixIdx = p + preloadPx;
			const bool omit = pixelCache % BLOCK_Y == 0; // Compile-time condition
			/*
			 * Don't load any image pixels corresponding to filter pixels that don't exist.
			 */
			if (pixIdx < filterPixels && (omit || preloadPx < pixelCache)) {
				const int x = imgLoadModPosX + pixIdx % FS;
				const int y = imgLoadModPosY + pixIdx / FS;

				if (y >= 0 && y < IH && x >= 0 && x < IW) {
					float* m = &images[imgStride * (y * IW + x)];

#pragma unroll
					for (int c = 0; c < numColors; c++) {
#pragma unroll
						for (int i = 0; i < imgsPerThread; i++) {
							if (!checkImgBounds || myImgIdx + i * BLOCK_X < N) {
								shImages[preloadPx + c * pixelCache][threadIdx.x * imgsPerThread + i] = m[c * imgStride * imgPixels + i * BLOCK_X];
							}
							else {
								shImages[preloadPx + c * pixelCache][threadIdx.x * imgsPerThread + i] = 0;
							}
						}
					}
				}
				else { // Padding
#pragma unroll
					for (int i = 0; i < imgsPerThread; i++) {
#pragma unroll
						for (int c = 0; c < numColors; c++) {
							shImages[preloadPx + c * pixelCache][threadIdx.x * imgsPerThread + i] = 0;
						}
					}
				}
			}
		}

		__syncthreads();

#pragma unroll
		for (int i = 0; i < pixelCache*numColors; i++) {
#pragma unroll
			for (int f = 0; f < filtersPerThread; f++) {
#pragma unroll
				for (int g = 0; g < imgsPerThread; g++) {
					prod[f][g] += shImages[i][g + threadIdx.x * imgsPerThread] * shFilters[i][threadIdx.y * filtersPerThread + f];
				}
			}
		}
		__syncthreads();
	}

	if (scale) {
#pragma unroll
		for (int f = 0; f < filtersPerThread; f++) {
#pragma unroll
			for (int g = 0; g < imgsPerThread; g++) {
				if (!checkImgBounds || myImgIdx + g * BLOCK_X < N) {
					targets[g * BLOCK_X + f * N * numModules] = scaleTargets * targets[g * BLOCK_X + f * N * numModules] + scaleOutputs * prod[f][g];
				}
			}
		}
	}
	else {
#pragma unroll
		for (int g = 0; g < imgsPerThread; g++) {
			if (!checkImgBounds || myImgIdx + g * BLOCK_X < N) {
#pragma unroll
				for (int f = 0; f < filtersPerThread; f++) {
					targets[g * BLOCK_X + f * N * numModules] = scaleOutputs * prod[f][g];
				}
			}
		}
	}
}

/*
 * Block size B_YxB_X. Each block applies B_Y * filtersPerThread filters to B_X * imgsPerThread images.
 * threadIdx.x determines image
 * threadIdx.y determines filter
 *
 * blockIdx.x determines image batch of B_X * imgsPerThread
 * blockIdx.y determines filter batch of B_Y * filtersPerThread
 *
 * images:      (numImgColors, imgSizeY, imgSizeX, numImages) with stride given
 * filters:     (numFilterColors, filterPixels, numFilters) if conv
 *              (numModules, numFilterColors, filterPixels, numFilters) otherwise
 *
 * targets:     (numFilters, numModulesY, numModulesX, numImages)
 *
 * B_Y one of 4, 8, 16
 * B_X one of 16, 32
 * imgsPerThread one of 1, 2, 4
 * filtersPerThread one of 1, 2, 4, 8
 * colorCache: how many colors to put into shmem
 *
 * numFilters should be divisible by B_Y * filtersPerThread
 * numImages be divisible by B_X * imgsPerThread
 * numFilterColors should be divisible by colorCache.
 * numImgColors must be even.
 * numFilters must be divisible by numGroups.
 * no restrictions on pixelCache
 * The imgSize here is the size of the actual image without the padding.
 * As always, try to make B_X * imgsPerThread == B_Y * filtersPerThread for maximum efficiency.
 *
 */
template <int BLOCK_Y, int BLOCK_X, int imgsPerThread, int filtersPerThread, int colorCache,
	bool scale, bool checkImgBounds>
	__global__ void filterActs_YxX_sparse2(float* images, float* filters, float* targets,
		const int numImages, const int numFilters,
		const int imgSizeY, const int imgSizeX, const int filterSize, const int paddingStart,
		const int moduleStride,
		const int numModulesY, const int numModulesX, const int imgStride, const int numImgColors,
		const int numGroups,
		const float scaleTargets, const float scaleOutputs,
		const bool conv) {
	__shared__ float shFilters[colorCache][BLOCK_Y * filtersPerThread]; // pre-load 1 pixel from B_Y*filtersPerThread filters
	__shared__ float shImages[colorCache][BLOCK_X * imgsPerThread]; // pre-load 1 pixel from B_X*imgsPerThread images
	const int imgPixels = IH * IW;
	const int filterPixels = filterSize * filterSize;
	const int numFilterColors = numImgColors / numGroups;
	const int blocksPerModule = OC / (BLOCK_Y*filtersPerThread);
	const int moduleIdx = blockIdx.y / blocksPerModule;
	const int blockFilterIdx = filtersPerThread * BLOCK_Y * (blockIdx.y % blocksPerModule);
	const int numFiltersPerGroup = OC / numGroups;
	const int blockGroupIdx = blockFilterIdx / numFiltersPerGroup;

	const int numModules = OW * OH;
	const int blockColorIdx = numFilterColors * blockGroupIdx;

	const int tidx = threadIdx.y * BLOCK_X + threadIdx.x;

	const int imgLoadModPosY = paddingStart + (moduleIdx / OW) * moduleStride;
	const int imgLoadModPosX = paddingStart + (moduleIdx % OW) * moduleStride;

	const int shFilterLoadY = tidx / (BLOCK_Y * filtersPerThread);
	const int shFilterLoadX = tidx % (BLOCK_Y * filtersPerThread);
	const int myImgIdx = blockIdx.x * BLOCK_X * imgsPerThread + threadIdx.x;

	images += (blockColorIdx + threadIdx.y) * imgPixels * imgStride + myImgIdx;
	filters += blockFilterIdx
		+ shFilterLoadY * OC * filterPixels + shFilterLoadX;
	if (!conv) {
		filters += moduleIdx * numFilterColors * filterPixels * OC;
	}

	targets += moduleIdx * N
		+ (blockFilterIdx + threadIdx.y) * N * numModules
		+ myImgIdx;

	float prod[filtersPerThread][imgsPerThread];
#pragma unroll
	for (int f = 0; f < filtersPerThread; f++) {
#pragma unroll
		for (int g = 0; g < imgsPerThread; g++) {
			prod[f][g] = 0;
		}
	}
	const int imgStartX = MAX(0, imgLoadModPosX);
	const int imgStartY = MAX(0, imgLoadModPosY);
	const int imgEndX = MIN(imgLoadModPosX + filterSize, IW);
	const int imgEndY = MIN(imgLoadModPosY + filterSize, IH);
	//    __shared__ int imgPos[]

	for (int imgY = imgStartY; imgY < imgEndY; ++imgY) {
		const int filterPxY = imgY - imgLoadModPosY;
		for (int imgX = imgStartX; imgX < imgEndX; ++imgX) {
			const int filterPxX = imgX - imgLoadModPosX;
			const int p = filterPxY * FS + filterPxX;
			for (int oc = 0; oc < numFilterColors; oc += colorCache) { // oc stands for outer color (loop)

				/*
				 * Load a pixel from B_Y*filtersPerThread filters
				 * This condition covers the case when B_X is not divisible by filtersPerThread.
				 * In this case, not all of the threads will participate in the loading operation.
				 * This ensures that in each loop iteration, an integer number of rows of shFilters
				 * are filled, which makes indexing simple.

				 * nvcc is behaving in a completely insane way: removing this condition under
				 * template parameters that guarantee it to be true actually slows down
				 * the computation.
				 *
				 */
				if (/*B_X % filtersPerThread == 0 ||*/ shFilterLoadY < BLOCK_X / filtersPerThread) {
#pragma unroll
					for (int c = 0; c < colorCache; c += BLOCK_X / filtersPerThread) {
						if (colorCache % (BLOCK_X / filtersPerThread) == 0 || c + shFilterLoadY < colorCache) {
							shFilters[c + shFilterLoadY][shFilterLoadX] = filters[((oc + c) * filterPixels + p) * OC];
						}
					}
				}

				/*
				 * Load a pixel from B_X*imgsPerThread images.
				 */
				const int pixIdx = imgY * IW + imgX;// Pixel index in img

				float* m = &images[imgStride * (oc * imgPixels + pixIdx)];
#pragma unroll
				for (int c = 0; c < colorCache; c += BLOCK_Y) {
					if (colorCache % BLOCK_Y == 0 || threadIdx.y + c < colorCache) {
#pragma unroll
						for (int i = 0; i < imgsPerThread; i++) {
							if (!checkImgBounds || myImgIdx + i * BLOCK_X < N) {
								shImages[c + threadIdx.y][threadIdx.x + i * BLOCK_X] = m[c * imgStride * imgPixels + i * BLOCK_X];
							}
							else {
								shImages[c + threadIdx.y][threadIdx.x + i * BLOCK_X] = 0;
							}
						}
					}
				}

				__syncthreads();

				for (int c = 0; c < colorCache; c++) {
#pragma unroll
					for (int g = 0; g < imgsPerThread; g++) {
#pragma unroll
						for (int f = 0; f < filtersPerThread; f++) {
							prod[f][g] += shImages[c][g * BLOCK_X + threadIdx.x] * shFilters[c][threadIdx.y + f * BLOCK_Y];
						}
					}
				}
				__syncthreads();
			}
		}
	}

	if (scale) {
#pragma unroll
		for (int g = 0; g < imgsPerThread; g++) {
			if (!checkImgBounds || myImgIdx + g * BLOCK_X < N) {
#pragma unroll
				for (int f = 0; f < filtersPerThread; f++) {
					targets[g * BLOCK_X + f * BLOCK_Y * N * numModules] = scaleTargets * targets[g * BLOCK_X + f * BLOCK_Y * N * numModules] + scaleOutputs * prod[f][g];
				}
			}
		}
	}
	else {
		// Note: reversing order of these loops saves 2 registers, but costs time
#pragma unroll
		for (int f = 0; f < filtersPerThread; f++) {
#pragma unroll
			for (int g = 0; g < imgsPerThread; g++) {
				if (!checkImgBounds || myImgIdx + g * BLOCK_X < N) {
					targets[g * BLOCK_X + f * BLOCK_Y * N * numModules] = scaleOutputs * prod[f][g];
				}
			}
		}
	}
}


/*****************************Function Revision Record*****************************
 * Author: Tencent BestImage Team(ankerguo@tencent.com)                           *
 * Date:   2015-05-18                                                             *
 * Reason: Optimizing kernel to get faster speed according to GPU features        *
 * Method:                                                                        *
 *         1. reorganizing data structure to avoid bank conflict;                 *
 *         2. using vectorized data type;                                         *
 * Note:   This function can be used when each thread loads even number of filter *
 *         pixels(filtersPerThread * colorCache / B_X is even), and this can be   *
 *         optimized more when the number of loaded image's pixel is even.        *
 *********************************************************************************/
template <int BLOCK_Y, int BLOCK_X, int imgsPerThread, int filtersPerThread, int colorCache,
	bool scale, bool checkImgBounds>
	__global__ void filterActs_YxX_sparse2_f_vec(float* images, float* filters, float* targets,
		const int numImages, const int numFilters,
		const int imgSizeY, const int imgSizeX, const int filterSize, const int paddingStart,
		const int moduleStride,
		const int numModulesY, const int numModulesX, const int imgStride, const int numImgColors,
		const int numGroups,
		const float scaleTargets, const float scaleOutputs,
		const bool conv) {
	// improve shared memory's band width by using 'float2' instead of 'float'
	__shared__ float2 shFilters[colorCache / 2][BLOCK_Y * filtersPerThread]; // pre-load 1 pixel from B_Y*filtersPerThread filters
	__shared__ float shImages[colorCache][BLOCK_X * imgsPerThread]; // pre-load 1 pixel from B_X*imgsPerThread images

	const int tx = threadIdx.x % BLOCK_X, ty = threadIdx.y % BLOCK_Y;
	const int imgPixels = IH * IW;
	const int filterPixels = filterSize * filterSize;
	const int numFilterColors = numImgColors / numGroups;
	const int blocksPerModule = OC / (BLOCK_Y*filtersPerThread);
	const int moduleIdx = blockIdx.y / blocksPerModule;
	const int blockFilterIdx = filtersPerThread * BLOCK_Y * (blockIdx.y % blocksPerModule);
	const int numFiltersPerGroup = OC / numGroups;
	const int blockGroupIdx = blockFilterIdx / numFiltersPerGroup;

	const int numModules = OW * OH;
	const int blockColorIdx = numFilterColors * blockGroupIdx;

	const int tidx = ty * BLOCK_X + tx;

	const int imgLoadModPosY = paddingStart + (moduleIdx / OW) * moduleStride;
	const int imgLoadModPosX = paddingStart + (moduleIdx % OW) * moduleStride;

	// load position of filters' pixels for current thread
	const int shFilterLoadY = tidx / (BLOCK_Y * filtersPerThread);
	const int shFilterLoadX = tidx % (BLOCK_Y * filtersPerThread);
	// load position of images' pixels for current thread
	const int shImgLoadY = tidx / (BLOCK_X * imgsPerThread);
	const int shImgLoadX = tidx % (BLOCK_X * imgsPerThread);

	const int myImgIdx = blockIdx.x * BLOCK_X * imgsPerThread + shImgLoadX;
	images += (blockColorIdx + shImgLoadY) * imgPixels * imgStride + myImgIdx;

	filters += blockFilterIdx
		+ shFilterLoadY * OC * filterPixels + shFilterLoadX;
	if (!conv) {
		filters += moduleIdx * numFilterColors * filterPixels * OC;
	}

	targets += moduleIdx * N
		+ (blockFilterIdx + ty) * N * numModules
		+ blockIdx.x * BLOCK_X * imgsPerThread + tx;

	float prod[filtersPerThread][imgsPerThread];
#pragma unroll
	for (int f = 0; f < filtersPerThread; f++) {
#pragma unroll
		for (int g = 0; g < imgsPerThread; g++) {
			prod[f][g] = 0;
		}
	}

	const int imgStartX = MAX(0, imgLoadModPosX);
	const int imgStartY = MAX(0, imgLoadModPosY);
	const int imgEndX = MIN(imgLoadModPosX + filterSize, IW);
	const int imgEndY = MIN(imgLoadModPosY + filterSize, IH);

	// temporary buffer to store the filter's loaded pixels during each loop
	float fPreload[colorCache * filtersPerThread / BLOCK_X];
	// temporary buffer to store the image's loaded pixels during each loop
	float iPreload[colorCache * imgsPerThread / BLOCK_Y];

	// preload filter's pixels
#pragma unroll
	for (int c = 0; c < colorCache; c += BLOCK_X / filtersPerThread) {
		fPreload[c * filtersPerThread / BLOCK_X] = filters[(c * filterPixels + (imgStartY - imgLoadModPosY) * FS + (imgStartX - imgLoadModPosX)) * OC];
	}

	// preload image's pixels
	if (!checkImgBounds || myImgIdx < N) {
#pragma unroll
		for (int c = 0; c < colorCache; c += BLOCK_Y / imgsPerThread) {
			iPreload[c * imgsPerThread / BLOCK_Y] = images[(c * imgPixels + imgStartY * IW + imgStartX) * imgStride];
		}
	}
	else {
#pragma unroll
		for (int c = 0; c < colorCache; c += BLOCK_Y / imgsPerThread) {
			iPreload[c * imgsPerThread / BLOCK_Y] = 0;
		}
	}

	for (int imgY = imgStartY; imgY < imgEndY; ++imgY) {
		//const int filterPxY = imgY - imgLoadModPosY;
		for (int imgX = imgStartX; imgX < imgEndX; ++imgX) {
			for (int oc = 0; oc < numFilterColors; oc += colorCache) { // oc stands for outer color (loop)
				// store the preloaded filter's pixels into shared memory
#pragma unroll
				for (int c = 0; c < colorCache / 2; c += BLOCK_X / filtersPerThread) {
					shFilters[c + shFilterLoadY][shFilterLoadX].x = fPreload[c * filtersPerThread / BLOCK_X];
					shFilters[c + shFilterLoadY][shFilterLoadX].y = fPreload[(c + colorCache / 2) * filtersPerThread / BLOCK_X];
				}

				// store the preloaded image's pixels into shared memory
#pragma unroll
				for (int c = 0; c < colorCache; c += BLOCK_Y / imgsPerThread) {
					shImages[c + shImgLoadY][shImgLoadX] = iPreload[c * imgsPerThread / BLOCK_Y];
				}
				/*
				 * Load a pixel from B_Y*filtersPerThread filters
				 * This condition covers the case when B_X is not divisible by filtersPerThread.
				 * In this case, not all of the threads will participate in the loading operation.
				 * This ensures that in each loop iteration, an integer number of rows of shFilters
				 * are filled, which makes indexing simple.

				 * nvcc is behaving in a completely insane way: removing this condition under
				 * template parameters that guarantee it to be true actually slows down
				 * the computation.
				 *
				 */

				 /* preload image and filter pixels' data */
				if ((oc + colorCache) == numFilterColors) { // move to next pixel when all colors of current pixel have been finished
					int imgXn = (imgX < (imgEndX - 1)) ? (imgX + 1) : imgStartX;
					int imgYn = imgY + (imgXn != (imgX + 1));

#pragma unroll
					for (int c = 0; c < colorCache; c += BLOCK_X / filtersPerThread) {
						fPreload[c * filtersPerThread / BLOCK_X] = filters[(c * filterPixels + (imgYn - imgLoadModPosY) * FS + (imgXn - imgLoadModPosX)) * OC];
					}

					if (!checkImgBounds || myImgIdx < N) {
#pragma unroll
						for (int c = 0; c < colorCache; c += BLOCK_Y / imgsPerThread) {
							iPreload[c * imgsPerThread / BLOCK_Y] = images[(c * imgPixels + imgYn * IW + imgXn) * imgStride];
						}
					}
					else {
#pragma unroll
						for (int c = 0; c < colorCache; c += BLOCK_Y / imgsPerThread) {
							iPreload[c * imgsPerThread / BLOCK_Y] = 0;
						}
					}
				}
				else { // move next colorCache
#pragma unroll
					for (int c = 0; c < colorCache; c += BLOCK_X / filtersPerThread) {
						fPreload[c * filtersPerThread / BLOCK_X] = filters[((c + oc + colorCache) * filterPixels + (imgY - imgLoadModPosY) * FS + (imgX - imgLoadModPosX)) * OC];
					}

					if (!checkImgBounds || myImgIdx < N) {
#pragma unroll
						for (int c = 0; c < colorCache; c += BLOCK_Y / imgsPerThread) {
							iPreload[c * imgsPerThread / BLOCK_Y] = images[((c + oc + colorCache) * imgPixels + imgY * IW + imgX) * imgStride];
						}
					}
					else {
#pragma unroll
						for (int c = 0; c < colorCache; c += BLOCK_Y / imgsPerThread) {
							iPreload[c * imgsPerThread / BLOCK_Y] = 0;
						}
					}
				}

				__syncthreads();

				// convolution
				for (int c = 0; c < colorCache / 2; c++) {
#pragma unroll
					for (int g = 0; g < imgsPerThread; g++) {
#pragma unroll
						for (int f = 0; f < filtersPerThread; f++) {
							prod[f][g] += shImages[c][g * BLOCK_X + tx] * shFilters[c][ty + f * BLOCK_Y].x;
							prod[f][g] += shImages[c + colorCache / 2][g * BLOCK_X + tx] * shFilters[c][ty + f * BLOCK_Y].y;
						}
					}
				}
				__syncthreads();
			}
		}
	}

	// write convolution result into global memory
	if (scale) {
#pragma unroll
		for (int g = 0; g < imgsPerThread; g++) {
			if (!checkImgBounds || myImgIdx + g * BLOCK_X < N) {
#pragma unroll
				for (int f = 0; f < filtersPerThread; f++) {
					targets[g * BLOCK_X + f * BLOCK_Y * N * numModules] = scaleTargets * targets[g * BLOCK_X + f * BLOCK_Y * N * numModules] + scaleOutputs * prod[f][g];
				}
			}
		}
	}
	else {
		// Note: reversing order of these loops saves 2 registers, but costs time
#pragma unroll
		for (int f = 0; f < filtersPerThread; f++) {
#pragma unroll
			for (int g = 0; g < imgsPerThread; g++) {
				if (!checkImgBounds || myImgIdx + g * BLOCK_X < N) {
					targets[g * BLOCK_X + f * BLOCK_Y * N * numModules] = scaleOutputs * prod[f][g];
				}
			}
		}
	}
}
/*
 * images:      (numImgColors, imgSizeY, imgSizeX, numImages) with stride given
 * filters:     (numFilterColors, filterPixels, numFilters)             if conv
 *              (numModules, numFilterColors, filterPixels, numFilters) otherwise
 *
 * targets:     (numFilters, numModules, numImages)
 *
 * Note: all of these convolution routines are optimized for the case when
 * the number of images (i.e. the minibatch size) is a multiple of 128.
 * Other batch sizes will work, but but I made no attempt whatsoever
 * to make them work fast.
 */
void _filterActs(NVMatrix& images, NVMatrix& filters, NVMatrix& targets,
	int IH, int OH, int OW, int paddingStart, int moduleStride,
	int numImgColors, int numGroups,
	float scaleTargets, float scaleOutput, bool conv) {
	int numFilterColors = numImgColors / numGroups;
	int OC = filters.getNumCols();
	int numModules = OH * OW;
	int N = images.getNumCols();
	int imgPixels = images.getNumRows() / numImgColors;
	int IW = imgPixels / IH;
	int filterModuleMult = conv ? 1 : numModules;

	assert(numGroups > 1 || (numImgColors > 0 && (numImgColors <= 3 || numImgColors % 4 == 0)));
	assert(numGroups == 1 || numFilterColors % 4 == 0);
	assert(OC % (16 * numGroups) == 0);
	assert(numImgColors % numGroups == 0);
	//images.printShape("images");
	//printf("rows: %d, pixels: %d, colors: %d\n", images.getNumRows(), imgPixels, numImgColors);
	//images.printShape("images");
	assert(images.getNumRows() == imgPixels * numImgColors);
	assert(IH * IW == imgPixels);
	int numFiltersPerGroup = OC / numGroups;

	int imgStride = images.getStride(); // images does not need to be a contiguous matrix

	int filterPixels = filters.getNumRows() / (filterModuleMult * numFilterColors);
	int filterSize = int(sqrt(filterPixels));
	assert(filterSize * filterSize == filterPixels);
	assert(filters.getNumRows() == filterModuleMult * numFilterColors * filterPixels);

	// These routines don't handle the case when only part of the image is visited in the convolution
	assert(paddingStart <= 0);
	assert(paddingStart + (OW - 1)*moduleStride + filterSize >= IW);
	assert(paddingStart + (OH - 1)*moduleStride + filterSize >= IH);
	assert(moduleStride <= filterSize);

	assert(!images.isTrans());
	assert(!filters.isTrans());
	assert(!targets.isTrans());

	assert(filters.isContiguous());
	assert(targets.isContiguous());
	int imgsPerThread = N % 128 == 0 ? 4 : N % 64 == 0 ? 2 : 1;
	int filtersPerThread, threadsY = 4;
	if (numImgColors <= 3) {
		// Special kernels written for colors = 3, filters = 64 and colors = 3, filters = 48 cases.
		// The remaining cases use the old routines.
		// TODO: Modernize the remaining cases if you care about them.
		filtersPerThread = numFiltersPerGroup % 64 == 0 ? 16 : numFiltersPerGroup % 48 == 0 ? 12 : numFiltersPerGroup % 32 == 0 ? 8 : 4;
	}
	else {
		filtersPerThread = numFiltersPerGroup % 64 == 0 ? 16 : numFiltersPerGroup % 32 == 0 ? 8 : 4;
		threadsY = numFiltersPerGroup % 128 == 0 && numFilterColors % 8 == 0 && imgsPerThread != 4 ? 8 : 4;
	}
	int threadsX = 32;
	dim3 threads(threadsX, threadsY);
	dim3 blocks = dim3(DIVUP(numImages, threads.x * imgsPerThread), (numModules * numFilters) / (threads.y * filtersPerThread));

	bool checkImgBounds = N % (threads.x*imgsPerThread) != 0;
	bool scale = scaleTargets != 0;
	if (scaleTargets == 0) {
		targets.resize(OC * numModules, N);
	}
	else {
		assert(targets.getNumRows() == OC * numModules);
		assert(targets.getNumCols() == N);
	}
	cudaStream_t stream = NVMatrix::getDefaultStream();

	checkCudaErrors(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte)); // using wider band width

	// Auto-generated calling code...
	// NOTE: The calling code is set up such that if checkImgBounds is true, then imgsPerThread = 1.
	// In principle it doesn't have to be this way, and you may want to optimize for that case.

	if (scale == false) {
		if (checkImgBounds == false) {
			if (numFilterColors % 8 == 0) {
				if (N % 128 == 0) {
					if (numFiltersPerGroup % 128 == 0) {
						if (images.getNumDataBytes() < TEXTURE_SIZE_MAX) {
							cudaFuncSetCacheConfig(filterActs_YxX_sparse2_preload_ty_4_tx_32_i_4_f_16_c_4_tex < 4, 32, 4, 16, 4, false, false >, cudaFuncCachePreferL1);
							filterActs_YxX_sparse2_preload_ty_4_tx_32_i_4_f_16_c_4_tex < 4, 32, 4, 16, 4, false, false > << <blocks, threads, 0, stream >> > (images.getTextureObject(), filters.getTextureObject(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
						}
						else {
							cudaFuncSetCacheConfig(filterActs_YxX_sparse2_preload_ty_4_tx_32_i_4_f_16_c_4 < 4, 32, 4, 16, 4, false, false >, cudaFuncCachePreferL1);
							filterActs_YxX_sparse2_preload_ty_4_tx_32_i_4_f_16_c_4 < 4, 32, 4, 16, 4, false, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
						}
					}
					else if (numFiltersPerGroup % 64 == 0) {
						if (images.getNumDataBytes() < TEXTURE_SIZE_MAX) {
							cudaFuncSetCacheConfig(filterActs_YxX_sparse2_preload_ty_4_tx_32_i_4_f_16_c_4_tex < 4, 32, 4, 16, 4, false, false >, cudaFuncCachePreferL1);
							filterActs_YxX_sparse2_preload_ty_4_tx_32_i_4_f_16_c_4_tex < 4, 32, 4, 16, 4, false, false > << <blocks, threads, 0, stream >> > (images.getTextureObject(), filters.getTextureObject(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
						}
						else {
							cudaFuncSetCacheConfig(filterActs_YxX_sparse2_preload_ty_4_tx_32_i_4_f_16_c_4 < 4, 32, 4, 16, 4, false, false >, cudaFuncCachePreferL1);
							filterActs_YxX_sparse2_preload_ty_4_tx_32_i_4_f_16_c_4 < 4, 32, 4, 16, 4, false, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
						}
					}
					else if (numFiltersPerGroup % 32 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_sparse2_f_vec < 4, 32, 4, 8, 8, false, false >, cudaFuncCachePreferShared);
						filterActs_YxX_sparse2_f_vec < 4, 32, 4, 8, 8, false, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 1 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 4, 4, 8, false, false >, cudaFuncCachePreferShared);
						filterActs_YxX_sparse2 < 4, 32, 4, 4, 8, false, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
					}
				}
				else if (N % 64 == 0) {
					if (numFiltersPerGroup % 128 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_sparse2_f_vec < 8, 32, 2, 16, 8, false, false >, cudaFuncCachePreferShared);
						filterActs_YxX_sparse2_f_vec < 8, 32, 2, 16, 8, false, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 64 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_sparse2_f_vec < 4, 32, 2, 16, 8, false, false >, cudaFuncCachePreferShared);
						filterActs_YxX_sparse2_f_vec < 4, 32, 2, 16, 8, false, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 32 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_sparse2_f_vec < 4, 32, 2, 8, 8, false, false >, cudaFuncCachePreferShared);
						filterActs_YxX_sparse2_f_vec < 4, 32, 2, 8, 8, false, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 1 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 2, 4, 8, false, false >, cudaFuncCachePreferShared);
						filterActs_YxX_sparse2 < 4, 32, 2, 4, 8, false, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
					}
				}
				else if (N % 32 == 0) {
					if (numFiltersPerGroup % 128 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_sparse2_f_vec < 8, 32, 1, 16, 8, false, false >, cudaFuncCachePreferShared);
						filterActs_YxX_sparse2_f_vec < 8, 32, 1, 16, 8, false, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 64 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_sparse2_f_vec < 4, 32, 1, 16, 8, false, false >, cudaFuncCachePreferShared);
						filterActs_YxX_sparse2_f_vec < 4, 32, 1, 16, 8, false, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 32 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_sparse2_f_vec < 4, 32, 1, 8, 8, false, false >, cudaFuncCachePreferShared);
						filterActs_YxX_sparse2_f_vec < 4, 32, 1, 8, 8, false, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 1 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 4, 8, false, false >, cudaFuncCachePreferShared);
						filterActs_YxX_sparse2 < 4, 32, 1, 4, 8, false, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
					}
				}
			}
			else if (numFilterColors % 4 == 0) {
				if (N % 128 == 0) {
					if (numFiltersPerGroup % 128 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 4, 16, 4, false, false >, cudaFuncCachePreferShared);
						filterActs_YxX_sparse2 < 4, 32, 4, 16, 4, false, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 64 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 4, 16, 4, false, false >, cudaFuncCachePreferShared);
						filterActs_YxX_sparse2 < 4, 32, 4, 16, 4, false, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 32 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 4, 8, 4, false, false >, cudaFuncCachePreferShared);
						filterActs_YxX_sparse2 < 4, 32, 4, 8, 4, false, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 1 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 4, 4, 4, false, false >, cudaFuncCachePreferShared);
						filterActs_YxX_sparse2 < 4, 32, 4, 4, 4, false, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
					}
				}
				else if (N % 64 == 0) {
					if (numFiltersPerGroup % 128 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 2, 16, 4, false, false >, cudaFuncCachePreferShared);
						filterActs_YxX_sparse2 < 4, 32, 2, 16, 4, false, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 64 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 2, 16, 4, false, false >, cudaFuncCachePreferShared);
						filterActs_YxX_sparse2 < 4, 32, 2, 16, 4, false, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 32 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 2, 8, 4, false, false >, cudaFuncCachePreferShared);
						filterActs_YxX_sparse2 < 4, 32, 2, 8, 4, false, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 1 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 2, 4, 4, false, false >, cudaFuncCachePreferShared);
						filterActs_YxX_sparse2 < 4, 32, 2, 4, 4, false, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
					}
				}
				else if (N % 32 == 0) {
					if (numFiltersPerGroup % 128 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 16, 4, false, false >, cudaFuncCachePreferShared);
						filterActs_YxX_sparse2 < 4, 32, 1, 16, 4, false, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 64 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 16, 4, false, false >, cudaFuncCachePreferShared);
						filterActs_YxX_sparse2 < 4, 32, 1, 16, 4, false, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 32 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 8, 4, false, false >, cudaFuncCachePreferShared);
						filterActs_YxX_sparse2 < 4, 32, 1, 8, 4, false, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 1 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 4, 4, false, false >, cudaFuncCachePreferShared);
						filterActs_YxX_sparse2 < 4, 32, 1, 4, 4, false, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
					}
				}
			}
			else if (numFilterColors == 3) {
				if (N % 128 == 0) {
					if (numFiltersPerGroup % 64 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color_preload_ty_4_tx_32_i_4_f_16_px_4_cc_3_tex < 4, 32, 4, 16, 3, 4, false, false >, cudaFuncCachePreferShared);
						filterActs_YxX_color_preload_ty_4_tx_32_i_4_f_16_px_4_cc_3_tex < 4, 32, 4, 16, 3, 4, false, false > << <blocks, threads, 0, stream >> > (images.getTextureObject(), filters.getTextureObject(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 48 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color_preload_ty_4_tx_32_i_4_f_12_px_4_cc_3_tex < 4, 32, 4, 12, 3, 4, false, false >, cudaFuncCachePreferShared);
						filterActs_YxX_color_preload_ty_4_tx_32_i_4_f_12_px_4_cc_3_tex < 4, 32, 4, 12, 3, 4, false, false > << <blocks, threads, 0, stream >> > (images.getTextureObject(), filters.getTextureObject(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 32 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 8, 3, 4, false, false >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 4, 8, 3, 4, false, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 1 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 4, 3, 4, false, false >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 4, 4, 3, 4, false, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
				}
				else if (N % 64 == 0) {
					if (numFiltersPerGroup % 64 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 16, 3, 4, false, false >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 2, 16, 3, 4, false, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 48 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 12, 3, 4, false, false >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 2, 12, 3, 4, false, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 32 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 8, 3, 4, false, false >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 2, 8, 3, 4, false, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 1 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 4, 3, 4, false, false >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 2, 4, 3, 4, false, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
				}
				else if (N % 32 == 0) {
					if (numFiltersPerGroup % 64 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 16, 3, 4, false, false >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 1, 16, 3, 4, false, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 48 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 12, 3, 4, false, false >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 1, 12, 3, 4, false, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 32 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 8, 3, 4, false, false >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 1, 8, 3, 4, false, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 1 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 4, 3, 4, false, false >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 1, 4, 3, 4, false, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
				}
			}
			else if (numFilterColors == 2) {
				if (N % 128 == 0) {
					if (numFiltersPerGroup % 64 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 16, 2, 4, false, false >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 4, 16, 2, 4, false, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 48 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 12, 2, 4, false, false >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 4, 12, 2, 4, false, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 32 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 8, 2, 4, false, false >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 4, 8, 2, 4, false, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 1 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 4, 2, 4, false, false >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 4, 4, 2, 4, false, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
				}
				else if (N % 64 == 0) {
					if (numFiltersPerGroup % 64 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 16, 2, 4, false, false >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 2, 16, 2, 4, false, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 48 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 12, 2, 4, false, false >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 2, 12, 2, 4, false, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 32 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 8, 2, 4, false, false >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 2, 8, 2, 4, false, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 1 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 4, 2, 4, false, false >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 2, 4, 2, 4, false, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
				}
				else if (N % 32 == 0) {
					if (numFiltersPerGroup % 64 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 16, 2, 4, false, false >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 1, 16, 2, 4, false, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 48 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 12, 2, 4, false, false >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 1, 12, 2, 4, false, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 32 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 8, 2, 4, false, false >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 1, 8, 2, 4, false, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 1 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 4, 2, 4, false, false >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 1, 4, 2, 4, false, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
				}
			}
			else if (numFilterColors == 1) {
				if (N % 128 == 0) {
					if (numFiltersPerGroup % 64 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 16, 1, 4, false, false >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 4, 16, 1, 4, false, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 48 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 12, 1, 4, false, false >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 4, 12, 1, 4, false, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 32 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 8, 1, 4, false, false >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 4, 8, 1, 4, false, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 1 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 4, 1, 4, false, false >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 4, 4, 1, 4, false, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
				}
				else if (N % 64 == 0) {
					if (numFiltersPerGroup % 64 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 16, 1, 4, false, false >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 2, 16, 1, 4, false, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 48 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 12, 1, 4, false, false >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 2, 12, 1, 4, false, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 32 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 8, 1, 4, false, false >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 2, 8, 1, 4, false, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 1 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 4, 1, 4, false, false >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 2, 4, 1, 4, false, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
				}
				else if (N % 32 == 0) {
					if (numFiltersPerGroup % 64 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 16, 1, 4, false, false >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 1, 16, 1, 4, false, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 48 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 12, 1, 4, false, false >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 1, 12, 1, 4, false, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 32 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 8, 1, 4, false, false >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 1, 8, 1, 4, false, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 1 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 4, 1, 4, false, false >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 1, 4, 1, 4, false, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
				}
			}
		}
		else if (checkImgBounds == true) {
			if (numFilterColors % 8 == 0) {
				if (N % 1 == 0) {
					if (numFiltersPerGroup % 128 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 8, 32, 1, 16, 8, false, true >, cudaFuncCachePreferShared);
						filterActs_YxX_sparse2 < 8, 32, 1, 16, 8, false, true > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 64 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 16, 8, false, true >, cudaFuncCachePreferShared);
						filterActs_YxX_sparse2 < 4, 32, 1, 16, 8, false, true > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 32 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 8, 8, false, true >, cudaFuncCachePreferShared);
						filterActs_YxX_sparse2 < 4, 32, 1, 8, 8, false, true > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 1 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 4, 8, false, true >, cudaFuncCachePreferShared);
						filterActs_YxX_sparse2 < 4, 32, 1, 4, 8, false, true > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
					}
				}
			}
			else if (numFilterColors % 4 == 0) {
				if (N % 1 == 0) {
					if (numFiltersPerGroup % 128 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 16, 4, false, true >, cudaFuncCachePreferShared);
						filterActs_YxX_sparse2 < 4, 32, 1, 16, 4, false, true > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 64 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 16, 4, false, true >, cudaFuncCachePreferShared);
						filterActs_YxX_sparse2 < 4, 32, 1, 16, 4, false, true > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 32 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 8, 4, false, true >, cudaFuncCachePreferShared);
						filterActs_YxX_sparse2 < 4, 32, 1, 8, 4, false, true > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 1 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 4, 4, false, true >, cudaFuncCachePreferShared);
						filterActs_YxX_sparse2 < 4, 32, 1, 4, 4, false, true > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
					}
				}
			}
			else if (numFilterColors == 3) {
				if (N % 1 == 0) {
					if (numFiltersPerGroup % 64 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 16, 3, 4, false, true >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 1, 16, 3, 4, false, true > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 48 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 12, 3, 4, false, true >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 1, 12, 3, 4, false, true > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 32 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 8, 3, 4, false, true >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 1, 8, 3, 4, false, true > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 1 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 4, 3, 4, false, true >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 1, 4, 3, 4, false, true > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
				}
			}
			else if (numFilterColors == 2) {
				if (N % 1 == 0) {
					if (numFiltersPerGroup % 64 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 16, 2, 4, false, true >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 1, 16, 2, 4, false, true > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 48 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 12, 2, 4, false, true >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 1, 12, 2, 4, false, true > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 32 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 8, 2, 4, false, true >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 1, 8, 2, 4, false, true > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 1 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 4, 2, 4, false, true >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 1, 4, 2, 4, false, true > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
				}
			}
			else if (numFilterColors == 1) {
				if (N % 1 == 0) {
					if (numFiltersPerGroup % 64 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 16, 1, 4, false, true >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 1, 16, 1, 4, false, true > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 48 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 12, 1, 4, false, true >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 1, 12, 1, 4, false, true > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 32 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 8, 1, 4, false, true >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 1, 8, 1, 4, false, true > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 1 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 4, 1, 4, false, true >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 1, 4, 1, 4, false, true > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
				}
			}
		}
	}
	else if (scale == true) {
		if (checkImgBounds == false) {
			if (numFilterColors % 8 == 0) {
				if (N % 128 == 0) {
					if (numFiltersPerGroup % 128 == 0) {
						if (images.getNumDataBytes() < TEXTURE_SIZE_MAX) {
							cudaFuncSetCacheConfig(filterActs_YxX_sparse2_preload_ty_4_tx_32_i_4_f_16_c_4_tex < 4, 32, 4, 16, 4, true, false >, cudaFuncCachePreferL1);
							filterActs_YxX_sparse2_preload_ty_4_tx_32_i_4_f_16_c_4_tex < 4, 32, 4, 16, 4, true, false > << <blocks, threads, 0, stream >> > (images.getTextureObject(), filters.getTextureObject(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
						}
						else {
							cudaFuncSetCacheConfig(filterActs_YxX_sparse2_preload_ty_4_tx_32_i_4_f_16_c_4 < 4, 32, 4, 16, 4, true, false >, cudaFuncCachePreferL1);
							filterActs_YxX_sparse2_preload_ty_4_tx_32_i_4_f_16_c_4 < 4, 32, 4, 16, 4, true, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
						}
					}
					else if (numFiltersPerGroup % 64 == 0) {
						if (images.getNumDataBytes() < TEXTURE_SIZE_MAX) {
							cudaFuncSetCacheConfig(filterActs_YxX_sparse2_preload_ty_4_tx_32_i_4_f_16_c_4_tex < 4, 32, 4, 16, 4, true, false >, cudaFuncCachePreferL1);
							filterActs_YxX_sparse2_preload_ty_4_tx_32_i_4_f_16_c_4_tex < 4, 32, 4, 16, 4, true, false > << <blocks, threads, 0, stream >> > (images.getTextureObject(), filters.getTextureObject(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
						}
						else {
							cudaFuncSetCacheConfig(filterActs_YxX_sparse2_preload_ty_4_tx_32_i_4_f_16_c_4 < 4, 32, 4, 16, 4, true, false >, cudaFuncCachePreferL1);
							filterActs_YxX_sparse2_preload_ty_4_tx_32_i_4_f_16_c_4 < 4, 32, 4, 16, 4, true, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
						}
					}
					else if (numFiltersPerGroup % 32 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_sparse2_f_vec < 4, 32, 4, 8, 8, true, false >, cudaFuncCachePreferShared);
						filterActs_YxX_sparse2_f_vec < 4, 32, 4, 8, 8, true, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 1 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 4, 4, 8, true, false >, cudaFuncCachePreferShared);
						filterActs_YxX_sparse2 < 4, 32, 4, 4, 8, true, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
					}
				}
				else if (N % 64 == 0) {
					if (numFiltersPerGroup % 128 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_sparse2_f_vec < 8, 32, 2, 16, 8, true, false >, cudaFuncCachePreferShared);
						filterActs_YxX_sparse2_f_vec < 8, 32, 2, 16, 8, true, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 64 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_sparse2_f_vec < 4, 32, 2, 16, 8, true, false >, cudaFuncCachePreferShared);
						filterActs_YxX_sparse2_f_vec < 4, 32, 2, 16, 8, true, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 32 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_sparse2_f_vec < 4, 32, 2, 8, 8, true, false >, cudaFuncCachePreferShared);
						filterActs_YxX_sparse2_f_vec < 4, 32, 2, 8, 8, true, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 1 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 2, 4, 8, true, false >, cudaFuncCachePreferShared);
						filterActs_YxX_sparse2 < 4, 32, 2, 4, 8, true, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
					}
				}
				else if (N % 32 == 0) {
					if (numFiltersPerGroup % 128 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_sparse2_f_vec < 8, 32, 1, 16, 8, true, false >, cudaFuncCachePreferShared);
						filterActs_YxX_sparse2_f_vec < 8, 32, 1, 16, 8, true, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 64 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_sparse2_f_vec < 4, 32, 1, 16, 8, true, false >, cudaFuncCachePreferShared);
						filterActs_YxX_sparse2_f_vec < 4, 32, 1, 16, 8, true, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 32 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_sparse2_f_vec < 4, 32, 1, 8, 8, true, false >, cudaFuncCachePreferShared);
						filterActs_YxX_sparse2_f_vec < 4, 32, 1, 8, 8, true, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 1 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 4, 8, true, false >, cudaFuncCachePreferShared);
						filterActs_YxX_sparse2 < 4, 32, 1, 4, 8, true, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
					}
				}
			}
			else if (numFilterColors % 4 == 0) {
				if (N % 128 == 0) {
					if (numFiltersPerGroup % 128 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 4, 16, 4, true, false >, cudaFuncCachePreferShared);
						filterActs_YxX_sparse2 < 4, 32, 4, 16, 4, true, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 64 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 4, 16, 4, true, false >, cudaFuncCachePreferShared);
						filterActs_YxX_sparse2 < 4, 32, 4, 16, 4, true, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 32 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 4, 8, 4, true, false >, cudaFuncCachePreferShared);
						filterActs_YxX_sparse2 < 4, 32, 4, 8, 4, true, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 1 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 4, 4, 4, true, false >, cudaFuncCachePreferShared);
						filterActs_YxX_sparse2 < 4, 32, 4, 4, 4, true, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
					}
				}
				else if (N % 64 == 0) {
					if (numFiltersPerGroup % 128 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 2, 16, 4, true, false >, cudaFuncCachePreferShared);
						filterActs_YxX_sparse2 < 4, 32, 2, 16, 4, true, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 64 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 2, 16, 4, true, false >, cudaFuncCachePreferShared);
						filterActs_YxX_sparse2 < 4, 32, 2, 16, 4, true, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 32 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 2, 8, 4, true, false >, cudaFuncCachePreferShared);
						filterActs_YxX_sparse2 < 4, 32, 2, 8, 4, true, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 1 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 2, 4, 4, true, false >, cudaFuncCachePreferShared);
						filterActs_YxX_sparse2 < 4, 32, 2, 4, 4, true, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
					}
				}
				else if (N % 32 == 0) {
					if (numFiltersPerGroup % 128 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 16, 4, true, false >, cudaFuncCachePreferShared);
						filterActs_YxX_sparse2 < 4, 32, 1, 16, 4, true, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 64 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 16, 4, true, false >, cudaFuncCachePreferShared);
						filterActs_YxX_sparse2 < 4, 32, 1, 16, 4, true, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 32 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 8, 4, true, false >, cudaFuncCachePreferShared);
						filterActs_YxX_sparse2 < 4, 32, 1, 8, 4, true, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 1 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 4, 4, true, false >, cudaFuncCachePreferShared);
						filterActs_YxX_sparse2 < 4, 32, 1, 4, 4, true, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
					}
				}
			}
			else if (numFilterColors == 3) {
				if (N % 128 == 0) {
					if (numFiltersPerGroup % 64 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color_preload_ty_4_tx_32_i_4_f_16_px_4_cc_3_tex < 4, 32, 4, 16, 3, 4, true, false >, cudaFuncCachePreferShared);
						filterActs_YxX_color_preload_ty_4_tx_32_i_4_f_16_px_4_cc_3_tex < 4, 32, 4, 16, 3, 4, true, false > << <blocks, threads, 0, stream >> > (images.getTextureObject(), filters.getTextureObject(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 48 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color_preload_ty_4_tx_32_i_4_f_12_px_4_cc_3_tex < 4, 32, 4, 12, 3, 4, true, false >, cudaFuncCachePreferShared);
						filterActs_YxX_color_preload_ty_4_tx_32_i_4_f_12_px_4_cc_3_tex < 4, 32, 4, 12, 3, 4, true, false > << <blocks, threads, 0, stream >> > (images.getTextureObject(), filters.getTextureObject(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 32 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 8, 3, 4, true, false >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 4, 8, 3, 4, true, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 1 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 4, 3, 4, true, false >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 4, 4, 3, 4, true, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
				}
				else if (N % 64 == 0) {
					if (numFiltersPerGroup % 64 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 16, 3, 4, true, false >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 2, 16, 3, 4, true, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 48 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 12, 3, 4, true, false >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 2, 12, 3, 4, true, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 32 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 8, 3, 4, true, false >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 2, 8, 3, 4, true, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 1 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 4, 3, 4, true, false >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 2, 4, 3, 4, true, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
				}
				else if (N % 32 == 0) {
					if (numFiltersPerGroup % 64 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 16, 3, 4, true, false >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 1, 16, 3, 4, true, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 48 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 12, 3, 4, true, false >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 1, 12, 3, 4, true, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 32 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 8, 3, 4, true, false >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 1, 8, 3, 4, true, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 1 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 4, 3, 4, true, false >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 1, 4, 3, 4, true, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
				}
			}
			else if (numFilterColors == 2) {
				if (N % 128 == 0) {
					if (numFiltersPerGroup % 64 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 16, 2, 4, true, false >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 4, 16, 2, 4, true, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 48 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 12, 2, 4, true, false >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 4, 12, 2, 4, true, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 32 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 8, 2, 4, true, false >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 4, 8, 2, 4, true, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 1 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 4, 2, 4, true, false >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 4, 4, 2, 4, true, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
				}
				else if (N % 64 == 0) {
					if (numFiltersPerGroup % 64 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 16, 2, 4, true, false >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 2, 16, 2, 4, true, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 48 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 12, 2, 4, true, false >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 2, 12, 2, 4, true, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 32 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 8, 2, 4, true, false >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 2, 8, 2, 4, true, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 1 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 4, 2, 4, true, false >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 2, 4, 2, 4, true, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
				}
				else if (N % 32 == 0) {
					if (numFiltersPerGroup % 64 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 16, 2, 4, true, false >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 1, 16, 2, 4, true, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 48 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 12, 2, 4, true, false >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 1, 12, 2, 4, true, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 32 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 8, 2, 4, true, false >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 1, 8, 2, 4, true, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 1 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 4, 2, 4, true, false >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 1, 4, 2, 4, true, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
				}
			}
			else if (numFilterColors == 1) {
				if (N % 128 == 0) {
					if (numFiltersPerGroup % 64 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 16, 1, 4, true, false >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 4, 16, 1, 4, true, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 48 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 12, 1, 4, true, false >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 4, 12, 1, 4, true, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 32 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 8, 1, 4, true, false >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 4, 8, 1, 4, true, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 1 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 4, 4, 1, 4, true, false >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 4, 4, 1, 4, true, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
				}
				else if (N % 64 == 0) {
					if (numFiltersPerGroup % 64 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 16, 1, 4, true, false >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 2, 16, 1, 4, true, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 48 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 12, 1, 4, true, false >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 2, 12, 1, 4, true, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 32 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 8, 1, 4, true, false >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 2, 8, 1, 4, true, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 1 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 2, 4, 1, 4, true, false >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 2, 4, 1, 4, true, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
				}
				else if (N % 32 == 0) {
					if (numFiltersPerGroup % 64 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 16, 1, 4, true, false >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 1, 16, 1, 4, true, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 48 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 12, 1, 4, true, false >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 1, 12, 1, 4, true, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 32 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 8, 1, 4, true, false >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 1, 8, 1, 4, true, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 1 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 4, 1, 4, true, false >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 1, 4, 1, 4, true, false > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
				}
			}
		}
		else if (checkImgBounds == true) {
			if (numFilterColors % 8 == 0) {
				if (N % 1 == 0) {
					if (numFiltersPerGroup % 128 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 8, 32, 1, 16, 8, true, true >, cudaFuncCachePreferShared);
						filterActs_YxX_sparse2 < 8, 32, 1, 16, 8, true, true > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 64 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 16, 8, true, true >, cudaFuncCachePreferShared);
						filterActs_YxX_sparse2 < 4, 32, 1, 16, 8, true, true > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 32 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 8, 8, true, true >, cudaFuncCachePreferShared);
						filterActs_YxX_sparse2 < 4, 32, 1, 8, 8, true, true > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 1 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 4, 8, true, true >, cudaFuncCachePreferShared);
						filterActs_YxX_sparse2 < 4, 32, 1, 4, 8, true, true > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
					}
				}
			}
			else if (numFilterColors % 4 == 0) {
				if (N % 1 == 0) {
					if (numFiltersPerGroup % 128 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 16, 4, true, true >, cudaFuncCachePreferShared);
						filterActs_YxX_sparse2 < 4, 32, 1, 16, 4, true, true > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 64 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 16, 4, true, true >, cudaFuncCachePreferShared);
						filterActs_YxX_sparse2 < 4, 32, 1, 16, 4, true, true > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 32 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 8, 4, true, true >, cudaFuncCachePreferShared);
						filterActs_YxX_sparse2 < 4, 32, 1, 8, 4, true, true > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 1 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_sparse2 < 4, 32, 1, 4, 4, true, true >, cudaFuncCachePreferShared);
						filterActs_YxX_sparse2 < 4, 32, 1, 4, 4, true, true > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, numImgColors, numGroups, scaleTargets, scaleOutput, conv);
					}
				}
			}
			else if (numFilterColors == 3) {
				if (N % 1 == 0) {
					if (numFiltersPerGroup % 64 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 16, 3, 4, true, true >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 1, 16, 3, 4, true, true > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 48 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 12, 3, 4, true, true >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 1, 12, 3, 4, true, true > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 32 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 8, 3, 4, true, true >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 1, 8, 3, 4, true, true > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 1 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 4, 3, 4, true, true >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 1, 4, 3, 4, true, true > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
				}
			}
			else if (numFilterColors == 2) {
				if (N % 1 == 0) {
					if (numFiltersPerGroup % 64 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 16, 2, 4, true, true >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 1, 16, 2, 4, true, true > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 48 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 12, 2, 4, true, true >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 1, 12, 2, 4, true, true > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 32 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 8, 2, 4, true, true >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 1, 8, 2, 4, true, true > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 1 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 4, 2, 4, true, true >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 1, 4, 2, 4, true, true > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
				}
			}
			else if (numFilterColors == 1) {
				if (N % 1 == 0) {
					if (numFiltersPerGroup % 64 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 16, 1, 4, true, true >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 1, 16, 1, 4, true, true > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 48 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 12, 1, 4, true, true >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 1, 12, 1, 4, true, true > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 32 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 8, 1, 4, true, true >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 1, 8, 1, 4, true, true > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
					else if (numFiltersPerGroup % 1 == 0) {
						cudaFuncSetCacheConfig(filterActs_YxX_color < 4, 32, 1, 4, 1, 4, true, true >, cudaFuncCachePreferShared);
						filterActs_YxX_color < 4, 32, 1, 4, 1, 4, true, true > << <blocks, threads, 0, stream >> > (images.getDevData(), filters.getDevData(), targets.getDevData(), N, OC, IH, IW, FS, paddingStart, moduleStride, OH, OW, imgStride, scaleTargets, scaleOutput, conv);
					}
				}
			}
		}
	}
	checkCudaErrors(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte));

	getLastCudaError("filterActs: kernel execution failed");
}

void convFilterActs(NVMatrix& images, NVMatrix& filters, NVMatrix& targets,
	int IH, int OH, int OW, int paddingStart, int moduleStride,
	int numImgColors, int numGroups) {
	convFilterActs(images, filters, targets, IH, OH, OW, paddingStart, moduleStride, numImgColors, numGroups, 0, 1);
}

void convFilterActs(NVMatrix& images, NVMatrix& filters, NVMatrix& targets,
	int IH, int OH, int OW, int paddingStart, int moduleStride,
	int numImgColors, int numGroups,
	float scaleTargets, float scaleOutput) {
	_filterActs(images, filters, targets, IH, OH, OW, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, true);
}

void localFilterActs(NVMatrix& images, NVMatrix& filters, NVMatrix& targets,
	int IH, int OH, int OW, int paddingStart, int moduleStride,
	int numImgColors, int numGroups) {
	localFilterActs(images, filters, targets, IH, OH, OW, paddingStart, moduleStride, numImgColors, numGroups, 0, 1);
}

void localFilterActs(NVMatrix& images, NVMatrix& filters, NVMatrix& targets,
	int IH, int OH, int OW, int paddingStart, int moduleStride,
	int numImgColors, int numGroups,
	float scaleTargets, float scaleOutput) {
	_filterActs(images, filters, targets, IH, OH, OW, paddingStart, moduleStride, numImgColors, numGroups, scaleTargets, scaleOutput, false);
}

