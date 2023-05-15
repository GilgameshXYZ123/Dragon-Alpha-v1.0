#pragma once

#ifndef TEXTURE_H
#define TEXTURE_H

cudaTextureObject_t floatTexture(float* X, size_t size)
{
	cudaResourceDesc resDesc; memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeLinear;
	resDesc.res.linear.devPtr = X;
	resDesc.res.linear.sizeInBytes = sizeof(float) * size;
	resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
	resDesc.res.linear.desc.x = 32;

	cudaTextureDesc texDesc; memset(&texDesc, 0, sizeof(texDesc));
	texDesc.readMode = cudaReadModeElementType;
	texDesc.addressMode[0] = cudaAddressModeBorder;

	cudaTextureObject_t tex;
	cudaError_t error = cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL); printError(error);
	return tex;
}

cudaTextureObject_t floatTexture(float* X, size_t size, JNIEnv *env)
{
	cudaResourceDesc resDesc; memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeLinear;
	resDesc.res.linear.devPtr = X;
	resDesc.res.linear.sizeInBytes = sizeof(float) * size;
	resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
	resDesc.res.linear.desc.x = 32;

	cudaTextureDesc texDesc; memset(&texDesc, 0, sizeof(texDesc));
	texDesc.readMode = cudaReadModeElementType;
	texDesc.addressMode[0] = cudaAddressModeBorder;

	cudaTextureObject_t tex;
	cudaError_t error = cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL); handleError(error);
	return tex;
}

cudaTextureObject_t float4Texture(float* X, size_t size)
{
	cudaResourceDesc resDesc; memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeLinear;
	resDesc.res.linear.devPtr = X;
	resDesc.res.linear.sizeInBytes = sizeof(float) * size;
	resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
	resDesc.res.linear.desc.x = 32;
	resDesc.res.linear.desc.y = 32;
	resDesc.res.linear.desc.z = 32;
	resDesc.res.linear.desc.w = 32;

	cudaTextureDesc texDesc; memset(&texDesc, 0, sizeof(texDesc));
	texDesc.readMode = cudaReadModeElementType;
	texDesc.addressMode[0] = cudaAddressModeBorder;

	cudaTextureObject_t tex;
	cudaError_t error = cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL); printError(error);
	return tex;
}

cudaTextureObject_t float4Texture(float* X, size_t size, JNIEnv* env)
{
	cudaResourceDesc resDesc; memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeLinear;
	resDesc.res.linear.devPtr = X;
	resDesc.res.linear.sizeInBytes = sizeof(float) * size;
	resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
	resDesc.res.linear.desc.x = 32;
	resDesc.res.linear.desc.y = 32;
	resDesc.res.linear.desc.z = 32;
	resDesc.res.linear.desc.w = 32;

	cudaTextureDesc texDesc; memset(&texDesc, 0, sizeof(texDesc));
	texDesc.readMode = cudaReadModeElementType;
	texDesc.addressMode[0] = cudaAddressModeBorder;

	cudaTextureObject_t tex;
	cudaError_t error = cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL); handleError(error);
	return tex;
}

#endif
