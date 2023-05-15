#include "frame.cuh"
#include "CudaDevice.h"
#include "JNITool.cuh"

#define handleError(error) {if(error!=cudaSuccess) throwCudaException(env, error);}

JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_CudaDevice_setProperty(JNIEnv *env,
	jobject device, jint device_id)
{
	cudaDeviceProp prop;
	cudaError_t error = cudaGetDeviceProperties(&prop, device_id); handleError(error);
	//properies for the device=======================================
	jclass cls = env->GetObjectClass(device);

	jfieldID id = env->GetFieldID(cls, "name", "Ljava/lang/String;");
	env->SetObjectField(device, id, env->NewStringUTF(prop.name));

	id = env->GetFieldID(cls, "uuid", "[B");
	jbyteArray uuid = env->NewByteArray(16);
	env->SetByteArrayRegion(uuid, 0, 16, (jbyte*)prop.uuid.bytes);
	env->SetObjectField(device, id, uuid);

	id = env->GetFieldID(cls, "luid", "[B");
	jbyteArray luid = env->NewByteArray(8);
	env->SetByteArrayRegion(luid, 0, 8, (jbyte*)prop.luid);
	env->SetObjectField(device, id, luid);

	id = env->GetFieldID(cls, "luidDeviceNodeMask", "I");
	env->SetIntField(device, id, prop.luidDeviceNodeMask);

	id = env->GetFieldID(cls, "major", "I");
	env->SetIntField(device, id, prop.major);

	id = env->GetFieldID(cls, "minor", "I");
	env->SetIntField(device, id, prop.minor);

	id = env->GetFieldID(cls, "pciBusID", "I");
	env->SetIntField(device, id, prop.pciBusID);

	id = env->GetFieldID(cls, "pciDeviceID", "I");
	env->SetIntField(device, id, prop.pciDeviceID);

	id = env->GetFieldID(cls, "pciDomainID", "I");
	env->SetIntField(device, id, prop.pciDomainID);

	id = env->GetFieldID(cls, "multiGpuBoardGroupID", "I");
	env->SetIntField(device, id, prop.multiGpuBoardGroupID);

	id = env->GetFieldID(cls, "computeMode", "I");
	env->SetIntField(device, id, prop.computeMode);

	id = env->GetFieldID(cls, "deviceOverlap", "Z");
	env->SetBooleanField(device, id, prop.deviceOverlap);

	id = env->GetFieldID(cls, "integrated", "Z");
	env->SetBooleanField(device, id, prop.integrated);

	id = env->GetFieldID(cls, "kernelExecTimeoutEnabled", "Z");
	env->SetBooleanField(device, id, prop.kernelExecTimeoutEnabled);

	id = env->GetFieldID(cls, "unifiedAddressing", "Z");
	env->SetBooleanField(device, id, prop.unifiedAddressing);

	id = env->GetFieldID(cls, "canMapHostMemory", "Z");
	env->SetBooleanField(device, id, prop.canMapHostMemory);

	id = env->GetFieldID(cls, "canUseHostPointerForRegisteredMem", "Z");
	env->SetBooleanField(device, id, prop.canUseHostPointerForRegisteredMem);

	id = env->GetFieldID(cls, "directManagedMemAccessFromHost", "Z");
	env->SetBooleanField(device, id, prop.directManagedMemAccessFromHost);

	id = env->GetFieldID(cls, "concurrentKernels", "Z");
	env->SetBooleanField(device, id, prop.concurrentKernels);

	id = env->GetFieldID(cls, "ECCEnabled", "Z");
	env->SetBooleanField(device, id, prop.ECCEnabled);

	id = env->GetFieldID(cls, "tccDriver", "Z");
	env->SetBooleanField(device, id, prop.tccDriver);

	id = env->GetFieldID(cls, "streamPrioritiesSupported", "Z");
	env->SetBooleanField(device, id, prop.streamPrioritiesSupported);

	id = env->GetFieldID(cls, "globalL1CacheSupported", "Z");
	env->SetBooleanField(device, id, prop.globalL1CacheSupported);

	id = env->GetFieldID(cls, "localL1CacheSupported", "Z");
	env->SetBooleanField(device, id, prop.localL1CacheSupported);

	id = env->GetFieldID(cls, "managedMemory", "Z");
	env->SetBooleanField(device, id, prop.managedMemory);

	id = env->GetFieldID(cls, "multiGpuBoard", "Z");
	env->SetBooleanField(device, id, prop.isMultiGpuBoard);

	id = env->GetFieldID(cls, "hostNativeAtomicSupported", "Z");
	env->SetBooleanField(device, id, prop.hostNativeAtomicSupported);

	id = env->GetFieldID(cls, "pageableMemoryAccess", "Z");
	env->SetBooleanField(device, id, prop.pageableMemoryAccess);

	id = env->GetFieldID(cls, "cooperativeLaunch", "Z");
	env->SetBooleanField(device, id, prop.cooperativeLaunch);

	id = env->GetFieldID(cls, "concurrentManagedAccess", "Z");
	env->SetBooleanField(device, id, prop.concurrentManagedAccess);

	id = env->GetFieldID(cls, "computePreemptionSupported", "Z");
	env->SetBooleanField(device, id, prop.computePreemptionSupported);

	id = env->GetFieldID(cls, "clockRate", "I");
	env->SetIntField(device, id, prop.clockRate);

	id = env->GetFieldID(cls, "memoryClockRate", "I");
	env->SetIntField(device, id, prop.memoryClockRate);

	id = env->GetFieldID(cls, "memoryBusWidth", "I");
	env->SetIntField(device, id, prop.memoryBusWidth);

	id = env->GetFieldID(cls, "totalGlobalMemory", "J");
	env->SetLongField(device, id, prop.totalGlobalMem);

	id = env->GetFieldID(cls, "totalConstMemory", "J");
	env->SetLongField(device, id, prop.totalConstMem);

	id = env->GetFieldID(cls, "l2CacheSize", "I");
	env->SetIntField(device, id, prop.l2CacheSize);

	id = env->GetFieldID(cls, "persistingL2CacheMaxSize", "I");
	env->SetIntField(device, id, prop.persistingL2CacheMaxSize);

	id = env->GetFieldID(cls, "multiProcessorCount", "I");
	env->SetIntField(device, id, prop.multiProcessorCount);

	id = env->GetFieldID(cls, "asyncEngineCount", "I");
	env->SetIntField(device, id, prop.asyncEngineCount);

	id = env->GetFieldID(cls, "maxGridSize", "[I");
	jintArray maxGridSize = env->NewIntArray(3);
	env->SetIntArrayRegion(maxGridSize, 0, 3, (jint*)prop.maxGridSize);
	env->SetObjectField(device, id, maxGridSize);

	id = env->GetFieldID(cls, "singleToDoublePrecisionPerfRatio", "I");
	env->SetIntField(device, id, prop.singleToDoublePrecisionPerfRatio);

	id = env->GetFieldID(cls, "accessPolicyMaxWindowSize", "I");
	env->SetIntField(device, id, prop.accessPolicyMaxWindowSize);

	id = env->GetFieldID(cls, "memoryPitch", "J");
	env->SetLongField(device, id, prop.memPitch);

	//properties for a multiprocessor=============================
	id = env->GetFieldID(cls, "maxThreadsPerMultiProcessor", "I");
	env->SetIntField(device, id, prop.maxThreadsPerMultiProcessor);

	id = env->GetFieldID(cls, "sharedMemPerMultiprocessor", "J");
	env->SetLongField(device, id, prop.sharedMemPerMultiprocessor);

	id = env->GetFieldID(cls, "regsPerMultiprocessor", "I");
	env->SetIntField(device, id, prop.regsPerMultiprocessor);

	id = env->GetFieldID(cls, "maxBlocksPerMultiProcessor", "I");
	env->SetIntField(device, id, prop.maxBlocksPerMultiProcessor);

	//properties for a block ==============================
	id = env->GetFieldID(cls, "sharedMemPerBlock", "J");
	env->SetLongField(device, id, prop.sharedMemPerBlock);

	id = env->GetFieldID(cls, "reservedSharedMemPerBlock", "J");
	env->SetLongField(device, id, prop.reservedSharedMemPerBlock);

	id = env->GetFieldID(cls, "sharedMemPerBlockOptin", "J");
	env->SetLongField(device, id, prop.sharedMemPerBlockOptin);


	id = env->GetFieldID(cls, "regsPerBlock", "I");
	env->SetIntField(device, id, prop.regsPerBlock);

	id = env->GetFieldID(cls, "maxThreadsPerBlock", "I");
	env->SetIntField(device, id, prop.maxThreadsPerBlock);

	id = env->GetFieldID(cls, "maxThreadsDim", "[I");
	jintArray maxThreadsDim = env->NewIntArray(3);
	env->SetIntArrayRegion(maxThreadsDim, 0, 3, (jint*)prop.maxThreadsDim);
	env->SetObjectField(device, id, maxThreadsDim);

	id = env->GetFieldID(cls, "warpSize", "I");
	env->SetIntField(device, id, prop.warpSize);

	//properties from texture==========================
	id = env->GetFieldID(cls, "textureAlignment", "J");
	env->SetLongField(device, id, prop.textureAlignment);

	id = env->GetFieldID(cls, "texturePitchAlignment", "J");
	env->SetLongField(device, id, prop.texturePitchAlignment);

	id = env->GetFieldID(cls, "maxTexture1D", "I");
	env->SetLongField(device, id, prop.maxTexture1D);

	id = env->GetFieldID(cls, "maxTexture1DMipmap", "I");
	env->SetLongField(device, id, prop.maxTexture1DMipmap);

	id = env->GetFieldID(cls, "maxTexture1DMipmap", "I");
	env->SetLongField(device, id, prop.maxTexture1DLinear);

	id = env->GetFieldID(cls, "maxTexture2D", "[I");
	jintArray maxTexture2D = env->NewIntArray(2);
	env->SetIntArrayRegion(maxTexture2D, 0, 2, (jint*)prop.maxTexture2D);
	env->SetObjectField(device, id, maxTexture2D);

	id = env->GetFieldID(cls, "maxTexture2DMipmap", "[I");
	jintArray maxTexture2DMipmap = env->NewIntArray(2);
	env->SetIntArrayRegion(maxTexture2DMipmap, 0, 2, (jint*)prop.maxTexture2DMipmap);
	env->SetObjectField(device, id, maxTexture2DMipmap);

	id = env->GetFieldID(cls, "maxTexture2DLinear", "[I");
	jintArray maxTexture2DLinear = env->NewIntArray(3);
	env->SetIntArrayRegion(maxTexture2DLinear, 0, 3, (jint*)prop.maxTexture2DLinear);
	env->SetObjectField(device, id, maxTexture2DLinear);

	id = env->GetFieldID(cls, "maxTexture2DGather", "[I");
	jintArray maxTexture2DGather = env->NewIntArray(2);
	env->SetIntArrayRegion(maxTexture2DGather, 0, 2, (jint*)prop.maxTexture2DGather);

	id = env->GetFieldID(cls, "maxTexture3D", "[I");
	jintArray maxTexture3D = env->NewIntArray(3);
	env->SetIntArrayRegion(maxTexture3D, 0, 3, (jint*)prop.maxTexture3D);
	env->SetObjectField(device, id, maxTexture3D);

	id = env->GetFieldID(cls, "maxTexture3DAlt", "[I");
	jintArray maxTexture3DAlt = env->NewIntArray(3);
	env->SetIntArrayRegion(maxTexture3DAlt, 0, 3, (jint*)prop.maxTexture3DAlt);
	env->SetObjectField(device, id, maxTexture3D);

	id = env->GetFieldID(cls, "maxTexture1DLayered", "[I");
	jintArray maxTexture1DLayered = env->NewIntArray(2);
	env->SetIntArrayRegion(maxTexture1DLayered, 0, 2, (jint*)prop.maxTexture1DLayered);
	env->SetObjectField(device, id, maxTexture1DLayered);

	id = env->GetFieldID(cls, "maxTexture2DLayered", "[I");
	jintArray maxTexture2DLayered = env->NewIntArray(3);
	env->SetIntArrayRegion(maxTexture2DLayered, 0, 3, (jint*)prop.maxTexture2DLayered);
	env->SetObjectField(device, id, maxTexture2DLayered);

	id = env->GetFieldID(cls, "maxTextureCubemapLayered", "[I");
	jintArray maxTextureCubemapLayered = env->NewIntArray(2);
	env->SetIntArrayRegion(maxTextureCubemapLayered, 0, 2, (jint*)prop.maxTextureCubemapLayered);
	env->SetObjectField(device, id, maxTextureCubemapLayered);

	//properties for surface========================== 
	id = env->GetFieldID(cls, "surfaceAlignment", "J");
	env->SetLongField(device, id, prop.surfaceAlignment);

	id = env->GetFieldID(cls, "maxSurface1D", "I");
	env->SetIntField(device, id, prop.maxSurface1D);

	id = env->GetFieldID(cls, "maxSurface2D", "[I");
	jintArray maxSurface2D = env->NewIntArray(2);
	env->SetIntArrayRegion(maxSurface2D, 0, 2, (jint*)prop.maxSurface2D);
	env->SetObjectField(device, id, maxSurface2D);

	id = env->GetFieldID(cls, "maxSurface3D", "[I");
	jintArray maxSurface3D = env->NewIntArray(3);
	env->SetIntArrayRegion(maxSurface3D, 0, 3, (jint*)prop.maxSurface3D);
	env->SetObjectField(device, id, maxSurface3D);

	id = env->GetFieldID(cls, "maxSurface1DLayered", "[I");
	jintArray maxSurface1DLayered = env->NewIntArray(2);
	env->SetIntArrayRegion(maxSurface1DLayered, 0, 2, (jint*)prop.maxSurface1DLayered);
	env->SetObjectField(device, id, maxSurface1DLayered);

	id = env->GetFieldID(cls, "maxSurface2DLayered", "[I");
	jintArray maxSurface2DLayered = env->NewIntArray(3);
	env->SetIntArrayRegion(maxSurface2DLayered, 0, 3, (jint*)prop.maxSurface2DLayered);
	env->SetObjectField(device, id, maxSurface2DLayered);

	id = env->GetFieldID(cls, "maxSurfaceCubemap", "I");
	env->SetIntField(device, id, prop.maxSurfaceCubemap);

	id = env->GetFieldID(cls, "maxSurfaceCubemapLayered", "[I");
	jintArray maxSurfaceCubemapLayered = env->NewIntArray(2);
	env->SetIntArrayRegion(maxSurfaceCubemapLayered, 0, 2, (jint*)prop.maxSurfaceCubemapLayered);
	env->SetObjectField(device, id, maxSurfaceCubemapLayered);
}