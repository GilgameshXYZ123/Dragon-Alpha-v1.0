#pragma once

//Method:    getDeviceId
JNIEXPORT jint JNICALL Java_z_dragon_engine_cuda_impl_Cuda_getDeviceId(JNIEnv *env, jclass cls)
{
	int dev_id;
	cudaError_t error = cudaGetDevice(&dev_id); handleError(error);
	return dev_id;
}

//Method:    getDeviceCount
JNIEXPORT jint JNICALL Java_z_dragon_engine_cuda_impl_Cuda_getDeviceCount(JNIEnv *env, jclass cls)
{
	int count;
	cudaError_t error = cudaGetDeviceCount(&count); handleError(error);
	return count;
}

//Method:    setDevice
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_Cuda_setDevice(JNIEnv *env, jclass cls, 
	jint dev_id)
{
	int current_dev_id;
	cudaError_t error = cudaGetDevice(&current_dev_id); handleError(error);
	if (current_dev_id != dev_id) {
		error = cudaSetDevice(dev_id); handleError(error);
	}
}

//Method:    resetDevice
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_Cuda_resetDevice(JNIEnv *env, jclass cls)
{
	cudaError_t error = cudaDeviceReset();
	handleError(error);
}

//Method:    deviceSynchronize
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_Cuda_deviceSynchronize(JNIEnv *env, jclass cls)
{
	cudaError_t error = cudaDeviceSynchronize();
	handleError(error);
}

//Method:    isDeviceP2PAccessEnabled
JNIEXPORT jboolean JNICALL Java_z_dragon_engine_cuda_impl_Cuda_isDeviceP2PAccessEnabled(JNIEnv *env, jclass cls, 
	jint dev1_id, jint dev2_id)
{
	int can;
	cudaError_t error = cudaDeviceCanAccessPeer(&can, dev1_id, dev2_id); handleError(error);
	return can;
}

//Method:    enableDeviceP2PAccess
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_Cuda_enableDeviceP2PAccess(JNIEnv *env, jclass cls,
	jint dev2_id)
{
	cudaError_t error = cudaDeviceEnablePeerAccess(dev2_id, NULL);
	handleError(error);
}

//Method:    disableDeviceP2PAccess
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_Cuda_disableDeviceP2PAccess(JNIEnv *env, jclass cls, 
	jint dev2_id)
{
	cudaError_t error = cudaDeviceDisablePeerAccess(dev2_id);
	handleError(error);
}