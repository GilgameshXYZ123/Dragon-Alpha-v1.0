#pragma once

//Method:    newEvent_Default
JNIEXPORT jlong JNICALL Java_z_dragon_engine_cuda_impl_Cuda_newEvent_1Default(JNIEnv *env, jclass cls) 
{
	cudaEvent_t event;
	cudaError_t error = cudaEventCreateWithFlags(&event, cudaEventDefault); handleError(error);
	return (intptr_t)event;
}

//Method:    newEvent_BlockingSync
JNIEXPORT jlong JNICALL Java_z_dragon_engine_cuda_impl_Cuda_newEvent_1BlockingSync(JNIEnv *env, jclass cls)
{
	cudaEvent_t event;
	cudaError_t error = cudaEventCreateWithFlags(&event, cudaEventBlockingSync); handleError(error);
	return (intptr_t)event;
}

//Method:    newEvent_DisableTimings
JNIEXPORT jlong JNICALL Java_z_dragon_engine_cuda_impl_Cuda_newEvent_1DisableTiming(JNIEnv *env, jclass cls)
{
	cudaEvent_t event;
	cudaError_t error = cudaEventCreateWithFlags(&event, cudaEventDisableTiming); handleError(error);
	return (intptr_t)event;
}

//Method:    newEvent_Interprocess
JNIEXPORT jlong JNICALL Java_z_dragon_engine_cuda_impl_Cuda_newEvent_1Interprocess(JNIEnv *env, jclass cls)
{
	cudaEvent_t event;
	cudaError_t error = cudaEventCreateWithFlags(&event, cudaEventInterprocess); handleError(error);
	return (intptr_t)event;
}

//Method:    deleteEvent
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_Cuda_deleteEvent(JNIEnv *env, jclass cls, 
	jlong event_address) 
{
	cudaEvent_t event = (cudaEvent_t)(intptr_t)event_address;
	cudaError_t error = cudaEventDestroy(event); handleError(error);
}

//Method:    eventRecord
JNIEXPORT void JNICALL JNICALL Java_z_dragon_engine_cuda_impl_Cuda_eventRecord__JJ(JNIEnv *env, jclass cls,
	jlong event_address, jlong stream_address)
{
	cudaEvent_t event = (cudaEvent_t)(intptr_t)event_address;
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	cudaError_t error = cudaEventRecord(event, stream); handleError(error);
}

//Method:    eventRecord
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_Cuda_eventRecord__J_3JI(JNIEnv *env, jclass cls,
	jlong event_address, jlongArray stream_arr, jint length)
{
	jlong *streamAddrs = env->GetLongArrayElements(stream_arr, NULL);
	for (int i = 0; i < length; i++) {
		cudaStream_t stream = (cudaStream_t)(intptr_t)streamAddrs[i];
		cudaError_t error = cudaStreamSynchronize(stream); handleError(error);
	}
	env->ReleaseLongArrayElements(stream_arr, streamAddrs, JNI_ABORT);
}

//Method:    eventSynchronize
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_Cuda_eventSynchronize(JNIEnv *env, jclass cls,
	jlong event_addresss)
{
	cudaEvent_t event = (cudaEvent_t)(intptr_t)event_addresss;
	cudaError_t error = cudaEventSynchronize(event); handleError(error);
}

//Method:    eventSynchronize_delete
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_Cuda_eventSynchronize_1delete(JNIEnv *env, jclass cls, 
	jlong event_addresss) 
{
	cudaEvent_t event = (cudaEvent_t)(intptr_t)event_addresss;
	cudaError_t error = cudaEventSynchronize(event); handleError(error);
	error = cudaEventDestroy(event); handleError(error);
}

//Method:    eventElapsedTime
JNIEXPORT jfloat JNICALL Java_z_dragon_engine_cuda_impl_Cuda_eventElapsedTime(JNIEnv *env, jclass cls,
	jlong start_event_address, jlong stop_event_address)
{
	cudaEvent_t start_event = (cudaEvent_t)(intptr_t)start_event_address;
	cudaEvent_t stop_event = (cudaEvent_t)(intptr_t)stop_event_address;
	float ms;
	cudaError_t error = cudaEventElapsedTime(&ms, start_event, stop_event); handleError(error);
	return ms;
}

//Method:    streamWaitEvent_default
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_Cuda_streamWaitEvent_1default(JNIEnv *env, jclass cls,
	jlong stream_address, jlong event_address)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	cudaEvent_t event = (cudaEvent_t)(intptr_t)event_address;
	cudaError_t error = cudaStreamWaitEvent(stream, event, cudaEventWaitDefault); handleError(error);
}

//Method:    streamWaitEvent_external
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_Cuda_streamWaitEvent_1external(JNIEnv *env, jclass cls,
	jlong stream_address, jlong event_address)
{
	cudaStream_t stream = (cudaStream_t)(intptr_t)stream_address;
	cudaEvent_t event = (cudaEvent_t)(intptr_t)event_address;
	cudaError_t error = cudaStreamWaitEvent(stream, event, cudaEventWaitExternal); handleError(error);
}

//Method:    streamsWaitEvent_default
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_Cuda_streamsWaitEvent_1default(JNIEnv *env, jclass cls,
	jlongArray stream_arr, jint length, jlong event_address)
{
	cudaEvent_t event = (cudaEvent_t)(intptr_t)event_address;
	jlong *streamAddrs = env->GetLongArrayElements(stream_arr, NULL);
	for (int i = 0; i < length; i++)
	{
		cudaStream_t stream = (cudaStream_t)(intptr_t)streamAddrs[i];
		cudaError_t error = cudaStreamWaitEvent(stream, event, cudaEventWaitDefault); handleError(error);
	}
	env->ReleaseLongArrayElements(stream_arr, streamAddrs, JNI_ABORT);
}

//Method:    streamsWaitEvent_external
JNIEXPORT void JNICALL Java_z_dragon_engine_cuda_impl_Cuda_streamsWaitEvent_1external(JNIEnv *env, jclass cls,
	jlongArray stream_arr, jint length, jlong event_address)
{
	cudaEvent_t event = (cudaEvent_t)(intptr_t)event_address;
	jlong *streamAddrs = env->GetLongArrayElements(stream_arr, NULL);
	for (int i = 0; i < length; i++)
	{
		cudaStream_t stream = (cudaStream_t)(intptr_t)streamAddrs[i];
		cudaError_t error = cudaStreamWaitEvent(stream, event, cudaEventWaitExternal); handleError(error);
	}
	env->ReleaseLongArrayElements(stream_arr, streamAddrs, JNI_ABORT);
}