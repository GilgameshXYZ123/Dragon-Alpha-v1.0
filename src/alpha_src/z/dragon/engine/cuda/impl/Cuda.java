/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.engine.cuda.impl;

import z.dragon.engine.cuda.CudaFloat32EngineBase;

/**
 *
 * @author Gilgamesh
 */
public final class Cuda 
{
    private Cuda() {}
    
    //<editor-fold defaultstate="collapsed" desc="Low level Operators of Cuda Memory">
    public static native long malloc(long size);
    public static native void free(long address);
    
    public static native long mallocHost(long size);
    public static native void freeHost(long address);
    
    public static native void memsetAsync(long stream_address, long address, int value, long size);
    
    //<editor-fold defaultstate="collapsed" desc="cudaMemcpy">
    public static final int memcpyHostToDevice=0;
    public static final int memcpyHostToHost=1;
    public static final int memcpyDeviceToHost=2;
    public static final int memcpyDeviceToDevice=3;
    public static final int memcpyDefault=4;
    
    public static native void memcpyAsyncHostToDevice(long stream_address, long dst_address, long src_address, long size);
    public static native void memcpyAsyncHostToHost(long stream_address, long dst_address, long src_address, long size);
    public static native void memcpyAsyncDeviceToHost(long stream_address, long dst_address, long src_address, long size);
    public static native void memcpyAsyncDeviceToDevice(long stream_address, long dst_address, long src_address, long size);
    public static native void memcpyAsyncDefault(long stream_address, long dst_address, long src_address, long size);
    
    public static void cudaMemcpyAsync(long stream_address, long dst_address, long src_address, long size, int type)
    {
        if(type==memcpyHostToDevice) memcpyAsyncHostToDevice(stream_address, dst_address, src_address, size);
        else if(type==memcpyHostToHost) memcpyAsyncHostToHost(stream_address, dst_address, src_address, size);
        else if(type==memcpyDeviceToHost) memcpyAsyncDeviceToHost(stream_address, dst_address, src_address, size);
        else if(type==memcpyDeviceToDevice) memcpyAsyncDeviceToDevice(stream_address, dst_address, src_address, size);
        else memcpyAsyncDefault(stream_address, dst_address, src_address, size);
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="low level Operators of Cuda Device">
    public static native int getDeviceId();// current device id
    public static native int getDeviceCount();
    
    public static native void setDevice(int dev_id);
    //effect all threads of the current process, destroy all state and all allocation of the current device
    public static native void resetDevice();
    
    public static native void deviceSynchronize();
    
    public static native boolean isDeviceP2PAccessEnabled(int dev1_id, int dev2_id);
    public static native void enableDeviceP2PAccess(int dev2_id);
    public static native void disableDeviceP2PAccess(int dev2_id);
    public static void setDeviceP2PAccess(int dev2_id, boolean enabled) {
        
        if(enabled) Cuda.enableDeviceP2PAccess(dev2_id);
        else Cuda.disableDeviceP2PAccess(dev2_id);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="low level Operators of Cuda Stream">
    public static native int[] getStreamPriorityRange(); //[low bound, high bound]
     
    public static native long newStream_Blocking();
    public static native long newStream_Blocking(int priority);
    
    public static native long newStream_NonBlocking();
    public static native long newStream_NonBlocking(int priority);
    
    public static final int cudaStreamBlocking=0;
    public static final int cudaStreamNonBlocking=1;
    public static long newStream(int flag) {
        return flag==cudaStreamNonBlocking? 
                Cuda.newStream_NonBlocking():
                Cuda.newStream_Blocking();
    }
    public static long newStream(int flag, int priority) { 
        return flag==cudaStreamNonBlocking? 
                Cuda.newStream_NonBlocking(priority):
                Cuda.newStream_Blocking(priority);
    }
     
    public static native void deleteStream(long stream_address);
    public static native void deleteStream(long[] streams, int length);
    
    public static native boolean streamQuery(long stream_address);//isCompleted
    public static native void streamSynchronize(long stream_address);
    public static native void streamSynchronize(long[] streams, int length);
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="low level Operators of Cuda Event">
    public static native long newEvent_Default();
    public static native long newEvent_BlockingSync();
    public static native long newEvent_DisableTiming();
    public static native long newEvent_Interprocess();
    
    public static native void deleteEvent(long event_address);
    
    public static native void eventRecord(long event_address, long stream_address);
    public static native void eventRecord(long event_address, long[] streamArr, int length);
    
    public static native void eventSynchronize(long event_address);
    public static native float eventElapsedTime(long start_event_address, long stop_event_address);
    
    public static native void streamWaitEvent_default(long stream_address, long event_address);
    public static native void streamWaitEvent_external(long stream_address, long event_address);
    
    public static native void streamsWaitEvent_default(long[] streamArr, int length, long event_address);
    public static native void streamsWaitEvent_external(long[] streamArr, int length, long event_address);
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="low level Operators of Cuda Exception">
    public static native String getExceptionName(int type);//passed
    public static native String getExceptionInfo(int type);//passed
    public static native int getLastExceptionType();//passed
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Low Level Operators of Cuda float">
    public static native void get1D(long stream_address, long address, float[] value, int length);
    public static native void get2D(long stream_address, long address, float[] value, int height, int width, int stride);
    public static native void get1D_v2(long stream_address, long address, float[] value, long buf_address, int length);
    public static native void get2D_v2(long stream_address, long address, float[] value, long buf_address, int height, int width, int stride);
    
    public static native void set1D(long stream_address, long address, float value, int length);
    public static native void set2D(long stream_address, long address, float value, int height, int width, int stride);
    
    public static native void set1D(long stream_address, long address, byte[] value, int length);
    public static native void set1D(long stream_address, long address, float[] value, int length);
    public static native void set1D_v2(long stream_address, long address, byte[] value, long buf_address, int length);
    public static native void set1D_v2(long stream_address, long address, float[] value, long buf_address, int length);
    
    public static native void set2D(long stream_address, long address, byte[] value, int height, int width, int stride);
    public static native void set2D(long stream_address, long address, float[] value, int height, int width, int stride);
    public static native void set2D_v2(long stream_address, long address, byte[] value, long buf_address, int height, int width, int stride);
    public static native void set2D_v2(long stream_address, long address, float[] value, long buf_address, int height, int width, int stride);
    
    public static native void setFrom1Dto2D(long stream_address,
            long src_address, int src_length, 
            long dst_address, int dst_height, int dst_width, int dst_stride);
    public static native void setFrom2Dto1D(long stream_address,
            long src_address, int src_height, int src_width, int src_stride, 
            long dst_address, int dst_length);
    public static native void setFrom2Dto2D(long stream_address,
            long src_address, int src_height, int src_width, int src_stride, 
            long dst_address, int dst_height, int dst_width, int dst_stride);
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Low Level Operators of Cuda int(int32)">
    public static native void get1D_int(long stream_address, long address, int[] value, int length);
    public static native void get2D_int(long stream_address, long address, int[] value, int height, int width, int stride);
    public static native void get1D_v2_int(long stream_address, long address, int[] value, long buf_address, int length);
    public static native void get2D_v2_int(long stream_address, long address, int[] value, long buf_address, int height, int width, int stride);
    
    public static native void set1D_int(long stream_address, long address, int[] value, int length);
    public static native void set2D_int(long stream_address, long address, int[] value, int height, int width, int stride);
    
    public static native void set1D_v2_int(long stream_address, long address, int[] value, long buf_address, int length);
    public static native void set2D_v2_int(long stream_address, long address, int[] value, long buf_address, int height, int width, int stride);
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Low Level Operators of Cuda char(int8)">
    public static native void get1D_char(long stream_address, long address, byte[] value, int length);
    public static native void get2D_char(long stream_address, long address, byte[] value, int height, int width, int stride);
    public static native void get1D_v2_char(long stream_address, long address, byte[] value, long buf_address, int length);
    public static native void get2D_v2_char(long stream_address, long address, byte[] value, long buf_address, int height, int width, int stride);
    
    public static native void set1D_char(long stream_address, long address, byte[] value, int length);
    public static native void set2D_char(long stream_address, long address, byte[] value, int height, int width, int stride);
    
    public static native void set1D_v2_char(long stream_address, long address, byte[] value, long buf_address, int length);
    public static native void set2D_v2_char(long stream_address, long address, byte[] value, long buf_address, int height, int width, int stride);
    public static native void set2D_v2_char_W3S4(long stream_address,//designed for pictures with 3 channelss
            long address, byte[] value,
            long buf1_address, long buf2_address,
            int height);//[width, stride] = [3, 4]
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Init-Code">
    static
    {
        if(CudaFloat32EngineBase.__TEST_MODE__())
        synchronized(Cuda.class) {
            System.load("D:\\virtual disc Z-Gilgamesh\\Dargon-alpha\\engine-cuda-base\\Cuda\\x64\\Release\\Cuda.dll");
        }
    }
    //</editor-fold>  
}
