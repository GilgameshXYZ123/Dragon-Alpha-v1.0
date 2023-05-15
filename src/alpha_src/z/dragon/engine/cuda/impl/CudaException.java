/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.engine.cuda.impl;

/**
 *
 * @author Gilgamesh
 */

public final class CudaException extends RuntimeException
{
    private static final long serialVersionUID = 1L;
    
    private final int type;
    
    public CudaException() { this.type = CudaException.Unknown; }
    public CudaException(int type) { this.type = type; }
    public CudaException(String message) {
        super(message);
        this.type = CudaException.Unknown;
    }
    public CudaException(int type, String message) {
        super(message);
        this.type = type;
    }
    
    //<editor-fold defaultstate="collapsed" desc="CudaExceptionTypes">
    public static final int Success=0;
    public static final int InvalidValue=1;
    public static final int MemoryAllocation=2;
    public static final int InitializationError=3;
    public static final int CudartUnloading=4;
    public static final int ProfilerDisabled=5;
    public static final int ProfilerNotInitialized=6;
    public static final int ProfilerAlreadyStarted=7;
    public static final int ProfilerAlreadyStopped=8;
    public static final int InvalidConfiguration=9;
    public static final int InvalidPitchValue=12;
    public static final int InvalidSymbol=13;
    public static final int InvalidHostPointer=16;
    public static final int InvalidDevicePointer=17;
    public static final int InvalidTexture=18;
    public static final int InvalidTextureBinding=19;
    public static final int InvalidChannelDescriptor=20;
    public static final int InvalidMemcpyDirection=21;
    public static final int AddressOfConstant=22;
    public static final int TextureFetchFailed=23;
    public static final int TextureNotBound=24;
    public static final int SynchronizationError=25;
    public static final int InvalidFilterSetting=26;
    public static final int InvalidNormSetting=27;
    public static final int MixedDeviceExecution=28;
    public static final int NotYetImplemented=31;
    public static final int MemoryValueTooLarge=32;
    public static final int StubLibrary=34;
    public static final int InsufficientDriver=35;
    public static final int CallRequiresNewerDriver=36;
    public static final int InvalidSurface=37;
    public static final int DuplicateVariableName=43;
    public static final int DuplicateTextureName=44;
    public static final int DuplicateSurfaceName=45;
    public static final int DevicesUnavailable=46;
    public static final int IncompatibleDriverContext=49;
    public static final int MissingConfiguration=52;
    public static final int PriorLaunchFailure=53;
    public static final int LaunchMaxDepthExceeded=65;
    public static final int LaunchFileScopedTex=66;
    public static final int LaunchFileScopedSurf=67;
    public static final int SyncDepthExceeded=68;
    public static final int LaunchPendingCountExceeded=69;
    public static final int InvalidDeviceFunction=98;
    public static final int NoDevice=100;
    public static final int InvalidDevice=101;
    public static final int DeviceNotLicensed=102;
    public static final int SoftwareValidityNotEstablished=103;
    public static final int StartupFailure=127;
    public static final int InvalidKernelImage=200;
    public static final int DeviceUninitialized=201;
    public static final int MapBufferObjectFailed=205;
    public static final int UnmapBufferObjectFailed=206;
    public static final int ArrayIsMapped=207;
    public static final int AlreadyMapped=208;
    public static final int NoKernelImageForDevice=209;
    public static final int AlreadyAcquired=210;
    public static final int NotMapped=211;
    public static final int NotMappedAsArray=212;
    public static final int NotMappedAsPointer=213;
    public static final int ECCUncorrectable=214;
    public static final int UnsupportedLimit=215;
    public static final int DeviceAlreadyInUse=216;
    public static final int PeerAccessUnsupported=217;
    public static final int InvalidPtx=218;
    public static final int InvalidGraphicsContext=219;
    public static final int NvlinkUncorrectable=220;
    public static final int JitCompilerNotFound=221;
    public static final int UnsupportedPtxVersion=222;
    public static final int JitCompilationDisabled=223;
    public static final int UnsupportedExecAffinity=224;
    public static final int InvalidSource=300;
    public static final int FileNotFound=301;
    public static final int SharedObjectSymbolNotFound=302;
    public static final int SharedObjectInitFailed=303;
    public static final int OperatingSystem=304;
    public static final int InvalidResourceHandle=400;
    public static final int IllegalState=401;
    public static final int SymbolNotFound=500;
    public static final int NotReady=600;
    public static final int IllegalAddress=700;
    public static final int LaunchOutOfResources=701;
    public static final int LaunchTimeout=702;
    public static final int LaunchIncompatibleTexturing=703;
    public static final int PeerAccessAlreadyEnabled=704;
    public static final int PeerAccessNotEnabled=705;
    public static final int SetOnActiveProcess=708;
    public static final int ContextIsDestroyed=709;
    public static final int Assert=710;
    public static final int TooManyPeers=711;
    public static final int HostMemoryAlreadyRegistered=712;
    public static final int HostMemoryNotRegistered=713;
    public static final int HardwareStackError=714;
    public static final int IllegalInstruction=715;
    public static final int MisalignedAddress=716;
    public static final int InvalidAddressSpace=717;
    public static final int InvalidPc=718;
    public static final int LaunchFailure=719;
    public static final int CooperativeLaunchTooLarge=720;
    public static final int NotPermitted=800;
    public static final int NotSupported=801;
    public static final int SystemNotReady=802;
    public static final int SystemDriverMismatch=803;
    public static final int CompatNotSupportedOnDevice=804;
    public static final int MpsConnectionFailed=805;
    public static final int MpsRpcFailure=806;
    public static final int MpsServerNotReady=807;
    public static final int MpsMaxClientsReached=808;
    public static final int MpsMaxConnectionsReached=809;
    public static final int StreamCaptureUnsupported=900;
    public static final int StreamCaptureInvalidated=901;
    public static final int StreamCaptureMerge=902;
    public static final int StreamCaptureUnmatched=903;
    public static final int StreamCaptureUnjoined=904;
    public static final int StreamCaptureIsolation=905;
    public static final int StreamCaptureImplicit=906;
    public static final int CapturedEvent=907;
    public static final int StreamCaptureWrongThread=908;
    public static final int Timeout=909;
    public static final int GraphExecUpdateFailure=910;
    public static final int ExternalDevice=911;
    public static final int Unknown=999;
    public static final int ApiFailureBase=10000;
    //</editor-fold>
    
    public static CudaException lastException() {
        return new CudaException(Cuda.getLastExceptionType());
    }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public int type() { return type; }
    public String name() { return Cuda.getExceptionName(type); }
    public String info() { return Cuda.getExceptionInfo(type); }
    
    public void append(StringBuilder sb) {
        sb.append(super.toString());
        sb.append("{ type = ").append(type);
        sb.append(", name = ").append(name());
        sb.append(", Info = ").append(info());
        sb.append(" }");
    }
    
    @Override
    public String toString() {
        StringBuilder sb=new StringBuilder();
        this.append(sb);
        return sb.toString();
    }
    //</editor-fold>
}
