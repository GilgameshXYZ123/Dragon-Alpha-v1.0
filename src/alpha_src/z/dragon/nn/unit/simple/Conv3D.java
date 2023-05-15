/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.simple;

import z.dragon.common.state.State;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.Counter.CountGc;
import z.dragon.engine.Parameter;
import z.dragon.engine.Parameter.ParamMap;
import z.dragon.engine.Parameter.ParamSet;
import z.util.lang.annotation.Passed;

/**
 *
 * @author Gilgamesh
 */
@Passed("CudaFloat32Base")
public class Conv3D extends SimpleUnit
{
    protected int OC, IC;
    protected int FH, FW;
    protected int sh, sw;
    protected int ph, pw;
    
    private int IH, IW;
 
    protected boolean biased;
    protected Parameter W;
    protected Parameter B;
    
    public Conv3D(boolean biased,
            int in_channels, int out_channels, 
            int kernel_height, int kernel_width, 
            int stride_height, int stride_width,
            int padding_height, int padding_width) 
    {
        this.biased = biased;
        this.OC = out_channels;   this.IC = in_channels;
        this.FH = kernel_height;  this.FW = kernel_width;
        this.sh = stride_height;  this.sw = stride_width;
        this.ph = padding_height; this.pw = padding_width;
    }

    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public Tensor weight() { return W.ts(); }
    public Conv3D weight(Tensor weight) {
        if(Tensor.isNull(weight)) throw new NullPointerException("weight is null");
        if(!weight.dimEquals(OC, FH, FW, IC)) throw new IllegalArgumentException(String.format(
                name + ": weight.dim != [out_channels: %d, kernel: (%d, %d), in_channels: %d]",
                OC, FH, FW, IC));
        
        if(W != null) W.delete();
        W.tensor(weight);
        return this;
    }
    
    public Tensor bias() { return B.ts(); }
    public Conv3D bias(Tensor bias) {
        if(Tensor.isNull(bias)) throw new NullPointerException("bias is null");
        if(!bias.dimEquals(OC)) throw new IllegalArgumentException(String.format(
                name + ": bias.dim != [out_channels: %d]", OC));
        
        if(B != null) B.delete(); 
        B.tensor(bias);
        return this;
    }
    
    public boolean biased() { return biased; }
    public int out_channels() { return OC; }
    public int in_channels() { return IC; }
    
    public int[] kernel()  { return new int[]{ FH, FW }; }
    public int[] stride()  { return new int[]{ sh, sw }; }
    public int[] padding() { return new int[]{ ph, pw }; }
    
    public int[] fans() { return new int[] { FH*FW*IC, FH*FW*OC }; }//[fan_in, fan_out]
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append(" { biased = ").append(biased);
        sb.append(", [in_channels, out_channels] = [").append(IC).append(", ").append(OC).append(']');
        sb.append(", kernel = [").append(FH).append(", ").append(FW).append(']');
        sb.append(", stride = [").append(sh).append(", ").append(sw).append(']');
        sb.append(", padding = [").append(ph).append(", ").append(pw).append(" ] }");
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: others">
    @Override
    protected void __init__(Engine eg) {
        Parameter.delete(W, B);
        
        Tensor tW = eg.empty(OC, FH, FW, IC);
        W = new Parameter(tW).need_grads(true);
        eg.kaiming_uniform(tW.c(), fans(), (float)Math.sqrt(5.0));//inplace: tW
        
        if(biased) { 
            float bound = (float) (1.0 / Math.sqrt(FH * FW * IC));
            Tensor tB = eg.uniform(-bound, bound, OC);
            B = new Parameter(tB).need_grads(true);
        }
        
        Parameter.sync(W, B);
    }
    
    @Override 
    public void params(ParamSet set) { 
        set.add(W); 
        if(biased) set.add(B);
    }
    
    protected String weight_key() { return name + ".weight"; }
    protected String bias_key() { return name + ".bias"; }
       
    @Override
    public void param_map(ParamMap<String> map) {
        map.put(weight_key(), W);
        if(biased) map.put(bias_key(), B);
    }
    
    @Override
    public void state(State dic) {
        dic.put(weight_key(), W.ts());
        if(biased) dic.put(bias_key(), B.ts());
    }
    
    @Override
    public void update_state(State dic, boolean partial) {
        W.ts().set(dic.get(weight_key()), partial, name + ": fail to update state for weight");
        if(biased) B.ts().set(dic.get(bias_key()), partial, name + ": fail to update state for bias");
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
    @Override
    protected Tensor __forward__(Tensor X) { 
        int[] dimX = X.dim(); IH = dimX[1]; IW = dimX[2];
        return (biased? //X[N, IH, IW, IC] -> Y[N, OH, OW, OC]
                eg.conv3D_biased(X, W.ts(), sh, sw, ph, pw, B.ts()):
                eg.conv3D(X, W.ts(), sh, sw, ph, pw));
    }
    
    @Override
    protected Tensor __backward__(Tensor deltaY, boolean grad_inplace, boolean backward_grads) {
        Tensor deltaW = null, deltaB = null, deltaX = null;
        int gc_count = 0;
        
        if(W.need_grads()) {
            deltaW = eg.dconv3D_deltaW(holdX(), deltaY, FH, FW, sh, sw, ph, pw);
            W.grads().add(deltaW);
            if(grad_inplace) gc_count++;
        }
        
        if(biased && B.need_grads()) {
            deltaB = eg.field_sum(deltaY, OC);
            B.grads().add(deltaB);
            if(grad_inplace) gc_count++;
        }
        
        if(backward_grads) {
            deltaX = eg.dconv3D_deltaX(deltaY, W.ts(), IH, IW, sh, sw, ph, pw);
            W.ts().follow(deltaX);//When compute deltaX, W can't be changed
            if(grad_inplace) gc_count++;
        }
        
        if(gc_count != 0) {//when deltaW, deltaB, deltaX are cauculated, deltaY is not needed
            CountGc gc = new CountGc(gc_count, deltaY);
            if(deltaW != null) { deltaW.dual(()-> { gc.countDown(); }).remote_sync(); }
            if(deltaB != null) { deltaB.dual(()-> { gc.countDown(); }).remote_sync(); }
            if(deltaX != null) { deltaX.dual(()-> { gc.countDown(); }); }
        }
        
        return deltaX;
    }
    //</editor-fold>
}
