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
public class Deconv3D extends SimpleUnit
{
    protected int IC, OC;
    protected int FH, FW;
    protected int sh, sw;
    protected int ph, pw;
    
    private int OH, OW;//reversed conv3D: (OH, OW) -> (IH, IW)
 
    protected boolean biased;
    protected Parameter W;
    protected Parameter B;
    
    public Deconv3D(boolean biasd,
            int in_channels, int out_channels, 
            int kernel_height, int kernel_width, 
            int stride_height, int stride_width,
            int padding_height, int padding_width,
            int output_height, int output_width) 
    {
        this.biased = biasd;
        this.IC = in_channels;    this.OC = out_channels;//transposed: swap(OC, IC)
        this.FH = kernel_height;  this.FW = kernel_width;
        this.sh = stride_height;  this.sw = stride_width;
        this.ph = padding_height; this.pw = padding_width;
        this.OH = output_height;  this.OW = output_width;
    }

    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public Tensor weight() { return W.ts(); }
    public Deconv3D weight(Tensor weight) {
        if(Tensor.isNull(weight)) throw new NullPointerException("weight is null");
        if(!weight.dimEquals(IC, FH, FW, OC)) throw new IllegalArgumentException(String.format(
                name + ": weight.dim != [in_channels: %d, kernel: (%d, %d), out_channels: %d]", 
                IC, FH, FW, OC));
        
        if(W != null) W.delete(); W.tensor(weight);
        return this;
    }
    
    public Tensor bias() { return B.ts(); }
    public Deconv3D bias(Tensor bias) {
        if(Tensor.isNull(bias)) throw new NullPointerException("bias is null");
        if(!bias.dimEquals(IC)) throw new IllegalArgumentException(String.format(
                name + ": bias.dim != [in_channels: %d]", IC));
        
        if(B != null) B.delete(); B.tensor(bias);
        return this;
    }
    
    public boolean biased() { return biased; }
    public int out_channels() { return OC; }
    public int in_channels() { return IC; }
    
    public int[] kernel()  { return new int[]{ FH, FW }; }
    public int[] stride()  { return new int[]{ sh, sw }; }
    public int[] padding() { return new int[]{ ph, pw }; }
    public int[] out_size() { return new int[] { OH, OW }; }
  
    public int[] fans() { return new int[]{ FH*FW*OC, FH*FW*IC }; }//[fan_in, fan_out]
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append("{ biased = ").append(biased);
        sb.append(", [in_channels, out_channels] = [").append(IC).append(", ").append(OC).append("]");
        sb.append(", kernel  = [").append(FH).append(", ").append(FW);
        sb.append(", stride  = [").append(sh).append(", ").append(sw).append(']');
        sb.append(", padding = [").append(ph).append(", ").append(pw).append(']');
        sb.append(", output_size = [").append(OH).append(OW).append("] }");
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: others">
    @Override
    protected void __init__(Engine eg) {
        Parameter.delete(W, B);
        
        Tensor tW = eg.empty(IC, FH, FW, OC);
        W = new Parameter(tW).need_grads(true);
        eg.kaiming_uniform(tW.c(), fans(), (float)Math.sqrt(5.0));//inplace: tW
        
        if(biased) {
            float bound = (float) (1.0 / Math.sqrt(FH * FW * IC));
            Tensor tB = eg.uniform(-bound, bound, OC);
            B = new Parameter(tB).need_grads(true);
        }
        
       Parameter.delete(W, B);
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
    @Override//when forward prop: W logically -> Wr
    protected Tensor __forward__(Tensor X) {
        if(OH == -1) OH = (X.dim(1) - 1)*sh + FH - (ph << 1);//reversed IH
        if(OW == -1) OW = (X.dim(2) - 1)*sw + FW - (pw << 1);//reversed IW
        
        //transposed: swap(OC, IC), X[N, OH, OW, OC] -> Y[N, IH, IW, IC]
        Tensor Y = eg.dconv3D(X, W.ts(), OH, OW, sh, sw, ph, pw);
        if(biased) Y = eg.add_row(true, Y.c(), B.ts());
        return Y;
    }

    @Override//when backward prop: conv(X, deltaY) -> gradient(Wr)^r -> gradient(W)
    protected Tensor __backward__(Tensor deltaY, boolean grad_inplace, boolean backward_grads) {
        Tensor deltaW = null, deltaB = null, deltaX = null;
        int gc_count = 0;
        
        if(W.need_grads()) {//conv3D.deltaY.size = dconv3D.X.size, conv3D.X.size = dconv3D.deltaY.size
            deltaW = eg.conv3D_deltaW(deltaY, holdX(), FH, FW, sh, sw, ph, pw);
            W.grads().add(deltaW);
            if(grad_inplace) gc_count++;
        }
        
        if(biased && B.need_grads()) {//mean: for each line: Yi[IH*IW, IC]
            deltaB = eg.field_sum(deltaY, OC);
            B.grads().add(deltaB);
            if(grad_inplace) gc_count++;
        }
        
        if(backward_grads) {
            deltaX = eg.conv3D_deltaX(deltaY, W.ts(), sh, sw, ph, pw);
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
