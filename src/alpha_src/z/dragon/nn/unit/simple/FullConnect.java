/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.simple;

import z.dragon.common.state.State;
import z.dragon.common.state.State.StateValue;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.Counter.CountGc;
import z.dragon.engine.Parameter;
import z.dragon.engine.Parameter.ParamMap;
import z.dragon.engine.Parameter.ParamSet;
import z.util.lang.annotation.Passed;

/**
 * <pre>
 * read X from the last layer.
 * alloc: W, B
 * compute: Y(the next layer will read)
 * 
 * read deltaY from the next layer:
 * alloc: deltaW, deltaX(is need)
 * compute: deltaX(the last layer will read)
 * 
 * forward: Y = X*W + B 
 * back: deltaX = deltaY*W^T
 * </pre>
 * @author Gilgamesh
 */
@Passed("CudaFloat32Base")
public class FullConnect extends SimpleUnit
{
    protected int IW, OW;
    
    protected boolean biased;
    protected Parameter W;
    protected Parameter B;
    
    public FullConnect(boolean biased, int in_features, int out_features) {
        this.biased = biased;
        this.IW = in_features;
        this.OW = out_features;
    }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public Tensor weight() { return W.ts(); }
    public FullConnect weight(Tensor weight) { 
        if(Tensor.isNull(weight)) throw new NullPointerException("weight is null");
        if(!weight.dimEquals(IW, OW)) throw new IllegalArgumentException(String.format(
                name + ": weight.dim != [in_features: %d, out_features: %d]", IW, OW));
        
        if(W != null) W.delete(); 
        W.tensor(weight);
        return this; 
    }
    
    public Tensor bias() { return B.ts(); }
    public FullConnect bias(Tensor bias) {
        if(Tensor.isNull(bias)) throw new NullPointerException("bias is null");
        if(!bias.dimEquals(OW)) throw new IllegalArgumentException(String.format(
                name + ": bias.dim != [out_features: %d]", OW));
        
        if(B != null) B.delete(); 
        B.tensor(bias);
        return this;
    }
    
    public boolean biased() { return biased; }
    public int in_features() { return IW; }
    public int out_features() { return OW; }
    
    public int[] fans() { return new int[]{ IW, OW }; }
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append(" { biased = ").append(biased);
        sb.append(", [in_features, out_features] = [").append(IW).append(", ").append(OW).append("] }");
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running area: others">
    @Override
    protected void __init__(Engine eg) {
        Parameter.delete(W, B);
       
        Tensor tW = eg.empty(IW, OW);
        W = new Parameter(tW).need_grads(true);
        eg.kaiming_uniform(tW.c(), fans(), (float)Math.sqrt(5.0));//inplace: tW
        
        if(biased) {
            float bound = (float) (1.0 / Math.sqrt(IW));
            Tensor tB = eg.uniform(-bound, bound, OW);
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
    
    //<editor-fold defaultstate="collapsed" desc="running area: propagation">
    @Override
    protected Tensor __forward__(Tensor X) {
        return (biased? 
                eg.fullconnect(X, W.ts(), B.ts()):
                eg.fullconnect(X, W.ts()));
    }

    @Override
    protected Tensor __backward__(Tensor deltaY, boolean grad_inplace, boolean backward_grads) {
        Tensor deltaW = null, deltaB = null, deltaX = null;
        int gc_count = 0;
        
        if(W.need_grads()) {//[K, N] * [N, M] -> [K, M]
            deltaW = eg.fullconnect_deltaW(holdX(), deltaY);
            W.grads().add(deltaW);
            if(grad_inplace) gc_count++;
        }
        
        if(biased && B.need_grads()) {
            deltaB = eg.field_sum(deltaY, OW);
            B.grads().add(deltaB);
            if(grad_inplace) gc_count++;
        }
        
        if(backward_grads) {
            deltaX = eg.fullconnect_deltaX(deltaY, W.ts());
            W.ts().follow(deltaX);//When compute deltaX, W can't be changed
            if(grad_inplace) gc_count++;
        }
        
        if(gc_count != 0) {//when deltaW, deltaB, deltaX are cauculated, deltaY is not needed
            CountGc gc = new CountGc(gc_count, deltaY);
            if(deltaW != null) { deltaW.dual(()-> { gc.countDown() ;}).remote_sync(); }
            if(deltaB != null) { deltaB.dual(()-> { gc.countDown(); }).remote_sync(); }
            if(deltaX != null) { deltaX.dual(()-> { gc.countDown(); }); }
        }
        
        return deltaX;
    }
    //</editor-fold>
}
