/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.simple.affine;

import java.util.Arrays;
import z.dragon.common.state.State;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.Counter.CountGc;
import z.dragon.engine.Parameter;
import z.dragon.engine.Parameter.ParamMap;
import z.dragon.engine.Parameter.ParamSet;
import z.dragon.engine.Tensor.TensorSet;
import z.util.lang.annotation.Passed;

/**
 * Layer Normalization
 * @author Gilgamesh
 */
@Passed("CudaFloat32Base")
public class LayerNorm extends Affine
{
    protected boolean affine;
    
    protected Tensor X_mean;
    protected Tensor X_sqmean;
    protected float eps;
    
    public LayerNorm(boolean inplace, boolean affine,
            float eps, int... feature_dim) 
    {
        super(inplace, feature_dim);
        this.affine = affine;
        this.eps = eps;
    }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    @Override public LayerNorm weight(Tensor weight) { super.weight(weight); return this; }
    @Override public LayerNorm bias(Tensor bias) { super.bias(bias); return this; }
    
    public Tensor mean() { return X_mean; }
    public Tensor sqmean() { return X_sqmean; }
    
    public boolean affine() { return affine; }
    public LayerNorm affine(boolean flag) { affine = flag; return this; }
     
    public float eps() { return eps; }
    public LayerNorm eps(float eps) { this.eps = eps; return this;}
    
    @Override 
    public void append(String pre, StringBuilder sb) { 
        sb.append(pre).append(default_name());
        sb.append(" { inplace = ").append(inplace());
        sb.append(", [feature_num, param_dim] = [ ")
                .append(features).append(", ")
                .append(Arrays.toString(param_dim)).append(" ]");
        sb.append(", eps = ").append(eps).append(" }");
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="running-area: others">
    @Override
    protected void __init__(Engine eg) {
        eg.delete(A, B);//params are inited to match the lastDim of input
        if(affine) {//perform indentity transform
            A = new Parameter(eg.ones(param_dim)).need_grads(true);
            B = new Parameter(eg.zeros(param_dim)).need_grads(true);
            Parameter.sync(A, B);
        }
    }
    
    @Override
    public void vars(TensorSet set) {
        super.vars(set);
        set.add(X_mean, X_sqmean);
    }
    
    @Override
    public void gc() {
        super.gc();
        eg.delete(X_mean); X_mean = null;
        eg.delete(X_sqmean); X_sqmean = null;
    }
    
    @Override public void params(ParamSet set) { if(affine) super.params(set); }
    @Override public void param_map(ParamMap<String> map) { if(affine) super.param_map(map); }
    @Override public void state(State dic) { if(affine) super.state(dic); }
    @Override public void update_state(State dic, boolean partial) { if(affine) super.update_state(dic, partial); }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: forward-propagation">
    @Override
    protected Tensor __forward__(Tensor X, boolean inplace) {
        Tensor[] stats = eg.row_mean_sqmean(X, features);
        X_mean = stats[0];
        X_sqmean = stats[1];
        X_mean.c(); X_sqmean.c();
        
        return (affine? 
                eg.layerNorm(inplace, X, X_mean, X_sqmean, eps, A.ts(), B.ts()): 
                eg.layerNorm(inplace, X, X_mean, X_sqmean, eps));
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: backwards-propagation">
    protected Tensor __backward_no_affine__(Tensor deltaY, boolean grad_inplace, boolean backward_grads) {
        if(!backward_grads) return null;
        return (isHoldX()?
                eg.layerNorm_deltaX_v2(grad_inplace, deltaY, holdX(), X_mean, X_sqmean, eps): //V2: X is not changed);
                eg.layerNorm_deltaX_v1(grad_inplace, deltaY, holdY(), X_mean, X_sqmean, eps));//V1: Y is not changed);
    }
    
    @Override
    protected Tensor __backward__(Tensor deltaY, boolean grad_inplace, boolean backward_grads) {
        if(!affine) return __backward_no_affine__(deltaY, grad_inplace, backward_grads);//affine = false

        //======[separately finds gradients for {A, B, X} ]=====================
        Tensor deltaA = null, deltaB = null, deltaX = null;
        int gc_count = 0;
        
        if(A.need_grads() && B.need_grads()) {//A.need_grads = B.need_grads = true
            Tensor[] grads = (isHoldY()?
                    eg.layerNorm_deltaAB_v1(deltaY, holdY(), A.ts(), B.ts())://V1: Y is not changed
                    eg.layerNorm_deltaAB_v2(deltaY, holdX(), X_mean, X_sqmean, eps));//V2: X is not changed
            
            A.grads().add(deltaA = grads[0]);
            B.grads().add(deltaB = grads[1]);
            if(grad_inplace) gc_count += 2;
        }
        else if(A.need_grads()) {//B.need_grads = false
            deltaA = (isHoldY()?
                    eg.layerNorm_deltaA_v1(deltaY, holdY(), A.ts(), B.ts())://V1: Y is not changed
                    eg.layerNorm_deltaA_v2(deltaY, holdX(), X_mean, X_sqmean, eps));//V2: X is not changed
            A.grads().add(deltaA);
            if(grad_inplace) gc_count++;
        }
        else if(B.need_grads()) {//A.need_grads = false
            deltaB = eg.field_sum(deltaY, features);
            B.grads().add(deltaB);
            if(grad_inplace) gc_count++;
        }
        
        if(backward_grads) {
            deltaX = (isHoldX()? 
                    eg.layerNorm_deltaX_v2(false, deltaY, holdX(),//V1: Y is not changed
                            X_mean, X_sqmean, eps, A.ts()) :
                    eg.layerNorm_deltaX_v1(false, deltaY, holdY(),//V2: X is not changed
                            X_mean, X_sqmean, eps, A.ts(), B.ts()));
            
            A.ts().follow(deltaX);//When compute deltaX, A can't be changed
            if(isHoldY()) B.ts().follow(deltaX);//When compute deltaX, B can't be changed
            if(grad_inplace) gc_count++;
        }
        
        if(gc_count != 0) {//when deltaA, deltaB, deltaX are cauculated, deltaY is not needed
            CountGc gc = new CountGc(gc_count, deltaY);
            if(deltaA != null) { deltaA.dual(()-> { gc.countDown(); }).remote_sync(); } 
            if(deltaB != null) { deltaB.dual(()-> { gc.countDown(); }).remote_sync(); }
            if(deltaX != null) { deltaX.dual(()-> { gc.countDown(); }); }
        }
        
        return deltaX;
    }
    //</editor-fold>
}
