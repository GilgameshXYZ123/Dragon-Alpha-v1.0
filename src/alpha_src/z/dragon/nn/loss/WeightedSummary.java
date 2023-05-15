/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.loss;

import z.dragon.engine.Engine;
import z.dragon.engine.Result;
import z.dragon.engine.Syncer;
import z.dragon.engine.Tensor;
import z.util.math.vector.Vector;

/**
 *
 * @author Gilgamesh
 */
@SuppressWarnings("unchecked")
public class WeightedSummary extends LossFunction
{
    protected float[] weights;
    protected LossFunction[] loss_funcs;
    
    public WeightedSummary(float[] weights, LossFunction[] loss_funcs) {
        if(loss_funcs.length != weights.length) 
            throw new IllegalArgumentException("loss_functions.length != weights.length");
        Vector.requireNonNull(loss_funcs, "loss_functions");
        
        this.weights = weights;
        this.loss_funcs = loss_funcs;
    }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public float[] weights() { return weights; }
    public LossFunction[] loss_funcs() { return loss_funcs; }
    
    @Override
    public WeightedSummary average(boolean flag) { 
        super.average(flag);
        for(LossFunction lf : loss_funcs) lf.average(flag);
        return this; 
    }
    
    @Override
    public WeightedSummary zero_nan(boolean flag) { 
        super.zero_nan(flag);
        for(LossFunction lf : loss_funcs) lf.average(flag);
        return this;
    }
    
    @Override
    public void append(StringBuilder sb) 
    {
        super.append(sb); sb.deleteCharAt(sb.length() - 1);
        for(int i=0; i<weights.length; i++) {
            sb.append("\n[weight, loss_function] = ");
            sb.append(weights[i]).append(", ").append(loss_funcs[i]);
        }
        sb.append("\n}");
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="forward-prop">
    @Override protected Tensor __loss_tensor__(Tensor Yh, Tensor Y, Engine eg) { return null; }
    @Override
    public Tensor loss_tensor(Tensor Yh, Tensor Y) 
    {
        Engine eg = Yh.engine();
        Tensor[] loss = new Tensor[loss_funcs.length];
        for(int i=0; i<loss.length; i++) loss[i] = loss_funcs[i].__loss_tensor__(Yh, Y, eg);
        for(int i=0; i<loss.length; i++) loss[i] = eg.smul(true, loss[i].c(), weights[i]);
        for(Tensor ls : loss) ls.c();
        
        Tensor sum = eg.summary(true, loss);
        Syncer sc = sum.syncer(); sum.setSyncer(Syncer.dual(sc, ()->{
            for(int i=1; i<loss.length; i++) loss[i].delete();
        }));
        return sum;
    }
   
    //<editor-fold defaultstate="collapsed" desc="class: LossWeightedSum">
    public static final class LossWeightedSum extends Result<Float> 
    {
        private final float[] weights;
        private final Result[] loss;
        
        LossWeightedSum(float[] weights, Result[] loss) {
            this.weights = weights;
            this.loss = loss;
        }
        
        @Override
        protected Float waitResult() {
            float sum = 0.0f;
            for(int i=0; i<loss.length; i++) {
                Result<Float> ls = loss[i];
                sum += ls.get() * weights[i];
            }
            return sum;
        }
    }
    //</editor-fold>
    
    @Override protected Result<Float> mean_loss(Tensor loss, Engine eg) { return null; }
    @Override
    public Result<Float> loss(Tensor Yh, Tensor Y)  {
        Result[] loss = new Result[loss_funcs.length];
        for(int i=0; i<loss.length; i++) loss[i] = loss_funcs[i].loss(Yh, Y);
        return new LossWeightedSum(weights, loss);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="backward-prop">
    @Override protected Tensor __gradient__(Tensor Yh, Tensor Y, Engine eg) { return null; }
    @Override protected Tensor mean_gradient(Tensor grad, Engine eg) { return null; }
    @Override
    public Tensor gradient(Tensor Yh, Tensor Y) 
    {
        Engine eg = Yh.engine();
        Tensor[] grads = new Tensor[loss_funcs.length];
        for(int i=0; i<grads.length; i++) grads[i] = loss_funcs[i].__gradient__(Yh, Y, eg);
        for(int i=0; i<grads.length; i++) grads[i] = eg.smul(true, grads[i].c(), weights[i]);
        for(Tensor grad : grads) grad.c();
        
        Tensor sum = eg.summary(true, grads);
        Syncer sc = sum.syncer(); sum.setSyncer(Syncer.dual(sc, ()->{
             for(int i=1; i<grads.length; i++) eg.delete(grads[i]);
        }));
        return sum;
    }
    //</editor-fold>
}
