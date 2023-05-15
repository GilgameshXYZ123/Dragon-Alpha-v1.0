/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.simple.affine;

import java.util.Arrays;
import z.dragon.common.state.State;
import z.dragon.common.state.State.StateValue;
import z.dragon.engine.Counter;
import z.dragon.engine.Engine;
import z.dragon.engine.Parameter;
import z.dragon.engine.Parameter.ParamMap;
import z.dragon.engine.Parameter.ParamSet;
import z.dragon.engine.Tensor;
import z.dragon.nn.unit.Train2Eval;
import z.util.math.vector.Vector;

/**
 *
 * @author Gilgamesh
 */
public class GlobalBatchNorm extends Affine implements Train2Eval
{
    protected boolean affine;
    
    protected float beta1;//expoential average of mean(X)
    protected double expBeta1;//to corrrect the exponential average
    protected Parameter run_mean;
    
    protected float beta2, eps;//exponential average of variance(X)
    protected double expBeta2;//to corrrect the exponential average
    protected Parameter run_var;
       
    public GlobalBatchNorm(boolean inplace, boolean affine,
            float beta1, float beta2, float eps,
            int... feature_dim)
    {
        super(inplace, feature_dim);

        this.affine = affine;
        this.beta1 = beta1; expBeta1 = 1;
        this.beta2 = beta2; this.eps = eps; expBeta2 = 1;
    }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public Tensor run_mean() { return run_mean.ts(); }
    public GlobalBatchNorm run_mean(Tensor mean) {
        if(Tensor.isNull(mean)) throw new NullPointerException("run_mean is null");
        if(!mean.dimEquals(param_dim)) throw new IllegalArgumentException(
                name + ": run_mean.dim != param_dim" + Arrays.toString(param_dim));
        
        if(run_mean != null) run_mean.delete();
        run_mean.tensor(mean);
        return this;
    }
    
    public Tensor run_var() { return run_var.ts(); }
    public GlobalBatchNorm run_var(Tensor var) {
        if(Tensor.isNull(var)) throw new NullPointerException("");
        if(!var.dimEquals(param_dim)) throw new IllegalArgumentException(
                name + ": run_var.dim != param_dim" + Arrays.toString(param_dim));
        
        if(run_var != null) run_var.delete();
        run_var.tensor(var);
        return this;
    }
    
    @Override public GlobalBatchNorm weight(Tensor weight) { super.weight(weight); return this; }
    @Override public GlobalBatchNorm bias(Tensor bias)  { super.bias(bias); return this; }
    
    public boolean affine() { return affine; }
    public GlobalBatchNorm affine(boolean flag) { affine = flag; return this; }
    
    public float beta1() { return beta1; }
    public GlobalBatchNorm beta1(float beta1) { this.beta1 = beta1; return this; }

    public float beta2() { return beta2; } 
    public GlobalBatchNorm beta2(float beta2) { this.beta2 = beta2; return this; }
    
    public float eps() {return eps;}
    public GlobalBatchNorm eps(float eps) { this.eps = eps; return this; }
    
    @Override 
    public void append(String pre, StringBuilder sb) { 
        sb.append(pre).append(default_name());
        sb.append(" { inplace = ").append(inplace());
        sb.append(", training = ").append(training);
        sb.append(", affine = ").append(affine);
        sb.append(", [feature_num, param_dim] = [")
                .append(features).append(", ")
                .append(Arrays.toString(param_dim)).append("]");
        sb.append(", [beta1, beta2, eps] = [")
                .append(beta1).append(", ")
                .append(beta2).append(", ")
                .append(eps).append("] }");
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="running-area: others">
    @Override
    protected void __init__(Engine eg)  {
        Parameter.delete(A, B, run_mean, run_var);//params are inited to match the lastDim of input
        run_mean = Parameter.virual(eg.zeros(param_dim));//perform indentity transform
        run_var = Parameter.virual(eg.ones(param_dim));
        if(affine) {//perform indentity transform
           A = new Parameter(eg.ones(param_dim)).need_grads(true);
           B = new Parameter(eg.zeros(param_dim)).need_grads(true);
        }
        Parameter.sync(A, B, run_mean, run_var);
    }
    
    @Override
    public void params(ParamSet set) {
        set.add(A, B, run_mean, run_var);
    }
    
    protected String run_mean_key() { return name + ".run_mean"; }
    protected String run_var_key() { return name + ".run_var"; } 
    protected String params_key() { return name + ".params"; }
    
    @Override
    public void param_map(ParamMap<String> map) {
        if(affine) super.param_map(map);//put [A, B]
        map.put(run_mean_key(), run_mean);
        map.put(run_var_key(), run_var);
    }
    
    @Override
    public void state(State dic) {
        if(affine) super.state(dic);//put[A, B]
        dic.put(run_mean_key(), run_mean.ts());
        dic.put(run_var_key(), run_var.ts());
        dic.put(params_key(), State.floats((float)expBeta1, (float)expBeta2));
    }
    
    @Override
    public void update_state(State dic, boolean partial) {
        if(affine) super.update_state(dic, partial);//update [A, B]
        
        run_mean.ts().set(dic.get(run_mean_key()), partial, name + ": fail to update run_mean");
        run_var.ts().set(dic.get(run_var_key()), partial,  name + ": fail to update run_var");

        StateValue params = dic.get(params_key());
        State.set(params, name + "fail to update for params", partial, ()->{
            float[] arr = Vector.toFloatVector(params.toStringLines(), 2);
            expBeta1 = arr[0]; expBeta2 = arr[1];
        });
    }
    
    protected boolean training = true;
    @Override public boolean training() {return training;}
    @Override public GlobalBatchNorm train() { this.training = true; return this; }
    @Override public GlobalBatchNorm eval() { this.training = false; return this; }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: forward-propagation">
    protected Tensor __forward_compute__(Tensor X, boolean inplace) {
        return (affine?
                eg.batchNorm(inplace, X, run_mean.ts(), run_var.ts(), eps, A.ts(), B.ts()) : 
                eg.batchNorm(inplace, X, run_mean.ts(), run_var.ts(), eps));
    }
    
    @Override
    protected Tensor __forward__(Tensor X, boolean inplace)  {
        if(!training) return __forward_compute__(X, inplace);
        
        //====[Stage1: update run_mean and run_var]=============================
        Tensor[] stats = eg.field_var_mean(X, features);
        Tensor dX_var  = stats[0];
        Tensor dX_mean = stats[1];
        
        //Update: run_mean = a1*run_mean + a2*dX_mean
        //[1] a1 = alpha * (1 - alpha^(t-1)) / (1-alpha^t)
        //[2] a2 = (1 - alpha) / (1-alpha^t)
        double last_correct_beta1 = 1 - expBeta1; 
        expBeta1 *= beta1;
        double corrrect_beta1 = 1 - expBeta1;
        float a1 = (float) (beta1 * last_correct_beta1 / corrrect_beta1);
        float a2 = (float) ((1 - beta1) / corrrect_beta1);
        eg.add(true, a1, run_mean.ts(), a2, dX_mean.c());//inplace: run_mean

        //Update: run_var = b1*run_var + b2*dX_var
        //[1] K = N / (N - 1), N = batch_size, for unbiased  estimation
        //[1] b1 = beta * (1-beta^(t-1)) / (1-beta^t)
        //[2] b2 = K * (1 - beta) / (1 - beta^t)
        int N = X.length() / this.features;
        double K = N / (N - 1.0);
        double last_correct_beta2 = 1 - expBeta2;
        expBeta2 *= beta2;
        double correct_beta2 = 1 - expBeta2;
        float b1 = (float) (beta2 * last_correct_beta2 / correct_beta2);
        float b2 = (float) (K * (1 - beta2) / correct_beta2);
        eg.add(true, b1, run_var.ts(), b2, dX_var.c());//inplace: run_var
        
        //====[Stage2: Global Batch Normalization]==============================
        run_mean.c(); run_var.c();
        Tensor Y = (affine? 
                eg.batchNorm(inplace, X, run_mean.ts(), run_var.ts(), eps, A.ts(), B.ts()):
                eg.batchNorm(inplace, X, run_mean.ts(), run_var.ts(), eps));
        dX_mean.delete(); dX_var.delete();//intermediate variables are not needed
        return Y;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: backwward-propagation">
    protected Tensor __backward_no_affine__(Tensor deltaY, boolean grad_inplace, boolean backward_grads) {
        if(!backward_grads) return null;
        return (isHoldY() ?
                eg.batchNorm_deltaX_v1(grad_inplace, deltaY, holdY(), run_var.ts(), eps): //V1: Y is not changed
                eg.batchNorm_deltaX_v2(grad_inplace, deltaY, holdX(), run_mean.ts(), run_var.ts(), eps));//V2: X is not changed
    }
    
    @Override
    protected Tensor __backward__(Tensor deltaY, boolean grad_inplace, boolean backward_grads) {
        if(!affine) return __backward_no_affine__(deltaY, grad_inplace, backward_grads);//affine = false
     
        //======[integrally finds gradients for {A, B, X} ]=====================
        if(A.need_grads() && B.need_grads() && backward_grads) {
            Tensor[] grads = (isHoldX() ?
                    eg.batchNorm_gradients_v2(grad_inplace, deltaY, holdX(), run_mean.ts(), run_var.ts(), eps, A.ts()): //V2: X is not changed
                    eg.batchNorm_gradients_v1(grad_inplace, deltaY, holdY(), run_var.ts(), eps, A.ts(), B.ts()));//V1: Y is not changed
            
            Tensor deltaX = grads[0];
            A.grads().add(grads[1]);
            B.grads().add(grads[2]);
         
            A.ts().follow(deltaX);//When compute deltaX, A can't be changed
            if(isHoldY()) B.ts().follow(deltaX);//When holdY & compute deltaX, B can't be changed
            return deltaX;
        }
        
        //======[separately finds gradients for {A, B, X} ]=====================
        Tensor deltaA = null, deltaB = null, deltaX = null;
        int gc_count = 0;
        
        if(A.need_grads() && B.need_grads()) {
            Tensor[] grads = (isHoldY()? 
                    eg.batchNorm_deltaAB_v1(deltaY, holdY(), A.ts(), B.ts()) ://V1: Y is not changed
                    eg.batchNorm_deltaAB_v2(deltaY, holdX(), run_mean.ts(), run_var.ts(), eps));//V2: X is not changed
                
            A.grads().add(deltaA = grads[0]);
            B.grads().add(deltaB = grads[1]);
            if(grad_inplace) gc_count += 2;
        }
        else if(A.need_grads()) {//B.need_grads = false
            deltaA = (isHoldY() ?
                    eg.batchNorm_deltaA_v1(deltaY, holdY(), A.ts(), B.ts()) ://V1: Y is not changed
                    eg.batchNorm_deltaA_v2(deltaY, holdX(), run_mean.ts(), run_var.ts(), eps));//V2: X is not changed
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
                    eg.batchNorm_deltaX_v2(false, deltaY, holdX(), run_mean.ts(), run_var.ts(), eps, A.ts()) ://V2: X is not changed
                    eg.batchNorm_deltaX_v1(false, deltaY, holdY(), run_var.ts(), eps, A.ts(), B.ts())); //V1: Y is not changed);
                
            A.ts().follow(deltaX);//When compute deltaX, A can't be changed
            if(isHoldY()) B.ts().follow(deltaX);//When holdY & compute deltaX, B can't be changed
            if(grad_inplace) gc_count++;
        }
        
        if(gc_count != 0) {//when deltaA, deltaB, deltaX are cauculated, deltaY is not needed
            Counter.CountGc gc = new Counter.CountGc(gc_count, deltaY);
            if(deltaA != null) { deltaA.dual(()-> { gc.countDown(); }).remote_sync(); }
            if(deltaB != null) { deltaB.dual(()-> { gc.countDown(); }).remote_sync(); }
            if(deltaX != null) { deltaX.dual(()-> { gc.countDown(); }); }
        }
        
        return deltaX;
    }
    //</editor-fold>
}
