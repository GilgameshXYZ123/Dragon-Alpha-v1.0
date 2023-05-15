/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.simple.affine;

import java.util.Arrays;
import z.dragon.engine.Counter;
import z.dragon.engine.Tensor;
import z.util.lang.annotation.Passed;

/**
 * Global Batch Normalization.
 * @author Gilgamesh
 */
@Passed("CudaFloat32Base")
public class BatchNorm extends GlobalBatchNorm
{
    protected boolean track_stats = true;
    protected Tensor dX_mean;
    protected Tensor dX_var;
    
    public BatchNorm(boolean inplace, boolean affine,
            float beta1, float beta2, float eps, int... feature_dim) 
    {
        super(inplace, affine,
                beta1, beta2, eps,
                feature_dim);
    }

    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public Tensor mean() { return dX_mean; }
    public Tensor var() { return dX_var; }
    
    public boolean track_stats() { return track_stats; }
    public BatchNorm track_stats(boolean flag) { track_stats = flag; return this; }
    
    @Override public BatchNorm weight(Tensor weight) { super.weight(weight); return this; }
    @Override public BatchNorm bias(Tensor bias) { super.bias(bias); return this; }
    
    @Override public BatchNorm run_mean(Tensor mean) { super.run_mean(mean); return this; }
    @Override public BatchNorm run_var(Tensor var) { super.run_var(var); return this; }
    
    @Override public BatchNorm affine(boolean flag) { affine = flag; return this; }
    @Override public BatchNorm beta1(float beta1) { this.beta1 = beta1; return this; }
    @Override public BatchNorm beta2(float beta2) { this.beta2 = beta2; return this; }
    @Override public BatchNorm eps(float eps) { this.eps = eps; return this; }
    
    @Override public BatchNorm train() { this.training = true; return this; }
    @Override public BatchNorm eval() { this.training = false; return this; }
    
    @Override 
    public void append(String pre, StringBuilder sb) { 
        sb.append(pre).append(default_name());
        sb.append(" { inplace = ").append(inplace());
        sb.append(", training = ").append(training);
        sb.append(", affine = ").append(affine);
        sb.append(", track_stats = ").append(track_stats);
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
    public void vars(Tensor.TensorSet set) {
        super.vars(set);
        set.add(dX_mean, dX_var);
    }
    
    @Override
    public void gc() {
        super.gc();
        eg.delete(dX_mean); dX_mean = null;
        eg.delete(dX_var); dX_var = null; 
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: forward-propagation">
    @Override
    protected Tensor __forward__(Tensor X, boolean inplace)  {
        if(!training) return __forward_compute__(X, inplace);

        //====[Stage1: update run_mean and run_var]=============================
        Tensor[] stats = eg.field_var_mean(X, features);
        dX_var  = stats[0]; 
        dX_mean = stats[1];
        
        if(track_stats) {
            //Update: run_mean = a1*run_mean + a2*dX_mean
            //[1] a1 = alpha * (1 - alpha^(t-1)) / (1 - alpha^t)
            //[2] a2 = (1 - alpha) / (1 - alpha^t)
            double last_correct_beta1 = 1 - expBeta1; 
            expBeta1 *= beta1;
            double corrrect_beta1 = 1 - expBeta1;
            float a1 = (float) (beta1 * last_correct_beta1 / corrrect_beta1);
            float a2 = (float) ((1.0 - beta1) / corrrect_beta1);
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
            float b2 = (float) (K * (1.0 - beta2) / correct_beta2);
            eg.add(true, b1, run_var.ts(), b2, dX_var.c());//inplace: run_var
        }
        else { dX_mean.c(); dX_var.c(); }
        
        //====Stage2: Batch Normalization=======================================
        Tensor Y = (affine?
                eg.batchNorm(inplace, X, dX_mean, dX_var, eps, A.ts(), B.ts()):
                eg.batchNorm(inplace, X, dX_mean, dX_var, eps));
        if(track_stats) Y.dual(()-> { run_mean.c(); run_var.c(); });
        return Y;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: backward-propagation">
    @Override
    protected Tensor __backward_no_affine__(Tensor deltaY, boolean grad_inplace, boolean backward_grads) {
        if(!backward_grads) return null;
        return (isHoldY() ?
                eg.batchNorm_deltaX_v1(grad_inplace, deltaY, holdY(), dX_var, eps)://V1: Y is not changed;
                eg.batchNorm_deltaX_v2(grad_inplace, deltaY, holdX(), dX_mean, dX_var, eps));//V2: X is not changed;
    }
    
    @Override
    protected Tensor __backward__(Tensor deltaY, boolean grad_inplace, boolean backward_grads) {
        if(!affine) return __backward_no_affine__(deltaY, grad_inplace, backward_grads);//affine = false
        
        //======[integrally finds gradients for {A, B, X} ]=====================
        if(A.need_grads() && B.need_grads() && backward_grads) {
            Tensor[] grads = (isHoldX() ? //the complextity of V2 is lower
                    eg.batchNorm_gradients_v2(grad_inplace, deltaY, holdX(), dX_mean, dX_var, eps, A.ts()) ://V2: X is not changed
                    eg.batchNorm_gradients_v1(grad_inplace, deltaY, holdY(), dX_var, eps, A.ts(), B.ts()));//V1: Y is not changed 
            
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
            Tensor[] grads = (isHoldY() ? 
                    eg.batchNorm_deltaAB_v1(deltaY, holdY(), A.ts(), B.ts()) ://V1: Y is not changed
                    eg.batchNorm_deltaAB_v2(deltaY, holdX(), dX_mean, dX_var, eps));//V2: X is not changed
                
            A.grads().add(deltaA = grads[0]);
            B.grads().add(deltaB = grads[1]);
            if(grad_inplace) gc_count += 2;
        }
        else if(A.need_grads()) {//B.need_grads = false
            deltaA = (isHoldY()? 
                    eg.batchNorm_deltaA_v1(deltaY, holdY(), A.ts(), B.ts()) ://V1: Y is not changeds
                    eg.batchNorm_deltaA_v2(deltaY, holdX(), dX_mean, dX_var, eps));//V2: X is not changed
            A.grads().add(deltaA);
            if(grad_inplace) gc_count++;
        }
        else if(B.need_grads()) {//A.need_grads = false
            deltaB = eg.field_sum(deltaY, features);
            B.grads().add(deltaB);
            if(grad_inplace) gc_count++;
        }
        
        if(backward_grads) { 
            deltaX = (isHoldX() ?
                    eg.batchNorm_deltaX_v2(false, deltaY, holdX(), dX_mean, dX_var, eps, A.ts()) ://V2: X is not changed
                    eg.batchNorm_deltaX_v1(false, deltaY, holdY(), dX_var, eps, A.ts(), B.ts()));//V1: Y is not changed
            
            A.ts().follow(deltaX);//When compute deltaX, A can't be changed
            if(isHoldY()) B.ts().follow(deltaX);//When holdY & compute deltaX, B can't be changed
            if(grad_inplace) gc_count++;
        }
        
        if(gc_count != 0) {
            Counter.CountGc gc = new Counter.CountGc(gc_count, deltaY);
            if(deltaA != null) { deltaA.dual(()-> { gc.countDown(); }).remote_sync(); }
            if(deltaB != null) { deltaB.dual(()-> { gc.countDown(); }).remote_sync(); }
            if(deltaX != null) { deltaX.dual(()-> { gc.countDown(); }); }
        }
        
        return deltaX;
    }
    //</editor-fold>
}
