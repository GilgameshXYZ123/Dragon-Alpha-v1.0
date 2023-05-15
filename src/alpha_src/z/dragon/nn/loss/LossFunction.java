/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.loss;

import z.dragon.engine.Engine;
import z.dragon.engine.Result;
import z.dragon.engine.Tensor;

/**
 * @author Gilgamesh
 */
@SuppressWarnings("unchecked")
public abstract class LossFunction
{
    private boolean average = true;
    private boolean zero_nan = false;//zero all NAN in loss
    
    //<editor-fold defaultstate="collapsed" desc="Basic-functions">
    public boolean average() { return average; }  
    public LossFunction average(boolean flag) { average = flag; return this; }
    
    public boolean zero_nan() { return zero_nan; }
    public LossFunction zero_nan(boolean flag) { zero_nan = flag; return this; }

    public void append(StringBuilder sb) {
        sb.append(getClass().getSimpleName());
        sb.append("{ average = ").append(average);
        sb.append(", zero_nan = ").append(zero_nan);
        sb.append(" }");
    }
    
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder(64);
        this.append(sb);
        return sb.toString();
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="forward-propagation">
    protected abstract Tensor __loss_tensor__(Tensor Yh, Tensor Y, Engine eg);
    protected abstract Result<Float> mean_loss(Tensor loss, Engine eg);
    protected Result<Float> sum_loss(Tensor loss, Engine eg) {
        return eg.straight_sum(loss);
    }
    
    public Tensor loss_tensor(Tensor Yh, Tensor Y) {//return L
        Tensor L = __loss_tensor__(Yh.c(), Y.c(), Yh.engine());
        L.carry(Y);
        return L;
    }
    
    public Result<Float> loss(Tensor Yh, Tensor Y) {
        Engine eg = Yh.engine();
        Tensor loss = __loss_tensor__(Yh.c(), Y.c(), eg).c();
        
        if(zero_nan) loss.zero_nan().c();
        Result<Float> ls = (average? 
                mean_loss(loss, eg):
                sum_loss(loss, eg));
        
        return Result.dual(ls, ()->{ loss.delete(); });
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="backward-propagation">
    protected abstract Tensor __gradient__(Tensor Yh, Tensor Y, Engine eg);
    protected abstract Tensor mean_gradient(Tensor grad, Engine eg);
    
    public Tensor gradient(Tensor Yh, Tensor Y) {
        Engine eg = Yh.engine();
        Tensor grad = __gradient__(Yh.c(), Y.c(), eg);
        if(zero_nan) grad.c().zero_nan().c();
        if(average) grad = mean_gradient(grad.c(), eg);
        grad.carry(Y.need_carry(true)); 
        return grad;
    }
    //</editor-fold>
}
