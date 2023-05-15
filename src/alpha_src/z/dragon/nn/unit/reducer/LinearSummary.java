/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.reducer;

import z.dragon.engine.Counter;
import z.dragon.engine.Tensor;

/**
 *
 * @author Gilgamesh
 */
public class LinearSummary extends ReduceFunction
{
    protected float alpha;
    protected float beta;
    
    public LinearSummary(float alpha, float beta) {
        this.alpha = alpha;
        this.beta = beta;
    }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public float alpha() {return alpha;}
    public LinearSummary alpha(float alpha) {this.alpha = alpha; return this;}
    
    public float beta() {return beta;}
    public LinearSummary beta(float beta) {this.beta = beta; return this;}
    
    @Override
    public void append(String pre, StringBuilder sb) { 
        sb.append(pre).append(default_name());
        sb.append("{ alpha = ").append(alpha);
        sb.append(", beta = ").append(beta).append(" }");
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area">
    @Override
    protected Tensor __forward__(Tensor[] X) {
        return eg.linear_summary(false, alpha, beta, X);
    }

    @Override
    protected Tensor[] __backward__(Tensor deltaY, int input_tensor_num) {
        Tensor[] deltaX = new Tensor[input_tensor_num]; 
        
        int index = 0;//inplace = false, multi operators can't disturb each other
        int num2 = (input_tensor_num >> 2) << 2;
        while(index < num2) {//compute 2 grads integrately
            Tensor[] grads = eg.linear_2out(false, deltaY, alpha, 0, alpha, 0);
            deltaX[index++] = grads[0];
            deltaX[index++] = grads[1];
        }
        
        while(index < input_tensor_num) {//remainder = 0 or 1
            deltaX[index++] = eg.linear(false, alpha, deltaY, 0);
        }
        
        return deltaX;
    }
    
    @Override
    protected Tensor[] __backward__(Tensor deltaY, int input_tensor_num,
            boolean grad_inplace, boolean backward_grads)
    {
        if(!backward_grads) return null;
        if(input_tensor_num == 1) return new Tensor[] { eg.linear(grad_inplace, alpha, deltaY, 0) };
        if(input_tensor_num == 2) return eg.linear_2out(grad_inplace, deltaY, alpha, 0, alpha, 0);
        
        Tensor[] deltaX = __backward__(deltaY, input_tensor_num);

        if(grad_inplace) {//when deltaX[] are found, deltaY is not needed
            Counter.CountGc gc = new Counter.CountGc(input_tensor_num, deltaY);
            for (Tensor grad : deltaX) grad.dual(()-> { gc.countDown(); });
        }
        
        return deltaX;
    }
    //</editor-fold>
}
