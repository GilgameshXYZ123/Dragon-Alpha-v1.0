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
public class LinearMean extends LinearSummary
{
    protected float mean_alpha;
    protected float mean_beta;
    
    public LinearMean(float alpha, float beta) {
        super(alpha, beta);
    }
    
    //<editor-fold defaultstate="collapsed" desc="running-area">
    @Override
    protected Tensor __forward__(Tensor[] X) {
        mean_alpha = (alpha / X.length);
        mean_beta  = (beta  / X.length);
        return eg.linear_summary(false, mean_alpha, mean_beta, X);
    }

    @Override
    protected Tensor[] __backward__(Tensor deltaY, int input_tensor_num) {
        Tensor[] deltaX = new Tensor[input_tensor_num];
        
        int index = 0;//inplace = false, multi operators can't disturb each other
        int num2 = (input_tensor_num >> 2) << 2;
        while(index < num2) {//compute 2 grads integrately
            Tensor[] grads = eg.linear_2out(false, deltaY, mean_alpha, 0, mean_alpha, 0);
            deltaX[index++] = grads[0];
            deltaX[index++] = grads[1];
        }
        
        while(index < input_tensor_num) {//remainder = 0 or 1
            deltaX[index++] = eg.linear(false, mean_alpha, deltaY, 0);
        }
        
        return deltaX;
    }
    
    @Override
    protected Tensor[] __backward__(Tensor deltaY, int input_tensor_num,
            boolean grad_inplace, boolean backward_grads)
    {
        if(!backward_grads) return null;
        if(input_tensor_num == 1) return new Tensor[] { eg.linear(grad_inplace, mean_alpha, deltaY, 0) };
        if(input_tensor_num == 2) return eg.linear_2out(grad_inplace, deltaY, mean_alpha, 0, mean_alpha, 0);
        
        Tensor[] deltaX = __backward__(deltaY, input_tensor_num);

        if(grad_inplace) {//when deltaX[] are found, deltaY is not needed
            Counter.CountGc gc = new Counter.CountGc(input_tensor_num, deltaY);
            for (Tensor grad : deltaX) grad.dual(()-> { gc.countDown(); });
        }
        
        return deltaX;
    }
    //</editor-fold>
}
