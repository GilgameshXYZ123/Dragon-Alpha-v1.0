/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.reducer;

import z.dragon.engine.Tensor;

/**
 *
 * @author Gilgamesh
 */
public class QuadraticMean extends QuadraticSummary
{
    protected float mean_alpha;
    protected float mean_beta;
    protected float mean_gamma;

    public QuadraticMean(float alpha, float beta, float gamma) {
        super(alpha, beta, gamma);
    }
    
    //<editor-fold defaultstate="collapsed" desc="functions">
    @Override
    protected Tensor __forward__(Tensor[] X) {
        mean_alpha = (alpha / X.length);
        mean_beta  = (beta  / X.length);
        mean_gamma = (gamma / X.length);
        return eg.quadratic_summary(false, mean_alpha, mean_beta, mean_gamma, X);
    }
    
    @Override //deltaX = 2*mean_alpha * (deltaY*X) + mean_beta * deltaY
    protected Tensor[] __backward__(Tensor deltaY, int input_tensor_num) {
        Tensor[] deltaX = new Tensor[input_tensor_num];
        
        float mean_alpha2 = 2.0f * mean_alpha;
        for(int i=0; i<deltaX.length; i++) {
            deltaX[i] = eg.quadratic2(false, deltaY, holdX(i), 
                    0.0f, mean_alpha2, 0.0f, 
                    mean_beta, 0.0f, 0.0f);
        }

        return deltaX;
    }
    //</editor-fold>
}
