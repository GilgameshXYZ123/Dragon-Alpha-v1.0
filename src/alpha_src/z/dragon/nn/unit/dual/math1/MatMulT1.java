/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.dual.math1;

import z.dragon.engine.Counter.CountGc;
import z.dragon.engine.Tensor;
import z.util.lang.annotation.Passed;

/**
 *
 * @author Gilgamesh
 */
@Passed("CudaFloat32Base")
public class MatMulT1 extends DualFunction
{
    //<editor-fold defaultstate="collapsed" desc="running-area">
    @Override
    protected Tensor __forward__(Tensor X1, Tensor X2) {
        return eg.matMulT1(X1, X2);
    }

    @Override
    protected Tensor[] __backward__(Tensor deltaY, boolean grad_inplace, boolean backward_grads) {
        if(!backward_grads) return null;

        Tensor deltaX1 = eg.matMulT2(holdX2(), deltaY); //deltaX1[K, N] = X2[K, M] * deltaY^T[M, N]
        Tensor deltaX2 = eg.matMul(holdX1(), deltaY); //deltaX2[K, M] = X1[K, N] * deltaY[N, M]
        
        if(grad_inplace) {//when deltaX1 and deltaX2 are cauculated, the deltaY is not needed
            CountGc gc = new CountGc(2, deltaY);
            deltaX1.dual(()-> { gc.countDown(); });
            deltaX2.dual(()-> { gc.countDown(); });
        }
        
        return new Tensor[]{ deltaX1, deltaX2 };
    }
     //</editor-fold>
}
