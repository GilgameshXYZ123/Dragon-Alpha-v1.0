/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.dual.math2;

import z.dragon.engine.Counter.CountGc;
import z.dragon.engine.Tensor;
import z.util.lang.annotation.Passed;

/**
 *
 * @author Gilgamesh
 */
@Passed("CudaFloat32Base")
public class BatchMatMul extends DualLikeFunction<BatchMatMul>
{
    public BatchMatMul(boolean likeX1) {
        super(likeX1);
    }

    //<editor-fold defaultstate="collapsed" desc="running-area">
    @Override
    protected Tensor __forward__(Tensor X1, Tensor X2, boolean likeX1) {
        return eg.batchMatMul(likeX1, X1, X2);
    }

    @Override
    protected Tensor[] __backward__(Tensor deltaY, boolean grad_inplace, boolean backward_grads) {
        if(!backward_grads) return null;

        //deltaX1[batch, N, K] = deltaY[batch, N, M] * X2^T[batch, M, K]
        Tensor deltaX1 = eg.batchMatMulT2(deltaY, holdX2());
        
        //deltaX2[batch, K, M] = X1^T[batch, K, N] * deltaY[batch, N, M]
        Tensor deltaX2 = eg.batchMatMulT1(holdX1(), deltaY);
        
        if(grad_inplace) {//when deltaX1 and deltaX2 are cauculated, the deltaY is not needed
            CountGc gc = new CountGc(2, deltaY);
            deltaX1.dual(()-> { gc.countDown(); });
            deltaX2.dual(()-> { gc.countDown(); });
        }
        
        return new Tensor[]{ deltaX1, deltaX2 };
    }
    //</editor-fold>
}
