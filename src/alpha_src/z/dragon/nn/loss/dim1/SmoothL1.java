/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.loss.dim1;

import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.util.lang.annotation.Passed;

/**
 *
 * @author Gilgamesh
 */
@Passed("CudaFloat32Base")
public class SmoothL1 extends Loss1D
{
    //<editor-fold defaultstate="collapsed" desc="functions">
    @Override
    protected Tensor __loss_tensor__(Tensor Yh, Tensor Y, Engine eg) {
        return eg.smoothL1(Yh, Y);
    }
    
    @Override
    protected Tensor __gradient__(Tensor Yh, Tensor Y, Engine eg) {
        return eg.smoothL1_deltaYh(Yh, Y);
    }
    //</editor-fold>
}
