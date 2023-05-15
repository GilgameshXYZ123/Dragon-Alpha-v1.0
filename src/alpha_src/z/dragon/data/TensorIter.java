/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.data;

import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;

/**
 * @author Gilgamesh
 */
public interface TensorIter 
{
    public static class TensorPair extends Pair<Tensor, Tensor> {
        public TensorPair(Tensor feature, Tensor label) {
            super(feature, label);
        }
    }
    
    default void reset() { reset(false); }
    public void reset(boolean shuffle);
    
    public boolean hasNext();
   
    default TensorPair next(Engine eg) { return next(eg, 1); }
    public TensorPair next(Engine eg, int batch);
}
