/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.data;

import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;

/**
 *
 * @author Gilgamesh
 * @param <T>
 */
public interface Transform<T> 
{
    public Tensor transform(Engine eg, T value);
}
