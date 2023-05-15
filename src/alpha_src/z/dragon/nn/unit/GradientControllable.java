/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit;

/**
 *
 * @author Gilgamesh
 */
public interface GradientControllable
{
    public abstract boolean backward_grads();
    public abstract Unit backward_grads(boolean flag);
}
