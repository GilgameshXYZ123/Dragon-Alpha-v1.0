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
public interface BackwardHookable
{
    public static interface Hook {
        public void callback(Unit self);
    }
    
    public <T extends Unit> T hook_before_backward(Hook hook);
    public <T extends Unit> T hook_after_backward(Hook hook);
    
    default <T extends Unit> T remove_hook_before_backward() {
        this.hook_before_backward(null);
        return (T) this;
    }
    
     default <T extends Unit> T remove_hook_after_backward() {
        this.hook_after_backward(null);
        return (T) this;
    }
}
