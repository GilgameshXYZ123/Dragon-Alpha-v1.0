/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.furcation;

import java.util.Arrays;
import z.dragon.engine.Tensor;

/**
 *
 * @author Gilgamesh
 */
public class Split extends FurcationFunction
{
    private int idx;
    private int[] sec;
    
    public Split(int dimIdx, int... section) {
        this.idx = dimIdx;
        this.sec = section;
    }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public int[] section() {return sec;}
    public Split section(int... section) { this.sec = section; return this; }
    
    public int dimIdx() {return idx;}
    public Split dimIdx(int dimIndex) { this.idx = dimIndex; return this; }
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append(" { dimIdx = ").append(idx);
        sb.append(", section = ").append(Arrays.toString(sec)).append(" }");
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area">
    @Override
    protected Tensor[] __forward__(Tensor X) {
        return eg.split(X, idx, sec);
    }

    @Override
    protected Tensor __backward__(Tensor[] deltaY) {
        return eg.concat(idx, deltaY);
    }
    //</editor-fold>
}
