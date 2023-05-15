/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.furcation;

import z.dragon.engine.Tensor;

/**
 *
 * @author Gilgamesh
 */
public class Chunk extends FurcationFunction
{
    private int dimIdx;
    private int num;
    
    public Chunk(int dimIdx, int num) {
        this.dimIdx = dimIdx;
        this.num = num;
    }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public int num() {return num;}
    public Chunk num(int num) { this.num = num; return this; }
    
    public int dimIdx() {return dimIdx;}
    public Chunk dimIdx(int dimIdx) { this.dimIdx = dimIdx; return this; }
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append(" { dimIdx = ").append(dimIdx);
        sb.append(", num = ").append(num).append(" }");
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area">  
    @Override
    protected Tensor[] __forward__(Tensor X) {
        return eg.chunk(X, dimIdx, num);
    }

    @Override
    protected Tensor __backward__(Tensor[] deltaY) {
        return eg.concat(dimIdx, deltaY);
    }
    //</editor-fold>
}
