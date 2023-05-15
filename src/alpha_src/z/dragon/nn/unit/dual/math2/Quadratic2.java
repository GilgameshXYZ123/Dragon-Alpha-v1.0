/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.dual.math2;

import z.dragon.engine.Tensor;
import z.dragon.engine.Counter.CountGc;
import z.util.lang.annotation.Passed;

/**
 *
 * @author Gilgamesh
 */
@Passed("CudaFloat32Base")
public class Quadratic2 extends DualLikeFunction<Quadratic2>
{
    protected float k11, k12, k22, k1, k2, C;
    
    public Quadratic2(boolean likeX1, 
            float k11, float k12, float k22,
            float k1, float k2,
            float C) 
    {
        super(likeX1);
        this.k11 = k11; this.k12 = k12; this.k22 = k22;
        this.k1  = k1;  this.k2  = k2; 
        this.C = C;
    }

    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public float k11() { return k11; }
    public Quadratic2 k11(float k11) { this.k11 = k11; return this; }
    
    public float k12() { return k12; }
    public Quadratic2 k12(float k12) { this.k12 = k12; return this;}
    
    public float k22() { return k22; }
    public Quadratic2 k22(float k22) { this.k22 = k22; return this;}
    
    public float k1() { return k1; }
    public Quadratic2 k1(float k1) { this.k1 = k1; return this;}
    
    public float k2() { return k2; }
    public Quadratic2 k2(float k2) { this.k2 = k2; return this; }
    
    public float C() { return C; }
    public Quadratic2 C(float C) { this.C = C; return this;}

    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append(" { likeX1 = ").append(likeX1());
        sb.append(", [k11, k12, k22] = [")
                .append(k11).append(", ")
                .append(k12).append(", ")
                .append(k22).append(']');
        sb.append(", [k1, k2, C] = ")
                .append(k1).append(", ")
                .append(k2).append(", ")
                .append(C).append("] }");
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area">
    @Override
    protected Tensor __forward__(Tensor X1, Tensor X2, boolean likeX1) {
        return eg.quadratic2(false, likeX1, X1, X2, // inplace = false
                k11, k12, k22,
                k1, k2, C);
    }

    @Override
    protected Tensor[] __backward__(Tensor deltaY, boolean grad_inplace, boolean backward_grads) {
        if(!backward_grads) return null;
        
        Tensor[] deltaX = eg.quadratic2_deltaX(false, deltaY,
                holdX1(), holdX2(),
                k11, k12, k22, 
                k1, k2);
        
        if(grad_inplace) {//when deltaX1 and deltaX2 are cauculated, the deltaY is not needed
            Tensor deltaX1 = deltaX[0], deltaX2 = deltaX[1];
            CountGc gc = new CountGc(2, deltaY);
            deltaX1.dual(()-> { gc.countDown(); });
            deltaX2.dual(()-> { gc.countDown(); });
        }
        
        return deltaX;
    }   
    //</editor-fold>
}
