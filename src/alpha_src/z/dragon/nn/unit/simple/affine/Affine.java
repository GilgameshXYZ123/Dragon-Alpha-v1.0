/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.simple.affine;

import java.util.Arrays;
import z.dragon.common.state.State;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.Counter.CountGc;
import z.dragon.engine.Parameter;
import z.dragon.engine.Parameter.ParamMap;
import z.dragon.engine.Parameter.ParamSet;
import z.dragon.nn.unit.simple.math2.SimpleInplaceFunction;
import z.util.lang.annotation.Passed;
import z.util.math.vector.Vector;

/**
 * <pre>
 * read X from the last layer.
 * alloc: A, B
 * compute: Y(the next layer will read)
 * 
 * read deltaY from the next layer:
 * alloc: deltaW, deltaX(is need)
 * compute: deltaX(the last layer will read)
 * 
 * forward: Y = X(*)A + B #element multiply
 * back: deltaX = deltaY*W^T
 * </pre>
 * @author Gilgamesh
 */
@Passed("CudaFloat32Base")
public class Affine extends SimpleInplaceFunction<Affine>
{
    protected int[] param_dim;//[...., mem_width = input.lastDim]
    protected int features;
    
    protected Parameter A;
    protected Parameter B;
    
    public Affine(boolean inplace, int... feature_dim) {
        super(inplace);

        if(feature_dim == null || feature_dim.length == 0)
            throw new NullPointerException("feature_dim is null");
        
        if(feature_dim.length == 1) {//input feature is 2D
            this.features = feature_dim[0];
            this.param_dim = new int[]{ feature_dim[0] };
        }
        else {//input feature >= 3D. to save memory: the param is 2D[mem_height, mem_width]
            this.features = Vector.mul(feature_dim);
            int lastDim = feature_dim[feature_dim.length - 1];
            this.param_dim = new int[]{ features / lastDim, lastDim };
        }
    }
     
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public int[] param_dim() { return param_dim; }
    public int features() { return features; }
    
    public Tensor weight() { return A.ts(); }
    public Affine weight(Tensor weight) { 
        if(Tensor.isNull(weight)) throw new NullPointerException("weight is null");
        if(!weight.dimEquals(param_dim)) throw new IllegalArgumentException(
                name + ": weight.dim != param_dim" + Arrays.toString(param_dim));
        
        if(A != null) A.delete();
        A.tensor(weight);
        return this;
    }
    
    public Tensor bias() { return B.ts(); }
    public Affine bias(Tensor bias) {
        if(Tensor.isNull(bias)) throw new NullPointerException("bias is null");
        if(!bias.dimEquals(param_dim)) throw new IllegalArgumentException(
                name + ": bias.dim != param_dim: " + Arrays.toString(param_dim));
        
        if(B != null) B.delete();
        B.tensor(bias);
        return this;
    }
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append(" { inplace = ").append(inplace());
        sb.append(", [feature_num, param_dim] = [")
                .append(features).append(", ")
                .append(Arrays.toString(param_dim)).append("] }");
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: others"> 
    @Override
    protected void __init__(Engine eg) {
        Parameter.delete(A, B);//params are inited to match the lastDim of input
        A = new Parameter(eg.ones(param_dim)).need_grads(true);//perform indentity transform
        B = new Parameter(eg.zeros(param_dim)).need_grads(true);
        Parameter.sync(A, B);
    }
    
    @Override 
    public void params(ParamSet set) { 
        set.add(A, B);
    }
    
    protected String weight_key() { return name + ".weight"; }
    protected String bias_key() { return name + ".bias"; }
    
    @Override
    public void param_map(ParamMap<String> map) {
        map.put(weight_key(), A);
        map.put(bias_key(), B);
    }
    
    @Override 
    public void state(State dic) {
        dic.put(weight_key(), A.ts());
        dic.put(bias_key(), B.ts());
    }
    
    @Override
    public void update_state(State dic, boolean partial)  {
        A.ts().set(dic.get(weight_key()), partial, name + ": fail to update state for weight");
        B.ts().set(dic.get(bias_key()), partial, name + ": fail to update state for bias");
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: propagation"> 
    @Override
    protected Tensor __forward__(Tensor X, boolean inplace) {
        return eg.affine(inplace, X, A.ts(), B.ts());
    }

    @Override
    protected Tensor __backward__(Tensor deltaY, boolean grad_inplace, boolean backward_grads) {
        Tensor deltaA = null, deltaB = null, deltaX = null;
        int gc_count = 0;
        
        if(A.need_grads() && B.need_grads()) {//A.need_grads = B.need_grads = true
            Tensor[] delta = (isHoldY()? 
                    eg.affine_deltaAB_v1(deltaY, holdY(), A.ts(), B.ts())://V1: Y is not changed
                    eg.affine_deltaAB_v2(deltaY, holdX(), features));//V2: X is not changed
            
            deltaA = delta[0]; A.grads().add(deltaA);
            deltaB = delta[1]; B.grads().add(deltaB);
            if(grad_inplace) gc_count += 2;
        }
        else if(A.need_grads()) {//B.need_grads = false
            deltaA = (isHoldY()?
                    eg.affine_deltaA_v1(deltaY, holdY(), A.ts(), B.ts())://V1: Y is not changed
                    eg.affine_deltaA_v2(deltaY, holdX(), features));//V2: X is not changed
            A.grads().add(deltaA);
            if(grad_inplace) gc_count++;
        }
        else if(B.need_grads()) {//A.need_grads = false
            deltaB = eg.field_sum(deltaY, features);
            B.grads().add(deltaB);
            if(grad_inplace) gc_count++;
        }
        
        if(backward_grads) {
            deltaX = eg.mul_row(false, deltaY, A.ts());//false: deltaY is read only
            A.ts().follow(deltaX);//When compute deltaX, A can't be changed
            if(grad_inplace) gc_count++;
        }
        
        //the final gc process--------------------------------------------------
        if(gc_count != 0) {//when deltaA, deltaB, deltaX are cauculated, deltaY is not needed
            CountGc gc = new CountGc(gc_count, deltaY);
            if(deltaA != null) { deltaA.dual(()-> { gc.countDown(); }).remote_sync(); }
            if(deltaB != null) { deltaB.dual(()-> { gc.countDown(); }).remote_sync(); }
            if(deltaX != null) { deltaX.dual(()-> { gc.countDown(); }); }
        }
        
        return deltaX;
    }
    //</editor-fold>
}
