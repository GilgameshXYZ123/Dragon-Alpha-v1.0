/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.optim;

import java.util.Collection;
import java.util.Map;
import z.dragon.common.state.State;
import z.dragon.common.state.State.StateValue;
import z.dragon.engine.Engine;
import z.dragon.engine.Parameter;
import z.dragon.engine.Tensor;
import z.util.math.vector.Vector;

/**
 *
 * @author Gilgamesh
 */
public class Adamax extends Optimizer
{
    protected float lr_t;
    
    protected float beta1, a1, a2, expBeta1;
    protected Tensor[] V;
     
    protected float beta2, eps, expBeta2;
    protected Tensor[] S;
    
    protected float L1coef = 0.0f;
    protected float L2coef = 0.0f;
    
    //<editor-fold defaultstate="collapsed" desc="__init__">
    private void __init__(float beta1, float beta2, float eps) {
        this.beta1 = beta1; 
        this.beta2 = beta2; this.eps = eps;
        
        V = Tensor.zero_like(params); expBeta1 = 1.0f; 
        S = Tensor.zero_like(params); expBeta2 = 1.0f;
        Tensor.sync(V); Tensor.sync(S);
    } 
    //</editor-fold>
    public Adamax(Parameter[] params, float lr, float beta1, float beta2, float eps) {
        super(params, lr);
        __init__(beta1, beta2, eps);
    }
    
    public Adamax(Collection<Parameter> params, float lr, float beta1, float beta2, float eps) {
        super(params, lr);
        __init__(beta1, beta2, eps);
    }
    
    public Adamax(Map<String, Parameter> param_map, float lr, float beta1, float beta2, float eps) {
        super(param_map, lr);
        __init__(beta1, beta2, eps);
    }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    @Override public Adamax learning_rate(float lr) { super.learning_rate(lr); return this; }
    
    public Tensor[] exp_avg() { return V; }
    public Tensor[] exp_avg_sq() { return S; }
    
    public float beta1() {return beta1;}
    public Adamax beta1(float beta1) { this.beta1 = beta1;  return this; }
    
    public float beta2() { return beta2; }
    public Adamax beta2(float beta2) { this.beta2 = beta2; return this; }
    
    public float eps() { return eps; }
    public Adamax eps(float eps) { this.eps = eps; return this;}
    
    public float L1coef() { return L1coef; }
    public Adamax L1coef(float L1coef) { this.L1coef = L1coef; return this; }
    
    public float L2cof() {return L2coef;}
    public Adamax L2coef(float L2coef) {this.L2coef = L2coef; return this;}
    
    @Override
    public void append(StringBuilder sb) {
        sb.append(getClass().getSimpleName());
        sb.append(" { learning_rate = ").append(lr);
        sb.append(", [beta1, beta2, eps] = [")
                .append(beta1).append(", ")
                .append(beta2).append(", ")
                .append(eps).append("]");
        sb.append(", [L1coef, L2coef] = [")
                .append(L1coef).append(", ")
                .append(L2coef).append("] }");
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: state">
   //hypher-state--------------------------------------------------------------
    @Override
    protected void hypher_state(State dic) {
        dic.put("expBetas", State.floats(expBeta1, expBeta2));
    }
    
    @Override
    protected void update_hypher_state(State dic, boolean partial) {
        StateValue expBetas = dic.get("expBetas");
        State.set(expBetas, "fail to load expBetas", partial, ()->{
            float[] arr = Vector.toFloatVector(expBetas.toStringLines(), 2);
            expBeta1 = arr[0]; expBeta2 = arr[1];
        });
    }

    //param-state---------------------------------------------------------------
    protected String exp_avg_key(String param_name) { return param_name + ".exp_avg"; }
    protected String exp_avg_sq_key(String param_name) { return param_name + ".exp_avg_sq"; }
    
    @Override
    protected void param_state(State dic, int index, String paramName) {
        dic.put(exp_avg_key(paramName), V[index]);
        dic.put(exp_avg_sq_key(paramName), S[index]);
    }
    
    @Override
    protected void update_param_state(State dic, boolean partial, int index, String paramName) {
        String exp_avg_key = exp_avg_key(paramName);
        V[index].set(dic.get(exp_avg_key), partial, "fail to load " + exp_avg_key);
        
        String exp_avg_sq_key = exp_avg_sq_key(paramName);
        S[index].set(dic.get(exp_avg_sq_key), partial, "fail to load " + exp_avg_sq_key);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: update">
    @Override
    protected void __before_update__() {//init: Us = Uv = 1
        expBeta1 *= beta1; expBeta2 *= beta2;
        
        a1 = beta1; a2 = 1.0f - beta1; //exp_avg_mean
        
        double correct_beta2 = Math.sqrt(1 - expBeta2);
        lr_t = (float) (lr * correct_beta2 / (1 - expBeta1));
    }
    
    @Override
    protected void __update__(int index, Tensor gradient, Engine eg) {
        if(L1coef != 0 || L2coef != 0) {
            eg.adamax(params[index].ts(),//inplace: param.datas
                    V[index], a1, a2, 
                    S[index], beta2, eps, 
                    gradient, lr_t,
                    L1coef, L2coef); 
            return;
        }
        
        eg.adamax(params[index].ts(),//inplace: param.datas
                V[index], a1, a2, 
                S[index], beta2, eps,
                gradient, lr_t);
    }
    
    @Override
    protected void __update__(int index, Collection<Tensor> gradients, Engine eg) {
        if(L1coef != 0 || L2coef != 0) {
            eg.adamax_decay(params[index].ts(),//inplace: param.datas
                    V[index], a1, a2,
                    S[index], beta2, eps,
                    gradients, lr_t,
                    L1coef, L2coef);
            return;
        }
        
        eg.adamax(params[index].ts(),//inplace: param.datas
                V[index], a1, a2, 
                S[index], beta2, eps, 
                gradients, lr_t);
    }
    
    @Override
    protected void __clear__() {
        Tensor.delete(V); V = null;
        Tensor.delete(S); S = null;
    }
    //</editor-fold>
}
