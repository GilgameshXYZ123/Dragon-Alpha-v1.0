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
 * <pre>
 *Adamod(W):
 * Init: Uv = beta^t, Us = beta2^t
 * Iterate:
 *      a1 = beta1, a2 = 1 - beta1
 *      b1 = beta2, b2 = 1 - beta2
 *      (1) V = a1*V + a2*grad       
 *      (2) S = b1*S + b2*grad^2      
 *      (3) lr_t = lr * sqrt(1 - Us) / (1 - Uv)
 *      (4) eps_t = eps * sqrt(1 - Us)
 *      (5) neta = lr_t / (sqrt(S) + eps_t)
 *      (6) G = c1*G + c2*neta
 *      (7) neta = min(neta, G) * V
 *      (8) W -= stepSize  # gradient descent
 *      Uv *= alpha; Us *= beta.
 * </pre>
 * @author Gilgamesh
 */
public class Adamod  extends Optimizer
{
    protected float lr_t, eps_t;
    
    protected float beta1, a1, a2, expBeta1;
    protected Tensor[] V;
     
    protected float beta2, eps, b1, b2, expBeta2;
    protected Tensor[] S;
    
    protected float beta3, c1, c2;
    protected Tensor[] G;
    
    protected float L1coef = 0.0f;
    protected float L2coef = 0.0f;
    
    //<editor-fold defaultstate="collapsed" desc="__init__">
    private void __init__(float beta1, float beta2, float eps, float beta3) {
        this.beta1 = beta1; //V
        this.beta2 = beta2; this.eps = eps;//S
        this.beta3 = beta3; //G
        
        V = Tensor.zero_like(params); expBeta1 = 1.0f; 
        S = Tensor.zero_like(params); expBeta2 = 1.0f;
        G = Tensor.zero_like(params);
        Tensor.sync(V); Tensor.sync(S); Tensor.sync(G);
    } 
    //</editor-fold>
    public Adamod(Parameter[] params, float lr, float beta1, float beta2, float eps, float beta3)  {
        super(params, lr);
        __init__(beta1, beta2, eps, beta3);
    }
    
    public Adamod(Collection<Parameter> params, float lr, float beta1, float beta2, float eps, float beta3) {
        super(params, lr);
       __init__(beta1, beta2, eps, beta3);
    }
    
    public Adamod(Map<String, Parameter> param_map, float lr, float beta1, float beta2, float eps, float beta3) {
        super(param_map, lr);
       __init__(beta1, beta2, eps, beta3);
    }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    @Override public Adamod learning_rate(float lr) { super.learning_rate(lr); return this; }
    
    public Tensor[] exp_avg() { return V; }
    public Tensor[] exp_avg_sq() { return S; }
    public Tensor[] exp_avg_step() { return G; }
    
    public float beta1() {return beta1;}
    public Adamod beta1(float beta1) { this.beta1 = beta1;  return this; }
    
    public float beta2() { return beta2; }
    public Adamod beta2(float beta2) { this.beta2 = beta2; return this; }
    
    public float eps() { return eps; }
    public Adamod eps(float eps) { this.eps = eps; return this;}
    
    public float beta3() { return beta3; }
    public Adamod beta3(float beta3) { this.beta3 = beta3; return this; }
    
    public float L1coef() { return L1coef; }
    public Adamod L1coef(float L1coef) {this.L1coef = L1coef; return this;}
    
    public float L2coef() { return L2coef; }
    public Adamod L2coef(float L2coef) {this.L2coef = L2coef; return this;}
    
    @Override
    public void append(StringBuilder sb) {
        sb.append(getClass().getSimpleName());
        sb.append("{ learning_rate = ").append(lr);
        sb.append(", [beta1, beta2, eps, beta3] = (")
                .append(beta1).append(", ")
                .append(beta2).append(", ")
                .append(eps).append(", ")
                .append(beta3).append("]");
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
    protected String exp_avg_step_key(String param_name) { return param_name + ".exp_avg_step"; }
    
    @Override
    protected void param_state(State dic, int index, String param_name) {
        dic.put(exp_avg_key(param_name), V[index]);
        dic.put(exp_avg_sq_key(param_name), S[index]);
        dic.put(exp_avg_step_key(param_name), G[index]);
    }

    @Override
    protected void update_param_state(State dic, boolean partial, int index, String param_name) {
        String exp_avg_key = exp_avg_key(param_name);
        V[index].set(dic.get(exp_avg_key), partial, "fail to load " + exp_avg_key);
        
        String exp_avg_sq_key = exp_avg_sq_key(param_name);
        S[index].set(dic.get(exp_avg_sq_key), partial, "fail to load " + exp_avg_sq_key);
        
        String exp_avg_step_key = exp_avg_step_key(param_name);
        G[index].set(dic.get(exp_avg_step_key), partial, "fail to load" + exp_avg_step_key);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: update">
    @Override
    protected void __before_update__() {
        expBeta1 *= beta1; expBeta2 *= beta2; //init: expBeta1 = expBeta2 = 1
        
        a1 = beta1; a2 = 1.0f - beta1; //exp_avg
        b1 = beta2; b2 = 1.0f - beta2; //exp_avg_sq
        c1 = beta3; c2 = 1.0f - beta3; //exp_step
        
        double correct_beta2 =  Math.sqrt(1 - expBeta2);
        lr_t =  (float) (lr  * correct_beta2 / (1 - expBeta1));
        eps_t = (float) (eps * correct_beta2);
    }

    @Override
    protected void __update__(int index, Tensor gradient, Engine eg) {
        if(L1coef !=0 || L2coef != 0) {
            eg.adamod(params[index].ts(),//inplace: param.datas
                    V[index], a1, a2,
                    S[index], b1, b2, eps_t,
                    G[index], c1, c2,
                    gradient, lr_t,
                    L1coef, L2coef);
            return;
        }
        
        eg.adamod(params[index].ts(),//inplace: param.datas
                V[index], a1, a2,
                S[index], b1, b2, eps_t,
                G[index], c1, c2,
                gradient, lr_t);
    }

    @Override
    protected void __update__(int index, Collection<Tensor> gradients, Engine eg) {
        if(L1coef !=0 || L2coef != 0) {
            eg.adamod(params[index].ts(),//inplace: param.datas
                    V[index], a1, a2,
                    S[index], b1, b2, eps_t,
                    G[index], c1, c2,
                    gradients, lr_t,
                    L1coef, L2coef);
            return;
        }
        
       eg.adamod(params[index].ts(),//inplace: param.datas
                V[index], a1, a2, 
                S[index], b1, b2, eps_t,
                G[index], c1, c2,
                gradients, lr_t);
    }

    @Override
    protected void __clear__() {
        Tensor.delete(V); V = null;
        Tensor.delete(S); S = null;
        Tensor.delete(G); G = null;
    }
    //</editor-fold>
}
