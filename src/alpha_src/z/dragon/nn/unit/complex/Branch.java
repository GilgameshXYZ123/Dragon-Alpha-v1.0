/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.complex;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import z.dragon.common.state.State;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.nn.unit.GradientControllable;
import z.dragon.nn.unit.Train2Eval;
import z.dragon.nn.unit.Unit;
import z.dragon.common.state.State.AsyncStateUpdate;
import z.dragon.engine.Parameter;
import z.dragon.engine.Parameter.ParamMap;
import z.dragon.engine.Parameter.ParamSet;
import z.dragon.engine.Tensor.TensorSet;

/**
 *
 * @author Gilgamesh
 */
 @SuppressWarnings("unchecked")
public class Branch extends Unit
        implements GradientControllable, Train2Eval, AsyncStateUpdate
{
    protected Engine eg;
    
    protected final Unit[] brh;
    protected int input_tensor_num;
    
    protected Tensor[] X, deltaX;
    protected Tensor[] Y, deltaY;
    
    protected final int[] Ywidth; 
    protected final Tensor[][] Ymat, mdeltaY, mdeltaX;
    
    public Branch(Collection<Unit> units) {
        if(units == null || units.isEmpty()) throw new NullPointerException("units is null");
        if(units.size() < 2) throw new IllegalArgumentException("Branch at leaset has 2 units");
        
        this.brh = new Unit[units.size()]; int index = 0;
        for(Unit u : brh) { unitNonNull(u, index++); brh[index++] = u; }
        
        Ymat = new Tensor[brh.length][];
        Ywidth = new int[brh.length];
        mdeltaY = new Tensor[brh.length][];
        mdeltaX = new Tensor[brh.length][];
    }
    
    public Branch(Unit... units) {
        if(units == null || units.length == 0)  throw new NullPointerException("Branch: units is null");
        if(units.length < 2) throw new IllegalArgumentException("Branch at leaset has 2 units");
        for(int i=0; i<units.length; i++) unitNonNull(units[i], i);
        
        this.brh = new Unit[units.length];
        System.arraycopy(units, 0, brh, 0, brh.length);
        
        Ymat = new Tensor[brh.length][];
        Ywidth = new int[brh.length];
        mdeltaY = new Tensor[brh.length][];
        mdeltaX = new Tensor[brh.length][];
    }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    private static void unitNonNull(Unit u, int index) {
        if(u == null) throw new NullPointerException(
                String.format("Branch: units[%d] is null", index));
    }

    public int width() { return brh.length; }
    public Unit[] brh() { return brh;}
    public Unit brh(int index) {
        if(index < 0) index = brh.length + index;
        return brh[index];
    }
     
    public Engine engine() { return eg;}
    public int input_tensor_num() {return input_tensor_num;}
     
    @Override
    public void append(String pre, StringBuilder sb) { 
        sb.append(default_name());
        sb.append("{ width = ").append(width());
        String next_pre = pre + add_pre; int index = 0;
        for(Unit u : brh) {
            sb.append('\n').append(pre);
            sb.append('(').append(index++).append(") ");
            if(u.isComplex()) u.append(next_pre, sb);
            else u.append(sb);
        }
        sb.append('\n').append(pre).append('}');
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: others">
    @Override public boolean isComplex() { return true; }
     
    @Override 
    public Collection<Unit> next() {
        UnitSet set = new UnitSet(brh.length);
        for(Unit u : brh) set.addAll(u.next()); return set;
    }   
    
    protected boolean backward_grads = true;
    @Override public boolean backward_grads() {return backward_grads;}
    @Override public Branch backward_grads(boolean flag) {
        backward_grads = flag;
        for(Unit u : brh) 
            if(u instanceof GradientControllable) 
                ((GradientControllable)u).backward_grads(flag);
        return this;
    }
    
    //<editor-fold defaultstate="collapsed" desc="functions: find">
    @Override
    public <T extends Unit> Set<T> find(Class<T> cls) {
        UnitSet set = new UnitSet(brh.length);
        find(set, cls); 
        return (Set<T>) set;
    }
    
    @Override
    public <T extends Unit> Set<T> find() {
        UnitSet set = new UnitSet(brh.length << 1);
        find(set, Unit.class); 
        return (Set<T>) set;
    }
    
    @Override
    public void find(UnitSet set, Class<? extends Unit> cls) {
        super.find(set, cls);
        for(Unit sc : brh) sc.find(set, cls);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="functions: params & vars">
    @Override public void params(ParamSet set) { for(Unit u : brh) u.params(set); }
    @Override public Set<Parameter> params() { ParamSet set = new ParamSet(brh.length << 1); params(set); return set; }
  
    @Override public void param_map(ParamMap<String> map) { for(Unit u : brh) u.param_map(map); }
    @Override public Map<String, Parameter> param_map() { ParamMap<String> map = new ParamMap<>(brh.length << 1); param_map(map); return map; }
    
    //variables && gc-----------------------------------------------------------
    @Override public void vars(TensorSet vars) { for(Unit sc : brh) sc.vars(vars); vars.add(deltaX); }
    @Override public Set<Tensor> vars() { TensorSet set = new TensorSet(brh.length << 1); vars(set); return set; }

    @Override
    public void gc() {
        for(Unit u : brh) u.gc();
        
        eg.delete(X); X = null; 
        eg.delete(Y); Y = null; 
        eg.delete(deltaX); deltaX = null; 
        eg.delete(deltaY); deltaY = null; 
      
        for(int i=0; i<brh.length; i++) {
            eg.delete(Ymat[i]); Ymat[i] = null;
            eg.delete(mdeltaY[i]); mdeltaY[i] = null;
            eg.delete(mdeltaX[i]); mdeltaX[i] = null;
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="functions: state">
    @Override public void state(State dic) { for(Unit u : brh) u.state(dic); }
    @Override public State state() { State dic = new State(); state(dic); return dic;  }
    
    //update state async--------------------------------------------------------
    private boolean update_state_sync = false;
    @Override public boolean update_state_sync() { return update_state_sync; }
    @Override public <T extends State.AsyncStateUpdate> T update_state_sync(boolean flag) { update_state_sync = flag; return (T) this; }
    
    @Override
    public void update_state(State dic, boolean partial, List<Future<?>> fts) {
        if(fts == null) {//sync update_state mode
            for(Unit u : brh) {
                if(u instanceof AsyncStateUpdate)
                    ((AsyncStateUpdate)u).update_state(dic, partial, null);
                else u.update_state(dic, partial); 
            }
            return;
        }
        
        for(Unit u : brh) {//async update_state mode
            if(u instanceof AsyncStateUpdate) 
                ((AsyncStateUpdate)u).update_state(dic, partial, fts);
            else fts.add(exec.submit(()-> { u.update_state(dic, partial); }));
        }
    }
    
    @Override 
    public void update_state(State dic, boolean partial) {
        ArrayList<Future<?>> fts = (update_state_sync? null:
                new ArrayList<>(brh.length));
        this.update_state(dic, partial, fts);
        
        if(fts == null || fts.isEmpty()) return;
        try { for(Future<?> ft : fts)  ft.get(); }
        catch(InterruptedException | ExecutionException e) {
            throw new RuntimeException(e); 
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="functions: Train2Eval">
    @Override
    public boolean training() {
        for(Unit u : brh) 
            if(u instanceof Train2Eval) 
                if(!((Train2Eval)u).training()) return false;
        return true;
    }
    
    @Override
    public Branch train() {
        for(Unit u : brh) 
            if(u instanceof Train2Eval) ((Train2Eval)u).train();
        return this;
    }

    @Override
    public Branch eval() {
        for(Unit u : brh) 
            if(u instanceof Train2Eval) ((Train2Eval)u).eval();
        return this;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="functions: init & name">
    @Override 
    public Branch name(String name) {  
        this.name = name;
        for(int i=0; i<brh.length; i++) 
            brh[i].name(name + '.' + brh[i].getClass().getSimpleName() + i);
        return this;
    }
    
    @Override
    public <T extends Unit> T init(Engine eg, String name) { 
        this.eg = eg; this.name = name;
        for(int i=0; i<brh.length; i++) {
            String subName = name + '.' + brh[i].getClass().getSimpleName() + i;
            brh[i].init(eg, subName);
        }
        return (T) this;
    }
    //</editor-fold>
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="running-area: forward-propagation">
    @Override
    public synchronized Tensor[] forward(Tensor... input) {
        X = input; 
        input_tensor_num = X.length;
        
        int length = 0;
        for(int i=0; i<brh.length; i++) {//store output of the ith brh
            Ymat[i] = brh[i].forward(X);
            Ywidth[i] = Ymat[i].length;
            length += Ymat[i].length;
        }
        
        //Ymat(2D) -> Y(1D)
        Y = new Tensor[length]; int index = 0;
        for (Tensor[] Yarr : Ymat) for (Tensor y : Yarr) Y[index++] = y;
        return Y;
    }
    
    @Override protected void traceBack(Unit next, int last_out_index, int next_in_index) {}
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="running-area: backward-propagation">
    @Override public Tensor[] collectGradientFromNext() { return null; }
    
    @Override//gradient must be aligned with the gradient
    public synchronized Tensor[] backward(Tensor... gradient) {
        deltaY = gradient; 
            
        //branchL__backward__()-------------------------------------------------
        Tensor[][] mdeltaXT = new Tensor[input_tensor_num][brh.length];
        for(int i=0, index = 0; i<brh.length; i++)
        {
            //split the gradient for the ith brh
            int width = Ywidth[i];
            mdeltaY[i] = new Tensor[width];
            for(int j=0; j<width; j++) mdeltaY[i][j] = gradient[index++];
            
            //brh backward
            Tensor[] dx = brh[i].backward(mdeltaY[i]);//must size = input.size
            for(int j=0; j<input_tensor_num; j++) mdeltaXT[i][j] = dx[j];
        }
        //branch:__backward__()-------------------------------------------------
        
        if(!backward_grads) return null;
        
        //reduce: brh.length -> 1
        deltaX = new Tensor[input_tensor_num];
        for(int i=0; i<input_tensor_num; i++) {
            Tensor.sync(mdeltaXT[i]);
            deltaX[i] = eg.summary(true, mdeltaXT[i]);
        }
        return deltaX;
    }     
    
    @Override public Tensor gradient(int index) { return null; }
    //</editor-fold>
}
