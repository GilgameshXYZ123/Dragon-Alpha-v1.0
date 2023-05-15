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
public class Sequence extends Unit
        implements GradientControllable, Train2Eval, AsyncStateUpdate
{
    //<editor-fold defaultstate="collapsed" desc="static class: Empty: EmptySeq">
    protected static final class EmptySeq extends Sequence {
        protected EmptySeq() { }
        
        @Override public int length() { return 0; }
        @Override public Unit[] seq() { return null; }
        @Override public Unit seq(int index) { return null; }
    
        @Override public void append(String pre, StringBuilder sb) {
            sb.append(pre).append("Sequence { length = ").append(0).append(" }");
        }
        
        @Override public Collection<Unit> next() { return null; }
        @Override public boolean training() { return true; }
        @Override public Sequence train() { return this; }
        @Override public Sequence eval() { return this; }
    }
    //</editor-fold>
    
    protected final Unit[] seq;
    
    public Sequence(Collection<Unit> units) {
        if(units == null) throw new NullPointerException("units is null");
        seq = new Unit[units.size()]; int index = 0;
        for(Unit u : units) { unitNonNull(u, index); seq[index++] = u; }
    }
    
    public Sequence(Unit... units) {
        if(units == null) throw new NullPointerException("units is null");
        for(int i=0; i<units.length; i++) unitNonNull(units[i], i);
        seq = new Unit[units.length];
        System.arraycopy(units, 0, seq, 0, seq.length);
    }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public int length() { return seq.length; }
    public Unit[] seq() { return seq; }
    public Unit seq(int index) {
        if(index < 0) index = seq.length + index;
        return seq[index];
    }
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(default_name());
        sb.append("{ length = ").append(length());
        String next_pre = pre + add_pre; int index = 0;
        for(Unit u : seq) {
            sb.append('\n').append(pre);//start a new line
            sb.append('(').append(index++).append(") ");
            if(u.isComplex()) u.append(next_pre, sb);
            else u.append(sb);
        }
        sb.append('\n').append(pre).append('}');
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: others">
    private static void unitNonNull(Unit u, int index) {
        if(u == null) throw new NullPointerException(
                String.format("Sequence: units[%d] is null", index));
    }
    
    @Override public boolean isComplex() { return true; }
    
    @Override public Collection<Unit> next() { return seq[seq.length - 1].next(); }
    
    private boolean backward_grads = true;
    @Override public boolean backward_grads() { return backward_grads;}
    @Override public Unit backward_grads(boolean flag) {
        backward_grads = flag;
        if(seq[0] instanceof GradientControllable) 
            ((GradientControllable)seq[0]).backward_grads(flag);
        return this;
    }
     
    //<editor-fold defaultstate="collapsed" desc="functions: find">
    @Override
    public <T extends Unit> Set<T> find(Class<T> cls) {
        UnitSet set = new UnitSet(seq.length << 1);
        find(set, cls); 
        return (Set<T>) set;
    }
    
    @Override
    public <T extends Unit> Set<T> find() {
        UnitSet set = new UnitSet(seq.length << 1);
        find(set, Unit.class); 
        return (Set<T>) set;
    }
    
    @Override
    public void find(UnitSet set, Class<? extends Unit> cls) {
        super.find(set, cls);//add this, if this.class is a subclass of clazz
        for(Unit u : seq) u.find(set, cls);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="functions: params & vars">
    @Override public void params(ParamSet params) { for(Unit u : seq) u.params(params); }
    @Override public Set<Parameter> params() { ParamSet set = new ParamSet(seq.length << 1); params(set); return set;  }
    
    @Override public void param_map(ParamMap<String> map) { for(Unit u : seq) u.param_map(map); }
    @Override public Map<String, Parameter> param_map() { ParamMap<String> map = new ParamMap<>(seq.length << 1); param_map(map); return map; }
    
    @Override public Set<Tensor> vars() { TensorSet set = new TensorSet(seq.length << 1); vars(set); return set; }
    @Override public void vars(TensorSet vars) { for(Unit u : seq) u.vars(vars); }
    @Override public void gc() { for(Unit u : seq) u.gc(); }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="functions: state">
    @Override public void state(State dic) { for(Unit u : seq) u.state(dic); }
    @Override public State state() { State dic = new State(seq.length); state(dic); return dic; }
    
    private boolean update_state_sync = false;
    @Override public boolean update_state_sync() { return update_state_sync; }
    @Override public <T extends AsyncStateUpdate> T update_state_sync(boolean flag) { this.update_state_sync = flag; return (T) this; }
    
    //update state async--------------------------------------------------------
    @Override
    public void update_state(State dic, boolean partial, List<Future<?>> fts) {
        if(fts == null) {//sync update_state mode
            for(Unit u : seq) {
                if(u instanceof AsyncStateUpdate)
                    ((AsyncStateUpdate)u).update_state(dic, partial, null);
                else u.update_state(dic, partial); 
            }
            return;
        }
        
        for(Unit u : seq) {//async update_state mode
            if(u instanceof AsyncStateUpdate) 
                ((AsyncStateUpdate)u).update_state(dic, partial, fts);
            else fts.add(exec.submit(()-> { u.update_state(dic, partial); }));
        }
    }
    
    @Override 
    public void update_state(State dic, boolean partial) {
        ArrayList<Future<?>> fts = (update_state_sync? null: new ArrayList<>(seq.length));
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
        for(Unit u : seq) 
            if(u instanceof Train2Eval) 
                if(!((Train2Eval)u).training()) return false;
        return true;
    }
    
    @Override
    public Sequence train() {
        for(Unit u : seq) 
            if(u instanceof Train2Eval) ((Train2Eval)u).train();
        return this;
    }

    @Override
    public Sequence eval() {
         for(Unit u : seq) 
            if(u instanceof Train2Eval) ((Train2Eval)u).eval();
        return this;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="functions: name & init"> 
    @Override
    public Sequence name(String name) {
        this.name = name;
        for(int i=0; i < seq.length; i++)
            seq[i].name(name + '.' + seq[i].getClass().getSimpleName() + i);
        return this;
    }
    
    @Override
    public <T extends Unit> T init(Engine eg, String name) {
        this.name = name;
        for(int i=0; i<seq.length; i++) {
            String subName = name + '.' + seq[i].getClass().getSimpleName() + i;
            seq[i].init(eg, subName);
        }
        return (T) this;
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: propagation"> 
    @Override protected void traceBack(Unit next, int out_index, int next_in_index) {}
    @Override //forward: directly connect to seq.last, no need do traceback
    public synchronized Tensor[] forward(Tensor... input) {
        Tensor[] Y = input;
        for(int i=0; i<seq.length;i++) Y = seq[i].forward(Y);
        return Y;
    }
    
    @Override public Tensor gradient(int index) { return null; }
    @Override public Tensor[] collectGradientFromNext() { return null; }
    @Override //backward: directly connect to seq.last, no need do collect, or get gradient
    public synchronized Tensor[] backward(Tensor... gradient) {
        Tensor[] dX = gradient;
        for(int i = seq.length - 1; i>=0; i--) dX = seq[i].backward(dX);
        return dX;
    }
    //</editor-fold>
}
