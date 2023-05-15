/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit;

import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadFactory;
import z.dragon.common.state.State;
import z.dragon.common.state.State.StateReader;
import z.dragon.engine.Tensor;
import z.dragon.common.state.State.Stateful;
import z.dragon.engine.Engine;
import z.dragon.engine.Parameter;
import z.dragon.engine.Parameter.ParamMap;
import z.dragon.engine.Parameter.ParamSet;
import z.dragon.engine.Tensor.TensorSet;

/**
 *
 * @author Gilgamesh
 */
@SuppressWarnings("unchecked")
public abstract class Unit implements Stateful, StateReader
{
    //<editor-fold defaultstate="collapsed" desc="daemon exec">
    protected static final ThreadFactory daemonThreadFactory = (Runnable r) -> {
        Thread t = new Thread(r);
        t.setDaemon(true);
        return t;
    };
    protected static final ExecutorService exec = Executors.newFixedThreadPool(4, daemonThreadFactory); 
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="static class: UnitMap">
    public static class UnitMap<V> extends HashMap<Unit, V> 
    {
        private static final long serialVersionUID = 141558956307138L;
        
        public UnitMap() {super();}
        public UnitMap(int initialCapacity) { super(initialCapacity); }

        @Override
        public V put(Unit unit, V value) {
            return (unit == null ? null : super.put(unit, value));
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="static class: UnitSet">
    public static class UnitSet extends HashSet<Unit> 
    {
        private static final long serialVersionUID = 1L;
        
        public UnitSet() { super(); }
        public UnitSet(int init_capacity) { super(init_capacity); }

        @Override
        public boolean add(Unit unit) {
            return (unit == null ? false :  super.add(unit));
        }
        
        public boolean add(Unit...units) {
            if(units == null || units.length == 0) return false;
            boolean result = true;
            for(Unit unit : units) {
                result &= (unit == null ? false : super.add(unit));
            }
            return result;
        }
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="functions: others">
    public abstract Collection<Unit> next();
    public abstract boolean isComplex();
    
    public static final String add_pre = "  ";
    public abstract void append(String pre, StringBuilder sb);
    public final void append(StringBuilder sb) { append("", sb); }
    public <T extends Unit> T println() { System.out.println(toString()); return (T) this; }
    
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder(64);
        this.append(sb);
        return sb.toString();
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="functions: vars & params">  
    public void delete() {
        gc();//delete all variables, and set references to nullptr
        params().forEach((Parameter t)-> { t.delete(); });
    }

    public abstract void vars(TensorSet set);
    public Set<Tensor> vars() { TensorSet set = new TensorSet(4); vars(set); return set; }
    public abstract void gc();

    public abstract void params(ParamSet set);
    public Set<Parameter> params() { ParamSet set = new ParamSet(4); params(set); return set; }
    
    public abstract void param_map(ParamMap<String> map);
    public Map<String, Parameter> param_map() { ParamMap<String> map = new ParamMap<>(4); param_map(map);  return map; }
    
    public boolean need_grads() {
        boolean flag = false;
        for(Parameter param : params()) flag = (flag || param.need_grads());
        return flag;
    }

    public Unit need_grads(boolean flag) {
        for(Parameter param : params()) param.need_grads(flag);
        return this;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="functions: find">
    public void find(UnitSet set, Class<? extends Unit> cls) {
        if(cls.isAssignableFrom(getClass())) set.add(this);
    }
    
    public <T extends Unit> Set<T> find() {
        UnitSet set = new UnitSet();
        find(set, Unit.class);
        return (Set<T>) set; 
    }
    
    public <T extends Unit> Set<T> find(Class<T> cls) { 
        UnitSet set = new UnitSet();
        find(set, cls);
        return (Set<T>) set; 
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="functions: state"> 
    @Override public abstract void update_state(State dic, boolean partial);
    @Override public State read() { return state(); }
    @Override public State state() { State dic = new State(2);  this.state(dic); return dic; }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area">
    protected String name = default_name();
    protected String default_name() { return getClass().getSimpleName(); }
    public String name() { return name; }
    public Unit name(String name) { this.name = name; return this;}
    
    public <T extends Unit> T init(Engine eg) { init(eg, name); return (T) this; }
    public abstract <T extends Unit> T init(Engine eg, String name);
     
    public abstract Tensor[] forward(Tensor... input);
    protected abstract void traceBack(Unit next, int out_index, int next_in_index);//called by the next node
    
    public abstract Tensor[] collectGradientFromNext();
    public abstract Tensor[] backward(Tensor... gradient);
    public abstract Tensor gradient(int index);
    //</editor-fold>
}
