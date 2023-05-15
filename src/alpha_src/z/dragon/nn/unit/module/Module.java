/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.module;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import z.dragon.common.state.State;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.nn.unit.Train2Eval;
import z.dragon.nn.unit.Unit;
import z.dragon.nn.unit.GradientControllable;
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
public abstract class Module extends Unit
        implements Train2Eval, AsyncStateUpdate, GradientControllable 
{   
    protected Engine eg;
     
    private final UnitMap<String>   unit_map  = new UnitMap<>();//<unit, fid_name>
    private final Map<String, Unit> runit_map = new HashMap<>();//<fid_name, unit>
    
    private final GraphHead head = new GraphHead(this);
    private final GraphTail tail = new GraphTail(this);
    
    private Tensor[] X, deltaX;//input, input.gradient
    private Tensor[] Y, deltaY;//output, output.gradient
    
    //<editor-fold defaultstate="collapsed" desc="Basic-functions">
    public Engine engine() { return eg; }
    
    public int size() {
        if(!constructed) { register_member_units(); constructed = true; }
        return unit_map.size();
    }
    
    public Set<Unit> units() { 
        if(!constructed) { register_member_units(); constructed = true; }
        return unit_map.keySet(); 
    }
    
    public <T extends Unit> T unit(String fid_name) {
        if(!constructed) { register_member_units(); constructed = true; }
        return (T) runit_map.get(fid_name);
    }
    
    public Tensor[] X() { return X; }
    public Tensor[] Y() { return Y; }
    public Tensor[] deltaX() { return deltaX; }
    public Tensor[] deltaY() { return deltaY; }
    
    @Override
    public void append(String pre, StringBuilder sb) {
        if(!constructed) { register_member_units(); constructed = true; }
        
        sb.append(default_name());
        sb.append(" { size = ").append(size());
        String next_pre = pre + add_pre;
        try {//can't use unit_map, as it may shuffle the order of member params
            for(Field fid : getClass().getDeclaredFields()) {
                fid.setAccessible(true);
                Object member = fid.get(this);
                if(member instanceof Unit) {
                    Unit u = (Unit) member; 
                    String fid_name = fid.getName();
                    
                    sb.append('\n').append(pre);//start a new line
                    sb.append('(').append(fid_name).append(") ");
                    if(u.isComplex()) u.append(next_pre, sb);
                    else u.append(sb);
                }
                fid.setAccessible(false);
            }
        }
        catch(IllegalAccessException | IllegalArgumentException | SecurityException e) {
            throw new RuntimeException(e);
        }
        sb.append('\n').append(pre).append('}');
    }
    
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder(512);
        this.append(sb);
        return sb.toString();
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: others">
    @Override public boolean isComplex() { return true; }
    
    @Override public Collection<Unit> next() { return tail.nexts; }

    @Override public boolean backward_grads() { return tail.backward_grads(); }
    @Override public Module backward_grads(boolean flag) { tail.backward_grads(flag); return this; }
    
    //<editor-fold defaultstate="collapsed" desc="functions: find">
    @Override
    public <T extends Unit> Set<T> find(Class<T> cls) {
        UnitSet set = new UnitSet(unit_map.size() << 1);
        find(set, cls);
        return (Set<T>) set;
    }
    
    @Override
    public <T extends Unit> Set<T> find() {
        UnitSet set = new UnitSet(unit_map.size() << 1);
        find(set, Unit.class); 
        return (Set<T>) set;
    }
    
    @Override
    public void find(UnitSet set, Class<? extends Unit> cls) {
        if(!constructed) { register_member_units(); constructed = true; }
        super.find(set, cls);//add this, if this.class is a subclass of clazz
        unit_map.keySet().forEach((u) -> { u.find(set, cls); });
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="functions: params & vars">
    @Override public Set<Parameter> params() { ParamSet set = new ParamSet(unit_map.size() << 1); params(set); return set; }
    @Override public void params(ParamSet set) { 
        if(!constructed) { register_member_units(); constructed = true; }
        for(Unit u : unit_map.keySet()) u.params(set);  
    }
    
    @Override public Map<String, Parameter> param_map() { ParamMap<String> map = new ParamMap<>(unit_map.size() << 1); param_map(map); return map; }
    @Override public void param_map(ParamMap<String> map) { 
        if(!constructed) { register_member_units(); constructed = true; }
        for(Unit u : unit_map.keySet()) u.param_map(map); 
    }
    
    //variables & gc------------------------------------------------------------
    @Override public Set<Tensor> vars() { TensorSet set = new TensorSet(unit_map.size() << 1); vars(set);  return set; }
    @Override public void vars(TensorSet set) { 
        if(!constructed) { register_member_units(); constructed = true; }
        head.vars(set); tail.vars(set); 
        for(Unit u : unit_map.keySet()) u.vars(set);  
    }
    
    @Override
    public void gc() {
        X = null; deltaX = null;
        Y = null; deltaY = null;
        head.gc(); tail.gc();
        for(Unit u : unit_map.keySet()) u.gc();
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="functions: state">
    @Override public State state() { State dic = new State(unit_map.size() << 1); state(dic); return dic;  }
    
    @Override 
    public void state(State dic) { 
        if(!constructed) { register_member_units(); constructed = true; }
        for(Unit u : unit_map.keySet()) u.state(dic);  
    }
    
    //update state async--------------------------------------------------------
    private boolean update_state_sync = false;
    @Override public boolean update_state_sync() { return update_state_sync; }
    @Override public <T extends AsyncStateUpdate> T update_state_sync(boolean flag) { update_state_sync = flag; return (T) this; }
    
    @Override
    public void update_state(State dic, boolean partial, List<Future<?>> fts) {
        if(!constructed) { register_member_units(); constructed = true; }
        
        if(fts == null) {//sync update_state mode
            for(Unit u : unit_map.keySet()) {
                if(u instanceof AsyncStateUpdate) 
                    ((AsyncStateUpdate)u).update_state(dic, partial, null);
                else u.update_state(dic, partial); 
            }
            return; 
        }
        
        for(Unit u : unit_map.keySet()) {//async update_state mode
            if(u instanceof AsyncStateUpdate) 
                ((AsyncStateUpdate)u).update_state(dic, partial, fts);
            else fts.add(exec.submit(()-> { u.update_state(dic, partial); }));
        }
    }
    
    @Override
    public void update_state(State dic, boolean partial) { 
        ArrayList<Future<?>> fts = (update_state_sync? null: new ArrayList<>(unit_map.size()));
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
         if(!constructed) { register_member_units(); constructed = true; }
        for(Unit u : unit_map.keySet())
            if(u instanceof Train2Eval) 
                if(!((Train2Eval)u).training()) return false;
        return true;
    }
    
    @Override
    public Module train() {
        if(!constructed) { register_member_units(); constructed = true; }
        for(Unit u : unit_map.keySet()) 
            if(u instanceof Train2Eval) ((Train2Eval)u).train();
        return this;
    }

    @Override
    public Module eval() {
        if(!constructed) { register_member_units(); constructed = true; }
        for(Unit u : unit_map.keySet()) 
            if(u instanceof Train2Eval) ((Train2Eval)u).eval();
        return this;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="functions: name & init">
    private boolean constructed = false;
    
    private synchronized void register_member_units() {
        if(constructed) return;//constructed only once
        try {
            for(Field fid : getClass().getDeclaredFields()) {
                fid.setAccessible(true);
                Object member = fid.get(this);
                if(member instanceof Unit) {
                    Unit u = (Unit) member; 
                    String fid_name = fid.getName();
                    
                    u.name(name + '.' + fid_name);//recusively set unit.name
                    unit_map.put(u, fid_name);//<unit, fid_name>
                    runit_map.put(fid_name, u);//<fid_name, unit>
                }
                fid.setAccessible(false);
            }
        }
        catch(IllegalAccessException | IllegalArgumentException | SecurityException e) {
            throw new RuntimeException(e);
        }
    }
    
    @Override 
    public Module name(String name) {
        if(!constructed) { register_member_units(); constructed = true; }

        this.name = name;
        tail.name(name + ".tail"); 
        head.name(name + ".head"); 
        unit_map.forEach((Unit u, String fid_name)->{  
            u.name(name + '.' + fid_name); 
        });
        return this;
    }
    
    public void __init__(Engine eg) {}
    
    @Override 
    public synchronized <T extends Unit> T init(Engine eg, String name)  {
        if(!constructed) { register_member_units(); constructed = true; }
        
        this.name = name;
        head.init(eg, name + ".head"); 
        tail.init(eg, name + ".tail");
        unit_map.forEach((Unit u, String fid_name)-> {
            u.init(eg, name + '.' + fid_name);
        });
        
        __init__(this.eg = eg);//customized init
        return (T) this;
    }   
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: forward-propagation">
    public abstract Tensor[] __forward__(Tensor... X);
    
    @Override //module.last -> module[starts -> starts.next -> .... -> tail] -> module.next
    public synchronized Tensor[] forward(Tensor... input) {
        X = head.forward(input);//this.starts = head.next
        return Y = tail.forward(__forward__(X));//this.next = tail.next;
    }

    @Override//all next unit is module.innerUnits[starts]
    protected synchronized void traceBack(Unit next, int out_index, int next_in_index) {
        tail.traceBack(next, out_index, next_in_index);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: backward-propagation">
    @Override //module.trackback(): tail.traceBack();
    public synchronized Tensor[] collectGradientFromNext() { 
        return tail.collectGradientFromNext(); 
    }
   
    private final HashSet<Unit> visited = new HashSet<>(4);
    
    /**
     * <pre>.
     * if Unit instance of module
     * (1) for(Unit next : module.nexts == module.tail.next) backward(next, backed);
     * (2) module.collectGradientFromNext() = module.tail.collectGradientFromNext()
     *  As: module.nexts = tail.nexts, 
     * (3) module.backward(gradient): {@code 
     *      module.deltaX <- head.deltaX <- head.collectFromNext()
     *          <- head.next.backward(head.next.collectFromNext()) - ....
     *          <- tail.last.backward(tail.last.collectFromNext()) <- tail.backward(gradient) }
     * [1] when: module.hasNext -> tail.hasNext
     *  module.collectFromNext is called: -> module.tail.collectGradientFromNext is called
     * [2] when: !module.hasNext -> tail.hasNoNext:  
     *  so module, module.tail is the end of global compute graph, 
     *  just set: module.deltaY = tail.deltaX = tail.deltaY
     * @param unit 
     */
    private void backward(Unit unit) {
        if(visited.contains(unit)) return;//tail is visited, head will be visited
      
        for(Unit next : unit.next()) backward(next);//go to the leaf node in the compute graph
        unit.backward(unit.collectGradientFromNext());//ends.collect gradients from tail
        
        visited.add(unit);
    }
    
    @Override //the gradients must be alinged with input, no need to reorder
    public synchronized Tensor[] backward(Tensor... gradient) {
        //the end node of graph, tail.deltaY = this.deltaY = gradient
        tail.backward(deltaY = gradient);//assign gradient to tail.gradient
        visited.add(tail);
        
        for(Unit start : head.nexts) backward(start);//head.next = graph.start
        visited.clear(); 
       
        //head computes the gradient for input, based on sub arcs of sub graph of Module
        deltaX = head.collectGradientFromNext();
        
        //final process---------------------------------------------------------
        if(deltaX != null) {//collect gradient for deltaX
            for(int i=0; i<X.length; i++) 
                if(X[i].need_grad()) X[i].grad(deltaX[i]);
        }
        
        return deltaX;
    }

    @Override 
    public synchronized Tensor gradient(int index) {
        return head.gradient(index); 
    }
    //</editor-fold>
}
