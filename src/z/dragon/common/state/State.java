/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.common.state;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.concurrent.Future;
import z.dragon.common.state.State.StateValue;

public class State extends HashMap<String, StateValue>
{
    private static final long serialVersionUID = 1L;
 
    //<editor-fold defaultstate="collapsed" desc="interface: StateValue">
    public static int MAX_STRING_LINE_SIZE = Integer.MAX_VALUE >> 4;
    
    public static interface StateValue { 
        public Object value();
        public Class<?> type();
        public ArrayList<String> toStringLines(); 
    }
    //</editor-fold>
    
    public State(int init_capacity) { super(init_capacity); }
    public State() { super(); }

    public static void set(StateValue value, String msg, boolean partial, Runnable opt) {
        if(value == null && !partial) throw new RuntimeException(msg);
        if(value != null) try { opt.run(); } 
        catch(Exception e) { throw new RuntimeException(msg, e); }
    }    
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public void append(StringBuilder sb) {
        sb.append(getClass().getSimpleName()).append(" {");
        this.forEach((String key, Object value) -> {
            sb.append("\n[key, value] = [")
                    .append(key).append(", ")
                    .append(value).append(']');
        });
        sb.append("}");
    }
    
    @Override
    public String toString()  {
        StringBuilder sb = new StringBuilder(256);
        this.append(sb);
        return sb.toString();
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="interface: Stateful">
    public static interface Stateful {
        public void state(State dic);
        
        default State state() {  State dic = new State(4); state(dic);  return dic; }
        
        public void update_state(State dic, boolean partial) throws Exception;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="interface: StatefulTransformer">
    public static interface StatefulTransformer { 
        public Stateful transform(Stateful st) throws Exception;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="interface: AsyncStateUpdatable">
    public static interface AsyncStateUpdate 
    {
        public boolean update_state_sync();
        
        public <T extends AsyncStateUpdate>  T update_state_sync(boolean flag);
        
        public void update_state(State dic, boolean partial, List<Future<?>> fts);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="interface: StateReader">
    public static interface StateReader { 
        public State read() throws Exception; 
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="interface: StateWriter">
    public static interface StateWriter { 
        public void write(State dic) throws Exception;
    }
    //</editor-fold>
    
    public static FloatArrayValue floats(float... vals) {
        return new FloatArrayValue(vals);
    }
}
