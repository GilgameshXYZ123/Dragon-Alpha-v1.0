/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.engine;

public abstract class Counter 
{
    public Counter(int count) {
        if (count <= 0) throw new IllegalArgumentException("count must be a positive number");
        this.count = count;
    }
    
    //<editor-fold defaultstate="collapsed" desc="member params & Basic-Functions">
    private int count;
    private boolean runned = false;

    public int count() { return count;}
    public boolean runned() { return runned; }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="running-area">
    public abstract void run();
    public synchronized void countDown() {
        count--;
        if (count <= 0 && !runned) {
            run();
            runned = true;
        }
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="class: CountGc">
    public static class CountGc extends Counter  
    { 
        private final Tensor ts;
        
        public CountGc(int count, Tensor ts) {
            super(count);
            this.ts = ts;
        }
        
        public Tensor tensor() { return ts; }
        
        @Override 
        public void run() { 
            if(ts != null) ts.delete(); 
        }
    }
    //</editor-fold>
    public static CountGc countGc(int count, Tensor ts) {
        return new CountGc(count, ts);
    }
}
