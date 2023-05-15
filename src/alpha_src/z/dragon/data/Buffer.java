/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.data;

import java.util.concurrent.Callable;

/**
 *
 * @author Gilgamesh
 * @param <T>
 */
public class Buffer<T> 
{
    private final byte[] lock = new byte[0];
    private final Thread worker;
    private volatile boolean start = false;
    private volatile boolean running = true;
    
    protected final Callable<T> getter;
    protected T data;
    
    //<editor-fold defaultstate="collapsed" desc="class Worker">
    class Worker implements Runnable
    {
        @Override
        public void run() {
            while(running) 
            {
                synchronized(lock) {//load the next data, and notify all consulers in lock.waitList
                    try{ if(data == null) data = getter.call(); }
                    catch(Exception e) { throw new RuntimeException(e); }
                    finally { lock.notifyAll(); }
                }
                
                try{ synchronized(getter) { getter.wait(); } } //worker is blocked
                catch (InterruptedException e) { }
            }
        }
    }
    //</editor-fold>
    public Buffer(Callable<T> getter)  {
        this.getter = getter;
        worker = new Thread(new Worker());
        worker.setDaemon(true);
    }
    
    public boolean started() { return start; }
    public boolean alive() { return running; }
    
    //<editor-fold defaultstate="collapsed" desc="operators">
    public T get() {
        T result;
        synchronized(lock) {
            if(!start) { worker.start(); start = true; }
            while(data == null)//add current thread to lock.waitList
            try { lock.wait(); } catch(InterruptedException e) {}
            result = data; data = null; 
        }
        synchronized(getter) { getter.notify(); } //wake up worker to get next value
        return result;
    }
    
    public void close() {
        synchronized(getter) {
            getter.notify(); 
            if(!worker.isInterrupted() && worker.isAlive()) worker.interrupt();
        }
        synchronized(lock) { running = false; lock.notifyAll();}
    }
    //</editor-fold>
}