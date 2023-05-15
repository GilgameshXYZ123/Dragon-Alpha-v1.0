/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.concurrent;

import java.io.Serializable;
import z.util.lang.annotation.Passed;

/**
 * locker occupied the lock of this instance,then others can't own the lock of
 * this instace so whenever the blocking deadline, the this.wait()[in locker
 * thread] is called, when need to block, the this.notify is called, to awake
 * the locker thread
 *
 * @author dell
 * @param <T>
 */
public class FugitiveValue<T> implements Serializable
{
    private class Locker implements Runnable
    {
        @Override
        public void run() 
        {
            while(running)
            try
            {
                synchronized(this)
                {
                    blocked=true;
                    this.wait();
                    System.out.println(value);
                    Thread.sleep(blockTime);
                    value=(value==A? B:A);
                     System.out.println(value);
                }
            }
            catch(InterruptedException e)
            {throw new RuntimeException(e);}
        }
    }
    
    protected boolean blocked=false;
    
    protected volatile T value;
    protected final T A;
    protected final T B;
    
    protected long blockTime;
    private volatile boolean running=false;
    private Locker locker;
    private Thread ext;
    
    public FugitiveValue(T A, T B, long blockTime)
    {
        this.A=A;
        this.B=B;
        this.value=A;
        this.blockTime=blockTime;
    }
    //<editor-fold defaultstate="collapsed" desc="Basic-Function">
    public boolean isBlocked()
    {
        return blocked;
    }
    public T getValue()
    {
        return value;
    }
    public T getA()
    {
        return A;
    }
    public T getB()
    {
        return B;
    }
    public long getBlockTime()
    {
        return blockTime;
    }
    public synchronized void setBlockTime(long blockTime)
    {
        this.blockTime=blockTime;
    }
    public synchronized T get()
    {
        return this.value;
    }
    public void append(StringBuilder sb)
    {
        sb.append(this.getClass()).append('{');
        sb.append("\n\tA = ").append(A);
        sb.append("\n\tB = ").append(B);
        sb.append("\n\tvalue = ").append(value==A? "A":"B");
        sb.append("\n\tblocked = ").append(blocked);
    }
    @Override
    public String toString()
    {
        StringBuilder sb=new StringBuilder();
        this.append(sb);
        return sb.toString();
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Inner-Operator">
    private void set(T val)
    {
        synchronized(locker)
        {
            this.value=val;
            blocked=false;
            locker.notify();
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Operator">
    @Passed
    public synchronized void open()
    {
        if(running) throw new RuntimeException("The BlockedValue has been opened");
        
        ext=new Thread(locker=new Locker());
        ext.setPriority(Thread.MAX_PRIORITY);
        
        running=true;
        blocked=true;
        ext.start();
    }
    public void setA() {set(A);}
    public void setB() {set(B);}
    public void change() {set(value==A? B:A);};
    @Passed
    public synchronized void close()
    {
        if(!running) throw new RuntimeException("The BlockedValue has been stopped. or hasn't been opend");
        
        running=false;
        blocked=false;
        this.notify();
        locker=null;
        ext=null;
    }
    //</editor-fold>
}
