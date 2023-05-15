/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.concurrent;

/**
 *
 * @author dell
 */
public class Lock
{
    volatile int value;
    final int initValue;
    
    public Lock(int value)
    {
        this.initValue=value;
        this.value=value;
    }
    public void lock()
    {
        try
        {
            synchronized(this)
            {
                while(value>0) this.wait();
                this.value=initValue;
            }
        }
        catch(InterruptedException e)
        {throw new RuntimeException(e);}
    }
    public synchronized void unlock()
    {
        value--;
        if(value==0) this.notify();
    }
    public synchronized void unlock(int n)
    {
        value-=n;
        if(value==0) this.notify();
    }
}
