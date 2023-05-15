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
public class BinarySemaphore
{
    volatile boolean value=false;
    
    public void P()
    {
        try
        {
            synchronized(this)
            {
                while(value) this.wait();
                value=true;
            }
        }
        catch(InterruptedException e)
        {throw new RuntimeException(e);}
    }
    public void P(long waitTime)
    {
        try
        {
            synchronized(this)
            {
                while(value) this.wait(waitTime);
                value=true;
            }
        }
        catch(InterruptedException e)
        {throw new RuntimeException(e);}
    }
    public void sleep(long sleepTime)
    {
        try
        {
            synchronized(this)
            {
                Thread.sleep(sleepTime);
            }
        }
        catch(Exception e)
        {throw new RuntimeException(e);};
    }
    public synchronized void V()
    {
        if(value) 
        {
            this.notify();
            value=false;
        }
    }
}
