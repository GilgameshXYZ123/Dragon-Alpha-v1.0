/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.concurrent;

import java.util.logging.Level;
import java.util.logging.Logger;
import z.util.lang.Lang;
import z.util.math.ExRandom;

/**
 * singleWait(L,t,need):
  signal:
  *     if(singleWait>0) singleLock.notify();
  *     if(multiWait>0&&value>=2) multiWait.notifyAll();
 * @author dell
 */
public class Semaphore
{
    //columns------------------------------------------------------------------
    String name;
    volatile int value;
    volatile int singleWait;
    
    private final byte[] singleLock=new byte[0];
    private final byte[] multiLock=new byte[0];
    
    public Semaphore(int value, String name)
    {
        this.value=value;
        this.name=name;
        this.singleWait=0;
    }
    public void P(int n)
    {
        singleWait++;
        synchronized(this.multiLock)
        {
            try
            {
                while(value<n) this.multiLock.wait();
            }
            catch(InterruptedException e)
            {throw new RuntimeException(e);}
        }
        value-=n;
        singleWait--;
    }
    public synchronized void P() 
    {
        singleWait++;
        synchronized(this.singleLock)
        {
            try
            {
                while(value<1) this.singleLock.wait();
            }
            catch(InterruptedException e)
            {throw new RuntimeException(e);}
        }
        value--;
        singleWait--;
    }
    public synchronized void V()
    {
        value++;
        this.notify();
    }
    
    static int in=0;
    static int out=0;
    static int[] buffer=new int[5];
    
    public static void test1()
    {
        Semaphore full=new Semaphore(0,"1");
        Semaphore empty=new Semaphore(5,"2");
        Semaphore mutex=new Semaphore(1,"3");
        
        ExRandom ex=Lang.exRandom();
        
        Thread producer=new Thread(()->{
            while(true)
            {
                empty.P();
                mutex.P();
                int v=ex.nextInt(500);
                System.out.println("producer:index="+in+"  value="+v);
                buffer[in]=v;
                in=(in+1)%5;
                mutex.V();
                full.V();
                
                
                 try {
                    Thread.sleep(100);
                } catch (InterruptedException ex1) {
                    Logger.getLogger(Semaphore.class.getName()).log(Level.SEVERE, null, ex1);
                }
            }
        });
        Thread consumer=new Thread(()->{
            while(true)
            {
                full.P();
                mutex.P();
                int v=buffer[out];
                out=(out+1)%5;
                System.out.println("consumer:index="+out+"  value="+v);
                mutex.V();
                empty.V();
                
                
                try {
                    Thread.sleep(500);
                } catch (InterruptedException ex1) {
                    Logger.getLogger(Semaphore.class.getName()).log(Level.SEVERE, null, ex1);
                }
            }
        });
       
        consumer.start();
         producer.start();
    }
}
