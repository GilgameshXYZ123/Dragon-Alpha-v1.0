/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.lang;

import z.util.factory.Meta;
import z.util.math.vector.Vector;

/**
 *
 * @author dell
 */
public class SimpleTimer 
{
    public static SimpleTimer instance() {
        return new SimpleTimer();
    }
    
    public static SimpleTimer clock() {
        return new SimpleTimer().record();
    }
    
    long last;
    long cur;
    
    public SimpleTimer record() {
        last = cur;
        cur = System.currentTimeMillis();
        return this;
    }
    
    public long timeStampDifMills() {
        return cur - last;
    }
    
    @Override
    public String toString() { 
        return "Mills:"+this.timeStampDifMills(); 
    }
    
    //extensive-----------------------------------------------------------------
    private static final String TEST_AVG_TIME = "test.avg.time";
    private static final String TEST_STDDEV_TIME = "test.stddev.time";
    private static final String TEST_EACH_TIME = "test.each.time";
    
    public Meta test(int times, Runnable run)
    {
        long[] t=new long[times];
        for(int i=0;i<times;i++)
        {
            this.record();
            run.run();
            this.record();
            t[i]=this.timeStampDifMills();
        }
        Meta mt=new Meta();
        mt.put(TEST_EACH_TIME, t);
        mt.put(TEST_AVG_TIME, Vector.average(t));
        mt.put(TEST_STDDEV_TIME, Vector.stddev(t));
        return mt;
    }
}
