/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.math;

import java.io.IOException;
import java.io.OutputStream;
import static java.lang.Math.PI;
import static java.lang.Math.exp;
import static java.lang.Math.sqrt;
import z.util.lang.exception.IAE;
import z.util.lang.annotation.Passed;

/**
 *
 * @author dell
 */
public final class ExMath 
{         
    private ExMath() {}
    
    //<editor-fold defaultstate="collapsed" desc="static class Function">
    public static interface Function
    {
        public double value(double x);
    }
    public static interface HashCoder
    {   
        public int hashCode(Object key);
    }
    public static final HashCoder DEF_HASHCODER=new HashCoder() {
        @Override
        public int hashCode(Object key) 
        {
            if(key==null) return 0;
            int h=key.hashCode();
            return ((h=h^(h>>>16))>0? h:-h);
        }
    };
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Function">
    public static int clip(int v, int min, int max) {
        if(v < min) v = min;
        if(v > max) v = max;
        return v;
    }
    
    @Passed
    public static boolean isPerfectSqurae(int v) {
        int r=v;
        while(r*r > v) r = (r + v/r) >> 1;// Xn+1 = Xn - f(Xn)/f'(Xn)
        return r*r == v;
    }
    
    @Passed
    public static boolean isPrime(int v) {
        if(v<0) v=-v;
        if(v<=1) return false;
        for(int i = 2,len = (int)sqrt(v); i <= len; i++)
            if(v % i == 0) return false;
        return true;
    }
    
    /**
     * <pre>
     * get the value of a specific index {@code n} of Fibonacci Sequence,
     * while the index of sequence is from 0;
     * 1,1,2,3,5,8,13,21,44,65......
     * </pre>
     * @param n
     * @return 
     */
    @Passed
    public static long fibonacci(int n) {
        if(n<0) throw new IAE("The index of the sequence mustn't be negative"); 
        if(n<=1) return 1;
        int a=1,b=1,c;
        for(int i=2;i<=n;i++) {
            c = a + b; a = b; b = c;
        }
        return b;
    }
    
    public static final double LOG2 = Math.log(2);
    public static double log2(double x){return Math.log(x)/LOG2;}
    public static void log2Table(OutputStream out, int x)
    {
        try
        {
            for(int i=3;i<=x;i++)
            {
                String line="log2("+i+") = "+log2(i)+'\n';
                out.write(line.getBytes());
            }
        }
        catch(IOException e) {e.printStackTrace();throw new RuntimeException(e);}
    }
    public static void log2Table(int x) {log2Table(System.out, x);}
            
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Gaussian-Distribute">
    //<editor-fold defaultstate="collapsed" desc="class NDdensity implements Fuction">
    public static final class GaussianDensity implements Function
    {
        private final double avg;
        private final double div;
        private final double k;
        
        public GaussianDensity()
        {
            this(0,1);
        }
        public GaussianDensity(double avg, double stddev)
        {
            this.avg=avg;
            this.div=-2*stddev*stddev;
            this.k=1/(sqrt(2*PI)*stddev);
        }
        @Override
        public double value(double x)
        {
            x-=avg;
            return k*exp(x*x/div);
        }
    }
    private static final GaussianDensity N_1_0=new GaussianDensity();
    //</editor-fold>
    /**
     * <pre>
     * This function is used to approximate the standard normal distribution,
     * with the min=0 and variance=1.
     * we regard the the densitiy function of Normal Distribute(NF) as f(x):
     * compute:
     * (1)I1=the integral of f(x) in(-infinite, x);
     * (2)I2=the integral of f(x) in (-x,x);
     * Theoritically, I1=(I2+1)/2, but practically, there exists samll difference
     * between I1 and I2, so we return (I1+I2)/2 to improve the presition.
     * </pre>
     * @param x
     * @return 
     */
    @Passed
    public static double gaussianDistribute(double x)
    {
        if(x<1e-9&&x>-1e-9) return 0.5;
        double r1=(ExMath.integral(-x, x, N_1_0, 1e-7)+1)/2;
        double r2=ExMath.integralFromNF(x, N_1_0);
        return (r1+r2)/2;
    }
    /**
     * <pre>
     * This function is used to approximate the standard normal distribution,
     * with the min=0 and variance=1.
     * we regard the the densitiy function of Normal Distribute(NF) as f(x):
     * compute the integral for f(x) in(start,end);
     * </pre>
     * @param start
     * @param end
     * @return 
     */
    @Passed
    public static double gaussianDistribute(double start, double end)
    {
        return ExMath.integral(start, end, N_1_0, 1e-7);
    }
    /**
     * By equivalent transformation, we use standard NF to get the value
     * of NF with a specific avg and stddev.
     * @see #gaussianDistribute(double) 
     * @param x
     * @param avg
     * @param stddev
     * @return 
     */
    @Passed
    public static double gaussianDistribute(double x, double avg, double stddev)
    {
        return ExMath.gaussianDistribute((x-avg)/stddev)/stddev;
    }
    /**
     * By equivalent transformation, we use standard NF to get the value
     * of NF with a specific avg and stddev.
     * @see #gaussianDistribute(double, double) 
     * @param start
     * @param end
     * @param avg
     * @param stddev
     * @return 
     */
    @Passed
    public static double gaussianDistribute(double start, double end, double avg, double stddev)
    {
        return ExMath.gaussianDistribute((start-avg)/stddev, (end-avg)/stddev)/stddev;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Narrow-Function">
    //<editor-fold defaultstate="collapsed" desc="Inner-Code">
    private static final double[] EXP_NARROW_K=ExMath.createExpNarrowK();
    private static final double[] SINH_NARROW_K=ExMath.createSinhNarrowK();
    private static final double[] COSH_NARROW_K=ExMath.createCoshNarrowK();
    private static final double[] SIGMOID_NARROW_K=ExMath.createSigmoidNarrowK();
    
    @Passed
    private static double[] createExpNarrowK()//exp(x)
    {
        double k[]=new double[12],c=1;
        for(int i=0,j=0;i<k.length;i++) {c/=++j;k[i]=c;}
        return k;
    }
    @Passed
    private static double[] createSinhNarrowK()//(exp(x)-exp(-x))/2
    {
        double k[]=new double[12],c=1;
        k[0]=c;
        for(int i=1,j=1;i<k.length;i++) {c/=++j*++j;k[i]=c;}
        return k;
    }
    @Passed
    private static double[] createCoshNarrowK()
    {
        double k[]=new double[12],c=0.5;
        k[0]=c;
        for(int i=1,j=2;i<k.length;i++) {c/=++j*++j;k[i]=c;}
        return k;
    }
    @Passed
    private static double[] createLnNarrowK()
    {
        double k[]=new double[30],c=1;
        k[0]=c;
        for(int i=1,j=0;i<k.length;i++) {c/=++j*++j;k[i]=c;}
        return k;
    }
    private static double[] createSigmoidNarrowK()
    {
       double k[]=new double[12];
       return k;
    }
    //</editor-fold>
    @Passed
    public static double exp_narrow(double x)
    {
        double y=1,cur=x;
        for(int i=0;i<EXP_NARROW_K.length;i++)
            {y+=cur*EXP_NARROW_K[i];cur*=x;}
        return y;
    }
    @Passed
    public static double sinh_narrow(double x)
    {
        double y=0,cur=x;
        x*=x;
        for(int i=0;i<SINH_NARROW_K.length;i++)
            { y+=cur*SINH_NARROW_K[i]; cur*=x; }
        return y;
    }
    @Passed
    public static double cosh_narrow(double x)
    {
        double y=1,cur=x*=x;
        for(int i=0;i<COSH_NARROW_K.length;i++)
            {y+=cur*COSH_NARROW_K[i];cur*=x;}
        return y;
    }
    public static double sigmoid_narrow(double x)
    {
        throw new RuntimeException();
    }
    public static double tanh_narrow(double x)
    {
        throw new RuntimeException();
    }
    public static double softplus_narrow(double x)
    {
        throw new RuntimeException();
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Integral-Function">
    private static final int MIN_INTEGRAL_P=500000;
    private static final int MAX_INTEGRAL_P=1000000;
    
    public static final double DEF_INTEGRAL_STEP=1e-7;
    public static final double INTEGRAL_POSITIVE_ZERO=1e-19;
    public static final double INEGRAL_POSITIVE_ZERO_RECIPROCAL=1e4;
    //<editor-fold defaultstate="collapsed" desc="Inner-Code:Compute-Integral">
    /**
     * <pre>
     * (1)Compute the integral for {@ Function func} on interval from start
     * to end; 
     * (2)This method use an approximation Algorithm, the precision
     * is based on step, while the number of steps is p;
     * (3)This Method is an lower interface, kindly use, other higher 
     * interface to avoid use this function directly
     * (4)to avoid spend too mush resource, the number of steps is limited
     * on interval [50000,100000];
     * </pre>
     * @param start the Lower bound of this intergral
     * @param step
     * @param p
     * @param func
     * @return 
     */
    @Passed
    private static double avgIntegral(double start, double step, int p, Function func)
    {
        double sum=func.value(start)/2;
        start+=step;
        for(int i=1;i<p;i++,start+=step)
            sum+=func.value(start);
        sum-=func.value(start)/2;
        return sum*step;
    }
    /**
     * <pre>
     * The integral from start to end {@code Function func}.
     * (1)check the start and the end of integral; sometimes {@code start < end}, 
     * in this case exchange(start, end), and set flag=-1;
     * (2)compute the step and step_number based on the interval and dx
     * (3)for integral from negative infinite or to positive infinite,
     * you need do some transformation on the function and boundary, then 
     * you can use this method, but somtimes there may exist an larger 
     * deviation between result and practial value;
     * </pre>
     * @param start the lower bound of intergral
     * @param end the higher bound of integral
     * @param func
     * @param dx the expected step
     * @throws IAE {@code  if div<dx*100}, the step number is too few for integral
     * @throws IAE {@code if step>1e-6*dx}, the step is too big for intergral
     * @return 
     */
    @Passed
    public static double integral(double start, double end, Function func, double dx)
    {
        double step=dx,div=end-start,flag=1;
        if(div<0) {flag=start;start=end;end=flag;div*=-1;flag=-1;}
        if(div<=dx*100) throw new IAE("Need more segment for dx to integral[div<dx*100]");
        
        int p=(int) ((end-start)/dx);
        if(p<MIN_INTEGRAL_P) {p=MIN_INTEGRAL_P;step=div/MIN_INTEGRAL_P;}
        else if(p>MAX_INTEGRAL_P) 
        {
            p=MAX_INTEGRAL_P;step=div/MAX_INTEGRAL_P;
            if(step>dx*1e6) throw new IAE("Need smaller step to integral[step>dx*1e6]");
        }
        return flag*ExMath.avgIntegral(start, step, p, func);
    }
    /**
     * <pre>
 The integral from +infinite to base for Function func.
     * Divide the interval to two parts: [greater than 0], [less than 0],
     * as base is not an infinite value;
     * (1)for integral (0, +infinite) do some transimition and compute
     *  =>fx->1/(x*x)*f(1/x)
     * (2)for integral [-base, 0), just compute the integral
     * must avoid the integral at 0, so in effect the integral algorithm
     * is like this:
     * {@code if(base>0) return ExMath.integral(1e-19, end, funcs, dx);
     *   else return ExMath.integral(base, 0, func, dx)
     *           +ExMath.integral(1e-19, 1e3, func, dx);}
     * </pre>
     * @param base
     * @param func
     * @param dx
     * @return 
     */
    @Passed
    public static double integralToPF(double base, Function func,double dx)
    {
        if(base==0) throw new IAE("bound can not equal to zero");
        double end=1/base;
        Function funcs=(double x) -> (x=1/x)*func.value(x)*x;
        if(base>0) return ExMath.integral(INTEGRAL_POSITIVE_ZERO, end, funcs, dx);
        else return ExMath.integral(base, 0, func, dx)
                +ExMath.integral(INTEGRAL_POSITIVE_ZERO, INEGRAL_POSITIVE_ZERO_RECIPROCAL, funcs, dx);
    }
    /**
     * <pre>
     * The integral from -infinite to bound for {@code Function func}.
     * (1)do such a transimition:
     * {@code ExMath.integral(-infinite, bound, function(x))
     *          =ExMath.integral(-bound, +infinite, function(-x))},as:
     * {@code ExMath.integralFromNF(bound, funcion(x))
     *          =ExMath.integralFromPF(-bound, function(-x));}
     * 
     * </pre>
     * @param bound
     * @param func
     * @param dx
     * @return 
     */
    @Passed
    public static double integralFromNF(double bound, Function func, double dx)
    {
        if(bound==0) throw new IAE("bound can not equal to zero");
        double end=-1/bound;
        Function funcs=(double x) -> (x=1/-x)*func.value(x)*x;
        if(bound<0) return ExMath.integral(INTEGRAL_POSITIVE_ZERO, end, funcs, dx);
        else return ExMath.integral(-bound, 0, func, dx)
                +ExMath.integral(INTEGRAL_POSITIVE_ZERO, INEGRAL_POSITIVE_ZERO_RECIPROCAL, funcs, dx);
    }
    //</editor-fold>
    public static double integral(double start, double end, Function func)
    {
        return ExMath.integral(start, end, func, DEF_INTEGRAL_STEP);
    }
    public static double integralToPF(double start, Function func)
    {
        return ExMath.integralToPF(start, func, DEF_INTEGRAL_STEP);
    }
    public static double integralFromNF(double bound, Function func)
    {
        return ExMath.integralFromNF(bound, func, DEF_INTEGRAL_STEP);
    }
    //<editor-fold defaultstate="collapsed" desc="Inner-Code:Find-Integral-Boundary">
    @Passed
    private static double findAvgIntegralHB(double start, double step, double expect, Function func)
    {
        expect/=step;
        double sum=func.value(start)/2, lastdiv=expect,nextdiv=expect-sum;
        if(nextdiv<0) nextdiv=-nextdiv;
        while(lastdiv>=nextdiv)
        {
            lastdiv=nextdiv;
            sum+=func.value(start+=step);
            nextdiv=expect-sum;
            if(nextdiv<0) nextdiv=-nextdiv;
        }
        sum-=func.value(step)/2;
        nextdiv=expect-sum;
        if(nextdiv<0) nextdiv=-nextdiv;
        return (lastdiv<nextdiv? start:start+step);
    }
    public static double findIntegralHB(double start, double expect, double precision, Function func)
    { 
        if(precision<0)
            throw new IAE("The precision used to find the highe boundary of this Integral must positive");
        double ex=(expect>=0? expect:-expect);
        if(ex<precision*10) 
            throw new IAE("the precision is too big to find the highe boundary of this Integral must positive");
        return ExMath.findAvgIntegralHB(start, precision, expect, func);
    }
    //</editor-fold>
    
    //</editor-fold>
}
