/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.math.vector;

import static java.lang.Math.cos;
import static java.lang.Math.sin;
import java.util.Comparator;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import z.util.concurrent.BinarySemaphore;
import z.util.concurrent.Lock;
import z.util.ds.linear.ZArrayList;
import z.util.lang.Lang;
import z.util.lang.SimpleTimer;
import z.util.lang.annotation.Passed;
import z.util.lang.exception.IAE;
import z.util.math.Sort;

/**
 *
 * @author dell
 */
@Passed("Need optimization")
public final class ExMatrix
{  
    //<editor-fold defaultstate="collapsed" desc="Multiply-Test">
    public static void test1() 
    {
        try
        {
//            for(int x=100;x<130;x++)
//            for(int y=30;y<40;y++)
            for(int z=10;z<11;z++)
            {
                int kx=8, ky=8, kz=8;
                double[][] a=Matrix.randomDoubleMatrix(kx, ky, 10);
                double[][] b=Matrix.randomDoubleMatrix(ky, kz, 10);
                double[][] c1=new double[kx][kz];
                double[][] c2=new double[kx][kz];
//                ExMatrix.multiplyM(c1, a, b, exec);
                Matrix.multiply(c2, a, b);
                double l=Matrix.samePercent(c2, c1);
                if(l!=1) System.out.println(l);
                Lang.zprintln(c1, c2);
            }
        }
        catch(Exception e)
        {
            e.printStackTrace();
        }
    }
    public static void test2() throws InterruptedException
    {
        SimpleTimer st=new SimpleTimer();
        for(int i=10;i<100;i++)
        {
            int x=i,y=i,z=i;
            double[][] a=Matrix.randomDoubleMatrix(x, y, 10);
            double[][] b=Matrix.randomDoubleMatrix(y, z, 10);
            double[][] c=new double[x][z];
            int len=30;
            double sum1=0, sum2=0;
            for(int k=0;k<len;k++)
            {
                st.record();
                Matrix.multiply(c, a, b);//M=66
                st.record();
                sum1+=st.timeStampDifMills();
            }
            sum1/=len;
            for(int k=0;k<len;k++)
            {
                st.record();
                ExMatrix.multiply(c, a, b, new double[4][b.length]);//M=66
                st.record();
                sum2+=st.timeStampDifMills();
            }
            sum2/=len;
            System.out.println("turn="+i+"\tNormal="+sum1+"\tEx="+sum2);
        }
    }
    public static void test3()
    {
        SimpleTimer st=new SimpleTimer();
        int x=1000,y=1000,z=1000;
        double[][] a=Matrix.randomDoubleMatrix(x, y, 10);
        double[][] b=Matrix.randomDoubleMatrix(y, z, 10);
        double[][] c=new double[x][z];
        
        ExMatrix.multiplyM(c, a, b, exec);//141.5 14
        int len=40;
        double sum1=0;
        for(int k=0;k<len;k++)
        {
            st.record();
//             Matrix.multiply(c, a, b);
            ExMatrix.multiplyM(c, a, b, exec);
            st.record();
            System.out.println(k+" "+st);
            sum1+=st.timeStampDifMills();
        }
        System.out.println(sum1/len);
    }
    //</editor-fold> 
   
    public static void main(String[] args)
    {
       test3();
       ExMatrix.shutDown();
    }
    //<editor-fold defaultstate="collapsed" desc="JNI-Kernal-X1:Basic">
    static
    {
        System.load("D:\\virtual disc Z-Gilgamesh\\Gilgamesh java2\\ZUTIL-STD-1.1\\src\\z\\util\\math\\vector\\mk44n.dll");
    }
    public static void multiply44MU(double[][] c, double[][] a, double[][] b, ExecutorService exec)
    {
        Lock ss=new Lock((a.length/4)*(b[0].length/4));
        for(int i,j=0,k;j<b[0].length-3;j+=4)
        {
            double[][] sb=new double[4][b.length];
            for(k=0;k<b.length;k++)//get 4 column from b
            {
                sb[0][k]=b[k][j];
                sb[1][k]=b[k][j+1];
                sb[2][k]=b[k][j+2];
                sb[3][k]=b[k][j+3];
            }
            for(i=0;i<a.length;i+=4)
            {
                int row=i, col=j;
                exec.submit(() -> {
                    mk44N(c, a, sb, row, col);
                    ss.unlock();
                });
            }
        }
        ss.lock();
    }
    private static void mk44N(double[][] c, double[][] a, double[][] sb, int i, int j)
    { 
        mk44N(c[0], c[1], c[2], c[3],
              a[0], a[1], a[2], a[3],
              sb[0], sb[1], sb[2], sb[3],
              i, j, sb.length);
    }
    private native static void mk44N(
            double[] c0, double[] c1, double[] c2, double[] c3,
            double[] a0, double[] a1, double[] a2, double[] a3, 
            double[] b0,double[] b1, double[] b2, double[] b3,
            int i, int j, int width);
    //</editor-fold>
    
    private ExMatrix() {}
    public static final int OPT_MULTIPLY_THRESHOLD=10000;
    public static final int EX_MULTIPLY_THRESHOLD=800000;
    
    //<editor-fold defaultstate="collapsed" desc="Multiply-Function">
    //<editor-fold defaultstate="collapsed" desc="Multiply-Kernel">
    @Passed
    private static void mk11(double[][] c, double[][] a, double[][] sb, int i, int j)
    {
        double t00=0;
        double[] a0=a[i],
                 b0=sb[0];
        /**
         * a--> b
         * t0  |
         */
        for(int k=0;k<b0.length;k++) t00+=a0[k]*b0[k];
        c[i][j]=t00;
    }
    @Passed
    private static void mk12(double[][] c, double[][] a, double[][] sb, int i, int j)
    {
        double t00=0, t01=0;
        double[] a0=a[i],
                 b0=sb[0],  b1=sb[1];
        double va0, vb0, vb1;
        /**
         * a-->   b
         * t0  t1 |
         * t4  t5 |
         */
        for(int k=0;k<b0.length;k++)
        {
            va0=a0[k];
            vb0=b0[k];  vb1=b1[k];
            
            t00+=va0*vb0;   t01+=va0*vb1;
        }
        c[i][j]=t00;   c[i][j+1]=t01;
    }
    @Passed
    private static void mk22(double[][] c, double[][] a, double[][] sb, int i, int j)
    {
        double t00=0,  t01=0,
               t11=0,  t12=0;
        double[] a0=a[i],   a1=a[i+1],
                 b0=sb[0],  b1=sb[1];
        double va0, va1, vb0, vb1;
        /**
         * a-->   b
         * t0  t1 |
         * t4  t5 |
         */
        for(int k=0;k<b0.length;k++)
        {
            va0=a0[k];  va1=a1[k];
            vb0=b0[k];  vb1=b1[k];
            
            t00+=va0*vb0;   t01+=va0*vb1;
            t11+=va1*vb0;   t12+=va1*vb1;  
        }
        c[i][j]=t00;   c[i++][j+1]=t01;
        c[i][j]=t11;   c[i][j+1]=t12;
    }
    @Passed
    private static void mk14(double[][] c, double[][] a, double[][] sb, int i, int j)
    {
        double t00=0,  t01=0,  t02=0,  t03=0;
        double[] a0=a[i],   
                 b0=sb[0],  b1=sb[1],   b2=sb[2],   b3=sb[3];
        double va0, va1, va2, va3, vb0, vb1, vb2, vb3;
        /**
         * a-->            b
         * t0  t1  t2  t3  |
         */
        for(int k=0, width=sb[0].length;k<width;k++)
        {
            va0=a0[k];
            vb0=b0[k];  vb1=b1[k];  vb2=b2[k];  vb3=b3[k];
            
            t00+=va0*vb0;   t01+=va0*vb1;   t02+=va0*vb2;   t03+=va0*vb3;  
        }
        c[i][j]=t00;   c[i][j+1]=t01;   c[i][j+2]=t02;   c[i][j+3]=t03;
    }
    @Passed
    private static void mk24(double[][] c, double[][] a, double[][] sb, int i, int j)
    {
        double t00=0,  t01=0,  t02=0,  t03=0, 
               t10=0,  t11=0,  t12=0,  t13=0;
        double[] a0=a[i],   a1=a[i+1],
                 b0=sb[0],  b1=sb[1],   b2=sb[2],   b3=sb[3];
        double va0, va1, va2, va3, vb0, vb1, vb2, vb3;
        /**
         * a-->            b
         * t0  t1  t2  t3  |
         * t4  t5  t6  t7  |
         */
        for(int k=0, width=sb[0].length;k<width;k++)
        {
            va0=a0[k];  va1=a1[k];
            vb0=b0[k];  vb1=b1[k];  vb2=b2[k];  vb3=b3[k];
            
            t00+=va0*vb0;   t01+=va0*vb1;   t02+=va0*vb2;   t03+=va0*vb3;  
            t10+=va1*vb0;   t11+=va1*vb1;   t12+=va1*vb2;   t13+=va1*vb3;
        }
        c[i][j]=t00;   c[i][j+1]=t01;   c[i][j+2]=t02;   c[i++][j+3]=t03;
        c[i][j]=t10;   c[i][j+1]=t11;   c[i][j+2]=t12;   c[i][j+3]=t13;
    }
    @Passed
    private static void mk34(double[][] c, double[][] a, double[][] sb, int i, int j)
    {
        double t00=0,  t01=0,  t02=0,  t03=0, 
               t10=0,  t11=0,  t12=0,  t13=0,
               t20=0,  t21=0,  t22=0,  t23=0;
        double[] a0=a[i],   a1=a[i+1],  a2=a[i+2],
                 b0=sb[0],  b1=sb[1],   b2=sb[2],   b3=sb[3];
        double va0, va1, va2, va3, vb0, vb1, vb2, vb3;
        /**
         * a-->            b
         * t0  t1  t2  t3  |
         * t4  t5  t6  t7  |
         * t8  t8  t9  t11 
         */
        for(int k=0, width=sb[0].length;k<width;k++)
        {
            va0=a0[k];  va1=a1[k];  va2=a2[k];
            vb0=b0[k];  vb1=b1[k];  vb2=b2[k];  vb3=b3[k];
            
            t00+=va0*vb0;   t01+=va0*vb1;   t02+=va0*vb2;   t03+=va0*vb3;  
            t10+=va1*vb0;   t11+=va1*vb1;   t12+=va1*vb2;   t13+=va1*vb3;
            t20+=va2*vb0;   t21+=va2*vb1;   t22+=va2*vb2;   t23+=va2*vb3;
        }
        c[i][j]=t00;   c[i][j+1]=t01;   c[i][j+2]=t02;   c[i++][j+3]=t03;
        c[i][j]=t10;   c[i][j+1]=t11;   c[i][j+2]=t12;   c[i++][j+3]=t13;
        c[i][j]=t20;   c[i][j+1]=t21;   c[i][j+2]=t22;   c[i][j+3]=t23;
    }
    /**
     * the size for the input Matrix a,b and c must be multiple of 4.
     * the input Matrix sb is used for buffering Matrix b to reduce the dimenson
     * to increase the productivity of CPU, when you use this function, you 
     * can set {@code sb=new double[4][b[0].length} or the 2-D Array with the
     * same size.
     * by using buffer sb, we read 4 line of a and 4 row of b for one turn 
     * computing to get 16(4*4) values for c.
     * @param c
     * @param a
     * @param sb
     * @param i
     * @param j 
     */
    @Passed
    private static void mk44(double[][] c, double[][] a, double[][] sb, int i, int j)
    {
        double t00=0,  t01=0,  t02=0,  t03=0, 
               t10=0,  t11=0,  t12=0,  t13=0,
               t20=0,  t21=0,  t22=0,  t23=0,
               t30=0,  t31=0,  t32=0,  t33=0;
        double[] a0=a[i],   a1=a[i+1],  a2=a[i+2],  a3=a[i+3],
                 b0=sb[0],  b1=sb[1],   b2=sb[2],   b3=sb[3];
        double va0, va1, va2, va3, vb0, vb1, vb2, vb3;
        /**
         * a-->            b
         * t0  t1  t2  t3  |
         * t4  t5  t6  t7  |
         * t8  t8  t9  t11 
         * t12 t13 t14 t15
         */
        for(int k=0, width=sb[0].length;k<width;k++)
        {
            va0=a0[k];  va1=a1[k];  va2=a2[k];  va3=a3[k];
            vb0=b0[k];  vb1=b1[k];  vb2=b2[k];  vb3=b3[k];
            
            t00+=va0*vb0;   t01+=va0*vb1;   t02+=va0*vb2;   t03+=va0*vb3;  
            t10+=va1*vb0;   t11+=va1*vb1;   t12+=va1*vb2;   t13+=va1*vb3;
            t20+=va2*vb0;   t21+=va2*vb1;   t22+=va2*vb2;   t23+=va2*vb3;
            t30+=va3*vb0;   t31+=va3*vb1;   t32+=va3*vb2;   t33+=va3*vb3;
        }
        c[i][j]=t00;   c[i][j+1]=t01;   c[i][j+2]=t02;   c[i++][j+3]=t03;
        c[i][j]=t10;   c[i][j+1]=t11;   c[i][j+2]=t12;   c[i++][j+3]=t13;
        c[i][j]=t20;   c[i][j+1]=t21;   c[i][j+2]=t22;   c[i++][j+3]=t23;
        c[i][j]=t30;   c[i][j+1]=t31;   c[i][j+2]=t32;   c[i][j+3]=t33;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="MutiplyT-Kernel">
    @Passed
    private static void mtk11(double[][] c, double[][] a, double[][] b, int i, int j)
    {
        double t00=0;
        double[] a0=a[i],
                 b0=b[j];
        /**
         * a--> b
         * t0  |
         */
        for(int k=0;k<b0.length;k++) t00+=a0[k]*b0[k];
        c[i][j]=t00;
    }
    @Passed
    private static void mtk12(double[][] c, double[][] a, double[][] b, int i, int j)
    {
        double t00=0, t01=0;
        double[] a0=a[i],
                 b0=b[j],  b1=b[j+1];
        double va0, vb0, vb1;
        /**
         * a-->   b
         * t0  t1 |
         * t4  t5 |
         */
        for(int k=0;k<b0.length;k++)
        {
            va0=a0[k];
            vb0=b0[k];  vb1=b1[k];
            
            t00+=va0*vb0;   t01+=va0*vb1;
        }
        c[i][j]=t00;   c[i][j+1]=t01;
    }
    @Passed
    private static void mtk22(double[][] c, double[][] a, double[][] b, int i, int j)
    {
        double t00=0,  t01=0,
               t11=0,  t12=0;
        double[] a0=a[i],  a1=a[i+1],
                 b0=b[j],  b1=b[j+1];
        double va0, va1, vb0, vb1;
        /**
         * a-->   b
         * t0  t1 |
         * t4  t5 |
         */
        for(int k=0;k<b0.length;k++)
        {
            va0=a0[k];  va1=a1[k];
            vb0=b0[k];  vb1=b1[k];
            
            t00+=va0*vb0;   t01+=va0*vb1;
            t11+=va1*vb0;   t12+=va1*vb1;  
        }
        c[i][j]=t00;   c[i++][j+1]=t01;
        c[i][j]=t11;   c[i][j+1]=t12;
    }
    @Passed
    private static void mtk14(double[][] c, double[][] a, double[][] b, int i, int j)
    {
         double t00=0,  t01=0,  t02=0,  t03=0;
        double[] a0=a[i],
                 b0=b[j],  b1=b[j+1],   b2=b[j+2],   b3=b[j+3];
        double va0, vb0, vb1, vb2, vb3;
        /**
         * a-->            b
         * t0  t1  t2  t3  |
         */
        for(int k=0;k<b0.length;k++)
        {
            va0=a0[k];
            vb0=b0[k];  vb1=b1[k];  vb2=b2[k];  vb3=b3[k];
            
            t00+=va0*vb0;   t01+=va0*vb1;   t02+=va0*vb2;   t03+=va0*vb3;  
        }
        c[i][j]=t00;   c[i][j+1]=t01;   c[i][j+2]=t02;   c[i][j+3]=t03;
    }
    @Passed
    private static void mtk24(double[][] c, double[][] a, double[][] b, int i, int j)
    {
         double t00=0,  t01=0,  t02=0,  t3=0, 
               t10=0,  t11=0,  t12=0,  t13=0;
        double[] a0=a[i],   a1=a[i+1],
                 b0=b[j],  b1=b[j+1],   b2=b[j+2],   b3=b[j+3];
        double va0, va1, vb0, vb1, vb2, vb3;
        /**
         * a-->            b
         * t0  t1  t2  t3  |
         * t4  t5  t6  t7  |
         */
        for(int k=0;k<b0.length;k++)
        {
            va0=a0[k];  va1=a1[k];
            vb0=b0[k];  vb1=b1[k];  vb2=b2[k];  vb3=b3[k++];
            
            t00+=va0*vb0;   t01+=va0*vb1;   t02+=va0*vb2;   t3+=va0*vb3;  
            t10+=va1*vb0;   t11+=va1*vb1;   t12+=va1*vb2;   t13+=va1*vb3;
        }
        c[i][j]=t00;   c[i][j+1]=t01;   c[i][j+2]=t02;   c[i++][j+3]=t3;
        c[i][j]=t10;   c[i][j+1]=t11;   c[i][j+2]=t12;   c[i][j+3]=t13;
    }
     @Passed
    private static void mtk34(double[][] c, double[][] a, double[][] b, int i, int j)
    {   
        double t00=0,  t01=0,  t02=0,  t03=0, 
               t10=0,  t11=0,  t12=0,  t13=0,
               t20=0,  t21=0,  t22=0,  t23=0;
        double[] a0=a[i],   a1=a[i+1],  a2=a[i+2],
                 b0=b[j],  b1=b[j+1],   b2=b[j+2],   b3=b[j+3];
        double va0, va1, va2, vb0, vb1, vb2, vb3;
        /**
         * a-->            b
         * t0  t1  t2  t3  |
         * t4  t5  t6  t7  |
         * t8  t8  t9  t11 
         */
        for(int k=0;k<b0.length;k++)
        {
            va0=a0[k];  va1=a1[k];  va2=a2[k];
            vb0=b0[k];  vb1=b1[k];  vb2=b2[k];  vb3=b3[k++];
            
            t00+=va0*vb0;   t01+=va0*vb1;   t02+=va0*vb2;   t03+=va0*vb3;  
            t10+=va1*vb0;   t11+=va1*vb1;   t12+=va1*vb2;   t13+=va1*vb3;
            t20+=va2*vb0;   t21+=va2*vb1;   t22+=va2*vb2;   t23+=va2*vb3;
        }
        c[i][j]=t00;   c[i][j+1]=t01;   c[i][j+2]=t02;   c[i++][j+3]=t03;
        c[i][j]=t10;   c[i][j+1]=t11;   c[i][j+2]=t12;   c[i++][j+3]=t13;
        c[i][j]=t20;   c[i][j+1]=t21;   c[i][j+2]=t22;   c[i][j+3]=t23;
    }
    @Passed
    private static void mtk44(double[][] c, double[][] a, double[][] b, int i, int j)
    {
        double t00=0,  t01=0,  t02=0,  t03=0, 
               t10=0,  t11=0,  t12=0,  t13=0,
               t20=0,  t21=0,  t22=0,  t23=0,
               t30=0,  t31=0,  t32=0,  t33=0;
        double[] a0=a[i],  a1=a[i+1],  a2=a[i+2],  a3=a[i+3],
                 b0=b[j],  b1=b[j+1],  b2=b[j+2],  b3=b[j+3];
        double va0, va1, va2, va3, vb0, vb1, vb2, vb3;
        /**
         * a-->            b
         * t0  t1  t2  t3  |
         * t4  t5  t6  t7  |
         * t8  t8  t9  t11 
         * t12 t13 t14 t15
         */
        for(int k=0;k<b0.length;k++)
        {
            va0=a0[k];  va1=a1[k];  va2=a2[k];  va3=a3[k];
            vb0=b0[k];  vb1=b1[k];  vb2=b2[k];  vb3=b3[k];
            
            t00+=va0*vb0;   t01+=va0*vb1;   t02+=va0*vb2;   t03+=va0*vb3;  
            t10+=va1*vb0;   t11+=va1*vb1;   t12+=va1*vb2;   t13+=va1*vb3;
            t20+=va2*vb0;   t21+=va2*vb1;   t22+=va2*vb2;   t23+=va2*vb3;
            t30+=va3*vb0;   t31+=va3*vb1;   t32+=va3*vb2;   t33+=va3*vb3;
        }
        c[i][j]=t00;   c[i][j+1]=t01;   c[i][j+2]=t02;   c[i++][j+3]=t03;
        c[i][j]=t10;   c[i][j+1]=t11;   c[i][j+2]=t12;   c[i++][j+3]=t13;
        c[i][j]=t20;   c[i][j+1]=t21;   c[i][j+2]=t22;   c[i++][j+3]=t23;
        c[i][j]=t30;   c[i][j+1]=t31;   c[i][j+2]=t32;   c[i][j+3]=t33;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="SingleThread-Array-Mutiplication">
    @Passed("sb=new double[4][b.length]")
    public static void multiply44(double[][] c, double[][] a, double[][] b, double[][] sb)
    {
        for(int i,j=0,k;j<b[0].length;j+=4)
        {
            for(k=0;k<b.length;k++)//get 4 column from b
            {
                sb[0][k]=b[k][j];
                sb[1][k]=b[k][j+1];
                sb[2][k]=b[k][j+2];
                sb[3][k]=b[k][j+3];
            }
            for(i=0;i<a.length;i+=4) mk44(c, a, sb, i, j);
        }
    }
    @Passed("sb=new double[4][b.length]")
    public static void multiply(double[][] c, double[][] a, double[][] b, double[][] sb) 
    {
        int i,j=0,k;
        for(;j<b[0].length-3;j+=4)
        {
            for(k=0;k<b.length;k++)//get 4 column from b
            {
                sb[0][k]=b[k][j];
                sb[1][k]=b[k][j+1];
                sb[2][k]=b[k][j+2];
                sb[3][k]=b[k][j+3];
            }
            for(i=0;i<a.length-3;i+=4) mk44(c, a, sb, i, j);
            if(i==a.length-3) mk34(c, a, sb, i, j);
            else if(i==a.length-2) mk24(c, a, sb, i, j);
            else if(i==a.length-1) mk14(c, a, sb, i, j);
        }
        if(j<b[0].length-1)
        {
            for(k=0;k<b.length;k++)//get 2 column from b
            {
                sb[0][k]=b[k][j];
                sb[1][k]=b[k][j+1];
            }
            for(i=0;i<a.length-1;i+=2) mk22(c, a, sb, i, j);
            if(i<a.length) mk12(c, a, sb, i, j);
            j+=2;
        }
        if(j<b[0].length)
        {
            for(k=0;k<b.length;k++)//get 1 column from b
                sb[0][k]=b[k][j];
            for(i=0;i<a.length;i++) mk11(c, a, sb, i, j);
        }
    }
    @Passed("sb=new double[4][sb[0].length")
    public static void multiplyT(double[][] c, double[][] a, double[][] b) 
    {
        int i,j=0;
        for(;j<b.length-3;j+=4)
        {
            for(i=0;i<a.length-3;i+=4) mtk44(c, a, b, i, j);
            if(i==a.length-3) mtk34(c, a, b, i, j);
            else if(i==a.length-2) mtk24(c, a, b, i, j);
            else if(i==a.length-1) mtk14(c, a, b, i, j);
        }
        if(j<b.length-1)
        {
            for(i=0;i<a.length-1;i+=2) mtk22(c, a, b, i, j);
            if(i<a.length) mtk12(c, a, b, i, j);
            j+=2;
        }
        if(j<b.length)
        {
            for(i=0;i<a.length;i++) mtk11(c, a, b, i, j);
        }
    }
    public static void multiply(double[][] c, double[][] a, double[][] b) 
    {
        multiply(c, a, b, new double[4][b.length]);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="MultiThread-Array-Multiplication">
    private static final ExecutorService exec=Executors.newFixedThreadPool(14);
    private static final BinarySemaphore mutex=new BinarySemaphore();
    
    public static void shutDown()
    {
        mutex.P();
        if(!exec.isShutdown()) exec.shutdown();
        mutex.V();
    }
    @Passed
    public static void multiply44M(double[][] c, double[][] a, double[][] b, ExecutorService exec)
    {
        Lock ss=new Lock((a.length/4)*(b[0].length/4));
        for(int i,j=0,k;j<b[0].length-3;j+=4)
        {
            double[][] sb=new double[4][b.length];
            for(k=0;k<b.length;k++)//get 4 column from b
            {
                sb[0][k]=b[k][j];
                sb[1][k]=b[k][j+1];
                sb[2][k]=b[k][j+2];
                sb[3][k]=b[k][j+3];
            }
            for(i=0;i<a.length;i+=4)
            {
                int row=i, col=j;
                exec.submit(() -> {mk44(c, a, sb, row, col);ss.unlock();});
            }
        }
        ss.lock();
    }
    @Passed("8x/4x/2x/1x-->12x becomes slower")
    public static void multiplyM(double[][] c, double[][] a, double[][] b, ExecutorService exec)
    {
        int i,j=0,k,bwidth=b[0].length;
        Lock ss=new Lock((a.length>>2)*(b[0].length>>2));
        for(;j<bwidth-3;j+=4)
        {
            double[][] sb=new double[4][b.length];
            for(k=0;k<b.length;k++)//get 4 column from b
            {
                sb[0][k]=b[k][j];
                sb[1][k]=b[k][j+1];
                sb[2][k]=b[k][j+2];
                sb[3][k]=b[k][j+3];
            }
            for(i=0;i<a.length-7;i+=8)
            {
                int row=i, col=j;//you can't coding this line outside, as i is changing
                exec.submit(() -> {mk44(c, a, sb, row, col);
                    mk44(c, a, sb, row+4, col);
                    ss.unlock(2);});
            }
            if(i<a.length-3)
            {
                int row=i, col=j;
                exec.submit(() -> {mk44(c, a, sb, row, col);ss.unlock();});
                i+=4;
            }
            if(i==a.length-3) {mk34(c, a, sb, i, j);}//remained=3
            else if(i==a.length-2) {mk24(c, a, sb, i, j);}//remaind=2
            else if(i==a.length-1) mk14(c, a, sb, i, j);//remaind=1
        }
        if(j<bwidth-1)
        {
            double[][] sb=new double[2][b.length];
            for(k=0;k<b.length;k++)//get 2 column from b
            {
                sb[0][k]=b[k][j];
                sb[1][k]=b[k][j+1];
            }
            for(i=0;i<a.length-1;i+=2) mk22(c, a, sb, i, j);
            if(i<a.length) mk12(c, a, sb, i, j);
            j+=2;
        }
        if(j<bwidth)
        {
            double[][] sb=new double[1][b.length];
            for(k=0;k<b.length;k++)//get 1 column from b
                sb[0][k]=b[k][j];
            for(i=0;i<a.length;i++) mk11(c, a, sb, i, j);
        }
        ss.lock();
    }
    @Passed("8x/4x/2x/1x")
    public static void multiplyTM(double[][] c, double[][] a, double[][] b, ExecutorService exec)
    {
        int i,j=0;
        Lock ss=new Lock((a.length/4)*(b.length/4));
        for(;j<b.length-3;j+=4)
        {
            for(i=0;i<a.length-7;i+=8)
            {
                int row=i, col=j;
                exec.submit(()->{
                    mtk44(c, a, b, row, col);
                    mtk44(c, a, b, row+4, col);
                    ss.unlock(2);
                });
            }
            if(i<a.length-3)
            {
                int row=i, col=j;
                exec.submit(() ->{mtk44(c, a, b, row, col);ss.unlock();});
                i+=4;
            }
            if(i==a.length-3) {mtk34(c, a, b, i, j);}
            else if(i==a.length-2) {mtk24(c, a, b, i, j);i+=2;}
            else if(i==a.length-1) mtk14(c, a, b, i, j);
        }
        if(j<b.length-1)
        {
            for(i=0;i<a.length-1;i+=2) mtk22(c, a, b, i, j);
            if(i<a.length) mtk12(c, a, b, i, j);
            j+=2;
        }
        if(j<b.length) for(i=0;i<a.length;i++) mtk11(c, a, b, i, j);
        ss.lock();
    }
    public static void multiplyM(double[][] c, double[][] a, double[][] b)
    {
        multiplyM(c, a, b, exec);
    }
    public static void multiplyTM(double[][] c, double[][] a, double[][] b)
    {
        multiplyTM(c, a, b, exec);
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Eigens of Matrix">
    //<editor-fold defaultstate="collapsed" desc="static class Eigens">
    public static class Eigen implements Comparable
    {
        //columns---------------------------------------------------------------
        double value;
        double[] vector;
        int index;
        
        //functions------------------------------------------------------------
        Eigen(int index ,double value, double[] vector)
        {
            this.index=index;
            this.value=value;
            this.vector=vector;
        }
        public double getValue()
        {
            return value;
        }
        public double[] getVector()
        {
            return vector;
        }
        @Override
        public int compareTo(Object o)
        {
            double av=((Eigen)o).value;
            if(value>av) return 1;
            else if(value<av) return -1;
            else return 0;
        }
        public void append(StringBuilder sb)
        {
            sb.append('(').append(index).append(")\t").append(value).append("\t[");
            Vector.append(sb, vector);
            sb.append(']');
        }
        @Override
        public String toString()
        {
            StringBuilder sb=new StringBuilder();
            this.append(sb);
            return sb.toString();
        }
    }
    public static class Eigens extends ZArrayList<Eigen>
    {
        private static final Comparator NORMAL_CMP=new Comparator<Eigen>() {
            @Override
            public int compare(Eigen e1, Eigen e2)
            {
                if(e1.value>e2.value) return 1;
                else if(e1.value<e2.value) return -1;
                else return 0;
            }
        };
        private static final Comparator INDEX_CMP=new Comparator<Eigen>() {
            @Override
            public int compare(Eigen e1, Eigen e2)
            {
                return e1.index-e2.index;
            }
        };
        private static final Comparator ABS_CMP=new Comparator<Eigen>() {
            @Override
            public int compare(Eigen e1, Eigen e2)
            {
                double d1=(e1.value<0? -e1.value:e1.value),
                       d2=(e2.value<0? -e2.value:e2.value);
                if(d1<d2) return -1;
                else if(d1>d2) return 1;
                else return 0;
            }
        };
        //columns---------------------------------------------------------------
        double[] values;
        double[][] vectors;
        
        //functions-------------------------------------------------------------
        public double[] getValues()
        {
            if(values==null)
            {
                values=new double[num];
                for(int i=0;i<values.length;i++) values[i]=((Eigen)data[i]).value;
            }
            return values;
        }
        public double[][] getVectors() 
        {
            if(vectors==null)
            {
                vectors=new double[num][];
                for(int i=0;i<vectors.length;i++) 
                    vectors[i]=((Eigen)data[i]).getVector();
            }
            return vectors;
        }
        public Eigens sort()
        {
            values=null;
            vectors=null;
            Sort.sort(data, NORMAL_CMP);
            return this;
        }
        public Eigens sortByAbs()
        {
            values=null;
            vectors=null;
            Sort.sort(data, ABS_CMP);
            return this;
        }
        public Eigens sortByIndex()
        {
            values=null;
            vectors=null;
            Sort.sort(data, INDEX_CMP);
            return this;
        }
        @Override
        public String toString()
        {
            StringBuilder sb=new StringBuilder();
            for(int i=0;i<num;i++) {((Eigen)data[i]).append(sb);sb.append('\n');}
            return sb.toString();
        }
    }
    //</editor-fold>
    public static void requreSqureMatrix(double[][] x)
    {
        if(x==null) throw new NullPointerException();
        if(x.length!=x[0].length) throw new IAE("Not a Squre Matrix");
    }
    //<editor-fold defaultstate="collapse" desc="JacbiCor-Rotatrion">
    final static Eigens eigenJacbiCorRotation(double[][] x, int iterLimit, double precision)
    {
        requreSqureMatrix(x);
        int n=x.length;//number of iterations
        double[][] vectors=Matrix.eyeDouble(n);//the Eigens Vectors
        double[] values=new double[n];//the Eigens Values
 
	for(int count=0;count<iterLimit;count++)
	{
            //find the max value of PM that not on the diagonal
            double max=x[0][1];
            int row =0, col=1;
            for(int i=0,j;i<n;i++)			//行
            for(j=0;j<n;j++)		//列
            if(i!=j)
            {
                double d=(x[i][j]<0? -x[i][j]:x[i][j]);
                if(d>max){max=d;row=i;col=j;}
            } 
            //if the precision ,or the iteration times greater than need, end Loop
            if(max<=precision) break;
            
            double App=x[row][row],Aqq=x[col][col],Apq=x[row][col];
             
            //Calculate the rotation Angle and do rotation
            double u=-2*Apq/(Aqq-App);
            double angle=0.5*Math.atan(u);
            double sina=sin(angle), cosa=cos(angle);
            double sin2a=2*sina*cosa, cos2a=(cosa+sina)*(cosa-sina);
            
	    x[row][row]=App*cosa*cosa + Aqq*sina*sina + 2*Apq*cosa*sina;
            x[col][col]=App*sina*sina + Aqq*cosa*cosa- 2*Apq*cosa*sina;
	    x[col][row]=x[row][col]=0.5*(Aqq-App)*sin2a + Apq*cos2a;
 
	    for(int i=0;i<n;i++) 
            if(i!=col&&i!=row) 
            { 
		max = x[i][row]; 
		x[i][row]= x[i][col]*sina + max*cosa; 
                x[i][col]= x[i][col]*cosa - max*sina; 
            } 
 
            for(int j=0;j<n;j++)
            if(j!=col&&j!=row) 
            { 
                max = x[row][j]; 
            	x[row][j]= x[col][j]*sina + max*cosa; 
                x[col][j]= x[col][j]*cosa - max*sina; 
            } 
 
            for(int i=0;i<n;i ++)//compute the eigen vectors
            { 
		max=vectors[i][row]; 
		vectors[i][row] = vectors[i][col]*sina + max*cosa; 
		vectors[i][col] = vectors[i][col]*cosa - max*sina; 
            } 
	}
        
        for(int i=0,j;i<n;i++)//make sure all all Vects is positive
	{
            double sum=0;
            values[i]=x[i][i];
            for(j=0;j<n;j++) sum+=vectors[j][i];
            if(sum>=0) continue;
            for(j=0;j<n;j++) vectors[j][i]=-vectors[j][i];
	}
       
        Eigens egs=new Eigens();
        for(int i=0;i<values.length;i++) 
            egs.add(new Eigen(i, values[i], vectors[i]));
        return egs;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="QR-Decomposition">
    @Passed
    final static void QR(double x[][])
    {
        double[][] mt=Matrix.eyeDouble(x.length);
        double sum, cr, hr;
        double vw[]=new double[x.length], vu[]=new double[x.length], 
               vp[]=new double[x.length];
        
        boolean allzero;
        for(int r=0,i,j;r<x.length-1;r++)
        {
            //check whether the main diagonal elements of the matrix are 0
            for(allzero=true,i=r+1;i<x.length;i++)
                if(x[i][r]!=0) {allzero=false;break;}
            if(allzero) break;
            
            for(sum=0, i=r;i<x.length;i++) sum+=x[i][r]*x[i][r];
            cr=(x[r][r]==0? Math.sqrt(sum):-Math.signum(x[r][r])*Math.sqrt(sum));
            hr=cr*(cr-x[r][r]);
            
            for(i=0;i<r;i++) vu[i]=0;//evaluate vectorU
            vu[r]=x[r][r]-cr;
            for(i=r+1;i<x.length;i++) vu[i]=x[i][r];
            
            for(i=0;i<x.length;i++) //计算Q与R（An）
            for(vw[i]=0,j=0;j<x.length;j++) vw[i]+=mt[i][j]*vu[j];
           
            for(i=0;i<x.length;i++)
            for(j=0;j<x.length;j++) mt[i][j]-=vw[i]*vu[j]/hr;
            
            for(i=0;i<x.length;vp[i]/=hr,i++)
            for(vp[i]=0, j=0;j<x.length;j++) vp[i]+=x[j][i]*vu[j];
            
            for(i=0;i<x.length;i++)
            for(j=0;j<x.length;j++) x[i][j]-=vu[i]*vp[j];
        }
    }
    @Passed
    final static void dualQR(double x[][])
    {
        //子程序：对matrixA进行双步位移QR分解，计算A=Q'AQ
        int m=x.length-1;
        double s=x[m-1][m-1]+x[m][m];
        double t=x[m-1][m-1]*x[m][m] - x[m][m-1]*x[m-1][m];
        double[][] mat=new double[x.length][x.length];
        
        int i,j, r;
        for(i=0;i<x.length;i++)
        for(j=0;j<x.length;j++)
        for(r=0;r<x.length;r++) mat[i][j]+=x[i][r]*x[r][j];//square a

        for(i=0;i<x.length;i++)
        for(j=0;j<x.length;j++)//assign value for diagnoal and not 
            mat[i][j]= (i==j? mat[i][j]-s*x[i][j]+t : mat[i][j]-s*x[i][j]);
        
        double sum, cr, hr, tr, w;
        double vu[]=new double[x.length], vv[]=new double[x.length],
               vp[]=new double[x.length], vq[]=new double[x.length];
        
        boolean allzero;
        for(r=0;r<m;r++)
        {
            for(allzero=true, i=r+1;i<x.length;i++)
                if(mat[i][r]!=0){allzero=false;break;}
            if(allzero) continue;
            
            for(sum=0,i=r;i<x.length;i++) sum+=mat[i][r]*mat[i][r];
            cr=(mat[r][r]==0?  Math.sqrt(sum): -Math.signum(mat[r][r])*Math.sqrt(sum));
            hr=cr*(cr-mat[r][r]);
            
            for(i=0;i<r;i++) vu[i]=0;
            vu[r]=mat[r][r]-cr;
            for(i=r+1;i<x.length;i++) vu[i]=mat[i][r];
               
            for(j=0;j<x.length;vv[j]/=hr,j++)
            for(vv[j]=0, i=0;i<x.length;i++) vv[j]+=mat[i][j]*vu[i];
                
            for(i=0;i<x.length;i++)
            for(j=0;j<x.length;j++) mat[i][j]=mat[i][j]-vu[i]*vv[j];   

            for(i=0;i<x.length;vp[i]/=hr,i++)//Pr
            for(vp[i]=0, j=0;j<x.length;j++) vp[i]+=x[j][i]*vu[j];
            
            for(i=0;i<x.length;vq[i]/=hr, i++)//Qr
            for(vq[i]=0, j=0;j<x.length;j++) vq[i]+=x[i][j]*vu[j];
               
            for(tr=0, i=0;i<x.length;i++) tr+=vp[i]*vu[i];
            
            for(tr/=hr,i=0;i<x.length;i++)
            for(w=vq[i]-tr*vu[i],j=0;j<x.length;j++)
                x[i][j]=x[i][j]-w*vu[j]-vu[i]*vp[j];
        }
    }
    @Passed
    public static Eigens eigenQRDecomposition(double x[][], int iterLimit, double precision)
    {
        requreSqureMatrix(x);
        
        Matrix.toHessenberg(x, x);
                 
        int da=x.length, m=da-1;//cauculate eigen values
        double egValue[][]=new double[da][2];
        double[] s1={0, 0}, s2={0, 0};
        double detD, sum;
        for(int k=0;;k++)
        {
            if(Math.abs(x[m][m-1])<=precision)
            {
                egValue[m][0]=x[m][m];
                if(--m==0) {egValue[m][0]=x[m][m];break;}
                if(m==-1) {break;}
                continue;
            }
            detD=x[m-1][m-1]*x[m][m]-x[m-1][m]*x[m][m-1];
            sum=(x[m-1][m-1]+x[m][m])*(x[m-1][m-1]+x[m][m])-4*detD;
            s1[1]=s1[0]=s2[1]=s2[0]=0;
            if(sum>=0)
            {
                s1[0]=(x[m-1][m-1]+x[m][m]+Math.sqrt(sum))/2;
                s2[0]=(x[m-1][m-1]+x[m][m]-Math.sqrt(sum))/2;
            }
            else
            {
                s1[0]=(x[m-1][m-1]+x[m][m])/2;
                s1[1]=Math.sqrt(Math.abs(sum))/2;
                s2[0]=(x[m-1][m-1]+x[m][m])/2;
                s2[1]=-Math.sqrt(Math.abs(sum))/2;
            }
            if(m==1)
            {
                egValue[m][0]=s1[0];
                egValue[m][1]=s1[1];
                egValue[m-1][0]=s2[0];
                egValue[m-1][1]=s2[1];
                break;
            }
            if(Math.abs(x[m-1][m-2])<=precision)
            {
                egValue[m][0]=s1[0];
                egValue[m][1]=s1[1];
                egValue[m-1][0]=s2[0];
                egValue[m-1][1]=s2[1];
                m=m-2;
                if(m==0) {egValue[m][0]=x[m][m];break;}
                if(m==-1) break;
                continue;
            }
            if(k==iterLimit) return null;
            dualQR(x);
        }
        
        Eigens egs=new Eigens();
        double b[]=new double[da];//cauculate eigen vectors
        for(int n=0,i,j,k,index=0;n<da;n++)
        {
            if(egValue[n][1]!=0) continue;
            
            double egVector[]=new double[da];
            for(i=0;i<da;i++)
            for(j=0;j<da;j++)
                if(i!=j) x[i][j]=Math.sin(0.5*(i+1)+0.2*(j+1));
                else x[i][j]=1.5*Math.cos(i+1+1.2*(j+1));
            
            for(i=0;i<da;i++) x[i][i]-=egValue[n][0];
                
            double u;//置线性方程组的右端结果为0向量 matrixA-eigenvalue*I
            for(k=0;x[k][k]!=0&&k<da-1;k++)
            for(i=k+1;i<da;i++)
            {
                u=x[i][k]/x[k][k];
                b[i]-=u*b[k];
                for(j=k+1;j<da;j++) x[i][j]-=u*x[k][j];
            }

            egVector[da-1]=1; //矩阵消元化为上三角阵
            for(k=da-2; k>=0; k--)
            {
                double h=0;
                for(j=k+1;j<da;j++) h+=x[k][j]*egVector[j];
                egVector[k]=(b[k]-h)/x[k][k];
            }
            egs.add(new Eigen(index++, egValue[n][0], egVector));
        }
        return egs;
    }
    //</editor-fold>
    //</editor-fold>
}

