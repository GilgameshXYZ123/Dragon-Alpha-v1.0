/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.math.vector;

import java.io.PrintStream;
import static java.lang.Math.exp;
import static java.lang.Math.log;
import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.Collection;
import java.util.LinkedList;
import java.util.List;
import z.util.lang.Lang;
import static z.util.lang.Lang.NULL;
import z.util.math.vector.Vector.MaxMin;
import z.util.lang.annotation.Passed;
import static z.util.math.vector.ExMatrix.EX_MULTIPLY_THRESHOLD;
import z.util.math.vector.ExMatrix.Eigens;
import static z.util.math.vector.ExMatrix.OPT_MULTIPLY_THRESHOLD;

/**
 *
 * @author dell
 */
public final class Matrix 
{ 
    private Matrix() {}
    
    //<editor-fold defaultstate="collapsed" desc="Matrix-Checker-Function">
    public static boolean equals(boolean[][] a, boolean[][] b)
    {
        if(a==b) return true;
        if(a==null||b==null) return false;
        if(a.length!=b.length) return false;
        
        for(int i=0;i<a.length;i++) 
            if(!Arrays.equals(a[i], b[i])) return false;
        return true;
    }
    public static boolean equals(byte[][] a, byte[][] b)
    {
        if(a==b) return true;
        if(a==null||b==null) return false;
        if(a.length!=b.length) return false;
        
        for(int i=0;i<a.length;i++) 
            if(!Arrays.equals(a[i], b[i])) return false;
        return true;
    }
    public static boolean equals(char[][] a, char[][] b)
    {
        if(a==b) return true;
        if(a==null||b==null) return false;
        if(a.length!=b.length) return false;
        
        for(int i=0;i<a.length;i++) 
            if(!Arrays.equals(a[i], b[i])) return false;
        return true;
    }
    public static boolean equals(short[][] a, short[][] b)
    {
        if(a==b) return true;
        if(a==null||b==null) return false;
        if(a.length!=b.length) return false;
        
        for(int i=0;i<a.length;i++) 
            if(!Arrays.equals(a[i], b[i])) return false;
        return true;
    }
    public static boolean equals(int[][] a, int[][] b)
    {
        if(a==b) return true;
        if(a==null||b==null) return false;
        if(a.length!=b.length) return false;
        
        for(int i=0;i<a.length;i++) 
            if(!Arrays.equals(a[i], b[i])) return false;
        return true;
    }
    public static boolean equals(long[][] a, long[][] b)
    {
        if(a==b) return true;
        if(a==null||b==null) return false;
        if(a.length!=b.length) return false;
        
        for(int i=0;i<a.length;i++) 
            if(!Arrays.equals(a[i], b[i])) return false;
        return true;
    }
    public static boolean equals(float[][] a, float[][] b)
    {
        if(a==b) return true;
        if(a==null||b==null) return false;
        if(a.length!=b.length) return false;
        
        for(int i=0;i<a.length;i++) 
            if(!Arrays.equals(a[i], b[i])) return false;
        return true;
    }
    public static boolean equals(double[][] a, double[][] b)
    {
        if(a==b) return true;
        if(a==null||b==null) return false;
        if(a.length!=b.length) return false;
        
        for(int i=0;i<a.length;i++) 
            if(!Arrays.equals(a[i], b[i])) return false;
        return true;
    }
    
    
    public static void checkMatrix(boolean[][] mat)
    {
        if(mat==null) throw new NullPointerException();
        if(mat[0]==null) throw new NullPointerException();
        int width=mat[0].length;
        for(int i=1;i<mat.length;i++)
        {
            if(mat[i]==null) throw new NullPointerException();
            if(mat[i].length!=width) throw new IllegalArgumentException("mat["+i+"].length!=width");
        }
    }
    public static void checkMatrix(boolean[][] mat, int height, int width)
    {
        if(mat==null) throw new NullPointerException();
        if(mat.length!=height) throw new IllegalArgumentException("mat.length!=height");
        for(int i=0;i<mat.length;i++)
        {
            if(mat[i]==null) throw new NullPointerException();
            if(mat[i].length!=width) throw new IllegalArgumentException("mat["+i+"].length!=width");
        }
    }
    
    public static void checkMatrix(byte[][] mat) {checkMatrix(mat, "mat");}
    public static void checkMatrix(byte[][] mat, String name)
    {
        if(mat == null) throw new NullPointerException();
        if(mat[0]==null) throw new NullPointerException(name + "[0] is null");
        for(int i = 1, width = mat[0].length; i<mat.length; i++)
        {
            if(mat[i] == null) throw new NullPointerException(name + "[" + i + "] is null");
            if(mat[i].length != width) throw new IllegalArgumentException(
                    name + "[" + i + "].length != width");
        }
    }
    public static void checkMatrix(byte[][] mat, int height, int width)
    {
        if(mat == null) throw new NullPointerException();
        if(mat.length != height) throw new IllegalArgumentException("mat.length!=height");
        for(int i=0;i<mat.length;i++)
        {
            if(mat[i]==null) throw new NullPointerException();
            if(mat[i].length!=width) throw new IllegalArgumentException("mat["+i+"].length!=width");
        }
    }
    
    public static void checkMatrix(char[][] mat)
    {
        if(mat==null) throw new NullPointerException();
        if(mat[0]==null) throw new NullPointerException();
        int width=mat[0].length;
        for(int i=1;i<mat.length;i++)
        {
            if(mat[i]==null) throw new NullPointerException();
            if(mat[i].length!=width) throw new IllegalArgumentException("mat["+i+"].length!=width");
        }
    }
     public static void checkMatrix(char[][] mat, int height, int width)
    {
        if(mat==null) throw new NullPointerException();
        if(mat.length!=height) throw new IllegalArgumentException("mat.length!=height");
        for(int i=0;i<mat.length;i++)
        {
            if(mat[i]==null) throw new NullPointerException();
            if(mat[i].length!=width) throw new IllegalArgumentException("mat["+i+"].length!=width");
        }
    }
    
    public static void checkMatrix(short[][] mat)
    {
        if(mat==null) throw new NullPointerException();
        if(mat[0]==null) throw new NullPointerException();
        int width=mat[0].length;
        for(int i=1;i<mat.length;i++)
        {
            if(mat[i]==null) throw new NullPointerException();
            if(mat[i].length!=width) throw new IllegalArgumentException("mat["+i+"].length!=width");
        }
    }
    public static void checkMatrix(short[][] mat, int height, int width)
    {
        if(mat==null) throw new NullPointerException();
        if(mat.length!=height) throw new IllegalArgumentException("mat.length!=height");
        for(int i=0;i<mat.length;i++)
        {
            if(mat[i]==null) throw new NullPointerException();
            if(mat[i].length!=width) throw new IllegalArgumentException("mat["+i+"].length!=width");
        }
    }
    
    public static void checkMatrix(int[][] mat)
    {
        if(mat==null) throw new NullPointerException();
        if(mat[0]==null) throw new NullPointerException();
        int width=mat[0].length;
        for(int i=1;i<mat.length;i++)
        {
            if(mat[i]==null) throw new NullPointerException();
            if(mat[i].length!=width) throw new IllegalArgumentException("mat["+i+"].length!=width");
        }
    }
    public static void checkMatrix(int[][] mat, int height, int width)
    {
        if(mat==null) throw new NullPointerException();
        if(mat.length!=height) throw new IllegalArgumentException("mat.length!=height");
        for(int i=0;i<mat.length;i++)
        {
            if(mat[i]==null) throw new NullPointerException();
            if(mat[i].length!=width) throw new IllegalArgumentException("mat["+i+"].length!=width");
        }
    }
    
    public static void checkMatrix(long[][] mat)
    {
        if(mat==null) throw new NullPointerException();
        if(mat[0]==null) throw new NullPointerException();
        int width=mat[0].length;
        for(int i=1;i<mat.length;i++)
        {
            if(mat[i]==null) throw new NullPointerException();
            if(mat[i].length!=width) throw new IllegalArgumentException("mat["+i+"].length!=width");
        }
    }
    public static void checkMatrix(long[][] mat, int height, int width)
    {
        if(mat==null) throw new NullPointerException();
        if(mat.length!=height) throw new IllegalArgumentException("mat.length!=height");
        for(int i=0;i<mat.length;i++)
        {
            if(mat[i]==null) throw new NullPointerException();
            if(mat[i].length!=width) throw new IllegalArgumentException("mat["+i+"].length!=width");
        }
    }
    
    public static void checkMatrix(float[][] mat, String name)
    {
        if(mat == null) throw new NullPointerException();
        if(mat[0] == null) throw new NullPointerException(name + "[0] is null");
        int width = mat[0].length;
        for(int i = 1; i<mat.length; i++)
        {
            if(mat[i] == null) throw new NullPointerException(name + "[" + i + "] is null");
            if(mat[i].length != width) throw new IllegalArgumentException(name + "[" + i + "].length!=width");
        }
    }
    public static void checkMatrix(float[][] mat, int height, int width)
    {
        if(mat==null) throw new NullPointerException();
        if(mat.length!=height) throw new IllegalArgumentException("mat.length!=height");
        for(int i=0;i<mat.length;i++)
        {
            if(mat[i]==null) throw new NullPointerException();
            if(mat[i].length!=width) throw new IllegalArgumentException("mat["+i+"].length!=width");
        }
    }
    
    public static void checkMatrix(double[][] mat)
    {
        if(mat==null) throw new NullPointerException();
        if(mat[0]==null) throw new NullPointerException();
        int width=mat[0].length;
        for(int i=1;i<mat.length;i++)
        {
            if(mat[i]==null) throw new NullPointerException();
            if(mat[i].length!=width) throw new IllegalArgumentException("mat["+i+"].length!=width");
        }
    }
    public static void checkMatrix(double[][] mat, int height, int width)
    {
        if(mat==null) throw new NullPointerException();
        if(mat.length!=height) throw new IllegalArgumentException("mat.length!=height");
        for(int i=0;i<mat.length;i++)
        {
            if(mat[i]==null) throw new NullPointerException();
            if(mat[i].length!=width) throw new IllegalArgumentException("mat["+i+"].length!=width");
        }
    }
    
    public static void checkMatrix(Object[][] mat)
    {
        if(mat==null) throw new NullPointerException();
        if(mat[0]==null) throw new NullPointerException();
        int width=mat[0].length;
        for(int i=1;i<mat.length;i++)
        {
            if(mat[i]==null) throw new NullPointerException();
            if(mat[i].length!=width) throw new IllegalArgumentException("mat["+i+"].length!=width");
        }
    }
    public static void checkMatrix(Object[][] mat, int height, int width)
    {
        if(mat==null) throw new NullPointerException();
        if(mat.length!=height) throw new IllegalArgumentException("mat.length!=height");
        for(int i=0;i<mat.length;i++)
        {
            if(mat[i]==null) throw new NullPointerException();
            if(mat[i].length!=width) throw new IllegalArgumentException("mat["+i+"].length!=width");
        }
    }
    
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Matrix-String-Function">
    //<editor-fold defaultstate="collapsed" desc="String-Function:append">
    public static void append(StringBuilder sb, boolean[][] v)
    {
        if(v==null) {sb.append(NULL);return;}
        Vector.append(sb, v[0]);
        for(int i=1;i<v.length;i++)
            {sb.append('\n');Vector.append(sb, v[i]);}
    }
    public static void append(StringBuilder sb, byte[][] v)
    {
        if(v==null) {sb.append(NULL);return;}
        for(int i=1;i<v.length;i++)
            {Vector.append(sb, v[i]);sb.append('\n');}
    }
    public static void append(StringBuilder sb, char[][] v)
    {
        if(v==null) {sb.append(NULL);return;}
        Vector.append(sb, v[0]);
        for(int i=1;i<v.length;i++)
            {sb.append('\n');Vector.append(sb, v[i]);}
    }
    public static void append(StringBuilder sb, short[][] v)
    {
        if(v==null) {sb.append(NULL);return;}
        Vector.append(sb, v[0]);
        for(int i=1;i<v.length;i++)
            {sb.append('\n');Vector.append(sb, v[i]);}
    }
    public static void append(StringBuilder sb, int[][] v)
    {
        if(v==null) {sb.append(NULL);return;}
        Vector.append(sb, v[0]);
        for(int i=1;i<v.length;i++)
            {sb.append('\n');Vector.append(sb, v[i]);}
    }
    public static void append(StringBuilder sb, long[][] v)
    {
       if(v==null) {sb.append(NULL);return;}
        Vector.append(sb, v[0]);
        for(int i=1;i<v.length;i++)
            {sb.append('\n');Vector.append(sb, v[i]);}
    }
    public static void append(StringBuilder sb, float[][] v)
    {
       if(v==null) {sb.append(NULL);return;}
        Vector.append(sb, v[0]);
        for(int i=1;i<v.length;i++)
            {sb.append('\n');Vector.append(sb, v[i]);}
    }
    public static void append(StringBuilder sb, double[][] v)
    {
        if(v==null) {sb.append(NULL);return;}
        Vector.append(sb, v[0]);
        for(int i=1;i<v.length;i++)
            {sb.append('\n');Vector.append(sb, v[i]);}
    }
    public static <T> void append(StringBuilder sb, T[][] v)
    {
        if(v==null) {sb.append(NULL);return;}
        Vector.append(sb, v[0]);
        for(int i=1;i<v.length;i++)
            {sb.append('\n');Vector.append(sb, v[i]);}
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="String-Function:toString">
    public static String toString(boolean[][] arr)
    {
        if(arr==null) return NULL;
        StringBuilder sb=new StringBuilder();
        Matrix.append(sb, arr);
        return sb.toString();
    }
    public static String toString(byte[][] arr)
    {
        if(arr==null) return NULL;
        StringBuilder sb=new StringBuilder();
        Matrix.append(sb, arr);
        return sb.toString();
    }
    public static String toString(short[][] arr)
    {
        if(arr==null) return NULL;
        StringBuilder sb=new StringBuilder();
        Matrix.append(sb, arr);
        return sb.toString();
    }
    public static String toString(int[][] arr)
    {
        if(arr==null) return NULL;
        StringBuilder sb=new StringBuilder();
        Matrix.append(sb, arr);
        return sb.toString();
    }
    public static String toString(long[][] arr)
    {
        if(arr==null) return NULL;
        StringBuilder sb=new StringBuilder();
        Matrix.append(sb, arr);
        return sb.toString();
    }
    public static String toString(float[][] arr)
    {
        if(arr==null) return NULL;
        StringBuilder sb=new StringBuilder();
        Matrix.append(sb, arr);
        return sb.toString();
    }
    public static String toString(double[][] arr)
    {
        if(arr==null) return NULL;
        StringBuilder sb=new StringBuilder();
        Matrix.append(sb, arr);
        return sb.toString();
    }
    public static <T> String toString(T[][] arr)
    {
        if(arr==null) return NULL;
        StringBuilder sb=new StringBuilder();
        Matrix.append(sb, arr);
        return sb.toString();
    }
    public static String toString(Collection<double[]> arr)
    {
        if(arr==null) return "null";
        StringBuilder sb=new StringBuilder();
        int j;
        for(double[] ar:arr) 
        {
            sb.append(ar[0]);
            for(j=0;j<ar.length;j++) sb.append(',').append(ar[j]);
            sb.append('\n');
        }
        return sb.toString();
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="String-Function:print">
    private static final PrintStream DEF_OUT=System.out;
    public static synchronized void setDefaultPrintStream(PrintStream out){}
    public static PrintStream getDefaultPrintStream() {return DEF_OUT;}
    
    public static void println(boolean[][] v) {println(DEF_OUT, v);}
    public static void println(byte[][] v) {println(DEF_OUT, v);}
    public static void println(short[][] v) {println(DEF_OUT, v);}
    public static void println(char[][] v) {println(DEF_OUT, v);}
    public static void println(int[][] v) {println(DEF_OUT, v);}
    public static void println(long[][] v) {println(DEF_OUT, v);}
    public static void println(float[][] v) {println(DEF_OUT, v);}
    public static void println(double[][] v) {println(DEF_OUT, v);}
    public static void println(double[][] v, char ldiv) {println(DEF_OUT, v, ldiv);}
    public static void println(Object[][] v) {println(DEF_OUT, v);}
    
    public static void println(PrintStream out, boolean[][] v)
    {
        if(v==null) {out.println(NULL);return;}
        for(int i=0;i<v.length;i++) Vector.println(out, v[i]);
    }
    public static void println(PrintStream out, byte[][] v)
    {
        if(v==null) {out.println(NULL);return;}
        for(int i=0;i<v.length;i++) Vector.println(out, v[i]);
    }
    public static void println(PrintStream out, short[][] v)
    {
        if(v==null) {out.println(NULL);return;}
        for(int i=0;i<v.length;i++) Vector.println(out, v[i]);
    }
    public static void println(PrintStream out, char[][] v)
    {
        if(v==null) {out.println(NULL);return;}
        for(int i=0;i<v.length;i++) Vector.println(out, v[i]);
    }
    public static void println(PrintStream out, int[][] v)
    {
        if(v==null) {out.println(NULL);return;}
        for(int i=0;i<v.length;i++) Vector.println(out, v[i]);
    }
    public static void println(PrintStream out, long[][] v)
    {
        if(v==null) {out.println(NULL);return;}
        for(int i=0;i<v.length;i++) Vector.println(out, v[i]);
    }
    public static void println(PrintStream out, float[][] v)
    {
        if(v==null) {out.println(NULL);return;}
        for(int i=0;i<v.length;i++) Vector.println(out, v[i]);
    }
    public static void println(PrintStream out, double[][] v)
    {
        if(v==null) {out.println(NULL);return;}
        for(int i=0;i<v.length;i++) Vector.println(out, v[i]);
    }
    public static void println(PrintStream out, double[][] v, char ldiv)
    {
        if(v==null) {out.println(NULL);return;}
        for(int i=0;i<v.length;i++) Vector.println(out, v[i], ldiv);
    }
    public static void println(PrintStream out, Object[][] v)
    {
        if(v==null) {out.println(NULL);return;}
        for(int i=0;i<v.length;i++) Vector.println(out, v[i]);
    }
    //</editor-fold>
    //</editor-fold>
    
    public static Object[] field_max_indexed(float[][] X) 
    {
        int height = X.length, width = X[0].length;
        float[] Y = new float[width];
        int[] Index = new int[width];
        for(int i=0; i<width; i++) Y[i] = -(Float.MAX_VALUE - 1);
        
        for(int i=0; i<height; i++) {
            for(int j=0; j<width; j++) 
                if(Y[j] < X[i][j]) { Y[j] = X[i][j]; Index[j] = i; }
        }
        return new Object[] { Y, Index };
    }
    
    public static Object[] field_min_indexed(float[][] X) 
    {
        int height = X.length, width = X[0].length;
        float[] Y = new float[width];
        int[] Index = new int[width];
        for(int i=0; i<width; i++) Y[i] = Float.MAX_VALUE;
        
        for(int i=0; i<height; i++) {
            for(int j=0; j<width; j++) 
                if(Y[j] > X[i][j]) { Y[j] = X[i][j]; Index[j] = i; }
        }
        return new Object[] { Y, Index };
    }
    
    public static Object[] row_max_indexed(float[][] X) 
    {
        int height = X.length, width = X[0].length;
        float[] Y = new float[height];
        int[] Index = new int[height];
        for(int i=0; i<height; i++) Y[i] = -(Float.MAX_VALUE - 1);
        
        for(int i=0; i<height; i++) {
            for(int j=0; j<width; j++) 
                if(Y[i] < X[i][j]) { Y[i] = X[i][j]; Index[i] = j; }
        }
        return new Object[] { Y, Index };
    }
    
    public static Object[] row_min_indexed(float[][] X) 
    {
        int height = X.length, width = X[0].length;
        float[] Y = new float[height];
        int[] Index = new int[height];
        for(int i=0; i<height; i++) Y[i] = Float.MAX_VALUE;
        
        for(int i=0; i<height; i++) {
            for(int j=0; j<width; j++) 
                if(Y[i] > X[i][j]) { Y[i] = X[i][j]; Index[i] = j; }
        }
        return new Object[] { Y, Index };
    }
    
    //<editor-fold defaultstate="collapsed" desc="MaxMin For Matrix">
    public static double max(double[][] x)
    {
        double max=x[0][0];
        for(int i=0,j;i<x.length;i++)
        for(j=0;j<x[i].length;j++) 
            if(max<x[i][j]) max=x[i][j];
        return max;
    }
    /**
     * Find the max value of Matrix {@code double[][]x} that are not 
     * on the diagonal.
     * @param x
     * @return 
     */
    public static double maxND(double[][] x)
    {
        double max=Double.MIN_VALUE;
        for(int i=0,j;i<x.length;i++)
        for(j=0;j<x[i].length;j++) 
            if(max<x[i][j]&&i!=j) max=x[i][j];
        return max;
    }
    public static MaxMin maxMin(double[][] x)
    {
        MaxMin<Double> mm=Vector.maxMin(x[0]);
        double max=mm.max, min=mm.min;
        for(int i=1,j;i<x.length;i++)
        {
            for(j=0;j<x[i].length;j+=2)
            {
                if(x[i][j]>x[i][j+1])
                {
                    if(max<x[i][j]) max=x[i][j];
                    if(min>x[i][j+1]) min=x[i][j+1];
                }
                else 
                {
                    if(max<x[i][j+1]) max=x[i][j+1];
                    if(min>x[i][j]) min=x[i][j];
                }
            }
            if((x[i].length&1)==1)
            {
                double ev=x[i][x.length-1];
                if(max<ev) max=ev;
                else if(min>ev) min=ev;
            }
        }
        mm.max=max;mm.min=min;
        return mm;
    }
    public static double maxAbs(double[][] x)
    {
        MaxMin<Double> mm=Matrix.maxMin(x);
        if(mm.min<0) mm.min*=-1;
        return (mm.max>mm.min? mm.max:mm.min);
    }
    /**
     * Find the element of Matrix {@code double[][] x} that are not
     * on the diagonal with the maximum absolute value.
     * @param x
     * @return 
     */
    public static double maxAbsNd(double[][] x)
    {
        double max=0, d;
        for(int i=0,j;i<x.length;i++)
        for(j=0;j<x[i].length;j++)
            if(i!=j&&(d=(x[i][j]<0? -x[i][j]:x[i][j]))>max) max=d;
        return max; 
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="MaxMin For each Line">
    public static double[] minValueEachRow(double[][] val) {
        double[] min = new double[val.length];
        for(int i=0; i<val.length; i++) min[i] = Vector.minValue(val[i]);
        return min;
    }
    public static void rowMin(float[][] val, float[] min) {
        for(int i=0; i<val.length; i++) min[i] = Vector.minValue(val[i]);
    }
    public static float[] minValueEachRow(float[][] val) {
        float[] min = new float[val.length];
        for(int i=0; i<val.length; i++) min[i] = Vector.minValue(val[i]);
        return min;
    }
    public static int[] minValueEachRow(int[][] val) {
        int[] min = new int[val.length];
        for(int i=0;i<val.length;i++) min[i] = Vector.minValue(val[i]);
        return min;
    }
    
    public static int[] maxValueEachRow(int[][] val) {
        int[] max  = new int[val.length];
        for(int i=0; i<val.length; i++) max[i] = Vector.maxValue(val[i]);
        return max;
    }
    public static void rowMax(float[][] val, float[] max) {
        for(int i=0; i<val.length; i++) max[i] = Vector.maxValue(val[i]);
    }
    public static float[] maxValueEachRow(float[][] val) {
        float[] max  = new float[val.length];
        for(int i=0; i<val.length; i++) max[i] = Vector.maxValue(val[i]);
        return max;
    }
    public static double[] maxValueEachRow(double[][] val) {
        double[] max = new double[val.length];
        for(int i=0; i<val.length; i++) max[i] = Vector.maxValue(val[i]);
        return max;
    }
    
    public static <T extends Comparable> T[] maxValueEachRow(T[][] val, Class<T> clazz) {
        T[] max=(T[]) Array.newInstance(clazz, val.length);
        for(int i=0; i<val.length; i++) max[i]=Vector.maxValue(val[i]);
        return max;
    }
    public static MaxMin<int[]> maxMinForEachLine(int[][] val)
    {
        int[] max=new int[val.length];
        int[] min=new int[val.length];
        MaxMin<Integer> mm=new MaxMin<>();
        for(int i=0;i<val.length;i++)
        {
            Vector.maxMin(val[i], 0, val[i].length-1, mm);
            max[i]=mm.max;
            min[i]=mm.min;
        }
        return new MaxMin(max,min);
    }
    public static MaxMin<int[]> maxMinForEachLine(double[][] val)
    {
        double[] max=new double[val.length];
        double[] min=new double[val.length];
        MaxMin<Double> mm=new MaxMin<>();
        for(int i=0;i<val.length;i++)
        {
            Vector.maxMin(val[i], 0, val[i].length-1, mm);
            max[i]=mm.max;
            min[i]=mm.min;
        }
        return new MaxMin(max,min);
    }
    public static <T extends Comparable> MaxMin<T[]> maxMinForEachLine(T[][] val, Class<T> clazz)
    {
        T[] max=(T[]) Array.newInstance(clazz, val.length);
        T[] min=(T[]) Array.newInstance(clazz, val.length);
        MaxMin<T> mm=new MaxMin<>();
        for(int i=0;i<val.length;i++)
        {
            Vector.maxMin(val[i], 0, val[i].length-1, mm);
            max[i]=mm.max;
            min[i]=mm.min;
        }
        return new MaxMin(max,min);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="MaxMin For each Field">
    public static void maxMinForEachField(int[][] val, int[] minArr, int[] maxArr)
    {
        int max,min,ev;
        boolean flag= (val.length&1)==1;
        for(int i,j=0,width=val[0].length;j<width;j++)
        {
            if(val[0][j]>val[1][j]) {max=val[0][j];min=val[1][j];}
            else {max=val[1][j];min=val[0][j];}
            for(i=2;i<val.length;i+=2)
            {
                if(val[i][j]>val[i+1][j])
                {
                    if(max<val[i][j]) max=val[i][j];
                    if(min>val[i+1][j]) min=val[i+1][j];
                }
                else
                {
                    if(max<val[i+1][j]) max=val[i+1][j];
                    if(min>val[i][j]) min=val[i][j];
                }
            }
            if(flag)//val.length is an odd number
            {
                ev=val[val.length-1][j];
                if(max<ev) max=ev;
                else if(min>ev) min=ev;
            }
            maxArr[j]=max;
            minArr[j]=min;
        }
    }
    public static void maxMinForEachField(double[][] val, double[] minArr, double[] maxArr)
    {
        double max,min,ev;
        boolean flag= (val.length&1)==1;
        for(int i,j=0,width=val[0].length;j<width;j++)
        {
            if(val[0][j]>val[1][j]) {max=val[0][j];min=val[1][j];}
            else {max=val[1][j];min=val[0][j];}
            for(i=2;i<val.length;i+=2)
            {
                if(val[i][j]>val[i+1][j])
                {
                    if(max<val[i][j]) max=val[i][j];
                    if(min>val[i+1][j]) min=val[i+1][j];
                }
                else
                {
                    if(max<val[i+1][j]) max=val[i+1][j];
                    if(min>val[i][j]) min=val[i][j];
                }
            }
            if(flag)//val.length is an odd number
            {
                ev=val[val.length-1][j];
                if(max<ev) max=ev;
                else if(min>ev) min=ev;
            }
            maxArr[j]=max;
            minArr[j]=min;
        }
    }
    public static <T extends Comparable> void maxMinForEachField(T[][] val, T[] minArr, T[] maxArr)
    {
        T max,min,ev;
        boolean flag= (val.length&1)==1;
        for(int i,j=0,width=val[0].length;j<width;j++)
        {
            if(val[0][j].compareTo(val[1][j])>0) {max=val[0][j];min=val[1][j];}
            else {max=val[1][j];min=val[0][j];}
            for(i=2;i<val.length;i+=2)
            {
                if(val[i][j].compareTo(val[i+1][j])>0)
                {
                    if(max.compareTo(val[i][j])<0) max=val[i][j];
                    if(min.compareTo(val[i+1][j])>0) min=val[i+1][j];
                }
                else
                {
                    if(max.compareTo(val[i+1][j])<0) max=val[i+1][j];
                    if(min.compareTo(val[i][j])>0) min=val[i][j];
                }
            }
            if(flag)//val.length is an odd number
            {
                ev=val[val.length-1][j];
                if(max.compareTo(ev)<0) max=ev;
                else if(min.compareTo(ev)>0) min=ev;
            }
            maxArr[j]=max;
            minArr[j]=min;
        }
    }
    public static MaxMin<double[]> maxMinForEachField(double[][] val)
    {
        double[] maxArr=new double[val[0].length];
        double[] minArr=new double[val[0].length];
        Matrix.maxMinForEachField(val, minArr, maxArr);
        return new MaxMin(maxArr,minArr);
    }
    
    public static MaxMin<int[]> maxMinForEachField(int[][] val)
    {
        int[] maxArr=new int[val[0].length];
        int[] minArr=new int[val[0].length];
        Matrix.maxMinForEachField(val, minArr, maxArr);
        return new MaxMin(maxArr,minArr);
    }
    public static <T extends Comparable> MaxMin<T[]> maxMinForEachField(T[][] val, Class<T> clazz)
    {
        T[] maxArr=(T[]) Array.newInstance(clazz, val[0].length);
        T[] minArr=(T[]) Array.newInstance(clazz, val[0].length);
        Matrix.maxMinForEachField(val, minArr, maxArr);
        return new MaxMin(maxArr,minArr);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Distance between Matrix">
    public static void distanceVectorLine(double[] result, double[][] left, double[][] right)
    {
        for(int i=0;i<result.length;i++)
            result[i]=Vector.distance(left[i], right[i]);
    }
    public static double[] distanceVectorLine(double[][] left, double[][] right)
    {
        double[] result=new double[left.length];
        Matrix.distanceVectorLine(result, left, right);
        return result;
    }
    public static void distanceVectorField(double[] result, double[][] value, double[] vector)
    {
        double t;
        for(int i,j=0;j<result.length;j++)
        {
            for(i=0;i<value.length;i++)
            {
                t=value[i][j]-vector[i];
                result[i]+=t*t;
            }
            result[i]=Math.sqrt(result[i]);
        }
    }
    public static double[] distanceVectorField(double[][] value, double[] vector)
    {
        double[] result=new double[value[0].length];
        Matrix.distanceVectorField(result, value, vector);
        return result;
    }
    public static double distanceLine(double[][] left, double[][] right)
    {
        double dis=0;
        for(int i=0;i<left.length;i++)
            dis+=Vector.distanceSquare(left[i], right[i]);
        return Math.sqrt(dis);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Matrix Aggregation">
    public static void averageLine(double[] result, double[][] value)
    {
        for(int i,j=0;j<result.length;j++)
        {
            result[j]=value[0][j];
            for(i=1;i<value.length;i++) result[j]+=value[i][j];
            result[j]/=value.length;
        }
    }
    public static void normalizeLine(double[][] result, double[] avg, double[] std)
    {
        for(int i,j=0;j<avg.length;j++)
        {
            avg[j]=result[0][j];
            std[j]=avg[j]*avg[j];
            for(i=1;i<result.length;i++)
            {
                avg[j]+=result[i][j];
                std[j]+=result[i][j]*result[i][j];
            }
            avg[j]/=result.length;
            std[j]/=result.length;
            std[j]=Math.sqrt(std[j]-avg[j]*avg[j]);
            
            for(i=0;i<result.length;i++)
                result[i][j]=(result[i][j]-avg[j])/std[j];
        }
    }
    public static void normalizeField(double[][] result, double[] avg, double[] std)
    {
        for(int i=0,j;i<result.length;i++)
        {
            avg[i]=result[i][0];
            std[i]=avg[i]*avg[i];
            for(j=1;j<result[i].length;i++) 
            {
                avg[i]+=result[i][j];
                std[i]+=result[i][j]*result[i][j];
            }
            avg[i]/=result[i].length;
            std[i]/=result[i].length;
            std[i]=Math.sqrt(std[i]-avg[i]*avg[i]);
            
            for(j=0;j<result[i].length;j++)
                result[i][j]=(result[i][j]-avg[j])/std[j];
        }
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Basic Matrix Manipulation">
    public static void printShape(double[][] result)
    {
        System.out.println(result.length+" * "+result[0].length);
    }
    @Passed
    public static void transpose(double[][] result, double[][] value)
    {
        if(result==value)//only allowed for square Matrix
        {
            double t;
            for(int i=0,j;i<value.length;i++)
            for(j=i;j<value.length;j++)
                {t=value[i][j];value[i][j]=value[j][i];value[j][i]=t;}
            return;
        }
        for(int i=0,j;i<result.length;i++)
        for(j=0;j<value.length;j++) result[i][j]=value[j][i];
    }
    public static void transpose(float[][] A, float[][] AT)
    {
        int N=A.length, M=A[0].length;
        for(int i=0;i<N;i++)
            for(int j=0;j<M;j++) AT[j][i]=A[i][j];
    }
    @Passed
    public static double[][] transpose(double[][] value)
    {
        double[][] result=new double[value[0].length][value.length];
        for(int i=0,j;i<result.length;i++)
        for(j=0;j<value.length;j++) result[i][j]=value[j][i];
        return result;
    } 
    @Passed
    public static double samePercent(double[][] left, double[][] right, double threshold)
    {
        double sum=0;
        for(int i=0,j;i<left.length;i++)
        for(j=0;j<left[i].length;j++)
        {
            double div=Math.abs(left[i][j]-right[i][j]);
            if(div<threshold) sum++;
        }
        return sum/(left.length*left[0].length);
    }
    public static double samePercent(double[][] left, double[][] right) {return Matrix.samePercent(left, right, 1e-3);}
    
    public static float samePercent(float[][] left, float[][] right, float threshold)
    {
        float sum=0;
        for(int i=0,j;i<left.length;i++)
        for(j=0;j<left[i].length;j++)
        {
            double div=Math.abs(left[i][j]-right[i][j]);
            if(div<threshold) sum++;
        }
        return sum/(left.length*left[0].length);
    }
    public static float samePercent(float[][] left, float[][] right) {return Matrix.samePercent(left, right, 1e-3f);}
    
    
    @Passed
    public static void add(double[][] result, double[][] left, double s)
    {
        int width=result[0].length;
        for(int i=0,j;i<result.length;i++)
        for(j=0;j<width;j++) 
            result[i][j]=left[i][j]+s;
    }
    @Passed
    public static void add(double[][] result, double[][] left, double[][] right)
    {
        int width=result[0].length;
        for(int i=0,j;i<result.length;i++)
        for(j=0;j<width;j++) 
            result[i][j]=left[i][j]+right[i][j];
    }
    @Passed
    public static double[][] add(double[][] left, double[][] right)
    {
        double[][] result=new double[left.length][left[0].length];
        Matrix.add(result, left, right);
        return result;
    }
    @Passed
    public static void rowVectorAdd(double[][] result, double[][] left, double[] line)
    {
        for(int i=0,j;i<result.length;i++)
        for(j=0;j<line.length;j++)
            result[i][j]=left[i][j]+line[j];
    }
    @Passed
    public static void sub(double[][] result, double[][] left, double[][] right)
    {
        int width=result[0].length;
        for(int i=0,j;i<result.length;i++)
        for(j=0;j<width;j++)
            result[i][j]=left[i][j]-right[i][j];
    }
    @Passed
    public static void sub(double[][] result, double[][] left, double k, double[][] right)
    {
        for(int i=0,j;i<result.length;i++)
        for(j=0;j<result[i].length;j++)
            result[i][j]=left[i][j]-k*right[i][j];
    }
    @Passed
    public static double[][] sub(double[][] left, double[][] right)
    {
        double[][] result=new double[left.length][left[0].length];
        Matrix.sub(result, left, right);
        return result;
    }
    @Passed
    public static void rowVectorSub(double[][] result, double[][] left, double[] line)
    {
        for(int i=0,j;i<result.length;i++)
        for(j=0;j<line.length;j++)
            result[i][j]=left[i][j]-line[j];
    }
    @Passed
    public static void abs(double[][] result, double[][] left)
    {
        for(int i=0,j;i<result.length;i++)
        for(j=0;j<result[i].length;j++)
            result[i][j]=(left[i][j]<0? -left[i][j]:left[i][j]);
    }
    @Passed
    public static double[][] abs(double[][] x)
    {
        double[][] result=new double[x.length][x[0].length];
        Matrix.abs(result, x);
        return result;
    }
    @Passed
    public static void multiply(double[][] result, double[][] left, double k)
    {
        for(int i=0,j;i<result.length;i++)
        for(j=0;j<result[i].length;j++)
            result[i][j]*=k;
    }
    @Passed
    public static void multiply(double[][] result, double[][] left, double[][] right)
    {
        int complex=left.length*right.length*right[0].length;
        if(complex>EX_MULTIPLY_THRESHOLD) {ExMatrix.multiplyM(result, left, right);return;}
        else if(complex>OPT_MULTIPLY_THRESHOLD) {ExMatrix.multiply(result, left, right);return;}
        
        double s;
        for(int i=0,j,k;i<result.length;i++)
        {
            for(j=0;j<result[i].length;j++) result[i][j]=left[i][0]*right[0][j];
            for(k=1;k<left[i].length;k++)
                for(s=left[i][k],j=0;j<result[i].length;j++) 
                    result[i][j]+=s*right[k][j];
        }
    }
    @Passed
    public static double[][] multiply(double[][] left, double[][] right)
    {
        double[][] result=new double[left.length][right[0].length];
        Matrix.multiply(result, left, right);
        return result;
    }
    @Passed
    public static void multiplyT(double[][] result, double[][] left, double[][] right)
    {
        int complex=left.length*right.length*right[0].length;
        if(complex>EX_MULTIPLY_THRESHOLD) {ExMatrix.multiplyTM(result, left, right);return;}
        else if(complex>OPT_MULTIPLY_THRESHOLD) {ExMatrix.multiplyT(result, left, right);return;}
        
        double s;
        for(int i=0,j,k;i<result.length;i++)
        {
            for(j=0;j<right.length;j++) result[i][j]=left[i][0]*right[j][0];
            for(k=1;k<left[i].length;k++)
                for(s=left[i][k],j=0;j<right.length;j++) 
                    result[i][j]+=s*right[j][k];
        }
    }
    @Passed
    public static double[][] multiplyT(double[][] left, double[][] right)
    {
        double[][] result=new double[left.length][right.length];
        Matrix.multiplyT(result, left, right);
        return result;
    }
    @Passed
    public static void elementMultiply(double[][] result, double[][] left, double[][] right)
    {
        int width=result[0].length;
        for(int i=0,j;i<result.length;i++)
        for(j=0;j<width;j++)
            result[i][j]=left[i][j]*right[i][j];
    }
    @Passed
    public static double[][] elementMultiply(double[][] left, double[][] right)
    {
        double[][] result=new double[left.length][left[0].length];
        Matrix.elementMultiply(result, left, right);
        return result;
    }
    @Passed
    public static void elementMultiplyLine(double[][] result, double[][] left, double[] right)
    {
        for(int i=0,j;i<result.length;i++)
        for(j=0;j<right.length;j++)
            result[i][j]=left[i][j]*right[j];
    }
    @Passed
    public static void elementDivideLine(double[][] result, double[][] left, double[] right)
    {
        for(int i,j=0;j<right.length;j++)
        for(i=0;i<result.length;i++)
            result[i][j]=left[i][j]/right[j];
    }
    @Passed
    public static double elementSummary(double[][] value)
    {
        double sum=0;
        for(double[] v:value)
        for(double d:v) sum+=d;
        return sum;
    }
    @Passed
    public static double elemntAverage(double[][] value)
    {
        double sum=0;
        for(double[] line:value)
        for(double e:line) sum+=e;
        return sum/(value.length*value[0].length);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Extensive Matrix Manipulation1">
    @Passed
    public static void approximate(double[][] result, double[][] val, double threshold, double standard)
    {
        double d;
        for(int i=0,j;i<result.length;i++)
        for(j=0;j<result.length;j++)
        {
            d=val[i][j]-standard;
            if(d<0) d=-d;
            if(d<=threshold) result[i][j]=standard;
        }
    }
    @Passed
    public static void approximateZero(double[][] val, double threshold)
    {
        double d;
        for(int i=0,j;i<val.length;i++)
        for(j=0;j<val.length;j++)
        {
            d=(val[i][j]<0? -val[i][j]:val[i][j]);
            if(d<=threshold) val[i][j]=0.0;
        }
    }
    @Passed
    public static void approximateZero(double[][] val)
    {
        Matrix.approximateZero(val, 1e-8);
    }
    @Passed
    public static void reciprocal(double[][] result, double[][] val)
    {
        for(int i=0,j;i<result.length;i++)
        for(j=0;j<result[i].length;j++) result[i][j]=1/val[i][j];
    }
    @Passed  
    public double[][] reciprocal(double[][] x)
    {
        double[][] result=new double[x.length][x[0].length];
        Matrix.reciprocal(result, x);
        return result;
    }
    @Passed
    public static void sqrt(double[][] result, double[][] val)
    {
        for(int i=0,j;i<result.length;i++)
        for(j=0;j<result[i].length;j++) result[i][j]=Math.sqrt(val[i][j]);
    }
    @Passed
    public static void momentum(double[][] vdw, double b1,double[][] dw)
    {
        double b2=1-b1;
        for(int i=0,j,width=vdw[0].length;i<vdw.length;i++)
        for(j=0;j<width;j++) 
            vdw[i][j]=b1*vdw[i][j]+b2*dw[i][j];
    }
    @Passed
    public static void rmsProp(double[][] sdw, double b1, double[][] dw)
    {
        double b2=1-b1;
        for(int i=0,j,width=sdw[0].length;i<sdw.length;i++)
        for(j=0;j<width;j++)
            sdw[i][j]=b1*sdw[i][j]+b2*dw[i][j]*dw[i][j];
    }
    /**
     * <pre>
     * Regard each line of Matrix a as a tuple, regard each field as an
     * attribute, find the covariance Matrix for the specific fields.
     * covMat=(A * At)/(a.width-1), but it is approximte to: (A * At)/a.width, 
     * which avoid the case of a.width==1 to let the divisor to be zero.
     * </pre>
     * @param cov
     * @param a 
     */
    public static void covMatrix(double[][] cov, double[][] a)
    {
        double[][] at=Matrix.transpose(a);
        Matrix.multiply(cov, at, a);
        Matrix.multiply(cov, cov, 1.0/a.length);
    }
    public static double[][] covMatrix(double[][] a)
    {
        double[][] cov=new double[a[0].length][a[0].length];
        Matrix.covMatrix(cov, a);
        return cov;
    }
    
    public static void giniForEachLine(double[] result, double[][] value)
    {
        for(int i=0;i<result.length;i++) 
            result[i]=Vector.gini(value[i]);
    }
    public static void entropyForEachLine(double[] result, double[][] value)
    {
        for(int i=0;i<result.length;i++) 
            result[i]=Vector.entropyE(value[i]);
    }
    public static void relu(double[][] result, double[][] val)
    {
        for(int i=0,j,width=result[0].length;i<result.length;i++)
        for(j=0;j<width;j++)
            result[i][j]=(val[i][j]>0? val[i][j]:0);
    }
    public static void sigmoid(double[][] result, double[][] val)
    {
        for(int i=0,j;i<result.length;i++)
        for(j=0;j<result[i].length;j++)
            result[i][j]=1/(1+exp(-val[i][j]));
    }
    public static void unSigmoid(double[][] result, double[][] val)
    {
        for(int i=0,j;i<result.length;i++)
        for(j=0;j<result[i].length;j++)
            result[i][j]=-log(1/val[i][j]-1);
    }
    public static void tanh(double[][] result, double[][] val)
    {
        for(int i=0,j;i<result.length;i++)
        for(j=0;j<result[i].length;j++)
            result[i][j]=1-2/(exp(2*val[i][j])+1);
    }
     public static void unTanh(double[][] result, double[][] val)
    {
        for(int i=0,j;i<result.length;i++)
        for(j=0;j<result[i].length;j++)
            result[i][j]=-0.5*log(1/(1-val[i][j])-1);
    }
    public static void softPlus(double[][] result, double[][] val)
    {
        for(int i=0,j;i<result.length;i++)
        for(j=0;j<result[i].length;j++)
            result[i][j]=log(exp(val[i][j]+1));
    }
    public static void unSoftPlus(double[][] result, double[][] val)
    {
        for(int i=0,j;i<result.length;i++)
        for(j=0;j<result[i].length;j++)
            result[i][j]=log(exp(val[i][j]-1));
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Linear-Algebra">
    /**
     * 
     * @param result
     * @param x 
     */
    @Passed
    public static void toHessenberg(double[][] result, double[][] x)
    {
        double sum, c, h, w;
        double u[]=new double[x.length], p[]=new double[x.length],
               q[]=new double[x.length];
        
        boolean allzero;
        for(int r=0,i,j;r<x.length-2;r++)
        {
            //check whether the diagonal elements of the matrix are 0
            for(allzero=true, i=r+2; i<x.length;i++) 
                if(x[i][r]!=0) {allzero=false;break;}
            if(allzero) continue;
                
            for(sum=0, i=r+1;i<x.length;i++) sum+=x[i][r]*x[i][r];
            c=-Math.signum(x[r+1][r])*Math.sqrt(sum);
            h=c*c-c*x[r+1][r];
            
            for(i=0;i<=r;i++) u[i]=0;
            u[r+1]=x[r+1][r]-c;
            for(i=r+2;i<x.length;i++) u[i]=x[i][r];
            
            for(j=0;j<x.length; p[j]/=h,j++)
            for(p[j]=0,i=0;i<x.length;i++) p[j]+=x[i][j]*u[i];
            
            for(i=0;i<x.length;q[i]/=h,i++)
            for(q[i]=0,j=0;j<x.length;j++) q[i]+=x[i][j]*u[j];
            
            for(sum=0,i=0;i<x.length;i++) sum+=p[i]*u[i];
            
            for(sum/=h, i=0;i<x.length;i++)
            for(w=q[i]-sum*u[i], j=0;j<x.length;j++)
                result[i][j]-=w*u[j]+u[i]*p[j];
        }
    }
    /**
     * <pre>
     * Get all engien vectors from {@code Matrix x}:
     * (1)without detecting duplicate Eigen Values;
     * (2)without doing Normalization; if it's need, 
     * try to use {@link #schmidtNoramlization(double[][], double[][]) }.
     * We use {@link ExMatrix#eigenQRDecomposition(double[][], int, double) } First,
     * if it Fails it will try to use {@link ExMatrix#eigenJacbiCorRotation(double[][], int, double) },
     * All eigen values and vetors is encapsulated in an Engins instance {@link ExMatrix.Eigens}.
     * </pre>
     * @param x
     * @return 
     */
    @Passed
    public static Eigens eigens(double[][] x)
    {
        double e=1e-16;
        int L= x.length*x.length*1000;
        Eigens eg=ExMatrix.eigenQRDecomposition(x, L, e);
        if(eg==null) eg=ExMatrix.eigenJacbiCorRotation(x, L, e);
        return eg;
    }
    /**
     * <pre>
     * Regard each line of the input {@code Matrix a} as an Vector, 
 then convert all of the vectors to unit vectors and assgin valies
 to the input {@code Matrix b}.
     * </pre>
     * @param b
     * @param a 
     */
    @Passed
    public static void unitalize(double[][] b, double[][] a)
    {
        for(int i=0;i<a.length;i++)
            Vector.elementScalarDiv(a[i], Vector.norm(a[i]), b[i]);
    }
    /**
     * {@link #unitalize(double[][], double[][]) }.
     * @param a
     * @return 
     */
    @Passed
    public static double[][] unitalize(double[][] a)   
    {
        double[][] b=new double[a.length][a[0].length];
        Matrix.unitalize(b, a);
        return b;
    }
    /**
     * <pre>
     * regard each line of the input Matrix {@code double[][] a} as a 
     * vector, and do schmidt Normalzation on it;
     * After Normalization, we unitalize all line vectors.
     * as: b1=a1
     *     br=ar - sum([bi, ar]/[bi, bi]*b1) from 1 to r-1
     * </pre>
     * @param b
     * @param a 
     */
    @Passed
    public static void schmidtNoramlization(double[][] b, double[][] a)
    {
        for(int i=0,j,width=a[0].length;i<a.length;i++)
        {
            System.arraycopy(a[i], 0, b[i], 0, width);
            for(j=0;j<i;j++)
            {
                double k=Vector.dot(b[j], a[i])/Vector.dot(b[j], b[j]);
                Vector.elementSub(1, b[i], k, b[j], b[i]);
            }
        }
        Matrix.unitalize(b, b);
    }
    /**
     * {@link #schmidtNoramlization(double[][], double[][]) }
     * @param a
     * @return 
     */
    @Passed
    public static double[][] schmidtNormalization(double[][] a)
    {
        double[][] b=new double[a.length][a[0].length];
        Matrix.schmidtNoramlization(b, a);
        return b;
    }
    public static void zeroLine(double[][] b, double[][] a, int k)
    {
        for(int j=0;j<a[k].length;j++) b[k][j]=0;
    }
    public static double[][] zeroLine(double[][] a, int k)
    {
        int width=a[0].length;
        double[][] b=new double[a.length][width];
        for(int i=0;i<k;i++) System.arraycopy(a[i], 0, b[i], 0, width);
        for(int i=k+1;i<a.length;i++) System.arraycopy(a[i], 0, b[i], 0, width);
        return b;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Matrix - Row Vector operation">
    public static float[][] rowVectorAffine(float[][] X, float[] A, float[] B)
    {
        int height = X.length, width = A.length;
        float[][] Y = new float[height][width];
        for(int i=0; i<height; i++)
            for(int j=0; j<width; j++)
                Y[i][j] = A[j] * X[i][j] + B[j];
        return Y;
    }
    
    public static void rowVectorAdd(float[][] A, float[] V, float[][] B)
    {
        for(int i=0;i<B.length;i++)
            for(int j=0;j<V.length;j++) B[i][j]=A[i][j]+V[j];
    }
    public static void rowVectorAdd(float alpha, float[][] A, float beta, float[] V, float[][] B)
    {
        for(int i=0;i<B.length;i++)
            for(int j=0;j<V.length;j++) B[i][j]=alpha*A[i][j] + beta*V[j];
    }
    public static void rowVectorBinomial(float[][] X1, float[] X2,
            float k11, float k12, float k22,
            float k1, float k2, float C,
            float[][] Y, int height, int width)
    {
        for(int i=0; i<height; i++)
           for(int j=0; j<width; j++)
           {
               float x1 = X1[i][j];
               float x2 = X2[j];
               Y[i][j] = k11*(x1*x1) + k12*(x1*x2) + k22*(x2*x2) +
                       k1*x1 + k2*x2 + C;
           }
    }
    
     public static void rowVectorAddSquare(float[][] A, float[] V, float[][] B)
    {
        for(int i=0;i<B.length;i++)
            for(int j=0;j<V.length;j++) B[i][j]=A[i][j] + V[j]*V[j];
    }
    public static void rowVectorAddSquare(float alpha, float[][] A, float beta, float[] V, float[][] B)
    {
        for(int i=0;i<B.length;i++)
            for(int j=0;j<V.length;j++) B[i][j]=alpha*A[i][j] + beta*V[j]*V[j];
    }
    public static void rowVectorWeightedAdd(float[][] X, float[] A, float[][] Y, float[][] Z)
    {
        for(int i=0;i<X.length;i++)
            for(int j=0;j<A.length;j++)
                Z[i][j] = X[i][j] + A[j]*Y[i][j];
    }
    
    public static void rowVectorSub(float[][] A, float[] V, float[][] B)
    {
        for(int i=0;i<B.length;i++)
            for(int j=0;j<V.length;j++) B[i][j]=A[i][j]-V[j];
    }
    public static void rowVectorSub(float alpha, float[][] A, float beta, float[] V, float[][] B)
    {
        for(int i=0;i<B.length;i++)
            for(int j=0;j<V.length;j++) B[i][j]=alpha*A[i][j] - beta*V[j];
    }
    public static void rowVectorMul(float[][] A, float[] V, float[][] B)
    {
        for(int i=0;i<B.length;i++)
            for(int j=0;j<V.length;j++) B[i][j]=A[i][j]*V[j];
    }
     public static void rowVectorMul(float a1, float[][] A, float b1,
             float a2, float[] V, float b2, float[][] B)
    {
        for(int i=0;i<B.length;i++)
            for(int j=0;j<V.length;j++) B[i][j]=(a1*A[i][j] + b1)*(a2*V[j] + b2);
    }
    public static void rowVectorDiv(float[][] A, float[] V, float[][] B)
    {
        for(int i=0;i<B.length;i++)
            for(int j=0;j<V.length;j++) B[i][j]=A[i][j]/V[j];
    }
    public static void rowVectorDiv(float a1, float[][] A, float b1,
            float a2, float[] V, float b2, float[][] B)
    {
        for(int i=0;i<B.length;i++)
            for(int j=0;j<V.length;j++) B[i][j]=(a1*A[i][j] + b1)/(a2*V[j] + b2);
    }
    
    public static void batchNorm(float[][] X, 
            float[] X_mean, float[] X_square_mean,
            float[] A, float[] B,
            float[][] Y, int N, int M)
    {
        for(int i=0;i<N;i++)
            for(int j=0;j<M;j++)
            {
                float X_var = X_square_mean[j] - X_mean[j]*X_mean[j];
                Y[i][j] = (X[i][j] - X_mean[j]) / ((float)Math.sqrt(X_var) + 1e-5f);
                Y[i][j] = A[j] * Y[i][j] + B[j];
            }
    }
    
    public static void batchNorm_deltaX1(float[][] deltaY, //affined = false
            float[][] X,
            float[] X_mean,
            float[] X_square_mean, float eps,
            float[][] deltaX, int N, int M)
    {
        float[] X_std = new float[M];
        for(int i=0; i<M; i++) {
            float mean = X_mean[i];
            float smean = X_square_mean[i];
            X_std[i] = (float) Math.sqrt(smean - mean*mean + eps);
        }
        
        float[][] dX1 = new float[N][M];
        for(int i=0; i<N; i++)
        for(int j=0; j<M; j++) {
            dX1[i][j] = deltaY[i][j] / X_std[j];
        }
        
        float[] dX2 = new float[M];
        for(int i=0; i<N; i++)
        for(int j=0; j<M; j++)
        {
            float dy = deltaY[i][j];
            float x = X[i][j];
            float mean = X_mean[j];
            float smean = X_square_mean[j];
            float std = X_std[j];
            
            dX2[j] += dy * (x*mean - smean - eps) / (std*std*std);
        }
        
        float[] dX3 = new float[M];
        for(int i=0; i<N; i++)
        for(int j=0; j<M; j++)
        {
            float dy = deltaY[i][j];
            float x = X[i][j];
            float mean = X_mean[j];
            float std = X_std[j];
            dX3[j] += -0.5f * dy * (x - mean) / (std * std * std);
        }
        
        for(int i=0; i< N; i++)
        for(int j=0; j< M; j++) {
            deltaX[i][j] = dX1[i][j] + dX2[j] / N + dX3[j] * (2 * X[i][j] / N);
        }
    }
    
    public static void batchNorm_deltaX1(float[][] deltaY, //affined = false
            float[][] X,
            float[] X_mean,
            float[] X_square_mean, float eps,
            float[] A,
            float[][] deltaX, int N, int M)
    {
        float[] X_std = new float[M];
        for(int i=0; i<M; i++) {
            float mean = X_mean[i];
            float smean = X_square_mean[i];
            X_std[i] = (float) Math.sqrt(smean - mean*mean + eps);
        }
        
        double[][] dX1 = new double[N][M];
        for(int i=0; i<N; i++)
        for(int j=0; j<M; j++) {
            dX1[i][j] = (double)deltaY[i][j] * A[j] / X_std[j];
        }
        
        double[] dX2 = new double[M];
        for(int i=0; i<N; i++)
        for(int j=0; j<M; j++)
        {
            double dy = deltaY[i][j];
            float x = X[i][j];
            float mean = X_mean[j];
            float smean = X_square_mean[j];
            float std = X_std[j];
            
            dX2[j] += dy * A[j] * (x*mean - smean - eps) / (std*std*std);
        }
        
        float[] dX3 = new float[M];
        for(int i=0; i<N; i++)
        for(int j=0; j<M; j++)
        {
            float dy = deltaY[i][j];
            float x = X[i][j];
            float mean = X_mean[j];
            float std = X_std[j];
            dX3[j] += -0.5f * dy * A[j] * (x - mean) / (std * std * std);
        }
        
        for(int i=0; i< N; i++)
        for(int j=0; j< M; j++) {
            deltaX[i][j] = (float) (dX1[i][j] + dX2[j] / N + dX3[j] * (2 * X[i][j] / N));
        }
    }
    
    public static void batchNorm_deltaX2(float[][] deltaY, //affined = false
            float[][] X,
            float[] X_mean,
            float[] X_square_mean, float eps,
            float[][] deltaX, int N, int M)
    {
        float[] X_std = new float[M];
        for(int i=0; i<M; i++) {
            float mean = X_mean[i];
            float smean = X_square_mean[i];
            X_std[i] = (float) Math.sqrt(smean - mean*mean + eps);
        }
        
        float[] dX1 = new float[M];
        for(int i=0; i<N; i++)
        for(int j=0; j<M; j++)
        {
            float dy = deltaY[i][j];
            float x = X[i][j];
            float mean = X_mean[j];
            float smean = X_square_mean[j];
            
            dX1[j] += dy * (x*mean - smean - eps);
        }
        
        float[] dX2 = new float[M];
        for(int i=0; i<N; i++)
        for(int j=0; j<M; j++)
        {
            float dy = deltaY[i][j];
            float x = X[i][j];
            float mean = X_mean[j];
            
            dX2[j] += dy * (x - mean);
        }
        
        for(int i=0; i< N; i++)
        for(int j=0; j< M; j++) 
        {
            float std = X_std[j];
            float deltaXp1 = dX1[j];
            float deltaXp2 = dX2[j];
            
            deltaX[i][j] = (deltaY[i][j] + (deltaXp1 - deltaXp2*X[i][j]) / (N * std * std)) / std;
        }
    }
    
    public static void layerNorm(float[][] X, 
            float[] X_mean, float[] X_square_mean,
            float[] A, float[] B,
            float[][] Y, int N, int M)
    {
        for(int i=0;i<N;i++)
        {
            float x_mean = X_mean[i];
            float x_square_mean = X_square_mean[i];
            float x_stddev = (float) Math.sqrt(x_square_mean - x_mean*x_mean + 1e-5f);
            for(int j=0;j<M;j++)
            {
                Y[i][j] = (X[i][j] - x_mean) / x_stddev;
                Y[i][j] = A[j] * Y[i][j] + B[j];
            }
        }
    }
    
    public static void layerNorm_deltaX1(float[][] deltaY, //affined = false
            float[][] X,
            float[] X_mean,
            float[] X_square_mean, float eps,
            float[][] deltaX, int N, int M)
    {
        float[] X_std = new float[N];
        for(int i=0; i<N; i++) {
            float mean = X_mean[i];
            float smean = X_square_mean[i];
            X_std[i] = (float) Math.sqrt(smean - mean*mean + eps);
        }
        
        float[][] dX1 = new float[N][M];
        for(int i=0; i<N; i++)
        for(int j=0; j<M; j++) {
            dX1[i][j] = deltaY[i][j] / X_std[i];
        }
        
        float[] dX2 = new float[N];
        for(int i=0; i<N; i++)
        for(int j=0; j<M; j++)
        {
            float dy = deltaY[i][j];
            float x = X[i][j];
            float mean = X_mean[i];
            float smean = X_square_mean[i];
            float std = X_std[i];
            
            dX2[i] += dy * (x*mean - smean - eps) / (std*std*std);
        }
        
        float[] dX3 = new float[N];
        for(int i=0; i<N; i++)
        for(int j=0; j<M; j++)
        {
            float dy = deltaY[i][j];
            float x = X[i][j];
            float mean = X_mean[i];
            float std = X_std[i];
            
            dX3[i] += -0.5f * dy * (x - mean) / (std * std * std);
        }
        
        for(int i=0; i< N; i++)
        for(int j=0; j< M; j++) {
            deltaX[i][j] = dX1[i][j] + dX2[i] / M + dX3[i] * (2 * X[i][j] / M);
        }
    }
    
    public static void layerNorm_deltaX1(float[][] deltaY, //affined = false
            float[][] X,
            float[] X_mean,
            float[] X_square_mean, float eps,
            float[] A,
            float[][] deltaX, int N, int M)
    {
        float[] X_std = new float[N];
        for(int i=0; i<N; i++) {
            float mean = X_mean[i];
            float smean = X_square_mean[i];
            X_std[i] = (float) Math.sqrt(smean - mean*mean + eps);
        }
        
        float[][] dX1 = new float[N][M];
        for(int i=0; i<N; i++)
        for(int j=0; j<M; j++) {
            dX1[i][j] = deltaY[i][j] * A[j] / X_std[i];
        }
        
        float[] dX2 = new float[N];
        for(int i=0; i<N; i++)
        for(int j=0; j<M; j++)
        {
            float dy = deltaY[i][j];
            float x = X[i][j];
            float mean = X_mean[i];
            float smean = X_square_mean[i];
            float std = X_std[i];
            
            dX2[i] += dy * A[j] * (x*mean - smean - eps) / (std*std*std);
        }
        
        float[] dX3 = new float[N];
        for(int i=0; i<N; i++)
        for(int j=0; j<M; j++)
        {
            float dy = deltaY[i][j];
            float x = X[i][j];
            float mean = X_mean[i];
            float std = X_std[i];
            
            dX3[i] += -0.5f * dy * A[j] * (x - mean) / (std * std * std);
        }
        
        for(int i=0; i< N; i++)
        for(int j=0; j< M; j++) {
            deltaX[i][j] = dX1[i][j] + dX2[i] / M + dX3[i] * (2 * X[i][j] / M);
        }
    }
    //</editor-fold>
     
    //<editor-fold defaultstate="collapsed" desc="Matrix - Field Vector Operation">
    public static void fieldVectorBinomial(float[][] X1, float[] X2,
            float k11, float k12, float k22,
            float k1, float k2, float C,
            float[][] Y, int height, int width)
    {
        for(int i=0; i<height; i++)
           for(int j=0; j<width; j++)
           {
               float x1 = X1[i][j];
               float x2 = X2[i];
               Y[i][j] = k11*(x1*x1) + k12*(x1*x2) + k22*(x2*x2) +
                       k1*x1 + k2*x2 + C;
           }
    }
    public static void fieldVectorDiv(
            float alpha1, float[][] X1, float beta1,
            float alpha2, float[]   X2, float beta2,
            float gamma,
            float[][] Y, int height, int width)
    {
        for(int i=0; i<height; i++)
           for(int j=0; j<width; j++)
           {
               float x1 = alpha1*X1[i][j] + beta1;
               float x2 = alpha2*X2[i]    + beta2;
               Y[i][j] = x1 / x2 + gamma; 
           }
    }
     
    public static void fieldVectorAdd(float[][] A, float[] V, float[][] B)
    {
        for(int i=0;i<B.length;i++)
            for(int j=0;j<B[i].length;j++) B[i][j]=A[i][j]+V[i];
    }
    public static void fieldVectorSub(float[][] A, float[] V, float[][] B)
    {
        for(int i=0;i<B.length;i++)
            for(int j=0;j<B[i].length;j++) B[i][j]=A[i][j]-V[i];
    }
    public static void fieldVectorMul(float[][] A, float[] V, float[][] B)
    {
        for(int i=0;i<B.length;i++)
            for(int j=0;j<B[i].length;j++) B[i][j]=A[i][j]*V[i];
    }
    public static void fieldVectorDiv(float[][] A, float[] V, float[][] B)
    {
        for(int i=0;i<B.length;i++)
            for(int j=0;j<B[i].length;j++) B[i][j]=A[i][j]/V[i];
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Matrix - Reduce">
    //<editor-fold defaultstate="collapsed" desc="Row Vector reduce">
    public static void sumOfEachRow(float[][] A, float[] V)
    {
        int N=A.length, M=A[0].length;
        for(int i=0;i<N;i++)
        {
            V[i]=A[i][0];
            for(int j=1;j<M;j++) V[i]+=A[i][j];
        }
    }
     public static void sumOfEachRow(float alpha, float[][] A, float beta, float[] V)
    {
        int N=A.length, M=A[0].length;
        for(int i=0;i<N;i++)
        {
            V[i] = alpha*A[i][0] + beta;
            for(int j=1;j<M;j++) V[i] += alpha*A[i][j] + beta;
        }
    }
    public static void squareSumOfEachRow(float[][] A, float[] V)
    {
        int N=A.length, M=A[0].length;
        for(int i=0;i<N;i++)
        {
            V[i]=A[i][0] * A[i][0];
            for(int j=1;j<M;j++) V[i] += A[i][j] * A[i][j];
        }
    }
    public static void squareSumOfEachRow(float[][] A, float alpha, float[] V)
    {
        int N=A.length, M=A[0].length;
        for(int i=0;i<N;i++)
        {
            float v = A[i][0] + alpha; V[i] = v*v;
            for(int j=1;j<M;j++) {v = A[i][j] + alpha; V[i] += v*v;}
        }
    }
     public static void squareSumOfEachRow(float alpha, float[][] A, float beta, float[] V)
    {
        int N=A.length, M=A[0].length;
        for(int i=0;i<N;i++)
        {
            float v = alpha*A[i][0] + beta; V[i] = v*v;
            for(int j=1;j<M;j++) {v = alpha*A[i][j] + beta; V[i] += v*v;}
        }
    }
    public static void absSumOfEachRow(float[][] A, float[] V)
    {
        int N=A.length, M=A[0].length;
        for(int i=0;i<N;i++)
        {
            V[i] = Math.abs(A[i][0]);
            for(int j=1;j<M;j++) V[i] += Math.abs(A[i][j]);
        }
    }
    public static void absSumOfEachRow(float[][] A, float alpha, float[] V)
    {
        int N=A.length, M=A[0].length;
        for(int i=0;i<N;i++)
        {
            V[i] = Math.abs(A[i][0] + alpha);
            for(int j=1;j<M;j++) V[i] += Math.abs(A[i][j] + alpha);
        }
    }
    
     public static void row_linear(float[][] A, 
            float alpha, float beta, float[] V)
    {
        int N = A.length, M = A[0].length;
        for(int i = 0; i < N; i++)
        {
            V[i] = 0.0f;
            for(int j = 0; j < M; j++)  V[i] += alpha*A[i][j] + beta;
        }
    }
    
    public static void row_quadratic(float[][] A, 
            float alpha, float beta, float gamma, float[] V)
    {
        int N = A.length, M = A[0].length;
        for(int i = 0; i < N; i++)
        {
            V[i] = 0.0f;
            for(int j = 0; j < M; j++) {
                float a = A[i][j];
                V[i] += alpha*(a*a) + beta*a + gamma;
            }
        }
    }
    
    public static float[] row_mean(float[][] X) //[height, width] -> [height]
    {
        int height = X.length, width = X[0].length;
        float[] Y = new float[height];
        float alpha = 1.0f / width;
        
        for(int i=0; i<height; i++)
        for(int j=0; j<width; j++) {
            Y[i] += alpha * X[i][j];
        }
        return Y;
    }
    
    public static float[] row_squareMean(float[][] X) //[height, width] -> [height]
    {
        int height = X.length, width = X[0].length;
        float[] Y = new float[height];
        float alpha = 1.0f / width;
        
        for(int i=0; i<height; i++)
        for(int j=0; j<width; j++) {
            float x = X[i][j];
            Y[i] += alpha * x*x;
        }
        return Y;
    }
    
    public static float[][] row_var(float[][] X) {
        float[] mean = row_mean(X);
        float[] squareMean = row_squareMean(X);
        float[] var = new float[mean.length];
        
        for(int i=0; i<var.length; i++) {
            float m = mean[i];
            float sm = squareMean[i];
            var[i] = sm - m*m;
        }
        return new float[][] { var, mean, squareMean };
    }
    
    public static float[][] row_stddev(float[][] X) {
        float[] mean = row_mean(X);
        float[] squareMean = row_squareMean(X);
        float[] stddev = new float[mean.length];
        
        for(int i=0; i<stddev.length; i++) {
            float m = mean[i];
            float sm = squareMean[i];
            stddev[i] = (float) Math.sqrt(sm - m*m);
        }
        return new float[][] { stddev, mean, squareMean };
    }
    
    public static void maxValueOfEachRow(float[][] A, float[] V)
    {
        int N=A.length, M=A[0].length;
        for(int i=0;i<N;i++)
        {
            V[i]=A[i][0];
            for(int j=1;j<M;j++) if(V[i]<A[i][j]) V[i]=A[i][j];
        }
    }
    public static void minValueOfEachRow(float[][] A, float[] V)
    {
        int N=A.length, M=A[0].length;
        for(int i=0;i<N;i++)
        {
            V[i]=A[i][0];
            for(int j=1;j<M;j++) if(V[i]>A[i][j]) V[i]=A[i][j];
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Field Vector reduce">
    public static void sumOfEachField(float[][] A, float[] V)
    {
        int N=A.length, M=A[0].length;
        for(int j=0;j<M;j++)
        {
            V[j]=A[0][j];
            for(int i=1;i<N;i++) V[j]+=A[i][j];
        }
    }
    
    public static void field_max(float[][] A, float[] V)
    {
        int N = A.length, M = A[0].length;
        for(int j=0;j<M;j++)
        {
            V[j] = - Float.MAX_VALUE;
            for(int i = 0; i < N; i++)  V[j] = Math.max(V[j], A[i][j]);
        }
    }
    public static void minOfEachField(float[][] A, float[] V)
    {
        int N = A.length, M = A[0].length;
        for(int j=0;j<M;j++)
        {
            V[j] = Float.MAX_VALUE;
            for(int i = 0; i < N; i++)  V[j] = Math.min(V[j], A[i][j]);
        }
    }
   
    public static float[] field_mean(float[][] X) {//[height, width] -> [width]
        int height = X.length, width = X[0].length;
        float alpha = 1.0f / height;
        
        float[] Y = new float[width];
        for(int i=0; i<height; i++)
        for(int j=0; j<width; j++)
            Y[j] += alpha * X[i][j];
        return Y;
    }
    
    public static float[] field_squareMean(float[][] X) {
        int height = X.length, width = X[0].length;
        float alpha = 1.0f / height;
        
        float[] Y = new float[width];
        for(int i=0; i<height; i++)
        for(int j=0; j<width; j++) {
            float x = X[i][j];
            Y[j] += alpha * x*x;
        }
        return Y;
    }
    
    public static float[][] field_var(float[][] X) {
        float[] mean = field_mean(X);
        float[] squareMean = field_squareMean(X);
        float[] var = new float[mean.length];
        
        for(int i=0; i<var.length; i++) {
            float m = mean[i];
            float sm = squareMean[i];
            var[i] = sm - m*m;
        }
        return new float[][] { var, mean, squareMean };
    }
    
    public static float[][] field_stddev(float[][] X) {
        float[] mean = field_mean(X);
        float[] squareMean = field_squareMean(X);
        float[] stddev = new float[mean.length];
        
        for(int i=0; i<stddev.length; i++) {
            float m = mean[i];
            float sm = squareMean[i];
            stddev[i] = (float) Math.sqrt(sm - m*m);
        }
        return new float[][] {stddev, mean, squareMean };
    }
            
    public static void field_linear(float[][] A, 
            float alpha, float beta, float[] V)
    {
        int N = A.length, M = A[0].length;
        for(int j = 0; j < M; j++) {
            V[j] = 0;
            for(int i=0; i<N; i++)  V[j] += alpha* A[i][j] + beta;
        }
    }
    
    public static void field_mean(float[][] A, float[] V)
    {
        int N = A.length, M = A[0].length;
        for(int j = 0; j < M; j++) {
            V[j] = 0;
            for(int i=0; i<N; i++) V[j] += A[i][j];
            V[j] /= N;
        }
    }
    
    public static void field_squareMean(float[][] A, float[] V)
    {
        int N = A.length, M = A[0].length;
        for(int j = 0; j < M; j++) {
            V[j] = 0;
            for(int i=0; i<N; i++) {
                float a = A[i][j];
                V[j] += a*a;
            }
            V[j] /= N;
        }
    }
    
    public static void field_linear2(float[][] X1,  float[][] X2,
            float alpha, float beta, float gamma, float[] V)
    {
        int N = X1.length, M = X1[0].length;
        for(int j = 0; j < M; j++)
        {
            V[j] = 0;
            for(int i=0; i<N; i++) {
                V[j] += alpha* X1[i][j] + beta*X2[i][j] + gamma;
            }
        }
    }
    
    public static void field_quadratic(float[][] A, 
            float alpha, float beta, float gamma, float[] V)
    {
        int N = A.length, M = A[0].length;
        for(int j = 0; j < M; j++)
        {
            V[j] = 0;
            for(int i=0; i<N; i++) {
                float a = A[i][j];
                V[j] += alpha*a*a + beta*a + gamma;
            }
        }
    }
    public static void field_quadratic2(float[][] X1, float[][] X2, 
            float k11, float k12, float k22, float k1, float k2, float C, float[] V)
    {
        int N = X1.length, M = X1[0].length;
        for(int j = 0; j < M; j++)
        {
            V[j] = 0;
            for(int i=0; i<N; i++) {
                float x1 = X1[i][j];
                float x2 = X2[i][j];
                V[j] += k11*(x1*x1) + k12*(x1*x2) + k22*(x2*x2) + k1*x1 + k2*x2 + C;
            }
        }
    }
    
    public static void batchNorm_deltaA(float[][] deltaY, float[][] X,
            float[] X_mean, float[] X_square_mean, float e,
            int N, int M, 
            float[] deltaA)
    {
        float[][] A = new float[N][M];
        for(int i=0; i<N; i++)
            for(int j=0;j<M;j++)
            {
                float X_var = X_square_mean[j] - X_mean[j]*X_mean[j];
                float X_norm = (X[i][j] - X_mean[j] + 1e-9f) / ((float)Math.sqrt(X_var) + 1e-9f);
                A[i][j] = deltaY[i][j] * X_norm;
            }
        Matrix.sumOfEachField(A, deltaA);
    }
    
    public static void batchNorm_deltaA(float[][] deltaY, float[][] Y,
            float[] A, float[] B,
            int N, int M, 
            float[] deltaA)
    {
        for(int j=0; j<M; j++)
        {
            deltaA[j] = 0;
            for(int i=0; i<N; i++)
            {
                float X_norm = (Y[i][j] - B[j]) / (A[j]);
                deltaA[j] += deltaY[i][j] * X_norm;
            }
        }
    }
    
    public static void layerNorm_deltaA(float[][] deltaY, float[][] X,
            float[] X_mean, float[] X_square_mean, float e,
            int N, int M, 
            float[] deltaA)
    {
        float[][] A = new float[N][M];
        
        for(int i=0;i<N;i++)
        {
            float X_var = (float)Math.sqrt(X_square_mean[i] - X_mean[i]*X_mean[i]) + e;
            for(int j=0;j<M;j++)
            {
                float X_norm = (X[i][j] - X_mean[i]) / X_var;
                A[i][j] = deltaY[i][j] * X_norm;
            }
        }
        Matrix.sumOfEachField(A, deltaA);
    }
    
   public static void sumOfEachField(float alpha, float[][] A, float beta, float[] V)
    {
        int N=A.length, M=A[0].length;
        for(int j=0;j<M;j++)
        {
            V[j] = (alpha*A[0][j] + beta);
            for(int i=1;i<N;i++) V[j] += (alpha*A[i][j] + beta);
        }
    }
    
    public static void squareSumOfEachField(float[][] A, float[] V)
    {
        int N=A.length, M=A[0].length;
        for(int j=0;j<M;j++)
        {
            V[j]=A[0][j]*A[0][j];
            for(int i=1;i<N;i++) V[j]+=A[i][j]*A[i][j];
        }
    }
    public static void squareSumOfEachField(float alpha, float[][] A, float beta, float[] V)
    {
        int N=A.length, M=A[0].length;
        for(int j=0;j<M;j++)
        {
            float v = alpha*A[0][j] + beta;
            V[j] = v*v;
            for(int i=1;i<N;i++)
            {
                v = alpha*A[i][j] + beta;
                V[j] += v*v;
            }
        }
    }
    
    public static void maxValueOfEachField(float[][] A, float[] V)
    {
        int N=A.length, M=A[0].length;
        for(int j=0;j<M;j++)
        {
            V[j]=A[0][j];
            for(int i=1;i<N;i++) if(V[j]<A[i][j]) V[j]=A[i][j];
        }
    }
    public static void minValueOfEachField(float[][] A, float[] V)
    {
        int N=A.length, M=A[0].length;
        for(int j=0;j<M;j++)
        {
            V[j]=A[0][j];
            for(int i=1;i<N;i++) if(V[j]>A[i][j]) V[j]=A[i][j];
        }
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Matrix-Splice-Function">
    public static double[][] toDoubleArray2D(Object[] vars)//pass
    {
        double[][] r=null;
        if(vars==null) throw new NullPointerException();
        List<double[]> list=new LinkedList<>();
        double[][] var;
        
        for(int i=0;i<vars.length;i++)
        if(vars[i] instanceof double[]) list.add((double[]) vars[i]);
        else if(vars[i] instanceof double[][])
        {
            var=(double[][]) vars[i];
            for(int j=0;j<var.length;j++) list.add(var[j]);
        }
        
        int index=0;
        r=new double[list.size()][];
        for(double[] cur:list) r[index++]=cur;
        return r;
    }
    public static double[][] toDoubleArray2D(Collection vars)//pass
    {
        double[][] r=null;
        if(vars==null) throw new NullPointerException();
        List<double[]> list=new LinkedList<>();
        double[][] var;
        
        for(Object o:vars)
        if(o instanceof double[]) list.add((double[]) o);
        else if(o instanceof double[][])
        {
            var=(double[][]) o;
            for(int j=0;j<var.length;j++) list.add(var[j]);
        }
        
        int index=0;
        r=new double[list.size()][];
        for(double[] cur:list) r[index++]=cur;
        return r;
    }
    public static int[][] toIntArray2D(Object[] vars)//pass
    {
        int[][] r=null;
        if(vars==null) throw new NullPointerException();
        List<int[]> list=new LinkedList<>();
        int[][] var;
        
        for(int i=0;i<vars.length;i++)
        if(vars[i] instanceof int[]) list.add((int[]) vars[i]);
        else if(vars[i] instanceof int[][]) 
        {
            var=(int[][]) vars[i];
            for(int j=0;j<var.length;j++) list.add(var[j]);
        }
        
        int index=0;
        r=new int[list.size()][];
        for(int[] cur:list) r[index++]=cur;
        return r;
    }
    public static int[][] toIntArray2D(Collection vars)//pass
    {
        int[][] r=null;
        if(vars==null) throw new NullPointerException();
        List<int[]> list=new LinkedList<>();
        int[][] var;
        
        for(Object o:vars)
        if(o instanceof int[]) list.add((int[]) o);
        else if(o instanceof int[][]) 
        {
            var=(int[][]) o;
            for(int j=0;j<var.length;j++) list.add(var[j]);
        }
        
        int index=0;
        r=new int[list.size()][];
        for(int[] cur:list) r[index++]=cur;
        return r;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Matrix-Creator">
    //<editor-fold defaultstate="collapsed" desc="Common-Creator">
    /**
     * <pre>
     * give you an unit matrix with a specific dimension {@code size}.
     * eyeDouble(2)={1,0}
     *              {0,1}
     * </pre>
     * @param size
     * @return 
     */
    @Passed
    public static double[][] eyeDouble(int size)
    {
        double[][] eye=new double[size][size];
        for(int i=0;i<size;i++) eye[i][i]=1;
        return eye;
    }
    /**
     * <pre>
     * give you an unit matrix with a specific dimesion {@code height, width},
     * the element on the diagnoal of the Matrix is 1.
     * eyeDouble(2,3)={1,0,0}   eyeDouble(3,2)={1,0}
     *                {0,1,0}                 ={0,1}
     *                                         {0,0}
     * </pre>
     * @param height
     * @param width
     * @return 
     */
    @Passed
    public static double[][] eyeDouble(int height, int width)
    {
        double[][] eye=new double[height][width];
        int len=(height<width? height: width);
        for(int i=0;i<len;i++) eye[i][i]=1;
        return eye;
    }
     @Passed
    public static double[][] eyeDouble(double[] dia)
    {
        double[][] eye=new double[dia.length][dia.length];
        for(int i=0;i<dia.length;i++) eye[i][i]=dia[i];
        return eye;
    }
    /**
     * give you an unit matrix with a specific dimension, and set the
     * elements in {@code double dia} for each on the diagnoal.
     * @param height
     * @param width
     * @param dia contains the diagnoal elements
     * @return 
     */
    @Passed
    public static double[][] eyeDouble(int height, int width, double[] dia)
    {
        if(dia==null) return eyeDouble(height, width);
        if(height<dia.length) height=dia.length;
        if(width<dia.length) width=dia.length;
        double[][] eye=new double[height][width];
        if(height>width) height=width;
        if(height>dia.length) height=dia.length;
        for(int i=0;i<height;i++) eye[i][i]=dia[i];
        return eye;
    }
    public static int[][] intMatrix(int height, int width, int val)
    {
        int[][] arr=new int[height][width];
        for(int i=0,j;i<height;i++)
        for(j=0;j<width;j++) arr[i][j]=val;
        return arr;
    }
    public static int[][] intMatrix(double[][] x)
    {
        int[][] arr=new int[x.length][x[0].length];
        for(int i=0,j;i<arr.length;i++)
        for(j=0;j<arr[i].length;j++) arr[i][j]=(int) x[i][j];
        return arr;
    }
    public static double[][] doubleMatrix(int height, int width, double val)
    {
        double[][] arr=new double[height][width];
        for(int i=0,j;i<height;i++)
        for(j=0;j<width;j++) arr[i][j]=val;
        return arr;
    }
    public static double[][] doubleMatrix(int[][] x)
    {
        double[][] arr=new double[x.length][x[0].length];
        for(int i=0,j;i<arr.length;i++)
        for(j=0;j<arr[i].length;j++) arr[i][j]=x[i][j];
        return arr;
    }
    public static double[][] doubleMatrix(float[][] x)
    {
        double[][] arr=new double[x.length][x[0].length];
        for(int i=0,j;i<arr.length;i++)
        for(j=0;j<arr[i].length;j++) arr[i][j]=x[i][j];
        return arr;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Mathematic-Function:exRandom:int">
    @Passed
    public static int[][] randomIntMatrix(int height, int width)
    {
        return Lang.exRandom().nextIntMatrix(height, width);
    }
    @Passed
    public static int[][] randomIntMatrix(int[][] v)
    {
        return Lang.exRandom().nextIntMatrix(v);
    }
    @Passed
    public static int[][] randomIntMatrix(int width, int height, int max)
    {
        return Lang.exRandom().nextIntMatrix(width, height, 0,  max);
    }
    @Passed
    public static int[][] randomIntMatrix(int[][] v, int max)
    {
        return Lang.exRandom().nextIntMatrix(v, 0, max);
    }
    @Passed
    public static int[][] randomIntMatrix(int width, int height, int max ,int min)
    {
        return Lang.exRandom().nextIntMatrix(width, height, max, min);
    }
    @Passed
    public static int[][] randomIntMatrix(int[][] v, int max, int min)
    {
        return Lang.exRandom().nextIntMatrix(v, min, max);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Mathematic-Function:exRandom:double">
    @Passed
    public static double[][] randomDoubleMatrix(int height, int width)
    {
        return Lang.exRandom().nextDoubleMatrix(height, width);
    }
    @Passed
    public static double[][] randomGaussianMatrix(int height ,int width)
    {
        return Lang.exRandom().nextGaussianMatrix(height, width);
    }
    @Passed
    public static double[][] randomDoubleMatrix(double[][] v)
    {
        return Lang.exRandom().nextDoubleMatrix(v);
    }
    @Passed
    public static double[][] randomGaussianMatrix(double[][] v)
    {
        return Lang.exRandom().nextGaussianMatrix(v);
    }
    @Passed
    public static double[][] randomDoubleMatrix(int height, int width, double max)
    {
        return Lang.exRandom().nextDoubleMatrix(height, width, max);
    }
    @Passed
    public static double[][] randomDoubleMatrix(double[][] v, double max)
    {
        return Lang.exRandom().nextDoubleMatrix(v, 0, max);
    }
    @Passed
    public static double[][] randomDoubleMatrix(int height, int width, double min ,double max)
    {
        return Lang.exRandom().nextDoubleMatrix(height, width, min, max);
    }
    @Passed
    public static double[][] randomGaussianMatrix(int height, int width, double min, double max)
    {
        return Lang.exRandom().nextGaussianMatrix(height, width, min, max);
    }
    @Passed
    public static double[][] randomDoubleMatrix(double[][] v, double min, double max)
    {
        return Lang.exRandom().nextDoubleMatrix(v, min, max);
    }
    @Passed
    public static double[][] randomGaussianMatrix(double[][] v, double min, double max)
    {
        return Lang.exRandom().nextGaussianMatrix(v, min, max);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Mathematic-Function:exRandom:float">
    @Passed
    public static float[][] randomFloatMatrix(int height, int width)
    {
        return Lang.exRandom().nextFloatMatrix(height, width);
    }
    @Passed
    public void nextFloatMatrix(float[][] v)
    {
        Lang.exRandom().nextFloatMatrix(v);
    }
    @Passed
    public float[][] nextFloatMatrix(int height, int width, float max)
    {
        return Lang.exRandom().nextFloatMatrix(height ,width, 0, max);
    }
    @Passed
    public void nextFloatMatrix(float[][] v, float max)
    {
        Lang.exRandom().nextFloatMatrix(v, 0, max);
    }
    @Passed
    public float[][] nextFloatMatrix(int height, int width, float min, float max)
    {
        return Lang.exRandom().nextFloatMatrix(height, width, min, max);
    }
    @Passed
    public void nextFloatMatrix(float[][] v, float min, float max)
    {
        Lang.exRandom().nextFloatMatrix(v, min, max);
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Matrix-Convert-Function">
    //<editor-fold defaultstate="collapsed" desc="Matrix-Convert-Function:elementary">
    public static boolean[][] valueOfBooleanMatrix(String value) throws Exception
    {
        String[] tks=value.split("\n");
        boolean[][] arr=new boolean[tks.length][];
        for(int i=0;i<arr.length;i++)
            arr[i]=Vector.valueOfBooleanVector(tks[i]);
        return arr;
    }
    public static byte[][] valueOfByteMatrix(String value) throws Exception
    {
        String[] tks=value.split("\n");
        byte[][] arr=new byte[tks.length][];
        for(int i=0;i<arr.length;i++)
            arr[i]=Vector.valueOfByteVector(tks[i]);
        return arr;
    }
    public static short[][] valueOfShortMatrix(String value) throws Exception
    {
        String[] tks=value.split("\n");
        short[][] arr=new short[tks.length][];
        for(int i=0;i<arr.length;i++)
            arr[i]=Vector.valueOfShortVector(tks[i]);
        return arr;
    }
    public static int[][] valueOfIntMatrix(String value) throws Exception
    {
        String[] tks=value.split("\n");
        int[][] arr=new int[tks.length][];
        for(int i=0;i<arr.length;i++)
            arr[i]=Vector.valueOfIntVector(tks[i]);
        return arr;
    }
    public static long[][] valueOfLongMatrix(String value) throws Exception
    {
        String[] tks=value.split("\n");
        long[][] arr=new long[tks.length][];
        for(int i=0;i<arr.length;i++)
            arr[i]=Vector.valueOfLongVector(tks[i]);
        return arr;
    }
    public static float[][] valueOfFloatMatrix(String value) throws Exception
    {
        String[] tks=value.split("\n");
        float[][] arr=new float[tks.length][];
        for(int i=0;i<arr.length;i++)
            arr[i]=Vector.toFloatVector(tks[i]);
        return arr;
    }
    public static double[][] valueOfDoubleMatrix(String value) throws Exception
    {
        String[] tks=value.split("\n");
        double[][] arr=new double[tks.length][];
        for(int i=0;i<arr.length;i++)
            arr[i]=Vector.valueOfDoubleVector(tks[i]);
        return arr;
    }
    public static void valueOfDoubleArray(String value, double[][] arr) throws Exception
    {
        String[] tks=value.split("\n");
        for(int i=0;i<arr.length;i++)
            Vector.valueOfDoubleVector(tks[i], arr[i]);
    }
    public static String[][] valueOfStringMatrix(String value)
    {
        String[] tks=value.split("\n"),curTks=null;
        String[][] r=new String[tks.length][];
        for(int i=0,j;i<r.length;i++)
        {
            curTks=tks[i].split(" {0,}, {0,}");
            r[i]=new String[curTks.length];
            for(j=0;j<curTks.length;j++) r[i][j]=curTks[j];
        }
        return r;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Matrix-Convert-Function:elementary">
    public static Boolean[][] valueOfNBooleanMatrix(String value) throws Exception
    {
        String[] tks=value.split("\n");
        Boolean[][] arr=new Boolean[tks.length][];
        for(int i=0;i<arr.length;i++)
            arr[i]=Vector.valueOfNBooleanVector(tks[i]);
        return arr;
    }
    public static Byte[][] valueOfNByteMatrix(String value) throws Exception
    {
        String[] tks=value.split("\n");
        Byte[][] arr=new Byte[tks.length][];
        for(int i=0;i<arr.length;i++)
            arr[i]=Vector.valueOfNByteArray(tks[i]);
        return arr;
    }
    public static Short[][] valueOfNShortMatrix(String value) throws Exception
    {
        String[] tks=value.split("\n");
        Short[][] arr=new Short[tks.length][];
        for(int i=0;i<arr.length;i++)
            arr[i]=Vector.valueOfNShortArray(tks[i]);
        return arr;
    }
    public static Integer[][] valueOfNIntMatrix(String value) throws Exception
    {
        String[] tks=value.split("\n");
        Integer[][] arr=new Integer[tks.length][];
        for(int i=0;i<arr.length;i++)
            arr[i]=Vector.valueOfNIntVector(tks[i]);
        return arr;
    }
    public static Long[][] valueOfNLongMatrix(String value) throws Exception
    {
        String[] tks=value.split("\n");
        Long[][] arr=new Long[tks.length][];
        for(int i=0;i<arr.length;i++)
            arr[i]=Vector.valueOfNLongVector(tks[i]);
        return arr;
    }
    public static Float[][] valueOfNFloatMatrix(String value) throws Exception
    {
        String[] tks=value.split("\n");
        Float[][] arr=new Float[tks.length][];
        for(int i=0;i<arr.length;i++)
            arr[i]=Vector.valueOfNFloatVector(tks[i]);
        return arr;
    }
    public static Double[][] valueOfNDoubleMatrix(String value) throws Exception
    {
        String[] tks=value.split("\n");
        Double[][] arr=new Double[tks.length][];
        for(int i=0;i<arr.length;i++)
            arr[i]=Vector.valueOfNDoubleVector(tks[i]);
        return arr;
    }
    //</editor-fold>
    //</editor-fold>
    
    
    public static float[][] softmax(float[][] X) {
        int N = X.length, M = X[0].length;
        float[][] Y = new float[N][M];
        softmax(X, Y, N, M);
        return Y;
    }
    public static void softmax(float[][] X, float[][] Y, int N, int M){
        for(int i=0;i<N;i++) {
            double v = 0;
            for(int j=0; j<M; j++) {
                Y[i][j] = (float) Math.exp(X[i][j]);
                v += Y[i][j];
            }
            for(int j=0; j<M; j++) Y[i][j] /= v;
        }
    }
    
    
    /**
     * Fo the input Matrx x, regard rows as tuples and fields as columns.
     * as the covariance Matrix=xT*x
     * @param x
     * @return 
     */
    public static double[][] covarianceMatrixForLine(double[][] x)
    {
        int width=x.length;
        double k=1.0/x.length;
        double[][] cov = new double[width][width];
        double[][] xt = Matrix.transpose(x);
        Matrix.multiply(cov, xt, x);
        Matrix.multiply(cov, cov, k);
        return cov;
    }
    public static double[][] inverse(double[][] x) 
    {
        double[][] r=new double[x.length][x[0].length];
        return r;
    }
    
    //<editor-fold defaultstate="collapsed" desc="toMatrix">
    public static void toMatrix(float[] vec, float[][] mat, int width) {
        int start=0;
        for (float[] line : mat) {
            System.arraycopy(vec, start, line, 0, width);
            start += width;
        }
    }
    public static float[][] toMatrix(float[] vec, int width) {
        int height = (vec.length + width - 1 ) / width;
        float[][] mat = new float[height][width];
        Matrix.toMatrix(vec, mat, width);
        return mat;
    }
    
    public static void toMatrix(char[] vec, char[][] mat, int width) {
        int start = 0;
        for(char[] line : mat) {
            System.arraycopy(vec, start, line, 0, width);
            start += width;
        }
    }
    public static char[][] toMatrix(char[] vec, int width) {
        int height = (vec.length + width - 1) / width;
        char[][] mat = new char[height][width];
        Matrix.toMatrix(vec, mat, width);
        return mat;
    }
    
    
    public static void toMatrix(byte[] vec, byte[][] mat, int width) {
        int start = 0;
        for(byte[] line : mat) {
            System.arraycopy(vec, start, line, 0, width);
            start += width;
        }
    }
    public static byte[][] toMatrix(byte[] vec, int width) {
        int height = (vec.length + width - 1) / width;
        byte[][] mat = new byte[height][width];
        Matrix.toMatrix(vec, mat, width);
        return mat;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="matrixToVector">
    public static void toVector(float[] vec, float[][] mat) {
        for(int i=0, index=0; i<mat.length; i++)
            for(int j=0; j<mat[i].length; j++)  {
                if(index >= vec.length) return;
                vec[index++] = mat[i][j];
            }
    }
    public static float[] toVector(float[][] mat, int length) {
        float[] vec = new float[length];
        Matrix.toVector(vec, mat);
        return vec;
    }
    
    public static float[] toVector(float[][] mat) {
        int length = mat.length * mat[0].length;
        float[] vec = new float[length];
        Matrix.toVector(vec, mat);
        return vec;
    }
    
    
    public static void toVector(byte[] vec, byte[][] mat) 
    {
        for(int i=0, index=0; i<mat.length; i++)
            for(int j=0;j<mat[i].length;j++) {
                if(index>=vec.length) return;
                vec[index++] = mat[i][j];
            }
    }
    public static byte[] toVector(byte[][] mat, int length)
    {
        byte[] vec = new byte[length];
        Matrix.toVector(vec, mat);
        return vec;
    }
    public static byte[] toVector(byte[][] mat)
    {
        int length = mat.length * mat[0].length;
        byte[] vec = new byte[length];
        Matrix.toVector(vec, mat);
        return vec;
    }
    
    
    public static void toFloatVector(float[] vec, byte[][] mat)
    {
        for(int i=0, index=0; i<mat.length; i++)
            for(int j=0; j<mat[i].length; j++)  {
                if(index >= vec.length) return;
                vec[index++] = mat[i][j];
            }
    }
    public static float[] toFloatVector(byte[][] mat, int length)
    {
        float[] vec = new float[length];
        Matrix.toFloatVector(vec, mat);
        return vec;
    }
    
    public static float[] toFloatVector(byte[][] mat)
    {
        int length = mat.length * mat[0].length;
        float[] vec = new float[length];
        Matrix.toFloatVector(vec, mat);
        return vec;
    }
    //</editor-fold>
    
    public static float[][] onehot(byte[] X, float alpha, float beta, int num_class)
    {
        int height = X.length;
        float[][] Y = new float[height][num_class];
        
        for(int i=0; i<height; i++) {
            for(int j=0; j<num_class; j++) Y[i][j] = beta;
            byte label_index = X[i];
            Y[i][label_index] = alpha;
        }
        return Y;
    }
    
    public static float[][] onehot(int[] X, float alpha, float beta, int num_class)
    {
        int height = X.length;
        float[][] Y = new float[height][num_class];
        
        for(int i=0; i<height; i++) {
            for(int j=0; j<num_class; j++) Y[i][j] = beta;
            int label_index = X[i];
            Y[i][label_index] = alpha;
        }
        return Y;
    }
}
