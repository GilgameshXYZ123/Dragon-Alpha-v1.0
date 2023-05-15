/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.math.vector;

import java.io.PrintStream;
import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.Collection;
import java.util.Comparator;
import java.util.Enumeration;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.function.BiPredicate;
import java.util.function.Predicate;
import z.util.ds.linear.ZArrayList;
import z.util.function.SequenceCreator;
import z.util.lang.Lang;
import static z.util.lang.Lang.NULL;
import static z.util.lang.Lang.NULL_LN;
import z.util.math.ExRandom.RandomSupplier;
import z.util.lang.annotation.Passed;
import z.util.lang.exception.IAE;
import z.util.math.Sort;
import static z.util.math.Sort.INSERT_SORT_THRESHOLD;

/**
 *
 * @author dell
 */
public final class Vector
{
    private Vector() {}
    
    //<editor-fold defaultstate="collapsed" desc="Vector-String-Function">
    //<editor-fold defaultstate="collapsed" desc="String-Function:append">
    @Passed
    public static void append(StringBuilder sb, boolean[] v)
    {
        if(v==null) {sb.append(NULL);return;}
        sb.append(v[0]);
        for(int i=1;i<v.length;i++) sb.append(',').append(v[i]);
    }
    @Passed
    public static void append(StringBuilder sb, char[] v)
    {
        if(v==null) {sb.append(NULL);return;}
        sb.append(v[0]);
        for(int i=1;i<v.length;i++) sb.append(',').append(v[i]);
    }
    @Passed
    public static void append(StringBuilder sb, byte[] v)
    {
        if(v==null) {sb.append(NULL);return;}
        sb.append(v[0]);
        for(int i=1;i<v.length;i++) sb.append(',').append(v[i]);
    }
    @Passed
    public static void append(StringBuilder sb, short[] v)
    {
        if(v==null) {sb.append(NULL);return;}
        sb.append(v[0]);
        for(int i=1;i<v.length;i++) sb.append(',').append(v[i]);
    }
    @Passed
    public static void append(StringBuilder sb, int[] v)
    {
        if(v==null) {sb.append(NULL);return;}
        sb.append(v[0]);
        for(int i=1;i<v.length;i++) sb.append(',').append(v[i]);
    }
    
    @Passed
    public static void append(StringBuilder sb, int[] v, int start, int end)
    {
        if(v == null || start >= v.length) {sb.append(NULL);return;}
        sb.append(v[start]);
        for(int i = start + 1;i <=end && i<v.length; i++)
            sb.append(',').append(v[i]);
    }
    
    
    @Passed
    public static void append(StringBuilder sb, long[] v) {
        if(v==null) { sb.append(NULL);return; }
        sb.append(v[0]);
        for(int i=1; i<v.length; i++) 
            sb.append(',').append(Float.toString(v[i]));
    }
    
    @Passed
    public static void append(StringBuilder sb, float[] v)
    {
        if(v==null) {sb.append(NULL);return;}
        sb.append(v[0]);
        for(int i=1;i<v.length;i++) sb.append(',').append(v[i]);
    }
    @Passed
    public static void append(StringBuilder sb, double[] v)
    {
        if(v==null) {sb.append(NULL);return;}
        sb.append(v[0]);
        for(int i=1;i<v.length;i++) sb.append(',').append(v[i]);
    }
    @Passed
    public static <T> void append(StringBuilder sb, T[] v)
    {
        if(v==null) {sb.append(NULL);return;}
        sb.append(v[0]);
        for(int i=1;i<v.length;i++) sb.append(',').append(v[i]);
    }
    @Passed
    public static <T> void append(StringBuilder sb, String prefix, T[] v)
    {
        if(v==null) {sb.append(NULL);return;}
        sb.append(v[0]);
        String pf=','+prefix;
        for(int i=1;i<v.length;i++) sb.append(pf).append(v[i]);
    }
     @Passed
    public static void append(StringBuilder sb, Collection v)
    {
        if(v==null) {sb.append(NULL);return;}
        for(Object val:v) sb.append(val).append(',');
        if(sb.charAt(sb.length()-1)==',') sb.setLength(sb.length()-1);
    }
    @Passed
    public static<T> void appendLn(StringBuilder sb, T[] v)
    {
        if(v==null) {sb.append(NULL_LN);return;}
        for(T val:v) sb.append(val).append('\n');
    }
    @Passed
    public static void appendLn(StringBuilder sb, Collection v)
    {
        if(v==null) {sb.append(NULL_LN);return;}
        for(Object o:v) sb.append(o).append('\n');
    }
    @Passed
    public static void appendLn(StringBuilder sb, Collection v, String prefix)
    {
        if(v==null) {sb.append(NULL_LN);return;}
        for(Object o:v) sb.append(prefix).append(o).append('\n');
    }
    @Passed
    public static void appendLn(StringBuilder sb, Map map, String prefix)
    {
        if(map==null) {sb.append(NULL_LN);return;}
        map.forEach((Object key, Object value)->{sb.append(prefix).append(key).append(" = ").append(value);});
    }
    @Passed
    public static void appendLn(StringBuilder sb, Map map)
    {
        if(map==null) {sb.append(NULL_LN);return;}
        map.forEach((Object key, Object value)->{sb.append('\n').append(key).append(" = ").append(value);});
    }
    
    public static void appendLimitCapacity(StringBuilder sb, float[] value)
    {
        int capacity = sb.capacity() - 10, index = 0, lineFeed = 0;
        while(index<value.length-1 && sb.length()<capacity)
        {
            sb.append(value[index]).append(',');
            index++; lineFeed++;
            if((lineFeed&7) ==0) sb.append("\n\t");
        }
        if(sb.length()<capacity) sb.append(value[value.length - 1]);
        if(index<value.length) sb.append(".......");
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="String-Function:toString">
    @Passed
    public static String toString(boolean[] v)
    {
        StringBuilder sb=new StringBuilder();
        Vector.append(sb, v);
        return sb.toString();
    }
    @Passed
    public static String toString(byte[] v)
    {
        StringBuilder sb=new StringBuilder();
        Vector.append(sb, v);
        return sb.toString();
    }
    @Passed
    public static String toString(short[] v)
    {
        StringBuilder sb=new StringBuilder();
        Vector.append(sb, v);
        return sb.toString();
    }
    @Passed
    public static String toString(int[] v)
    {
        StringBuilder sb=new StringBuilder();
        Vector.append(sb, v);
        return sb.toString();
    }
    @Passed
    public static String toString(long[] v)
    {
        StringBuilder sb=new StringBuilder();
        Vector.append(sb, v);
        return sb.toString();
    }
    @Passed
    public static String toString(float[] v)
    {
        StringBuilder sb=new StringBuilder();
        Vector.append(sb, v);
        return sb.toString();
    }
    @Passed
    public static String toString(double[] v)
    {
        StringBuilder sb=new StringBuilder();
        Vector.append(sb, v);
        return sb.toString();
    }
    public static <T> String toString(T[] v)
    {
        StringBuilder sb=new StringBuilder();
        Vector.append(sb, v);
        return sb.toString();
    }
    public static String toStringln(Map m)
    {
        StringBuilder sb=new StringBuilder();
        m.forEach((k,v)->{sb.append('\n').append(k).append(" = ").append(v);});
        return sb.toString();
    }
    public static String toStringln(Collection v)
    {
        StringBuilder sb=new StringBuilder();
        for(Object o:v) sb.append(o).append('\n');
        return sb.toString();
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="String-Function:print">
    private static final PrintStream DEF_OUT=System.out;
    public static synchronized void setDefaultPrintStream(PrintStream out){}
    public static PrintStream getDefaultPrintStream() {return DEF_OUT;}
    
    public static void println(boolean[] v) {println(DEF_OUT, v);}
    
    public static void println(char[] v) {println(DEF_OUT, v, 0, v.length - 1);}
    public static void println(char[] v, int low ,int high) {println(DEF_OUT, v, low, high);}
    public static void println(PrintStream out, char[] v) {println(DEF_OUT, v, 0, v.length - 1);}
    
    public static void println(byte[] v) {println(DEF_OUT, v);}
    public static void println(byte[] v, int low, int high) {println(DEF_OUT, v, low, high);}
    public static void println(String msg, byte[] v, int low, int high) {
        DEF_OUT.print(msg);
        println(DEF_OUT, v, low, high);
    }
    
    public static void println(String msg, int[] v, int low, int high) {
        DEF_OUT.print(msg);
        println(DEF_OUT, v, low, high);
    }
     
    public static void println(short[] v) {println(DEF_OUT, v);}
    
    public static void println(int[] v) {println(DEF_OUT, v);}
    public static void println(int[] v, int low, int high) {println(DEF_OUT, v, low, high);}
    
    public static void println(long[] v) {println(DEF_OUT, v);}
    
    public static void println(float[] v) {println(DEF_OUT, v);}
    public static void println(float[] v, int low, int high) {println(DEF_OUT, v, low, high);}
    public static void println(String msg, float[] v, int low, int high) {
        DEF_OUT.print(msg);
        println(DEF_OUT, v, low, high);
    }
    
    public static void println(double[] v) {println(DEF_OUT, v);}
    
    public static void println(Object[] v) {println(DEF_OUT, v);}
    public static void println(Object[] v, char deli) {println(DEF_OUT, v, deli);}
    public static void println(Object[] v, int low, int high) {println(DEF_OUT, v, low, high, ',');}
    public static void println(Object[] v, int low, int high, char deli) {println(DEF_OUT, v, low, high, deli);}
    public static void println(Collection v) {println(DEF_OUT, v);}
    public static void println(Map m) {println(DEF_OUT, m);}
    
    @Passed
    public static void println(PrintStream out, boolean[] v)
    {
        if(v==null) {out.println(NULL);return;}
        out.print(v[0]);
        for(int i=1;i<v.length;i++) 
            {out.print(',');out.print(v[i]);}
        out.println();
    }
    @Passed
    public static void println(PrintStream out, byte[] v)
    {
        if(v==null) {out.println(NULL);return;}
        out.print(v[0]);
        for(int i=1;i<v.length;i++) 
            {out.print(',');out.print(v[i]);}
        out.println();
    }
    @Passed
    public static void println(PrintStream out, byte[] v, int low, int high)
    {
        if(v==null) {out.println(NULL);return;}
        if(high >= v.length) high=v.length-1;
        if(low<=high)
        {
            out.print(v[low]);
            for(int i=low+1;i<=high;i++) 
                {out.print(',');out.print(v[i]);}
        }
        out.println();
    }
    
    @Passed
    public static void println(PrintStream out, char[] v, int low, int high)
    {
        if(v == null) { out.println(NULL); return; }
        if(high >= v.length) high = v.length-1;
        if(low<=high) {
            out.print(v[low]);
            for(int i = low+1; i<= high; i++) 
                { out.print(',');out.print(v[i]); }
        }
        out.println();
    }
    
    
    @Passed
    public static void println(PrintStream out, short[] v)
    {
        if(v==null) {out.println(NULL);return;}
        out.print(v[0]);
        for(int i=1;i<v.length;i++) 
            {out.print(',');out.print(v[i]);}
        out.println();
    }
    public static void println(PrintStream out, int[] v) {Vector.println(out, v, 0, v.length-1);}
    
    @Passed
    public static void println(PrintStream out, int[] v, int low, int high)
    {
        if(v==null) {out.println(NULL);return;}
        if(high>=v.length) high=v.length-1;
        if(low<=high)
        {
            out.print(v[low]);
            for(int i=low+1;i<=high;i++) 
            {out.print(',');out.print(v[i]);}
        }
        out.println();
    }
    @Passed
    public static void println(PrintStream out, long[] v)
    {
        if(v==null) {out.println(NULL);return;}
        out.print(v[0]);
        for(int i=1;i<v.length;i++) 
            {out.print(',');out.print(v[i]);}
        out.println();
    }
    @Passed
    public static void println(PrintStream out, float[] v)
    {
        if(v==null) {out.println(NULL);return;}
        out.format("% 6f", v[0]);
        for(int i=1;i<v.length;i++) {
                out.print(','); out.print(v[i]);
                //out.format("% 6f", v[i]); 
        }
        out.println();
    }
    @Passed
    public static void println(PrintStream out, float[] v, int low, int high)
    {
        if(v==null) {out.println(NULL);return;}
        if(high >= v.length) high=v.length-1;
        if(low <= high) {
            out.format("% 6f", v[0]);
            for(int i=low+1;i<=high;i++) {
                    out.print(','); out.format("% 6f", v[i]); 
            }
        }
        out.println();
    }
    
    public static void println(PrintStream out, double[] v)
    {
        Vector.println(out, v, ',');
    }
    @Passed
    public static void println(PrintStream out, double[] v, char ldiv)
    {
        if(v==null) {out.println(NULL);return;}
        out.print(v[0]);
        for(int i=1;i<v.length;i++) 
            {out.print(ldiv);out.print(v[i]);}
        out.println();
    }
    public static void println(PrintStream out, Object[] v)
    {
        Vector.println(out, v, 0, v.length-1, ',');
    }
    public static void println(PrintStream out, Object[] v, char deli)
    {
        Vector.println(out, v, 0, v.length-1, deli);
    }
    @Passed
    public static void println(PrintStream out, Object[] v, int low ,int high, char deli)
    {
        if(v==null) {out.println(NULL);return;}
        out.print(v[low]);
        for(int i=low+1;i<=high;i++) 
            {out.print(deli);out.print(v[i]);}
        out.println();
    }
    @Passed
    public static void println(PrintStream out, Collection v)
    {
        if(v==null) {out.println(NULL);return;}
        for(Object o:v) out.println(o);
    }
    @Passed
    public static void println(PrintStream out, Map m)
    {
        if(m==null) {out.println(NULL);return;}
        m.forEach((k,v)->{out.println(k + " = " + v);});
    }
    public static void println(PrintStream out, Enumeration en)
    {
        if(en==null) {out.println(NULL);return;}
        while(en.hasMoreElements())
            out.println(en.nextElement());
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Vector-Convert-Function">
    //<editor-fold defaultstate="collapsed" desc="Array-Convert-Function:elementary">
    public static byte[] valueOfByteVector(String str) throws Exception {
        return Vector.valueOfByteVector(str.split(","));
    }
    public static byte[] valueOfByteVector(String[] tokens) throws Exception  {
        byte[] r = new byte[tokens.length];
        for (int i = 0; i < tokens.length; i++)  r[i] = Byte.valueOf(tokens[i]);
        return r;
    }
    
    public static boolean[] valueOfBooleanVector(String str) throws Exception  {
        return Vector.valueOfBooleanVector(str.split(","));
    }
    
    public static boolean[] valueOfBooleanVector(String[] tokens) throws Exception {
        boolean[] r = new boolean[tokens.length];
        for (int i = 0; i < tokens.length; i++) r[i] = Boolean.valueOf(tokens[i]);
        return r;
    }
    
    public static short[] valueOfShortVector(String str) throws Exception {
        return Vector.valueOfShortVector(str.split(","));
    }
    
    public static short[] valueOfShortVector(String[] tokens) throws Exception {
        short[] r = new short[tokens.length];
        for (int i = 0; i < tokens.length; i++) r[i] = Short.valueOf(tokens[i]);
        return r;
    }
    
    public static int[] valueOfIntVector(String str) throws Exception {
        return Vector.valueOfIntVector(str.split(","));
    }
    
    public static int[] valueOfIntVector(String[] tokens) throws Exception {
        int[] r = new int[tokens.length];
        for (int i = 0; i < tokens.length; i++) r[i] = Integer.valueOf(tokens[i]);
        return r;
    }
    
    public static float[] toFloatVector(byte[] arr) {
        float[] farr = new float[arr.length];
        for(int i=0; i<farr.length; i++) farr[i] = arr[i];
        return farr;
    }
    public static float[] toFloatVector(int[] arr) {
        float[] farr = new float[arr.length];
        for(int i=0; i<farr.length; i++) farr[i] = arr[i];
        return farr;
    }
    
    public static float[] toFloatVector(String str) throws Exception {
        return Vector.toFloatVector(str.split(","));
    }
    public static float[] toFloatVector(String[] tokens) throws Exception {
        float[] r = new float[tokens.length];
        for (int i = 0; i < tokens.length; i++) r[i] = Float.valueOf(tokens[i]);
        return r;
    }
    public static float[] toFloatVector(List<String> lines, int length) 
    {
        float[] value = new float[length];
        int index = 0;
        for(String line : lines) {
            for(String token : line.split(","))
                value[index++] = Float.valueOf(token);
        }
        return value;
    }
    
    
    public static void valueOfDoubleVector(String value, double[] arr) throws Exception {
        String[] tks = value.split(",");
        for(int i=0;i<arr.length;i++) arr[i]=Double.valueOf(tks[i]);
    }
    public static double[] valueOfDoubleVector(String str) throws Exception {
        return Vector.valueOfDoubleVector(str.split(","));
    }
    public static double[] valueOfDoubleVector(String[] tokens) throws Exception 
    {
        double[] r = new double[tokens.length];
        for (int i = 0; i < tokens.length; i++) 
            r[i] = Double.valueOf(tokens[i]);
        return r;
    }
    public static long[] valueOfLongVector(String str) throws Exception
    {
        return Vector.valueOfLongVector(str.split(","));
    }
    public static long[] valueOfLongVector(String[] tokens) throws Exception 
    {
        long[] r = new long[tokens.length];
        for (int i = 0; i < tokens.length; i++) 
            r[i] = Long.valueOf(tokens[i]);
        return r;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Array-Convert-Function:numbers">
    public static Byte[] valueOfNByteArray(String str) throws Exception 
    {
        return Vector.valueOfNByteArray(str.split(","));
    }
    public static Byte[] valueOfNByteArray(String[] tokens) throws Exception 
    {
        Byte[] r = new Byte[tokens.length];
        for (int i = 0; i < tokens.length; i++) 
            r[i] = Byte.valueOf(tokens[i]);
        return r;
    }
    public static Boolean[] valueOfNBooleanVector(String str) throws Exception 
    {
        return Vector.valueOfNBooleanArray(str.split(","));
    }
    public static Boolean[] valueOfNBooleanArray(String[] tokens) throws Exception 
    {
        Boolean[] r = new Boolean[tokens.length];
        for (int i = 0; i < tokens.length; i++) 
            r[i] = Boolean.valueOf(tokens[i]);
        return r;
    }
    public static Short[] valueOfNShortArray(String str) throws Exception 
    {
        return Vector.valueOfNShortArray(str.split(","));
    }
    public static Short[] valueOfNShortArray(String[] tokens) throws Exception 
    {
        Short[] r = new Short[tokens.length];
        for (int i = 0; i < tokens.length; i++) 
            r[i] = Short.valueOf(tokens[i]);
        return r;
    }
    public static Integer[] valueOfNIntVector(String str) throws Exception
    {
        return Vector.valueOfNIntArray(str.split(","));
    }
    public static Integer[] valueOfNIntArray(String[] tokens) throws Exception 
    {
        Integer[] r = new Integer[tokens.length];
        for (int i = 0; i < tokens.length; i++) 
            r[i] = Integer.valueOf(tokens[i]);
        return r;
    }
    public static Float[] valueOfNFloatVector(String str) throws Exception 
    {
        return Vector.valueOfNFloatArray(str.split(","));
    }
    public static Float[] valueOfNFloatArray(String[] tokens) throws Exception 
    {
        Float[] r = new Float[tokens.length];
        for (int i = 0; i < tokens.length; i++) 
            r[i] = Float.valueOf(tokens[i]);
        return r;
    }
    public static Double[] valueOfNDoubleVector(String str) throws Exception
    {
        return Vector.valueOfNDoubleArray(str.split(","));
    }
    public static Double[] valueOfNDoubleArray(String[] tokens) throws Exception 
    {
        Double[] r = new Double[tokens.length];
        for (int i = 0; i < tokens.length; i++) 
            r[i] = Double.valueOf(tokens[i]);
        return r;
    }
    public static Long[] valueOfNLongVector(String str) throws Exception
    {
        return Vector.valueOfNLongArray(str.split(","));
    }
    public static Long[] valueOfNLongArray(String[] tokens) throws Exception 
    {
        Long[] r = new Long[tokens.length];
        for (int i = 0; i < tokens.length; i++) 
            r[i] = Long.valueOf(tokens[i]);
        return r;
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Vector-Math-Function">
    public static <T> T firstNotNull(T[] val)
    {
        for(int i=0;i<val.length;i++)
            if(val[i]!=null) return val[i];
        return null;
    }
    public static <T> T firstNotNull(T[] val, int index)
    {
        for(int i=index;i<val.length;i++)
            if(val[i]!=null) return val[i];
        return null;
    }
    public static <T> void putFirstNoNullToHead(T[] val)
    {
        T fn=Vector.firstNotNull(val);
        T t=fn;fn=val[0];val[0]=t;
    }
    public static <T> void putFirstNoNullToHead(T[] val, int index)
    {
        T fn=Vector.firstNotNull(val, index);
        T t=fn;fn=val[index];val[index]=t;
    }
    @Passed
    public static double minValue(double[] val)
    {
        double min=val[0];
        for(int i=1;i<val.length;i++) if(min>val[i]) min=val[i];
        return min;
    }
    @Passed
    public static float minValue(float[] val, int low, int high)
    {
        float min=val[low];
        for(int i=1+low;i<=high;i++) if(min>val[i]) min=val[i];
        return min;
    }
    public static float minValue(float[] val) {return Vector.minValue(val, 0, val.length-1);}
    
    public static int minValue(byte[] val, int low, int high)
    {
        byte min=val[low];
        for(int i=1+low;i<=high;i++) if(min>val[i]) min=val[i];
        return min;
    }
    public static int minValue(byte[] val) {return Vector.minValue(val, 0, val.length-1);}
    
    @Passed
    public static int minValue(int[] val, int low, int high)
    {
        int min=val[low];
        for(int i=1+low;i<=high;i++) if(min>val[i]) min=val[i];
        return min;
    }
    public static int minValue(int[] val) {return Vector.minValue(val, 0, val.length-1);}
    
    @Passed
    public static int minValueIndex(int[] val, int low, int high) {
        int min = val[low], index=0;
        for(int i=low+1; i<=high; i++) 
            if(min > val[i]) { min = val[i]; index = i; }
        return index;
    }
    public static int minValueIndex(int[] val) {return Vector.minValueIndex(val, 0, val.length-1);}
    
    
    @Passed
    public static int minValueIndex(float[] val, int low, int high) {
        float min = val[low]; int index=0;
        for(int i=low+1; i<=high; i++) 
            if(min > val[i]) { min = val[i]; index = i; }
        return index;
    }
    public static int minValueIndex(float[] val) {return Vector.minValueIndex(val, 0, val.length-1);}
    
    @Passed
    public static <T extends Comparable> T minValue(T[] val)
    {
        Vector.putFirstNoNullToHead(val);
        T min=val[0];
        if(val[0]==null) return null;
        for(int i=1;i<val.length;i++)
            if(min.compareTo(val[i])>0) min=val[i];
        return min;
    }
    
    @Passed
    public static int maxValueIndex(int[] val, int low, int high) {
        int max = val[low], index = low;
        for(int i=low+1; i<=high; i++) 
            if(max < val[i]) { max = val[i]; index = i; }
        return index;
    }
    public static int maxValueIndex(int[] val) {
        return Vector.maxValueIndex(val, 0, val.length-1);
    }
    
    @Passed
    public static int maxValueIndex(float[] val, int low, int high) {
        float max = val[low]; int index = low;
        for(int i = low + 1; i<=high; i++) 
            if(max < val[i]) { max = val[i]; index = i; }
        return index;
    }
    public static int maxValueIndex(float[] val) {
        return Vector.maxValueIndex(val, 0, val.length-1);
    }
    
    
    @Passed
    public static char maxValue(char[] val) {
        char max = val[0];
        for(int i=1; i<val.length; i++) if(max < val[i]) max = val[i];
        return max;
    }
     @Passed
    public static char minValue(char[] val) {
        char min = val[0];
        for(int i=1; i<val.length; i++) if(min > val[i]) min = val[i];
        return min;
    }
    
    
    @Passed
    public static double maxValue(double[] val) {
        double max = val[0];
        for(int i=1; i<val.length; i++) if(max < val[i]) max=val[i];
        return max;
    }
    
    public static float maxValue(float[] val, int low, int high) {
        float max = val[low];
        for(int i=low+1;i<=high;i++) if(max<val[i]) max=val[i];
        return max;
    }
    
    public static float maxValue(float[] val) {return Vector.maxValue(val, 0, val.length-1);}
    
    @Passed
    public static int maxValue(byte[] val)
    {
        byte max=val[0];
        for(int i=1;i<val.length;i++) if(max<val[i]) max=val[i];
        return max;
    }
    @Passed
    public static int maxValue(int[] val)
    {
        int max=val[0];
        for(int i=1;i<val.length;i++) if(max<val[i]) max=val[i];
        return max;
    }
    public static <T extends Comparable> T maxValue(T[] val)
    {
        Vector.putFirstNoNullToHead(val);
        T max=val[0]; 
        if(val[0]==null) return null;
        for(int i=1;i<val.length;i++)
            if(max.compareTo(val[i])<0) max=val[i];
        return max;
    }
    
    //<editor-fold defaultstate="collapsed" desc="class MaxMin">
    public static class MaxMin<T>
    {
        //columns---------------------------------------------------------------
        T max;
        T min;
        
        //functions-------------------------------------------------------------
        MaxMin() {}
        MaxMin(T max, T min)
        {
            this.max=max;
            this.min=min;
        }
        public T getMax() 
        {
            return max;
        }
        public T getMin() 
        {
            return min;
        }
        @Override
        public String toString()
        {
            return "max="+max+"\tmin="+min;
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Inner-Code:MaxMin">
    @Passed
    static void maxMin(int[] val, int low, int high, MaxMin<Integer> mm)
    {
        int max,min;
        if(val[low]>val[low+1]) {max=val[low];min=val[low+1];}
        else {max=val[low+1];min=val[low];}
        for(int i=2+low;i<high;i+=2)
        {
            if(val[i]>val[i+1])
            {
                if(max<val[i]) max=val[i];
                if(min>val[i+1]) min=val[i+1];
            }
            else
            {
                if(max<val[i+1]) max=val[i+1];
                if(min>val[i]) min=val[i];
            }
        }
        if(((high-low+1)&1)==1)//val.length is an odd number
        {
            int ev=val[high];
            if(max<ev) max=ev;
            else if(min>ev) min=ev;
        }
        mm.max=max;
        mm.min=min;
    }
    @Passed
    static void maxMinABS(int[] val, int low, int high, MaxMin<Integer> mm)
    {
        int max,min;
        int abs1= (val[low]>=0? val[low]: -val[low]);
        int abs2= (val[low+1]>=0? val[low+1]: -val[low+1]);
        if(abs1>abs2)  {max=abs1;min=abs2;}
        else {max=abs1;min=abs2;}
        for(int i=2+low;i<high;i+=2)
        {
            abs1=(val[i]>=0? val[i]: -val[i]);
            abs2=(val[i+1]>=0? val[i+1]: -val[i+1]);
            if(abs1>abs2)
            {
                if(max<abs1) max=abs1;
                if(min>abs2) min=abs2;
            }
            else
            {
                if(max<abs2) max=abs2;
                if(min>abs1) min=abs1;
            }
        }
        if(((high-low+1)&1)==1)//val.length is an odd number
        {
            int ev=(val[high]>=0? val[high]:-val[high]);
            if(max<ev) max=ev;
            else if(min>ev) min=ev;
        }
        mm.max=max;
        mm.min=min;
    }
    @Passed
    static void maxMinABSIndex(int[] val, int low, int high, MaxMin<Integer> mm)
    {
        int max,min, maxIndex, minIndex;
        int abs1= (val[low]>=0? val[low]: -val[low]);
        int abs2= (val[low+1]>=0? val[low+1]: -val[low+1]);
        if(abs1>abs2)  {max=abs1; maxIndex=0; min=abs2; minIndex=1;}
        else {max=abs2; maxIndex=1; min=abs1; minIndex=0;}
        for(int i=2+low;i<high;i+=2)
        {
            abs1=(val[i]>=0? val[i]: -val[i]);
            abs2=(val[i+1]>=0? val[i+1]: -val[i+1]);
            if(abs1>abs2)
            {
                if(max<abs1) {max=abs1; maxIndex=i;}
                if(min>abs2) {min=abs2; minIndex=i+1;}
            }
            else
            {
                if(max<abs2) {max=abs2; maxIndex=i+1;}
                if(min>abs1) {min=abs1; minIndex=i;}
            }
        }
        if(((high-low+1)&1)==1)//val.length is an odd number
        {
            int ev=(val[high]>=0? val[high]:-val[high]);
            if(max<ev) maxIndex=high;
            else if(min>ev) minIndex=high;
        }
        mm.max=maxIndex;
        mm.min=minIndex;
    }
    @Passed
    static void maxMin(double[] val, int low, int high, MaxMin<Double> mm)
    {
        double max,min;
        if(val[low]>val[low+1]) {max=val[low];min=val[low+1];}
        else {max=val[low+1];min=val[low];}
        for(int i=2+low;i<high;i+=2)
        {
            if(val[i]>val[i+1])
            {
                if(max<val[i]) max=val[i];
                if(min>val[i+1]) min=val[i+1];
            }
            else
            {
                if(max<val[i+1]) max=val[i+1];
                if(min>val[i]) min=val[i];
            }
        }
        if(((high-low+1)&1)==1)//val.length is an odd number
        {
            double ev=val[high];
            if(max<ev) max=ev;
            else if(min>ev) min=ev;
        }
        mm.max=max;
        mm.min=min;
    }
    @Passed
    static void maxMinABS(double[] val, int low, int high, MaxMin<Double> mm)
    {
        double max,min;
        double abs1= (val[low]>=0? val[low]: -val[low]);
        double abs2= (val[low+1]>=0? val[low+1]: -val[low+1]);
        if(abs1>abs2)  {max=abs1;min=abs2;}
        else {max=abs2;min=abs1;}
        for(int i=2+low;i<high;i+=2)
        {
            abs1=(val[i]>=0? val[i]: -val[i]);
            abs2=(val[i+1]>=0? val[i+1]: -val[i+1]);
            if(abs1>abs2)
            {
                if(max<abs1) max=abs1;
                if(min>abs2) min=abs2;
            }
            else
            {
                if(max<abs2) max=abs2;
                if(min>abs1) min=abs1;
            }
        }
        if(((high-low+1)&1)==1)//val.length is an odd number
        {
            double ev=(val[high]>=0? val[high]:-val[high]);
            if(max<ev) max=ev;
            else if(min>ev) min=ev;
        }
        mm.max=max;
        mm.min=min;
    }
    @Passed
    static <T extends Comparable> void maxMin(T[] val, int low, int high, MaxMin<T> mm)
    {
        T max,min;
        if(val[low].compareTo(val[low+1])>0) {max=val[low];min=val[low+1];}
        else {max=val[low+1];min=val[low];}
        for(int i=2+low;i<high;i+=2)
        {
            if(val[i].compareTo(val[i+1])>0)
            {
                if(max.compareTo(val[i])<0) max=val[i];
                if(min.compareTo(val[i+1])>0) min=val[i+1];
            }
            else
            {
                if(max.compareTo(val[i+1])<0) max=val[i+1];
                if(min.compareTo(val[i])>0) min=val[i];
            }
        }
        if(((high-low+1)&1)==1)//val.length is an odd number
        {
            T ev=val[high];
            if(max.compareTo(ev)<0) max=ev;
            else if(min.compareTo(ev)>0) min=ev;
        }
        mm.max=max;
        mm.min=min;
    }
    //</editor-fold>
    public static MaxMin<Integer> maxMin(int[] val)
    {
        MaxMin<Integer> mm=new MaxMin<>();
        Vector.maxMin(val, 0, val.length-1, mm);
        return mm;
    }
    public static MaxMin<Integer> maxMin(int[] val, int low, int high)
    {
        MaxMin<Integer> mm=new MaxMin<>();
        Vector.maxMin(val, low, high, mm);
        return mm;
    }
    public static MaxMin<Integer> maxMinABS(int[] val)
    {
        MaxMin<Integer> mm=new MaxMin<>();
        Vector.maxMinABS(val, 0, val.length-1, mm);
        return mm;
    }
    public static MaxMin<Integer> maxMinABS(int[] val, int low, int high)
    {
        MaxMin<Integer> mm=new MaxMin<>();
        Vector.maxMinABS(val, low, high, mm);
        return mm;
    }
    public static MaxMin<Integer> maxMinABSIndex(int[] val)
    {
        MaxMin<Integer> mm=new MaxMin<>();
        Vector.maxMinABSIndex(val, 0, val.length-1, mm);
        return mm;
    }
    public static MaxMin<Integer> maxMinABSIndex(int[] val, int low, int high)
    {
        MaxMin<Integer> mm=new MaxMin<>();
        Vector.maxMinABSIndex(val, low, high, mm);
        return mm;
    }
    /**
     * if there exists max-min>threshold, return null and early stopping.
     * @param val
     * @param low
     * @param high
     * @param threshold
     * @return 
     */
    @Passed
    public static MaxMin<Integer> maxMin(int[] val, int low, int high,  int threshold)
    {
        MaxMin<Integer> mm=new MaxMin<>();
        int max,min;
        if(val[low]>val[low+1]) {max=val[low];min=val[low+1];}
        else {max=val[low+1];min=val[low];}
        for(int i=2+low;i<high;i+=2)
        {
            if(max-min>threshold) {return null;}
            if(val[i]>val[i+1])
            {
                if(max<val[i]) max=val[i];
                if(min>val[i+1]) min=val[i+1];
            }
            else
            {
                if(max<val[i+1]) max=val[i+1];
                if(min>val[i]) min=val[i];
            }
        }
        if(((high-low+1)&1)==1)//val.length is an odd number
        {
            int ev=val[high];
            if(max<ev) max=ev;
            else if(min>ev) min=ev;
        }
        mm.max=max;
        mm.min=min;
        return mm;
    }
    
    public static MaxMin<Double> maxMin(double[] val)
    {
        MaxMin<Double> mm=new MaxMin<>();
        Vector.maxMin(val, 0, val.length-1, mm);
        return mm;
    }
    public static MaxMin<Double> maxMin(double[] val, int low, int high)
    {
        MaxMin<Double> mm=new MaxMin<>();
        Vector.maxMin(val, low, high, mm);
        return mm;
    }
     public static MaxMin<Double> maxMinABS(double[] val)
    {
        MaxMin<Double> mm=new MaxMin<>();
        Vector.maxMinABS(val, 0, val.length-1, mm);
        return mm;
    }
    public static MaxMin<Double> maxMinABS(double[] val, int low, int high)
    {
        MaxMin<Double> mm=new MaxMin<>();
        Vector.maxMinABS(val, low, high, mm);
        return mm;
    }
    
    public static <T extends Comparable> MaxMin<T> maxMin(T[] val)
    {
        return Vector.maxMin(val, 0, val.length-1);
    }
    public static <T extends Comparable> MaxMin<T> maxMin(T[] val, int low, int high)
    {
        MaxMin<T> mm=new MaxMin<>();
        Vector.maxMin(val, low, high, mm);
        return mm;
    }
    
    /**
     * <pre>
     * deassign a relative value from 0 to 1, to all element of the input
     * Array {@code val}.
     * 1.find the max and min value of {@code val}
     * 2.let {@code base=max-min}
     * 2.for each element of val:{@code result[i]=(val[i]-min)/base}
     * </pre>
     * @param result
     * @param val 
     */
    @Passed
    public static void relative(double[] result, double[] val)
    {
        MaxMin<Double> mm=Vector.maxMin(val);
        double min=mm.min,base=mm.max-min;
        for(int i=0;i<result.length;i++) result[i]=(val[i]-min)/base;
    }
     /**
     * consider the input Array{@code left}, {@code right} as two points
     * in a multi-dimensional space, find the squre of distance between 
     * the specific two points.
     * @param left the first point 
     * @param right the second point
     * @return distance between the two points
     */
    @Passed
    public static double distanceSquare(double[] left, double[] right)
    {
        double dis=0,r;
        for(int i=0;i<left.length;i++)
            {r=left[i]-right[i];dis+=r*r;}
        return dis;
    }
    /**
     * consider the input Array{@code left}, {@code right} as two points
     * in a multi-dimensional space, cauculate the distance between the 
     * specific two points.
     * @param left the first point 
     * @param right the second point
     * @return distance between the two points
     */
    @Passed
    public static double distance(double[] left, double[] right)
    {
        double dis=0,r;
        for(int i=0;i<left.length;i++)
            {r=left[i]-right[i];dis+=r*r;}
        return Math.sqrt(dis);
    }
    public static double sum(double[] a, int low ,int high)
    {
        double sum=0;
        for(int i=low;i<=high;i++) sum+=a[i];
        return sum;
    }
    public static float sum(float[] a, int low ,int high)
    {
        float sum=0;
        for(int i=low;i<=high;i++) sum+=a[i];
        return sum;
    }
    public static int sum(int[] a, int low ,int high)
    {
        int sum=0;
        for(int i=low;i<=high;i++) sum+=a[i];
        return sum;
    }
    public static double sum(double[] a) {return sum(a, 0, a.length-1);}
    public static float sum(float[] a) {return sum(a, 0, a.length-1);}
    public static int sum(int[] a) {return sum(a, 0, a.length-1);}
    
    public static float straight_quadratic(float[] X, float alpha, float beta, float gamma)
    {
        float v = 0;
        for(int i=0; i < X.length; i++) {
            float x = X[i];
            v += alpha*(x*x) + beta*x + gamma;
        }
        return v;
    }
    
     public static float straight_linear(float[] X, float alpha, float beta)
    {
        float v = 0;
        for(int i=0; i < X.length; i++) {
            float x = X[i];
            v += alpha*x + beta;
        }
        return v;
    }
    
    public static float squareMean(float[] X) {
        float v = 0, alpha = 1.0f / X.length;
        for(int i=0; i<X.length; i++) {
            float x = X[i];
            v += alpha * x*x;
        }
        return v;
    } 
    
    public static float mean(float[] X) {
        float v = 0, alpha = 1.0f / X.length;
        for(int i=0; i<X.length; i++) {
            v += alpha * X[i] ;
        }
        return v;
    }
    
    public static float[] var(float[] X) {
        float mean = mean(X);
        float squareMean = squareMean(X);
        float var = squareMean - mean*mean;
        return new float[]{ var, mean, squareMean };
    }
    
    public static float[] stddev(float[] X) {
        float mean = mean(X);
        float squareMean = squareMean(X);
        float stddev = (float) Math.sqrt(squareMean - mean*mean);
        return new float[]{ stddev, mean, squareMean };
    }
    
    public static float sum(float alpha, float[] a, float beta, int low, int high) {
        float sum=0;
        for(int i=low;i<=high;i++) sum += alpha*a[i] + beta;
        return sum;
    }
    public static float[] sum(float[]... Xs) {
        float[] Y = new float[Xs[0].length];
        for(float[] X : Xs) {
            for(int i=1; i<Y.length; i++) Y[i] += X[i];
        }
        return Y;
    }
    
    
    public static float sum(float alpha, float[] a, float beta) {return sum(alpha, a, beta, 0, a.length-1);}
    
    public static float squareSum(float[] a, int low, int high)
    {
        float sum=0;
        for(int i=low;i<=high;i++) sum+=a[i]*a[i];
        return sum;
    }
    public static float squareSum(float[] a) {return squareSum(a, 0, a.length-1);}
    
    public static float squareSum(float[] a, float alpha, int low, int high)
    {
        float sum=0;
        float k1=alpha*2, k2 = alpha*alpha;
        for(int i=low;i<=high;i++) sum+=a[i]*(a[i]+k1);
        return sum + k2*(high-low+1);
    }
    public static float squareSum(float[] a, float alpha) {return squareSum(a, alpha, 0, a.length-1);}
    
    public static float absSum(float[] a, int low, int high)
    {
        float sum=0;
        for(int i=low;i<=high;i++) sum+=Math.abs(a[i]);
        return sum;
    }
    public static float absSum(float[] a) {return absSum(a, 0, a.length-1);}
    
    public static float absSum(float[] a, float alpha, int low, int high)
    {
        float sum=0;
        for(int i=low;i<=high;i++) sum+=Math.abs(a[i]+alpha);
        return sum;
    }
    public static float absSum(float[] a, float alpha) {return absSum(a, alpha, 0, a.length-1);}
    
    @Passed
    public static double average(int[] val)
    {
        double avg=0;
        for(int i=0;i<val.length;i++) avg+=val[i];
        return avg/val.length;
    }
    @Passed
    public static double average(long[] val)
    {
        double avg=0;
        for(int i=0;i<val.length;i++) avg+=val[i];
        return avg/val.length;
    }
    
    @Passed
    public static double average(double[] a)
    {
        double avg=0;
        for(int i=0;i<a.length;i++) avg+=a[i];
        return avg/a.length;
    }
    
    @Passed
    public static double average(double[] a, int start, int end)
    {
        if(start > end) throw new IllegalArgumentException("start > end");
        double avg = 0, k = 1.0 / (end - start + 1);
        for(int i = start; i <= end; i++) avg += k*a[i];
        return avg;
    }
    
    @Passed
    public static double averageABS(double[] a)
    {
        double avg=average(a), sum=0, t;
        for(double v:a) {t=v-avg;sum+=(t<0? -t: t);}
        return sum/a.length;
    }
    
    @Passed
    private static double minN(double[] a, int n, int low, int high)
    {
        //Find the Nth minest factor of the input Array {@code double[] a}.
        if(low>=high) return a[low];
        int mid=Vector.partition(a, low, high);
        if(mid==n) return a[mid];
        else if(n<mid) return minN(a, n, low, mid-1);
        else return minN(a, n, mid+1, high);
    }
    public static double minN(double[] a, int n, int low, int high, boolean copied)
    {
        if(copied) a=Vector.arrayCopy(a);
        return minN(a, n, low, high);
    }
    public static double minN(double[] a, int n, boolean copied)
    {
        return minN(a, n, 0, a.length-1, copied);
    }
    public static double minN(double[] a, int n)
    {
        return minN(a, n, 0, a.length-1, true);
    }
    public static double maxN(double[] a, int n, int low, int high, boolean copied)
    {
        if(copied) a=Vector.arrayCopy(a);
        return minN(a, a.length-1-n, low, high);
    }
    public static double maxN(double[] a, int n, boolean copied)
    {
        return maxN(a, n, 0, a.length-1, copied);
    }
    public static double maxN(double[] a, int n)
    {
        return minN(a, n, 0, a.length-1, true);
    }
    
    /**
     * find the median of the input Array. It may change the order
     * of elements, if you don't want that, set copied=true, to do this on a
     * new copied Array.
     * @param a
     * @param copied
     * @return 
     */
    @Passed
    public static double median(double[] a, boolean copied)
    {
        if(a.length==0) throw new IAE("Can't find the median in an Array with zero length");
        if(copied) a=Vector.arrayCopy(a);
        double mid1=Vector.minN(a, a.length>>1, 0, a.length-1, false);
        if((a.length&1)==1) return mid1;
        return (mid1+Vector.minN(a, a.length>>1+1, 0, a.length-1, false))/2;
    }
    public static double median(double[] a)
    {
        return median(a, true);
    }
    
    
    @Passed
    public static double variance(int[] val)
    {
       double avg=0,avgs=0;
        for(int i=0;i<val.length;i++)
            {avg+=val[i];avgs+=val[i]*val[i];}
        avg/=val.length;
        avgs/=val.length;
        return avgs-avg*avg;
    }
    @Passed
    public static double variance(long[] val)
    {
       double avg=0,avgs=0;
        for(int i=0;i<val.length;i++)
            {avg+=val[i];avgs+=val[i]*val[i];}
        avg/=val.length;
        avgs/=val.length;
        return avgs-avg*avg;
    }
    @Passed
    public static double variance(double[] val)
    {
        double avg=0,avgs=0;
        for(int i=0;i<val.length;i++)
            {avg+=val[i];avgs+=val[i]*val[i];}
        avg/=val.length;
        avgs/=val.length;
        return avgs-avg*avg;
    }
    public static double stddev(int[] val)
    {
        return Math.sqrt(Vector.variance(val));
    }
    public static double stddev(long[] val)
    {
        return Math.sqrt(Vector.variance(val));
    }
    public static double stddev(double[] val)
    {
        return Math.sqrt(Vector.variance(val));
    }
 
    /**
     * <pre>
     * do normalization on the input Array {@code val}.
     * 1.find the average {@code avg} and the standard derivation {@code stddev} of {@ val}
     * 2.for each element of {@code val}: {@code result[i]=(val[i]-avg)/stddev}
     * </pre>
     * @param result
     * @param val 
     */
    @Passed
    public static void normalize(double[] result, double[] val)
    {
        double avg=0,avgs=0;
        for(int i=0;i<val.length;i++)
            {avg+=val[i];avgs+=val[i]*val[i];}
        avg/=val.length;
        avgs/=val.length;
        avgs=Math.sqrt(avgs);
        for(int i=0;i<val.length;i++)
            result[i]=(val[i]-avg)/avgs;
    }
    
    /**
     * result(1,n)=Vt(1,p)* A(p,n).
     * find the product of between left and each column vector of the input
     * Matrix {@code A}. In linear algebra, we need to transpose V first
     * @param result
     * @param v
     * @param A 
     */
    @Passed
    public static void multiply(double[] result, double[] v, double[][] A)
    {
        for(int j=0,i,width=A[0].length;j<width;j++)
        {
            result[j]=v[0]*A[j][0];
            for(i=1;i<A.length;i++) result[j]+=v[i]*A[j][i];
        }
    }
    /**
     * {@link #multiply(double[], double[], double[][]) }
     * @param v
     * @param A
     * @return 
     */
    public static double[] multiply(double[] v, double[][] A)
    {
        double[] result=new double[A[0].length];
        Vector.multiply(result, v, A);
        return result;
    }
    /**
     * result(n, 1)=At(n,p)*v(p,1).
     * @param result
     * @param v
     * @param A 
     */
    public static void multiply(double[] result, double[][] A, double[] v)
    {
        for(int i=0,j,width=A[0].length;i<v.length;i++)
        {
            result[i]=A[i][0]*v[0];
            for(j=0;j<width;j++) result[i]+=A[i][j]*v[j];
        }
    }
    public static double[] multiply(double[][] A, double[] v)
    {
        double[] result=new double[A.length];
        Vector.multiply(result, A, v);
        return result;
    }
    /**
     * Consider the input Arrays {@code left}, a vectors, find the product
     * of {@left} and the input constant {@code k}.
     * @param result
     * @param left
     * @param k
     */
    @Passed
    public static void multiply(double[] result, double[] left, double k)
    {
        for(int i=0;i<result.length;i++)
            result[i]=left[i]*k;
    }
    @Passed
    public static double dot(double[] A, double[] B)
    {
        double result=0;
        for(int i=0;i<A.length;i++) result+=A[i]*B[i];
        return result;
    }
    public static float dot(float[] A, float[] B)
    {
        float result=0;
        for(int i=0;i<A.length;i++) result+=A[i]*B[i];
        return result;
    }
    /**
     * Consider the two input Arrays {@code left}, {@code right} as two 
     * vectors, find the cosine of the Angle between the two vectors, as 
     * the Angle belongs to [0, Pi].
     * @param left
     * @param right
     * @return 
     */
    @Passed
    public static double cosAngle(double[] left, double[] right)
    {
        double product=0,sub,mod=0;
        for(int i=0;i<left.length;i++)
        {
            product+=left[i]*right[i];
            sub=right[i]-left[i];
            mod+=sub*sub;
        }
        return product/mod;
    }
    /**
     * Consider the two input Arrays {@code left}, {@code right} as two 
     * vectors, find the sine of the Angle between the two vectors, as 
     * the Angle belongs to [0, Pi].
     * @param left
     * @param right
     * @return 
     */
    @Passed
    public static double sinAngle(double[] left, double[] right)
    {
        double cos=Vector.cosAngle(left, right);
        return Math.sqrt(1-cos*cos);
    }
    
    public static double[] posibility(int[] f)
    {
        double[] p=new double[f.length];
        int num=0;
        for(int i=0;i<f.length;i++) num+=f[i];
        for(int i=0;i<p.length;i++) p[i]=f[i]*1.0/num;
        return p;
    }
    public static double[][] posibility(int[][] fs)
    {
        double[][] ps=new double[fs.length][];
        for(int i=0;i<ps.length;i++) ps[i]=posibility(fs[i]);
        return ps;
    }
    public static double gini(double[] p)
    {
        double result=p[0]*p[0];
        for(int i=1;i<p.length;i++) result+=p[i]*p[i];
        return 1-result;
    }
    public static double entropyE(double[] p)
    {
        double result=0;
        for(int i=0;i<p.length;i++) result+=(p[i]==0? 0:p[i]*Math.log(p[i]));
        return -result;
    }
    public static double entropyE(int[] f) {return entropyE(posibility(f));}
    public static double entropy2(double[] p) {return entropyE(p)/Math.log(2);}
    public static double entropy2(int[] f) {return entropyE(f)/Math.log(2);}
    
    private static double InfoEntropyE(double[][] p, int[] nums)
    {
        double[] infos=new double[p.length];
        int num=0;
        for(int i=0;i<infos.length;i++) {infos[i]=entropyE(p[i]);num+=nums[i];}
        
        double info=0;
        for(int i=0;i<infos.length;i++) {info+=infos[i]*nums[i]/num;}
        return info;
    }
    public static double InfoEntropyE(int[][] f) 
    {
        int[] nums=new int[f.length];
        for(int i=0;i<f.length;i++) nums[i]=sum(f[i]);
        return InfoEntropyE(posibility(f), nums);
    }
    public static double InfoEntropy2(int[][] f) {return InfoEntropyE(f)/Math.log(2);}
      /**
     * find the reciprocal for each component of the input Array {@code val}.
     * @param result
     * @param val
     */
    @Passed
    public static void reciprocal(double[] result, double[] val)
    {
        for(int i=0;i<result.length;i++) result[i]=1/val[i];
    }
    @Passed
    public static void relu(double[] result, double[] val)
    {
        for(int i=0;i<result.length;i++)
            result[i]=(val[i]>0? val[i]:0);
    }
    
    public static void sigmoid(double[] Y) {
        for(int i=0; i < Y.length; i++)  Y[i] = 1 / (1 + Math.exp(-Y[i]));
    }
    
    @Passed
    public static void sigmoid(double[] X, double[] Y)
    {
        for(int i=0;i<Y.length;i++)
            Y[i]=1/(1+Math.exp(-X[i]));
    }
    @Passed
    public static void unSigmoid(double[] result)
    {
        for(int i=0;i<result.length;i++)
            result[i]=-Math.log(1/result[i]-1);
    }
    @Passed
    public static void unSigmoid(double[] result, double[] val)
    {
        for(int i=0;i<result.length;i++)
            result[i]=-Math.log(1/val[i]-1);
    }
    @Passed
    public static void tanh(double[] result)
    {
        for(int i=0;i<result.length;i++)
            result[i] = 1 - 2/(Math.exp(2*result[i])+1);
    }
    @Passed
    public static void tanh(double[] result, double[] val)
    {
        for(int i=0;i<result.length;i++)
            result[i]=1-2/(Math.exp(2*val[i])+1);
    }
    @Passed
    public static void unTanh(double[] result)
    {
        for(int i=0;i<result.length;i++)
            result[i]=-0.5*Math.log(1/(1-result[i])-1);
    }
    @Passed
    public static void unTanh(double[] result, double[] val)
    {
        for(int i=0;i<result.length;i++)
            result[i]=-0.5*Math.log(1/(1-val[i])-1);
    }
    @Passed
    public static void softPlus(double[] result, double[] val)
    {
        for(int i=0;i<result.length;i++)
            result[i]=Math.log(Math.exp(val[i]+1));
    }
    @Passed
    public static void unSoftPlus(double[] result, double[] val)
    {
        for(int i=0;i<result.length;i++)
            result[i]=Math.log(Math.exp(val[i]-1));
    }
    @Passed
    public static double norm(double[] x)
    {
        double sum=0;
        for(double v:x) sum+=v*v;
        return Math.sqrt(sum);
    }
    @Passed
    public static double normSquare(double[] x)
    {
        double sum=0;
        for(double v:x) sum+=v*v;
        return sum;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Vector-math-Function">
    //<editor-fold defaultstate="collapsed" desc="Trigonometric Function">
    public static float[] sin(float[] X) {
        float[] Y = new float[X.length];
        for(int i=0; i<X.length; i++) Y[i] = (float) Math.sin(X[i]);
        return Y;
    }
    public static float[] sin(float alpha, float[] X, float beta) {
        float[] Y = new float[X.length];
        for(int i=0; i<X.length; i++) Y[i] = (float) Math.sin(alpha*X[i] + beta);
        return Y;
    }
    
    public static float[] cos(float[] X) {
        float[] Y = new float[X.length];
        for(int i=0; i<X.length; i++) Y[i] = (float) Math.cos(X[i]);
        return Y;
    }
    public static float[] cos(float alpha, float[] X, float beta) {
        float[] Y = new float[X.length];
        for(int i=0; i<X.length; i++) Y[i] = (float) Math.cos(alpha*X[i] + beta);
        return Y;
    }
    
    
    public static void halfSin(float amp, float alpha, float[] X, float beta, float[] Y, int length) {
        for(int i=0; i<length; i++) {
            float x = alpha*X[i] + beta;
            x -= Math.floor(x/Math.PI + 0.5f) * Math.PI;
            Y[i] = amp * (float) Math.sin(x);
        }
    }
    public static void halfSin_Deri(float alpha, float[] Y, float[] deriY, int length)
    {
        for(int i=0; i<length; i++) {
            float y = Y[i];
            deriY[i] = (float) (alpha * Math.sqrt(1 - y*y));
        }
    }
    
    public static float[] equal(float[] X1, float[] X2, float min, float max) {
        float[] Y = new float[X1.length];
        for(int i=0; i<Y.length; i++) {
            float div = Math.abs(X1[i] - X2[i]);
            boolean flag = div <= max && div >= min;
            Y[i] = flag? 1 : 0;
        }
        return Y;
    }
    
    public static float[] equal(byte[] X1, byte[] X2, byte min, byte max) {
        float[] Y = new float[X1.length];
        for(int i=0; i<Y.length; i++) {
            int div = Math.abs(X1[i] - X2[i]);
            boolean flag = div <= max && div >= min;
            Y[i] = flag? 1 : 0;
        }
        return Y;
    }

    public static float[] equal(int[] X1, int[] X2, int min, int max) {
        float[] Y = new float[X1.length];
        for(int i=0; i<Y.length; i++) {
            int div = Math.abs(X1[i] - X2[i]);
            boolean flag = div <= max && div >= min;
            Y[i] = flag? 1 : 0;
        }
        return Y;
    }

    public static float[] tan(float[] X) {
        float[] Y = new float[X.length];
        for(int i=0; i<X.length; i++) Y[i] = (float) Math.tan(X[i]);
        return Y;
    }        
    public static float[] tan(float alpha, float[] X, float beta) {
        float[] Y = new float[X.length];
        for(int i=0; i<X.length; i++) Y[i] = (float) Math.tan(alpha*X[i] + beta);
        return Y;
    }   
    public static void tan_Deri(float alpha, float[] X, float beta, float[] deriY, int length) {
        for(int i=0; i<length; i++) {
            float x = alpha*X[i] + beta;
            float cos_x = (float) Math.cos(x);
            deriY[i] = alpha / (cos_x * cos_x);
        }
    }
    
    public static float[] cot(float[] X) {
        float[] Y = new float[X.length];
        for(int i=0; i<X.length; i++) Y[i] = (float) (1.0f / Math.tan(X[i]));
        return Y;
    }
    public static float[] cot(float alpha, float[] X, float beta) {
        float[] Y = new float[X.length];
        for(int i=0; i<X.length; i++) Y[i] = (float) (1/Math.tan(alpha*X[i] + beta));
        return Y;
    }
    public static void cot_Deri(float alpha, float[] X, float beta, float[] deriY, int length) {
        for(int i=0;i<length;i++) {
            float x = alpha*X[i] + beta;
            float sin_x = (float) Math.sin(x);
            deriY[i] = -alpha / (sin_x * sin_x);
        }
    }
    
    
    public static void sec(float[] X, float[] Y, int length) {
        for(int i=0;i<length;i++) Y[i] = (float) (1/Math.cos(X[i]));
    }
    public static void sec(float alpha, float[] X, float beta, float[] Y, int length) {
        for(int i=0;i<length;i++) Y[i] = (float) (1/Math.cos(alpha*X[i] + beta));
    }
    
    public static void csc(float[] X, float[] Y, int length)
    {
        for(int i=0;i<length;i++) Y[i] =(float) (1/Math.sin(X[i]));
    }
    public static void csc(float alpha, float[] X, float beta, float[] Y, int length)
    {
        for(int i=0;i<length;i++) Y[i] =(float) (1/Math.sin(alpha*X[i] + beta));
    }
    
    public static float[] arcsin(float[] X) {
        float[] Y = new float[X.length];
        for(int i=0; i<X.length; i++) Y[i] = (float) Math.asin(X[i]);
        return Y;
    }
    public static float[] arcsin(float alpha, float[] X, float beta) {
        float[] Y = new float[X.length];
        for(int i=0; i<X.length; i++) Y[i] = (float) Math.asin(alpha * X[i] + beta);
        return Y;
    }
    public static void asin_Deri(float alpha, float[] X, float beta, float[] deriY, int length) {
        for(int i=0; i<length; i++) {
            float x = alpha*X[i] + beta;
            deriY[i] = (float) (alpha / Math.sqrt(1 - x*x));
        }
    }
  
    public static float[] arccos(float[] X) {
        float[] Y = new float[X.length];
        for(int i=0; i<X.length; i++) Y[i] = (float) Math.acos(X[i]);
        return Y;
    }
    public static float[] arccos(float alpha, float[] X, float beta) {
        float[] Y = new float[X.length];
        for(int i=0; i<X.length; i++) Y[i] = (float) Math.acos((alpha * X[i] + beta));
        return Y;
    }
    
    public static float[] arctan(float[] X) {
        float[] Y = new float[X.length];
        for(int i=0; i<X.length; i++) Y[i] = (float) Math.atan(X[i]);
        return Y;
    }        
    public static float[] arctan(float alpha, float[] X, float beta) {
        float[] Y = new float[X.length];
        for(int i=0; i<X.length; i++) Y[i] = (float) (Math.atan(alpha*X[i] + beta));
        return Y;
    }     
    public static void atan_Deri(float alpha, float[] X, float beta, float[] deriY, int length) {
        for(int i=0; i<length; i++) {
            float x = alpha*X[i] + beta;
            deriY[i] = alpha / (1 + x*x);
        }
    }
    
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Exponential Function">
    public static float[] sqrt(float[] X) {
        float[] Y = new float[X.length];
        for(int i=0; i<X.length; i++) Y[i] = (float) Math.sqrt(X[i]);
        return Y;
    }
    public static float[] sqrt(float alpha, float[] X, float beta) {
        float[] Y = new float[X.length];
        for(int i=0; i<X.length; i++) Y[i] = (float) Math.sqrt(alpha*X[i] + beta);
        return Y;
    }
    public static void sqrt_Deri(float alpha, float[] X, float beta, float[] deriY, int length) {
        for(int i=0; i<length; i++) {
            float x = (float) Math.sqrt(alpha * X[i] + beta);
            deriY[i] =  alpha / (2 * x);
        }
    }
    public static void sqrt_Deri(float[] Y, float alpha, float[] deriY, int length) {
        for(int i=0; i<length; i++) deriY[i] = 0.5f * alpha / Y[i];
    }
    
    
    public static float[] sqrt_quadratic2(float[] X1, float[] X2, 
            float k11, float k12, float k22, 
            float k1, float k2, float C) 
    {
        float[] Y = new float[X1.length];
        for(int i=0; i<Y.length; i++) {
            float x1 = X1[i];
            float x2 = X2[i];
            float y = k11*x1*x1 + k12*x1*x2 + k22*x2*x2 +
                    k1*x1 + k2*x2 + C;
            Y[i] = (float) Math.sqrt(y);
        }
        return Y;
    }
    
    public static float[] linear_greater(float alpha, float[] X, float  beta) {
        float[] Y = new float[X.length];
        for(int i=0; i<Y.length; i++) {
            if(alpha * X[i] + beta > 0) Y[i] = 1;
            else Y[i] = 0;
        }
        return Y;
    }
    
    public static float[] linear_greater2(float[] X1, float[] X2, float alpha, float beta, float gamma) {
        float[] Y = new float[X1.length];
        for(int i=0; i<Y.length; i++) {
            if(alpha*X1[i] + beta*X2[i] + gamma > 0) Y[i] = 1;
            else Y[i] = 0;
        }
        return Y;
    }
    
    
    public static void cbrt(float[] X, float[] Y, int length)
    {
        for(int i=0;i<length;i++) Y[i] = (float) Math.cbrt(X[i]);
    }
    public static void cbrt(float alpha, float[] X, float beta, float[] Y, int length)
    {
        for(int i=0;i<length;i++) Y[i] = (float) Math.cbrt(alpha*X[i] + beta);
    }
    
    public static float[] exp(float[] X) {
        float[] Y = new float[X.length];
        for(int i=0; i<X.length; i++) Y[i] = (float) Math.exp(X[i]);
        return Y;
    } 
    public static float[] exp(float alpha, float[] X, float beta) {
        float[] Y = new float[X.length];
        for(int i=0; i<X.length; i++) Y[i] = (float) Math.exp(alpha*X[i] + beta);
        return Y;
    }
    
    public static float[] log(float[] X) {
        float[] Y = new float[X.length];
        for(int i=0; i<X.length; i++) Y[i] = (float) Math.log(X[i]);
        return Y;
    }
    public static float[] log(float alpha, float[] X, float beta) {
        float[] Y = new float[X.length];
        for(int i=0; i<X.length; i++) Y[i] = (float) Math.log(alpha * X[i] + beta);
        return Y;
    }
    
    
    public static void pow(float[] X, float k, float[] Y, int length)
    {
        for(int i=0;i<length;i++) Y[i] = (float) Math.pow(X[i], k);
    } 
    public static void pow(float alpha, float[] X, float beta, float k, float[] Y, int length)
    {
        for(int i=0;i<length;i++) Y[i] = (float) Math.pow(alpha*X[i] + beta, k);
    } 
    
    public static void pow(float k, float[] X, float[] Y, int length)
    {
        for(int i=0;i<length;i++) Y[i] = (float) Math.pow(k, X[i]);
    } 
    public static void pow(float k, float alpha, float[] X, float beta, float[] Y, int length)
    {
        for(int i=0;i<length;i++) Y[i] = (float) Math.pow(k, alpha*X[i] + beta);
    } 
    
    public static void pow(float[] X, float[] Y, float[] Z, int length)
    {
        for(int i=0;i<length;i++) Z[i] = (float) Math.pow(X[i], Y[i]);
    } 
    public static void pow(float a1, float[] X, float b1, float a2, float[] Y, float b2, float[] Z, int length)
    {
        for(int i=0;i<length;i++) Z[i] = (float) Math.pow(a1*X[i]+b1, a2*Y[i]+b2);
    } 
    
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Hyperbolic Function">
    public static void sinh(float[] X, float[] Y, int length)
    {
        for(int i=0;i<length;i++)
        {
            float exp_x=(float) Math.exp(X[i]);
            Y[i] = (exp_x - 1/exp_x) / 2;
        }
    }
    public static void sinh(float alpha, float[] X, float beta, float[] Y, int length)
    {
        for(int i=0;i<length;i++)
        {
            float exp_x=(float) Math.exp(alpha*X[i] + beta);
            Y[i] = (exp_x - 1/exp_x) / 2;
        }
    }
    
    public static void cosh(float[] X, float[] Y, int length)
    {
        for(int i=0;i<length;i++) 
        {
            float exp_x=(float) Math.exp(X[i]);
            Y[i] = (exp_x + 1/exp_x) / 2;
        }
    }
    public static void cosh(float alpha, float[] X, float beta, float[] Y, int length)
    {
        for(int i=0; i<length; i++) 
        {
            float exp_x=(float) Math.exp(alpha*X[i] + beta);
            Y[i] = (exp_x + 1/exp_x) / 2;
        }
    }
    
    public static float[] tanh(float[] X) {
        float[] Y = new float[X.length];
        for(int i=0; i<X.length; i++) {
            float expX = (float) Math.exp(X[i]);
            Y[i] = (expX - 1/expX) / (expX + 1/expX);
        }
        return Y;
    }
    public static void tanh_Dri(float[] Y, float[] deriY, int length) {
        for(int i=0; i<length; i++) deriY[i] = 1 - Y[i]*Y[i];
    }
    
    public static void sigmoid(float[] X, float[] Y, int length) {
        for(int i=0; i<length; i++)
            Y[i] = (float) (1 / (1 + Math.exp(-X[i])));
    }
    public static float[] sigmoid(float[] X) {
        float[] Y = new float[X.length];
        for(int i=0; i<X.length; i++)
            Y[i] = (float) (1 / (1 + Math.exp(-X[i])));
        return Y;
    }
    public static void sigmoid_Deri(float[] Y, float[] deriY, int length) {
        for(int i=0; i<length; i++)
            deriY[i] = Y[i] * (1 - Y[i]);
    }
    
    public static float[] softmax(float[] X) {
        float[] Y = new float[X.length];
        float sum = 0, maxX = Vector.maxValue(X);
        for(int i=0; i<X.length; i++) {
            Y[i] = (float) Math.exp(X[i] - maxX);
            sum += Y[i];
        }
        for(int i=0; i<Y.length; i++) Y[i] /= sum;
        return Y;
    }
    
   
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Basic Function">
    //<editor-fold defaultstate="collapsed" desc="linear">
    public static float[] sadd(float[] X, float C) {
        return linear(1.0f, X, C);
    }
    public static float[] ssub(float[] X, float C) {
        return linear(1.0f, X, -C);
    }
    public static float[] smul(float[] X, float C) {
        return linear(C, X, 0.0f);
    }
    public static float[] sdiv(float[] X, float C) {
        return linear(1.0f / C, X, 0.0f);
    }
    public static float[] linear(float alpha, float[] X, float beta) {
        float[] Y = new float[X.length];
        for(int i=0; i<X.length; i++) Y[i] = alpha*X[i] + beta;
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="rpl">
    public static float[] rpl(float[] X) {
        float[] Y = new float[X.length];
        for(int i=0; i<X.length; i++) Y[i] = 1.0f / X[i];
        return Y;
    }
    
    public static float[] rpl(float alpha, float[] X, float beta, float gamma) {
        float[] Y = new float[X.length];
        for(int i=0; i<X.length; i++) Y[i] = alpha / (X[i] + beta) + gamma;
        return Y;
    }
    
    public static void rpl_Deri(float[] X, float alpha, float beta, float[] deriY, int length){
        for(int i=0; i<length; i++) {
            float x = X[i] + beta;
            deriY[i] = -alpha / (x * x);
        }
    }
    //</editor-fold>
    
    public static void div(
            float alpha1, float[] X1, float beta1,
            float alpha2, float[] X2, float beta2, 
            float gamma,
            float[] Y, int length) 
    {
        for(int i=0; i<length; i++) 
            Y[i] = (alpha1*X1[i] + beta1) / (alpha2*X2[i] + beta2) + gamma;
    }
    
    public static void div_Deri(float[] deriX1, float[] deriX2,
            float[] X1, float alpha1, float beta1,
            float[] X2, float alpha2, float beta2,
            int length)
    {
        for(int i=0; i<length; i++) {
            float x1 = X1[i];
            float x2 = X2[i];
            deriX1[i] = alpha1 / (alpha2*x2 + beta2);
            deriX2[i] = -alpha2 * (alpha1*x1 + beta1) /(alpha2*x2 + beta2)/ (alpha2*x2 + beta2);
        }
    }
    
    //<editor-fold defaultstate="collapsed" desc="linear2">
    public static float[] add(float[] X1, float[] X2) {
        float[] Y = new float[X1.length];
        for(int i=0; i<X1.length; i++) Y[i] = X1[i] + X2[i];
        return Y;
    }
    public static float[] sub(float[] X1, float[] X2) {
        float[] Y = new float[X1.length];
        for(int i=0; i<X1.length; i++) Y[i] = X1[i] - X2[i];
        return Y;
    }
    public static float[] add(float alpha, float[] X1, float beta, float[] X2) {
        float[] Y = new float[X1.length];
        for(int i=0; i<X1.length; i++) Y[i] = alpha*X1[i] + beta*X2[i];
        return Y;
    }
    public static float[] linear2(float[] X1, float[] X2,  float alpha, float beta, float gamma) {
        float[] Y = new float[X1.length];
        for(int i=0; i<X1.length; i++) Y[i] = alpha*X1[i] + beta*X2[i] + gamma;
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="quadratic2">
    public static float[] mul(float[] X1, float[] X2) {
        float[] Y = new float[X1.length];
        for(int i=0; i<X1.length; i++) Y[i] = X1[i] * X2[i];
        return Y;
    }
    public static float[] mul(float alpha, float[] X1, float[] X2) {
        float[] Y = new float[X1.length];
        for(int i=0; i<X1.length; i++) Y[i] = alpha * X1[i] * X2[i];
        return Y;
    }
    public static float[] squareAdd(float[] X1, float[] X2) {
        float[] Y = new float[X1.length];
        for(int i=0; i<X1.length; i++) {
            float x1 = X1[i];
            float x2 = X2[i];
            Y[i] = x1*x1 + x2*x2;
        }
        return Y;
    }
    public static float[] squareSub(float[] X1, float[] X2) {
        float[] Y = new float[X1.length];
        for(int i=0; i<X1.length; i++) {
            float x1 = X1[i];
            float x2 = X2[i];
            Y[i] = x1*x1 - x2*x2;
        }
        return Y;
    }
    public static float[] squareAdd(float alpha, float[] X1, float beta, float[] X2) {
        float[] Y = new float[X1.length];
        for(int i=0; i<X1.length; i++) {
            float x1 = X1[i];
            float x2 = X2[i];
            Y[i] = alpha*x1*x1 + beta*x2*x2;
        }
        return Y;
    }
    public static float[] quadratic2(float[] X1, float[] X2,
            float k11, float k12, float k22,
            float k1, float k2, float C)
    {
        float[] Y = new float[X1.length];
        for(int i=0; i<X1.length; i++) {
            float x1 = X1[i];
            float x2 = X2[i];
            Y[i] = k11*x1*x1 + k12*x1*x2 + k22*x2*x2 + k1*x1 + k2*x2 + C;
        }
        return Y;
    }
    public static void binomial_Deri(
            float[] deriX1, float[] deriX2,
            float[] X1, float[] X2,
            float k11, float k12, float k22, 
            float k1, float k2,
            int length)
    {
        for(int i=0; i<length; i++) {
            float x1 = X1[i];
            float x2 = X2[i];
            deriX1[i] = 2*k11*x1 + k12 * x2 + k1;
            deriX2[i] = 2*k22*x2 + k12 * x1 + k2;
        }
    }
    //</editor-fold>
    
    public static float[] abs(float[] X) {
        float[] Y = new float[X.length];
        for(int i=0; i<X.length; i++) Y[i] = Math.abs(X[i]);
        return Y;
    }
    public static float[] abs(float alpha, float[] X, float beta) {
        float[] Y = new float[X.length];
        for(int i=0; i<X.length; i++) Y[i] = Math.abs(alpha*X[i] + beta);
        return Y;
    }
    
    public static void sign(float[] X, float[] Y, int length) {
        for(int i=0;i<length;i++)  {
            if(X[i]>0) Y[i] = 1;
            else if(X[i] == 0) Y[i] = 0;
            else Y[i] = -1;
        }
    }
    public static void sign(float alpha, float[] X, float beta, float[] Y, int length) {
        for(int i=0; i<length; i++)  {
            float x = alpha*X[i]+beta;
            if(x>0) Y[i] = 1;
            else if(x == 0) Y[i]=0;
            else Y[i] = -1;
        }
    }
    
    public static float[] square(float[] X) {
        float[] Y = new float[X.length];
        for(int i=0; i<X.length; i++) {
            float x = X[i];
            Y[i] = x * x;
        }
        return Y;
    }
    public static float[] square(float alpha, float[] X) {
        float[] Y = new float[X.length];
        for(int i=0; i<X.length; i++) {
            float x = X[i];
            Y[i] = alpha * x * x;
        }
        return Y;
    }
    public static float[] quadratic(float[] X, float alpha, float beta, float gamma) {
        float[] Y = new float[X.length];
        for(int i=0; i<X.length; i++) {
            float x = X[i];
            Y[i] = alpha*x*x +  beta*x + gamma;
        }
        return Y;
    }
    
    public static void quadratic_Deri(float[] X, float alpha, float beta, float[] deriY, int length){
        System.out.println(alpha + ":" + beta);
        for(int i=0; i<length; i++) {
            float x = X[i];
            deriY[i] = 2*alpha*x + beta;
        }
    }
    
    public static void ceil(float[] X, float[] Y, int length)
    {
        for(int i=0;i<length;i++) Y[i]=(float) Math.ceil(X[i]);
    }
    public static void ceil(float alpha, float[] X, float beta, float[] Y, int length)
    {
        for(int i=0;i<length;i++) Y[i]=(float) Math.ceil(alpha*X[i] + beta);
    }
    
    public static void floor(float[] X, float[] Y, int length)
    {
        for(int i=0;i<length;i++) Y[i]=(float) Math.floor(X[i]);
    }
    public static void floor(float alpha, float[] X, float beta, float[] Y, int length)
    {
        for(int i=0;i<length;i++) Y[i]=(float) Math.floor(alpha*X[i] + beta);
    }
    
    public static void max(float[] X, float vmax, float[] Y, int length)
    {
        for(int i=0;i<length;i++) Y[i] = Math.max(X[i], vmax);
    }
    
    public static void min(float[] X, float vmin, float[] Y, int length)
    {
        for(int i=0;i<length;i++) Y[i] = Math.min(X[i], vmin);
    }
    
    public static void clip(float[] X, float min, float max, float[] Y, int length)
    {
        if(min > max) {float t = min; min = max; max = t;}
        for(int i=0;i<length;i++) 
        {
            if(X[i] > max) Y[i] = max;
            else if(X[i] > min) Y[i] = X[i];
            else Y[i] = min;
        }
    }
   
    public static float[] relu(float[] X) {
        float[] Y = new float[X.length];
        for(int i=0; i<X.length; i++) Y[i] = Math.max(X[i], 0);
        return Y;
    }
    public static void relu_deri(float[] Y, float[] deriY, int length)
    {
        for(int i=0;i<length;i++) {
            if(Y[i] > 0) deriY[i] = 1.0f;
            else deriY[i] = 0;
        }
    }
    
    public static float[] leakyRelu(float[] X, float k) {
        float[] Y = new float[X.length];
        for(int i=0; i<X.length; i++) {
            if(X[i] > 0) Y[i] = X[i];
            else Y[i] = k*X[i];
        }
        return Y;
    }
    
    public static void leakyRelu_Deri_X(float[] X, float k, float[] deriY, int length) {
        for(int i=0; i<length; i++)
            deriY[i] = (X[i] > 0 ? 1.0f : k);
    }
    public static void leakyRelu_Deri(float[] Y, float k, float[] deriY, int length) {
        for(int i=0; i<length; i++)
            deriY[i] = (Y[i] > 0 ? 1.0f : k);
    }
    
    public static void sin_Deri(float[] X, float alpha, float beta, float[] deriY, int length){
        for(int i=0; i<length; i++)
            deriY[i] = (float) (alpha * Math.cos(alpha * X[i] + beta));
    }
    
    public static void abs_Deri(float[] X, float alpha, float[] deriY, int length) {
        for(int i=0; i<length; i++){
            float x = X[i];
            if(x > 0) deriY[i] = alpha;
            else if(x < 0 ) deriY[i] = -alpha;
            else deriY[i] = 0;
        }
    }
    
    public static float[] elu(float[] X, float alpha, float k) {
        float[] Y = new float[X.length];
        for(int i=0; i<X.length; i++) {
            if(X[i] > 0) Y[i] = X[i];
            else Y[i]= k * (float) (Math.exp(X[i]) - 1.0f);
            Y[i] *= alpha;
        }
        return Y;
    }
    
    public static void elu_Deri(float[] Y, float alpha, float beta, float[] deriY, int length)
    {
        for(int i=0;i<length;i++) {
            if(Y[i] > 0) deriY[i] = alpha;
            else deriY[i] = Y[i] + alpha*beta;
        }
    }
    public static void elu_Deri_2(float[] X, float alpha, float beta, float[] R, int length)
    {
        for(int i=0;i<length;i++)
        {
            if(X[i]>0) R[i] = 1;
            else R[i] = beta * (float)Math.exp(X[i]);
            R[i] *= alpha;
        }
    }
    
    public static float[] softplus(float[] X) {
        float[] Y = new float[X.length];
        for(int i=0; i<X.length; i++) 
            Y[i] = (float) Math.log1p(Math.exp(X[i]));
        return Y;
    }
    public static void softPlus_Deri(float[] Y, float[] deriY, int length)
    {
        for(int i=0;i<length;i++)
            deriY[i] = 1 - (float) Math.exp(-Y[i]);
    }
    public static void softPlus_Deri_2(float[] X, float[] R, int length)
    {
        for(int i=0;i<length;i++)
            R[i] = 1 - 1/(1 + (float)Math.exp(X[i]));
    }
    
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Logarithm">
    public static void log2(float[] X, float[] Y, int length)
    {
        float log2=(float) Math.log(2);
        for(int i=0;i<length;i++) Y[i]=(float) Math.log(X[i])/log2;
    } 
    public static void log2(float alpha, float[] X, float beta, float[] Y, int length)
    {
        float log2=(float) Math.log(2);
        for(int i=0;i<length;i++) Y[i]=(float) Math.log(alpha*X[i] + beta)/log2;
    } 
    
    public static void log10(float[] X, float[] Y, int length)
    {
        for(int i=0;i<length;i++) Y[i]=(float) Math.log10(X[i]);
    }
    public static void log10(float alpha, float[] X, float beta, float[] Y, int length)
    {
        for(int i=0;i<length;i++) Y[i]=(float) Math.log10(alpha*X[i] + beta);
    }
    
   
    
    public static void log(float v, float[] X, float[] Y, int length) {
        float logv = (float) Math.log(v);
        for(int i=0;i<length;i++) Y[i]=(float) (Math.log(X[i]) / logv);
    }
    public static void log(float v, float alpha, float[] X, float beta, float[] Y, int length)
    {
        float logv=(float) Math.log(v);
        for(int i=0;i<length;i++) Y[i]=Y[i]=(float) (Math.log(alpha*X[i] + beta) / logv);
    }
   
    public static void log(float[] X, float[] Y, float[] Z, int length)
    {
        for(int i=0;i<length;i++) Z[i]=(float) (Math.log(Y[i])/Math.log(X[i]));
    }
    public static void log(float a1, float[] X, float b1, float a2, float[] Y, float b2, float[] Z, int length)
    {
        for(int i=0;i<length;i++) Z[i]=(float) (Math.log(a2*Y[i]+b2)/Math.log(a1*X[i] + b1));
    }
    
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Difference & Pertinence">
    public static void L1(float[] Yh, float[] Y, float[] L) {
        for(int i=0; i<L.length; i++)
            L[i] = Math.abs(Yh[i] - Y[i]);
    }
    public static void L1_deltaYh(float[] Yh, float[] Y, float[] deltaYh) {
        for(int i=0;i<deltaYh.length;i++)
            deltaYh[i] = Math.signum(Yh[i] - Y[i]);
    }
      
    public static void L2(float[] Yh, float[] Y, float[] L) {
        for(int i=0;i<L.length;i++) {
            float div = Yh[i]-Y[i];
            L[i] = 0.5f*div*div;
        }
    }
    public static void L2_deltaYh(float[] Yh, float[] Y, float[] deltaYh) {
        for(int i=0; i<deltaYh.length; i++) 
            deltaYh[i] = Yh[i] - Y[i];
    }
    
    public static void smoothL1(float[] Yh, float[] Y, float[] L) {
        for(int i=0; i< L.length; i++) {
            float div = Math.abs(Yh[i] - Y[i]);
            if(div <= 1) L[i] = 0.5f * div * div;
            else L[i] = div - 0.5f;
        }
    }
    public static void smoothL1_deltaYh(float[] Yh, float[] Y, float[] deltaYh) {
        for(int i=0; i<deltaYh.length; i++) {
            float div = Math.abs(Yh[i] - Y[i]);
            if(div <= 1) deltaYh[i] = Yh[i] - Y[i];
            else deltaYh[i] = Math.signum(Yh[i] - Y[i]);
        }
    }
    
    public static void crossEntropy(float[] Yh, float[] Y, float[] L) {
        for(int i=0;i<L.length;i++)
            L[i] = (float) (-Y[i] * Math.log(Yh[i]) 
                    + (Y[i] - 1) * Math.log(1 - Yh[i]));
    }
    public static void crossEntropy_deltaYh(float[] Yh, float[] Y, float[] deltaYh) {
        for(int i=0; i<deltaYh.length; i++)
            deltaYh[i] = (-Y[i] / Yh[i] + (Y[i] - 1) / (Yh[i] - 1));
    }
    
    public static void balancedCrossEntropy(float[] Yh, float[] Y, float alpha, float beta, float[] L) {
        for(int i=0;i<L.length;i++)
            L[i] = (float) (-alpha * Y[i] * Math.log(Yh[i])
                    + beta * (Y[i] - 1) * Math.log(1 - Yh[i]));
    }
    public static void balancedCrossEntropy_deltaYh(float[] Yh, float[] Y, float alpha, float beta, float[] deltaYh) {
        for(int i=0; i<deltaYh.length; i++)
            deltaYh[i] = -alpha * (Y[i] / Yh[i]) + beta * (Y[i] - 1) / (Yh[i] - 1);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Vector-math-Function">
    public static int mul(int[] a) {return multiple(a, 0, a.length-1);}
    public static int multiple(int[] a, int start, int end) {
        int mul = 1;
        for(int i=start; i<=end; i++) mul *= a[i];
        return mul;
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Vector-Element-Operation">
       /**
     * <pre>
     * consider the input Array{@code left}, {@code right} as two vector
     * int the space with the same dimension, find the summary:
     * for each each components of
     * {@code left}, {@code right}:
     *      {@code c[i] = a[i] + b[i]}
     * </pre>
     * @param a
     * @param b 
     * @param c the summary of vector left and right
     */
    public static void elementAdd(double[] a, double[] b, double[] c)
    {for(int i=0;i<c.length;i++) c[i]=a[i]+b[i];}
    
    public static void elementAdd(float[] a, float[] b, float[] c)
    {for(int i=0;i<c.length;i++) c[i]=a[i]+b[i];}
    
    public static void assign(float[] a, float value)
    {
        for(int i=0; i<a.length; i++) a[i] = value;
    }
    
    
    /**
    * c[i] = alpha*a[i] + beta*b[i]
    * @param alpha
    * @param a
    * @param beta
    * @param b
    * @param c 
    */
    public static void elementAdd(double alpha, double[] a, double beta ,double[] b, double[] c)
    {for(int i=0;i<c.length;i++) c[i]= alpha*a[i] + beta*b[i];}
    
    public static void elementAdd(float alpha, float[] a, float beta ,float[] b, float[] c)
    {for(int i=0;i<c.length;i++) c[i]= alpha*a[i] + beta*b[i];}
    
    
    public static void elementAddSquare(float[] a, float[] b, float[] c)
    {
        for(int i=0;i<c.length;i++) c[i] = a[i] + b[i]*b[i];
    }
     
    public static void elementAddSquare(float alpha, float[] a, float beta ,float[] b, float[] c)
    {
        for(int i=0;i<c.length;i++) c[i] = alpha*a[i] + beta*b[i]*b[i];
    }
    
     /**
     * <pre>
     * consider the input Array{@code left}, {@code right} as two vector
     * int the space with the same dimension, find the difference.
     * for each each components of
     * {@code left}, {@code right}:
     *      {@code result[i]=left[i]-right[i]}
     * </pre>
     * @param a
     * @param b 
     * @param c the difference between vector left and right
     */
    public static void elementSub(double[] a, double[] b, double[] c)
    {for(int i=0;i<c.length;i++) c[i]=a[i]-b[i];}
    
    public static void elementSub(float[] a, float[] b, float[] c)
    {for(int i=0;i<c.length;i++) c[i]=a[i]-b[i];}
    
    /**
     * c[i] = alpha*a[i] - beta*b[i]
     * @param alpha
     * @param a
     * @param beta
     * @param b
     * @param c 
     */
    public static void elementSub(double alpha, double[] a, double beta, double[] b, double[] c)
    {for(int i=0;i<c.length;i++) c[i] = alpha*a[i] - beta*b[i];}
    
    public static void elementSub(float alpha, float[] a, float beta, float[] b, float[] c)
    {for(int i=0;i<c.length;i++) c[i] = alpha*a[i] - beta*b[i];}
    
    /**
     * <pre>
     * This function may be widely used int Neural Network.
     * for each element of the input Arrays {@code left}, {@code right}:
     *      {@code c[i] = k * a[i] + (1-k) * b[i];}
     * </pre>
     * @param c
     * @param a
     * @param k
     * @param b 
     */
    public static void momentum(double[] c, double[] a, double k, double[] b)
    {Vector.elementAdd(k, a, 1-k, b, c);}
    
    /**
     * <pre>
     * Consider the two input Arrays {@code left}, {@code right} as two 
     * vectors, compute the Hadamard product of them.
     * for each component of {@code left}, {@code right}:
     *      {@code c[i] = a[i]*b[i]}
     * </pre>
     * @param a
     * @param b 
     * @param c
     */
    public static void elementMul(double[] a, double[] b, double[] c)
    {for(int i=0;i<c.length;i++) c[i]=a[i]*b[i];}
    
    public static void elementMul(float[] a, float[] b, float[] c)
    {for(int i=0;i<c.length;i++) c[i]=a[i]*b[i];}
    
    /**
     * c[i] = k*a[i]*b[i]
     * @param k
     * @param a
     * @param b
     * @param c 
     */
    public static void elementMul(double k, double[] a, double[] b, double[] c){
        for(int i=0;i<c.length;i++) c[i]=k*a[i]*b[i];
    }
    
    public static void elementMul(float k, float[] a, float[] b, float[] c){
        for(int i=0;i<c.length;i++) c[i]=k*a[i]*b[i];
    }
    
    public static void linear(float alpha, float[] a, float beta, float[] b){
        for(int i=0; i<a.length; i++) b[i] = alpha*a[i] + beta;
    }
    
    public static void elementMul(float a1, float[] A, float b1, 
            float a2, float[] B, float b2, float[] C)
    {
        for(int i=0;i<C.length;i++) C[i]=(a1*A[i] + b1)*(a2*B[i] + b2);
    }
    /**
     * <pre>
     * Consider the two input Arrays {@code left}, {@code right} as two 
     * vectors, compute the division by each component of them.
     * for each component of {@code left}, {@code right}:
     *      {@code c[i] = a[i]/b[i]}
     * </pre>
     * @param c
     * @param a
     * @param b 
     */
    public static void elementDiv(double[] a, double[] b, double[] c)
    {for(int i=0;i<c.length;i++) c[i]=a[i]/b[i];}
    
    public static void elementDiv(float[] a, float[] b, float[] c)
    {for(int i=0;i<c.length;i++) c[i]=a[i]/b[i];}
    
    /**
     * c[i] = k*a[i]/b[i].
     * @param k
     * @param c
     * @param a
     * @param b 
     */
    public static void elementDiv(float k, double[] a, double[] b, double[] c)
    {for(int i=0;i<c.length;i++) c[i] = k*a[i]/b[i];}
    
    public static void elementDiv(float k, float[] a, float[] b, float[] c)
    {for(int i=0;i<c.length;i++) c[i] = k*a[i]/b[i];}
    
    /**
     * b[i] = k/a[i].
     * @param k
     * @param a
     * @param b 
     */
    public static void elementRpl(float k, double[] a, double[] b)
    {for(int i=0;i<b.length;i++) b[i] = k/a[i];}
    
    public static void elementRpl(float k, float[] a, float[] b)
    {for(int i=0;i<b.length;i++) b[i] = k/a[i];}
    
    /**
     * b[i] = a[i]+k.
     * @param a
     * @param k
     * @param b 
     */
    public static void elementScalarAdd(double[] a, double k, double[] b)
    {for(int i=0;i<b.length;i++) b[i] = a[i]+k;}
    
    public static void elementScalarAdd(float[] a, float k, float[] b)
    {for(int i=0;i<b.length;i++) b[i] = a[i]+k;}
    
    /**
     * b[i] = alpha*a[i] + beta.
     * @param alpha
     * @param a
     * @param beta
     * @param b 
     */
    public static void elementScalarAdd(double alpha, double[] a, double beta, double[] b)
    {for(int i=0;i<b.length;i++) b[i] = alpha*a[i] + beta;}
    
    public static void elementScalarAdd(float alpha, float[] a, float beta, float[] b)
    {for(int i=0;i<b.length;i++) b[i] = alpha*a[i] + beta;}
    
    /**
     * b[i] = a[i]*k.
     * @param a
     * @param k
     * @param b 
     */
    public static void elementScalarMul(double[] a, double k, double[] b)
    {for(int i=0;i<b.length;i++) b[i] = a[i]*k;}
    
    public static void elementScalarMul(float[] a, float k, float[] b)
    {for(int i=0;i<b.length;i++) b[i] = a[i]*k;}
    
    /**
     * b[i] = a[i]/k.
     * @param a
     * @param k
     * @param b 
     */
    public static void elementScalarDiv(double[] a, double k, double[] b) {
        for(int i=0;i<b.length;i++) b[i] = a[i]/k;
    }
    
    public static void elementScalarDiv(float[] a, float k, float[] b) {
        for(int i=0;i<b.length;i++) b[i] = a[i]/k;
    }
    
    public static void Momentum(float[] W, float[] deltaW,
            float[] V, float a1, float a2,
            float lr_t, int length)
    {
        for(int i=0; i<length; i++) {
            V[i] = a1*V[i] + a2*deltaW[i];
            W[i] = W[i] - lr_t*V[i];
        }
    }
    
    public static void SGDMN(float[] W, float[] deltaW,
            float[] V, float momentum, float dampen, float nesterov, 
            float lr, int length)
    {
        float K = (nesterov * momentum) + (1.0f - nesterov);
        for(int i=0; i<length; i++) {
            V[i] = momentum*V[i] + (1 - dampen)*deltaW[i];
            float step = nesterov * deltaW[i] + K*V[i];
            W[i] -= lr*step;
        }
    }
    
    public static void RMSprop(float[] W, float[] deltaW,
            float[] S, float a1, float a2, float e,
            float k, int length)
    {
        for(int i=0;i<length;i++)
        {
            S[i] = a1*S[i] + a2*deltaW[i]*deltaW[i];
            W[i] = W[i] - k * deltaW[i]/((float)Math.sqrt(S[i]) + e);
        }
    }
    
    public static void Adam(float[] W, float[] deltaW,
            float[] V, float a1, float a2,
            float[] S, float b1, float b2, float e,
            float lr, int length)
    {
        for(int i=0;i<length;i++)
        {
            V[i] = a1*V[i] + a2*deltaW[i];
            S[i] = b1*S[i] + b2*deltaW[i]*deltaW[i];
            W[i] = W[i] - lr * V[i]/((float)Math.sqrt(S[i]) + e);
        }
    }
    
      public static void Adamod(float[] W, float[] deltaW,
            float[] V, float a1, float a2,
            float[] S, float b1, float b2, float e,
            float[] G, float c1, float c2,
            float lr, int length)
    {
        for(int i=0;i<length;i++)
        {
            V[i] = a1*V[i] + a2*deltaW[i];
            S[i] = b1*S[i] + b2*deltaW[i]*deltaW[i];
            
            float neta = (float) (lr / (Math.sqrt(S[i]) + e));
            G[i] = c1*G[i] + c2*neta;
            
            W[i] -= Math.min(neta, G[i]) * V[i];
        }
    }
    
    
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Vector-Creator">
    public static double[] sequence(int length, double base, double div)
    {
        double[] arr=new double[length];
        arr[0]=base;
        for(int i=1;i<arr.length;i++) arr[i]=arr[i-1]+div;
        return arr;
    }
    public static int[] sequence(int length, int base, int div)
    {
        int[] arr=new int[length];
        arr[0]=base;
        for(int i=1;i<arr.length;i++) arr[i]=arr[i-1]+div;
        return arr;
    }
    public static int[] sequence(int length)
    {
        return sequence(length, 0, 1);
    }
    public static <T> T[] sequence(int length, Class<T> clazz, SequenceCreator<T> sc)
    {
        T[] arr=(T[]) Array.newInstance(clazz, length);
        for(int i=0;i<arr.length;i++) arr[i]=sc.create(i);
        return arr;
    }
    
    public static float[] zeros(int length) { return constants(0.0f, length); }
    public static float[] ones(int length) { return constants(1.0f, length); }
    public static float[] constants(float C, int length) {
        float[] arr = new float[length];
        for(int i=0;i<arr.length;i++) arr[i] = C;
        return arr;
    }
    
    public static<T> Collection<T> collection(T[] arr)
    {
        ZArrayList r=new ZArrayList<>();
        for(T v:arr) r.add(v);
        return r;
    }
    //<editor-fold defaultstate="collapsed" desc="ExRandom:int">
    public static int[] randomNRIntVector(int len) {
        return Lang.exRandom().nextNRIntVector(len);
    }
    public static int[] randomNRIntVector(int len, int max) {
        return Lang.exRandom().nextNRIntVector(len, 0, max);
    }
    public static int[] randomNRIntVector(int len, int min, int max) {
        return Lang.exRandom().nextNRIntVector(len, min, max);
    }
    
    public static int[] randomIntVector(int width) {
        return Lang.exRandom().nextIntVector(width);
    }
    public static void randomIntVector(int[] v) {
        Lang.exRandom().nextIntVector(v);
    }
    public static int[] randomIntVector(int width, int max) {
        return Lang.exRandom().nextIntVector(width, 0,  max);
    }
    public static void randomIntVector(int[] v, int max) {
        Lang.exRandom().nextIntVector(v, 0, max);
    }
    public static int[] randomIntVector(int width, int min, int max ) {
        return Lang.exRandom().nextIntVector(width, min, max);
    }
    public static void randomIntVector(int[] v, int max, int min) {
        Lang.exRandom().nextIntVector(v, min, max);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="ExRandom:byte">
    public static byte[] randomByteVector(int width) {
        return Lang.exRandom().nextBytes(width);
    }
    public static byte[] randomByteVector(int width, byte min, byte max) {
        return Lang.exRandom().nextBytes(width, min, max);
    }
    public static byte[] randomByteVector(int width, byte max) {
        return Lang.exRandom().nextBytes(width, (byte)0, max);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="ExRandom:double">
    public static double[] randomNRDoubleArray(int len)
    {
        return Lang.exRandom().nextNRDoubleVector(len);
    }
    public static double[] randomNRGaussianArray(int len)
    {
        return Lang.exRandom().nextNRGaussianVector(len);
    }
    public static double[] randomNRDoubleArray(int len, double max)
    {
        return Lang.exRandom().nextNRDoubleVector(len, 0, max);
    }
    public static double[] randomNRGaussianArray(int len, double max)
    {
        return Lang.exRandom().nextNRDoubleVector(len, 0, max);
    }
    public static double[] randomNRDoubleArray(int len, double min, double max)
    {
        return Lang.exRandom().nextNRDoubleVector(len, min, max);
    }
    public static double[] randomNRGaussianArray(int len, double min, double max)
    {
        return Lang.exRandom().nextNRGaussianVector(len, min, max);
    }
    @Passed
    public static double[] randomDoubleVector(int width)
    {
        return Lang.exRandom().nextDoubleVector(width);
    }
    @Passed
    public static double[] randomGaussianVector(int width)
    {
        return Lang.exRandom().nextGaussianVector(width);
    }
    @Passed
    public static void randomDoubleVector(double[] v)
    {
        Lang.exRandom().nextDoubleVector(v);
    }
    @Passed
    public static void randomGaussianVector(double[] v)
    {
        Lang.exRandom().nextGaussianVector(v);
    }
    @Passed
    public static double[] randomDoubleVector(int width, double max)
    {
        return Lang.exRandom().nextDoubleVector(width, max);
    }
    @Passed
    public static void randomDoubleVector(double[] v, double max)
    {
        Lang.exRandom().nextDoubleVector(v, max);
    }
    @Passed
    public static double[] randomDoubleVector(int width, double min ,double max)
    {
        return Lang.exRandom().nextDoubleVector(width, min, max);
    }
    @Passed
    public static double[] randomGaussianVector(int width, double min, double max)
    {
        return Lang.exRandom().nextGaussinVector(width, min, max);
    }
    @Passed
    public static void randomDoubleVector(double[] v, double min, double max)
    {
        Lang.exRandom().nextDoubleVector(v, min, max);
    }
    @Passed
    public static void randomGaussianVector(double[] v, double min, double max)
    {
        Lang.exRandom().nextGaussianVector(v, min, max);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="ExRandom:float">
    @Passed
    public static float[] randomFloatVector(int width) {
        return Lang.exRandom().nextFloatVector(width);
    }
    @Passed
    public static void randomFloatVector(float[] v) {
        Lang.exRandom().nextFloatVector(v);
    }
    @Passed
    public static float[] randomFloatVector(int width, float max) {
        return Lang.exRandom().nextFloatVector(width, 0, max);
    }
    @Passed
    public void randomFloatVector(float[] v, float max) {
        Lang.exRandom().nextFloatVector(v, 0, max);
    }
    @Passed
    public static float[] randomFloatVector(int width, float min, float max) {
        return Lang.exRandom().nextFloatVector(width, min, max);
    }
    @Passed
    public void randomFloatVector(float[] v, float min, float max) {
        Lang.exRandom().nextFloatVector(v, min, max);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="ExRandom:Object">
    public static <T> T[] randomObjectVector(int width, Class<T> clazz, RandomSupplier<T> sp)
    {
        return Lang.exRandom().nextObjectVector(width, clazz, sp);
    }
    public static Object[] randomObjectVector(int width, RandomSupplier sp)
    {
        return Lang.exRandom().nextObjectVector(width, sp);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="ExRandom:Extensive">
    @Passed
    public static String[] randomStringVector(int len, int strlen)
    {
        return Lang.exRandom().nextStringVector(len, strlen);
    }
    public static String[] randomNeatStringVector(int len, int strlen)
    {
        return Lang.exRandom().nextNeatStringVector(len, strlen);
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Vector-Manage-Function">
    //<editor-fold defaultstate="collapsed" desc="Array-Copy">
    public static int[] arrayCopy(int[] a)
    {
        int[] arr = new int[a.length];
        System.arraycopy(a, 0, arr, 0, a.length);
        return arr;
    }
    public static int[] arrayCopy(int[] a, int low, int high)
    {
        int[] arr = new int[high - low + 1];
        System.arraycopy(a, low, arr, 0, high-low+1);
        return arr;
    }
    
    public static float[] arrayCopy(float[] a)
    {
        float[] arr=new float[a.length];
        System.arraycopy(a, 0, arr, 0, a.length);
        return arr;
    }
    public static float[] arrayCopy(float[] a, int low, int high)
    {
        float[] arr=new float[high-low+1];
        System.arraycopy(a, low, arr, 0, high-low+1);
        return arr;
    }
    
    public static double[] arrayCopy(double[] a)
    {
        double[] arr=new double[a.length];
        System.arraycopy(a, 0, arr, 0, a.length);
        return arr;
    }
    public static double[] arrayCopy(double[] a, int low, int high)
    {
        double[] arr=new double[high-low+1];
        System.arraycopy(a, low, arr, 0, high-low+1);
        return arr;
    }
    
    public static Object[] arrayCopy(Object[] a)
    {
        Object[] arr=new Object[a.length];
        System.arraycopy(a, 0, arr, 0, a.length);
        return arr;
    }
    public static Object[] arrayCopy(Object[] a, int low, int high)
    {
        Object[] arr=new Object[high-low+1];
        System.arraycopy(a, low, arr, 0, high-low+1);
        return arr;
    }
    
    public static Comparable[] arrayCopy(Comparable[] a)
    {
        Comparable[] arr=new Comparable[a.length];
        System.arraycopy(a, 0, arr, 0, a.length);
        return arr;
    }
    public static Comparable[] arrayCopy(Comparable[] a, int low, int high)
    {
        Comparable[] arr=new Comparable[high-low+1];
        System.arraycopy(a, low, arr, 0, high-low+1);
        return arr;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Next-Permutation">
    @Passed
    public static void reverse(int[] a, int low, int high)
    {
        for(int t;low<high;low++, high--)
            {t=a[low];a[low]=a[high];a[high]=t;}
    }
    public static void reverse(int[] a)
    {
        reverse(a, 0, a.length-1);
    }
    @Passed
    public static void reverse(char[] a, int low, int high)
    {
        for(char t;low<high;low++, high--)
            {t=a[low];a[low]=a[high];a[high]=t;}
    }
    public static void reverse(char[] a)
    {
        reverse(a, 0, a.length-1);
    }
    @Passed
    public static boolean nextPermutation(int[] a, int low, int high)
    {
        int cur=high, pre=cur-1;
        for(;cur>low&&a[cur]<=a[pre];cur--,pre--);
        if(cur<=low) return false;
        
        for(cur=high;a[cur]<=a[pre];cur--);
        int t=a[cur];a[cur]=a[pre];a[pre]=t;
        reverse(a, pre+1, high);//让最小的一段开头最大，结尾最小
        return true;
    }
    public static boolean nextPermutation(int[] a)
    {
        return nextPermutation(a, 0, a.length-1);
    }
    @Passed
    public static boolean nextPermutation(char[] a, int low, int high)
    {
        int cur=high, pre=cur-1;
        for(;cur>low&&a[cur]<=a[pre];cur--,pre--);
        if(cur<=low) return false;
        
        for(cur=high;a[cur]<=a[pre];cur--);
        char t=a[cur];a[cur]=a[pre];a[pre]=t;
        reverse(a, pre+1, high);
        return true;
    }
    public static boolean nextPermutation(char[] a)
    {
        return nextPermutation(a, 0, a.length-1);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Max-Multiple">
    public static final long DEF_MAX_MULTIPLE_MOD=1000000009;
    @Passed
    public static long maxMultiple(int[] a, int low, int high, int k, long mod)
    {
        if(a.length<k) throw new IAE("K is greater than the length of Array.");
        long r=1, sign=1;
        Sort.sort(a, low ,high);
        
        if((k&1)==1)
        {
            r=a[high--];k--;
            if(r<0) sign=-1;
        }
        for(long lows, highs;k>0;k-=2)
        {
            lows=a[low]*a[low+1];highs=a[high]*a[high-1];
            if(sign*lows>sign*highs) {r=(r*(lows%mod))%mod;low-=2;}
            else {r=(r*(highs%mod))%mod;high+=2;}
        }
        return r;
    }
    public static long maxMultiple(int[] a, int k, int mod)
    {
        return maxMultiple(a, 0, a.length-1, k, mod);
    }
    @Passed
    public static long maxMultiple(int[] a, int low, int high, int k)
    {
        if(a.length<k) throw new IAE("K is greater than the length of Array.");
        long r=1, sign=1;
        Sort.sort(a, low ,high);
        
        if((k&1)==1)
        {
            r=a[high--];k--;
            if(r<0) sign=-1;
        }
        for(long lows, highs;k>0;k-=2)
        {
            lows=a[low]*a[low+1];highs=a[high]*a[high-1];
            if(sign*lows>sign*highs) {r*=lows;low-=2;}
            else {r*=highs;high+=2;}
        }
        return r;
    }
    public static long maxMultiple(int[] a, int k)
    {
        return maxMultiple(a, 0, a.length-1, k);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Remove-Duplcate"> 
    public static int removeDuplcateIfSorted(int[] arr)
    {
        if(arr==null||arr.length==0) return 0;
        int last=arr[0];
        int j=1;
        for(int i=1;i<arr.length;i++)
            if(last!=arr[i]) last=arr[j++]=arr[i];
        return j;
    }
    public static int removeDuplcateIfSorted(double[] arr)
    {
        if(arr==null||arr.length==0) return 0;
        double last=arr[0];
        int j=1;
        for(int i=1;i<arr.length;i++)
            if(last!=arr[i]) last=arr[j++]=arr[i];
        return j;
    }
    public static int removeDuplcateIfSorted(Object[] arr)
    {
        if(arr==null||arr.length==0) return 0;
        Object last=arr[0];
        int j=1;
        for(int i=1;i<arr.length;i++)
            if(last!=arr[i]||!last.equals(arr[i])) last=arr[j++]=arr[i];
        return j;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="remove function">
    /**
     * <pre>
     * We used an optimized way to compact array, find all no-null
     * elements of the input Array {@code arr}, and move all them together
     * from the start of {@code arr}， without changing the order of elements
     * which is not null. 
     * This algorithm work like this:
     * (1)find the first element block in the list that's not null
     * (2)find the next null element block
     * (3)find move the block to take place of the null block
     * (4)looped
     * </pre>
     * @param arr
     * @param low the start index
     * @param high the end index
     * @return the length of new Array
     */
    @Passed
    public static int removeNull(Object[] arr, int low, int high)//checked
    {
        int start,end,nstart;
        //find the first element block in the list that's not null
        for(end=high;end>=low&&arr[end]==null;end--);
        //looped block move to take place of null block
        while(end>low)//if end==0, means there is no null element
        {
            //find the not null block
            for(start=end-1;start>=low&&arr[start]!=null;start--);
            if(start<low) break;//all element is not null

            //find the null block
            for(nstart=start-1;nstart>=low&&arr[nstart]==null;nstart--);
            
            //move block
            System.arraycopy(arr, start+1, arr, nstart+1, end-=start);
            end+=nstart;
        }
        System.out.println("old:"+end);
        return end;
    }
    /**
     * <pre>
     * The operating priciple is similar to {@link Vector#removeNull(java.lang.Object[], int, int) },
     * the only difference is: this function remove all elements which 
     * is equal to {@code value}.
     * </pre>
     * @param arr
     * @param val 
     * @param low the start index
     * @param high the end index
     * @return the length of new Array
     */
    @Passed
    public static int remove(Object[] arr, Object val, int low, int high)
    {
        if(val==null) return Vector.removeNull(arr, low, high);
        int start,end,nstart;
        //find the first element block in the list that isn't equal to value
        for(end=high;end>=low&&val.equals(arr[end]);end--);
        //looped block move to take place of equaled block
        while(end>low)//if end==0, means there is no null element
        {
            //find the not-equal block
            for(start=end-1;start>=low&&!val.equals(arr[start]);start--);
            if(start<low) break;//all element is not null

            //find the equaled block
            for(nstart=start-1;nstart>=low&&val.equals(arr[nstart]);nstart--);
            
            //move block
            System.arraycopy(arr, start+1, arr, nstart+1, end-=start);
            end+=nstart;
        }
        return end;
    }
    /**
     * <pre>
     * The operating priciple is similar to {@link Vector#removeNull(java.lang.Object[], int, int) },
     * the only difference is: this function remove all elements which 
     * is equal to {@code value}.
     * </pre>
     * @param arr
     * @param val 
     * @param low the start index
     * @param high the end index
     * @return the length of new Array
     */
    public static int remove(int[] arr, int val, int low, int high)
    {
        int start,end,nstart;
         //find the first element block in the list that isn't equal to value
        for(end=high;end>=low&&arr[end]==(val);end--);
        while(end>low)//if end==0, means there is no null element
        {
            //find the not-equal block
            for(start=end-1;start>=low&&arr[start]!=val;start--);
            if(start<low) break;//all element is not null

            //find the equaled block
            for(nstart=start-1;nstart>=low&&arr[nstart]==val;nstart--);
            
            //move block
            System.arraycopy(arr, start+1, arr, nstart+1, end-=start);
            end+=nstart;
        }
        return end;
    }
    /**
     * <pre>
     * The operating priciple is similar to {@link Vector#removeNull(java.lang.Object[], int, int) },
     * the only difference is: this function remove all elements which 
     * meets the need of {@code pre}.
     * </pre>
     * @param arr
     * @param pre
     * @param low the start index
     * @param high the end index
     * @return he length of new Array
     */
    @Passed
    public static int remove(Object[] arr, Predicate pre, int low, int high)
    {
        int start,end,nstart;
        //find the first element block in the list that doesn't meet pre
        for(end=high;end>=low&&pre.test(arr[end]);end--);
        //looped block move to take place of satisfied block
        while(end>low)
        {
            //find the not-satisfied block
            for(start=end-1;start>=low&&!pre.test(arr[start]);start--);
            if(start<low) break;
            
            //find the satisfied block
            for(nstart=start-1;nstart>low&&pre.test(arr[nstart]);nstart--);
            
            //move block
            System.arraycopy(arr, start+1, arr, nstart+1, end-=start);
            end+=nstart;
        }
        return high;
    }
      /**
     * <pre>
     * The operating priciple is similar to {@link Vector#removeNull(java.lang.Object[], int, int) },
     * the only difference is: this function remove all elements which 
     * meets the need of {@code pre} and {@code condition}:
     *      {@code pre.test(each element, condition)}.
     * </pre>
     * @param arr
     * @param pre
     * @param condition the second parameter of {@link BiPredicate#test(Object, Object) }
     * @param low the start index
     * @param high the end index
     * @return he length of new Array
     */
    @Passed
    public static int remove(Object[] arr, BiPredicate pre, Object condition, int low, int high)
    {
        int start,end,nstart;
        //find the first element block in the list that doesn't meet pre
        for(end=high;end>=low&&pre.test(arr[end], condition);end--);
        //looped block move to take place of satisfied block
        while(end>low)
        {
            //find the not-satisfied block
            for(start=end-1;start>=low&&!pre.test(arr[start], condition);start--);
            if(start<low) break;
            
            //find the satisfied block
            for(nstart=start-1;nstart>low&&pre.test(arr[nstart], condition);nstart--);
            
            //move block
            System.arraycopy(arr, start+1, arr, nstart+1, end-=start);
            end+=nstart;
        }
        return end;
    }
    
    public static int removeNull(Object[] arr, int high)
    {
        return Vector.removeNull(arr, 0, high);
    }
    public static int removeNull(Object[] arr)
    {
        return Vector.removeNull(arr, 0, arr.length-1);
    }
    
    public static int remove(Object[] arr, Object val, int high)
    {
        return Vector.remove(arr, val, 0, high);
    }
    public static int remove(Object[] arr, Object val)
    {
        return Vector.remove(arr, val, 0, arr.length-1);
    }
    
    public static int remove(int[] arr, int val, int high)
    {
        return Vector.remove(arr, val, 0, high);
    }
    public static int remove(int[] arr, int val)
    {
        return Vector.remove(arr, val, 0, arr.length-1);
    }
    
    public static int remove(Object[] arr, Predicate pre, int high)
    {
        return Vector.remove(arr, pre, 0, high);
    }
    public static int remove(Object[] arr, Predicate pre)
    {
        return Vector.remove(arr, pre, 0, arr.length-1);
    }
    
    public static int remove(Object[] arr, BiPredicate pre, Object condition, int high)
    {
        return Vector.remove(arr, pre, condition, 0, high);
    }
    public static int remove(Object[] arr, BiPredicate pre, Object condition)
    {
        return Vector.remove(arr, pre, condition, 0, arr.length-1);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="QuickSort:partition">
    @Passed
    public static int partition(Comparable[] a, int low, int high) 
    {
        Comparable t,p=a[low];
        while(low<high) 
        {
            while(low<high&&a[high].compareTo(p)>=0) high--;
            t=a[low];a[low]=a[high];a[high]=t;
            while(low<high&&a[low].compareTo(p)<=0) low++;
            t=a[low];a[low]=a[high];a[high]=t;
        }
        return low;
    }
    @Passed
    public static int partition(int[] a, int low, int high) 
    {
        int t,p=a[low];
        while(low<high) 
        {
            while(low<high&&a[high]>=p) high--;
            t=a[low];a[low]=a[high];a[high]=t;
            while(low<high&&a[low]<=p) low++;
            t=a[low];a[low]=a[high];a[high]=t;
        }
        return low;
    }
    @Passed
    public static int partition(float[] a, int low, int high) 
    {
        float t,p=a[low];
        while(low<high) 
        {
            while(low<high&&a[high]>=p) high--;
            t=a[low];a[low]=a[high];a[high]=t;
            while(low<high&&a[low]<=p) low++;
            t=a[low];a[low]=a[high];a[high]=t;
        }
        return low;
    }
     @Passed
    public static int partition(double[] a, int low, int high) 
    {
        double t,p=a[low];
        while(low<high) 
        {
            while(low<high&&a[high]>=p) high--;
            t=a[low];a[low]=a[high];a[high]=t;
            while(low<high&&a[low]<=p) low++;
            t=a[low];a[low]=a[high];a[high]=t;
        }
        return low;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="QuickSort:threePartition">
    @Passed
    public static long threePartition(Comparable[] a, int low, int high)
    {
        Comparable p=a[low],t;
        for(int k=low+1;k<=high;)
        {
            while(k<=high)
            {
                if(a[k].compareTo(p)<0) {t=a[low];a[low]=a[k];a[k]=t;low++;}
                else if(a[k].compareTo(p)>0) break;
                k++;
            }
            while(k<=high)
            {
                if(a[high].compareTo(p)>0) high--;
                else if(a[high].compareTo(p)==0) 
                {
                    t=a[k];a[k]=a[high];a[high]=t;
                    k++;high--;break;
                }
                else 
                {
                    t=a[low];a[low]=a[high];a[high]=a[k];a[k]=t;
                    low++;k++;high--;break;
                }
            }
        }
        return ((long)low & 0xFFFFFFFFl) | (((long)high << 32) & 0xFFFFFFFF00000000l);
    }
    @Passed
    public static long threePartition(int[] a, int low, int high)
    {
        int p=a[low],t;
        for(int k=low+1;k<=high;)
        {
            while(k<=high)
            {
                if(a[k]<p) {t=a[low];a[low]=a[k];a[k]=t;low++;}
                else if(a[k]>p) break;
                k++;
            }
            while(k<=high)
            {
                if(a[high]>p) high--;
                else if(a[high]==p) 
                {
                    t=a[k];a[k]=a[high];a[high]=t;
                    k++;high--;break;
                }
                else 
                {
                    t=a[low];a[low]=a[high];a[high]=a[k];a[k]=t;
                    low++;k++;high--;break;
                }
            }
        }
        return ((long)low & 0xFFFFFFFFl) | (((long)high << 32) & 0xFFFFFFFF00000000l);
    }
    @Passed
    public static long threePartition(float[] a, int low, int high)
    {
        float p=a[low],t;
        for(int k=low+1;k<=high;)
        {
            while(k<=high)
            {
                if(a[k]<p) {t=a[low];a[low]=a[k];a[k]=t;low++;}
                else if(a[k]>p) break;
                k++;
            }
            while(k<=high)
            {
                if(a[high]>p) high--;
                else if(a[high]==p) 
                {
                    t=a[k];a[k]=a[high];a[high]=t;
                    k++;high--;break;
                }
                else 
                {
                    t=a[low];a[low]=a[high];a[high]=a[k];a[k]=t;
                    low++;k++;high--;break;
                }
            }
        }
        return ((long)low & 0xFFFFFFFFl) | (((long)high << 32) & 0xFFFFFFFF00000000l);
    }
    @Passed
    public static long threePartition(double[] a, int low, int high)
    {
        double p=a[low],t;
        for(int k=low+1;k<=high;)
        {
            while(k<=high)
            {
                if(a[k]<p) {t=a[low];a[low]=a[k];a[k]=t;low++;}
                else if(a[k]>p) break;
                k++;
            }
            while(k<=high)
            {
                if(a[high]>p) high--;
                else if(a[high]==p) 
                {
                    t=a[k];a[k]=a[high];a[high]=t;
                    k++;high--;break;
                }
                else 
                {
                    t=a[low];a[low]=a[high];a[high]=a[k];a[k]=t;
                    low++;k++;high--;break;
                }
            }
        }
        return ((long)low & 0xFFFFFFFFl) | (((long)high << 32) & 0xFFFFFFFF00000000l);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="QuickSort:DualPivotPartition">
    @Passed
    public static long dualPivotPartition(Comparable[] a, int low ,int high)
    {
        Comparable t;
        if(a[low].compareTo(a[high])>=0) {t=a[low];a[low]=a[high];a[high]=t;}
        Comparable p1=a[low], p2=a[high];
        int i=low+1,k=low+1,j=high-1;
        while(k<=j)
        {
            while(k<=j)
            {
                if(a[k].compareTo(p1)<0) {t=a[i];a[i]=a[k];a[k]=t;i++;}
                else if(a[k].compareTo(p2)>=0) break;
                k++;
            }
            while(k<=j)
            {
                if(a[j].compareTo(p2)>0) j--;
                else if(a[j].compareTo(p1)>=0&&a[j].compareTo(p2)<=0) 
                {
                    t=a[j];a[j]=a[k];a[k]=t;
                    k++;j--;break;
                }
                else
                {
                    t=a[j];a[j]=a[k];a[k]=a[i];a[i]=t;
                    k++;i++;j--;break;
                }
            }
        }
        i--;j++;
        t=a[low];a[low]=a[i];a[i]=t;
        t=a[high];a[high]=a[j];a[j]=t;
        return ((long)i & 0xFFFFFFFFl) | (((long)j << 32) & 0xFFFFFFFF00000000l);
    }
    @Passed
    public static long dualPivotPartition(Object[] a, Comparator cmp, int low ,int high)
    {
        Object t;
        if(cmp.compare(a[low], a[high])>=0) {t=a[low];a[low]=a[high];a[high]=t;}
        Object p1=a[low],p2=a[high];
        int i=low+1,k=low+1,j=high-1;
        while(k<=j)
        {
            while(k<=j)
            {
                if(cmp.compare(a[k], p1)<0) {t=a[i];a[i]=a[k];a[k]=t;i++;}
                else if(cmp.compare(a[k], p2)>=0) break;
                k++;
            }
            while(k<=j)
            {
                if(cmp.compare(a[j], p2)>0) j--;
                else if(cmp.compare(a[j], p1)>=0&&cmp.compare(a[j], p2)<=0) 
                {
                    t=a[j];a[j]=a[k];a[k]=t;
                    k++;j--;break;
                }
                else
                {
                    t=a[j];a[j]=a[k];a[k]=a[i];a[i]=t;
                    k++;i++;j--;break;
                }
            }
        }
        i--;j++;
        t=a[low];a[low]=a[i];a[i]=t;
        t=a[high];a[high]=a[j];a[j]=t;
        return ((long)i & 0xFFFFFFFFl) | (((long)j << 32) & 0xFFFFFFFF00000000l);
    }
    @Passed
    public static long dualPivotPartition(int[] a, int low ,int high)
    {
        int t;
        if(a[low]>a[high]) {t=a[low];a[low]=a[high];a[high]=t;}
        int p1=a[low],p2=a[high];
        int i=low+1,k=low+1,j=high-1;
        while(k<=j)
        {
            while(k<=j)
            {
                if(a[k]<p1) {t=a[i];a[i]=a[k];a[k]=t;i++;}
                else if(a[k]>=p2) break;
                k++;
            }
            while(k<=j)
            {
                if(a[j]>p2) j--;
                else if(a[j]>=p1&&a[j]<=p2) 
                {
                    t=a[j];a[j]=a[k];a[k]=t;
                    k++;j--;break;
                }
                else
                {
                    t=a[j];a[j]=a[k];a[k]=a[i];a[i]=t;
                    k++;i++;j--;break;
                }
            }
        }
        i--;j++;
        t=a[low];a[low]=a[i];a[i]=t;
        t=a[high];a[high]=a[j];a[j]=t;
        return ((long)i & 0xFFFFFFFFl) | (((long)j << 32) & 0xFFFFFFFF00000000l);
    }
    @Passed
    public static long dualPivotPartition(float[] a, int low ,int high)
    {
        float t;
        if(a[low]>a[high]) {t=a[low];a[low]=a[high];a[high]=t;}
        float p1=a[low],p2=a[high];
        int i=low+1,k=low+1,j=high-1;
        while(k<=j)
        {
            while(k<=j)
            {
                if(a[k]<p1) {t=a[i];a[i]=a[k];a[k]=t;i++;}
                else if(a[k]>=p2) break;
                k++;
            }
            while(k<=j)
            {
                if(a[j]>p2) j--;
                else if(a[j]>=p1&&a[j]<=p2) 
                {
                    t=a[j];a[j]=a[k];a[k]=t;
                    k++;j--;break;
                }
                else
                {
                    t=a[j];a[j]=a[k];a[k]=a[i];a[i]=t;
                    k++;i++;j--;break;
                }
            }
        }
        i--;j++;
        t=a[low];a[low]=a[i];a[i]=t;
        t=a[high];a[high]=a[j];a[j]=t;
        return ((long)i & 0xFFFFFFFFl) | (((long)j << 32) & 0xFFFFFFFF00000000l);
    }
    @Passed
    public static long dualPivotPartition(double[] a, int low ,int high)
    {
        double t;
        if(a[low]>a[high]) {t=a[low];a[low]=a[high];a[high]=t;}
        double p1=a[low],p2=a[high];
        int i=low+1, k=low+1, j=high-1;
        while(k<=j)
        {
            while(k<=j)
            {
                if(a[k]<p1) {t=a[i];a[i]=a[k];a[k]=t;i++;}
                else if(a[k]>=p2) break;
                k++;
            }
            while(k<=j)
            {
                if(a[j]>p2) j--;
                else if(a[j]>=p1&&a[j]<=p2) 
                {
                    t=a[j];a[j]=a[k];a[k]=t;
                    k++;j--;break;
                }
                else
                {
                    t=a[j];a[j]=a[k];a[k]=a[i];a[i]=t;
                    k++;i++;j--;break;
                }
            }
        }
        i--;j++;
        t=a[low];a[low]=a[i];a[i]=t;
        t=a[high];a[high]=a[j];a[j]=t;
        return ((long)i & 0xFFFFFFFFl) | (((long)j << 32) & 0xFFFFFFFF00000000l);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="MergeSort:merge">
    @Passed
    public static void merge(Comparable[] a, int low, int mid, int high) 
    {
        if(high-low+1<=INSERT_SORT_THRESHOLD) {Sort.innerInsertSort(a, low, high);return;}
        Comparable[] b=new Comparable[high-mid];//from mid+1 to high
        System.arraycopy(a, mid+1, b, 0, b.length);
        int i=mid, j=b.length-1, end=high;
        while(i>=low&&j>=0)
            a[end--]=(a[i].compareTo(b[j])>0? a[i--]: b[j--]);
        if(j>=0) System.arraycopy(b, 0, a, end-j, j+1);
    }
    @Passed
    public static void merge(Object[] a, Comparator cmp, int low, int mid, int high) 
    {
        if(high-low+1<=INSERT_SORT_THRESHOLD) {Sort.innerInsertSort(a, cmp, low, high);return;}
        Object[] b=new Object[high-mid];//from mid+1 to high
        System.arraycopy(a, mid+1, b, 0, b.length);
        int i=mid, j=b.length-1, end=high;
        while(i>=low&&j>=0)
            a[end--]=(cmp.compare(a[i], b[j])>0? a[i--]: b[j--]);
        if(j>=0) System.arraycopy(b, 0, a, end-j, j+1);
    }
    @Passed
    public static void merge(int[] a, int low, int mid, int high) 
    {
        if(high-low+1<=INSERT_SORT_THRESHOLD) {Sort.innerInsertSort(a, low, high);return;}
        int[] b=new int[high-mid];//from mid+1 to high
        System.arraycopy(a, mid+1, b, 0, b.length);
        int i=mid, j=b.length-1, end=high;
        while(i>=low&&j>=0)
            a[end--]=(a[i]>b[j]?  a[i--]:b[j--]);
        if(j>=0) System.arraycopy(b, 0, a, end-j, j+1);
    }
    @Passed
    public static void merge(float[] a, int low, int mid, int high) 
    {
        if(high-low+1<=INSERT_SORT_THRESHOLD) {Sort.innerInsertSort(a, low, high);return;}
        float[] b=new float[high-mid];//from mid+1 to high
        System.arraycopy(a, mid+1, b, 0, b.length);
        int i=mid, j=b.length-1, end=high;
        while(i>=low&&j>=0)
            a[end--]=(a[i]>b[j]?  a[i--]:b[j--]);
        if(j>=0) System.arraycopy(b, 0, a, end-j, j+1);
    }
    @Passed
    public static void merge(double[] a, int low, int mid, int high) 
    {
        if(high-low+1<=INSERT_SORT_THRESHOLD) {Sort.innerInsertSort(a, low, high);return;}
        double[] b=new double[high-mid];//from mid+1 to high
        System.arraycopy(a, mid+1, b, 0, b.length);
        int i=mid, j=b.length-1, end=high;
        while(i>=low&&j>=0)
            a[end--]=(a[i]>b[j]?  a[i--]:b[j--]);
        if(j>=0) System.arraycopy(b, 0, a, end-j, j+1);
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Vector-Check-Fucntion">
    public static int toneOfSequence(int[] a) {return toneOfSequence(a, 0 ,a.length-1);}
    public static int toneOfSequence(double[] a) {return toneOfSequence(a, 0 ,a.length-1);}
    
    public static boolean isDesending(int[] a) {return isDesending(a, 0, a.length-1);}
    public static boolean isDesending(double[] a) {return isDesending(a, 0, a.length-1);}
    public static boolean isDesending(Comparable[] a) {return isDesending(a, 0, a.length-1);}
    
    public static boolean isStrictDesending(int[] a) {return isStrictDesending(a, 0, a.length-1);}
    public static boolean isStrictDesending(double[] a) {return isStrictDesending(a, 0, a.length-1);}
    public static boolean isStrictDesending(Comparable[] a) {return isStrictDesending(a, 0, a.length-1);}
    
    public static boolean isAscendingOrder(int[] a) {return isAscendingOrder(a, 0, a.length-1);}
    public static boolean isAscendingOrder(float[] a) {return isAscendingOrder(a, 0, a.length-1);}
    public static boolean isAscendingOrder(double[] a) {return isAscendingOrder(a, 0, a.length-1);}
    public static boolean isAscendingOrder(Comparable[] a) {return isAscendingOrder(a, 0, a.length-1);}
    
    public static boolean isStrictAscendingOrder(int[] a) {return isStrictAscendingOrder(a, 0, a.length-1);}
    public static boolean isStrictAscendingOrder(double[] a) {return isStrictAscendingOrder(a, 0, a.length-1);}
    public static boolean isStrictAscendingOrder(Comparable[] a) {return isStrictAscendingOrder(a, 0, a.length-1);}
    
    public static final int NOT_SURE=0;
    public static final int INCREASE=1;
    public static final int STRICT_INCREASE=2;
    public static final int DECREASE=-1;
    public static final int STRICT_DECREASE=-2;
    
    public static void requireNonNull(Object[] arr, String name) {
        requireNonNull(arr, name, 0, arr.length - 1);
    }
    public static void requireNonNull(Object[] arr, String name, int low, int high) {
        if(arr == null) throw new NullPointerException(name + "is null");
        for(int i=low; i<= high; i++)
            if(arr[i] == null)
                throw new NullPointerException(name + "[" + i + "is null");
    }
    
    public static void requireNonNull(long[] arr, String name) {
        requireNonNull(arr, name, 0, arr.length - 1);
    }
    public static void requireNonNull(long[] arr, String name, int low, int high) {
        if(arr == null) throw new NullPointerException(name + "is null");
        for(int i=low; i<= high; i++)
            if(arr[i] == 0L)
                throw new NullPointerException(name + "[" + i + "is null");
    }
    
    public static int toneOfSequence(int[] a, int low, int high)
    {
        if(high<=low) return NOT_SURE;
        if(a[low+1]>a[low]) 
        {
            int base=STRICT_INCREASE;
            for(int i=low+1;i<high;i++) 
                if(a[i+1]<a[i]) {base=NOT_SURE;break;}
                else if(a[i+1]==a[i]) base=INCREASE;
            return base;
        }
        else
        {
            int base=STRICT_DECREASE;
            for(int i=low+1;i<high;i++)
                if(a[i+1]>a[i]) {base=NOT_SURE;break;}
                else if(a[i+1]==a[i]) base=DECREASE;
            return base;
        }
    }
    public static int toneOfSequence(double[] a, int low, int high)
    {
        if(high<=low) return NOT_SURE;
        if(a[low+1]>a[low]) 
        {
            int base=STRICT_INCREASE;
            for(int i=low+1;i<high;i++) 
                if(a[i+1]<a[i]) {base=NOT_SURE;break;}
                else if(a[i+1]==a[i]) base=INCREASE;
            return base;
        }
        else
        {
            int base=STRICT_DECREASE;
            for(int i=low+1;i<high;i++)
                if(a[i+1]>a[i]) {base=NOT_SURE;break;}
                else if(a[i+1]==a[i]) base=DECREASE;
            return base;
        }
    }
    
    public static boolean isDesending(int[] a, int low, int high)
    {
        while(low<high)
            if(a[low]<a[++low]) return false;
        return true;
    }
    public static boolean isDesending(double[] a, int low, int high)
    {
        while(low<high)
            if(a[low]<a[++low]) return false;
        return true;
    }
     public static boolean isDesending(Comparable[] a, int low, int high)
    {
        while(low<high)
            if(a[low].compareTo(a[++low])<0) return false;
        return true;
    }
     
    public static boolean isStrictDesending(int[] a, int low, int high)
    {
        while(low<high)
            if(a[low]<=a[++low]) return false;
        return true;
    }
    public static boolean isStrictDesending(double[] a, int low, int high)
    {
        while(low<high)
            if(a[low]<=a[++low]) return false;
        return true;
    }
     public static boolean isStrictDesending(Comparable[] a, int low, int high)
    {
        while(low<high)
            if(a[low].compareTo(a[++low])<=0) return false;
        return true;
    }
    
    public static boolean isAscendingOrder(int[] a, int low, int high)
    {
        while(low<high)
            if(a[low]>a[++low]) return false;
        return true;
    }
     public static boolean isAscendingOrder(float[] a, int low, int high)
    {
        while(low<high)
            if(a[low]>a[++low]) return false;
        return true;
    }
    public static boolean isAscendingOrder(double[] a, int low, int high)
    {
        while(low<high)
            if(a[low]>a[++low]) return false;
        return true;
    }
    public static boolean isAscendingOrder(Comparable[] a, int low, int high)
    {
        while(low<high)
            if(a[low].compareTo(a[++low])>0) return false;
        return true;
    }
    
    public static boolean isStrictAscendingOrder(int[] a, int low, int high)
    {
        while(low<high)
            if(a[low]>=a[++low]) return false;
        return true;
    }
    public static boolean isStrictAscendingOrder(double[] a, int low, int high)
    {
        while(low<high)
            if(a[low]>=a[++low]) return false;
        return true;
    }
    public static boolean isStrictAscendingOrder(Comparable[] a, int low, int high)
    {
        while(low<high)
            if(a[low].compareTo(a[++low])>=0) return false;
        return true;
    }
    //</editor-fold>
    
    public static int[] append(int[] arr, int v)  {
        int[] arr2 = new int[arr.length + 1];
        System.arraycopy(arr, 0, arr2, 0, arr.length);
        arr2[arr.length] = v;
        return arr2;
    }
    
    public static int[] append(int v, int[] arr) {
        int[] arr2 = new int[arr.length + 1];
        arr2[0] = v;
        System.arraycopy(arr, 0, arr2, 1, arr.length);
        return arr2;
    }
    
    public static float[] nan(int length) {
        float[] v = new float[length];
        for(int i=0; i<length; i++) v[i] = Float.NaN;
        return v;
    }
    
    public static float[] constant(float value, int length) {
        float[] v = new float[length];
        for(int i=0; i<length; i++) v[i] = value;
        return v;
    }
    
    public static float[] region(float[] X, int start, int end) {
        float[] Y = new float[end - start + 1];
        for(int i=start; i<=end; i++) Y[i] = X[i];
        return Y;
    }
    
    public static boolean PRINT_DIFFERENT = false;
    
    //<editor-fold defaultstate="collapsed" desc="sampePercentAbsolute(byte)">
    public static float samePercentAbsolute(byte[] A, byte[] B, int length) { return samePercentAbsolute(A, B, length, 0); }
    public static float samePercentAbsolute(byte[] A, byte[] B) { return samePercentAbsolute(A, B, (A.length<B.length? A.length:B.length), 0); }
    public static float samePercentAbsolute(byte[] A, byte[] B, int length, int threshold)
    {
        if(length >= A.length) length = A.length;
        if(length >= B.length) length = B.length;
        
        int sum=0;
        for(int i=0; i<length; i++)
        {
            if(A[i] == B[i]) { sum++; continue; }
            float div = Math.abs(A[i] - B[i]);
            if(div < threshold) sum++;
            else if(PRINT_DIFFERENT) System.out.println("different: " + i + ":" + div+"  "+A[i]+"  "+B[i]);
        }
        return ((float)sum)/length;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="sampePercentAbsolute(int)">
    public static float samePercentAbsolute(int[] A, int[] B, int length) { return samePercentAbsolute(A, B, length, 0); }
    public static float samePercentAbsolute(int[] A, int[] B) { return samePercentAbsolute(A, B, (A.length<B.length? A.length:B.length), 0); }
    public static float samePercentAbsolute(int[] A, int[] B, int length, int threshold)
    {
        if(length >= A.length) length = A.length;
        if(length >= B.length) length = B.length;
        
        int sum=0;
        for(int i=0;i<length;i++)
        {
            if(A[i] == B[i]) { sum++; continue; }
            float div = Math.abs(A[i] - B[i]);
            if(div < threshold) sum++;
            else if(PRINT_DIFFERENT) System.out.println("different: " + i + ":" + div+"  "+A[i]+"  "+B[i]);
        }
        return ((float)sum)/length;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="sampePercentAbsolute(float)">
    public static float samePercentAbsolute(float[] A, float[] B, int length) {return samePercentAbsolute(A, B, length, 1e-3f);}
    public static float samePercentAbsolute(float[] A, float[] B, float threshold) { return samePercentAbsolute(A, B, (A.length<B.length? A.length:B.length), threshold); }
    public static float samePercentAbsolute(float[] A, float[] B) { return samePercentAbsolute(A, B, (A.length<B.length? A.length:B.length), 1e-3f); }
    public static float samePercentAbsolute(float[] A, float[] B, int length, float threshold)
    {
        if(length >= A.length) length = A.length;
        if(length >= B.length) length = B.length;
        
        int sum=0;
        for(int i=0;i<length;i++)
        {
            if(A[i] == B[i]) { sum++; continue; }
            float div = Math.abs(A[i] - B[i]);
            if(div < threshold) { sum++; continue; }
            if(Float.isNaN(A[i]) && Float.isNaN(B[i])) { sum++; continue; }
            if(PRINT_DIFFERENT) System.out.println("different: " + i + ":" + div+"  "+A[i]+"  "+B[i]);
        }
        return ((float)sum)/length;
    }
    //</editor-fold>
  
    //<editor-fold defaultstate="collapsed" desc="sampePercentRelative(float)">
    public static float samePercentRelative(float[] A, float[] B, int length){return samePercentRelative(A, B, length, 1e-3f);}
    public static float samePercentRelative(float[] A, float[] B, float threshold) {return samePercentRelative(A, B, (A.length<B.length? A.length:B.length), threshold);}
    public static float samePercentRelative(float[] A, float[] B) { return samePercentRelative(A, B, (A.length<B.length? A.length:B.length), 1e-3f); }
    public static float samePercentRelative(float[] A, float[] B, int length, float threshold)
    {
        if(length<A.length) length=A.length;
        if(length<B.length) length=B.length;
        
        int sum=0;
        for(int i=0;i<length;i++)
        {
            if(A[i] == B[i]) { sum++; continue; }
            
            float div = Math.abs((A[i] - B[i] + Float.MIN_VALUE) / (A[i] + B[i] + Float.MIN_VALUE));
            if(div < threshold) { sum++; continue; }
            if(Float.isNaN(A[i]) && Float.isNaN(B[i])) { sum++; continue; }
            if(PRINT_DIFFERENT) System.out.println("different: " + i + ":" + div+", "+A[i]+", "+B[i]);
        }
        return ((float)sum)/length;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="sampePercentRelative(double, float)">
    public static float samePercentRelative(double[] A, float[] B, int length){return samePercentRelative(A, B, length, 1e-3f);}
    public static float samePercentRelative(double[] A, float[] B, float threshold) {return samePercentRelative(A, B, (A.length<B.length? A.length:B.length), threshold);}
    public static float samePercentRelative(double[] A, float[] B) {return samePercentRelative(A, B, (A.length<B.length? A.length:B.length), 1e-3f);}
    public static float samePercentRelative(double[] A, float[] B, int length, float threshold)
    {
        if(length<A.length) length=A.length;
        if(length<B.length) length=B.length;
        
        int sum=0;
        for(int i=0;i<length;i++)
        {
            if(A[i] == B[i]) {sum++; continue;}
            double div = Math.abs((A[i] - B[i] + Float.MIN_VALUE) / (A[i] + B[i] + Float.MIN_VALUE));
            if(div < threshold) sum++;
            else if(PRINT_DIFFERENT) System.out.println("different: " + i + ":" + div+", "+A[i]+", "+B[i]);
        }
        return ((float)sum)/length;
    }
    //</editor-fold>
 
    public static float zeroPercent(float[] A) {return zeroPercent(A, 0, A.length - 1);}
    public static float zeroPercent(float[] A, int low, int high)
    {
        int sum=0;
        for(int i=low;i<=high;i++) if(A[i]==0) sum++;
        return ((float)sum) / (high - low + 1);
    }
    
    public static float zeroPercent(byte[] A) {return zeroPercent(A, 0, A.length - 1);}
    public static float zeroPercent(byte[] A, int low, int high)
    {
        int sum=0;
        for(int i=low;i<=high;i++) if(A[i]==0) sum++;
        return ((float)sum) / (high - low + 1);
    }
    
    
    public static float matchPercent(float[] A, int low, int high, Predicate<Float> checker)
    {
        int sum = 0;
        for(int i=low; i<=high; i++)
            if(checker.test(A[i])) sum++;
        return ((float)sum)/(high - low + 1);
    }
    public static float matchPercent(float[] A, Predicate<Float> checker){return matchPercent(A, 0, A.length - 1, checker);}
    
    public static int differentNum(float[] A, int low, int high) {
        HashSet<Float> set = new HashSet<>();
        for(int i=low; i<=high; i++) set.add(A[i]);
        return set.size();
    }
    public static int differentNum(float[] A) {return differentNum(A, 0, A.length - 1);}
    
    
    public static float[] toVector(float[][] mat) {
        int dim0 = mat.length, dim1 = mat[0].length;
        float[] v = new float[dim0 * dim1];
        int index = 0;
        for(int i=0; i<dim0; i++)
            for(int j=0; j<dim1; j++)
                v[index++] = mat[i][j];
        return v;
    }
    
    
    public static float[] toVector(float[][][] tense) {
        int dim0 = tense.length;
        int dim1 = tense[0].length;
        int dim2 = tense[0][0].length;
        float[] v = new float[dim0 * dim1 * dim2];
        int index = 0;
        for(int i=0; i<dim0; i++)
            for(int j=0; j<dim1; j++)
                for(int k=0; k<dim2; k++)
                    v[index++] = tense[i][j][k];
        return v;
    }
    
    public static double[] toVector_double(double[][][] tense) {
        int dim0 = tense.length;
        int dim1 = tense[0].length;
        int dim2 = tense[0][0].length;
        double[] v = new double[dim0 * dim1 * dim2];
        int index = 0;
        for(int i=0; i<dim0; i++)
            for(int j=0; j<dim1; j++)
                for(int k=0; k<dim2; k++)
                    v[index++] = tense[i][j][k];
        return v;
    }
   
    public static float[] toVector(float[][][][] tense) {
        int dim0 = tense.length;
        int dim1 = tense[0].length;
        int dim2 = tense[0][0].length;
        int dim3 = tense[0][0][0].length;
        
        float[] v = new float[dim0 * dim1 * dim2 * dim3];
        int index = 0;
        for(int d0=0; d0<dim0; d0++)
            for(int d1=0; d1<dim1; d1++)
                for(int d2=0; d2<dim2; d2++)
                    for(int d3=0; d3<dim3; d3++)
                    v[index++] = tense[d0][d1][d2][d3];
        return v;
    }
    
     public static double[] toVector_double(double[][][][] tense) {
        int dim0 = tense.length;
        int dim1 = tense[0].length;
        int dim2 = tense[0][0].length;
        int dim3 = tense[0][0][0].length;
        
        double[] v = new double[dim0 * dim1 * dim2 * dim3];
        int index = 0;
        for(int d0=0; d0<dim0; d0++)
            for(int d1=0; d1<dim1; d1++)
                for(int d2=0; d2<dim2; d2++)
                    for(int d3=0; d3<dim3; d3++)
                    v[index++] = tense[d0][d1][d2][d3];
        return v;
    }
           
   
    //<editor-fold defaultstate="collapsed" desc="toND(byte)">  
    public static byte[][] to2D(byte[] X, int dim0, int dim1) {
        byte[][] Y = new byte[dim0][dim1];
        int index = 0;
        for(int i=0; i<dim0; i++)
            for(int j=0; j<dim1; j++)
                Y[i][j] = X[index++];//X[index++]
        return Y;
    }
    
    public static byte[][][] to3D(byte[] X, int dim0, int dim1, int dim2) {
        byte[][][] Y = new byte[dim0][dim1][dim2];
        int index = 0;
        for(int i=0; i<dim0; i++)
            for(int j=0; j<dim1; j++)
                for(int k=0; k<dim2; k++)
                    Y[i][j][k] = X[index++];
        return Y;
    }
    
    public static byte[][][][] to4D(byte[] X, int dim0, int dim1, int dim2, int dim3) {
        byte[][][][] Y = new byte[dim0][dim1][dim2][dim3];
        int index = 0;
        for(int d0=0; d0<dim0; d0++)
            for(int d1=0; d1<dim1; d1++)
                for(int d2=0; d2<dim2; d2++)
                    for(int d3=0; d3<dim3; d3++)
                        Y[d0][d1][d2][d3] = X[index++];
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="toND(int)">  
    public static int[][] to2D(int[] X, int dim0, int dim1) {
        int[][] Y = new int[dim0][dim1];
        int index = 0;
        for(int i=0; i<dim0; i++)
            for(int j=0; j<dim1; j++)
                    Y[i][j] = X[index++];//X[index++]
        return Y;
    }
    
    public static int[][][] to3D(int[] X, int dim0, int dim1, int dim2) {
        int[][][] Y = new int[dim0][dim1][dim2];
        int index = 0;
        for(int i=0; i<dim0; i++)
            for(int j=0; j<dim1; j++)
                for(int k=0; k<dim2; k++)
                    Y[i][j][k] = X[index++];
        return Y;
    }
    
    public static int[][][][] to4D(int[] X, int dim0, int dim1, int dim2, int dim3) {
        int[][][][] Y = new int[dim0][dim1][dim2][dim3];
        int index = 0;
        for(int d0=0; d0<dim0; d0++)
            for(int d1=0; d1<dim1; d1++)
                for(int d2=0; d2<dim2; d2++)
                    for(int d3=0; d3<dim3; d3++)
                        Y[d0][d1][d2][d3] = X[index++];
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="toND(float)">  
    public static float[][] to2D(float[] X, int dim0, int dim1) {
        float[][] Y = new float[dim0][dim1];
        int index = 0;
        for(int i=0; i<dim0; i++)
            for(int j=0; j<dim1; j++)
                    Y[i][j] = X[index++];//X[index++]
        return Y;
    }
    
    public static float[][][] to3D(float[] X, int dim0, int dim1, int dim2) {
        float[][][] Y = new float[dim0][dim1][dim2];
        int index = 0;
        for(int i=0; i<dim0; i++)
            for(int j=0; j<dim1; j++)
                for(int k=0; k<dim2; k++)
                    Y[i][j][k] = X[index++];
        return Y;
    }
    
    public static float[][][][] to4D(float[] X, int dim0, int dim1, int dim2, int dim3) {
        float[][][][] Y = new float[dim0][dim1][dim2][dim3];
        int index = 0;
        for(int d0=0; d0<dim0; d0++)
            for(int d1=0; d1<dim1; d1++)
                for(int d2=0; d2<dim2; d2++)
                    for(int d3=0; d3<dim3; d3++)
                        Y[d0][d1][d2][d3] = X[index++];
        return Y;
    }
    //</editor-fold>
    
    public static double[][][][] toTense4D_double(float[] X, int dim0, int dim1, int dim2, int dim3)
    {
        double[][][][] tense = new double[dim0][dim1][dim2][dim3];
        int index = 0;
        for(int d0=0; d0<dim0; d0++)
            for(int d1=0; d1<dim1; d1++)
                for(int d2=0; d2<dim2; d2++)
                    for(int d3=0; d3<dim3; d3++)
                        tense[d0][d1][d2][d3] = X[index++];
        return tense;
    }
    
    
    public static float[][][] toTense3D(float[][][][] tense, int dim0, int dim1, int dim2) {
        float[] v = Vector.toVector(tense);
        return Vector.to3D(v, dim0, dim1, dim2);
    }
    
    //<editor-fold defaultstate="collapsed" desc="Tensor transpose">
    public static float[][][][] transpose4D(float[][][][] X, int dimIdx1, int dimIdx2) {
        if(dimIdx1 > dimIdx2) {
            int t = dimIdx1; dimIdx1 = dimIdx2; dimIdx2 = t;
        }
        
        int dim0 = X.length;
        int dim1 = X[0].length;
        int dim2 = X[0][0].length;
        int dim3 = X[0][0][0].length;
        
        if(dimIdx1 == 0 && dimIdx2 == 1) return transepose4D_0_1(X, dim0, dim1, dim2, dim3);
        if(dimIdx1 == 0 && dimIdx2 == 2) return transepose4D_0_2(X, dim0, dim1, dim2, dim3);
        if(dimIdx1 == 0 && dimIdx2 == 3) return transepose4D_0_3(X, dim0, dim1, dim2, dim3);
        if(dimIdx1 == 1 && dimIdx2 == 2) return transepose4D_1_2(X, dim0, dim1, dim2, dim3);
        if(dimIdx1 == 1 && dimIdx2 == 3) return transepose4D_1_3(X, dim0, dim1, dim2, dim3);
        if(dimIdx1 == 2 && dimIdx2 == 3) return transepose4D_2_3(X, dim0, dim1, dim2, dim3);
        else throw new IllegalArgumentException();
    }
    
    //<editor-fold defaultstate="collapsed" desc="transpose 4D">
    static float[][][][] transepose4D_0_1(float[][][][] X, int dim0, int dim1, int dim2, int dim3)
    {
        float[][][][] Y = new float[dim1][dim0][dim2][dim3];
        for(int d0=0; d0<dim0; d0++)
            for(int d1=0; d1<dim1; d1++)
                for(int d2=0; d2<dim2; d2++)
                    System.arraycopy(X[d0][d1][d2], 0, Y[d1][d0][d2], 0, dim3);
        return Y;
    }
    static float[][][][] transepose4D_0_2(float[][][][] X, int dim0, int dim1, int dim2, int dim3)
    {
        float[][][][] Y = new float[dim2][dim1][dim0][dim3];
        for(int d0=0; d0<dim0; d0++)
            for(int d1=0; d1<dim1; d1++)
                for(int d2=0; d2<dim2; d2++)
                    System.arraycopy(X[d0][d1][d2], 0, Y[d2][d1][d0], 0, dim3);
        return Y;
    }
    static float[][][][] transepose4D_0_3(float[][][][] X, int dim0, int dim1, int dim2, int dim3)
    {
        float[][][][] Y = new float[dim3][dim1][dim2][dim0];
        for(int d0=0; d0<dim0; d0++)
            for(int d1=0; d1<dim1; d1++)
                for(int d2=0; d2<dim2; d2++)
                    for(int d3=0; d3<dim3; d3++)
                        Y[d3][d1][d2][d0] = X[d0][d1][d2][d3];
        return Y;
    }
    
    static float[][][][] transepose4D_1_2(float[][][][] X, int dim0, int dim1, int dim2, int dim3)
    {
        float[][][][] Y = new float[dim0][dim2][dim1][dim3];
        for(int d0=0; d0<dim0; d0++)
            for(int d1=0; d1<dim1; d1++)
                for(int d2=0; d2<dim2; d2++)
                    System.arraycopy(X[d0][d1][d2], 0, Y[d0][d2][d1], 0, dim3);
        return Y;
    }
    
    static float[][][][] transepose4D_1_3(float[][][][] X, int dim0, int dim1, int dim2, int dim3)
    {
        float[][][][] Y = new float[dim0][dim3][dim2][dim1];
        for(int d0=0; d0<dim0; d0++)
            for(int d1=0; d1<dim1; d1++)
                for(int d2=0; d2<dim2; d2++)
                    for(int d3=0; d3<dim3; d3++)
                        Y[d0][d3][d2][d1] = X[d0][d1][d2][d3];
        return Y;
    }
    
    static float[][][][] transepose4D_2_3(float[][][][] X, int dim0, int dim1, int dim2, int dim3)
    {
        float[][][][] Y = new float[dim0][dim1][dim3][dim2];
        for(int d0=0; d0<dim0; d0++)
            for(int d1=0; d1<dim1; d1++)
                for(int d2=0; d2<dim2; d2++)
                    for(int d3=0; d3<dim3; d3++)
                        Y[d0][d1][d3][d2] = X[d0][d1][d2][d3];
        return Y;
    }
    //</editor-fold>
    
    public static float[][][] transpose3D(float[][][] X, int dimIdx1, int dimIdx2) {
        if(dimIdx1 > dimIdx2) {
            int t = dimIdx1; dimIdx1 = dimIdx2; dimIdx2 = t;
        }
        
        int dim0 = X.length;
        int dim1 = X[0].length;
        int dim2 = X[0][0].length;
        
        if(dimIdx1 == 0 && dimIdx2 == 1) return transepose3D_0_1(X, dim0, dim1, dim2);
        if(dimIdx1 == 0 && dimIdx2 == 2) return transepose3D_0_2(X, dim0, dim1, dim2);
        if(dimIdx1 == 1 && dimIdx2 == 2) return transepose3D_1_2(X, dim0, dim1, dim2);
        else throw new IllegalArgumentException();
    }
    //<editor-fold defaultstate="collapsed" desc="transpose 3D">
    static float[][][] transepose3D_0_1(float[][][] X, int dim0, int dim1, int dim2)
    {
        float[][][] Y = new float[dim1][dim0][dim2];
        for(int d0=0; d0<dim0; d0++)
            for(int d1=0; d1<dim1; d1++)
                for(int d2=0; d2<dim2; d2++)
                        Y[d1][d0][d2] = X[d0][d1][d2];
        return Y;
    }
    
    
    static float[][][] transepose3D_0_2(float[][][] X, int dim0, int dim1, int dim2)
    {
        float[][][] Y = new float[dim2][dim1][dim0];
        for(int d0=0; d0<dim0; d0++)
            for(int d1=0; d1<dim1; d1++)
                for(int d2=0; d2<dim2; d2++)
                        Y[d2][d1][d0] = X[d0][d1][d2];
        return Y;
    }
    
    static float[][][] transepose3D_1_2(float[][][] X, int dim0, int dim1, int dim2)
    {
        float[][][] Y = new float[dim0][dim2][dim1];
        for(int d0=0; d0<dim0; d0++)
            for(int d1=0; d1<dim1; d1++)
                for(int d2=0; d2<dim2; d2++)
                        Y[d0][d2][d1] = X[d0][d1][d2];
        return Y;
    }
    //</editor-fold>
    
    
    public static float[][] transpose2D(float[][] X)
    {
        int dim0 = X.length;
        int dim1 = X[0].length;
        
        float[][] Y = new float[dim1][dim0];
        for(int d0=0; d0<dim0; d0++)
            for(int d1=0; d1<dim1; d1++)
                        Y[d1][d0] = X[d0][d1];
        return Y;
    }
    //</editor-fold>
    
    public static float[][][][] concat4D(int dimIdx, float[][][][]...X) 
    {
        if(dimIdx < 0) dimIdx += 4;
        
        if(dimIdx == 0) return concat4D_0(X);
        if(dimIdx == 1) return concat4D_1(X);
        if(dimIdx == 2) return concat4D_2(X);
        if(dimIdx == 3) return concat4D_3(X);
        throw new IllegalArgumentException();
    }
    
    //<editor-fold defaultstate="collapsed" desc="concat 4D">
    public static float[][][][] concat4D_0(float[][][][]... X) {
        int dimSize = 0;
        for(int i=0; i<X.length;i++) dimSize += X[i].length;
        
        int dim1 = X[0][0].length;
        int dim2 = X[0][0][0].length;
        int dim3 = X[0][0][0][0].length;
        
        float[][][][] Y = new float[dimSize][dim1][dim2][dim3];
        int yd0 = 0;
        for(int i=0; i<X.length; i++) {
            int dim0 = X[i].length;
            for(int d0=0; d0<dim0; d0++, yd0++)
            for(int d1=0; d1<dim1; d1++)
            for(int d2=0; d2<dim2; d2++) 
                System.arraycopy(X[i][d0][d1][d2], 0, Y[yd0][d1][d2], 0, dim3);
        }
        return Y;
    }
    
    public static float[][][][] concat4D_1(float[][][][]... X) {
        int dimSize = 0;
        for(int i=0; i<X.length; i++) dimSize += X[i][0].length;
        
        int dim0 = X[0].length;
        int dim2 = X[0][0][0].length;
        int dim3 = X[0][0][0][0].length;
        
        float[][][][] Y = new float[dim0][dimSize][dim2][dim3];
        int yd1 = 0;
        for(int i=0; i<X.length; i++){
            int dim1 = X[i][0].length;
            for(int d1=0; d1<dim1; d1++, yd1++)
            for(int d0=0; d0<dim0; d0++)
            for(int d2=0; d2<dim2; d2++)
                System.arraycopy(X[i][d0][d1][d2], 0, Y[d0][yd1][d2], 0, dim3);
        }
        return Y;
    }
    
    public static float[][][][] concat4D_2(float[][][][]... X) {
        int dimSize = 0;
        for(int i=0; i<X.length; i++) dimSize += X[i][0][0].length;
        
        int dim0 = X[0].length;
        int dim1 = X[0][0].length;
        int dim3 = X[0][0][0][0].length;
        
        float[][][][] Y = new float[dim0][dim1][dimSize][dim3];
        int yd2 = 0;
        for(int i=0; i<X.length; i++) {
            int dim2 = X[i][0][0].length;
            for(int d2=0; d2<dim2; d2++, yd2++)
            for(int d0=0; d0<dim0; d0++)
            for(int d1=0; d1<dim1; d1++)
                System.arraycopy(X[i][d0][d1][d2], 0, Y[d0][d1][yd2], 0, dim3);
        }
        return Y;
    }
    
    public static float[][][][] concat4D_3(float[][][][]...X){
        int dimSize = 0;
        for(int i=0; i<X.length; i++) dimSize += X[i][0][0][0].length; 
        
        int dim0 = X[0].length;
        int dim1 = X[0][0].length;
        int dim2 = X[0][0][0].length;
        
        float[][][][] Y = new float[dim0][dim1][dim2][dimSize];
        int yd3 = 0;
        for(int i=0; i<X.length; i++) {
            int dim3 = X[i][0][0][0].length;
            for(int d3=0; d3<dim3; d3++, yd3++)
            for(int d0=0; d0<dim0; d0++)
            for(int d1=0; d1<dim1; d1++)
            for(int d2=0; d2<dim2; d2++)
                Y[d0][d1][d2][yd3] = X[i][d0][d1][d2][d3];
        }
        return Y;
    }
    //</editor-fold>
    
    
    public static float[][][][][] split4D(float[][][][] X, int dimIdx, int[] section) {
        if(dimIdx < 0) dimIdx += 4;
        if(dimIdx == 0) return split4D_0(X, section);
        if(dimIdx == 1) return split4D_1(X, section);
        if(dimIdx == 2) return split4D_2(X, section);
        if(dimIdx == 3) return split4D_3(X, section);
        throw new IllegalArgumentException();
    }
    
    //<editor-fold defaultstate="collapsed" desc="split 4d">
    public static float[][][][][] split4D_0(float[][][][] X, int[] section) 
    {
        int dim1 = X[0].length;
        int dim2 = X[0][0].length;
        int dim3 = X[0][0][0].length;
        
        float[][][][][] Y = new float[section.length][][][][];
        for(int i=0; i<section.length; i++) {
            Y[i] = new float[section[i]][dim1][dim2][dim3];
        }
        
        int yd0 = 0;
        for(int i=0; i<section.length; i++)
        {
            int dim0 = section[i];
            for(int d0=0; d0<dim0; d0++, yd0++)
            for(int d1=0; d1<dim1; d1++)
            for(int d2=0; d2<dim2; d2++)
                System.arraycopy(X[yd0][d1][d2], 0, Y[i][d0][d1][d2], 0, dim3);
                
        }
        return Y;
    }
    
    public static float[][][][][] split4D_1(float[][][][] X, int[] section) 
    {
        int dim0 = X.length;
        int dim2 = X[0][0].length;
        int dim3 = X[0][0][0].length;
        
        float[][][][][] Y = new float[section.length][][][][];
        for(int i=0; i<section.length; i++) {
            Y[i] = new float[dim0][section[i]][dim2][dim3];
        }
        
        int yd1 = 0;
        for(int i=0; i<section.length; i++)
        {
            int dim1 = section[i];
            for(int d1=0; d1<dim1; d1++, yd1++)
            for(int d0=0; d0<dim0; d0++)
            for(int d2=0; d2<dim2; d2++)
                System.arraycopy(X[d0][yd1][d2], 0, Y[i][d0][d1][d2], 0, dim3);
                
        }
        return Y;
    }
    
    public static float[][][][][] split4D_2(float[][][][] X, int[] section) 
    {
        int dim0 = X.length;
        int dim1 = X[0].length;
        int dim3 = X[0][0][0].length;
        
        float[][][][][] Y = new float[section.length][][][][];
        for(int i=0; i<section.length; i++) {
            Y[i] = new float[dim0][dim1][section[i]][dim3];
        }
        
        int yd2 = 0;
        for(int i=0; i<section.length; i++)
        {
            int dim2 = section[i];
            for(int d2=0; d2<dim2; d2++, yd2++)
            for(int d0=0; d0<dim0; d0++)
            for(int d1=0; d1<dim1; d1++)
                System.arraycopy(X[d0][d1][yd2], 0, Y[i][d0][d1][d2], 0, dim3);
        }
        return Y;
    }
     
    public static float[][][][][] split4D_3(float[][][][] X, int[] section) 
    {
        int dim0 = X.length;
        int dim1 = X[0].length;
        int dim2 = X[0][0].length;
        
        float[][][][][] Y = new float[section.length][][][][];
        for(int i=0; i<section.length; i++) {
            Y[i] = new float[dim0][dim1][dim2][section[i]];
        }
        
        int yd3 = 0;
        for(int i=0; i<section.length; i++)
        {
            int dim3 = section[i];
            for(int d3=0; d3<dim3; d3++, yd3++)
            for(int d0=0; d0<dim0; d0++)
            for(int d1=0; d1<dim1; d1++)
            for(int d2=0; d2<dim2; d2++)
                Y[i][d0][d1][d2][d3] = X[d0][d1][d2][yd3];
        }
        return Y;
    }
    //</editor-fold>
    
    public static float[][][][] rot180(float[][][][] X)
    {
        int dim0 = X.length;
        int dim1 = X[0].length;
        int dim2 = X[0][0].length;
        int dim3 = X[0][0][0].length;
        
        float[][][][] Y = new float[dim0][dim1][dim2][dim3];
        
        for(int d0 = 0; d0 < dim0; d0++)
        for(int d1 = 0; d1 < dim1; d1++)
        for(int d2 = 0; d2 < dim2; d2++)
        for(int d3 = 0; d3 < dim3; d3++)
            Y[d0][d1][d2][d3] = X[d0][dim1 - 1 - d1][dim2 - 1 - d2][d3];
        return Y;
    }
}