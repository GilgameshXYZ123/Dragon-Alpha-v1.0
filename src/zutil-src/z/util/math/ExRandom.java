/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.math;
import java.lang.reflect.Array;
import java.util.HashSet;
import java.util.Objects;
import java.util.Random;
import z.util.lang.exception.IAE;
import static z.util.lang.Lang.LENGTH_TIMES_INT_CHAR;
import z.util.lang.annotation.Passed;
import static z.util.lang.Lang.INT_BYTE;

/**
 *
 * @author dell
 */
public class ExRandom extends Random
{
    private static final String TOO_LONG = "len>max-min:";
    private static final String NEGATIVE_LENGTH = "len<=0:";
    private static final String NEGATIVE_WIDTH = "Matrix.height<=0";
    private static final String NEGATIVE_HEIGHT = "Matrix.width<=0";
    private static final String NO_RANDOM_SPACE = "max==min:";
    private static final String NEGATIVE_BOUND = "bound must be positive";
    private static final String NEGATIVE_THRESHOLD = "threshold must be positive";
    private static final String INT_OVER_BOUND = "base+bound>Integer.MAX";
    private static final int NEAT_CHAR_BASE = '!';
    private static final int NEAT_CHAR_THRESHOLD = '~'-'!';
    
    //<editor-fold defaultstate="collapsed" desc="class: BadBoundException">
    public static class BadBoundException extends RuntimeException
    {
        public static final String MSG="BadBoundException:";
        public BadBoundException() {}
        public BadBoundException(String message) 
        {
            super(MSG+message);
        }
        public BadBoundException(Throwable cause) 
        {
            super(cause);
        }
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="ExtensiveRandom:next">
    /**
     * Generates exRandom bytes and places them into a user-supplied byte array.
     * The number of exRandom bytes produced is equal to the length of the byte
     * array.
     *
     * @param bytes the byte array to fill with exRandom bytes.
     */
    @Override
    public void nextBytes(byte[] bytes) 
    {
        int intLen = bytes.length / INT_BYTE;
        int byteLen = (intLen == 0? bytes.length : bytes.length % intLen);
        
        int index=0;
        for(int i=0; i < intLen; i++)
        for(int j=0, r = next(32); j < INT_BYTE; j++) { 
            bytes[index++] = (byte)r;
            r = r>>Byte.SIZE;
        }
            
        for(int i=0, r=next(32); i < byteLen; i++) {
            bytes[index++] = (byte)r;
            r = r>> Byte.SIZE;
        }
    }
    /**
     * Generates exRandom bytes, and the length of array is user-supplied
     * @param length the length of byte array.
     * @return 
     */
    @Passed
    public byte[] nextBytes(int length)
    {
        byte[] bytes = new byte[length];
        this.nextBytes(bytes);
        return bytes;
    }
    
    public byte[] nextBytes(int length, byte min, byte max) 
    {
        if(max == min) throw new IllegalArgumentException("min == max");
        if(max < min) { byte t = min; min = max; max = t; }
        
        byte threshold = (byte) (max - min + 1), base = min;
        byte[] arr = new byte[length]; nextBytes(arr);
        
        for(int i=0; i<arr.length; i++) {
            int v = arr[i];
            if(v < 0) v = -v;
            v = v % threshold + base;
            arr[i] = (byte) v;
        }
        return arr;
    }
    /**
     * Generates exRandom chars and places them into a user-supplied char array.
     * The number of exRandom chars produced is equal to the length of the char
     * array.
     *
     * @param chars the char array to fill with exRandom bytes.
     */
    @Passed
    public void nextChars(char[] chars)
    {
         int intLen=chars.length/LENGTH_TIMES_INT_CHAR,
            byteLen=chars.length%intLen;
        
        int index=0,r,i,j;
        for(i=0;i<intLen;i++)
        for(j=0,r=next(32);j<LENGTH_TIMES_INT_CHAR;j++)
            {chars[index++]=(char)r;r=r>>Character.SIZE;}
            
        for(i=0,r=next(32);i<byteLen;i++)
            {chars[index++]=(char)r;r=r>>Character.SIZE;}
    }
    /**
     * Generates exRandom chars, and the length of array is user-supplied
     * @param length the length of char array.
     * @return 
     */
    @Passed
    public char[] nextChars(int length)
    {
        char[] chars=new char[length];
        this.nextChars(chars);
        return chars;
    }
     /**
     * Generates exRandom String, and the length of String is user-supplied
     * @param length the length of String.
     * @return 
     */
    @Passed
    public String nextString(int length)
    {
        return new String(this.nextChars(length));
    }
    /**
     * Generates exRandom chars and places them into a user-supplied char
     * array,all character in this String is printable, from '!' to '~'. The
     * number of exRandom chars produced is equal to the length of the char
     * array.
     *
     * @param chars the char array to fill with exRandom bytes.
     */
    @Passed
    public void nextNeatChars(char[] chars)
    {
        int intLen=chars.length/LENGTH_TIMES_INT_CHAR,
            byteLen=chars.length%intLen;
        
        int index=0,r,i,j;
        for(i=0;i<intLen;i++)
        for(j=0,r=next(32);j<LENGTH_TIMES_INT_CHAR;j++)
        {
            chars[index++]=(char) ((NEAT_CHAR_BASE+(char)r%NEAT_CHAR_THRESHOLD));
            r=r>>Character.SIZE;
        }
        for(i=0,r=next(32);i<byteLen;i++)
        {
            chars[index++]=(char) ((NEAT_CHAR_BASE+(char)r%NEAT_CHAR_THRESHOLD));
            r=r>>Character.SIZE;
        }
    }
    /**
     * Generates exRandom chars, and the length of array is user-supplied, all
     * character in this String is printable, from '!' to '~'.
     *
     * @param length the length of char array.
     * @return
     */
    @Passed
    public char[] nextNeatChars(int length)
    {
        char[] chars=new char[length];
        this.nextNeatChars(chars);
        return chars;
    }
    /**
     * Generates exRandom String, and the length of String is user-supplied, all
     * character in this String is printable, from '!' to '~'.
     *
     * @param length the length of String.
     * @return
     */
    @Passed
    public String nextNeatString(int length)
    {
        return new String(this.nextNeatChars(length));
    }
    /**
     * Returns the next pseudorandom, uniformly distributed {@code short} value
     * from this exRandom number generator's sequence.
     *
     * @return
     */
    @Passed
    public short nextShort()
    {
        return (short) next(16);
    }
    /**
     * Returns a pseudorandom, uniformly distributed {@code int} value between 0
     * (inclusive) and the specified value (inclusive), drawn from this exRandom
     * number generator's sequence.
     *
     * @param max the upper bound (exclusive). Must be positive.
     * @throws BadBoundException if bound is not positive.
     * @return
     */
    @Override
    public int nextInt(int max)
    {
        if(max<=0) throw new BadBoundException(NEGATIVE_BOUND);
        int r=next(31),m=++max-1;
        if((max&m)==0) r=(int)((max*(long)r)>>31);
        else for(int u=r;u+m<(r=u%max);u=next(31));
        return r;
    }
    /**
     * Returns a pseuduorandom, uniformly distributed {@code int} value between
     * base(includsive) and base+threshold-1(inclusive), drawn from this
     * exRandom number generator's sequence.
     *
     * @param base
     * @param threshold
     * @throws BadBoundException if threshold less than or equal to 0
     * @throws BadBoundException if (threshold+base)>Integer.max
     * @return
     */
    @Passed
    public int nextIntFromBase(int base, int threshold)
    {
        if(threshold<=0) throw new BadBoundException(NEGATIVE_THRESHOLD);
        if(base+threshold>Integer.MAX_VALUE)  throw new BadBoundException(INT_OVER_BOUND);
        int r = next(31),m = threshold-1;
        if((threshold&m) ==0) r=(int)(((threshold)*(long)r)>>31);
        else for(int u=r;u+m<(r=u%threshold);u=next(31));
        return r+base;
    }
    /**
     * Returns a pseuduorandom, uniformly distributed {@code int} value between
     * min(includsive) and max(inclusive), drawn from this exRandom number
     * generator's sequence. if max less than min, it will exchange the value
     * between min and max.
     *
     * @param min the lower bound.
     * @param max the upper bound.
     * @throws BadBoundException if max==min.
     * @return
     */
    @Passed
    public int nextInt(int min, int max)
    {
        if(min == max) return min;
        if(min > max) {int t = min; min = max; max = t;}
        
        int threshold = max - min + 1,r = next(31),m = threshold-1;
        if((threshold&m) == 0) 
            r = (int)((threshold * (long)r) >> 31);
        else
            for(int u = r; u + m < (r = u % threshold); u=next(31));
        return r + min;
    }
    
    public float nextFloat(float min, float max)
    {
        if(min == max) throw new BadBoundException(NO_RANDOM_SPACE);
        if(min > max) {float t = min; min = max; max = t;}
        return (max - min) * nextFloat() + min;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Array-Checker">
    @Passed
    private static int checkRNIntArray(int len, int min, int max)
    {
        if(len<=0) throw new BadBoundException(NEGATIVE_LENGTH+len);
        int num=max-min;
        if(len>num) throw new BadBoundException(TOO_LONG+'<'+max+','+min+','+len+'>');
        return num;
    }
    @Passed
    private static void checkArrayLength(int len)
    {
        if(len<=0) throw new BadBoundException(NEGATIVE_LENGTH+len);
    }
    @Passed
    private static void checkMatrixSize(int height, int width)
    {
        if(height<=0) throw new BadBoundException(NEGATIVE_HEIGHT);
        if(width<=0) throw new BadBoundException(NEGATIVE_WIDTH);
    }
    @Passed
    private static void checkIntArray(int len, long min, long max)
    {
        if(len<=0) throw new BadBoundException(NEGATIVE_LENGTH+len);
        if(min==max) throw new BadBoundException(NO_RANDOM_SPACE);
    }
    @Passed
    private static void checkIntMatrix(int height, int width, long min, long max)
    {
        if(height<=0) throw new BadBoundException(NEGATIVE_HEIGHT);
        if(width<=0) throw new BadBoundException(NEGATIVE_WIDTH);
        if(min==max) throw new BadBoundException(NO_RANDOM_SPACE);
    }
    @Passed
    private static void checkIntArray(int[] v, int min, int max)
    {
        if(v==null) throw new NullPointerException();
        if(min==max) throw new BadBoundException(NO_RANDOM_SPACE);
    }
    @Passed
    private static void checkIntMatrix(int[][] v, int min, int max)
    {
        if(v==null) throw new NullPointerException();
        if(min==max) throw new BadBoundException(NO_RANDOM_SPACE);
    }
    @Passed
    private static void checkIntArray(long[] v, long min, long max)
    {
        if(v==null) throw new NullPointerException();
        if(min==max) throw new BadBoundException(NO_RANDOM_SPACE);
    }
    @Passed
    private static void checkIntMatrix(long[][] v, long min, long max)
    {
        if(v==null) throw new NullPointerException();
        if(min==max) throw new BadBoundException(NO_RANDOM_SPACE);
    }
    @Passed
    private static void checkRealArray(int len, double min, double max)
    {
        if(len<=0) throw new BadBoundException(NEGATIVE_LENGTH+len);
        if(min==max) throw new BadBoundException(NO_RANDOM_SPACE);
    }
    @Passed
    private static void checkRealMatrix(int height, int width, double min, double max)
    {
        if(height<=0) throw new BadBoundException(NEGATIVE_HEIGHT);
        if(width<=0) throw new BadBoundException(NEGATIVE_WIDTH);
        if(min==max) throw new BadBoundException(NO_RANDOM_SPACE);
    }
    @Passed
    private static void checkRealArray(float[] v, double min, double max)
    {
        if(v==null) throw new NullPointerException();
        if(min==max) throw new BadBoundException(NO_RANDOM_SPACE);
    }
    @Passed
    private static void checkRealMatrix(float[][] v, float min, float max)
    {
        if(v==null) throw new NullPointerException();
        if(min==max) throw new BadBoundException(NO_RANDOM_SPACE);
    }
    @Passed
    private static void checkRealArray(double[] v, double min, double max)
    {
        if(v==null) throw new NullPointerException();
        if(min==max) throw new BadBoundException(NO_RANDOM_SPACE);
    }
    @Passed
    private static void checkRealMatrix(double[][] v, double min, double max)
    {
        if(v==null) throw new NullPointerException();
        if(min==max) throw new BadBoundException(NO_RANDOM_SPACE);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Vector:exRandom">
    //<editor-fold defaultstate="collapsed" desc="Vector:NoRepetiveRandom">
    //<editor-fold defaultstate="collapsed" desc="Core-Code">
    @Passed
    private int[] nextNRIntVectorLessThanHalf(int len, int min, int max)
    {
        int threshold=max-min;
        HashSet<Integer> set=new HashSet<>(len<<1);
        for(int count=0;count<len;)
            if(set.add(this.nextInt(threshold)+min)) count++;
        int[] arr=new int[len];
        int index=0;
        for(Integer val:set) arr[index++]=val;
        return arr;
    }
    @Passed
    private int[] nextNRIntVectorGreaterThanHalf(int len, int min, int max)
    {
        int threshold=max-min;
        HashSet<Integer> set=new HashSet<>(len<<1);
        for(int count=len;count>=len;)
            if(set.remove(this.nextInt(threshold)+min)) count--;
        int[] arr=new int[len];
        int index=0;
        for(Integer val:set) arr[index++]=val;
        return arr;
    }
    //</editor-fold>
    @Passed 
    public int[] nextNRIntVector(int len)
    {
        ExRandom.checkArrayLength(len);
        HashSet<Integer> set=new HashSet<>(len*2);
        for(int i=0;i<len;i++) set.add(this.nextInt());
        int[] arr=new int[len];
        int index=0;
        for(Integer val:set) arr[index++]=val;
        return arr;
    }
    @Passed
    public int[] nextNRIntVector(int len, int max)
    {
        return this.nextNRIntVector(len, 0, max);
    }
    @Passed
    public int[] nextNRIntVector(int len, int min, int max)
    {
        if(max<min) {int t=max;max=min;min=t;}
        int half=ExRandom.checkRNIntArray(len, min, max)>>1;
        if(len>half) return this.nextNRIntVectorGreaterThanHalf(len, min, max);
        else return this.nextNRIntVectorLessThanHalf(len, min, max);
    }
    @Passed
    public double[] nextNRDoubleVector(int len)
    {
        ExRandom.checkArrayLength(len);
        HashSet<Double> set=new HashSet<>(len*2);
        for(int i=0;i<len;i++) set.add(this.nextDouble());
        double[] arr=new double[len];
        int index=0;
        for(Double val:set) arr[index++]=val;
        return arr;
    }
    @Passed
    public double[] nextNRGaussianVector(int len)
    {
        ExRandom.checkArrayLength(len);
        HashSet<Double> set=new HashSet<>(len*2);
        for(int i=0;i<len;i++) set.add(this.nextGaussian());
        double[] arr=new double[len];
        int index=0;
        for(Double val:set) arr[index++]=val;
        return arr;
    }
    @Passed
    public double[] nextNRDoubleVector(int len, double min, double max)
    {
        if(max<min) {double t=max;max=min;min=t;}
        ExRandom.checkArrayLength(len);
        double threshold=max-min;
        HashSet<Double> set=new HashSet(len<<1);
        for(int count=0;count<len;)
            if(set.add(this.nextDouble()*threshold+min)) count++;
        double[] arr=new double[len];
        int index=0;
        for(Double val:set) arr[index++]=val;
        return arr;
    }
    @Passed
    public double[] nextNRDoubleVector(int len, double max)
    {
        return this.nextNRDoubleVector(len, 0, max);
    }
    @Passed
    public double[] nextNRGaussianVector(int len, double min, double max)
    {
        if(max<min) {double t=max;max=min;min=t;}
        ExRandom.checkArrayLength(len);
        double threshold=max-min;
        HashSet<Double> set=new HashSet(len<<1);
        for(int count=0;count<len;)
            if(set.add(this.nextGaussian()*threshold+min)) count++;
        double[] arr=new double[len];
        int index=0;
        for(Double val:set) arr[index++]=val;
        return arr;
    }
    @Passed
    public double[] nextNRGaussianVector(int len, double max)
    {
        return this.nextNRDoubleVector(len, 0, max);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Vector:exRandom:float">
    @Passed
    public float[] nextFloatVector(int width)
    {
        ExRandom.checkArrayLength(width);
        float[] arr=new float[width];
        for(int i=0;i<width;i++) arr[i]=this.nextFloat();
        return arr;
    }
    @Passed
    public void nextFloatVector(float[] v)
    {
        Objects.requireNonNull(v);
        for(int i=0;i<v.length;i++) v[i]=this.nextFloat();
    }
    @Passed
    public float[] nextFloatVector(int width, float max)
    {
        return this.nextFloatVector(width, 0, max);
    }
    @Passed
    public void nextFloatVector(float[] v, float max)
    {
        this.nextFloatVector(v, 0, max);
    }
    @Passed
    public float[] nextFloatVector(int width, float min, float max)
    {
        if(max<min) {float t=max;max=min;min=t;}
        ExRandom.checkRealArray(width, min, max);
        float threshold=max-min;
        float[] arr=new float[width];
        for(int i=0;i<width;i++) arr[i]=this.nextFloat()*threshold+min;
        return arr;
    }
    @Passed
    public void nextFloatVector(float[] v, float min, float max)
    {
        if(max<min) {float t=max;max=min;min=t;}
        ExRandom.checkRealArray(v, min, max);
        float threshold=max-min;
        for(int i=0;i<v.length;i++) v[i]=this.nextFloat()*threshold+min;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Vector:exRandom:double">
    @Passed
    public double[] nextDoubleVector(int width)
    {
        ExRandom.checkArrayLength(width);
        double[] arr=new double[width];
        for(int i=0;i<width;i++) arr[i]=this.nextDouble();
        return arr;
    }
    @Passed
    public double[] nextGaussianVector(int width)
    {
        ExRandom.checkArrayLength(width);
        double[] arr=new double[width];
        for(int i=0;i<width;i++) arr[i]=this.nextGaussian();
        return arr;
    }
    @Passed
    public void nextDoubleVector(double[] v)
    {
        Objects.requireNonNull(v);
        for(int i=0;i<v.length;i++) v[i]=this.nextDouble();
    }
    @Passed
    public void nextGaussianVector(double[] v)
    {
        Objects.requireNonNull(v);
        for(int i=0;i<v.length;i++) v[i]=this.nextGaussian();
    }
    @Passed
    public double[] nextDoubleVector(int width, double max)
    {
        return this.nextDoubleVector(width, 0, max);
    }
    @Passed
    public void nextDoubleVector(double[] v, double max)
    {
        this.nextDoubleVector(v, 0, max);
    }
    @Passed
    public double[] nextDoubleVector(int width, double min ,double max)
    {
        if(max<min) {double t=max;max=min;min=t;}
        ExRandom.checkRealArray(width, min, max);
        double threshold=max-min;
        double[] arr=new double[width];
        for(int i=0;i<width;i++) arr[i]=this.nextDouble()*threshold+min;
        return arr;
    }
    @Passed
    public double[] nextGaussinVector(int width, double min, double max)
    {
        if(max<min) {double t=max;max=min;min=t;}
        ExRandom.checkRealArray(width, min, max);
        double threshold=max-min;
        double[] arr=new double[width];
        for(int i=0;i<width;i++) arr[i]=this.nextGaussian()*threshold+min;
        return arr;
    }
    @Passed
    public void nextDoubleVector(double[] v, double min, double max)
    {
        if(max<min) {double t=max;max=min;min=t;}
        ExRandom.checkRealArray(v, min, max);
        double threshold=max-min;
        for(int i=0;i<v.length;i++) v[i]=this.nextDouble()*threshold+min;
    }
    @Passed
    public void nextGaussianVector(double[] v, double min, double max)
    {
        if(max<min) {double t=max;max=min;min=t;}
        ExRandom.checkRealArray(v, min, max);
        double threshold=max-min;
        for(int i=0;i<v.length;i++) v[i]=this.nextGaussian()*threshold+min;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapased" desc="Vector:exRandom:int">
    @Passed
    public int[] nextIntVector(int width)
    {
        ExRandom.checkArrayLength(width);
        int[] arr=new int[width];
        for(int i=0;i<width;i++) arr[i]=this.nextInt();
        return arr;
    }
    @Passed
    public void nextIntVector(int[] v)
    {
        Objects.requireNonNull(v);
        for(int i=0;i<v.length;i++) v[i]=this.nextInt();
    }
    @Passed
    public int[] nextIntVector(int width, int max)
    {
       return this.nextIntVector(width, 0, max);
    }
    @Passed
    public void nextIntVector(int[] v, int max)
    {
        this.nextIntVector(v, 0, max);
    }
    @Passed
    public int[] nextIntVector(int width, int min ,int max)
    {
        if(max<min) {int t=max;max=min;min=t;}
        ExRandom.checkIntArray(width, min, max);
        int threshold=max-min;
        int[] arr=new int[width];
        for(int i=0;i<width;i++) arr[i]=this.nextInt(threshold)+min;
        return arr;
    }
    @Passed
    public void nextIntVector(int[] v, int max, int min)
    {
        if(max<min) {int t=max;max=min;min=t;}
        ExRandom.checkIntArray(v, min, max);
        int threshold=max-min;
        for(int i=0;i<v.length;i++) v[i]=this.nextInt(threshold)+min;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Vector:exRandom:long">
    @Passed
    public long[] nextLongVector(int width)
    {
        ExRandom.checkArrayLength(width);
        long[] arr=new long[width];
        for(int i=0;i<width;i++) arr[i]=this.nextLong();
        return arr;
    }
    @Passed
    public void nextLongVector(long[] v)
    {
        Objects.requireNonNull(v);
        for(int i=0;i<v.length;i++) v[i]=this.nextLong();
    }
    @Passed
    public long[] nextLongVector(int width, long max)
    {
        return this.nextLongVector(width, 0, max);
    }
    @Passed
    public void nextLongVector(long[] v, long max)
    {
        this.nextLongVector(v, 0, max);
    }
    @Passed
    public long[] nextLongVector(int width, long min, long max)
    {
        if(max<min) {long t=max;max=min;min=t;}
        ExRandom.checkIntArray(width, min, max);
        long threshold=max-min;
        long[] arr=new long[width];
        for(int i=0;i<width;i++) arr[i]=(long) (this.nextDouble()*threshold+min);
        return arr;
    }
    @Passed
    public void nextLongVector(long[] v, long min, long max)
    {
        if(max<min) {long t=max;max=min;min=t;}
        ExRandom.checkIntArray(v, min, max);
        long threshold=max-min;
        for(int i=0;i<v.length;i++) v[i]=(long) (this.nextDouble()*threshold);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Vector:Extensive">
    public static interface RandomSupplier<T>
    {
        public T get(Random ran);
    }
    public <T> T[] nextObjectVector(int width, Class<T> clazz, RandomSupplier<T> sp)
    {
        ExRandom.checkArrayLength(width);
        T[] arr=(T[]) Array.newInstance(clazz, width);
        for(int i=0;i<width;i++) arr[i]=sp.get(this);
        return arr;
    }
    public Object[] nextObjectVector(int width, RandomSupplier sp)
    {
        ExRandom.checkArrayLength(width);
        Object[] arr=new Object[width];
        for(int i=0;i<width;i++) arr[i]=sp.get(this);
        return arr;
    }
    public String[] nextStringVector(int width, int length)
    {
        ExRandom.checkArrayLength(width);
        if(length<=0) throw new IAE("the length of String must be positive");
        String[] arr=new String[width];
        for(int i=0;i<width;i++) arr[i]=this.nextString(length);
        return arr;
    }
    public String[] nextNeatStringVector(int width, int length)
    {
        ExRandom.checkArrayLength(width);
        if(length<=0) throw new IAE("the length of String must be positive");
        String[] arr=new String[width];
        for(int i=0;i<width;i++) arr[i]=this.nextNeatString(length);
        return arr;
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Matrix:exRandom">
    //<editor-fold defaultstate="collapsed" desc="Matrix:exRandom:float">
    @Passed
    public float[][] nextFloatMatrix(int height, int width)
    {
        ExRandom.checkMatrixSize(height, width);
        float[][] arr=new float[height][width];
        for(int i=0,j;i<height;i++)
        for(j=0;j<width;j++) arr[i][j]=this.nextFloat();
        return arr;
    }
    @Passed
    public void nextFloatMatrix(float[][] v)
    {
        Objects.requireNonNull(v);
        for(int i=0,j;i<v.length;i++)
        {
            Objects.requireNonNull(v[i]);
            for(j=0;j<v[i].length;j++) v[i][j]=this.nextFloat();
        }
    }
    @Passed
    public float[][] nextFloatMatrix(int height, int width, float max)
    {
        return this.nextFloatMatrix(height, width, 0, max);
    }
    @Passed
    public void nextFloatMatrix(float[][] v, float max)
    {
        this.nextFloatMatrix(v, 0, max);
    }
    @Passed
    public float[][] nextFloatMatrix(int height, int width, float min, float max)
    {
        if(max<min) {float t=max;max=min;min=t;}
        ExRandom.checkRealMatrix(height, width, min, max);
        float threshold=max-min;
        float[][] arr=new float[height][width];
        for(int i=0,j;i<height;i++)
        for(j=0;j<width;j++) arr[i][j]=this.nextFloat()*threshold+min;
        return arr;
    }
    @Passed
    public void nextFloatMatrix(float[][] v, float min, float max)
    {
        if(max<min) {float t=max;max=min;min=t;}
        ExRandom.checkRealMatrix(v, min, max);
        float threshold=max-min;
        for(int i=0,j;i<v.length;i++)
        {
            Objects.requireNonNull(v[i]);
            for(j=0;j<v[i].length;j++) v[i][j]=this.nextFloat()*threshold+min;
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Matrix:exRandom:double">
    @Passed
    public double[][] nextDoubleMatrix(int height, int width)
    {
        ExRandom.checkMatrixSize(height, width);
        double[][] arr=new double[height][width];
        for(int i=0,j;i<height;i++)
        for(j=0;j<width;j++) arr[i][j]=this.nextDouble();
        return arr;
    }
    @Passed
    public double[][] nextGaussianMatrix(int height, int width)
    {
        ExRandom.checkMatrixSize(height, width);
        double[][] arr=new double[height][width];
        for(int i=0,j;i<height;i++)
        for(j=0;j<width;j++) arr[i][j]=this.nextGaussian();
        return arr;
    }
    @Passed
    public double[][] nextDoubleMatrix(double[][] v)
    {
        Objects.requireNonNull(v);
        for(int i=0,j;i<v.length;i++)
        {
            Objects.requireNonNull(v[i]);
            for(j=0;j<v[i].length;i++) v[i][j]=this.nextDouble();
        }
        return v;
    }
    @Passed
    public double[][] nextGaussianMatrix(double[][] v)
    {
        Objects.requireNonNull(v);
        for(int i=0,j;i<v.length;i++)
        {
            Objects.requireNonNull(v[i]);
            for(j=0;j<v[i].length;j++) v[i][j]=this.nextGaussian();
        }
        return v;
    }
    @Passed
    public double[][] nextDoubleMatrix(int height, int width, double max)
    {
        return this.nextDoubleMatrix(height, width, 0, max);
    }
    @Passed
    public double[][] nextGaussianMatrix(int height, int width, double max)
    {
        return this.nextDoubleMatrix(height, width, 0, max);
    }
    @Passed
    public double[][] nextDoubleMatrix(double[][] v, double max)
    {
        this.nextDoubleMatrix(v, 0, max);
        return v;
    }
    @Passed
    public double[][] nextGaussianMatrix(double[][] v, double max)
    {
        this.nextGaussianMatrix(v, 0, max);
        return v;
    }
    @Passed
    public double[][] nextDoubleMatrix(int height, int width, double max, double min)
    {
        if(max<min) {double t=max;max=min;min=t;}
        ExRandom.checkRealMatrix(height, width, min, max);
        double[][] arr=new double[height][width];
        for(int i=0,j;i<height;i++)
        for(j=0;j<width;j++) arr[i][j]=this.nextDouble()*max+min;
        return arr;
    }
    @Passed
    public double[][] nextGaussianMatrix(int height, int width, double max, double min)
    {
        if(max<min) {double t=max;max=min;min=t;}
        ExRandom.checkRealMatrix(height, width, min, max);
        double[][] arr=new double[height][width];
        for(int i=0,j;i<height;i++)
        for(j=0;j<width;j++) arr[i][j]=this.nextGaussian()*max+min;
        return arr;
    }
    @Passed
    public double[][] nextDoubleMatrix(double[][] v, double max, double min)
    {
        if(max<min) {double t=max;max=min;min=t;}
        ExRandom.checkRealMatrix(v, min, max);
        double threshold=max-min;
        for(int i=0,j;i<v.length;i++)
        {
            Objects.requireNonNull(v[i]);
            for(j=0;j<v[i].length;j++) v[i][j]=this.nextDouble()*threshold+min;
        }
        return v;
    }
    @Passed
    public double[][] nextGaussianMatrix(double[][] v, double max, double min)
    {
        if(max<min) {double t=max;max=min;min=t;}
        ExRandom.checkRealMatrix(v, min, max);
        double threshold=max-min;
        for(int i=0,j;i<v.length;i++)
        {
            Objects.requireNonNull(v[i]);
            for(j=0;j<v[i].length;j++) v[i][j]=this.nextGaussian()*threshold+min;
        }
        return v;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Matrix:exRandom:int">
    @Passed
    public int[][] nextIntMatrix(int height, int width)
    {
        ExRandom.checkMatrixSize(height, width);
        int[][] arr=new int[height][width];
        for(int i=0,j;i<height;i++)
        for(j=0;j<width;j++) arr[i][j]=this.nextInt();
        return arr;
    }
    @Passed
    public int[][] nextIntMatrix(int[][] v)
    {
        Objects.requireNonNull(v);
        for(int i=0,j;i<v.length;i++)
        {
            Objects.requireNonNull(v[i]);
            for(j=0;j<v[i].length;j++) v[i][j]=this.nextInt();
        }
        return v;
    }
    @Passed
    public int[][] nextIntMatrix(int height, int width, int max)
    {
        return this.nextIntMatrix(height, width, 0, max);
    }
    @Passed
    public int[][] nextIntMatrix(int[][] v, int max)
    {
        this.nextIntMatrix(v, 0, max);
        return v;
    }
    @Passed
    public int[][] nextIntMatrix(int height, int width, int min, int max)
    {
        if(max<min) {int t=max;max=min;min=t;}
        ExRandom.checkIntMatrix(height, width, min, max);
        int threshold=max-min;
        int[][] arr=new int[height][width];
        for(int i=0,j;i<height;i++)
        for(j=0;j<width;j++) arr[i][j]=this.nextInt(threshold)+min;
        return arr;
    }
    @Passed
    public int[][] nextIntMatrix(int[][] v, int min, int max)
    {
        if(max<min) {int t=max;max=min;min=t;}
        ExRandom.checkIntMatrix(v, min, max);
        int threshold=max-min;
        for(int i=0,j;i<v.length;i++)
        {
            Objects.requireNonNull(v[i]);
            for(j=0;j<v[i].length;j++) v[i][j]=this.nextInt(threshold)+min;
        }
        return v;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Matrix:exRandom:long">
    @Passed
    public long[][] nextLongMatrix(int height, int width)
    {
        ExRandom.checkMatrixSize(height, width);
        long[][] arr=new long[height][width];
        for(int i=0,j;i<height;i++)
        for(j=0;j<width;j++) arr[i][j]=this.nextLong();
        return arr;
    }
    @Passed
    public void nextLongMatrix(long[][] v)
    {
        Objects.requireNonNull(v);
        for(int i=0,j;i<v.length;i++)
        {
            Objects.requireNonNull(v[i]);
            for(j=0;j<v[i].length;j++) v[i][j]=this.nextInt();
        }
    }
    @Passed
    public long[][] nextLongMatrix(int height, int width, long max)
    {   
        return this.nextLongMatrix(height, width, 0, max);
    }
    @Passed
    public void nextLongMatrix(long[][] v, long max)
    {
        this.nextLongMatrix(v, 0, max);
    }
    @Passed
    public long[][] nextLongMatrix(int width, int height, long min, long max)
    {
        if(max<min) {long t=max;max=min;min=t;}
        ExRandom.checkIntMatrix(height, width, min, max);
        long[][] arr=new long[height][width];
        long threshold=max-min;
        for(int i=0,j;i<height;i++)
        for(j=0;j<width;j++) arr[i][j]=(long) (this.nextDouble()*threshold+min);
        return arr;
    }
    @Passed
    public void nextLongMatrix(long[][] v, long min ,long max)
    {
        if(max<min) {long t=max;max=min;min=t;}
        ExRandom.checkIntMatrix(v, min, max);
        long threshold=max-min;
        for(int i=0,j;i<v.length;i++)
        {
            Objects.requireNonNull(v[i]);
            for(j=0;j<v[i].length;i++) v[i][j]=(long) (this.nextDouble()*threshold+min);
        }
    }
    //</editor-fold>
    //</editor-fold>
}
