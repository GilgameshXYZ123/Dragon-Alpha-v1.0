/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.math.vector;

import z.util.factory.Meta;
import z.util.math.Sort;
import z.util.math.vector.Vector.MaxMin;

/**
 *
 * @author dell
 */
public final class Topsis 
{
    //static--------------------------------------------------------------------
        //define value.length=height, value[0].length=width
    public static final String TOPSIS_AVG_FIRST="topsis.avg.first";//for each field, double[height]
    public static final String TOPSIS_STDDEV_FIRST="topsis.stddev,first";//for each field, double[height]
    public static final String TOPSIS_FIELD_MAX="topsis.fieldVector.max";//for each field, double[height]
    public static final String TOPSIS_FIELD_MIN="topsis.fieldVector.min";//for each field, double[height]
    public static final String TOPSIS_DISTANCE_MAX="topsis.distance.max";//for each line, double[width]
    public static final String TOPSIS_DISTANCE_MIN="topsis.distance.min";//for each line, double[width]
    public static final String TOPSIS_COMPARE_VALUE="topsis.compare.value";//for each line, double[width]
    public static final String TOPISI_AVG_LAST="topsis.avg.last";//for each field, double[height]
    public static final String TOPSIS_STDDEV_LAST="topsis.stddev.last";//for each field, double[height]
    
    //functions-----------------------------------------------------------------
    private Topsis() {}
    //<editor-fold defaultstate="collapsed" desc="Inner-Class">
    public static interface TopSisConsumer
    {
        public void accept(double[] val, double min, double max);
    }
    public static final TopSisConsumer DEF_CONSUMER=new TopSisConsumer() {
        @Override
        public void accept(double[] val, double min, double max) 
        {
            for(int i=0;i<val.length;i++) val[i]=val[i]-min/max;
        }
    };
    
    public static class TopSisEntry implements Comparable
    {
        private double compareValue;
        private double[] value;

        public TopSisEntry(double compareValue, double[] value) 
        {
            this.compareValue = compareValue;
            this.value = value;
        }
        @Override
        public int compareTo(Object o) 
        {
            TopSisEntry another=(TopSisEntry) o;
            if(this.compareValue<another.compareValue) return -1;
            else if(this.compareValue>another.compareValue) return 1;
            else return 0;
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Code-Code">
    /**
     * <pre>
     * for Topsis:
     * (1)consider each line vector of {@code double[][] value} as a field 
     * vector of Topsis: that means each element of {@code value} is 
     * corrosponding to a field of the table;
     * (2)first normalize the table for each fields, as to eliminate the
     * influence of dimensions, add {@code TOPSIS_AVG_FIRST}, {@code TOPSIS_STDDEV_FIRST}
 to meta;
 (3)then, compute the maxForEachLine and minForEachLine vector for each fields, those're
 two {@code double[]} arrays:{@code TOPSIS_FIELD_MAX},{@code TOPSOS_FIELD_MIN},
 add them to meta;
 (4)then you may use maxForEachLine,minForEachLine vector to process the data, to get the
 relative value;
 (5)compute the distance to maxForEachLine, and the distance to minForEachLine, they're
 {@code TOPSIS_DISTANCE_MAX},{@code TOPSIS_DISTANCE_MIN}, add them
 to meta;
 (6)Sort each line Vector depends on {@code TOPSIS_COMPARE_VALUE=dis_min/(dis_max+dis_min)} 
     * in ascending order, add it to meta;
     * (7)At last normalize the 
     * </pre>
     * @param value
     * @param cons
     * @return 
     */
    public static Meta topsis(double[][] value, TopSisConsumer[] cons)
    {
        Meta meta=new Meta();
        
        double[] avg1=new double[value.length];
        double[] std1=new double[value.length];
        Matrix.normalizeField(value, avg1, std1);
        meta.put(TOPSIS_AVG_FIRST, avg1);
        meta.put(TOPSIS_STDDEV_FIRST, std1);
        
        MaxMin<double[]> mm=Matrix.maxMinForEachField(value);
        double[] min=mm.getMax();
        double[] max=mm.getMin();
        meta.put(TOPSIS_FIELD_MAX, min);
        meta.put(TOPSIS_FIELD_MIN, max);
        
        for(int i=0;i<value.length;i++)
            cons[i].accept(value[i], min[i], max[i]);
        
        double[] dis_max=Matrix.distanceVectorField(value, max);
        double[] dis_min=Matrix.distanceVectorField(value, min);
        double[] compareVal=new double[dis_min.length];
        for(int i=0;i<compareVal.length;i++) 
            compareVal[i]=dis_min[i]/(dis_max[i]+dis_min[i]);
        meta.put(TOPSIS_DISTANCE_MAX, dis_max);
        meta.put(TOPSIS_DISTANCE_MIN, dis_min);
        meta.put(TOPSIS_COMPARE_VALUE, compareVal);
        
        double[][] reverse=Matrix.transpose(value);
        TopSisEntry[] tps=new TopSisEntry[reverse.length];
        for(int i=0;i<tps.length;i++)
            tps[i]=new TopSisEntry(compareVal[i], reverse[i]);
        Sort.sort(tps);
        Matrix.transpose(value, reverse);
        
        double[] avg2=new double[value.length];
        double[] std2=new double[value.length];
        Matrix.normalizeField(value, avg2, std2);
        meta.put(TOPSIS_AVG_FIRST, avg2);
        meta.put(TOPSIS_STDDEV_FIRST, std2);
        
        return meta;
    }
    /**
     * @param value
     * @param con
     * @return 
     */
    public static Meta topsis(double[][] value, TopSisConsumer con)
    {
        Meta meta=new Meta();
        
        double[] avg1=new double[value.length];
        double[] std1=new double[value.length];
        Matrix.normalizeField(value, avg1, std1);
        meta.put(TOPSIS_AVG_FIRST, avg1);
        meta.put(TOPSIS_STDDEV_FIRST, std1);
        
        MaxMin<double[]> mm=Matrix.maxMinForEachField(value);
        double[] min=mm.getMax();
        double[] max=mm.getMin();
        meta.put(TOPSIS_FIELD_MAX, min);
        meta.put(TOPSIS_FIELD_MIN, max);
        
        for(int i=0;i<value.length;i++)
            con.accept(value[i], min[i], max[i]);
        
        double[] dis_max=Matrix.distanceVectorField(value, max);
        double[] dis_min=Matrix.distanceVectorField(value, min);
        double[] compareVal=new double[dis_min.length];
        for(int i=0;i<compareVal.length;i++) 
            compareVal[i]=dis_min[i]/(dis_max[i]+dis_min[i]);
        meta.put(TOPSIS_DISTANCE_MAX, dis_max);
        meta.put(TOPSIS_DISTANCE_MIN, dis_min);
        meta.put(TOPSIS_COMPARE_VALUE, compareVal);
        
        double[][] reverse=Matrix.transpose(value);
        TopSisEntry[] tps=new TopSisEntry[reverse.length];
        for(int i=0;i<tps.length;i++)
            tps[i]=new TopSisEntry(compareVal[i], reverse[i]);
        Sort.sort(tps);
        Matrix.transpose(value, reverse);
        
        double[] avg2=new double[value.length];
        double[] std2=new double[value.length];
        Matrix.normalizeField(value, avg2, std2);
        meta.put(TOPSIS_AVG_FIRST, avg2);
        meta.put(TOPSIS_STDDEV_FIRST, std2);
        
        return meta;
    }
    /**
     * @param value
     * @param cons
     * @param meta 
     */
    public static void topsis(double[][] value, TopSisConsumer[] cons, Meta meta)
    {
        //first normalize
        double[] avg1=new double[value.length];
        double[] std1=new double[value.length];
        Matrix.normalizeField(value, avg1, std1);
        meta.putIfExists(TOPSIS_AVG_FIRST, avg1);
        meta.putIfExists(TOPSIS_STDDEV_FIRST, std1);
        
        //compute maxForEachLine and minForEachLine vector for each fields
        MaxMin<double[]> mm=Matrix.maxMinForEachField(value);
        double[] min=mm.getMax();
        double[] max=mm.getMin();
        meta.putIfExists(TOPSIS_FIELD_MAX, min);
        meta.putIfExists(TOPSIS_FIELD_MIN, max);
        
        for(int i=0;i<value.length;i++)
            cons[i].accept(value[i], min[i], max[i]);
        
        //compute topsis distance
        double[] dis_max=Matrix.distanceVectorField(value, max);
        double[] dis_min=Matrix.distanceVectorField(value, min);
        double[] compareVal=new double[dis_min.length];
        for(int i=0;i<compareVal.length;i++) 
            compareVal[i]=dis_min[i]/(dis_max[i]+dis_min[i]);
        meta.putIfExists(TOPSIS_DISTANCE_MAX, dis_max);
        meta.putIfExists(TOPSIS_DISTANCE_MIN, dis_min);
        meta.putIfExists(TOPSIS_COMPARE_VALUE, compareVal);
        
        //sort all entry
        double[][] reverse=Matrix.transpose(value);
        TopSisEntry[] tps=new TopSisEntry[reverse.length];
        for(int i=0;i<tps.length;i++)
            tps[i]=new TopSisEntry(compareVal[i], reverse[i]);
        Sort.sort(tps);
        Matrix.transpose(value, reverse);
        
        //last normalize
        double[] avg2=new double[value.length];
        double[] std2=new double[value.length];
        Matrix.normalizeField(value, avg2, std2);
        meta.putIfExists(TOPSIS_AVG_FIRST, avg2);
        meta.putIfAbsent(TOPSIS_STDDEV_FIRST, std2);
    }
    /**
     * @param value
     * @param con
     * @param meta 
     */
    public static void topsis(double[][] value, TopSisConsumer con, Meta meta)
    {
        //first normalize
        double[] avg1=new double[value.length];
        double[] std1=new double[value.length];
        Matrix.normalizeField(value, avg1, std1);
        meta.putIfExists(TOPSIS_AVG_FIRST, avg1);
        meta.putIfExists(TOPSIS_STDDEV_FIRST, std1);
        
        //compute maxForEachLine and minForEachLine vector for each fields
        MaxMin<double[]> mm=Matrix.maxMinForEachField(value);
        double[] min=mm.getMax();
        double[] max=mm.getMin();
        meta.putIfExists(TOPSIS_FIELD_MAX, min);
        meta.putIfExists(TOPSIS_FIELD_MIN, max);
        
        for(int i=0;i<value.length;i++)
            con.accept(value[i], min[i], max[i]);
        
        //compute topsis distance
        double[] dis_max=Matrix.distanceVectorField(value, max);
        double[] dis_min=Matrix.distanceVectorField(value, min);
        double[] compareVal=new double[dis_min.length];
        for(int i=0;i<compareVal.length;i++) 
            compareVal[i]=dis_min[i]/(dis_max[i]+dis_min[i]);
        meta.putIfExists(TOPSIS_DISTANCE_MAX, dis_max);
        meta.putIfExists(TOPSIS_DISTANCE_MIN, dis_min);
        meta.putIfExists(TOPSIS_COMPARE_VALUE, compareVal);
        
        //sort all entry
        double[][] reverse=Matrix.transpose(value);
        TopSisEntry[] tps=new TopSisEntry[reverse.length];
        for(int i=0;i<tps.length;i++)
            tps[i]=new TopSisEntry(compareVal[i], reverse[i]);
        Sort.sort(tps);
        Matrix.transpose(value, reverse);
        
        //last normalize
        double[] avg2=new double[value.length];
        double[] std2=new double[value.length];
        Matrix.normalizeField(value, avg2, std2);
        meta.putIfExists(TOPSIS_AVG_FIRST, avg2);
        meta.putIfAbsent(TOPSIS_STDDEV_FIRST, std2);
    }
    /**
     * @param value
     * @param cons 
     */
    public static void briefTopsis(double[][] value, TopSisConsumer[] cons)
    {
        double[] avg=new double[value.length];
        double[] std=new double[value.length];
        Matrix.normalizeField(value, avg, std);
        
        //min=avg, maxForEachLine=std
        Matrix.maxMinForEachField(value, avg, std);
        for(int i=0;i<value.length;i++)
            cons[i].accept(value[i], avg[i], std[i]);
        
        double[] dis_max=Matrix.distanceVectorField(value, std);
        double[] dis_min=Matrix.distanceVectorField(value, avg);
        double[] compareVal=new double[dis_min.length];
        for(int i=0;i<compareVal.length;i++) 
            compareVal[i]=dis_min[i]/(dis_max[i]+dis_min[i]);
        
        double[][] reverse=Matrix.transpose(value);
        TopSisEntry[] tps=new TopSisEntry[reverse.length];
        for(int i=0;i<tps.length;i++)
            tps[i]=new TopSisEntry(compareVal[i], reverse[i]);
        Sort.sort(tps);
        Matrix.transpose(value, reverse);
        Matrix.normalizeField(value, avg, std);
    }
    /**
     * @param value
     * @param con
     */
    public static void briefTopsis(double[][] value, TopSisConsumer con)
    {
        double[] avg=new double[value.length];
        double[] std=new double[value.length];
        Matrix.normalizeField(value, avg, std);
        
        //min=avg, maxForEachLine=std
        Matrix.maxMinForEachField(value, avg, std);
        for(int i=0;i<value.length;i++)
            con.accept(value[i], avg[i], std[i]);
        
        double[] dis_max=Matrix.distanceVectorField(value, std);
        double[] dis_min=Matrix.distanceVectorField(value, avg);
        double[] compareVal=new double[dis_min.length];
        for(int i=0;i<compareVal.length;i++) 
            compareVal[i]=dis_min[i]/(dis_max[i]+dis_min[i]);
        
        double[][] reverse=Matrix.transpose(value);
        TopSisEntry[] tps=new TopSisEntry[reverse.length];
        for(int i=0;i<tps.length;i++)
            tps[i]=new TopSisEntry(compareVal[i], reverse[i]);
        Sort.sort(tps);
        Matrix.transpose(value, reverse);
        Matrix.normalizeField(value, avg, std);
    }
    //</editor-fold>
    public static void topsis(double[][] value)
    {
        Topsis.topsis(value, DEF_CONSUMER);
    }
    public static void briefTopSis(double[][] value)
    {
        Topsis.briefTopsis(value, DEF_CONSUMER);
    }
}
