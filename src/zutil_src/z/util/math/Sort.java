/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.math;

import java.util.Arrays;
import java.util.Comparator;
import z.util.math.vector.Vector;
import java.util.Objects;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import z.util.concurrent.BinarySemaphore;
import z.util.concurrent.Lock;
import z.util.lang.Lang;
import static z.util.lang.Lang.MB_BIT;
import z.util.lang.SimpleTimer;
import z.util.math.vector.Vector.MaxMin;
import z.util.lang.annotation.Passed;

/**
 * @author dell
 */
public final class Sort
{
    public static final int INSERT_SORT_THRESHOLD=4;
    public static final int SELECT_SORT_THRESHOLD=8;
    public static final int SHELL_SORT_THRESHOLD=32;
    public static final int QUICK_SORT_THRESHOLD=186;
    public static final int COUNTING_SORT_RANGE_INT=(MB_BIT)/Integer.SIZE;
    public static final int TIM_SORT_SEGMENT_NUM=48;
    public static final int SINGLE_THREAD_THRESHOLD=130000;
    public static final int MULTI_THREAD_LEAF= 500000;
    
    //<editor-fold defaultstate="collapsed" desc="Speed-Test">
    public static void test1()
    {
        int turn=10;
        long sum=0;
        SimpleTimer ss=new SimpleTimer();
        for(int i=0;i<turn;i++)
        {
            double[] v=Vector.randomDoubleVector(2000000, 1000);
            ss.record();
//            Sort.sort(v);
            Arrays.sort(v);
            ss.record();
            sum+=ss.timeStampDifMills();
            System.out.println(Vector.isAscendingOrder(v));
            System.out.println(ss);
        }
        System.out.println(sum/turn);
    }
    //</editor-fold>
    
    private Sort() {}
    //<editor-fold defaultstate="collapsed" desc="Sorter:Inner-Code">
    //<editor-fold defaultstate="collapsed" desc="Counting-Sort">
    @Passed
    public static boolean innerCountingSort(int[] a, int low, int high)
    {
        MaxMin<Integer> mm=Vector.maxMin(a, low, high, COUNTING_SORT_RANGE_INT);
        if(mm==null) return false;
        int min=mm.getMin(),dif=mm.getMax()-min+1;
        if(dif> COUNTING_SORT_RANGE_INT) return false;
        int[] h=new int[dif];
        int i,j,index;
        for(i=low;i<=high;i++) h[a[i]-min]++;
        for(i=0,index=low;i<dif;i++)
            for(j=0;j<h[i];j++) a[index++]=min+i;
        return true;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Insert-Sort">
    @Passed
    public static void innerInsertSort(Comparable[] a, int low, int high)
    {
        Comparable k;
        for(int i=low+1,j;i<=high;i++)
        {
            for(k=a[low], a[low]=a[i], j=i-1;a[low].compareTo(a[j])<0;j--) a[j+1]=a[j];//a[i]<a[j
            if(j!=low) {a[j+1]=a[low];a[low]=k;continue;}
            if(a[low].compareTo(k)<0) a[low+1]=k;//k==a[0]
            else {a[low+1]=a[low];a[low]=k;}
        }
    }
    @Passed
    public static void innerInsertSort(Object[] a, Comparator cmp, int low, int high)
    {
         Object k;
        for(int i=low+1,j;i<=high;i++)
        {
            for(k=a[low], a[low]=a[i], j=i-1;cmp.compare(a[low], a[j])<0;j--) a[j+1]=a[j];//a[i]<a[j
            if(j!=low) {a[j+1]=a[low];a[low]=k;continue;}
            if(cmp.compare(a[low], k)<0) a[low+1]=k;//k==a[0]
            else {a[low+1]=a[low];a[low]=k;}
        }
    }
    @Passed
    public static void innerInsertSort(int[] a, int low, int high)
    {
        int k;
        for(int i=low+1,j;i<=high;i++)
        {
            for(k=a[low], a[low]=a[i], j=i-1;a[low]<a[j];j--) a[j+1]=a[j];//a[i]<a[j
            if(j!=low) {a[j+1]=a[low];a[low]=k;continue;}
            if(a[low]<k) a[low+1]=k;//k==a[0]
            else a[low+1]=a[low];a[low]=k;
        }
    }
    @Passed
    public static void innerInsertSort(float[] a, int low, int high)
    {
        float k;
        for(int i=low+1,j;i<=high;i++)
        {
            for(k=a[low], a[low]=a[i], j=i-1;a[low]<a[j];j--) a[j+1]=a[j];//a[i]<a[j
            if(j!=low) {a[j+1]=a[low];a[low]=k;continue;}
            if(a[low]<k) a[low+1]=k;//k==a[0]
            else {a[low+1]=a[low];a[low]=k;}
        }
    }
    @Passed
    public static void innerInsertSort(double[] a, int low, int high)
    {
        double k;
        for(int i=low+1,j;i<=high;i++)
        {
            for(k=a[low], a[low]=a[i], j=i-1;a[low]<a[j];j--) a[j+1]=a[j];//a[i]<a[j
            if(j!=low) {a[j+1]=a[low];a[low]=k;continue;}
            if(a[low]<k) a[low+1]=k;//k==a[0]
            else {a[low+1]=a[low];a[low]=k;}
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="SelectSort">
    @Passed
    public static void innerSelectSort(Comparable[] a, int low, int high)
    {
        Comparable p;
        for(int i=low,j;i<high;i++)
        for(j=i+1;j<=high;j++)
            if(a[j].compareTo(a[i])<0) {p=a[i];a[i]=a[j];a[j]=p;}
    } 
    @Passed
    public static void innerSelectSort(Object[] a, Comparator cmp, int low, int high)
    {
        Object p;
        for(int i=low,j;i<high;i++)
        for(j=i+1;j<=high;j++)
            if(cmp.compare(a[j], a[i])<0) {p=a[i];a[i]=a[j];a[j]=p;}
    }
    @Passed
    public static void innerSelectSort(int[] a, int low, int high)
    {
        int p;
        for(int i=low,j;i<=high;i++)
        for(j=i+1;j<=high;j++)
            if(a[j]<a[i]) {p=a[i];a[i]=a[j];a[j]=p;}
    }
    @Passed
    public static void innerSelectSort(float[] a, int low, int high)
    {
        float p;
        for(int i=low,j;i<=high;i++)
        for(j=i+1;j<=high;j++)
            if(a[j]<a[i]) {p=a[i];a[i]=a[j];a[j]=p;}
    }
    @Passed
    public static void innerSelectSort(double[] a, int low, int high)
    {
        double p;
        for(int i=low,j;i<=high;i++)
        for(j=i+1;j<=high;j++)
            if(a[j]<a[i]) {p=a[i];a[i]=a[j];a[j]=p;}
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="ShellSort">
    @Passed
    public static void innerShellSort(Comparable[] a, int low, int high)
    {
        int i,j,h=1,p;
        for(int top=(high-low+1)/3;h<top;h=h*3+1);
        Comparable t;
        for(;h>0;h/=3)
        {
            for(i=h+low;i<=high;i++)
            for(j=i;j>=h&&a[j].compareTo(a[p=j-h])<0;j=p)
                {t=a[j];a[j]=a[p];a[p]=t;}
        }
    }
    @Passed
    public static void innerShellSort(Object[] a, Comparator cmp, int low, int high)
    {
        int i,j,h=1,p;
        for(int top=(high-low+1)/3;h<top;h=h*3+1);
        Object t;
        for(;h>0;h/=3)
        {
            for(i=h+low;i<=high;i++)
            for(j=i;j>=h&&cmp.compare(a[j], a[p=j-h])<0;j=p)
                {t=a[j];a[j]=a[p];a[p]=t;}
        }
    }
    @Passed
    public static void innerShellSort(int[] a, int low, int high)
    {
        int i,j,h=1,p;
        for(int top=(high-low+1)/3;h<top;h=h*3+1);
        int t;
        for(;h>0;h/=3)
        {
            for(i=h+low;i<=high;i++)
            for(j=i;j>=h&&a[j]<a[p=j-h];j=p)
                {t=a[j];a[j]=a[p];a[p]=t;}
        }
    }
    @Passed
    public static void innerShellSort(float[] a, int low, int high)
    {
        int i,j,h=1,p;
        for(int top=(high-low+1)/3;h<top;h=h*3+1);
        float t;
        for(;h>0;h/=3)
        {
            for(i=h+low;i<=high;i++)
            for(j=i;j>=h&&a[j]<a[p=j-h];j=p)
                {t=a[j];a[j]=a[p];a[p]=t;}
        }
    }
    @Passed
    public static void innerShellSort(double[] a, int low, int high)
    {
        int i,j,h=1,p;
        for(int top=(high-low+1)/3;h<top;h=h*3+1);
        double t;
        for(;h>0;h/=3)
        {
            for(i=h+low;i<=high;i++)
            for(j=i;j>=h&&a[j]<a[p=j-h];j=p)
                {t=a[j];a[j]=a[p];a[p]=t;}
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="QuickSort">
    @Passed
    public static void innerQuickSort(Comparable[] a, int low, int high)
    {
        if(low<=high) return;
        int index=Lang.exRandom().nextInt(low,high);
        Comparable t=a[low];a[low]=a[index];a[index]=t;
        int p=Vector.partition(a, low, high);
        Sort.innerQuickSort(a, low, p-1);
        Sort.innerQuickSort(a, p+1, high);
    }
    @Passed
    public static  void innerQuickSort(int[] a, int low, int high)
    {
        if(low>=high) return;
        int index=Lang.exRandom().nextInt(low,high);
        int t=a[low];a[low]=a[index];a[index]=t;
        int p=Vector.partition(a, low, high);
        Sort.innerQuickSort(a, low, p-1);
        Sort.innerQuickSort(a, p+1, high);
    }
    @Passed
    public static  void innerQuickSort(float[] a, int low, int high)
    {
        if(low>=high) return;
        int index=Lang.exRandom().nextInt(low,high);
        float t=a[low];a[low]=a[index];a[index]=t;
        int p=Vector.partition(a, low, high);
        Sort.innerQuickSort(a, low, p-1);
        Sort.innerQuickSort(a, p+1, high);
    }
    @Passed
    public static  void innerQuickSort(double[] a, int low, int high)
    {
        if(low>=high) return;
        int index=Lang.exRandom().nextInt(low,high);
        double t=a[low];a[low]=a[index];a[index]=t;
        int p=Vector.partition(a, low, high);
        Sort.innerQuickSort(a, low, p-1);
        Sort.innerQuickSort(a, p+1, high);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="QuickSort:ThreeQuickSort">
    @Passed
    public static void innerThreeQuickSort(Comparable[] a, int low, int high)
    {
        if(low>=high) return;
        int index=Lang.exRandom().nextInt(low,high);
        Comparable t=a[low];a[low]=a[index];a[index]=t;
        long p=Vector.threePartition(a, low, high);
        int p0=Num.low32(p),p1=Num.high32(p);
        Sort.innerThreeQuickSort(a, low, p0-1);
        Sort.innerThreeQuickSort(a, p1+1, high);
    }
    @Passed
    public static void innerThreeQuickSort(int[] a, int low, int high)
    {
        if(low>=high) return;
        int index=Lang.exRandom().nextInt(low,high);
        int t=a[low];a[low]=a[index];a[index]=t;
        long p=Vector.threePartition(a, low, high);
        int p0=Num.low32(p),p1=Num.high32(p);
        Sort.innerThreeQuickSort(a, low, p0-1);
        Sort.innerThreeQuickSort(a, p1+1, high);
    }
    @Passed
    public static void innerThreeQuickSort(float[] a, int low, int high)
    {
        if(low>=high) return;
        int index=Lang.exRandom().nextInt(low,high);
        float t=a[low];a[low]=a[index];a[index]=t;
        long p=Vector.threePartition(a, low, high);
        int p0=Num.low32(p),p1=Num.high32(p);
        Sort.innerThreeQuickSort(a, low, p0-1);
        Sort.innerThreeQuickSort(a, p1+1, high);
    }
   @Passed
    public static void innerThreeQuickSort(double[] a, int low, int high)
    {
        if(low>=high) return;
        int index=Lang.exRandom().nextInt(low,high);
        double t=a[low];a[low]=a[index];a[index]=t;
        long p=Vector.threePartition(a, low, high);
        int p0=Num.low32(p),p1=Num.high32(p);
        Sort.innerThreeQuickSort(a, low, p0-1);
        Sort.innerThreeQuickSort(a, p1+1, high);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="QuickSort:DualPivotQuickSort">
    @Passed
    public static void innerDualPivotQuickSort(Comparable[] a, int low, int high)
    {
        if(low>=high) return;
        
        ExRandom ran=Lang.exRandom();
        int index=ran.nextInt(low, high);
        Comparable t=a[low];a[low]=a[index];a[index]=t;
        index=ran.nextInt(low,high);
        t=a[high];a[high]=a[index];a[index]=t;
        
        long p=Vector.dualPivotPartition(a, low, high);
        int p0=Num.low32(p),p1=Num.high32(p);
        Sort.innerDualPivotQuickSort(a, low, p0-1);
        Sort.innerDualPivotQuickSort(a, p0+1, p1-1);
        Sort.innerDualPivotQuickSort(a, p1+1, high);
    }
     @Passed
    public static void innerDualPivotQuickSort(Object[] a, Comparator cmp, int low, int high)
    {
        if(low>=high) return;
        
        ExRandom ran=Lang.exRandom();
        int index=ran.nextInt(low, high);
        Object t=a[low];a[low]=a[index];a[index]=t;
        index=ran.nextInt(low,high);
        t=a[high];a[high]=a[index];a[index]=t;
        
        long p=Vector.dualPivotPartition(a, cmp, low, high);
        int p0=Num.low32(p),p1=Num.high32(p);
        Sort.innerDualPivotQuickSort(a, cmp, low, p0-1);
        Sort.innerDualPivotQuickSort(a, cmp, p0+1, p1-1);
        Sort.innerDualPivotQuickSort(a, cmp, p1+1, high);
    }
    @Passed
    public static void innerDualPivotQuickSort(int[] a, int low, int high)
    {
        if(low>=high) return;
        
        ExRandom ran=Lang.exRandom();
        int index=ran.nextInt(low, high);
        int t=a[low];a[low]=a[index];a[index]=t;
        index=ran.nextInt(low,high);
        t=a[high];a[high]=a[index];a[index]=t;
        
        long p=Vector.dualPivotPartition(a, low, high);
        int p0=Num.low32(p),p1=Num.high32(p);
        Sort.innerDualPivotQuickSort(a, low, p0-1);
        Sort.innerDualPivotQuickSort(a, p0+1, p1-1);
        Sort.innerDualPivotQuickSort(a, p1+1, high);
    }
    @Passed
    public static void innerDualPivotQuickSort(float[] a, int low, int high)
    {
        if(low>=high) return;
        
        ExRandom ran=Lang.exRandom();
        int index=ran.nextInt(low, high);
        float t=a[low];a[low]=a[index];a[index]=t;
        index=ran.nextInt(low,high);
        t=a[high];a[high]=a[index];a[index]=t;
        
        long p=Vector.dualPivotPartition(a, low, high);
        int p0=Num.low32(p),p1=Num.high32(p);
        Sort.innerDualPivotQuickSort(a, low, p0-1);
        Sort.innerDualPivotQuickSort(a, p0+1, p1-1);
        Sort.innerDualPivotQuickSort(a, p1+1, high);
    }
    @Passed
    public static void innerDualPivotQuickSort(double[] a, int low, int high)
    {
        if(low>=high) return;
        
        ExRandom ran=Lang.exRandom();
        int index=ran.nextInt(low, high);
        double t=a[low];a[low]=a[index];a[index]=t;
        index=ran.nextInt(low,high);
        t=a[high];a[high]=a[index];a[index]=t;
        
        long p=Vector.dualPivotPartition(a, low, high);
        int p0=Num.low32(p),p1=Num.high32(p);
        Sort.innerDualPivotQuickSort(a, low, p0-1);
        Sort.innerDualPivotQuickSort(a, p0+1, p1-1);
        Sort.innerDualPivotQuickSort(a, p1+1, high);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="MergeSort">
    @Passed
    public static void innerMergeSort(Comparable[] a, int low, int high)
    {
        if(low>=high) return;
        int mid=(low+high)>>1;
        Sort.innerMergeSort(a, low, mid);
        Sort.innerMergeSort(a, mid+1, high);
        Vector.merge(a, low, mid, high);
    }
    public static void innerMergeSort(int[] a, int low, int high)
    {
        if(low>=high) return;
        int mid=(low+high)>>1;
        Sort.innerMergeSort(a, low, mid);
        Sort.innerMergeSort(a, mid+1, high);
        Vector.merge(a, low, mid, high);
    }
    public static void innerMergeSort(float[] a, int low, int high)
    {
        if(low>=high) return;
        int mid=(low+high)>>1;
        Sort.innerMergeSort(a, low, mid);
        Sort.innerMergeSort(a, mid+1, high);
        Vector.merge(a, low, mid, high);
    }
    public static void innerMergeSort(double[] a, int low, int high)
    {
        if(low>=high) return;
        int mid=(low+high)>>1;
        Sort.innerMergeSort(a, low, mid);
        Sort.innerMergeSort(a, mid+1, high);
        Vector.merge(a, low, mid, high);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="MergeSort:TimSort">
    @Passed
    public static void timMerge(int[] edge, int len, Comparable[] a)
    {
        //len>=1
        //if there are three segments in stack, then combine y with smaller(z,x)
        int[] stack=new int[4];
        for(int i=0,index=0;i<=len;i++)
        {
            stack[index++]=edge[i];
            if(index==4)
            {
                if(stack[3]-stack[2]<stack[1]-stack[0]) 
                    Vector.merge(a, stack[1], stack[2]-1, stack[3]-1);//merge y and x
                else
                {
                    Vector.merge(a, stack[0], stack[1]-1, stack[2]-1);//merge y and z
                    stack[1]=stack[2];
                }
                stack[2]=stack[3];
                index--;
            }
        }
        Vector.merge(a, stack[0], stack[1]-1, stack[2]-1);
    }
    @Passed
    public static void timMerge(int[] edge, int len, Object[] a, Comparator cmp)
    {
        //len>=1
        //if there are three segments in stack, then combine y with smaller(z,x)
        int[] stack=new int[4];
        for(int i=0,index=0;i<=len;i++)
        {
            stack[index++]=edge[i];
            if(index==4)
            {
                if(stack[3]-stack[2]<stack[1]-stack[0]) 
                    Vector.merge(a, cmp, stack[1], stack[2]-1, stack[3]-1);//merge y and x
                else
                {
                    Vector.merge(a, cmp, stack[0], stack[1]-1, stack[2]-1);//merge y and z
                    stack[1]=stack[2];
                }
                stack[2]=stack[3];
                index--;
            }
        }
        Vector.merge(a, cmp, stack[0], stack[1]-1, stack[2]-1);
    }
    @Passed
    public static void timMerge(int[] edge, int len, int[] a)
    {
        //len>=1
        //if there are three segments in stack, then combine y with smaller(z,x)
        int[] stack=new int[4];
        for(int i=0,index=0;i<=len;i++)
        {
            stack[index++]=edge[i];
            if(index==4)
            {
                if(stack[3]-stack[2]<stack[1]-stack[0]) 
                    Vector.merge(a, stack[1], stack[2]-1, stack[3]-1);//merge y and x
                else
                {
                    Vector.merge(a, stack[0], stack[1]-1, stack[2]-1);//merge y and z
                    stack[1]=stack[2];
                }
                stack[2]=stack[3];
                index--;
            }
        }
        Vector.merge(a, stack[0], stack[1]-1, stack[2]-1);
    }
    @Passed
    public static void timMerge(int[] edge, int len, float[] a)
    {
        //len>=1
        //if there are three segments in stack, then combine y with smaller(z,x)
        int[] stack=new int[4];
        for(int i=0,index=0;i<=len;i++)
        {
            stack[index++]=edge[i];
            if(index==4)
            {
                if(stack[3]-stack[2]<stack[1]-stack[0]) 
                    Vector.merge(a, stack[1], stack[2]-1, stack[3]-1);//merge y and x
                else
                {
                    Vector.merge(a, stack[0], stack[1]-1, stack[2]-1);//merge y and z
                    stack[1]=stack[2];
                }
                stack[2]=stack[3];
                index--;
            }
        }
        Vector.merge(a, stack[0], stack[1]-1, stack[2]-1);
    }
    @Passed
    public static void timMerge(int[] edge, int len, double[] a)
    {
        //len>=1
        //if there are three segments in stack, then combine y with smaller(z,x)
        int[] stack=new int[4];
        for(int i=0,index=0;i<=len;i++)
        {
            stack[index++]=edge[i];
            if(index==4)
            {
                if(stack[3]-stack[2]<stack[1]-stack[0]) 
                    Vector.merge(a, stack[1], stack[2]-1, stack[3]-1);//merge y and x
                else
                {
                    Vector.merge(a, stack[0], stack[1]-1, stack[2]-1);//merge y and z
                    stack[1]=stack[2];
                }
                stack[2]=stack[3];
                index--;
            }
        }
        Vector.merge(a, stack[0], stack[1]-1, stack[2]-1);
    }
    @Passed
    public static boolean innerTimSort(Comparable[] a, int low, int high)
    {
        Comparable t;
        int index=0,start,end;
        int[] edge=new int[TIM_SORT_SEGMENT_NUM+1];
        edge[index]=low;
        //find element-ordered segment:each[edge[i], edge[i+1]]
        for(int k=low;k<high&&index<TIM_SORT_SEGMENT_NUM;edge[++index]=++k)
        {
            if(a[k].compareTo(a[k+1])<=0) 
                while(++k<high&&a[k].compareTo(a[k+1])<=0);
            else
            {
                while(++k<high&&a[k].compareTo(a[k+1])>=0);
                for(start=edge[index],end=k;start<end;start++,end--)
                    {t=a[start];a[start]=a[end];a[end]=t;}
            }
        }
        if(index==0) return true;
        if(index<TIM_SORT_SEGMENT_NUM) 
        {
            edge[++index]=high+1;
            Sort.timMerge(edge, index, a);
            return true;
        }
        return false;
    }
    @Passed
    public static  boolean innerTimSort(Object[] a, Comparator cmp, int low, int high)
    {
        Object t;
        int index=0,start,end;
        int[] edge=new int[TIM_SORT_SEGMENT_NUM+1];
        edge[index]=low;
        //find element-ordered segment:each[edge[i], edge[i+1]]
        for(int k=low;k<high&&index<TIM_SORT_SEGMENT_NUM;edge[++index]=++k)
        {
            if(cmp.compare(a[k], a[k+1])<=0) 
                while(++k<high&&cmp.compare(a[k], a[k+1])<=0);
            else
            {
                while(++k<high&&cmp.compare(a[k], a[k+1])>=0);
                for(start=edge[index],end=k;start<end;start++,end--)
                    {t=a[start];a[start]=a[end];a[end]=t;}
            }
        }
        if(index==0) return true;
        if(index<TIM_SORT_SEGMENT_NUM) 
        {
            edge[++index]=high+1;
            Sort.timMerge(edge, index, a, cmp);
            return true;
        }
        return false;
    }
    @Passed
    public static boolean innerTimSort(int[] a, int low, int high)
    {
        int t;
        int index=0,start,end;
        int[] edge=new int[TIM_SORT_SEGMENT_NUM+1];
        edge[index]=low;
        //find element-ordered segment:each[edge[i], edge[i+1]]
        for(int k=low;k<high&&index<TIM_SORT_SEGMENT_NUM;edge[++index]=++k)
        {
            if(a[k]<=a[k+1]) while(++k<high&&a[k]<=a[k+1]);
            else
            {
                while(++k<high&&a[k]>=a[k+1]);
                for(start=edge[index],end=k;start<end;start++,end--)
                    {t=a[start];a[start]=a[end];a[end]=t;}
            }
        }
        if(index==0) return true;
        if(index<TIM_SORT_SEGMENT_NUM) 
        {
            edge[++index]=high+1;
            Sort.timMerge(edge, index, a);
            return true;
        }
        return false;
    }
    @Passed
    public static boolean innerTimSort(float[] a, int low, int high)
    {
        float t;
        int index=0,start,end;
        int[] edge=new int[TIM_SORT_SEGMENT_NUM+1];
        edge[index]=low;
        //find element-ordered segment:each[edge[i], edge[i+1]]
        for(int k=low;k<high&&index<TIM_SORT_SEGMENT_NUM;edge[++index]=++k)
        {
            if(a[k]<=a[k+1]) while(++k<high&&a[k]<=a[k+1]);
            else
            {
                while(++k<high&&a[k]>=a[k+1]);
                for(start=edge[index],end=k;start<end;start++,end--)
                    {t=a[start];a[start]=a[end];a[end]=t;}
            }
        }
        if(index==0) return true;
        if(index<TIM_SORT_SEGMENT_NUM) 
        {
            edge[++index]=high+1;
            Sort.timMerge(edge, index, a);
            return true;
        }
        return false;
    }
    @Passed
    public static boolean innerTimSort(double[] a, int low, int high)
    {
        double t;
        int index=0,start,end;
        int[] edge=new int[TIM_SORT_SEGMENT_NUM+1];
        edge[index]=low;
        //find element-ordered segment:each[edge[i], edge[i+1]]
        for(int k=low;k<high&&index<TIM_SORT_SEGMENT_NUM;edge[++index]=++k)
        {
            if(a[k]<=a[k+1]) while(++k<high&&a[k]<=a[k+1]);
            else
            {
                while(++k<high&&a[k]>=a[k+1]);
                for(start=edge[index],end=k;start<end;start++,end--)
                    {t=a[start];a[start]=a[end];a[end]=t;}
            }
        }
        if(index==0) return true;
        if(index<TIM_SORT_SEGMENT_NUM) 
        {
            edge[++index]=high+1;
            Sort.timMerge(edge, index, a);
            return true;
        }
        return false;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="HeapSort:BinaryHeap">
    @Passed
    public static void innerHeapSort(int[] a, int low ,int high)
    {
        int p,left, max, t, i;
        for(i=(low+high)>>1;i>=low;i--)//create max-heap
        for(p=i,left=(p<<1)+1;left<=high;left=(p<<1)+1)
        {
            max=(a[p]>=a[left]? p:left);
            if(++left<=high) max=(a[max]>=a[left]? max:left);
            if(max==p) break;
            t=a[max];a[max]=a[p];a[p]=t;
            p=max;
        }
        for(i=high;i>low;)//exchange A[0] and A[i], then max-down().
        {
            t=a[low];a[low]=a[i];a[i]=t;
            --i;
            for(p=0,left=1;left<=i;left=(p<<1)+1)
            {
                max=(a[p]>=a[left]? p:left);
                if(++left<=i) max=(a[max]>=a[left]? max:left);
                if(max==p) break;
                t=a[max];a[max]=a[p];a[p]=t;
                p=max;
            }
        }
    }
    @Passed
    public static void innerHeapSort(float[] a, int low ,int high)
    {
        int p,left, max,i;
        float t;
        for(i=(low+high)>>1;i>=low;i--)//create max-heap
        for(p=i,left=(p<<1)+1;left<=high;left=(p<<1)+1)
        {
            max=(a[p]>=a[left]? p:left);
            if(++left<=high) max=(a[max]>=a[left]? max:left);
            if(max==p) break;
            t=a[max];a[max]=a[p];a[p]=t;
            p=max;
         }
        for(i=high;i>low;)//exchange A[0] and A[i], then max-down().
        {
            t=a[low];a[low]=a[i];a[i]=t;
            --i;
            for(p=0,left=1;left<=i;left=(p<<1)+1)
            {
                max=(a[p]>=a[left]? p:left);
                if(++left<=i) max=(a[max]>=a[left]? max:left);
                if(max==p) break;
                t=a[max];a[max]=a[p];a[p]=t;
                p=max;
            }
        }
    }
    @Passed
    public static void innerHeapSort(double[] a, int low ,int high)
    {
        int p,left, max, i;
        double t;
        for(i=(low+high)>>1;i>=low;i--)//create max-heap
        for(p=i,left=(p<<1)+1;left<=high;left=(p<<1)+1)
        {
            max=(a[p]>=a[left]? p:left);
            if(++left<=high) max=(a[max]>=a[left]? max:left);
            if(max==p) break;
            t=a[max];a[max]=a[p];a[p]=t;
            p=max;
         }
        for(i=high;i>low;)//exchange A[0] and A[i], then max-down().
        {
            t=a[low];a[low]=a[i];a[i]=t;
            --i;
            for(p=0,left=1;left<=i;left=(p<<1)+1)
            {
                max=(a[p]>=a[left]? p:left);
                if(++left<=i) max=(a[max]>=a[left]? max:left);
                if(max==p) break;
                t=a[max];a[max]=a[p];a[p]=t;
                p=max;
            }
        }
    }
    @Passed
    public static void innerHeapSort(Comparable[] a, int low ,int high)
    {
        int p,left, max, i;
        Comparable t;
        for(i=(low+high)>>1;i>=low;i--)//create max-heap
        for(p=i,left=(p<<1)+1;left<=high;left=(p<<1)+1)
        {
            max=(a[p].compareTo(a[left])>=0? p:left);
            if(++left<=high) max=(a[max].compareTo(a[left])>=0? max:left);
            if(max==p) break;
            t=a[max];a[max]=a[p];a[p]=t;
            p=max;
         }
        for(i=high;i>low;)//exchange A[0] and A[i], then max-down().
        {
            t=a[low];a[low]=a[i];a[i]=t;
            --i;
            for(p=0,left=1;left<=i;left=(p<<1)+1)
            {
                max=(a[p].compareTo(a[left])>=0? p:left);
                if(++left<=i) max=(a[max].compareTo(a[left])>=0? max:left);
                if(max==p) break;
                t=a[max];a[max]=a[p];a[p]=t;
                p=max;
            }
        }
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Sorter:Outer-Code">
    //<editor-fold defaultstate="collapased" desc="CountingSort">
    public static boolean countingSort(int[] a)
    {
        if(a==null) throw new NullPointerException();
        return Sort.countingSort(a);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="SelectSort">
    public static void selectSort(Comparable[] a)
    {
        if(a==null) throw new NullPointerException();
        Sort.innerSelectSort(a, 0, a.length-1);
    }
    public static void selectSort(Object[] a, Comparator cmp)
    {
        if(a==null) throw new NullPointerException();
        Sort.innerSelectSort(a, cmp, 0, a.length-1);
    }
    public static void selectSort(int[] a)
    {
        if(a==null) throw new NullPointerException();
        Sort.innerSelectSort(a, 0, a.length-1);
    }
    public static void selectSort(float[] a)
    {
        if(a==null) throw new NullPointerException();
        Sort.innerSelectSort(a, 0, a.length-1);
    }
    public static void selectSort(double[] a)
    {
        Objects.requireNonNull(a);
        Sort.innerSelectSort(a, 0, a.length-1);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="ShellSort">
    public static void shellSort(Comparable[] a)
    {
        if(a==null) throw new NullPointerException();
        Sort.innerShellSort(a, 0, a.length-1);
    }
    public void shellSort(Object[] a, Comparator cmp)
    {
        if(a==null) throw new NullPointerException();
        Sort.innerShellSort(a, cmp, 0, a.length-1);
    }
    public static void shellSort(int[] a)
    {
        if(a==null) throw new NullPointerException();
        Sort.innerShellSort(a, 0, a.length-1);
    }
    public static void shellSort(float[] a)
    {
        if(a==null) throw new NullPointerException();
        Sort.innerShellSort(a, 0, a.length-1);
    }
    public static void shellSort(double[] a)
    {
       if(a==null) throw new NullPointerException();
        Sort.innerShellSort(a, 0, a.length-1);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="QuickSort">
    public static void quickSort(Comparable[] a)
    {
        if(a==null) throw new NullPointerException();
        Sort.innerQuickSort(a, 0, a.length-1);
    }
    public static void quickSort(int[] a)
    {
        if(a==null) throw new NullPointerException();
        Sort.innerQuickSort(a, 0, a.length-1);
    }
    public static void quickSort(float[] a)
    {
        if(a==null) throw new NullPointerException();
        Sort.innerQuickSort(a, 0, a.length-1);
    }
    public static void quickSort(double[] a)
    {
        if(a==null) throw new NullPointerException();
        Sort.innerQuickSort(a, 0, a.length-1);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="QuickSort:ThreeQuickSort">
    public static void threeQuickSort(Comparable[] a)
    {
        if(a==null) throw new NullPointerException();
        Sort.innerThreeQuickSort(a, 0, a.length-1);
    }
    public static void threeQuickSort(int[] a)
    {
        if(a==null) throw new NullPointerException();
        Sort.innerThreeQuickSort(a, 0, a.length-1);
    }
    public static void threeQuickSort(float[] a)
    {
        if(a==null) throw new NullPointerException();
        Sort.innerThreeQuickSort(a, 0, a.length-1);
    }
    public static void threeQuickSort(double[] a)
    {
        if(a==null) throw new NullPointerException();
        Sort.innerThreeQuickSort(a, 0, a.length-1);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="QuickSort:DualPivotQuickSort">
    public static void dualPivotQuickSort(Comparable[] a)
    {
        if(a==null) throw new NullPointerException();
        Sort.innerDualPivotQuickSort(a, 0, a.length-1);
    }
    public static void dualPivotQuickSort(Object[] a, Comparator cmp)
    {
        if(a==null) throw new NullPointerException();
        Sort.innerDualPivotQuickSort(a, cmp, 0, a.length-1);
    }
    public static void dualPivotQuickSort(int[] a)
    {
        if(a==null) throw new NullPointerException();
        Sort.innerDualPivotQuickSort(a, 0, a.length-1);
    }
    public static void dualPivotQuickSort(float[] a)
    {
       if(a==null) throw new NullPointerException();
        Sort.innerDualPivotQuickSort(a, 0, a.length-1);
    }
    public static void dualPivotQuickSort(double[] a)
    {
       if(a==null) throw new NullPointerException();
        Sort.innerDualPivotQuickSort(a, 0, a.length-1);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="MergeSort">
    public static void mergeSort(Comparable[] a)
    {
        if(a==null) throw new NullPointerException();
        Sort.innerMergeSort(a, 0, a.length-1);
    }
    public static void mergeSort(int[] a)
    {
        if(a==null) throw new NullPointerException();
        Sort.innerMergeSort(a, 0, a.length-1);
    }
    public static void mergeSort(float[] a)
    {
        if(a==null) throw new NullPointerException();
        Sort.innerMergeSort(a, 0, a.length-1);
    }
    public static void mergeSort(double[] a)
    {
        if(a==null) throw new NullPointerException();
        Sort.innerMergeSort(a, 0, a.length-1);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="MergeSort:TimSort">
    public static boolean timSort(Comparable[] a)
    {
        if(a==null) throw new NullPointerException();
        return Sort.innerTimSort(a, 0, a.length-1);
    }
    public static boolean timSort(Object[] a, Comparator cmp)
    {
        if(a==null) throw new NullPointerException();
        return Sort.innerTimSort(a, cmp, 0, a.length-1);
    }
    public static boolean timSort(int[] a)
    {
        if(a==null) throw new NullPointerException();
        return Sort.innerTimSort(a, 0, a.length-1);
    }
    public static boolean timSort(float[] a)
    {
        if(a==null) throw new NullPointerException();
        return Sort.innerTimSort(a, 0, a.length-1);
    }
    public static boolean timSort(double[] a)
    {
        if(a==null) throw new NullPointerException();
        return Sort.innerTimSort(a, 0, a.length-1);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="HeapSort:BinaryHeap">
    public static void heapSort(int[] a)
    {
        if(a==null) throw new NullPointerException();
        Sort.innerHeapSort(a, 0, a.length-1);
    }
    public static void heapSort(float[] a)
    {
        if(a==null) throw new NullPointerException();
        Sort.innerHeapSort(a, 0, a.length-1);
    }
    public static void heapSort(double[] a)
    {
        if(a==null) throw new NullPointerException();
        Sort.innerHeapSort(a, 0, a.length-1);
    }
    public static void heapSort(Comparable[] a)
    {
        if(a==null) throw new NullPointerException();
        Sort.innerHeapSort(a, 0, a.length-1);
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Integration-For-Sorters:Inner-Code">
    private static final ExecutorService exec=Executors.newFixedThreadPool(15);
    private static final BinarySemaphore mutex=new BinarySemaphore();
    
    public static void shutDown()
    {
        mutex.P();
        if(!exec.isShutdown()) exec.shutdown();
        mutex.V();
    }
    //<editor-fold defaultstate="collapsed" desc="InnerSort-Comparable">
    @Passed
    private static void innerSortM(Comparable[] a, int low, int high)
    {
        //skip the redundant check of single thread sort
        int len=high-low+1;
        ExRandom ran=Lang.exRandom();
        int index=ran.nextInt(low, high);
        Comparable t=a[low];a[low]=a[index];a[index]=t;
        index=ran.nextInt(low,high);
        t=a[high];a[high]=a[index];a[index]=t;
        
        long p=Vector.dualPivotPartition(a, low, high);
        int p0=(int) (p&0x000000ffffffffL),p1=(int) (p>>32);
        
        if(len<=SINGLE_THREAD_THRESHOLD)
        {
            Sort.innerSort(a, low, p0-1);
            Sort.innerSort(a, p0+1, p1-1);
            Sort.innerSort(a, p1+1, high);
            return;
        }
        if(len<=MULTI_THREAD_LEAF)//the leaf of multi-thread sorting Tree
        {
            Lock ss=new Lock(3);
            exec.execute(() -> {Sort.innerSort(a, low, p0-1);ss.unlock();});
            exec.execute(() -> {Sort.innerSort(a, p0+1, p1-1);ss.unlock();});
            exec.execute(() -> {Sort.innerSort(a, p1+1, high);ss.unlock();});
            ss.lock();
            return;
        }
        //the parent node of multi-thread sorting Tree, you must create child nodes asynchronously
        Lock ss=new Lock(3);
        exec.execute(() ->{Sort.innerSortM(a, low, p0-1);ss.unlock();});
        exec.execute(() ->{Sort.innerSortM(a, p0+1, p1-1);ss.unlock();});
        exec.execute(() ->{Sort.innerSortM(a, p1+1, high);ss.unlock();});
        ss.lock();
    }
    @Passed
    public static void innerSort(Comparable[] a, int low, int high)
    {
        int len=high-low+1;
        if(len<=1) return;
        if(len<=SELECT_SORT_THRESHOLD) {Sort.innerSelectSort(a, low, high);return;}//checked
        if(len<=SHELL_SORT_THRESHOLD) {Sort.innerShellSort(a, low, high);return;}//checked
        if(len<=QUICK_SORT_THRESHOLD) 
        {
            ExRandom ran=Lang.exRandom();
            int index=ran.nextInt(low, high);
            Comparable t=a[low];a[low]=a[index];a[index]=t;
            index=ran.nextInt(low,high);
            t=a[high];a[high]=a[index];a[index]=t;
        
            long p=Vector.dualPivotPartition(a, low, high);
            int p0=(int) (p&0x000000ffffffffL),p1=(int) (p>>32);
            Sort.innerSort(a, low, p0-1);
            Sort.innerSort(a, p0+1, p1-1);
            Sort.innerSort(a, p1+1, high);
            return;
        }
        if(Sort.innerTimSort(a, low, high)) return;
        
        if(len>MULTI_THREAD_LEAF) {Sort.innerSortM(a, low, high);return;}
       
        ExRandom ran=Lang.exRandom();
        int index=ran.nextInt(low, high);
        Comparable t=a[low];a[low]=a[index];a[index]=t;
        index=ran.nextInt(low,high);
        t=a[high];a[high]=a[index];a[index]=t;
        
        long p=Vector.dualPivotPartition(a, low, high);
        int p0=(int) (p&0x000000ffffffffL),p1=(int) (p>>32);
        Sort.innerSort(a, low, p0-1);
        Sort.innerSort(a, p0+1, p1-1);
        Sort.innerSort(a, p1+1, high);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="InnerSort-Object&Comparator">
    @Passed
    private static void innerSortM(Object[] a, Comparator cmp, int low, int high)
    {
        //skip the redundant check of single thread sort
        int len=high-low+1;
        ExRandom ran=Lang.exRandom();
        int index=ran.nextInt(low, high);
        Object t=a[low];a[low]=a[index];a[index]=t;
        index=ran.nextInt(low,high);
        t=a[high];a[high]=a[index];a[index]=t;
        
        long p=Vector.dualPivotPartition(a, cmp, low, high);
        int p0=(int) (p&0x000000ffffffffL),p1=(int) (p>>32);
        
        if(len<=SINGLE_THREAD_THRESHOLD)
        {
            Sort.innerSort(a, cmp, low, p0-1);
            Sort.innerSort(a, cmp, p0+1, p1-1);
            Sort.innerSort(a, cmp, p1+1, high);
            return;
        }
        if(len<=MULTI_THREAD_LEAF)//the leaf of multi-thread sorting Tree
        {
            Lock ss=new Lock(3);
            exec.execute(() -> {Sort.innerSort(a, cmp, low, p0-1);ss.unlock();});
            exec.execute(() -> {Sort.innerSort(a, cmp, p0+1, p1-1);ss.unlock();});
            exec.execute(() -> {Sort.innerSort(a, cmp, p1+1, high);ss.unlock();});
            ss.lock();
            return;
        }
        //the parent node of multi-thread sorting Tree, you must create child nodes asynchronously
        Lock ss=new Lock(3);
        exec.execute(() ->{Sort.innerSortM(a, cmp, low, p0-1);ss.unlock();});
        exec.execute(() ->{Sort.innerSortM(a, cmp, p0+1, p1-1);ss.unlock();});
        exec.execute(() ->{Sort.innerSortM(a, cmp, p1+1, high);ss.unlock();});
        ss.lock();
    }
    @Passed
    public static void innerSort(Object[] a, Comparator cmp, int low, int high)
    {
        int len=high-low+1;
        if(len<=1) return;
        if(len<=SELECT_SORT_THRESHOLD) {Sort.innerSelectSort(a, cmp, low, high);return;}//checked
        if(len<=SHELL_SORT_THRESHOLD) {Sort.innerShellSort(a, cmp, low, high);return;}//checked
        if(len<=QUICK_SORT_THRESHOLD) 
        {
            ExRandom ran=Lang.exRandom();
            int index=ran.nextInt(low, high);
            Object t=a[low];a[low]=a[index];a[index]=t;
            index=ran.nextInt(low,high);
            t=a[high];a[high]=a[index];a[index]=t;
        
            long p=Vector.dualPivotPartition(a, cmp, low, high);
            int p0=(int) (p&0x000000ffffffffL),p1=(int) (p>>32);
            Sort.innerSort(a, cmp, low, p0-1);
            Sort.innerSort(a, cmp, p0+1, p1-1);
            Sort.innerSort(a, cmp, p1+1, high);
            return;
        }
        if(Sort.innerTimSort(a, cmp, low, high)) return;
        
         if(len>MULTI_THREAD_LEAF) {Sort.innerSortM(a, cmp, low, high);return;}
       
        ExRandom ran=Lang.exRandom();
        int index=ran.nextInt(low, high);
        Object t=a[low];a[low]=a[index];a[index]=t;
        index=ran.nextInt(low,high);
        t=a[high];a[high]=a[index];a[index]=t;
        
        long p=Vector.dualPivotPartition(a, cmp, low, high);
        int p0=(int) (p&0x000000ffffffffL),p1=(int) (p>>32);
      
        Sort.innerSort(a, cmp, low, p0-1);
        Sort.innerSort(a, cmp, p0+1, p1-1);
        Sort.innerSort(a, cmp, p1+1, high);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="InnerSort-Int">
     @Passed
    private static void innerSortM(int[] a, int low, int high)
    {
        //skip the redundant check of single thread sort
        int len=high-low+1;
        ExRandom ran=Lang.exRandom();
        int index=ran.nextInt(low, high);
        int t=a[low];a[low]=a[index];a[index]=t;
        index=ran.nextInt(low,high);
        t=a[high];a[high]=a[index];a[index]=t;
        
        long p=Vector.dualPivotPartition(a, low, high);
        int p0=(int) (p&0x000000ffffffffL),p1=(int) (p>>32);
        
        if(len<=SINGLE_THREAD_THRESHOLD)
        {
            Sort.innerSort(a, low, p0-1);
            Sort.innerSort(a, p0+1, p1-1);
            Sort.innerSort(a, p1+1, high);
            return;
        }
        if(len<=MULTI_THREAD_LEAF)//the leaf of multi-thread sorting Tree
        {
            Lock ss=new Lock(3);
            exec.execute(() -> {Sort.innerSort(a, low, p0-1);ss.unlock();});
            exec.execute(() -> {Sort.innerSort(a, p0+1, p1-1);ss.unlock();});
            exec.execute(() -> {Sort.innerSort(a, p1+1, high);ss.unlock();});
            ss.lock();
            return;
        }
        //the parent node of multi-thread sorting Tree, you must create child nodes asynchronously
        Lock ss=new Lock(3);
        exec.execute(() ->{Sort.innerSortM(a, low, p0-1);ss.unlock();});
        exec.execute(() ->{Sort.innerSortM(a, p0+1, p1-1);ss.unlock();});
        exec.execute(() ->{Sort.innerSortM(a, p1+1, high);ss.unlock();});
        ss.lock();
    }
    @Passed
    public static void innerSort(int[] a, int low, int high)
    {
        int len=high-low+1;
        if(len<=1) return;
        if(len<=SELECT_SORT_THRESHOLD) {Sort.innerSelectSort(a, low, high);return;}//checked
        if(len<=SHELL_SORT_THRESHOLD) {Sort.innerShellSort(a, low, high);return;}//checked
        if(len<=QUICK_SORT_THRESHOLD) 
        {
            ExRandom ran=Lang.exRandom();
            int index=ran.nextInt(low, high);
            int t=a[low];a[low]=a[index];a[index]=t;
            index=ran.nextInt(low,high);
            t=a[high];a[high]=a[index];a[index]=t;
        
            long p=Vector.dualPivotPartition(a, low, high);
            int p0=(int) (p&0x000000ffffffffL),p1=(int) (p>>32);
            Sort.innerSort(a, low, p0-1);
            Sort.innerSort(a, p0+1, p1-1);
            Sort.innerSort(a, p1+1, high);
            return;
        }
        if(Sort.innerTimSort(a, low, high)) return;
        if(Sort.innerCountingSort(a, low, high)) return;
        
        if(len>MULTI_THREAD_LEAF) {Sort.innerSortM(a, low, high);return;}
        
        ExRandom ran=Lang.exRandom();
        int index=ran.nextInt(low, high);
        int t=a[low];a[low]=a[index];a[index]=t;
        index=ran.nextInt(low,high);
        t=a[high];a[high]=a[index];a[index]=t;
        
        long p=Vector.dualPivotPartition(a, low, high);
        int p0=(int) (p&0x000000ffffffffL),p1=(int) (p>>32);
        Sort.innerSort(a, low, p0-1);
        Sort.innerSort(a, p0+1, p1-1);
        Sort.innerSort(a, p1+1, high);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="InnerSort-Float">
    @Passed
    private static void innerSortM(float[] a, int low, int high)
    {
        //skip the redundant check of single thread sort
        int len=high-low+1;
        ExRandom ran=Lang.exRandom();
        int index=ran.nextInt(low, high);
        float t=a[low];a[low]=a[index];a[index]=t;
        index=ran.nextInt(low,high);
        t=a[high];a[high]=a[index];a[index]=t;
        
        long p=Vector.dualPivotPartition(a, low, high);
        int p0=(int) (p&0x000000ffffffffL),p1=(int) (p>>32);
        
        if(len<=SINGLE_THREAD_THRESHOLD)
        {
            Sort.innerSort(a, low, p0-1);
            Sort.innerSort(a, p0+1, p1-1);
            Sort.innerSort(a, p1+1, high);
            return;
        }
        if(len<=MULTI_THREAD_LEAF)//the leaf of multi-thread sorting Tree
        {
            Lock ss=new Lock(3);
            exec.execute(() -> {Sort.innerSort(a, low, p0-1);ss.unlock();});
            exec.execute(() -> {Sort.innerSort(a, p0+1, p1-1);ss.unlock();});
            exec.execute(() -> {Sort.innerSort(a, p1+1, high);ss.unlock();});
            ss.lock();
            return;
        }
        //the parent node of multi-thread sorting Tree, you must create child nodes asynchronously
        Lock ss=new Lock(3);
        exec.execute(() ->{Sort.innerSortM(a, low, p0-1);ss.unlock();});
        exec.execute(() ->{Sort.innerSortM(a, p0+1, p1-1);ss.unlock();});
        exec.execute(() ->{Sort.innerSortM(a, p1+1, high);ss.unlock();});
        ss.lock();
    }
    @Passed
    private static void innerSort(float[] a, int low, int high)
    {
        int len=high-low+1;
        if(len<=1) return;
        if(len<=SELECT_SORT_THRESHOLD) {Sort.innerSelectSort(a, low, high);return;}//checked
        if(len<=SHELL_SORT_THRESHOLD) {Sort.innerShellSort(a, low, high);return;}//checked
        if(len<=QUICK_SORT_THRESHOLD) 
        {
            ExRandom ran=Lang.exRandom();
            int index=ran.nextInt(low, high);
            float t=a[low];a[low]=a[index];a[index]=t;
            index=ran.nextInt(low,high);
            t=a[high];a[high]=a[index];a[index]=t;
        
            long p=Vector.dualPivotPartition(a, low, high);
            int p0=(int) (p&0x000000ffffffffL),p1=(int) (p>>32);
            Sort.innerSort(a, low, p0-1);
            Sort.innerSort(a, p0+1, p1-1);
            Sort.innerSort(a, p1+1, high);
            return;
        }
        if(Sort.innerTimSort(a, low, high)) return;
        
        if(len>MULTI_THREAD_LEAF) {Sort.innerSortM(a, low, high);return;}
       
        ExRandom ran=Lang.exRandom();
        int index=ran.nextInt(low, high);
        float t=a[low];a[low]=a[index];a[index]=t;
        index=ran.nextInt(low,high);
        t=a[high];a[high]=a[index];a[index]=t;
        
        long p=Vector.dualPivotPartition(a, low, high);
        int p0=(int) (p&0x000000ffffffffL),p1=(int) (p>>32);
        
        Sort.innerSort(a, low, p0-1);
        Sort.innerSort(a, p0+1, p1-1);
        Sort.innerSort(a, p1+1, high);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="InnerSort-Double">
    @Passed
    private static void innerSortM(double[] a, int low, int high)
    {
        //skip the redundant check of single thread sort
        int len=high-low+1;
        ExRandom ran=Lang.exRandom();
        int index=ran.nextInt(low, high);
        double t=a[low];a[low]=a[index];a[index]=t;
        index=ran.nextInt(low,high);
        t=a[high];a[high]=a[index];a[index]=t;
        
        long p=Vector.dualPivotPartition(a, low, high);
        int p0=(int) (p&0x000000ffffffffL),p1=(int) (p>>32);
        
        if(len<=SINGLE_THREAD_THRESHOLD)
        {
            Sort.innerSort(a, low, p0-1);
            Sort.innerSort(a, p0+1, p1-1);
            Sort.innerSort(a, p1+1, high);
            return;
        }
        if(len<=MULTI_THREAD_LEAF)//the leaf of multi-thread sorting Tree
        {
            Lock ss=new Lock(3);
            exec.execute(() -> {Sort.innerSort(a, low, p0-1);ss.unlock();});
            exec.execute(() -> {Sort.innerSort(a, p0+1, p1-1);ss.unlock();});
            exec.execute(() -> {Sort.innerSort(a, p1+1, high);ss.unlock();});
            ss.lock();
            return;
        }
        //the parent node of multi-thread sorting Tree, you must create child nodes asynchronously
        Lock ss=new Lock(3);
        exec.execute(() ->{Sort.innerSortM(a, low, p0-1);ss.unlock();});
        exec.execute(() ->{Sort.innerSortM(a, p0+1, p1-1);ss.unlock();});
        exec.execute(() ->{Sort.innerSortM(a, p1+1, high);ss.unlock();});
        ss.lock();
    }
    @Passed
    private static void innerSort(double[] a, int low, int high)
    {
        int len=high-low+1;
        if(len<=1) return;
        if(len<=SELECT_SORT_THRESHOLD) {Sort.innerSelectSort(a, low, high);return;}//checked
        if(len<=SHELL_SORT_THRESHOLD) {Sort.innerShellSort(a, low, high);return;}//checked
        if(len<=QUICK_SORT_THRESHOLD) 
        {
            ExRandom ran=Lang.exRandom();
            int index=ran.nextInt(low, high);
            double t=a[low];a[low]=a[index];a[index]=t;
            index=ran.nextInt(low,high);
            t=a[high];a[high]=a[index];a[index]=t;
        
            long p=Vector.dualPivotPartition(a, low, high);
            int p0=(int) (p&0x000000ffffffffL),p1=(int) (p>>32);
            Sort.innerSort(a, low, p0-1);
            Sort.innerSort(a, p0+1, p1-1);
            Sort.innerSort(a, p1+1, high);
            return;
        }
        if(Sort.innerTimSort(a, low, high)) return;
        
        if(len>MULTI_THREAD_LEAF) {Sort.innerSortM(a, low, high);return;}
        
        ExRandom ran=Lang.exRandom();
        int index=ran.nextInt(low, high);
        double t=a[low];a[low]=a[index];a[index]=t;
        index=ran.nextInt(low,high);
        t=a[high];a[high]=a[index];a[index]=t;
        
        long p=Vector.dualPivotPartition(a, low, high);
        int p0=(int) (p&0x000000ffffffffL),p1=(int) (p>>32);
        Sort.innerSort(a, low, p0-1);
        Sort.innerSort(a, p0+1, p1-1);
        Sort.innerSort(a, p1+1, high);
        return;
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Integration-For-Sorters:Outer-Code">
    public static void sort(Comparable[] a)
    {
        if(a==null) throw new NullPointerException();
        Sort.innerSort(a, 0, a.length-1);
    }
    public static void sort(Comparable[] a, int low, int high)
    {
        if(a==null) throw new NullPointerException();
        Sort.innerSort(a, low, high);
    }
    public static void sort(Object[] a, Comparator cmp)
    {
        if(a==null) throw new NullPointerException();
        Sort.innerSort(a, cmp, 0, a.length-1);
    }
    public static void sort(int[] a)
    {
        if(a==null) throw new NullPointerException();
        Sort.innerSort(a, 0, a.length-1);
    }
    public static void sort(int[] a, int low, int high)
    {
        if(a==null) throw new NullPointerException();
        Sort.innerSort(a, low, high);
    }
    public static void sort(float[] a)
    {
        if(a==null) throw new NullPointerException();
        Sort.innerSort(a, 0, a.length-1);
    }
    public static void sort(float[] a, int low, int high)
    {
        if(a==null) throw new NullPointerException();
        Sort.innerSort(a, low, high);
    }
    public static void sort(double[] a)
    {
        if(a==null) throw new NullPointerException();
        Sort.innerSort(a, 0, a.length-1);
    }
    public static void sort(double[] a, int low, int high)
    {
        if(a==null) throw new NullPointerException();
        Sort.innerSort(a, low, high);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Find a element in an Array sorted in asscending order">
    public static int biFind(int[] arr, int val)
    {
        return Sort.biFind(arr, val, 0, arr.length-1);
    }
    @Passed
    public static int biFind(int[] arr, int val, int low, int high)
    {
        int mid;
        while(low<=high)
        {
            mid=(low+high)>>1;
            if(val<arr[mid]) high=mid-1;
            else if(val>arr[mid]) low=mid+1;
            else return mid;
        }
        return -1;
    }
    public static int threeFind(int[] arr, int val)
    {
        return Sort.threeFind(arr, val, 0, arr.length-1);
    }
    @Passed
    public static int threeFind(int[] arr, int val, int low, int high)
    {
        int p1,p2,d;
        while(low<=high)
        {
            d=(high-low+1)/3;
            p1=d+low;p2=p1+d;
            if(val<arr[p1]) high=p1-1;
            else if(val>arr[p2]) low=p2+1;
            else if(val>arr[p1]&&val<arr[p2]){low=p1+1;high=p2-1;}
            else if(val==arr[p1]) return p1;
            else return p2;
        }
        return -1;
    }
    //</editor-fold>
}
