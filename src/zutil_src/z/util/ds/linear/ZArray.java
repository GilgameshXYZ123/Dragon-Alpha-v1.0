/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.ds.linear;

import java.lang.reflect.Array;
import java.util.Queue;
import java.util.function.Predicate;
import java.util.function.BiPredicate;
import z.util.ds.imp.ZStack;
import z.util.lang.annotation.Passed;

/**
 *
 * @author dell
 * @param <T>
 */
public abstract class ZArray<T> extends ZLinear<T> implements Queue<T>, ZStack<T>
{
    protected static final Object[] EMPTY_DATA=new Object[]{};//default empty data
    protected Object[] data;
    protected int num;
    //<editor-fold defaultstate="collapsed" desc="Basic-Function"> 
    @Override
    public int number() 
    {
        return num;
    }
    @Override
    public boolean isEmpty() 
    {
        return num==0||size==0||data==EMPTY_DATA;
    }  
    @Override
    public String toString()
    {
        if(this.isEmpty()) return "";
        StringBuilder sb=new StringBuilder();
        sb.append('{').append(data[0]);
        for(int i=1;i<num;i++) sb.append(", ").append(data[i]);
        sb.append('}');
        return sb.toString();
    }
    public Object[] getData()
    {
        return data;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Inner-Code:Remove">
    /**
     * {@link z.util.math.Vector#removeNull(Object[], int, int) }.
     */
    @Passed
    protected void removeNull()//checked
    {
        int end,start,nstart;
        //find the first element block in the list that's not null
        for(end=num-1;end>=0&&data[end]==null;end--);
        //looped block move to take place of null block
        while(end>0)//if end==0, means there is no null element
        {
            //find the block
            for(start=end-1;start>=0&&data[start]!=null;start--) ;
            if(start<0) break;//beyond the range of the arrary

            //find the null block
            for(nstart=start-1;nstart>=0&&data[nstart]==null;nstart--); 
            
            //move block
            System.arraycopy(data, start+1, data, 1+nstart, end-=start);
            end+=nstart;
        }
        num=end+1;
    }
    /**
     * {@link z.util.math.Vector#removeValue(Object[], Object, int, int) }.
     * @param value 
     */
    @Passed
    protected void removeValue(Object value)
    {
        int end,start,nstart;
        //find the first element block in the list that!=value
        for(end=num-1;end>=0&&value.equals(data[end]);end--);
        //looped block move to take place of equal block
        while(end>0)//if end==0, means there is no null element
        {
            for(start=end-1;start>=0&&!value.equals(data[start]);start--);
            if(start<0) break;//beyond the range of the arrary
           
            //find the equal block
            for(nstart=start-1;nstart>=0&&value.equals(data[nstart]);nstart--);
            
            //move block
           System.arraycopy(data, start+1, data, 1+nstart, end-=start);
           end+=nstart;
        }
        num=end+1;
    }
    /**
     * {@link z.util.math.Vector#removeValue(Object[], Predicate, int, int) }.
     * @param pre 
     */
    @Passed
    protected void removeValue(Predicate pre)
    {
        int end,start,nstart;
        //find the first element doesn't meet the condition
        for(end=num-1;end>=0&&pre.test(data[end]);end--);
        //looped block move to take place of satisfied block
        while(end>0)//if end==0, means there is no null element
        {
            for(start=end-1;start>=0&&!pre.test(data[start]);start--);
            if(start<0) break;//beyond the range of the arrary
           
            //find the equal block
            for(nstart=start-1;nstart>=0&&pre.test(data[nstart]);nstart--);
            
            //move block
            System.arraycopy(data, start+1, data, 1+nstart, end-=start);
            end+=nstart;
        }
        num=end+1;
    }
    /**
     * {@link z.util.math.Vector#removeValue(Object[], BiPredicate, Object, int, int) }.
     * @param pre
     * @param condition 
     */
    @Passed
    protected void removeValue(BiPredicate pre, Object condition)
    {
        int end,start,nstart;
        //find the first element doesn't meet the condition
        for(end=num-1;end>=0&&pre.test(data[end], condition);end--);
        //looped block move to take place of satisfied block
        while(end>0)//if end==0, means there is no null element
        {
            for(start=end-1;start>=0&&!pre.test(data[start], condition);start--);
            if(start<0) break;//beyond the range of the arrary
           
            //find the equal block
            for(nstart=start-1;nstart>=0&&pre.test(data[nstart], condition);nstart--);
            
            //move block
            System.arraycopy(data, start+1, data, 1+nstart, end-=start);
            end+=nstart;
        }
        num=end+1;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapased" desc="ZLinear">
    /**
     * use the last element as the guard.
     * @param o
     * @return if any element in the list equals(o)
     * @Checked
     */
    @Override
    public boolean contains(Object o)//checked
    {
        if(this.isEmpty()) return false;
        Object ov=data[num-1];
        data[num-1]=o;
        int index=0;
        if(o==null)
        {
            if(o==ov) return true;
            while(data[index]!=null) index++;
        }
        else
        {
            if(o.equals(ov)) return true;
            while(!o.equals(data[index])) index++;
        }
        data[num-1]=ov;
        return index!=num-1;
    }
    @Override
    public Object[] toArray()//checked
    {
        Object[] arr=new Object[num];
        System.arraycopy(data, 0, arr, 0, num);
        return arr;
    }
    @Override
    public <T> T[] toArray(T[] a)//checked
    {
        if(a.length<num)
            a=(T[]) Array.newInstance(a.getClass().getComponentType(), num);
        System.arraycopy(data, 0, a, 0, num);
        if(a.length>num) a[num]=null;
        return a;
    }
    @Override
    public void clear()//checked
    {
        for(int i=0;i<num;i++) data[i]=null;
        size=num=0;
        data=EMPTY_DATA;
    }
    //</editor-fold>
}
