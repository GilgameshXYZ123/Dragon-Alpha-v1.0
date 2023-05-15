/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.ds.heap;

import static java.lang.Math.log;
import java.util.Collection;
import java.util.function.BiPredicate;
import java.util.function.Predicate;
import static z.util.ds.Constants.Array.DEF_EXPEND_RATE;
import static z.util.ds.Constants.Hash.DEF_INIT_SIZE;
import z.util.ds.imp.Indexable;
import z.util.ds.imp.Numberable;
import z.util.ds.imp.ZHeap;
import z.util.ds.imp.ZStack;
import z.util.ds.linear.ZArrayList;
import z.util.ds.linear.ZLinkedList;
import z.util.ds.linear.ZLinkedStack;
import z.util.lang.annotation.Passed;
import z.util.lang.exception.UOE;

/**
 * <pre>
 * An implementation of Binary Heap, don't Support null value.
 * (1)index=(num)/2;
 *  data[index] is the first leaf node
 *  data[index-1] is the last parent node.
 * </pre>
 * @author dell
 * @param <T>
 */
@Passed
public class ZBinaryHeap<T extends Comparable> extends ZArrayList<T> implements ZHeap<T>
{
    //constants------------------------------------------------------------------
    private static final int LINEAR_FIND_THRESHOLD=16;
    
    //constructors--------------------------------------------------------------
    public ZBinaryHeap(int size, double expendRate)
    {
        super(size, expendRate);
    }
    public ZBinaryHeap(int size)
    {
        super(size, DEF_EXPEND_RATE);
    }
    public ZBinaryHeap()
    {
        super(DEF_INIT_SIZE, DEF_EXPEND_RATE);
    }
    //<editor-fold defaultstate="collapsed" desc="Basic-Function">
    @Override
    public int getIndexType()
    {
        return Indexable.HEAP;
    }
    //<editor-fold defaultstate="collapsed" desc="Inner-Operator">
    /**
     * before using this method, you may need to exchange a[0] with 
     * a[num-1], to remove the root node of this heap.
     * @param low the index of the root node of a sub heap
     * @param high the index of the last leaf node of a sub heap
     */
    @Passed
    private void maxDown(int low, int high)
    {
        Object t;
        for(int p=low,c=(p<<1)+1,min;c<=high;p=min,c=(p<<1)+1)
        {
            min=(((Comparable) data[p]).compareTo(data[c])<=0? p:c);//compare parent to the left
            if(++c<=high) min=(((Comparable) data[min]).compareTo(data[c])<=0? min:c);
            if(min==p) return;
            t=data[min];data[min]=data[p];data[p]=t;
        }
    }
    /**
     * the num-increase and expend option has been done before calling
     * this method. 
     * @param low the index of the last leaf node of a sub heap
     * @param high the index of the root node of a sub heap
     */
    @Passed
    private void minUp(int low, int high)
    {
        Object t;
        for(int c=low,p=((c+1)>>1)-1;
                p>=high&&((Comparable) data[c]).compareTo(data[p])<0;
                c=p,p=((c+1)>>1)-1)
            {t=data[c];data[c]=data[p];data[p]=t;}
    }
    /**
     * this method can be complete in a O(n) time complexity.
     * @param high the index of the root node of a sub heap
     */
    private void createHeap(int low, int high)
    {
        int p, left, min;
        Object t;
        for(int i=(low+high)>>1;i>=low;i--)
        for(p=i,left=(p<<1)+1;left<=high;p=min,left=(p<<1)+1)
        {
            min=(((Comparable)data[p]).compareTo(data[left])<=0? p:left);
            if(++left<=high) min=(((Comparable)data[min]).compareTo(data[left])<=0? min:left);
            if(min==p) break;
            t=data[min];data[min]=data[p];data[p]=t;
        }
    }
    @Override
    protected boolean innerRemoveAll(Predicate pre)
    {
        boolean r=super.innerRemoveAll(pre);//num and size has been decreased
        this.createHeap(0, num-1);
        return r;
    }
    @Override
    protected boolean innerRemoveAll(BiPredicate pre, Object condition)
    {
        boolean r=super.innerRemoveAll(pre, condition);//num and size has been decreased
        this.createHeap(0, num-1);
        return r;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Outer-Operator">
    @Override
    public boolean add(T e)//passed
    {
        if(e==null) return false;
        boolean r=super.add(e);
        this.minUp(num-1, 0);
        return r;
    }
    @Override
    public void push(T e)//passed
    {
        if(e==null) return;
        super.push(e);
        this.minUp(num-1, 0);
    }
    @Override
    public boolean addAll(Collection<? extends T> c)//[assed
    {
        if(c==null) throw new NullPointerException();
        
        int need=0; 
        for(T o:c) if(o!=null) need++;
        if(need==0) return false;
        
        if(num+need>size) this.expend(need,num);
        
        int n2=(c instanceof Numberable? ((Numberable)c).number():c.size());
        double x=(double)num/n2;
        if(log(1+x)+1>x*(2-log(num+n2)))//use the algorithm with lower complexity
        {
            for(T o:c) if(o!=null) data[num++]=o;
            this.createHeap(0, num-1);
            return true;
        }
        for(T o:c) if(o!=null) {data[num]=o;this.minUp(num++, 0);}
        return true;
    }
    @Override
    public T remove()//passed
    {
        if(this.isEmpty()) return null;
        Object t=data[num-1];data[num-1]=data[0];data[0]=t;
        this.maxDown(0, num-2);
        return this.pop();//as we have exchanged data[num-1] and data[0]
    }
    @Override
    public boolean remove(Object o)//passed
    {
        boolean r=super.remove(o);
        this.createHeap(0, num-1);
        return r;
    }
    @Override
    public T remove(int index)//passed
    {
        if(this.isEmpty()) return null;
        Object t=data[num-1];data[num-1]=data[index];data[index]=t;
        this.maxDown(index, num-2);
        return this.pop();//as we have exchanged data[num-1] and data[0]
    }
    @Override
    public T findMin()
    {
        return (this.isEmpty()? null:(T) data[0]);
    }
    @Override
    public T removeMin()
    {
        return this.remove();
    }
    @Override
    public boolean contains(Object o)//passed
    {
        if(num<LINEAR_FIND_THRESHOLD) return super.contains(o);
        if(o==null) return false;
        Comparable key=(Comparable) o;
        ZLinkedStack<Integer> s=new ZLinkedStack<>();
        s.push(0);
        int v,p,c;
        while(!s.isEmpty())
        {
            v=key.compareTo(data[p=s.pop()]);
            if(v<0) {}//less than the parent, no need to check the children
            else if(v>0)
            {
                if((c=(p<<1)+2)<num) s.push(c);//push right first
                if(--c<num) s.push(c);
            }
            else return true;
        }
        return false;
    }
    /**
     * <pre>
     * We can use the character of heap to find a value quicker, when 
     * testing, it actually worker fine.
     * The algorithm works like this:
     * (1) if a key is less than the parent, there is no need to check the children
     * (2) else check its child
     * For the time complexity: 
     *      for S>2, we have: O1(n)=(1-1/n)*O(n/2)+K
     * as the average time complexity for linear search is: O2(n/2)
     * when O1 less than O2, we have:
     * </pre>
     * @param o
     * @return 
     */
    @Override
    public int indexOf(Object o)
    {
        if(num<LINEAR_FIND_THRESHOLD) return super.indexOf(o);
        if(o==null) return -1;
        Comparable key=(Comparable) o;
        ZStack<Integer> s=new ZLinkedStack<>();
        s.push(0);
        int v,p,c;
        while(!s.isEmpty())
        {
            v=key.compareTo(data[p=s.pop()]);
            if(v<0) {}//less than the parent, no need to check the children
            else if(v>0)
            {
                if((c=(p<<1)+2)<num) s.push(c);//push right first
                if(--c<num) s.push(c);
            }
            else return p;
        }
        return -1;
    }
    /**
     * We can use the character of heap to find a value quicker.
     * The algorithm works like this:
     * (1) if a key is greater than the child, no need to check the parent
     * (2)only check left child, don't check the right, as they have the same parent
     * @param o
     * @return 
     */
    @Override
    public int lastIndexOf(Object o)//passed
    {
        if(num<LINEAR_FIND_THRESHOLD) return super.lastIndexOf(o);
        ZLinkedList<Integer> l=new ZLinkedList<>();
        for(int i=num-1,end=num>>1;i>=end;i--) l.push(i);//add all leaves from big to small in index
        
        Comparable key=(Comparable) o;
        int c,v;
        while(!l.isEmpty())
        {
           
            v=key.compareTo(data[ c=l.remove()]);
            if(v>0){}//greater than the child, no need to check the parent
            else if(v<0)
            {
                //only check left child, don't check the right, as they have the same parent
                if((c&1)==1) l.add(((c+1)>>1)-1);
            }
            else return c;
        }
        return -1;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Banned-Super-Operator">
    @Override
    public void add(int index, T element)
    {
        throw new UOE("You can't add a element anywhere in the heap optinaly");
    }
    @Override
    public boolean addAll(int index, Collection<? extends T> c)
    {
        throw new UOE("You can't add a element anywhere in the heap optinaly");
    }
    @Override
    public T set(int index, T e)
    {
        throw new UOE("You can't add a element anywhere in the heap optinaly");
    }
    //</editor-fold>
}
