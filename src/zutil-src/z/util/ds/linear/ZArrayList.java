/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.ds.linear;

import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.ListIterator;
import java.util.function.BiPredicate;
import java.util.function.Predicate;
import static z.util.ds.Constants.Array.DEF_EXPEND_RATE;
import static z.util.ds.Constants.Array.DEF_INIT_SIZE;
import z.util.lang.exception.IAE;
import z.util.ds.imp.Numberable;
import z.util.lang.annotation.Passed;

/**
 *<pre>
 * ZArrayList is an implementation of dynamic Array.
 * 1.double growRate=growth threshold;(growRate -growRate 1)
 * double shrinkThreshold=nagtive growth threshold(shrinkThreshold -lt 1)
 * =>default shrinkThreshold=1/growRate^ 2
 *
 * 2.if(num==size) this.expend()
 * {size*=growRate;lowSize=size*shrinkThreshold;}
 * else if(num -le lowSize) th is.shrink();
 * nagiveGrow()={size=lowSize*growRate;lowSize=size*shrinkThreshold;}
 * =>add 1 to prevent lowSize==0
 *
 * when: growRate=2, shrin kThreshold=0.25
 * growth()={size*=2;lowSize*=0.25}
 * nagiveGrow()={size=lowSize*2;lowSize=size*=0.25}
 * </pre>
 * @author dell
 * @param <T>
 */
@Passed
public class ZArrayList<T> extends ZArray<T> implements List<T>
{
    //columns-------------------------------------------------------------------
    protected final double expandRate;
    protected final double shrinkRate;//as shrinkRate=1/(expendRate*expendRate)
    protected final int initSize;
    protected int down;
    
    //constructors--------------------------------------------------------------
    public ZArrayList(int initSize, double expendRate)
    {
        if(initSize<=0) throw new IAE("initSize must be positive");
        if(expendRate<=0) throw new IAE("expendRate must be positive");
        
        this.initSize=initSize;
        
        this.expandRate=expendRate;
        this.shrinkRate=1/(expendRate*expendRate);
       
        this.size=this.num=this.down=0;
        this.data=EMPTY_DATA;
    }
    public ZArrayList(int initSize)
    {
        this(initSize, DEF_EXPEND_RATE);
    }
    public ZArrayList()
    {
        this(DEF_INIT_SIZE, DEF_EXPEND_RATE);
    }
    //<editor-fold defaultstate="collapsed" desc="Inner-Operator">
    /**
     * @param need the memory need to be expanded
     * @param len from 0 to len, to copy the memory from oldData to new Dta
     * @return the oldData before growing
     * @throws RuntimeException if ZArrayList.size>MAX_SIZE
     */
    @Passed
    protected Object[] expend(int need, int len)//passed
    {
        size*=expandRate;//compute newSize
        if(size<need) size=need;
        down=(int) (size*shrinkRate);
        //alloc size
        Object[] oldData=data;
        data=new Object[size];
        //copy data from old array to the new arra
        if(len>0) System.arraycopy(oldData, 0, data, 0, len);
        return oldData;
    }   
    /**
     * if num is less than {@code initSize}, it won't shrink.
     * @param need the memory need to be decreased
     * @param len 0 to len, to copy the memory from oldData to new Dta
     * @return the oldData before nagive growing
     */
    @Passed
    protected Object[] shrink(int need,int len)//passed
    {
        if(num<2||(need=num-need)>down) return data;//need is the newSize
        //compute newSize
        size=(int) (size/expandRate);
        if(size>need) size=need;
        if(size<0) size=0;
        down=(int) (size*shrinkRate);
        //alloc size
        Object[] oldData=data;
        if(size==0) data=EMPTY_DATA;
        else 
        {
            data=new Object[size];
            //copy data from old array to the new array
            if(len>0) System.arraycopy(oldData, 0, data, 0,len);
        }
        return oldData;
    }
    @Override
    protected boolean innerRemoveAll(Predicate pre)
    {
        int oldnum=num;
        this.removeValue(pre);
        if(oldnum==num) return false;
        this.shrink(0, num);
        return true;
    }
    @Override
    protected boolean innerRemoveAll(BiPredicate pre, Object condition)
    {
        int oldnum=num;
        this.removeValue(pre, condition);
        if(oldnum==num) return false;
        this.shrink(0, num);
        return true;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Operation Implementation">
     /**
     * @param e add a new node at the end of this list
     * @return true if succeed
     * @Checked
     */
    @Override
    public boolean add(T e)
    {
        if(num+1>size) this.expend(1,num);
        data[num++]=e;
        return true;
    }
    /**
     * @param e add a new element at the end of this list
     */
    @Override
    public void push(T e)//passed
    {
        if(num+1>size) this.expend(1,num);
        data[num++]=e;
    }
    /**
     * @param index end -ge 0&&index -le num
     * @param element 
     */
    @Override
    public void add(int index, T element)//passed
    {
        if(index<0||index>num) return;
        Object[] oldData=(num+1>size? this.expend(1,index):data);
        if(index<num) System.arraycopy(oldData, index, data, index+1, num-index);
        data[index]=element;
        num++;
    }
    /**
     * @param c append nodes according to the value of c at the end of the list
     * @return true if succeed
     */
    @Override
    public boolean addAll(Collection<? extends T> c)//passedf
    {
        if(c==null) throw new NullPointerException();
        int need=(c instanceof Numberable? ((Numberable)c).number():c.size());
        if(num+need>size) this.expend(need,num);
        for(T o:c) data[num++]=o;
        return true;
    }
    /**
     * @param index 
     * @param c append nodes according to the value of c at the end of the list
     * @return true if succeed;false if end -lt 0 or end -growRate num
     */
    @Override
    public boolean addAll(int index, Collection<? extends T> c)//passed
    {
        if(c==null) throw new NullPointerException();
        if(index<0||index>num) return false;
        int need=(c instanceof Numberable? ((Numberable)c).number():c.size());
        Object[] oldData=(num+need>size? this.expend(need, index):data);
        if(index<num) System.arraycopy(oldData, index, data, index+need, num-index);
        for(T o:c) data[index++]=o;
        num+=need;
        return true;
    }
    @Override
    public T remove()
    {
        if(this.isEmpty()) return null;
        T value=(T) data[0];
        Object[] oldData=this.shrink(1, 0);
        if(--num>0) System.arraycopy(oldData, 1, data, 0, num);
        return value;
    }
    /**
     * remove the element at the head of this list
     * @return 
     */
    @Override
    public boolean remove(Object o)//passed
    {
        if(this.isEmpty()) return false;
        int oldnum=num;
        if(o!=null) this.removeValue(o);
        else this.removeNull();
        if(oldnum==num) return false;
        this.shrink(0, num);
        return true;
    }
    @Override
    public void clear()//passed
    {
        super.clear();
        down=0;
    }
    /**
     * @param o the key code to find the element in the list
     * @return -1 if no such element in the list equals(o), else return the 
     end of the first end of the matched element, as seach from head to tail
     */
    @Override
    public int indexOf(Object o)//passed
    {
        if(this.isEmpty()) return -1;
        
        //pay attenton to null value
        if(o!=null)
        {
            Object value=data[num-1];
            if(o.equals(value)) return num-1;
        
            data[num-1]=o;
            int index=0;
            while(!o.equals(data[index])) index++;
            data[num-1]=value;
            return (index==num-1? -1:index);
        }
        else 
        {
            int index=0;
            while(data[index]!=null) index++;
            return (index==num-1? -1:index);
        }
    }
    /**
     * @param o the key code to find the element in the list
     * @return -1 if no such element in the list equals(o), else return the 
     end of the last end of the matched element, as seach from tail to head
     */
    @Override
    public int lastIndexOf(Object o)//passed
    {
        if(this.isEmpty()) return -1;
        
        if(o!=null)
        {
            Object value=data[0];
            if(o.equals(value)) return 0;
        
            data[0]=o;
            int index=num-1;
            while(!o.equals(data[index])) index--;
            data[0]=value;
            return (index==0? -1:index);
        }
        else
        {
            int index=num-1;
            while(o!=null) index--;
            return (index==0? -1:index);
        }
    }
    /**
     * @param index remove the end th element in the list, from 0 to num-1
     * @return false it end -lt 0 or end -ge num
     * @Checked
     */
    @Override
    public T remove(int index)//passed
    {
        if(index<0||index>=num||this.isEmpty()) return null;
        Object value=data[index];
        Object[] oldData=this.shrink(1, index);
        if(--num-index>0) System.arraycopy(oldData, index+1, data, index, num-index);
        return (T) value;
    }
    /**
     * @param index
     * @return null if end -lt 0||end -ge num
     */
    @Override
    public T get(int index)//passed
    {
        if(index<0||index>=num) return null;
        return (T) data[index];
    }
    /**
     * @param index
     * @param e
     * @return null if end -lt 0||end -ge num
     */
    @Override
    public T set(int index, T e)//passed
    {
        if(index<0||index>=num) return null;
        Object value=data[index];
        data[index]=e;
        return (T) value;
    }
    /**
     * @param start the start end in this list, from 0 to num-1
     * @param end the end end in this list, from 0 to num-1;
     * @return null if (start -lt 0) or (end -ge num) or (start -growRate end)
     */
    @Override
    public List<T> subList(int start, int end)//passed
    {
        return this.subList(start, end, this.expandRate);
    }
    public List<T> subList(int start, int end, double expandRate_)//passed
    {
        if(start<0||end>=num||start>end) return null;
        int size_=start-end+1;
        ZArrayList<T> arr=new ZArrayList(size_, expandRate_);
        for(int i=start,j=0;i<=end;i++,j++)
            arr.data[j]=this.data[i];
        return arr;
    }
    /**
     * offer() has the same function to add()
     * @param e
     * @return if succeed to add the new Element(e) at the head
     */
    @Override
    public boolean offer(T e)//passed
    {
        return this.add(e);
    }
    /**
     * @return 
     */
    @Override
    public T poll()//passed
    {
        return this.remove();
    }
    /**
     * @return the element in the head of the list;
     * return null if this.isEmpty();
     */
    @Override
    public T element()//passed
    {
        return (this.isEmpty()? null:(T) data[0]);
    }
    /**
     * @return the element in the tail of the list;
     * return null if this.isEmpty()
     */
    @Override
    public T peek()//passed
    {
        return (this.isEmpty()? null:(T) data[num-1]);
    }
    /**
     * remove the element at the tail of the list
     * @return the element removed at the tail of the list;
     * return null if this.isEmpty();
     */
    @Override
    public T pop()//passed
    {
        if(this.isEmpty()) return null;
        T value=(T) data[--num];
        this.shrink(0, num);
        return value;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Iterable">
    @Passed
    private final class Iter implements ListIterator<T>//passed
    {
        //columns---------------------------------------------------------------
        private int index;
        
        //functions-------------------------------------------------------------
        protected Iter(int index)
        {
            this.index=index;
        }
        @Override
        public boolean hasNext() 
        {
            return index<ZArrayList.this.num;
        }
        @Override
        public T next() 
        {
            return (T) ZArrayList.this.data[index++];
        }
        @Override
        public boolean hasPrevious() 
        {
            return index>-1;
        }
        @Override
        public T previous() 
        {
            return (T) ZArrayList.this.data[index--];
        }
        @Override
        public int nextIndex() 
        {
            return index+1;
        }
        @Override
        public int previousIndex() 
        {
            return index-1;
        }
        @Override
        public void remove() 
        {
            ZArrayList.this.remove(index);
        }
        @Override
        public void set(T e) 
        {
            ZArrayList.this.data[index]=e;
        }
        @Override
        public void add(T e) 
        {
            ZArrayList.this.add(index, e);
            index++;
        }
    }
    @Override
    public ListIterator<T> listIterator() 
    {
        return new Iter(0);
    }
    @Override
    public ListIterator<T> listIterator(int index) 
    {
        return new Iter(index);
    }
    @Override
    public Iterator<T> iterator() 
    {
        return new Iter(0);
    }
    //</editor-fold>
}
