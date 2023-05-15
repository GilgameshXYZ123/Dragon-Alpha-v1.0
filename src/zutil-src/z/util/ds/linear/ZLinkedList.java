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
import z.util.lang.Lang;
import z.util.lang.annotation.Passed;
import z.util.math.ExRandom;

/**
 * @author dell
 * @param <T>
 */
@Passed
public class ZLinkedList<T> extends ZDualLinked<T> implements List<T>
{
    //<editor-fold defaultstate="collapsed" desc="Outer-Operator">
    /**
     * @param index a index in the list, from 0 to num, 0=phead, num=ptail
     * @param c
     * @return true if success
     * @Checked
     */
    @Override
    public boolean addAll(int index, Collection<? extends T> c) 
    {
        if(index<0||index>size||c==null||c.isEmpty()) return false;
        this.link(c, this.nodeAt(index));
        return true;
    }
    /**
     * @param index 0=phead, num=ptail,thats:
     * >phead:add a new node behind phead=list[0]
     * >ptail:add a new node behined ptail=list[num-1]
     * @param e 
     * @Checked
     */
     @Override
    public void add(int index, T e)//checked
    {
        if(index<0||index>size) return;
        this.link(new DualNode(e), this.nodeAt(index));
    }
    /**
     * @return if this is empty=null;else ptail.value
     */
    @Override
    public T peek()//check
    {
        return this.isEmpty()? null:ptail.value;
    }
    /**
     * @param e add a new node at the tail
     */
     @Override
    public void push(T e) 
    {
        this.link(new DualNode(e), ptail);
    }
    /**
     * Retrieve and Remove The Node at the tail of this List.
     * @return if this is empty=null;else ptail.value
     */
    @Override
    public T pop() 
    {
        T value=ptail.value;
        this.unlink(ptail);
        return value;
    }
    /**
     * @param index 
     * @return null if index -lt 0 or index -ge num
     */
    @Override
    public T remove(int index) 
    {
        if(index<0||index>=size) return null;
        DualNode<T> n=this.nodeAt(index+1);
        T value=n.value;
        this.unlink(n);
        return value;
    }
    /**
     * @param index
     * @return null if index -lt 0 or index -ge num
     */
    @Override
    public T get(int index)//checked
    {
        if(index<0||index>=size) return null;
        return this.nodeAt(index).value;
    }
    /**
     * @param index
     * @param element
     * @return null if index -lt 0 or index -ge num, else return the old 
     * value of the specified node in the list
     */
    @Override
    public T set(int index, T element)
    {
        if(index<0||index>=size) return null;
        return this.nodeAt(index+1).setVaue(element);
    }
    /**
     * @param e
     * @return true if succeed to add new node in the list at the tail
     */
    @Override
    public boolean offer(T e)
    {
        this.link(new DualNode(e), ptail);
        return true;
    }
    /**
     * @return list[0].value=phead.next.value;if the list is empty,then return null;
     */
    @Override
    public T element()//return 
    {
        return this.phead.value;
    }
    //<>
    /**
     * @return null if the list is empty;
     * or the value in the head of the list, that is phead.value;
     */
    @Override
    public T poll() 
    {
        if(this.isEmpty()) return null;
        T value=phead.value;
        this.unlink(phead);
        return value;
    }
    /**
     * @param o 
     * @return -1 is no such node whose value.equals(o);
     * from 0 to num-1 if there has such node. seach from head totail, so
     * the index is the first index
     */
    @Override
    public int indexOf(Object o) 
    {
        if(this.isEmpty()) return -1;
        phead.value=(T) o;
        int index=0;
        DualNode n=this.phead;
        while(!n.value.equals(o)) index++;
        return (n==phead? -1:index);
    }
    /**
     * @param o 
     * @return -1 is no such node whose value.equals(o);
     * from 0 to num-1 if there has such node. seach from tail to head, so
     * the index is the last index
     */
    @Override
    public int lastIndexOf(Object o) 
    {
        if(this.isEmpty()) return -1;
        int index=size-1;
        phead.value=(T) o;
        DualNode<T> n=this.ptail;
        while(!n.value.equals(o)) 
        {
            n=n.plast;
            index--;
        }
        return index;
    }
    /**
     * @param start the start index in the list, from 0 to num-1
     * @param end the end index in the list 0 to num-1
     * @return null, if start>end or (end-start+1)>num;
     */
     @Override
    public List<T> subList(int start, int end) 
    {
        int len=end-start+1;
        if(start>end||len>size) return null;
        ZLinkedList<T> list=new ZLinkedList<>();
        DualNode<T> n=this.nodeAt(start);
        for(int i=start;i<=end;i++)
        {
            list.ptail.pnext=new DualNode(n.value);
            list.ptail.pnext.plast=list.ptail;
            list.ptail=list.ptail.pnext;
            n=n.pnext;
        }
        list.ptail.pnext=list.phead;
        list.phead.plast=list.ptail;
        list.size=len;
        return list;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Interable">
    private final class Iter implements ListIterator<T>
    {
        //columns---------------------------------------------------------------
        private DualNode<T> cur;
        private int index;
        
        //functions-------------------------------------------------------------
        private Iter(int index)
        {
            this.cur=ZLinkedList.this.nodeAt(index);
            this.index=index;
        }
        private Iter()
        {
            this.cur=ZLinkedList.this.phead;
            this.index=0;
        }
        @Override
        public boolean hasNext() 
        {
            return index<ZLinkedList.this.size;
        }
        @Override
        public T next() 
        {
            T value=cur.value;
            cur=cur.pnext;
            index++;
            return value;
        }
        @Override
        public boolean hasPrevious() 
        {
            return index>0;
        }
        @Override
        public T previous() 
        {
            T value=cur.value;
            cur=cur.plast;
            index--;
            return value;
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
            DualNode last=cur;
            cur=cur.pnext;
            ZLinkedList.this.unlink(last);
        }
        @Override
        public void set(T e) 
        {
            cur.value=e;
        }
        @Override
        public void add(T e) 
        {
            ZLinkedList.this.link(new DualNode(e), cur);
        }
    }
    @Override
    public Iterator<T> iterator()
    {
        return new Iter();
    }
    @Override
    public ListIterator<T> listIterator() 
    {
        return new Iter();
    }
    @Override
    public ListIterator<T> listIterator(int index) 
    {
        return new Iter(index);
    }
    //</editor-fold>
}
