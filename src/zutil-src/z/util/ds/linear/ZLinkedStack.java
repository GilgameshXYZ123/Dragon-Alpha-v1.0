/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.ds.linear;
import java.lang.reflect.Array;
import java.util.Collection;
import java.util.Iterator;
import java.util.function.BiPredicate;
import java.util.function.Predicate;
import z.util.ds.imp.ZStack;
import z.util.lang.annotation.Passed;

/**
 * This is Stack is basd on One-Way LinkedList, with no limitation of size.
 * It's very simple and cost lower time and space.
 * @author dell
 * @param <T>
 */
@Passed
public class ZLinkedStack<T> extends ZLinked<T> implements ZStack<T>
{
    //<editor-fold defaultstate="collapsed" desc="class ZNode<E>">
    private final static class ZNode<E>//checked
    { 
        private E value;
        private ZNode<E> plast;
        
        private ZNode(E value, ZNode<E> plast) {
            this.value = value;
            this.plast = plast;
        }
    }
    //</editor-fold>
    private ZNode<T> ptail;
    
    public ZLinkedStack() {
        this.ptail = null;
        this.size=0;
    }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Function">
    @Override
    public String toString()
    {
        StringBuilder sb = new StringBuilder(128);
        sb.append("{ ");
        if(!isEmpty()) {
            ZNode n=this.ptail;
            sb.append(n.value);
            n=n.plast;
            while(n!=null)
            {
                sb.append(',').append(n.value);
                n=n.plast;
            }
        }
        sb.append(" }");
        return sb.toString();
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Inner-Code">
    @Passed
    private void unlink(ZNode<T> last,ZNode<T> target, ZNode<T> next)
    {
        if(next!=null) next.plast=last;//tar!=phead
        else  ptail=ptail.plast;//tar=phead
        target.value=null;
        target.plast=null;
        size--;
    }
    
    @Override
    protected final boolean innerRemoveAll(Predicate pre) 
    {
        int osize=size;
        ZNode n=this.ptail,last=null,next=null;
        while(n!=null)
            {
                if(pre.test(n.value))
                {
                    last=n.plast;
                    this.unlink(last, n, next);
                    n=last;
                }
                else {next=n;n=n.plast;}
            }
        return size!=osize;    
    }
    @Override
    protected final boolean innerRemoveAll(BiPredicate pre, Object condition) 
    {
        int osize=size;
        ZNode n=this.ptail,last=null,next=null;
        while(n!=null)
            {
                if(pre.test(n.value,condition))
                {
                    last=n.plast;
                    this.unlink(last, n, next);
                    n=last;
                }
                else {next=n;n=n.plast;}
            }
        return size!=osize;    
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="ZLinked">
    /**
     * add an new Node at the end of the Stack.
     * @param value
     * @return 
     */
    @Override
    public boolean add(T value)//checked
    {
        ptail=new ZNode<>(value,this.ptail);
        size++;
        return true;
    }
    /**
     * add all Values of {@code Collection c} at the end of the Stack.
     * @param c
     * @return 
     */
    @Override
    public boolean addAll(Collection<? extends T> c)//checked
    {
        for(T o:c) ptail=new ZNode<>(o, ptail);
        size+=c.size();
        return true;
    }
    
    @Override
    public boolean contains(Object o)
    {
        for(ZNode n=this.ptail;n!=null;n=n.plast)
            if(n.value.equals(o)) return true;
        return false;
    }
    @Override
    public boolean remove(Object o)//checked
    {
        int osize=size;
        ZNode n=this.ptail,last=null,next=null;
        if(o==null)
            while(n!=null)
            {
                if(n.value==null)
                {
                    last=n.plast;
                    this.unlink(last, n, next);
                    n=last;
                }
                else {next=n;n=n.plast;}
            }
        else // o != null
            while(n!=null)
            {
                if(o.equals(n.value))
                {
                    last=n.plast;
                    this.unlink(last, n, next);
                    n=last;
                }
                else {next=n;n=n.plast;}
            }
        return size!=osize;    
    }
    @Override
    public void clear()
    {
        ZNode<T> cur=null;
        while(ptail!=null)
        {
            cur=ptail;
            ptail=ptail.plast;
            cur.value=null;
            cur.plast=null;
        }
        size=0;
    }
    @Override
    public Object[] toArray()
    {
        Object[] arr=new Object[this.size];
        int index=0;
        for(ZNode n=this.ptail;n!=null;n=n.plast) 
            arr[index++]=n.value;
        return arr;
    }
    @Override
    public <E> E[] toArray(E[] a)
    {
        if(a.length<size)
            a=(E[]) Array.newInstance(a.getClass().getComponentType(), size);
        
        int index=0;
        Object[] o=a;
        for(ZNode n=this.ptail;n!=null;n=n.plast)
            o[index++]=n.value;
        if(a.length>size) a[size]=null;
        
        return a;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="ZStack">
    /**
     * add a new Node at the end of the Stack.
     * @param e 
     */
    @Override
    public void push(T e) 
    {
        ptail=new ZNode<>(e,this.ptail);
        size++;
    }
    @Override
    public T peek()
    {
        return (this.isEmpty()? null:ptail.value);
    }
    /**
     * @return the last value of the Stack.
     */
    @Override
    public T pop()//passed
    {
        if(this.isEmpty()) return null;
        ZNode<T> n=this.ptail;
        ptail=ptail.plast;
        n.plast=null;
        size--;
        return n.value;
    }
    public T remove()
    {
        if(this.isEmpty()) return null;
        ZNode<T> n=this.ptail;
        ptail=ptail.plast;
        n.plast=null;
        size--;
        return n.value;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Iterable">
    @Passed
    private final class Iter implements Iterator<T>
    {
        //columns---------------------------------------------------------------
        ZNode<T> cur;
        
        //functions-------------------------------------------------------------
        private Iter(ZNode cur)
        {
            this.cur=cur;
        }
        @Override
        public boolean hasNext() 
        {
            return cur!=null&&cur.plast==null;
        }
        @Override
        public T next() 
        {
            if(cur==null) return null;
            T value=cur.value;
            cur=cur.plast;
            return value;
        }
    }
    @Override
    public Iterator<T> iterator() 
    {
        return new Iter(ptail);
    }
    //</editor-fold>
}
