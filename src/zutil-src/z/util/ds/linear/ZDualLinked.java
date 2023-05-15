/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.ds.linear;

import java.lang.reflect.Array;
import java.util.Collection;
import java.util.Iterator;
import java.util.Queue;
import java.util.function.BiPredicate;
import java.util.function.Predicate;
import z.util.ds.imp.ZStack;
import z.util.lang.annotation.Passed;

/**
 *
 * @author dell
 * @param <T>
 */
public abstract class ZDualLinked<T> extends ZLinked<T> implements Queue<T>, ZStack<T>
{
    //<editor-fold defaultstate="collapsed" desc="class DualNode<E>">
    protected static final class DualNode<E>
    {
        //columns---------------------------------------------------------------
        protected E value;
        protected DualNode<E> pnext;
        protected DualNode<E> plast;
        
        //functions-------------------------------------------------------------
        protected DualNode()
        {
            value=null;
            pnext=plast=null;
        }
        protected DualNode(E value)
        {
            this.value=value;
            pnext=plast=null;
        }
        protected E setVaue(E value)
        {
            E ov=this.value;
            this.value=value;
            return ov;
        }
    }
    //</editor-fold>
    protected DualNode<T> phead;
    protected DualNode<T> ptail;
    
    public ZDualLinked()
    {
        size=0;
        ptail=phead=null;
    }
    //<editor-fold defaultstate="collapsed" desc="Basic-Function">
    @Override
    public String toString()//checked
    {
        StringBuilder sb=new StringBuilder();
        sb.append('{');
        if(!this.isEmpty())
        {
            DualNode<T> n=this.phead;
            sb.append(n.value);
            n=n.pnext;
            while(n!=this.phead)
            {
                sb.append(',').append(n.value);
                n=n.pnext;
            }
         }
        sb.append("}");
        return sb.toString();
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Inner-Code">
    /**
     * @param index 0=this.phead,num=this.ptail;
     * @return 
     */
    @Passed
    protected DualNode<T> nodeAt(int index)//checked
    {
        DualNode n=null;
        int bindex=size-index-1;
        if(bindex<index) for(n=this.ptail,index=0;index<bindex;index++) n=n.plast;
        else for(n=this.phead,bindex=0;bindex<index;bindex++) n=n.pnext;
        return n;
    }
    /**
     * if the new node is linked behine ptail, then set ptail=new node.
     * @param n the new added node behind the target
     * @param target the target node to be appended a new node
     */
    @Passed
    protected void link(DualNode n, DualNode target)//checked
    { 
        if(this.isEmpty()) 
        {
            this.phead=this.ptail=n;
            phead.pnext=phead.plast=ptail;
            ptail.pnext=ptail.plast=phead;
        }
        else
        {
            target.pnext.plast=n;
            n.pnext=target.pnext;
            target.pnext=n;
            n.plast=target;
            if(ptail==target) ptail=n;
        }
        size++;
    }
    /**
     * if the new node is linked behine ptail, then set ptail=new node.
     * @param c c is not null, and c.size()>0
     * @param target the target node to be appended new nodes
     */
    @Passed
    protected void link(Collection c, DualNode target)//checked
    {
        //create the link to append
        Iterator iter=c.iterator();
        DualNode head = new DualNode(iter.next());
        DualNode tail = head;
        while(iter.hasNext()) 
        {   
            tail.pnext=new DualNode(iter.next());
            tail.pnext.plast=tail;
            tail=tail.pnext;
        }
        tail.pnext=head;
        head.plast=tail;
        //append th link
        if(this.isEmpty()) 
        {
            this.phead=head;
            this.ptail=tail;
        }
        else
        {
            target.pnext.plast=tail;
            tail.pnext=target.pnext;
            target.pnext=head;
            head.plast=target;
            if(ptail==target) ptail=tail;
        }
        size+=c.size();
    }
    /**
     * @param target unlink a node in the linked List
     */
    @Passed
    protected void unlink(DualNode target)//checked
    {
        if(size>1)
        {
            target.pnext.plast=target.plast;
            target.plast.pnext=target.pnext;
            if(target==ptail) ptail=target.plast;
            else if(target==phead) phead=target.pnext;
        }
        else if(size==1) phead=ptail=null;
        target.plast=null;
        target.pnext=null;
        target.value=null;
        size--;
    }
    protected void unlink(DualNode start, DualNode end)
    {
        //unlink at the start and end
        start.plast.pnext=end.pnext;
        end.pnext.plast=start.plast;
        start.plast=null;
        end.pnext=null;    
        
        //destroy start to end
        DualNode last;
        while(start!=end)
        {
            last=start;
            start=start.pnext;
            last.value=null;
            last.pnext=null;
            last.plast=null;
            size--;
        }
        end.value=null;
        end.pnext=null;
        end.plast=null;
        if(--size==0) phead=ptail=null;
    }
    /**
     * Form the start node, find the first node whose value is null
     * @param start
     * @return 
     */
    @Passed
    protected DualNode<T> firstNodeNull(DualNode start)
    {
        DualNode n=start;
        do
        {
            if(n.value==null) return n;
            n=n.pnext;
        }
        while(n!=start);
        return null;
    }
    /**
     * Form the start node, find the first node whose value.equalts v
     * @param v
     * @param start
     * @return 
     */
    @Passed
    protected DualNode<T> firstNodeEquals(DualNode start, Object v)
    {
        DualNode n=start;
        do
        {
            if(v.equals(n.value)) return n;
            n=n.pnext;
        }
        while(n!=start);
        return null;
    }
    @Override
    protected final boolean innerRemoveAll(Predicate pre)
    {
        DualNode n = this.phead;
        DualNode last;
        int osize=size;
        while(n!=this.ptail)
            if(pre.test(n.value))
            {
                last=n;
                n=n.pnext;
                this.unlink(last);
            }
            else n=n.pnext;
        if(pre.test(this.ptail.value)) this.unlink(ptail);
        return size!=osize;
    }
    @Override
    protected final boolean innerRemoveAll(BiPredicate pre, Object condition)
    {
        DualNode n = this.phead;
        DualNode last;
        int osize=size;
        while(n!=this.ptail)
            if(pre.test(n.value,condition))
            {
                last=n;
                n=n.pnext;
                this.unlink(last);
            }
            else n=n.pnext;
        if(pre.test(this.ptail.value,condition)) this.unlink(ptail);
        return size!=osize;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Ourter-Operator">
    /**
     * @param e add a new DualNode(e) at the tail of the linked list
     * @return true is sunccess
     * @Checked
     */
    @Override
    public boolean add(T e)//checked
    {
        this.link(new DualNode(e), ptail);
        return true;
    }
    /**
     * @param c for each element e in collection c, append  a new node at the patil
     * @return true if success
     * @Checked
     */
    @Override
    public boolean addAll(Collection<? extends T> c) 
    {
        if(c==null||c.isEmpty()) return false;
        this.link(c, ptail);
        return true;
    }
     /**
     * @param o 
     * @return true if contains
     * @Checked
     */
    @Override
    public boolean contains(Object o)//check
    {
        if(this.isEmpty()) return false;
        T ov=ptail.value;
        ptail.value=(T) o;
        DualNode n=this.phead;
        if(o==null)
        {
            if(o==ov) return true;
            while(n.value!=null) n=n.pnext;
        }
        else
        {
            if(o.equals(ov)) return true;
            while(!o.equals(n.value)) n=n.pnext;
        }
        ptail.value=ov;
        return n!=ptail;
    }
    /**
     * @return 
     * @Checked
     */
    @Override
    public T remove()//checked
    {
        if(this.isEmpty()) return null;
        T value=phead.value;
        this.unlink(phead);
        return value;
    }
    /**
     * @param o remove all objects in the list equals to o
     * @return true if o is in the list and is removed successfully
     * @Checked
     */
    @Override
    public boolean remove(Object o)
    {
        if(this.isEmpty()) return false;
        int osize=size;
        DualNode n = this.phead;
        DualNode last;
        if(o==null)
        {
            while(n!=this.ptail)
                if(n.value==null)
                {
                    last=n;
                    n=n.pnext;
                    this.unlink(last);
                }
                else n=n.pnext;
            if(this.ptail.value==null) this.unlink(ptail);
        }
        else
        {
            while(n!=this.ptail)
                if(o.equals(n.value))
                {
                    last=n;
                    n=n.pnext;
                   this.unlink(last);
                }
            else n=n.pnext;
        }
        if(n.equals(this.ptail.value)) this.unlink(ptail);
        return size!=osize;
    }
    /**
     * remove all elemnts in the linked list, except for the sentinal node=phead,
     * then set num=0.
     */
    @Override
    public void clear()//checked
    {
        if(this.isEmpty()) return;
        DualNode<T> last=null;
        while(ptail!=this.phead)
        {
            last=ptail;
            ptail=ptail.plast;
            last.plast=null;
            last.pnext=null;
            last.value=null;
        }
        ptail.value=null;
        ptail.plast=null;
        ptail.pnext=null;
        ptail=phead=null;
        size=0;
    }
      /**
     * @return new Array contains the references of the elements in the list,
     * if the list is empty, the array is not null.
     */
    @Override
    public Object[] toArray()//checked
    {
        if(this.isEmpty()) return null;
        Object[] arr=new Object[size];
        int index=0;
        DualNode n=phead;
        do
        {
            arr[index++]=n.value;
            n=n.pnext;
        }
        while(n!=phead);
        return arr;
    }
    /**
     * @param <E>
     * @param a
     * @return new Array contains the references of the elements in the list,
     * if the list is empty, the array is not null, but array.length is 0.
     */
    @Override
    public <E> E[] toArray(E[] a)//checked
    {
        if(a.length<size)
            a=(E[]) Array.newInstance(a.getClass().getComponentType(), size);
        int index=0;
        Object[] arr=a;
        DualNode n=phead.pnext;
        do
        {
            arr[index++]=a;
            n=n.pnext;
        }
        while(n!=this.phead);
        if(a.length>size) a[size]=null;
        return a;
    }
    //</editor-fold>
}
