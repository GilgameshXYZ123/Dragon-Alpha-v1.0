/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.ds.tree;

import z.util.ds.linear.ZLinkedStack;
import z.util.ds.linear.ZLinkedList;
import java.lang.reflect.Array;
import java.util.Collection;
import java.util.Iterator;
import java.util.function.BiPredicate;
import java.util.function.Consumer;
import java.util.function.Predicate;
import z.util.ds.ZBaseSet;
import z.util.ds.imp.Indexable;
import z.util.ds.imp.Structurable;
import z.util.ds.imp.ZStack;
import z.util.lang.annotation.Passed;

/**
 * ZTreeSet is an implementation of Red-Back Tree Set.
 * We use false to represent black, while true corrsponds to red.
 * @author dell
 * @param <T>
 */
@Passed
public class ZTreeSet<T extends Comparable> extends ZBaseSet<T> implements Structurable, Cloneable
{
    //<editor-fold defaultstate="collapsed" desc="static class ZTreeSet.ZNode<M>">
    protected final static class ZNode<M>//checked
    {
        //columns---------------------------------------------------------------
        private M key;
        private boolean color;
        
        private ZNode<M> pleft;
        private ZNode<M> pright;
        private ZNode<M> parent;
        
        //functions-------------------------------------------------------------
        protected ZNode()
        {
            color=false;
            pleft=pright=parent=null;
        }
        protected ZNode(M key)
        {
            this.color=true;
            this.key=key;
            
            this.parent=nullptr;
            this.pleft=nullptr;
            this.pright=nullptr;
        }
        protected ZNode(M key, ZNode<M> parent)
        {
            this.color=true;
            this.key=key;
            
            this.parent=parent;
            this.pleft=nullptr;
            this.pright=nullptr;
        }
        protected void setKey(M key)
        { 
            this.key=key;
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="nullptr">
    protected static final ZNode nullptr;
    
    static
    {
        nullptr=new ZNode();
        nullptr.pleft=nullptr;
        nullptr.pright=nullptr;
        nullptr.parent=nullptr;
    }
    //</editor-fold>
    protected ZNode<T> root;
    
    public ZTreeSet()
    {
        size=0;
        root=nullptr;
    }
    //<editor-fold defaultstate="collapsed" desc="Basic-Function">
    @Override
    public int size() 
    {
        return size;
    }
    @Override
    public int number() 
    {
        return size;
    }
    public int height()
    {
        return this.heightFrom(root);
    }
    @Override
    public boolean isEmpty() 
    {
        return size==0||root==nullptr;
    }
    @Passed
    public void append(StringBuilder sb)
    {
        sb.append('{');
        if(!this.isEmpty())
        {
            ZLinkedStack<ZNode> s=new ZLinkedStack<>();
            ZNode r=this.root;
            do
            {
                if(r!=nullptr)
                {
                    s.push(r);
                    r=r.pleft;
                }
                else
                {
                    r=s.pop();
                    sb.append(r.key).append(", ");
                    r=r.pright;
                }
            }
            while(!s.isEmpty()||r!=nullptr);
        }
        if(size>1) sb.setCharAt(sb.length()-2, '}');
        else sb.append('}');
    }
    @Override
    public String toString()
    {
        StringBuilder sb=new StringBuilder();
        this.append(sb);
        return sb.toString();
    }
    @Override
    public int getIndexType() 
    {
        return Indexable.TREE;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Inner-Operator">
    @Passed
    protected ZNode<T> findNode(Object value)
    {
        ZNode<T> r=this.root;
        int v;
        while(r!=nullptr)
        {
            v=r.key.compareTo(value);
            if(v>0) r=r.pleft;
            else if(v<0) r=r.pright;
            else break;
        }
        return r;
    }
    @Passed
    protected final void inOrderTraverse(ZNode root, Consumer con, ZStack<ZNode> s)
    {
        ZNode<T> r=root;
        do
        {
            if(r!=nullptr)
            {
                s.push(r);
                r=r.pleft;
            }
            else
            {
                r=s.pop();
                con.accept(r.key);
                r=r.pright;
            }
        }
        while(!s.isEmpty()||r!=nullptr);
    }
    @Override
    public final void struc()//need to optimize
    {
        ZNode r=root;
        int deep=1;
        ZStack<ZNode> s=new ZLinkedStack<>();
        ZStack<Integer> d=new ZLinkedStack<>();
        s.push(r);
        d.push(deep);
        while(!s.isEmpty())
        {
            r=s.pop();
            deep=d.pop();
            System.out.println(r.key+">>>"+deep);
            if(r.pleft!=nullptr) {s.push(r.pleft);d.push(deep+1);}
            if(r.pright!=nullptr) {s.push(r.pright);d.push(deep+1);}
        }
    }
    @Passed
    protected void transplant(ZNode p, ZNode c)//checked
    {
        ZNode g=p.parent;
        if(g==nullptr) this.root=c;
        else if(g.pleft==p) g.pleft=c;
        else g.pright=c;
        c.parent=g;
    }
    @Passed
    protected ZNode<T> maxNode(ZNode root)//checked
    {
        try
        {
        while(root.pright!=nullptr)
            root=root.pright;
        }
        catch(Exception e)
        {
            System.out.println(root);
        }
        return root;
    }
    @Passed
    protected ZNode<T> lastNode(ZNode root)//checked
    {
        if(root.pleft!=nullptr) return maxNode(root.pleft);
        ZNode y=root.parent;
        while(y!=nullptr&&y.pleft==root)
        {
            root=y;
            y=y.parent;
        }
        return y;
    }
    @Passed
    protected ZNode<T> minNode(ZNode root)
    {
        while(root.pleft!=nullptr)
            root=root.pleft;
        return root;
    }
    @Passed
    protected ZNode<T> nextNode(ZNode target)
    {
        if(target.pright!=null) return minNode(target.pright);
        ZNode y=target.parent;
        while(y!=nullptr&&y.pright==target)
        {
            target=y;
            y=y.parent;
        }
        return y;
    }
    @Passed
    protected void leftRotate(ZNode target)//checked
    {
        ZNode right=target.pright;
        if(right==nullptr) return;
        target.pright=right.pleft;
        if(right.pleft!=nullptr) right.pleft.parent=target;
        this.transplant(target, right);
        right.pleft=target;
        target.parent=right;
    }
    @Passed
    protected void rightRotate(ZNode target)//checked
    {
        ZNode left=target.pleft;
        if(left==nullptr) return;
        target.pleft=left.pright;
        if(left.pright!=nullptr) left.pright.parent=target;
        this.transplant(target, left);
        left.pright=target;
        target.parent=left;
    }
    @Passed
    protected void insertFix(ZNode z)//checked
    {
        ZNode y=nullptr;
        while(z.parent.color)
        {
            if(z.parent==z.parent.parent.pleft)
            {
                y=z.parent.parent.pright;
                if(y.color)
                {
                    y.color=z.parent.color=false;
                    z=z.parent.parent;
                    z.color=true;
                }
                 else
                {
                    if(z==z.parent.pright)
                    {
                        z=z.parent;
                        this.leftRotate(z);
                    }
                    z.parent.color=false;
                    z.parent.parent.color=true;
                    this.rightRotate(z.parent.parent);
                }
            }
            else
            {
                y=z.parent.parent.pleft;
                if(y.color)
                {
                    y.color=z.parent.color=false;
                    z=z.parent.parent;
                    z.color=true;
                }
                else
                {
                    if(z==z.parent.pleft)
                    {
                        z=z.parent;
                        this.rightRotate(z);
                    }
                    z.parent.color=false;
                    z.parent.parent.color=true;
                    this.leftRotate(z.parent.parent);
                }
            }
        }
        this.root.color=false;
    }
    @Passed
    protected final void removeFix(ZNode z)
    {
        ZNode w=nullptr;
        while(z!=this.root&&!z.color)
        {
            if(z==z.parent.pleft)
            {
                w=z.parent.pright;
                if(w.color)
                {
                    w.color=false;
                    z.parent.color=true;
                    this.leftRotate(z.parent); 
                    w=z.parent.pright;
                }
                if(!w.pleft.color&&!w.pright.color)
                {
                    w.color=true;
                    z=z.parent;
                }
                else if(!w.pright.color)
                {
                    w.pleft.color=false;
                    w.color=true;
                    this.rightRotate(w);
                    w=z.parent.pright;
                }
                w.color=z.parent.color;
                z.parent.color=false;
                w.pright.color=false;
                this.leftRotate(z.parent);
                z=this.root;
            }
            else
            {
                w=z.parent.pleft;
                if(w.color)
                {
                    w.color=false;
                    z.parent.color=true;
                    this.rightRotate(z.parent);
                    w=z.parent.pleft;
                }
                if(!w.pleft.color&&!w.pright.color)
                {
                    w.color=true;
                    z=z.parent;
                }
                else if(!w.pleft.color)
                {
                    w.pright.color=false;
                    w.color=true;
                    this.leftRotate(w);
                    w=z.parent.pleft;
                }
                w.color=z.parent.color;
                z.parent.color=false;
                w.parent.color=false;
                this.rightRotate(z.parent);
                z=this.root;
            }
        }
        z.color=false;
    }
    @Passed
    protected void removeNode(ZNode z)
    {
        ZNode x=nullptr;
        boolean oc=z.color;
        if(z.pleft==nullptr) this.transplant(z, x=z.pright);
        else if(z.pright==nullptr) this.transplant(z, x=z.pleft);
        else
        {
            ZNode y=maxNode(z.pleft);
            oc=y.color;
            x=y.pleft;
            if(y.parent!=z)
            {
                this.transplant(y, x);
                y.pleft=z.pleft;
                z.pleft.parent=y;
            }
            this.transplant(z, y);
            y.pright=z.pright;
            z.pright.parent=y;
            y.color=z.color;
        }
        z.pright=z.pleft=z.parent=null;
        z.key=null;
        if(x!=nullptr&&!oc) this.removeFix(x);
        if(--this.size==0) this.root=nullptr;
    }
    @Override
    protected boolean innerRemoveAll(Predicate pre) 
    {
        ZNode r=this.root;
        ZLinkedList<ZNode> l1=new ZLinkedList<>();
        ZLinkedList<ZNode> l2=new ZLinkedList<>();
        l1.add(r);
        while(!l1.isEmpty())
        {
            r=l1.remove();
            if(pre.test(r.key)) l2.add(r);
            if(r.pleft!=nullptr) l1.add(r.pleft);
            if(r.pright!=nullptr) l1.add(r.pright);
        }
        boolean result=!l2.isEmpty();
        while(!l2.isEmpty())
            this.removeNode(l2.pop());
        return result;
    }
    @Override
    protected boolean innerRemoveAll(BiPredicate pre, Object condition) 
    {
        ZNode r=this.root;
        ZLinkedList<ZNode> l1=new ZLinkedList<>();
        ZLinkedList<ZNode> l2=new ZLinkedList<>();
        l1.add(r);
        while(!l1.isEmpty())
        {
            r=l1.remove();
            if(pre.test(r.key, condition)) l2.add(r);
            if(r.pleft!=nullptr) l1.add(r.pleft);
            if(r.pright!=nullptr) l1.add(r.pright);
        }
        boolean result=!l2.isEmpty();
        while(!l2.isEmpty())
            this.removeNode(l2.pop());
        return result;
    }
    @Passed
    protected final int copyTree(ZNode<T> r, ZNode<T> aroot)
    {
        ZStack<ZNode> s1=new ZLinkedStack<>();
        ZStack<ZNode> s2=new ZLinkedStack<>();
        int num=0;
        while(!s1.isEmpty())
        {
            r=s1.pop();
            aroot=s2.pop();
            if(r.pleft!=nullptr)
            {
                aroot.pleft=new ZNode(r.pleft.key, aroot);
                num++;
                s1.push(r.pleft);
                s2.push(aroot.pleft);
            }
            if(r.pright!=nullptr) 
            {
                aroot.pright=new ZNode(r.pright.key, aroot);
                num++;
                s1.push(r.pleft);
                s2.push(aroot.pleft);
            }
        }
        return num;
    }
    @Passed
    protected final int heightFrom(ZNode r)
    {
        int h=0, maxh=0;
        ZStack<ZNode> s=new ZLinkedStack<>();
        ZStack<Integer> d=new ZLinkedStack<>();
        do
        {
            if(r!=nullptr)
            {
                s.push(r);d.push(h);
                r=r.pleft;h++;
            }
            else
            {
                if(maxh<h) maxh=h;
                r=s.pop();h=d.pop();
                r=r.pright;
                h++;
            }
        }
        while(!s.isEmpty()||r!=nullptr);
        return maxh;
    }
    @Passed
    protected final void createHeadSet(T endKey, boolean inclusive, ZNode<T> r, ZTreeSet atree, ZStack<ZNode> s)
    {
        int v;
        while(r!=nullptr)
        {
            v=r.key.compareTo(endKey);
            if(v>0) r=r.pleft;
            else if(v<0)
            {
                atree.add(r.key);
                if(r.pleft!=nullptr) this.inOrderTraverse(r.pleft, atree::add, s);
                r=r.pright;
            }
            else
            {
                if(inclusive) atree.add(r.key);
                if(r.pleft!=nullptr) this.inOrderTraverse(r.pleft, atree::add, s);
                return;
            }
        }
    }
    @Passed
    protected final void createTailSet(T fromKey, boolean inclusive, ZNode<T> r, ZTreeSet atree, ZStack<ZNode> s)
    {
        int v;
        while(r!=nullptr)
        {
            v=r.key.compareTo(fromKey);
            if(v<0) r=r.pright;
            else if(v>0)
            {
                atree.add(r.key);
                if(r.pright!=nullptr) this.inOrderTraverse(r.pright, atree::add, s);
                r=r.pleft;
            }
            else
            {
                if(inclusive) atree.add(r.key);
                if(r.pright!=nullptr) this.inOrderTraverse(r.pright, atree::add, s);
                return;
            }
        }
    }
    @Passed
    protected final void createSubSetNoEqual(T lowKey, T highKey, ZNode<T> r, ZTreeSet atree, ZStack<ZNode> s)
    {
        s.push(r);
        while(!s.isEmpty())
        {
            r=s.pop();
            while(r!=nullptr)
            {
                if(r.key.compareTo(lowKey)<0) {r=r.pright;continue;}//r.key<lowKey
                if(r.key.compareTo(highKey)>0) {r=r.pleft;continue;}//r.key>highKey
                atree.add(r.key);
                if(r.pleft!=nullptr) s.push(r.pleft);
                if(r.pright!=nullptr) s.push(r.pright);
                break;
            }
        }
    }
    @Passed
    protected final void createSubSet(T lowKey, T highKey, boolean inclusive, ZNode<T> r, ZTreeSet<T> atree, ZStack<ZNode> s)
    {
        s.push(r);
        int v1,v2;//v1:lowKey, v2:highKey
        while(!s.isEmpty())
        {
            r=s.pop();
            while(r!=nullptr)
            {
                v1=r.key.compareTo(lowKey);
                if(v1<0) {r=r.pright;continue;}//r.key<lowKey
                v2=r.key.compareTo(highKey);
                if(v2>0) {r=r.pleft;continue;}//r.key>highKey
                if(v1==0)//r.key=lowKey
                {
                    if(inclusive) atree.add(r.key);
                    this.createSubSetNoEqual(lowKey, highKey, r.pright, atree, s);
                    return;
                }
                if(v2==0) 
                {
                    if(inclusive) atree.add(r.key);
                    this.createSubSetNoEqual(lowKey, highKey, r.pleft, atree, s);
                    return;
                }
                atree.add(r.key);
                if(r.pleft!=nullptr) s.push(r.pleft);
                if(r.pright!=nullptr) s.push(r.pright);
                break;
            }
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Outer-Operator">
    @Override
    public boolean contains(Object o)//checked
    {
        return this.findNode(o)==nullptr;
    }
   
    @Override
    public boolean add(T e)//checked
    {
        if(e==null) throw new NullPointerException();
        ZNode<T> r=this.root,last=nullptr;
        int v;
        while(r!=nullptr)
        {
            last=r;
            v=r.key.compareTo(e);
            if(v>0) r=r.pleft;
            else if(v<0) r=r.pright;
            else 
            {
                r.setKey(e);
                return false;
            }
        }
        r=new ZNode<>(e, last);
        if(last==nullptr) this.root=r;
        else if(last.key.compareTo(e)>0) last.pleft=r;
        else last.pright=r;
        this.insertFix(r);
        this.size++; 
        return false;
    }
    @Override
    public boolean addAll(Collection<? extends T> c)//checked
    {
        if(c==null) throw new NullPointerException();
        int osize=this.size;
        for(T o:c) this.add(o);
        return osize!=size;
    }
    @Override
    public boolean remove(Object o)//checked
    {
        if(o==null) throw new NullPointerException();
        ZNode r=this.findNode(o);
        if(r==nullptr) return false;
        this.removeNode(r);
        return true;
    }
    @Override
    public void clear()
    {
        if(this.isEmpty()) return;
        ZNode r=this.root;
        ZLinkedList<ZNode> l1=new ZLinkedList<>();
        l1.add(r);
        while(!l1.isEmpty())
        {
            r=l1.remove();
            if(r.pleft!=nullptr) l1.add(r.pleft);
            if(r.pright!=nullptr) l1.add(r.pright);
            r.parent=r.pleft=r.pright=null;
            r.key=null;
        }
        root=nullptr;
        size=0;
    }
    protected void copyArray(Object[] arr)//checked
    {
        int index=0;
        ZNode r=this.root;
        ZLinkedStack<ZNode> s=new ZLinkedStack<>();
        do
        {
            if(r!=nullptr)
            {
                s.add(r);
                r=r.pleft;
            }
            else
            {
                arr[index++]=r=s.pop();
                r=r.pright;
            }
        }
        while(!s.isEmpty()||r!=nullptr);
    }
    @Override
    public Object[] toArray()//checked
    {
        Object[] arr=new Object[this.size];
        this.copyArray(arr);
        return arr;
    }
    @Override
    public <E> E[] toArray(E[] a)//checked
    {
        if(a.length<size)
            a=(E[]) Array.newInstance(a.getClass().getComponentType(), size);
        this.copyArray(a);
        if(a.length>size) a[size]=null;
        return a;
    }
    @Override
    public void forEach(Consumer<? super T> con) 
    {
        if(this.isEmpty()) return;
        this.inOrderTraverse(root, con, new ZLinkedStack<>());
    }
    @Override
    public ZTreeSet<T> clone()
    {
        ZTreeSet<T> atree=new ZTreeSet<>();
        if(this.isEmpty()) return atree;
        atree.root=new ZNode(root.key);
        atree.size=this.copyTree(root, atree.root);
        return atree;
    }
    public ZTreeSet<T> tailMap(T fromKey, boolean inclusive)
    {
        ZTreeSet atree=new ZTreeSet();
        ZNode<T> r=this.root;
        this.createTailSet(fromKey, inclusive, r, atree, new ZLinkedStack<>());
        return atree;
    }
    public ZTreeSet<T> tailMap(T fromKey)
    {
        ZTreeSet atree=new ZTreeSet();
        ZNode<T> r=this.root;
        this.createTailSet(fromKey, true, r, atree, new ZLinkedStack<>());
        return atree;
    }
    public ZTreeSet<T> headSet(T endKey, boolean inclusive)
    {
        ZTreeSet atree=new ZTreeSet();
        ZNode<T> r=this.root;
        this.createHeadSet(endKey, inclusive, r, atree, new ZLinkedStack<>());
        return atree;
    }
    public ZTreeSet<T> headSet(T endKey)
    {
        ZTreeSet atree=new ZTreeSet();
        ZNode<T> r=this.root;
        this.createHeadSet(endKey, true, r, atree, new ZLinkedStack<>());
        return atree;
    }
    public ZTreeSet<T> subSet(T lowKey, T highKey, boolean inclusive) 
    {
        ZTreeSet atree=new ZTreeSet();
        if(inclusive)
            this.createSubSet(lowKey, highKey, inclusive, root, atree, new ZLinkedStack<>());
        else this.createSubSetNoEqual(lowKey, highKey, root, atree, new ZLinkedStack<>());
        return atree;
    }
    public ZTreeSet<T> subSet(T lowKey, T highKey) 
    {
        ZTreeSet atree=new ZTreeSet();
        this.createSubSet(lowKey, highKey, true, root, atree, new ZLinkedStack<>());
        return atree;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapased" desc="Iterable">
    @Passed
    private final class Iter implements Iterator<T>
    {
        //columns---------------------------------------------------------------
        private final ZLinkedStack<ZNode> s;
        private ZNode<T> cur;
        
        //functions-------------------------------------------------------------
        private Iter()
        {
            s=new ZLinkedStack<>();
            cur=ZTreeSet.this.root;
        }
        @Override
        public boolean hasNext() 
        {
            return !s.isEmpty()||cur!=nullptr;
        }
        @Override
        public T next() 
        {
            for(;cur!=nullptr;s.push(cur),cur=cur.pleft);
            cur=s.pop();
            T value=cur.key;
            cur=cur.pright;
            return value;
        }
        @Override
        public void remove()//when delete, go back to root Node
        {
            if(ZTreeSet.this.isEmpty()) return;
            for(;cur!=nullptr;s.push(cur),cur=cur.pleft);
            ZTreeSet.this.removeNode(cur=s.pop());
            cur=ZTreeSet.this.root;
        }
    }
    @Override
    public Iterator iterator() 
    {
        return new Iter();
    }
    //</editor-fold>
}
