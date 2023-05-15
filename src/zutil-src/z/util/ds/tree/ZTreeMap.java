/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.ds.tree;

import java.util.AbstractCollection;
import java.util.AbstractSet;
import java.util.Collection;
import java.util.Iterator;
import z.util.ds.linear.ZLinkedStack;
import z.util.ds.linear.ZLinkedList;
import java.util.Map;
import java.util.Set;
import java.util.function.BiConsumer;
import java.util.function.BiPredicate;
import java.util.function.Consumer;
import java.util.function.Predicate;
import z.util.ds.imp.Indexable;
import z.util.ds.ZEntry;
import z.util.ds.ZMap;
import z.util.ds.imp.ZStack;
import z.util.lang.exception.IAE;
import z.util.ds.imp.Structurable;
import z.util.ds.linear.ZArrayList;
import z.util.lang.annotation.Passed;

/**
 * <pre>
 ZTreeMap is an implementation of Red-Black Map.
 * We use false to represent black, while true corrsponds to red.
 To use a sub class of ZTreeMap.ZNode, you need to overwrite {@link #createNode(Comparable, Object, ZNode) },
 * by using method {@link #put(Comparable, Object, Consumer)}, you can give corrosponing new
 features to your overridden sub class of ZTreeMap.ZNode.
 </pre>
 * @author dell
 * @param <K>
 * @param <V>
 */
@Passed
public class ZTreeMap<K extends Comparable,V> extends ZMap<K,V> implements Cloneable,Structurable 
{
    //<editor-fold defaultstate="collapsed" desc="static class ZtreeMap.ZNode">
    protected static class ZNode<K1 extends Comparable, V1> extends ZEntry<K1,V1> implements Comparable//checked
    {
        //columns---------------------------------------------------------------
        private boolean color;//red=true,black=false
        private ZNode<K1,V1> pleft;
        private ZNode<K1,V1> pright;
        private ZNode<K1,V1> parent;
                
        //functions-------------------------------------------------------------
        protected ZNode()//only used for nullptr
        {
            color=false;
            pleft=pright=parent=null;
        }
        protected ZNode(K1 key, V1 value)
        {
            super(key,value);
            this.color=true;
            
            this.parent=nullptr;
            this.pleft=nullptr;
            this.pright=nullptr;
        }
        protected ZNode(K1 key, V1 value, ZNode<K1,V1> parent)
        {
            super(key,value);
            this.color=true;
            
            this.parent=parent;
            this.pleft=nullptr;
            this.pright=nullptr;
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="nullptr">
    protected static final ZNode nullptr;
    
    static
    {
        nullptr=new ZNode();
        nullptr.parent=nullptr;
        nullptr.pright=nullptr;
        nullptr.pleft=nullptr;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Iterable">
    @Passed
    private abstract class TreeIter
    {
        //columns---------------------------------------------------------------
        private final ZLinkedStack<ZNode> s;
        private ZNode<K, V> cur;
        
        //functions-------------------------------------------------------------
        TreeIter()
        {
            cur=ZTreeMap.this.root;
            s=new ZLinkedStack<>();
        }
        public final boolean hasNext() 
        {
            return !s.isEmpty()||cur!=nullptr;
        }
        final ZNode<K,V> nextNode()
        {
            for(;cur!=nullptr;s.push(cur),cur=cur.pleft);
            ZNode next=cur=s.pop();
            cur=cur.pright;
            return next;
        }
    }
    private final class KeyIter extends TreeIter implements Iterator<K>
    {
        @Override
        public K next() {return nextNode().key;}
    }
    private final class ValueIter extends TreeIter implements Iterator<V>
    {
        @Override
        public V next() {return nextNode().value;}
    }
    private final class EntryIter extends TreeIter implements Iterator<Entry<K,V>>
    {
        @Override
        public Entry<K,V> next() {return nextNode();}
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="class KeySet">
    @Passed
    protected final class KeySet extends AbstractSet<K>
    {
        @Override
        public int size() {return size;}
        @Override
        public void clear() {ZTreeMap.this.clear();}
        @Override
        public boolean contains(Object key) {return ZTreeMap.this.containsKey(key);}
        @Override
        public Iterator<K> iterator() {return new KeyIter();}
        @Override
        public void forEach(Consumer<? super K> con) 
        {
            ZTreeMap.this.inOrderTraverseKey(ZTreeMap.this.root, con, new ZLinkedStack<>());
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="class ValueCollection">
    @Passed
    protected final class ValueCollection extends AbstractCollection<V>
    {
        @Override
        public int size() {return size;}
        @Override
        public void clear() {ZTreeMap.this.clear();}
        @Override
        public boolean contains(Object value) {return ZTreeMap.this.containsValue(value);}
        @Override
        public Iterator<V> iterator() {return new ValueIter();}
        @Override
        public void forEach(Consumer<? super V> con) 
        {
            ZTreeMap.this.inOrderTraverseValue(ZTreeMap.this.root, con, new ZLinkedStack<>());
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="class EntrySet">
    @Passed
    protected final class EntrySet extends AbstractSet<Entry<K,V>>
    {
        @Override
        public int size() {return size;}
        @Override
        public void clear() {ZTreeMap.this.clear();}
        @Override
        public boolean contains(Object entry) 
        {
            Entry<K,V> kv=(Entry<K,V>) entry;
            return ZTreeMap.this.contains(kv.getKey(), kv.getValue());
        }
        @Override
        public Iterator<Entry<K,V>> iterator() {return new EntryIter();}
        @Override
        public void forEach(Consumer<? super Entry<K, V>> con) 
        {
            ZTreeMap.this.inOrderTraverseEntry(ZTreeMap.this.root, con, new ZLinkedStack<>());
        }
    }
    //</editor-fold>
    protected ZNode<K,V> root;
    protected KeySet keySet;
    protected ValueCollection valueCollection;
    protected EntrySet entrySet;
    
    public ZTreeMap()
    {
        this.size=0;
        root=nullptr;
    }
    //<editor-fold defaultstate="collapsed" desc="Basic-Function">
    @Override
    public int number() 
    {
        return size;
    }
    @Passed
    public int height()
    {
        return this.heightFrom(root);
    }
    @Override
    public boolean isEmpty()//passed
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
                    sb.append(r=s.pop()).append(", ");
                    r=r.pright;
                }
            }
            while(!s.isEmpty()||r!=nullptr);
        }
        if(size>1) sb.setCharAt(sb.length()-2, '}');
        else sb.append('}');
    }
    @Override
    public String toString()//passed
    {
       StringBuilder builder=new StringBuilder();
       this.append(builder);
       return builder.toString();
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
            System.out.println(r+">>>"+deep);
            if(r.pleft!=nullptr) {s.push(r.pleft);d.push(deep+1);}
            if(r.pright!=nullptr) {s.push(r.pright);d.push(deep+1);}
        }
    }
    @Override
    public int getIndexType() 
    {
        return Indexable.TREE;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Inner-Operator">
    public ZNode createNode(K key, V value, ZNode last)
    {
        return new ZNode(key, value, last);
    }
    @Passed
    protected final ZNode<K,V> findNode(Object key)
    {
        ZNode<K,V> r=this.root;
        int v;
        while(r!=nullptr)
        {
            if((v=r.key.compareTo(key))>0) r=r.pleft;
            else if(v<0) r=r.pright;
            else break;
        }
        return r;
    }
    @Passed
    protected final void inOrderTraverseKey(ZNode root, Consumer con, ZStack<ZNode> s)
    {
        ZNode<K,V> r=root;
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
    @Passed
    protected final void inOrderTraverseValue(ZNode root, Consumer con, ZStack<ZNode> s)
    {
        ZNode<K,V> r=root;
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
                con.accept(r.value);
                r=r.pright;
            }
        }
        while(!s.isEmpty()||r!=nullptr);
    }
   @Passed
    protected final void inOrderTraverseEntry(ZNode root, Consumer con, ZStack<ZNode> s)
    {
        ZNode<K,V> r=root;
        do
        {
            if(r!=nullptr)
            {
                s.push(r);
                r=r.pleft;
            }
            else
            {
                con.accept(r=s.pop());
                r=r.pright;
            }
        }
        while(!s.isEmpty()||r!=nullptr);
    }
    @Passed
    protected final void inOrderTraverse(ZNode root, BiConsumer con, ZStack<ZNode> s)
    {
        ZNode<K,V> r=root;
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
                con.accept(r.key, r.value);
                r=r.pright;
            }
        }
        while(!s.isEmpty()||r!=nullptr);
    }
    @Passed
    protected final void transplant(ZNode p, ZNode c)
    {
        ZNode g=p.parent;
        if(g==nullptr) this.root=c;
        else if(g.pleft==p) g.pleft=c;
        else g.pright=c;
        c.parent=g;
    }
    @Passed
    protected final ZNode<K,V> maxNode(ZNode r)
    {
        while(r.pright!=nullptr) r=r.pright;
        return r;
    }
    @Passed
    protected final ZNode<K,V> lastNode(ZNode r)
    {
        if(r.pleft!=nullptr) return maxNode(r.pleft);
        ZNode y=r.parent;
        while(y!=nullptr&&y.pleft==r)
        {
            r=y;
            y=y.parent;
        }
        return y;
    }
    @Passed
    protected final ZNode<K,V> minNode(ZNode r)
    {
        while(r.pleft!=nullptr) r=r.pleft;
        return r;
    }
    @Passed
    protected final ZNode<K,V> nextNode(ZNode r)
    {
        if(r.pright!=nullptr) return (r.pright);
        ZNode y=r.parent;
        while(y!=nullptr&&y.pright==r)
        {
            r=y;
            y=y.parent;
        }
        return y;
    }
    @Passed
    protected void leftRotate(ZNode target)
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
    protected void rightRotate(ZNode target)
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
    protected final void insertFix(ZNode z)
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
    protected final void removeNode(ZNode z)
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
        z.value=null;
        if(x!=nullptr&&!oc) this.removeFix(x);//oc is black
        if(--this.size==0) this.root=nullptr;
    }
    @Override
    protected boolean innerRemoveAll(Predicate pre) 
    {
        ZLinkedList<ZNode> l1=new ZLinkedList<>();
        ZLinkedList<ZNode> l2=new ZLinkedList<>();
        ZNode r=this.root;
        l1.add(r);
        while(!l1.isEmpty())
        {
            r=l1.pop();
            if(pre.test(r.key)) l2.add(r);
            if(r.pleft!=nullptr) l1.add(r.pleft);
            if(r.pright!=nullptr) l1.add(r.pright);
        }
        boolean result=!l2.isEmpty();
        while(!l2.isEmpty()) this.removeNode(l2.pop());
        return result;
    }
    @Override
    protected boolean innerRemoveAll(BiPredicate pre) 
    {
        ZLinkedList<ZNode> l1=new ZLinkedList<>();
        ZLinkedList<ZNode> l2=new ZLinkedList<>();
        ZNode r=this.root;
        l1.add(r);
        while(!l1.isEmpty())
        {
            r=l1.pop();
            if(pre.test(r.key, r.value)) l2.add(r);
            if(r.pleft!=nullptr) l1.add(r.pleft);
            if(r.pright!=nullptr) l1.add(r.pright);
        }
        boolean result=!l2.isEmpty();
        while(!l2.isEmpty()) this.removeNode(l2.pop());
        return result;
    }
    /**
     * @param r the start node of this
     * @param aroot the start node of a new Tree
     * @return  how many node is copied
     */
    @Passed
    protected final int copyTree(ZNode<K,V> r, ZNode<K,V> aroot)
    {
        ZStack<ZNode> s1=new ZLinkedStack<>();
        ZStack<ZNode> s2=new ZLinkedStack<>();
        s1.push(r);
        s2.push(aroot);
        int num=0;
        while(!s1.isEmpty())
        {
            r=s1.pop();
            aroot=s2.pop();
            if(r.pleft!=nullptr) 
            {
                aroot.pleft=new ZNode(r.pleft.key, r.pleft.value, aroot);
                num++;
                s1.push(r.pleft);
                s2.push(aroot.pleft);
            }
            if(r.pright!=nullptr) 
            {
                aroot.pright=new ZNode(r.pright.key, r.pright.value, aroot);
                num++;
                s1.push(r.pright);
                s2.push(aroot.pright);
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
    protected final void createHeadMap(K endKey, boolean inclusive, ZNode<K,V> r, ZTreeMap atree,  ZStack<ZNode> s)
    {
        int v;
        while(r!=nullptr)
        {
            v=r.key.compareTo(endKey);
            if(v>0) r=r.pleft;
            else if(v<0)
            {
                atree.put(r.key, r.value);
                if(r.pleft!=nullptr) this.inOrderTraverse(r.pleft, atree::put, s);
                r=r.pright;
            }
            else
            {
                if(inclusive) atree.put(r.key, r.value);
                if(r.pleft!=nullptr) this.inOrderTraverse(r.pleft, atree::put, s);
                return;
            }
        }
    }
    @Passed
    protected final void createTailMap(K fromKey, boolean inclusive, ZNode<K,V> r, ZTreeMap atree, ZStack<ZNode> s)
    {
        int v;
        while(r!=nullptr)
        {
            v=r.key.compareTo(fromKey);
            if(v<0) r=r.pright;
            else if(v>0)
            {
                atree.put(r.key, r.value);
                if(r.pright!=nullptr) this.inOrderTraverse(r.pright, atree::put, s);
                r=r.pleft;
            }
            else
            {
                if(inclusive) atree.put(r.key, r.value);
                if(r.pright!=nullptr) this.inOrderTraverse(r.pright, atree::put, s);
                return;
            }
        }
    }
    @Passed
    protected final void createSubMapNoEqual(K lowKey, K highKey, ZNode<K,V> r, ZTreeMap<K,V> atree, ZStack<ZNode> s)
    {
        s.push(r);
        while(!s.isEmpty())
        {
            r=s.pop();
            while(r!=nullptr)
            {
                if(r.key.compareTo(lowKey)<0) {r=r.pright;continue;}//r.key<lowKey
                if(r.key.compareTo(highKey)>0) {r=r.pleft;continue;}//r.key>highKey
                atree.put(r.key, r.value);
                if(r.pleft!=nullptr) s.push(r.pleft);
                if(r.pright!=nullptr) s.push(r.pright);
                break;
            }
        }
    }
    @Passed
    protected final void createSubMap(K lowKey, K highKey, boolean inclusive, ZNode<K,V> r, ZTreeMap<K,V> atree, ZStack<ZNode> s)
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
                    if(inclusive) atree.put(r.key, r.value);
                    this.createSubMapNoEqual(lowKey, highKey, r.pright, atree, s);
                    return;
                }
                if(v2==0) 
                {
                    if(inclusive) atree.put(r.key, r.value);
                    this.createSubMapNoEqual(lowKey, highKey, r.pleft, atree, s);
                    return;
                }
                atree.put(r.key, r.value);
                if(r.pleft!=nullptr) s.push(r.pleft);
                if(r.pright!=nullptr) s.push(r.pright);
                break;
            }
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Outer-Operator">
    @Override
    public boolean containsKey(Object key)
    {
        return this.findNode(key)!=nullptr;
    }
    @Override
    public boolean containsValue(Object value)
    {
        ZLinkedStack<ZNode> s=new ZLinkedStack<>();
        ZNode r=this.root;
        if(value==null)
        {
            do
            {
                if(r!=nullptr) s.push(r=r.pleft);
                else
                {
                    r=s.pop();
                    if(r.value==value) return true;
                    r=r.pright;
                }
            }
            while(!s.isEmpty()||r!=nullptr);
        }
        else
        {
            do
            {
                if(r!=nullptr) s.push(r=r.pleft);
                else
                {
                    r=s.pop();
                    if(r.value.equals(value)) return true;
                    r=r.pright;
                }
            }
            while(!s.isEmpty()||r!=nullptr);
        }
        return false;
    }
    @Override
    public boolean contains(Object key, Object value)
    {
        ZNode r=this.findNode(key);
        if(r==nullptr) return false;
        if(value==null) return r.value==null;
        return value.equals(r.value);
    }
    @Override
    public V put(K key, V value)
    {
        if(key==null) throw new NullPointerException();
        ZNode<K,V> r = this.root, last = nullptr;
        int v;
        while(r!=nullptr)
        {
            last=r;
            if((v=r.key.compareTo(key))>0) r=r.pleft;
            else if(v<0) r=r.pright;
            else return r.setValue(value);
        }
        r=this.createNode(key, value, last);
        if(last==nullptr) this.root=r;
        else if(last.key.compareTo(key)>0) last.pleft=r;
        else last.pright=r;
        this.insertFix(r);
        this.size++;
        return null;
    }
    @Passed
    public V put(K key, V value, Consumer con)
    {
        if(key==null) throw new NullPointerException();
        ZNode<K, V> r = this.root;
        ZNode<K,V> last = nullptr;
        int v;
        while(r!=nullptr)
        {
            last=r;
            if((v=r.key.compareTo(key))>0) r=r.pleft;
            else if(v<0) r=r.pright;
            else return r.setValue(value);
        }
        r=this.createNode(key, value, last);
        con.accept(r);
        if(last==nullptr) this.root=r;
        else if(last.key.compareTo(key)>0) last.pleft=r;
        else last.pright=r;
        this.insertFix(r);
        this.size++;
        return null;
    }
    @Override
    public void putAll(Map<? extends K, ? extends V> m) //optimizable
    {
        m.forEach((key, value) -> this.put(key, value));
    }
    public void putAll(K[] keys, V[] values)
    {
        if(keys.length!=values.length) throw new IAE("keys.length != values.length");
        for(int i=0;i<keys.length;i++) this.put(keys[i], values[i]);
    }
    public void putAllIf(Map<? extends K, ? extends V> m, BiPredicate pre) //optimizable
    {
        m.forEach((key, value) ->  {
            if(pre.test(key, value))  this.put(key, value);
        });
    }
    public void putAllIf(K[] keys, V[] values, BiPredicate pre)
    {
        if(keys.length!=values.length) throw new IAE("keys.length != values.length");
        for(int i=0;i<keys.length;i++) 
            if(pre.test(keys[i], values)) this.put(keys[i], values[i]);
    }
    @Override
    public V remove(Object key)
    {
        if(key==null) throw new NullPointerException();
        ZNode<K,V> r=this.findNode(key);
        if(r==nullptr) return null;
        V value=r.value;
        this.removeNode(r);
        return value;
    }
    public Entry<K,V> maxFactor()
    { 
        return this.isEmpty()? null:this.maxNode(root);
    }
    public Entry<K,V> minFactor()
    {
        return this.isEmpty()? null:this.minNode(root);
    }
    public K firstKey() 
    {
        return this.isEmpty()? null:this.minNode(root).key;
    }
    public K lastKey() 
    {
        return this.isEmpty()? null:this.maxNode(root).key;
    }
    @Override
    public V get(Object key)
    {
        return this.findNode(key).value;
    }
    public Entry<K,V> next(K key)
    {
        return this.nextNode(this.findNode(key));
    }
    public Entry<K,V> last(K key)
    {
        return this.lastNode(this.findNode(key));
    }
    @Override
    public Set<K> keySet()
    {
        if(this.keySet==null) this.keySet=new KeySet();
        return this.keySet;
    }
    @Override
    public ZTreeSet<K> replicaKeySet()
    {
        ZTreeSet<K> set= new ZTreeSet<>();
        ZLinkedStack<ZNode<K,V>> s=new ZLinkedStack<>();
        ZNode<K,V> r=this.root;
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
                set.add(r.key);
                r=r.pright;
            }
        }
        while(!s.isEmpty()||r!=nullptr);
        return set;
    }
    @Override
    public Collection<V> values()
    {
        if(this.valueCollection==null) this.valueCollection=new ValueCollection();
        return this.valueCollection;
    }
    @Override
    public ZArrayList<V> replicaValues()
    {
        if(this.isEmpty()) return null;
        ZArrayList<V> arr=new ZArrayList<>(size);
        ZLinkedStack<ZNode<K,V>> s=new ZLinkedStack<>();
        ZNode<K,V> r=this.root;
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
                arr.add(r.value);
                r=r.pright;
            }
        }
        while(!s.isEmpty()||r!=nullptr);
        return arr;
    }
    @Override
    public Set<Entry<K,V>> entrySet()
    {
        if(this.entrySet==null) this.entrySet=new EntrySet();
        return this.entrySet;
    }
    @Override
    public ZTreeSet<? extends ZEntry<K,V>> replicaEntrySet()
    {
        ZTreeSet<ZNode<K,V>> set=new ZTreeSet<>();
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
                set.add(r=s.pop());
                r=r.pright;
            }
        }
        while(!s.isEmpty()||r!=nullptr);
        return set;
    }
    @Override
    public void clear()
    {
        if(this.isEmpty()) return;//the points of nullptr can't be changed
        ZNode r=this.root;
        ZLinkedList<ZNode> l=new ZLinkedList<>();
        l.add(r);
        while(!l.isEmpty())
        {
            r=l.remove();
            if(r.pleft!=nullptr) l.add(r.pleft);
            if(r.pright!=nullptr) l.add(r.pright);
            r.parent=r.pleft=r.pright=null;
            r.key=null;
            r.value=null;
        }
        root=nullptr;
        size=0;
    }
    @Override
    public void forEachKey(Consumer<? super K> con)
    {
        if(this.isEmpty()) return;
        this.inOrderTraverseKey(root, con, new ZLinkedStack<>());
    }
    @Override
    public void forEachValue(Consumer<? super V> con) 
    {
         if(this.isEmpty()) return;
        this.inOrderTraverseValue(root, con, new ZLinkedStack<>());
    }
    @Override
    public void forEachEntry(Consumer<? super Entry<K, V>> con)
    {
         if(this.isEmpty()) return;
        this.inOrderTraverseEntry(root, con, new ZLinkedStack<>());
    }
    @Override
    public void forEach(BiConsumer<? super K, ? super V> con)
    {
        if(this.isEmpty()) return;
        this.inOrderTraverse(root, con, new ZLinkedStack<>());
    }
    @Override
    protected ZTreeMap<K,V> clone() 
    {
        ZTreeMap<K,V> atree=new ZTreeMap<>();
        if(this.isEmpty()) return atree;
        atree.root=new ZNode(root.key, root.value);
        atree.size=this.copyTree(root, atree.root);
        return atree;
    }
     /**
     * give all Entries in the Map that greater than to endKey. if inclusive is
     * tree: we will also add the Entry equal to endKey to the returned
     * ZTreeMap.
     * @param fromKey
     * @param inclusive
     * @return
     */
    public ZTreeMap<K,V> tailMap(K fromKey, boolean inclusive)
    {
        ZTreeMap atree=new ZTreeMap();
        ZNode<K,V> r=this.root;
        this.createTailMap(fromKey, inclusive, r, atree, new ZLinkedStack<>());
        return atree;
    }
    public ZTreeMap<K,V> tailMap(K fromKey)
    {
        ZTreeMap atree=new ZTreeMap();
        ZNode<K,V> r=this.root;
        this.createTailMap(fromKey, true, r, atree, new ZLinkedStack<>());
        return atree;
    }
    /**
     * give all Entries in the Map that less than to endKey. if inclusive is
     * tree: we will also add the Entry equal to endKey to the returned
     * ZTreeMap.
     * @param endKey
     * @param inclusive
     * @return
     */
    public ZTreeMap<K,V> headMap(K endKey, boolean inclusive)
    {
        ZTreeMap atree=new ZTreeMap();
        ZNode<K,V> r=this.root;
        this.createHeadMap(endKey, inclusive, r, atree, new ZLinkedStack<>());
        return atree;
    }
    public ZTreeMap<K,V> headMap(K endKey)
    {
         ZTreeMap atree=new ZTreeMap();
        ZNode<K,V> r=this.root;
        this.createHeadMap(endKey, true, r, atree, new ZLinkedStack<>());
        return atree;
    }
    public ZTreeMap<K, V> subMap(K lowKey, K highKey, boolean inclusive) 
    {
        ZTreeMap atree=new ZTreeMap();
        if(inclusive)
            this.createSubMap(lowKey, highKey, inclusive, root, atree, new ZLinkedStack<>());
        else this.createSubMapNoEqual(lowKey, highKey, root, atree, new ZLinkedStack<>());
        return atree;
    }
    public ZTreeMap<K, V> subMap(K lowKey, K highKey) 
    {
        ZTreeMap atree=new ZTreeMap();
        this.createSubMap(lowKey, highKey, true, root, atree, new ZLinkedStack<>());
        return atree;
    }
    //</editor-fold>
}
