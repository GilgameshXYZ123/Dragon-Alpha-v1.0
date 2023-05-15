/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.ds.heap;

import java.util.AbstractCollection;
import java.util.AbstractSet;
import java.util.Collection;
import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.function.BiConsumer;
import java.util.function.BiPredicate;
import java.util.function.Consumer;
import java.util.function.Predicate;
import z.util.ds.ZEntry;
import z.util.ds.ZMap;
import z.util.ds.imp.Indexable;
import z.util.ds.imp.ZHeap;
import z.util.ds.imp.ZStack;
import z.util.ds.linear.ZArrayList;
import z.util.ds.linear.ZLinkedList;
import z.util.ds.linear.ZLinkedStack;
import z.util.ds.tree.ZTreeSet;
import z.util.lang.annotation.Passed;

/**
 * ZLeftTreeHeapMap is an impelemtation of Min-Heap based ont Left-Tree, 
 * it doesn't Support Null Key, as it's a multi-map, means one key may
 * corrospond to mutl-values.
 * @author dell
 * @param <K>
 * @param <V>
 */
@Passed
public class ZLeftTreeHeapMap<K extends Comparable, V> extends ZMap<K,V> implements ZHeap<Entry<K,V>>
{
    //<editor-fold defaultstate="collapsed" desc="static class ZNode<K1,V1>">
    private static final class ZNode<K1 extends Comparable, V1> extends ZEntry<K1,V1>
    {
       //columns----------------------------------------------------------------
        private ZNode pleft;
        private ZNode pright;
        private ZNode parent;
        private int dist;
        
        //functions-------------------------------------------------------------
        public ZNode(K1 key, V1 value)
        {
            super(key, value);
            this.parent=null;
            this.pleft=null;
            this.pright=null;
            this.dist=0;
        }
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
            cur=ZLeftTreeHeapMap.this.root;
            s=new ZLinkedStack<>();
            s.push(cur);
        }
        public final boolean hasNext() 
        {
            return !s.isEmpty();
        }
        final ZNode<K,V> nextNode()
        {
            cur=s.pop();
            if(cur.pleft!=null)
            {
                s.push(cur.pleft);
                if(cur.pright!=null) s.push(cur.pright);
            }
            return cur;
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
        public void clear() {ZLeftTreeHeapMap.this.clear();}
        @Override
        public boolean contains(Object key) {return ZLeftTreeHeapMap.this.containsKey(key);}
        @Override
        public Iterator<K> iterator() {return new KeyIter();}
        @Override
        public void forEach(Consumer<? super K> con) 
        {
            ZLeftTreeHeapMap.this.forEachKey(con);
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
        public void clear() {ZLeftTreeHeapMap.this.clear();}
        @Override
        public boolean contains(Object value) {return ZLeftTreeHeapMap.this.containsValue(value);}
        @Override
        public Iterator<V> iterator() {return new ValueIter();}
        @Override
        public void forEach(Consumer<? super V> con) 
        {
            ZLeftTreeHeapMap.this.forEachValue(con);
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
        public void clear() {ZLeftTreeHeapMap.this.clear();}
        @Override
        public boolean contains(Object entry) 
        {
            Entry<K,V> kv=(Entry<K,V>) entry;
            return ZLeftTreeHeapMap.this.contains(kv.getKey(), kv.getValue());
        }
        @Override
        public Iterator<Entry<K,V>> iterator() {return new EntryIter();}
        @Override
        public void forEach(Consumer<? super Entry<K, V>> con) 
        {
            ZLeftTreeHeapMap.this.forEachEntry(con);
        }
    }
    //</editor-fold>
    private ZNode root;
    protected KeySet keySet;
    protected ValueCollection valueCollection;
    protected EntrySet entrySet;
    
    public ZLeftTreeHeapMap()
    {
        this.size=0;
        root=null;
    }
    //<editor-fold defaultstate="collapsed" desc="Basic-Function">
     @Override
    public int number()
    {
        return size;
    }
    @Override
    public int getIndexType()
    {
        return Indexable.HEAP;
    }
    @Passed
    public void append(StringBuilder sb)
    {
        sb.append('{');
        if(!this.isEmpty())
        {
            ZNode r=this.root;
            ZStack<ZNode> s=new ZLinkedStack<>();
            s.push(r);
            while(!s.isEmpty())
            {
                sb.append(r=s.pop()).append(", ");
                if(r.pleft!=null)//if the parent node has no pleft, it must have no right nide
                {
                    s.push(r.pleft);
                    if(r.pright!=null) s.push(r.pright);
                }
            }
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
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Inner-Operator">
    @Passed
    private static ZNode merge(ZNode ra, ZNode rb)
    {
        if(ra==null) return rb;
        if(rb==null) return ra;
        ZNode t;
        if(ra.compareTo(rb)>0) {t=ra;ra=rb;rb=t;}
        ra.pright=merge(ra.pright, rb);
        ra.pright.parent=ra;
        
        if((ra.pright!=null? ra.pright.dist:-1)>(ra.pleft!=null? ra.pleft.dist:-1))
            {t=ra.pleft;ra.pleft=ra.pright;ra.pright=t;}
        ra.dist=(ra.pright!=null? ra.pright.dist+1:0);//-1+1=0
        return ra;
    }
    @Passed
    private ZNode<K, V> findNode(Object key)
    {
        if(this.isEmpty()) return null;
        ZStack<ZNode> s=new ZLinkedStack<>();
        ZNode<K,V> r=this.root;
        s.push(r);
        int v;
        while(!s.isEmpty())
        {
            r=s.pop();
            v=r.key.compareTo(key);
            if(v>0) {}//less than the parent
            else if(v<0)
            {
                if(r.pleft!=null) 
                {
                    s.push(r.pleft);
                    if(r.pright!=null) s.push(r.pright);
                }
            }
            else return r;
        }
        return null;
    }
    @Passed
    private void transplant(ZNode p, ZNode c)
    {
        ZNode g=p.parent;
        if(g==null) this.root=c;
        else if(g.pleft==p) g.pleft=c;
        else g.pright=c;
        if(c!=null) c.parent=g;
    }
    @Passed
    private void removeNode(ZNode x)
    {
        ZNode g=x.parent, p=merge(x.pright, x.pleft),t ;
        this.transplant(x, p);//the q is the parent of newly merged tree
        int dright;
        while(g!=null)
        {
            if((g.pleft==null? -1:g.pleft.dist)<(g.pright==null? -1:g.pright.dist))
                {t=g.pleft;g.pleft=g.pright;g.pright=t;}
            //the left and right may be exchanged
            dright=(g.pright==null? -1:g.pright.dist);
            if(g.dist==dright+1) break;
            g.dist=dright+1;
            p=g;//set p the grand parent
            g=g.parent;
        }
        this.size--;
    }
    @Override
    protected boolean innerRemoveAll(Predicate pre)//passed
    {
        ZLinkedList<ZNode> l1=new ZLinkedList<>();
        ZLinkedList<ZNode> l2=new ZLinkedList<>();
        ZNode r=this.root;
        l1.add(r);
        while(!l1.isEmpty())
        {
            r=l1.pop();
            if(pre.test(r.key)) l2.add(r);
            if(r.pleft!=null)
            {
                l1.add(r.pleft);
                if(r.pright!=null) l1.add(r.pright);
            }
        }
        boolean result=!l2.isEmpty();
        while(!l2.isEmpty()) this.removeNode(l2.pop());
        return result;
    }
    @Override
    protected boolean innerRemoveAll(BiPredicate pre)//passed
    {
         ZLinkedList<ZNode> l1=new ZLinkedList<>();
        ZLinkedList<ZNode> l2=new ZLinkedList<>();
        ZNode r=this.root;
        l1.add(r);
        while(!l1.isEmpty())
        {
            r=l1.pop();
            if(pre.test(r.key, r.value)) l2.add(r);
            if(r.pleft!=null)
            {
                l1.add(r.pleft);
                if(r.pright!=null) l1.add(r.pright);
            }
        }
        boolean result=!l2.isEmpty();
        while(!l2.isEmpty()) this.removeNode(l2.pop());
        return result;
    } 
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Outer-Operator">
    @Override
    public V put(K key, V value)//passed
    {
        if(key==null) throw new NullPointerException();
        this.root=merge(this.root, new ZNode(key, value));
        this.size++;
        return null;
    }
    @Override
    public void putAll(Map<? extends K, ? extends V> m)//passed
    {
        if(m==null) throw new NullPointerException();
        if(m instanceof ZLeftTreeHeapMap)
        {
            ZLeftTreeHeapMap zm=(ZLeftTreeHeapMap) m;
            this.root=merge(this.root, zm.root);
            this.size+=zm.size;
            return;
        }
        m.forEach((K key, V value)->
        {
            if(key==null) return;
            this.root=merge(this.root, new ZNode(key, value));
            this.size++;
        });
    }
    @Override
    public V get(Object key)//passed
    {
        return (this.isEmpty()? null: this.findNode(key).value);
    }
    @Override
    public V remove(Object key)//passed
    {
        if(this.isEmpty()) return null;
        ZNode<K,V> r=this.findNode(key);
        if(r==null) return null;
        V value=r.value;
        this.removeNode(r);
        return value;
    }
    @Override
    public boolean containsKey(Object key)//passed
    {
        return (this.isEmpty()? false: this.findNode(key)!=null);
    }
    @Override
    public boolean containsValue(Object value)//passed
    {
        if(this.isEmpty()) return false;
        ZStack<ZNode> s=new ZLinkedStack<>();
        ZNode<K,V> r=this.root;
        s.push(r);
        if(value==null)
        {
            while(!s.isEmpty())
            {
                r=s.pop();
                if(r.value==null) return true;
                if(r.pleft!=null) 
                {
                    s.push(r.pright);
                    if(r.pright!=null) s.push(r.pleft);
                }
            }
        }
        else
        {
            while(!s.isEmpty())
            {
                r=s.pop();
                if(value.equals(r.value)) return true;
                if(r.pleft!=null) 
                {
                    s.push(r.pright);
                    if(r.pright!=null) s.push(r.pleft);
                }
            }
        }
        return false;
    }
    @Override
    public boolean contains(Object key, Object value)//passed
    {
        if(this.isEmpty()) return false;
        ZNode<K,V> r=this.findNode(key);
        if(r==null) return false;
        if(value==null) return r.value==null;
        else return value.equals(r.value);
    }
    @Override
    public void forEachKey(Consumer<? super K> con)//passed
    {
        if(this.isEmpty()) return;
        ZStack<ZNode> s=new ZLinkedStack<>();
        ZNode<K,V> r=this.root;
        s.push(r);
        while(!s.isEmpty())
        {
            r=s.pop();
            if(r.pleft!=null) 
            {
                s.push(r.pleft);
                if(r.pright!=null) s.push(r.pright);
            }
            con.accept(r.key);
        }
    }
    @Override
    public void forEachValue(Consumer<? super V> con)//passed
    {
        if(this.isEmpty()) return;
        ZStack<ZNode> s=new ZLinkedStack<>();
        ZNode<K,V> r=this.root;
        s.push(r);
        while(!s.isEmpty())
        {
            r=s.pop();
            if(r.pleft!=null) 
            {
                s.push(r.pleft);
                if(r.pright!=null) s.push(r.pright);
            }
            con.accept(r.value);
        }
    }
    @Override
    public void forEachEntry(Consumer<? super Entry<K, V>> con)//passed
    {
        if(this.isEmpty()) return;
        ZStack<ZNode> s=new ZLinkedStack<>();
        ZNode<K,V> r=this.root;
        s.push(r);
        while(!s.isEmpty())
        {
            r=s.pop();
            if(r.pleft!=null) 
            {
                s.push(r.pleft);
                if(r.pright!=null) s.push(r.pright);
            }
            con.accept(r);
        }
    }
    @Override
    public void forEach(BiConsumer<? super K, ? super V> con)
    {
         if(this.isEmpty()) return;
        ZStack<ZNode> s=new ZLinkedStack<>();
        ZNode<K,V> r=this.root;
        s.push(r);
        while(!s.isEmpty())
        {
            r=s.pop();
            if(r.pleft!=null) 
            {
                s.push(r.pleft);
                if(r.pright!=null) s.push(r.pright);
            }
            con.accept(r.key, r.value);
        }
    }
    @Override
    public void clear()//passed
    {
        if(this.isEmpty()) return;
        ZNode r=this.root;
        ZLinkedList<ZNode> l=new ZLinkedList<>();
        l.add(r);
        while(!l.isEmpty())
        {
            r=l.remove();
            if(r.pleft!=null) 
            {
                l.add(r.pleft);
                if(r.pright!=null) l.add(r.pright);
            }
            r.parent=r.pleft=r.pright=null;
            r.key=null;
            r.value=null;
        }
        this.root=null;
        this.size=0;
    }
    @Override
    public Set<K> keySet()
    {
        if(this.keySet==null) this.keySet=new KeySet();
        return this.keySet;
    }
    @Override
    public ZTreeSet<K> replicaKeySet()//passed
    {
        ZTreeSet<K> set=new ZTreeSet<>();
        ZStack<ZNode<K,V>> l=new ZLinkedStack<>();
        ZNode<K,V> r=this.root;
        l.push(r);
        while(!l.isEmpty())
        {
            r=l.pop();
            set.add(r.key);
            if(r.pleft!=null)
            {
                l.push(r.pleft);
                if(r.pright!=null) l.push(r.pright);
            }
        }
        return set;
    }
    @Override
    public Collection<V> values()//passed
    {
        if(this.valueCollection==null) this.valueCollection=new ValueCollection();
        return this.valueCollection;
    }
    @Override
    public ZArrayList<V> replicaValues()//passed
    {
        ZArrayList<V> arr=new ZArrayList<>(size);
        ZNode<K,V> r=this.root;
        ZLinkedStack<ZNode> s=new ZLinkedStack<>();
        s.push(r);
        while(!s.isEmpty())
        {
            r=s.pop();
            if(r.pleft!=null) 
            {
                s.push(r.pleft);
                if(r.pright!=null) s.push(r.pright);
            }
            arr.add(r.value);
        }
        return arr;
    }
    @Override
    public Set<Entry<K, V>> entrySet()//passed
    {
        if(this.entrySet==null) this.entrySet=new EntrySet();
        return this.entrySet;
    }
    @Override
    public ZTreeSet<? extends ZEntry<K, V>> replicaEntrySet()//passed
    {
        ZTreeSet<ZNode<K,V>> set=new ZTreeSet<>();
        ZStack<ZNode<K,V>> l=new ZLinkedStack<>();
        ZNode<K,V> r=this.root;
        l.push(r);
        while(!l.isEmpty())
        {
            set.add(r=l.pop());
            if(r.pleft!=null)
            {
                l.push(r.pleft);
                if(r.pright!=null) l.push(r.pright);
            }
        }
        return set;
    }
    @Override
    public Entry<K, V> findMin()//passed
    {
        return root;
    }
    @Override
    public Entry<K, V> removeMin()//passed
    {
        if(this.isEmpty()) return null;
        ZNode left=root.pleft, right=root.pright;
        right=merge(right, left);
        root.pleft=null;
        root.pright=null;
        this.root=right;
        this.size--;
        return root;
    }
    //</editor-fold>
}
