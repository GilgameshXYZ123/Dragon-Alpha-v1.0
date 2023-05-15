/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.ds.hash;

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
import static z.util.ds.Constants.Hash.DEF_EXPEND_RATE;
import static z.util.ds.Constants.Hash.DEF_EXPEND_THRESHOLD;
import static z.util.ds.Constants.Hash.DEF_INIT_SIZE;
import static z.util.ds.Constants.Hash.DEF_SHRINK_THRESHOLD;
import z.util.ds.ZEntry;
import z.util.ds.ZMap;
import z.util.lang.exception.IAE;
import z.util.ds.imp.Indexable;
import z.util.ds.imp.Structurable;
import z.util.ds.linear.ZArrayList;
import z.util.math.ExMath.HashCoder;
import static z.util.math.ExMath.DEF_HASHCODER;
import z.util.lang.annotation.Passed;

/**
 * <pre>
 * Support null key.
 * To talk about the Alogorithm. null {@code
 * (1)put(K,V): if(++num>=up) this.expend();
 * (2)remove(K): if(--num<=down) this.shrink();
 * (3)expend():
 * size*=expendRate;
 * up=size*expendThreshold;
 * down=ize*shrinkThreshold;
 * insert all values to new HashTable;
 * (4)shrink():
 * size*=shrinkRate
 * if(size<DEF_INIT_SIZE) size=DEF_INIT_SIZE;
 * up=size*expendThreshold;
 * down=ize*shrinkThreshold;
 * insert all values to new HashTable
 * }
 *</pre>
 * @author dell
 * @param <K>
 * @param <V>
 */
@Passed
public class ZLinkedHashMap<K extends Comparable,V>  extends ZMap<K,V> implements Structurable
{
    //<editor-fold defaultstate="collapsed" desc="static class ZLinkedHashMap.ZNode<K,V>">
    protected static class ZNode<K1 extends Comparable, V1> extends ZEntry<K1,V1>
    {
        //columns---------------------------------------------------------------
        protected int hash;
        protected ZNode pnext;
        
        //constructors----------------------------------------------------------
        protected ZNode(){}
        protected ZNode(K1 key, V1 value, int hash)
        {
            super(key, value);
            this.hash=hash;
            this.pnext=null;
        }
        protected ZNode(ZNode<K1,V1> node)
        {
            super(node.key, node.value);
            this.hash=node.hash;
            this.pnext=null;
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Iterable">
    private abstract class HashIter
    {
        //columns---------------------------------------------------------------
        private int index=0;
        private ZNode phead;
        
        //functions-------------------------------------------------------------
        HashIter()
        {
            index=0;
            phead=ZLinkedHashMap.this.buckets[0];
        }
        public final boolean hasNext() 
        {
            if(phead!=null) return true;
            while(++index<size&&buckets[index]==null);
            if(index>=size) return false;
            phead=buckets[index];
            return true;
        }
        ZNode<K,V> nextNode()
        {
            ZNode<K,V> last=phead;
            phead=phead.pnext;
            return last;
        }
    }
    private final class KeyIter extends HashIter implements Iterator<K>
    {
        @Override
        public K next() {return this.nextNode().key;}
    }
    private final class ValueIter extends HashIter implements Iterator<V>
    {
        @Override
        public V next() {return this.nextNode().value;}
    }
    private final class EntryIter extends HashIter implements Iterator<Entry<K,V>>
    {
        @Override
        public Entry<K, V> next() {return this.nextNode();}
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="class KeySet">
    @Passed
    protected class KeySet extends AbstractSet<K>
    {
        @Override
        public int size() {return size;}
        @Override
        public void clear() {ZLinkedHashMap.this.clear();}
        @Override
        public boolean contains(Object key) {return ZLinkedHashMap.this.containsKey(key);}
        @Override
        public boolean remove(Object key) {return ZLinkedHashMap.this.remove(key)!=null;}
        @Override
        public Iterator<K> iterator() {return new KeyIter();}
        @Override
        public void forEach(Consumer<? super K> con) 
        {
            ZLinkedHashMap.this.forEachKey(con);
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="class ValueCollection">
    @Passed
    protected class ValueCollection extends AbstractCollection<V>
    {
        @Override
        public int size() {return size;}
        @Override
        public void clear() {ZLinkedHashMap.this.clear();}
        @Override
        public boolean contains(Object value) {return ZLinkedHashMap.this.containsValue(value);}
        @Override
        public Iterator<V> iterator() {return new ValueIter();}
        @Override
        public void forEach(Consumer<? super V> con) 
        {
            ZLinkedHashMap.this.forEachValue(con);
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="class EntrySet">
    @Passed
    protected class EntrySet extends AbstractSet<Entry<K,V>>
    {
        @Override
        public int size() {return size;}
        @Override
        public void clear() {ZLinkedHashMap.this.clear();}
        @Override
        public boolean contains(Object entry) 
        {
            Entry<K,V> kv=(Entry<K,V>) entry;
            return ZLinkedHashMap.this.contains(kv.getKey(), kv.getValue());
        }
        @Override
        public boolean remove(Object key) 
        {
            return ZLinkedHashMap.this.remove(key)!=null;
        }
        @Override
        public Iterator<Entry<K,V>> iterator() {return new EntryIter();}
        @Override
        public void forEach(Consumer<? super Entry<K, V>> con) 
        {
        }
    }
    //</editor-fold>
  
    protected static final ZNode[] EMPTY_DATA={};
    
    protected final double expendThreshold;
    protected final double shrinkThreshold;
    
    protected final double expandRate;
    protected final double shrinkRate;//as shrinkRate=1/expendRate
    
    protected final HashCoder coder;
    
    protected final int initSize;
    protected int up;
    protected int down;
    protected int num;
    protected boolean unInit;
    
    protected ZNode<K,V>[] buckets; 
    
    protected KeySet keySet;
    protected ValueCollection valueCollection;
    protected EntrySet entrySet;
    
    @Passed
    public ZLinkedHashMap(int initSize, double expendThreshold, double shrinkThreshold, 
            double expendRate, HashCoder coder)
    {
        if(initSize<=0) throw new IAE("initSize must positive");
        if(expendThreshold>=1||expendThreshold<=0)
                throw new IAE("(expendThreshold must be between 0 and 1");
        if(shrinkThreshold>=1||shrinkThreshold<=0)
                throw new IAE("shrinkThreshold must be between 0 and 1");
        
        this.initSize=(initSize>DEF_INIT_SIZE? initSize:DEF_INIT_SIZE);
        
        if(expendThreshold<=shrinkThreshold*expendRate)
        {
            this.expendThreshold=DEF_EXPEND_THRESHOLD;
            this.shrinkThreshold=DEF_SHRINK_THRESHOLD;
        }
        else
        {
            this.expendThreshold=expendThreshold;
            this.shrinkThreshold=shrinkThreshold;
        }
        
        this.expandRate=(expendRate>1? expendRate: DEF_EXPEND_RATE);
        this.shrinkRate=1/this.expandRate;
        
        this.coder=(coder!=null? coder:DEF_HASHCODER);
        
        this.num=0;
        this.size=this.down=this.up=0;
        this.unInit=true;
        this.buckets=EMPTY_DATA;
    }
    public ZLinkedHashMap(int intiSize, HashCoder coder)
    {
        this(intiSize, DEF_EXPEND_THRESHOLD, DEF_SHRINK_THRESHOLD, DEF_EXPEND_RATE, coder);
    }
    public ZLinkedHashMap(int initSize)
    {
        this(initSize, DEF_EXPEND_THRESHOLD, DEF_SHRINK_THRESHOLD, DEF_EXPEND_RATE, DEF_HASHCODER);
    }
    public ZLinkedHashMap(HashCoder coder)
    {
        this(DEF_INIT_SIZE, DEF_EXPEND_THRESHOLD, DEF_SHRINK_THRESHOLD, DEF_EXPEND_RATE, coder);
    }
    public ZLinkedHashMap()
    {
        this(DEF_INIT_SIZE, DEF_EXPEND_THRESHOLD, DEF_SHRINK_THRESHOLD, DEF_EXPEND_RATE, DEF_HASHCODER);
    }
    //<editor-fold defaultstate="collapsed" desc="Basic-Function">
    @Override
    public int number() 
    {
        return num;
    }
    @Passed
    public void append(StringBuilder sb)
    {
        sb.append('{');
        if(!this.isEmpty())
        {
            ZNode<K,V> phead=null;
            for(int i=0;i<size;i++)
                for(phead=buckets[i];phead!=null;phead=phead.pnext)
                    sb.append(phead).append(", ");
        }
        if(num>1) sb.setCharAt(sb.length()-2, '}');
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
    public void struc() 
    {
        ZNode<K,V> phead=null;
        for(int i=0;i<size;i++)
        {
            if((phead=buckets[i])==null) continue;
            System.out.print(i+":>>");
            for(;phead!=null;phead=phead.pnext)
                System.out.print("["+phead+"] ");
             System.out.println();
        }
    }
    @Override
    public int getIndexType() 
    {
        return Indexable.HASH;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Inner-Operator">
    @Passed
    protected void init()
    {
        if(size==0) 
        {
            size=initSize;
            up=(int) (size*this.expendThreshold);
            down=(int) (size*this.shrinkThreshold);
        }
        buckets=new ZNode[size];
        unInit=false;
    }
    /**
     * <pre>
     * Before expend, you need to check {@code need+num<up }, and increase
     * {@code num} if need.
     * </pre>
     * @param need 
     */
    @Passed
    protected void expend(int need)
    {
        if(size<initSize) size=initSize;
        if(size<(need+=num)) size=need;
        size*=this.expandRate;
        up=(int) (size*this.expendThreshold);
        down=(int) (size*this.shrinkThreshold);       
        this.copyOldNodes(buckets);
    }
    /**
     * <pre>
     * Before expend, you need to check {@code (num-=need)>down||size<initSize },
     * and decrease {@code num} if need.
     * </pre>
     * @param need 
     */
    @Passed
    protected void shrink(int need)
    {
        if(size<initSize) return;
        size*=this.shrinkRate;
        need=(int) (num/this.shrinkRate);
        if(size>need) size=need;
        if(size<initSize) size=initSize;
        
        up=(int) (size*this.expendThreshold);
        down=(int) (size*this.shrinkThreshold);
        this.copyOldNodes(buckets);
    }
    /**
     * During expending, don't call {@link #addKV(Comparable, Object, int, int) },
     * as it may cause dead cycle of method calling, So we use 
     * {@link #copyNode(ZNode)} to avoid this.
     * @param oldBuckets
     */
    @Passed
    protected void copyOldNodes(ZNode[] oldBuckets)
    {
        buckets=new ZNode[size];
        ZNode last;
        for(ZNode bucket:oldBuckets)
            while(bucket!=null)
            {
                last=bucket;
                bucket=bucket.pnext;
                last.pnext=null;
                this.copyNode(last);
            }
    }
    @Passed
    private void copyNode(ZNode node)
    {
        int index=node.hash%size;
        if(buckets[index]==null) buckets[index]=node;
        else
        {
            ZNode phead=buckets[index];
            while(phead.pnext!=null) phead=phead.pnext;
            phead.pnext=node;
        }
    }
    /**
     * you need to add the node first, then call expend, otherwise, Entries 
     * with different hash-index may add to the same bucket.
     * @param key
     * @param value
     * @param hash
     * @return 
     */
    @Passed
    protected V addKV(K key, V value, int hash)
    {
        int index=hash%size;//after expend,and back to here, the size is changed
        if(buckets[index]==null) 
        {
            if(num+1>=up) {expend(1);return addKV(key, value, hash);}
            buckets[index]=new ZNode(key, value, hash);
            num++;
            return null;
        }
        ZNode phead=buckets[index];
        //check if there exists conflict
        if(key==null)
        {
            for(;phead.pnext!=null;phead=phead.pnext)
                if(phead.key==null) return (V) phead.setValue(value);
            if(phead.key==null) return (V) phead.setValue(value);
        }
        else
        {
            for(;phead.pnext!=null;phead=phead.pnext)
                if(key.equals(phead.key)) return (V) phead.setValue(value);
            if(key.equals(phead.key)) return (V) phead.setValue(value);
        }
        //no confict exists
        if(num+1>=up) {this.expend(1);return this.addKV(key, value, hash);}
        phead.pnext=new ZNode(key, value, hash);
        num++;
        return null;
    }
    @Passed
    protected V removeNode(K key, int hash)
    {
        int index=hash%size;
        if(buckets[index]==null) return null;
        ZNode<K,V> phead=buckets[index],last=null;
        if(key==null) 
        {
            for(;phead!=null;last=phead,phead=phead.pnext)
            if(phead.key==null)
            {
                if(last!=null) last.pnext=phead.pnext;
                else buckets[index]=phead.pnext;
                phead.pnext=null;
                phead.key=null;
                if(--num<down) this.shrink(1);
                return phead.setValue(null);
            }
        }
        else
        {
            for(;phead!=null;last=phead,phead=phead.pnext)
            if(key.equals(phead.key))
            {
                if(last!=null) last.pnext=phead.pnext;
                else buckets[index]=phead.pnext;
                phead.pnext=null;
                phead.key=null;
                if(--num<down) this.shrink(1);
                return phead.setValue(null);
            }
        }
        return null;
    }
    @Passed
    protected ZNode<K,V> findNode(K key, int hash)
    {
        ZNode phead=buckets[hash%size];
        if(key==null)
        {
            while(phead!=null&&phead.key!=null)
                phead=phead.pnext;
        }
        else
        {
             while(phead!=null&&!key.equals(phead.key))
                phead=phead.pnext;
        }
        return phead;
    }
    @Override
    protected boolean innerRemoveAll(Predicate pre)//checked
    {
        int count=0;
        ZNode last,phead;
        for(int i=0;i<size;i++)
        {
            if(buckets[i]==null) continue;
            for(phead=buckets[i],last=null;phead!=null;)
            {
                if(pre.test(phead.key)) 
                {
                    if(last!=null) last.pnext=phead.pnext;
                    else buckets[i]=phead.pnext;
                    last=phead;
                    phead=phead.pnext;
                    last.pnext=null;
                    last.key=null;
                    last.value=null;
                    count++;
                }
                else {last=phead;phead=phead.pnext;}
           }
        }
        if((num-=count)<down) this.shrink(count);
        return count!=0;
    }
    @Override
    protected boolean innerRemoveAll(BiPredicate pre)//checked 
    {
        int count=0;
        ZNode last,phead;
        for(int i=0;i<size;i++)
        {
            if(buckets[i]==null) continue;
            for(phead=buckets[i],last=null;phead!=null;)
            {
                if(pre.test(phead.key, phead.value)) 
                {
                    if(last!=null) last.pnext=phead.pnext;
                    else buckets[i]=phead.pnext;
                    last=phead;
                    phead=phead.pnext;
                    last.pnext=null;
                    last.key=null;
                    last.value=null;
                    count++;
                }
                else {last=phead;phead=phead.pnext;}
           }
        }
        if((num-=count)<down) this.shrink(count);
        return count!=0;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Outer-Operator">
    @Override
    public boolean containsKey(Object key) 
    {
        if(this.isEmpty()) return false;
        int hash=coder.hashCode(key);
        return this.findNode((K) key, hash)!=null;
    }
    @Override
    public boolean containsValue(Object value) 
    {
        if(this.isEmpty()) return false;
        if(value==null)
        {
            for(ZNode phead:buckets)
                while(phead!=null) 
                {
                    if(phead.value==null) return true;
                    phead=phead.pnext;
                }
        }
        else
        {
            for(ZNode phead:buckets)
                while(phead!=null) 
                {
                    if(value.equals(phead.value)) return true;
                    phead=phead.pnext;
                }
        }
        return false;
    }
    @Override
    public boolean contains(Object key, Object value)
    {
        if(this.isEmpty()) return false;
        int hash=coder.hashCode(key);
        ZNode r=this.findNode((K) key, hash);
        if(r==null) return false;
        if(r.value==null) return value==null;
        return r.value.equals(value);
    }
    @Override
    public V get(Object key)//checked
    {
        if(this.isEmpty()) return null;
        int hash=coder.hashCode(key);
        ZNode<K,V> node=this.findNode((K) key, hash);
        return (node!=null? node.value:null);
    }
    @Override
    public V put(K key, V value)//checked
    {
        if(unInit) this.init();
        return this.addKV(key, value, coder.hashCode(key));
    }
    @Override
    public V remove(Object key)
    {
        return (isEmpty()? null:removeNode((K) key,coder.hashCode(key)));
    }
    @Override
    public void putAll(Map<? extends K,? extends V> m)
    {
        if(unInit) this.init();
        m.forEach((K key, V value)-> {addKV(key, value, coder.hashCode(key));});
    }
    @Override
    public void forEachKey(Consumer<? super K> con)//checked
    {
       for(ZNode<K,V> phead:buckets)
            for(;phead!=null;phead=phead.pnext)
                con.accept(phead.key);
    }
    @Override
    public void forEachValue(Consumer<? super V> con)//checked
    {
        for(ZNode<K,V> phead:buckets)
            for(;phead!=null;phead=phead.pnext)
                con.accept(phead.value);
    }
    @Override
    public void forEachEntry(Consumer<? super Entry<K, V>> con)//checked
    {
        for(ZNode<K,V> phead:buckets)
            for(;phead!=null;phead=phead.pnext)
                con.accept(phead);
    }
    @Override
    public void forEach(BiConsumer<? super K, ? super V> con)//checked
    {
        for(ZNode<K,V> phead:buckets)
            for(;phead!=null;phead=phead.pnext)
                con.accept(phead.key, phead.value);
    }
    @Override
    public void clear() 
    {
        ZNode phead,last;
        for(int i=0;i<size;i++)
        {
            for(phead=buckets[i],last=null;phead!=null;)
            {
                last=phead;
                phead=phead.pnext;
                last.value=null;
                last.key=null;
                last.pnext=null;
            }
            this.buckets[i]=null;
        }
        this.size=0;
        this.num=this.up=this.down=0;
        this.buckets=EMPTY_DATA;
        this.unInit=true;
    }
    @Override
    public Set<K> keySet() 
    {
        if(this.keySet==null) this.keySet=new KeySet();
        return this.keySet;
    }
    @Override
    public ZLinkedHashSet<K> replicaKeySet()//checked
    {
        ZLinkedHashSet<K> set=new ZLinkedHashSet<>(size);
        for(ZNode<K,V> phead:this.buckets)
            for(;phead!=null;phead=phead.pnext) set.add(phead.key);
        return set;
    }
    @Override
    public Collection<V> values()//checked
    {
        if(this.valueCollection==null) this.valueCollection=new ValueCollection();
        return this.valueCollection;
    }
    @Override
    public ZArrayList<V> replicaValues()//checked
    {
        ZArrayList<V> arr=new ZArrayList<>(size);
        for(ZNode<K,V> phead:this.buckets)
            for(;phead!=null;phead=phead.pnext) arr.add(phead.value);
        return arr;
    }
    @Override
    public Set entrySet() 
    {
        if(this.entrySet==null) this.entrySet=new EntrySet();
        return entrySet;
    }
    @Override
    public ZLinkedHashSet<? extends ZEntry<K, V>> replicaEntrySet()//checked
    {
        ZLinkedHashSet<ZNode<K,V>> set=new ZLinkedHashSet<>(size);
        for(ZNode<K,V> phead:this.buckets)
            for(;phead!=null;phead=phead.pnext) set.add(phead);
        return set;
    }
    //</editor-fold>
}
