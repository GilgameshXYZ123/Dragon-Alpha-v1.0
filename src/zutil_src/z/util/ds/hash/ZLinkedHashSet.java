/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.ds.hash;

import java.lang.reflect.Array;
import java.util.Collection;
import java.util.Iterator;
import java.util.function.BiPredicate;
import java.util.function.Consumer;
import java.util.function.Predicate;
import static z.util.ds.Constants.Hash.DEF_EXPEND_RATE;
import static z.util.ds.Constants.Hash.DEF_EXPEND_THRESHOLD;
import static z.util.ds.Constants.Hash.DEF_INIT_SIZE;
import static z.util.ds.Constants.Hash.DEF_SHRINK_THRESHOLD;
import z.util.ds.ZBaseSet;
import z.util.ds.imp.Structurable;
import z.util.lang.exception.IAE;
import z.util.ds.imp.Indexable;
import z.util.math.ExMath;
import static z.util.math.ExMath.DEF_HASHCODER;
import z.util.math.ExMath.HashCoder;
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
 * (5)The different for adding elements between HashSet and ArrayList is:
 * ArrayList doesn't need to check conflict before actually adding new elements, but 
 * HashSet need. So HashSet only should expend after find there is no conflict,
 * while ArrayList can expend before adding.
 * }
 * </pre>
 * @author dell
 * @param <T>
 */
@Passed
public class ZLinkedHashSet<T extends Comparable> extends ZBaseSet<T> implements Structurable
{
    //<editor-fold defaultstate="collapsed" desc="static class ZlinkedHashSet.ZNode<T>">
    protected static class ZNode<T1 extends Comparable> 
    {
        //columns---------------------------------------------------------------
        protected T1 key;
        protected int hash;
        protected ZNode pnext;
        
        //constructors----------------------------------------------------------
        protected ZNode() {}
        protected ZNode(T1 key, int hash)
        {
            this.key=key;
            this.hash=hash;
            this.pnext=null;
        }
        protected ZNode(ZNode<T1> node)
        {
            this.key=node.key;
            this.hash=node.hash;
            this.pnext=null;
        }
        protected T1 setKey(T1 key)
        {
            T1 ov=this.key;
            this.key=key;
            return ov;
        }
    }
    //</editor-fold>
    protected static final ZNode[] EMPTY_DATA={};
    
    protected final double expendThreshold;
    protected final double shrinkThreshold;
    
    protected final double expandRate;
    protected final double shrinkRate;//as shrinkRate=1/expendRate
    
    protected final ExMath.HashCoder coder;
    
    protected final int initSize;
    protected int up;
    protected int down;
    protected int num;
    protected boolean unInit;
    
    protected ZNode<T>[] buckets;
    
    @Passed
    public ZLinkedHashSet(int size, double expendThreshold, double shrinkThreshold, 
            double expendRate, HashCoder coder)
    {
       if(expendThreshold>=1||expendThreshold<=0)
                throw new IAE("(expendThreshold must be between 0 and 1");
        if(shrinkThreshold>=1||shrinkThreshold<=0)
                throw new IAE("shrinkThreshold must be between 0 and 1");
        
        this.initSize=this.size=(size>DEF_INIT_SIZE? size:DEF_INIT_SIZE);
        
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
        this.up=(int) (size*this.expendThreshold);
        this.down=(int) (size*this.shrinkThreshold);
        this.unInit=true;
        this.buckets=EMPTY_DATA;
    }
    public ZLinkedHashSet(int size, HashCoder coder)
    {
        this(size, DEF_EXPEND_THRESHOLD, DEF_SHRINK_THRESHOLD, DEF_EXPEND_RATE, coder);
    }
    public ZLinkedHashSet(int size)
    {
        this(size, DEF_EXPEND_THRESHOLD, DEF_SHRINK_THRESHOLD, DEF_EXPEND_RATE, DEF_HASHCODER);
    }
    public ZLinkedHashSet()
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
            ZNode<T> phead=null;
            for(int i=0;i<size;i++)
                for(phead=buckets[i];phead!=null;phead=phead.pnext)
                    sb.append(phead.key).append(", ");
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
        ZNode<T> phead=null;
        for(int i=0;i<size;i++)
        {
            if((phead=buckets[i])==null) continue;
            System.out.print(i+":>>");
            for(;phead!=null;phead=phead.pnext)
                System.out.print("["+phead.key+"] ");
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
        if(unInit)
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
    protected void copyNode(ZNode node)
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
     * @param hash
     * @return 
     */
    @Passed
    protected boolean addT(T key, int hash)
    {
        int index=hash%size;//after expend,and back to here, the size is changed
        if(buckets[index]==null) 
        {
            if(num+1>=up) {expend(1);return addT(key, hash);}
            buckets[index]=new ZNode(key, hash);
            num++;
            return true;
        }
        ZNode phead=buckets[index];
        //check if there exists conflict
        if(key==null)
        {
            for(;phead.pnext!=null;phead=phead.pnext)
                if(phead.key==null) return false;
            if(phead.key==null) return false;
        }
        else
        {
            for(;phead.pnext!=null;phead=phead.pnext)
                if(key.equals(phead.key)) return false;
            if(key.equals(phead.key)) return false;
        }
        //no confict exists
        if(num+1>=up) {this.expend(1);return this.addT(key, hash);}
        phead.pnext=new ZNode(key, hash);
        num++;
        return true;
    }
    @Passed
    protected boolean removeNode(T key, int hash)
    {
        int index=hash%size;
        if(buckets[index]==null) return false;
        ZNode<T> phead=buckets[index],last=null;
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
                return true;
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
                return true;
            }
        }
        return false;
    }
    @Passed
    protected ZNode<T> findNode(T key, int hash)
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
    protected boolean innerRemoveAll(Predicate pre) 
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
                    count++;
                }
                else {last=phead;phead=phead.pnext;}
           }
        }
        if((num-=count)<down) this.shrink(count);
        return count!=0;
    }
    @Override
    protected boolean innerRemoveAll(BiPredicate pre, Object condition)
    {
        int count=0;
        ZNode last,phead;
        for(int i=0;i<size;i++)
        {
            if(buckets[i]==null) continue;
            for(phead=buckets[i],last=null;phead!=null;)
            {
                if(pre.test(phead.key, condition)) 
                {
                    if(last!=null) last.pnext=phead.pnext;
                    else buckets[i]=phead.pnext;
                    last=phead;
                    phead=phead.pnext;
                    last.pnext=null;
                    last.key=null;
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
    public boolean contains(Object o)
    {
        if(this.isEmpty()) return false;
        return this.findNode((T) o, coder.hashCode(o))!=null;
    }
    @Override
    public boolean add(T e) 
    {
        if(unInit) this.init();
        return this.addT(e, coder.hashCode(e));
    }
     @Override
    public boolean remove(Object o)
    {
        return (this.isEmpty()? null:this.removeNode((T) o,coder.hashCode(o)));
    }
    @Override
    public boolean addAll(Collection<? extends T> c) 
    {
        if(unInit) this.init();
        int onum=num;
        for(T o:c) addT(o,coder.hashCode(o));
        return onum>num;
    }
    @Override
    public void forEach(Consumer<? super T> con)
    {
        ZNode<T> phead;
        for(int i=0;i<size;i++)
            for(phead=buckets[i];phead!=null;phead=phead.pnext)
                con.accept(phead.key);
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
    public Object[] toArray() 
    {
        Object[] arr=new Object[num];
        int index=0;
        for(ZNode phead:buckets)
            for(;phead!=null;phead=phead.pnext)
                arr[index++]=phead.key;
        return arr;
    }
    @Override
    public <P> P[] toArray(P[] a) 
    {
        if(a.length<num)
            a=(P[]) Array.newInstance(a.getClass().getComponentType(), num);
        Object[] arr=a;
        int index=0;
        for(ZNode phead:buckets)
            for(;phead!=null;phead=phead.pnext)
                arr[index++]=phead.key;
        if(a.length>num) a[num]=null;
        return a;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Iterable">
    @Passed
    private final class Iter implements Iterator<T>
    {
        //columns---------------------------------------------------------------
        private ZNode<T> phead;
        private int index;
        
        //functions-------------------------------------------------------------
        Iter()
        {
            index=0;
            phead=ZLinkedHashSet.this.buckets[0];
        }
        @Override
        public boolean hasNext() 
        {
            if(phead!=null) return true;
            while(++index<size&&buckets[index]==null);
            if(index>=size) return false;
            phead=buckets[index];
            return true;
        }
        @Override
        public T next() 
        {
            T key=phead.key;
            phead=phead.pnext;
            return key;
        }
    }
    @Override
    public Iterator<T> iterator()
    {
        return new Iter();
    }
    //</editor-fold>
}
