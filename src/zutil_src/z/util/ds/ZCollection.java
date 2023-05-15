/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.ds;

import z.util.ds.imp.CollectionRemoveExtensive;
import z.util.ds.imp.Indexable;
import static java.lang.Math.log;
import java.util.Collection;
import java.util.HashSet;
import java.util.TreeSet;
import java.util.function.BiPredicate;
import java.util.function.Predicate;
import z.util.ds.imp.Numberable;

/**
 *
 * @author dell
 * @param <T>
 */
public abstract class ZCollection<T> implements Collection<T>, Indexable, CollectionRemoveExtensive, Numberable
{
    @Override
    public boolean containsAll(Collection<?> c)//checked
    {
        if(this.isEmpty()) return true;
        if(c==null||c.isEmpty()) return false;
        for(Object o:c)
            if(!this.contains(o)) return false;
        return true;
    }
    //<editor-fold defaultstate="collapsed" desc="Inner-Code">
    /**
     * <pre>
     * (1) this.length=n
     * (2) t.length=m.
     * The Time Complexity to check all elements of this, whether it's equal to
     * any elements of t based on some condition:
     * The collision probability of the hash table can be described in a normal distribution,
     * O_Array(m,n)=m*n
     * O_Tree(m,n)=log2(m)*(n+m)+10*m-0.5m^2
     * O_Hash=k*n+m*10
     * {@code
     *      if O_Array < O_Tree  return t, O(m*n);
     *      else if O_Tree <= O_Hash return TreeMap(t), O(log(m)*n+m*log(m))
     *      else return HashMap(t), O(c*n+m*c)
     * }
     * </pre>
     * @param t
     * @return 
     */
    protected Collection optimizedCheckInCollection(Collection t)
    {
        int anum=t instanceof ZCollection? ((ZCollection)t).number():t.size();
        if(Indexable.isHash(t)) return t;
        int num=this.number();
        double tree_o=1.442695040888954*log(anum)*(num/anum)+1+10-(anum>>1);
        if(num <= tree_o) return t;
        else if(tree_o<=3*num/anum) return (Indexable.isTree(t)? t:new TreeSet(t));
        else return new HashSet(t);
    }
    protected abstract boolean innerRemoveAll(Predicate pre);
    protected abstract boolean innerRemoveAll(BiPredicate pre, Object condition);
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Remove Extensive">
    @Override
    public boolean removeAll(Predicate pre) 
    {
        return this.isEmpty()? false:this.innerRemoveAll(pre);
    }
    @Override
    public boolean removeAll(BiPredicate pre, Object condition) 
    {
        return this.isEmpty()? false:this.innerRemoveAll(pre, condition);
    }
    @Override
    public boolean retainAll(Predicate pre) 
    {
        return this.isEmpty()? false:this.innerRemoveAll(pre.negate());
    }
    @Override
    public boolean retainAll(BiPredicate pre, Object condition) 
    {
        return this.isEmpty()? false:this.innerRemoveAll(pre.negate(), condition);
    }
    @Override
    public boolean removeAll(Collection<?> c)
    {
        if(this.isEmpty()) return false;
        if(c==null||c.isEmpty()) return true;
        final Collection fc=this.optimizedCheckInCollection(c);
        return this.innerRemoveAll((Predicate) (o) -> fc.contains(o));
    }
    @Override
    public boolean retainAll(Collection<?> c)//checked
    {
        if(this.isEmpty()) return false;
        if(c==null||c.isEmpty()) return true;
        final Collection fc=this.optimizedCheckInCollection(c);
        return this.innerRemoveAll((Predicate) (o) -> !fc.contains(o));
    }
    //</editor-fold>
}
