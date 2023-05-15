/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.ds.hash;

import java.util.Collection;
import java.util.Iterator;
import java.util.Set;
import java.util.function.BiConsumer;
import java.util.function.BiPredicate;
import java.util.function.Consumer;
import java.util.function.Predicate;
import static z.util.ds.Constants.Hash.DEF_EXPEND_RATE;
import static z.util.ds.Constants.Hash.DEF_EXPEND_THRESHOLD;
import static z.util.ds.Constants.Hash.DEF_INIT_SIZE;
import static z.util.ds.Constants.Hash.DEF_SHRINK_THRESHOLD;
import static z.util.ds.Constants.Hash.DEF_TREE_THRESHOLD;
import static z.util.ds.Constants.Hash.DEF_LINKED_THRESHOLD;
import z.util.ds.ZEntry;
import z.util.ds.linear.ZArrayList;
import z.util.ds.linear.ZLinkedList;
import static z.util.math.ExMath.DEF_HASHCODER;
import z.util.math.ExMath.HashCoder;
import z.util.lang.exception.IAE;
import z.util.ds.linear.ZLinkedStack;
import z.util.lang.annotation.Passed;
import z.util.math.vector.Vector;

/**
 * <pre>
 * ZHashMap is an optimized implemetation by using Hash-Index.
 * 1.A ZHashMap has many buckets, Elements with the same hashCode will be 
 * added to the same bucket;
 * 2.Firstly, We use a LinkedList to link all elements in the same bucket;
 * 3.As more and more elements are added to the same bucket, when there are more
 * than {@code TREE_THRESHOLD} elements, the linked list will be reconstructed 
 * to a black red tree;
 * 4.when the quantity of the elements in the same bucket is less than 
 * {@code LINKEDLIST_THRESHOLD} the tree will be reconstructed to a linkedlist
 * 5.When the size of table is bigger than {@code DEF_GROW_THRESHOLD} elements, 
 * it will expend to have more buckets. 
 * 6.The rule of shrink and expend is similar to {@see ZArrayList}.
 * 7.When testing, use this code to turn a linkedlist-bucket to a tree-bucket:
 * {@code 
 * int num=10;
 * ExRandom ex=Lang.exRandom();
 * int[] v=new int[num];
 * for(int i=0;i<num;i++) map.put(v[i]=map.size*i+1, i);}
 * as it can cause a high posibility for element conflict.
 * </pre>
 *
 * @author dell
 * @param <K>
 * @param <V>
 */
@Passed
public class ZHashMap<K extends Comparable, V> extends ZLinkedHashMap<K,V>
{
    //<editor-fold defaultstate="collapsed" desc="static class TreeNode<K2,K2>">
    private static final class TreeNode<K2 extends Comparable, V2> extends ZLinkedHashMap.ZNode<K2,V2>
    {
        //columns---------------------------------------------------------------
        private boolean color;//red=true,black=false
        private TreeNode pleft;
        private TreeNode pright;
        private TreeNode parent;
        
        //constructors----------------------------------------------------------
        TreeNode()//only used to create nullptr
        {
            color=false;
            pleft=pright=parent=null;
        }
        TreeNode(K2 key, V2 value, int hash) 
        {
            super(key, value, hash);
            this.color=true;
            
            this.parent=nullptr;
            this.pleft=nullptr;
            this.pright=nullptr;
        }
        TreeNode(K2 key, V2 value, int hash, TreeNode<K2,V2> parent) 
        {
            super(key, value, hash);
            this.color=true;
            
            this.parent=parent;
            this.pleft=nullptr;
            this.pright=nullptr;
        }
        TreeNode(ZNode<K2,V2> node)
        {
            super(node.key, node.value, node.hash);
            this.color=true;
            
            this.parent=nullptr;
            this.pleft=nullptr;
            this.pright=nullptr;
        }
    }
    //<editor-fold defaultstate="collapsed" desc="nullptr">
    protected static final TreeNode nullptr;
    
    static
    {
        nullptr=new TreeNode();
        nullptr.parent=nullptr;
        nullptr.pright=nullptr;
        nullptr.pleft=nullptr;
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Iterable">
    @Passed
    private abstract class HashIter
    {
        //columns---------------------------------------------------------------
        private int index=0;
        private ZNode phead;
        private boolean inTree;
        private ZLinkedStack<TreeNode> s;
        
        //functions-------------------------------------------------------------
        HashIter()
        {
            index=0;
            phead=ZHashMap.this.buckets[0];
            inTree=phead instanceof TreeNode;
            s=new ZLinkedStack<>();
        }
        public final boolean hasNext() 
        {
            if(inTree)
            {
                if((!s.isEmpty()||phead!=nullptr))return true;
            }
            else if(phead!=null) return true;
            
            while(++index<size&&buckets[index]==null);
            if(index>=size) return false;
            phead=buckets[index];
            inTree=phead instanceof TreeNode;
            return true;
        } 
        ZNode<K,V> nextNode()
        {
           if(inTree)
           {
                TreeNode r=(TreeNode) phead, next;
                for(;r!=nullptr;s.push(r),r=r.pleft);
                next=r=s.pop();
                phead=r.pright;
                return next;
           }
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
    private final class KeySet2 extends KeySet
    {
        @Override
        public int size() {return size;}
        @Override
        public void clear() {ZHashMap.this.clear();}
        @Override
        public boolean contains(Object key) {return ZHashMap.this.containsKey(key);}
        @Override
        public boolean remove(Object key) {return ZHashMap.this.remove(key)!=null;}
        @Override
        public Iterator<K> iterator() 
        {
            if(ZHashMap.this.treeCount==0) return super.iterator();
            return new KeyIter();
        }
        @Override
        public void forEach(Consumer<? super K> con)
        {
            ZHashMap.this.forEachKey(con);
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="class ValueCollection">
    @Passed
    private final class ValueCollection2 extends ValueCollection
    {
        @Override
        public int size() {return size;}
        @Override
        public void clear() {ZHashMap.this.clear();}
        @Override
        public boolean contains(Object value) {return ZHashMap.this.containsValue(value);}
        @Override
        public Iterator<V> iterator() 
        {
            if(ZHashMap.this.treeCount==0) return super.iterator();
            return new ValueIter();
        }
        @Override
        public void forEach(Consumer<? super V> con) 
        {
            ZHashMap.this.forEachValue(con);
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="class EntrySet">
    @Passed
    private final class EntrySet2 extends EntrySet 
    {
        @Override
        public int size() {return size;}
        @Override
        public void clear() {ZHashMap.this.clear();}
        @Override
        public boolean contains(Object entry) 
        {
            Entry<K,V> kv=(Entry<K,V>) entry;
            return ZHashMap.this.contains(kv.getKey(), kv.getValue());
        }
        @Override
        public boolean remove(Object key) 
        {
            return ZHashMap.this.remove(key)!=null;
        }
        @Override
        public Iterator<Entry<K,V>> iterator()
        {
            if(ZHashMap.this.treeCount==0) return super.iterator();
            return new EntryIter();
        }
        @Override
        public void forEach(Consumer<? super Entry<K, V>> con) 
        {
            ZHashMap.this.forEachEntry(con);
        }
    }
    //</editor-fold>
    private static final int[] EMPTY_COUNTS={};
    
    private final int treeThreshold;
    private final int linkedThreshold;
    
    private int[] counts;
    private int treeCount;
    
    @Passed
    public ZHashMap(int initSize, double expendThreshold, double shrinkThreshold, 
           double expendRate, HashCoder coder,
           int treeThreshold, int linkedThreshold)
    {
        super(initSize, expendThreshold, shrinkThreshold, expendRate, coder);
        
        if(treeThreshold<=0) throw new IAE("threeThreshold must be positive");
        if(linkedThreshold<=0) throw new IAE("linkedThreshold must be positive");
        
        if((treeThreshold<0.9*linkedThreshold)||(treeThreshold-linkedThreshold<2))
        {
            this.treeThreshold=DEF_TREE_THRESHOLD;
            this.linkedThreshold=DEF_LINKED_THRESHOLD;
        }
        else
        {
            this.treeThreshold=treeThreshold;
            this.linkedThreshold=linkedThreshold;
        }
        treeCount=0;
    }
    public ZHashMap(int initSize, HashCoder coder, int treeThreshold, int linkedThreshold)
    {
       this(initSize, DEF_EXPEND_THRESHOLD, DEF_SHRINK_THRESHOLD, DEF_EXPEND_RATE, coder,
                treeThreshold, linkedThreshold);
    }
    public ZHashMap(int initSize, HashCoder coder)
    {
         this(initSize, DEF_EXPEND_THRESHOLD, DEF_SHRINK_THRESHOLD, DEF_EXPEND_RATE, coder,
                DEF_TREE_THRESHOLD, DEF_LINKED_THRESHOLD);
    }
    public ZHashMap(int size)
    {
        this(size, DEF_EXPEND_THRESHOLD, DEF_SHRINK_THRESHOLD, DEF_EXPEND_RATE, DEF_HASHCODER,
                DEF_TREE_THRESHOLD, DEF_LINKED_THRESHOLD);
    }
    public ZHashMap(HashCoder coder)
    {
         this(DEF_INIT_SIZE, DEF_EXPEND_THRESHOLD, DEF_SHRINK_THRESHOLD, DEF_EXPEND_RATE, coder,
                DEF_TREE_THRESHOLD, DEF_LINKED_THRESHOLD);
    }
    public ZHashMap()
    {
        this(DEF_INIT_SIZE, DEF_EXPEND_THRESHOLD, DEF_SHRINK_THRESHOLD, DEF_EXPEND_RATE, DEF_HASHCODER,
                 DEF_TREE_THRESHOLD, DEF_LINKED_THRESHOLD);
    }
    //<editor-fold defaultstate="collapsed" desc="Basic-Function">
    @Override
    public void append(StringBuilder sb)//checked
    {
        if(treeCount==0) {super.append(sb);return;}
        sb.append('{');
        if(!this.isEmpty())
        {
            ZLinkedStack<TreeNode> s=new ZLinkedStack<>();
            TreeNode<K,V> r;
            for(ZNode<K,V> phead:buckets)
            {
                if(phead instanceof TreeNode)
                {
                    r=(TreeNode<K, V>) phead;
                    do{
                        if(r!=nullptr) {s.push(r);r=r.pleft;}
                        else
                        {
                            sb.append(r= r=s.pop()).append(", ");
                            r=r.pright;
                        }
                    }while(!s.isEmpty()||r!=nullptr);
                    continue;
                }
                for(;phead!=null;phead=phead.pnext) sb.append(phead).append(", ");
            }
        }
        if(num>1) sb.setCharAt(sb.length()-2, '}');
        else sb.append('}');
    }
    @Override
    public void struc() 
    {
        Vector.println(System.out, counts);
        if(treeCount==0) {super.struc();return;}
        TreeNode r;
        ZNode<K,V> phead=null;
        ZLinkedStack<TreeNode> s=new ZLinkedStack<>();
        for(int i=0;i<size;i++)
        {
            if(buckets[i]==null) continue;
            System.out.print(i+":count="+counts[i]);
            if(buckets[i] instanceof TreeNode)
            {
                System.out.print("tree>>>");
                r=(TreeNode) buckets[i];
                do{
                    if(r!=nullptr) {s.push(r);r=r.pleft;}
                    else{
                        r=s.pop();
                        System.out.print("["+r+"] ");
                        r=r.pright;
                    }
                }while(!s.isEmpty()||r!=nullptr);
                continue;
            }
            System.out.print("linkedlist>>>");
            for(phead=buckets[i];phead!=null;phead=phead.pnext)
                System.out.print("["+phead+"] ");
             System.out.println();
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Inner-Operator">
    @Override
    protected void init()
    {
        if(size==0) 
        {
            size=initSize;
            up=(int) (size*this.expendThreshold);
            down=(int) (size*this.shrinkThreshold);
        }
        buckets=new ZNode[size];
        counts=new int[size];
        treeCount=0;
        unInit=false;
    }
    //<editor-fold defaultstate="collapsed" desc="Tree-Basic-Operator">
    @Passed
    private void transplant(TreeNode p, TreeNode c, int index)
    {
        TreeNode g=p.parent;
        if(g==nullptr) buckets[index]=c;
        else if(g.pleft==p) g.pleft=c;
        else g.pright=c;
        c.parent=g;
    }
    @Passed
    private void leftRotate(TreeNode target, int index)
    {
        TreeNode right=target.pright;
        if(right==nullptr) return;
        target.pright=right.pleft;
        if(right.pleft!=nullptr) right.pleft.parent=target;
        this.transplant(target, right, index);
        right.pleft=target;
        target.parent=right;
    }
    @Passed
    private void rightRotate(TreeNode target, int index)
    {
        TreeNode left=target.pleft;
        if(left==nullptr) return;
        target.pleft=left.pright;
        if(left.pright!=nullptr) left.pright.parent=target;
        this.transplant(target, left, index);
        left.pright=target;
        target.parent=left;
    }
    @Passed
    private void insertFix(TreeNode z, int index)
    {
        TreeNode y=nullptr;
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
                        this.leftRotate(z, index);
                    }
                    z.parent.color=false;
                    z.parent.parent.color=true;
                    this.rightRotate(z.parent.parent, index);
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
                        this.rightRotate(z, index);
                    }
                    z.parent.color=false;
                    z.parent.parent.color=true;
                    this.leftRotate(z.parent.parent, index);
                }
            }
        }
        ((TreeNode)buckets[index]).color=false;
    }
    @Passed
    private void removeFix(TreeNode z, int index)
    {
        TreeNode w=nullptr;
        while(z!=buckets[index]&&!z.color)
        {
            if(z==z.parent.pleft)
            {
                w=z.parent.pright;
                if(w.color)
                {
                    w.color=false;
                    z.parent.color=true;
                    this.leftRotate(z.parent, index); 
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
                    this.rightRotate(w, index);
                    w=z.parent.pright;
                }
                w.color=z.parent.color;
                z.parent.color=false;
                w.pright.color=false;
                this.leftRotate(z.parent, index);
                z=(TreeNode) buckets[index];
            }
            else
            {
                w=z.parent.pleft;
                if(w.color)
                {
                    w.color=false;
                    z.parent.color=true;
                    this.rightRotate(z.parent, index);
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
                    this.leftRotate(w, index);
                    w=z.parent.pleft;
                }
                w.color=z.parent.color;
                z.parent.color=false;
                w.parent.color=false;
                this.rightRotate(z.parent, index);
                z=(TreeNode) buckets[index];
            }
        }
        z.color=false;
    }
    /**
     * simply add a node to tree bucket, without increase the num and 
     * counts[index].
     * @param node
     * @param index 
     */
    @Passed
    private void treeAdd(TreeNode node, int index)
    {
        TreeNode<K,V> r=(TreeNode<K,V>) buckets[index],last=nullptr;
        int v;
        if(node.key==null)
        {
            while(r!=nullptr)
            {
                last=r;
                if(r.key==null) return;
                else if((v=r.key.compareTo(node.key))>0) r=r.pleft;
                else r=r.pright;
            }
        }
        else
        {
            while(r!=nullptr)
            {
                last=r;
                if((v=node.key.compareTo(r.key))<0) r=r.pleft;
                else if(v>0) r=r.pright;
                else return;
            }
        }
        r=node;
        r.parent=last;
        if(last==nullptr) buckets[index]=r;
        else if(last.key.compareTo(node.key)>0) last.pleft=r;
        else last.pright=r;
        this.insertFix(r, index);
    }
    @Passed
    private TreeNode<K,V> treeFind(K key, int index)
    {
        TreeNode<K,V> r=(TreeNode<K,V>) buckets[index];
        int v;
        if(key==null)
        {
            while(r!=nullptr)
            {
                if(r.key==null) break;
                else if((v=r.key.compareTo(key))>0) r=r.pleft;
                else r=r.pright;
            }
        }
        else
        {
            while(r!=nullptr)
            {
                if((v=key.compareTo(r.key))<0) r=r.pleft;
                else if(v>0) r=r.pright;
                else break;
            }
        }
        return r;
    }
    /**
     * decease num and call {@link #shrink(int) } or {@link #toLinked(int)}
     * automatically. {@code
     * (1) if --num<=down: call shrink and return. 
     * (2) if --counts[index]<linkedThreshold: call toLinked() and return
     * (3) removeFix();
     * }
     * @param z
     * @param index 
     */
    @Passed
    private void treeRemove(TreeNode z, int index)
    {
        TreeNode x=nullptr;
        boolean oc=z.color;
        if(z.pleft==nullptr) this.transplant(z, x=z.pright, index);
        else if(z.pright==nullptr) this.transplant(z, x=z.pleft, index);
        else
        {
            TreeNode y=z.pleft;
            while(y.pright!=null) y=y.pright;
            oc=y.color;
            x=y.pleft;
            if(y.parent!=z)
            {
                this.transplant(y, x, index);
                y.pleft=z.pleft;
                z.pleft.parent=y;
            }
            this.transplant(z, y, index);
            y.pright=z.pright;
            z.pright.parent=y;
            y.color=z.color;
        }
        z.pright=z.pleft=z.parent=null;
        z.key=null;
        z.value=null;
        
        if(--num<=down) {this.shrink(1);return;}//need to shrink, no need to removeFix
        if(--counts[index]<=this.linkedThreshold) {this.toLinked(index);return;}
        if(x!=nullptr&&!oc) this.removeFix(x, index);//oc is black
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Tree-LinkedList-Transfer">
    @Passed
    private void toTree(int index)
    {
        ZNode<K,V> phead=buckets[index],last=null;
        buckets[index]=nullptr;
        while(phead!=null)
        {
            last=phead;
            phead=phead.pnext;
            this.treeAdd(new TreeNode(last), index);
            last.key=null;
            last.value=null;
            last.pnext=null;
        }
        this.treeCount++;
    }
    @Passed
    private void toLinked(int index)
    {
        TreeNode r=(TreeNode) buckets[index];
        ZLinkedList<TreeNode> l=new ZLinkedList<>(); 
        l.add(r);
        ZNode phead=new ZNode(),pnow=phead;
        while(!l.isEmpty())
        {
            r=l.remove();
            if(r.pleft!=nullptr) l.add(r.pleft);
            if(r.pright!=nullptr) l.add(r.pright);
            //add new linked node
            pnow.key=r.key;
            pnow.value=r.value;
            pnow.pnext=new ZNode();
            pnow=pnow.pnext;
            //delete old treenode
            r.parent=r.pleft=r.pright=null;
            r.key=null;
            r.value=null;
        }
        pnow.pnext=null;//delete the single redundent node
        buckets[index]=phead;
        this.treeCount--;
        System.out.println("toLinked:"+this.treeCount);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="function-group:copyOldNodes">
    @Passed
    protected void treeCopyNode(TreeNode node)
    {
        int index=node.hash%size;
        counts[index]++;
        if(buckets[index]==null) 
        {
            buckets[index]=new ZNode(node);//newNode.pnext=null;
            node.parent=node.pright=node.pleft=null;//delete the node
            node.key=null;
            node.value=null;
            return;
        }
        if(buckets[index] instanceof TreeNode)//the bucket is a tree
        {
            node.parent=node.pright=node.pleft=nullptr;//must reset the pointer before add the node
            this.treeAdd(node, index);
            return;
        }
        if(counts[index]>=this.treeThreshold) //the bukcet is a linkedList
        {
            this.toTree(index);
            node.parent=node.pright=node.pleft=nullptr;
            this.treeAdd(node, index);
            return;
        }
        ZNode phead=buckets[index];
        while(phead.pnext!=null) phead=phead.pnext;
        phead.pnext=new ZNode(node);
        node.parent=node.pright=node.pleft=null;//delete the node
        node.key=null;
        node.value=null;
    }
    @Passed
    protected void linkdCopyNode(ZNode node)
    {
        node.pnext=null;
        int index=node.hash%size;
        counts[index]++;
        if(buckets[index]==null) 
        {
            buckets[index]=node;
            return;
        }
        if(buckets[index] instanceof TreeNode)//the bucket is a tree
        {
            this.treeAdd(new TreeNode(node), index);
            node.key=null;
            node.value=null;
            return;
        }        
        if(counts[index]>=this.treeThreshold) 
        {
            this.toTree(index);
            this.treeAdd(new TreeNode(node), index);
            return;
        }
        ZNode phead=buckets[index];//the bucket is a linkedlist
        while(phead.pnext!=null) phead=phead.pnext;
        phead.pnext=node;
    }
    @Override   
    protected void copyOldNodes(ZNode[] oldBuckets)//passed
    {
        if(treeCount==0) 
        {
            counts=new int[size];
            super.copyOldNodes(oldBuckets);
            return;
        }
        counts=new int[size];
        buckets=new ZNode[size];
        treeCount=0;//set treeCount=0, let all trees to be created during copyNode
        //Befor building the tree creating a linkedList to the elements in each buckets,
        //cost no extra memory.
        ZNode last;
        TreeNode r;
        ZLinkedList<TreeNode> l=new ZLinkedList<>();
        for(ZNode bucket:oldBuckets)
        {
            if(bucket instanceof TreeNode)//it is a tree
            {
                r=(TreeNode) bucket;
                l.add(r);
                while(!l.isEmpty())
                {
                    r=l.remove();
                    if(r.pleft!=nullptr) l.add(r.pleft);
                    if(r.pright!=nullptr) l.add(r.pright);
                    this.treeCopyNode(r);
                }
                continue;
            }
            while(bucket!=null)//it is a linkedList
            {
                last=bucket;
                bucket=bucket.pnext;
                this.linkdCopyNode(last);
            }
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="function-group:addKV">
    @Passed  
    private V linkedAddKV(K key, V value, int hash, int index)
    {
        ZNode phead=buckets[index];
        if(key==null)//check if there exists conflict
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
        //no conflict, add new node
        if(num+1>=up) {this.expend(1);return this.addKV(key, value, hash);}//to a tree or linked
        
        if((counts[index]+1)>this.treeThreshold)
        {
            this.toTree(index);
            num++;
            return this.treeAddKV(key, value, hash, index);//num++
        }
        
        phead.pnext=new ZNode(key, value, hash);
        num++;    
        counts[index]++;
        return null;
    }
    @Passed
    private V treeAddKV(K key, V value, int hash, int index)
    {
        //if a bucket is a tree, it's not null.
        TreeNode<K,V> r=(TreeNode<K,V>) buckets[index],last=nullptr;
        int v;
        if(key==null)//check if there exists conflict
        {
            while(r!=nullptr)
            {
                last=r;
                if(r.key==null) return r.setValue(value);
                else if((v=r.key.compareTo(key))>0) r=r.pleft;
                else r=r.pright;
            }
        }
        else
        {
            while(r!=nullptr)
            {
                last=r;
                if((v=key.compareTo(r.key))<0) r=r.pleft;
                else if(v>0) r=r.pright;
                else return r.setValue(value);
            }
        }
        r=new TreeNode(key, value, hash, last);
        if(last==nullptr) buckets[index]=r;
        else if(last.key.compareTo(key)>0) last.pleft=r;
        else last.pright=r;
        
        if(num+1>=up) {this.expend(1);return this.addKV(key, value, hash);}//to a tree or linked
        
        this.insertFix(r, index);
        num++;
        counts[index]++;
        return null;
    }
    @Override   
    protected V addKV(K key, V value, int hash)//passed
    {
        int index=hash%size;//after expend,and back to here, the size is changed
        if(buckets[index]==null)
        {
            if(num+1>=up) {expend(1);return addKV(key, value, hash);}
            buckets[index]=new ZNode(key, value, hash);
            num++;
            counts[index]++;
            return null;
        }
        return (buckets[index] instanceof TreeNode? 
                this.treeAddKV(key, value, hash, index):
                this.linkedAddKV(key, value, hash, index));
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="ZLinkedHashMap"> 
    @Override
    protected V removeNode(K key, int hash)//passed
    {
        int index=hash%size;
        if(buckets[index]==null) return null;
        if(buckets[index] instanceof TreeNode)//the bucket is a tree
        {
            TreeNode<K,V> r=this.treeFind(key, index);
            if(r==null) return null;
            V ov=r.value;
            this.treeRemove(r, index);//num--,counts[index--]
            return ov;
        }
        ZNode<K,V> phead=buckets[index],last=null;//the bucket is a linkedList
        if(key==null) 
        {
            for(;phead!=null;last=phead,phead=phead.pnext)
            if(phead.key==null)
            {
                if(last!=null) last.pnext=phead.pnext;
                else buckets[index]=phead.pnext;
                phead.pnext=null;
                phead.key=null;
                if(--num<=down) this.shrink(1);
                else counts[index]--;
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
                if(--num<=down) this.shrink(1);
                else counts[index]--;
                return phead.setValue(null);
            }
        }
        return null;
    }
    @Override
    protected ZNode<K,V> findNode(K key, int hash)//passed
    {
        int index=hash%size;
        ZNode phead=buckets[index];
        if(phead instanceof TreeNode)//the bucket is a tree
            return this.treeFind(key, index);
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
    protected boolean innerRemoveAll(Predicate pre)//passed
    {
        int count=0,l2_num;
        boolean[] toLink=new boolean[size];
        TreeNode r;
        ZNode last,phead;
        ZLinkedList<TreeNode> l1=new ZLinkedList<>();
        ZLinkedList<TreeNode> l2=new ZLinkedList<>();
        for(int i=0;i<size;i++)
        {
            if(buckets[i]==null) continue;
            if(buckets[i] instanceof TreeNode)//the bucket is a tree
            {
                r=(TreeNode) buckets[i];
                l1.add(r);
                while(!l1.isEmpty())
                {
                    r=l1.pop();
                    if(r.pleft!=nullptr) l1.add(r.pleft);
                    if(r.pright!=nullptr) l1.add(r.pright);
                    if(pre.test(r.key)) l2.add(r);
                }
                count+=l2_num=l2.number();
                while(!l2.isEmpty()) this.treeRemove(l2.pop(), i);
                if((counts[i]-=l2_num)<=this.shrinkThreshold) toLink[i]=true;
                continue;
            }
            for(phead=buckets[i],last=null;phead!=null;)//the bucket is a linkedList
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
                    counts[i]--;
                }
                else {last=phead;phead=phead.pnext;}
           }
        }
        if((num-=count)<down) this.shrink(count);
        for(int i=0;i<size;i++) 
            if(toLink[i]) this.toLinked(i);
        return count!=0;
    }
    @Override
    protected boolean innerRemoveAll(BiPredicate pre)//passed
    {
         int count=0,l2_num;
        boolean[] toLink=new boolean[size];
        TreeNode r;
        ZNode last,phead;
        ZLinkedList<TreeNode> l1=new ZLinkedList<>();
        ZLinkedList<TreeNode> l2=new ZLinkedList<>();
        for(int i=0;i<size;i++)
        {
            if(buckets[i]==null) continue;
            if(buckets[i] instanceof TreeNode)//the bucket is a tree
            {
                r=(TreeNode) buckets[i];
                l1.add(r);
                while(!l1.isEmpty())
                {
                    r=l1.pop();
                    if(r.pleft!=nullptr) l1.add(r.pleft);
                    if(r.pright!=nullptr) l1.add(r.pright);
                    if(pre.test(r.key, r.value)) l2.add(r);
                }
                count+=l2_num=l2.number();
                while(!l2.isEmpty()) this.treeRemove(l2.pop(), i);
                if((counts[i]-=l2_num)<=this.shrinkThreshold) toLink[i]=true;
                continue;
            }
            for(phead=buckets[i],last=null;phead!=null;)//the bucket is a linkedList
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
                    counts[i]--;
                }
                else {last=phead;phead=phead.pnext;}
           }
        }
        if((num-=count)<down) this.shrink(count);
        for(int i=0;i<size;i++) 
            if(toLink[i]) this.toLinked(i);
        return count!=0;
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Outer-Operator">
    @Override
    public boolean containsValue(Object value)//passed
    {
        if(treeCount==0) {return super.containsValue(value);}
        ZLinkedStack<TreeNode> s=new ZLinkedStack<>();
        TreeNode<K,V> r;
        if(value==null)
        {
            for(ZNode<K,V> phead:buckets)
            {
                if(phead instanceof TreeNode)
                {
                    r=(TreeNode<K, V>) phead;
                    do{
                        if(r!=nullptr) {s.push(r);r=r.pleft;}
                        else{
                            r=s.pop();
                            if(r.value==null) return true;
                            r=r.pright;
                        }
                    }while(!s.isEmpty()||r!=nullptr);
                    continue;
                }
                for(;phead!=null;phead=phead.pnext)
                    if(phead.value==null) return true;
            }
        }
        else
        {
            for(ZNode<K,V> phead:buckets)
            {
                if(phead instanceof TreeNode)
                {
                    r=(TreeNode<K, V>) phead;
                    do{
                        if(r!=nullptr) {s.push(r);r=r.pleft;}
                        else
                        {
                            r=s.pop();
                            if(value.equals(r.value)) return true;
                            r=r.pright;
                        }
                    }while(!s.isEmpty()||r!=nullptr);
                    continue;
                }
                for(;phead!=null;phead=phead.pnext)
                    if(value.equals(phead.value)) return true;
            }
        }
        return false;
    }
    @Override
    public void forEachKey(Consumer<? super K> con)//passed
    {
        if(treeCount==0) {super.forEachKey(con);return;}
        ZLinkedStack<TreeNode> s=new ZLinkedStack<>();
        TreeNode<K,V> r;
        for(ZNode<K,V> phead:buckets)
        {
            if(phead instanceof TreeNode)
            {
                r=(TreeNode<K, V>) phead;
                do{
                    if(r!=nullptr) {s.push(r);r=r.pleft;}
                    else{
                        r=s.pop();
                        con.accept(r.key);
                        r=r.pright;
                    }
                }while(!s.isEmpty()||r!=nullptr);
                continue;
            }
            for(;phead!=null;phead=phead.pnext) con.accept(phead.key);
        }
    }
    @Override
    public void forEachValue(Consumer<? super V> con)//passed
    {
        if(treeCount==0) {super.forEachValue(con);return;}
        ZLinkedStack<TreeNode> s=new ZLinkedStack<>();
        TreeNode<K,V> r;
        for(ZNode<K,V> phead:buckets)
        {
            if(phead instanceof TreeNode)
            {
                r=(TreeNode<K, V>) phead;
                do{
                    if(r!=nullptr) {s.push(r);r=r.pleft;}
                    else{
                        r=s.pop();
                        con.accept(r.value);
                        r=r.pright;
                    }
                }while(!s.isEmpty()||r!=nullptr);
                continue;
            }
            for(;phead!=null;phead=phead.pnext) con.accept(phead.value);
        }
    }
    @Override
    public void forEachEntry(Consumer<? super Entry<K, V>> con)//passed
    {
        if(treeCount==0) {super.forEachEntry(con);return;}
        ZLinkedStack<TreeNode> s=new ZLinkedStack<>();
        TreeNode<K,V> r;
        for(ZNode<K,V> phead:buckets)
        {
            if(phead instanceof TreeNode)
            {
                r=(TreeNode<K, V>) phead;
                do{
                    if(r!=nullptr) {s.push(r);r=r.pleft;}
                    else{
                        r=s.pop();
                        con.accept(r);
                        r=r.pright;
                    }
                }while(!s.isEmpty()||r!=nullptr);
                continue;
            }
            for(;phead!=null;phead=phead.pnext) con.accept(phead);
        }
    }
    @Override
    public void forEach(BiConsumer<? super K, ? super V> con)//passed
    {
        if(treeCount==0) {super.forEach(con);return;}
        ZLinkedStack<TreeNode> s=new ZLinkedStack<>();
        TreeNode<K,V> r;
        for(ZNode<K,V> phead:buckets)
        {
            if(phead instanceof TreeNode)
            {
                r=(TreeNode<K, V>) phead;
                do{
                    if(r!=nullptr) {s.push(r);r=r.pleft;}
                    else{
                        r=s.pop();
                        con.accept(r.key, r.value);
                        r=r.pright;
                    }
                }while(!s.isEmpty()||r!=nullptr);
                continue;
            }
            for(;phead!=null;phead=phead.pnext) con.accept(phead.key, phead.value);
        }
    }
    @Override
    public void clear()//passed
    {
        if(treeCount==0) 
        {
            super.clear();
            this.treeCount=0;
            this.counts=EMPTY_COUNTS;
            return;
        }
        ZLinkedList<TreeNode> l1=new ZLinkedList<>();
        ZNode phead,last;
        for(int i=0;i<size;i++)
        {
            phead=buckets[i];
            if(phead instanceof TreeNode)//if the bucket is tree
            {
                TreeNode r=(TreeNode) phead;
                l1.add(r);
                while(!l1.isEmpty())
                {
                    r=l1.remove();
                    if(r.pleft!=nullptr) l1.add(r.pleft);
                    if(r.pright!=nullptr) l1.add(r.pright);
                    r.parent=r.pleft=r.pright=null;
                    r.key=null;
                    r.value=null;
                }
            }
            else//if the buckeg is a linkedList
            {
                for(last=null;phead!=null;)
                {
                    last=phead;
                    phead=phead.pnext;
                    last.value=null;
                    last.key=null;
                    last.pnext=null;
                }
            }
            buckets[i]=null;
        }
        this.size=0;
        this.num=this.up=this.down=0;
        this.buckets=EMPTY_DATA;
        this.unInit=true;
        this.treeCount=0;
        this.counts=EMPTY_COUNTS;
    }
    @Override
    public Set<K> keySet()//passed
    {
        if(this.keySet==null) this.keySet=new KeySet2();
        return this.keySet;
    }
    @Override
    public ZLinkedHashSet<K> replicaKeySet()//passed
    {
        if(treeCount==0) return super.replicaKeySet();
        ZHashSet<K> set=new ZHashSet<>(size);
        TreeNode<K,V> r;
        ZLinkedStack<TreeNode> s=new ZLinkedStack<>();
        for(ZNode<K,V> phead:buckets)
        {
            if(phead instanceof TreeNode)
            {
                r=(TreeNode<K, V>) phead;
                do{
                    if(r!=nullptr) {s.push(r);r=r.pleft;}
                    else
                    {
                        r=s.pop();
                        set.add(r.key);
                        r=r.pright;
                    }
                }while(!s.isEmpty()||r!=nullptr);
                continue;
            }
            for(;phead!=null;phead=phead.pnext) set.add(phead.key);
        }
        return set;
    }
    @Override
    public Collection<V> values()//passed
    {
        if(this.valueCollection==null) this.valueCollection=new ValueCollection2();
        return this.valueCollection;
    }
    @Override
    public ZArrayList<V> replicaValues()//passed
    {
        if(treeCount==0) return super.replicaValues();
        TreeNode<K,V> r;
        ZLinkedStack<TreeNode> s=new ZLinkedStack<>();
        ZArrayList<V> arr=new ZArrayList<>(size);
        for(ZNode<K,V> phead:buckets)
        {
            if(phead instanceof TreeNode)
            {
                r=(TreeNode<K, V>) phead;
                do{
                    if(r!=nullptr) {s.push(r);r=r.pleft;}
                    else
                    {
                        r=s.pop();
                        arr.add(r.value);
                        r=r.pright;
                    }
                }while(!s.isEmpty()||r!=nullptr);
                continue;
            }
            for(;phead!=null;phead=phead.pnext) arr.add(phead.value);
        }
         return arr;
    }
    @Override
    public Set entrySet()//passed
    {
        if(this.entrySet==null) this.entrySet=new EntrySet2();
        return entrySet;
    }
    @Override
    public ZLinkedHashSet<? extends ZEntry<K, V>> replicaEntrySet()//passed
    {
        if(treeCount==0) return super.replicaEntrySet();
        ZHashSet<ZNode<K,V>> set=new ZHashSet<>(size);
        TreeNode<K,V> r;
        ZLinkedStack<TreeNode> s=new ZLinkedStack<>();
        for(ZNode<K,V> phead:buckets)
        {
            if(phead instanceof TreeNode)
            {
                r=(TreeNode<K, V>) phead;
                do{
                    if(r!=nullptr) {s.push(r);r=r.pleft;}
                    else
                    {
                        r=s.pop();
                        set.add(r);
                        r=r.pright;
                    }
                }while(!s.isEmpty()||r!=nullptr);
                continue;
            }
            for(;phead!=null;phead=phead.pnext) set.add(phead);
        }
        return set;
    }
    //</editor-fold>
}