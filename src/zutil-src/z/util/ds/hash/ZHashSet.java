/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.ds.hash;

import z.util.ds.hash.ZLinkedHashSet;
import java.lang.reflect.Array;
import java.util.Iterator;
import java.util.function.BiPredicate;
import java.util.function.Consumer;
import java.util.function.Predicate;
import static z.util.ds.Constants.Hash.DEF_EXPEND_RATE;
import static z.util.ds.Constants.Hash.DEF_EXPEND_THRESHOLD;
import static z.util.ds.Constants.Hash.DEF_INIT_SIZE;
import static z.util.ds.Constants.Hash.DEF_SHRINK_THRESHOLD;
import static z.util.ds.Constants.Hash.DEF_TREE_THRESHOLD;
import static z.util.ds.Constants.Hash.DEF_LINKED_THRESHOLD;
import z.util.lang.exception.IAE;
import z.util.ds.linear.ZLinkedList;
import z.util.ds.linear.ZLinkedStack;
import z.util.lang.annotation.Passed;
import static z.util.math.ExMath.DEF_HASHCODER;
import z.util.math.ExMath.HashCoder;
import z.util.math.vector.Vector;

/**
 *<pre>
 * ZHashSet is an optimized implemetation by using Hash-Index.
 * 1.A ZHashSet has many buckets, Elements with the same hashCode will be 
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
 * @code{ 
 * int num=10;
 * ExRandom ex=Lang.exRandom();
 * int[] v=new int[num];
 * for(int i=0;i<num;i++) map.put(v[i]=map.size*i+1, i);}
 * as it can cause a high posibility for element conflict.
 * </pre>
 * 
 * @author dell
 * @param <T>
 */
@Passed
public class ZHashSet<T extends Comparable> extends ZLinkedHashSet<T>
{
    //<editor-fold defaultstate="collapsed" desc="static class TreeNode<T2>">
    private static final class TreeNode<T2 extends Comparable> extends ZLinkedHashSet.ZNode<T2>
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
        TreeNode(T2 key, int hash) 
        {
            super(key, hash);
            this.color=true;
            
            this.parent=nullptr;
            this.pleft=nullptr;
            this.pright=nullptr;
        }
        TreeNode(T2 key, int hash, TreeNode<T2> parent) 
        {
            super(key, hash);
            this.color=true;
            
            this.parent=parent;
            this.pleft=nullptr;
            this.pright=nullptr;
        }
        TreeNode(ZNode<T2> node)
        {
            super(node.key, node.hash);
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
    private static final int[] EMPTY_COUNTS={};
    private final int treeThreshold;
    private final int linkedThreshold;
    
    private int[] counts;
    private int treeCount;
    
    @Passed
    public ZHashSet(int size, double expendThreshold, double shrinkThreshold, 
           double expendRate, HashCoder coder,
           int treeThreshold, int linkedThreshold)
    {
        super(size, expendThreshold, shrinkThreshold, expendRate, coder);
        
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
    public ZHashSet(int size, HashCoder coder, int treeThreshold, int linkedThreshold)
    {
       this(size, DEF_EXPEND_THRESHOLD, DEF_SHRINK_THRESHOLD, DEF_EXPEND_RATE, coder,
                treeThreshold, linkedThreshold);
    }
    public ZHashSet(int size, HashCoder coder)
    {
         this(size, DEF_EXPEND_THRESHOLD, DEF_SHRINK_THRESHOLD, DEF_EXPEND_RATE, coder,
                DEF_TREE_THRESHOLD, DEF_LINKED_THRESHOLD);
    }
    public ZHashSet(int size)
    {
        this(size, DEF_EXPEND_THRESHOLD, DEF_SHRINK_THRESHOLD, DEF_EXPEND_RATE, DEF_HASHCODER,
                DEF_TREE_THRESHOLD, DEF_LINKED_THRESHOLD);
    }
    public ZHashSet()
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
            TreeNode<T> r;
            for(ZNode<T> phead:buckets)
            {
                if(phead instanceof TreeNode)
                {
                    r=(TreeNode<T>) phead;
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
        ZNode<T> phead=null;
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
        TreeNode<T> r=(TreeNode<T>) buckets[index],last=nullptr;
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
    private TreeNode<T> treeFind(T key, int index)
    {
        TreeNode<T> r=(TreeNode<T>) buckets[index];
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
        
        if(--num<=down) {this.shrink(1);return;}//need to shrink, no need to removeFix
        if(--counts[index]<=this.linkedThreshold) {this.toLinked(index);return;}
        if(x!=nullptr&&!oc) this.removeFix(x, index);//oc is black
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Tree-LinkedList-Transfer">
    @Passed
    private void toTree(int index)
    {
        ZNode<T> phead=buckets[index],last=null;
        buckets[index]=nullptr;
        while(phead!=null)
        {
            last=phead;
            phead=phead.pnext;
            this.treeAdd(new TreeNode(last), index);
            last.key=null;
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
            pnow.pnext=new ZNode();
            pnow=pnow.pnext;
            //delete old treenode
            r.parent=r.pleft=r.pright=null;
            r.key=null;
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
    //<editor-fold defaultstate="collapsed" desc="function-group:addT">
    @Passed  
    private boolean linkedAddT(T key, int hash, int index)
    {
        ZNode phead=buckets[index];
        if(key==null)//check if there exists conflict
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
        //no conflict, add new node
        if(num+1>=up) {this.expend(1);return this.addT(key, hash);}//to a tree or linked
        
        if((counts[index]+1)>this.treeThreshold)
        {
            this.toTree(index);
            num++;
            return this.treeAddT(key, hash, index);//num++
        }
        
        phead.pnext=new ZNode(key, hash);
        num++;    
        counts[index]++;
        return true;
    }
    @Passed
    private boolean treeAddT(T key, int hash, int index)
    {
        //if a bucket is a tree, it's not null.
        TreeNode<T> r=(TreeNode<T>) buckets[index],last=nullptr;
        int v;
        if(key==null)//check if there exists conflict
        {
            while(r!=nullptr)
            {
                last=r;
                if(r.key==null) return false;
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
                else return false;
            }
        }
        r=new TreeNode(key, hash, last);
        if(last==nullptr) buckets[index]=r;
        else if(last.key.compareTo(key)>0) last.pleft=r;
        else last.pright=r;
        
        if(num+1>=up) {this.expend(1);return this.addT(key, hash);}//to a tree or linked
        
        this.insertFix(r, index);
        num++;
        counts[index]++;
        return true;
    }
    @Override   
    protected boolean addT(T key, int hash)//passed
    {
        int index=hash%size;//after expend,and back to here, the size is changed
        if(buckets[index]==null)
        {
            if(num+1>=up) {expend(1);return addT(key, hash);}
            buckets[index]=new ZNode(key, hash);
            num++;
            counts[index]++;
            return true;
        }
        return (buckets[index] instanceof TreeNode? 
                this.treeAddT(key, hash, index):
                this.linkedAddT(key, hash, index));
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="ZLinkedHashSet"> 
    @Override
    protected boolean removeNode(T key, int hash)//passed
    {
        int index=hash%size;
        if(buckets[index]==null) return false;
        if(buckets[index] instanceof TreeNode)//the bucket is a tree
        {
            TreeNode<T> r=this.treeFind(key, index);
            if(r==null) return false;
            this.treeRemove(r, index);//num--,counts[index--]
            return true;
        }
        ZNode<T> phead=buckets[index],last=null;//the bucket is a linkedList
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
                if(--num<=down) this.shrink(1);
                else counts[index]--;
                return true;
            }
        }
        return false;
    }
    @Override
    protected ZNode<T> findNode(T key, int hash)//passed
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
    protected boolean innerRemoveAll(BiPredicate pre, Object condition)//passed
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
                    if(pre.test(r.key, condition)) l2.add(r);
                }
                count+=l2_num=l2.number();
                while(!l2.isEmpty()) this.treeRemove(l2.pop(), i);
                if((counts[i]-=l2_num)<=this.shrinkThreshold) toLink[i]=true;
                continue;
            }
            for(phead=buckets[i],last=null;phead!=null;)//the bucket is a linkedList
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
    public void forEach(Consumer<? super T> con)//passed
    {
        if(treeCount==0) {super.forEach(con);return;}
        ZLinkedStack<TreeNode> s=new ZLinkedStack<>();
        TreeNode<T> r;
        for(ZNode<T> phead:buckets)
        {
            if(phead instanceof TreeNode)
            {
                r=(TreeNode<T>) phead;
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
                }
            }
            else//if the buckeg is a linkedList
            {
                for(last=null;phead!=null;)
                {
                    last=phead;
                    phead=phead.pnext;
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
    public Object[] toArray() 
    {
        if(treeCount==0) return super.toArray();
        Object[] arr=new Object[num];
        ZLinkedStack<TreeNode> s=new ZLinkedStack<>();
        TreeNode<T> r;
        int index=0;
        for(ZNode<T> phead:buckets)
        {
            if(phead instanceof TreeNode)
            {
                r=(TreeNode<T>) phead;
                do{
                    if(r!=nullptr) {s.push(r);r=r.pleft;}
                    else{
                        r=s.pop();
                        arr[index++]=r;
                        r=r.pright;
                    }
                }while(!s.isEmpty()||r!=nullptr);
                continue;
            }
            for(;phead!=null;phead=phead.pnext) arr[index++]=phead;
        }
        return arr;
    }
    @Override
    public <P> P[] toArray(P[] a) 
    {
        if(treeCount==0) return super.toArray(a);
        if(a.length<num)
        a=(P[]) Array.newInstance(a.getClass().getComponentType(), num);
        Object[] arr=new Object[num];
        ZLinkedStack<TreeNode> s=new ZLinkedStack<>();
        TreeNode<T> r;
        int index=0;
        for(ZNode<T> phead:buckets)
        {
            if(phead instanceof TreeNode)
            {
                r=(TreeNode<T>) phead;
                do{
                    if(r!=nullptr) {s.push(r);r=r.pleft;}
                    else{
                        r=s.pop();
                        arr[index++]=r;
                        r=r.pright;
                    }
                }while(!s.isEmpty()||r!=nullptr);
                continue;
            }
            for(;phead!=null;phead=phead.pnext) arr[index++]=phead;
        }
        if(a.length>num) a[num]=null;
        return a;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Iterable">
    @Passed
    private final class Iter implements Iterator<T>
    {
        //columns---------------------------------------------------------------
        private int index;
        private ZNode<T> phead;
        private boolean inTree;
        private ZLinkedStack<TreeNode> s;
        
        //functions-------------------------------------------------------------
        Iter()
        {
            index=0;
            phead=ZHashSet.this.buckets[0];
             inTree=phead instanceof TreeNode;
            s=new ZLinkedStack<>();
        }
        @Override
        public boolean hasNext() 
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
        @Override
        public T next() 
        {
           if(inTree)
           {
                TreeNode<T> r=(TreeNode<T>) phead, next;
                for(;r!=nullptr;s.push(r),r=r.pleft);
                next=r=s.pop();
                phead=r.pright;
                return next.key;
           }
           ZNode<T> last=phead;
           phead=phead.pnext;
           return last.key;
        }
    }
    @Override
    public Iterator<T> iterator()
    {
        return new Iter();
    }
    //</editor-fold>
}
