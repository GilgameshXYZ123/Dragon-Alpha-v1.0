///*
// * To change this license header, choose License Headers in Project Properties.
// * To change this template file, choose Tools | Templates
// * and open the template in the editor.
// */
//package z.util.ds.heap;
//
///**
// *
// * @author dell
// * @param <T>
// */
//public class ZBinomialHeap
//{
//    private static final class Node
//    {
//        int key;
//        int degree;
//        Node parent;
//        Node pleft;
//        Node psibling;
//    }
//    
//    private Node root;
//    
//    public ZBinomialHeap()
//    {
//        root=new Node();
//    }
//    private Node findMinHeap(Node heap)
//    {
//        Node min=null,cur=null;
//        int minK=Integer.MAX_VALUE;
//        while(cur!=null)
//        {
//            if(cur.key<minK) {minK=cur.key;min=cur;}
//            cur=cur.psibling;
//        }
//        return min;
//    }
//    //即将两棵根节点度数相同的二项树Bk-1连接成一棵Bk
//    private void Link()
//    {
//        
//    }
//    //将H1和H2的根表合并成一个按度数的单调递增次序排列的链表。
//    private Node heapMerge(Node h1, Node h2)
//    {
//        
//    }
//    //反复连接根节点的度数相同的各二项树
//    private Node heapUnion(Node h1, Node h2)
//    {
//        Node heap=null, pre_x=null, x ,next_x=null;
//        heap=this.heapMerge(h1, h2);
//        if(heap==null) return heap;
//        
//        pre_x=null;
//        x=heap;
//        next_x=x.psibling;
//        
//    } 
//        
//    
//}
