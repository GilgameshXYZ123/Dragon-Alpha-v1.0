/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.common.int64;

import java.util.Arrays;

/**
 *
 * @author Gilgamesh
 */
public class LinkedList_int64 
{
    //<editor-fold defaultstate="collapsed" desc="class: Lnode">
    protected static final class Node 
    {
        protected long value = 0;
        protected Node last = null;
        
        protected Node(long value, Node last) {
            this.value = value;
            this.last = last;
        }
        
        @Override
        public String toString() {
            return "{ value = " + value + ", has_last = " + (last != null) + "}";
        }
    }
    //</editor-fold>
    
    public LinkedList_int64() {}
    
    private Node ptail = null;
    private int size = 0;
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public int size() { return size; }
    
    public boolean isEmpty() { return size == 0; }
    
    public long[] toArray_int64() {
        long[] arr = new long[size]; int index = 0;
        for(Node n = ptail; n!= null; n=n.last) arr[index++] = n.value;
        return arr;
    }
    
    @Override
    public String toString() {
        return Arrays.toString(toArray_int64());
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area">
    private void unlink(Node last, Node target, Node next) {
        if(next != null) next.last = last;
        else ptail = ptail.last;//next = null, target == ptail
        target.last = null; 
        size--;
    }
    
    public boolean add(long value) {
        ptail = new Node(value, ptail); size++;
        return true;
    }
    
    public boolean addAl(long[] values) {
        for(long value : values) ptail = new Node(value, ptail);
        size += values.length;
        return true;
    }
    
    public long end() { return ptail.value; }
    
    public boolean contains(long value) {
        for(Node n = ptail; n != null; n = n.last)
            if(n.value == value) return true;
        return false;
    }

    public boolean remove(long value) {//only remove the last value
        int osize = size;
        for(Node n = ptail, next = null; n != null; ) {
            if(n.value != value) { next = n; n = n.last; continue; }
            this.unlink(n.last, n, next);
            break;
        }
        return osize != size;
    } 
    
    public long remove() {//use isEmpty before remove
        Node n = ptail;
        ptail = ptail.last;
        n.last = null;
        size--;
        return n.value;
    }
    
    public void clear() {
        while(ptail != null) {
            Node cur = ptail;
            ptail = ptail.last;
            cur.last = null;
        }
        size = 0;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="class: Iter<long>">
    public static class Iter implements Iter_int64{
        private Node cur;
        
        private Iter(Node cur) {
            this.cur = cur;
        }
        
        @Override
        public boolean hasNext() {
            return (cur != null) && (cur.last != null);
        }
        
        @Override
        public long next() {
            long value = cur.value;
            cur = cur.last;
            return value;
        }
    }
    //</editor-fold>
    public Iter_int64 iterator() { return new Iter(ptail); }
}
