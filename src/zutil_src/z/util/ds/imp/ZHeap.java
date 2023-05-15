/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.ds.imp;

import z.util.math.vector.Vector;

/**
 * An interface to define the minimum heap.
 * @author dell
 * @param <T>
 */
public interface ZHeap<T> 
{
    public T findMin();
    public T removeMin();
    public static boolean isHeap(ZHeap heap, int size)
    {
        if(heap==null) throw new NullPointerException();
        Comparable[] v=new Comparable[size];
        for(int i=0;i<size;i++) v[i]=(Comparable) heap.removeMin();
        Vector.println(System.out, v);
        return Vector.isAscendingOrder(v, 0, size-1);
    }
}
