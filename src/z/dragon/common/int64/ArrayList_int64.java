/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.common.int64;

import java.util.Arrays;
import java.util.Collection;
import java.util.ConcurrentModificationException;
import java.util.NoSuchElementException;

/**
 *
 * @author Gilgamesh
 */
public class ArrayList_int64
{
    public static final int MAX_ARRAY_SIZE = Integer.MAX_VALUE - 8;
    public static final int DEFAULT_CAPACITY = 10;

    private static final long[] EMPTY_ELEMENTDATA = {};
    private static final long[] DEFAULT_CAPACITY_EMPTY_ELEMENTDATA = {};

    public ArrayList_int64(int init_capacity) {
        if(init_capacity < 0) throw new IllegalArgumentException("Illegal Capacity: "+ init_capacity);
        this.element = (init_capacity > 0 ? 
                new long[init_capacity] : 
                EMPTY_ELEMENTDATA);
    }

    public ArrayList_int64() {
        this.element = DEFAULT_CAPACITY_EMPTY_ELEMENTDATA;
    }
    
    //<editor-fold defaultstate="collapsed" desc="member-params & Basic-Functions">
    protected long[] element; // non-private to simplify nested class access
    protected int size;
    
    public int size() { return size; }

    public long[] element() { return element; }
    
    public boolean isEmpty() { return size == 0; }

    public long[] toArray_int64() { return Arrays.copyOf(element, size); }
    
    public void clear() { size = 0; }
    
    @Override public String toString() { return Arrays.toString(element); }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="inner-code">
    private String outOfBoundsMsg(int index) { return "Index: "+index+", Size: "+size; }
    
    public void ensureCapacity(int min_capacity) {
        int minExpand = (element != DEFAULT_CAPACITY_EMPTY_ELEMENTDATA) ? 
                0: 
                DEFAULT_CAPACITY; 
        if (min_capacity > minExpand) ensureExplicitCapacity(min_capacity);
    }

    private static int calculateCapacity(long[] elementData, int min_capacity) {
        if (elementData == DEFAULT_CAPACITY_EMPTY_ELEMENTDATA) {
            return (DEFAULT_CAPACITY > min_capacity? 
                    DEFAULT_CAPACITY:
                    min_capacity);//max(DEFAULT_CAPACITY, minCapacity);
        }
        return min_capacity;
    }

    private void ensureCapacityInternal(int min_capacity) {
        ensureExplicitCapacity(calculateCapacity(element, min_capacity));
    }

    private void ensureExplicitCapacity(int min_capacity) {
        if (min_capacity - element.length > 0) grow(min_capacity);//overflow-conscious code
    }

    private void grow(int min_capacity) {
        int old_capacity = element.length;
        int new_capacity = old_capacity + (old_capacity >> 1);//1.5 * old_capacity
        if (new_capacity - min_capacity < 0) new_capacity = min_capacity;//new_capcity >= min_capacity
        if (new_capacity - MAX_ARRAY_SIZE > 0) new_capacity = hugeCapacity(min_capacity);
        
        element = Arrays.copyOf(element, new_capacity);
    }

    private static int hugeCapacity(int min_capacity) {
        if (min_capacity < 0) throw new OutOfMemoryError("ArrayList_int64: out of memory");
        return ((min_capacity > MAX_ARRAY_SIZE) ?
                Integer.MAX_VALUE :
                MAX_ARRAY_SIZE);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="operators">
    public void trimToSize() {
        if(size >= element.length) return;
        element = ((size > 0)? Arrays.copyOf(element, size): EMPTY_ELEMENTDATA);
    }
      
    public boolean contains(long value) {  return indexOf(value) >= 0; }
    
    public int indexOf(long value) {
        for(int i=0; i < size; i++) if (value == element[i]) return i;
        return -1;
    }
    
    public int lastIndexOf(long value) {
        for(int i = size-1; i>=0; i--) if (value == element[i]) return i;
        return -1;
    }
    
    public long get(int index) {
       if (index >= size) throw new IndexOutOfBoundsException(outOfBoundsMsg(index));
        return element[index];
    }
    
    public long set(int index, long value) {
        if (index >= size) throw new IndexOutOfBoundsException(outOfBoundsMsg(index));
        long old_value = element[index];
        element[index] = value;
        return old_value;
    }
    
    public boolean add(long value) {
        ensureCapacityInternal(size + 1);  // Increments modCount!!
        element[size++] = value;
        return true;
    }
    
     public void add(int index, long element)  {
        if(index > size || index < 0) throw new IndexOutOfBoundsException(outOfBoundsMsg(index));
        ensureCapacityInternal(size + 1);// Increments modCount!!
        System.arraycopy(this.element, index, this.element, index + 1, size - index);
        this.element[index] = element;
        size++;
    }
     
    public long remove(int index) {
        if(index >= size) throw new IndexOutOfBoundsException(outOfBoundsMsg(index));

        long old_value = element[index];

        int numMoved = size - index - 1;
        if (numMoved > 0) System.arraycopy(element, index+1, element, index, numMoved);
        
        size--;
        return old_value;
    }
    
    public boolean remove(long value) {
        for (int index = 0; index < size; index++)
            if (value == element[index]) {
                fastRemove(index);
                return true;
            }
        return false;
    }
    
    private void fastRemove(int index) {
        int numMoved = size - index - 1;
        if (numMoved > 0)
            System.arraycopy(element, index+1, element, index, numMoved);
        size--;
    }
    
    public boolean addAll(long[] values) {
        if(values == null || values.length == 0) return false;
        int total_num = values.length;
        
        ensureCapacityInternal(size + total_num);  // Increments modCount
        System.arraycopy(values, 0, element, size, total_num);
        size += total_num;
        return true;
    }
    
    public boolean addAll(int index, long[] values) {
        if(values == null || values.length == 0) return false;
        if(index > size || index < 0) throw new IndexOutOfBoundsException(outOfBoundsMsg(index));

        int total_num = values.length;
        ensureCapacityInternal(size + total_num);  // Increments modCount

        int numMoved = size - index;
        if (numMoved > 0)
            System.arraycopy(element, index, element, index + total_num, 
                    numMoved);
        System.arraycopy(values, 0, element, index, total_num);
        
        size += total_num;
        return true;
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="class Iter<long>">
    private class Itr implements Iter_int64{
        int cursor;       //index of next element to return
        int lastRet = -1; //index of last element returned; -1 if no such

        Itr() {}

        @Override public boolean hasNext() { return cursor != size; }

        @Override
        public long next() {
            int i = cursor;
            if (i >= size) throw new NoSuchElementException();
            
            long[] elementData = ArrayList_int64.this.element;
            if (i >= elementData.length) throw new ConcurrentModificationException();
            
            cursor = i + 1;
            return elementData[lastRet = i];
        }

        public void remove() {
            if (lastRet < 0) throw new IllegalStateException();

            try {
                ArrayList_int64.this.remove(lastRet);
                cursor = lastRet;
                lastRet = -1;
            } catch (IndexOutOfBoundsException ex) {
                throw new ConcurrentModificationException();
            }
        }
    }
    //</editor-fold>
    public Iter_int64 iterator() { return new Itr(); }
}
