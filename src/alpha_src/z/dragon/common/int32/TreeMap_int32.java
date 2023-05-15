/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.common.int32;

import java.util.NoSuchElementException;
import java.util.Objects;
import z.dragon.common.int32.Entry_int32.FinalEnrty_int32;

/**
 * <int, V>
 * @author Gilgamesh
 * @param <V>
 */
public class TreeMap_int32<V>
{
    private static final boolean RED   = false;
    private static final boolean BLACK = true;
    
    //<editor-fold defaultstate="collapsed" desc="static class: Entry<V>">
    protected static final class Entry<V> implements Entry_int32<V> 
    {
        private int key;
        private V value;
        
        private Entry<V> left = null;
        private Entry<V> right = null;
        private Entry<V> parent = null;
        private boolean color = BLACK;

        Entry(int key, V value, Entry<V> parent) {
            this.key = key;
            this.value = value;
            this.parent = parent;
        }
       
        //<editor-fold defaultstate="collapsed" desc="functions">
        @Override public int getKey() { return key; }
        @Override public V getValue() {  return value; }

        @Override
        public V setValue(V value) {
            V oldValue = this.value;
            this.value = value;
            return oldValue;
        }

        @Override
        public boolean equals(Object o) {
            if (!(o instanceof Entry_int32)) return false;
            Entry_int32<?> e = (Entry_int32<?>) o;
            return valEquals(key,e.getKey()) && valEquals(value,e.getValue());
        }

        @Override
        public int hashCode() {
            int keyHash = (int) key;
            int valueHash = (value==null ? 0 : value.hashCode());
            return keyHash ^ valueHash;
        }

        @Override public String toString() {  return key + "=" + value; }
        //</editor-fold>
    }
    //</editor-fold>
    
    private Entry<V> root = null;
    private int size = 0;
    
    public TreeMap_int32() {}
    
    //<editor-fold defaultstate="collapsed" desc="Basic-functions">    
    public int size() {  return size; }
    
    public boolean isEmpty() { return size == 0; }
    
    public void clear() { size = 0; root = null; }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="inner-code: others">
    protected static final int key(Entry<?> e) {
        if(e==null) throw new NoSuchElementException();
        return e.key;
    }
    
    protected static final boolean valEquals(Object o1, Object o2) {
        return (o1==null ? o2==null : o1.equals(o2));
    }
    
    protected static final <V> Entry_int32<V> exportEntry(Entry<V> e) {
        return ((e != null) ?
                new FinalEnrty_int32<>(e.key, e.value) :
                null);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="inner-code: { get, first, last } Entry">
    protected final Entry<V> getEntry(int key) {//entry.key = key
        for(Entry<V> p = root; p != null; ) { 
            if(key < p.key) p = p.left;//left is smaller
            else if(key > p.key) p = p.right;//left is greater 
            else return p;
        }
        return null;
    }
      
    protected final Entry<V> getFirstEntry() {//find the minumum key
        Entry<V> p = root;
        if(p != null) while(p.left != null) p = p.left;
        return p;
    }
    
    protected final Entry<V> getLastEntry() {//find the maximum key
        Entry<V> p = root;
        if(p != null) while(p.right != null) p = p.right;
        return p;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="inner-code: { ceiling, floor } Entry">
    protected final Entry<V> getCeilingEntry(int key) {
        for(Entry<V> p = root; p != null; ) {
            if(key < p.key) {
                if (p.left != null) p = p.left;
                else return p;
            } 
            else if(key > p.key){ //key > p.key
                if (p.right != null) p = p.right;
                else {
                    Entry<V> parent = p.parent, child = p;
                    while(parent != null && child == parent.right) {
                        child = parent; parent = parent.parent;
                    }
                    return parent;
                }
            } 
            else return p;//key = p.key
        }
        return null;
    }
    
    protected final Entry<V> getFloorEntry(int key) {
        for(Entry<V> p = root; p != null; ) {
            if(key > p.key) {
                if(p.right != null) p = p.right;
                else return p;
            } 
            else if(key < p.key) {
                if(p.left != null) p = p.left;
                else{
                    Entry<V> parent = p.parent, child = p;
                    while(parent != null && child == parent.left) {
                        child = parent; parent = parent.parent;
                    }
                    return parent;
                }
            }
            else return p;//key = p.key
        }
        return null;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="inner-code: { higher, lower } Entry">
    protected final Entry<V> getHigherEntry(int key) {
        for(Entry<V> p = root; p != null; ) {
            if(key < p.key) {
                if(p.left != null) p = p.left;
                else return p;
            } 
            else {//key >= p.key
                if(p.right != null)  p = p.right;
                else {
                    Entry<V> parent = p.parent, child = p;
                    while (parent != null && child == parent.right) {
                        child = parent; parent = parent.parent;
                    }
                    return parent;
                }
            }
        }
        return null;
    }

    protected final Entry<V> getLowerEntry(int key) {
        for(Entry<V> p = root; p != null; ) {
            if(key > p.key) {
                if (p.right != null) p = p.right;
                else return p;
            } 
            else {//key <= p.key
                if (p.left != null) p = p.left;
                else {
                    Entry<V> parent = p.parent, child = p;
                    while (parent != null && child == parent.left) {
                        child = parent; parent = parent.parent;
                    }
                    return parent;
                }
            }
        }
        return null;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="inner-code: successor & predecessor">
    protected final static <V> Entry<V> successor(Entry<V> t) {//the next_node.key > t.key
        if(t == null) return null;
        else if (t.right != null) {
            Entry<V> p = t.right;
            while(p.left != null) p = p.left;
            return p;
        } 
        else {
            Entry<V> parent = t.parent, child = t;
            while(parent != null && child == parent.right) {
                child = parent; parent = parent.parent;
            }
            return parent;
        }
    }
    
    protected final static <V> Entry<V> predecessor(Entry<V> t) {//the last_node.key < t.key
        if(t == null) return null;
        else if (t.left != null) {
            Entry<V> p = t.left;
            while(p.right != null) p = p.right;
            return p;
        }
        else {
            Entry<V> parent = t.parent, child = t;
            while (parent != null && child == parent.left) {
                child = parent; parent = parent.parent;
            }
            return parent;
        }
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="inner-code: tree fix primitives">
    private static <V> boolean colorOf(Entry<V> p) { return (p == null ? BLACK : p.color); }
    private static <V> void setColor(Entry<V> p, boolean c) { if(p != null) p.color = c; }

    private static <V> Entry<V> parentOf(Entry<V> p) { return (p == null ? null: p.parent); }
    private static <V> Entry<V> leftOf(Entry<V> p) { return (p == null) ? null: p.left; }
    private static <V> Entry<V> rightOf(Entry<V> p) { return (p == null) ? null: p.right; }
    
    private void rotateLeft(Entry<V> p)  {
        if(p == null) return;

        Entry<V> r = p.right;//parent.right
        p.right = r.left;
            
        if(r.left != null) r.left.parent = p;
        r.parent = p.parent;
            
        if(p.parent == null) root = r;
        else if(p.parent.left == p) p.parent.left = r;
        else p.parent.right = r;
            
        r.left = p;
        p.parent = r;
    }
    
    private void rotateRight(Entry<V> p) {
        if(p == null) return;
        
        Entry<V> l = p.left;//parent.right
        p.left = l.right;
        
        if(l.right != null) l.right.parent = p;
        l.parent = p.parent;
        
        if(p.parent == null) root = l;
        else if(p.parent.right == p) p.parent.right = l;
        else p.parent.left = l;
        
        l.right = p;
        p.parent = l;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="inner-code: tree fix">
    private void fixAfterInsertion(Entry<V> x)
    {
        x.color = RED;
        while (x != null && x != root && x.parent.color == RED) 
        {
            if (parentOf(x) == leftOf(parentOf(parentOf(x)))) {
                Entry<V> y = rightOf(parentOf(parentOf(x)));
                if (colorOf(y) == RED) {
                    setColor(parentOf(x), BLACK);
                    setColor(y, BLACK);
                    setColor(parentOf(parentOf(x)), RED);
                    x = parentOf(parentOf(x));
                } 
                else {
                    if (x == rightOf(parentOf(x))) {
                        x = parentOf(x);
                        rotateLeft(x);
                    }
                    setColor(parentOf(x), BLACK);
                    setColor(parentOf(parentOf(x)), RED);
                    rotateRight(parentOf(parentOf(x)));
                }
            }
            else {
                Entry<V> y = leftOf(parentOf(parentOf(x)));
                if (colorOf(y) == RED) {
                    setColor(parentOf(x), BLACK);
                    setColor(y, BLACK);
                    setColor(parentOf(parentOf(x)), RED);
                    x = parentOf(parentOf(x));
                } 
                else {
                    if (x == leftOf(parentOf(x))) {
                        x = parentOf(x);
                        rotateRight(x);
                    }
                    setColor(parentOf(x), BLACK);
                    setColor(parentOf(parentOf(x)), RED);
                    rotateLeft(parentOf(parentOf(x)));
                }
            }
        }
        root.color = BLACK;
    }
    
    private void fixAfterDeletion(Entry<V> x) 
    {
        while (x != root && colorOf(x) == BLACK) 
        {
            if (x == leftOf(parentOf(x))) {
                Entry<V> sib = rightOf(parentOf(x));

                if (colorOf(sib) == RED) {
                    setColor(sib, BLACK);
                    setColor(parentOf(x), RED);
                    rotateLeft(parentOf(x));
                    sib = rightOf(parentOf(x));
                }

                if (colorOf(leftOf(sib))  == BLACK &&
                    colorOf(rightOf(sib)) == BLACK) {
                    setColor(sib, RED);
                    x = parentOf(x);
                } 
                else {
                    if (colorOf(rightOf(sib)) == BLACK) {
                        setColor(leftOf(sib), BLACK);
                        setColor(sib, RED);
                        rotateRight(sib);
                        sib = rightOf(parentOf(x));
                    }
                    setColor(sib, colorOf(parentOf(x)));
                    setColor(parentOf(x), BLACK);
                    setColor(rightOf(sib), BLACK);
                    rotateLeft(parentOf(x));
                    x = root;
                }
            } 
            else {//symmetric
                Entry<V> sib = leftOf(parentOf(x));

                if (colorOf(sib) == RED) {
                    setColor(sib, BLACK);
                    setColor(parentOf(x), RED);
                    rotateRight(parentOf(x));
                    sib = leftOf(parentOf(x));
                }

                if (colorOf(rightOf(sib)) == BLACK &&
                    colorOf(leftOf(sib)) == BLACK) {
                    setColor(sib, RED);
                    x = parentOf(x);
                } 
                else {
                    if (colorOf(leftOf(sib)) == BLACK) {
                        setColor(rightOf(sib), BLACK);
                        setColor(sib, RED);
                        rotateLeft(sib);
                        sib = leftOf(parentOf(x));
                    }
                    setColor(sib, colorOf(parentOf(x)));
                    setColor(parentOf(x), BLACK);
                    setColor(leftOf(sib), BLACK);
                    rotateRight(parentOf(x));
                    x = root;
                }
            }
        }
        setColor(x, BLACK);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="inner-code: delete Entry">
    private void deleteEntry(Entry<V> p) {
        size--;
        if(p.left != null && p.right != null) {
            Entry<V> s = successor(p);
            p.key = s.key;
            p.value = s.value;
            p = s;
        }

        //Start fixup at replacement node, if it exists.
        Entry<V> replacement = (p.left != null ? p.left : p.right);
        if (replacement != null) {
            replacement.parent = p.parent;// Link replacement to parent
            if (p.parent == null) root = replacement;
            else if(p == p.parent.left) p.parent.left = replacement;
            else p.parent.right = replacement;

            // Null out links so they are OK to use by fixAfterDeletion.
            p.left = p.right = p.parent = null;

            if(p.color == BLACK) fixAfterDeletion(replacement); //Fix replacement
        } 
        else if (p.parent == null) root = null;//return if we are the only node.
        else {// No children. Use self as phantom replacement and unlink.
            if(p.color == BLACK) fixAfterDeletion(p);
            if(p.parent != null) {
                if(p == p.parent.left) p.parent.left = null;
                else if(p == p.parent.right) p.parent.right = null;
                p.parent = null;
            }
        }
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="read-operators">    
    public boolean containsKey(int key) { return getEntry(key) != null; }
    public boolean containsValue(Object value) {
        for(Entry<V> e = getFirstEntry(); e != null; e = successor(e))
            if(valEquals(value, e.value)) return true;
        return false;
    }
    
    public V get(int key) {
        Entry<V> p = getEntry(key);
        return (p==null ? null : p.value);
    }
    
    public int firstKey() { return key(getFirstEntry()); }
    public int lastKey() { return key(getLastEntry()); }
    public int lowerKey(int key) { return key(getLowerEntry(key)); }
    public int higherKey(int key) { return key(getHigherEntry(key)); }
    public int floorKey(int key) { return key(getFloorEntry(key)); }
    public int ceilingKey(int key) { return key(getCeilingEntry(key)); }
    
    public Entry_int32<V> firstEntry() { return exportEntry(getFirstEntry()); }
    public Entry_int32<V> lastEntry() { return exportEntry(getLastEntry()); }
    public Entry_int32<V> lowerEntry(int key) { return exportEntry(getLowerEntry(key)); }
    public Entry_int32<V> higherEntry(int key) { return exportEntry(getHigherEntry(key)); }
    public Entry_int32<V> floorEntry(int key) { return exportEntry(getFloorEntry(key)); }
    public Entry_int32<V> ceilingEntry(int key) { return exportEntry(getCeilingEntry(key)); }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="write-operators">    
    public V put(int key, V value) 
    {
        Entry<V> t = root;
        if(t == null) {//create root node
            root = new Entry<>(key, value, null);
            size = 1;
            return null;
        }
       
        Entry<V> parent; 
        do {//find the parent
            parent = t;
            if(key < t.key) t = t.left;//left is smaller
            else if (key > t.key) t = t.right;//right is bigger
            else return t.setValue(value);//t.key == key
        } while (t != null);

        Entry<V> e = new Entry<>(key, value, parent);
        if(key < parent.key) parent.left = e;//parent.key > key
        else parent.right = e;
        
        fixAfterInsertion(e);
        size++;
        return null;
    }
    
    public V remove(int key) {
        Entry<V> p = getEntry(key);
        if(p == null) return null;

        V old_value = p.value;
        deleteEntry(p);
        return old_value;
    }
    
    public Entry_int32<V> pollFirstEntry() {
        Entry<V> p = getFirstEntry();
        Entry_int32<V> result = exportEntry(p);
        if(p != null) deleteEntry(p);
        return result;
    }
    
    public Entry_int32<V> pollLastEntry() {
        Entry<V> p = getLastEntry();
        Entry_int32<V> result = exportEntry(p);
        if(p != null) deleteEntry(p);
        return result;
    }
    
    public boolean replace(int key, V old_value, V new_value) {
        Entry<V> p = getEntry(key);
        if(p!=null && Objects.equals(old_value, p.value)) {
            p.value = new_value;
            return true;
        }
        return false;
    }

    public V replace(int key, V value) {
        Entry<V> p = getEntry(key);
        if(p != null) {
            V old_value = p.value;
            p.value = value;
            return old_value;
        }
        return null;
    }

    public void forEach(BiConsumer_int32<? super V> action) {
        Objects.requireNonNull(action);
        for(Entry<V> e = getFirstEntry(); e != null; e = successor(e)) {
            action.accept(e.key, e.value);
        }
    }
    //</editor-fold>
}
