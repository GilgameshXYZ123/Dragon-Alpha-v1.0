/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.data;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Map.Entry;
import java.util.function.BiConsumer;
import java.util.function.Consumer;

/**
 *
 * @author Gilgamesh
 * @param <K>
 * @param <V>
 */
public class EntryList<K, V> extends ArrayList<Entry<K, V>>
{
    private static final long serialVersionUID = 2023320L;
    
    public EntryList() { super(); }
    public EntryList(int init_capacity) {
        super(init_capacity);
    }
    public EntryList(Collection<? extends Entry<K, V>> c) {
        super(c);
    }
    
    public boolean put(K key, V value) {
        return add(new Pair<>(key, value));
    }

    public void forEach(BiConsumer<K, V> action) {
        forEach((Entry<K, V> kv)->{ action.accept(kv.getKey(), kv.getValue()); });
    }
}
