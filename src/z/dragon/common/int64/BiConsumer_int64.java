/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.common.int64;

/**
 * <long, V>
 * @author Gilgamesh
 * @param <V>
 */
public interface BiConsumer_int64<V> {
    void accept(long key, V value);
}
