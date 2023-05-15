/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.common.int32;

/**
 * <int, V>
 * @author Gilgamesh
 * @param <V>
 */
public interface BiConsumer_int32<V> {
    void accept(int key, V value);
}
