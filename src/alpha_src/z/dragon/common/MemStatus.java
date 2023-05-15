/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.common;

/**
 *
 * @author Gilgamesh
 */
public interface MemStatus 
{
    public long max_mem_size();
    public long total_mem_size();
    public long used_mem_size();
     
    default long max_mem_size_MB() { return max_mem_size() >> 20; }
    default long total_mem_size_MB() { return total_mem_size() >> 20; }
    default long used_mem_size_MB() { return used_mem_size() >> 20; }
    default long buffered_mem_size_MB() { return buffered_mem_size() >> 20; }
    
    default float used_mem_percent() { return 1.0f * used_mem_size() / total_mem_size(); }
    default long buffered_mem_size() { return (total_mem_size() - used_mem_size()); }
    default float buffered_mem_percent() { return 1.0f - used_mem_percent(); }
}
