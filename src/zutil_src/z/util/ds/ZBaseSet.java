/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.ds;

/**
 *
 * @author dell
 * @param <T>
 */
public abstract class ZBaseSet<T extends Comparable> extends ZSet<T>
{
    protected int size;
    
    @Override
    public int size()
    {
        return size;
    }
    @Override
    public boolean isEmpty()
    {
        return size==0;
    }
}
