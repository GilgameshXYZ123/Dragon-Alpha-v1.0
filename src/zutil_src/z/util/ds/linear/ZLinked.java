/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.ds.linear;

/**
 *
 * @author dell
 * @param <T>
 */
public abstract class ZLinked<T> extends ZLinear<T>
{
    @Override
    public int number()
    {
        return size;
    }
    @Override
    public boolean isEmpty()//checked
    {
        return size==0;
    }
}
