/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.ds.imp;

/**
 *
 * @author dell
 * @param <T>
 */
public interface ZStack<T>
{
    public T peek();
    public T pop();
    public void push(T e);
    public boolean isEmpty();
}
