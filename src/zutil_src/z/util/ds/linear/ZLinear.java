/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.ds.linear;

import z.util.ds.ZCollection;

/**
 *
 * @author dell
 * @param <T>
 */
public abstract class ZLinear<T> extends ZCollection<T>
{
    protected int size;
    //<editor-fold defaultstate="collapsed" desc="Basic-Function">
    @Override
    public int size()
    {
        return size;
    }
    @Override
    public int getIndexType() 
    {
        return LINEAR;
    }
    //</editor-fold>
}
