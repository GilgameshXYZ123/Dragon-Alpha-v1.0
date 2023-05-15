/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.ds.imp;

import java.util.Collection;
import java.util.HashSet;
import java.util.TreeSet;

/**
 *
 * @author dell
 */
public interface Indexable 
{
    public static final int LINEAR=1;
    public static final int TREE=2;
    public static final int HASH=3;
    public static final int HEAP=4;
    
    public int getIndexType();//use conf-file to instead if possible
    public static boolean isHash(Collection c)
    {
        if(c instanceof Indexable) return ((Indexable)c).getIndexType()==HASH;
        else if(c instanceof HashSet) return true;
        return false;
    }
    public static boolean isTree(Collection c)
    {
        if(c instanceof Indexable) return ((Indexable)c).getIndexType()==TREE;
        else if(c instanceof TreeSet) return true;
        return false;
    }
}
