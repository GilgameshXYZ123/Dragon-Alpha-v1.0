/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.ds.imp;

import java.util.function.Predicate;

/**
 *
 * @author dell
 */
public interface RemoveExtensive 
{
    public boolean removeAll(Predicate pre);
    public boolean retainAll(Predicate pre);
}
