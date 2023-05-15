/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.ds.imp;

import java.util.function.BiPredicate;

/**
 *
 * @author dell
 */
public interface CollectionRemoveExtensive extends RemoveExtensive
{
    public boolean removeAll(BiPredicate pre, Object condition);
    public boolean retainAll(BiPredicate pre, Object condition);
}
