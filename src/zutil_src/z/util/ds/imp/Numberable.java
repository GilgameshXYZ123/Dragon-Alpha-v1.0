/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.ds.imp;

/**
 *
 * @author dell
 */
public interface Numberable
{
    /**
     * There exists such cases: the size and the num of a specific Map 
     * or Collection is not the same. 
     * When you create an ZHashMap:
     * (1) size refers to the number of Bucket
     * (2) num refers to the number of Entries.
     * Also, When you create an ZArrayList:
     * @return 
     */
    public int number();
}
