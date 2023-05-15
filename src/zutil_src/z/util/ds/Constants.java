/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.ds;

import z.util.lang.annotation.Desc;

/**
 *
 * @author dell
 */
public final class Constants 
{
    public static final class Hash
    {
        public static final double DEF_EXPEND_THRESHOLD = 0.75; //0.375 -gt 0.67
        public static final double DEF_SHRINK_THRESHOLD = 0.25;
        public static final int DEF_INIT_SIZE = 16;
        public static final double DEF_EXPEND_RATE = 2.0;
        
        public static final int DEF_TREE_THRESHOLD=8;
        public static final int DEF_LINKED_THRESHOLD=6;
    }
    public static final class Array
    {
        @Desc(value="To expend the space of this Array List, as newSize=size*growthRate")
        public static final double DEF_EXPEND_RATE=2.0;
        @Desc(value="The default initialized size of the List")
        public static final int DEF_INIT_SIZE=16;
    }
}
