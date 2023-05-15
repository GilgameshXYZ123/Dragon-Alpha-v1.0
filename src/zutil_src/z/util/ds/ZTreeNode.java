/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.ds;

import java.util.LinkedList;
import java.util.Map.Entry;

/**
 *
 * @author dell
 */
public class ZTreeNode extends LinkedList<ZTreeNode>
{
    //columns-------------------------------------------------------------------
    String value;
    String hidden;
    
    //constructors--------------------------------------------------------------
    public ZTreeNode(String value, String hidden) 
    {
        this.value = value;
        this.hidden = hidden;
    }
    public ZTreeNode(Entry<String,String> kv)
    {
        this.value=kv.getKey();
        this.hidden=kv.getValue();
    }
    //functions-----------------------------------------------------------------
    public String getValue() 
    {
        return value;
    }
    public void setValue(String value) 
    {
        this.value = value;
    }
    public String getHidden() 
    {
        return hidden;
    }
    public void setHidden(String hidden) 
    {
        this.hidden = hidden;
    }
    public ZTreeNode add(Entry<String,String> kv)
    {
        this.add(new ZTreeNode(kv));
        return this;
    }
     public ZTreeNode add(String value, String hidden)
    {
        this.add(new ZTreeNode(value,hidden));
        return this;
    }
}
