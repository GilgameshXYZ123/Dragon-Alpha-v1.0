/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.ds.exception;

/**
 *
 * @author dell
 */
public class LackOfResException extends Exception
{
    //static--------------------------------------------------------------------
    private static String msg="[Lack of Resource Exception]";
    
    //functions-----------------------------------------------------------------
    public LackOfResException()
    {
        super(msg);
    }
    public LackOfResException(String message)
    {
        super(msg+message);
    }
}
