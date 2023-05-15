package z.util.ds.exception;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */


/**
 *
 * @author dell
 */
public class WrongLenderException extends Exception
{
    //static--------------------------------------------------------------------
    private static final String msg="[The Resource is returned to the Wrong Lender]";
    
    //functions-----------------------------------------------------------------
    public WrongLenderException()
    {
        super(msg);
    }
    public WrongLenderException(String message)
    {
        super(msg+message);
    }
}
