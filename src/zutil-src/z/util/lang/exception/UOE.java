/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.lang.exception;

/**
 *
 * @author dell
 */
public class UOE extends UnsupportedOperationException
{
    private static final String MSG="UnsupportedOperationException";

    public UOE()
    {
        super(MSG);
    }
    public UOE(String message)
    {
        super(MSG+message);
    }
    public UOE(Throwable cause)
    {
        super(cause);
    }
}
