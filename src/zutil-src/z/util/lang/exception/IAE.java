/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.lang.exception;

/**
 * A short Name of {@link java.lang.IllegalArgumentException}
 * @author dell
 */
public class IAE extends IllegalArgumentException
{
    private static final String MSG="[IllegalArgumentException]";
    
    public IAE()
    {
        super(MSG);
    }
    public IAE(String message)
    {
        super(MSG+message);
    }
    public IAE(Throwable ex)
    {
        super(ex);
    }
}
