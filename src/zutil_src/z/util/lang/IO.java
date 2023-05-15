/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.lang;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.OutputStreamWriter;

/**
 *
 * @author dell
 */
public class IO 
{
    //<editor-fold defaultstate="collapsed" desc="National-Buffered-IO">
    public static BufferedReader bufferedReader(String filename) {return IO.bufferedReader(new File(filename));}
    public static BufferedWriter bufferedWriter(String filename) {return IO.bufferedWriter(new File(filename));}
    
    public static BufferedReader bufferedReader(File f)
    {
        FileReader fr = null;
        BufferedReader bufr = null;
        try
        {
            fr = new FileReader(f);
            bufr = new BufferedReader(fr);
        }
        catch(IOException e)
        {
            try
            {
                if(bufr!=null) bufr.close();
                if(fr!=null) fr.close();
            }
            catch(IOException ex) {throw new RuntimeException(ex);}
            throw new RuntimeException(e);
        }
        return bufr;
    }
    public static BufferedReader bufferedReader(InputStream in)
    {
        InputStreamReader ir=null;
        BufferedReader bufr=null;
        try
        {
            ir=new InputStreamReader(in);
            bufr=new BufferedReader(ir);
        }
        catch(Exception e)
        {
            try
            {
                if(bufr!=null) bufr.close();
                if(ir!=null) ir.close();
            }
            catch(IOException ex) {throw new RuntimeException(ex);}
            throw new RuntimeException(e);
        }
        return bufr;
    }
    public static BufferedWriter bufferedWriter(File f)
    {
        FileWriter fw=null;
        BufferedWriter bufw=null;
        try
        {
            fw=new FileWriter(f);
            bufw=new BufferedWriter(fw);
        }
        catch(IOException e)
        {
            try
            {
                if(bufw!=null) bufw.close();
                if(fw!=null) fw.close();
            }
            catch(IOException ex) {throw new RuntimeException(ex);}
            throw new RuntimeException(e);
        }
        return bufw;
    }
    public static BufferedWriter bufferedWriter(OutputStream out)
    {
        OutputStreamWriter ow=null;
        BufferedWriter bufw=null;
        try
        {
            ow=new OutputStreamWriter(out);
            bufw=new BufferedWriter(ow);
        }
        catch(Exception e)
        {
            try
            {
                if(bufw!=null) bufw.close();
                if(ow!=null) ow.close();
            }
            catch(IOException ex) {throw new RuntimeException(ex);}
            throw new RuntimeException(e);
        }
        return bufw;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Quick-Test-IO">
    public static void quickWrite(String dst, byte[] buffer)
    {
        File f=new File(dst);
        FileOutputStream out=null;
        try
        {
            out = new FileOutputStream(f);
            out.write(buffer);
        }
        catch(IOException e) {throw new RuntimeException(e);}
        finally
        {
            try {if(out!=null) out.close();}
            catch(IOException e) {throw new RuntimeException(e);}
        }
    }
    
    public static void quickWrite(String dst, String buffer) {
        IO.quickWrite(dst, buffer.getBytes());
    }
    
    public static int quickRead(String src, byte[] buffer)
    {
        File f=new File(src);
        FileInputStream in=null;
        int index=0;
        try
        {
            in=new FileInputStream(f);
            index=in.read(buffer);
        }
        catch(IOException e) {throw new RuntimeException(e);}
        finally
        {
            try {if(in!=null) in.close();}
            catch(IOException e) {throw new RuntimeException(e);}
        }
        return index;
    }
    //</editor-fold>
}
