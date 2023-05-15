/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.common;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;

/**
 *
 * @author Gilgamesh
 */
public class DragonFile 
{
    private DragonFile() {}
    public static final DragonFile instance() {return fl;}
    public static final DragonFile fl = new DragonFile();
    
    public BufferedFile create(File file) { return new BufferedFile(file); }
    public BufferedFile create(String path) {return new BufferedFile(new File(path));}

    public byte[] to_bytes(BufferedFile file) throws IOException { return file.toBytes(); }
    public byte[] to_bytes(String path) throws IOException { return to_bytes(create(path)); }
    public byte[] to_bytes(File file) throws IOException { return to_bytes(create(file)); }
    
    public char[] to_chars(BufferedFile file) throws IOException { return file.toChars(); }
    public char[] to_chars(String path) throws IOException { return to_chars(create(path)); }
    public char[] to_chars(File file) throws IOException { return to_chars(create(file)); }
    
    //<editor-fold defaultstate="collapsed" desc="class: BufferedFile">
    public static class BufferedFile 
    {
        final File file;
       
        public BufferedFile(File file) {
            this.file = file;
        }
        
        FileReader reader;
        BufferedReader bufReader;
        //<editor-fold defaultstate="collapsed" desc="file read">
        private void readClose() throws IOException {
            if(bufReader != null) { bufReader.close(); bufReader = null; }
            if(reader != null) { reader.close(); reader = null;}
        }

        private void readOpen() throws IOException {
            try {
                reader = new FileReader(file);
                bufReader = new BufferedReader(reader);
            }
            catch (FileNotFoundException e) {
                readClose();throw e;
            }
        }
        
        public String nextLine() throws IOException {
            if (bufReader == null) readOpen();
            String line = bufReader.readLine();
            if (line == null) readClose();
            return line;
        }
        
        public byte[] toBytes() throws IOException
        {
            byte[] buf = null;
            FileInputStream in = null;
            BufferedInputStream bfin = null;
            try
            {
                in = new FileInputStream(file);
                bfin = new BufferedInputStream(in);
                
                if(file.length() >= Integer.MAX_VALUE)
                    throw new IllegalArgumentException("file.length exceeds the maximum array length");
                
                int file_length = (int)file.length();
                buf = new byte[file_length];
                
                int index = 0, len;
                int read_size = Math.min(buf.length, 2048);
                while((len = bfin.read(buf, index, read_size)) != -1 && read_size != 0) {
                    index += len;
                    read_size = buf.length - index;
                    if(read_size > 2048) read_size = 2048;
                }
            }
            catch(IOException e) { 
                buf = null; throw e;
            }
            finally {
                if(bfin != null) bfin.close();
                if(in != null) in.close();
            }
            return buf;
        }
        
        public char[] toChars() throws IOException
        {
            char[] buf = null;
            FileReader fr = null;
            BufferedReader bufr = null;
            try
            {
                fr = new FileReader(file);
                bufr = new BufferedReader(fr);
                
                int file_length = (int)file.length();
                if(file_length < 0 || file.length() >= Integer.MAX_VALUE) 
                    throw new IllegalArgumentException("file.length exceeds the maximum array length");
                buf = new char[file_length];
                
                int index = 0, len;
                int read_size = Math.min(buf.length, 2048);
                while((len = bufr.read(buf, index, read_size)) != -1 && read_size != 0) {
                    index += len;
                    read_size = buf.length - index;
                    if(read_size > 2048) read_size = 2048;
                }
            }
            catch(IOException e){ 
                buf = null; throw e;
            }
            finally {
                if(bufr != null) bufr.close();
                if(fr != null) fr.close();
            }
            return buf;
        }
        //</editor-fold>
        
        FileOutputStream writer;
        BufferedOutputStream bufWriter;
        //<editor-fold defaultstate="collapsed" desc="file write">
        private void writeClose() throws IOException {
            if(writer != null) writer.close();
            if(bufWriter != null) bufWriter.close();
        }
        
        private void writeOpen() throws IOException {
            try {
                writer = new FileOutputStream(file);
                bufWriter = new BufferedOutputStream(writer);
            }
            catch(IOException e) {
                writeClose(); throw e;
            }
        }
        
        public BufferedFile write(byte[] buf) throws IOException {
            if(bufWriter == null) writeOpen();
            bufWriter.write(buf);
            return this;
        }
        
        public BufferedFile write(String buf) throws IOException {
            if(bufWriter == null) writeOpen();
            bufWriter.write(buf.getBytes());
            return this;
        }
        
        public BufferedFile flush() throws IOException {
            if(bufWriter != null) bufWriter.flush();
            return this;
        }
        
        public BufferedFile finish() throws IOException {
            writeClose();
            return this;
        }
        //</editor-fold>
    }
    //</editor-fold>
}
