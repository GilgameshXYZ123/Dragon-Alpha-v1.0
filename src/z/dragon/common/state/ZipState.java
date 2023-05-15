/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.common.state;

import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.function.BiConsumer;
import java.util.zip.Deflater;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.util.zip.ZipOutputStream;
import z.dragon.common.state.State.StateReader;
import z.dragon.common.state.State.StateValue;
import z.dragon.common.state.State.StateWriter;

/**
 *
 * @author Gilgamesh
 */
public class ZipState 
{
    private static final Charset UTF_8 = Charset.forName("UTF-8");
    
    //<editor-fold defaultstate="collapsed" desc="class: ZipValue">
    public static final class ZipValue implements StateValue
    {
        private final ZipEntry entry;
        private final ZipFile file;
        
        protected ZipValue(ZipFile file, ZipEntry entry) {
            this.file = file;
            this.entry = entry;
        }
        
        //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
        public String name() { return entry.getName(); }
        public String comment() { return entry.getComment(); }
        
        public long compressedSize() { return entry.getCompressedSize(); }
        public long size() { return entry.getSize(); }
        
        public void append(StringBuilder sb) {
            sb.append(getClass().getName() + " {");
            sb.append(" name = ").append(this.name());
            sb.append(", compressedSize = ").append(this.compressedSize());
            sb.append(", size = ").append(this.size());
            sb.append(", comment = ").append(this.comment());
            sb.append(" }");
        }
            
        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder(128);
            this.append(sb);
            return sb.toString();
        }
        //</editor-fold>
        
        @Override public ZipEntry value() {return entry;}
        @Override public Class<?> type() {return entry.getClass();}

        @Override
        public ArrayList<String> toStringLines() 
        {
            ArrayList<String> arr = null;
            InputStream in = null;
            InputStreamReader reader = null;
            BufferedReader bufr = null;
            try 
            {
                in = file.getInputStream(entry);
                reader = new InputStreamReader(in);
                bufr = new BufferedReader(reader);
                arr = new ArrayList<>(4);
                
                //被截断的float不好评判是一个还是两个float, 因此逐行读取，而非缓冲区读取
                //可以逐行split -> float, 来为float[]赋值
                String line;
                while((line = bufr.readLine()) != null) {
                    arr.add(line);
                }
            }
            catch(IOException e) { throw new RuntimeException(e); }
            finally 
            {
                try{
                    if(bufr != null) bufr.close();
                    if(reader != null) reader.close();
                    if(in != null) in.close();
                } 
                catch(IOException e) { throw new RuntimeException(e); }
            }
            return arr;
        }
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="class: ZipStateReader">
    public static class ZipStateReader implements StateReader
    {
        protected final ZipFile file;
        
        public ZipStateReader(String path, Charset charset) throws IOException {
            this.file = new ZipFile(path, charset);
        }
        public ZipStateReader(File file, Charset charset) throws IOException{
            this.file = new ZipFile(file, charset);
        }
        
        @Override
        public State read() {
            State dic = new State();
            Enumeration<? extends ZipEntry> entries = file.entries(); 
            while(entries.hasMoreElements()) 
            {
                ZipEntry entry = entries.nextElement();
                if(entry.isDirectory()) continue;
                ZipValue zv = new ZipValue(file, entry);
                dic.put(entry.getName(), zv);
            }
            return dic;
        }
    }
    //</editor-fold>
    public static State state_read(String path) throws IOException {
        return state_read(path, UTF_8); 
    }
    public static State state_read(String path, Charset charset) throws IOException {
        return new ZipStateReader(path, charset).read(); 
    }
    
    //<editor-fold defaultstate="collapsed" desc="class: ZipStateWriter">
    public static final int BEST_SPEED = Deflater.BEST_SPEED;
    public static final int BEST_COMPRESSION = Deflater.BEST_COMPRESSION;
    private static int zip_level = BEST_SPEED;
    
    public static int zip_level() { return zip_level; }
    public static void zip_level(int level) { zip_level = level; }
    
    //<editor-fold defaultstate="collapsed" desc="class: SimpleZipConsumer"> 
    private static final class SimpleZipConsumer implements BiConsumer<String, StateValue>
    {
        protected FileOutputStream fos = null;
        protected ZipOutputStream zos = null;
        protected BufferedOutputStream bos = null;

        public SimpleZipConsumer(File file, Charset charset) {
            try 
            {
                fos = new FileOutputStream(file);
                zos = new ZipOutputStream(fos, charset);
                zos.setLevel(zip_level);
                bos = new BufferedOutputStream(zos);
            }
            catch(FileNotFoundException e) {
               this.clear();
               throw new RuntimeException(e);
            }
        }
        
        public void clear(){
            try 
            {
                if (bos != null) bos.close();
                if (zos != null) zos.close(); 
                if (fos != null) fos.close();
            } 
            catch(IOException e) {
                throw new RuntimeException(e);
            }
        }
        
        private static final byte[] line_separator = new byte[]{'\n'};
        
        @Override
        public void accept(String name, StateValue value) {
            try 
            {
                zos.putNextEntry(new ZipEntry(name));
                boolean first = true;
                for(String line : value.toStringLines()) {
                    if(first) first = false;
                    else  bos.write(line_separator);//start a new line
                    bos.write(line.getBytes());
                }
                bos.flush();
            }
            catch (IOException ex) { 
                this.clear();
                throw new RuntimeException(name, ex);
            }
        } 
    }
    //</editor-fold>
    
    public static class ZipStateWriter implements StateWriter
    {
        private final File file;
        private final Charset charset;
        
        public ZipStateWriter(String path, Charset charset) throws IOException {
            this(new File(path), charset);
        }
        public ZipStateWriter(File file, Charset charset) {
            this.file = file;
            this.charset = charset;
        }
        
        @Override
        public void write(State dic) {
            SimpleZipConsumer consumer = new SimpleZipConsumer(file, charset);
            dic.forEach(consumer);
            consumer.clear();
        }
    }
    //</editor-fold>
    public static void state_write(State dic, String path) throws IOException  { 
        state_write(dic, path, UTF_8); 
    } 
    public static void state_write(State dic, String path, Charset charset) throws IOException {
        new ZipStateWriter(path, charset).write(dic);
    }
}
