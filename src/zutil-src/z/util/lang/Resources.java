/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.lang;

import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.util.Enumeration;
/**
 *
 * @author dell
 */
public class Resources
{
    private static ClassLoaderWrapper clw;
    static
    {
        synchronized(Resources.class)
        {
            if(clw==null) clw=new ClassLoaderWrapper();
        }
    }
    //<editor-fold defaultstate="collapsed" desc="ClassLoaderWrapper">
    public static class ClassLoaderWrapper 
    {
        //columns---------------------------------------------------------------
        ClassLoader defaultCL;
        ClassLoader systemCL;
        ClassLoader[] classLoaders;
    
        //constructors----------------------------------------------------------
        public ClassLoaderWrapper()
        {
            try
            {
                this.systemCL=ClassLoader.getSystemClassLoader();
                this.classLoaders=new ClassLoader[]{
                    this.systemCL,
                    this.defaultCL,
                    null,
                    this.getClass().getClassLoader()};
            }
            catch(SecurityException e)
            {
                System.err.println(e);
                throw new RuntimeException(e);
            }
        }
        //<editor-fold defaultstate="collapsed" desc="CORE_CODE">
        ClassLoader[] getClassLoaders()
        {
            this.classLoaders[2]=Thread.currentThread().getContextClassLoader();
            return this.classLoaders;
        }
        ClassLoader[] getClassLoaders(ClassLoader classLoader)
        {
            return new ClassLoader[]{
                classLoader,
                this.systemCL,
                this.defaultCL,
                Thread.currentThread().getContextClassLoader(),
                this.getClass().getClassLoader()};
        }
        URL getResourceAsURL(String resource, ClassLoader[] cl)
        {
            URL url=null;
            for(int i=0;i<cl.length;i++)
            {
                if(cl[i]==null) continue;
                url=cl[i].getResource(resource);
                if(url==null) url=cl[i].getResource('/'+resource);
                if(url!=null) break;
            }
            return url;
        }
        InputStream getResourceAsStream(String resource, ClassLoader[] cl)
        {
            InputStream rv=null;
            for(int i=0;i<cl.length;i++)
            {
                if(cl[i]==null) continue;
                rv=cl[i].getResourceAsStream(resource);
                if(rv==null) rv=cl[i].getResourceAsStream("/"+resource);
                if(rv!=null) break;
            }   
            return rv;
        }
        Class<?> classForName(String name, ClassLoader[] cl) throws ClassNotFoundException
        {
            Class<?> c=null;
            for(int i=0;i<cl.length;i++)
            {
                if(cl[i]==null) continue;
                try
                {
                    c=Class.forName(name, true, cl[i]);
                    if(c!=null) return c;
                }   
                catch(ClassNotFoundException e){}
            }
            throw new ClassNotFoundException("No such Class: "+name);
        }
        Enumeration<URL> getResources(String resource, ClassLoader[] cl) throws IOException
        {
            Enumeration<URL> urls=null;
            for(int i=0;i<cl.length;i++)
            {
                if(cl[i]==null) continue;
                try
                {
                    urls=cl[i].getResources(resource);
                    if(urls==null) urls=cl[i].getResources("/"+resource);
                    if(urls!=null) return urls;
                }
                catch(IOException e) {}
            }
            throw new IOException("No such Resource: "+resource);
        }
        //</editor-fold>
        public URL getResourceAsURL(String resource)
        {
            return this.getResourceAsURL(resource, this.getClassLoaders());
        }
        public URL getResourceAsURL(String resource, ClassLoader classLoader)
        {
            return this.getResourceAsURL(resource, this.getClassLoaders(classLoader));
        }
        public InputStream getResourceAsStream(String resource)
        {
             return this.getResourceAsStream(resource, this.getClassLoaders());
        }
        public InputStream getResourceAsStream(String resource, ClassLoader classLoader)
        {
            return this.getResourceAsStream(resource, this.getClassLoaders(classLoader));
        }
        public Class<?> classForName(String name) throws ClassNotFoundException
        {
           return this.classForName(name, this.getClassLoaders());
        }
        public Class<?> classForName(String name, ClassLoader classLoader) throws ClassNotFoundException
        {
            return this.classForName(name, this.getClassLoaders(classLoader));
        }
        public Enumeration<URL> getResources(String resource) throws IOException
        {
            return this.getResources(resource, this.getClassLoaders());
        }
        public Enumeration<URL> getResources(String path, ClassLoader classLoader) throws IOException
        {
            return this.getResources(path, this.getClassLoaders(classLoader));
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Outer-Operator">
    private static void requireResNonNull(Object res, String resource) throws IOException
    {
        if(res==null) throw new IOException("Can't find resource:"+resource);
    }
    public static void setDefaultClassLoader(ClassLoader defClassLoader)
    {
        clw.defaultCL=defClassLoader;
    }
    public static URL getResourceURL(String resource) throws IOException
    {
        URL url=clw.getResourceAsURL(resource);
        Resources.requireResNonNull(url, resource);
        return url;
    }
    public static URL getResourceURL(String resource, ClassLoader loader) throws IOException
    {
        URL url=clw.getResourceAsURL(resource, loader);
        Resources.requireResNonNull(url, resource);
        return url;
    }
    public static InputStream getResourceAsStream(String resource) throws IOException
    {
        InputStream in=clw.getResourceAsStream(resource);
        Resources.requireResNonNull(in, resource);
        return in;
    }
    public static InputStream getReourceAsStream(String resource, ClassLoader loader) throws IOException
    {
        InputStream in=clw.getResourceAsStream(resource, loader);
        Resources.requireResNonNull(in, resource);
        return in;
    }
    public static Enumeration<URL> getResources(String resource) throws IOException
    {
        Enumeration<URL> urls=clw.getResources(resource);
        Resources.requireResNonNull(urls, resource);
        return urls;
    }
    public static Enumeration<URL> getResources(String resource, ClassLoader loader) throws IOException
    {
        Enumeration<URL> urls=clw.getResources(resource, loader);
        Resources.requireResNonNull(urls, resource);
        return urls;
    }
    //</editor-fold>
}
