/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.lang;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.PrintStream;
import java.io.Serializable;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.lang.reflect.ParameterizedType;
import java.lang.reflect.Type;
import java.net.JarURLConnection;
import java.net.URL;
import java.net.URLDecoder;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Collection;
import java.util.Date;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.function.Consumer;
import java.util.function.Predicate;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;
import z.util.factory.Meta;
import z.util.function.Converter;
import z.util.function.Printer;
import z.util.function.Stringer;
import z.util.math.ExRandom;
import z.util.math.vector.Matrix;
import z.util.math.vector.Vector;
import z.util.lang.annotation.Passed;

/**
 * <pre>
 * z.util.lang.Lang is the base of all class in Package z.util, before using
 * relative class, you must load z.util.lang.Lang.
 * (1)The convert-function and class-mapping are necessary to init
 * (2)while the print-function and to-String function is optional to load, according
 * to the configuration of 'z.util.lang.lang-site.xml'
 * </pre>
 * @author dell
 */
public final class Lang 
{
    public static final String NULL="null";
    public static final String NULL_LN="null\n";
    
    public static final int INT_BYTE=Integer.SIZE/Byte.SIZE;
    public static final int LENGTH_TIMES_INT_CHAR=Integer.SIZE/Character.SIZE;
    public static final int LENGTH_DEF_INT_CHAR=Integer.SIZE-Character.SIZE;
    
    public static final int KB_BIT=1<<13;
    public static final int MB_BIT=1<<23;
    public static final long GB_BIT=1L<<33;
    
    private Lang() {}
    //<editor-fold defaultstate="collapsed" desc="Basic-Function">
    private static final ExRandom RANDOM=new ExRandom();
    
    public static final ExRandom exRandom()
    {
        return RANDOM;
    }
    public static final boolean test(Object v)
    {
        if(v instanceof Boolean) return (Boolean) v;
        else if(v instanceof Number) return ((Number)v).intValue()!=0; 
        else if(v instanceof Collection) return ((Collection)v).isEmpty();
        else if(v instanceof Map) return((Map)v).isEmpty();
        else return v!=null;
    }
    public static final void line()
    {
        DEF_OUT.println("------------------------------------------");
    }
    public static final void line(char c)
    {
        char[] cs=new char[10];
        for(int i=0;i<cs.length;i++) cs[i]=c;
        DEF_OUT.println(new String(cs));
    }
    
    private static final SimpleDateFormat DEF_SDF=new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
    public static String currentDateTime()
    {
        return DEF_SDF.format(new Date());
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="HashMap with Function-Pointer">
    //<editor-fold defaultstate="collapsed" desc="class Converter">
    //elementary converters-----------------------------------------------------
    private static final Converter doubleConverter = new Converter() {
        @Override
        public <T> T convert(String val) 
        {
            return val==null? null:(T) Double.valueOf(val);
        }
    };
    private static final Converter intConverter = new Converter() {
        @Override
        public <T> T convert(String val) 
        {
            return val==null? null:(T) Integer.valueOf(val);
        }
    };
    private static final Converter booleanConverter = new Converter() {
        @Override
        public <T> T convert(String val) 
        {
            return val==null? null:(T) Boolean.valueOf(val);
        }
    };
    private static final Converter floatConverter = new Converter() {
        @Override
        public <T> T convert(String val) 
        {
            return val==null? null:(T) Float.valueOf(val);
        }
    };
    private static final Converter longConverter = new Converter() {
        @Override
        public <T> T convert(String val) 
        {
            return val==null? null:(T) Long.valueOf(val);
        }
    };
    private static final Converter byteConverter = new Converter() {
        @Override
        public <T> T convert(String val) 
        {
            return val==null? null:(T) Byte.valueOf(val);
        }
    };
    private static final Converter shortConverter = new Converter() {
        @Override
        public <T> T convert(String val) 
        {
            return val==null? null: (T) Short.valueOf(val);
        }
    };
    private static final Converter stringConverter = new Converter() {
        @Override
        public <T> T convert(String val) 
        {
            return (T) val;
        }
    };
    private static final Converter clazzConverter = new Converter() {
        @Override
        public <T> T convert(String val) throws Exception
        {
            return (T) Class.forName(val);
        }
    };
    
    //vector converters---------------------------------------------------------
    private static final Converter doubleVectorConverter = new Converter() {
        @Override
        public <T> T convert(String val) throws Exception 
        {
            return val==null? null:(T) Vector.valueOfDoubleVector(val);
        }
    };
    private static final Converter intVectorConverter = new Converter() {
        @Override
        public <T> T convert(String val) throws Exception
        {
            return val==null? null:(T) Vector.valueOfIntVector(val);
        }
    };
    private static final Converter booleanVectorConverter = new Converter() {
        @Override
        public <T> T convert(String val) throws Exception 
        {
            return val==null? null:(T) Vector.valueOfBooleanVector(val);
        }
    };
    private static final Converter floatVectorConverter = new Converter() {
        @Override
        public <T> T convert(String val) throws Exception 
        {
            return val==null? null:(T) Vector.toFloatVector(val);
        }
    };
    private static final Converter longVectorConverter = new Converter() {
        @Override
        public <T> T convert(String val) throws Exception 
        {
            return val==null? null:(T) Vector.valueOfLongVector(val);
        }
    };
    private static final Converter byteVectorConverter = new Converter() {
        @Override
        public <T> T convert(String val) throws Exception 
        {
            return val==null? null:(T) Vector.valueOfByteVector(val);
        }
    };
    private static final Converter shortVectorConverter = new Converter() {
        @Override
        public <T> T convert(String val) throws Exception 
        {
            return val==null? null:(T) Vector.valueOfShortVector(val);
        }
    };
    private static final Converter stringVectorConverter=new Converter() {
        @Override
        public <T> T convert(String val) throws Exception 
        {
            return (val==null? null:(T) val.split(" {0,}, {0,}"));
        }
    };
        
    private static final Converter nDoubleVectorConverter = new Converter() {
        @Override
        public <T> T convert(String val) throws Exception 
        {
            return val==null? null:(T) Vector.valueOfNDoubleVector(val);
        }
    };
    private static final Converter nIntVectorConverter = new Converter() {
        @Override
        public <T> T convert(String val) throws Exception
        {
            return val==null? null:(T) Vector.valueOfNIntVector(val);
        }
    };
    private static final Converter nBooleanVectorConverter = new Converter() {
        @Override
        public <T> T convert(String val) throws Exception 
        {
            return val==null? null:(T) Vector.valueOfNBooleanVector(val);
        }
    };
    private static final Converter nFloatVectorConverter = new Converter() {
        @Override
        public <T> T convert(String val) throws Exception 
        {
            return val==null? null:(T) Vector.valueOfNFloatVector(val);
        }
    };
    private static final Converter nLongVectorConverter = new Converter() {
        @Override
        public <T> T convert(String val) throws Exception 
        {
            return val==null? null:(T) Vector.valueOfNLongVector(val);
        }
    };
    private static final Converter nByteVectorConverter = new Converter() {
        @Override
        public <T> T convert(String val) throws Exception 
        {
            return val==null? null:(T) Vector.valueOfNByteArray(val);
        }
    };
    private static final Converter nShortVectorConverter = new Converter() {
        @Override
        public <T> T convert(String val) throws Exception 
        {
            return val==null? null:(T) Vector.valueOfNShortArray(val);
        }
    };
        
    //matrix converters---------------------------------------------------------
    private static final Converter doubleMatrixConverter = new Converter() {
        @Override
        public <T> T convert(String val) throws Exception 
        {
            return val==null? null:(T) Matrix.valueOfDoubleMatrix(val);
        }
    };
    private static final Converter intMatrixConverter = new Converter() {
        @Override
        public <T> T convert(String val) throws Exception
        {
            return val==null? null:(T) Matrix.valueOfIntMatrix(val);
        }
    };
    private static final Converter booleanMatrixConverter = new Converter() {
        @Override
        public <T> T convert(String val) throws Exception 
        {
            return val==null? null:(T) Matrix.valueOfBooleanMatrix(val);
        }
    };
    private static final Converter floatMatrixConverter = new Converter() {
        @Override
        public <T> T convert(String val) throws Exception 
        {
            return val==null? null:(T) Matrix.valueOfFloatMatrix(val);
        }
    };
    private static final Converter longMatrixConverter = new Converter() {
        @Override
        public <T> T convert(String val) throws Exception 
        {
            return val==null? null:(T) Matrix.valueOfLongMatrix(val);
        }
    };
    private static final Converter byteMatrixConverter = new Converter() {
        @Override
        public <T> T convert(String val) throws Exception 
        {
            return val==null? null:(T) Matrix.valueOfByteMatrix(val);
        }
    };
    private static final Converter shortMatrixConverter = new Converter() {
        @Override
        public <T> T convert(String val) throws Exception 
        {
            return val==null? null:(T) Matrix.valueOfShortMatrix(val);
        }
    };
    private static final Converter stringMatrixConverter=new Converter() {
        @Override
        public <T> T convert(String val) throws Exception 
        {
            return (val==null? null:(T) Matrix.valueOfStringMatrix(val));
        }
    };
        
    private static final Converter nDoubleMatrixConverter = new Converter() {
        @Override
        public <T> T convert(String val) throws Exception 
        {
            return val==null? null:(T) Matrix.valueOfNDoubleMatrix(val);
        }
    };
    private static final Converter nIntMatrixConverter = new Converter() {
        @Override
        public <T> T convert(String val) throws Exception
        {
            return val==null? null:(T) Matrix.valueOfNIntMatrix(val);
        }
    };
    private static final Converter nBooleanMatrixConverter = new Converter() {
        @Override
        public <T> T convert(String val) throws Exception 
        {
            return val==null? null:(T) Matrix.valueOfNBooleanMatrix(val);
        }
    };
    private static final Converter nFloatMatrixConverter = new Converter() {
        @Override
        public <T> T convert(String val) throws Exception 
        {
            return val==null? null:(T) Matrix.valueOfNFloatMatrix(val);
        }
    };
    private static final Converter nLongMatrixConverter = new Converter() {
        @Override
        public <T> T convert(String val) throws Exception 
        {
            return val==null? null:(T) Matrix.valueOfNLongMatrix(val);
        }
    };
    private static final Converter nByteMatrixConverter = new Converter() {
        @Override
        public <T> T convert(String val) throws Exception 
        {
            return val==null? null:(T) Matrix.valueOfNByteMatrix(val);
        }
    };
    private static final Converter nShortMatrixConverter = new Converter() {
        @Override
        public <T> T convert(String val) throws Exception 
        {
            return val==null? null:(T) Matrix.valueOfNShortMatrix(val);
        }
    };
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="class Stringer">
    //vector stringers----------------------------------------------------------
    private static final Stringer booleanVectorStringer=new Stringer<boolean[]>() {
        @Override
        public String process(boolean[] val) 
        {
            return Vector.toString(val);
        }
    };
    private static final Stringer byteVectorStringer=new Stringer<byte[]>() {
        @Override
        public String process(byte[] val) 
        {
            return Vector.toString(val);
        }
    };
    private static final Stringer shortVectorStringer=new Stringer<short[]>() {
        @Override
        public String process(short[] val) 
        {
            return Vector.toString(val);
        }
    };
    private static final Stringer intVectorStringer=new Stringer<int[]>() {
        @Override
        public String process(int[] val) 
        {
            return Vector.toString(val);
        }
    };
    private static final Stringer longVectorStringer=new Stringer<long[]>() {
        @Override
        public String process(long[] val) 
        {
            return Vector.toString(val);
        }
    };
    private static final Stringer floatVectorStringer=new Stringer<float[]>() {
        @Override
        public String process(float[] val) 
        {
            return Vector.toString(val);
        }
    };
    private static final Stringer doubleVectorStringer=new Stringer<double[]>() {
        @Override
        public String process(double[] val) 
        {
            return Vector.toString(val);
        }
    };
    
    //matrix stringers----------------------------------------------------------
    private static final Stringer booleanMatrixStringer=new Stringer<boolean[][]>() {
        @Override
        public String process(boolean[][] val) 
        {
            return Matrix.toString(val);
        }
    };
    private static final Stringer byteMatrixStringer=new Stringer<byte[][]>() {
        @Override
        public String process(byte[][] val) 
        {
            return Matrix.toString(val);
        }
    };
    private static final Stringer shortMatrixStringer=new Stringer<short[][]>() {
        @Override
        public String process(short[][] val) 
        {
            return Matrix.toString(val);
        }
    };
    private static final Stringer intMatrixStringer=new Stringer<int[][]>() {
        @Override
        public String process(int[][] val) 
        {
            return Matrix.toString(val);
        }
    };
    private static final Stringer longMatrixStringer=new Stringer<long[][]>() {
        @Override
        public String process(long[][] val) 
        {
            return Matrix.toString(val);
        }
    };
    private static final Stringer floatMatrixStringer=new Stringer<float[][]>() {
        @Override
        public String process(float[][] val) 
        {
            return Matrix.toString(val);
        }
    };
    private static final Stringer doubleMatrixStringer=new Stringer<double[][]>() {
        @Override
        public String process(double[][] val) 
        {
            return Matrix.toString(val);
        }
    };
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="class Printer"> 
    //vector printers-----------------------------------------------------------
    private static final Printer booleanVectorPrinter=new Printer<boolean[]>() {
        @Override
        public void println(PrintStream out, boolean[] val) 
        {
            Vector.println(out, val);
        }
    };
    private static final Printer byteVectorPrinter=new Printer<byte[]>() {
        @Override
        public void println(PrintStream out, byte[] val)
        {
            Vector.println(out, val);
        }
    };
    private static final Printer shortVectorPrinter=new Printer<short[]>() {
        @Override
        public void println(PrintStream out, short[] val) 
        {
            Vector.println(out, val);
        }
    };
    private static final Printer intVectorPrinter=new Printer<int[]>() {
        @Override
        public void println(PrintStream out, int[] val) 
        {
            Vector.println(out, val);
        }
    };
    private static final Printer longVectorPrinter=new Printer<long[]>() {
        @Override
        public void println(PrintStream out, long[] val) 
        {
            Vector.println(out, val);
        }
    };
    private static final Printer floatVectorPrinter=new Printer<float[]>() {
        @Override
        public void println(PrintStream out, float[] val) 
        {
            Vector.println(out, val);
        }
    };
    private static final Printer doubleVectorPrinter=new Printer<double[]>() {
        @Override
        public void println(PrintStream out, double[] val) 
        {
            Vector.println(out, val);
        }
    };
    
    //matrix printers-----------------------------------------------------------
    private static final Printer booleanMatrixPrinter=new Printer<boolean[][]>() {
        @Override
        public void println(PrintStream out, boolean[][] val) 
        {
            Matrix.println(out, val);
        }
    };
    private static final Printer byteMatrixPrinter=new Printer<byte[][]>() {
        @Override
        public void println(PrintStream out, byte[][] val) {
            Matrix.println(out, val);
        }
    };
    private static final Printer shortMatrixPrinter=new Printer<short[][]>() {
        @Override
        public void println(PrintStream out, short[][] val) {
            Matrix.println(out, val);
        }
    };
    private static final Printer intMatrixPrinter=new Printer<int[][]>() {
        @Override
        public void println(PrintStream out, int[][] val) {
            Matrix.println(out, val);
        }
    };
    private static final Printer longMatrixPrinter=new Printer<long[][]>() {
        @Override
        public void println(PrintStream out, long[][] val) {
            Matrix.println(out, val);
        }
    };
    private static final Printer floatMatrixPrinter=new Printer<float[][]>() {
        @Override
        public void println(PrintStream out, float[][] val) {
            Matrix.println(out, val);
        }
    };
    private static final Printer doubleMatrixPrinter=new Printer<double[][]>() {
        @Override
        public void println(PrintStream out, double[][] val) {
            Matrix.println(out, val);
        }
    };
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Init-State">
    private static final String TOSTRING_INIT_CONF="lang.toString.init";
    private static final String PRINT_INIT_CONF="lang.print.init";
    
    static
    {
        try
        {
            Lang.initClassMapping();
            Lang.initConvert();
            Meta mt=Meta.valueOf("z/util/lang/conf/zlang-site.xml", null, "configuration");
            if(mt.getValue(TOSTRING_INIT_CONF)) Lang.initToString();
            if(mt.getValue(PRINT_INIT_CONF)) Lang.initPrint();
        }
        catch(Exception e)
        {
            e.printStackTrace();
            throw new RuntimeException("Fail to init z.util.lang.Lang");
        }
    }
    //</editor-fold>
    
    private static Set<Class> ELEMENT_TYPE;
    private static Set<Class> ELEMENT_VECTOR_TYPE;
    private static Set<Class> ELEMENT_MATRIX_TYPE;
    private static Map<Class, String> CLASS_NAME_MAP;
    private static Map<String, Class> NAME_CLASS_MAP;
    
    private static HashMap<Class,Converter> CLASS_CONVERTER_MAP;
    private static HashMap<String,Converter> NAME_CONVERTER_MAP;  
    
    private static boolean TOSTRING_INIT=false;
    private static HashMap<Class, Stringer> CLASS_STRINGER_MAP;
    
    private static boolean PRINT_INIT=false;
    private static HashMap<Class, Printer> CLASS_PRINTER_MAP;
    
    //<editor-fold defaultstate="collapsed" desc="Init or CleanUp">
    @Passed
    private synchronized static void initClassMapping()
    {
        ELEMENT_TYPE=new HashSet<>();
        ELEMENT_VECTOR_TYPE=new HashSet<>();
        ELEMENT_MATRIX_TYPE=new HashSet<>();
        CLASS_NAME_MAP=new HashMap<>();
        NAME_CLASS_MAP=new HashMap<>();
        
        //for element data type-------------------------------------------------
        Class[] clazz=Lang.getElementClass();
        String[] name=Lang.getElementClassName();
        for(int i=0;i<clazz.length;i++) 
        {
            ELEMENT_TYPE.add(clazz[i]);
            CLASS_NAME_MAP.put(clazz[i], name[i]);
            NAME_CLASS_MAP.put(name[i], clazz[i]);
        }
        
        //for element vector-----------------------------------------------------
        clazz=Lang.getElementVectorClass();
        name=Lang.getElementVectorClassName();
        for(int i=0;i<clazz.length;i++) 
        {
            ELEMENT_VECTOR_TYPE.add(clazz[i]);
            CLASS_NAME_MAP.put(clazz[i], name[i]);
            NAME_CLASS_MAP.put(name[i], clazz[i]);
        }
        
        //for element matrix----------------------------------------------------
        clazz=Lang.getElementMatrixClass();
        name=Lang.getElementMatrixClassName();
        for(int i=0;i<clazz.length;i++) 
        {
            ELEMENT_MATRIX_TYPE.add(clazz[i]);
            CLASS_NAME_MAP.put(clazz[i], name[i]);
            NAME_CLASS_MAP.put(name[i], clazz[i]);
        }
    }
    @Passed
    private synchronized static void initConvert()
    {
        CLASS_CONVERTER_MAP=new HashMap<>();
        NAME_CONVERTER_MAP=new HashMap<>();
        
        //for elemnt converter--------------------------------------------------
        Class[] clazz=Lang.getElementClass();
        String[] name=Lang.getElementClassName();
        Converter[] converter=new Converter[]{
            byteConverter, booleanConverter, shortConverter, intConverter, longConverter,
            floatConverter, doubleConverter,
            stringConverter, clazzConverter,
            booleanConverter, byteConverter, shortConverter, intConverter, longConverter,
            floatConverter, doubleConverter};
        for(int i=0;i<clazz.length;i++) 
        {
            CLASS_CONVERTER_MAP.put(clazz[i], converter[i]);
            NAME_CONVERTER_MAP.put(name[i], converter[i]);
        }
        
        //for vector converter--------------------------------------------------
        clazz=Lang.getElementVectorClass();
        name=Lang.getElementVectorClassName();
        converter=new Converter[]{
            byteVectorConverter, booleanVectorConverter, shortVectorConverter, intVectorConverter, longVectorConverter,
            floatVectorConverter, doubleVectorConverter,
            stringVectorConverter,
            nBooleanVectorConverter, nByteVectorConverter, nShortVectorConverter, nIntVectorConverter, nLongVectorConverter,
            nFloatVectorConverter, nDoubleVectorConverter};
        for(int i=0;i<clazz.length;i++) 
        {
            CLASS_CONVERTER_MAP.put(clazz[i], converter[i]);
            NAME_CONVERTER_MAP.put(name[i], converter[i]);
        }
        
        //for matrix converter--------------------------------------------------
        clazz=Lang.getElementMatrixClass();
        name=Lang.getElementMatrixClassName();
        converter=new Converter[]{
            byteMatrixConverter, booleanMatrixConverter, shortMatrixConverter, intMatrixConverter, longMatrixConverter,
            floatMatrixConverter, doubleMatrixConverter,
            stringMatrixConverter,
            nBooleanMatrixConverter, nByteMatrixConverter, nShortMatrixConverter, nIntMatrixConverter, nLongMatrixConverter,
            nFloatMatrixConverter, nDoubleMatrixConverter};
        for(int i=0;i<clazz.length;i++) 
        {
            CLASS_CONVERTER_MAP.put(clazz[i], converter[i]);
            NAME_CONVERTER_MAP.put(name[i], converter[i]);
        }
    }
    @Passed
    public synchronized static void initToString()
    {
        if(TOSTRING_INIT) 
        {
            System.out.println("z.util.Lang-Stringer has been initialized,"
                    + " don't call this function repeatedly");
            return;
        }
        CLASS_STRINGER_MAP=new HashMap<>();
        //for vector Stringer---------------------------------------------------
        Class[] clazz=Lang.getElementVectorClass();
        Stringer[] stringer=new Stringer[]{
            booleanVectorStringer, byteVectorStringer, shortVectorStringer, intVectorStringer, longVectorStringer,
            floatVectorStringer, doubleVectorStringer};
        for(int i=0;i<stringer.length;i++)
            CLASS_STRINGER_MAP.put(clazz[i], stringer[i]);

        //for matrix Stringer---------------------------------------------------
        clazz=Lang.getElementMatrixClass();
        stringer=new Stringer[]{
            booleanMatrixStringer, byteMatrixStringer, shortMatrixStringer, intMatrixStringer, longMatrixStringer,
            floatMatrixStringer, doubleMatrixStringer};
        for(int i=0;i<stringer.length;i++)
            CLASS_STRINGER_MAP.put(clazz[i], stringer[i]);
        
        TOSTRING_INIT=true;
    }
    @Passed
    public synchronized static void cleanUpToString()
    {
        if(!TOSTRING_INIT) 
        {
            System.out.println("z.util.Lang-Stringer has been cleaned up,"
                    + " don't call this function repeatedly");
            return;
        }
        CLASS_STRINGER_MAP.clear();
        CLASS_STRINGER_MAP=null;
        TOSTRING_INIT=false;
    }
    @Passed
    public synchronized static void initPrint()
    {
        if(PRINT_INIT) 
        {
            System.out.println("z.util.Lang-Printer has been initialized,"
                    + " don't call this function repeatedly");
            return;
        }
        CLASS_PRINTER_MAP=new HashMap<>();
        //for vector Stringer---------------------------------------------------
        Class[] clazz=Lang.getElementVectorClass();
        Printer[] printer=new Printer[]{
            booleanVectorPrinter, byteVectorPrinter, shortVectorPrinter, intVectorPrinter, longVectorPrinter,
            floatVectorPrinter, doubleVectorPrinter};
        for(int i=0;i<printer.length;i++)
            CLASS_PRINTER_MAP.put(clazz[i], printer[i]);

        //for matrix Stringer---------------------------------------------------
        clazz=Lang.getElementMatrixClass();
        printer=new Printer[]{
            booleanMatrixPrinter, byteMatrixPrinter, shortMatrixPrinter, intMatrixPrinter, longMatrixPrinter,
            floatMatrixPrinter, doubleMatrixPrinter};
        for(int i=0;i<printer.length;i++)
            CLASS_PRINTER_MAP.put(clazz[i], printer[i]);
        
        PRINT_INIT=true;
    }
    @Passed
    public synchronized static void cleanUpPrint()
    {
        if(!PRINT_INIT) 
        {
            System.out.println("z.util.Lang-Printer has been initialized,"
                    + " don't call this function repeatedly");
            return;
        }
        CLASS_PRINTER_MAP.clear();
        CLASS_PRINTER_MAP=null;
        PRINT_INIT=false;
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Reflect-Function">
    //<editor-fold defaultstate="collapsed" desc="Normal-Function">
    @Passed
    public static Class[] getElementClass()
    {
        return new Class[]{
            byte.class, boolean.class, short.class, int.class, long.class,
            float.class, double.class,
            String.class, Class.class,
            Byte.class, Boolean.class, Short.class, Integer.class, Long.class,
            Float.class, Double.class};
    }
    @Passed
    public static Class[] getElementVectorClass()
    {
        return new Class[]{
            byte[].class, boolean[].class, short[].class, int[].class, long[].class,
            float[].class, double[].class,
            String[].class,  
            Byte[].class, Boolean[].class, Short[].class, Integer[].class, Long[].class,
            Float[].class, Double[].class,};
    }
    @Passed
    public static Class[] getElementMatrixClass()
    {
        return new Class[]{
            byte[][].class, boolean[][].class, short[][].class, int[][].class, long[][].class,
            float[][].class, double[][].class,
            String[][].class,  
            Byte[][].class, Boolean[][].class, Short[][].class, Integer[][].class, Long[][].class,
            Float[][].class, Double[][].class};
    }
    @Passed
    public static String[] getElementClassName()
    {
          return new String[]{
            "byte", "boolean", "short", "int", "long",
            "float", "double",
            "String", "Class",
            "Byte", "Boolean", "Short", "Integer", "Long",
            "Float", "Double"};
    }
    @Passed
    public static String[] getElementVectorClassName()
    {
        return new String[]{
            "byte[]", "boolean[]", "short[]", "int[]","long[]",
            "float[]", "double[]",
            "String[]",
            "Byte[]", "Boolean[]", "Short[]", "Integer[]", "Long[]",
            "Float[]", "Double[]"};
    }
    @Passed
    public static String[] getElementMatrixClassName()
    {
        return new String[]{
            "byte[][]", "boolean[][]", "short[][]", "int[][]","long[][]",
            "float[][]", "double[][]",
            "String[][]",
            "Byte[][]", "Boolean[][]", "Short[][]", "Integer[][]", "Long[][]",
            "Float[][]", "Double[][]"};
    }
    @Passed
    public static String getClassName(Object o)
    {
        Class clazz=o.getClass();
        String name=CLASS_NAME_MAP.get(clazz);
        return (name!=null? name:clazz.getSimpleName());
    }
    @Passed
    public static boolean isElementType(Class clazz)
    {
        return ELEMENT_TYPE.contains(clazz);
    }
    @Passed
    public static boolean isElementType(Field field)
    {
        return ELEMENT_TYPE.contains(field.getType());
    }
    @Passed
    public static boolean isElementVectorType(Class clazz)
    {
        return ELEMENT_VECTOR_TYPE.contains(clazz);
    }
    @Passed
    public static boolean isElementVectorType(Field field)
    {
        return ELEMENT_VECTOR_TYPE.contains(field.getType());
    }
    @Passed
    public static boolean isElementMatrixType(Class clazz)
    {
        return ELEMENT_MATRIX_TYPE.contains(clazz);
    }
    @Passed
    public static boolean isElementMatrixType(Field field)
    {
        return ELEMENT_MATRIX_TYPE.contains(field.getType());
    }
    @Passed
    public static boolean isSubClass(Class subClass, Class superClass)
    {
        try 
        {
            subClass.asSubclass(superClass);
            return true;
        }
        catch(Exception e){return false;}
    }
    /**
     * @param clazz
     * @return if the class is an Array Class, return the class of CompontentType.
     * else return null.
     */
    @Passed
    public static Class isArray(Class clazz)
    {
        if(Lang.isElementVectorType(clazz)) return clazz.getComponentType();
        if(!clazz.isArray()) return null;
        else return clazz.getComponentType();
    }
    /**
     * @param clazz
     * @return if the class is an Matrix Class, return the class of CompontentType.
     * else return null.
     */
    @Passed
    public static Class isMatrix(Class clazz)
    {
        if(Lang.isElementMatrixType(clazz)) return clazz.getComponentType().getComponentType();
        Class d1=Lang.isArray(clazz);
        return (d1==null? null:Lang.isArray(d1));
    }
    @Passed
    public static String toDetailString(Class clazz)
    {
        StringBuilder sb=new StringBuilder();
        sb.append("Class - ").append(clazz.getName()).append("{\n");
        sb.append("Fields:\n");
        Vector.appendLn(sb, getExtensiveFields(clazz));
        
        sb.append("\nFunctions\n");
        Vector.appendLn(sb, clazz.getDeclaredMethods());
        sb.append("}");
        return sb.toString();
    }
    @Passed
    public static String toGenericString(Class clazz)
    {
        StringBuilder sb=new StringBuilder();
        sb.append("Class - ").append(clazz.getName()).append("{\n");
        
        Collection<Field> fields=Lang.getExtensiveFields(clazz);
        sb.append("Fields:\n");
        Class[] interfs=null;
        for(Field field:fields)
        {
            sb.append(field.toGenericString());
            interfs=field.getType().getInterfaces();
            if(interfs!=null) sb.append(Arrays.toString(interfs));
            sb.append('\n');
        }
        
        Method[] methods=clazz.getDeclaredMethods();
        sb.append("\nFunctions\n");
        for(Method m:methods)
            sb.append(m.toGenericString()).append('\n');
        
        sb.append("}");
        return sb.toString();
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Field-Function">
    public static final Predicate fieldIsStatic=new Predicate<Field>() {
        @Override
        public boolean test(Field t)
        {
            return Modifier.isStatic(t.getModifiers());
        }
    };
    public static final Predicate fieldNotStatic=new Predicate<Field>() {
        @Override
        public boolean test(Field t) 
        {
            return !Modifier.isStatic(t.getModifiers());
        }
    };
    public static final Predicate fieldIsFinal=new Predicate<Field>() {
        @Override
        public boolean test(Field t) 
        {
            return Modifier.isFinal(t.getModifiers());
        }
    };
    public static final Predicate fieldNotFinal=new Predicate<Field>() {
        @Override
        public boolean test(Field t) 
        {
            return !Modifier.isFinal(t.getModifiers());
        }
    };
    
    //<editor-fold defaultstate="collapsed" desc="Extensive:Core-Code">
    /**
     * get all declared fields of a specified class, both its or its 
     * super classes'.
     * @param cls 
     * @return 
     */
    public static Collection<Field> getExtensiveFields(Class cls)
    {
        Objects.requireNonNull(cls);
        LinkedList<Field> c = new LinkedList<>();
        
        for(; cls !=null && cls != Object.class; cls = cls.getSuperclass()) {
            for(Field fid : cls.getDeclaredFields()) c.add(fid);
        }
        return c;
    }
     /**
     * get all declared fields of a specified class, both its or its 
     * super classes'. for each field you can use {@code Predicate pre} to check
     * and decide whether to add it to the Collection.
     * @param cls 
     * @param pre 
     * @return 
     */
    public static Collection<Field> getExtensiceFields(Class cls, Predicate<Field> pre)
    {
        Objects.requireNonNull(cls);
        Collection<Field> c=new LinkedList<>();
        
        Field[] fids=null;
        for(int i; cls!=null && cls!=Object.class; cls=cls.getSuperclass())
        {
            fids = cls.getDeclaredFields();
            for(i=0;i<fids.length;i++)
                if(pre.test(fids[i])) c.add(fids[i]);
        }
        return c;
    }
    /**
     * get all declared fields of a specified class, both its or its 
     * super classes'. for each field you can use {@code Predicate pre} to check
     * and decide whether to add it to the Collection. Besides, before you add
     * the field to the Collection, you can use con to process the field, like
     * {@code field.setAccessible(true);} and so on.
     * @param cls 
     * @param pre 
     * @param con 
     * @return 
     */
    public static Collection<Field> getExtensiceFields(Class cls, Predicate<Field> pre, Consumer<Field> con)
    {
        Objects.requireNonNull(cls);
        Collection<Field> c=new LinkedList<>();
        
        Field[] fids=null;
        for(int i;cls!=null&&cls!=Object.class;cls=cls.getSuperclass())
        {
            fids=cls.getDeclaredFields();
            for(i=0;i<fids.length;i++)
                if(pre.test(fids[i])) {con.accept(fids[i]);c.add(fids[i]);}
        }
        return c;
    }
    //</editor-fold>
    public static Collection<Field> getExtensiveMemberFields(Class clazz)
    {
        return Lang.getExtensiceFields(clazz, fieldNotStatic);
    }
    public static Collection<Field> getExtensiveMemberFields(Class clazz, Consumer<Field> con)
    {
        return Lang.getExtensiceFields(clazz, fieldNotStatic, con);
    }
    public static Collection<Field> getExtensiveStaticFields(Class clazz)
    {
        return Lang.getExtensiceFields(clazz, fieldIsStatic);
    }
    public static Collection<Field> getExtensiveStaticFields(Class clazz, Consumer<Field> con)
    {
        return Lang.getExtensiceFields(clazz, fieldIsStatic, con);
    }
    public static Collection<Field> getExtensiveFinalFields(Class clazz)
    {
        return Lang.getExtensiceFields(clazz, fieldIsFinal);
    }
    public static Collection<Field> getExtensiveFinalFields(Class clazz, Consumer<Field> con)
    {
        return Lang.getExtensiceFields(clazz, fieldIsFinal, con);
    }
    public static Collection<Field> getExtensiveNotFinalFields(Class clazz)
    {
        return Lang.getExtensiceFields(clazz, fieldNotFinal);
    }
    public static Collection<Field> getExtensiveNotFinalFields(Class clazz, Consumer<Field> con)
    {
        return Lang.getExtensiceFields(clazz, fieldNotFinal, con);
    }
    //<editor-fold defaultstate="collapsed" desc="Local:Core-Code"> 
    public static Collection<Field> getFields(Class clazz, Predicate<Field> pre)
    {
        Field[] fids = clazz.getFields();
        Collection<Field> c = new LinkedList<>();
        for(int i = 0; i < fids.length; i++) 
            if(pre.test(fids[i])) c.add(fids[i]);
        return c;
    }
    public static Collection<Field> getFields(Class clazz, Predicate<Field> pre, Consumer<Field> con)
    {
        Field[] fids = clazz.getFields();
        Collection<Field> c = new LinkedList<>();
        for(int i=0; i< fids.length;i++) 
            if(pre.test(fids[i])) { con.accept(fids[i]);c.add(fids[i]); }
        return c;
    }
    //</editor-fold>
    public static Collection<Field> getStaticFields(Class clazz)
    {
        return Lang.getFields(clazz, fieldIsStatic);
    }
    public static Collection<Field> getStaticFields(Class clazz, Predicate<Field> pre)
    {
        return Lang.getFields(clazz, fieldIsStatic.and(pre));
    }
    public static Collection<Field> getStaticFields(Class clazz, Predicate<Field> pre, Consumer<Field> con)
    {
        return Lang.getFields(clazz, fieldIsStatic.and(pre), con);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Method-Function">
    public static final Predicate methodIsStatic=new Predicate<Method>() {
        @Override
        public boolean test(Method t)
        {
            return Modifier.isStatic(t.getModifiers());
        }
    };
    public static final Predicate methodNotStatic=new Predicate<Method>() {
        @Override
        public boolean test(Method t) 
        {
            return !Modifier.isStatic(t.getModifiers());
        }
    };
    public static final Predicate methodIsFinal=new Predicate<Method>() {
        @Override
        public boolean test(Method t) 
        {
            return Modifier.isFinal(t.getModifiers());
        }
    };
    public static final Predicate methodNotFinal=new Predicate<Method>() {
        @Override
        public boolean test(Method t) 
        {
            return !Modifier.isFinal(t.getModifiers());
        }
    };
    
    //<editor-fold defaultstate="collapsed" desc="Core-Code">
    /**
     * get all declared methods of a specified class, both its or its 
     * super classes'.
     * @param cls 
     * @return 
     */
    public static Collection<Method> getExtensiveMethods(Class cls)
    {
        Objects.requireNonNull(cls);
        Collection<Method> c=new LinkedList<>();
        
        Method[] mths=null;
        for(int i;cls!=null&&cls!=Object.class;cls=cls.getSuperclass())
        {
            mths=cls.getDeclaredMethods();
            for(i=0;i<mths.length;i++) c.add(mths[i]);
        }
        return c;
    }
     /**
     * get all declared Methods of a specified class, both its or its 
     * super classes'. for each Method you can use {@code Predicate pre} to check
     * and decide whether to add it to the Collection.
     * @param cls 
     * @param pre 
     * @return 
     */
    public static Collection<Method> getExtensiveMethods(Class cls, Predicate<Method> pre)
    {
        Objects.requireNonNull(cls);
        Collection<Method> c=new LinkedList<>();
        
        Method[] mths=null;
        for(int i;cls!=null&&cls!=Object.class;cls=cls.getSuperclass())
        {
            mths=cls.getDeclaredMethods();
            for(i=0;i<mths.length;i++) 
                if(pre.test(mths[i])) c.add(mths[i]);
        }
        return c;
    }
    /**
     * get all declared Methods of a specified class, both its or its 
     * super classes'. for each Method you can use {@code Predicate pre} to check
     * and decide whether to add it to the Collection. Besides, before you add
     * the field to the Collection, you can use con to process the field, like
     * {@code field.setAccessible(true);} and so on.
     * @param cls 
     * @param pre 
     * @param con 
     * @return 
     */
    public static Collection<Method> getExtensiveMethods(Class cls, Predicate<Method> pre, Consumer<Method> con)
    {
        Objects.requireNonNull(cls);
        Collection<Method> c=new LinkedList<>();
        
        Method[] mths=null;
        for(int i;cls!=null&&cls!=Object.class;cls=cls.getSuperclass())
        {
            mths=cls.getDeclaredMethods();
            for(i=0;i<mths.length;i++) 
                if(pre.test(mths[i])) {con.accept(mths[i]);c.add(mths[i]);}
        }
        return c;
    }
    //</editor-fold>
    public static Collection<Method> getExensiveMemberMethods(Class clazz)
    {
        return Lang.getExtensiveMethods(clazz, methodNotStatic);
    }
    public static Collection<Method> getExensiveMemberMethods(Class clazz, Consumer<Method> con)
    {
        return Lang.getExtensiveMethods(clazz, methodNotStatic, con);
    }
    public static Collection<Method> getExtensiveStaticMethods(Class clazz)
    {
        return Lang.getExtensiveMethods(clazz, methodIsStatic);
    }
    public static Collection<Method> getExtensiveStaticMethods(Class clazz, Consumer<Method> con)
    {
        return Lang.getExtensiveMethods(clazz, methodIsStatic, con);
    }
    public static Collection<Method> getExtensiveFinalMethods(Class clazz)
    {
        return Lang.getExtensiveMethods(clazz, methodIsFinal);
    }
    public static Collection<Method> getExtensiveFinalMethods(Class clazz, Consumer<Method> con)
    {
        return Lang.getExtensiveMethods(clazz, methodIsFinal, con);
    }
    public static Collection<Method> getExtensiveNotFinalMethods(Class clazz)
    {
        return Lang.getExtensiveMethods(clazz, methodNotFinal);
    }
    public static Collection<Method> getExtensiveNotFinalMethods(Class clazz, Consumer<Method> con)
    {
        return Lang.getExtensiveMethods(clazz, methodNotFinal, con);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Package-Function">
    private static final ClassLoader DEF_CLL=ClassLoader.getSystemClassLoader();
    
    //<editor-fold defaultstate="collapsed" desc="Core-Code:getClass">
    @Passed
    private static void addClassesFromFile(Set<Class> set, String pack, String path, ClassLoader cll) throws Exception
    {
        File dir=new File(path);  
        if(!dir.exists()||!dir.isDirectory()) return;  
        File[] files=dir.listFiles();
        String fileName=null;
        for(int i=0;i<files.length;i++)
        {  
            fileName=files[i].getName();
            if(files[i].isDirectory())
                addClassesFromFile(set, pack+'.'+fileName, files[i].getAbsolutePath(), cll);  
            else if(fileName.endsWith(".class"))
                set.add(cll.loadClass(pack+'.'+fileName.substring(0, fileName.length()-6)));//add class    
        }  
    }  
    @Passed
    private static void addClassesFromJar(Set<Class> set, URL url, String path, ClassLoader cll) throws Exception
    {
        JarFile jar=((JarURLConnection) url.openConnection()).getJarFile();  
        Enumeration<JarEntry> entries=jar.entries();  
        for(JarEntry entry;entries.hasMoreElements();)
        {  
            entry=entries.nextElement();  
            if(entry.isDirectory()) continue;
            String entryName=entry.getName(); 
            if(entryName.endsWith(".class")&&entryName.startsWith(path))
            {
                int len=entryName.length();
                if(entryName.charAt(0)=='/') entryName=entryName.substring(1);
                if(entryName.charAt(len-1)=='/') entryName=entryName.substring(0, len-1);
                set.add(cll.loadClass(entryName.substring(0, len-6).replace('/', '.')));//class full path
            }
        }  
    }
    @Passed
    public static void getClasses(Set<Class> set, String pack, ClassLoader cll) throws Exception
    {  
        String path=pack.replace('.', '/');
        Enumeration<URL> dirs=cll.getResources(path);  
        
        for(URL url;dirs.hasMoreElements();) 
        {  
            url=dirs.nextElement();  
            switch(url.getProtocol())
            {
                case "file":Lang.addClassesFromFile(set, pack, URLDecoder.decode(url.getFile(), "UTF-8"), cll);break;
                case "jar":Lang.addClassesFromJar(set, url, path, cll);break;
            }
        }  
    }  
    public static Set<Class> getClasses(String pack, ClassLoader cll) throws Exception
    {  
        Set<Class> set=new HashSet<>();  
        Lang.getClasses(set, pack, cll);
        return set;  
    }  
    public static Set<Class> getClasses(String[] packs, ClassLoader cll) throws Exception
    {
        Set<Class> set=new HashSet<>();
        for(int i=0;i<packs.length;i++) Lang.getClasses(set, packs[i], cll);
        return set;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Core-Code:getClassNeat">
    @Passed
    private static void addClassesFromFileNeat(Set<Class> set, String pack, String path, ClassLoader cll) throws Exception
    {
        File dir=new File(path);  
        if(!dir.exists()||!dir.isDirectory()) return;  
        File[] files=dir.listFiles();
        String fileName=null;
        for(int i=0;i<files.length;i++)
        {  
            fileName=files[i].getName();
            if(files[i].isDirectory())
                addClassesFromFileNeat(set, pack+'.'+fileName, files[i].getAbsolutePath(), cll);  
            else if(fileName.endsWith(".class"))
            {
                fileName=fileName.substring(0, fileName.length()-6);
                int index=fileName.lastIndexOf('$');
                if(index!=-1) fileName=fileName.substring(0, index);
                set.add(cll.loadClass(pack+'.'+fileName));//add class    
            }
        }  
    }  
    @Passed
    private static void addClassesFromJarNeat(Set<Class> set, URL url, String path, ClassLoader cll) throws Exception
    {
        JarFile jar=((JarURLConnection) url.openConnection()).getJarFile();  
        Enumeration<JarEntry> entries=jar.entries();  
        for(JarEntry entry;entries.hasMoreElements();)
        {  
            entry=entries.nextElement();  
            if(entry.isDirectory()) continue;
            String entryName=entry.getName(); 
            if(entryName.endsWith(".class")&&entryName.startsWith(path))
            {
                int len=entryName.length();
                if(entryName.charAt(0)=='/') entryName=entryName.substring(1);
                if(entryName.charAt(len-1)=='/') entryName=entryName.substring(0, len-1);
                
                entryName=entryName.substring(0, len-6).replace('/', '.');
                int index=entryName.lastIndexOf('$');
                if(index!=-1) entryName=entryName.substring(0, index);
                set.add(cll.loadClass(entryName));//class full path
            }
        }  
    }
    @Passed
    public static void getClassesNeat(Set<Class> set, String pack, ClassLoader cll) throws Exception
    {  
        String path=pack.replace('.', '/');
        Enumeration<URL> dirs=cll.getResources(path);  
        
        for(URL url;dirs.hasMoreElements();) 
        {  
            url=dirs.nextElement();  
            switch(url.getProtocol())
            {
                case "file":Lang.addClassesFromFileNeat(set, pack, URLDecoder.decode(url.getFile(), "UTF-8"), cll);break;
                case "jar":Lang.addClassesFromJarNeat(set, url, path, cll);break;
            }
        }  
    }  
    @Passed
    public static Set<Class> getClassesNeat(String pack, ClassLoader cll) throws Exception
    {  
        Set<Class> set=new HashSet<>();  
        Lang.getClassesNeat(set, pack, cll);
        return set;  
    }  
    public static void makeClassSetNeat(Set<Class> set, ClassLoader cll) throws ClassNotFoundException 
    {
        Set<String> names=new HashSet<>();
        int index=0;
        String name=null;
        for(Class cls:set)
        {
            name=cls.getName();
            index=name.lastIndexOf('$');
            if(index!=-1) name=name.substring(0, index);
            names.add(name);
        }
        set.clear();
        for(String nam:names) set.add(cll.loadClass(nam));
    }
    //</editor-fold>
    public static Set<Class> getClasses(String pack) throws Exception
    {
        return Lang.getClasses(pack, DEF_CLL);
    }
    public static Set<Class> getClassesNeat(String pack) throws Exception
    {
        return Lang.getClassesNeat(pack, DEF_CLL);
    }
    public static Set<Class> getClasses(String[] packs) throws Exception
    {
        Set<Class> set=new HashSet<>();
        for(int i=0;i<packs.length;i++) Lang.getClasses(set, packs[i], DEF_CLL);
        return set;
    }
    public static Set<Class> getClassesNeat(String[] packs) throws Exception
    {
        Set<Class> set=new HashSet<>();
        for(int i=0;i<packs.length;i++) Lang.getClassesNeat(set, packs[i], DEF_CLL);
        return set;
    }
    public static void makeClassSetNeat(Set<Class> set) throws ClassNotFoundException 
    {
        Lang.makeClassSetNeat(set, DEF_CLL);
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Clone-Function">
    public static <T extends Serializable> T clone(Object object) throws Exception
    {
        T result=null;
        ByteArrayOutputStream bout=null;
        ObjectOutputStream oos=null;
        ByteArrayInputStream bin=null;
        ObjectInputStream ois=null;
        try
        {
            bout=new ByteArrayOutputStream();
            oos=new ObjectOutputStream(bout);
            oos.writeObject(object);
            bin=new ByteArrayInputStream(bout.toByteArray());
            ois=new ObjectInputStream(bin);
            result=(T) ois.readObject();
        }
        catch(IOException | ClassNotFoundException e)
        {
            throw new RuntimeException(e);
        }
        finally
        {
            try
            {
                if(ois!=null) ois.close();
                if(bin!=null) bin.close();
                if(oos!=null) oos.close();
                if(bout!=null) bout.close();
            }
            catch(IOException e)
            {
                throw new RuntimeException(e);
            }
        }
        return result;
    }
    public static <T extends Serializable> List<T> clone(Object object, int num) throws Exception
    {
        List<T> result=null;
        ByteArrayOutputStream bout=null;
        ObjectOutputStream oos=null;
        ByteArrayInputStream bin=null;
        ObjectInputStream ois=null;
        try
        {
            bout=new ByteArrayOutputStream();
            oos=new ObjectOutputStream(bout);
            bin=new ByteArrayInputStream(bout.toByteArray());
            ois=new ObjectInputStream(bin);
            
            result=new LinkedList<>();
            for(int i=0;i<num;i++)
            {
                oos.writeObject(object);
                result.add((T) ois.readObject());
            }
        }
        catch(IOException | ClassNotFoundException e)
        {
            throw new RuntimeException(e);
        }
        finally
        {
            try
            {
                if(ois!=null) ois.close();
                if(bin!=null) bin.close();
                if(oos!=null) oos.close();
                if(bout!=null) bout.close();
            }
            catch(IOException e)
            {
                throw new RuntimeException(e);
            }
        }
        return result;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Convert-Function">
    @Passed
    public static String converterDetail()
    {
        StringBuilder sb=new StringBuilder();
        
        sb.append("Map: class->Converter = {");
        Vector.appendLn(sb, CLASS_CONVERTER_MAP, "\t\n");
        sb.append("\n}\n");
        
        sb.append("Map: name->Converter = {");
        Vector.appendLn(sb, NAME_CONVERTER_MAP, "\t\n");
        sb.append("\n}\n");
        
        return sb.toString();
    }
    @Passed
    public static Converter getConverter(Class clazz) 
    {
        Converter con=CLASS_CONVERTER_MAP.get(clazz);
        if(con==null) throw new RuntimeException("There is no matched Converter:"+clazz);
        return con;
    }
    @Passed
    public static Converter getConverter(String name)
    {
        Converter con=NAME_CONVERTER_MAP.get(name);
        if(con==null) throw new RuntimeException("There is no matched Converter:"+name);
        return con;
    }
    @Passed
    public static <T> T convert(String str, Class clazz) throws Exception
    {
        Converter con=CLASS_CONVERTER_MAP.get(clazz);
        if(con==null) throw new RuntimeException("There is no matched Converter:"+clazz);
        return con.convert(str);
    }
    @Passed
    public static <T> T convert(String str, String clazzName) throws Exception
    {
        Converter con=NAME_CONVERTER_MAP.get(clazzName);
        if(con==null) throw new RuntimeException("There is no matched Converter:"+clazzName);
        return con.convert(str);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="toString-Function">
    @Passed
    public static String stringerDetail()
    {
        StringBuilder sb=new StringBuilder();
        
        sb.append("Map: class->Stringer = {");
        Vector.appendLn(sb, CLASS_STRINGER_MAP, "\t\n");
        sb.append("\n}\n");
        
        return sb.toString();
    }
    @Passed
    public static Stringer getStringer(Class clazz) 
    {
        Stringer str=CLASS_STRINGER_MAP.get(clazz);
        if(str==null) throw new RuntimeException("There is no matched Stringer:"+clazz);
        return str;
    }
    public static String toString(Object val, Class clazz)
    {
        Stringer str=CLASS_STRINGER_MAP.get(clazz);
        if(str!=null) return str.process(val);//find the corrosponding stringer
        if((clazz=Lang.isArray(clazz))!=null) 
        {
            if((clazz=Lang.isArray(clazz))!=null) return Matrix.toString((Object[][])val);
            return Vector.toString((Object[])val);
        }
        return val.toString();
    }
    public static String toString(Object val)
    {
        return (val==null? "NULL":Lang.toString(val, val.getClass()));
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Print-Function">
    private static PrintStream DEF_OUT=System.out;
    public static synchronized void setDefaultPrintStream(PrintStream out) {DEF_OUT=out;}
    public static PrintStream getDefaultPrintStream(){return DEF_OUT;}
    
    @Passed
    public static String printerDetail()
    {
        StringBuilder sb=new StringBuilder();
        
        sb.append("Map: class->Printer = {");
        Vector.appendLn(sb, CLASS_PRINTER_MAP, "\t\n");
        sb.append("\n}\n");
        
        return sb.toString();
    }
    @Passed
    public static Printer getPrinter(Class clazz) 
    {
        Printer pr=CLASS_PRINTER_MAP.get(clazz);
        if(pr==null) throw new RuntimeException("There is no matched Printer:"+clazz);
        return pr;
    }
    @Passed
    public static void println(Object val, Class clazz)
    {
        Printer pr=CLASS_PRINTER_MAP.get(clazz);
        if(pr!=null) {pr.println(DEF_OUT, val);return;}
        if((clazz=Lang.isArray(clazz))!=null)
        {
            if((clazz=Lang.isArray(clazz))!=null) {Matrix.println(DEF_OUT, (Object[][])val);return;}
            Vector.println(DEF_OUT, (Object[])val);return;
        }
        DEF_OUT.println(val);
    }
    public static void println(Object val)
    {
        if(val==null) {DEF_OUT.println(val);}
        else Lang.println(val, val.getClass());
    }
    @Passed
    public static void zprintln(Object... args)
    {
        if(args==null) {DEF_OUT.println(NULL);return;}
        for(int i=0;i<args.length;i++)
        {
            DEF_OUT.print("args ["+i+"] :");
            Lang.println(args[i]);
            DEF_OUT.println();
        }
    }
    //</editor-fold>
    
    private static final Field StringBuilder_value;
    private static final Field String_value;
    static
    {
        try
        {
            Class clazz = StringBuilder.class.getSuperclass();
            StringBuilder_value = clazz.getDeclaredField("value");
            StringBuilder_value.setAccessible(true);
            
            String_value = String.class.getDeclaredField("value");
            String_value.setAccessible(true);
        }
        catch(Exception e)
        {
            e.printStackTrace();
            throw new RuntimeException(e);
        }
    }
    
    public static final char[] getChars(StringBuilder sb) throws Exception
    {
        return (char[]) StringBuilder_value.get(sb);
    }
    
    public static final char[] getChars(String sb) throws Exception
    {
        return (char[]) String_value.get(sb);
    }
}
