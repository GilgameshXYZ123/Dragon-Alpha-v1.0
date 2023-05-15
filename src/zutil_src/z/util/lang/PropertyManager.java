/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.lang;

import java.lang.reflect.Field;
import java.util.Collection;
import java.util.HashMap;
import java.util.Objects;
import java.util.Set;
import java.util.TreeSet;
import java.util.function.Predicate;
import z.util.lang.exception.IAE;
import z.util.math.vector.Vector;
/**
 *
 * @author dell
 */
public class PropertyManager 
{
    //<editor-fold defaultstate="collapsed" desc="class PropertyCreator">
    public static interface PropertyCreator
    {
        public String createProperty(Field f);
    }
    public static final PropertyCreator FIELD_NAME=new PropertyCreator() {
        @Override
        public String createProperty(Field f) 
        {
            return f.getName();
        }
    };
    public static final PropertyCreator FIELD_VALUE=new PropertyCreator() {
        @Override
        public String createProperty(Field f)
        {
            try
            {
                return f.get(null).toString();
            }
            catch(IllegalAccessException | IllegalArgumentException e)
            {throw new RuntimeException(e);}
        }
    };
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="class PropertyNameSpace">
    static class PropertyNameSpace extends TreeSet<String>
    {
        //columns---------------------------------------------------------------
        String name;
        Predicate<Field> pre;//add a property's name when it match some condition
        
        //functions-------------------------------------------------------------
        PropertyNameSpace(String name, Predicate<Field> pre) 
        {
            Objects.requireNonNull(name, "name");
            Objects.requireNonNull(pre, "Predicate");
            this.name=name;
            this.pre=pre;
        }
        public String getName() 
        {
            return name;
        }
        @Override
        public String toString()
        {
            StringBuilder sb=new StringBuilder();
            sb.append("[PropertyNameSpace] ").append(name).append(" = {\n");
            Vector.appendLn(sb, this, "\t");
            sb.append("}");
            return sb.toString();
        }
    }
    //</editor-fold>
    protected HashMap<String, PropertyNameSpace> pnss=new HashMap<>();
    protected PropertyCreator pc;    
    
    public PropertyManager(PropertyCreator pc)
    {
        Objects.requireNonNull(pc, "PropertyCreator");
        this.pc=pc;
    }
    public PropertyManager()
    {
        this.pc=FIELD_VALUE;
    }
    //<editor-fold defaultstate="collapsed" desc="Basic-Function">
    @Override
    public String toString()
    {
        StringBuilder sb=new StringBuilder();
        sb.append("PropertyManager = {");
        pnss.forEach((String key, PropertyNameSpace value)->{sb.append("\n").append(value);});
        sb.append("\n}");
        return sb.toString();
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Operators">
    private PropertyNameSpace[] getAllPropertyNameSpace()
    {
        PropertyNameSpace[] arr=new PropertyNameSpace[pnss.size()];
        int index=0;
        for(PropertyNameSpace pns:pnss.values()) arr[index++]=pns;
        return arr;
    }
        //load-----------------------------------------------------------------
    private void loadFromClass(PropertyNameSpace pns, Class clazz)
    {
        Collection<Field> fids=Lang.getStaticFields(clazz, pns.pre);
        for(Field f:fids) pns.add(pc.createProperty(f));
    }
    public void loadFromClass(String pnsName, Class clazz)
    {
        PropertyNameSpace pns=pnss.get(pnsName);
        if(pns==null) throw new NullPointerException("No such PropertyNameSpace: "+pnsName);
        this.loadFromClass(pns, clazz);
    }
    public void loadFromPackage(String pnsName, String packName) throws Exception
    {
        PropertyNameSpace pns=pnss.get(pnsName);
        if(pns==null) throw new NullPointerException("No such PropertyNameSpace: "+pnsName);
        Set<Class> set=Lang.getClasses(packName);
        for(Class cls:set) this.loadFromClass(pns, cls);
    }
    public void loadFromPackage(String pnsName, String[] packs) throws Exception
    {
        PropertyNameSpace pns=pnss.get(pnsName);
        if(pns==null) throw new NullPointerException("No such PropertyNameSpace: "+pnsName);
        Set<Class> set=Lang.getClasses(packs);
        for(Class cls:set) this.loadFromClass(pns, cls);
    }
        //get-set-remove--------------------------------------------------------
    public void setPropertyNameSpace(String pnsName, Predicate<Field> pre)
    {
        if(pnss.get(pnsName)!=null)
            throw new IAE("There exists PropertyNameSpace with the same Name: "+pnsName);
        pnss.put(pnsName, new PropertyNameSpace(pnsName, pre));
    }
    public PropertyNameSpace getPropertyNameSpace(String pnsName)
    {
        PropertyNameSpace pns=pnss.get(pnsName);
        if(pns==null) throw new NullPointerException("No such PropertyNameSpace: "+pnsName);
        return pns;
    }
    public void removePropertyNameSpace(String pnsName)
    {
        PropertyNameSpace pns=pnss.get(pnsName);
        if(pns==null) System.err.println("No such PropertyNameSpace: "+pnsName);
        pnss.remove(pnsName);
    }
    public Set<String> getProperties(String pnsName)
    {
        PropertyNameSpace pns=pnss.get(pnsName);
        if(pns==null) throw new NullPointerException("No such PropertyNameSpace: "+pnsName);
        return pns;
    }
        //complex---------------------------------------------------------------
    public void load(String[] paths) throws Exception
    {
        Set<Class> set=Lang.getClasses(paths);
        
        Collection<Field> fids=null;
        PropertyNameSpace[] arr=this.getAllPropertyNameSpace();
        for(Class cls:set) 
        {
            fids=Lang.getStaticFields(cls);
            for(Field f:fids)
            {
                for(int i=0;i<arr.length;i++)
                    if(arr[i].pre.test(f)) arr[i].add(pc.createProperty(f));
            }
        }
    }
    //</editor-fold>
}
