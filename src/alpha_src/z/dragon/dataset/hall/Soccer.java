/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.dataset.hall;

import java.io.File;
import javax.xml.parsers.DocumentBuilderFactory;
import org.w3c.dom.Document;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import static z.dragon.alpha.Alpha.Datas.data;
import static z.dragon.alpha.Alpha.alpha;
import static z.dragon.common.DragonCV.cv;
import z.dragon.data.DataSet;
import z.dragon.data.Pair;
import z.dragon.data.container.AutoLoadContainer;
import z.dragon.data.container.AutoLoadContainer.Loader;
import z.util.xml.XML;

/**
 *
 * @author Gilgamesh
 */
public final class Soccer
{
    public static final int label_size = 4;
    public static final int[] picture_dim = {128, 128, 3};
    
    private Soccer() {}
    
    //<editor-fold defaultstate="collapsed" desc="data-loader">
    static class SLoader implements Loader<byte[], float[]>
    {
        private final String picPath;
        private final String xmlPath;
        
        private final File[] pics;
        private final int start, mod;
        private volatile int picIndex = 0;
        
        SLoader(int start, int mod) {
            picPath = alpha.home() + "\\data\\soccer\\JPEGImages\\";
            xmlPath = alpha.home() + "\\data\\soccer\\Annotations\\";
            pics = new File(picPath).listFiles();
            this.start = start;
            this.mod = mod; 
        }
        
        public float[] getY(File file) throws Exception
        {
            DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
            Document doc= factory.newDocumentBuilder().parse(file);
            NodeList list = doc.getElementsByTagName("bndbox").item(0).getChildNodes();
            
            float[] Y = new float[4]; int index = 0;
            for(int i = 0, len = list.getLength(); i<len;i++) {
                Node cur = list.item(i);
                if(cur.getNodeType()!=Node.ELEMENT_NODE) continue;
                float value = Float.valueOf(XML.getFirstChildNodeValue(cur));
                value /= (index < 2 ? 1280f : 720f);
                Y[index++] = value;
            }
            return Y;
        }
        
        @Override
        public Pair<byte[], float[]> load() 
        {
            File pic = pics[picIndex];//test: start = 1200, mod = 88; train: start = 0, mod = 1200
            synchronized(this) { picIndex = start + (picIndex + 1) % mod; }
            try
            {
                //read X--------------------------------------------------------
                byte[] X = cv.pixel(cv.reshape(cv.imread(pic), 128, 128));
                
                //read Y--------------------------------------------------------
                String xmlName = pic.getName();
                int end = xmlName.lastIndexOf('.');
                File xml = new File(xmlPath + xmlName.substring(0, end) + ".xml");
                float[] Y =this.getY(xml);
                return new Pair<>(X, Y);
            }
            catch(Exception e) {
                throw new RuntimeException(e);
            }
        }
    };
    //</editor-fold>
    
    public static DataSet<byte[], float[]> test() {
        AutoLoadContainer<byte[], float[]> conta = data.autoLoad(byte[].class, float[].class)
                .loader(new SLoader(1200, 88));
        return data.dataset(conta)
                .input_transform(data.image_transform(picture_dim))
                .label_transform(data.floats_transform(picture_dim));
    }
    
    public static DataSet<byte[], float[]> train() {
        AutoLoadContainer<byte[], float[]> conta = data.autoLoad(byte[].class, float[].class)
                .loader(new SLoader(0, 1200));
        return data.dataset(conta)
                .input_transform(data.image_transform(picture_dim))
                .label_transform(data.floats_transform(label_size));
    }
}
