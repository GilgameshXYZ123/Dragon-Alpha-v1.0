/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cifar10.googlenet;

import z.dragon.alpha.Alpha.Line;
import static z.dragon.alpha.Alpha.UnitBuilder.nn;
import static z.dragon.alpha.Alpha.UnitFunctional.F;
import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Tensor;
import z.dragon.nn.unit.Unit;
import z.dragon.nn.unit.module.Module;

/**
 *
 * @author Gilgamesh
 */
public class Net
{
    public static class Inception extends Module
    {
        Unit branch1, branch2, branch3, branch4;
        public Inception(int in_channels,
                int ch1x1, 
                int ch3x3red, int ch3x3,
                int ch5x5red, int ch5x5,
                int pool_proj)
        {
            branch1 = nn.sequence(
                    nn.conv3D(true, in_channels, ch1x1, 1, 1, 0),
                    nn.leakyRelu());
            
            branch2 = nn.sequence(
                    nn.conv3D(true, in_channels, ch3x3red, 1, 1, 0),
                    nn.leakyRelu(),
                    nn.conv3D(true, ch3x3red, ch3x3, 3, 1, 1),
                    nn.leakyRelu());
            
            branch3 = nn.sequence(
                    nn.conv3D(true, in_channels, ch5x5red, 1, 1, 0),
                    nn.leakyRelu(),
                    nn.conv3D(true, ch5x5red, ch5x5, 5, 1, 2),
                    nn.leakyRelu());
            
            branch4 = nn.sequence(
                    nn.maxPool2D(3, 1, 1),
                    nn.conv3D(true, in_channels, pool_proj, 1, 1, 0),
                    nn.leakyRelu()
            );
        }

        @Override
        public Tensor[] __forward__(Tensor... X) {
            Line<Tensor> X1 = alpha.line(()-> branch1.forward(X)[0]);
            Line<Tensor> X2 = alpha.line(()-> branch2.forward(X)[0]);
            Line<Tensor> X3 = alpha.line(()-> branch3.forward(X)[0]);
            Line<Tensor> X4 = alpha.line(()-> branch4.forward(X)[0]);
            return F.concat(X1.c(), X2.c(), X3.c(), X4.c());
        }
    }

    public static class GoogLeNet extends Module
    {
        Unit conv1 = nn.conv3D(true, 3, 64, 5, 1, 2);
        Unit conv2 = nn.conv3D(true, 64, 192, 3, 1, 1);
        Unit bn1 = nn.batchNorm(false, 192);
        
        //in_channels, ch1x1, ch3x3red, ch3x3,ch5x5red, ch5x5, pool_proj
        Unit inception3a = new Inception(192,  64,  96, 128, 16, 32, 32);//[ 64, 128, 32, 32] = 256
        Unit inception3b = new Inception(256, 128, 128, 192, 32, 96, 64);//[128, 192, 96, 64] = 480
        Unit bn2 = nn.batchNorm(480);
        
        Unit inception4a = new Inception(480, 192,  96, 208, 16,  48,  64);//[192, 208,  48,  64] = 512
        Unit inception4b = new Inception(512, 160, 112, 224, 24,  64,  64);//[160, 224,  64,  64] = 512
        Unit inception4c = new Inception(512, 128, 128, 256, 24,  64,  64);//[128, 256,  64,  64] = 512
        Unit inception4d = new Inception(512, 112, 144, 288, 32,  64,  64);//[112, 288,  64,  64] = 580
        Unit inception4e = new Inception(528, 256, 160, 320, 32, 128, 128);//[256, 320, 128, 128] = 832 
        
        Unit inception5a = new Inception(832, 256, 160, 320, 32, 128, 128);//[256, 320, 128, 128] = 832
        Unit inception5b = new Inception(832, 384, 192, 384, 48, 128, 128);//[384, 384, 128, 128] = 1024

        Unit classifier = nn.sequence(
                nn.dropout(0.9f),
                nn.fullconnect(true, 1024, 256),
                nn.leakyRelu(),
                nn.dropout(false, 0.9f),
                nn.fullconnect(true, 256, 10)
        );
        
        @Override
        public Tensor[] __forward__(Tensor... X) {
            X = conv1.forward(X);//(32, 32, 3) -> (32, 32, 64)
            X = F.leakyRelu(X);
            X = F.maxPool2D(2, X);//(32, 32, 64) -> (16, 16, 64)
            
            X = conv2.forward(X);//(16, 16, 64) -> (16, 16, 192)
            X = bn1.forward(X);
            X = F.leakyRelu(X);
            X = F.maxPool2D(2, X);//(16, 16, 192) -> (8, 8, 192)
            
            X = inception3a.forward(X);//(8, 8, 192) -> (8, 8, 256)
            X = inception3b.forward(X);//(8, 8, 192) -> (8, 8, 480)
            X = bn2.forward(X);
            X = F.maxPool2D(2, X);//(8, 8, 480) -> (4, 4, 480)
            
            X = inception4a.forward(X);//(4, 4, 480) -> (4, 4, 512)
            X = inception4b.forward(X);//(4, 4, 512) -> (4, 4, 512)
            X = inception4c.forward(X);//(4, 4, 512) -> (4, 4, 512)
            X = inception4d.forward(X);//(4, 4, 512) -> (4, 4, 528)
            X = inception4e.forward(X);//(4, 4, 528) -> (4, 4, 832) 
            X = F.maxPool2D(2, X);//(4, 4, 832) -> (2, 2,832)
            
            X = inception5a.forward(X);//(4, 4, 832) -> (2, 2, 832)
            X = inception5b.forward(X);//(2, 2, 832) -> (2, 2, 1024)
            X = F.adaptive_avgPool2D(1, X);//(2, 2, 1024) -> (1, 1, 1024)
            
            X = classifier.forward(F.flaten(X));
            return X;
        }
        
        String weight = "C:\\Users\\Gilgamesh\\Desktop\\cifar10-GoogLeNet.zip";
        public void load() { alpha.stat.load_zip(this, weight); }
        public void save() { alpha.stat.save_zip(this, weight);}
    }
}
