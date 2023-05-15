/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cifar10.resnet34;

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
    public static class BasicBlock extends Module
    {
        Unit conv1, bn1, conv2, bn2, downsample;
        public BasicBlock(int in_channel, int out_channel, int stride) {
            conv1 = nn.conv3D(false, in_channel, out_channel, 3, stride, 1);
            bn1 = nn.batchNorm(false, out_channel);
            conv2 = nn.conv3D(false, out_channel, out_channel, 3, 1, 1);
            bn2 = nn.batchNorm(out_channel);
           
            if(stride != 1 || out_channel != in_channel) 
                downsample = nn.sequence(
                        nn.conv3D(false, in_channel, out_channel, 3, stride, 1),
                        nn.batchNorm(out_channel)
                );
        }

        @Override
        public Tensor[] __forward__(Tensor... X) {
            Tensor[] res = X;

            X = bn1.forward(conv1.forward(X));
            X = F.leakyRelu(X);
            X = bn2.forward(conv2.forward(X));
            
            if(downsample != null) res = downsample.forward(res);
            
            X = F.add(X[0], res[0]);
            X = F.leakyRelu(X);
            return X;
        }
    }
    
    public static class ResNet34 extends Module
    {
        Unit prepare = nn.sequence(//div2, channel: 3 -> 64
                nn.conv3D(true, 3, 64, 3, 2, 1),
                nn.batchNorm(false, 64),
                nn.leakyRelu()
        );
        
        Unit layer1 = nn.sequence(//div2, channel: 64 -> 64
                new BasicBlock(64, 64, 1),
                new BasicBlock(64, 64, 1),
                new BasicBlock(64, 64, 1),
                new BasicBlock(64, 64, 1)
        );
        
        Unit layer2 = nn.sequence(//div4, channel: 64 -> 128
                new BasicBlock(64, 128, 2),
                new BasicBlock(128, 128, 1),
                new BasicBlock(128, 128, 1),
                new BasicBlock(128, 128, 1)
        );
        
        Unit layer3 = nn.sequence(//div8, channel: 128 -> 256
                new BasicBlock(128, 256, 2),
                new BasicBlock(256, 256, 1),
                new BasicBlock(256, 256, 1),
                new BasicBlock(256, 256, 1),
                new BasicBlock(256, 256, 1),
                new BasicBlock(256, 256, 1)
        );
        
        Unit layer4 = nn.sequence(//div16, channel: 256 -> 512
                new BasicBlock(256, 512, 2),
                new BasicBlock(512, 512, 1),
                new BasicBlock(512, 512, 1)
        );
        
        Unit pool = nn.adaptive_avgPool2D(1);
        
        Unit fc = nn.sequence(
                nn.dropout(0.9f),
                nn.fullconnect(true, 512, 256),
                nn.leakyRelu(),
                nn.dropout(false, 0.9f),
                nn.fullconnect(true, 256, 10)
        );

        @Override
        public Tensor[] __forward__(Tensor... X) {
            X = prepare.forward(X);
            
            X = layer1.forward(X);
            X = layer2.forward(X);
            X = layer3.forward(X);
            X = layer4.forward(X);
            X = pool.forward(X);
            
            X = fc.forward(F.flaten(X));
            return X;
        }
        
        String weight = "C:\\Users\\Gilgamesh\\Desktop\\cifar10-Resnet34.zip";
        public void load() { alpha.stat.load_zip(this, weight); };
        public void save() { alpha.stat.save_zip(this, weight); }
    }
}
