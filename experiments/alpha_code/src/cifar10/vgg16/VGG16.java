/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cifar10.vgg16;

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
public class VGG16 extends Module
{
    Unit conv1 = nn.conv3D(false, 3, 64, 3, 1, 1);
    Unit conv2 = nn.conv3D(false, 64, 64, 3, 1, 1);
    Unit bn1 = nn.batchNorm(false, 64);
    
    Unit conv3 = nn.conv3D(false, 64, 128, 3, 1, 1);
    Unit conv4 = nn.conv3D(false, 128, 128, 3, 1, 1);
    Unit bn2 = nn.batchNorm(false, 128);
    
    Unit conv5 = nn.conv3D(false, 128, 256, 3, 1, 1);
    Unit conv6 = nn.conv3D(false, 256, 256, 3, 1, 1);
    Unit conv7 = nn.conv3D(false, 256, 256, 3, 1, 1);
    Unit bn3 = nn.batchNorm(false, 256);
    
    Unit conv8 = nn.conv3D(false, 256, 512, 3, 1, 1);
    Unit conv9 = nn.conv3D(false, 512, 512, 3, 1, 1);
    Unit conv10 = nn.conv3D(false, 512, 512, 3, 1, 1);
    Unit bn4 = nn.batchNorm(false, 512);
    
    Unit conv11 = nn.conv3D(false, 512, 512, 3, 1, 1);
    Unit conv12 = nn.conv3D(false, 512, 512, 3, 1, 1);
    Unit conv13 = nn.conv3D(false, 512, 512, 3, 1, 1);
    
    Unit classifier = nn.sequence(
            nn.fullconnect(false, 512, 256),
            nn.leakyRelu(),
            nn.dropout(false, 0.9f),
            nn.fullconnect(false, 256, 128),
            nn.leakyRelu(),
            nn.dropout(false, 0.9f),
            nn.fullconnect(true, 128, 10));
    
    @Override
    public Tensor[] __forward__(Tensor... X) {
        X = F.leakyRelu(conv1.forward(X));//div2, channel: 3 -> 64
        X = F.leakyRelu(conv2.forward(X));
        X = bn1.forward(X);
        X = F.maxPool2D(2, X);
        
        X = F.leakyRelu(conv3.forward(X));//div4, channel: 64 -> 128
        X = F.leakyRelu(conv4.forward(X));
        X = bn2.forward(X);
        X = F.maxPool2D(2, X);
          
        X = F.leakyRelu(conv5.forward(X));//div8, channel: 128 -> 256
        X = F.leakyRelu(conv6.forward(X));
        X = F.leakyRelu(conv7.forward(X));
        X = bn3.forward(X);
        X = F.maxPool2D(2, X);

        X = F.leakyRelu(conv8.forward(X));//div16,  channel: 256 -> 512
        X = F.leakyRelu(conv9.forward(X));
        X = F.leakyRelu(conv10.forward(X));
        X = bn4.forward(X);
        X = F.maxPool2D(2, X);
        
        X = F.leakyRelu(conv11.forward(X));//div32, channel: 512 -> 512
        X = F.leakyRelu(conv12.forward(X));
        X = F.leakyRelu(conv13.forward(X));
        X = F.maxPool2D(2, X);
        
        X = classifier.forward(F.flaten(X));
        
        return X;
    }
    
    String weight = "C:\\Users\\Gilgamesh\\Desktop\\VGG16-cifar10.zip";
    public void load() { alpha.stat.load_zip(this, weight); };
    public void save() { alpha.stat.save_zip(this, weight); }
}
