/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cifar10.alexnet;

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
public class AlexNet extends Module
{
    Unit prepare = nn.sequence(
            nn.conv3D(true, 3, 96, 6, 2, 2),
            nn.leakyRelu(),
            nn.batchNorm(false, 96),
            nn.maxPool2D(3, 2, 1)
    );
    
    Unit layer1 = nn.sequence(
            nn.conv3D(true, 96, 256, 5, 1, 2),
            nn.leakyRelu(),
            nn.batchNorm(false, 256),
            nn.maxPool2D(3, 2, 1)
    );
    
    Unit layer2 = nn.sequence(
            nn.conv3D(true, 256, 384, 3, 1, 1),
            nn.leakyRelu(),
            nn.conv3D(true, 384, 384, 3, 1, 1),
            nn.leakyRelu(),
            nn.conv3D(true, 384, 256, 3, 1, 1),
            nn.batchNorm(false, 256),
            nn.leakyRelu(),
            nn.maxPool2D(3, 2, 1)
    );
   
   Unit classifier = nn.sequence(
           nn.dropout(false, 0.9f),
           nn.fullconnect(true, 256 * 2 * 2, 512),
           nn.leakyRelu(),
           nn.dropout(false, 0.9f),
           nn.fullconnect(true, 512, 256),
           nn.leakyRelu(),
           nn.fullconnect(true, 256, 10)
   );

    @Override
    public Tensor[] __forward__(Tensor... X) {
        X = prepare.forward(X);
        
        X = layer1.forward(X);
        X = layer2.forward(X);
        
        X = classifier.forward(F.flaten(X));
        return X;
    }
         
    String weight = "C:\\Users\\Gilgamesh\\Desktop\\cifar10-AlexNet.zip";
    public void save() { alpha.stat.save_zip(this, weight); }
    public void load() { alpha.stat.load_zip(this, weight); }
}
