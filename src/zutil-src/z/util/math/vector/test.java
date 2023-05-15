/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.math.vector;

import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;

/**
 *
 * @author dell
 */
public class test 
{
    public native static void print(double[] x);
  
    public static void main(String[] args)
    {
        System.load("D:\\virtual disc Z-Gilgamesh\\Gilgamesh java2\\ZUTIL-STD-1.1\\src\\z\\util\\math\\vector\\test.dll");
        print(Vector.randomDoubleVector(10));
    }
}
