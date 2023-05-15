/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.engine.c;

import z.dragon.engine.EngineBase;

/**
 *
 * @author Gilgamesh
 */
public abstract class CFloat32Base extends EngineBase
{
    public CFloat32Base(String dataType, long LsizeofDatatype) {
        super("Cfloat32", 2, "cint32", "cint8");
    }
}