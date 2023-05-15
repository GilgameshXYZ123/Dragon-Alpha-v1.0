/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.dual.math1;

import z.dragon.common.state.State;
import z.dragon.engine.Engine;
import z.dragon.engine.Parameter.ParamMap;
import z.dragon.engine.Parameter.ParamSet;
import z.dragon.nn.unit.dual.DualUnit;

/**
 *
 * @author Gilgamesh
 */
public abstract class DualFunction extends DualUnit
{
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append("{ }");
    }
    
    @Override protected void __init__(Engine eg) {}
    @Override public void params(ParamSet set) {}
    @Override public void param_map(ParamMap<String> set) {}
    @Override public void state(State dic) {}
    @Override public void update_state(State dic, boolean partial) {}
}
