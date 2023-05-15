/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.common.state;

import java.util.ArrayList;
import z.dragon.common.state.State.StateValue;

public final class FloatArrayValue implements StateValue 
{
    private final float[] value;

    public FloatArrayValue(float... value) {
        if(value == null) throw new RuntimeException();
        this.value = value;
    }

    @Override public float[] value() { return value; }

    @Override public Class<?> type() { return float[].class; }

    @Override
    public ArrayList<String> toStringLines() {
        StringBuilder sb = new StringBuilder(128);
        for (float v : value) sb.append(v).append(',');
        String line = sb.deleteCharAt(sb.length() - 1).toString();
        ArrayList<String> lines = new ArrayList<>(1);
        lines.add(line);
        return lines;
    }
}
