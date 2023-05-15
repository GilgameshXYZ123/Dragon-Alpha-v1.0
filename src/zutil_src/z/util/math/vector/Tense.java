/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.math.vector;

import z.util.math.ExRandom;

/**
 *
 * @author Gilgamesh
 */
public class Tense 
{
    public static float[][][][] copy(float[][][][] T, int W, int Z, int Y, int X)
    {
        float[][][][] nT=new float[W][Z][Y][X];
        for(int w=0;w<W;w++)
        for(int z=0;z<Z;z++)
            for(int y=0;y<Y;y++)
            System.arraycopy(T[w][z][y], 0, nT[w][z][y], 0, X);
        return nT;
    }
    
    public static void println(float[][][][] T, int W, int Z, int Y, int X)
    {
        for(int w=0;w<W;w++)
        for(int z=0;z<W;z++)
        {
            System.out.println("=============================================");
            for(int y=0;y<Y;y++)
            {
                for(int x=0;x<X;x++)
                    System.out.print(T[w][z][y][x]+", ");
                System.out.println();
            }
            System.out.println("=============================================");
        }
    }
    
    /**
     * <pre>
     * Convert a 1d vector to a 4D Tensor.
     * Make sure: V.length >= dimW*dimZ*dimY*dimX
     * </pre>
     * @param T
     * @param V
     * @param dimW dim size on W axis of Tensor T
     * @param dimZ dim size on Z axis of Tensor T
     * @param dimY dim size on Y axis of Tensor T
     * @param dimX dim size on X axis of Tensor T
     */
    public static void vectorToTensor_4D(float[][][][] T, 
            float[] V, int dimW, int dimZ, int dimY, int dimX)
    {
        int index=0;
        for(int w=0; w<dimW; w++)
        for(int z=0; z<dimZ; z++)
        for(int y=0; y<dimY; y++)
        for(int x=0; x<dimX; x++) T[w][z][y][x] = V[index++];
    }
    public static float[][][][] vectorToTensor_4D(float[] V, int dimW, int dimZ, int dimY, int dimX)
    {
        
        float[][][][] T=new float[dimW][dimZ][dimY][dimX];
        Tense.vectorToTensor_4D(T, V, dimW, dimZ, dimY, dimX);
        return T;
    }
    
    /**
     * <pre>
     * Convert a 4D Tensor to a 1D Vector.
     * </pre>
     * @param V
     * @param T 
     */
    public static void tensor_4DToVector(float[] V, float[][][][] T)
    {
        int index=0;
        for(int w=0; w<T.length; w++)
        for(int z=0; z<T[w].length; z++)
        for(int y=0; y<T[w][z].length; y++)
        for(int x=0; x<T[w][z][y].length; x++) 
        {
            V[index++] = T[w][z][y][x];
//            System.out.println(T[w][z][y][x]);
        }
    }
    public static float[] tensor_4DToVector(float[][][][] T, int length)
    {
        float[] V=new float[length];
        tensor_4DToVector(V, T);
        return V;
    }
    
    //X[XstartC, XstratC+channelNum] = Y[YstratC, ]
    public static void channelCopy3D(
            float[][][][] X, int XstartC,
            float[][][][] Y, int YstartC,
            int N, int IH, int IW, int cnum)
    {
        for(int n=0; n<N; n++)
        for(int c=0; c<cnum; c++)
        {
            int xc = XstartC + c;
            int yc = YstartC + c;
            
            for(int ih=0; ih<IH; ih++)
            for(int iw=0; iw<IW; iw++)
            {
                Y[n][yc][ih][iw] = X[n][xc][ih][iw];
//                System.out.println(Y[n][yc][ih][iw]);
            }
        }
    }
    
    //<editor-fold defaultstate="collapsed" desc="3D convolution">
    /**
     * <pre>
     * 3d convolution implemented in a simple way.
     * X: 4D Tensor[N , IC, IH, IW]
     * W: 4d Tensor[OC, IC, FH, FW]
     * Y: 4D Tensor[N , OC, OH, OW]
     * </pre>
     * @param X 
     * @param IH dimension size on Y axis of Tensor X
     * @param IW dimension size on X axis of Tensor X
     * @param W
     * @param FH dimension size on Y axis of Tensor W
     * @param FW dimension size on X axis of Tensor W
     * @param Y
     * @param OH dimension size on Y axis of Tensor Y
     * @param OW dimension size on X axis of Tensor Y
     * @param N batchSize
     * @param IC input channel num
     * @param OC output channel num
     * @param sh stride on Y axis
     * @param sw stride on X axis
     * @param ph padding on Y axis
     * @param pw padding on X axis
     */
    public static void conv3D_naive(
        float[][][][] X, int IH, int IW,
	float[][][][] W, int FH, int FW,
	float[][][][] Y, int OH, int OW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw)
    {
        System.out.format("(IH, IW, FH, FW, OH, OW) = (%d, %d, %d, %d, %d, %d)\n", IH, IW, FH, FW, OH, OW);
        System.out.format("(N, IC, OC) = (%d, %d, %d)\n", N, IC, OC);
        System.out.format("(ph, pw, sh, sw) = (%d, %d, %d, %d)\n", ph, pw, sh, sw);
        
	for (int n = 0; n < N; n++)
        for (int oc = 0; oc < OC; oc++)
	{
            int ih_s, iw_s, oh, ow;
            for (ih_s = -ph, oh = 0; ih_s <= (IH + ph - FH) && oh<OH; ih_s += sh, oh++)//oh < OH
            for (iw_s = -pw, ow = 0; iw_s <= (IW + pw - FW) && ow<OW; iw_s += sw, ow++)//ow < OW
            {
                double v = 0;
                for (int fh = 0; fh < FH; fh++)
                for (int fw = 0; fw < FW; fw++)
                for (int ic = 0; ic < IC; ic++)
		{  
                    int ih = ih_s + fh, iw = iw_s + fw;
                    if (ih < 0 || iw < 0 || ih >= IH || iw >= IW) continue;
                    v += X[n][ih][iw][ic] * W[oc][fh][fw][ic];
		}
                Y[n][oh][ow][oc] = (float) v;
            }
        }
    }
    
    public static void conv3D_naive_double(
        double[][][][] X, int IH, int IW,
	double[][][][] W, int FH, int FW,
	double[][][][] Y, int OH, int OW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw)
    {
        System.out.format("(IH, IW, FH, FW, OH, OW) = (%d, %d, %d, %d, %d, %d)\n", IH, IW, FH, FW, OH, OW);
        System.out.format("(N, IC, OC) = (%d, %d, %d)\n", N, IC, OC);
        System.out.format("(ph, pw, sh, sw) = (%d, %d, %d, %d)\n", ph, pw, sh, sw);
        
	for (int n = 0; n < N; n++)
        for (int oc = 0; oc < OC; oc++)
	{
            int ih_s, iw_s, oh, ow;
            for (ih_s = -ph, oh = 0; ih_s <= (IH + ph - FH) && oh<OH; ih_s += sh, oh++)//oh < OH
            for (iw_s = -pw, ow = 0; iw_s <= (IW + pw - FW) && ow<OW; iw_s += sw, ow++)//ow < OW
            {
                double v = 0;
                for (int fh = 0; fh < FH; fh++)
                for (int fw = 0; fw < FW; fw++)
                for (int ic = 0; ic < IC; ic++)
		{
                    int ih = ih_s + fh, iw = iw_s + fw;
                    if (ih < 0 || iw < 0 || ih >= IH || iw >= IW) continue;
                    v += X[n][ih][iw][ic] * W[oc][fh][fw][ic];
		}
                Y[n][oh][ow][oc] = v;
            }
        }
    }
    
    /**
     * <pre>
     * 3d convolution implemented using Img2col Algorithn.
     * X: 4D Tensor[N , IC, IH, IW]
     * W: 4d Tensor[OC, IC, FH, FW]
     * Y: 4D Tensor[N , OC, OH, OW]
     * Convert 4D Tensor X, W, Y to Matrix:
     * (1) X -> Matrix A[GN, GK]
     * (2) W -> Matrix B[GK, GM]
     * (3) Y -> Matrix C[GN, GM]
     * GN = OC;
     * GM = N * OH * OW;
     * GK = IC * FH * FW;
     * </pre>
     * @param X 
     * @param IH dimension size on Y axis of Tensor X
     * @param IW dimension size on X axis of Tensor X
     * @param W
     * @param FH dimension size on Y axis of Tensor W
     * @param FW dimension size on X axis of Tensor W
     * @param Y
     * @param OH dimension size on Y axis of Tensor Y
     * @param OW dimension size on X axis of Tensor Y
     * @param N batchSize
     * @param IC input channel num
     * @param OC output channel num
     * @param sh stride on Y axis
     * @param sw stride on X axis
     * @param ph padding on Y axis
     * @param pw padding on X axis
     */
    public static void conv3D_img2col(
	float[][][][] X, int IH, int IW, //X[N , IC, IH, IW] => A[GN, GK]
	float[][][][] W, int FH, int FW, //W[OC, IC, KH, KW] => B[GK, GM]
	float[][][][] Y, int OH, int OW, //Y[N , OC, OH, OW] => C[GN, GM]
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw)
    {
	int GN = OC;
        int GM = N * OH * OW;
	int GK = IC * FH * FW;
	for (int i = 0; i < GN; i++)
	{
            int oc = i;
            for (int j = 0; j < GM; j++)
            {
		int n = j / (OH*OW);
		int j_res = j % (OH*OW);
		int oh = j_res / OW, ow = j_res % OW;
                
		float v = 0;
		for (int k = 0; k < GK; k++)
		{
                    int ic = k / (FH*FW);
                    int k_res = k % (FH*FW);
                    int fh = k_res / FW, fw = k_res % FW;
                    int ih = oh * sh - ph + fh;
                    int iw = ow * sw - pw + fw;

                    if (ih < 0 || iw < 0 || ih >= IH || iw >= IW) continue;
                    v+=X[n][ih][iw][ic] * W[oc][fh][fw][ic];
		}
                Y[n][oh][ow][oc]=v;
            }
	}
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="3D deconvolution deltaX">
    public static float[][][][] innerPadding_Y_X_axis(
            float[][][][] deltaY, int N, int OC, int OH, int OW, int sh, int sw)
    {
        int OH_p = OH + (OH-1)*(sh-1), OW_p = OW + (OW-1)*(sw-1);
        float[][][][] deltaY_p=new float[N][OC][OH_p][OW_p];
        for(int n=0;n<N;n++)
        for(int oc=0;oc<OC;oc++)
        for(int oh=0;oh<OH;oh++)
        {
            int oh_p=oh*sh;
            for(int ow=0;ow<OW;ow++)
                deltaY_p[n][oc][oh_p][ow*sw]=deltaY[n][oc][oh][ow];
        }
        return deltaY_p;
    }
    
    public static float[][][][] rot180_Y_X_axis_exchange_IC_OC_axis(
            float[][][][] W, int OC, int IC, int FH, int FW)
    {
        float[][][][] W_r=new float[IC][OC][FH][FW];
        for(int oc=0;oc<OC;oc++)
        for(int ic=0;ic<IC;ic++)
            for(int fh=0;fh<FH;fh++)
            for(int fw=0;fw<FW;fw++)
                W_r[ic][oc][fh][fw] = W[oc][ic][FH-1-fh][FW-1-fw];
        return W_r;
    }
    
    //the native implementation
    //(1) exchange the IC and OC axis of the 4d Tensor W, while rot 180 in XoY surface: 
    //    W -> W_r: W_r[ic][oc][FH-fh-1][FW-fw-1]  = W[oc][ic][fh][fw]
    //(2) inner padding 4d tensor deltaX on X and Y axis: 
    //    iph = (sh-1)*OH
    //    ipw = (sw-1)*OW
    //deltaY = deconv(deltaX, )
    public static void deconv3d_deltaX_naive(
        float[][][][] deltaY, int OH, int OW,
	float[][][][] W, int FH, int FW,
	float[][][][] deltaX, int IH, int IW,
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw)
    {
        
        int OH_p = OH + (OH-1)*(sh-1), OW_p = OW + (OW-1)*(sw-1);
        int oph = FH-ph-1, opw = FW-pw-1;
        float[][][][] deltaY_p=Tense.innerPadding_Y_X_axis(deltaY, N, OC, OH, OW, sh, sw);
        //System.out.format("dYp: %d, %d, %d\n", deltaY_p.length, deltaY_p[0].length, deltaY_p[0][0].length, deltaY_p[0][0][0].length);
        
        float[][][][] W_r=Tense.rot180_Y_X_axis_exchange_IC_OC_axis(W, OC, IC, FH, FW);
        
        Tense.conv3D_naive(deltaY_p, OH_p, OW_p, W_r, FH, FW, deltaX, IH, IW, N, OC, IC, 1, 1, oph, opw);
        //Tensor.conv3D_img2col(deltaY_p, OH_p, OW_p, W_r, FH, FW, deltaX, IH, IW, N, OC, IC, 1, 1, oph, opw);
    }
    
    //<editor-fold defaultstate="collapsed" desc="deconv3D_deltaX_img2col">
    //GN = IC
    //GM = N*IH*IW
    //GK = OC*FH*FW
    public static void deconv3D_deltaX_img2col(
	float[][][][] deltaY, int OH, int OW, 
	float[][][][] W, int FH, int FW, 
	float[][][][] deltaX, int IH, int IW, 
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw)
    {
	int GN = IC;
        int GM = N * IH * IW;
	int GK = OC * FH * FW;
        
        int OHp = OH + (OH-1)*(sh-1), OWp = OW + (OW-1)*(sw-1);
        int oph = FH-ph-1, opw = FW-pw-1;
        
	for (int i = 0; i < GN; i++)
	{
            int ic = i;
            for (int j = 0; j < GM; j++)
            {
		int n = j / (IH*IW);
		int j_res = j % (IH*IW);
		int ih = j_res / IW, iw = j_res % IW;
                
		float v = 0;
		for (int k = 0; k < GK; k++)
		{
                    int oc = k / (FH*FW);
                    int k_res = k % (FH*FW);
                    int fh = k_res / FW, fw = k_res % FW;
                    
                    int oh = ih - oph + fh;
                    int ow = iw - opw + fw;
                    
                    if (oh < 0 || ow < 0 || oh >= OHp || ow >= OWp) continue;
                    if (oh%sh != 0 || ow%sw!=0) continue;
                    
                    v += deltaY[n][oh / sh][ow / sw][oc] * W[oc][FH - 1 - fh][FW - 1 - fw][ic];
		}
                deltaX[n][ih][iw][ic] = v;
            }
	}
    }
    
    public static void deconv3D_deltaX_img2col2(
	float[][][][] deltaY, int OH, int OW, 
	float[][][][] W, int FH, int FW, 
	float[][][][] deltaX, int IH, int IW, 
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw)
    {
        System.out.println("deconv3d deltaX img2col2");
        
	int GN = IC;
        int GM = N * IH * IW;
        
        int OH_p = OH + (OH-1)*(sh-1), OW_p = OW + (OW-1)*(sw-1);
        int oph = FH-ph-1, opw = FW-pw-1;
        
	for (int i = 0; i < GN; i++)
	{
            int ic = i;
            for (int j = 0; j < GM; j++)
            {
		int n = j / (IH*IW);
		int j_res = j % (IH*IW);
		int ih = j_res / IW, iw = j_res % IW;
                
                
                int fw_s=0, fh_s=0;
                loop:
                for(;fh_s<FH;fh_s++)
                {
                    int oh = ih - oph + fh_s;
                    if(oh<0 || oh>=OH_p || oh%sh!=0) continue;
                    for (fw_s = 0; fw_s < FW; fw_s++) 
                    {
                        int ow = iw - opw + fw_s;
                        if(ow >= 0 && ow < OW_p && ow % sw == 0) break loop;
                    }
                }
                
                int FH_r = (FH - fh_s + sh-1)/sh;
                int FW_r = (FW - fw_s + sw-1)/sw;
                int GK_r = FH_r*FW_r*OC;
                
		float v = 0;
		for (int k = 0; k < GK_r; k++)
		{
                    int oc = k/(FH_r * FW_r);
                    int k_res = k%(FH_r * FW_r);
                    int fh_r = k_res/FW_r, fw_r = k_res%FW_r;
                    int fh = fh_r*sh + fh_s;
                    int fw = fw_r*sw + fw_s;
                    
                    int oh = ih - oph + fh;
                    int ow = iw - opw + fw;
                    
                    if (oh >= OH_p || ow >= OW_p) continue;
                    v += deltaY[n][oc][oh/sh][ow/sw] * W[oc][ic][FH - 1 - fh][FW-1-fw];
		}
                deltaX[n][ic][ih][iw] = v;
            }
	}
    }
    //</editor-fold>
     
    //<editor-fold defaultstate="collapsed" desc="deconv3D_deltaX_CrossAdd">
    public static void deconv3D_deltaX_crossAdd(
        float[][][][] deltaY, int OH, int OW, //Y[N , OC, OH, OW] => C[GN, GM]
        float[][][][] W, int FH, int FW, //W[OC, IC, KH, KW] 
	float[][][][] deltaX, int IH, int IW, //X[N , IC, IH, IW] => A[GN, GK]
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw)
    {
        
        for(int oc=0; oc<OC; oc++)//IC可以很小, blockIdx.y
        for(int n=0; n<N; n++) 
        for(int oh=0; oh<OH; oh++)//oh, ow, n, blockIdx.x
        for(int ow=0; ow<OW; ow++)     
        {
            float dy = deltaY[n][oh][ow][oc];
            
            for(int fh=0; fh<FH; fh++)
            for(int fw=0; fw<FW; fw++) 
            {
                int ih = oh * sh - ph + fh;
                int iw = ow * sw - pw + fw;
                if(ih < 0 || iw <0 || ih >= IH || iw >= IW) continue;
                for(int ic=0; ic<IC; ic++)
                {
                    float w = W[oc][fh][fw][ic];
                    deltaX[n][ih][iw][ic] += w*dy;
                }
            }
        }
    }
    //</editor-fold>
   
    //<editor-fold defaultstate="collapsed" desc="deconv3D_deltaX_kernelSplit">
    /**
     *   //===================[STEP 1] split kernel==============================
        //sh * sw smallW: smallW (y, x)
        //(y, x) from (0, 0) to (sh, sw)
        
        //smallW(y, x)[fh'][fw'] = W[y + fh'*sh][x + fw'*sw]
        //transposed: smallWT(y, x)[fh'][fw'] = smallW(y, x)[smallFH - 1 - fh'][smallFW - 1 - fw']
        //so: smallWT(y, x)[smallFH - 1 -fh'][smallFW - 1 - fw'] = W[y + fh'*sh][x + fw'*sw]
        //let: fhr' = smallFH - 1 -fh'
        //     fwr' = smallFW - 1 - fw'
        //we have: smallWT(y, x)[fhr'][fwr'] = W[y + (smallFH - 1 - fhr')*sh][x + (smallFW - 1 - fwr')*sw]
        //fhr' -> fh = y + (smallFH - 1 - fhr')*sh
        //fwr' -> fw = x + (smallFW - 1 - fwr')*sw
        //if(fh <0 || fw<0 || fh>=FH || fw>=FW) smallWT(y, x) = 0;
        
        //======================[STEP 2] do convolution=========================
        //padding: oph, opw
        //for: smallWT(y, x)
        //for n from 0 to N:
        //for ic from 0 to IC;
        //
        //  for ohs from -oph to [OH - smallFH]:
        //  for ows from -opw to [OW - smallFW]:
        //      float dx = 0;
        //      for fhr' from 0 to smallFH:
        //      for fwr' from 0 to smallFW:
        //      for oc from 0 to OC:
        //          oh = ohs + fhr';
        //          ow = ows + fwr';
        //          dy = (oh<0 || ow<0) ? 0 :deltaY[n][oh][ow][oc];
        //
        //          fh = y + (smallFH - 1 - fhr')*sh;
        //          fw = x + (smallFW - 1 - fwr')*sw;
        //          w = (fh<0 || fh>=FH || fw<0 || fw>=FW)? 0 : W[oc][fh][fw][ic];
        //
        //          dx += w*dy;
        
        //      depth2Space:
        //      ih = y + (ohs + oph) * sh - ph
        //      iw = x + (ows + opw) * sw - pw
        //      if(ih<0 || iw <0) continue;
        //      deltaX[n][ih][iw][ic] = dx;
        //======================[STEP 3] find oph, opw==========================
        //As: ih = y + (ohs + oph) * sh - ph
        //so: IH - 1 = max: y + (ohs + oph) * sh - ph
        //max: y = sh
        //max: ohs + oph = OH - smallFH + oph
        //IH = sh + (OH - smallFH + oph) * sh - ph  
        //so: IH - sh + ph = (OH - smallFH + oph) * sh
        //IH + ph = (OH - smallFH + oph + 1)*sh
        //IH + ph = OH * sh - smallFH * sh + (oph + 1) * sh
        // = ((IH + 2ph - FH)/sh + 1)*sh - (FH + sh - 1)/sh*sh + (oph + 1)*sh
        // = (IH + 2ph - FH) + sh - (FH + sh - 1) + (oph + 1)*sh
        //oph = (2*FH - 1 - ph)/sh - 1
        
        //(1) oph = (IH - sh + ph)/sh + smallFH - OH
        //(2) opw = (IW - sw + pw)/sw + smallFW - OW
     * @param deltaY
     * @param OH
     * @param OW
     * @param W
     * @param FH
     * @param FW
     * @param deltaX
     * @param IH
     * @param IW
     * @param N
     * @param IC
     * @param OC
     * @param sh
     * @param sw
     * @param ph
     * @param pw 
     */
    public static void deconv3D_deltaX_kernelSplit(
        float[][][][] deltaY, int OH, int OW, //Y[N , OC, OH, OW] => C[GN, GM]
        float[][][][] W, int FH, int FW, //W[OC, IC, KH, KW] 
	float[][][][] deltaX, int IH, int IW, //X[N , IC, IH, IW] => A[GN, GK]
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw)
    {
        //0 -> o -> overlap
        float Nh0 = (FH / 2) * (1.0f / sh);
        float Nw0 = (FW / 2) * (1.0f / sw);
        
        boolean lh5 = (Nh0 - (int)Nh0) < 0.5f;
        boolean lw5 = (Nw0 - (int)Nw0) < 0.5f;
       
        int CFH = lh5 ? 2*(int)Nh0 + 1 : 2*(int)Nh0 + 2;
        int CFW = lw5 ? 2*(int)Nw0 + 1 : 2*(int)Nw0 + 2;
        
        //int oph = (IH + ph - sh + 1) / sh + CFH - OH;
        //int opw = (IW + pw - sh + 1) / sw + CFW - OW;
        //sh - 1 + sh*(OH - CFH + 2oph) - ph = IH - 1 + ph;
        //sh + sh*(OH - CFH + 2oph) = IH + ph*2
        //sh*(OH - CFH + 2oph + 1) = IH + ph*2
        //2oph + OH - CFH + 1 = (IH + 2*ph) / sh
        //2oph = (IH + 2*ph) / sh - OH + CFH - 1
        //oph = ((IH + 2*ph) / sh - OH + CFH - 1)/2
        //int oph = ((IH + 2*ph) / sh - OH + CFH - 1)/2;
        //int opw = ((IW + 2*pw) / sw - OW + CFW - 1)/2;
        //CFH - 1: 非常有趣，最边缘的input-feature恰好只和一个梯度（output-feature）有关
        int oph = CFH - 1;//(FH/sh + CFH - 2)/2;
        int opw = CFW - 1;//(FW/sw + CFW - 2)/2;
        
        int ih_max = 0, ih_min = 0;
        int iw_max = 0, iw_min = 0;
        
        System.out.println(CFH + ":" + CFW);
        
        //blockX = y*x*ohs*ows
        //(OH - CFH + 2oph + 1) * (OW - CFW + 2opw + 1) * sh * sw <-(x, y, ohs, ows)
        for(int ic=0; ic<IC; ic++)
        {
            for(int n=0; n<N; n++)
            for(int ohs = -oph; ohs < OH; ohs++) // for(int ohs = -oph; ohs <= (OH - CFH + oph); ohs++) OH - CFH + CFH - 1
            for(int ows = -opw; ows < OW; ows++)
            {
                for(int y=0; y<sh; y++)
                for(int x=0; x<sw; x++)
                {
                    int ih = y + (ohs + oph)*sh - ph;
                    int iw = x + (ows + opw)*sw - pw;
                    
                    ih_max = Math.max(ih, ih_max);
                    ih_min = Math.min(ih, ih_min);
                    
                    iw_max = Math.max(iw, iw_max);
                    iw_min = Math.min(iw, iw_min);
                    
                    if(ih < 0) System.exit(-2);
                    
                    if(ih<0 || iw<0 || ih>=IH || iw>=IW) continue;
                    
                    float dx = 0;
                    for(int fhr=0; fhr < CFH; fhr++)
                    for(int fwr=0; fwr < CFW; fwr++)
                    for(int oc=0; oc<OC; oc++) 
                    {
                        int oh = ohs + fhr;
                        int ow = ows + fwr;
                        float dy = (oh<0 || ow<0 || oh>=OH || ow>=OW)? 0 :deltaY[n][oh][ow][oc];
                        
                        int fh = y + (CFH - 1 - fhr)*sh;
                        int fw = x + (CFW - 1 - fwr)*sw;
                        
                        float w =  (fh>=FH || fw>=FW) ? 0 : W[oc][fh][fw][ic];
                        //float w =  (fh>=FH || fw>=FW) ? 0 : W[oc][fh][fw][ic];
//                        float w =  (fh <0 || fw<0 || fh>=FH || fw>=FW) ? 0 : W[oc][fh][fw][ic];
                        
                        dx += w * dy;
                    }
                    deltaX[n][ih][iw][ic] = dx;
                }
            }
            
            System.out.println(ih_max + ":" + ih_min + "\\" + (IH + oph - 1));
            System.out.println(iw_max + ":" + iw_min + "\\" + (IW + opw - 1));
        }
    }
    
    public static int deconv3D_deltaX_kernelSplit_v2(
        float[][][][] deltaY, int OH, int OW, //Y[N , OC, OH, OW] => C[GN, GM]
        float[][][][] W, int FH, int FW, //W[OC, IC, KH, KW] 
	float[][][][] deltaX, int IH, int IW, //X[N , IC, IH, IW] => A[GN, GK]
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw)
    {
        float Nh0 = (FH / 2) * (1.0f / sh);
        float Nw0 = (FW / 2) * (1.0f / sw);
        
        boolean lh5 = (Nh0 - (int)Nh0) < 0.5f;
        boolean lw5 = (Nw0 - (int)Nw0) < 0.5f;
       
        int CFH = lh5 ? 2*(int)Nh0 + 1 : 2*(int)Nh0 + 2;
        int CFW = lw5 ? 2*(int)Nw0 + 1 : 2*(int)Nw0 + 2;
       
        int oph = CFH - 1, opw = CFW - 1;
        
        //COH = (OH - CFH + 2*oph)/1 - 1
        //COH = (OH - CFH + 2*CFH - 2) - 1
        //COH = (OH + CFH - 3)*sh
        //= OH*sh + CFH*sh - 3*sh
        //= (IH - FH) + 2*ph + sh + CFH*sh - 3*sh
        //= IH - FH + 2*ph - 2*sh + CFH*sh
        //= IH + 2*ph - FH + (CFH - 2)*sh
        //= IH + 2*ph, 
        
        int ih_max = 0, ih_min = 0;
        int iw_max = 0, iw_min = 0;
        
        for(int y=0; y<sh; y++)
        for(int x=0; x<sw; x++)
        {
            for(int ic=0; ic<IC; ic++)
            {
                int hc = (ph - y + sh - 1) / sh;
                int wc = (pw - x + sw - 1) / sw;
                
                for(int n=0; n<N; n++)
                for(int ohs = -oph + hc; ohs <= (OH - CFH + oph - hc); ohs++) 
                for(int ows = -opw + wc; ows <= (OW - CFW + opw - wc); ows++)
                {
                    //ih = y + (ohs + oph) * sh - ph
                    //ohs = (ih + ph - y)/sh - oph
                    int ih = y + (ohs + oph)*sh - ph;
                    int iw = x + (ows + opw)*sw - pw;
                    
                    int ohs2 = (ih + ph - y)/sh - oph;
                    int ows2 = (iw + pw - x)/sw - opw;
                    
                    System.out.format("(ih, iw, ohs, ows, ohs2, ows) = (%d, %d, %d, %d, %d, %d)\n", 
                            ih, iw, ohs, ows, ohs2, ows);
                    
                    if(ohs2 != ohs || ows2 != ows) System.out.println(123123);
                    
                    
                    ih_max = Math.max(ih, ih_max);
                    ih_min = Math.min(ih, ih_min);
                            
                    iw_max = Math.max(iw, iw_max);
                    iw_min = Math.min(iw, iw_min);
                    
                    if(ih<0 || iw<0 || ih>=IH || iw>=IW) continue;
                    
                    float dx = 0;
                    for(int fhr=0; fhr < CFH; fhr++)
                    for(int fwr=0; fwr < CFW; fwr++)
                    for(int oc=0; oc<OC; oc++) 
                    {
                        int oh = ohs + fhr;
                        int ow = ows + fwr;
                        float dy = (oh<0 || ow<0 || oh>=OH || ow>=OW)? 0 :deltaY[n][oh][ow][oc];
                        
                        int fh = y + (CFH - 1 - fhr)*sh;
                        int fw = x + (CFW - 1 - fwr)*sw;
                        
                        float w =  (fh>=FH || fw>=FW) ? 0 : W[oc][fh][fw][ic];
                        dx += w * dy;
                    }
                    deltaX[n][ih][iw][ic] = dx;
                }
            }
        }
        
        int div = ih_max - IH;
        System.out.format("(lh5, ih_max, ih_min, IH - 1) = (%b, %d, %d, %d)\n", lh5, ih_max, ih_min, IH);
        System.out.format("(lw5, iw_max, iw_min, IW - 1) = (%b, %d, %d, %d)\n", lw5, iw_max, iw_min, IW);
        
        return div;
    }
    
    public static void deconv3D_deltaX_kernelSplit_v3(
        float[][][][] deltaY, int OH, int OW, //Y[N , OC, OH, OW] => C[GN, GM]
        float[][][][] W, int FH, int FW, //W[OC, IC, KH, KW] 
	float[][][][] deltaX, int IH, int IW, //X[N , IC, IH, IW] => A[GN, GK]
	int N, int IC, int OC,
	int sh, int sw, int ph, int pw)
    {
        float Nh0 = (FH / 2) * (1.0f / sh);
        float Nw0 = (FW / 2) * (1.0f / sw);
        
        boolean lh5 = (Nh0 - (int)Nh0) < 0.5f;
        boolean lw5 = (Nw0 - (int)Nw0) < 0.5f;
       
        int CFH = lh5 ? 2*(int)Nh0 + 1 : 2*(int)Nh0 + 2;
        int CFW = lw5 ? 2*(int)Nw0 + 1 : 2*(int)Nw0 + 2;
       
        int oph = CFH - 1, opw = CFW - 1;
        
        for(int ic=0; ic<IC; ic++)
        {
            for(int n=0; n<N; n++)
            for(int ohs = -oph; ohs <= (OH - CFH + oph); ohs++) 
            for(int ows = -opw; ows <= (OW - CFW + opw); ows++)
            {
                for(int y=0; y<sh; y++)
                for(int x=0; x<sw; x++)
                {
                    int ih = y + (ohs + oph)*sh - ph;
                    int iw = x + (ows + opw)*sw - pw;
                    
                    if(ih<0 || iw<0 || ih>=IH || iw>=IW) continue;
                     System.out.println(x + ", " + y + ":" + ih + ", " + iw);
                    
                    float dx = 0;
                    for(int fhr=0; fhr < CFH; fhr++)
                    for(int fwr=0; fwr < CFW; fwr++)
                    for(int oc=0; oc<OC; oc++) 
                    {
                        int oh = ohs + fhr;
                        int ow = ows + fwr;
                        float dy = (oh<0 || ow<0 || oh>=OH || ow>=OW)? 0 :deltaY[n][oh][ow][oc];
                        
                        int fh = y + (CFH - 1 - fhr)*sh;
                        int fw = x + (CFW - 1 - fwr)*sw;
                        
                        float w =  (fh>=FH || fw>=FW) ? 0 : W[oc][fh][fw][ic];
                        dx += w * dy;
                    }
                    deltaX[n][ih][iw][ic] = dx;
                }
            }
        }
    } 
    
    public static void deconv3D_deltaX_kernelSplit_v4(
            float[][][][] deltaY, int OH, int OW, //Y[N , OC, OH, OW] => C[GN, GM]
            float[][][][] W, int FH, int FW, //W[OC, IC, KH, KW] 
            float[][][][] deltaX, int IH, int IW, //X[N , IC, IH, IW] => A[GN, GK]
            int N, int IC, int OC,
            int sh, int sw, int ph, int pw)
    {
        float Nh0 = (FH / 2) * (1.0f / sh);
        float Nw0 = (FW / 2) * (1.0f / sw);
        
        boolean lh5 = (Nh0 - (int)Nh0) < 0.5f;
        boolean lw5 = (Nw0 - (int)Nw0) < 0.5f;
       
        int CFH = lh5 ? 2*(int)Nh0 + 1 : 2*(int)Nh0 + 2;
        int CFW = lw5 ? 2*(int)Nw0 + 1 : 2*(int)Nw0 + 2;
       
        int oph = CFH - 1, opw = CFW - 1;
        for(int ic=0; ic<IC; ic++)
        {
            for(int n=0; n<N; n++)
            for(int ih=0; ih<IH; ih++)
            for(int iw=0; iw<IW; iw++)
            {
//                int ohs = (ih + ph - y)/sh - oph;
//                int ows = (iw + pw - x)/sw - opw;
//                int ih = y + (ohs + oph)*sh - ph;
                int ohs = (ih + ph)/sh - oph;
                int ows = (iw + pw)/sw - opw;
                
//                int y = ih + ph - (ohs + oph)*sh;
//                int x = iw + pw - (ows + opw)*sw;
//                int y = (ih + ph) - (ih + ph)/sh*sh;
//                int x = (iw + pw) - (iw + pw)/sw*sw;
                int y = (ih + ph) % sh;
                int x = (iw + pw) % sw;
                
                //ih, iw -> ohs, ows, y, x
                //ohs, ows -> oh, ow -> deltaY
                //y, x, ic -> fh, fw -> W
                
                float dx = 0;
                for(int fhr=0; fhr < CFH; fhr++)
                for(int fwr=0; fwr < CFW; fwr++)
                for(int oc=0; oc<OC; oc++) 
                {
                    int oh = ohs + fhr;
                    int ow = ows + fwr;
                    float dy = (oh<0 || ow<0 || oh>=OH || ow>=OW)? 0 :deltaY[n][oh][ow][oc];
                        
                    int fh = y + (CFH - 1 - fhr)*sh;
                    int fw = x + (CFW - 1 - fwr)*sw;
                    
                    float w = (fh>=FH || fw>=FW) ? 0 : W[oc][fh][fw][ic];
                    dx += w * dy;
                }
                deltaX[n][ih][iw][ic] = dx;
            }
        }
    }
    
    public static void deconv3D_deltaX_kernelSplit_v5(
            float[][][][] deltaY, int OH, int OW, //Y[N , OC, OH, OW] => C[GN, GM]
            float[][][][] W, int FH, int FW, //W[OC, IC, KH, KW] 
            float[][][][] deltaX, int IH, int IW, //X[N , IC, IH, IW] => A[GN, GK]
            int N, int IC, int OC,
            int sh, int sw, int ph, int pw)
    {
        float Nh0 = (FH / 2) * (1.0f / sh);
        float Nw0 = (FW / 2) * (1.0f / sw);
        
        boolean lh5 = (Nh0 - (int)Nh0) < 0.5f;
        boolean lw5 = (Nw0 - (int)Nw0) < 0.5f;
       
        int CFH = lh5 ? 2*(int)Nh0 + 1 : 2*(int)Nh0 + 2;
        int CFW = lw5 ? 2*(int)Nw0 + 1 : 2*(int)Nw0 + 2;
       
        int oph = CFH - 1, opw = CFW - 1;
        
//        int y = (ih + ph) % sh; -> ih = G*sh + y - ph
//        int x = (iw + pw) % sw;
        for(int y=0; y<sh; y++)
        for(int x=0; x<sw; x++)
        {
            for(int ic=0; ic<IC; ic++)
            {
                for(int n=0; n<N; n++)
                for(int ih = y-ph; ih<IH; ih += sh)
                for(int iw = x-pw; iw<IW; iw += sw)
                {
                    if(ih <0 || iw < 0) continue;
                    int ohs = (ih + ph - y)/sh - oph;
                    int ows = (iw + pw - x)/sw - opw;

                    float dx = 0;
                    for(int fhr=0; fhr < CFH; fhr++)
                    for(int fwr=0; fwr < CFW; fwr++)
                    for(int oc=0; oc<OC; oc++) 
                    {
                        int oh = ohs + fhr;
                        int ow = ows + fwr;
                        float dy = (oh<0 || ow<0 || oh>=OH || ow>=OW)? 0 :deltaY[n][oh][ow][oc];
                        
                        int fh = y + (CFH - 1 - fhr)*sh;
                        int fw = x + (CFW - 1 - fwr)*sw;
                        float w = (fh>=FH || fw>=FW) ? 0 : W[oc][fh][fw][ic];
                        dx += w * dy;
                    }
                    deltaX[n][ih][iw][ic] = dx;
                }
            }
        }
    }
    
    public static void deconv3D_deltaX_kernelSplit_v6(
            float[][][][] deltaY, int OH, int OW, //Y[N , OC, OH, OW] => C[GN, GM]
            float[][][][] W, int FH, int FW, //W[OC, IC, KH, KW] 
            float[][][][] deltaX, int IH, int IW, //X[N , IC, IH, IW] => A[GN, GK]
            int N, int IC, int OC,
            int sh, int sw, int ph, int pw)
    {
        float Nh0 = (FH / 2) * (1.0f / sh);
        float Nw0 = (FW / 2) * (1.0f / sw);
        
        boolean lh5 = (Nh0 - (int)Nh0) < 0.5f;
        boolean lw5 = (Nw0 - (int)Nw0) < 0.5f;
       
        int CFH = lh5 ? 2*(int)Nh0 + 1 : 2*(int)Nh0 + 2;
        int CFW = lw5 ? 2*(int)Nw0 + 1 : 2*(int)Nw0 + 2;
       
        int oph = CFH - 1, opw = CFW - 1;
        
//        int y = (ih + ph) % sh; -> ih = G*sh + y - ph
//        int x = (iw + pw) % sw;
        for(int y=0; y<sh; y++)//Gz(y, x) = sh*sw
        for(int x=0; x<sw; x++)
        {
            for(int ic=0; ic<IC; ic++)//GN(ic) = IC
            {
                int ihs = y - ph + (ph - y + sh - 1)/sh*sh;
                int iws = x - pw + (pw - x + sw - 1)/sw*sw;
                
                //GM(m, ih, iw)
                //IH_slice = (IH - ihs + sh - 1)/sh 
                //IH_slice_max = (IH - min(ihs) + sh - 1) / sh
                //IH_slice_max = (IH + sh - 1)/sh
                for(int n=0; n<N; n++)
                for(int ih = ihs; ih<IH; ih += sh)
                for(int iw = iws; iw<IW; iw += sw)
                {
                    int ohs = (ih + ph - y)/sh - oph;
                    int ows = (iw + pw - x)/sw - opw;

                    float dx = 0;
                    for(int fhr=0; fhr < CFH; fhr++)
                    for(int fwr=0; fwr < CFW; fwr++)
                    for(int oc=0; oc<OC; oc++) 
                    {
                        int oh = ohs + fhr;
                        int ow = ows + fwr;
                        float dy = (oh<0 || ow<0 || oh>=OH || ow>=OW)? 0 :deltaY[n][oh][ow][oc];
                        
                        int fh = y + (CFH - 1 - fhr)*sh;
                        int fw = x + (CFW - 1 - fwr)*sw;
                        float w = (fh>=FH || fw>=FW) ? 0 : W[oc][fh][fw][ic];
                        dx += w * dy;
                    }
                    deltaX[n][ih][iw][ic] = dx;
                }
            }
        }
    }
    
    public static void deconv3D_deltaX_kernelSplit_v7(
            float[][][][] deltaY, int OH, int OW, //Y[N , OC, OH, OW] => C[GN, GM]
            float[][][][] W, int FH, int FW, //W[OC, IC, KH, KW] 
            float[][][][] deltaX, int IH, int IW, //X[N , IC, IH, IW] => A[GN, GK]
            int N, int IC, int OC,
            int sh, int sw, int ph, int pw)
    {
        float Nh0 = (FH / 2) * (1.0f / sh);
        float Nw0 = (FW / 2) * (1.0f / sw);
        
        boolean lh5 = (Nh0 - (int)Nh0) < 0.5f;
        boolean lw5 = (Nw0 - (int)Nw0) < 0.5f;
       
        int CFH = lh5 ? 2*(int)Nh0 + 1 : 2*(int)Nh0 + 2;
        int CFW = lw5 ? 2*(int)Nw0 + 1 : 2*(int)Nw0 + 2;
       
        int oph = CFH - 1, opw = CFW - 1;
        int IH_slice = (IH + sh - 1) / sh;
        int IW_slice = (IW + sw - 1) / sw;
        
        //GM(m, ih, iw)
        //IH_slice = (IH - ihs + sh - 1)/sh 
        //IH_slice_max = (IH - min(ihs) + sh - 1) / sh
        //IH_slice_max = (IH + sh - 1)/sh
        //GM_max = {(IH + sh - 1)/sh} * {(IW + sw - 1)/sw} * N
        int GM = IH_slice * IW_slice * N; 
        
        int ih_max = 0, fh_max = 0;
        
        for(int y=0; y<sh; y++)//Gz(y, x) = sh*sw
        for(int x=0; x<sw; x++)
        {
            for(int ic=0; ic<IC; ic++)//GN(ic) = IC
            {
                int ihs = (y - ph);  
                int iws = (x - pw);
                if(ihs < 0) ihs += (ph - y + sh - 1)/sh*sh;
                if(iws < 0) iws += (pw - x + sw - 1)/sw*sw;
                
                for(int j=0; j<GM; j++)
                {
                    int n = j/(IH_slice * IW_slice), jr = j%(IH_slice * IW_slice);
                    int ih = (jr / IW_slice)*sh + ihs;
                    int iw = (jr % IW_slice)*sw + iws;
                    
                    ih_max = Math.max(ih, ih_max);
                    
                    int ohs = (ih + ph - y)/sh - oph;
                    int ows = (iw + pw - x)/sw - opw;
                    
                    float dx = 0;
                    for(int fhr=0; fhr < CFH; fhr++)
                    for(int fwr=0; fwr < CFW; fwr++)
                    for(int oc=0; oc<OC; oc++) 
                    {
                        int oh = ohs + fhr;
                        int ow = ows + fwr;
                        boolean ldy = (oh>=0) && (oh<OH) && (ow>=0) && (ow<OW); 
                        float dy = (ldy? deltaY[n][oh][ow][oc] : 0);
                        
                        int fh = y + (CFH - 1 - fhr)*sh;
                        int fw = x + (CFW - 1 - fwr)*sw;
                        boolean lw = (fh<FH) && (fw<FW);
                        float w = (lw? W[oc][fh][fw][ic] : 0);
                        
                        dx += w * dy;
                        
                        fh_max = Math.max(fh, fh_max);
                    }
                    
                    if(ih < IH && iw < IW) deltaX[n][ih][iw][ic] = dx;
                }
            }
        }
        
        System.out.println(IH + ", " + ih_max + " : " + FH + ", " + (fh_max+1) + "\\" + sh);
    }
    
    public static void deconv3D_deltaX_kernelSplit_v8(
            float[][][][] deltaY, int OH, int OW, //Y[N , OC, OH, OW] => C[GN, GM]
            float[][][][] W, int FH, int FW, //W[OC, IC, KH, KW] 
            float[][][][] deltaX, int IH, int IW, //X[N , IC, IH, IW] => A[GN, GK]
            int N, int IC, int OC,
            int sh, int sw, int ph, int pw)
    {
        float Nh0 = (FH / 2) * (1.0f / sh);
        float Nw0 = (FW / 2) * (1.0f / sw);
        
        boolean lh5 = (Nh0 - (int)Nh0) < 0.5f;
        boolean lw5 = (Nw0 - (int)Nw0) < 0.5f;
       
        int CFH = lh5 ? 2*(int)Nh0 + 1 : 2*(int)Nh0 + 2;
        int CFW = lw5 ? 2*(int)Nw0 + 1 : 2*(int)Nw0 + 2;
        
//        System.out.println(FH + ", " + sh + ", " + CFH);
       
        int oph = CFH - 1, opw = CFW - 1;
        int IH_slice = (IH + sh - 1) / sh;
        int IW_slice = (IW + sw - 1) / sw;
        
        int GM = IH_slice * IW_slice * N; 
        int GK = CFH*CFW*OC;
        
        boolean flag = (IH % sh == 0) && (IW % sw == 0);
        
        for(int y=0; y<sh; y++)//Gz(y, x) = sh*sw
        for(int x=0; x<sw; x++)
        {
            for(int ic=0; ic<IC; ic++)//GN(ic) = IC
            {
                int ihs = (y - ph); if(ihs < 0) ihs += (ph - y + sh - 1)/sh*sh;
                int iws = (x - pw); if(iws < 0) iws += (pw - x + sw - 1)/sw*sw;
               
                for(int j=0; j<GM; j++)
                {
                    int n = j/(IH_slice * IW_slice), jr = j%(IH_slice * IW_slice);
                    int ih = (jr / IW_slice)*sh + ihs;
                    int iw = (jr % IW_slice)*sw + iws;
                    
                    int ohs = (ih + ph - y)/sh - oph;
                    int ows = (iw + pw - x)/sw - opw;
                    
                    //FH = 3, sh = 2
                    //ohs_max = (IH - 1 + ph)/2 - 2
                    //ohs_max = (32 - 1 + 1)/2 - 2 = 32/2-2 = 16 - 2 = 14
                    
                    float dx = 0;
                    for(int k=0; k<GK; k++)
                    {
                        int fhr = k / (CFW*OC), kr = k % (CFW*OC);
                        int fwr = kr / OC, oc = kr % OC;
                        
                        int oh = ohs + fhr;
                        int ow = ows + fwr;
                        boolean ldy = (oh>=0) && (oh<OH) && (ow>=0) && (ow<OW); 
                        float dy = (ldy? deltaY[n][oh][ow][oc] : 0);
                        
                        int fh = y + (CFH - 1 - fhr)*sh;
                        int fw = x + (CFW - 1 - fwr)*sw;
                        boolean lw = (fh<FH) && (fw<FW);
                        float w = (lw? W[oc][fh][fw][ic] : 0);
                        
                        dx += w * dy;
                    }
                  
                    if(ih < IH && iw < IW) deltaX[n][ih][iw][ic] = dx;
                }
            }
        }
    }
    
    public static void deconv3D_deltaX_kernelSplit_v9(
            float[][][][] deltaY, int OH, int OW, //Y[N , OC, OH, OW] => C[GN, GM]
            float[][][][] W, int FH, int FW, //W[OC, IC, KH, KW] 
            float[][][][] deltaX, int IH, int IW, //X[N , IC, IH, IW] => A[GN, GK]
            int N, int IC, int OC,
            int sh, int sw, int ph, int pw)
    {
        float Nh0 = (FH / 2) * (1.0f / sh);
        float Nw0 = (FW / 2) * (1.0f / sw);
        
        boolean lh5 = (Nh0 - (int)Nh0) < 0.5f;
        boolean lw5 = (Nw0 - (int)Nw0) < 0.5f;
       
        int CFH = lh5 ? 2*(int)Nh0 + 1 : 2*(int)Nh0 + 2;
        int CFW = lw5 ? 2*(int)Nw0 + 1 : 2*(int)Nw0 + 2;
        
//        System.out.println(FH + ", " + sh + ", " + CFH);
       
        int oph = CFH - 1, opw = CFW - 1;
        int IH_slice = (IH + sh - 1) / sh;
        int IW_slice = (IW + sw - 1) / sw;
        
        int GM = IH_slice * IW_slice * N; 
        int GK = CFH*CFW*OC;
        
        for(int y=0; y<sh; y++)//Gz(y, x) = sh*sw
        for(int x=0; x<sw; x++)
        {
            for(int ic=0; ic<IC; ic++)//GN(ic) = IC
            {
                int ihs = (y - ph); if(ihs < 0) ihs += (ph - y + sh - 1)/sh*sh;
                int iws = (x - pw); if(iws < 0) iws += (pw - x + sw - 1)/sw*sw;
               
                for(int j=0; j<GM; j++)
                {
                    int n = j/(IH_slice * IW_slice), jr = j%(IH_slice * IW_slice);
                    int ih = (jr / IW_slice)*sh + ihs;
                    int iw = (jr % IW_slice)*sw + iws;
                    
                    int ohs = (ih + ph - y)/sh - oph;
                    int ows = (iw + pw - x)/sw - opw;
                    
                    float dx = 0;
                    for(int k=0; k<GK; k++)
                    {
                        int fhr = k / (CFW*OC), kr = k % (CFW*OC);
                        int fwr = kr / OC, oc = kr % OC;
                        
                        int oh = ohs + fhr;
                        int ow = ows + fwr;
                        boolean ldy = (oh>=0) && (oh<OH) && (ow>=0) && (ow<OW); 
                        float dy = (ldy? deltaY[n][oh][ow][oc] : 0);
                        
                        int fh = y + (CFH - 1 - fhr)*sh;
                        int fw = x + (CFW - 1 - fwr)*sw;
                        
                        //fh_max = sh - 1 + (CFH - 1)*sh
                        //FHp = sh + (CFH - 1)*sh
                        
                        boolean lw = (fh<FH) && (fw<FW);
                        float w = (lw? W[oc][fh][fw][ic] : 0);
                        
                        dx += w * dy;
                    }
                    deltaX[n][ih][iw][ic] = dx;
                }
            }
        }
    }
    
    
    //较小的卷积核由W转化为CW时，会padding过多的0，影响计算效率
    //fh = y + (CFH - 1 - fhr)*sh < FH
    //we have: 
    //CFH - 1 < (FH - y)/sh + fhr
    //CFH <= (FH - y)/sh + fhr 
    //CFH <= min{ (FH - y)/sh + fhr }
    //CFH <= (FH - y)/sh
    //So: compress (CFH, CFW) -> (CFH', CFW')
    //we have: CFH' = (FH - y + sh - 1)/sh
    //we have: CFW' = (FW - x + sw - 1)/sw
    public static void deconv3D_deltaX_kernelSplit_v10(
            float[][][][] deltaY, int OH, int OW, //Y[N , OC, OH, OW] => C[GN, GM]
            float[][][][] W, int FH, int FW, //W[OC, IC, KH, KW] 
            float[][][][] deltaX, int IH, int IW, //X[N , IC, IH, IW] => A[GN, GK]
            int N, int IC, int OC,
            int sh, int sw, int ph, int pw)
    {
        int IH_slice = (IH + sh - 1) / sh;
        int IW_slice = (IW + sw - 1) / sw;
        int GM = IH_slice * IW_slice * N; 

        for(int y=0; y<sh; y++)
        for(int x=0; x<sw; x++)
        {
            int CFH = (FH - y + sh - 1) / sh, oph = CFH - 1;
            int CFW = (FW - x + sw - 1) / sw, opw = CFW - 1;
            int GK = CFH*CFW*OC;
            
            for(int ic=0; ic<IC; ic++)//GN(ic) = IC
            {
                int ihs = (y - ph); if(ihs < 0) ihs += (ph - y + sh - 1)/sh*sh; //make sure: ihs, iws >= 0
                int iws = (x - pw); if(iws < 0) iws += (pw - x + sw - 1)/sw*sw;
               
                for(int j=0; j<GM; j++)
                {
                    int n = j / (IH_slice * IW_slice), jr = j%(IH_slice * IW_slice);
                    int ih = (jr / IW_slice)*sh + ihs;
                    int iw = (jr % IW_slice)*sw + iws;
                    
                    int ohs = (ih + ph - y)/sh - oph;
                    int ows = (iw + pw - x)/sw - opw;
                    
                    float dx = 0;
                    for(int k=0; k<GK; k++)
                    {
                        int fhr = k / (CFW*OC), kr = k % (CFW*OC);
                        int fwr = kr / OC, oc = kr % OC;
                        
                        int oh = ohs + fhr;
                        int ow = ows + fwr;
                        boolean ldy = (oh>=0) && (oh<OH) && (ow>=0) && (ow<OW); 
                        float dy = (ldy? deltaY[n][oh][ow][oc] : 0);
                        
                        int fh = y + (CFH - 1 - fhr)*sh;
                        int fw = x + (CFW - 1 - fwr)*sw;
                        
                        boolean lw = (fh<FH) && (fw<FW);
                        float w = (lw? W[oc][fh][fw][ic] : 0);
                        dx += w * dy;
                    }
                    if(ih < IH && iw < IW) deltaX[n][ih][iw][ic] = dx;
                }
            }
        }
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="3D deconvolution deltaW">
     public static void  exchange_N_IC_axis(
            float[][][][] X_e, float[][][][] X, int N, int IH, int IW, int IC)
    {
        for(int n=0 ;n<N ;n++)
            for(int ih=0; ih<IH; ih++)
                for(int iw=0; iw<IW; iw++)
                    for(int ic=0; ic<IC; ic++)
                        X_e[ic][ih][iw][n] = X[n][ih][iw][ic];
    }
    
    public static float[][][][]  exchange_N_IC_axis(
            float[][][][] X, int N, int IH, int IW, int IC)
    {
        float[][][][] X_e = new float[IC][IH][IW][N];
        Tense.exchange_N_IC_axis(X_e, X, N, IH, IW, IC);
        return X_e;
    }
    
    public static float[][][][] innerPadding_Y_X_axis_exchange_N_OC_axis(
            float[][][][] deltaY, int N, int OH, int OW, int OC, int sh, int sw)
    {
        int OHp = OH + (OH-1)*(sh-1);
        int OWp = OW + (OW-1)*(sw-1);
        float[][][][] deltaY_pe = new float[N][OHp][OWp][OC];
        
        System.out.println(deltaY.length + ":" + deltaY[0].length + ":" + deltaY[0][0].length);
        
        for(int n=0; n<N; n++)
            for(int oh=0; oh<OH; oh++)
                for(int ow=0; ow<OW; ow++)
                    for(int oc=0; oc<OC; oc++){
                        deltaY_pe[n][oh * sh][ow * sw][oc] = deltaY[n][oh][ow][oc];
                    }
        return deltaY_pe;
    }
    
    public static void deconv3d_deltaW_naive(
        float[][][][] X, int IH, int IW,
        float[][][][] deltaY, int OH, int OW,
        float[][][][] deltaW, int FH, int FW,
        int N, int IC, int OC, 
        int sh, int sw, int ph, int pw)
    {
        int OHp = OH + (OH-1)*(sh-1);
        int OWp = OW + (OW-1)*(sw-1);
        int oph = ph, opw = pw;
        
        //create X_e------------------------------------------------------------
        float[][][][] Xe = new float[IC][IH][IW][N];
        Tense.exchange_N_IC_axis(Xe, X, N, IH, IW, IC);
        for(int n=0 ;n<N ;n++)
            for(int ih=0; ih<IH; ih++)
                for(int iw=0; iw<IW; iw++)
                    for(int ic=0; ic<IC; ic++)
                        Xe[ic][ih][iw][n] = X[n][ih][iw][ic];
        
        //create deltaYpe-------------------------------------------------------
        float[][][][] deltaYpe = new float[OC][OHp][OWp][N];
        for(int n=0; n<N; n++)
            for(int oh=0; oh<OH; oh++)
                for(int ow=0; ow<OW; ow++)
                    for(int oc=0; oc<OC; oc++)
                        deltaYpe[oc][oh * sh][ow * sw][n] = deltaY[n][oh][ow][oc];
        
        //create deltaWe--------------------------------------------------------
        float[][][][] deltaWe = new float[IC][FH][FW][OC];
        Tense.conv3D_naive(
                Xe, IH, IW,
                deltaYpe, OHp, OWp, 
                deltaWe, FH, FW,
                IC, N, OC,
                1, 1, oph, opw);
        
        for(int ic=0 ;ic<IC ;ic++)
            for(int fh=0; fh<FH; fh++)
                for(int fw=0; fw<FW; fw++)
                    for(int oc=0; oc<OC; oc++)
                        deltaW[oc][fh][fw][ic] = deltaWe[ic][fh][fw][oc];
    }
    
    public static void deconv3D_deltaW_img2col1(
        float[][][][] X, int IH, int IW,
        float[][][][] deltaY, int OH, int OW,
        float[][][][] deltaW, int FH, int FW,
        int N, int IC, int OC, 
        int sh, int sw, int ph, int pw)
    {
      int OH_p = OH + (OH-1)*(sh-1), OW_p = OW + (OW-1)*(sw-1);
        int oph = ph, opw = pw;
        
	int GN = OC;
        int GM = IC * FH * FW;
	int GK = N * OH_p * OW_p;

	for (int i = 0; i < GN; i++)
	{
            int oc = i;
            for (int j = 0; j < GM; j++)
            {
		int ic = j / (FH*FW);
		int j_res = j % (FH*FW);
		int fh = j_res / FW, fw = j_res % FW;
                
		float v = 0;
		for (int k = 0; k < GK; k++)
		{
                    int n = k / (OH_p*OW_p);
                    int k_res = k % (OH_p*OW_p);
                    int oh = k_res / OW_p, ow = k_res % OW_p;
                    if(oh%sh!=0 || ow%sw!=0) continue;
                     
                    int ih = fh - oph + oh;
                    int iw = fw - opw + ow;

                    if (ih < 0 || iw < 0 || ih >= IH || iw >= IW) continue;
                    v+=X[n][ic][ih][iw] * deltaY[n][oc][oh/sh][ow/sw];
		}
                deltaW[oc][ic][fh][fw]=v;
            }
        }
    }
    
       
    public static void deconv3D_deltaW_img2col2(
        float[][][][] X, int IH, int IW,
        float[][][][] deltaY, int OH, int OW,
        float[][][][] deltaW, int FH, int FW,
        int N, int IC, int OC, 
        int sh, int sw, int ph, int pw)
    {
	int GN = OC;
        int GM = IC * FH * FW;
	int GK = N * OH * OW;
        int oph = ph, opw = pw;
        
	for (int i = 0; i < GN; i++)
	{
            int oc = i;
            for (int j = 0; j < GM; j++)
            {
		int ic = j / (FH*FW);
		int j_res = j % (FH*FW);
		int fh = j_res / FW, fw = j_res % FW;
                
		float v = 0;
		for (int k = 0; k < GK; k++)
		{
                    int n = k / (OH*OW);
                    int k_res = k % (OH*OW);
                    int oh = k_res / OW, ow = k_res % OW;
                     
                    int ih = fh - oph + (oh*sh);
                    int iw = fw - opw + (ow*sw);

                    if (ih < 0 || iw < 0 || ih >= IH || iw >= IW) continue;
                    v += X[n][ih][iw][ic] * deltaY[n][oh][ow][oc];
		}
                deltaW[oc][fh][fw][ic] = v;
            }
        }
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="2D pooling">
    public static void pool2D_avg_naive_ignore_padding(
        float[][][][] X, int IH, int IW,
	int FH, int FW,
	float[][][][] Y, int OH, int OW,
	int N, int IC,
	int sh, int sw, int ph, int pw)
    {
	for (int n = 0; n < N; n++)
        for (int ic = 0; ic < IC; ic++)
	{
            for (int oh = 0; oh<OH; oh++)//oh < OH
            for (int ow = 0; ow<OW; ow++)//ow < OW
            {
                float v = 0; int count=0;//the padding part is not included
                for (int fh = 0; fh < FH; fh++)
                for (int fw = 0; fw < FW; fw++)
		{
                    int ih = oh*sh - ph + fh;
                    int iw = ow*sw - pw + fw;
                    
                    if (ih < 0 || iw < 0 || ih >= IH || iw >= IW) continue;
                    v += X[n][ih][iw][ic]; count++;
		}
                Y[n][oh][ow][ic] = v/count;
            }
        }
    }
    
      public static void pool2D_avg_naive(
        float[][][][] X, int IH, int IW,
	int FH, int FW,
	float[][][][] Y, int OH, int OW,
	int N, int IC,
	int sh, int sw, int ph, int pw)
    {
	for (int n = 0; n < N; n++)
        for (int ic = 0; ic < IC; ic++)
	{
            for (int oh = 0; oh<OH; oh++)//oh < OH
            for (int ow = 0; ow<OW; ow++)//ow < OW
            {
                float v = 0; 
                for (int fh = 0; fh < FH; fh++)
                for (int fw = 0; fw < FW; fw++)
		{
                    int ih = oh*sh - ph + fh;
                    int iw = ow*sw - pw + fw;
                    
                    if (ih < 0 || iw < 0 || ih >= IH || iw >= IW) continue;
                    v += X[n][ih][iw][ic]; 
		}
                Y[n][oh][ow][ic] = v / (FH * FW);
            }
        }
    }
    
    public static void pool2D_max_naive(
            float[][][][] X, int IH, int IW,
            int FH, int FW,
            float[][][][] Y, int OH, int OW,
            int N, int IC,
            int sh, int sw, int ph, int pw)
    {
	for (int n = 0; n < N; n++)
        for (int ic = 0; ic < IC; ic++)
	{
            for (int oh = 0; oh < OH; oh++)//oh < OH
            for (int ow = 0; ow < OW; ow++)//ow < OW
            {
                float v = - Float.MAX_VALUE;//the padding part is not included
                for (int fh = 0; fh < FH; fh++)
                for (int fw = 0; fw < FW; fw++)
		{
                    int ih = oh*sh - ph + fh;
                    int iw = ow*sw - pw + fw;
                    if (ih < 0 || iw < 0 || ih >= IH || iw >= IW) continue;
                    float x = X[n][ih][iw][ic];
                    if(v<x) v = x;
		}
                Y[n][oh][ow][ic] = v;
            }
        }
    }
      
    public static void pool2D_max_img2col(
	float[][][][] X, int IH, int IW, //X[N , IC, IH, IW] => A[GN, GK]
	int FH, int FW, //W[OC, IC, KH, KW] => B[GK, GM]
	float[][][][] Y, int OH, int OW, //Y[N , OC, OH, OW] => C[GN, GM]
	int N, int IC,
	int sh, int sw, int ph, int pw)
    {
        System.out.println("Pooling2D max img2col");
        
	int GN = IC;
        int GM = N * OH * OW;
	int GK = FH * FW;
	for (int i = 0; i < GN; i++)
	{
            int ic = i;
            for (int j = 0; j < GM; j++)
            {
		int n = j / (OH*OW);
		int j_res = j % (OH*OW);
		int oh = j_res / OW, ow = j_res % OW;
                
		float v = - Float.MAX_VALUE;
		for (int k = 0; k < GK; k++)
		{
                    int fh = k / FW, fw = k % FW;
                    int ih = oh * sh - ph + fh;
                    int iw = ow * sw - pw + fw;
                    if (ih < 0 || iw < 0 || ih >= IH || iw >= IW) continue;
                    float x = X[n][ih][iw][ic];
                    if(v < x) v = x;
		}
                Y[n][oh][ow][ic] = v;
            }
	}
    }
      
      
    
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="2D unpooing_average">
    public static void unpool2D_avgerage_naive(
        float[][][][] deltaY, int OH, int OW,
	int FH, int FW,
	float[][][][] deltaX, int IH, int IW,
	int N, int IC,
	int sh, int sw, int ph, int pw)
    {
        System.out.println("unPooling average naive");
        //System.out.format("(IH, IW, FH, FW, OH, OW) = (%d, %d, %d, %d, %d, %d)\n", IH, IW, FH, FW, OH, OW);
        //System.out.format("(N, IC) = (%d, %d, %d)\n", N, IC);
        //System.out.format("(ph, pw, sh, sw) = (%d, %d, %d, %d)\n", ph, pw, sh, sw);
        
        float[][][][] deltaY_p=Tense.innerPadding_Y_X_axis(deltaY, N, IC, OH, OW, sh, sw);
        int OH_p = OH + (OH-1)*(sh-1), OW_p = OW + (OW-1)*(sw-1);
        int oph = FH-ph-1, opw = FW-pw-1;
        
	for (int n = 0; n < N; n++)
        for (int ic = 0; ic < IC; ic++)
	{
            for(int ih=0;ih<IH;ih++)
            for(int iw=0;iw<IW;iw++)
            {
                float v=0;
                
                for(int fh=0;fh<FH;fh++)
                for(int fw=0;fw<FW;fw++)
                {
                    int oh = ih -oph +fh;
                    int ow = iw -opw +fw;
                    
                    if(oh<0 || ow<0 || oh>=OH_p || ow>=OW_p) continue;
                    if(deltaY_p[n][ic][oh][ow] == 0) continue;
                    
                    int ih_min = (oh/sh) * sh - ph, ih_max = ih_min + FH;
                    if(ih_min <0) ih_min = 0;
                    if(ih_max >= IH) ih_max = IH; 
                    
                    int iw_min = (ow/sw) * sw - pw, iw_max = iw_min + FW;
                    if(iw_min < 0) iw_min = 0;
                    if(iw_max >= IW) iw_max = IW ;
                    
                    int div = (ih_max - ih_min)*(iw_max - iw_min);
                    
                    v += deltaY_p[n][ic][oh][ow]/div;
                }
             
                deltaX[n][ic][ih][iw]=v;
            }
        }
    }
     
     //the prototype of GPU function
    public static void unpool2D_avgerage_img2col_plus2(
        float[][][][] deltaY, int OH, int OW,
	int FH, int FW,
	float[][][][] deltaX, int IH, int IW,
	int N, int IC,
	int sh, int sw, int ph, int pw)
    {
        int GN = IC;
        int GM = N*IH*IW;
        int OH_p = OH + (OH-1)*(sh-1), OW_p = OW + (OW-1)*(sw-1);
        int oph = FH-ph-1, opw = FW-pw-1;
        
        for(int i=0; i<GN; i++)
        {
            int ic = i;
            for(int j = 0;j < GM; j++)
            {
                int n = j/(IH*IW);
                int j_res = j%(IH*IW);
                int ih = j_res/IW, iw = j_res%IW;
                
                int fw_s = 0, fh_s = 0;
                loop:
                for(;fh_s<FH;fh_s++)
                {
                    int oh = ih - oph + fh_s;
                    if(oh<0 || oh>=OH_p || oh%sh!=0) continue;
                    for (fw_s = 0; fw_s < FW; fw_s++) 
                    {
                        int ow = iw - opw + fw_s;
                        if(ow >= 0 && ow < OW_p && ow % sw == 0) break loop;
                    }
                }
                
                int FH_r = (FH - fh_s + sh-1)/sh;//为什么加sh-1，最少执行一次，把发现的第一个执行掉
                int FW_r = (FW - fw_s + sw-1)/sw;
                //why:
                // if (FH-fh_s) % sh==0: then fh <= FH - sh
                // if (FH-fh_s) % sh!=0: then fh <= FH - b
                // 0=< b <sh
                int GK_r = FH_r*FW_r;
                
                float v=0;
                for(int k=0; k<GK_r; k++)
                {
                    int fh_r = k/FW_r, fw_r = k%FW_r;
                    int fh = fh_r*sh + fh_s;
                    int fw = fw_r*sw + fw_s;
                    
                    int oh = ih-oph+fh;
                    int ow = iw-opw+fw;
                    
                    if(oh>=OH_p || ow>=OW_p) continue;
                    //(1): oh = ih - oph + fh>= ih -oph + fh_s>=0
                    //(2): oh = ih - oph + fh = ih - (FH - ph -1) + fh
                    //     < ih - FH + ph + 1 + FH - b
                    //     < ih + ph + 1 - b
                    //充分条件: ih + ph + 1 - b < (OH-1)*(sh-1) + OH
                    //         ih + ph + 1 -b < OH*sh - OH -sh + 1 +OH
                    //         ih + ph -b < OH*sh -sh
                    //         ih + ph -b < ((IH + 2*ph - FH)/sh + 1)*sh - sh
                    //         ih + ph -b < IH +2*ph -FH 
                    //充分条件: IH + ph - b <=IH + 2*ph - FH
                    //          FH <= ph + b <= ph + sh, 所以需要保留
                    oh /= sh; ow /= sw;
                    v += deltaY[n][oh][ow][ic] / (FH * FW);
                }
                deltaX[n][ih][iw][ic] = v;
            }
        }
    }
    
    //the prototype of GPU function
    public static void unpool2D_avgerage_img2col_plus2_ignore_padding(
        float[][][][] deltaY, int OH, int OW,
	int FH, int FW,
	float[][][][] deltaX, int IH, int IW,
	int N, int IC,
	int sh, int sw, int ph, int pw)
    {
        int GN = IC;
        int GM = N*IH*IW;
        int OH_p = OH + (OH-1)*(sh-1), OW_p = OW + (OW-1)*(sw-1);
        int oph = FH-ph-1, opw = FW-pw-1;
        
        for(int i=0; i<GN; i++)
        {
            int ic = i;
            for(int j = 0;j < GM; j++)
            {
                int n = j/(IH*IW);
                int j_res = j%(IH*IW);
                int ih = j_res/IW, iw = j_res%IW;
                
                int fw_s = 0, fh_s = 0;
                loop:
                for(;fh_s<FH;fh_s++)
                {
                    int oh = ih - oph + fh_s;
                    if(oh<0 || oh>=OH_p || oh%sh!=0) continue;
                    for (fw_s = 0; fw_s < FW; fw_s++) 
                    {
                        int ow = iw - opw + fw_s;
                        if(ow >= 0 && ow < OW_p && ow % sw == 0) break loop;
                    }
                }
                
                int FH_r = (FH - fh_s + sh-1)/sh;//为什么加sh-1，最少执行一次，把发现的第一个执行掉
                int FW_r = (FW - fw_s + sw-1)/sw;
                //why:
                // if (FH-fh_s) % sh==0: then fh <= FH - sh
                // if (FH-fh_s) % sh!=0: then fh <= FH - b
                // 0=< b <sh
                int GK_r = FH_r*FW_r;
                
                float v=0;
                for(int k=0; k<GK_r; k++)
                {
                    int fh_r = k/FW_r, fw_r = k%FW_r;
                    int fh = fh_r*sh + fh_s;
                    int fw = fw_r*sw + fw_s;
                    
                    int oh = ih-oph+fh;
                    int ow = iw-opw+fw;
                    
                    if(oh>=OH_p || ow>=OW_p) continue;
                    //(1): oh = ih - oph + fh>= ih -oph + fh_s>=0
                    //(2): oh = ih - oph + fh = ih - (FH - ph -1) + fh
                    //     < ih - FH + ph + 1 + FH - b
                    //     < ih + ph + 1 - b
                    //充分条件: ih + ph + 1 - b < (OH-1)*(sh-1) + OH
                    //         ih + ph + 1 -b < OH*sh - OH -sh + 1 +OH
                    //         ih + ph -b < OH*sh -sh
                    //         ih + ph -b < ((IH + 2*ph - FH)/sh + 1)*sh - sh
                    //         ih + ph -b < IH +2*ph -FH 
                    //充分条件: IH + ph - b <=IH + 2*ph - FH
                    //          FH <= ph + b <= ph + sh, 所以需要保留
                    oh /= sh; ow /= sw;
                    
                    int ih_min = oh * sh - ph, ih_max = ih_min + FH;
                    if (ih_min < 0) ih_min = 0;
                    if (ih_max >= IH) ih_max = IH;

                    int iw_min = ow * sw - pw, iw_max = iw_min + FW;
                    if (iw_min < 0) iw_min = 0;
                    if (iw_max >= IW) iw_max = IW;

                    int div = (ih_max - ih_min)*(iw_max - iw_min);
                    
                    v += deltaY[n][oh][ow][ic] / div;
                }
                deltaX[n][ih][iw][ic] = v;
            }
        }
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="2D unpooling_max">
    public static void unpool2D_max_naive(
        float[][][][] deltaY, float[][][][] Y, int OH, int OW,
	int FH, int FW,
        float[][][][] deltaX, float[][][][] X, int IH, int IW,
	int N, int IC,
	int sh, int sw, int ph, int pw)
    {
        System.out.println("unPooling average Max");
        //System.out.format("(IH, IW, FH, FW, OH, OW) = (%d, %d, %d, %d, %d, %d)\n", IH, IW, FH, FW, OH, OW);
        //System.out.format("(N, IC) = (%d, %d, %d)\n", N, IC);
        //System.out.format("(ph, pw, sh, sw) = (%d, %d, %d, %d)\n", ph, pw, sh, sw);
        
        float[][][][] deltaY_p=Tense.innerPadding_Y_X_axis(deltaY, N, IC, OH, OW, sh, sw);
        int OH_p = OH + (OH-1)*(sh-1), OW_p = OW + (OW-1)*(sw-1);
        int oph = FH-ph-1, opw = FW-pw-1;
        
	for (int n = 0; n < N; n++)
        for (int ic = 0; ic < IC; ic++)
	{
            for(int ih=0;ih<IH;ih++)
            for(int iw=0;iw<IW;iw++)
            {
                float x=X[n][ic][ih][iw];
                
                float v=0;
                for(int fh=0;fh<FH;fh++)
                for(int fw=0;fw<FW;fw++)
                {
                    int oh = ih -oph +fh;
                    int ow = iw -opw +fw;
                    
                    if(oh<0 || ow<0 || oh>=OH_p || ow>=OW_p) continue;
                    if(oh%sh!=0 || ow%sw!=0) continue;
                    
                    float y=Y[n][ic][oh/sh][ow/sw];
                    if(y>x) continue;
                    v += deltaY_p[n][ic][oh][ow];
                }
                deltaX[n][ic][ih][iw]=v;
            }
        }
    }
    
    public static void unpool2D_max_img2col_plus2(
        float[][][][] deltaY, float[][][][] Y, int OH, int OW,
	int FH, int FW,
        float[][][][] deltaX, float[][][][] X, int IH, int IW,
	int N, int IC,
	int sh, int sw, int ph, int pw)
    {
        int GN = IC;
        int GM = N*IH*IW;
        int OHp = OH + (OH-1)*(sh-1), OWp = OW + (OW-1)*(sw-1);
        int oph = FH-ph-1, opw = FW-pw-1;
        
        for(int i=0;i<GN;i++)
        {
            int ic=i;
            for(int j=0;j<GM;j++)
            {
                int n = j/(IH*IW);
                int j_res = j%(IH*IW);
                final int ih = j_res/IW, iw = j_res%IW;
                float x = X[n][ih][iw][ic];
                
                //--------------------------------------------------------------
                int fw_s = 0, fh_s = 0;
                loop:
                for(;fh_s<FH;fh_s++)
                {
                    int oh = ih - oph + fh_s;
                    if(oh<0 || oh>=OHp || oh%sh!=0) continue;
                    for (fw_s = 0; fw_s < FW; fw_s++) 
                    {
                        int ow = iw - opw + fw_s;
                        if(ow >= 0 && ow < OWp && ow % sw == 0) break loop;
                    }
                }
                int FH_r = (FH - fh_s + sh-1)/sh;
                int FW_r = (FW - fw_s + sw-1)/sw;
                int GK_r = FH_r*FW_r;
                //--------------------------------------------------------------
                
                float v=0;
                for(int k=0; k<GK_r; k++)
                {
                    int fh_r = k/FW_r, fw_r = k%FW_r;
                    int fh = fh_r*sh + fh_s, fw = fw_r*sw + fw_s;
                    
                    int oh = ih-oph+fh, ow = iw-opw+fw;
                    if(oh>=OHp || ow>=OWp) continue;
                    
                    oh/=sh; ow/=sw;
                    float y = Y[n][oh][ow][ic];
                    if(y > x) continue;
                    v += deltaY[n][oh][ow][ic];
                }
                
                deltaX[n][ih][iw][ic] = v;
            }
        }
    }
    //</editor-fold>
}
