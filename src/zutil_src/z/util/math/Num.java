/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.math;

import z.util.ds.linear.ZArrayList;
import z.util.lang.SimpleTimer;
import z.util.lang.annotation.Passed;
import z.util.lang.exception.IAE;
import z.util.math.vector.Vector;
import z.util.math.vector.Vector.MaxMin;

/**
 *
 * @author dell
 */
public final class Num
{
    public static final int INT_MAX_POWER_3=1162261467;
    public static final int INT_MAX_POWER_5=1220703125;
    public static final int INT_MAX_POWER_7=282475249;
    
    public static final long LONG_MAX_POWER_3=4052555153018976267L;
    public static final long LONG_MAX_POWER_5=7450580596923828125L;
    public static final long LONG_MAX_POWER_7=3909821048582988049L;
    
    public static final double LOG_2=Math.log(2);
    
    public static void main(String[] args)
    {
        for(int i=1; i < (1 << 30); i <<= 1)
        {
            int y = log2(i);
            System.out.println(i + " : " + y);
        }
    }
    
    public static int log2(int n) //n >= 0
    {
        int result = 0;  
        if((n & 0xffff0000) != 0) {result += 16; n >>= 16; }  
        if((n & 0x0000ff00) != 0) {result += 8; n >>= 8; }  
        if((n & 0x000000f0) != 0) {result += 4; n >>= 4; }  
        if((n & 0x0000000c) != 0) {result += 2; n >>= 2; }  
        if((n & 0x00000002) != 0) {result += 1; n >>= 1; }  
        return result; 
    }
    
    //<editor-fold defaultstate="collapsed" desc="Number-Theory">
    @Passed
    public static int oddNumber(int low, int high) {
        if(low < high) { int t = low; low=high;high=t; }
        return (high + 1)>> 1 - (low) >>1;
    }
    public static int evenNumber(int low, int high)
    {
        if(low<high) {int t=low;low=high;high=t;}
        return (high+1)>>1-(low)>>1;
    }
    //<editor-fold defaultstate="collapsed" desc="IsPowerOf">
    @Passed
    public static boolean isPowerOf2(int x)
    {
        if(x<=0 || (x&(x-1))!=0) return false;
        return (x & 0xffffffff)!=0;
    }
    public static boolean isPowerOf2(int... arr)
    {
        for(int x : arr) 
            if(x<=0 || (x&(x-1))!=0 || (x&0xffffffff)==0) return false;
        return true;
    }
    
    @Passed
    public static boolean isPowerOf2(long x)
    {
        if(x<=0||(x&(x-1))!=0) return false;
        return (x & 0xffffffffffffffffL)!=0;
    }
    @Passed
    public static boolean isPowerOf4(int x)
    {
        if(x<=0||(x&(x-1))!=0) return false;//check if there is only one 1 on bits of x
        return (x & 0x55555555)!=0;
    }
    @Passed
    public static boolean isPowerOf4(long x)
    {
        if(x<=0||(x&(x-1))!=0) return false;//check if there is only one 1 on bits of x
        return (x & 0x5555555555555555L)!=0;
    }
    @Passed
    public static boolean isPowerOf8(int x)
    {
        if(x<=0||(x&(x-1))!=0) return false;
        return (x & 0x49249249)!=0;
    }
    @Passed
    public static boolean isPowerOf8(long x)
    {
        if(x<=0||(x&(x-1))!=0) return false;
        return (x & 0x9249249249249249L)!=0;
    }
    @Passed
    public static boolean isPowerOf16(int x)
    {
        if(x<=0||((x&(x-1))!=0)) return false;
        return (x & 0x11111111)!=0;
    }
    @Passed
    public static boolean isPowerOf16(long x)
    {
        if(x<=0||((x&(x-1))!=0)) return false;
        return (x & 0x1111111111111111L)!=0;
    }
    /**
     * check whether x is the power of 3.
     * The prime factor to a power of 3 is only 3 so the factor of 'power of 3'
     * must be 'the power of 3' It is also suitable for the power of other
     * 'prime factors'.
     * @param x
     * @return
     */
    @Passed
    public static boolean isPowerOf3(int x)
    {
        return x>0 && INT_MAX_POWER_3%x==0;
    }
    @Passed
    public static boolean isPowerOf3(long x)
    {
        return x>0 && LONG_MAX_POWER_3%x==0;
    }
    @Passed
    public static boolean isPowerOf5(int x)
    {
        return x>0 && INT_MAX_POWER_5%x==0;
    }
    @Passed
    public static boolean isPowerOf5(long x)
    {
        return x>0 && LONG_MAX_POWER_5%x==0;
    }
    @Passed
    public static boolean isPowerOf7(int x)
    {
        return x>0 && INT_MAX_POWER_7%x==0;
    }
    @Passed
    public static boolean isPowerOf7(long x)
    {
        return x>0 && LONG_MAX_POWER_7%x==0;
    }
    //</editor-fold>
    @Passed
    public static byte[] base2(long num, int len)
    {
        if(num==0) return new byte[len];
        if(num<0) num=-num;
        byte[] r=new byte[len];
        for(int i=len-1;num>0;i--,num>>=1) r[i--]=(byte) (num&1);
        return r;
    }
    @Passed static byte[] baseX(long num, int x, int len)
    {
        if(num==0) return new byte[len];
        if(num<0) num=-num;
        byte[] r=new byte[len];
        for(int i=len-1;num>0;i--,num/=x) r[i--]=(byte) (num%x);
        return r;
    }
    @Passed
    public static String toStringBaseX(int num, int base)
    {
        if(base<=0) throw new IAE("Base must be positive");
        if(base==2) return toStringBase2(num);
        else if(base==16) return toStringBase16(num);
        
        if(num==0) return "0";
        StringBuilder sb=new StringBuilder();
        if(num<0) {sb.append('-');num=-num;}
        int len=(int)(Math.log(num)/Math.log(base))+1;
        char[] a=new char[len];
        int index=0;
        while(num>0)
        {
            a[index++]=(char) (num%base);
            num/=base;
        }
        for(int i=index-1;i>=0;i--) sb.append((char)('0'+a[i]));
        return sb.toString();
    }
    @Passed
    public static String toStringBase2(int num)
    {
        if(num==0) return "0";
        StringBuilder sb=new StringBuilder();
        if(num<0) {sb.append('-');num=-num;}
        int len=(int)(Math.log(num)/LOG_2)+1;
        char[] a=new char[len];
        int index=0;
        while(num>0)
        {
            a[index++]=(char) (num&1);
            num>>=1;
        }
        for(int i=index-1;i>=0;i--) sb.append((char)('0'+a[i]));
        return sb.toString();
    }
    @Passed
    public static String toStringBase16(int num)
    {
        if(num==0) return "0";
        StringBuilder sb=new StringBuilder();
        if(num<0) {sb.append("-0x");num=-num;}
        else sb.append("0x");
        int len=(int)(Math.log(num)/LOG_2)+1;
        char[] a=new char[len];
        int index=0;
        while(num>0)
        {
            a[index++]=(char) (num&15);
            num>>=4;
        }
        for(int i=index-1;i>=0;i--)
        {
            if(a[i]<10) sb.append((char)(a[i]+'0'));
            else sb.append((char)('a'+(char)(a[i]-10)));
        }
        return sb.toString();
    }
    @Passed
    public static int gcd(int a, int b)
    {
        if(a==0||b==0) return -1;
        for(int c=a%b;c!=0;a=b, b=c,c=a%b);
        return b;
    }
    @Passed
    public static int gcd(int[] a, int low, int high)
    {
        if(low==high) return a[low];
        if(high-low==1) return gcd(a[low], a[low+1]);
        
        MaxMin<Integer> mm=Vector.maxMinABSIndex(a, low, high);
        int maxIndex=mm.getMax(), minIndex=mm.getMin(); 
        int gcd=gcd(a[maxIndex], a[minIndex]);
        
        //exchange a[maxIndex], a[minIndex] with a[low] and a[low+1]
        int t=a[low];a[low]=a[maxIndex];a[maxIndex]=t;
        t=a[low+1];a[low+1]=a[minIndex];a[minIndex]=t;
        
        //Time Complexity: O((max*min)^0.5 + sum(gcd*a[i])*0,5)
        for(int i=low+2;i<=high;i++) gcd=gcd(gcd, a[i]);// 
        
        //exchange a[maxIndex], a[minIndex] with a[low] and a[low+1]
        t=a[low];a[low]=a[maxIndex];a[maxIndex]=t;
        t=a[low+1];a[low+1]=a[minIndex];a[minIndex]=t;
        
        return gcd;
    }
    public static int gcd(int[] a)
    {
        return gcd(a, 0, a.length-1);
    }
    @Passed
    public static int[] primeTable(int n)
    {
        int[] prime=new int[n+1];
        prime[0]=prime[1]=1;
        for(int i=2,j,end=(int) Math.sqrt(n);i<=end;i++)
            if(prime[i]==0)
            for(j=i*i;j<=n;j+=i) prime[j]=1;
        return prime;
    }
    @Passed
    public static int[] primeFactors(int x)
    {
        if(x<=1) return null;
        ZArrayList<Integer> arr=new ZArrayList<>();
        for(int i=2, end=(int) Math.sqrt(x);i<=end;i++)
        if(x%i==0)
        {
            while(x%i==0) x/=i;
            arr.add(i);
        }
        arr.add(x);
        int[] r=new int[arr.number()];
        int index=0;
        for(int v:arr) r[index++]=v;
        arr.clear();
        return r;
    }
    /**
     * return x^y mod p.
     * @param x the base 
     * @param y the index
     * @param p the mod
     * @return 
     */
    @Passed
    public static long quickPow(long x, long y, long p)
    {
        x%=p;
        long r=1;
        while(y>0)
        {
            if((y&1)==1) r=(r*x)%p;
            x=(x*x)%p;
            y>>=1;
        }
        return r;
    }
    @Passed
    public static long quickPow(int x, int y)
    {
        long r=1, nx=x;
        while(y>0)
        {
            if((y&1)==1) r*=nx;
            nx*=nx;
            y>>=1;
        }
        return r;
    }
    //</editor-fold>
    
    public static int MAX_INTEGER_TWO_POWERS=1<<30;
    public static long MAX_LONG_TWO_POWERS=1L<<60;
    
    //<editor-fold defaultstate="collapsed" desc="BitWise-Function">
    /**
     * return the value greater than or equal to {@code int n}, which
     * is a specific power of 2.
     * Pay attention: 
     * (1)if n is less than 0, then return 1.
     * (2)if n is greater than {@code MAX_INTEGER_TWO_POWERS}, then return
     * {@code MAX_INTEGER_TWO_POWERS}.
     * For example:
     * (1)f(-1)=1 (2)f(1)=2 (3)f(127)=128 (4)f(0)=1
     * @param n
     * @return 
     */
    @Passed
    public static int approximate2PowerSmaller(int n)//need to think the principle
    {
        n--;
        n |= n >>> 1;
        n |= n >>> 2;
        n |= n >>> 4;
        n |= n >>> 8;
        n |= n >>> 16;
        if(n<0) return 1;
        return (n >=MAX_INTEGER_TWO_POWERS ? MAX_INTEGER_TWO_POWERS:++n);
    }
    @Passed
    public static int approximate2PowerGreater(int n)//need to think the principle
    {
        n--;
        n |= n >> 1;
        n |= n >> 2;
        n |= n >> 4;
        n |= n >> 8;
        n |= n >> 16;
        if(n<0) return 1;
        return (n >=MAX_INTEGER_TWO_POWERS ? MAX_INTEGER_TWO_POWERS:++n);
    }
    /**
     * return the low 32bit of a Long Integer.
     * @param x
     * @return 
     */
    @Passed
    public static int low32(long x)
    {
        return (int) (x&0x000000ffffffffL);
    }
    /**
     * return the high 32bit of a Long Integer;
     * @param x
     * @return 
     */
    @Passed
    public static int high32(long x)
    {
        return (int) (x>>32);
    }
    public static void swap(int x, int y)
    {
        x^=y;y^=x;x^=y;//逐个2二进制位的思考n
    }
    /**
     * compine two Integer to one Long.
     * @param low regard as the low 32 bit of long
     * @param high regard as the high 32 bit of long
     * @return 
     */
    @Passed
    public static long toLong(int low, int high)
    {
        return ((long)low & 0xFFFFFFFFl) | (((long)high << 32) & 0xFFFFFFFF00000000l);
    }
    /**
     * exchange the value of even bits and odd bits;
     * In detail, change the value of bit[2n] and bit[2n-1].
     * @param num
     * @return 
     */
    @Passed
    public static int swapEvenOdd(int num)
    {
        return ((num & 0x55555555) << 1) | ((num & 0xaaaaaaaa) >> 1);
    }
    /**
     * @see ExMath#exchangeBits(int). 
     * @param num
     * @return 
     */
    @Passed
    public static long swapEvenOdd(long num)
    {
         return ((num&0x5555555555555555L)<<1 | (num & 0xaaaaaaaaaaaaaaaaL) >> 1);
    }
    @Passed
    public static int reverse(int n) 
    {
        n=(n >>> 16)|(n << 16);//exchange the first and last 16 bit
        n=((n & 0xff00ff00) >>>  8) | ((n & 0x00ff00ff) << 8);//8 bit
        n=((n & 0xf0f0f0f0) >>>  4) | ((n & 0x0f0f0f0f) << 4);//4 bit
        n=((n & 0xcccccccc) >>>  2) | ((n & 0x33333333) << 2);//2 bit
        n=((n & 0xaaaaaaaa) >>>  1) | ((n & 0x55555555) << 1);// 1 bit
        return n;
    }
    @Passed
    public static long reverse(long n) 
    {
        n=(n >>> 32)| (n<<32);//exchange the first and last 16 bits
        n=((n&0xffff0000ffff0000L) >>> 16| (n&0x0000ffff0000ffffL) <<16);//16bits
        n=((n&0xff00ff00ff00ff00L) >>> 8 | (n&0x00ff00ff00ff00ffL) <<8 );//8bits
        n=((n&0xf0f0f0f0f0f0f0f0L) >>> 4 | (n&0x0f0f0f0f0f0f0f0fL) <<4 );//4bits
        n=((n&0xccccccccccccccccL) >>> 2 | (n&0xccccccccccccccccL) <<2 );//2bits
        n=((n&0xaaaaaaaaaaaaaaaaL) >>> 1 | (n&0xaaaaaaaaaaaaaaaaL) <<1 );//1bits
        return n;
    }
    /**
     * return a+b, From the point of view of the computer adder.
     * @param a
     * @param b
     * @return 
     */
    @Passed
    public static int add(int a, int b)
    {
        int carry;
        while(b!=0)
        {
            carry=(a&b)<<1;
            a=a^b;//if not zero, must 1, so only the different can remain
            b=carry;//carry
        }
        return a;
    }
    /**
     * <pre>
     * From low to high set bits of x to 0.
     * As low=5 ,high=8:
     * {@code
     * (1)x = 1<<(high-low+1) = 1<<4 = 10000
     * (2)x-1 = 01111
     * (3)x<<low = x<<5 =0111100000
     * (4)~x = 1000011111}
     * </pre>
     * @param x
     * @param low
     * @param high
     * @return 
     */
    @Passed
    public static int setBitsZero(int x, int low, int high)
    {
        return  x& ~(((1<<(high-low+1))-1) << low);
    }
    @Passed
    public static long setBitsZero(long x, int low, int high)
    {
        return  x& ~(((1<<(high-low+1))-1) << low);
    }
    /**
     * <pre>
     * From low to hig set bits of x to 0, then insert y to the segment,
     * (align it's tail to the low, the extra parts are filled with 0).
     * {@link #setBitsZero(int, int, int)}
     * </pre>
     * @param x
     * @param y
     * @param low
     * @param high
     * @return 
     */
    @Passed
    public static int insertBits(int x, int y, int low, int high)
    {
        return  (x& ~(((1<<(high-low+1))-1) << low)) |(y << low);
    }
    @Passed
    public static long insertBits(long x, long y, int low, int high)
    {
        return  (x& ~(((1<<(high-low+1))-1) << low)) |(y << low);
    }
    //</editor-fold>
}
