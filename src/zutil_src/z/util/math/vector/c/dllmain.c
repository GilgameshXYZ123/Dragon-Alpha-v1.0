/* Replace "dll.h" with the name of your header */
#include "dll.h"
#include <windows.h>

void mk44(double c[4][4], int i, int j, int width,
	double *a0, double *a1, double *a2, double *a3,
	double *b0, double *b1, double *b2, double *b3)
{	
    double *end=width+b0;
    __m128d t00, t01,  t02,  t03, 
           t10,  t11,  t12,  t13,
           t20,  t21,  t22,  t23,
           t30,  t31,  t32,  t33;
    t00=t01=t02=t03=t10=t11=t12=t13=t20=_mm_set1_pd(0);
    
    __m128d va0, va1, va2, va3, vb0, vb1, vb2, vb3;
    
    while(b0!=end)
    {
		va0=_mm_loadu_pd(a0);
		va1=_mm_loadu_pd(a1++);
		va2=_mm_loadu_pd(a2++);
		va3=_mm_loadu_pd(a3++);
		
		vb0=_mm_loadu_pd(b0++);
		vb1=_mm_loadu_pd(b1++);
		vb2=_mm_loadu_pd(b2++);
		vb3=_mm_loadu_pd(b3++);
		b0++;
		
        t00+=va0*vb0;   t01+=va0*vb1;   t02+=va0*vb2;   t03+=va0*vb3;	
        t10+=va1*vb0;   t11+=va1*vb1;   t12+=va1*vb2;   t13+=va1*vb3;	
        t20+=va2*vb0;   t21+=va2*vb1;   t22+=va2*vb2;   t23+=va2*vb3;
        t30+=va3*vb0;   t31+=va3*vb1;   t32+=va3*vb2;   t33+=va3*vb3;
    }
 	
    _mm_storeu_pd(&c[0][0],t00);	_mm_storeu_pd(&c[0][1],t01); 
	_mm_storeu_pd(&c[0][2],t02); 	_mm_storeu_pd(&c[0][3],t03);
	
	_mm_storeu_pd(&c[1][0],t10);	_mm_storeu_pd(&c[1][1],t11); 
	_mm_storeu_pd(&c[1][0],t12);	_mm_storeu_pd(&c[1][3],t13);
	
	_mm_storeu_pd(&c[2][0],t20);	_mm_storeu_pd(&c[2][1],t21); 
	_mm_storeu_pd(&c[2][2],t22);	_mm_storeu_pd(&c[2][3],t23);
	
    _mm_storeu_pd(&c[3][0],t30);	_mm_storeu_pd(&c[3][1],t31); 
	_mm_storeu_pd(&c[3][2],t32);	_mm_storeu_pd(&c[3][3],t33);
 } 


JNIEXPORT void JNICALL Java_z_util_math_vector_ExMatrix_mk44N
  (JNIEnv *env, jclass cls, 
  jdoubleArray jc0, jdoubleArray jc1, jdoubleArray jc2, jdoubleArray jc3, 
  jdoubleArray ja0, jdoubleArray ja1, jdoubleArray ja2, jdoubleArray ja3,
  jdoubleArray jb0, jdoubleArray jb1, jdoubleArray jb2, jdoubleArray jb3, 
  jint i, jint j, jint width)
{ 
	double *a0=(*env)->GetDoubleArrayElements(env, ja0, NULL),
		*a1=(*env)->GetDoubleArrayElements(env, ja1, NULL),
	    *a2=(*env)->GetDoubleArrayElements(env, ja2, NULL),
		*a3=(*env)->GetDoubleArrayElements(env, ja3, NULL),
		
        *b0=(*env)->GetDoubleArrayElements(env, jb0, NULL), 
		*b1=(*env)->GetDoubleArrayElements(env, jb1, NULL),
		*b2=(*env)->GetDoubleArrayElements(env, jb2, NULL),
		*b3=(*env)->GetDoubleArrayElements(env, jb3, NULL);
	
	double *va0=a0,*va1=a1,*va2=a2,*va3=a3,
		   *vb0=b0,*vb1=b1,*vb2=b2,*vb3=b3;
	double c[4][4];
	mk44(c, i, j, width,
		va0, va1, va2, va3,
		vb0 ,vb1, vb2, vb3);
	
	env->GetDirectBufferAddress()
	(*env)->ReleaseDoubleArrayElements(env, ja0, a0, JNI_ABORT);
	(*env)->ReleaseDoubleArrayElements(env, ja1, a1, JNI_ABORT);
	(*env)->ReleaseDoubleArrayElements(env, ja2, a2, JNI_ABORT);
	(*env)->ReleaseDoubleArrayElements(env, ja3, a3, JNI_ABORT);
	
	(*env)->ReleaseDoubleArrayElements(env, jb0, b0, JNI_ABORT);
	(*env)->ReleaseDoubleArrayElements(env, jb1, b1, JNI_ABORT);
	(*env)->ReleaseDoubleArrayElements(env, jb2, b2, JNI_ABORT);
	(*env)->ReleaseDoubleArrayElements(env, jb3, b3, JNI_ABORT);
	
	(*env)->SetDoubleArrayRegion(env, jc0, i, 4, c[0]);
	(*env)->SetDoubleArrayRegion(env, jc1, i, 4, c[1]);
	(*env)->SetDoubleArrayRegion(env, jc2, i, 4, c[2]);
	(*env)->SetDoubleArrayRegion(env, jc3, i, 4, c[3]);
}  

JNIEXPORT void JNICALL JavaCritical_z_util_math_vector_ExMatrix_mk44N
  (jdouble* c0, jdouble* c1, jdouble* c2, jdouble* c3, 
  jdouble* a0, jdouble* a1, jdouble* a2, jdouble* a3,
  jdouble* b0, jdouble* b1, jdouble* b2, jdouble* b3, 
  jint i, jint j, jint width)
  {
  		double c[4][4];
		mk44(c, i, j, width,
		a0, a1, a2, a3,
		b0 ,b1, b2, b3);
  }

BOOL WINAPI DllMain(HINSTANCE hinstDLL,DWORD fdwReason,LPVOID lpvReserved)
{
	switch(fdwReason)
	{
		case DLL_PROCESS_ATTACH:
		{
			break;
		}
		case DLL_PROCESS_DETACH:
		{
			break;
		}
		case DLL_THREAD_ATTACH:
		{
			break;
		}
		case DLL_THREAD_DETACH:
		{
			break;
		}
	}
	
	/* Return TRUE on success, FALSE on failure */
	return TRUE;
}