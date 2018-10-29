//parallel program
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <algorithm>
#include <curand_kernel.h>
#include <stdio.h>
#define K 401296
#define max(a, b)  (((a) > (b)) ? (a) : (b))
#define min(a, b)  (((a) < (b)) ? (a) : (b))
#define NT 4
#define PS  20
#define PS2 10
#define Y 365
#define EP 15
#define CHECK(res) if(res!=cudaSuccess){printf("Error:%d\n", __LINE__);exit(-1);}

__device__ double LIMIT(double min, double max, double X)
{
	if (min>X)
		return min;
	if (max<X)
		return max;
	return X;
}

__device__ double AFGEN(double *x, int n, double X)
{
	double Y1 = 0;
	if (X <= x[0])
		Y1 = x[1];
	if (X >= x[n - 2])
		Y1 = x[n - 1];
	for (int i = 2; i<n - 1;)
	{
		if ((x[i] >= X) && (X >= x[i - 2]))
		{
			double slope = (x[i + 1] - x[i - 1]) / (x[i] - x[i - 2]);
			Y1 = x[i - 1] + (X - x[i - 2])*slope;
		}
		i += 2;
	}
	return Y1;
}

__device__ double sum(double *f, int k)
{
	double sum = 0;
	for (int i = 0; i<k; i++)
	{
		if (f[i]>0)
			sum = sum + f[i];
	}
	return sum;
}

__device__ double max2(double *f, int k)
{
	double max2 = 0;
	for (int i = 0; i<k - 1; i++)
	{
		if (f[i + 1]>f[i])
			max2 = f[i + 1];
	}
	return max2;
}

//WOFOST model
__device__ void LAIcal(double SPAN, double *param, double *TMIN, double *TMAX, double *AVRAD, double *JCLAImoni)
{
	double IDEM = param[0];	
	int IDAY = 0;       
	int DELT = 1;	
	double DTSMTB[] = { 0.00, 0.00, 10.00, 0.00, 30.00, 20.00, 40.00, 30.00 };
	int ILDTSM = sizeof(DTSMTB) / sizeof(double);
	double TSUM1 = 1800;         
	double TSUM2 = 620;          
	double DVSI = 0.250;         
	double DVSEND = 2.00;          	
	double TDWI = 65.00;        
	double RGRLAI = 0.0070;      
	double SLATB[] = { 0.00, 0.0045, 0.16, 0.0033, 0.61, 0.0030, 0.80, 0.0029, 1.00, 0.0025, 1.55, 0.0024, 2.02, 0.0018 };
	int ILSLA = sizeof(SLATB) / sizeof(double);	
	double LAIEM = 0.10;          	
	double TBASE = 15.0;          	
	double KDIFTB[] = { 0.00, 0.40, 0.65, 0.40, 1.00, 0.60, 2.00, 0.60 };
	int ILKDIF = sizeof(KDIFTB) / sizeof(double);	
	double EFFTB[] = { 10, 0.54, 40, 0.36 };
	int ILEFF = sizeof(EFFTB) / sizeof(double);	
	double AMAXTB[] = { 0.00, 40.00, 1.00, 40, 1.90, 40, 2.00, 40.00 };
	int ILAMAX = sizeof(AMAXTB) / sizeof(double);	
	double TMPFTB[] = { 0.00, 0.00, 12.00, 0.69, 18.00, 0.85, 24.00, 1.00, 30.00, 1.00, 36.00, 0.87, 42.00, 0.27 };
	int ILTMPF = sizeof(TMPFTB) / sizeof(double);
	double TMNFTB[] = { 0.00, 0.00, 3.00, 1.00 };
	int ILTMNF = sizeof(TMNFTB) / sizeof(double);
	double CVL = 0.754;        
	double CVO = 0.684;       
	double CVR = 0.754;       
	double CVS = 0.754;       
	double Q10 = 2.0;     
	double RML = 0.0200; 
	double RMO = 0.0030; 
	double RMR = 0.0100; 
	double RMS = 0.0150; 
	double RFSETB[] = { 0.00, 1.00, 2.00, 1.00 };
	int ILRFSE = sizeof(RFSETB) / sizeof(double);
	double FLTB[] = { 0.00, 0.65, 0.31, 0.60, 0.53, 0.57, 0.80, 0.35, 0.94, 0.14, 1.00, 0.10, 1.2, 0.00, 2.10, 0.00 };
	int ILFL = sizeof(FLTB) / sizeof(double);
	double FRTB[] = { 0.00, 0.50, 0.43, 0.45, 0.65, 0.40, 0.80, 0.37, 0.85, 0.27, 0.99, 0.10, 1.00, 0.00, 2.00, 0.00 };
	int ILFR = sizeof(FRTB) / sizeof(double);
	double FSTB[] = { 0.00, 0.35, 0.31, 0.40, 0.53, 0.43, 0.80, 0.637, 0.94, 0.553, 1.00, 0.10, 1.20, 0.00, 2.10, 0.00 };
	int ILFS = sizeof(FSTB) / sizeof(double);
	double FOTB[] = { 0.00, 0.00, 0.50, 0.00, 0.80, 0.013, 0.94, 0.316, 1.00, 0.80, 1.20, 1.000, 1.50, 1.000, 2.00, 1.00 };
	int ILFO = sizeof(FOTB) / sizeof(double);
    double RDRRTB[] = { 0.00, 0.000, 1.50, 0.000, 1.5001, 0.020, 2.00, 0.020 };
	int ILRDRR = sizeof(RDRRTB) / sizeof(double);
	double RDRSTB[] = { 0.00, 0.000, 1.50, 0.000, 1.5001, 0.020, 2.00, 0.020 };
	int ILRDRS = sizeof(RDRSTB) / sizeof(double);
	double IDANTH = -99;       
	double DVS = DVSI;   
	double TSUM = 0;     
	double FR = AFGEN(FRTB, ILFR, DVS);
	double FL = AFGEN(FLTB, ILFL, DVS);
	double FS = AFGEN(FSTB, ILFS, DVS);
	double FO = AFGEN(FOTB, ILFO, DVS);
	double SLA[Y]; SLA[0] = AFGEN(SLATB, ILSLA, DVS);
	double LVAGE[Y]; LVAGE[0] = 0;             
	double ILVOLD = 1;                        
	double WRT = FR*TDWI;        
	double TADW = (1 - FR)*TDWI;  
	double WST = FS*TADW;        
	double WSO = FO*TADW;        
	double WLV = FL*TADW;        
	LAIEM = WLV * SLA[0];        
	double LV[Y]; LV[0] = WLV;	
	double LASUM = LAIEM;      
	double LAIEXP = LAIEM;     
	double LAIMAX = LAIEM;     
	double LAI[Y]; LAI[int(IDEM) - 1] = LASUM;    
	double TMINRA = 0;
	double TMNSAV[7];
	for (int i1 = 0; i1<7; i1++)
	{
		TMNSAV[i1] = -99;
	}
	double TEMP;
	double DTEMP;
	double GASS;
	double MRES;
	double DMI;
	double DTSUM;
	double DVR;
	double AMAX;
	double KDIF;
	double EFF;
	double XGAUSS[3] = { 0.1127017, 0.5000000, 0.8872983 };
	double WGAUSS[3] = { 0.2777778, 0.4444444, 0.2777778 };
	double PI = 3.1415926;
	double DALV;
	for (IDAY = IDEM - 1; IDAY < Y; IDAY++)
	{
		TEMP = (TMIN[IDAY] + TMAX[IDAY]) / 2;      
		DTEMP = (TMAX[IDAY] + TEMP) / 2;       
		for (int i2 = 0; i2 < 6; i2++)
			TMNSAV[i2] = TMNSAV[i2 + 1];
		TMNSAV[6] = TMIN[IDAY];
		TMINRA = 0;
		int I4 = 0;
		for (int i3 = 0; i3 < 7; i3++)
		{
			if (TMNSAV[i3] != -99)
			{
				TMINRA = TMINRA + TMNSAV[i3];
				I4 = I4 + 1;
			}
		}
		TMINRA = TMINRA / I4;
		DTSUM = AFGEN(DTSMTB, ILDTSM, TEMP);
		if (DVS < 1)
			DVR = DTSUM / TSUM1;   
		else
			DVR = DTSUM / TSUM2;   
		AMAX = AFGEN(AMAXTB, ILAMAX, DVS);
		AMAX = AMAX * AFGEN(TMPFTB, ILTMPF, DTEMP);
		KDIF = AFGEN(KDIFTB, ILKDIF, DVS);
		EFF = AFGEN(EFFTB, ILEFF, DTEMP);
		double DTGA = 0;
		double DAYL;
		for (int i5 = 0; i5 < 3; i5++)
		{
			double DEC = -asin(sin(23.45*0.0174533)*cos(2 * PI*(IDAY + 1 + 10) / Y));
			double SINLD = sin(0.017453292*43.85)*sin(DEC);
			double COSLD = cos(0.017453292*43.85)*cos(DEC);
			double AOB = SINLD / COSLD;
			DAYL = 12 * (1 + 2 * asin(AOB) / PI);           
			double HOUR = 12 + 0.5*DAYL*XGAUSS[i5];         
			double HOUR2 = SINLD + COSLD*cos(2 * PI*(HOUR + 12) / 24);
			double SINB = max(0, HOUR2);
			double DSINBE = 3600 * (DAYL*(SINLD + 0.4*(SINLD*SINLD + COSLD*COSLD*0.5)) + 12 * COSLD*(2 + 3 * 0.4*SINLD)*sqrt(1 - AOB*AOB) / PI);
			double PAR = 0.5*AVRAD[IDAY] * SINB*(1 + 0.4*SINB) / DSINBE;
			double SC = 1370 * (1 + 0.033*cos(2 * PI*(IDAY + 1) / Y));
			double DSINB = 3600 * (DAYL*SINLD + 24 * COSLD*sqrt(1 - AOB*AOB) / PI);
			double ANGOT = SC*DSINB;
			double ATMTR = AVRAD[IDAY] / ANGOT;
			double FRDIF;
			if (ATMTR > 0.75)
			{
				FRDIF = 0.23;
			}
			else
			{
				if (0.35 < ATMTR&ATMTR <= 0.75)
				{
					FRDIF = 1.33 - 1.46*ATMTR;
				}
				else
				{
					if (0.07 < ATMTR&ATMTR <= 0.35)
					{
						FRDIF = 1 - 2.3*(ATMTR - 0.07)*(ATMTR - 0.07);
					}
					else
						FRDIF = 1;
				}
			}
			double DIFPP = FRDIF*ATMTR*0.5*SC;
			double PARDIF = min(PAR, SINB*DIFPP);
			double PARDIR = PAR - PARDIF;
			double SCV = 0.2;									
			double REFH = (1 - sqrt(1 - SCV)) / (1 + sqrt(1 - SCV));
			double REFS = REFH * 2 / (1 + 1.6*SINB);				 
			double KDIRBL = (0.5 / SINB)*KDIF / (0.8*sqrt(1 - SCV));  
			double KDIRT = KDIRBL*sqrt(1 - SCV);                 
			double FGROS = 0;
			for (int i6 = 0; i6 < 3; i6++)
			{
				double LAIC = LAI[IDAY] * XGAUSS[i6];
				double VISDF = (1 - REFS)*PARDIF*KDIF*exp(-KDIF*LAIC);
				double VIST = (1 - REFS)*PARDIR*KDIRT*exp(-KDIRT*LAIC);
				double VISD = (1 - SCV)*PARDIR*KDIRBL*exp(-KDIRBL*LAIC);
				double VISSHD = VISDF + VIST - VISD;
				double FGRSH = AMAX*(1 - exp(-VISSHD*EFF / max(2.0, AMAX)));
				double VISPP = (1 - SCV)*PARDIR / SINB;
				double FGRSUN;
				if (VISPP <= 0)
					FGRSUN = FGRSH;
				else
					FGRSUN = AMAX*(1 - (AMAX - FGRSH)*(1 - exp(-VISPP*EFF / max(2.0, AMAX))) / (EFF*VISPP));

				double FSLLA = exp(-KDIRBL*LAIC);
				double FGL = FSLLA*FGRSUN + (1 - FSLLA)*FGRSH;
				FGROS = FGROS + FGL*WGAUSS[i6];
			}
			FGROS = FGROS*LAI[IDAY];
			DTGA = DTGA + FGROS*WGAUSS[i5];
		}
		DTGA = DTGA*DAYL;
		DTGA = DTGA * AFGEN(TMNFTB, ILTMNF, TMINRA); 
		GASS = DTGA * 30 / 44;    
		double RMRES = (RMR*WRT + RML*WLV + RMS*WST + RMO*WSO)* AFGEN(RFSETB, ILRFSE, DVS);
		double TEFF = pow(Q10, (TEMP - 25) / 10);   
		MRES = min(GASS, RMRES*TEFF);
		double ASRC = GASS - MRES;
		FR = AFGEN(FRTB, ILFR, DVS);
		FL = AFGEN(FLTB, ILFL, DVS);
		FS = AFGEN(FSTB, ILFS, DVS);
		FO = AFGEN(FOTB, ILFO, DVS);
		double CVF = 1 / ((FL / CVL + FS / CVS + FO / CVO)*(1 - FR) + FR / CVR); 
		DMI = CVF*ASRC;                   
		double GRRT = FR*DMI;                               
		double DRRT = WRT * AFGEN(RDRRTB, ILRDRR, DVS);
		double GWRT = GRRT - DRRT;       
		double ADMI = (1 - FR)*DMI;
		double GRST = FS*ADMI;                                  
		double DRST = AFGEN(RDRSTB, ILRDRS, DVS)*WST;
		double GWST = GRST - DRST;                   
		double GWSO = FO*ADMI;              
		double GRLV = FL*ADMI;									  
		double LAICR = 3.2 / KDIF;
		double DSLV = WLV * LIMIT(0, 0.03, 0.03*(LAI[IDAY] - LAICR) / LAICR);
		int I7 = ILVOLD - 1;                                 
		while (DSLV > LV[I7] & I7 >= 0)
		{
			DSLV = DSLV - LV[I7];
			I7 = I7 - 1;
		}
		DALV = 0;
		if (LVAGE[I7] > SPAN & DSLV > 0 & I7 >= 0)
		{
			DALV = LV[I7] - DSLV;
			DSLV = 0;
			I7 = I7 - 1;
		}
		while (I7 >= 0 & LVAGE[I7] > SPAN)
		{
			DALV = DALV + LV[I7];
			I7 = I7 - 1;
		}
		DALV = DALV / DELT;		
		double FYSDEL = max(0, (TEMP - TBASE) / (35 - TBASE)); 
		double SLAT = AFGEN(SLATB, ILSLA, DVS);
		double DTEFF;
		double GLAIEX;
		double GLASOL;
		double GLA;
		if (LAIEXP<6)
		{
			DTEFF = max(0, TEMP - TBASE);
			GLAIEX = LAIEXP*RGRLAI*DTEFF;
			GLASOL = GRLV*SLAT;
			GLA = min(GLAIEX, GLASOL);
			if (GRLV>0)
				SLAT = GLA / GRLV;
		}
		DVS = DVS + DVR*DELT;
		TSUM = TSUM + DTSUM*DELT;
		if (DVS >= 1 & IDANTH == -99)
		{
			IDANTH = IDAY - IDEM;
			DVS = 1;
		}
		double DSLVT = DSLV*DELT;
		int I8 = ILVOLD - 1;
		while (DSLVT > 0 & I8 >= 0)
		{
			if (DSLVT >= LV[I8])
			{
				DSLVT = DSLVT - LV[I8];
				LV[I8] = 0;
				I8 = I8 - 1;
			}
			else
			{
				LV[I8] = LV[I8] - DSLVT;
				DSLVT = 0;
			}
		}
		while (LVAGE[I8] >= SPAN&I8 >= 0)
		{
			LV[I8] = 0;
			I8 = I8 - 1;
		}
		ILVOLD = I8 + 1;
		int I9;
		for (I9 = ILVOLD - 1; I9 > -1; I9--)
		{
			LV[I9 + 1] = LV[I9];
			SLA[I9 + 1] = SLA[I9];
			LVAGE[I9 + 1] = LVAGE[I9] + FYSDEL*DELT;
		}
		ILVOLD = ILVOLD + 1;
		LV[0] = GRLV*DELT;
		SLA[0] = SLAT;
		LVAGE[0] = 0;
		LASUM = 0;
		double tt = 0;
		int I10;
		for (I10 = 0; I10 < ILVOLD; I10++)
			tt = tt + LV[I10] * SLA[I10];
		LASUM = tt;
		WLV = sum(LV, Y);
		LAIEXP = LAIEXP + GLAIEX*DELT;
		WRT = WRT + GWRT*DELT;   
		WST = WST + GWST*DELT;
		WSO = WSO + GWSO*DELT;
		TADW = WLV + WST + WSO;		
		LAI[IDAY + 1] = LASUM;
		LAIMAX = max(LAI[IDAY + 1], LAIMAX);
		if (ILVOLD > 364)
			break;
		if (DVS >= DVSEND)
			break;
		LAIMAX = max2(LAI, Y);
		if (LAIMAX <= 0.002 & DVS > 0.5)
			break;
	}	
	for (int i = 0; i < NT; i++)
	{		
		JCLAImoni[i]=LAI[int(param[i+10])];
	}
	return;	
}

__device__ void min2(double *a, int n, int &min_idx)
{
	double min = a[0];
	for (int i = 0; i < n; i++)
	{
		if (min > a[i])
		{
			min = a[i];
			min_idx = i;
		}
	}
}

//PSO algorithm
__global__ void pso(double *param, double *MLAI, double *TMIN, double *TMAX, double *AVRAD, long  rand, double *gbest)
{
	curandState state;
	int idx = threadIdx.x+blockDim.x*blockIdx.x;	
	if(idx<K)
	{
	long seed = rand;
	curand_init(seed, idx, 0, &state);
	double mv = (param[3] - param[2]) / 2;	
	double MLAI1[NT];
	for (int s = 0; s < NT; s++)
	{
		MLAI1[s] = MLAI[idx*2 + s];
	}	
	double vel[PS];
	double pos[PS];
	double pbest[PS2];	
	double randnum1;
	double randnum2;
	for (int i = 0; i < PS; i++)
	{
		randnum1 = abs(curand_uniform_double(&state));
		vel[i] = 2*mv*randnum1 - mv;
		randnum2 = abs(curand_uniform_double(&state));
		pos[i] = (param[3] - param[2])*randnum2 + param[2];	
		if(i%2==0)
		{
		   pbest[i/2] = pos[i];
		}
	}
	double pbestval[PS2];
	double cost;
	double JCLAImoni[NT];
	for (int i = 0; i < PS; i++)
	{
		LAIcal(pos[i], param, TMIN, TMAX, AVRAD, JCLAImoni);		
		if(i%2==0)
		{
			cost=0;
			for (int j = 0; j < NT; j++)
			{			
				cost = cost + (MLAI1[j] - JCLAImoni[j])*(MLAI1[j] - JCLAImoni[j]);
			}
			pbestval[i/2] = sqrt(cost) / NT;		
		}
	}			
	int min_idx = 0;
	min2(pbestval, PS2, min_idx);
	double gbestval = pbestval[min_idx];
	gbest[idx] = pbest[min_idx];
	double cnt2 = 0;
	double iwt[EP] = { param[6] };	
    double pout[PS2];
	double randnum3;
	double randnum4;
	double tmp1=0;
	double tr=0;
	for (int j = 0; j < EP; j++)
	{
		for (int i = 0; i < PS; i++)
		{
			LAIcal(pos[i], param, TMIN, TMAX, AVRAD, JCLAImoni);							
			cost = 0;
			if(i%2==0)
			{
				for (int jj = 0; jj < NT; jj++)
				{
					cost = cost + (MLAI1[jj] - JCLAImoni[jj])*(MLAI1[jj] - JCLAImoni[jj]);
				}
				pout[i/2] = sqrt(cost) / NT;
			}
		}
		tr = gbestval;     
		for (int i = 0; i < PS2; i++)
		{
			if (pbestval[i] >= pout[i])
			{
				pbestval[i] = pout[i];
				pbest[i] = pos[2*i];
			}
		}
		int min_idx2;
		min2(pbestval, PS2, min_idx2);
		double iterbestval = pbestval[min_idx2];
		if (gbestval >= iterbestval)
		{
			gbestval = iterbestval;
			gbest[idx] = pbest[min_idx2];
		}      
		if (j <= EP)
			iwt[j] = ((param[7] - param[6]) / 14 )*(j - 1) + param[6];		    
		else
			iwt[j] = param[7];
		double ac11;
		double ac22;
		double pbest2[PS];		
		for (int i = 0; i < PS; i++)
		{			
			if(i%2==0)
			{
				pbest2[i]=pbest[i];
			}
			else
			{
				pbest2[i]=pbest[i-1];
			}
			randnum3=abs(curand_uniform_double(&state));
			randnum4=abs(curand_uniform_double(&state));
			ac11 = randnum3 * param[4];
			ac22 = randnum4 * param[5];
			vel[i] = iwt[i] * vel[i] + ac11 * (pbest2[i] - pos[i]) + ac22 * (gbest[idx] - pos[i]);
			vel[i] = LIMIT(-mv, mv, vel[i]);
			pos[i] = pos[i] + vel[i];
			pos[i] = LIMIT(param[2], param[3], pos[i]);
		}
		tmp1 = abs(tr - gbestval);
		if (tmp1 > param[8])
			cnt2 = 0;
		else
		{
			if (tmp1 <= param[8])
			{
				cnt2 = cnt2 + 1;
				if (cnt2 >= param[9])
					break;
			}
		}		
	}
	}
	
}


using namespace std;

//main function
int main()
{	
	srand((unsigned int)time(NULL));
	const int nop = NT + 10;
	double param[nop];
	param[0] = 154;                   
	param[1] = Y - param[0] + 1;      
	param[2] = 10;                   
	param[3] = 50;                  
	param[4] = 2.1;                 
	param[5] = 1.6;                 
	param[6] = 0.9;                 
	param[7] = 0.6;                 
	param[8] = 1e-99;               
	param[9] = 10;                  
    param[10] = 194;                
	param[11] = 210;                
	param[12] = 216;
	param[13] = 259;	
	int i, j;
	double** MLAI= new double* [K];
	for (i=0;i<K;i++)
	{
		MLAI[i]=new double [NT];		
	}	
	ifstream fin("MLAI_2015.txt");         
	for (i = 0; i < K; i++)
	{
	    for (j = 0; j<NT; j++)
		{
		     fin >> MLAI[i][j];
		}	    
	}
	fin.close();
	const int totalpix = K*NT;
	double *MLAI1=new double [totalpix];
	
	for (int p = 0; p < K; p++)
	{
		for (int q = 0; q < NT; q++)
		{
			MLAI1[p*NT+q] = MLAI[p][q];			
		}
	}
	double TMIN[Y];
	ifstream fin1("TMIN_2015.txt");       
	for (i = 0; i<Y; i++)
		fin1 >> TMIN[i];
	fin1.close();
	double TMAX[Y];
	ifstream fin2("TMAX_2015.txt");
	for (i = 0; i<Y; i++)
		fin2 >> TMAX[i];
	fin2.close();
	double AVRAD[Y];
	ifstream fin3("AVRAD_2015.txt");
	for (i = 0; i<Y; i++)
		fin3 >> AVRAD[i];
	fin3.close();
	for (int i = 0; i < Y; i++)
	{
		AVRAD[i] = AVRAD[i] * 1000;
	}
	double *gbest=new double [K];
	double *d_param;
	double *d_MLAI;
	double *d_TMIN;
	double *d_TMAX;
	double *d_AVRAD;
	double *d_gbest;	
	cudaSetDevice(5);
	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start, 0);	
	cudaError_t cudastate;
	cudastate = cudaMalloc((void**)&d_param, sizeof(double)* nop); CHECK(cudastate)
	cudastate = cudaMalloc((void**)&d_MLAI, sizeof(double)*totalpix); CHECK(cudastate)
	cudastate = cudaMalloc((void**)&d_TMIN, sizeof(double)*Y); CHECK(cudastate)
	cudastate = cudaMalloc((void**)&d_TMAX, sizeof(double)*Y); CHECK(cudastate)
	cudastate = cudaMalloc((void**)&d_AVRAD, sizeof(double)*Y); CHECK(cudastate)
	cudastate = cudaMalloc((void**)&d_gbest, sizeof(double)*K); CHECK(cudastate)
	cudaMemcpy(d_param, param, sizeof(double)* nop, cudaMemcpyHostToDevice);
	cudaMemcpy(d_MLAI, MLAI1, sizeof(double)*totalpix, cudaMemcpyHostToDevice);
	cudaMemcpy(d_TMIN, TMIN, sizeof(double)*Y, cudaMemcpyHostToDevice);
	cudaMemcpy(d_TMAX, TMAX, sizeof(double)*Y, cudaMemcpyHostToDevice);
	cudaMemcpy(d_AVRAD, AVRAD, sizeof(double)*Y, cudaMemcpyHostToDevice);
	for(i=0; i<K; ++i) 
	{
		delete[] MLAI[i];
	}
	delete[] MLAI;
	delete[] MLAI1;	
	int  thread = 256;
	int  block = K / thread;
	dim3 dimGrid(block + 1);
	dim3 dimBlock(thread);
	pso <<<dimGrid, dimBlock >>>(d_param, d_MLAI, d_TMIN, d_TMAX, d_AVRAD,rand(), d_gbest);
    cudastate = cudaDeviceSynchronize(); CHECK(cudastate)
	cudaMemcpy(gbest, d_gbest, sizeof(double)*K, cudaMemcpyDeviceToHost);	
	float GPU_time;
	cudaEventRecord(end, 0);
	cudaEventSynchronize(end); 		
	cudaEventElapsedTime(&GPU_time, start, end);	
	cout << "parallel time："<< GPU_time/1000 << endl;	
	FILE *p = fopen("gbest_2015.txt", "wt");
	for (int i = 0; i<K; i++) 
		fprintf(p, "%4.2f\n", gbest[i]);
	fclose(p);	
	delete[] gbest;
	cudastate = cudaFree(d_param); CHECK(cudastate)
	cudastate = cudaFree(d_MLAI); CHECK(cudastate)
	cudastate = cudaFree(d_TMIN); CHECK(cudastate)
	cudastate = cudaFree(d_TMAX); CHECK(cudastate)
	cudastate = cudaFree(d_AVRAD); CHECK(cudastate)
	cudastate = cudaFree(d_gbest); CHECK(cudastate)
}

