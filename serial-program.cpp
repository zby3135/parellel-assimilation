#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <algorithm>
#define Y 365
#define max(a, b)  (((a) > (b)) ? (a) : (b))
#define min(a, b)  (((a) < (b)) ? (a) : (b))
#define CHECK(res) if(res!=cudaSuccess){printf("Error:%d\n", __LINE__);exit(-1);}
using namespace std;
double LIMIT(double min, double max, double X)
{
	if (min>X)
		return min;
	if (max<X)
		return max;
	return X;
}
double AFGEN(double *x, int n, double X)
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
double sum(double *f, int k)
{
	double sum = 0;
	for (int i = 0; i<k; i++)
	{
		if (f[i]>0)
			sum = sum + f[i];
	}
	return sum;
}
double max2(double *f, int k)
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
double *LAIcal(int SPAN, double *param, double *TMIN, double *TMAX, double *AVRAD)
{
	double f = 1;
	double IDEM = param[0];
	double TIME = param[1];
	double *out = new double[Y];
	int IDAY = 0;                  
	double TSUM0 = 300;              
	int DELT = 1;             
	int IDSL = 0;             
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
	double PERDL = 0.030;        
	double RDRRTB[] = { 0.00, 0.000, 1.50, 0.000, 1.5001, 0.020, 2.00, 0.020 };
	int ILRDRR = sizeof(RDRRTB) / sizeof(double);
	double RDRSTB[] = { 0.00, 0.000, 1.50, 0.000, 1.5001, 0.020, 2.00, 0.020 };
	int ILRDRS = sizeof(RDRSTB) / sizeof(double);
	double IDANTH = -99;             
	double DVS = DVSI;  
	double FR = AFGEN(FRTB, ILFR, DVS);  
	double FL = AFGEN(FLTB, ILFL, DVS);  
	double FS = AFGEN(FSTB, ILFS, DVS);  
	double FO = AFGEN(FOTB, ILFO, DVS);  
	double SLA[Y]; SLA[0] = AFGEN(SLATB, ILSLA, DVS);
	double LVAGE[Y]; LVAGE[0] = 0;             
	double ILVOLD = 1;                        
	double DWRT = 0;                          
	double DWLV = 0;                          
	double DWST = 0;                          
	double DWSO = 0;                          
	double WRT = FR*TDWI;        
	double TADW = (1 - FR)*TDWI; 
	double WST = FS*TADW;        
	double WSO = FO*TADW;        
	double WLV[Y]; WLV[int(IDEM) - 1] = FL*TADW; 
	LAIEM = WLV[int(IDEM) - 1] * SLA[0];         
	double LV[Y]; LV[0] = WLV[int(IDEM) - 1];	
	double LASUM[Y]; LASUM[int(IDEM) - 1] = LAIEM;
	double LAIEXP = LAIEM;                    
	double LAIMAX = LAIEM;                    
	double LAI[Y]; LAI[int(IDEM) - 1] = LASUM[int(IDEM) - 1];
	double GASST = 0;
	double MDSLV = 0;
	double TRAT = 0;
	double TMINRA = 0;
	double TMNSAV[7];
	for (int i1 = 0; i1<7; i1++)
	{
		TMNSAV[i1] = -99;
	}
	double TSUME = 0;
	double DTSUME = 0;
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
	double TWRT;
	double TWLV;
	double TWST;
	double TWSO;
	double XGAUSS[3] = { 0.1127017, 0.5000000, 0.8872983 };
	double WGAUSS[3] = { 0.2777778, 0.4444444, 0.2777778 };
	double PI = 3.1415926;
	double DALV;
	double DRSO;
	for (int IDAY = IDEM - 1; IDAY < Y; IDAY++)
	{
		TEMP = (TMIN[IDAY] + TMAX[IDAY]) / 2;            
		DTEMP = (TMAX[IDAY] + TEMP) / 2;      
		for (int i2 = 0; i2 < 6; i2++)
			TMNSAV[i2] = TMNSAV[i2 + 1];
		TMNSAV[6] = TMIN[IDAY];
		double TMINRA = 0;
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
		double RMRES = (RMR*WRT + RML*WLV[IDAY] + RMS*WST + RMO*WSO)* AFGEN(RFSETB, ILRFSE, DVS);
		double TEFF = pow(Q10, (TEMP - 25) / 10);
		MRES = min(GASS, RMRES*TEFF);
		double ASRC = GASS - MRES;
		FR = AFGEN(FRTB, ILFR, DVS);
		FL = AFGEN(FLTB, ILFL, DVS);
		FS = AFGEN(FSTB, ILFS, DVS);
		FO = AFGEN(FOTB, ILFO, DVS);
		double CVF = 1 / ((FL / CVL + FS / CVS + FO / CVO)*(1 - FR) + FR / CVR);  
		CVF = CVF*f;
		DMI = CVF*ASRC;              
		double GRRT = FR*DMI;        
		double DRRT = WRT * AFGEN(RDRRTB, ILRDRR, DVS);
		double GWRT = GRRT - DRRT;                     
		double ADMI = (1 - FR)*DMI;
		double GRST = FS*ADMI;     
		double DRST = AFGEN(RDRSTB, ILRDRS, DVS)*WST; 
		double GWST = GRST - DRST;                    
		double GWSO = FO*ADMI;                        
		DRSO = 0;
		double GRLV = FL*ADMI;	
		double LAICR = 3.2 / KDIF;
		double DSLV = WLV[IDAY] * LIMIT(0, 0.03, 0.03*(LAI[IDAY] - LAICR) / LAICR);
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
		double DRLV = DSLV + DALV;
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
		LASUM[IDAY + 1] = 0;
		WLV[IDAY + 1] = 0;
		double tt = 0;
		int I10;
		for (I10 = 0; I10 < ILVOLD; I10++)
			tt = tt + LV[I10] * SLA[I10];
		LASUM[IDAY + 1] = tt;
		WLV[IDAY + 1] = sum(LV, Y);
		LAIEXP = LAIEXP + GLAIEX*DELT;
		WRT = WRT + GWRT*DELT;  
		WST = WST + GWST*DELT;
		WSO = WSO + GWSO*DELT;
		TADW = WLV[IDAY + 1] + WST + WSO;
		DWRT = DWRT + DRRT*DELT;
		DWLV = DWLV + DRLV*DELT;
		DWST = DWST + DRST*DELT;
		DWSO = DWSO + DRSO*DELT;
		LAI[IDAY + 1] = LASUM[IDAY + 1];
		LAIMAX = max(LAI[IDAY + 1], LAIMAX);
		if (ILVOLD > 364)
			break;
		if (DVS >= DVSEND)
			break;
		LAIMAX = max2(LAI, Y);
		if (LAIMAX <= 0.002 & DVS> 0.5)
			break;
	}
	for (int i = 0; i < Y; i++)
	{
		out[i] = LAI[i];
	}
	return out;
}
void min2(double *a, int n, int &min_idx)
{
	double min = a[0];
	min_idx = 0;
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
void pso(int k, double *param, double *MLAI, double *TMIN, double *TMAX, double *AVRAD, double *gbest)
{
	const int NT = 4;
	const int iwe = 15;
	const int PS = 10;
	double varrange0 = param[2];
	double varrange1 = param[3];
	double iw1 = param[6];
	double iw2 = param[7];
	double ac1 = param[4];
	double ac2 = param[5];
	double ergrd = param[8];
	double ergrdep = param[9];
	double mv = (varrange1 - varrange0) / 2;         
	double velmaskmin = -mv;
	double velmaskmax = mv;
	double posmaskmin = varrange0;
	double posmaskmax = varrange1;
	double vel[PS];
	for (int i = 0; i < PS; i++)
	{
		vel[i] = (rand() % ((int)((velmaskmax - velmaskmin) * 10000 + 1)) + (velmaskmin * 10000)) / 10000;
	}
	double pos[PS];
	for (int i = 0; i < PS; i++)
	{
		pos[i] = (rand() % ((int)((posmaskmax - posmaskmin) * 10000 + 1)) + (posmaskmin * 10000)) / 10000;
	}
	double pbest[PS];
	for (int i = 0; i < PS; i++)
	{
		pbest[i] = pos[i];
	}
	double pout1[PS];
	int T[NT];
	for (int i = 0; i < NT; i++)
	{
		T[i] = param[i + 10];
	}
	double MLAI1[NT];
	for (int s = 0; s < NT; s++)
	{
		MLAI1[s] = MLAI[k + s];
	}
	for (int i = 0; i < PS; i++)
	{
		double* LAImoni = LAIcal(pos[i], param, TMIN, TMAX, AVRAD);
		double JCLAImoni[NT];
		double cost = 0;
		for (int j = 0; j < NT; j++)
		{
			JCLAImoni[j] = LAImoni[T[j] - 1];
			cost = cost + (MLAI1[j] - JCLAImoni[j])*(MLAI1[j] - JCLAImoni[j]);
		}
		pout1[i] = sqrt(cost) / NT;
	}
	double pbestval[PS];
	for (int i = 0; i<PS; i++)
	{
		pbestval[i] = pout1[i];
	}
	double gbestval;
	int min_idx = 0;
	min2(pbestval, PS, min_idx);
	gbestval = pbestval[min_idx];
	gbest[k] = pbest[min_idx];
	double cnt = 0;
	double cnt2 = 0;
	double iwt[15] = { iw1 };
	double bestpos[15][2];
	double tr[15];
	double te[15];
	for (int j = 0; j < 15; j++)
	{
		double pout2[PS];
		for (int i = 0; i < PS; i++)
		{
			double* LAImoni = LAIcal(pos[i], param, TMIN, TMAX, AVRAD);
			double JCLAImoni[NT];
			double cost = 0;
			for (int j = 0; j < NT; j++)
			{
				JCLAImoni[j] = LAImoni[T[j] - 1];
				cost = cost + (MLAI1[j] - JCLAImoni[j])*(MLAI1[j] - JCLAImoni[j]);
			}
			pout2[i] = sqrt(cost) / NT;
		}
		tr[j] = gbestval;      
		te[j] = j;             
		bestpos[j][0] = gbest[k];
		bestpos[j][1] = gbestval;
		for (int i = 0; i < PS; i++)
		{
			int pbestval_ = floor(pbestval[i] * 10000);
			int pout2_ = floor(pout2[i] * 10000);
			if (pbestval_ >= pout2_)
			{
				pbestval[i] = pout2[i];
				pbest[i] = pos[i];
			}
		}
		int min_idx2;
		min2(pbestval, PS, min_idx2);
		double iterbestval = pbestval[min_idx2];
		int gbestval_ = floor(gbestval * 10000);
		int iterbestval_ = floor(iterbestval * 10000);
		if (gbestval_ >= iterbestval_)
		{
			gbestval = iterbestval;
			gbest[k] = pbest[min_idx2];
		}
		double rannum1[PS];
		int N = 9999;
		for (int i = 0; i < PS; i++)
		{
			rannum1[i] = rand() % (N + 1) / (double)(N + 1);
		}
		double rannum2[PS];
		for (int i = 0; i < PS; i++)
		{
			rannum2[i] = rand() % (N + 1) / (double)(N + 1);
		}
		if (j <= iwe)
			iwt[j] = ((iw2 - iw1) / (iwe - 1))*(j - 1) + iw1;
		else
			iwt[j] = iw2;
		double ac11[PS];
		double ac22[PS];
		for (int i = 0; i < PS; i++)
		{
			ac11[i] = rannum1[i] * ac1;
			ac22[i] = rannum2[i] * ac2;
			vel[i] = iwt[i] * vel[i] + ac11[i] * (pbest[i] - pos[i]) + ac22[i] * (gbest[k] - pos[i]);
			vel[i] = LIMIT(velmaskmin, velmaskmax, vel[i]);
			pos[i] = pos[i] + vel[i];
			pos[i] = LIMIT(posmaskmin, posmaskmax, pos[i]);
		}
		double tmp1 = abs(tr[j] - gbestval);
		if (tmp1 > ergrd)
			cnt2 = 0;
		else
		{
			if (tmp1 <= ergrd)
			{
				cnt2 = cnt2 + 1;
				if (cnt2 >= ergrdep)
					break;
			}
		}
	}
}
//main function
int main()
{
	const int NT = 4;
	const int K = 100;
	const int nop = NT + 10;
	double param[nop];
	const int totalpix = K*NT;
	param[0] = 154;                   
	param[1] = Y - param[0] + 1;      
	param[2] = 10;                    
	param[3] = 50;                    
	param[4] = 2.1;                   
	param[5] = 1.6;                   
	param[6] = 0.9;                   
	param[7] = 0.6;                   
	param[8] = 1e-99;                 
	param[9] = 15;                    
	param[10] = 203;                  
	param[11] = 215;                  
	param[12] = 232;
	param[13] = 261;
	int i, j;
	double** MLAI = new double*[K];
	for (i = 0; i<K; i++)
	{
		MLAI[i] = new double[NT];
	}
	ifstream fin("MLAI10.txt");
	for (i = 0; i < K; i++)
	{
		for (j = 0; j<NT; j++)
		{
			fin >> MLAI[i][j];
		}
	}
	double *TMIN = new double[Y];
	ifstream fin1("TMIN_2017.txt");        
	for (i = 0; i<Y; i++)
		fin1 >> TMIN[i];
	fin1.close();
	double *TMAX = new double[Y];
	ifstream fin2("TMAX_2017.txt");
	for (i = 0; i<Y; i++)
		fin2 >> TMAX[i];
	fin2.close();
	double *AVRAD = new double[Y];
	ifstream fin3("AVRAD_2017.txt"); 
	for (i = 0; i<Y; i++)
		fin3 >> AVRAD[i];
	fin3.close();
	for (int i = 0; i < Y; i++)
	{
		AVRAD[i] = AVRAD[i] * 1000;
	}
	double *gbest = new double[totalpix];
	double *MLAI1 = new double[totalpix];
	for (int p = 0; p < K; p++)
	{
		for (int q = 0; q < NT; q++)
		{
			MLAI1[p*NT + q] = MLAI[p][q];
		}
	}
	srand((unsigned)time(NULL));
	clock_t start, stop;
	start = clock();
	for (int k = 0; k < K * NT; k = k + NT)
	{
		pso(k, param, MLAI1, TMIN, TMAX, AVRAD, gbest);
	}
	FILE *p1 = fopen("gbest10.txt", "wt");
	for (int i = 0; i < K * NT; i = i + NT) 
	{
		fprintf(p1, "%4.2f\t\n", gbest[i]);
	}
	fclose(p1);
	for (i = 0; i<K; ++i)
	{
		delete[] MLAI[i];
	}
	delete[] MLAI;
	delete[] TMAX;
	delete[] gbest;
	delete[] AVRAD;
	delete[] TMIN;
	delete[] MLAI1;
	stop = clock();
	cout << "serial time" << (double)(stop - start) / CLOCKS_PER_SEC << endl;
	return 0;
}

