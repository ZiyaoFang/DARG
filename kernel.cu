#include<math.h>
#include<stdio.h>
#define num_toa $NTOA
#define num_range $Nrange
#define num_linear $Nlinear
#define pi 3.141592653589793
#define deg_rad 0.017453292519943295
#define km_s 3.335641e-06
#define mean_mjd $mean_MJD

__global__ void kernel(float start_ra, float start_dec, float step, float RAJ, float DECJ, float p0, double *mjd, float *ssb, double *resids, float *errors_inverse, float *result) 
{
    int thread_count = blockIdx.x * blockDim.x + threadIdx.x;
    int count = blockIdx.y * blockDim.y + threadIdx.y;

    float x = (-1 * num_range * 0.5 + thread_count + start_ra) * step * deg_rad;     
    float y = (-1 * num_range * 0.5 + count + start_dec) * step * deg_rad;          

    float RA = RAJ * deg_rad;
    float DEC = DECJ * deg_rad;

    double mjd_value;
    double new_resids_linear;
    double mean_x=0.0; 
    double mean_y=0.0;
    double mean_resids=0.0;
    double chi2 = 0.0;
                            
    float dx = -cos(RA)*cos(DEC)*(x*x+y*y)*0.5 - cos(RA)*sin(DEC)*y - sin(RA)*cos(DEC)*x + sin(DEC)*sin(RA)*x*y;
    float dy = -sin(RA)*cos(DEC)*(x*x+y*y)*0.5 - sin(DEC)*sin(RA)*y + cos(RA)*cos(DEC)*x - cos(RA)*sin(DEC)*x*y;
    float dz = -sin(DEC)*y*y*0.5 + cos(DEC)*y;
    
    //float dx = cos(RA + x)*cos(DEC + y) - cos(RA)*cos(DEC);
    //float dy = sin(RA+x)*cos(DEC+y) - sin(RA)*cos(DEC);
    //float dz = sin(DEC+y) - sin(DEC);
    
    double delt_resids;
    double new_resids_bk;
    double linear_resids,destination;
    
    double w1=0.0, w2=0.0;
    double m1=0.0, m2=0.0;
    float k1,b1,k2,b2; 
    
    for(int ii=0; ii<num_linear; ii++)
    {
    delt_resids = dx * ssb[ii*3] + dy * ssb[ii*3+1] + dz * ssb[ii*3+2];
    new_resids_linear = resids[ii] + delt_resids*km_s;
    mean_x += mjd[ii];
    mean_y += new_resids_linear;
    }
    mean_x = mean_x / num_linear;
    mean_y = mean_y / num_linear;
    
    for(int jj=0; jj<num_linear; jj++)
    {
    mjd_value = mjd[jj];
    delt_resids = dx * ssb[jj*3] + dy * ssb[jj*3+1] + dz * ssb[jj*3+2];
    new_resids_linear = resids[jj] + delt_resids*km_s;
    w1 += (mjd_value-mean_x)*(new_resids_linear-mean_y);
    w2 += pow((mjd_value-mean_x),2);
    }
    k1 = w1 / w2;
    b1 = mean_y - mean_x * k1;
    
    for(int kk=0; kk<num_toa; kk++)
    {
    mjd_value = mjd[kk];
    delt_resids = dx * ssb[kk*3] + dy * ssb[kk*3+1] + dz * ssb[kk*3+2];
    linear_resids = resids[kk] + delt_resids*km_s;
    destination = mjd_value * k1 + b1;
    if(linear_resids - destination > 0)
    {
        while(destination - linear_resids > p0)
        {
        linear_resids = linear_resids-p0;
        }
    }
    else
    {
        while(linear_resids - destination > p0)
        {
        linear_resids = linear_resids+p0;
        }
    }      
    new_resids_bk = linear_resids;
    mean_resids += new_resids_bk;
    }
    mean_resids = mean_resids / num_toa;
    
    for(int pp=0; pp<num_toa; pp++)
    {
    mjd_value = mjd[pp];
    delt_resids = dx * ssb[pp*3] + dy * ssb[pp*3+1] + dz * ssb[pp*3+2];
    linear_resids = resids[pp] + delt_resids*km_s;
    destination = mjd_value * k1 + b1;
    if(linear_resids - destination > 0)
    {
        while(destination - linear_resids > p0)
        {
        linear_resids = linear_resids-p0;
        }
    }
    else
    {
        while(linear_resids - destination > p0)
        {
        linear_resids = linear_resids+p0;
        }
    }
    new_resids_bk = linear_resids;
    m1 += (mjd_value-mean_mjd)*(new_resids_bk-mean_resids);
    m2 += pow((mjd_value-mean_mjd),2);
    }
    k2 = m1 / m2;
    b2 = k2 * mean_mjd - mean_resids;
    
    for(int tt=0; tt<num_toa; tt++)
    {
    mjd_value = mjd[tt];
    delt_resids = dx * ssb[tt*3] + dy * ssb[tt*3+1] + dz * ssb[tt*3+2];
    linear_resids = resids[tt] + delt_resids*km_s;
    destination = mjd_value * k1 + b1;
    if(linear_resids - destination > 0)
    {
        while(destination - linear_resids > p0)
        {
        linear_resids = linear_resids-p0;
        }
    }
    else
    {
        while(linear_resids - destination > p0)
        {
        linear_resids = linear_resids+p0;
        }
    }
    new_resids_bk = linear_resids;
    destination = k2 * mjd_value + b2;
    chi2 += pow((new_resids_bk-destination)*errors_inverse[tt],2);
    }
        
    result[(thread_count*num_range+count)*3] = chi2 / num_toa;
    result[(thread_count*num_range+count)*3+1] = (-1 * num_range * 0.5 + thread_count + start_ra) * step + RAJ;
    result[(thread_count*num_range+count)*3+2] =  (-1 * num_range * 0.5 + count + start_dec) * step + DECJ;
    
    __syncthreads();
} 
