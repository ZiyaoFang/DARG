#define pi 3.141592653589793f
#define deg_rad 0.017453292519943295f
#define km_s 3.335641e-06f

__global__ void quick_search(float RAJ, float DECJ, int num_toa, float *mjd, float *ssb, float mean_mjd, float *resids, float *errors_inverse,
int range, float step,float *result)
{ 
    // extern __shared__ char shared_memory[];
    int thread_count = blockIdx.x * blockDim.x + threadIdx.x;
    int count = blockIdx.y * blockDim.y + threadIdx.y;

    float x = (-1 * range * 0.5 + thread_count) * step * deg_rad;     
    float y = (-1 * range * 0.5 + count) * step * deg_rad;          

    float RA = RAJ * deg_rad;
    float DEC = DECJ * deg_rad;

    double mjd_value;
    double new_resids_linear;
    double mean_x=0.0; 
    double mean_y=0.0;
    double mean_resids=0.0;
    double chi2 = 0.0;
                            
    float dx = -__cosf(RA)*__cosf(DEC)*(x*x+y*y)*0.5 - __cosf(RA)*__sinf(DEC)*y - __sinf(RA)*__cosf(DEC)*x + __sinf(DEC)*__sinf(RA)*x*y;
    float dy = -__sinf(RA)*__cosf(DEC)*(x*x+y*y)*0.5 - __sinf(DEC)*__sinf(RA)*y + __cosf(RA)*__cosf(DEC)*x - __cosf(RA)*__sinf(DEC)*x*y;
    float dz = -__sinf(DEC)*y*y*0.5 + __cosf(DEC)*y;
    
    double delt_resids;
    double new_resids_bk;
    double linear_resids,destination;
    
    double w1=0.0, w2=0.0;
    double m1=0.0, m2=0.0;
    float k,b; 

    for(int kk=0; kk<num_toa; kk++)
    {
    mjd_value = mjd[kk];
    delt_resids = dx * ssb[kk*3] + dy * ssb[kk*3+1] + dz * ssb[kk*3+2];
    new_resids_bk = resids[kk] + delt_resids*km_s;
    mean_resids += new_resids_bk;
    }
    mean_resids = mean_resids / num_toa;
    
    for(int pp=0; pp<num_toa; pp++)
    {
    mjd_value = mjd[pp];
    delt_resids = dx * ssb[pp*3] + dy * ssb[pp*3+1] + dz * ssb[pp*3+2];
    new_resids_bk = resids[pp] + delt_resids*km_s;
    m1 += (mjd_value-mean_mjd)*(new_resids_bk-mean_resids);
    m2 += pow((mjd_value-mean_mjd),2);
    }
    k = m1 / m2;
    b = mean_mjd - mean_resids * k ;
    
    for(int tt=0; tt<num_toa; tt++)
    {
    mjd_value = mjd[tt];
    delt_resids = dx * ssb[tt*3] + dy * ssb[tt*3+1] + dz * ssb[tt*3+2];
    new_resids_bk = resids[tt] + delt_resids*km_s;
    destination = k * mjd_value + b;
    chi2 += pow((new_resids_bk-destination)*errors_inverse[tt],2);
    }
        
    result[(thread_count*range+count)*3] = chi2 / num_toa;
    result[(thread_count*range+count)*3+1] = (-1 * range * 0.5 + thread_count) * step + RAJ;
    result[(thread_count*range+count)*3+2] =  (-1 * range * 0.5 + count) * step + DECJ;
    
    //__syncthreads();
}

__global__ void complete_search(float RAJ, float DECJ, float p0, float num_toa, int *linear_direction, int num_linear, float start_ra, float start_dec, float *mjd, float *ssb, float mean_mjd, float *resids, float *errors_inverse,
int range, float step,float *result)
  {
    // extern __shared__ char shared_memory[];
    int thread_count = blockIdx.x * blockDim.x + threadIdx.x;
    int count = blockIdx.y * blockDim.y + threadIdx.y;

    float x = (-1 * range * 0.5 + thread_count + start_ra) * step * deg_rad;     
    float y = (-1 * range * 0.5 + count + start_dec) * step * deg_rad;        

    float RA = RAJ * deg_rad;
    float DEC = DECJ * deg_rad;

    int linear_index;
    double mjd_value;
    double new_resids_linear;
    double mean_x=0.0; 
    double mean_y=0.0;
    double mean_resids=0.0;
    double mean_resids_phase = 0.0;
    double chi2 = 0.0;
                              
    float dx = -__cosf(RA)*__cosf(DEC)*(x*x+y*y)*0.5 - __cosf(RA)*__sinf(DEC)*y - __sinf(RA)*__cosf(DEC)*x + __sinf(DEC)*__sinf(RA)*x*y;
    float dy = -__sinf(RA)*__cosf(DEC)*(x*x+y*y)*0.5 - __sinf(DEC)*__sinf(RA)*y + __cosf(RA)*__cosf(DEC)*x - __cosf(RA)*__sinf(DEC)*x*y;
    float dz = -__sinf(DEC)*y*y*0.5 + __cosf(DEC)*y;
    
    double delt_resids;
    double new_resids_bk;
    double resids0,linear_resids,destination;
    
    double w1=0.0, w2=0.0;
    double m1=0.0, m2=0.0;
    double n1=0.0, n2=0.0;
    float k1,b1,k2,b2,k3,b3; 
    
    for(int ii=0; ii<num_linear; ii++)
    {
      //float3 ssb_value = make_float3(ssb[ii*3], ssb[ii*3+1], ssb[ii*3+2]);
      linear_index = linear_direction[ii];
      delt_resids = dx * ssb[linear_index*3] + dy * ssb[linear_index*3+1] + dz * ssb[linear_index*3+2];
      new_resids_linear = resids[linear_index] + delt_resids*km_s;
      mean_x += mjd[linear_index];
      mean_y += new_resids_linear;
    }
    mean_x = mean_x / num_linear;
    mean_y = mean_y / num_linear;
    
    for(int jj=0; jj<num_linear; jj++)
    {
      linear_index = linear_direction[jj];
      mjd_value = mjd[linear_index];
      delt_resids = dx * ssb[linear_index*3] + dy * ssb[linear_index*3+1] + dz * ssb[linear_index*3+2];
      new_resids_linear = resids[linear_index] + delt_resids*km_s;
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
        while(linear_resids - destination > 0.5*p0)
        {
          linear_resids = linear_resids-p0;
        }
      }
      else
      {
        while(destination - linear_resids > 0.5*p0)
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
        while(linear_resids - destination > 0.5*p0)
        {
          linear_resids = linear_resids-p0;
        }
      }
      else
      {
        while(destination - linear_resids > 0.5*p0)
        {
          linear_resids = linear_resids+p0;
        }
      } 
      new_resids_bk = linear_resids;
      m1 += (mjd_value-mean_mjd)*(new_resids_bk-mean_resids);
      m2 += pow((mjd_value-mean_mjd),2);
    }
    k2 = m1 / m2;
    b2 = mean_resids - k2 * mean_mjd;
    p0 = p0 * (1 + k2/86400);
    
    mjd_value = mjd[0];
    delt_resids = dx * ssb[0] + dy * ssb[1] + dz * ssb[2];
    resids0 = resids[0] + delt_resids*km_s;
    destination = mjd_value * k1 + b1;
    if(resids0 - destination > 0)
    {
      while(resids0 - destination > 0.5*p0)
      {
        resids0 = resids0-p0;
      }
    }
    else
    {
      while(destination - resids0 > 0.5*p0)
      {
        resids0 = resids0+p0;
      }
    }
    resids0 = resids0 - k2*mjd_value - b2;
    new_resids_bk = resids0;
    
    for(int tt=1; tt<num_toa; tt++)
    {
      mjd_value = mjd[tt];
      delt_resids = dx * ssb[tt*3] + dy * ssb[tt*3+1] + dz * ssb[tt*3+2];
      linear_resids = resids[tt] + delt_resids*km_s;
      destination = mjd_value * k1 + b1;
      if(linear_resids - destination > 0)
      {
        while(linear_resids - destination > 0.5*p0)
        {
          linear_resids = linear_resids-p0;
        }
      }
      else
      {
        while(destination - linear_resids > 0.5*p0)
        {
          linear_resids = linear_resids+p0;
        }
      }
      linear_resids =  linear_resids - k2 * mjd_value - b2 - resids0;
      if(abs(linear_resids) > 0.5*p0)
      {
        if(linear_resids > 0.5*p0)
        {
          linear_resids = linear_resids - p0;
        }
        else
        {
          linear_resids = linear_resids + p0;
        }
      }
      else
      {
        linear_resids = linear_resids;
      } 
      if(linear_resids - new_resids_bk > 0.5*p0)
      {
        linear_resids = linear_resids - p0;
      }
      else if(linear_resids - new_resids_bk < -0.5*p0)
      {
        linear_resids = linear_resids + p0;
      }
      else
      {
        linear_resids = linear_resids;
      }
      new_resids_bk = linear_resids;
      mean_resids_phase += new_resids_bk;
    }
    mean_resids_phase = mean_resids_phase / num_toa;
    new_resids_bk = resids0;
    
    for(int zz=1; zz<num_toa; zz++)
    {
      mjd_value = mjd[zz];
      delt_resids = dx * ssb[zz*3] + dy * ssb[zz*3+1] + dz * ssb[zz*3+2];
      linear_resids = resids[zz] + delt_resids*km_s;
      destination = mjd_value * k1 + b1;
      if(linear_resids - destination > 0)
      {
        while(linear_resids - destination > 0.5*p0)
        {
          linear_resids = linear_resids-p0;
        }
      }
      else
      {
        while(destination - linear_resids > 0.5*p0)
        {
          linear_resids = linear_resids+p0;
        }
      }
      linear_resids =  linear_resids - k2 * mjd_value - b2 - resids0;
      if(abs(linear_resids) > 0.5*p0)
      {
        if(linear_resids > 0.5*p0)
        {
          linear_resids = linear_resids - p0;
        }
        else
        {
          linear_resids = linear_resids + p0;
        }
      }
      else
      {
        linear_resids = linear_resids;
      }      
      if(linear_resids - new_resids_bk > 0.5*p0)
      {
        linear_resids = linear_resids - p0;
      }
      else if(linear_resids - new_resids_bk < -0.5*p0)
      {
        linear_resids = linear_resids + p0;
      }
      else
      {
        linear_resids = linear_resids;
      }
      new_resids_bk = linear_resids;
      n1 += (mjd_value-mean_mjd)*(new_resids_bk-mean_resids_phase);
    }
    n1 += (mjd[0]-mean_mjd)*(0.0-mean_resids_phase);
    n2 = m2;
    k3 = n1 / n2;
    b3 = mean_resids_phase - k3 * mean_mjd;
    p0 = p0 * (1 + k3/86400);
    new_resids_bk = resids0;
  
    for(int qq=1; qq<num_toa; qq++)
    {
      mjd_value = mjd[qq];
      delt_resids = dx * ssb[qq*3] + dy * ssb[qq*3+1] + dz * ssb[qq*3+2];
      linear_resids = resids[qq] + delt_resids*km_s;
      destination = mjd_value * k1 + b1;
      if(linear_resids - destination > 0)
      {
        while(linear_resids - destination > 0.5*p0)
        {
          linear_resids = linear_resids-p0;
        }
      }
      else
      {
        while(destination - linear_resids > 0.5*p0)
        {
          linear_resids = linear_resids+p0;
        }
      }
      linear_resids =  linear_resids - k2 * mjd_value - b2 - resids0;
      if(abs(linear_resids) > 0.5*p0)
      {
        if(linear_resids > 0.5*p0)
        {
          linear_resids = linear_resids - p0;
        }
        else
        {
          linear_resids = linear_resids + p0;
        }
      }
      else
      {
        linear_resids = linear_resids;
      }      
      if(linear_resids - new_resids_bk > 0.5*p0)
      {
        linear_resids = linear_resids - p0;
      }
      else if(linear_resids - new_resids_bk < -0.5*p0)
      {
        linear_resids = linear_resids + p0;
      }
      else
      {
        linear_resids = linear_resids;
      }
      new_resids_bk = linear_resids;
      new_resids_linear = new_resids_bk - k3*mjd_value - b3;
      chi2 = chi2+pow(new_resids_linear*errors_inverse[qq],2);
    }  
    chi2 += pow((k3*mjd[0]+b3)*errors_inverse[0],2);
        
    result[(thread_count*range+count)*3] = chi2 / num_toa;
    result[(thread_count*range+count)*3+1] = (-1 * range * 0.5 + thread_count + start_ra) * step + RAJ;
    result[(thread_count*range+count)*3+2] =  (-1 * range * 0.5 + count + start_dec) * step + DECJ;
    // __syncthreads();
  }


