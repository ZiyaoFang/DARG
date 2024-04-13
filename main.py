#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import numpy as np
import matplotlib.pyplot as plt
import pint.models as models
import pint.toa as toa
import pint.fitter as fitter
import astropy
from pint.residuals import Residuals as Residuals
import astropy.coordinates
import astropy.units as u
import copy
import logging
import pycuda.autoinit
from pycuda import gpuarray
import pycuda.driver as cuda
import time
from pycuda.elementwise import ElementwiseKernel
from pycuda.compiler import SourceModule
import astropy
import string
import astropy.units as u
from astropy.coordinates.angles import Angle
from pathlib import Path
import argparse
import sys

# __all__ = ["main"]
def reduce_toa_number(toas,number_per_observation,outfile=None):
    mjd = toas.get_mjds()
    integer_parts = mjd.value.astype(int)  # 取整数部分
    unique_integers = np.unique(integer_parts)  # 获取唯一的整数部分值

    reduced_toas = []
    observation = []
    indices = []
    for integer in unique_integers:
        day_indices = np.where(integer_parts == integer)[0]  # 找到与当前整数部分相同的索引
        num_group = mjd[day_indices]  # 根据索引获取对应的原始数值
        sorted_indices = day_indices[np.argsort(num_group)][0:number_per_observation]  # 对这些数值排序并取前n个的索引
        # num_group_sorted = num_group[np.argsort(num_group)][0:number_per_observation]  # 对这些数值排序并取前n个
        observation.append(toas[sorted_indices])
        indices.append(sorted_indices)
    
    reduced_toas = toas[np.concatenate(indices)].renumber()
    if isinstance(outfile, str) :
        reduced_toas.write_TOA_file(str(outfile)+'.tim',name='unk', format='tempo2', commentflag=None, order_by_index=True, include_pn=True, include_info=None, comment=None)
    return reduced_toas

def resids_offset(resids_initial,ssb,ra,dec,delt_ra,delt_dec):
    c = astropy.constants.c.value
    r_E = ssb
    RA = (ra/180)*np.pi
    DEC = (dec/180)*np.pi
    D_ra = (delt_ra/180)*np.pi
    D_dec = (delt_dec/180)*np.pi
    x = np.cos(RA)*np.cos(DEC)
    y = np.sin(RA)*np.cos(DEC)
    z = np.sin(DEC)
    x_i = np.cos(RA+D_ra)*np.cos(DEC+D_dec)
    y_i = np.sin(RA+D_ra)*np.cos(DEC+D_dec)
    z_i = np.sin(DEC+D_dec)
    delt_x = x_i - x
    delt_y = y_i - y
    delt_z = z_i - z
    Roemer_offset = np.dot(r_E,np.array([delt_x,delt_y,delt_z]).T)
    resids = resids_initial + Roemer_offset*1.e3/c
    return resids

def calculate_new_resids_offset(resids_initial,ssb,ra,dec,delt_ra,delt_dec,mjd,p0,linear):
    c = astropy.constants.c.value
    r_E = ssb
    RA = (ra/180)*np.pi
    DEC = (dec/180)*np.pi
    D_ra = (delt_ra/180)*np.pi
    D_dec = (delt_dec/180)*np.pi
    x = np.cos(RA)*np.cos(DEC)
    y = np.sin(RA)*np.cos(DEC)
    z = np.sin(DEC)
    x_i = np.cos(RA+D_ra)*np.cos(DEC+D_dec)
    y_i = np.sin(RA+D_ra)*np.cos(DEC+D_dec)
    z_i = np.sin(DEC+D_dec)
    delt_x = x_i - x
    delt_y = y_i - y
    delt_z = z_i - z
    Roemer_offset = np.dot(r_E,np.array([delt_x,delt_y,delt_z]).T)
    resids = resids_initial + Roemer_offset*1.e3/c
    predict = mjd* linear[0] + linear[1]
    for i in range(len(resids)):
        if(resids[i] > predict[i]):
            while resids[i] - predict[i] > p0 :
                resids[i] = resids[i] - p0
        else:
            while  predict[i] - resids[i] > p0 :
                resids[i] = resids[i] + p0
    linear_fit = np.polyfit(mjd,resids,1)
    return resids,linear_fit

def generate_grid(range,step,num_thread):
    Nrow = np.ceil(range / step / num_thread).astype(int)
    Ncol = np.ceil(range / step / num_thread).astype(int)
    row = np.arange(-Nrow,(Nrow+1),1) * step * num_thread          #unit:arcsec
    col = np.arange(-Ncol,(Ncol+1),1) * step * num_thread          #unit:arcsec 
    return row.astype(np.float32),col.astype(np.float32)

def generate_new_model(model,RAJ_new,DECJ_new,p0_new,outfile):
    new_model = copy.deepcopy(model)
    getattr(new_model, "RAJ").frozen = True
    getattr(new_model, "DECJ").frozen = True
    getattr(new_model, "F0").frozen = True
    getattr(new_model, "F1").frozen = True
    ra = Angle(angle=RAJ_new,unit='deg')
    dec = Angle(angle=DECJ_new,unit='deg')
    new_model.RAJ.quantity = ra
    new_model.DECJ.quantity = dec
    new_model.F1.quantity = (1/p0_new) * u.Hz
    print(new_model.print_summary)
    if isinstance(outfile, str) :
        new_model.write_parfile(outfile,"wt")
    return new_model

def plot_residuals(toas,resids_initial,new_model,errors):
    f = fitter.Fitter.auto(toas, new_model)
    f.fit_toas()
    resids_plot = []
    resids_plot.append(resids_initial*1.e3)
    resids_plot.append(Residuals(toas,new_model).time_resids.to(u.ms))
    resids_plot.append(f.resids.time_resids.to(u.ms).value)
    mjd = toas.get_mjds()
    plt.figure(figsize=(15,4))
    plt.subplot(1, 3, 1) 
    plt.errorbar(mjd,resids_plot[0],yerr=errors.value,fmt='x')
    plt.legend(loc='upper left')
    plt.title('Prefit residuals')
    plt.subplot(1, 3, 2) 
    plt.errorbar(mjd,resids_plot[1],yerr=errors.value,fmt='x')
    plt.legend(loc='upper centre')
    plt.title('Solution residuals')
    plt.subplot(1, 3, 3) 
    plt.errorbar(mjd,resids_plot[2],yerr=errors.value,fmt='x')
    plt.legend(loc='upper right')
    plt.title('post-PINT-fit residuals')
    plt.errorbar(mjd,resids_initial,yerr=errors,fmt='x')
    plt.savefig('solution.png')

import multiprocessing

def main():
    parser = argparse.ArgumentParser(description='Description of your program')
    # parser.add_argument('arg', help='Description of argument')
    parser.add_argument("parfile", help="par file to read model from")
    parser.add_argument("timfile", help="tim file to read toas from")
    # parser.add_argument('-c', '--ncpus', type=int, help='Number of CPU threads, default = 1',default=1)
    parser.add_argument('-r', '--range', type=float, help='Range of position searching, default = 3 arcminites',default=0.02)
    parser.add_argument('-s', '--step', type=float, help='Precision of position searching, default = 1.0 milliarcsec',default=1.e-3)
    parser.add_argument('-t','--threshold', type=float, help='Threshold of chisquar utoff, default = 2.0',default=2.0)
    parser.add_argument('-o','--outfile', type=str, help='Name of output parfile',default='output')
    args = parser.parse_args()
    
    """Main function of program"""
    parfile = Path(args.parfile)
    timfile = Path(args.timfile)
    outputfile = Path(args.outfile)
    
    search_range = args.range
    search_step = args.step
    num_thread = int(200*32)
    num_block = int(200)
    
    model = models.get_model(parfile,planet_ephem)
    toas = toa.get_TOAs(timfile,planet_ephem)
    reduce_number = 10
    toas = reduce_toa_number(toas,reduce_number,'reduced')
    RAJ = np.float32(model.RAJ.value)
    DECJ = np.float32(model.DECJ.value)
    p0 = np.float32(1/model.F0.value)
    MJD = toas.get_mjds().value
    mean_mjd = np.float32(np.mean(MJD))
    SSB = toas.table['ssb_obs_pos'].value
    SSB_1D = SSB.flatten()
    Resids = Residuals(toas,model).time_resids.to(u.s).value
    Errors = toas.get_errors().to(u.s)
    Errors_inverse = 1/(Errors.value)
    num_linear = int(2*reduce_number)
    num_toa = len(toas)

    mjd_gpu = gpuarray.to_gpu(MJD.astype(np.float64,order='C'))
    ssb_gpu = gpuarray.to_gpu(SSB_1D.astype(np.float32,order='C'))
    resids_gpu = gpuarray.to_gpu(Resids.astype(np.float64,order='C'))
    errors_inverse = gpuarray.to_gpu(Errors_inverse.astype(np.float32,order='C'))
    result_gpu = gpuarray.zeros(num_thread*num_thread*3, dtype=np.float64)
   code_single = string.Template("""
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
    """)
    
    code = code_single.substitute(NTOA=num_toa,Nrange=num_thread,mean_MJD=np.float32(mean_mjd),Nlinear=int(num_linear))
    mod = SourceModule(code)
    calculate = mod.get_function("kernel")
    
    grid_ra,grid_dec = generate_grid(search_range,search_step,num_thread)
    result_all = []
    result_grid = np.empty(num_thread*num_thread*3).astype(np.float64)
    for i in range(len(grid_ra)):
        start_ra = np.float32(grid_ra[i])
        for j in range(len(grid_dec)):
            # result_gpu = gpuarray.empty(num_thread*num_thread*3, dtype=np.float64)
            start_dec = np.float32(grid_dec[j])
            calculate(start_ra, start_dec, search_step, RAJ, DECJ, p0, mjd_gpu, ssb_gpu, resids_gpu, errors_inverse, result_gpu, grid=(num_block,num_block,1),block=(32,32,1))
            result_grid = result_gpu.get()
            chi2_grid = result_grid[0::3]
            ra_grid = result_grid[1::3]
            dec_grid = result_grid[2::3]
            least = chi2_grid.argmin()
            chi2_least = (chi2_grid[least],ra_grid[least],dec_grid[least])
            result_all.append(chi2_least)
            if chi2_least < args.threshold:
                solution = chi2_least
                linear1_k,linear1_b = np.polyfit(MJD[0:2*reduce_number],Residuals[0:2*reduce_number],1)
                resids_phase_predict = calculate_new_resids_offset(Resids,SSB,RAJ,DECJ,solution[1],solution[2],MJD,p0,(linear1_k,linear1_b))
                k0,b0 = np.polyfit(MJD,resids_phase_predict,1)
                p0_new = p0 * (1 + k0 / (24*3600))
                new_model = generate_new_model(model,solution[1],solution[2],p0_new,outputfile)
                plot_residuals(toas,Resids,new_model)
                break
        print ("RA_offset: %f ~ %f"  % (start_ra - num_thread*search_step*0.5 - RAJ,start_ra + num_thread*search_step*0.5-RAJ))
        print ("DEC_offset: %f ~ %f"  % (start_dec - num_thread*search_step*0.5-DECJ, start_dec + num_thread*search_step*0.5-DECJ))
        print ("least chi2 squar is %f" % chi2_least[0])
    
