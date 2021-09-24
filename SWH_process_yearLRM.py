#! /home/hds/.conda/envs/harry/bin/python3

# quick dirty processing
# lets open a netcdf l1b file
import numpy as np
import pandas as pd
import datetime as dt
import glob
import os
from netCDF4 import Dataset
from scipy import stats
from scipy import interpolate
from scipy.optimize import fsolve, minimize
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import gc
# import SWH
import SWH_wfm
import SWH_tables
from numba import jit

# batch processing of SWH from l1b files

# directory to process 
# root directory above the day/month structure
# l1b_dir = '/cpnet/li1_cpdata/SATS/RA/CRY/processed/arco/201403_SAR/rads/'
l1b_root = '/cpnet/li1_cpdata/SATS/RA/CRY/processed/arco/'
# l1b_dir = '/cpdata/SATS/RA/CRY/cryosat_Baseline-D_TDS/L1B/SAR/2014/03/'

# directory to save in
save_root= 'SWH_nc_dir/LRM/'
# save_dir= '/raid6/userdata/hds/SWH_nc/SAR/2014/02/'
# save_dir= '/home/hds/SWH_nc/LRM/2014/02/'

# set year to process
year = 2012

# position limits
lon_lims = [-180,180]
lat_lims = [60,89]# 
# data no limit - important for filtering peakiness
no_wf_lim = 5


print(l1b_root)

# files = glob.glob(l1b_dir+'*.wfm')
# files = glob.glob(l1b_dir+'*.nc')

# print('files in dir = ',np.shape(files)[0])

# load the lookup tables LRM
LU_dir = '/home/hds/Python/SWH/LookupTables/Original/'
LU_PL_h0_f = LU_dir+'/PL h0'
LU_PL_h2_f = LU_dir+'/PL h2'
LU_PL_h11_f = LU_dir+'/PL h11'
LU_PL_h12_f = LU_dir+'/PL h12'

LRMEcho = SWH_tables.ideal_echo()
LRMEcho.load_tables(LU_PL_h0_f,LU_PL_h11_f,
                    LU_PL_h12_f,LU_PL_h2_f)
LRMEcho.scale = 5.5e6
LRMEcho.sampling = 1/(3.2e8) * 1e9

file = LU_dir+'LRMweights.txt'
read = np.empty([128])
p = 0
LRM_f = open(file,'r')
for line in LRM_f:
    ls = line.split('    ')
    for n in ls:
        try:
            nf = float(n)
        except ValueError:
            pass
        else:
            # save them if successful
            read[p] = nf
            p +=1 
LRM_f.close()
LRMEcho.weight = read[19:79]/np.max(read)
LRMEcho.weight_s = np.sum(LRMEcho.weight)
#

# load the lookup tables SAR
LU_SAR_rec_b51_h0_f = LU_dir+'/SAR Rectangular 51 beams h0 V2'
LU_SAR_rec_b51_h2_f = LU_dir+'/SAR Rectangular 51 beams h2 V2'
LU_SAR_rec_b51_h11_f = LU_dir+'/SAR Rectangular 51 beams h11 V2'
LU_SAR_rec_b51_h12_f = LU_dir+'/SAR Rectangular 51 beams h12 V2'

SAREcho = SWH_tables.ideal_echo()
SAREcho.load_tables(LU_SAR_rec_b51_h0_f,LU_SAR_rec_b51_h11_f,
                     LU_SAR_rec_b51_h12_f,LU_SAR_rec_b51_h2_f)
SAREcho.scale = 4.5e8
SAREcho.sampling = 1/(2*3.2e8) * 1e9

# open sarweights and lrmweights
file = LU_dir+'SARweights.txt'
read = np.array(np.loadtxt(file))
SAREcho.weight = read[29:89]/np.max(read)
SAREcho.weight_s = np.sum(SAREcho.weight)

# all the months
# for mn in range(7,12):
for mn in range(0,8):
    # build the directory we want to save in
    #save_root= 'SWH_nc_dir/SAR/' add /yyyymm/
    
    dt_mn = dt.datetime(year,mn+1,1)
    # save_root= 'SWH_nc_dir/SAR/' # add /yyyymm/
    save_dir = os.path.dirname(
        save_root+dt_mn.strftime('%Y%m')+'/.')
    print(dt_mn.strftime('%Y%m'))
    print(save_dir)
    # check it exists
    if not os.path.exists(save_dir):
    # make if it doesn't
        os.makedirs(save_dir)
        print('Creating directory: ',save_dir)
    else: print('directory: ',save_dir)
    
    
    # find the files to process
    l1b_dir = l1b_root+dt_mn.strftime('%Y%m')+'_LRM/rads/'
    
    print(l1b_dir)

    files = glob.glob(l1b_dir+'*.wfm')
    # files = glob.glob(l1b_dir+'*.nc')

    print('files in dir = ',np.shape(files)[0])


    f_tot = np.shape(files)[0]
    use =False

# for f in range(1000):
# for f in range(10):
    for f in range(f_tot):
        # this_path = SWH.l1b_track(files[f],lon_lims,lat_lims,'SAR',time_open='20_ku')
        this_path = SWH_wfm.l1b_track(files[f],lon_lims,lat_lims,'LRM')
        use = this_path.use
        if use:
            if this_path.n_p<no_wf_lim: use = False
        if use: 
            # for dev - make sure it's of reasonable size!
            # if this_path.n_p<no_wf_lim: use = False

            print('file',f,'of',f_tot,'waves to use =',this_path.n_p)
        # extracting the usable open ocean waves
            # this_path.open_echos()
        # diagnostics on these waves
            this_path.get_noise_floor()
            this_path.get_peakiness()
            # this_path.get_power_std()
        # down sampling and some filtering
            this_path.filter_for_SWH(reduce_WF=20,LE_lim=0.5)
            print('calculating SWH')
        # the main retracking routine
            this_path.get_SWH(LRMEcho,[1.0,0.1,28.0],method='sig_t_A',
                    status = 3 ,save_err=True)
            # save the file
            this_path.save_swh_nc(save_dir+'/',SAREcho,method='sig_t_A',
        save_err=True,reduce_WF=20,m0=[1.0,0.1,28.0])

    #         print('saving in',save_dir+'/')


            del this_path.pWF
            del this_path.pWFrt
            del this_path
            gc.collect()


