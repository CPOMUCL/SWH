#! /home/hds/.conda/envs/harry/bin/python3
# SWH collection script

# for SWH collection

# class and defs for processing the CryoSat tracks into DOT
import numpy as np
import pandas as pd
import datetime as dt
import copy
import itertools
import glob
import sys
import gc
import SWH_collect_prh as sc
import data_year as dy
import grid_set as gs
from netCDF4 import Dataset
from dateutil.relativedelta import relativedelta
from mpl_toolkits.basemap import Basemap
from astropy.convolution import convolve
from astropy.convolution import kernels


# set grid - 50 km
t_start = dt.datetime(2014,2,1)
tp = 3*30 -1 # 365
# dyp = tp * 4
dyp = 365
period = relativedelta(days=1)

# SWH_dir =  '/raid6/userdata/hds/SWH_nc/'
# SWH_dir =  '/home/hds/Python/SWH/SWH_nc_dir/'
SWH_dir =  '/home/hds/Python/SWH/SWH_nc_dir/'
save_info = 'SWH_100km_1d_2014_fixed'
save_dir = '/home/hds/Python/SWH/swh_arrays/'

m = Basemap(projection='stere', lon_0=0.0, lat_0=90, lat_ts=70, 
                height = 3335000*2, width = 3335000*2)
this_day = sc.wave_array(period,t_start,m)
this_day.set_grid_dxy(dxRes=100e3,dyRes=100e3)
# this_day.get_grid_info()
# this_day.save_grid('/Users/H/WAVES/DOT_processing/grids/Polar_stereo_100km.npz')
this_day.load_grid('/home/hds/Python/SWH/grids/Polar_stereo_100km.npz')
# this_day.get_grid_mask()
# this_day.save_mask('/Users/H/WAVES/DOT_processing/grids/Polar_stereo_100km_mask_no_inf.npz')
this_day.load_mask('/home/hds/Python/SWH/grids/Polar_stereo_100km_mask_no_inf.npz')


swh_all = np.empty([tp,this_day.m,this_day.n])
swh_fill= np.empty([tp,this_day.m,this_day.n])
swh_SAR_bin = np.empty([tp,this_day.m,this_day.n])
swh_LRM_bin = np.empty([tp,this_day.m,this_day.n])
swh_SAR_binc = np.empty([tp,this_day.m,this_day.n])
swh_LRM_binc = np.empty([tp,this_day.m,this_day.n])
dates = []

for t in range(tp):
    this_day.find_tracks(SWH_dir)
    dates.append(this_day.time)
    print(this_day.time.strftime('%Y-%m-%d'))

    # this_day.load_tracks(LRM_lim=10e-3,SAR_lim=10e-4,check_error=True)
    this_day.load_tracks(LRM_lim=2e-3,SAR_lim=4e-4,check_error=True) # Toms set up

    this_day.LRM.binned=False
    this_day.SAR.binned=False
    this_day.bin_tracks()

    this_day.get_swh_weights(mask=True,save_w=True)
    this_day.fill_swh(radius=250e3)
    
    swh_all[t] = this_day.swh
    swh_fill[t] = this_day.swh_fill
    swh_SAR_bin[t] = this_day.SAR.swh_bin
    swh_LRM_bin[t] = this_day.LRM.swh_bin
    swh_SAR_binc[t] = this_day.SAR.swh_bin_count
    swh_LRM_binc[t] = this_day.LRM.swh_bin_count
    this_day.saved =True
    this_day.next_time()
    
swh_all_dy = dy.data_year(swh_all,dates,dyp)
swh_fill_dy= dy.data_year(swh_fill,dates,dyp)
swh_SAR_bin_dy = dy.data_year(swh_SAR_bin,dates,dyp)
swh_LRM_bin_dy= dy.data_year(swh_LRM_bin,dates,dyp)
swh_SAR_binc_dy = dy.data_year(swh_SAR_binc,dates,dyp)
swh_LRM_binc_dy= dy.data_year(swh_LRM_binc,dates,dyp)
swh_all_dy.save(save_dir+save_info+'_raw.npz')
swh_fill_dy.save(save_dir+save_info+'_filled.npz')
swh_SAR_bin_dy.save(save_dir+save_info+'_SAR_bin.npz')
swh_LRM_bin_dy.save(save_dir+save_info+'_LRM_bin.npz')
swh_SAR_binc_dy.save(save_dir+save_info+'_SAR_binc.npz')
swh_LRM_binc_dy.save(save_dir+save_info+'_LRM_binc.npz')


# set time and period
