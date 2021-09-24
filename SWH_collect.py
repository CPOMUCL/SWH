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
sys.path.insert(0, '/Users/H/WAVES/geo_data_group/')
import data_year as dy
import grid_set as gs
from netCDF4 import Dataset
from invoke import run
from numba import jit
from scipy import stats
from dateutil.relativedelta import relativedelta
from mpl_toolkits.basemap import Basemap
from astropy.convolution import convolve
from astropy.convolution import kernels


# set grid - 50 km
t_start = dt.datetime(2014,2,1)
tp = 3 * 3
dyp = tp * 4
period = relativedelta(days=1)

save_info = 'SWH_100km_10_day'

m = Basemap(projection='stere', lon_0=0.0, lat_0=90, lat_ts=70, 
                height = 3335000*2, width = 3335000*2)
this_day = wave_array(period,t_start,m)
# this_day.set_grid_dxy(dxRes=100e3,dyRes=100e3)
# this_day.get_grid_info()
# this_day.save_grid('/Users/H/WAVES/DOT_processing/grids/Polar_stereo_100km.npz')
this_day.load_grid('/Users/H/WAVES/DOT_processing/grids/Polar_stereo_100km.npz')
# this_day.get_grid_mask()
# this_day.save_mask('/Users/H/WAVES/DOT_processing/grids/Polar_stereo_100km_mask_no_inf.npz')
this_day.load_mask('/Users/H/WAVES/DOT_processing/grids/Polar_stereo_100km_mask_no_inf.npz')

save_dir = 'First_saves_2014/'

swh_all = np.empty([tp,this_day.m,this_day.n])
swh_fill= np.empty([tp,this_day.m,this_day.n])
swh_SAR_bin = np.empty([tp,this_day.m,this_day.n])
swh_LRM_bin = np.empty([tp,this_day.m,this_day.n])
swh_SAR_binc = np.empty([tp,this_day.m,this_day.n])
swh_LRM_binc = np.empty([tp,this_day.m,this_day.n])
dates = []

for t in range(tp):
    this_day.find_tracks('/Volumes/BU_extra/CryoSat/SWH/SWH_nc/')
    dates.append(this_day.time)
    print(this_day.time.strftime('%Y-%m-%d'))

    this_day.load_tracks(LRM_lim=2e-3,SAR_lim=2e-4,check_error=True)

    this_day.LRM.binned=False
    this_day.SAR.binned=False
    this_day.bin_tracks()

    this_day.get_swh_weights(mask=True,save_w=True)
    this_day.fill_swh(radius=250e3)
    
    swh_all[t] = this_day.swh
    swh_fill[t] = this_day.swh_fill
    swh_SAR_bin[t] = this_day.swh.SAR.swh_bin
    swh_LRM_bin[t] = this_day.swh.LRM.swh_bin
    swh_SAR_binc[t] = this_day.swh.SAR.swh_bin_count
    swh_LRM_binc[t] = this_day.swh.LRM.swh_bin_count
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
class wave_array(gs.grid_set):
    def __init__(self,period,time_start,mplot):
        super().__init__(mplot)
        self.time = time_start
        self.period = period 
        # here we can do clever time period stuff 
        # to make sure we fill months/years correctly
        self.files = False
        self.saved = False
        self.masked = False
        self.data = False
         
    def find_tracks(self,CS2_path):
	# given the time and the period return a list of the tracks
	# make sure we save that we have the tracks
        LRM_path = CS2_path+'LRM/'
        self.LRM   = Track_list(self.time, self.period, LRM_path,'/CS')
        print(len(self.LRM.files),'LRM files')
        SAR_path = CS2_path+'SAR/'
        self.SAR   = Track_list(self.time, self.period, SAR_path,'/CS')
        print(len(self.SAR.files),'SAR files')
#         self.SARIN = Track_list(self.time, self.period, CS2_path,'_SARIN/elev/CS')
        self.files = True
    
    def load_tracks(self,LRM_lim =False,SAR_lim =False,check_error =False):
        # from the track files get data and projection
        self.LRM.get_data_from_CS_list(this_day.mplot,error_lim = LRM_lim,check_error=check_error)
        self.SAR.get_data_from_CS_list(this_day.mplot,error_lim = SAR_lim,check_error=check_error)
        
        
    def bin_tracks(self,error_lim =False):
        # from the track files get data and projection
        self.LRM.bin_the_tracks(this_day)
        self.SAR.bin_the_tracks(this_day)

    def get_swh_weights(self,weight_lim = 1,mask=False,save_w=False,
        mode_mask=False):
        """
        Added a weight lim to filter data holes a little better
        Also save weight and mode mask flags.
        Save_weight so to create the mode mask -  see other methods
        mode_mask = True to use mode mask when processing DOT
        """
        total_w = (self.LRM.swh_bin_count+
                   self.SAR.swh_bin_count)
        LRM_w = self.LRM.swh_bin_count/total_w
        SAR_w = self.SAR.swh_bin_count/total_w
        if save_w:
            self.LRM_w = LRM_w
            self.SAR_w = SAR_w
            self.total_w = total_w
        if mode_mask:
            LRM_mask = self.LRM_mask
            SAR_mask = self.SAR_mask
        else:
            LRM_mask = np.ones([self.m,self.n])
            SAR_mask = np.ones([self.m,self.n])
        LRMdw = self.LRM.swh_bin.copy()
        SARdw = self.SAR.swh_bin.copy()
        LRMdw = LRMdw*LRM_w
        SARdw = SARdw*SAR_w
        LRMdw[np.isnan(LRMdw)] = 0.0
        SARdw[np.isnan(SARdw)] = 0.0
        SAR_w[np.isnan(SAR_w)] = 0.0
        temp_mask = np.ones([self.m,self.n])
        # make sure there were enough binned data to use
        temp_mask[total_w<weight_lim] = np.nan
#         for i in range(self.m):
#             for j in range(self.n):
#                 if total_w[:,i,j]<1: temp_mask[i,j] = np.nan
        self.swh = np.empty([self.m,self.n])
        self.swh[:] = (LRMdw*LRM_mask+ 
                SARdw*SAR_mask )
                  # offsets if needed + self.SAR_off_c[mn]*SAR_w[tt]*SAR_mask )
        self.swh_mask = temp_mask
        if mask:
            self.swh[total_w<weight_lim] = np.nan

    def fill_swh(self,radius):
        """
        experimental convolution to fill the gaps
        requires masking afterward
        """
        self.swh_fill = gs.geo_convolve(self.swh,self,
                                radius,[0.0,10.0],mask=False)*self.mask
        
        self.swh_fill[self.lats.T>88] = np.nan
        self.swh_fill[self.lats.T<60] = np.nan
        # also convolve the weights
        w_fill = gs.geo_convolve(self.total_w*self.swh_mask,self,
                                 radius/2,[0.0,10.0],mask=False)
        w_mask = np.ones_like(w_fill)
        w_mask[np.isnan(w_fill)] = np.nan
        w_mask[w_fill<2.0] = np.nan
        self.swh_fill_mask = w_mask
        self.swh_fill      = self.swh_fill*w_mask

    def next_time(self):
    # delete everyhting made sofar if saved
        if self.saved:
            
            # be a bit more clever about clearing arrays
            # maybe reuse??
            # Tracks
            self.LRM.clear
            del self.LRM
            self.SAR.clear
            del self.SAR
            # flags
            self.files = False
            self.LRM_tracks = False
            self.SAR_tracks = False
            self.binned = False
            self.offset = False
            self.saved = False
            
            gc.collect()

            # next time point
            self.time += self.period
            
            

class Track_list:
# this function gets the files 
# now in a dataframe for speed - individual arrays no longer used
    def __init__(self,time_start, period, CS2_path,type_path):
        files = []
        self.time = time_start
        self.period = period
        self.read = False
        self.binned = False
        self.g_files = False
        time_use = time_start
        while (time_use - (time_start + period)).days < 0:
            files.append(glob.glob(CS2_path+time_use.strftime('%Y/%m')
                +type_path+'_*_'+time_use.strftime('%Y%m%d')+'*.nc'))
            time_use += relativedelta(days=1)
        #load in files 
        if (np.size(files)==0):
            print ('no data this year')
            self.found_files = False
        else:
            self.found_files = True
            # finish by flattening the list of files
            self.files = list(itertools.chain(*files))

    def get_data_from_CS_list(self,mplot,error_lim = False,check_error = False):       
        # we need to get:
        # swh, lon, lat
        # then error filtering
        # get projection coords too
        d_tot = 0
        e_tot = 0
        self.swh= []
        self.fit= []
        self.lon= []
        self.lat= []
        for f in self.files:
            # load the track
            track = Dataset(f)
            # and the variables
            swh = track['swh']
            pwA = track['pwA']
            fit_error = track['fit_error']
            lon = track['lon']
            lat = track['lat']
            if check_error: d_tot += np.shape(lon)[0]
            if type(error_lim) == float:
                msk = fit_error[:]<error_lim
            else:
                msk = np.ones_like(fit_error,dtype=bool)
            if check_error: e_tot += np.shape(lon)[0] - np.sum(msk) 
            # fill up the list
            [self.swh.append(s) for s in swh[msk]]
            [self.fit.append(s) for s in fit_error[msk]]
            [self.lon.append(s) for s in lon[msk]]
            [self.lat.append(s) for s in lat[msk]]
            track.close()
        if check_error:
            print(d_tot,'data points',e_tot,'removed')
            
        self.xpts, self.ypts = mplot(self.lon,self.lat)
        
        self.read = True

    def bin_the_tracks(self,CS_track,SH=False):
        # takes the track data read in and bins it on the parent grid
        # the info needed for binning is xpts info - nx/ny and range - x/y min/max from parent
        # need to produce mean and count from measurements
        if self.binned:
            print('Binned already!!!')
        else:
            edges_x = CS_track.xpts[0,:] - CS_track.dxRes/2
            edges_y = CS_track.ypts[:,0] - CS_track.dyRes/2
            if SH:
                edges_x = np.append(edges_x,CS_track.xpts[0,-1] - CS_track.dxRes)
                edges_y = np.append(edges_y,CS_track.ypts[-1,0] - CS_track.dyRes)
                edges_x = np.flip(edges_x)
                edges_y = np.flip(edges_y)
            else:
                edges_x = np.append(edges_x,CS_track.xpts[0,-1] + CS_track.dxRes)
                edges_y = np.append(edges_y,CS_track.ypts[-1,0] + CS_track.dyRes)
                # change the last entry and flip
#                 edges_x[-1] = edges_x[-1] - 1.5*CS_track.dxRes
#                 edges_x[-1] = edges_x[-1] - 1.5*CS_track.dxRes
            # print(edges_x)
            # print(edges_y)
            ret = stats.binned_statistic_2d( self.xpts, self.ypts, self.swh,
	    		statistic='mean', bins=[edges_x,edges_y]) #150km bins at this resolution
            if SH: self.swh_bin= np.flip(ret.statistic)
            else:  self.swh_bin= ret.statistic
            ret = stats.binned_statistic_2d( self.xpts, self.ypts, self.fit,
	    		statistic='mean', bins=[edges_x,edges_y]) #150km bins at this resolution
            if SH: self.fit_bin= np.flip(ret.statistic)
            else:  self.fit_bin= ret.statistic
            ret = stats.binned_statistic_2d( self.xpts, self.ypts, None,
	    		statistic='count', bins=[edges_x,edges_y]) #150km bins at this resolution
            if SH: self.swh_bin_count = np.flip(ret.statistic)
            else:  self.swh_bin_count = ret.statistic
            
            self.binned = True
       
    def clear(self,geoid=True):
        # empty the pd dfs for the next time step
        if self.read and self.binned:
            self.pd_cs = self.pd_cs.iloc[0:0]
            del self.lon
            del self.lat
            del self.xpts
            del self.ypts
            del self.swh
            del self.fit
