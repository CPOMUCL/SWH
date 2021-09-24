
# classes and functions for SWH processing

import numpy as np
import datetime as dt
import glob
from netCDF4 import Dataset
from scipy import stats
from scipy import interpolate
from scipy.optimize import minimize
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
# from mpl_toolkits.basemap import Basemap
from numba import jit


# need 20hz/1hz option
# normalisation
# match
# save


class l1b_track:
    """
    # open a Lib file, check if it is in the desired region, 
    # if false, close the file, return false
    # else open it and get ready....
    # surface_type/lon-lat, oftern different sizes,
    # will assume they start together
    # set time_open for the data to read
    #      default is variable_'avg_01_ku'
    #      also available is variable_'20_ku'
    """
    def __init__(self,file,lon_lims,lat_lims,mode,st = 0,time_open = "avg_01_ku"):
        self.name = file
        self.mode = mode
        self.var_ext = time_open
        self.swh_calc = False
        data_nc = Dataset(file)
        surf_type = data_nc.variables["surf_type_01"]
        lats = data_nc.variables["lat_avg_01_ku"]
        lons = data_nc.variables["lon_avg_01_ku"]
        n_p = np.minimum(
            surf_type.shape[0],lats.shape[0])
        st_correct = (surf_type[:] == st)
        lon_correct = (lons[:] > lon_lims[0])&(lons[:] < lon_lims[1] )
        lat_correct = (lats[:] > lat_lims[0])&(lats[:] < lat_lims[1] )
        d_u = st_correct[0:n_p]&lon_correct[0:n_p]&lat_correct[0:n_p]
        n_p = n_p
#         print(np.sum(d_u))
        if np.sum(d_u)>2:
            lons = data_nc.variables["lon_"+self.var_ext]
            lats = data_nc.variables["lat_"+self.var_ext]
            if self.var_ext == '20_ku':
                d_u = np.repeat(d_u,20)
                n_p = n_p*20
                if n_p>lats.shape[0]:
                    n_p = np.minimum(n_p,lats.shape[0])
                    d_u = d_u[0:n_p]
                if self.mode == 'LRM':
                    self.w_bin = 128
                else:
                    self.w_bin = 256
            else:
                self.w_bin = 128
            print(n_p)
            self.use = True
            self.lons = lons[0:n_p][d_u]
            self.lats = lats[0:n_p][d_u]
            self.n_p = n_p
            self.n_u = np.sum(d_u)
            self.d_u = d_u
            data_nc.close()
        else:
            self.use = False
            data_nc.close()
            
#     def scale_locations():
        # DOESN"T SEEM TO BE A PROBLEM
        # if location is correct
        # lin interpolate lat/lon to get correct location
        
    def open_echos(self):
        """
        # ;define the secondary parameters to read (if file is suitable)
        # read_params = ['power_echo_waveform','data_record_time','echo_scale_power','echo_scale_factor']
        """
        if self.use:
            data_nc = Dataset(self.name)
            power_WF= data_nc.variables["pwr_waveform_"+self.var_ext]
            data_T  = data_nc.variables["time_"+self.var_ext]
            echo_SF = data_nc.variables["echo_scale_factor_"+self.var_ext]
            echo_SP = data_nc.variables["echo_scale_pwr_"+self.var_ext]
            alt     = data_nc.variables["alt_"+self.var_ext]
            roll    = data_nc.variables["off_nadir_roll_angle_str_20_ku"]
            pitch   = data_nc.variables["off_nadir_pitch_angle_str_20_ku"]
            # then some conversion - times
            day0 = dt.datetime(2000,1,1)
            day0_secs = day0.timestamp()
            self.times = [dt.datetime.fromtimestamp(dT + day0_secs)
                          for dT in data_T[0:self.n_p][self.d_u]]
            # only the filtered times
            self.pWF = power_WF[0:self.n_p][self.d_u]
            self.eSF = echo_SF[0:self.n_p][self.d_u]
            self.eSP = echo_SP[0:self.n_p][self.d_u]
            self.alt = alt[0:self.n_p][self.d_u]
            self.roll = roll[0:self.n_p][self.d_u]
            self.pitch = pitch[0:self.n_p][self.d_u]
            # extra info for saving
            self.orbit_no = data_nc.getncattr('abs_orbit_number')
            self.echos = True
            data_nc.close()
            
            # slice if this is SARIN - future as we only have SAR at the mo
#     def filer_waves(self):
            # CHECK MASKING ON ALL THIS
            # Find the peak power standard deviation - echo_SF/SP
            
            #  ;define diffuse waveforms from peakiness and power standard deviation thresholds
            # 'LRM' THEN diff = where(peak lt 4.0 AND power_std lt 0.07)
            # 'SAR' THEN diff = where(peak lt 6.0 AND power_std LT 0.1)   
            # 'SIN' THEN diff = where(peak lt 6.0 AND power_std LT 0.15)      
            
            # ;For a move window 100 samples wide find whether there is more than one
            # ;diffuse echo present and if so set the subset flag to 1
    def get_noise_floor(self,window=[10,20]):
        self.nf = (np.mean(self.pWF[:,window[0]:window[1]],axis=1)+
              3*np.std(self.pWF[:,window[0]:window[1]],axis=1))
        self.over_nf = np.zeros([self.n_u,self.w_bin],dtype = bool)
        self.over_nf[:,:] = [self.pWF[i,:].data < self.nf[i] for i in range(self.n_u)]
        
    def get_peakiness(self,filter_window = 21):
        temp = np.ma.array(self.pWF,mask=self.over_nf)
        self.peaky = np.max(temp.data,axis = 1)*np.sum(self.over_nf,axis=1)/np.sum(temp,axis = 1)
        w = min((filter_window,self.n_u))
        # w has to be odd
        w=w+np.mod(w,2)-1
        self.peak = savgol_filter(self.peaky,w,4)
            
    def get_power_std(self,window = 10):
        w = min((window,int(self.n_u/2)))
        scale_factor =  np.log10(self.eSF*(2.0**self.eSP))
        self.power_std = [np.std(scale_factor[:p+w][-2*w:]) 
                     for p in range(self.n_u)]

    def filter_for_SWH(self,reduce_WF = False,LE_lim = 0.7,check_window=50,check_lim=2):
        """
        # returns the pWFrt and rt_mask,
        # used to have the windowed pWF ready for retracking
        # rt_mask is a logical list giving which entries from n_u list to be used
        # rt_mask - useful for saing lon/lat etc values later
        
        # when reducing WF rt_mask will only reference 
        
        """
        # extra constants
        day0 = dt.datetime(2000,1,1)
        day0_secs = day0.timestamp()
        
        w = min((check_window,int(self.n_u/2)))
        if self.mode == 'LRM':
            p_lim = 4.0 
            p_std = 0.07
            w_use = [18,78]
            LE0 = 40
        elif self.mode == 'SAR':
            p_lim = 6.0 
            p_std = 0.1
            w_use = [28,88]
            LE0 = 50
        elif self.mode == 'SIN':
            p_lim = 6.0 
            p_std = 0.15
            w_use = [28,88]
            LE0 = 50
        else:
            print('mode: '+self.mode+' not recognised, using LRM')
            p_lim = 4.0 
            p_std = 0.07
            w_use = [18,78]
            LE0 = 40
        # pulse peakiness and power running std criterior
        diff = [self.peak[i]<p_lim and self.power_std[i] < p_std
                for i in range(self.n_u)]
        # pass those within a window
        extra_check = [False] * self.n_u
        for i in range(self.n_u):
            if sum(diff[:i+w][-2*w:])>min((check_lim,i+w-1)):
                extra_check[i] = True
        self.rt_mask = extra_check
        maxpWF = np.max(self.pWF[:,w_use[0]:w_use[1]],axis=1)
        if type(reduce_WF) == int: 
            # reduce the wf's by reduce_WF window
            # copy waves to a new array
            n_rWF = int(self.n_u/reduce_WF)
            self.n_rt = n_rWF
            self.pWFrt = np.empty([n_rWF,60])
            self.lonrt = np.empty([n_rWF])
            self.latrt = np.empty([n_rWF])
            #extra info to average times
            self.timesrt = np.empty([n_rWF])
            self.pitchrt = np.empty([n_rWF])
            self.rollrt = np.empty([n_rWF])
            self.altrt = np.empty([n_rWF])
            # new array to save peakiness
            self.pprt = np.empty([n_rWF])
            self.pprt[:] = np.nan
            for i in range(n_rWF):
                # reduce elements i*rWF:(i+1)*rWF
                # only if rt_mask says so
                # and move WF to leading edge
                wf_use = [reduce_WF*i + j 
                    for j,n in enumerate(self.rt_mask[reduce_WF*i:reduce_WF*(i+1)+1]) if n]
#                 print(wf_use)
                LE_t= [find_LE(self.pWF[j],LE_lim) 
                        for j in wf_use]
                LE_av = np.nanmean([self.pWF[j,w_use[0]+LE_t[k]-LE0:w_use[1]+LE_t[k]-LE0]
                        for k,j in enumerate(wf_use)
            # extra check to avoid those outside a sensible window to Leading edges
                        if (w_use[0]+LE_t[k]-LE0>1)and(w_use[1]+LE_t[k]-LE0<127)],axis=0)
                LE_av = LE_av/np.nanmean([maxpWF[j] for j in wf_use])
                # save the new LE 'aved WF
                self.pWFrt[i,:] = LE_av
                # save the reduced lon/lat
                self.lonrt[i] = np.mean([self.lons[j] for j in wf_use])
                self.latrt[i] = np.mean([self.lats[j] for j in wf_use])
                self.timesrt[i] = np.mean([self.times[j].timestamp() - day0_secs for j in wf_use])
                self.altrt[i] = np.mean([self.alt[i] for i in wf_use])
                self.rollrt[i] = np.mean([self.roll[i] for i in wf_use])
                self.pitchrt[i] = np.mean([self.pitch[i] for i in wf_use])
                self.pprt[i] = np.mean([self.peaky[i] for i in wf_use])
                # need an extra list to only process actualy exisiting pWFrt
            m_check = np.isfinite(self.pWFrt)
            rt_use = np.sum(m_check,axis=1)>0
            self.rt_use = [i for i,n in enumerate(np.sum(m_check,axis=1)) if n>0]
        else:
            self.n_rt = self.n_u
            self.pWFrt = np.empty([self.n_rt,60])
            self.pWFrt[:,:] = [self.pWF[i,w_use[0]:w_use[1]]/maxpWF[i] for i,n in enumerate(extra_check) if n]
            m_check = np.isfinite(self.pWFrt)
            self.rt_use = [i for i,n in enumerate(np.sum(m_check,axis=1)) if n>0]
            
            

        
# now need a method that matches the waveforms and save the parameters
# parameters to match:
# sig - yes, maybe: A, t0, 
    def get_SWH(self,ideal_echo,start_values,method = 'sig',status = False,save_err=False,orbitinfo = False):
        """
        methods: (sig), sig_t, sig_t_A
        for searching 1, 2, 3, params respectively
        start value = [*] for 1,2,3 points
        ranges = [(*,*)] for 1,2,3 points
        if searching for 1 param, method 'sig'
            the start value of the other two params are kept constant
            similarly for searching for sig, t
        set status to int for 'status' number progress messages
        """
        if type(status) == int:
            print_status = True
            step = np.int(self.n_rt/status)
        else: 
            step = 1000
            print_status = False
        self.pwA = np.empty([self.n_rt])
        self.sig = np.empty([self.n_rt])
        self.wt0 = np.empty([self.n_rt])
        self.pwA[:] = np.nan
        self.sig[:] = np.nan
        self.wt0[:] = np.nan
        if save_err:
            self.fit_err = np.empty([self.n_rt])
            self.fit_err[:] = np.nan
            # solve  - various options 
        if method == 'sig':
            print('Calculating 1 parameter SWH fit')
            for i in self.rt_use:
                if orbitinfo:
                    OIp = np.deg2rad(self.pitchrt[i])
                    OIr = np.deg2rad(self.rollrt[i])
                    OIdh = self.altrt[i] - ideal_echo.h
                else:
                    OIp = 0.0
                    OIr = 0.0
                    OIdh = 0.0
                self.pwA[i],self.sig[i],self.wt0[i]   = ideal_echo.wavefit_1(
                    self.pWFrt[i],start_values[0],start_values[1],start_values[2],
                    p=OIp,r=OIr,delh=OIdh,mode=self.mode)
                if (np.mod(i,step) == 0) and print_status:
                    print(i,'of',self.n_rt,'m =',self.pwA[i],self.sig[i],self.wt0[i])
        elif method == 'sig_t':
            print('Calculating 2 parameter SWH fit')
            for i in self.rt_use:
                if orbitinfo:
                    OIp = np.deg2rad(self.pitchrt[i])
                    OIr = np.deg2rad(self.rollrt[i])
                    OIdh = self.altrt[i] - ideal_echo.h
                else:
                    OIp = 0.0
                    OIr = 0.0
                    OIdh = 0.0
                self.pwA[i],self.sig[i],self.wt0[i]   = ideal_echo.wavefit_2(
                    self.pWFrt[i],start_values[0],start_values[1],start_values[2],
                    p=OIp,r=OIr,delh=OIdh,mode=self.mode)
                if (np.mod(i,step) == 0) and print_status:
                    print(i,'of',self.n_rt,'m =',self.pwA[i],self.sig[i],self.wt0[i])
        elif method == 'sig_t_A':
            print('Calculating 3 parameter SWH fit')
            for i in self.rt_use:
                if orbitinfo:
                    OIp = np.deg2rad(self.pitchrt[i])
                    OIr = np.deg2rad(self.rollrt[i])
                    OIdh = self.altrt[i] - ideal_echo.h
                else:
                    OIp = 0.0
                    OIr = 0.0
                    OIdh = 0.0
                self.pwA[i],self.sig[i],self.wt0[i]   = ideal_echo.wavefit_3(
                    self.pWFrt[i],start_values[0],start_values[1],start_values[2],
                    p=OIp,r=OIr,delh=OIdh,mode=self.mode)
                if (np.mod(i,step) == 0) and print_status:
                    print(i,'of',self.n_rt,'m =',self.pwA[i],self.sig[i],self.wt0[i])
        
        if save_err:
            print('Calculating errors')
            for i in self.rt_use:
                self.fit_err[i] = ideal_echo.wave_fit_mf(self.pWFrt[i],self.pwA[i],self.sig[i],self.wt0[i],mode=self.mode)
        self.swh_calc = True

    def save_swh_nc(self,directory,ideal_echo,method,save_err=False,reduce_WF=20,m0=[1.0,0.1,30.0]):
        """
        saves the processed track as a netcdf file
        adds all the attributes you'll need for later use
        exta variables for saving info:
        ideal_echo so to save it's name and files used for it
        reduce_WF option should be set as the previous filter method
        m0 saves the first guess for parameter search, same as set in the get SWH method
        """
        if self.swh_calc:
            # then save the track as an nc file
            # file name  -  based on dates
            # elev file: CS_LTA__SIR_LRM_2__20140101T004230_20140101T004421_C001.elev
            file = ('CS_SWH_'+method+'_'+self.mode+'_'+
                    self.times[0].strftime('%Y%m%dT%H%M%S')+'_'+
                    self.times[-1].strftime('%Y%m%dT%H%M%S')+
                    '.nc')
            print('saving in',directory+file)
            # CS_SWH_3p_fit_1Hz_(mode)_DateT(0)_DateT(-1)_(reduced_wf).nc
            NC_f = Dataset(directory+file, 'w', format='NETCDF4')
            NC_f.description = 'Significant Wave Height retracking of CryoSat 2 Echoes' 
            
            NC_f.createDimension('time', self.n_rt)
            # to save:
            # A,swh,t0
            pwA = NC_f.createVariable('pwA', 'f4', ('time',))
            swh = NC_f.createVariable('swh', 'f4', ('time',))
            wt0 = NC_f.createVariable('wt0', 'f4', ('time',))
            ppeaky = NC_f.createVariable('PPeakiness', 'f4', ('time',))
            # lat,lon,time(time in timestamp format)
            lat = NC_f.createVariable('lat', 'f4', ('time',))
            lon = NC_f.createVariable('lon', 'f4', ('time',))
            time_av = NC_f.createVariable('time_av', 'f4', ('time',))
            if save_err:
                # err`_fit
                fit_error = NC_f.createVariable('fit_error', 'f4', ('time',))
            # atributes:
            # date processed
            NC_f.setncattr_string('date_processed',
                        dt.date.today().strftime('%Y%m%d'))
            NC_f.setncattr_string('time_units:',
                        'seconds_since_2000_01_01_00000')
            # original nc file_name
            NC_f.setncattr_string('original L1B file',
                        self.name)
            # orbit no
            NC_f.setncattr_string('rel_orbit_number',
                        self.orbit_no)
            NC_f.setncattr_string('data_mode_method',
                        self.mode)
            if type(reduce_WF) == int: 
            # reduce_WF no.
                NC_f.setncattr_string('waveform_averaging_factor',
                        reduce_WF)
                NC_f.setncattr_string('sampling_freq(Hz)',
                        20/reduce_WF)
            else: 
            # reduce_WF no.
                NC_f.setncattr_string('waveform_averaging_factor',
                        0)
                NC_f.setncattr_string('sampling_freq_(Hz)',
                        20)
            if self.mode == 'LRM':
                p_lim = 4.0 
                p_std = 0.07
                w_use = [18,78]
                LE0 = 40
            elif self.mode == 'SAR':
                p_lim = 6.0 
                p_std = 0.1
                w_use = [28,88]
                LE0 = 50
            elif self.mode == 'SIN':
                p_lim = 6.0 
                p_std = 0.15
                w_use = [28,88]
                LE0 = 50
            # options:
#                  elif self.mode == 'SAR':
            NC_f.setncattr_string('filtering_option:pulse_peakiness',
                        p_lim)
            NC_f.setncattr_string('filtering_option:power_std',
                        p_std)
            NC_f.setncattr_string('filtering_option:wave_bin_window_0',
                       w_use[0] )
            NC_f.setncattr_string('filtering_option:wave_bin_window_1',
                       w_use[1] )
            NC_f.setncattr_string('filtering_option:leading_edge_bin_0',
                       LE0 )
            # p_search m_0
            NC_f.setncattr_string('processing_option:pwA_start',
                       m0[0])
            NC_f.setncattr_string('processing_option:sig_start(SWH*0.25)',
                       m0[1])
            NC_f.setncattr_string('processing_option:wt0_start',
                       m0[2])
            NC_f.setncattr_string('look_up_table_h0:',
                       ideal_echo.h0.name)
            NC_f.setncattr_string('look_up_table_h11:',
                       ideal_echo.h11.name)
            NC_f.setncattr_string('look_up_table_h12:',
                       ideal_echo.h12.name)
            NC_f.setncattr_string('look_up_table_h2:',
                       ideal_echo.h2.name)

            
            # fill dimensions with data
           
            pwA[:] = self.pwA
            swh[:] = self.sig*4.0
            wt0[:] = self.wt0
            ppeaky[:] = self.pprt
            # lat,lon,time(time in timestamp format)
            lat[:] = self.latrt
            lon[:] = self.lonrt
            time_av[:] = self.timesrt
            if save_err:
                # err`_fit
                fit_error[:] = self.fit_err

            NC_f.close()
            
        else: print('SWH not calculated so can not save it')
                
#                 g = lambda x: SAR_wave_fit_mf(self.pWFrt[i]/wave_max,x[0],x[1],x[2],error_return = 1e1)
#                 result = minimize(g, [1.0,1.0,30.0],bounds=[(0.5,1.5),(0.0,2.5),(10.0,40.0)],tol=1e-6,method='L-BFGS-B')
#                 path_3[i,0:3] = result.x[:]
#                 path_3[i,3] = SAR_wave_fit_mf(self.pWFrt[i]/wave_max,result.x[0],result.x[1],result.x[2])
#             if (status and np.mod(i,step) == 0):
#                 print(i,'of',self.n_rt,', m =',result.x)
        
@jit
def find_LE(wf,thresh):
    # go through the WF,  find max
    # find point of first over thresh
    # resolution of LE is in bins
    # can possibly interpolate later?
    thresh_use = np.max(wf)*thresh
    fLE = False
    i = -1
    while not fLE:
        i+=1
        if wf[i]>thresh_use:
            fLE = True
            t_bin = i 
    return t_bin