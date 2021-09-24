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


# lookup tables for ideal echoes
# reading mathematic look ups


# need to give a sig tor,
# then get 4 nearest neighbours from the table
class CSecho_Table:
    def __init__(self,file):
        LU_x, self.__data = read_LUT(file)
        self.sigd,self.tord = np.shape(self.__data)
        self.name = file
    
    def set_ranges(self,sig_lim,tor_lim,error_return = np.nan):
        """
        sig_lim  = [min,max]
        tor_lim  = [min,max]
        both for upper lower bounds on the 
        lookup table
        """
        self.sig_min = sig_lim[0]
        self.sig_max = sig_lim[1]
        self.tor_min = tor_lim[0]
        self.tor_max = tor_lim[1]
        self.sig_range = self.sig_max - self.sig_min
        self.tor_range = self.tor_max - self.tor_min
        self.dsig = self.sig_range/(self.sigd-1)
        self.dtor = self.tor_range/(self.tord-1)
        
        self.error_return = error_return
        
#         kx  = int((x-piq-pi)*invdx) + 1
#     kxw = c1 - ((x-piq-pi)*invdx - (kx-1))

    def __getitem__(self,sigtor):
        # take the sig,tor
        # use sig/tor lims to get 4 index
        sig = sigtor[0]
        tor = sigtor[1]
        sigi  = int((sig-self.sig_min)/self.dsig)
        tori  = int((tor-self.tor_min)/self.dtor)
        # and also 4 weights
        sigw = 1.0 - (sig-self.sig_min)/self.dsig + sigi
        torw = 1.0 - (tor-self.tor_min)/self.dtor + tori
        
#         print(sigi,sigw,tori,torw)
        
        if (sigi in range(self.sigd) and
            tori in range(self.tord)):
            # try to extract them
            try:
                pwf1 =      sigw *     torw *self.__data[sigi  ,tori]
            except IndexError:
                pwf1 = np.nan
            try:
                pwf2 = (1.0-sigw)*     torw *self.__data[sigi+1,tori]
            except IndexError:
                pwf2 = np.nan
            try:
                pwf3 =      sigw *(1.0-torw)*self.__data[sigi  ,tori+1]
            except IndexError:
                pwf3 = np.nan
            try:
                pwf4 = (1.0-sigw)*(1.0-torw)*self.__data[sigi+1,tori+1]
            except IndexError:
                pwf4 = np.nan
            
            return np.nansum([pwf1,pwf2,pwf3,pwf4])
            
            # linearly intrpolate to return desired value
#             return self.__data[idx]
        else:
            return self.error_return
        
        
class ideal_echo:
    """
    builds functions from lookup tables
    """
    def __init__(self):
        self.load_interp = False
        self.ns = 1e-9;
        self.c = 2.99792458e8;
        self.h = 720000;
        self.r = 6380000;
        self.istart = -50 ;
        self.iend = 180 ;
        self.icre = 0.1;
        self.npoints = 1 + (self.iend - self.istart)/0.1;
        self.nsigma = 25;
        self.sigmaint = 0.1;
        self.lambda_use = self.c/13.575e9;
        self.k0 = 2*np.pi/self.lambda_use;
        self.Del = 7200./18182;
        self.zeta = np.pi/(64*self.k0*self.Del);
        self.gammabar = 0.012215368000378016
        self.gammahat = 0.0381925958945466
        self.gamma1 = np.sqrt(2/(2/self.gammabar**2 + 2/self.gammahat**2));
        self.gamma2 = np.sqrt(2/(2/self.gammabar**2 - 2/self.gammahat**2));

    def load_tables(self,h0_f,h11_f,h12_f,h2_f,error_return = np.nan):
        self.h0  = CSecho_Table(h0_f)
        self.h11 = CSecho_Table(h11_f)
        self.h12 = CSecho_Table(h12_f)
        self.h2  = CSecho_Table(h2_f)
        self.h0.set_ranges([0,self.nsigma*self.sigmaint],[self.istart,self.iend],error_return = error_return)
        self.h11.set_ranges([0,self.nsigma*self.sigmaint],[self.istart,self.iend],error_return = error_return)
        self.h12.set_ranges([0,self.nsigma*self.sigmaint],[self.istart,self.iend],error_return = error_return)
        self.h2.set_ranges([0,self.nsigma*self.sigmaint],[self.istart,self.iend],error_return = error_return)
        self.load_interp = True
    
    def echo(self,sig, tor, p, r,delh,
              torscale = False,error_return = np.nan,mode='SAR'):
        """
        torscale for whether the echo is in bin no or physical dimensions
        set true if putting in physical values (of order 1e-9)
        else tor is a bin no range -50:140
        set error_return for sig out of scale - change depending on minimisation method
        """
        if not self.load_interp: 
            print("tables are not loaded and interpolated so function unavailable")
            return error_return
        if torscale:
            tor = tor*1e9
            ctor = self.c*tor
        else:
            ctor = self.c*tor*self.ns
        if mode == 'LRM':
            h1mult = torscale/(self.h*(1.0 + self.h/self.r))
        else:
            h1mult = 1.0
        # check ranges first
        # sigma within 0:2.5 strickly
        if sig < 0 or sig > self.nsigma*self.sigmaint:
#             print("sig out of range (0,2.5), returning error")
            return error_return
#         print(gamma1,gamma2)
        else:
        # other values analytical so fine
            h0 = (1.0 - 2.0*((p/self.gamma1)**2 + (r/self.gamma2)**2))*self.h0[sig,tor]
            h1 = 8*h1mult*((p**2/self.gamma1**4)*self.h11[sig,tor]
                  + (r**2/self.gamma2**4)*self.h12[sig,tor]) 
            h2 = delh*self.h2[sig,tor]
    #         print(h0,h1,h2)
            return h0 + h1 + h2
        # then return the required function of the interpolation table
        

    # add the weights - do externaly then load it
    def echo_vec(self,x_vec,A,sig,t0,p=0.0,r=0.0,delh=0.0,yoff=0.0,mode='SAR'):
        """
        given the parameter values, return vector the echo power for given x_vec
        """
        power_vec = [A*self.scale*self.echo(sig,(t-t0)*self.sampling,
                                        p, r,delh,mode=mode)+yoff for t in x_vec]
        return power_vec
    
    # the misfit function - needs : 
    #     sampling
    #     scale
    # then the three find the values methods - 
    
    def wave_fit_mf(self,wf,A,sig,t0,p=0.0,r=0.0,delh=0.0,return_vec = False,error_return = np.nan,mode='SAR'):
        """
        takes an input waveform and returns a misfit to an idealised waveform
        wf is the input waveform
        idealised given by power A, shape sig, leading edge t0
        the weighting function is presupplied
        normally returns a float that is the magnitude of mf (so to be minimised)
        return_vec for diagnostic, so to see the shape of the misfit
            the x vec, and corresponding vector of difference returned.
        """
        # get the shape of the waveform
        wf_s = np.shape(wf)
        # make x vec that corresponds
        x_vec = np.linspace(0,wf_s[0]-1,wf_s[0])
        diff_vec = wf - [A*self.scale*self.echo(sig,(t-t0)*self.sampling,
                                p, r,delh,error_return = error_return,mode=mode) 
                         for t in x_vec]
        diff_vec = self.weight/self.weight_s*diff_vec**2
        if np.isnan(diff_vec).all():
            diff_vec = np.nan
        if return_vec:
            return x_vec, diff_vec
        else:
            return np.nansum(diff_vec)
        
    def wavefit_1(self,wf,A0,sig0,t0,p=0.0,r=0.0,delh=0.0,mode='SAR'):
        g = lambda x: self.wave_fit_mf(wf,A0,x[0],t0,p,r,delh,error_return = 1e1,mode=mode)
        result = minimize(g, [sig0],bounds=[(0.0,2.5)],
                          tol=1e-7,method='L-BFGS-B')
        return A0,result.x[0],t0

    def wavefit_2(self,wf,A0,sig0,t0,p=0.0,r=0.0,delh=0.0,mode='SAR'):
        g = lambda x: self.wave_fit_mf(wf,A0,x[0],x[1],p,r,delh,error_return = 1e1,mode=mode)
        result = minimize(g, [sig0,t0],bounds=[(0.0,2.5),(20.0,40.0)],
                          tol=1e-7,method='L-BFGS-B')
        return A0,result.x[0],result.x[1]

    # @jit
    def wavefit_3(self,wf,A0,sig0,t0,p=0.0,r=0.0,delh=0.0,mode='SAR'):
        g = lambda x: self.wave_fit_mf(wf,x[0],x[1],x[2],p,r,delh,error_return = 1e1,mode=mode)
        result = minimize(g, [A0,sig0,t0],bounds=[(0.1,2.0),(0.0,2.5),(10.0,40.0)],
                          tol=1e-7,method='L-BFGS-B')
        return result.x[0],result.x[1],result.x[2]

# reading a mathematica look up table

def read_LUT(file):
    test = open(file,'r')
    p_start = 0
    l = 0
    LU_array = np.empty([25,2300])
    tb_c = 0
    read_x = False
    read_d = False
    test_x = []
    test_l = []
    # find seventh {
    for line in test :
        # First block finds the x coords for all the next
    #     print(line)
        p_start+=line.count('{')
        if read_x:
            #reading a table
            read = (line.split(',')[0:-1])
    #         print(read)
            try:
                test_x.append([float(r.replace('*^','e')) for r in read])
            except ValueError:
                # found the end of a table
                read[1] = read[1].split('}}')[0]
                test_x.append([float(r.replace('*^','e')) for r in read])
                # so stop reading it
                read_x = False
                p_start = 0
                # fill the array
                LU_x = np.array([n for tl in test_x for n in tl])

        if p_start == 9 and not read_x:
            # found the beginning of a table
            read = (line.split('{{'))[1].split(',')[0:-1]
    #         print(read)
            test_x.append([float(r.replace('*^','e')) for r in read])
            read_x = True

        # then read all number inside {}
        if read_d:
        # until we find '}}'
            if line.count('}}')>0:
                # final line
                read = line.split(',')
    #             print(read)
                temp_n = [r[r.find("{")+1:r.find("}")
                               ].replace('*^','e')
                                     for r in read]
                for n in temp_n:
                    n = n.replace('}','')
    #                 print(n)
                    try:
                        n = float(n)
                    except ValueError:
                        pass
                    else:
                        # save them if successful
                        test_l.append(float(n)) 
    #                     print(n)
                read_d = False
                # found the end of a table
                LU_array[tb_c,:] = np.array(test_l)
                tb_c +=1
                test_l = []
            else:
            # just another line
                read = line.split(',')
        #         print(read)
                temp_n = [r[r.find("{")+1:r.find("}")
                               ].replace('*^','e') 
                                     for r in read]
                for n in temp_n:
                    try:
                        n = float(n)
                    except ValueError:
                        pass
                    else:
                        # save them if successful
                        test_l.append(float(n)) 
    #                     print(n)


        # now find '{List}'
        if line.count('{List}')>0 and not read_d:
            # start reaing data
            # after generating list of strings/or not, we need to
            # make list in the correct format to convert to strings
            read_d = True
            # find all the numbers
            read = line.split(',')
    #         print(read)
            temp_n = [r[r.find("{")+1:r.find("}")
                               ].replace('*^','e') 
                                     for r in read]
            # try to convert the numbers to actual numbers
            for n in temp_n:
                n = n.replace('}','')
    #             print(n)
                try:
                    n = float(n)
                except ValueError:
                    pass
                else:
                    # save them if successful
                    test_l.append(float(n)) 
    #                 print(n)
    test.close()
    return LU_x, LU_array