'''
Computes volume averaged salt squared and volume mean salinity variance for the coarse model.
Used for Fig. 7 in Schlichting et al. 
'''

#Packages 
import numpy as np
import xgcm
from xgcm import Grid
import xarray as xr
import xroms
from xhistogram.xarray import histogram
from glob import glob

#Open model output
path = glob.glob('/d1/shared/TXLA_ROMS/numerical_mixing/non-nest/ver1/1hr/ocean_avg_0000*.nc')
ds_avg = xroms.open_mfnetcdf(path)
ds_avg, grid_avg = xroms.roms_dataset(ds_avg)

#Parent model slices
xislice = slice(271,404)
etaslice = slice(31,149)

#Compute sbar
dV = ds_avg.dV.isel(eta_rho = etaslice, xi_rho = xislice)
V = dV.sum(dim = ['eta_rho', 's_rho', 'xi_rho'])
salt = ds_avg.salt.isel(eta_rho = etaslice, xi_rho = xislice)
sbar = (1/V)*(salt*dV).sum(dim = ['eta_rho', 'xi_rho','s_rho'])

sprime = salt-sbar #salinity anomaly
salt2 = salt**2 #salt squared
sprime2 = sprime**2 #volume-mean salinity variance

#Volume average both quantities. 
salt2_vavg = (1/V)*(salt2*dV).sum(dim = ['eta_rho', 'xi_rho','s_rho'])
salt2_vavg.attrs = ''
salt2_vavg.name = 'salt2_vavg'
salt2_vavg.to_netcdf('/d2/home/dylan/JAMES/salt2_vavg.nc') 

sprime2_vavg = (1/V)*(sprime2*dV).sum(dim = ['eta_rho', 'xi_rho','s_rho'])
sprime2_vavg.attrs = ''
sprime2_vavg.name = 'sprime2_vavg'
sprime2_vavg.to_netcdf('/d2/home/dylan/JAMES/sprime2_vavg.nc')
