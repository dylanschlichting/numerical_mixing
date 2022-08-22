'''
Computes the volume averaged salinity 'sbar' for the child TXLA model corresponding using average files.  
'''
#Packages 
import numpy as np
import xgcm
from xgcm import Grid
import xarray as xr
import xroms
from xhistogram.xarray import histogram
from glob import glob

path = glob('/d1/shared/TXLA_ROMS/numerical_mixing/nest/ver1/ocean_avg_child_0000*.nc')
ds_avg = xroms.open_mfnetcdf(path)
ds_avg, grid = xroms.roms_dataset(ds_avg)

xislice = slice(11, 677-11)
etaslice = slice(11, 602-11)

dV = ds_avg.dV.isel(eta_rho = etaslice, xi_rho = xislice)
V = dV.sum(dim = ['eta_rho', 's_rho', 'xi_rho'])
salt = ds_avg.salt.isel(eta_rho = etaslice, xi_rho = xislice)
sbar = (1/V)*(salt*dV).sum(dim = ['eta_rho', 'xi_rho','s_rho'])
sbar.attrs = ''

print('saving outputs')
dates = np.arange('2010-06-03', '2010-07-15', dtype = 'datetime64[D]') 
for d in range(len(dates)):    
    sbar_sel = sbar.sel(ocean_time = str(dates[d]))
    path = '/d2/home/dylan/JAMES/budget_outputs/sbar/sbar_child_2010_%s.nc' %d
    sbar_sel.name = 'sbar'
    sbar_sel.to_netcdf(path, mode = 'w')
