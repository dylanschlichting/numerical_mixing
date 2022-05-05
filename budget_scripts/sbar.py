'''
Computes the volume averaged salinity 'sbar' for the native parent TXLA model corresponding 
to the location of the nested grid using average files. Run this script before running any other 
budget scripts!!!
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
path = glob('/d1/shared/TXLA_ROMS/numerical_mixing/non-nest/ver0/ocean_avg_0000*.nc')
ds_avg = xroms.open_mfnetcdf(path)
ds_avg, grid = xroms.roms_dataset(ds_avg)

#Slice the grid
xislice = slice(271,404)
etaslice = slice(31,149)

dV = ds_avg.dV.isel(eta_rho = etaslice, xi_rho = xislice)
V = dV.sum(dim = ['eta_rho', 's_rho', 'xi_rho'])
salt = ds_avg.salt.isel(eta_rho = etaslice, xi_rho = xislice)
sbar = (1/V)*(salt*dV).sum(dim = ['eta_rho', 'xi_rho','s_rho'])
sbar.attrs = ''

print('saving outputs')
dates = np.arange('2010-06-03', '2010-07-15', dtype = 'datetime64[D]') 
for d in range(len(dates)):    
    sbar_sel = sbar.sel(ocean_time = str(dates[d]))
    path = '/d2/home/dylan/JAMES/budget_outputs/sbar/sbar_parent_2010_%s.nc' %d
    sbar_sel.name = 'sbar'
    sbar_sel.to_netcdf(path, mode = 'w')
