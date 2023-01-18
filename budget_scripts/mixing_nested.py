'''
Computes the volume integrated numerical and physical mixing for the 
nested TXLA model. Change path depending on whether hourly, 30min, or 10min output is desired. 
'''
#Packages 
import numpy as np
import xgcm
from xgcm import Grid
import xarray as xr
import xroms

from xhistogram.xarray import histogram
from glob import glob
from dask.distributed import Client

path = glob('/d1/shared/TXLA_ROMS/numerical_mixing/nest/ver1/ocean_avg_child_0000*.nc')
ds_avg = xroms.open_mfnetcdf(path)
ds_avg, grid = xroms.roms_dataset(ds_avg)

xislice = slice(8, 677-8)
etaslice = slice(8, 602-8)

#Resolved mixing. Need to interpolate from the w to the rho points 
#for consistency. 
Akr_rho = grid.interp(ds_avg.AKr, 'Z', boundary = 'extend')
chi_online = (AKr_rho*ds_avg.dV).isel(eta_rho = etaslice, xi_rho = xislice).sum(['eta_rho', 'xi_rho', 's_rho'])
chi_online.attrs = ''

#Numerical mixing - dye_03 in ROMS syntax. 
mnum_online = (ds_avg.dye_03*ds_avg.dV).isel(eta_rho = etaslice, xi_rho = xislice).sum(['eta_rho', 'xi_rho', 's_rho'])
mnum_online.attrs = ''

def chunks(lst, n):
    """Yield successive n-sized chunks from lst. Thanks stack exchange!"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
        
#Save outputs to netcdf. Save every two hours to avoid cluster crashing.
print('saving outputs')
daterange = list(chunks(ds_avg.ocean_time.sel(ocean_time = slice('2010-06-03', '2010-07-13')), 2))
for d in range(len(daterange)):
    chi_online_sel = chi_online.sel(ocean_time = daterange[d])
    chi_online_sel.name = 'chi_online'
    path = '/d2/home/dylan/JAMES/budget_outputs/mixing/chi_online_nested_2010_%s.nc' %d
    chi_online_sel.to_netcdf(path, mode = 'w')
    
    mnum_online_sel = mnum_online.sel(ocean_time = daterange[d])
    mnum_online_sel.name = 'mnum_online'
    path = '/d2/home/dylan/JAMES/budget_outputs/mixing/mnum_online_nested_2010_%s.nc' %d
    mnum_online_sel.to_netcdf(path, mode = 'w')