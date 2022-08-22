'''
Computes the volume integrated numerical and physical mixing for the 
native TXLA model.
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
path = glob('/d1/shared/TXLA_ROMS/numerical_mixing/non-nest/ver1/1hr/ocean_avg_0000*.nc')
# path = glob('/d1/shared/TXLA_ROMS/numerical_mixing/non-nest/ver1/30min/ocean_avg_0000*.nc')
# path = glob('/d1/shared/TXLA_ROMS/numerical_mixing/non-nest/ver1/10min/ocean_avg_0000*.nc')
ds_avg = xroms.open_mfnetcdf(path)
ds_avg, grid_avg = xroms.roms_dataset(ds_avg)

#Indices for the location of the nested grid
xislice = slice(271,404)
etaslice = slice(31,149)

#Physical mixing. Need to interpolate to the s-rho points from the s-w points
Akr_rho = grid_avg.interp(ds_avg.AKr, 'Z')
chi_online = (Akr_rho*ds_avg.dV).isel(eta_rho = etaslice, xi_rho = xislice).sum(['eta_rho', 'xi_rho', 's_rho'])
chi_online.attrs = ''

#Numerical mixing. No need to intepolate, outputted on the s-rho points. 
mnum_online = (ds_avg.dye_03*ds_avg.dV).isel(eta_rho = etaslice, xi_rho = xislice).sum(['eta_rho', 'xi_rho', 's_rho'])
mnum_online.attrs = ''

#Subset the data daily and save to a netcdf file. The reason we do this is to avoid overloading
#the cluseter. 
print('saving outputs')
dates = np.arange('2010-06-03', '2010-07-15', dtype = 'datetime64[D]') 
for d in range(len(dates)):    
    chi_online_sel = chi_online.sel(ocean_time = str(dates[d]))
    chi_online_sel.name = 'chi_online'
    path = '/d2/home/dylan/JAMES/budget_outputs/mixing/chi_online_ver1_2010_%s.nc' %d
#     path = '/d2/home/dylan/JAMES/budget_outputs/30min/mixing/chi_online_2010_30min_%s.nc' %d
#     path = '/d2/home/dylan/JAMES/budget_outputs/10min/mixing/chi_online_2010_10min_%s.nc' %d
    chi_online_sel.to_netcdf(path, mode = 'w')
    
    mnum_online_sel = mnum_online.sel(ocean_time = str(dates[d]))
    mnum_online_sel.name = 'mnum_online'
    path = '/d2/home/dylan/JAMES/budget_outputs/mixing/mnum_online_ver1_2010_%s.nc' %d
#     path1 = '/d2/home/dylan/JAMES/budget_outputs/30min/mixing/mnum_online_2010_30min_%s.nc' %d
#     path1 = '/d2/home/dylan/JAMES/budget_outputs/10min/mixing/mnum_online_2010_10min_%s.nc' %d
    mnum_online_sel.to_netcdf(path, mode = 'w')
