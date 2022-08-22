import numpy as np
import xgcm
from xgcm import Grid
from xhistogram.xarray import histogram
import xarray as xr
import xroms
from scipy import signal
import glob
from datetime import datetime

path = glob.glob('/d1/shared/TXLA_ROMS/numerical_mixing/non-nest/ver1/1hr/ocean_avg_0000*.nc')
ds_avg = xroms.open_mfnetcdf(path)
ds_avg, grid_avg = xroms.roms_dataset(ds_avg)

path = glob.glob('/d1/shared/TXLA_ROMS/numerical_mixing/nest/ver1/ocean_avg_child_0000*.nc')
ds_avg_child = xroms.open_mfnetcdf(path)
ds_avg_child, grid_avg_child = xroms.roms_dataset(ds_avg_child)

xislice = slice(271,404)
etaslice = slice(31,149)

timedrop = [np.datetime64('2010-06-18T18:30:00.000000000'), 
            np.datetime64('2010-06-19T18:30:00.000000000'), 
            np.datetime64('2010-07-09T18:30:00.000000000')]

ds_avg_child = ds_avg_child.where((ds_avg_child.ocean_time!= timedrop[0])
                           & (ds_avg_child.ocean_time!= timedrop[1])
                           & (ds_avg_child.ocean_time!= timedrop[2]),
                              drop=True)

ds_avg = ds_avg.where(ds_avg_child.ocean_time==ds_avg.ocean_time)

#Resolved mixing
AKr_rho = grid_avg.interp(ds_avg.AKr, 'Z')
chi_dz = (AKr_rho*ds_avg.dz).isel(eta_rho = etaslice, xi_rho = xislice).sum(['s_rho'])
chi_dz.attrs = ''

#Numerical mixing
mnum_dz = (ds_avg.dye_03*ds_avg.dz).isel(eta_rho = etaslice, xi_rho = xislice).sum(['s_rho'])
mnum_dz.attrs = ''

print('saving outputs')
dates = np.arange('2010-06-03', '2010-07-15', dtype = 'datetime64[D]') 
for d in range(len(dates)):    
    chi_dz_sel = chi_dz.sel(ocean_time = str(dates[d]))
    chi_dz_sel.name = 'chi_dz'
    path = '/d2/home/dylan/JAMES/histogram_outputs/depth_int_mixing/chi_dz_2010_srho_%s.nc' %d
    chi_dz_sel.to_netcdf(path, mode = 'w')
    
    mnum_dz_sel = mnum_dz.sel(ocean_time = str(dates[d]))
    mnum_dz_sel.name = 'mnum_dz'
    path = '/d2/home/dylan/JAMES/histogram_outputs/depth_int_mixing/mnum_dz_2010_srho_%s.nc' %d
    mnum_dz_sel.to_netcdf(path, mode = 'w')
    
# xislicechild = slice(11, 677-11)
# etaslicechild = slice(11, 602-11)
xislicechild = slice(8, 677-8)
etaslicechild = slice(8, 602-8)

#Resolved mixing
AKr_rho_child = grid_avg_child.interp(ds_avg_child.AKr, 'Z')
chi_dz_child = (AKr_rho_child*ds_avg_child.dz).isel(eta_rho = etaslicechild, xi_rho = xislicechild).sum(['s_rho'])
chi_dz_child.attrs = ''

#Numerical mixing
mnum_dz_child = (ds_avg_child.dye_03*ds_avg_child.dz).isel(eta_rho = etaslicechild, xi_rho = xislicechild).sum(['s_rho'])
mnum_dz_child.attrs = ''

print('saving outputs')
dates = np.arange('2010-06-03', '2010-07-15', dtype = 'datetime64[D]') 
for d in range(len(dates)):    
    chi_dz_child_sel = chi_dz_child.sel(ocean_time = str(dates[d]))
    chi_dz_child_sel.name = 'chi_dz'
    path = '/d2/home/dylan/JAMES/histogram_outputs/depth_int_mixing/chi_dz_child_2010_%s.nc' %d
    chi_dz_child_sel.to_netcdf(path, mode = 'w')
    
    mnum_dz_child_sel = mnum_dz_child.sel(ocean_time = str(dates[d]))
    mnum_dz_child_sel.name = 'mnum_dz'
    path = '/d2/home/dylan/JAMES/histogram_outputs/depth_int_mixing/mnum_dz_child_2010_%s.nc' %d
    mnum_dz_child_sel.to_netcdf(path, mode = 'w')