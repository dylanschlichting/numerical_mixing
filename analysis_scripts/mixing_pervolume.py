'''
Calculates the numerical and physical mixing per unit volume as a function of normalized 
vertical relative vorticity for the the parent and child model output. 
'''
import numpy as np
import xgcm
from xgcm import Grid
from xhistogram.xarray import histogram
import xarray as xr
import xroms
from scipy import signal
import glob
from datetime import datetime

path = glob.glob('/d1/shared/TXLA_ROMS/numerical_mixing/non-nest/ver0/ocean_avg_0000*.nc')
ds_avg = xroms.open_mfnetcdf(path)
ds_avg, grid_avg = xroms.roms_dataset(ds_avg)

path = glob.glob('/d1/shared/TXLA_ROMS/numerical_mixing/nest/ver1/ocean_avg_child_0000*.nc')
ds_avg_child = xroms.open_mfnetcdf(path)
ds_avg_child, grid_avg_child = xroms.roms_dataset(ds_avg_child)

xisliceparent = slice(271,404)
etasliceparent = slice(31,149)
# xislicechild = slice(11, 677-11)
# etaslicechild = slice(11, 602-11)
xislicechild = slice(8, 677-8)
etaslicechild = slice(8, 602-8)

#Compute the relative vorticity on the psi points for the 
#entire water column, so use xroms bc it requires hgrad
rvort = xroms.relative_vorticity(ds_avg.u, ds_avg.v, grid_avg)
#Slice the vorticity to the locsation of the nested grid
rvort_slice = rvort.isel(eta_v = etasliceparent, xi_u = xisliceparent, s_w = slice(1,-1))
#Interpolate f to the psi points
f_psi = xroms.to_psi(ds_avg.f, grid_avg).isel(eta_v = etasliceparent, xi_u = xisliceparent)
#Normalize
rvort_norm = rvort_slice/f_psi
#Fix attributes for saving as a .nc file
rvort_norm.attrs = ''
rvort_norm.name = 'rvort'

#Compute the numerical mixing
mnum_rho = xroms.to_psi(grid_avg.interp(ds_avg.dye_03, 'Z', boundary = 'extend'), grid_avg)
mnum_dV = (mnum_rho*ds_avg.dV_w_psi).isel(eta_v = etasliceparent, xi_u = xisliceparent, s_w = slice(1,-1))
mnum_dV.attrs = ''
mnum_dV.name = 'mnum'

#Compute the resolved mixing 
chi = xroms.to_psi(ds_avg.AKr, grid_avg)
chi_dV = (chi*ds_avg.dV_w_psi).isel(eta_v = etasliceparent, xi_u = xisliceparent, s_w = slice(1,-1))

#Bins for the histogram
rvortbins = np.linspace(-3,3,150)

rvort_hist_dV = histogram(rvort_norm, 
                         bins = [rvortbins], 
                         weights = mnum_dV, 
                         dim = ['s_w', 'eta_v', 'xi_u'])
rvort_hist_dV.name = 'rvort_histogram_numerical_dV'

rvort_hist_resolved_dV = histogram(rvort_norm, 
                         bins = [rvortbins], 
                         weights = chi_dV, 
                         dim = ['s_w', 'eta_v', 'xi_u'])
rvort_hist_resolved_dV.name = 'rvort_histogram_numerical_dV'

# Make vorticity histogram weighted by dV instead
dV_w = ds_avg.dV_w_psi.isel(eta_v = etasliceparent, xi_u = xisliceparent, s_w = slice(1,-1))
dV_w.attrs = ''
dV_w.name = 'dV_w'

rvort_hist = histogram(rvort_norm, 
                         bins = [rvortbins], 
                         weights = dV_w, 
                         dim = ['s_w', 'eta_v', 'xi_u'])
rvort_hist.name = 'rvort_histogram_numerical_dV'

#Mixing per unit volume
mixing_pervolume = rvort_hist_dV/rvort_hist
mixing_pervolume.attrs = '' 

rmixing_pervolume = rvort_hist_resolved_dV/rvort_hist
rmixing_pervolume.attrs = '' 

#Repeat with child model
rvort_child = xroms.relative_vorticity(ds_avg_child.u, ds_avg_child.v, grid_avg_child)
rvort_slice_child = rvort_child.isel(eta_v = etaslicechild, xi_u = xislicechild, s_w = slice(1,-1))
f_psi_child = xroms.to_psi(ds_avg_child.f, grid_avg_child).isel(eta_v = etaslicechild, xi_u = xislicechild)
rvort_norm_child = rvort_slice_child/f_psi_child
rvort_norm_child.attrs = ''
rvort_norm_child.name = 'rvort'

mnum_rho_child = xroms.to_psi(grid_avg_child.interp(ds_avg_child.dye_03, 'Z', boundary = 'extend'), grid_avg_child)
mnum_child_dV = (mnum_rho_child*ds_avg_child.dV_w_psi).isel(eta_v = etaslicechild, xi_u = xislicechild, s_w = slice(1,-1))
mnum_child_dV.attrs = ''
mnum_child_dV.name = 'mnum'

#resolved mixing
chi_child = xroms.to_psi(ds_avg_child.AKr, grid_avg_child)
chi_dV_child = (chi_child*ds_avg_child.dV_w_psi).isel(eta_v = etaslicechild, xi_u = xislicechild, s_w = slice(1,-1))

#Volume integrated numerical mixing
rvort_hist_child_dV = histogram(rvort_norm_child, 
                                 bins = [rvortbins], 
                                 weights = mnum_child_dV, 
                                 dim = ['s_w', 'eta_v', 'xi_u'])
rvort_hist_child_dV.name = 'rvort_histogram_numerical_dV'

rvort_hist_resolved_child_dV = histogram(rvort_norm_child, 
                                 bins = [rvortbins], 
                                 weights = chi_dV_child, 
                                 dim = ['s_w', 'eta_v', 'xi_u'])
rvort_hist_resolved_child_dV.name = 'rvort_histogram_numerical_dV'

dV_w_child = ds_avg_child.dV_w_psi.isel(eta_v = etaslicechild, xi_u = xislicechild, s_w = slice(1,-1))
dV_w_child.attrs = ''
dV_w_child.name = 'dV_w'

#Volume
rvort_hist_child = histogram(rvort_norm_child, 
                         bins = [rvortbins], 
                         weights = dV_w_child, 
                         dim = ['s_w', 'eta_v', 'xi_u'])
rvort_hist_child.name = 'rvort_histogram_numerical_dV'

#Mixing per unit volume
mixing_pervolume_child = rvort_hist_child_dV/rvort_hist_child
mixing_pervolume_child.attrs = '' 

rmixing_pervolume_child = rvort_hist_resolved_child_dV/rvort_hist_child
rmixing_pervolume_child.attrs = '' 

def chunks(lst, n):
    """Yield successive n-sized chunks from lst. Thanks stack exchange!"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

#Save output subsetted every two hours so the child model doesn't crash
daterange = list(chunks(ds_avg_child.ocean_time.sel(ocean_time = slice('2010-06-03', '2010-07-15')), 2))
for d in range(len(daterange)):
# Numerical mixing per unit volume
#     mixing_pervolume_slice = mixing_pervolume.sel(ocean_time = daterange[d])
#     mixing_pervolume_slice.name = 'mixing_perunitvolume'
#     path = '/d2/home/dylan/JAMES/histogram_outputs/weighted/numerical_mixing_perunitvolume_parent_2010_%s.nc' %d
#     mixing_pervolume_slice.to_netcdf(path, mode = 'w')
    
    mixing_pervolume_child_slice = mixing_pervolume_child.sel(ocean_time = daterange[d])
    mixing_pervolume_child_slice.name = 'mixing_perunitvolume'
    path = '/d2/home/dylan/JAMES/histogram_outputs/weighted/numerical_mixing_perunitvolume_child_2010_%s.nc' %d
    mixing_pervolume_child_slice.to_netcdf(path, mode = 'w')

#Resolved mixing per unit volume
#     mixing_pervolume_slice = rmixing_pervolume.sel(ocean_time = daterange[d])
#     mixing_pervolume_slice.name = 'rmixing_perunitvolume'
#     path = '/d2/home/dylan/JAMES/histogram_outputs/weighted/resolved_mixing_perunitvolume_parent_2010_%s.nc' %d
#     mixing_pervolume_slice.to_netcdf(path, mode = 'w')
    
    rmixing_pervolume_child_slice = rmixing_pervolume_child.sel(ocean_time = daterange[d])
    rmixing_pervolume_child_slice.name = 'rmixing_perunitvolume'
    path = '/d2/home/dylan/JAMES/histogram_outputs/weighted/resolved_mixing_perunitvolume_child_2010_%s.nc' %d
    rmixing_pervolume_child_slice.to_netcdf(path, mode = 'w')