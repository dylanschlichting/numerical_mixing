#Packages 
import numpy as np
import xgcm
from xgcm import Grid
import xarray as xr
import xroms
from xhistogram.xarray import histogram
import glob

path = glob.glob('/d1/shared/TXLA_ROMS/numerical_mixing/non-nest/ver1/1hr/ocean_avg_0000*.nc')
ds_avg = xroms.open_mfnetcdf(path)
ds_avg, grid_avg = xroms.roms_dataset(ds_avg)

path = glob.glob('/d1/shared/TXLA_ROMS/numerical_mixing/nest/ver1/ocean_avg_child_0000*.nc')
ds_avg_child = xroms.open_mfnetcdf(path)
ds_avg_child, grid_avg_child = xroms.roms_dataset(ds_avg_child)

#Match the times of the parent and child models. Drop the first timestep after
#each restart, then match parent / child. 
timedrop = [np.datetime64('2010-06-18T18:30:00.000000000'), 
            np.datetime64('2010-06-19T18:30:00.000000000'), 
            np.datetime64('2010-07-09T18:30:00.000000000')]

ds_avg_child = ds_avg_child.where((ds_avg_child.ocean_time!= timedrop[0])
                                & (ds_avg_child.ocean_time!= timedrop[1])
                                & (ds_avg_child.ocean_time!= timedrop[2]),
                                   drop=True)

ds_avg = ds_avg.where(ds_avg_child.ocean_time==ds_avg.ocean_time)

def velgrad_tensor(ds, grid):
    '''
Computes the relative vertical vorticity, divergence,
and strain rate for the entire water column normalized by the Coriolis
frequence. All quantities are either computed on or interpolated linearly to the
psi points horizontally and to the rho points vertically.
    '''
    f_psi = xroms.to_psi(ds.f, grid)
    
    rvort = xroms.relative_vorticity(ds.u, ds.v, grid)
    rvort_srho = grid.interp(rvort, 'Z', boundary = 'extend')
    
    rvort_norm = rvort_srho/f_psi
    rvort_norm.attrs = ''
    rvort_norm.name = 'rvort'

    dudxi, dudeta = xroms.hgrad(ds.u, grid)
    dvdxi, dvdeta = xroms.hgrad(ds.v, grid)

    #Calculate divergence
    divergence = dudxi+dvdeta
    divergence_psi = xroms.to_psi((grid.interp(divergence, 'Z', boundary = 'extend')), grid)
    divergence_norm = divergence_psi/f_psi
    divergence_norm.attrs = ''
    divergence_norm.name = 'divergence'

    #Calculate strain
    s1 = (dudxi-dvdeta)**2
    s2 = xroms.to_psi(s1, grid)
    strain = ((s2+(dvdxi+dudeta)**2)**(1/2))
    strain_psi = xroms.to_psi((grid.interp(strain, 'Z', boundary = 'extend')), grid)
    strain_norm = strain_psi/f_psi
    strain_norm.attrs = ''
    strain_norm.name = 'strain'

    return rvort_norm, divergence_norm, strain_norm

def salinity_gradient_mag(ds, grid):
    '''
Computes the salinity gradient magnitude for the entire
water column on the psi points.
    ''' 
    dsaltdxi, dsaltdeta = xroms.hgrad(ds.salt, grid)
    dsaltdxi_psi = xroms.to_psi(dsaltdxi, grid)
    dsaltdeta_psi = xroms.to_psi(dsaltdeta, grid)
    sgradmag = (dsaltdxi_psi**2+dsaltdeta_psi**2)**(1/2)
    sgradmag_rho = grid.interp(sgradmag, 'Z', boundary = 'extend')
    sgradmag_rho.attrs = ''
    sgradmag_rho.name = 'sgradmag'
    
    return sgradmag_rho
    
xisliceparent = slice(271,404)
etasliceparent = slice(31,149)
xislicechild = slice(8,677-8)
etaslicechild = slice(8,602-8)

rvort_norm, divergence_norm, strain_norm = velgrad_tensor(ds_avg, grid_avg)

rv_parent = rvort_norm.isel(eta_v = etasliceparent, xi_u = xisliceparent).sel(ocean_time = slice('2010-06-03', '2010-07-13'))
divergence_parent = divergence_norm.isel(eta_v = etasliceparent, xi_u = xisliceparent).sel(ocean_time = slice('2010-06-03', '2010-07-13'))
strain_parent = strain_norm.isel(eta_v = etasliceparent, xi_u = xisliceparent).sel(ocean_time = slice('2010-06-03', '2010-07-13'))

sgradmag_rho = salinity_gradient_mag(ds_avg, grid_avg)
sgradmag_parent = sgradmag_rho.isel(eta_v = etasliceparent, xi_u = xisliceparent).sel(ocean_time = slice('2010-06-03', '2010-07-13'))

#Subset outputs and save to netCDF4
rvort_norm_child, divergence_norm_child, strain_norm_child = velgrad_tensor(ds_avg_child, grid_avg_child)

rv_child = rvort_norm_child.isel(eta_v = etaslicechild, xi_u = xislicechild).sel(ocean_time = slice('2010-06-03', '2010-07-13'))
divergence_child = divergence_norm_child.isel(eta_v = etaslicechild, xi_u = xislicechild).sel(ocean_time = slice('2010-06-03', '2010-07-13'))
strain_child = strain_norm_child.isel(eta_v = etaslicechild, xi_u = xislicechild).sel(ocean_time = slice('2010-06-03', '2010-07-13'))

sgradmag_rho_child = salinity_gradient_mag(ds_avg_child, grid_avg_child)
sgradmag_child = sgradmag_rho_child.isel(eta_v = etaslicechild, xi_u = xislicechild).sel(ocean_time = slice('2010-06-03', '2010-07-13'))

#Compute the rms for each variable 
def rms(y_load):
    '''
    Computes the root-mean square of an Xarray DataArray
Inputs - 
Loaded Xarray DataArray
    '''
    rms = np.sqrt(np.mean(y_load**2))
    return rms 

rv_subset_parent = rv_parent[:,:,::3,::3]
div_subset_parent = divergence_parent[:,:,::3,::3]
strain_subset_parent = strain_parent[:,:,::3,::3]
sgradmag_subset_parent = sgradmag_parent[:,:,::3,::3]

rv_subset_child = rv_child[:,:,::15,::15]
div_subset_child = divergence_child[:,:,::15,::15]
strain_subset_child = strain_child[:,:,::15,::15]
sgradmag_subset_child = sgradmag_child[:,:,::15,::15]

rms_rv_parent = rms(rv_subset_parent.values)
rms_rv_child = rms(rv_subset_child.values)

rms_divergence_parent = rms(div_subset_parent.values)
rms_divergence_child = rms(div_subset_child.values)

rms_strain_parent = rms(strain_subset_parent.values)
rms_strain_child = rms(strain_subset_child.values)

rms_sgradmag_parent = rms(sgradmag_subset_parent.values)
rms_sgradmag_child = rms(gradmag_subset_child.values)

#Save to numpy arrays 
np.save('/d2/home/dylan/JAMES/histogram_outputs/whole/stats/relvort_whole_subset_parent_rv_rms.npy', rms_rv_parent)
np.save('/d2/home/dylan/JAMES/histogram_outputs/whole/stats/relvort_whole_subset_child_rv_rms.npy', rms_rv_child)

np.save('/d2/home/dylan/JAMES/histogram_outputs/whole/stats/relvort_whole_subset_parent_div_rms.npy', rms_divergence_parent)
np.save('/d2/home/dylan/JAMES/histogram_outputs/whole/stats/relvort_whole_subset_child_div_rms.npy', rms_divergence_child)

np.save('/d2/home/dylan/JAMES/histogram_outputs/whole/stats/relvort_whole_subset_parent_strain_rms.npy', rms_strain_parent)
np.save('/d2/home/dylan/JAMES/histogram_outputs/whole/stats/relvort_whole_subset_child_strain_rms.npy', rms_strain_child)

np.save('/d2/home/dylan/JAMES/histogram_outputs/whole/stats/relvort_whole_subset_parent_sgradmag_rms.npy', rms_sgradmag_parent)
np.save('/d2/home/dylan/JAMES/histogram_outputs/whole/stats/relvort_whole_subset_child_sgradmag_rms.npy', rms_sgradmag_child)
