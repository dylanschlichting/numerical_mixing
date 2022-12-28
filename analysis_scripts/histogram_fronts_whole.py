#Script to investigate changes in model properties associated with fronts, 
#corresponding to zeta/f > 1. Also examine the thomas angle 
#Note------
#We are looking at dsdx**2+dsdy**2 here, not the sqrt(dsdx**2+dsdy**2)

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

# def salinity_gradient_mag_rho(ds, grid):
#     '''
# Computes the salinity gradient magnitude for the entire
# water column on the psi points.
#     ''' 
#     dsaltdxi, dsaltdeta = xroms.hgrad(ds.salt, grid)
#     dsaltdxi_rho = xroms.to_rho(dsaltdxi, grid)
#     dsaltdeta_rho = xroms.to_rho(dsaltdeta, grid)
#     sgradmag = (dsaltdxi_rho**2+dsaltdeta_rho**2)**(1/2)
#     sgradmag_rho = grid.interp(sgradmag, 'Z', boundary = 'extend')
#     sgradmag_rho.attrs = ''
#     sgradmag_rho.name = 'sgradmag'
    
#     return sgradmag_rho
    
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

sgradmag_rho = salinity_gradient_mag_rho(ds_avg, grid_avg)
sgradmag_parent = sgradmag_rho.isel(eta_rho = etasliceparent, xi_rho = xisliceparent).sel(ocean_time = slice('2010-06-03', '2010-07-13'))


#Subset outputs and save to netCDF4
rvort_norm_child, divergence_norm_child, strain_norm_child = velgrad_tensor(ds_avg_child, grid_avg_child)

rv_child = rvort_norm_child.isel(eta_v = etaslicechild, xi_u = xislicechild).sel(ocean_time = slice('2010-06-03', '2010-07-13'))
divergence_child = divergence_norm_child.isel(eta_v = etaslicechild, xi_u = xislicechild).sel(ocean_time = slice('2010-06-03', '2010-07-13'))
strain_child = strain_norm_child.isel(eta_v = etaslicechild, xi_u = xislicechild).sel(ocean_time = slice('2010-06-03', '2010-07-13'))

sgradmag_rho_child = salinity_gradient_mag(ds_avg_child, grid_avg_child)
sgradmag_child = sgradmag_rho_child.isel(eta_v = etaslicechild, xi_u = xislicechild).sel(ocean_time = slice('2010-06-03', '2010-07-13'))

# sgradmag_rho_child = salinity_gradient_mag_rho(ds_avg_child, grid_avg_child)
# sgradmag_child = sgradmag_rho_child.isel(eta_rho = etaslicechild, xi_rho = xislicechild).sel(ocean_time = slice('2010-06-03', '2010-07-13'))

#Now sort all values based on cyclonic vorticity associated with fronts, i.e. zeta/f > 1. 
rv_pos_parent = rv_parent.where(rv_parent>1)
div_pos_parent = divergence_parent.where(rv_parent>1)
strain_pos_parent = strain_parent.where(rv_parent>1)
sgradmag_pos_parent = sgradmag_parent.where(rv_parent>1)

rv_pos_child = rv_child.where(rv_child>1)
div_pos_child = divergence_child.where(rv_child>1)
strain_pos_child = strain_child.where(rv_child>1)
sgradmag_pos_child = sgradmag_child.where(rv_child>1)

#Bin sizes for each variable: use 150 bins for each variable
#Corresponding location of the parent model subset. See 'check_grid.ipynb' for more information

rvortbins = np.linspace(-5,5,150)
divbins = np.linspace(-5,5,150)
strainbins = np.linspace(0,8,150)
sgradbins = np.linspace(0,0.002,150)

#Relative vorticity
vort_hist = histogram(rv_pos_parent, bins = [rvortbins], density = True)
vort_hist.name = 'rvort'
path = '/d2/home/dylan/JAMES/histogram_outputs/fronts/rvort_whole_parent_2010_fronts.nc'
vort_hist.to_netcdf(path, mode = 'w')

vort_hist_child = histogram(rv_pos_child, bins = [rvortbins], density = True)
vort_hist_child.name = 'rvort'
path = '/d2/home/dylan/JAMES/histogram_outputs/fronts/rvort_whole_child_2010_fronts.nc'
vort_hist_child.to_netcdf(path, mode = 'w')

#Divergence
div_hist = histogram(div_pos_parent, bins = [divbins], density = True)
div_hist.name = 'div'
path = '/d2/home/dylan/JAMES/histogram_outputs/fronts/divergence_whole_parent_2010_fronts.nc'
div_hist.to_netcdf(path, mode = 'w')

div_hist_child = histogram(div_pos_child, bins = [divbins], density = True)
div_hist_child.name = 'div'
path = '/d2/home/dylan/JAMES/histogram_outputs/fronts/divergence_whole_child_2010_fronts.nc'
div_hist_child.to_netcdf(path, mode = 'w')

# #Strain
strain_hist = histogram(strain_pos_parent, bins = [strainbins], density = True)
strain_hist.name = 'strain'
path = '/d2/home/dylan/JAMES/histogram_outputs/fronts/strain_whole_parent_2010_fronts.nc'
strain_hist.to_netcdf(path, mode = 'w')

strain_hist_child = histogram(strain_pos_child, bins = [strainbins], density = True)
strain_hist_child.name = 'strain'
path = '/d2/home/dylan/JAMES/histogram_outputs/fronts/strain_whole_child_2010_fronts.nc'
strain_hist_child.to_netcdf(path, mode = 'w')

#Salinity gradient magnitude - psi points
sgradmag_rho_hist = histogram(sgradmag_pos_parent, bins = [sgradbins], density = True)
sgradmag_rho_hist.name = 'sgradmag'
path = '/d2/home/dylan/JAMES/histogram_outputs/fronts/sgradmag_rho_whole_parent_2010_fronts.nc'
sgradmag_rho_hist.to_netcdf(path, mode = 'w')

sgradmag_rho_hist_child = histogram(sgradmag_pos_child, bins = [sgradbins], density = True)
sgradmag_rho_hist_child.name = 'sgradmag'
path = '/d2/home/dylan/JAMES/histogram_outputs/fronts/sgradmag_rho_whole_child_2010_fronts.nc'
sgradmag_rho_hist_child.to_netcdf(path, mode = 'w')

#Salinity gradient magnitude - rho points 
# sgradmag_rho_hist = histogram(sgradmag_pos_parent, bins = [sgradbins], density = True)
# sgradmag_rho_hist.name = 'sgradmag'
# path = '/d2/home/dylan/JAMES/histogram_outputs/fronts/sgradmag_whole_parent_2010_fronts_space_rho.nc'
# sgradmag_rho_hist.to_netcdf(path, mode = 'w')

# sgradmag_rho_hist_child = histogram(sgradmag_pos_child, bins = [sgradbins], density = True)
# sgradmag_rho_hist_child.name = 'sgradmag'
# path = '/d2/home/dylan/JAMES/histogram_outputs/fronts/sgradmag_whole_child_2010_fronts_space_rho.nc'
# sgradmag_rho_hist_child.to_netcdf(path, mode = 'w')


