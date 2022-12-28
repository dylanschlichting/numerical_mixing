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

#Bin sizes for each variable: use 150 bins for each variable
#Corresponding location of the parent model subset. See 'check_grid.ipynb' for more information

rvortbins = np.linspace(-5,5,150)
divbins = np.linspace(-5,5,150)
strainbins = np.linspace(0,8,150)
sgradbins = np.linspace(0,0.002,150)

def chunks(lst, n):
    """Yield successive n-sized chunks from lst. Thanks stack exchange!"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
        
daterange = list(chunks(ds_avg_child.ocean_time.sel(ocean_time = slice('2010-06-03', '2010-07-14')), 4))
for d in range(len(daterange)):
    #Relative vorticity
    vort_hist = histogram(rvort_norm.sel(ocean_time = daterange[d]), 
                          bins = [rvortbins], density = False)
    vort_hist.name = 'rvort'
    path = '/d2/home/dylan/JAMES/histogram_outputs/whole/rvort_whole_parent_2010_final_%s.nc' %d
    vort_hist.to_netcdf(path, mode = 'w')
    
    vort_hist_child = histogram(rvort_norm_child.sel(ocean_time = daterange[d]), 
                                bins = [rvortbins], density = False)
    vort_hist_child.name = 'rvort'
    path = '/d2/home/dylan/JAMES/histogram_outputs/whole/rvort_whole_child_2010_final_%s.nc' %d
    vort_hist_child.to_netcdf(path, mode = 'w')
    
    #Divergence
    div_hist = histogram(divergence_norm.sel(ocean_time = daterange[d]), 
                         bins = [divbins], density = False)
    div_hist.name = 'div'
    path = '/d2/home/dylan/JAMES/histogram_outputs/whole/divergence_whole_parent_2010_final_%s.nc' %d
    div_hist.to_netcdf(path, mode = 'w')
    
    div_hist_child = histogram(divergence_norm_child.sel(ocean_time = daterange[d]), 
                               bins = [divbins], density = False)
    div_hist_child.name = 'div'
    path = '/d2/home/dylan/JAMES/histogram_outputs/whole/divergence_whole_child_2010_final_%s.nc' %d
    div_hist_child.to_netcdf(path, mode = 'w')
    
    #Strain
    strain_hist = histogram(strain_norm.sel(ocean_time = daterange[d]), 
                            bins = [strainbins], density = False)
    strain_hist.name = 'strain'
    path = '/d2/home/dylan/JAMES/histogram_outputs/whole/strain_whole_parent_2010_final_%s.nc' %d
    strain_hist.to_netcdf(path, mode = 'w')
    
    strain_hist_child = histogram(strain_norm_child.sel(ocean_time = daterange[d]), 
                                  bins = [strainbins], density = False)
    strain_hist_child.name = 'strain'
    path = '/d2/home/dylan/JAMES/histogram_outputs/whole/strain_whole_child_2010_final_%s.nc' %d
    strain_hist_child.to_netcdf(path, mode = 'w')
    
    #Salinity gradient magnitude
    sgradmag_rho_hist = histogram(sgradmag_rho.sel(ocean_time = daterange[d]), 
                                  bins = [sgradbins], density = False)
    sgradmag_rho_hist.name = 'sgradmag'
    path = '/d2/home/dylan/JAMES/histogram_outputs/whole/sgradmag_rho_whole_parent_2010_final_%s.nc' %d
    sgradmag_rho_hist.to_netcdf(path, mode = 'w')
    
    sgradmag_rho_hist_child = histogram(sgradmag_rho_child.sel(ocean_time = daterange[d]), 
                                        bins = [sgradbins], density = False)
    sgradmag_rho_hist_child.name = 'sgradmag'
    path = '/d2/home/dylan/JAMES/histogram_outputs/whole/sgradmag_rho_whole_child_2010_final_%s.nc' %d
    sgradmag_rho_hist_child.to_netcdf(path, mode = 'w')

#Subset the data and compute median and standard deviation for Table 1. 
rv_subset = rv_parent[:,:,::3,::3]
newdims_parent = len(rv_subset.ocean_time)*len(rv_subset.xi_u)*len(rv_subset.eta_v)*len(rv_subset.s_rho)
rvnew = np.reshape(np.array(rv_subset), (newdims_parent), order = 'F')

rvmean = np.mean(rvnew)
np.save('/d2/home/dylan/JAMES/histogram_outputs/whole/stats/relvort_whole_subset_parent_mean.npy', rvmean)
rvmedian = np.median(rvnew)
np.save('/d2/home/dylan/JAMES/histogram_outputs/whole/stats/relvort_whole_subset_parent_median.npy', rvmedian)
rvstd = np.std(rvnew)
np.save('/d2/home/dylan/JAMES/histogram_outputs/whole/stats/relvort_whole_subset_parent_std.npy', rvstd)

div_subset = divergence_parent[:,:,::3,::3]
divnew = np.reshape(np.array(div_subset), (newdims_parent), order = 'F')

divmean = np.mean(divnew)
np.save('/d2/home/dylan/JAMES/histogram_outputs/whole/stats/divergence_whole_subset_parent_mean.npy', divmean)

divmedian = np.median(divnew)
np.save('/d2/home/dylan/JAMES/histogram_outputs/whole/stats/divergence_whole_subset_parent_median.npy', divmedian)
divstd = np.std(divnew)
np.save('/d2/home/dylan/JAMES/histogram_outputs/whole/stats/divergence_whole_subset_parent_std.npy', divstd)

sgradmag_subset = sgradmag_parent[:,:,::3,::3]
sgradmag_new = np.reshape(np.array(sgradmag_subset), (newdims_parent), order = 'F')

sgradmagmean = np.mean(sgradmag_new)
np.save('/d2/home/dylan/JAMES/histogram_outputs/whole/stats/sgradmag_whole_subset_parent_mean.npy', sgradmagmean)
sgradmagmedian = np.median(sgradmag_new)
np.save('/d2/home/dylan/JAMES/histogram_outputs/whole/stats/sgradmag_whole_subset_parent_median.npy', sgradmagmedian)
sgradmagstd = np.std(sgradmag_new)
np.save('/d2/home/dylan/JAMES/histogram_outputs/whole/stats/sgradmag_whole_subset_parent_std.npy', sgradmagstd)

strain_subset = strain_parent[:,:,::3,::3]
strainnew = np.reshape(np.array(strain_subset), (newdims_parent), order = 'F')

strainmean = np.mean(strainnew)
np.save('/d2/home/dylan/JAMES/histogram_outputs/whole/stats/strain_whole_subset_parent_mean.npy', strainmean)
strainmedian = np.median(strainnew)
np.save('/d2/home/dylan/JAMES/histogram_outputs/whole/stats/strain_whole_subset_parent_median.npy', strainmedian)
strainstd = np.std(strainnew)
np.save('/d2/home/dylan/JAMES/histogram_outputs/whole/stats/strain_whole_parent_parent_std.npy', strainstd)

#Child model - subsample every 15 points. 
#-----------

rv_child_subset = rv_child[:,:,::15,::15]
newdims_child = len(rv_child_subset.ocean_time)*len(rv_child_subset.xi_u)*len(rv_child_subset.eta_v)*len(rv_child_subset.s_rho)

rvnew_child = np.reshape(np.array(rv_child_subset), (newdims_child), order = 'F')

rvmean_child = np.mean(rvnew_child)
np.save('/d2/home/dylan/JAMES/histogram_outputs/whole/stats/relvort_whole_subset_child_mean.npy', rvmean_child)
rvmedian_child = np.median(rvnew_child)
np.save('/d2/home/dylan/JAMES/histogram_outputs/whole/stats/relvort_whole_subset_child_median.npy', rvmedian_child)
rvstd_child = np.std(rvnew_child)
np.save('/d2/home/dylan/JAMES/histogram_outputs/whole/stats/relvort_whole_subset_child_std.npy', rvstd_child)

dv_child_subset = divergence_child[:,:,::15,::15]
divnew_child = np.reshape(np.array(dv_child_subset), (newdims_child), order = 'F')

divmean_child = np.mean(divnew_child)
np.save('/d2/home/dylan/JAMES/histogram_outputs/whole/stats/divergence_whole_subset_child_mean.npy', divmean_child)
divmedian_child = np.median(divnew_child)
np.save('/d2/home/dylan/JAMES/histogram_outputs/whole/stats/divergence_whole_subset_child_median.npy', divmedian_child)
divstd_child = np.std(divnew_child)
np.save('/d2/home/dylan/JAMES/histogram_outputs/whole/stats/divergence_whole_subset_child_std.npy', divstd_child)

sgradmag_child_subset = sgradmag_child[:,:,::15,::15]
sgradmag_child = np.reshape(np.array(sgradmag_child_subset), (newdims_child), order = 'F')

sgradmagmean_child = np.mean(sgradmag_child)
np.save('/d2/home/dylan/JAMES/histogram_outputs/whole/stats/sgradmag_whole_subset_child_mean.npy', sgradmagmean_child)
sgradmagmedian_child = np.median(sgradmag_child)
np.save('/d2/home/dylan/JAMES/histogram_outputs/whole/stats/sgradmag_whole_subset_child_median.npy', sgradmagmedian_child)
sgradmagstd_child = np.std(sgradmag_child)
np.save('/d2/home/dylan/JAMES/histogram_outputs/whole/stats/sgradmag_whole_subset_child_std.npy', sgradmagstd_child)

strain_subset = strain_child[:,:,::15,::15]
strainnew_child = np.reshape(np.array(strain_subset), (newdims_child), order = 'F')

strainmean_child = np.mean(strainnew_child)
np.save('/d2/home/dylan/JAMES/histogram_outputs/whole/stats/strain_whole_subset_child_mean.npy', strainmean_child)
strainmedian_child = np.median(strainnew_child)
np.save('/d2/home/dylan/JAMES/histogram_outputs/whole/stats/strain_whole_subset_child_median.npy', strainmedian_child)
strainstd_child = np.std(strainnew_child)
np.save('/d2/home/dylan/JAMES/histogram_outputs/whole/stats/strain_whole_subset_child_std.npy', strainstd_child)

#Original code ------- depracated 
#Statistics - .sel(ocean_time = slice('2010-06-03', '2010-07-13'))
# newdims_parent = len(rv_parent.ocean_time)*len(rv_parent.xi_u)*len(rv_parent.eta_v)*len(rv_parent.s_rho)

# rvnew = np.reshape(np.array(rv_parent), (newdims_parent), order = 'F')

# # # rvmean = np.mean(rvnew)
# # # np.save('/d2/home/dylan/JAMES/histogram_outputs/whole/stats/relvort_whole_parent_mean.npy', rvmean)

# # rvmedian = np.median(rvnew)
# # np.save('/d2/home/dylan/JAMES/histogram_outputs/whole/stats/relvort_whole_parent_median.npy', rvmedian)

# # rvstd = np.std(rvnew)
# # np.save('/d2/home/dylan/JAMES/histogram_outputs/whole/stats/relvort_whole_parent_std.npy', rvstd)

# # divnew = np.reshape(np.array(divergence_parent), (newdims_parent), order = 'F')

# # # divmean = np.mean(divnew)
# # # np.save('/d2/home/dylan/JAMES/histogram_outputs/whole/stats/divergence_whole_parent_mean.npy', divmean)

# # divmedian = np.median(divnew)
# # np.save('/d2/home/dylan/JAMES/histogram_outputs/whole/stats/divergence_whole_parent_median.npy', divmedian)

# # divstd = np.std(divnew)
# # np.save('/d2/home/dylan/JAMES/histogram_outputs/whole/stats/divergence_whole_parent_std.npy', divstd)

# # sgradmag_surfnew = np.reshape(np.array(sgradmag_parent), (newdims_parent), order = 'F')

# # # sgradmagmean = np.mean(sgradmag_surfnew)
# # # np.save('/d2/home/dylan/JAMES/histogram_outputs/whole/stats/sgradmag_whole_parent_mean.npy', sgradmagmean)

# # sgradmagmedian = np.median(sgradmag_surfnew)
# # np.save('/d2/home/dylan/JAMES/histogram_outputs/whole/stats/sgradmag_whole_parent_median.npy', sgradmagmedian)

# # sgradmagstd = np.std(sgradmag_surfnew)
# # np.save('/d2/home/dylan/JAMES/histogram_outputs/whole/stats/sgradmag_whole_parent_std.npy', sgradmagstd)

# # strainnew = np.reshape(np.array(strain_parent), (newdims_parent), order = 'F')

# # # strainmean = np.mean(strainnew)
# # # np.save('/d2/home/dylan/JAMES/histogram_outputs/whole/stats/strain_whole_parent_mean.npy', strainmean)

# # strainmedian = np.median(strainnew)
# # np.save('/d2/home/dylan/JAMES/histogram_outputs/whole/stats/strain_whole_parent_median.npy', strainmedian)

# # strainstd = np.std(strainnew)
# # np.save('/d2/home/dylan/JAMES/histogram_outputs/whole/stats/strain_whole_parent_std.npy', strainstd)

# #Child: Reshaping into a numpy array will not work for the child model because the arrays are 
# # at least 96 GB, so too big for Copano (our cluster) to handle. Have to use xarray and dask 
# #--------
# newdims_child = len(rv_child.ocean_time)*len(rv_child.xi_u)*len(rv_child.eta_v)*len(rv_child.s_rho)

# rvnew_child = np.reshape(np.array(rv_child), (newdims_child), order = 'F')

# # rvmean_child = np.mean(rvnew_child)
# # np.save('/d2/home/dylan/JAMES/histogram_outputs/whole/stats/relvort_whole_child_mean.npy', rvmean_child)

# rvmedian_child = np.median(rvnew_child)
# np.save('/d2/home/dylan/JAMES/histogram_outputs/whole/stats/relvort_whole_child_median.npy', rvmedian_child)

# rvstd_child = np.std(rvnew_child)
# np.save('/d2/home/dylan/JAMES/histogram_outputs/whole/stats/relvort_whole_child_std.npy', rvstd_child)

# divnew_child = np.reshape(np.array(divergence_child), (newdims_child), order = 'F')

# # divmean_child = np.mean(divnew_child)
# # np.save('/d2/home/dylan/JAMES/histogram_outputs/whole/stats/divergence_whole_child_mean.npy', divmean_child)

# divmedian_child = np.median(divnew_child)
# np.save('/d2/home/dylan/JAMES/histogram_outputs/whole/stats/divergence_whole_child_median.npy', divmedian_child)

# divstd_child = np.std(divnew_child)
# np.save('/d2/home/dylan/JAMES/histogram_outputs/whole/stats/divergence_whole_child_std.npy', divstd_child)

# sgradmag_surfnew_child = np.reshape(np.array(sgradmag_surf_child), (newdims_child), order = 'F')

# # sgradmagmean_child = np.mean(sgradmag_surfnew_child)
# # np.save('/d2/home/dylan/JAMES/histogram_outputs/whole/stats/sgradmag_whole_child_median.npy', sgradmagmean_child)

# sgradmagmedian_child = np.median(sgradmag_surfnew_child)
# np.save('/d2/home/dylan/JAMES/histogram_outputs/whole/stats/sgradmag_whole_child_median.npy', sgradmagmedian_child)

# sgradmagstd_child = np.std(sgradmag_surfnew_child)
# np.save('/d2/home/dylan/JAMES/histogram_outputs/whole/stats/sgradmag_whole_child_std.npy', sgradmagstd_child)

# strainnew_child = np.reshape(np.array(strain_child), (newdims_child), order = 'F')

# # strainmean_child = np.mean(strainnew_child)
# # np.save('/d2/home/dylan/JAMES/histogram_outputs/whole/stats/strain_whole_child_median.npy', strainmean_child)

# strainmedian_child = np.median(strainnew_child)
# np.save('/d2/home/dylan/JAMES/histogram_outputs/whole/stats/strain_whole_child_median.npy', strainmedian_child)

# strainstd_child = np.std(strainnew_child)
# np.save('/d2/home/dylan/JAMES/histogram_outputs/whole/stats/strain_whole_child_std.npy', strainstd_child)