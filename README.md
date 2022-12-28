# `JAMES_numerical_mixing`

JAMES_numerical_mixing contains python tools needed to reproduce figures shown in Schlichting et al. JAMES. If you want to access the TXLA model output, see X. This repository is organized as follows:
* analysis_scripts: Contains scripts used to generate data for non-tracer budget figures (i.e. histograms, )
* budget_scripts: Contains scripts used to generate each term in the tracer budgets. Detailed description below.
* figures: Contains Jupyter notebooks used to generate all manuscript figures.
* quality_control: Contains a mix of notebooks and scripts to debug and verify various calculations. Detailed description below.  


## Installation
A list of all packages for the conda environment used is shown in 'copano_env.yml'. The crux of the work done here relies on xroms. See https://github.com/xoceanmodel for more information. xroms is not required for analysis... but it will help avoid re-inventing the wheel when it comes to cumbersome calculations like Jacobians or the nonlinear EOS for seawater density.

## Budget_scripts: All scripts use xroms to open the model ouput, then subset each variable to the location of the two-way nested grid. To check that the grids are aligned, see /analysis_notebooks/grid_sanity_check.ipynb. Each script saves the variable of interest one day at a time in a for loop to avoid crashing the cluster.
 * advection.py: Calculates volume integrated tracer advection terms
 * tendency_new.py: Calculates all integrated tracer tendency terms
 * mixing.py: Calculates volume integrated physical and numerical mixing
 * surface.py: Calculates all volume integrated surface fluxes
 * hmix_diffusion.py: Calculates horizontal physical mixing and horizontal diffusive boundary fluxes
 * sbar.py: Calculates volume-averaged salinity


## quality_control: Contains scripts relevant to QC tracer budgets, grid indexing, and analyses computed during first round of revisions to address reviewer comments
  * histograms_*_test.py: scripts used to check that discreetly computing PDFs by chunking in time is identical to computing all at once with xhistogram's "density=True" syntax.
  * grid_sanity_check.ipynb: notebook to verify location of parent/child grids match.
  * QC_diffusion.ipynb: notebook to ensure horizontal diffusive boundary fluxes are computed correctly, verifies math in Section 2.3. 

## figures
  * Each notebook here is named after figure number.

## analysis_scripts:
  * depth_integrated.py: computes depth and time-integrated numerical and physical mixing for coarse/fine simulations
  * histograms_fronts_whole.py: histograms for whole water column sorted by fronts, i.e. $/zeta/f>1$ as described in Section 3.
  * histograms_fronts_surface.py: histograms for surface water column sorted by fronts, i.e. $/zeta/f>1$ as described in Section 3.
  * histogram_surface_stats.py: computes statistics for histogram variables of surface values used in Table 1.
  * histogram_*_test.py: Another way to compute histograms, just a QC check if HPRC resources are limited and we have to try out less intensive, but less efficient methods.
  * thomas_angle.py: Computes instability metric $\phi_{Rib}$ as described in Thomas et al. DSR II (2013): Symmetric instability in the gulf stream. This is used in Section 5.2 to verify whether new dynamical processes have emerged in fine simulation. Not shown in main text of manuscript.
