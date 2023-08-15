# `numerical_mixing`
```numerical_mixing``` contains information needed to reproduce figures shown in Schlichting et al. (2023) *JAMES*. If you want to access the TXLA model output, see the directory "TXLA ROMS nested model for SUNRISE" at https://hafen.geos.tamu.edu/thredds/catalog/catalog.html. 
## How to use the code (if working in Python)
If you work in Python, we recommend creating a custom conda environment so package version control can be managed easily. To run scripts/notebooks in this repository, an environment can be installed by running 

        conda install --file copano_env_revised.yml
        
The environment used for initial submission can be installed with ```copano_env_initial.yml```. The yaml file for the revised submission is ```copano_env_revised.yml```. The crux of the work done here relies on ```xroms```. See https://github.com/xoceanmodel for more information. xroms is not required for analysis but it will substantially shorten cumbersome calculations like Jacobians or the nonlinear equation of state for seawater density. ```xroms``` has undergone significant development since creation of this repository, so be wary of version control issues! 

## Repository organization:
> - ```/analysis_scripts/```: Contains scripts used to generate data for non-tracer budget figures (i.e. histograms)
> > - ```depth_integrated.py```: computes depth and time-integrated numerical and physical mixing for coarse/fine simulations
> > - ```histograms_fronts_whole.py```: histograms for whole water column sorted by fronts only, i.e., normalized relative vorticity, divergence, strain, and horizontal salinity gradient magnitude as described in Section 3.
> > - ```histograms_fronts_surface.py```: histograms for surface water column sorted by fronts. 
> - ```/budget_scripts/```: Contains scripts used to generate each term in the tracer budgets. Each script subsets the variable of interest one day (or less) at a time in a for loop to avoid limitations with HPRC resources.
> > - ```tendency_new.py```: Calculates all integrated tracer tendency terms
> > - ```advection.py```: Calculates volume integrated tracer advection terms
> > - ```mixing.py```: Calculates volume integrated physical and numerical mixing
> > - ```surface.py```: Calculates volume integrated surface fluxes
> > - ```hmix_diffusion.py```: Calculates horizontal physical mixing and horizontal diffusive boundary fluxes
> > - ```sbar.py```: Calculates volume-averaged salinity
> - ```/figures/```: Contains Jupyter notebooks used to generate all manuscript figures. Notebooks are named numerically.
> - ```latex_backups/```: Contains backup of overleaft LaTex files for manuscript.
> - ```/quality_control/```: Contains a mix of notebooks and scripts to debug and verify several calculations. Also includes checks for calculations that came up during the review process. 
> > - ```histograms_*_test.py```: scripts used to check that discreetly computing PDFs by chunking in time is identical to computing all at once with ```xhistogram's``` ```density=True``` syntax.
> > - ```grid_sanity_check.ipynb```: notebook to verify location of parent/child grids match.
> > - ```QC_diffusion.ipynb```: notebook to ensure horizontal diffusion terms are computed correctly.
> > - ```QC_tendency.ipynb```: notebook to ensure volume-integrated time rate of change terms for tracers are computed correctly.
> > - ```QC_advection.ipynb```: notebook to ensure horizontal advective boundary fluxes are computed correctly, verifies math in Section 2.3.
> > - ```QC_surface.ipynb```: notebook to ensure surface diffusive boundary fluxes are computed correctly, verifies math in Section 2.2-2.3.
> > - ```histogram_surface_stats.py```: computes statistics for histogram variables of surface values used in Table 1.
> > - ```thomas_angle.py```: Computes instability metric $\phi_{Rib}$ as described in Thomas et al. DSR II (2013): Symmetric instability in the Gulf Stream. This is used in Section 5.2 to verify whether new dynamical processes have emerged in fine simulation. Not shown in main text of manuscript.
