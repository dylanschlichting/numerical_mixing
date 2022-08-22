# `JAMES_numerical_mixing`

JAMES_numerical_mixing contains all python scripts needed to reproduce figures shown in Schlichting et al. JAMES. If you want to access the TXLA model output, see X. This repository is organized as follows:
* analysis_scripts: Contains scripts used to generate data for all non-tracer budget figures (i.e. histograms, Hovmoller diagrams)
* budget_scripts: Contains scripts used to generate each term in the tracer budgets. Detailed description below.

## Installation
A list of all packages for the conda environment used is shown in 'copano_env.yml'. The crux of the work done here relies on xroms. See https://github.com/xoceanmodel for more information. 

## Tracer budget scripts: All scripts use xroms to open the model ouput, then subset each variable to the location of the two-way nested grid. To check that the grids are aligned, see /analysis_notebooks/grid_sanity_check.ipynb. Each script saves the variable of interest one day at a time in a for loop to avoid crashing the cluster.
 * advection.py: Calculates all volume integrated tracer advection terms
 * tendency_avg.py: Calculates all volume integrated tracer tendency terms
 * mixing.py: Calculates volume integrated physical and numerical mixing
 * surface.py: Calculates all volume integrated surface fluxes

## Analysis_notebooks:
 * budget_analysis: Computes time series based on tracer budgets, extra terms, and ratios of different mixing quantities.
 * horizontal_mixing: Compares horizontal to vertical diffusive salt fluxes to show that horizontal mixing can be neglected in the numerical mixing analysis
 * grid_sanity_check: Produces plots of the parent and child grids to show they are properly aligned for slices listed in the budget scripts.
