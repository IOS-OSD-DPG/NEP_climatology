# NEP_climatology
**INCOMPLETE** \
For treating data for NEP ocean climatology update

## Requirements:
* Python >= 3.7
* Unix-like environment or Windows environment
  
## Installation
To install the project requirements in a conda virtual environment (my_venv): \
`conda activate my_venv` \
`conda install pandas numpy gsw tqdm matplotlib r-oce conda-build dask` \
`conda install -c conda-forge rpy2` 

Install basemap: follow the instructions at https://matplotlib.org/basemap/users/index.html (or install easily with Pycharm if using)

Install Rtools: https://cran.r-project.org/bin/windows/Rtools/rtools40.html

Install Julia: https://julialang.org/downloads/

Install DIVAnd in Julia: \
`using Pkg` \
`Pkg.add("DIVAnd")`

## Miscellaneous links
* CIOOS Pacific search by organization: https://catalogue.cioospacific.ca/organization
* Institute of Ocean Sciences Water Properties: https://www.waterproperties.ca/
* NODC WODSelect: https://www.ncei.noaa.gov/access/world-ocean-database-select/bin/dbsearch.pl
* `oce` oceApprox() function documentation: https://dankelley.github.io/oce/reference/oceApprox.html
* DIVAnd documentation: https://gher-ulg.github.io/DIVAnd.jl/latest/#DIVAnd.jl-documentation
* DIVA GitHub repository: https://github.com/gher-ulg/DIVA
* DIVA User Guide: https://github.com/gher-ulg/Diva-User-Guide/raw/master/DivaUserGuide.pdf
