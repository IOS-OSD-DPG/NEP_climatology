# NEP_climatology
For producing temperature, salinity and oxygen climatology layers for the Northeast Pacific ocean, averaged over 1991-2020. The output is intended as an update from the 1981-2010 climatologies from Christian and Foreman (2013). 

Author: Hana Hourston (@hhourston)

## Requirements:
* Python >= 3.7
* Julia >= 1.6.3 (older versions weren't tested but may be ok)
* R >= 4.1.1 (older versions weren't tested but may be ok)
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

## Processing steps
### Data preparation
1. Create value-vs-depth (vvd) csv tables
2. Add duplicate check flags to the vvd tables
3. Apply the duplicate check flags to the vvd tables
4. Do a latitude-longitude check to screen out data outside of the predetermined area of 30$^\circ$ N $<=$ 60$^\circ$ N latitude and -160$^\circ$ E $<=$ -115$^\circ$ E longitude.
5. Apply the source quality depth and data flags to the vvd data
6. Remove data with NaN depths or values
7. WOA18 depth checks
8. WOA18 range checks
9. WOA18 gradient checks
10. Vertical interpolation using modified Reiniger-Ross (1968) method to standard levels (a combination of WOA18 standard levels and standard levels from Christian and Foreman, 2013)
11. Replicate value check
    1. To flag replicated values produced from vertical interpolation
12. Standard deviation checks on 5-degree squares
13. Separate data by standard level and season (needed for ODV DIVA interpolation method)
14. Separate data by standard level, season and year (for Julia DIVAnd method)
15. Deprec

### DIVAnd steps 
(Option B for variational analysis; option A uses ODV DIVA)
16. DIVAnd in Julia to create fields for NEP on regular GEBCO 6-minute grid
17. Linear interpolation to the unstructured triangle grid from Christian and Foreman (2013)
18. Average unstructured triangle grid data over 1991-2020 for each season/depth combination
19. DIVAnd the data from step 18 to the full NEP on regular GEBCO 6-minute grid
20. Linear interpolation to the unstructured triangle grid from Christian and Foreman (2013)

## Miscellaneous links
* CIOOS Pacific search by organization: https://catalogue.cioospacific.ca/organization
* Institute of Ocean Sciences Water Properties: https://www.waterproperties.ca/
* NODC WODSelect: https://www.ncei.noaa.gov/access/world-ocean-database-select/bin/dbsearch.pl
* `oce` oceApprox() function documentation: https://dankelley.github.io/oce/reference/oceApprox.html
* DIVAnd documentation: https://gher-ulg.github.io/DIVAnd.jl/latest/#DIVAnd.jl-documentation
* DIVA GitHub repository: https://github.com/gher-ulg/DIVA
* DIVA User Guide: https://github.com/gher-ulg/Diva-User-Guide/raw/master/DivaUserGuide.pdf
* GEBCO elevation data access: https://www.gebco.net/data_and_products/gridded_bathymetry_data/

## References
Christian, J. R. and Foreman, M.G.G. 2013. Climate Trends and Projections for the Pacific Large
Aquatic Basin.Can. Tech. Rep. Fish. Aquat. Sci. 3032: xi + 112 p.
