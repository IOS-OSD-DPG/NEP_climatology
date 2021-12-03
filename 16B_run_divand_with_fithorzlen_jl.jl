using DIVAnd
using CSV
using NCDatasets
using DataFrames
using Statistics

# To run in command prompt or something
# julia compute_DIVAnd_pmn.jl


function run_divand_with_fithorzlen(obs_dir, mask_dir, fitcor_dir, output_dir,
        var_name, depth, yr, szn, pm, pn)
    # Open the required files

    # Read in standard level data file
    obs_filename = string(obs_dir, var_name, "_", depth, "m_", yr, "_", szn, ".csv")

    println(obs_filename)

    # Pipe operator to dataframe
    obs_df = CSV.File(obs_filename) |> DataFrame

    if size(obs_df)[1] == 0
        println("DataFrame empty -- skip")
    end

    # Convert values from Float64 to Float32 in an effort to avoid memory errors
    xobs = convert(Array{Float64}, obs_df[!, :Longitude])
    yobs = convert(Array{Float64}, obs_df[!, :Latitude])
    vobs = obs_df[!, :SL_value]

    # Compute anomaly field
    vmean = mean(vobs)
    vanom = convert(Array{Float64}, vobs .- vmean)

    println("Computed observation anomalies")

    # --------------------Set correlation length from fithorzlen-------------------

    fitcor_filename = string(fitcor_dir, "Oxy_fithorzlen_mean_lenxy_100m.csv")

    fitcor_df = CSV.File(fitcor_filename) |> DataFrame

    year_rownum = yr - 1990
    fitcor_lenxy = fitcor_df[year_rownum, szn]

    # --------------------------------Read in mask---------------------------------

    mask_filename = string(mask_dir, var_name, "_", standard_depth, "m_", 
        yr, "_", szn, "_mask_6min.nc")

    mask_ds = Dataset(mask_filename)

    mask = mask_ds["mask"][:,:]  # [1:subsamp_interval:end, 1:subsamp_interval:end]
    mask = Bool.(mask)
    #     println(typeof(mask))

    # Get the 2d mesh grid longitude and latitude
    Lon1d = convert(Array{Float64}, mask_ds["lon"][:])
    Lat1d = convert(Array{Float64}, mask_ds["lat"][:])
    Lon2d, Lat2d = ndgrid(Lon1d, Lat1d)

    close(mask_ds)

    # -------------------------Set some more parameters---------------------------

    signal_to_noise_ratio = 50.  # Default from Lu ODV session
    epsilon2_guess = 1/signal_to_noise_ratio  # 1.

    # Choose number of testing points around the current value of L (corlen)
    nl = 1

    # Choose number of testing points around the current value of epsilon2
    ne = 1

    # Choose cross-validation method
    # 1: full CV; 2: sampled CV; 3: GCV; 0: automatic choice between the three
    method = 3

    # ----------------------------Run the analysis--------------------------------

    println("Running analysis...")

    va = DIVAndrunfi(mask, (pm, pn), (Lon2d, Lat2d), (xobs, yobs), vanom,
                     (fitcor_lenxy, fitcor_lenxy), epsilon2_guess)

    # Add the output anomaly back to the mean of the observations
    vout = vmean .+ va

    println("Completed analysis")

    # ----------Export vout as a netCDF file in the same dims as the mask---------

    vout_filename = string(output_dir, var_name, "_", depth, "m_", yr, "_", 
        szn, "_analysis2d.nc")

    vout_ds = Dataset(vout_filename, "c")

    # Define lat and lon dims
    lon_dim_vout = defDim(vout_ds, "lon", length(Lon1d))
    lat_dim_vout = defDim(vout_ds, "lat", length(Lat1d))

    # Add lon and lat vars if can't add data to dims
    lon_var = defVar(vout_ds, "longitude", Float64, ("lon",))
    lat_var = defVar(vout_ds, "latitude", Float64, ("lat",))

    lon_var[:] = Lon1d
    lat_var[:] = Lat1d

    vout_var = defVar(vout_ds, "vout", Float64, ("lon", "lat"))
    vout_var[:,:] = vout

    println(vout_ds)

    close(vout_ds)
    
    return vout_filename
end


# -----------------------------------------------------------------------------

variable_name = "Oxy"
standard_depth = 0
years = [1994]  # collect(1991:1999)  # Creates increasing array
season = "JAS"
# subsamp_interval = 1

# Paths for windows
obs_folder = string("C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\",
    "value_vs_depth\\14_sep_by_sl_and_year\\")

mask_folder = string("C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\",
    "value_vs_depth\\16_diva_analysis\\masks\\")

pmn_folder = string("C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\",
    "value_vs_depth\\16_diva_analysis\\pmn\\")

output_folder = string("C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\",
    "value_vs_depth\\16_diva_analysis\\analysis\\fithorzlen\\")

fitcor_folder = string("C:\\Users\\HourstonH\\Documents\\NEP_climatology\\",
    "diva_explore\\correlation_length\\")

# # Paths for Linux:
# obs_dir = string("/home/hourstonh/Documents/climatology/data/",
#     "value_vs_depth/14_sep_by_sl_and_year/")

# mask_dir = string("/home/hourstonh/Documents/climatology/data/",
#     "value_vs_depth/16_diva_analysis/masks/")

# pmn_dir = string("/home/hourstonh/Documents/climatology/data/",
#     "value_vs_depth/16_diva_analysis/pmn/")

# fitcor_dir = "/home/hourstonh/Documents/climatology/diva_explore/correlation_length/"

# years = [1991:1:2020;]
# println(years)
# szns = ["JFM", "AMJ", "JAS", "OND"]

# ------------------------------------------------------------------------------

# Open the pmn netCDF file
pmn_filename = string(pmn_folder, "divand_pmn_for_mask_6min.nc")

pmn_ds = Dataset(pmn_filename)

# Get the inverse of the resolution of the grid
pm_diva = convert(Array{Float64}, pmn_ds["pm"][:,:])
pn_diva = convert(Array{Float64}, pmn_ds["pn"][:,:])

close(pmn_ds)

for y=years
    ncname = run_divand_with_fithorzlen(obs_folder, mask_folder, fitcor_folder, 
        output_folder, variable_name, standard_depth, y, season, pm_diva, pn_diva)
    
    println(ncname)
end
