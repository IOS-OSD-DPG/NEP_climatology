using DIVAnd
using DataFrames
using CSV
using NCDatasets
using Statistics

# Interpolate 30-year averaged data on trigrid to full gebco 6 minute land-sea mask


function len_from_fithorzlen()
end


function len_from_GCV()
    # Use generalized cross-validation to estimate the correlation length
end


function divand_to_full_NEP(data_filename, output_dir, mask_filename,
                            var_name, depth, szn, pm, pn)
    # Prepare inputs for the analysis

    # --------------------------------data--------------------------------------
	
    println(data_filename)

    data_ds = Dataset(data_filename)
    xobs = data_ds["longitude"][:]
    yobs = data_ds["latitude"][:]
    vobs = data_ds["SL_value_30yr_avg"]

    # Compute anomaly field
    vmean = mean(vobs)
    # vanom = convert(Array{Float64}, vobs .- vmean)
    vanom = vobs .- vmean

    println("Computed observation anomalies")
	
    # --------------------------------mask--------------------------------------
    
    mask_ds = Dataset(mask_filename)

    mask = mask_ds["mask"][:,:]  # [1:subsamp_interval:end, 1:subsamp_interval:end]
    mask = Bool.(mask)
    #     println(typeof(mask))

    # Get the 2d mesh grid longitude and latitude
    Lon1d = convert(Array{Float64}, mask_ds["lon"][:])
    Lat1d = convert(Array{Float64}, mask_ds["lat"][:])
    Lon2d, Lat2d = ndgrid(Lon1d, Lat1d)

    close(mask_ds)

    # -------------------------Generalized cross-validation--------------------------

    # Set the first guesses for lenx and leny as 1/10 the domain of the observations
    lenx_guess = 500e3 # (maximum(xobs) - minimum(xobs))  # /10
    leny_guess = 500e3 # (maximum(yobs) - minimum(yobs))  # /10
    println("lenx and leny guesses for GCV: ", lenx_guess, " ", leny_guess)

    signal_to_noise_ratio = 50.  # Default from Lu ODV session
    epsilon2 = 1/signal_to_noise_ratio  # 1.

    # Choose number of testing points around the current value of L (corlen)
    nl = 1

    # Choose number of testing points around the current value of epsilon2
    ne = 1

    # Choose cross-validation method
    # 1: full CV; 2: sampled CV; 3: GCV; 0: automatic choice between the three
    method = 3

    # Run generalized cross-validation
    println("Running GCV...")
    bestfactorl,bestfactore,cvval,cvvalues,x2Ddata,y2Ddata,cvinter,xi2D,yi2D = DIVAnd_cv(
        mask, (pm, pn), (Lon2d, Lat2d), (xobs, yobs), vanom, (lenx_guess, leny_guess),
        epsilon2, nl, ne, method)
    println("Completed GCV")
    
    new_lenx = bestfactorl * lenx
    new_leny = bestfactorl * leny
    new_epsilon2 = bestfactore * epsilon2

    lenx = new_lenx
    leny = new_leny
    epsilon2 = new_epsilon2
    println("Final lenx, leny, epsilon2: ", lenx, " ",leny, " ", epsilon2)

    # ----------------------------Run the analysis--------------------------------

    println("Running analysis...")

    try
        va = DIVAndrunfi(mask, (pm, pn), (Lon2d, Lat2d), (xobs, yobs), vanom,
                         (lenx, leny), epsilon2)

        # Add the output anomaly back to the mean of the observations
        vout = vmean .+ va

        println("Completed analysis")

        # ----------Export vout as a netCDF file in the same dims as the mask------

        vout_filename = string(output_dir, var_name, "_", depth, "m_", szn, 
	    "_analysis2d.nc")

        vout_ds = Dataset(vout_filename, "c")

        # Define lat and lon dims
        lon_dim_vout = defDim(vout_ds, "lon", length(Lon1d))
        lat_dim_vout = defDim(vout_ds, "lat", length(Lat1d))

        # Add lon and lat vars if can't add data to dims
        lon_var = defVar(vout_ds, "longitude", Float64, ("lon",))
        lat_var = defVar(vout_ds, "latitude", Float64, ("lat",))

        lon_var[:] = Lon1d
        lat_var[:] = Lat1d

        vout_var = defVar(vout_ds, "SL_value_30yr_avg", Float64, ("lon", "lat"))
        vout_var[:,:] = vout

        println(vout_ds)

        close(vout_ds)
    
        return vout_filename
    catch err
        # Return the error, whether LoadError, OutOfMemoryError, or other
        println(err)
 	return err
    end
end


# --------------------------------------------------------------------------------

# Define parameters
variable_name = "Oxy"
standard_depth = 0
season = "OND"

# Linux paths
# mask_folder = string("/home/hourstonh/Documents/climatology/data/value_vs_depth/",
#     "19_divand_to_full_NEP/masks/")
# input_folder = string("/home/hourstonh/Documents/climatology/data/value_vs_depth/",
# 	"18_30yr_avg/")
# output_folder = string("/home/hourstonh/Documents/climatology/data/value_vs_depth/",
# 	"19_divand_to_full_NEP/")

# Windows paths
mask_folder = string("C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\",
    "value_vs_depth\\19_divand_to_full_NEP\\masks\\")
corlen_folder = string("C:\\Users\\HourstonH\\Documents\\NEP_climatology\\diva_explore\\",
    "correlation_length\\")
input_folder = string("C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\",
    "value_vs_depth\\18_30yr_avg\\")
output_folder = string("C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\",
    "value_vs_depth\\19_divand_to_full_NEP\\")
pmn_folder = string("C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\",
    "value_vs_depth\\16_diva_analysis\\pmn\\")

data_file = string(input_folder, variable_name, "_", standard_depth, "m_", season, 
    "_tg_30yr_avg.nc")
mask_file = string(mask_folder, variable_name, "_", standard_depth, "m_", season, 
    "_mask_6min.nc")
corlen_file = string(corlen_folder, variable_name, "_fithorzlen_mean_lenxy_100m.csv")

# -----------------------------------------------------------------------------------

# Open the pmn netCDF file
pmn_filename = string(pmn_folder, "divand_pmn_for_mask_6min.nc")

pmn_ds = Dataset(pmn_filename)

# Get the inverse of the resolution of the grid
pm_diva = convert(Array{Float64}, pmn_ds["pm"][:,:])
pn_diva = convert(Array{Float64}, pmn_ds["pn"][:,:])

close(pmn_ds)

# Check if necessary files exist before running analysis
if isfile(data_file) && isfile(mask_file)
    ncname = divand_to_full_NEP(data_file, output_folder, mask_file,
        variable_name, standard_depth, season, pm_diva, pn_diva)
end
