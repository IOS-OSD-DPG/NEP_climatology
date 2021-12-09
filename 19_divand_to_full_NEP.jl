using DIVAnd
using DataFrames
using CSV
using NCDatasets
using Statistics

# Interpolate 30-year averaged data on trigrid to full gebco 6 minute land-sea mask


function divand_to_full_NEP(input_dir, output_dir, mask_dir, var_name, depth, szn)
	# Prepare inputs for the analysis
	# --------------------------------data--------------------------------------
	data_filename = string(input_dir, var_name, "_", depth, "m_",
		szn, "_30yr_avg.csv")
	
	println(data_filename)

	data_df = CSV.File(data_filename) |> DataFrame

	xobs = convert(Array{Float64}, data_df[!, "Longitude [degrees East]"])
    yobs = convert(Array{Float64}, data_df[!, "Latitude [degrees North]"])
    vobs = obs_df[!, :SL_value_30yr_avg]

    # Compute anomaly field
    vmean = mean(vobs)
    vanom = convert(Array{Float64}, vobs .- vmean)

    println("Computed observation anomalies")

	# ------------------------correlation length--------------------------------
	fitcor_lenxy = nothing
	
	# --------------------------------mask--------------------------------------	
	mask_filename = string(mask_dir, var_name, "_", depth, "m_", 
    	szn, "_mask_6min.nc")

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

    vout_var = defVar(vout_ds, "vout", Float64, ("lon", "lat"))
    vout_var[:,:] = vout

    println(vout_ds)

    close(vout_ds)
    
    return vout_filename
end


# --------------------------------------------------------------------------------

# Linux paths
mask_folder = string("/home/hourstonh/Documents/climatology/data/value_vs_depth/", 	"19_divand_to_full_NEP/masks/")

input_folder = string("/home/hourstonh/Documents/climatology/data/value_vs_depth/",
	"18_30yr_avg/")

output_folder = string("/home/hourstonh/Documents/climatology/data/value_vs_depth/",
	"19_divand_to_full_NEP/")

corlen_folder = nothing

# Define parameters
variable_name = "Oxy"
standard_depth = 0
season = "JFM"

ncname = divand_to_full_NEP(input_folder, output_folder, mask_folder, corlen_folder,
	variable_name, standard_depth, season)

