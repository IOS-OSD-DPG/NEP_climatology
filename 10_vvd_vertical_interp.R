library(oce)
library(stringr)

# Copied from 10_vertical_interp_rr.py

var_name = 'Temp'

file_sl = '/home/hourstonh/Documents/climatology/lu_docs/WOA_MForeman_Standard_Depths.txt'

vvd_indir = '/home/hourstonh/Documents/climatology/data/value_vs_depth/9_gradient_check/'
vvd_outdir = '/home/hourstonh/Documents/climatology/data/value_vs_depth/10_vertical_interp/'

# Read standard levels into an array with increasing order
df_sl = read.csv(file_sl, header = FALSE)
sl_arr = sort(na.omit(unlist(df_sl, use.names = F)))

vvd_files = Sys.glob(paste0(vvd_indir, sprintf('*%s*done.csv', var_name)))
print(length(vvd_files))

for (i in 2:length(vvd_files)){
  print(basename(vvd_files[i]))
  df_vvd = read.csv(vvd_files[i])  # Indexing starts at 1 not 0
  
  # Initialize data frame for output
  df_out = data.frame()
  
  sl_colnames = c(colnames(df_vvd)[1:6], c('SL_depth_m', 'SL_value'))
  
  # https://stat.ethz.ch/pipermail/r-help/2012-August/334347.html
  unique_mask <- !duplicated(df_vvd$Profile_number)  ## logical vector of unique values
  prof_start_int = seq_along(df_vvd$Profile_number)[unique_mask]  ## indices
  
  # Iterate through all the profiles
  for (j in 1:length(prof_start_int)){
    if (j == length(prof_start_int)){
      end_ind = length(df_vvd)
    } else {
      end_ind = prof_start_int[j + 1] - 1
    }
    
    indices = seq(prof_start_int[j], end_ind)
    depths = df_vvd$Depth_m[indices]
    values = df_vvd$Value[indices]
    
    if (all(diff(depths) < 0)){
      print(paste('Warning: profile number', df_vvd$Profile_number[indices[1]],
            'is an upcast'))
      # Sort the depths in increasing order
      depths_sort_inds = order(depths, decreasing=F)
      depths = depths[depths_sort_inds]
      values = values[depths_sort_inds]
    }
    
    # Extract the subset of standard levels to use for vertical interpolation
    # The last element in depths is the deepest one
    if (depths[1] < 5){
      # If there are data above 5m depth, then the surface value is taken to
      # equal to the shallowest recorded value.
      sl_subsetter = which(sl_arr <= depths[length(depths)])
    } else {
      sl_subsetter = which((sl_arr >= depths[1]) & 
                             (sl_arr <= depths[length(depths)]))
    }
    
    # Skip computations if no standard level matches
    if (length(sl_subsetter) == 0){
      print('Warning: No standard level matches')
    } else if (length(sl_subsetter) > 0){
      z_out = sl_arr[sl_subsetter]
      
      # Interpolate to standard levels
      rsl_values = oceApprox(depths, values, z_out, 'unesco')
      
      # Update length of profile information to length of interpolated value array
      profile_number = rep(df_vvd$Profile_number[indices[1]], length(z_out))
      cruise_number = rep(df_vvd$Cruise_number[indices[1]], length(z_out))
      instrument_type = rep(df_vvd$Instrument_type[indices[1]], length(z_out))
      date_string = rep(df_vvd$Date_string[indices[1]], length(z_out))
      latitude = rep(df_vvd$Latitude[indices[1]], length(z_out))
      longitude = rep(df_vvd$Longitude[indices[1]], length(z_out))
    }
    
    df_add = data.frame(
      Profile_number = profile_number,
      Cruise_number = cruise_number,
      Instrument_type = instrument_type,
      Date_string = date_string,
      Latitude = latitude,
      Longitude = longitude,
      SL_depth_m = z_out,
      SL_value = rsl_values
    )
    
    df_out = rbind(df_out, df_add)
  }
  
  
  df_out_name = paste0(
    vvd_outdir, str_replace(basename(vvd_files[i]), 'grad_check_done', 'rr'))
  
  write.csv(df_out, df_out_name, row.names = F)
}
