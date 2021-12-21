import pandas as pd
import os
import glob
from numpy import zeros, where

# Count the number of observations per file
# Save the findings in a csv file

indir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data\\value_vs_depth\\' \
        '14_sep_by_sl_and_year\\'

infiles = glob.glob(indir + '*.csv')

outdf = pd.DataFrame(index=list(map(lambda x: os.path.basename(x), infiles)))

# Initialize column to store observation counts
outdf['num_obs'] = zeros(len(infiles), dtype='int')

for f in infiles:
    indf = pd.read_csv(f)
    outdf.loc[os.path.basename(f), 'num_obs'] = len(indf)

# Save df of counts
outdir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\data_explore\\oxygen\\' \
         'post_qc_obs_counts\\'
outdf_filename = os.path.join(outdir + 'post_qc9_obs_counts.csv')

outdf.to_csv(outdf_filename, index=True)


# Print summary statistics
print('Number of files:', len(outdf))
for i in range(11):
    print(i, 'obs:', len(where(outdf.num_obs == i)[0]))

print('Over 10 obs:', len(where(outdf.num_obs > 10)[0]))
