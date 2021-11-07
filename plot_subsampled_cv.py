import matplotlib.pyplot as plt
import os
from pandas import read_csv
import numpy as np

var_name = 'Oxy'
# files = [(10, 2013, 'JAS'), (5200, 1993, 'OND'), (20, 1991, 'AMJ')]
files = [(0, 1991, 'JFM'), (0, 1991, 'AMJ'), (0, 1991, 'JAS'), (0, 1991, 'OND'),
         (0, 2000, 'JFM'), (0, 2000, 'AMJ'), (0, 2000, 'JAS'), (0, 2000, 'OND')]
         # (50, 1995, 'JFM'), (50, 1995, 'AMJ'), (50, 1995, 'JAS'), (50, 1995, 'OND')]

# Number of testing points around each of corlen and epsilon2
nl = 1
ne = 1

# Which version of the files list used (have used different groups of files)
files_version = 3

# subsample_freq = [50, 40, 30, 20, 10, 5, 3]

# # Oxy 0m 2010 OND
# nl = 1
# ne = 1
# corlen = [771865.1501652099, 1977483.4682193287, 1480466.4698135417, 1377114.3516690833,
#           1377114.3516690833, 1377114.3516690833, 1377114.3516690833]
# epsilon2 = [0.8933671843019265, 0.08201383846810749, 0.07357737361295778,
#             0.06600873717044665, 0.06600873717044665, 0.06600873717044665,
#             0.06600873717044665]

# -------------------------------------------------------------------------
# Oxy 0m 2010 OND
# nl = 2
# ne = 2
# corlen = [1108368.7947165552, 6294627.058970837, 6294627.058970837, 1108368.7947165552,
#           6294627.058970837, 6294627.058970837, 6294627.058970837]
# epsilon2 = [0.8933671843019265, 0.06600873717044665, 0.01295563091280854,
#             0.8933671843019265, 0.017942672843332313, 0.01295563091280854,
#             0.014441138202687555]
# -------------------------------------------------------------------------
# # Oxy 10m 2010 OND
# nl = 1
# ne = 1
# subsample_freq = [50, 40, 30, 20, 10, 5]
# corlen = [717980.8509811073, 829793.4537187803, 1377114.3516690833, 1280977.297521509,
#           1280977.297521509, 1280977.297521509]
#
# epsilon2 = [0.8933671843019265, 0.8933671843019265, 0.07357737361295778, 0.05921865879254168,
#             0.05921865879254168, 0.05921865879254168]
# --------------------------------------------------------------------------

fig = plt.figure(dpi=100)

for f in files:
    standard_depth = f[0]
    year = f[1]
    szn = f[2]

    cv_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\diva_explore\\cross_validation\\'
    cv_filename = os.path.join(cv_dir + 'cv_{}_{}m_{}_{}_nle_1.csv'.format(
        var_name, standard_depth, year, szn))
    cv_df = read_csv(cv_filename)

    subsample_freq = np.array(cv_df.interval_size)
    corlen = np.array(cv_df.lenx)
    epsilon2 = np.array(cv_df.epsilon2)

    plt.plot(list(map(lambda y: 1/y, subsample_freq)), list(map(lambda x: x/1e6, corlen)),
             label='{}_{}m_{}_{}'.format(var_name, standard_depth, year, szn))

plt.legend(fontsize='small')

# Add vertical line
plt.axvline(x=1/10, color='r', linestyle='--')

plt.title('Correlation length estimate vs 6\' mask subsampling frequency, nl=ne={}'.format(nl))
plt.xlabel('Subsampling frequency')
plt.ylabel('Correlation length [*1e6]')

output_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\diva_explore\\cross_validation\\'
png_name = os.path.join(output_dir + '{}_corlen_vs_subsamp_freq_nle{}_v{}.png'.format(
    var_name, nl, files_version))
plt.savefig(png_name, dpi=100)
plt.close(fig)


fig = plt.figure(dpi=100)
for f in files:
    standard_depth = f[0]
    year = f[1]
    szn = f[2]

    cv_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\diva_explore\\cross_validation\\'
    cv_filename = os.path.join(cv_dir + 'cv_{}_{}m_{}_{}_nle_{}.csv'.format(
        var_name, standard_depth, year, szn, nl))
    cv_df = read_csv(cv_filename)

    subsample_freq = np.array(cv_df.interval_size)
    corlen = np.array(cv_df.lenx)
    epsilon2 = np.array(cv_df.epsilon2)

    plt.plot(list(map(lambda y: 1/y, subsample_freq)), epsilon2,
             label='{}_{}m_{}_{}'.format(var_name, standard_depth, year, szn))

plt.legend(fontsize='small')

# Add vertical line
plt.axvline(x=1/10, color='r', linestyle='--')

plt.title('Epsilon2 estimate vs 6\' mask subsampling frequency: nl=ne={}'.format(nl))
plt.xlabel('Subsampling frequency')
plt.ylabel('Epsilon2')

output_dir = 'C:\\Users\\HourstonH\\Documents\\NEP_climatology\\diva_explore\\cross_validation\\'
png_name = os.path.join(output_dir + '{}_epsilon2_vs_subsamp_freq_nle{}_v{}.png'.format(
    var_name, nl, files_version))
plt.savefig(png_name, dpi=100)
plt.close(fig)
