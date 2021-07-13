# pd.read_csv(engine='python')
# _csv.Error: line contains NUL
# https://stackoverflow.com/questions/7894856/line-contains-null-byte-in-csv-reader-python

from subprocess import check_call, CalledProcessError

datadir = '/home/hourstonh/Documents/climatology/data/MEDS_TSO/'
datlist = ['MEDS_ASCII_1991_2000.csv', 'MEDS_ASCII_2001_2010.csv',
           'MEDS_ASCII_2011_2020.csv']

PATH_TO_FILE = datadir + datlist[1]

try:
    check_call("sed -i -e 's|\\x0||g' {}".format(PATH_TO_FILE), shell=True)
except CalledProcessError as err:
    print(err)
