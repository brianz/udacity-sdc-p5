import urllib.request
import os
import zipfile


root_url = 'https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/'
vehicles = 'vehicles.zip'
non_vehicles = 'non-vehicles.zip'

for fn in (vehicles, non_vehicles):
    dir_name = fn.split('.')[0]
    if not os.path.isdir(dir_name):
        print('Downloading %s...' % fn)
        urllib.request.urlretrieve(root_url + fn, fn)
        print('Extracting %s...' % fn)
        with zipfile.ZipFile(fn, 'r') as z:
            z.extractall()
        print('Done!')
