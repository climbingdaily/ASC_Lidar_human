import numpy as np
import os

directory = 'c:\\Users\\Daiyudi\\Desktop\\temp\\6\\imgs'
files = os.listdir(directory)
for f in files:
    try:
        int_a = int(f.split('.')[0])
        str_a = f'{int_a:05d}.'
        os.rename(os.path.join(directory, f), os.path.join(directory, str_a + f.split('.')[-1]))
    except:
        pass