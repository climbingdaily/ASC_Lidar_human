from cv2 import sort
import numpy as np
import os

directory = 'C:\\Users\\DAI\\Desktop\\temp\\imgs'
files = sorted(os.listdir(directory))
count = 0
for f in files:
    try:
        int_a = int(f.split('.')[0])
        str_a = f'{count:04d}.'
        os.rename(os.path.join(directory, f), os.path.join(directory, str_a + f.split('.')[-1]))
        count +=1
    except:
        pass