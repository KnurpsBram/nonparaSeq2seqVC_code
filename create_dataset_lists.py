import os
import random
# import glob
import numpy as np

random.seed(1234)

data_dir = 'data/VCTK/'
# pathlist = glob.glob(data_dir+"*.spec.npy") # why doesn't this work?
pathlist = [os.path.join(root, fname) for root, _, fnames in os.walk('data/VCTK/') for fname in fnames if fname[-9:] == ".spec.npy" ]

random.shuffle(pathlist)

a = int(np.round(len(pathlist) * 0.7))
b = a + int(np.round(len(pathlist) * 0.2))

open('data/VCTK/vctk_train.list', 'w').write("\n".join(pathlist[ :a]))
open('data/VCTK/vctk_eval.list',  'w').write("\n".join(pathlist[a:b]))
open('data/VCTK/vctk_test.list',  'w').write("\n".join(pathlist[b: ]))
