import os
import random
# import glob
import numpy as np

random.seed(1234)

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'VCTK')
print(data_dir)

# pathlist = glob.glob(data_dir+"*.spec.npy") # why doesn't this work?
pathlist = [os.path.join(root, fname) for root, _, fnames in os.walk(data_dir) for fname in fnames if fname[-9:] == ".spec.npy" ]

# filter out bad items
pathlist = [path for path in pathlist if \
    os.path.exists(path.replace(".spec", ".mel")) and \
    os.path.exists(path.replace("wav48", "txt").replace(".spec.npy", ".txt")) and \
    os.path.exists(path.replace("wav48", "txt").replace(".spec.npy", ".phones")) \
]

random.shuffle(pathlist)

a = int(np.round(len(pathlist) * 0.7))
b = a + int(np.round(len(pathlist) * 0.2))

open('data/VCTK/vctk_train.list', 'w').write(" ".join(pathlist[ :a]))
open('data/VCTK/vctk_eval.list',  'w').write(" ".join(pathlist[a:b]))
open('data/VCTK/vctk_test.list',  'w').write(" ".join(pathlist[b: ]))
