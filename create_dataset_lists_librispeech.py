import os
import random
# import glob
import numpy as np

random.seed(1234)

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'LibriSpeech')
print(data_dir)

# pathlist = glob.glob(data_dir+"*.spec.npy") # why doesn't this work?
pathlist = [os.path.join(root, fname) for root, _, fnames in os.walk(data_dir) for fname in fnames if fname.endswith(".mel.npy")]

# filter out bad items
pathlist = [path for path in pathlist if \
    os.path.exists(path.replace("wav48", "txt").replace(".mel.npy", ".txt")) and \
    os.path.exists(path.replace("wav48", "txt").replace(".mel.npy", ".phones")) \
]

seen_speakers = os.listdir(os.path.join(data_dir, 'train-clean-360'))
seen_speakers.remove('836')
seen_speakers.remove('501')
seen_speakers.remove('479')
seen_speakers.remove('472')
print(len(seen_speakers))

test_list   = [path for path in pathlist if not any([spkr in path for spkr in seen_speakers])]
remain_list = [path for path in pathlist if not path in test_list]

random.shuffle(remain_list)

# TODO: create test list by holding out all data of some specific speakers
cut = int(np.round(len(pathlist) * 0.8))

open('data/LibriSpeech/librispeech_train.list', 'w').write(" ".join(remain_list[   :cut]))
open('data/LibriSpeech/librispeech_eval.list',  'w').write(" ".join(remain_list[cut:   ]))
open('data/LibriSpeech/librispeech_test.list',  'w').write(" ".join(test_list))
