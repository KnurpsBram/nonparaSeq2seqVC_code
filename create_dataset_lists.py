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

seen_speakers = ['p336', 'p240', 'p262', 'p333', 'p297', 'p339', 'p276', 'p269', 'p303', 'p260', 'p250', 'p345', 'p305', 'p283', 'p277', 'p302', 'p280', 'p295', 'p245', 'p227', 'p257', 'p282', 'p259', 'p311', 'p301', 'p265', 'p270', 'p329', 'p362', 'p343', 'p246', 'p247', 'p351', 'p263', 'p363', 'p249', 'p231', 'p292', 'p304', 'p347', 'p314', 'p244', 'p261', 'p298', 'p272', 'p308', 'p299', 'p234', 'p268', 'p271', 'p316', 'p287', 'p318', 'p264', 'p313', 'p236', 'p238', 'p334', 'p312', 'p230', 'p253', 'p323', 'p361', 'p275', 'p252', 'p374', 'p286', 'p274', 'p254', 'p310', 'p306', 'p294', 'p326', 'p225', 'p255', 'p293', 'p278', 'p266', 'p229', 'p335', 'p281', 'p307', 'p256', 'p243', 'p364', 'p239', 'p232', 'p258', 'p267', 'p317', 'p284', 'p300', 'p288', 'p341', 'p340', 'p279', 'p330', 'p360', 'p285']

test_list   = [path for path in pathlist if not any([spkr in path for spkr in seen_speakers])]
remain_list = [path for path in pathlist if not path in test_list]

random.shuffle(remain_list)

# TODO: create test list by holding out all data of some specific speakers
cut = int(np.round(len(pathlist) * 0.8))

open('data/VCTK/vctk_train.list', 'w').write(" ".join(remain_list[   :cut]))
open('data/VCTK/vctk_eval.list',  'w').write(" ".join(remain_list[cut:   ]))
open('data/VCTK/vctk_test.list',  'w').write(" ".join(test_list))
