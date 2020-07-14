# make sure data is present in data/VCTK

# LIBROSA BUG:
#   if librosa is giving you trouble, change the line `import numba.decorators` to `import numba.core.decorators` in decorators.py
#   https://github.com/deezer/spleeter/issues/419#issuecomment-643574915

# it's recommended to install pip packages in an isolated conda environment
pip install -r requirements.txt
sudo apt-get install festival # required for text-to-phoneme
cd pre-train/reader
python extract_features.py ../../data/VCTK audio 
python extract_features.py ../../data/VCTK text
cd ../../
python create_dataset_lists.py 

# Ready! 

# cd pre-train 
# bash run.sh