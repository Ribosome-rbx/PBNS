import os
import numpy as np
from IO import readPC2, loadInfo
import scipy.io as sio
import pickle 

def pickle_load(file):
    """
    Load a pickle file.
    """
    with open(file, 'rb') as f:
        loadout = pickle.load(f)

    return loadout

path = os.path.dirname(os.path.realpath(__file__))
body_data = loadInfo(os.path.join(path, 'Model/Body.mat'))
outfit_data = loadInfo(os.path.join(path, 'Model/Outfit_config.mat'))
body_pkl = pickle_load('/home/borong/Desktop/PBNS/PBNS/Data/smpl/model_f.pkl')

breakpoint()
# # Load PC2 file
# project_path = os.path.dirname(os.path.realpath(__file__))
# file_path = os.path.join(project_path, 'results/outfit/outfit.pc2')

# file = readPC2(file_path)
# np.save(file_path.replace(".pc2", ".npy"), file['V'])
# print("Done!")