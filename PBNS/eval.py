import os
import sys
import numpy as np
import tensorflow as tf
from math import floor
import shutil
import glob

from Data.data import Data
from Model.PBNS import PBNS

from util import parse_args, model_summary
from IO import writeOBJ, writePC2Frames
from util import *

"""
This script will load a PBNS checkpoint
Predict results for a set of poses (at 'Data/test.npy')
Store output animation data in 'results/' folder
	Body:
	- 'results/body.obj'
	- 'results/body.pc2'
	Outfit:
	- 'results/outfit.obj'
	- 'results/ouftit.pc2'
	- 'results/rest.pc2' -> before skinning
"""

""" PARSE ARGS """
# parse_args()
gpu_id, name, folder, checkpoint = '1', '00152_outer', '00152_outer', '00152_outer'
type = folder.split('_')[-1].lower()
object = f"../Templates/{folder}/{type}"
body = f"../Templates/{folder}/{folder}"

type_folder = type[0].upper() + type[1:]
_takes_path = glob.glob(os.path.join("/home/borong/Desktop/hood_data", folder.split('_')[0], type_folder, "Take*"))
take_list = [path.split("Take")[-1] for path in _takes_path]
rest_pose = pickle_load(os.path.dirname(os.path.realpath(__file__))+f"/Templates/{folder}/restpose.pkl")


# gpu_id, name, object, body, checkpoint = parse_args(train=False)
name = os.path.abspath(os.path.dirname(__file__)) + '/results/' + name + '/'
if os.path.isdir(name):
	shutil.rmtree(name)
os.mkdir(name)
checkpoint = os.path.abspath(os.path.dirname(__file__)) + '/checkpoints/' + checkpoint

""" PARAMS """
batch_size = 32

""" GPU """
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

""" MODEL """
print("Building model...")
model = PBNS(object=object, body=body, checkpoint=checkpoint)
tgts = model.gather() # model weights
model_summary(tgts)


for take in take_list:
	take_name = "Take"+take
	tmp_name = name + take_name + "/"
	os.mkdir(tmp_name)
	""" DATA """
	print("Reading data...")
	# val_poses = f"Data/vali.npy"
	val_poses = f"Templates/{folder}/{take_name}.npy"
	val_data = Data(val_poses, model._shape, model._gender, rest_pose, batch_size=batch_size, mode='test')
	val_steps = floor(val_data._n_samples / batch_size)

	""" CREATE BODY AND OUTFIT .OBJ FILES """
	print("store template obj")
	writeOBJ(tmp_name + 'body.obj', model._body, model._body_faces)
	writeOBJ(tmp_name + f'{type}.obj', model._T, model._F)

	""" EVALUATION """
	print("")
	print("Evaluating...")
	print("--------------------------")
	step = 0
	for poses, G, body in val_data._iterator:
		if poses.shape[-1] > 72: poses = truncate_pose_smplx2smpl(poses)
		pred = model(poses, G, np.array(val_data.SMPL.transl))
		writePC2Frames(tmp_name + 'body.pc2', body.numpy())
		writePC2Frames(tmp_name + f'{type}.pc2', pred.numpy())
		writePC2Frames(tmp_name + 'rest.pc2', (model._T[None] + model.D).numpy())

		sys.stdout.write('\r\tStep: ' + str(step + 1) + '/' + str(val_steps))
		sys.stdout.flush()
		step += 1
	print("")
	print("")
	print("DONE!")
	print("")

	# transfer pc2 into npy
	from IO import readPC2
	val_transl = f"Templates/{folder}/{take_name}_trans.npy"
	add_transl = os.path.isfile(val_transl)
	if add_transl: translation = np.load(val_transl)

	for f in ['body.pc2', f'{type}.pc2']:
		file_path = tmp_name + f
		file = readPC2(file_path)
		if add_transl:
			file['V'] += translation[:,None,:]
		np.save(file_path.replace(".pc2", ".npy"), file['V'])
	print("pc2 to npy --> Done!")