import pickle
import copy
import os

import numpy as np
from matplotlib import pyplot as plt
from aitviewer.renderables.meshes import Meshes
from aitviewer.scene.camera import PinholeCamera
from aitviewer.utils import path
from aitviewer.viewer import Viewer
from aitviewer.renderables.meshes import VariableTopologyMeshes as VTMeshes
from aitviewer.renderables.point_clouds import PointClouds as VTPoints
from collections import defaultdict

def pickle_load(file):
    """
    Load a pickle file.
    """
    with open(file, 'rb') as f:
        loadout = pickle.load(f)

    return loadout

def pickle_save(path, obj):
    """
    Save a pickle file.
    """
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def quads2tris(F):
	F_out = []
	for f in F:
		if len(f) <= 3: F_out += [f]
		elif len(f) == 4:
			F_out += [
				[f[0], f[1], f[2]],
				[f[0], f[2], f[3]]
			]
	return np.array(F_out, np.int32)

def read_obj(filename):
    # Read the OBJ file and store vertices and faces in a dictionary
    vertices = []
    faces = []
    
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 0:
                continue
                
            if parts[0] == 'v':
                vertex = tuple(map(float, parts[1:]))
                vertices.append(vertex)
            elif parts[0] == 'f':
                face = tuple(map(int, [p.split('/')[0] for p in parts[1:]]))
                faces.append(face)

    vertices = np.array(vertices)
    faces = np.array(faces) - 1

    obj_data = {'vertices': vertices, 'faces': faces}
    return obj_data

def main():
    results_folder, take = '00152_outer', 'Take10'
    _t = results_folder.split("_")[-1]
    type = "pants" if _t == "Template" else _t.lower()
    project_path = os.path.dirname(os.path.realpath(__file__))
    ########### load garment ###########
    garment_vertices = np.load(os.path.join(project_path, 'results', results_folder, take, f'{type}.npy'))
    garment_obj = read_obj(os.path.join(project_path, 'results', results_folder, take, f'{type}.obj'))
    garment_faces = quads2tris(garment_obj['faces'])

    cmap = plt.get_cmap('gist_rainbow')
    garment_frames = {'vertices':[], 'faces':[], "colors":[]}
    for i in range(len(garment_vertices)):
        if i == 0: garment_vertices[i] = garment_obj['vertices']
        garment_frames['vertices'].append(garment_vertices[i])
        garment_frames['faces'].append(garment_faces)
        garment_frames['colors'].append(cmap([0.7]*len(garment_vertices[i])))

    ########### load body ###########
    body_vertices = np.load(os.path.join(project_path, 'results', results_folder, take, 'body.npy'))
    body_obj = read_obj(os.path.join(project_path, 'results', results_folder, take, 'body.obj'))
    body_faces = body_obj['faces']

    cmap = plt.get_cmap('gist_rainbow')
    body_frames = {'vertices':[], 'faces':[], "colors":[]}
    for i in range(len(body_vertices)):
        if i == 0: body_vertices[i] = body_obj['vertices']
        body_frames['vertices'].append(body_vertices[i])
        body_frames['faces'].append(body_faces)
        # colors = np.array([cmap(1.)]*len(body_vertices[0]))
        # body_frames['colors'].append(colors)



    viewer = Viewer()
    # garment
    garment_mesh = VTMeshes(garment_frames['vertices'], garment_frames['faces'], vertex_colors=garment_frames['colors'], name='garment')
    garment_mesh.backface_culling = False
    garment_point = VTPoints(garment_frames['vertices'], garment_frames['colors'], name='garment')

    # body
    body_mesh = VTMeshes(body_frames['vertices'], body_frames['faces'], name='body')
    body_mesh.backface_culling = False
    # body_point = VTPoints(body_frames['vertices'], body_frames['colors'], name='body')

    # set camera position
    positions, targets = path.lock_to_node(garment_mesh, [0, 0, 3])
    camera = PinholeCamera(positions, targets, viewer.window_size[0], viewer.window_size[1], viewer=viewer)
    viewer.scene.nodes = viewer.scene.nodes[:5]
    viewer.scene.add(garment_mesh)
    # viewer.scene.add(garment_point)
    viewer.scene.add(body_mesh)
    # viewer.scene.add(body_point)

    viewer.scene.add(camera)
    viewer.set_temp_camera(camera)

    viewer.playback_fps = 30
    viewer.run()

if __name__ == "__main__":
    main()