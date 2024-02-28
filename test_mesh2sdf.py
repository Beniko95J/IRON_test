import os
import sys
import trimesh
import mesh2sdf
import numpy as np
import time
import mcubes

filename = 'datasets/datasets_single/000.obj'

mesh_scale = 0.9
size = 256
level = 2 / size

mesh = trimesh.load(filename, force='mesh')

# normalize mesh
vertices = mesh.vertices
bbmin = vertices.min(0)
bbmax = vertices.max(0)
center = (bbmin + bbmax) * 0.5
scale = 2.0 * mesh_scale / (bbmax - bbmin).max() # -0.8 ~ 0.8
vertices = (vertices - center) * scale
# import pdb; pdb.set_trace()

# fix mesh
t0 = time.time()
sdf, mesh = mesh2sdf.compute(
    vertices, mesh.faces, size, fix=True, level=level, return_mesh=True)
t1 = time.time()

sdf_grid = {
    'grid': sdf,
    'center': center,
    'scale': scale,
}

np.save('datasets/datasets_single/sdf_grid.npy', sdf_grid)

# output
# mesh.vertices = mesh.vertices / scale + center
# mesh.export(filename[:-4] + '.fixed.obj')
# np.save(filename[:-4] + '.npy', sdf)

vertices, triangles = mcubes.marching_cubes(-sdf, 0.0)
# import pdb; pdb.set_trace()
b_min_np = np.array([-1., -1., -1.])
b_max_np = np.array([1., 1., 1.])
vertices = vertices / (size - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]

# import pdb; pdb.set_trace()
mesh = trimesh.Trimesh(vertices, triangles)
mesh.export(filename[:-4] + '.mcube.obj')

print('It takes %.4f seconds to process %s' % (t1-t0, filename))