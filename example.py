from utils.gpr_on_point_cloud import *
import open3d as o3d
import numpy as np
from utils.pointcloud_utils import *
point_cloud_dir = "point_clouds/"
obj_name = "plate_shapes"

# Select the object and load the point cloud
# ==========================================
filename = f"{point_cloud_dir}{obj_name}.ply"

# obj_name = "bun270_X" # Stanford bunny with X projected as the target
obj_name = "plate_shapes"  # random IKEA plate with hand-drawn shapes
# obj_name = "cup_X" # random cup that we scanned with X projected as the target

experiment_index = 2  # choose which initial position to use from x0_arr_10.npz

class param:
    pass  # c-style struct


param.timesteps = 500  # total simulation timesteps

# tuning: [1,100] increasing alpha result in global exploration closer to SS
# decreasing alpha result in local exploration lower limited
param.alpha = 100
# tuning: [25, 300] increasing the value bring more flexibility in selecting
# the pairing alpha. Lowering the value results in faster computation
param.nb_eigen = 200

# eigen corresponds to using Laplacian eigenbasis and using spectral acceleration
# when integrating the diffusion (heat) equation
# exact corresponds to using implicit time stepping for integrating
param.method = "eigen"
# param.method = "exact"

# voxel filter size for downsampling the point cloud
param.voxel_size = 0.003
# radius for the agent footprint that'd be used in coverage
param.agent_radius = 2.5 * param.voxel_size # for the cup and the bunny
# param.agent_radius = 5 * param.voxel_size  # for the plate
# define speed and acceleration in terms of voxel size
param.max_velocity = 1 * param.voxel_size
param.max_acceleration = 1 * param.max_velocity

# tuning: doesn't have much effect on exploration so we keep it at 1
param.source_strength = 1

# number of iterations for the non-stationary heat equation
# we kept it at 1 in this work to decrease the computational cost and
# it was enough for the exploration task
param.ndt = 1
# max. num. of neighbors to consider for computing the neighbors in agent radius
param.nb_max_neighbors = 500
# num. of neighbors to consider for tangent space and gradient computation
param.nb_minimum_neighbors = 20
# num. of neighbors to consider for implicitly determining the boundary
# setting this lower in bunny resutls in right ear considered as a seperate body
# setting this higher in bunny results in the right ear being considered as part
# of the main body
param.nb_boundary_neighbors = 40

filename = f"{point_cloud_dir}{obj_name}.ply"
pcloud = process_point_cloud(filename, param)
# pcd_tmp = o3d.io.read_point_cloud(filename)

# pcd = pcd_tmp.voxel_down_sample(voxel_size=0.003)  # downsample

# vertices = np.asarray(pcd.points)
# colors = np.asarray(pcd.colors)[:, 0]
from utils.plotting_utils import *
import plotly.graph_objects as go

plot = visualize_point_cloud(
    pcloud.vertices, 
    colors=pcloud.u0,
 point_size=5
)

# print("Number of vertices: ", vertices.shape[0])

# Load gp on pc class
# ====================
l = 0.002
sigma = 1.0
n_eig = 50
gp = GPRPointCloudModel(pcloud.vertices, l, sigma, n_eig)



train_x_real = torch.tensor(pcloud.vertices, dtype=torch.float32)
train_y_real = torch.tensor(pcloud.u0, dtype=torch.float32) 


# Initialize the likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = gp.GPROnPointCloud(train_x_real, train_y_real, likelihood, gp)

# set to training mode and train
model.train()
likelihood.train()


# Get into evaluation (predictive posterior) mode and predict
model.eval()
likelihood.eval()
# observed_pred = predict(model, likelihood, train_x)
with torch.no_grad():
    observed_pred = likelihood(model(train_x_real))

mean = observed_pred.mean.cpu().numpy()
var = observed_pred.variance.cpu().numpy()

print(mean.shape)
camera = dict(
    up=dict(x=0, y=1, z=0),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=0, y=0.7, z=1.25)
)

plot = plot_point_cloud(train_x_real.cpu().numpy(), point_colors=mean)
fig = go.Figure(plot)
update_figure(fig)
fig.update_layout(
    scene_camera=camera
)

fig.show('browser')