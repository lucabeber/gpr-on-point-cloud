from utils.gpr_on_point_cloud import *
import open3d as o3d
import numpy as np
from utils.pointcloud_utils import *
# Check if there is a GPU and if so use it with torch as default
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)
torch.set_default_device(device)

point_cloud_dir = "point_clouds/"
obj_name = "plate_shapes"



filename = f"{point_cloud_dir}{obj_name}.ply"
pcd_tmp = o3d.io.read_point_cloud(filename)

pcd = pcd_tmp.voxel_down_sample(voxel_size=0.001)  # downsample

vertices = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)[:, 0]

from utils.plotting_utils import *
import plotly.graph_objects as go

# Load gp on pc class
# ====================
l = 0.002
sigma = 1.0
n_eig = 100
km = rbf_manifold_kernel(vertices, l, sigma, n_eig)

print(km.shape)


train_x_real = torch.tensor(vertices, dtype=torch.float32)
train_y_real = torch.tensor(colors, dtype=torch.float32) 
indices = torch.randperm(train_x_real.size()[0])

train_size = int(0.3 * train_x_real.size()[0])
train_indices = indices[:train_size]
test_indices = indices[train_size:]

# Split the data into training and test sets
train_x, test_x = train_x_real[train_indices], train_x_real[test_indices]
train_y, test_y = train_y_real[train_indices], train_y_real[test_indices]

# Initialize the likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = GPROnPointCloud(train_x, train_y, likelihood, km, vertices)

# set to training mode and train
model.train()
likelihood.train()


# Get into evaluation (predictive posterior) mode and predict
model.eval()
likelihood.eval()

import time
start = time.time()
with torch.no_grad():
    observed_pred = likelihood(model(train_x_real))
end = time.time()
print(f"Time to predict: {end - start}")
mean = observed_pred.mean.cpu().numpy()
# var = observed_pred.variance.cpu().numpy()


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




