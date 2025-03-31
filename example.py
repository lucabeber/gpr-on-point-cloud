from utils.gpr_on_point_cloud import *
import open3d as o3d
import numpy as np
# from utils.pointcloud_utils import *
# Check if there is a GPU and if so use it with torch as default
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)
torch.set_default_device(device)

point_cloud_dir = "point_clouds/"
obj_name = "plate_shapes"

l = 0.003
sigma = 1.0
n_eig = 200

hypers = {
    "lengthscale": l,
    "outputscale": 1.0,
    "noise": 0.1,
    "mean": 0.0
}

filename = f"{point_cloud_dir}{obj_name}.ply"
pcd_tmp = o3d.io.read_point_cloud(filename)

pcd = pcd_tmp.voxel_down_sample(voxel_size=0.002)  # downsample

vertices = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)[:, 0]

from utils.plotting_utils import *
import plotly.graph_objects as go

# Load gp on pc class
# ====================
km = rbf_manifold_kernel(vertices, l, sigma, n_eig)

print(km.shape)


train_x_real = torch.tensor(vertices, dtype=torch.float32)
train_y_real = torch.tensor(colors, dtype=torch.float32) 
indices = torch.randperm(train_x_real.size()[0])

train_size = int(0.1 * train_x_real.size()[0])
train_indices = indices[:train_size]
test_indices = indices[train_size:]

# Split the data into training and test sets
train_x, test_x = train_x_real[train_indices], train_x_real[test_indices]
train_y, test_y = train_y_real[train_indices], train_y_real[test_indices]

# Initialize the likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = GPROnPointCloud(train_x, train_y, likelihood, km, vertices)

model.likelihood.noise_covar.noise = hypers["noise"]

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

print("Noise: ", model.likelihood.noise_covar.noise.item())
print("Mean: ", model.mean_module.constant.item())

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


## Test with classic RBF kernel
class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood_real):
        super(GPModel, self).__init__(train_x, train_y, likelihood_real)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=3))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
likelihood_real = gpytorch.likelihoods.GaussianLikelihood()
model_real = GPModel(train_x, train_y, likelihood_real)

# Training the model
model_real.train()
likelihood_real.train()


model_real.covar_module.base_kernel.lengthscale = hypers["lengthscale"]
model_real.covar_module.outputscale = hypers["outputscale"]
model_real.likelihood.noise_covar.noise = hypers["noise"]
# model_real.mean_module.constant = hypers["mean"]



# Switch to evaluation mode
model_real.eval()
likelihood_real.eval()

# Plot the predicted density
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = likelihood_real(model_real(train_x_real))

mean_rbf = observed_pred.mean.cpu().numpy()

plot = plot_point_cloud(train_x_real.cpu().numpy(), point_colors=mean_rbf)
fig = go.Figure(plot)
update_figure(fig)
fig.update_layout(
    scene_camera=camera
)

fig.show('browser')

print("Mean root squared error RBF euclidian: ", np.sqrt(np.mean((mean_rbf - colors) ** 2)))
print("Mean root squared error RBF on point cloud: ", np.sqrt(np.mean((mean - colors) ** 2)))