import torch 
import gpytorch
import robust_laplacian
import scipy.sparse.linalg as sla
import plotly.graph_objects as go
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Using device: ", device)
torch.set_default_device(device)


def rbf_manifold_kernel(pc, lengthscale=0.01, sigma=1.0, n_eig=100):
    # Compute the Laplacian matrix
    C, M = robust_laplacian.point_cloud_laplacian(pc)

    # Compute eigenvalues and eigenvectors
    evals, evecs = sla.eigsh(C, n_eig, M, sigma=1e-8)

    # Convert to torch tensors on the correct device
    evals = torch.tensor(evals, dtype=torch.float32, device=device)
    evecs = torch.tensor(evecs.T, dtype=torch.float32, device=device)

    # Compute C_inf
    diag_M = torch.tensor(M.diagonal(), dtype=torch.float32, device=device)
    total_volume = M.diagonal().sum()

    tmp = torch.tensor([torch.sum(torch.exp(- lengthscale**2 * evals / 2) * evecs[:, idx] * evecs[:, idx]) for idx in range(len(diag_M))], dtype=torch.float32, device=device)
    C_inf = torch.sum(tmp * diag_M) / total_volume
    
    eig_vals_exp = torch.exp(-lengthscale**2.0 * evals / 2.0)

    # Compute the kernel matrix
    kernel_matrix_pc = (
            sigma**2 / C_inf
            * torch.einsum(
            "k,ki,kj->ij",
            eig_vals_exp,
            evecs,
            evecs
            )
        )
    del C, M, evals, evecs, diag_M, total_volume, tmp, C_inf, eig_vals_exp
    return kernel_matrix_pc

class RBF_RM(gpytorch.kernels.Kernel):
        is_stationary = True

        def __init__(self, kernel_matrix_pc, pc):
            super().__init__()
            self.kernel_matrix_pc = kernel_matrix_pc
            self.pc = torch.tensor(pc, dtype=torch.float32, device=device)

        def forward(self, x1, x2, diag=False, **params):
            start = time.time() 
            # Ensure x1 and x2 are 2D tensors
            if x1.dim() == 1:
                x1 = x1.unsqueeze(0)  # Add batch dimension
            if x2.dim() == 1:
                x2 = x2.unsqueeze(0)  # Add batch dimension
            # Move x1 and x2 to the correct device if necessary
            if device.type == "cuda" and not x1.is_cuda:
                x1 = x1.to(device)
            if device.type == "cuda" and not x2.is_cuda:
                x2 = x2.to(device)
            # Find the nearest neighbors of x1 and x2 in a single operation
            distances_x1_x2 = torch.cdist(torch.cat((x1, x2), dim=0), self.pc, p=2)  # Euclidean distance
            _, indices = torch.topk(distances_x1_x2, k=1, largest=False)  # Find the nearest neighbor
            neig_x1, neig_x2 = indices[:x1.size(0)].squeeze(), indices[x1.size(0):].squeeze()

            kernel_matrix = torch.zeros(x1.size(0), x2.size(0), dtype=torch.float32)  # Initialize kernel matrix

            if kernel_matrix.is_cuda:
                kernel_matrix = self.kernel_matrix_pc[neig_x1[:, None], neig_x2[None, :]]
            else:
                kernel_matrix = self.kernel_matrix_pc[neig_x1[:, None].cpu(), neig_x2[None, :].cpu()]

            if diag:
                return torch.diag(kernel_matrix)
            end = time.time()
            print(f"Time taken: {end - start}")
            return kernel_matrix

class GPROnPointCloud(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel_matrix_pc, pc):
        super(GPROnPointCloud, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = RBF_RM(kernel_matrix_pc, pc)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        

def plot_point_cloud(points, point_colors=None, **kwargs):
    scatter = go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=point_colors,
            colorscale='Inferno',
            opacity=0.8,
            showscale=True
        ),
        **kwargs
    )
    return scatter


def update_figure(fig):
    """Utility to clean up figure"""
    fig.update_layout(scene_aspectmode="data")
    fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)
    # fig.update_traces(showscale=False, hoverinfo="none")
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))

    fig.update_layout(
        plot_bgcolor="black",  # Set the plot background to black
        paper_bgcolor="black",  # Set the paper background to black
    )
    fig.update_layout(
        scene=dict(
            xaxis=dict(showbackground=False, showticklabels=False, visible=False),
            yaxis=dict(showbackground=False, showticklabels=False, visible=False),
            zaxis=dict(showbackground=False, showticklabels=False, visible=False),
        )
    )
    return fig
        
    
