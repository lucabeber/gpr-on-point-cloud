import torch 
import gpytorch
import robust_laplacian
import scipy.sparse.linalg as sla
import plotly.graph_objects as go

device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Using device: ", device)
# torch.set_default_device(device)

class GPRPointCloudModel:
    def __init__(self, pc, l=0.01, sigma=1.0, n_eig=100):
        # if not isinstance(pc, torch.Tensor):
        #     raise TypeError("pc must be a torch.Tensor")
        if not isinstance(l, float):
            raise TypeError("l must be a float")
        if not isinstance(sigma, float):
            raise TypeError("sigma must be a float")
            
        self.pc = pc
        self.lengthscale = l
        self.sigma = sigma
        self.n_eig = n_eig

        # Compute eigenvalues and eigenvectors
        self.precomputation()

    def precomputation(self):
        # Compute the Laplacian matrix
        C, M = robust_laplacian.point_cloud_laplacian(self.pc)
        self.pc = torch.tensor(self.pc, dtype=torch.float32, device=device)
        # Compute eigenvalues and eigenvectors
        evals, evecs = sla.eigsh(C, self.n_eig, M, sigma=1e-8)

        # Convert to torch tensors on the correct device
        self.evals = torch.tensor(evals, dtype=torch.float32, device=device)
        self.evecs = torch.tensor(evecs, dtype=torch.float32, device=device)

        # Compute C_inf
        diag_M = torch.tensor(M.diagonal(), dtype=torch.float32, device=device)
        total_volume = M.diagonal().sum()

        tmp = torch.tensor([torch.sum(torch.exp(- self.lengthscale**2 * self.evals / 2) * self.evecs.T[:, idx] * self.evecs.T[:, idx]) for idx in range(len(diag_M))])
        self.C_inf = torch.sum(tmp * diag_M) / total_volume

    class RBF_RM(gpytorch.kernels.Kernel):
        is_stationary = True

        def __init__(self, parent_model):
            super().__init__()
            self.parent = parent_model

        def forward(self, x1, x2, diag=False, **params):
            l = self.parent.lengthscale
            sigma = self.parent.sigma

            # Compute the kernel matrix
            kernel_matrix = torch.zeros(x1.size(0), x2.size(0), dtype=torch.float32)  # Initialize kernel matrix

                    # Ensure x1 and x2 are 2D tensors
            if x1.dim() == 1:
                x1 = x1.unsqueeze(0)  # Add batch dimension
            if x2.dim() == 1:
                x2 = x2.unsqueeze(0)  # Add batch dimension

            # Find the nearest neighbors of x1 and x2 in a single operation
            distances_x1_x2 = torch.cdist(torch.cat((x1, x2), dim=0), self.parent.pc, p=2)  # Euclidean distance
            _, indices = torch.topk(distances_x1_x2, k=1, largest=False)  # Find the nearest neighbor
            neig_x1, neig_x2 = indices[:x1.size(0)].squeeze(), indices[x1.size(0):].squeeze()
            
            # Compute the kernel matrix using broadcasting
            eig_vals_exp = torch.exp(-l**2.0 * self.parent.evals / 2.0)  # Precompute exponential terms
            eig_vecs1 = self.parent.evecs.T[:,neig_x1] # Convert eigenvectors to tensor
            if x1.size(0) == 1:
                eig_vecs1 = eig_vecs1.unsqueeze(1).clone().detach()
            eig_vecs2 = self.parent.evecs.T[:,neig_x2] # Convert eigenvectors to tensor
            if x2.size(0) == 1:
                eig_vecs2 = eig_vecs2.unsqueeze(1).clone().detach() # Convert eigenvectors to tensor
            kernel_matrix = (
                sigma**2 / self.parent.C_inf
                * torch.einsum(
                "k,ki,kj->ij",
                eig_vals_exp,
                eig_vecs1,
                eig_vecs2
                )
            )
            # Return the diagonal if diag=True
            if diag:
                return torch.diag(kernel_matrix)
            return kernel_matrix

    class GPROnPointCloud(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood, parent_model):
            super(GPRPointCloudModel.GPROnPointCloud, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = GPRPointCloudModel.RBF_RM(parent_model)

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
    fig.update_layout(scene_aspectmode="cube")
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
        
    
