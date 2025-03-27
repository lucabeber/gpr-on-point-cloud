import torch 
import gpytorch
import robust_laplacian
import scipy.sparse.linalg as sla


class GPRPointCloudModel:
    class RBF_RM(gpytorch.kernels.Kernel):
        is_stationary = True

        # this is the kernel function
        def forward(self, x1, x2, **params):
            l = l_dist

            # Ensure x1 and x2 are 2D tensors
            if x1.dim() == 1:
                x1 = x1.unsqueeze(0)  # Add batch dimension
            if x2.dim() == 1:
                x2 = x2.unsqueeze(0)  # Add batch dimension

            # Compute the kernel matrix
            kernel_matrix = torch.zeros(x1.size(0), x2.size(0), dtype=torch.float32)  # Initialize kernel matrix

            # Find the nearest neighbors of x1 and x2 in a single operation
            distances_x1_x2 = torch.cdist(torch.cat((x1, x2), dim=0), pc, p=2)  # Euclidean distance
            _, indices = torch.topk(distances_x1_x2, k=1, largest=False)  # Find the nearest neighbor
            neig_x1, neig_x2 = indices[:x1.size(0)].squeeze(), indices[x1.size(0):].squeeze()
            
            # Compute the kernel matrix using broadcasting
            eig_vals_exp = torch.exp(-l**2.0 * torch.tensor(evals, dtype=torch.float32) / 2.0)  # Precompute exponential terms
            eig_vecs1 = torch.tensor(evecs.T[:,neig_x1], dtype=torch.float32) # Convert eigenvectors to tensor
            if x1.size(0) == 1:
                eig_vecs1 = eig_vecs1.unsqueeze(1).clone().detach()
            eig_vecs2 = torch.tensor(evecs.T[:,neig_x2], dtype=torch.float32) # Convert eigenvectors to tensor
            if x2.size(0) == 1:
                eig_vecs2 = eig_vecs2.unsqueeze(1).clone().detach()

            # Compute the kernel matrix
            kernel_matrix = (
                1.0 / C_inf
                * torch.einsum(
                "k,ki,kj->ij",
                eig_vals_exp,
                eig_vecs1,
                eig_vecs2
                )
            )
            return kernel_matrix
        
    # Define the GP model
    class GPROnPointCloud(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(GPRPointCloudModel.GPROnPointCloud, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = GPRPointCloudModel.RBF_RM()

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        
    
    def __init__(self, pc):
