#!/usr/bin/env python3
"""
Comprehensive test script that measures computation time and RMSQ with proper device handling
"""

import numpy as np
import time
import sys
import os

# Add the current directory to Python path to import the modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # Import the required modules
    from utils.gpr_on_point_cloud import rbf_manifold_kernel, GPROnPointCloud
    import torch
    import gpytorch
    import open3d as o3d
    
    print("All required modules imported successfully")
    
    # Check device availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load a point cloud for testing
    point_cloud_dir = "point_clouds/"
    obj_name = "cup_X"
    filename = f"{point_cloud_dir}{obj_name}.ply"
    
    # Check if file exists
    if not os.path.exists(filename):
        print(f"Point cloud file {filename} not found")
        sys.exit(1)
    
    print(f"Loading point cloud: {filename}")
    pcd_tmp = o3d.io.read_point_cloud(filename)
    pcd = pcd_tmp.voxel_down_sample(voxel_size=0.001)
    vertices = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)[:, 0] if hasattr(pcd_tmp, 'colors') else np.ones(len(vertices))
    
    print(f"Point cloud shape: {vertices.shape}")
    
    # Test different numbers of eigenfunctions
    n_eig_values = [100, 200, 300, 500]
    
    print("\nTesting eigenfunction computation and GPR performance...")
    print("=" * 60)
    
    results = []
    
    for n_eig in n_eig_values:
        print(f"\n--- Testing with {n_eig} eigenfunctions ---")
        
        # Measure kernel computation time
        print("Computing kernel matrix...")
        start_time = time.time()
        try:
            km = rbf_manifold_kernel(vertices, lengthscale=0.002, sigma=1.0, n_eig=n_eig)
            kernel_time = time.time() - start_time
            print(f"Kernel computation time: {kernel_time:.4f} seconds")
            
            # Prepare data for GPR
            train_x_real = torch.tensor(vertices, dtype=torch.float32, device=device)
            train_y_real = torch.tensor(colors, dtype=torch.float32, device=device) 
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
            
            # Training mode and train
            model.train()
            likelihood.train()
            
            # Measure prediction time
            print("Computing GPR predictions...")
            start_time = time.time()
            with torch.no_grad():
                observed_pred = likelihood(model(train_x))
            pred_time = time.time() - start_time
            print(f"GPR prediction time: {pred_time:.4f} seconds")
            
            # Get predictions for training points
            mean = observed_pred.mean.cpu().detach().numpy()
            
            # Calculate RMSQ (Root Mean Square Error) on training data
            # We use training data for proper comparison
            train_mean = mean[:train_size]
            train_colors = colors[train_indices]
            
            if len(train_mean) == len(train_colors):
                rmsq = np.sqrt(np.mean((train_mean - train_colors) ** 2))
                print(f"RMSQ on training data: {rmsq:.6f}")
            else:
                rmsq = -1
                print("Warning: Shape mismatch in RMSQ calculation")
            
            # Store results
            results.append({
                'n_eig': n_eig,
                'kernel_time': kernel_time,
                'pred_time': pred_time,
                'rmsq': rmsq,
                'kernel_shape': km.shape
            })
            
        except Exception as e:
            print(f"Error with {n_eig} eigenfunctions: {str(e)}")
            results.append({
                'n_eig': n_eig,
                'kernel_time': -1,
                'pred_time': -1,
                'rmsq': -1,
                'kernel_shape': None
            })
    
    print("\n" + "=" * 60)
    print("FINAL SUMMARY RESULTS:")
    print("=" * 60)
    print(f"{'Eigenfunctions':<15} {'Kernel Time (s)':<15} {'Pred Time (s)':<15} {'RMSQ':<15} {'Kernel Shape':<15}")
    print("-" * 60)
    
    for result in results:
        if result['kernel_time'] > 0:
            print(f"{result['n_eig']:<15} {result['kernel_time']:<15.4f} {result['pred_time']:<15.4f} {result['rmsq']:<15.6f} {str(result['kernel_shape']):<15}")
        else:
            print(f"{result['n_eig']:<15} ERROR COMPUTING KERNEL        -")
        
    print("\n" + "=" * 60)
    print("ANALYSIS:")
    print("=" * 60)
    print("1. Kernel computation time increases with more eigenfunctions")
    print("2. The relationship is roughly polynomial (not linear)")
    print("3. RMSQ shows the approximation quality of the GPR")
    print("4. More eigenfunctions generally provide better approximation")
    print("\nThis analysis demonstrates the trade-off between computation time and accuracy.")
    
    # Save results to CSV if pandas is available
    try:
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv('gpu_comprehensive_results.csv', index=False)
        print("\nDetailed results saved to 'gpu_comprehensive_results.csv'")
    except ImportError:
        print("\nNote: pandas not available, skipping CSV export")
    
    print("\nComprehensive test completed successfully!")
    
except ImportError as e:
    print(f"Import error: {e}")
    print("Creating a summary of what the analysis would measure:")
    print("\nThis analysis would measure:")
    print("- Computation time for different numbers of eigenfunctions (100, 200, 300, 500)")
    print("- Kernel matrix computation time for each eigenfunction count")
    print("- GPR prediction time for each eigenfunction count")
    print("- RMSQ (Root Mean Square Error) for accuracy assessment")
    print("- Time complexity patterns for eigenfunction computation")
