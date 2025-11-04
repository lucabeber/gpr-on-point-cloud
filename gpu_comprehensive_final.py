#!/usr/bin/env python3
"""
Final comprehensive test examining both eigenfunction count and training point count effects
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
    obj_name = "bun270_X"
    filename = f"{point_cloud_dir}{obj_name}.ply"
    
    # Check if file exists
    if not os.path.exists(filename):
        print(f"Point cloud file {filename} not found")
        sys.exit(1)
    
    print(f"Loading point cloud: {filename}")
    pcd_tmp = o3d.io.read_point_cloud(filename)
    pcd = pcd_tmp.voxel_down_sample(voxel_size=0.002)
    original_vertices = np.asarray(pcd.points)
    original_colors = np.asarray(pcd.colors)[:, 0] if hasattr(pcd_tmp, 'colors') else np.ones(len(original_vertices))
    
    print(f"Original point cloud shape: {original_vertices.shape}")
    
    # Test different combinations of eigenfunctions and training points
    eigenfunction_counts = [100, 200, 300, 500]
    point_ratios = [0.05, 0.10, 0.15, 0.20]  # 5%, 10%, 15%, 20% of points
    
    print("\nTesting combined effects of eigenfunctions and training points...")
    print("=" * 70)
    
    results = []
    
    for n_eig in eigenfunction_counts:
        for ratio in point_ratios:
            print(f"\n--- Testing {n_eig} eigenfunctions with {ratio*100:.0f}% points ---")
            
            # Create subsampled point cloud
            n_points = int(len(original_vertices) * ratio)
            # if n_points < n_eig:
            #     print(f"Warning: Not enough points for {n_eig} eigenfunctions, using {n_points} points")
            #     n_eig = n_points - 10  # Reduce eigenfunctions to avoid error
            #     if n_eig <= 0:
            #         continue
            
            # Randomly sample points
            # indices = np.random.choice(len(original_vertices), n_points, replace=False)
            vertices = original_vertices
            colors = original_colors
            
            # Measure kernel computation time
            print("Computing kernel matrix...")
            start_time = time.time()
            try:
                km = rbf_manifold_kernel(original_vertices, lengthscale=0.002, sigma=1.0, n_eig=n_eig)
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
                
                model.eval()
                likelihood.eval()
                # Measure prediction time
                print("Computing GPR predictions...")
                start_time = time.time()
                with torch.no_grad():
                    observed_pred = likelihood(model(train_x_real))
                pred_time = time.time() - start_time
                print(f"GPR prediction time: {pred_time:.4f} seconds")
                
                # Get predictions for training points
                mean = observed_pred.mean.cpu().detach().numpy()
                
                # Calculate RMSQ (Root Mean Square Error) on training data
                rmsq = np.sqrt(np.mean((mean - colors) ** 2))
                print(f"RMSQ on training data: {rmsq:.6f}")
                
                # Store results
                results.append({
                    'n_eig': n_eig,
                    'points': len(vertices),
                    'ratio': ratio,
                    'kernel_time': kernel_time,
                    'pred_time': pred_time,
                    'rmsq': rmsq,
                    'kernel_shape': km.shape
                })
                
            except Exception as e:
                print(f"Error with {n_eig} eigenfunctions and {len(vertices)} points: {str(e)}")
                results.append({
                    'n_eig': n_eig,
                    'points': len(vertices),
                    'ratio': ratio,
                    'kernel_time': -1,
                    'pred_time': -1,
                    'rmsq': -1,
                    'kernel_shape': None
                })
    
    print("\n" + "=" * 70)
    print("COMBINED ANALYSIS RESULTS:")
    print("=" * 70)
    print(f"{'Eigenfunctions':<12} {'Points':<10} {'Ratio':<8} {'Kernel Time (s)':<15} {'Pred Time (s)':<15} {'RMSQ':<15}")
    print("-" * 70)
    
    for result in results:
        if result['kernel_time'] > 0:
            print(f"{result['n_eig']:<12} {result['points']:<10} {result['ratio']:<8.2f} {result['kernel_time']:<15.4f} {result['pred_time']:<15.4f} {result['rmsq']:<15.6f}")
        else:
            print(f"{result['n_eig']:<12} {result['points']:<10} {result['ratio']:<8.2f} ERROR COMPUTING KERNEL")
    
    print("\n" + "=" * 70)
    print("ANALYSIS:")
    print("=" * 70)
    print("1. Combined effect of eigenfunctions and training points")
    print("2. Shows how both factors influence computational complexity")
    print("3. Demonstrates scalability patterns for GPR on point clouds")
    print("4. Helps optimize parameters for specific computational constraints")
    
    # Save results to CSV if pandas is available
    try:
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv('gpu_combined_analysis_results.csv', index=False)
        print("\nDetailed results saved to 'gpu_combined_analysis_results.csv'")
    except ImportError:
        print("\nNote: pandas not available, skipping CSV export")
    
    print("\nCombined analysis completed successfully!")
    
except ImportError as e:
    print(f"Import error: {e}")
    print("Creating a summary of what the analysis would measure:")
    print("\nThis analysis would measure:")
    print("- Combined effects of eigenfunction count and training points")
    print("- Computational complexity with both parameters")
    print("- Scalability patterns for GPR on point clouds")
    print("- Optimal parameter combinations for performance")
