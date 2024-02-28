import torch


def trilinear_interpolation_torch(sdf_grid, points):
    """
    Perform trilinear interpolation for a batch of points within the SDF grid using PyTorch.
    
    Parameters:
    - sdf_grid (torch.Tensor): The SDF grid with shape (256, 256, 256).
    - points (torch.Tensor): The (x, y, z) coordinates of the points within [-1, 1] cube space, shape (512, 3).
    
    Returns:
    - torch.Tensor: The interpolated SDF values at the given points, shape (512,).
    """
    N = sdf_grid.shape[0]  # Assuming sdf_grid is cubic and of shape (256, 256, 256)
    grid_size = N - 1  # 255

    # Map points from [-1, 1] to [0, grid_size] grid space
    points = (points + 1) / 2 * grid_size
    points = torch.clamp(points, 0, grid_size - 1e-5)

    # Get the integer and fractional parts of the points
    points_floor = torch.floor(points).long()
    points_frac = points - points_floor

    # Prepare for gathering values by expanding the grid dimensions
    sdf_grid = sdf_grid.unsqueeze(0)  # Add batch dimension, (1, 256, 256, 256)

    # Calculate indices for the 8 surrounding points
    x0, y0, z0 = points_floor[:, 0], points_floor[:, 1], points_floor[:, 2]
    x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1

    # Ensure indices are within bounds
    x0, x1 = torch.clamp(x0, 0, N-1), torch.clamp(x1, 0, N-1)
    y0, y1 = torch.clamp(y0, 0, N-1), torch.clamp(y1, 0, N-1)
    z0, z1 = torch.clamp(z0, 0, N-1), torch.clamp(z1, 0, N-1)

    # Gather the values at the 8 points
    vals_000 = sdf_grid[:, x0, y0, z0]
    vals_100 = sdf_grid[:, x1, y0, z0]
    vals_010 = sdf_grid[:, x0, y1, z0]
    vals_110 = sdf_grid[:, x1, y1, z0]
    vals_001 = sdf_grid[:, x0, y0, z1]
    vals_101 = sdf_grid[:, x1, y0, z1]
    vals_011 = sdf_grid[:, x0, y1, z1]
    vals_111 = sdf_grid[:, x1, y1, z1]

    # Perform trilinear interpolation
    xd, yd, zd = points_frac[:, 0], points_frac[:, 1], points_frac[:, 2]
    c00 = vals_000 * (1 - xd) + vals_100 * xd
    c01 = vals_001 * (1 - xd) + vals_101 * xd
    c10 = vals_010 * (1 - xd) + vals_110 * xd
    c11 = vals_011 * (1 - xd) + vals_111 * xd
    c0 = c00 * (1 - yd) + c10 * yd
    c1 = c01 * (1 - yd) + c11 * yd
    c = c0 * (1 - zd) + c1 * zd

    return c.squeeze()


if __name__ == '__main__':
    # Example usage
    N = 256
    sdf_grid = torch.rand(N, N, N)  # Example SDF grid
    points = torch.rand(512, 3) * 2 - 1  # Example points within [-1, 1] cube space
    sdf_values = trilinear_interpolation_torch(sdf_grid, points)
    print("Interpolated SDF values shape:", sdf_values.shape)
