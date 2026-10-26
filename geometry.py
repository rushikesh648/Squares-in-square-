"""Geometric routines for rotated squares and overlap computations."""
import numpy as np
from shapely.geometry import Polygon
from shapely import unary_union

def get_square_vertices(x: float, y: float, theta: float, side: float = 1.0) -> np.ndarray:
    """
    Returns (4, 2) corner coordinates of a square centered at (x, y) 
    rotated counter-clockwise by theta (radians).
    """
    half = side / 2.0
    local_corners = np.array([
        [-half, -half],
        [ half, -half],
        [ half,  half],
        [-half,  half]
    ])
    
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rot_matrix = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    
    rotated = local_corners @ rot_matrix.T
    return rotated + np.array([x, y])

def create_square_polygon(x: float, y: float, theta: float, side: float = 1.0) -> Polygon:
    """Builds a Shapely Polygon representation for intersection checks."""
    verts = get_square_vertices(x, y, theta, side)
    return Polygon(verts)

def compute_total_overlap(polygons: list[Polygon]) -> float:
    """Calculates pair-wise overlap area among all square polygons."""
    n = len(polygons)
    overlap_area = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if polygons[i].intersects(polygons[j]):
                overlap_area += polygons[i].intersection(polygons[j]).area
    return overlap_area

def compute_bounding_square_size(polygons: list[Polygon]) -> float:
    """
    Computes side length L of an axis-aligned bounding square enclosing all polygons.
    """
    all_bounds = [poly.bounds for poly in polygons] # (minx, miny, maxx, maxy)
    min_x = min(b[0] for b in all_bounds)
    min_y = min(b[1] for b in all_bounds)
    max_x = max(b[2] for b in all_bounds)
    max_y = max(b[3] for b in all_bounds)
    
    width = max_x - min_x
    height = max_y - min_y
    return max(width, height)
