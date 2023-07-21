import numpy as np
    
def clip_boundary(boundary, threshold):
    dist = np.linalg.norm(boundary, axis=0)
    mask = dist > threshold
    if np.sum(mask) > 0:
        boundary[:, mask] = threshold * boundary[:, mask] / dist[mask] 
    return boundary
