from geometry_perception_utils.vispy_utils import plot_list_pcl
import numpy as np

def main():
    point_clouds = []
    for i in range(3):
        # Generate random point clouds
        point_clouds += [np.random.rand(3, 100) for _ in range(3)]
        
    # Plot the point clouds
    plot_list_pcl(point_clouds, size=2)

if __name__ == "__main__":
    main()