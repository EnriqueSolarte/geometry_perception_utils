import geometry_perception_utils.config_utils
import os

GEOM_UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
GEOM_UTILS_CFG_DIR = os.path.join(GEOM_UTILS_DIR, 'config')

os.environ['GEOM_UTILS_DIR'] = GEOM_UTILS_DIR
os.environ['GEOM_UTILS_CFG_DIR'] = GEOM_UTILS_CFG_DIR

