
import os
from pathlib import Path
import subprocess
import sys
from geometry_perception_utils.create_gif import create_gif

assert len(sys.argv) == 3, f"Expected 2 arguments, got {len(sys.argv) - 1}."

input_dir = Path(sys.argv[1]).resolve()
assert os.path.exists(input_dir), f"Input directory {input_dir} does not exist."

output_fn = f"{Path(sys.argv[2]).resolve()}/{input_dir.__str__().split('/')[-1]}.gif"
create_gif(input_dir, output_fn)