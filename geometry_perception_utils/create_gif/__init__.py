import os
from pathlib import Path
import subprocess
import sys


def create_gif(input_dir, output_fn):
    """
    Create a gif from a list of images.
    """
    run_file = f"{Path(__file__).parent.absolute()}/bash.sh"
    subprocess.run(["bash", f"{run_file}", f"{input_dir.__str__()}", f"{output_fn.__str__()}"])
