
import os
from pathlib import Path
import subprocess
import sys

run_file = f"{Path(__file__).parent.absolute()}/bash.sh"

assert len(sys.argv) == 3, f"Expected 2 arguments, got {len(sys.argv) - 1}."

input_dir = Path(sys.argv[1]).resolve()
assert os.path.exists(input_dir), f"Input directory {input_dir} does not exist."

output_dir = f"{Path(sys.argv[2]).resolve()}/{input_dir.__str__().split('/')[-1]}.gif"
subprocess.run(["bash", f"{run_file}", f"{input_dir.__str__()}", f"{output_dir.__str__()}"])