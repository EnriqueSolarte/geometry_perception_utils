import os
import git
from pathlib import Path

# ! Registering current git commit


def get_repo_commit(REPO_DIR):
    current_dir = os.getcwd()
    if os.path.isdir(REPO_DIR):
        os.chdir(REPO_DIR)
    else:
        os.chdir(os.path.dirname(REPO_DIR))

    repo = git.Repo(search_parent_directories=True)
    commit = repo.head._get_commit()
    repo_name = Path(repo.working_dir).stem
    data = f"{repo_name}: {commit.name_rev}"
    os.chdir(current_dir)
    return data


if __name__ == "__main__":
    print(get_repo_commit(os.path.dirname(os.path.abspath(__file__))))

