import logging
import os
import git
import yaml
from omegaconf import OmegaConf
from coolname import generate_slug
import datetime


def set_stamp_name(number_names, *, _parent_):
    _parent_['stamp_name'] = generate_slug(int(number_names))
    return _parent_['stamp_name']


def get_date(format=0, *, _parent_):
    if format == 0:
        return datetime.datetime.now().strftime("%b %d '%y %H:%M:%S")
    elif format == 1:
        return datetime.datetime.now().strftime("%b-%Y")
    elif format == 2:
        return datetime.datetime.now().strftime("%d-%b-%Y")
    elif format == 3:
        return datetime.datetime.now().strftime("%y%m%d.%H%M%S")
    else:
        return datetime.datetime.now().strftime("%Y-%m-%d")


def get_hostname(*, _parent_):
    return os.uname().nodename


def get_git_commit(*, _parent_):
    repo = git.Repo(search_parent_directories=True)
    git_commit = repo.head._get_commit().name_rev
    return git_commit


def get_timestamp(format, *, _parent_):
    _parent_['stamp'] = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return _parent_['stamp']


def load(input_file, *, _parent_):
    assert os.path.exists(input_file), f"File does not exist {input_file}"
    cfg = read_omega_cfg(input_file)
    return cfg

OmegaConf.register_new_resolver('set_stamp_name', set_stamp_name)
OmegaConf.register_new_resolver('get_hostname', get_hostname)
OmegaConf.register_new_resolver('get_git_commit', get_git_commit)
OmegaConf.register_new_resolver('get_timestamp', get_timestamp)
OmegaConf.register_new_resolver('get_date', get_date)
OmegaConf.register_new_resolver('load', load)


def get_empty_cfg():
    logging.basicConfig(
        format='[%(levelname)s] [%(asctime)s]:  %(message)s',
        level=logging.INFO
    )
    cfg_dict = dict()

    # ! add git commit
    # cfg_dict = add_git_commit(cfg_dict)

    cfg = OmegaConf.create(cfg_dict)
    return cfg


def save_cfg(cfg_file, cfg):
    # OmegaConf.resolve(cfg)
    # cfg_ = get_resolved_cfg(cfg)
    with open(cfg_file, 'w') as fn:
        OmegaConf.save(config=cfg, f=fn)


def read_omega_cfg(cfg_file):
    assert os.path.exists(cfg_file), f"File does not exist {cfg_file}"

    # ! Reading YAML file
    with open(cfg_file, "r") as f:
        cfg_dict = yaml.safe_load(f)

    cfg = OmegaConf.create(cfg_dict)
    return cfg


def read_cfg(cfg_file):
    cfg = read_omega_cfg(cfg_file)
    OmegaConf.resolve(cfg)
    return cfg
