from setuptools import find_packages, setup

with open("./requirements.txt", "r") as f:
    requirements = [l.strip() for l in f.readlines() if len(l.strip()) > 0]


setup(
    name="geometry-perception-utils-dev",
    version=f"1.0",
    packages=find_packages(),
    install_requires=requirements,
    package_data={
                  "geometry_perception_utils": ["data/**", "config/**"]
                  },
    author="Enrique Solarte",
    author_email="enrique.solarte.pardo@gmail.com",
    description=("Utils used in geometry perception tasks"),
    license="BSD",
)
