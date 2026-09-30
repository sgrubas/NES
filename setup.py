import re
from setuptools import setup, find_packages

NAME = "NES"
with open("NES/__init__.py", "r") as f:  # no import of NES: it needs keras and a backend
    VERSION = re.search(r"__version__\s*=\s*['\"]([^'\"]+)['\"]", f.read()).group(1)
DESCRIPTION = "Neural Eikonal Solver: Framework for solving the eikonal equation using neural networks"
URL = "https://github.com/sgrubas/NES"
LICENSE = "MIT"
AUTHOR = "Serafim Grubas, Anton Duchkov, Georgy Loginov, Nikolay Shilov"
EMAIL = "serafimgrubas@gmail.com"
KEYWORDS = ["Eikonal", "Seismic", "Traveltime"]
CLASSIFIERS = [
               "Development Status :: 4 - Beta",
               "Intended Audience :: Science/Research",
               "Natural Language :: English",
               "License :: OSI Approved :: MIT License",
               "Operating System :: OS Independent",
               "Programming Language :: Python :: 3",
               "Topic :: Scientific/Engineering",
               ]

with open("requirements.txt", mode='r') as f:
    INSTALL_REQUIRES = [line.strip() for line in f if line.strip() and not line.startswith('#')]

with open("README.md", "r") as f:
    LONG_DESCRIPTION = f.read()

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    maintainer=AUTHOR,
    maintainer_email=EMAIL,
    classifiers=CLASSIFIERS,
    keywords=KEYWORDS,
    packages=find_packages(exclude=["tests", "tests.*"]),
    url=URL,
    zip_safe=False,
    python_requires=">=3.10",
    install_requires=INSTALL_REQUIRES,
    # one Keras 3 backend is needed; Google Colab has all three preinstalled
    extras_require={"jax": ["jax"], "tensorflow": ["tensorflow>=2.16"], "torch": ["torch"],
                    "hpo": ["optuna>=5", "matplotlib"], "test": ["pytest"]},
    package_data={NAME: ["data/*.npy"]},
    )
