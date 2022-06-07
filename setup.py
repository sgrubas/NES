from setuptools import setup, find_packages

NAME            = "NES"
VERSION         = "0.2.0"
DESCRIPTION     = "Neural Eikonal Solver: Framework for solving the eikonal equation using neural networks"
URL             = "https://github.com/sgrubas/NES"
LICENSE         = "MIT"
AUTHOR          = "Serafim Grubas, Nikolay Shilov, Anton Duchkov, Georgy Loginov"
EMAIL           = "serafimgrubas@gmail.com"
KEYWORDS        = ["Eikonal", "Seismic", "Traveltime"]
CLASSIFIERS     = [
                    "Development Status :: Beta",
                    "Intended Audience :: Geophysicist",
                    "Natural Language :: English",
                    f"License :: {LICENSE}",
                    "Operating System :: OS Independent",
                    "Programming Language :: Python :: 3.7",
                    "Topic :: Scientific/Engineering",
                    ]
INSTALL_REQUIRES = [
                    'numpy',
                    'scipy',
                    'scikit-learn',
                    'tensorflow']
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
    packages=find_packages(),
    # package_dir={"": NAME},
    url=URL,
    zip_safe=False,
    install_requires=INSTALL_REQUIRES,
    # include_package_data=True,
    package_data={NAME: ["data/*.npy"]}
    )

