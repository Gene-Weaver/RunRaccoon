from pathlib import Path

from setuptools import find_namespace_packages, setup

'''
Release to PyPI (see "Releasing" in README.md):
    1. bump __version__ in runraccoon/_version.py and add a CHANGELOG.md entry
    2. python -m build
    3. python -m twine upload dist/* --skip-existing --verbose
'''

HERE = Path(__file__).parent
version = {}
exec((HERE / "runraccoon" / "_version.py").read_text(encoding="utf-8"), version)   # single source of truth

with open(HERE / "README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="runraccoon",
    version=version["__version__"],
    author="Will",
    author_email="willwe@umich.edu",
    description="Local-only, drop-in replacement for Weights & Biases: same API and file layout, "
                "per-epoch and summary plots, and a localhost dashboard. No cloud.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Gene-Weaver/RunRaccoon",
    project_urls={
        "Bug Tracker": "https://github.com/Gene-Weaver/RunRaccoon/issues",
        "Changelog": "https://github.com/Gene-Weaver/RunRaccoon/blob/main/CHANGELOG.md",
    },
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    keywords=["experiment tracking", "wandb", "machine learning", "training plots", "offline", "ultralytics"],
    # namespace discovery so the dashboard's static/ folder (html/js/css/svg) ships as package data
    packages=find_namespace_packages(include=["runraccoon", "runraccoon.*"]),
    include_package_data=True,
    package_data={"runraccoon.dashboard.static": ["*"]},
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21",
        "matplotlib>=3.5",
        "pillow>=8.0",
        "pyyaml>=5.4",
    ],
    extras_require={
        "dev": ["pytest>=7", "build", "twine"],
    },
    entry_points={
        "console_scripts": [
            "runraccoon=runraccoon.cli:main",
        ],
    },
)
