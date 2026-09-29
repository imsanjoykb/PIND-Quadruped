"""
PIND: Physics-Informed Neural Dynamics for Quadruped Robots
"""

from setuptools import setup, find_packages
import os

# Read version
__version__ = "0.1.0"

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="pind-quadruped",
    version=__version__,
    author="Your Name",
    author_email="your.email@example.com",
    description="Physics-Informed Neural Dynamics for Learning Quadruped Robot Locomotion",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/PIND-Quadruped",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/PIND-Quadruped/issues",
        "Documentation": "https://pind-quadruped.readthedocs.io/",
        "Source Code": "https://github.com/yourusername/PIND-Quadruped",
    },
    packages=find_packages(exclude=["tests", "tests.*", "experiments", "experiments.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
            "mypy>=1.3.0",
        ],
        "docs": [
            "sphinx>=6.2.0",
            "sphinx-rtd-theme>=1.2.0",
        ],
        "viz": [
            "plotly>=5.14.0",
            "seaborn>=0.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pind-train=scripts.train:main",
            "pind-eval=scripts.evaluate:main",
            "pind-generate=pind.data.generator:main",
        ],
    },
    include_package_data=True,
    package_data={
        "pind": ["configs/*.yaml"],
    },
    zip_safe=False,
)
