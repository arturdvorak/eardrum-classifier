#!/usr/bin/env python3
"""
Setup script for Eardrum Classification Package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements/requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="eardrum_classifier",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A deep learning classifier for eardrum conditions using EfficientNetV2",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/arturdvorak/eardrum-classifier",
    packages=find_packages(where="src") + ["scripts"],
    package_dir={"": "src", "scripts": "scripts"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
    },
    entry_points={
        "console_scripts": [
            "eardrum-train=scripts.train:main",
            "eardrum-evaluate=scripts.evaluate:main",
            "eardrum-predict=scripts.predict:main",
        ],
    },
)
