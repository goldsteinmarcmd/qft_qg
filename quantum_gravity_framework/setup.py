from setuptools import setup, find_packages

setup(
    name="quantum_gravity_framework",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "networkx",
        "numba"
    ],
    author="Quantum Gravity Research Team",
    author_email="example@example.com",
    description="A comprehensive framework for quantum gravity research with dimensional flow",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/quantum-gravity-framework",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
) 