import setuptools

setuptools.setup(
    name='FLutils',
    version='0.0.1',
    author='Davide Dominii',
    description='FL utils',
    long_description='FL utils - LMCS 2025 Experiments - Field-Based Coordination for Federated Learning',
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],  # Information to filter the project on PyPi website
    python_requires='>=3.6',  # Minimum version requirement of the package
    py_modules=["FLutils"],  # Name of the python package
    package_dir={'': 'src'},  # Directory of the source code of the package
    install_requires=["torch","torchvision"]  # Install other dependencies if any
)