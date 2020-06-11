import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="berries-ugurgudelek",
    version="0.0.1", # read PEP 440
    author="Ugur Gudelek",
    author_email="ugurgudelek@gmail.com",
    description="Quick experiment library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ugurgudelek/berries",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['A==1.0', 'B>=1,<2'],
)