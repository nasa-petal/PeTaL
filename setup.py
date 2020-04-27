import setuptools

'''
Default installation of PeTaL pipeline
'''

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PeTaL-pipeline",
    version="0.0.1",
    author="Lucas Saldyt",
    author_email="lucassaldyt@gmail.com",
    description="PeTaL's data pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LSaldyt/PeTaL-pipeline",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
