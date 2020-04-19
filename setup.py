import setuptools

with open("README.md") as fh:
    LONG_DESCRIPTION = fh.read()

setuptools.setup(
    name="torch-summary",
    version="1.1.4",
    author="Tyler Yep @tyleryep",
    author_email="tyep10@gmail.com",
    description="Model summary in PyTorch, based off of the original torchsummary",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/tyleryep/torch-summary",
    packages=["torchsummary"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
