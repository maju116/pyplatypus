import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="platypus",
    version="0.0.1",
    author="Michał Maj",
    author_email="michalmaj116@gmail.com",
    description="Set of tools for Computer Vision like YOLOv3 for object detection and U-Net for image segmentation.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/maju116/platypus",
    project_urls={
        "Bug Tracker": "https://github.com/maju116/platypus/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.9",
)