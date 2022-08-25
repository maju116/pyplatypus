from setuptools import setup

setup(
    name="pyplatypus",
    version="0.1.0",
    author="Michał Maj",
    author_email="michalmaj116@gmail.com",
    description="""
    Set of tools for Computer Vision handling the object detection and image
    segmentation as end-to-end process, from loading a data via training the
    multiple models of choice to plotting the models' predictions.
    """,
    long_description=open('README.md').read(),
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
    install_requires=[],
    python_requires=">=3.6",
)
