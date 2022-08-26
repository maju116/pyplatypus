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
    url="https://github.com/maju116/pyplatypus",
    project_urls={
        "Bug Tracker": "https://github.com/maju116/pyplatypus/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "tensorflow==2.9.1",
        "albumentations==1.1.0",
        "Keras-Preprocessing==1.1.2",
        "scikit-image==0.19.1",
        "scikit-learn==1.0.2",
        "scipy==1.8.1",
        "tensorflow-estimator==2.9.0",
        "pandas==1.3.1",
        "numpy==1.22.4",
        "tensorboard==2.9.0",
        "PyYAML==6.0",
        "pydicom==2.2.0",
        "pydantic==1.9.1"
        ],
    python_requires=">=3.6",
)
