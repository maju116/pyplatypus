---
title: PyPlatypus Docs
summary: A brief description of the project.
authors:
    - Michał Maj
    - Jakub Borkowski
date: 2022-09-24
---

# Welcome to PyPlatypus

For the code and its documentation visit [github.com](https://github.com/maju116/pyplatypus).
## The Computer Vision Interface

<img src=Images/hexsticker_platypus.png alt="platypus_logo" width=250/>
<br/>


## Project layout, essentials

    mkdocs.yml      # The docs configuration file.
    environment.yml     # Conda/Mamba- compliant YAML, defining the environment.
    README.md       # Essentials with a touch of development guidelines.
    docs/       # Here the documentation is defined.
        Images/      # Graphics.
        index.md  # The documentation homepage.
        ...       # Other markdown pages, images and other files.
    examples/   # Here the example notebooks are stored.
        data/       # Placeholder
        models/     # Placeholder
        data_science_bowl_config.yaml   # Config utilized by the Platypus Engine.
        data_science_bowl_notebook.ipynb    # Notebook associated with the DS Bowl.
    platypus/   # Main directory, here the magic scrolls are hidden.
        config/ # Here the variety of constants is stored.
            ...
        data_models/    # Folder with the Pydantic data models.
            ...
        detection/  # Object detection-related files, placeholder
            ...
        segmentation/ # Semantic segmentation toolbox.
            models/ # Semantic segmentation models
                u_shaped_models.py  # Encoder-decoder based architectures.
            generator.py    # Data generator template.
            loss_functions.py   $ Stores all implemented loss functions.
        solvers/    # Solvers that are to be invoking various engine.
            platypus_cv_solver.py   # Main CV-related solver.
        utils/  # Various CV and non-CV utils.
            ...
        engine.py   # The Platypus Engine, invoked by the solvers.
