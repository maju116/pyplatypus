# PyPlatypus

## Instalation

To install `pyplatypus` use command:

```commandline
pip install pyplatypus
```

See documentation at: [GitHub Pages](https://maju116.github.io/pyplatypus/).

## Development

To build and deploy your updated docs, use:
`mkodcs build`
Which will create site/ directory in the project folder. Then type in:
`mkdocs gh-deploy`
to automatically push the docs to GitHub Pages.

To build the package locally just move to its main directory run:
`pip install --use-feature=in-tree-build .`

Then it may be imported as any other package installed with the use of pip.

If you wish to create the source distribution (in the .egg format):
`python setup.py sdist`

To build a wheel run:
`python setup.py bdist_wheel --universal`

Important!
Before uploading the source distribution to PyPI test it using the TestPyPI using:
`twine upload --repository-url https://test.pypi.org/legacy/ source/package.tar.gz`

To upload all the distributions from a certain folder, for instance if you created
wheels aside from the archive source distribution, use this line:
`twine upload --repository-url https://test.pypi.org/legacy/ folder/*`

Then try installing from TestPyPI:
`pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple pyplatypus`

Only after testing it there should you proceed with the release:
`twine upload dist/*`

And then install your package!
`pip install pyplatypus`

Note!
Pip dependencies will be installed during the package build but there is one conda dependency that
needs to be installed by hand if you are planning to play around with the GPU.
`conda install -c conda-forge tensorflow==2.9.1`

We are working on creating the deployment script for this :)
