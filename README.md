# pyplatypus

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
`pip install --index-url https://test.pypi.org/simple/ platypus`

Only after testing it there should you proceed with the release:
`twine upload dist/*`

And then install your package!
`pip install platypus`