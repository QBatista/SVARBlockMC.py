# Use setuptools in preference to distutils
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
import os

DESCRIPTION = "A package for doing exact Bayesian inference on structural VAR"

setup(name='svar_block_mc',
      packages=['svar_block_mc'],
      version=0.1,
      description=DESCRIPTION,
      author='Quentin Batista',
      author_email='batista.quent@gmail.com',
      url='https://github.com/QBatista/SVARBlockMC.py',  # URL to the repo
      keywords=['quantitative', 'economics', 'structural', 'VAR', 'block MC'],
      install_requires=[
          'jax',
          'jaxlib',
          ]
      )

