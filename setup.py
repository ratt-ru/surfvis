from setuptools import setup, find_packages
import surfvis

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = [
                'matplotlib',
                'python-casacore',
                'argparse',
                'ipython',
                'codex-africanus[dask]',
                'astropy',
                'scipy',
                'dask-ms[xarray, zarr]',
                'pytest >= 6.2.2',
            ]


setup(
     name='surfvis',
     version=surfvis.__version__,
     author="Ian Heywood",
     author_email="ianh@astro.ox.ac.uk",
     description="Per-baseline time/freq and Chi-Square plots",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/ratt-ru/surfvis",
     packages=find_packages(),
     python_requires='>=3.7',
     install_requires=requirements,
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
     entry_points='''
                    [console_scripts]
                    surfvis=surfvis.surfvis:main
                    surfchi2=surfvis.surfchi2:main
                    flagchi2=surfvis.flagchi2:main
     '''
     ,
 )
