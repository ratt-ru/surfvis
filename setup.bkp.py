from setuptools import setup, find_packages
import surfvis

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = [
                'matplotlib',
                'argparse',
                'ipython',
                'codex-africanus[complete] >= 0.4.1',
                'dask-ms[xarray, zarr, s3] >= 0.4.2',
                'pytest >= 8.0.0',
                'datashader',
                'holoviews',
                'colorcet',
                'bokeh >= 3.1.0',
                'Click >= 8.1'
            ]


setup(
     name='surfvis',
     version=surfvis.__version__,
     author="Ian Heywood",
     author_email="ianh@astro.ox.ac.uk",
     description="Collection of visibility data inspection tools",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/ratt-ru/surfvis",
     packages=find_packages(),
     python_requires='>=3.10',
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
