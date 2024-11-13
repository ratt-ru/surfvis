# flake8: noqa
import click
from pfb import logo

@click.group()
def cli():
    logo()
    pass


from surfvis.workers import (surfchi2, flagchi2, qaplots)

if __name__ == '__main__':
    cli()
