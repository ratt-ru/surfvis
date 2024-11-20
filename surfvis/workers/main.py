# flake8: noqa
import click

@click.group()
def cli():
    pass


from surfvis.workers import (surfchi2, flagchi2, phaseball, cmratio)

if __name__ == '__main__':
    cli()
