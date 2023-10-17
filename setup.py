from setuptools import setup, find_packages

setup(
    name='PokemonRedExperiments',
    version='0.0.1',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'my_start=PokemonRedExperiments',
        ]
    }
)