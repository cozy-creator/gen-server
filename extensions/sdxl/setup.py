from setuptools import setup

setup(
    name='Flask-SuperFeature',
    version='0.1',
    packages=['flask_superfeature'],
    entry_points={
        'flask.extensions': [
            'superfeature = flask_superfeature:initialize'
        ],
    }
)