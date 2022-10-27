from distutils.core import setup

setup(name='mpcbenchrunner',
    version='1.0',
    author_email='amin.faez.inbox@gmail.com',
    packages=['mpcbenchrunner'],    
    install_requires=[
        'commonsnakes',
        'pyyaml',
        'attrs',
        'cattrs',
        'docker'
    ],
)
