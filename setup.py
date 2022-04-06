from setuptools import setup, find_packages

setup(
    name='mutate',
    version='0.1.0',
    description='Text data synthesize and pseudo labelling using LLMs',
    author='Logesh Kumar Umapathi',
    author_email='logeshkumaru@gmail.com',
    url='https://github.com/infinitylogesh/mutate',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'tqdm',
        'numpy',
        'torch>=1.0',
        'jinja2==3.0.3',
        'transformers<=4.12.3',
        'datasets==1.6.2'
    ]
)
