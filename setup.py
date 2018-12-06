import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


install_requires = [
    'six',
    'numpy',
    'Pillow',
    'matplotlib',
    'jinja2'
]

try:
    import tensorflow
except ImportError:
    install_requires.append('tensorflow==1.11.0')

setuptools.setup(
    name="tfold",
    version="0.2.6",
    author="Marius Ionescu",
    author_email="marius@mi.www.ro",
    description="Tensorflow Object Detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/telize-ai/tfold",
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    package_data={'tfold': [
        'data/*.pb',
        'data/*.ckpt',
        'data/*.pbtxt',
        'data/faster_checkpoint/*.pb',
        'data/faster_checkpoint/*.ckpt.*',
        'data/ssd_checkpoint/*.pb',
        'data/ssd_checkpoint/*.ckpt',
        'config/templates/*.config',
    ]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache License",
        "Operating System :: Linux",
    ],
)
