from setuptools import setup

setup(
    name='VAE-WL',
    version='1.0',
    packages=['vaewl'],
    url='https://ricardodcpereira.com',
    license='MIT',
    author='Ricardo Pereira',
    author_email='rdpereira@dei.uc.pt',
    description='VAE-WL - Variational Autoencoder with Weighted Loss',
    python_requires='>=3.6.*',
    install_requires=['numpy>=1.20.0', 'scikit-learn>=0.24.2'],
    extras_require={
        'tf': ['tensorflow>=2.5.0'],
        'tf_gpu': ['tensorflow-gpu>=2.5.0'],
    }
)
