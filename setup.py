from setuptools import setup

setup(name='tell',
      version='0.1',
      description='News captioning experiments',
      url='https://github.com/alasdairtran/tell',
      author='Alasdair Tran',
      author_email='alasdair.tran@anu.edu.au',
      license='MIT',
      packages=['tell'],
      install_requires=[],
      scripts=['bin/tell'],
      zip_safe=False)
