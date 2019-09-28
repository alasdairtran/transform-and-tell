from setuptools import setup

setup(name='newser',
      version='0.1',
      description='News captioning experiments',
      url='https://github.com/alasdairtran/newser',
      author='Alasdair Tran',
      author_email='alasdair.tran@anu.edu.au',
      license='MIT',
      packages=['newser'],
      install_requires=['allennlp'],
      scripts=['bin/newser'],
      zip_safe=False)
