from setuptools import setup

setup(name="donkey_gym",
      version="0.1",
      url="https://github.com/tawnkramer/sdsandbox/src/donkey_gym",
      author="Tawn Kramer",
      license="MIT",
      packages=["donkey_gym"],
      install_requires = ["gym", "numpy", 'pillow']
      )
