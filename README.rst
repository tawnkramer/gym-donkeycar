======================================================
OpenAI Gym Environments for Donkey Car
======================================================


.. image:: https://img.shields.io/pypi/v/gym-donkeycar.svg
        :target: https://pypi.python.org/pypi/gym-donkeycar

.. image:: https://img.shields.io/travis/leigh-johnson/gym-donkeycar.svg
        :target: https://travis-ci.org/leigh-johnson/gym-donkeycar

.. image:: https://readthedocs.org/projects/gym-donkeycar/badge/?version=latest
        :target: https://gym-donkeycar.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

Donkey Car OpenAI Gym


* Free software: MIT license
* Documentation: https://gym-donkeycar.readthedocs.io/en/latest/

Installation
--------------

* Install with pip

.. code-block:: shell

    pip install gym-donkeycar

* Download simulator binaries: https://github.com/tawnkramer/donkey_gym/releases


Example Usage
--------------

.. code-block:: python

    import gym

    env = gym.make("donkey-generated-track-v0)


Environments
---------------

* "donkey-warehouse-v0"
* "donkey-generated-roads-v0"
* "donkey-avc-sparkfun-v0"
* "donkey-generated-track-v0"


Credits
------------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
