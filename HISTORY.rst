=======
History
=======

1.2.0 (WIP)
------------------
* Added type hints to most core methods
* Added ``send_lidar_config()`` method to configure LIDAR
* Added car roll, pitch yaw angle
* Renamed lidar config to use snake case instead of CamelCase (for instance ``degPerSweepInc`` was renamed to ``deg_per_sweep_inc``)

1.1.1 (2021-02-28)
------------------
* Fix type checking error

1.1.0 (2021-02-28)
------------------
* black + isort for autoformatting
* Many flake8 fixes (removed unused imports, ...)
* The simulator can be launched separately
* Made the client Thread daemon (so we can use ctrl+c to kill it)

1.0.0 (2019-07-26)
------------------

* First release on PyPI.

1.0.1 - 1.0.11 (2019-08-04)
-----------------------------

* Testing out deploy system
* Update credits/authors
* flake8
