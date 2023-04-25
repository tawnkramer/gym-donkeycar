=======
History
=======

1.3.1 (WIP)
------------------
- Use forward velocity in the reward function
- Use ruff instead of flake8 and move most configs to ``pyproject.toml``

1.3.0 (2022-05-30)
------------------
* Dropped Python 3.6 support, pinned Gym to version 0.21
* Move steer limits and throttle limits to config dict
* Normalized reward and use squared error for CTE
* Enabled hand brake in ``send_control()`` and at reset time
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
