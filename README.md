# `eagerx_demo` package

[![license](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Continuous Integration](https://github.com/eager-dev/eagerx_demo/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/eager-dev/eagerx_demo/actions/workflows/ci.yml)


What is the `eagerx_demo` package
---------------------------------

This repository contains the code for an interactive imitation learning demo using EAGERx.
Specifically, it allows to train a [CLIPort](https://cliport.github.io/) model interactively using [PARTNR](https://partnr-learn.github.io/).

[The core repository is available here.](https://github.com/eager-dev/eagerx)
[Full documentation and tutorials (including package creation and contributing) are available here.](https://eagerx.readthedocs.io/en/master/)

Installation
------------

You can run the main code as follows:

*Prerequisites:* Install [Poetry](https://python-poetry.org/docs/master/#installation) and [ROS](http://wiki.ros.org/noetic/Installation).
Requirements for usage with real Franka robot:
- [ros noetic](http://wiki.ros.org/noetic/Installation/Ubuntu)
- [franka_human_friendly_controllers](https://github.com/franzesegiovanni/franka_human_friendly_controllers) see [installation instructions](https://github.com/tud-phi/franka-emika-guide)

```bash
    git clone git@github.com:eager-dev/eagerx_demo.git
    cd eagerx_demo
    poetry install
    poetry run python scripts/main.py
```

Demo Overview
-------------

<p align="center">
  <img src="graph.png" width="100%" alt="graph"/>
  <img src="use_case.gif" width="100%"  alt="use_case"/> 
</p>


Credits
-------

Code within `eagerx_demo.cliport` modified from the [original repo](https://github.com/cliport/cliport).

Cite EAGERx
-----------

If you are using EAGERx for your scientific publications, please cite:

``` {.sourceCode .bibtex}
@article{eagerx,
    author  = {van der Heijden, Bas and Luijkx, Jelle, and Ferranti, Laura and Kober, Jens and Babuska, Robert},
    title = {EAGERx: Engine Agnostic Graph Environments for Robotics},
    year = {2022},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/eager-dev/eagerx}}
}
```

Acknowledgements
----------------

EAGERx is funded by the [OpenDR](https://opendr.eu/) Horizon 2020
project.