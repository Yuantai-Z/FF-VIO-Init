#!/usr/bin/env python3
# Copyright (C) 2025-2026 Yuantai-Z
# SPDX-License-Identifier: GPL-3.0-or-later

from distutils.core import setup

from catkin_pkg.python_setup import generate_distutils_setup


setup_args = generate_distutils_setup(
    packages=["vggt_ros"],
    package_dir={"": "src"},
)

setup(**setup_args)
