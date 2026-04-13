import ast
import itertools
import os
import pkgutil
from dataclasses import dataclass
from typing import List, Optional

from setuptools import find_packages, setup

package_name = "giskardpy_ros"

data_files = [
    ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
    ("share/" + package_name, ["package.xml"]),
]
for dirpath, _, filenames in itertools.chain(os.walk("data"), os.walk("launch")):
    full_paths = [os.path.join(dirpath, f) for f in filenames]
    install_path = os.path.join("share", package_name, dirpath)
    data_files.append((install_path, full_paths))

script_folder = "giskardpy.middleware.ros2.scripts"
setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=data_files,
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Simon Stelter",
    maintainer_email="stelter@uni-bremen.de",
    description="TODO: Package description",
    license="LGPLv3",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            f"pr2_standalone = {script_folder}.iai_robots.pr2.pr2_standalone:main",
            f"hsr_standalone = {script_folder}.iai_robots.hsr.hsr_standalone:main",
            f"hsr_velocity = {script_folder}.iai_robots.hsr.hsr_velocity:main",
            f"hsr_trust_me_bro = {script_folder}.iai_robots.hsr.iai_hsr_real_time_trust_me_bro:main",
            f"tracy_standalone = {script_folder}.iai_robots.tracy.tracy_standalone:main",
            f"tracy_velocity = {script_folder}.iai_robots.tracy.tracy_velocity:main",
            f"stretch_velocity = {script_folder}.iai_robots.stretch.stretch_velocity:main",
            f"r6bot = {script_folder}.other_robots.test.r6bot:main",
            f"interactive_marker = {script_folder}.tools.interactive_marker:main",
            f"motion_statechart_inspector = {script_folder}.tools.motion_statechart_inspector:main",
            f"joystick_e_stop = {script_folder}.tools.joystick_e_stop:main",
        ],
    },
)
