import itertools
import os

from setuptools import find_packages, setup

package_name = 'giskardpy_ros'

data_files = [
    ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
]
for dirpath, _, filenames in itertools.chain(os.walk('data'), os.walk('launch')):
    full_paths = [os.path.join(dirpath, f) for f in filenames]
    install_path = os.path.join('share', package_name, dirpath)
    data_files.append((install_path, full_paths))

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Simon Stelter',
    maintainer_email='stelter@uni-bremen.de',
    description='TODO: Package description',
    license='LGPLv3',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'generic_giskard = scripts.generic_giskard:main',
            'pr2_standalone = scripts.iai_robots.pr2.pr2_standalone:main',
            'hsr_standalone = scripts.iai_robots.hsr.hsr_standalone:main',
            'generic_giskard_standalone = scripts.generic_giskard_standalone:main',
            'interactive_marker = scripts.tools.interactive_marker:main',
            'motion_statechart_inspector = scripts.tools.motion_statechart_inspector:main',
            'tracy_standalone = scripts.tracy_standalone:main',
            'tracy_velocity = scripts.tracy_velocity:main'
        ],
    },
)
