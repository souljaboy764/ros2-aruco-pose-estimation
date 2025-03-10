from setuptools import setup, find_packages

setup(
    name='aruco_pose_estimation',
    version='2.0.0',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=['setuptools'],
)