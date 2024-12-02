from setuptools import find_packages, setup

package_name = 'ros2_to_rlds'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'envlogger', 'tfds'],
    zip_safe=True,
    maintainer='NU Haptics',
    maintainer_email='buckley.toby@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ros2_rlds_server = ros2_to_rlds.ros2_rlds_server:main',
        ],
    },
)
