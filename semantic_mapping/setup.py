from setuptools import find_packages, setup

package_name = 'semantic_mapping'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Grupo Robótica Inteligente',
    maintainer_email='robotica@ua.es',
    description='Mapa semántico 3D a partir de LiDAR + cámaras (Oxford Spires).',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'cloud_projector = semantic_mapping.cloud_projector_node:main',
            'segmentation_node = semantic_mapping.segmentation_node:main',
            'semantic_mapper = semantic_mapping.semantic_mapper_node:main',
        ],
    },
)
