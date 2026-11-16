"""Pipeline completo en un único launch:

    ros2 bag play (--clock)  ->  rko_lio (odometría, TF odom->pandar)
                             ->  segmentation_node (solo mode:=semantic)
                             ->  cloud_projector    (LiDAR -> píxeles: color / etiqueta)
                             ->  semantic_mapper    (fusión voxelizada en 'odom')
                             ->  rviz2

Ejemplos:
    ros2 launch semantic_mapping_bringup semantic_mapping.launch.py
    ros2 launch semantic_mapping_bringup semantic_mapping.launch.py mode:=color
    ros2 launch semantic_mapping_bringup semantic_mapping.launch.py backend:=yolo rate:=0.5
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, EmitEvent, ExecuteProcess, GroupAction,
                            IncludeLaunchDescription, RegisterEventHandler, TimerAction)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('semantic_mapping_bringup')
    params = os.path.join(share, 'config', 'params.yaml')
    calib = os.path.join(share, 'config', 'calibration')
    lio_cfg = os.path.join(share, 'config', 'rko_lio_oxford_spires.yaml')
    rviz_cfg = os.path.join(share, 'rviz', 'semantic_mapping.rviz')

    bag = LaunchConfiguration('bag_path')
    mode = LaunchConfiguration('mode')
    semantic = PythonExpression(["'", mode, "' == 'semantic'"])
    sim = {'use_sim_time': True}

    args = [
        DeclareLaunchArgument('bag_path', default_value='/data/bag'),
        DeclareLaunchArgument('mode', default_value='semantic', choices=['semantic', 'color']),
        DeclareLaunchArgument('backend', default_value='segformer', choices=['segformer', 'yolo']),
        DeclareLaunchArgument('model', default_value=''),
        DeclareLaunchArgument('rate', default_value='1.0', description='Velocidad de reproducción del bag'),
        DeclareLaunchArgument('start_delay', default_value='15.0',
                              description='s de espera antes de reproducir el bag (carga del modelo)'),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('shutdown_on_bag_end', default_value='true'),
        DeclareLaunchArgument('output_dir', default_value='/home/developer/output'),
        # tiempo para que semantic_mapper guarde el mapa al cerrar
        DeclareLaunchArgument('sigterm_timeout', default_value='60'),
    ]

    # Ámbito propio: los argumentos de launch son globales y rko_lio también
    # declara 'mode' (online|offline) y 'rviz'; sin esto chocan con los nuestros.
    rko_lio = GroupAction(scoped=True, forwarding=True, actions=[IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('rko_lio'), 'launch', 'odometry.launch.py')),
        # Solo mode, autodetect, rviz y config_file se leen como argumentos de launch;
        # el resto de parámetros de rko_lio están en config/rko_lio_oxford_spires.yaml.
        launch_arguments={
            'mode': 'online',
            'autodetect': 'false',   # el bag arranca más tarde: no hay nada que detectar
            'rviz': 'false',
            'config_file': lio_cfg,
        }.items())])

    segmentation = Node(
        package='semantic_mapping', executable='segmentation_node', name='segmentation_node',
        output='screen', condition=IfCondition(semantic),
        parameters=[params, sim, {'backend': LaunchConfiguration('backend'),
                                  'model': LaunchConfiguration('model')}])

    projector = Node(
        package='semantic_mapping', executable='cloud_projector', name='cloud_projector',
        output='screen',
        parameters=[params, sim, {'calibration_dir': calib, 'use_labels': semantic}])

    mapper = Node(
        package='semantic_mapping', executable='semantic_mapper', name='semantic_mapper',
        output='screen',
        parameters=[params, sim, {
            'output_dir': LaunchConfiguration('output_dir'),
            'color_mode': PythonExpression(["'semantic' if '", mode, "' == 'semantic' else 'rgb'"]),
        }])

    rviz = Node(package='rviz2', executable='rviz2', arguments=['-d', rviz_cfg],
                parameters=[sim], condition=IfCondition(LaunchConfiguration('rviz')))

    bag_play = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', bag, '--clock', '-r', LaunchConfiguration('rate'),
             '--read-ahead-queue-size', '5000'],
        output='screen')

    # Al terminar el bag se deja 5 s para vaciar colas y se cierra todo
    # (semantic_mapper guarda los resultados al recibir la señal de cierre).
    end = RegisterEventHandler(
        OnProcessExit(target_action=bag_play, on_exit=[
            TimerAction(period=5.0, actions=[EmitEvent(event=Shutdown(reason='bag terminado'))])]),
        condition=IfCondition(LaunchConfiguration('shutdown_on_bag_end')))

    return LaunchDescription(args + [
        rko_lio, segmentation, projector, mapper, rviz, end,
        TimerAction(period=LaunchConfiguration('start_delay'), actions=[bag_play]),
    ])
