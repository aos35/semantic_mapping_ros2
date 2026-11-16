#!/usr/bin/env bash
# Compila el workspace (el repo está montado en ~/ros2_ws/src) y ejecuta el comando.
set -e
source /opt/ros/jazzy/setup.bash
cd /home/developer/ros2_ws
if [ "${SKIP_BUILD:-0}" != "1" ]; then
  colcon build --packages-select semantic_mapping semantic_mapping_bringup \
      --event-handlers console_cohesion+ > /tmp/colcon_build.log 2>&1 \
    || { cat /tmp/colcon_build.log; exit 1; }
fi
source install/setup.bash
exec "$@"
