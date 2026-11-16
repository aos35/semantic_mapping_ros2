# semantic_mapping_ros2

Mapa semántico 3D de entornos urbanos a partir de LiDAR + cámaras del
**Oxford Spires Dataset**, sobre **ROS 2 Jazzy** y la odometría LiDAR-inercial
**rko_lio**.
Robótica Inteligente — Práctica II, Opción 4 (Universidad de Alicante, 2026/27).

## Pipeline

```
 ros2 bag play ──/hesai/pandar────────────┬──────────────► rko_lio ──TF odom→pandar──┐
   (--clock)   ──/alphasense_driver_ros/imu┘                                          │
               ──/alphasense_driver_ros/cam{0,1,2}/…/compressed                       │
                        │                       │                                     ▼
                        ▼                       ▼                              semantic_mapper
              segmentation_node ──labels──► cloud_projector ──/semantic_mapping/cloud──►  (voxels + votos)
              (SegFormer / YOLO)             (LiDAR → píxel)                         │
                                                                     /semantic_mapping/map, *.ply
```

| Nodo | Qué hace |
|---|---|
| `segmentation_node` | Segmenta las 3 cámaras (SegFormer-Cityscapes o YOLO11-seg) y publica `/semantic_mapping/camX/labels` (mono8) y `labels_viz`. |
| `cloud_projector` | Transforma cada scan al frame de cada cámara con `T_cam_lidar`, proyecta con la K y el modelo ojo de pez equidistante y asigna a cada punto color y etiqueta. Si un punto lo ven dos cámaras, se queda con la de menor ángulo al eje óptico. |
| `semantic_mapper` | Lleva cada nube etiquetada al frame `odom` con la pose de rko_lio y la fusiona en un mapa voxelizado (0,15 m) con votación por clase (equivale a la actualización bayesiana con un modelo de sensor simétrico). Descarta cielo y personas. Servicio `~/save`. |

Clases: las 19 de Cityscapes (calzada, acera, edificio, vegetación, señal, coche…).

## Ejecución (un único comando)

Requisitos: Docker con soporte GPU NVIDIA (`nvidia-container-toolkit` o Docker Desktop + WSL2).

```bash
# El bag debe estar en ../1710255615_2024-03-12-15-00-16 o indicarse con BAG_DIR
./run.sh                      # mapa semántico completo (SegFormer)
./run.sh mode:=color          # paso 1: nube coloreada con el píxel
./run.sh backend:=yolo        # segmentación con YOLO11-seg
./run.sh rate:=0.5 rviz:=false
```

La primera vez se construye la imagen base de la asignatura (`docker/Dockerfile.base`,
copia literal del Dockerfile del curso) y encima la del proyecto (`docker/Dockerfile`:
PyTorch CUDA, transformers, ultralytics y pesos de los modelos).

Al terminar el bag el sistema se cierra solo y deja en `./output`:

| Fichero | Contenido |
|---|---|
| `semantic_map.ply` | Mapa coloreado por clase (+ campos `label`, `confidence`) |
| `rgb_map.ply` | Mismo mapa con el color real de las imágenes |
| `trajectory.csv` | Trayectoria de rko_lio |
| `stats.json` | Voxels por clase, % etiquetado, confianza media, tiempos |

## Utilidades sin ROS

```bash
# Validar la calibración / figuras para la memoria
python3 tools/project_frame.py --bag ../1710255615_2024-03-12-15-00-16 \
    --calib semantic_mapping_bringup/config/calibration --scan 600 --color depth --out proj.jpg
# Tests de proyección y del mapa voxelizado
python3 -m pytest semantic_mapping/test
```

## Notas sobre los datos

* El bag no contiene `/tf`: los extrínsecos IMU-LiDAR para rko_lio vienen de la
  configuración oficial de rko_lio para Oxford Spires (`config/rko_lio_oxford_spires.yaml`)
  y los LiDAR-cámara de `config/calibration/cam-lidar-imu.yaml` (DiffCal).
* El frame del Hesai (`pandar`) está girado 180° en z respecto al vehículo:
  lo que está delante es −x del LiDAR.
* Cámaras: 1440×1080, ojo de pez equidistante (Kalibr), 20 Hz. LiDAR: 10 Hz.

## Estructura

```
semantic_mapping/            nodos y lógica (projection.py y voxel_map.py no dependen de ROS)
semantic_mapping_bringup/    launch, params.yaml, calibración, config rko_lio, RViz
docker/                      Dockerfile.base (curso), Dockerfile (proyecto), entrypoint
tools/                       herramientas offline
docs/memoria/                LaTeX de la memoria
```
