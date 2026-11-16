# Imagen del proyecto: extiende la imagen normalizada de la asignatura
# (docker/Dockerfile.base, copia literal del Dockerfile del curso) con las
# dependencias de visión por computador y los pesos de los modelos.
ARG BASE_IMAGE=ri-jazzy-base
FROM ${BASE_IMAGE}

USER root
COPY requirements.txt /tmp/requirements.txt
# --ignore-installed: evita intentar desinstalar paquetes de apt (p. ej. sympy)
RUN pip3 install --no-cache-dir --break-system-packages --ignore-installed \
        -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

USER developer
# Descarga de pesos en tiempo de build: el experimento se ejecuta sin red.
RUN mkdir -p /home/developer/models /home/developer/output \
    && python3 -c "from transformers import SegformerForSemanticSegmentation as S; S.from_pretrained('nvidia/segformer-b2-finetuned-cityscapes-1024-1024')" \
    && cd /home/developer/models \
    && python3 -c "from ultralytics import YOLO; YOLO('yolo11m-seg.pt')"
ENV YOLO_MODEL=/home/developer/models/yolo11m-seg.pt

WORKDIR /home/developer/ros2_ws
ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "launch", "semantic_mapping_bringup", "semantic_mapping.launch.py"]
