#  PV Forecasting Australia

Proyecto de **predicción de generación fotovoltaica** basado en **series temporales**, utilizando modelos de **Deep Learning** implementados en **PyTorch**.  
El objetivo es estimar la producción de energía solar a partir de datos históricos y variables meteorológicas en Australia.

---

##  Características principales

-  Procesamiento de datos reales de generación fotovoltaica
-  Modelos de Deep Learning para forecasting de series temporales:
  - LSTM
  - GRU
  - LSTM-FCN
  - Transformer
-  Pipeline completo de Machine Learning:
  - Preprocesamiento de datos
  - Entrenamiento y validación
  - Evaluación con métricas
  - Inferencia
-  Guardado automático del mejor modelo (checkpoints)
- Visualización de resultados
-  Configuración flexible mediante archivos de configuración

---

##  Estructura del proyecto

pv-forecasting-australia/
│
├── data/ # Datos crudos y procesados
├── preprocess/ # Preprocesamiento de series temporales
├── models/ # Arquitecturas Deep Learning
├── utils/ # Métricas, gráficos y utilidades
├── config/ # Configuración de experimentos
├── checkpoints/ # Modelos entrenados
│
├── Main.py # Script principal
├── train.py # Entrenamiento y validación
├── inference.py # Inferencia y predicciones
├── requirements.txt # Dependencias
└── README.md


Para clonar repositorio:


##  Instalación

###  Clonar el repositorio
```bash
git clone https://github.com/USUARIO/pv-forecasting-australia.git
cd pv-forecasting-australia

### Instalar dependencias:
pip install -r requirements.txt

### Entrenar modelo:
python Main.py

### Inferencia:
python inference.py
