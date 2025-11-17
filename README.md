# Taller3_3corte


# Segundo Punto

Este proyecto implementa un sistema completo capaz de identificar dispositivos electrónicos en tiempo real, aplicando un modelo de TensorFlow Lite y mostrando un segmentado azul sobre la región donde se encuentra el objeto detectado.
Además, se desarrolla una arquitectura cliente–servidor usando sockets TCP, hilos, semáforos, colas y mutex, y finalmente todo se despliega dentro de contenedores Docker.

## Entrenamiento y preparación del modelo

El proceso comenzó creando un dataset propio con imágenes de los dispositivos. Con TensorFlow se entrenó un modelo base y luego se convirtió a TensorFlow Lite para hacerlo más rápido.


```
model_unquant.tflite 
```

Y un archivo con las etiquetas

```
labels.txt

```

En el servidor estos archivos se cargan así:

```
self.labels = self.cargar_etiquetas(LABELS_PATH)
self.interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
self.interpreter.allocate_tensors()

```

La conversión a TFLite hace que el modelo sea más liviano y que pueda correr rápido incluso dentro de un contenedor Docker.


## Arquitectura Cliente–Servidor
El sistema se dividió en dos partes independientes servidor.py, que corre el modelo de IA, y cliente.py, que captura la cámara y muestra la detección.

El servidor ejecuta el modelo dentro de un mutex, ya que TFLite no soporta múltiples inferencias simultáneas:

```

with self.modelo_mutex:
    self.interpreter.set_tensor(...)
    self.interpreter.invoke()

```

También se usa un semaforo para limitar el número de conexiones concurrentes:

```
self.semaforo_conexiones = threading.Semaphore(3)
```


Esto evita que muchos clientes saturen el servidor.

## Servidor

El servidor escucha en un puerto TCP, recibe frames enviados desde el cliente, los preprocesa y devuelve la predicción.

```
data_size = struct.unpack(">L", conn.recv(4))[0]
data = conn.recv(data_size)
frame = pickle.loads(data)

```

Y así ejecuta la predicción:
```

tensor = self.preprocesar(frame)
pred = self.predecir(tensor)
```

Luego envía un diccionario con el resultado:

```

resultado = {
    'clase': clase,
    'confianza': confianza,
    'clase_id': clase_id,
    'threshold': CONFIDENCE_THRESHOLD
}
self.enviar_resultado(conn, resultado)

```

## Cliente

El cliente corre tres hilos independientes, comunicados con colas (Queue) y protegidos con mutex.

- Hilo de captura de cámara
```
while self.ejecutando:
    ok, frame = cap.read()
    with self.frame_mutex:
        self.frame_actual = frame.copy()
    self.cola_frames.put(frame.copy())

```

- Hilo de envío y recepción de predicciones

```
frame = self.cola_frames.get()
self.enviar_frame(frame)
resultado = self.recibir_resultado()
self.cola_resultados.put(resultado)

```

- Hilo de visualización

´´´

resultado = self.cola_resultados.get()
if resultado['confianza'] >= resultado['threshold']:
    frame = self.segmentar_objeto(frame, bbox)
```


## Implementación con Docker

Una vez funcionando, se creó una estructura con dos contenedores: uno para el servidor y otro para el cliente.

Cada uno tiene su propio Dockerfile, instalando Python 3.10, OpenCV, TensorFlow Lite y demás dependencias desde:

```
requirements.txt
```


Y el docker-compose.yml conecta ambos servicios esto permite ejecutar IA dentro del contenedor del servidor capturar cámara desde dentro del contenedor cliente, luego mostrar ventanas OpenCV en el sistema operativo del usuario. Finalmente, se construyen e inician así:

```
docker compose build
docker compose up
```

# Tercer punto