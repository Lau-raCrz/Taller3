import socket
import threading
import numpy as np
import tflite_runtime.interpreter as tflite
import pickle
import struct

# ===============================
# CONFIGURACIÓN DEL MODELO
# ===============================
MODEL_PATH = "model_unquant.tflite"
LABELS_PATH = "labels.txt"
CONFIDENCE_THRESHOLD = 0.70

class ServidorDeteccion:
    def __init__(self, host='0.0.0.0', port=5001):
        self.host = host
        self.port = port
        self.socket = None
        
        # Mutex para proteger acceso al modelo
        self.modelo_mutex = threading.Lock()
        
        # Semáforo para limitar conexiones concurrentes
        self.semaforo_conexiones = threading.Semaphore(3)
        
        # Cargar modelo y etiquetas
        print("🔄 Cargando modelo...")
        self.labels = self.cargar_etiquetas(LABELS_PATH)
        self.interpreter = self.cargar_modelo(MODEL_PATH)
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.H = self.input_details[0]["shape"][1]
        self.W = self.input_details[0]["shape"][2]
        print("✅ Modelo cargado correctamente\n")
        
    def cargar_etiquetas(self, labels_path):
        with open(labels_path, 'r') as f:
            return [x.strip() for x in f.readlines()]
    
    def cargar_modelo(self, model_path):
	interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    
    def preprocesar(self, frame):
        import cv2
        img = cv2.resize(frame, (self.W, self.H))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        return np.expand_dims(img, 0)
    
    def predecir(self, tensor):
        # Usar mutex para proteger acceso al modelo
        with self.modelo_mutex:
            self.interpreter.set_tensor(self.input_details[0]["index"], tensor)
            self.interpreter.invoke()
            return self.interpreter.get_tensor(self.output_details[0]["index"])[0]
    
    def recibir_frame(self, conn):
        """Recibe un frame serializado del cliente"""
        # Primero recibir el tamaño del mensaje
        data_size = struct.unpack(">L", conn.recv(4))[0]
        
        # Recibir los datos del frame
        data = b""
        while len(data) < data_size:
            packet = conn.recv(4096)
            if not packet:
                return None
            data += packet
        
        # Deserializar el frame
        frame = pickle.loads(data)
        return frame
    
    def enviar_resultado(self, conn, resultado):
        """Envía el resultado de la predicción al cliente"""
        data = pickle.dumps(resultado)
        conn.sendall(struct.pack(">L", len(data)))
        conn.sendall(data)
    
    def manejar_cliente(self, conn, addr):
        """Maneja la conexión con un cliente (ejecutado en un hilo)"""
        print(f"✅ Cliente conectado desde {addr}")
        
        try:
            while True:
                # Recibir frame
                frame = self.recibir_frame(conn)
                if frame is None:
                    print(f"⚠ Cliente {addr} desconectado")
                    break
                
                # Procesar frame
                tensor = self.preprocesar(frame)
                pred = self.predecir(tensor)
                
                clase_id = int(np.argmax(pred))
                confianza = float(pred[clase_id])
                clase = self.labels[clase_id]
                
                # Debug: Imprimir predicción
                print(f"🔍 [{addr}] {clase}: {confianza*100:.1f}%")
                
                # Preparar resultado
                resultado = {
                    'clase': clase,
                    'confianza': confianza,
                    'clase_id': clase_id,
                    'threshold': CONFIDENCE_THRESHOLD
                }
                
                # Enviar resultado
                self.enviar_resultado(conn, resultado)
                
        except Exception as e:
            print(f"❌ Error con cliente {addr}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            conn.close()
            self.semaforo_conexiones.release()
            print(f"🔌 Conexión cerrada con {addr}")
    
    def iniciar(self):
        """Inicia el servidor"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Permitir reutilización inmediata del puerto
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Para sistemas Linux, también agregar SO_REUSEPORT si está disponible
        try:
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except AttributeError:
            pass  # SO_REUSEPORT no disponible en este sistema
        
        try:
            self.socket.bind((self.host, self.port))
        except OSError as e:
            print(f"❌ Error: Puerto {self.port} en uso")
            print(f"💡 Soluciones:")
            print(f"   1. Ejecuta: sudo lsof -i :{self.port}")
            print(f"   2. Mata el proceso: kill -9 PID")
            print(f"   3. Espera 1-2 minutos")
            print(f"   4. Cambia el puerto en servidor.py y cliente.py")
            raise e
        
        self.socket.listen(5)
        
        print(f"🚀 Servidor escuchando en {self.host}:{self.port}")
        print(f"📊 Modelo: {MODEL_PATH}")
        print(f"🎯 Umbral de confianza: {CONFIDENCE_THRESHOLD*100}%\n")
        
        try:
            while True:
                conn, addr = self.socket.accept()
                
                # Usar semáforo para limitar conexiones
                self.semaforo_conexiones.acquire()
                
                # Crear hilo para manejar cliente
                hilo_cliente = threading.Thread(
                    target=self.manejar_cliente,
                    args=(conn, addr),
                    daemon=True
                )
                hilo_cliente.start()
                
        except KeyboardInterrupt:
            print("\n⚠ Servidor interrumpido por el usuario")
        finally:
            self.socket.close()
            print("🔌 Servidor cerrado")


if __name__ == "__main__":
    servidor = ServidorDeteccion()
    servidor.iniciar()
