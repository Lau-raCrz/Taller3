import cv2
import socket
import threading
import numpy as np
import pickle
import struct
from queue import Queue

class ClienteDeteccion:
    def __init__(self, host='servidor', port=5001):
        self.host = host
        self.port = port
        self.socket = None
        
        # Colas para comunicación entre hilos
        self.cola_frames = Queue(maxsize=2)
        self.cola_resultados = Queue(maxsize=2)
        
        # Mutex para proteger frame actual
        self.frame_mutex = threading.Lock()
        self.frame_actual = None
        self.resultado_actual = None
        
        # Control de hilos
        self.ejecutando = True
        
        # Semáforo para controlar envío de frames
        self.semaforo_envio = threading.Semaphore(1)
        
    def conectar(self):
        """Conecta al servidor"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            print(f"✅ Conectado al servidor {self.host}:{self.port}\n")
            return True
        except Exception as e:
            print(f"❌ Error al conectar: {e}")
            return False
    
    def enviar_frame(self, frame):
        """Envía un frame al servidor"""
        data = pickle.dumps(frame)
        self.socket.sendall(struct.pack(">L", len(data)))
        self.socket.sendall(data)
    
    def recibir_resultado(self):
        """Recibe el resultado del servidor"""
        data_size = struct.unpack(">L", self.socket.recv(4))[0]
        
        data = b""
        while len(data) < data_size:
            packet = self.socket.recv(4096)
            if not packet:
                return None
            data += packet
        
        resultado = pickle.loads(data)
        return resultado
    
    def hilo_captura(self, cap):
        """Hilo para capturar frames de la cámara"""
        print("📹 Hilo de captura iniciado")
        
        while self.ejecutando:
            ok, frame = cap.read()
            if not ok:
                print("⚠ Error capturando frame")
                break
            
            frame = cv2.flip(frame, 1)
            
            # Usar mutex para actualizar frame actual
            with self.frame_mutex:
                self.frame_actual = frame.copy()
            
            # Agregar a cola si no está llena
            if not self.cola_frames.full():
                self.cola_frames.put(frame.copy())
        
        print("📹 Hilo de captura terminado")
    
    def hilo_procesamiento(self):
        """Hilo para enviar frames y recibir resultados"""
        print("⚙️ Hilo de procesamiento iniciado")
        
        while self.ejecutando:
            try:
                # Obtener frame de la cola
                frame = self.cola_frames.get(timeout=1)
                
                # Usar semáforo para controlar envío
                self.semaforo_envio.acquire()
                
                try:
                    # Enviar frame al servidor
                    self.enviar_frame(frame)
                    
                    # Recibir resultado
                    resultado = self.recibir_resultado()
                    
                    if resultado:
                        # Debug: mostrar resultado recibido
                        print(f"📦 Recibido: {resultado['clase']} - {resultado['confianza']*100:.1f}%")
                        
                        # Agregar a cola de resultados
                        if not self.cola_resultados.full():
                            self.cola_resultados.put(resultado)
                        else:
                            # Si la cola está llena, vaciar y agregar nuevo
                            try:
                                self.cola_resultados.get_nowait()
                            except:
                                pass
                            self.cola_resultados.put(resultado)
                
                finally:
                    self.semaforo_envio.release()
                    
            except Exception as e:
                if self.ejecutando:
                    print(f"⚠ Error en procesamiento: {e}")
                    import traceback
                    traceback.print_exc()
                continue  # Continuar en lugar de break
        
        print("⚙️ Hilo de procesamiento terminado")
    
    def segmentar_objeto(self, frame, bbox):
        """Segmenta el objeto detectado con overlay azul"""
        x1, y1, x2, y2 = bbox
        
        obj = frame[y1:y2, x1:x2].copy()
        
        # Crear máscara azul
        azul = np.zeros_like(obj)
        azul[:] = (255, 0, 0)  # Azul BGR
        
        alpha = 0.5
        segmentado = cv2.addWeighted(obj, 1 - alpha, azul, alpha, 0)
        
        # Insertar de nuevo en el frame
        resultado = frame.copy()
        resultado[y1:y2, x1:x2] = segmentado
        return resultado
    
    def hilo_visualizacion(self):
        """Hilo para visualizar frames con detecciones"""
        print("🖥️ Hilo de visualización iniciado")
        
        cv2.namedWindow("Detección + Segmentación Azul", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Detección + Segmentación Azul", 900, 700)
        
        while self.ejecutando:
            # Obtener resultado de la cola
            try:
                resultado = self.cola_resultados.get(timeout=0.1)
                with self.frame_mutex:
                    self.resultado_actual = resultado
            except:
                pass
            
            # Obtener frame actual
            with self.frame_mutex:
                if self.frame_actual is None:
                    continue
                frame = self.frame_actual.copy()
                resultado = self.resultado_actual
            
            # ============================
            # DIBUJAR RESULTADOS (IGUAL QUE EL CÓDIGO ORIGINAL)
            # ============================
            h, w = frame.shape[:2]
            x1 = int(w * 0.25)
            y1 = int(h * 0.20)
            x2 = int(w * 0.75)
            y2 = int(h * 0.80)
            bbox = (x1, y1, x2, y2)
            
            if resultado and resultado['confianza'] >= resultado['threshold']:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(frame,
                            f"{resultado['clase']}: {resultado['confianza']*100:.1f}%",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (0, 255, 0), 2)
                
                # ---- Segmentar SOLO el objeto ----
                frame = self.segmentar_objeto(frame, bbox)
            else:
                cv2.putText(frame, "Sin deteccion confiable",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, (0, 0, 255), 3)
            
            cv2.imshow("Detección + Segmentación Azul", frame)
            
            if cv2.waitKey(1) == ord("q"):
                self.ejecutando = False
                break
        
        cv2.destroyAllWindows()
        print("🖥️ Hilo de visualización terminado")
    
    def ejecutar(self):
        """Ejecuta el cliente con hilos"""
        if not self.conectar():
            return
        
        # Abrir cámara
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Error: cámara no disponible")
            return
        
        print("\n=== Detector de dispositivos ===")
        print("📌 Presiona 'q' para salir\n")
        
        # Crear hilos
        hilo_cap = threading.Thread(target=self.hilo_captura, args=(cap,), daemon=True)
        hilo_proc = threading.Thread(target=self.hilo_procesamiento, daemon=True)
        hilo_vis = threading.Thread(target=self.hilo_visualizacion, daemon=True)
        
        # Iniciar hilos
        hilo_cap.start()
        hilo_proc.start()
        hilo_vis.start()
        
        # Esperar a que termine visualización
        hilo_vis.join()
        
        # Detener otros hilos
        self.ejecutando = False
        cap.release()
        
        # Esperar a que terminen los otros hilos
        hilo_cap.join(timeout=2)
        hilo_proc.join(timeout=2)
        
        # Cerrar socket
        if self.socket:
            self.socket.close()
        
        print("\n✅ Cliente cerrado correctamente")


if __name__ == "__main__":
    cliente = ClienteDeteccion()
    cliente.ejecutar()
