# Taller 3

## Segundo Punto

Este proyecto implementa un sistema completo capaz de identificar dispositivos electrónicos en tiempo real, aplicando un modelo de TensorFlow Lite y mostrando un segmentado azul sobre la región donde se encuentra el objeto detectado.
Además, se desarrolla una arquitectura cliente–servidor usando sockets TCP, hilos, semáforos, colas y mutex, y finalmente todo se despliega dentro de contenedores Docker.

### Entrenamiento y preparación del modelo

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


### Arquitectura Cliente–Servidor
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

### Servidor

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

### Cliente

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

```
resultado = self.cola_resultados.get()
if resultado['confianza'] >= resultado['threshold']:
    frame = self.segmentar_objeto(frame, bbox)
```

### Implementación con Docker

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

## Tercer Punto

### ¿Qué es Kubernetes?

Kubernetes (K8s) es una plataforma de código abierto diseñada para automatizar el despliegue, escalado y gestión de aplicaciones en contenedores. Se caracteriza por tener:

1. Orquestación de contenedores
2. Auto-escalado
3. Auto-recuperación
4. Balanceo de carga
5. Gestión declarativa

La implementación de estos K8s se basa en Microservicios en producción, Aplicaciones web de alta disponibilidad, Procesamiento de datos distribuido y Aplicaciones que requieren escalabilidad dinámica.

#### Kubernetes y Dockers

Kubernetes gestionan contenedores (generalmente Docker), organizando dónde y cómo se ejecutan. Mientras Docker crea y ejecuta contenedores individuales, Kubernetes coordina miles de contenedores en múltiples máquinas, garantizando que siempre estén funcionando según lo especificado.

### Desarrollo del Juego

La idea principal de uso de K8s, es crear un juego con la siguiente estructura. Dentro de este juego, se desarrollaran dos implementaciones ( Sencilla y Mejorada ).

<img width="599" height="498" alt="image" src="https://github.com/user-attachments/assets/ea2740b4-6e93-48c6-927f-e603df4eb9ee" />

#### Implementación sencilla

Esta implementación nos deja demostrar el uso del K8s mediante docker, para esto se siguieron los siguientes pasos.

1. Primero se recomienda por organización crear una carpeta con la cual dentro de esta estara todos los archivos necesarios `` mkdir juego-multijugador ``.
2. Dentro de la carpeta, crearemos un archivo llamado ``server.js``.

```
// server.js
const express = require('express');
const http = require('http');
const { Server } = require('socket.io');

const app = express();
const server = http.createServer(app);
const io = new Server(server);

app.get('/', (req, res) => {
  res.send('<h1>🎮 Bienvenido al Juego Multijugador</h1>');
});

let players = {};

io.on('connection', (socket) => {
  console.log('Jugador conectado:', socket.id);
  players[socket.id] = { x: 0, y: 0 };

  socket.on('move', (data) => {
    players[socket.id] = data;
    io.emit('state', players);
  });

  socket.on('disconnect', () => {
    delete players[socket.id];
    io.emit('state', players);
  });
});

const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
  console.log(`Servidor del juego en puerto ${PORT}`);
});
 ```
3. Luego el `` Dockerfile ``, para luego construirlo.

```
# Dockerfile
FROM node:18-alpine

WORKDIR /app

COPY server.js .

RUN npm install express socket.io

EXPOSE 3000

CMD ["node", "server.js"]
```

4. Ahora crearemos el Deployment en Kubernetes `` juego-deployment.yaml ``.

```
apiVersion: apps/v1
kind: Deployment
metadata:
  name: juego-multijugador
spec:
  replicas: 3
  selector:
    matchLabels:
      app: juego
  template:
    metadata:
      labels:
        app: juego
    spec:
      containers:
      - name: servidor-juego
        image: juego-multijugador:v1
        ports:
        - containerPort: 3000
```
- Luego de crear este codigo, para poder construirlo debemos descargar y activar el K8s

``` # Instalar Minikube
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube

minikube start --driver=docker # Iniciar Minikube
minikube status

minikube image load juego-multijugador:v1 # Cargar la imagen Docker en Minikube
```

- Luego aplicamos el Deploymet `` kubectl apply -f juego-deployment.yaml `` y generaremos los pods `` kubectl get pods ``

<img width="1024" height="161" alt="image" src="https://github.com/user-attachments/assets/9f09bb4e-23ae-4017-82f6-f359449f5001" />

  
5.Ahora creareos el Service `` nano juego-service.yaml ``, para luego aplicarlo y luego verificarlo `` kubectl apply -f juego-service.yaml - kubectl get svc ``.

```
apiVersion: v1
kind: Service
metadata:
  name: juego-service
spec:
  type: NodePort
  selector:
    app: juego
  ports:
  - protocol: TCP
    port: 3000
    targetPort: 3000
    nodePort: 30080
```

<img width="1024" height="138" alt="image" src="https://github.com/user-attachments/assets/fe1deeed-27f4-4675-b731-b539ad9b90c9" />

6. Ahora, activaremos el servicio completo para verificar su funcionamiento -> `` minikube service juego-service --url `` este codigo nos dara el URL del juego.

   - En si debe aparecernos solo el titulo del juego. Ahora para un juego mas desarrollado, debemos seguir a la implementación mejorada.


#### Implementación Mejorada

Para nuestro juego mejorado lo unico que debemos hacer es ajustar toda nuestra estructura.

1. Para el ``server.js`` usaremos lo siguiente.

```
const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const path = require('path');

const app = express();
const server = http.createServer(app);
const io = new Server(server);

app.use(express.static('public'));

app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

class GameState {
  constructor() {
    this.players = {};
    this.mutex = false; // Simula un mutex para acceso exclusivo
    this.maxPlayers = 10; // Semáforo: límite de jugadores
    this.gameResources = {
      coins: [],
      obstacles: []
    };
    this.initializeResources();
  }

  initializeResources() {
    // Generar monedas aleatorias
    for (let i = 0; i < 5; i++) {
      this.gameResources.coins.push({
        id: `coin-${i}`,
        x: Math.floor(Math.random() * 750),
        y: Math.floor(Math.random() * 550),
        collected: false
      });
    }

    for (let i = 0; i < 3; i++) {
      this.gameResources.obstacles.push({
        id: `obstacle-${i}`,
        x: Math.floor(Math.random() * 750),
        y: Math.floor(Math.random() * 550)
      });
    }
  }

  async acquireLock() {
    while (this.mutex) {
      await new Promise(resolve => setTimeout(resolve, 10));
    }
    this.mutex = true;
  }

  releaseLock() {
    this.mutex = false;
  }

  canAddPlayer() {
    return Object.keys(this.players).length < this.maxPlayers;
  }

  async addPlayer(id, name) {
    await this.acquireLock(); // Entrada a SECCIÓN CRÍTICA
    
    try {
      if (this.canAddPlayer()) {
        const colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F'];
        this.players[id] = {
          id,
          name: name || `Jugador ${Object.keys(this.players).length + 1}`,
          x: Math.floor(Math.random() * 700) + 50,
          y: Math.floor(Math.random() * 500) + 50,
          color: colors[Object.keys(this.players).length % colors.length],
          score: 0,
          timestamp: Date.now()
        };
        console.log(`✅ [SECCIÓN CRÍTICA] Jugador agregado: ${name} (${id})`);
        return true;
      } else {
        console.log(`❌ [SEMÁFORO] Límite de jugadores alcanzado (${this.maxPlayers})`);
        return false;
      }
    } finally {
      this.releaseLock(); // Salida de SECCIÓN CRÍTICA
    }
  }

  async updatePlayer(id, data) {
    await this.acquireLock(); // Entrada a SECCIÓN CRÍTICA
    
    try {
      if (this.players[id]) {
        this.players[id].x = data.x;
        this.players[id].y = data.y;
        console.log(`📍 [SECCIÓN CRÍTICA] Posición actualizada: ${this.players[id].name}`);
      }
    } finally {
      this.releaseLock(); // Salida de SECCIÓN CRÍTICA
    }
  }

  async collectCoin(playerId, coinId) {
    await this.acquireLock(); // Entrada a SECCIÓN CRÍTICA
    
    try {
      const coin = this.gameResources.coins.find(c => c.id === coinId);
      
      if (coin && !coin.collected) {
        coin.collected = true;
        this.players[playerId].score += 10;
        console.log(`💰 [SECCIÓN CRÍTICA] ${this.players[playerId].name} recolectó ${coinId}`);
        return true;
      }
      return false;
    } finally {
      this.releaseLock(); // Salida de SECCIÓN CRÍTICA
    }
  }

  async removePlayer(id) {
    await this.acquireLock(); // Entrada a SECCIÓN CRÍTICA
    
    try {
      if (this.players[id]) {
        console.log(`👋 [SECCIÓN CRÍTICA] Jugador removido: ${this.players[id].name}`);
        delete this.players[id];
      }
    } finally {
      this.releaseLock(); // Salida de SECCIÓN CRÍTICA
    }
  }

  getState() {
    return {
      players: this.players,
      resources: this.gameResources
    };
  }
}

const gameState = new GameState();

io.on('connection', (socket) => {
  console.log(`🔌 Nueva conexión: ${socket.id}`);

  socket.on('join', async (name) => {
    const added = await gameState.addPlayer(socket.id, name);
    
    if (added) {
      socket.emit('init', {
        id: socket.id,
        state: gameState.getState()
      });

      io.emit('playerJoined', {
        player: gameState.players[socket.id],
        totalPlayers: Object.keys(gameState.players).length
      });

      io.emit('state', gameState.getState());
    } else {
      socket.emit('serverFull', { message: 'Servidor lleno. Intenta más tarde.' });
      socket.disconnect();
    }
  });

  socket.on('move', async (data) => {
    await gameState.updatePlayer(socket.id, data);
    io.emit('state', gameState.getState());
  });

  socket.on('collectCoin', async (coinId) => {
    const collected = await gameState.collectCoin(socket.id, coinId);
    
    if (collected) {
      io.emit('coinCollected', {
        coinId,
        playerId: socket.id,
        newScore: gameState.players[socket.id].score
      });
      io.emit('state', gameState.getState());
    }
  });

  socket.on('disconnect', async () => {
    await gameState.removePlayer(socket.id);
    io.emit('playerLeft', {
      id: socket.id,
      totalPlayers: Object.keys(gameState.players).length
    });
    io.emit('state', gameState.getState());
  });
});

setInterval(() => {
  io.emit('state', gameState.getState());
}, 100); // Actualización cada 100ms (simula hilo de broadcast)

const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
  console.log(`
╔═══════════════════════════════════════════════╗
║  🎮 Servidor de Juego Multijugador           ║
║  📡 Puerto: ${PORT}                              ║
║  👥 Máximo de jugadores: ${gameState.maxPlayers}                ║
║  🔒 Sección crítica: ACTIVA                  ║
║  🚦 Semáforo: ACTIVO                         ║
╚═══════════════════════════════════════════════╝
  `);
});
 ```
2. Ahora crearemos nuestra interfaz en ``mkdir public`` y crearemos el siguiente archivo ``index.html``.

```
<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>🎮 Juego Multijugador - Kubernetes</title>
  <link rel="stylesheet" href="style.css">
</head>
<body>
  <div class="container">
    <div class="info-panel">
      <h1>🎮 Juego Multijugador</h1>
      <div class="stats">
        <div class="stat">
          <span class="label">👥 Jugadores:</span>
          <span id="playerCount">0</span>
        </div>
        <div class="stat">
          <span class="label">🏆 Tu puntuación:</span>
          <span id="myScore">0</span>
        </div>
        <div class="stat">
          <span class="label">🆔 Tu ID:</span>
          <span id="myId">-</span>
        </div>
      </div>
      
      <div class="concepts">
        <h3>💡 Conceptos implementados:</h3>
        <ul>
          <li>🔒 <strong>Mutex:</strong> Control de acceso exclusivo al estado</li>
          <li>🚦 <strong>Semáforo:</strong> Límite de 10 jugadores concurrentes</li>
          <li>⚠️ <strong>Sección Crítica:</strong> Actualización sincronizada</li>
          <li>🧵 <strong>Hilos:</strong> Socket.IO maneja conexiones concurrentes</li>
        </ul>
      </div>
    </div>

    <div class="game-container">
      <canvas id="gameCanvas" width="800" height="600"></canvas>
      <div class="controls">
        <p>⌨️ Usa las flechas o WASD para moverte</p>
        <p>💰 Recolecta monedas para sumar puntos</p>
      </div>
    </div>

    <div class="ranking-panel">
      <h3>🏆 Ranking de Jugadores</h3>
      <div id="ranking"></div>
    </div>
  </div>

  <div id="nameModal" class="modal">
    <div class="modal-content">
      <h2>🎮 Bienvenido al Juego</h2>
      <input type="text" id="playerName" placeholder="Ingresa tu nombre" maxlength="15">
      <button onclick="joinGame()">🚀 Jugar</button>
    </div>
  </div>

  <script src="/socket.io/socket.io.js"></script>
  <script src="game.js"></script>
</body>
</html>
 ```
3. Luego crearemos otro archivo nombrado `` style.css ``.

```
{
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  min-height: 100vh;
  display: flex;
  justify-content: center;
  align-items: center;
  padding: 20px;
}

.container {
  display: grid;
  grid-template-columns: 300px 1fr 250px;
  gap: 20px;
  max-width: 1600px;
  width: 100%;
}

.info-panel {
  background: white;
  border-radius: 15px;
  padding: 20px;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
}

.info-panel h1 {
  font-size: 24px;
  margin-bottom: 20px;
  color: #667eea;
  text-align: center;
}

.stats {
  margin-bottom: 20px;
}

.stat {
  background: #f7f7f7;
  padding: 10px;
  margin-bottom: 10px;
  border-radius: 8px;
  display: flex;
  justify-content: space-between;
}

.label {
  font-weight: 600;
  color: #555;
}

.concepts {
  margin-top: 20px;
  padding-top: 20px;
  border-top: 2px solid #eee;
}

.concepts h3 {
  color: #667eea;
  margin-bottom: 10px;
  font-size: 16px;
}

.concepts ul {
  list-style: none;
}

.concepts li {
  padding: 8px 0;
  font-size: 13px;
  line-height: 1.5;
  color: #666;
}

.concepts strong {
  color: #333;
}

.game-container {
  display: flex;
  flex-direction: column;
  align-items: center;
}

#gameCanvas {
  background: #1a1a2e;
  border-radius: 15px;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
  cursor: crosshair;
}

.controls {
  background: white;
  padding: 15px;
  margin-top: 15px;
  border-radius: 10px;
  text-align: center;
  box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
}

.controls p {
  margin: 5px 0;
  color: #555;
  font-size: 14px;
}

.ranking-panel {
  background: white;
  border-radius: 15px;
  padding: 20px;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
  max-height: 600px;
  overflow-y: auto;
}

.ranking-panel h3 {
  color: #667eea;
  margin-bottom: 15px;
  text-align: center;
}

.player-rank {
  background: #f7f7f7;
  padding: 10px;
  margin-bottom: 8px;
  border-radius: 8px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  transition: transform 0.2s;
}

.player-rank:hover {
  transform: translateX(5px);
}

.player-rank.me {
  background: #667eea;
  color: white;
  font-weight: bold;
}

.modal {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(0, 0, 0, 0.8);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 1000;
}

.modal-content {
  background: white;
  padding: 40px;
  border-radius: 20px;
  text-align: center;
  box-shadow: 0 15px 50px rgba(0, 0, 0, 0.5);
  animation: slideIn 0.3s ease-out;
}

@keyframes slideIn {
  from {
    transform: translateY(-50px);
    opacity: 0;
  }
  to {
    transform: translateY(0);
    opacity: 1;
  }
}

.modal-content h2 {
  color: #667eea;
  margin-bottom: 20px;
}

.modal-content input {
  width: 100%;
  padding: 15px;
  border: 2px solid #ddd;
  border-radius: 10px;
  font-size: 16px;
  margin-bottom: 20px;
  transition: border-color 0.3s;
}

.modal-content input:focus {
  outline: none;
  border-color: #667eea;
}

.modal-content button {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  padding: 15px 40px;
  border-radius: 10px;
  font-size: 18px;
  cursor: pointer;
  transition: transform 0.2s;
}

.modal-content button:hover {
  transform: scale(1.05);
}

@media (max-width: 1400px) {
  .container {
    grid-template-columns: 1fr;
    grid-template-rows: auto auto auto;
  }
  
  .info-panel, .ranking-panel {
    max-width: 800px;
    margin: 0 auto;
    width: 100%;
  }
}
```

4. Crearemos la logica funcional del juego `` nano game.js `` y por ultimo el Dockerfile.

```
const socket = io();

const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');

let myId = null;
let myPlayer = null;
let gameState = { players: {}, resources: { coins: [], obstacles: [] } };

const keys = {};

function joinGame() {
  const name = document.getElementById('playerName').value.trim() || 'Jugador';
  document.getElementById('nameModal').style.display = 'none';
  socket.emit('join', name);
}

socket.on('init', (data) => {
  myId = data.id;
  gameState = data.state;
  myPlayer = gameState.players[myId];
  
  document.getElementById('myId').textContent = myId.substring(0, 8);
  
  console.log('✅ Conectado al servidor');
  console.log('🔒 Mutex y secciones críticas activas en el servidor');
});

socket.on('state', (state) => {
  gameState = state;
  if (myId && gameState.players[myId]) {
    myPlayer = gameState.players[myId];
    updateUI();
  }
});

socket.on('playerJoined', (data) => {
  console.log(`👋 ${data.player.name} se unió al juego`);
  document.getElementById('playerCount').textContent = data.totalPlayers;
});

socket.on('playerLeft', (data) => {
  console.log(`👋 Jugador ${data.id} salió del juego`);
  document.getElementById('playerCount').textContent = data.totalPlayers;
});

socket.on('coinCollected', (data) => {
  if (data.playerId === myId) {
    console.log(`💰 ¡Recolectaste una moneda! Puntos: ${data.newScore}`);
  }
});

socket.on('serverFull', (data) => {
  alert(data.message);
});

document.addEventListener('keydown', (e) => {
  keys[e.key] = true;
});

document.addEventListener('keyup', (e) => {
  keys[e.key] = false;
});

function gameLoop() {
  if (!myPlayer) {
    requestAnimationFrame(gameLoop);
    return;
  }

  // Movimiento (entrada del jugador)
  let moved = false;
  const speed = 5;

  if (keys['ArrowUp'] || keys['w'] || keys['W']) {
    myPlayer.y = Math.max(0, myPlayer.y - speed);
    moved = true;
  }
  if (keys['ArrowDown'] || keys['s'] || keys['S']) {
    myPlayer.y = Math.min(canvas.height - 30, myPlayer.y + speed);
    moved = true;
  }
  if (keys['ArrowLeft'] || keys['a'] || keys['A']) {
    myPlayer.x = Math.max(0, myPlayer.x - speed);
    moved = true;
  }
  if (keys['ArrowRight'] || keys['d'] || keys['D']) {
    myPlayer.x = Math.min(canvas.width - 30, myPlayer.x + speed);
    moved = true;
  }

  // Enviar actualización al servidor (entra a sección crítica)
  if (moved) {
    socket.emit('move', { x: myPlayer.x, y: myPlayer.y });
  }

  // Detectar colisión con monedas
  checkCoinCollision();

  // Renderizar
  render();

  requestAnimationFrame(gameLoop);
}

function checkCoinCollision() {
  gameState.resources.coins.forEach(coin => {
    if (!coin.collected) {
      const dx = myPlayer.x + 15 - coin.x;
      const dy = myPlayer.y + 15 - coin.y;
      const distance = Math.sqrt(dx * dx + dy * dy);

      if (distance < 25) {
        socket.emit('collectCoin', coin.id);
      }
    }
  });
}

function render() {
  // Limpiar canvas
  ctx.fillStyle = '#1a1a2e';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // Dibujar grid
  ctx.strokeStyle = '#2a2a3e';
  ctx.lineWidth = 1;
  for (let i = 0; i < canvas.width; i += 50) {
    ctx.beginPath();
    ctx.moveTo(i, 0);
    ctx.lineTo(i, canvas.height);
    ctx.stroke();
  }
  for (let i = 0; i < canvas.height; i += 50) {
    ctx.beginPath();
    ctx.moveTo(0, i);
    ctx.lineTo(canvas.width, i);
    ctx.stroke();
  }

  gameState.resources.obstacles.forEach(obstacle => {
    ctx.fillStyle = '#FF6B6B';
    ctx.fillRect(obstacle.x, obstacle.y, 30, 30);
    ctx.fillStyle = 'white';
    ctx.font = '20px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('🚧', obstacle.x + 15, obstacle.y + 22);
  });

  gameState.resources.coins.forEach(coin => {
    if (!coin.collected) {
      ctx.fillStyle = '#FFD700';
      ctx.beginPath();
      ctx.arc(coin.x, coin.y, 10, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = 'white';
      ctx.font = '16px Arial';
      ctx.textAlign = 'center';
      ctx.fillText('💰', coin.x, coin.y + 5);
    }
  });

  Object.values(gameState.players).forEach(player => {
    const isMe = player.id === myId;

    ctx.fillStyle = player.color;
    ctx.beginPath();
    ctx.arc(player.x + 15, player.y + 15, 15, 0, Math.PI * 2);
    ctx.fill();

    if (isMe) {
      ctx.strokeStyle = '#FFD700';
      ctx.lineWidth = 3;
      ctx.stroke();
    }

    ctx.fillStyle = 'white';
    ctx.font = 'bold 12px Arial';
    ctx.textAlign = 'center';
    ctx.fillText(player.name, player.x + 15, player.y - 5);

    ctx.font = '10px Arial';
    ctx.fillText(`${player.score} pts`, player.x + 15, player.y + 45);
  });

  ctx.fillStyle = 'rgba(255, 255, 255, 0.1)';
  ctx.fillRect(10, 10, 250, 60);
  ctx.fillStyle = 'white';
  ctx.font = 'bold 14px Arial';
  ctx.textAlign = 'left';
  ctx.fillText('🔒 Sección Crítica Activa', 20, 30);
  ctx.font = '11px Arial';
  ctx.fillText('Sincronización de estado en servidor', 20, 50);
}

function updateUI() {
  // Actualizar puntuación
  document.getElementById('myScore').textContent = myPlayer.score;

  // Actualizar ranking
  const ranking = Object.values(gameState.players)
    .sort((a, b) => b.score - a.score);

  const rankingHTML = ranking.map((player, index) => {
    const isMe = player.id === myId;
    return `
      <div class="player-rank ${isMe ? 'me' : ''}">
        <span>${index + 1}. ${player.name}</span>
        <span>${player.score} pts</span>
      </div>
    `;
  }).join('');

  document.getElementById('ranking').innerHTML = rankingHTML;
}

gameLoop();
```
- Para el docker usaremos el mismo que creamos anteriormente.

```
 FROM node:18-alpine

WORKDIR /app

COPY server.js .
COPY public ./public

RUN npm install express socket.io

EXPOSE 3000

CMD ["node", "server.js"]
```

5. Por ultimo reconstruiremos todo y nos aparecera lo siguiente en nuestro juego.

<img width="1847" height="930" alt="Captura desde 2025-11-21 19-56-05" src="https://github.com/user-attachments/assets/2aa1cacc-44c0-474d-8eba-683752495496" />

<img width="1847" height="930" alt="image" src="https://github.com/user-attachments/assets/fb3657a2-e818-4136-8039-2f0e4be646fb" />
