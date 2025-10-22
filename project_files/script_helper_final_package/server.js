const express = require('express');
const http = require('http');
const WebSocket = require('ws');
const cors = require('cors');
const path = require('path');
const fs = require('fs');

// --- Server Setup ---
const app = express();
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

const port = process.env.PORT || 3033;

// --- Middleware ---
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname)));

// --- File-Based Data Store ---
const DATA_FILE = path.join(__dirname, '..', 'Experimento Cline', 'datos', 'client_live_actualizado.json');
let latestClientData = {};

function initializeDataFile() {
    try {
        const emptyData = {
            "meta": {},
            "client": { "nombre": "", "apellidos": "", "email": "", "member_status": "", "telefono": "", "tiene_reservacion": false, "tipo_huesped": "", "motivo": "", "pais": "", "resort": "" },
            "apis": {}, "llamada": {}, "analisis_emociones": {}, "resumen_llamada_md": ""
        };
        fs.writeFileSync(DATA_FILE, JSON.stringify(emptyData, null, 2), 'utf8');
        latestClientData = emptyData;
        console.log('client_data.json has been reset on server start.');
    } catch (error) {
        console.error('Error initializing data file:', error);
    }
}

// --- WebSocket Logic ---
wss.on('connection', (ws) => {
    console.log('Client connected to WebSocket.');
    ws.send(JSON.stringify(latestClientData));
    ws.on('close', () => console.log('Client disconnected.'));
    ws.on('error', (error) => console.error('WebSocket error:', error));
});

function broadcast(data) {
    latestClientData = data;
    console.log('Broadcasting new data to all clients.');
    wss.clients.forEach((client) => {
        if (client.readyState === WebSocket.OPEN) {
            client.send(JSON.stringify(data));
        }
    });
}

// --- File Watcher ---
fs.watch(DATA_FILE, (eventType) => {
    if (eventType === 'change') {
        console.log('client_live_actualizado.json has changed. Reading and broadcasting...');
        try {
            const fileContent = fs.readFileSync(DATA_FILE, 'utf8');
            if (fileContent) {
                const newData = JSON.parse(fileContent);
                broadcast(newData);
            }
        } catch (error) {
            console.warn('Could not read or parse client_live_actualizado.json during change, will retry on next change.');
        }
    }
});


// --- Start the Server ---
server.listen(port, () => {
    initializeDataFile();
    console.log(`HTTP and WebSocket server running on http://localhost:${port}`);
});
