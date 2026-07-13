# server.py
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import eventlet
eventlet.monkey_patch()

app = Flask(__name__, static_folder="../client", template_folder="../client")
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route("/")
def index():
    return app.send_static_file("receiver.html")

@app.route("/api/gesture", methods=["POST"])
def api_gesture():
    data = request.json
    # broadcast via SocketIO to connected clients
    socketio.emit('new_gesture', data, broadcast=True)
    print("Received gesture:", data)
    return jsonify({"status":"ok"})

@socketio.on('connect')
def on_connect():
    print("Client connected")

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000)
