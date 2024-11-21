from flask import Flask, render_template, request, jsonify, send_from_directory
import os

app = Flask(__name__)

# Ruta del directorio donde se encuentra el video
VIDEO_DIR = os.path.join(os.getcwd(), 'static')

@app.route('/')
def index():
    # Aquí pasamos el nombre del video para que se pueda utilizar en el frontend
    video_filename = '1.mp4'  # El video ahora está en la carpeta static
    return render_template('index.html', video_filename=video_filename)

@app.route('/video/<filename>')
def video(filename):
    return send_from_directory(VIDEO_DIR, filename)

@app.route('/set_zones', methods=['POST'])
def set_zones():
    data = request.get_json()
    zones = data.get('zones', [])
    print(f"Zonas recibidas: {zones}")
    # Aquí puedes procesar y guardar las zonas
    return jsonify({"status": "success", "zones": zones})

if __name__ == '__main__':
    app.run(debug=True)
