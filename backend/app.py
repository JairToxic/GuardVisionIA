from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Almacenar las zonas con tipos
zones = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/set_zones', methods=['POST'])
def set_zones():
    global zones
    data = request.get_json()
    zones = data.get('zones', [])
    print("Zonas recibidas:", zones)
    return jsonify(success=True, message="Zonas guardadas correctamente.")

if __name__ == '__main__':
    app.run(debug=True)
