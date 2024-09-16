from flask import Flask
from routes import bp as routes_bp
from app import create_app

def create_app():
    app = Flask(__name__)
    app.config['UPLOAD_FOLDER'] = 'static/uploads'
    
    app.register_blueprint(routes_bp)
    
    return app

app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)

