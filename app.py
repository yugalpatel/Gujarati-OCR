from flask import Flask

def create_app():
    app = Flask(__name__)
    app.config['UPLOAD_FOLDER'] = 'static/uploads'
    
    from . import routes
    app.register_blueprint(routes.bp)
    
    return app

from app import create_app

app = create_app()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
