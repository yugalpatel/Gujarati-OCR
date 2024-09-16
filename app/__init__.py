from flask import Flask
import os

def create_app():
    app = Flask(__name__, template_folder='../templates')
    app.config['UPLOAD_FOLDER'] = 'static/uploads'
    
    from . import routes
    app.register_blueprint(routes.bp)
    
    return app