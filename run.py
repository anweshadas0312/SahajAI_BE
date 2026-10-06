import secrets
import os
import sys

# Ensure current directory is at the top of sys.path
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
parent_dir = os.path.dirname(backend_dir)
if parent_dir not in sys.path:
    sys.path.insert(1, parent_dir)


from bp import bp
from website import Website
try:
    from backend import Backend_Api
except ImportError:
    # pyrefly: ignore [missing-import]
    from backend.backend import Backend_Api
from babel_setup import create_babel
from json import load
from flask import Flask

def create_app():
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
    if not os.path.exists(config_path):
        config_path = 'config.json'
    config = load(open(config_path, 'r'))
    site_config = config['site_config']
    url_prefix = config.pop('url_prefix', '')

    app = Flask(__name__)
    app.secret_key = secrets.token_hex(16)

    # Set up Babel
    create_babel(app)

    # Set up the website routes
    site = Website(bp, url_prefix)
    for route in site.routes:
        bp.add_url_rule(
            route,
            view_func=site.routes[route]['function'],
            methods=site.routes[route]['methods'],
        )

    # Set up the backend API routes
    backend_api = Backend_Api(bp, config)
    for route in backend_api.routes:
        bp.add_url_rule(
            route,
            view_func=backend_api.routes[route]['function'],
            methods=backend_api.routes[route]['methods'],
        )

    # Register the blueprint
    app.register_blueprint(bp, url_prefix=url_prefix)

    # Register modern Auth, Workspace, File Intelligence, and Admin API blueprint
    try:
        from api_routes import api_bp
        app.register_blueprint(api_bp)
        print("[API] Successfully registered /api routes (auth, workspaces, files, admin)")
    except Exception as e:
        print(f"[API] ERROR loading api_routes: {e}")
        import traceback
        traceback.print_exc()

    # Enable CORS for frontend requests
    try:
        from flask_cors import CORS
        CORS(app, resources={r"/*": {"origins": "*"}})
    except Exception as e:
        print(f"[CORS] Note: {e}")

    # Initialize MySQL Database & Tables
    try:
        import db
        db.init_db()
    except Exception as dbe:
        print(f"[DB] Database startup check: {dbe}")

    return app, site_config

app, site_config = create_app()

if __name__ == '__main__':
    # Run the Flask server
    print(f"Running on {site_config['port']}")
    app.run(**site_config)
    print(f"Closing port {site_config['port']}")

