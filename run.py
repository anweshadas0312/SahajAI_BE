import secrets
import os
import sys

# Ensure the root directory is in the python path so 'backend.*' imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

if __name__ == '__main__':

    # Load configuration from config.json (which is now in the backend directory)
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
    if not os.path.exists(config_path):
        config_path = 'config.json'
    config = load(open(config_path, 'r'))
    site_config = config['site_config']
    url_prefix = config.pop('url_prefix')

    # Create the app
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

    # Run the Flask server
    print(f"Running on {site_config['port']}{url_prefix}")
    app.run(**site_config)
    print(f"Closing port {site_config['port']}")
