from flask import Blueprint

bp = Blueprint('bp', __name__,
               template_folder='./../frontend/html',
               static_folder='./../frontend',
               static_url_path='assets')
