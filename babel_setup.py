import os
import subprocess
from flask import request, session, jsonify
from flask_babel import Babel


def get_languages_from_dir(directory):
    """Return a list of directory names in the given directory."""
    return [name for name in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, name))]


BABEL_DEFAULT_LOCALE = 'en_US'
TRANSLATIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'translations')
BABEL_LANGUAGES = get_languages_from_dir(TRANSLATIONS_DIR)


def create_babel(app):
    """Create and initialize a Babel instance with the given Flask app."""
    babel = Babel(app)
    app.config['BABEL_DEFAULT_LOCALE'] = BABEL_DEFAULT_LOCALE
    app.config['BABEL_LANGUAGES'] = BABEL_LANGUAGES

    babel.init_app(app, locale_selector=get_locale)
    compile_translations()


def get_locale():
    """Get the user's locale from the session or the request's accepted languages."""
    return session.get('language') or request.accept_languages.best_match(BABEL_LANGUAGES)


def get_languages():
    """Return a list of available languages in JSON format."""
    return jsonify(BABEL_LANGUAGES)


import sys

def compile_translations():
    """Compile the translation files."""
    try:
        cmd = f'"{sys.executable}" -m babel.messages.frontend compile -d "{TRANSLATIONS_DIR}"'
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
        )

        if result.returncode != 0:
            # Fallback to direct pybabel command if installed
            result = subprocess.run(
                f'pybabel compile -d "{TRANSLATIONS_DIR}"',
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True,
            )

        if result.returncode == 0:
            print('Translations compiled successfully')
        else:
            print(f"[Babel] Warning: Could not compile translations: {result.stderr.decode('utf-8', errors='ignore')}")
    except Exception as e:
        print(f"[Babel] Compilation note: {e}")
