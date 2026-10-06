from flask import Blueprint, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
import db
from auth import generate_token, token_required, admin_required

api_bp = Blueprint('api', __name__, url_prefix='/api')


# ==========================================
# 1. AUTHENTICATION ENDPOINTS
# ==========================================

@api_bp.route('/auth/register', methods=['POST'])
def register():
    data = request.get_json() or {}
    username = data.get('username', '').strip()
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')

    if not username or not email or not password:
        return jsonify({'success': False, 'error': 'Username, email, and password are required'}), 400

    if len(password) < 6:
        return jsonify({'success': False, 'error': 'Password must be at least 6 characters'}), 400

    try:
        # Check if username or email already exists
        existing = db.get_user_by_username_or_email(username) or db.get_user_by_username_or_email(email)
        if existing:
            return jsonify({'success': False, 'error': 'Username or Email is already registered'}), 400

        pw_hash = generate_password_hash(password)
        user_id = db.create_user(username, email, pw_hash, role='user')
        user = db.get_user_by_id(user_id)
        token = generate_token(user['id'], user['role'])

        db.log_audit(user_id, 'USER_REGISTER', f"Registered user {username} ({email})")

        return jsonify({
            'success': True,
            'token': token,
            'user': {
                'id': user['id'],
                'username': user['username'],
                'email': user['email'],
                'role': user['role']
            }
        }), 201
    except Exception as e:
        return jsonify({'success': False, 'error': f"Database error: {str(e)}"}), 500


@api_bp.route('/auth/login', methods=['POST'])
def login():
    data = request.get_json() or {}
    identifier = data.get('identifier', '').strip()
    password = data.get('password', '')

    if not identifier or not password:
        return jsonify({'success': False, 'error': 'Username/email and password are required'}), 400

    try:
        user = db.get_user_by_username_or_email(identifier)
        if not user or not check_password_hash(user['password_hash'], password):
            return jsonify({'success': False, 'error': 'Invalid credentials'}), 401

        token = generate_token(user['id'], user['role'])
        db.log_audit(user['id'], 'USER_LOGIN', f"User {user['username']} logged in")

        return jsonify({
            'success': True,
            'token': token,
            'user': {
                'id': user['id'],
                'username': user['username'],
                'email': user['email'],
                'role': user['role']
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': f"Database error: {str(e)}"}), 500


@api_bp.route('/auth/me', methods=['GET'])
@token_required
def get_current_user():
    user = request.current_user
    return jsonify({
        'success': True,
        'user': {
            'id': user['id'],
            'username': user['username'],
            'email': user['email'],
            'role': user['role'],
            'created_at': str(user.get('created_at', ''))
        }
    })


# ==========================================
# 2. WORKSPACE MANAGEMENT ENDPOINTS
# ==========================================

@api_bp.route('/workspaces', methods=['GET'])
@token_required
def list_workspaces():
    try:
        user_id = request.current_user['id']
        workspaces = db.get_user_workspaces(user_id)
        return jsonify({'success': True, 'workspaces': workspaces})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@api_bp.route('/workspaces', methods=['POST'])
@token_required
def create_workspace():
    data = request.get_json() or {}
    name = data.get('name', '').strip()
    description = data.get('description', '').strip()
    icon_color = data.get('icon_color', '#FACC15')

    if not name:
        return jsonify({'success': False, 'error': 'Workspace name is required'}), 400

    try:
        user_id = request.current_user['id']
        ws_id = db.create_workspace(user_id, name, description, icon_color)
        ws = db.get_workspace(ws_id, user_id)
        db.log_audit(user_id, 'CREATE_WORKSPACE', f"Created workspace: {name}")
        return jsonify({'success': True, 'workspace': ws}), 201
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@api_bp.route('/workspaces/<int:workspace_id>', methods=['GET'])
@token_required
def get_workspace_detail(workspace_id):
    try:
        user_id = request.current_user['id']
        ws = db.get_workspace(workspace_id, user_id)
        if not ws:
            return jsonify({'success': False, 'error': 'Workspace not found'}), 404
        return jsonify({'success': True, 'workspace': ws})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@api_bp.route('/workspaces/<int:workspace_id>', methods=['PUT'])
@token_required
def update_workspace_detail(workspace_id):
    data = request.get_json() or {}
    name = data.get('name')
    description = data.get('description')
    icon_color = data.get('icon_color')

    try:
        user_id = request.current_user['id']
        updated = db.update_workspace(workspace_id, user_id, name, description, icon_color)
        if not updated:
            return jsonify({'success': False, 'error': 'Workspace not found or unauthorized'}), 404
        ws = db.get_workspace(workspace_id, user_id)
        return jsonify({'success': True, 'workspace': ws})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@api_bp.route('/workspaces/<int:workspace_id>', methods=['DELETE'])
@token_required
def remove_workspace(workspace_id):
    try:
        user_id = request.current_user['id']
        ws = db.get_workspace(workspace_id, user_id)
        if not ws:
            return jsonify({'success': False, 'error': 'Workspace not found'}), 404
        if ws.get('is_default'):
            return jsonify({'success': False, 'error': 'Cannot delete default workspace'}), 400

        db.delete_workspace(workspace_id, user_id)
        db.log_audit(user_id, 'DELETE_WORKSPACE', f"Deleted workspace ID: {workspace_id}")
        return jsonify({'success': True, 'message': 'Workspace deleted'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ==========================================
# 3. CONVERSATION & CHAT ENDPOINTS
# ==========================================

@api_bp.route('/workspaces/<int:workspace_id>/conversations', methods=['GET'])
@token_required
def list_workspace_conversations(workspace_id):
    try:
        user_id = request.current_user['id']
        conversations = db.get_workspace_conversations(workspace_id, user_id)
        return jsonify({'success': True, 'conversations': conversations})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@api_bp.route('/workspaces/<int:workspace_id>/conversations', methods=['POST'])
@token_required
def create_new_conversation(workspace_id):
    data = request.get_json() or {}
    conversation_id = data.get('conversation_id')
    title = data.get('title', 'New Conversation')
    model = data.get('model', 'mistral:latest')
    jailbreak = data.get('jailbreak', 'default')

    if not conversation_id:
        import secrets
        conversation_id = secrets.token_hex(16)

    try:
        user_id = request.current_user['id']
        db.create_conversation(conversation_id, workspace_id, user_id, title, model, jailbreak)
        conv = db.get_conversation(conversation_id, user_id)
        return jsonify({'success': True, 'conversation': conv}), 201
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@api_bp.route('/conversations/<string:conversation_id>', methods=['GET'])
@token_required
def get_conversation_history(conversation_id):
    try:
        user_id = request.current_user['id']
        conv = db.get_conversation(conversation_id, user_id)
        if not conv:
            return jsonify({'success': False, 'error': 'Conversation not found'}), 404

        messages = db.get_conversation_messages(conversation_id)
        return jsonify({
            'success': True,
            'conversation': conv,
            'messages': messages
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@api_bp.route('/conversations/<string:conversation_id>', methods=['PATCH'])
@token_required
def update_conversation(conversation_id):
    data = request.get_json() or {}
    title = data.get('title', '').strip()
    if not title:
        return jsonify({'success': False, 'error': 'Title is required'}), 400

    try:
        user_id = request.current_user['id']
        db.update_conversation_title(conversation_id, user_id, title)
        return jsonify({'success': True, 'title': title})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@api_bp.route('/conversations/<string:conversation_id>', methods=['DELETE'])
@token_required
def delete_conversation_route(conversation_id):
    try:
        user_id = request.current_user['id']
        db.delete_conversation(conversation_id, user_id)
        return jsonify({'success': True, 'message': 'Conversation deleted'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ==========================================
# 4. ADMIN MANAGEMENT ENDPOINTS
# ==========================================

@api_bp.route('/admin/stats', methods=['GET'])
@admin_required
def admin_stats():
    try:
        stats = db.get_admin_stats()
        return jsonify({'success': True, 'stats': stats})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@api_bp.route('/admin/users', methods=['GET'])
@admin_required
def admin_users():
    try:
        users = db.get_all_users()
        return jsonify({'success': True, 'users': users})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@api_bp.route('/admin/users/<int:target_user_id>/role', methods=['PATCH'])
@admin_required
def admin_change_role(target_user_id):
    data = request.get_json() or {}
    new_role = data.get('role')
    if new_role not in ['admin', 'user']:
        return jsonify({'success': False, 'error': 'Invalid role. Must be admin or user'}), 400

    try:
        admin_id = request.current_user['id']
        if admin_id == target_user_id and new_role != 'admin':
            return jsonify({'success': False, 'error': 'Cannot demote yourself'}), 400

        db.update_user_role(target_user_id, new_role)
        db.log_audit(admin_id, 'CHANGE_USER_ROLE', f"Changed user {target_user_id} role to {new_role}")
        return jsonify({'success': True, 'message': f"Role updated to {new_role}"})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@api_bp.route('/admin/users/<int:target_user_id>', methods=['DELETE'])
@admin_required
def admin_delete_user(target_user_id):
    try:
        admin_id = request.current_user['id']
        if admin_id == target_user_id:
            return jsonify({'success': False, 'error': 'Cannot delete your own admin account'}), 400

        db.delete_user(target_user_id)
        db.log_audit(admin_id, 'DELETE_USER', f"Deleted user ID: {target_user_id}")
        return jsonify({'success': True, 'message': 'User deleted successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@api_bp.route('/admin/workspaces', methods=['GET'])
@admin_required
def admin_all_workspaces():
    try:
        workspaces = db.get_admin_workspaces()
        return jsonify({'success': True, 'workspaces': workspaces})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@api_bp.route('/db/status', methods=['GET'])
def get_db_status():
    """Endpoint to check database connection status."""
    try:
        conn = db.get_connection()
        conn.close()
        return jsonify({'connected': True, 'type': 'MySQL', 'message': 'MySQL connected successfully'})
    except Exception as e:
        return jsonify({'connected': False, 'type': 'MySQL', 'error': str(e)}), 200
