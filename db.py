import json
import os
from werkzeug.security import generate_password_hash

try:
    import pymysql
    from pymysql.cursors import DictCursor
    HAS_PYMYSQL = True
except ImportError:
    pymysql = None
    DictCursor = None
    HAS_PYMYSQL = False

# Path to config.json
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')


def get_db_config():
    config = {
        'host': os.getenv('MYSQL_HOST', 'localhost'),
        'port': int(os.getenv('MYSQL_PORT', 3306)),
        'user': os.getenv('MYSQL_USER', 'root'),
        'password': os.getenv('MYSQL_PASSWORD', ''),
        'database': os.getenv('MYSQL_DATABASE', 'sahaj_ai'),
        'charset': 'utf8mb4'
    }
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, 'r') as f:
                cfg = json.load(f)
                if 'db_config' in cfg:
                    config.update(cfg['db_config'])
                    if 'port' in config:
                        config['port'] = int(config['port'])
        except Exception as e:
            print(f"[DB] Error loading config.json: {e}")
    return config


def get_connection(include_db=True):
    if not HAS_PYMYSQL or pymysql is None:
        raise RuntimeError("No module named 'pymysql'")
    cfg = get_db_config()
    connect_args = {
        'host': cfg['host'],
        'port': cfg['port'],
        'user': cfg['user'],
        'password': cfg['password'],
        'charset': cfg.get('charset', 'utf8mb4'),
        'cursorclass': DictCursor,
        'autocommit': True
    }
    if include_db:
        connect_args['database'] = cfg['database']
    return pymysql.connect(**connect_args)



def init_db():
    """Initializes MySQL database, schema, and seed accounts."""
    cfg = get_db_config()
    db_name = cfg['database']
    try:
        # Step 1: Ensure database exists
        server_conn = get_connection(include_db=False)
        with server_conn.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")
        server_conn.close()

        # Step 2: Create tables
        conn = get_connection(include_db=True)
        with conn.cursor() as cursor:
            # Users
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(100) NOT NULL UNIQUE,
                    email VARCHAR(191) NOT NULL UNIQUE,
                    password_hash VARCHAR(255) NULL,
                    auth_provider VARCHAR(50) DEFAULT 'local',
                    role ENUM('admin', 'user') DEFAULT 'user',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)

            # Ensure existing table has nullable password_hash and auth_provider column
            try:
                cursor.execute("ALTER TABLE users MODIFY COLUMN password_hash VARCHAR(255) NULL;")
            except Exception:
                pass
            try:
                cursor.execute("ALTER TABLE users ADD COLUMN auth_provider VARCHAR(50) DEFAULT 'local';")
            except Exception:
                pass

            # Workspaces
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS workspaces (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    name VARCHAR(150) NOT NULL,
                    description TEXT,
                    icon_color VARCHAR(30) DEFAULT '#FACC15',
                    is_default BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)

            # Conversations
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id VARCHAR(64) PRIMARY KEY,
                    workspace_id INT NOT NULL,
                    user_id INT NOT NULL,
                    title VARCHAR(255) NOT NULL DEFAULT 'New Conversation',
                    model VARCHAR(100) DEFAULT 'mistral:latest',
                    jailbreak VARCHAR(50) DEFAULT 'default',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)

            # Messages
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    conversation_id VARCHAR(64) NOT NULL,
                    role ENUM('user', 'assistant', 'system') NOT NULL,
                    content MEDIUMTEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)

            # Audit Logs
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NULL,
                    action VARCHAR(100) NOT NULL,
                    details TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)

            # Files
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS files (
                    id VARCHAR(64) PRIMARY KEY,
                    user_id INT NOT NULL,
                    workspace_id INT DEFAULT NULL,
                    original_name VARCHAR(255) NOT NULL,
                    storage_path VARCHAR(512) NOT NULL,
                    mime_type VARCHAR(100) NOT NULL,
                    file_size INT NOT NULL,
                    status ENUM('uploading', 'processing', 'ready', 'error') DEFAULT 'uploading',
                    error_message TEXT DEFAULT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)

            # Conversation Files
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversation_files (
                    conversation_id VARCHAR(64) NOT NULL,
                    file_id VARCHAR(64) NOT NULL,
                    attached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (conversation_id, file_id),
                    FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)

            # File Chunks
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS file_chunks (
                    id VARCHAR(64) PRIMARY KEY,
                    file_id VARCHAR(64) NOT NULL,
                    chunk_index INT NOT NULL,
                    content MEDIUMTEXT NOT NULL,
                    page_number INT DEFAULT NULL,
                    sheet_name VARCHAR(100) DEFAULT NULL,
                    start_line INT DEFAULT NULL,
                    end_line INT DEFAULT NULL,
                    embedding JSON DEFAULT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE,
                    INDEX idx_file_id (file_id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)

            # System Configs Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_configs (
                    key_name VARCHAR(100) PRIMARY KEY,
                    key_value TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)

            # LLM Providers Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS llm_providers (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    name VARCHAR(150) NOT NULL,
                    provider_type VARCHAR(50) NOT NULL,
                    model_name VARCHAR(150) NOT NULL,
                    endpoint VARCHAR(512) DEFAULT NULL,
                    api_key VARCHAR(512) DEFAULT NULL,
                    timeout INT DEFAULT 600,
                    is_active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)

            # Seed system configs if empty
            cursor.execute("SELECT COUNT(*) AS count FROM system_configs;")
            cfg_count = cursor.fetchone()['count']
            if cfg_count == 0:
                default_configs = [
                    ('llm_api_url', 'http://122.163.121.176:3041/v1/chat/completions'),
                    ('default_model', 'mistral:latest'),
                    ('secondary_model', 'qwen2.5-coder:1.5b'),
                    ('streaming_protocol', 'Server-Sent Events (SSE)')
                ]
                cursor.executemany(
                    "INSERT IGNORE INTO system_configs (key_name, key_value) VALUES (%s, %s)",
                    default_configs
                )

            # Clean and sync llm_providers table (keep only actual models mistral:latest and qwen2.5-coder:1.5b)
            cursor.execute("DELETE FROM llm_providers WHERE model_name NOT IN ('mistral:latest', 'qwen2.5-coder:1.5b');")

            # Check and insert mistral:latest
            cursor.execute("SELECT id FROM llm_providers WHERE model_name = 'mistral:latest';")
            row_mistral = cursor.fetchone()
            if not row_mistral:
                cursor.execute(
                    "INSERT INTO llm_providers (name, provider_type, model_name, endpoint, api_key, timeout, is_active) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    ("Mistral Local", "Mistral", "mistral:latest", "http://122.163.121.176:3041/v1/chat/completions", "NA", 600, True)
                )
            else:
                cursor.execute(
                    "UPDATE llm_providers SET endpoint = %s, provider_type = %s, name = %s WHERE model_name = %s",
                    ("http://122.163.121.176:3041/v1/chat/completions", "Mistral", "Mistral Local", "mistral:latest")
                )

            # Check and insert qwen2.5-coder:1.5b
            cursor.execute("SELECT id FROM llm_providers WHERE model_name = 'qwen2.5-coder:1.5b';")
            row_qwen = cursor.fetchone()
            if not row_qwen:
                cursor.execute(
                    "INSERT INTO llm_providers (name, provider_type, model_name, endpoint, api_key, timeout, is_active) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    ("Qwen Coder Local", "Qwen", "qwen2.5-coder:1.5b", "http://122.163.121.176:3041/v1/chat/completions", "NA", 600, True)
                )
            else:
                cursor.execute(
                    "UPDATE llm_providers SET endpoint = %s, provider_type = %s, name = %s WHERE model_name = %s",
                    ("http://122.163.121.176:3041/v1/chat/completions", "Qwen", "Qwen Coder Local", "qwen2.5-coder:1.5b")
                )


            # Step 3: Seed initial users if empty
            cursor.execute("SELECT COUNT(*) AS count FROM users;")
            user_count = cursor.fetchone()['count']
            if user_count == 0:
                print("[DB] Seeding default users (admin and user)...")
                # Admin account
                admin_pw = generate_password_hash("admin123")
                cursor.execute(
                    "INSERT INTO users (username, email, password_hash, role) VALUES (%s, %s, %s, %s)",
                    ("admin", "admin@sahaj.ai", admin_pw, "admin")
                )
                admin_id = cursor.lastrowid
                cursor.execute(
                    "INSERT INTO workspaces (user_id, name, description, is_default, icon_color) VALUES (%s, %s, %s, %s, %s)",
                    (admin_id, "Admin Workspace", "Primary administrative workspace", True, "#FACC15")
                )

                # Regular user account
                user_pw = generate_password_hash("user123")
                cursor.execute(
                    "INSERT INTO users (username, email, password_hash, role) VALUES (%s, %s, %s, %s)",
                    ("sahaj_user", "user@sahaj.ai", user_pw, "user")
                )
                regular_user_id = cursor.lastrowid
                cursor.execute(
                    "INSERT INTO workspaces (user_id, name, description, is_default, icon_color) VALUES (%s, %s, %s, %s, %s)",
                    (regular_user_id, "Sahaj Workspace", "General workspace for daily inquiries", True, "#FACC15")
                )

        conn.close()
        print(f"[DB] MySQL database `{db_name}` initialized successfully.")
        return True
    except Exception as e:
        print(f"[DB] MySQL connection/initialization warning: {e}")
        return False


# --- User Operations ---
def get_user_by_id(user_id):
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT id, username, email, role, auth_provider, created_at FROM users WHERE id = %s", (user_id,))
            return cursor.fetchone()
    finally:
        conn.close()


def get_user_by_username_or_email(identifier):
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM users WHERE username = %s OR email = %s", (identifier, identifier))
            return cursor.fetchone()
    finally:
        conn.close()


def create_user(username, email, password_hash=None, role='user', auth_provider='local'):
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO users (username, email, password_hash, auth_provider, role) VALUES (%s, %s, %s, %s, %s)",
                (username, email, password_hash, auth_provider, role)
            )
            user_id = cursor.lastrowid
            # Automatically create a default workspace
            cursor.execute(
                "INSERT INTO workspaces (user_id, name, description, is_default, icon_color) VALUES (%s, %s, %s, %s, %s)",
                (user_id, "Sahaj Workspace", "Personal AI Workspace", True, "#FACC15")
            )
            return user_id
    finally:
        conn.close()


def get_all_users():
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT u.id, u.username, u.email, u.role, u.auth_provider, u.created_at,
                       COUNT(DISTINCT w.id) AS workspace_count,
                       COUNT(DISTINCT c.id) AS conversation_count
                FROM users u
                LEFT JOIN workspaces w ON w.user_id = u.id
                LEFT JOIN conversations c ON c.user_id = u.id
                GROUP BY u.id
                ORDER BY u.created_at DESC
            """)
            return cursor.fetchall()
    finally:
        conn.close()


def update_user_role(user_id, new_role):
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("UPDATE users SET role = %s WHERE id = %s", (new_role, user_id))
            return cursor.rowcount > 0
    finally:
        conn.close()


def delete_user(user_id):
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("DELETE FROM users WHERE id = %s", (user_id,))
            return cursor.rowcount > 0
    finally:
        conn.close()


# --- Workspace Operations ---
def get_user_workspaces(user_id):
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT w.*, COUNT(c.id) AS conversation_count
                FROM workspaces w
                LEFT JOIN conversations c ON c.workspace_id = w.id
                WHERE w.user_id = %s
                GROUP BY w.id
                ORDER BY w.is_default DESC, w.created_at ASC
            """, (user_id,))
            return cursor.fetchall()
    finally:
        conn.close()


def create_workspace(user_id, name, description="", icon_color="#FACC15", is_default=False):
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            if is_default:
                cursor.execute("UPDATE workspaces SET is_default = FALSE WHERE user_id = %s", (user_id,))
            cursor.execute(
                "INSERT INTO workspaces (user_id, name, description, icon_color, is_default) VALUES (%s, %s, %s, %s, %s)",
                (user_id, name, description, icon_color, is_default)
            )
            return cursor.lastrowid
    finally:
        conn.close()


def get_workspace(workspace_id, user_id=None):
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            if user_id:
                cursor.execute("SELECT * FROM workspaces WHERE id = %s AND user_id = %s", (workspace_id, user_id))
            else:
                cursor.execute("SELECT * FROM workspaces WHERE id = %s", (workspace_id,))
            return cursor.fetchone()
    finally:
        conn.close()


def update_workspace(workspace_id, user_id, name, description=None, icon_color=None):
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            updates = []
            params = []
            if name is not None:
                updates.append("name = %s")
                params.append(name)
            if description is not None:
                updates.append("description = %s")
                params.append(description)
            if icon_color is not None:
                updates.append("icon_color = %s")
                params.append(icon_color)
            if not updates:
                return True
            params.extend([workspace_id, user_id])
            query = f"UPDATE workspaces SET {', '.join(updates)} WHERE id = %s AND user_id = %s"
            cursor.execute(query, params)
            return cursor.rowcount > 0
    finally:
        conn.close()


def delete_workspace(workspace_id, user_id):
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("DELETE FROM workspaces WHERE id = %s AND user_id = %s", (workspace_id, user_id))
            return cursor.rowcount > 0
    finally:
        conn.close()


# --- Conversation Operations ---
def get_workspace_conversations(workspace_id, user_id):
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT c.*, 
                       (SELECT content FROM messages WHERE conversation_id = c.id ORDER BY id ASC LIMIT 1) AS first_message,
                       (SELECT COUNT(*) FROM messages WHERE conversation_id = c.id) AS message_count
                FROM conversations c
                WHERE c.workspace_id = %s AND c.user_id = %s
                ORDER BY c.updated_at DESC
            """, (workspace_id, user_id))
            return cursor.fetchall()
    finally:
        conn.close()


def get_conversation(conversation_id, user_id=None):
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            if user_id:
                cursor.execute("SELECT * FROM conversations WHERE id = %s AND user_id = %s", (conversation_id, user_id))
            else:
                cursor.execute("SELECT * FROM conversations WHERE id = %s", (conversation_id,))
            return cursor.fetchone()
    finally:
        conn.close()


def create_conversation(conversation_id, workspace_id, user_id, title="New Conversation", model="mistral:latest", jailbreak="default"):
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """INSERT INTO conversations (id, workspace_id, user_id, title, model, jailbreak)
                   VALUES (%s, %s, %s, %s, %s, %s)
                   ON DUPLICATE KEY UPDATE updated_at = CURRENT_TIMESTAMP""",
                (conversation_id, workspace_id, user_id, title, model, jailbreak)
            )
            return conversation_id
    finally:
        conn.close()


def update_conversation_title(conversation_id, user_id, title):
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("UPDATE conversations SET title = %s WHERE id = %s AND user_id = %s", (title, conversation_id, user_id))
            return cursor.rowcount > 0
    finally:
        conn.close()


def delete_conversation(conversation_id, user_id):
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("DELETE FROM conversations WHERE id = %s AND user_id = %s", (conversation_id, user_id))
            return cursor.rowcount > 0
    finally:
        conn.close()


# --- Message Operations ---
def add_message(conversation_id, role, content, workspace_id=None, user_id=None):
    if not conversation_id or not content:
        return None
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            # Check if conversation exists; if not, create it first
            cursor.execute("SELECT id FROM conversations WHERE id = %s", (conversation_id,))
            if not cursor.fetchone():
                # Resolve user_id
                uid = user_id
                if not uid:
                    cursor.execute("SELECT id FROM users LIMIT 1")
                    u_row = cursor.fetchone()
                    uid = u_row['id'] if u_row else 1

                # Resolve workspace_id
                ws_id = workspace_id
                if not ws_id:
                    cursor.execute("SELECT id FROM workspaces WHERE user_id = %s LIMIT 1", (uid,))
                    ws_row = cursor.fetchone()
                    ws_id = ws_row['id'] if ws_row else 1

                title = (content[:32] + '...') if role == 'user' and len(content) > 32 else (content if role == 'user' else 'New Conversation')
                cursor.execute(
                    """INSERT INTO conversations (id, workspace_id, user_id, title)
                       VALUES (%s, %s, %s, %s)
                       ON DUPLICATE KEY UPDATE updated_at = CURRENT_TIMESTAMP""",
                    (conversation_id, ws_id, uid, title)
                )

            cursor.execute(
                "INSERT INTO messages (conversation_id, role, content) VALUES (%s, %s, %s)",
                (conversation_id, role, content)
            )
            cursor.execute("UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = %s", (conversation_id,))
            return cursor.lastrowid
    finally:
        conn.close()


def get_conversation_messages(conversation_id):
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT id, role, content, created_at FROM messages WHERE conversation_id = %s ORDER BY id ASC", (conversation_id,))
            return cursor.fetchall()
    finally:
        conn.close()


# --- Admin & Analytics Operations ---
def get_admin_stats():
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) AS total_users FROM users;")
            total_users = cursor.fetchone()['total_users']

            cursor.execute("SELECT COUNT(*) AS total_workspaces FROM workspaces;")
            total_workspaces = cursor.fetchone()['total_workspaces']

            cursor.execute("SELECT COUNT(*) AS total_conversations FROM conversations;")
            total_conversations = cursor.fetchone()['total_conversations']

            cursor.execute("SELECT COUNT(*) AS total_messages FROM messages;")
            total_messages = cursor.fetchone()['total_messages']

            return {
                'total_users': total_users,
                'total_workspaces': total_workspaces,
                'total_conversations': total_conversations,
                'total_messages': total_messages,
                'database_type': 'MySQL'
            }
    finally:
        conn.close()


def get_admin_workspaces():
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT w.*, u.username, u.email, COUNT(c.id) AS conversation_count
                FROM workspaces w
                JOIN users u ON u.id = w.user_id
                LEFT JOIN conversations c ON c.workspace_id = w.id
                GROUP BY w.id
                ORDER BY w.created_at DESC
            """)
            return cursor.fetchall()
    finally:
        conn.close()


def log_audit(user_id, action, details=""):
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO audit_logs (user_id, action, details) VALUES (%s, %s, %s)",
                (user_id, action, details)
            )
        conn.close()
    except Exception as e:
        print(f"[AUDIT] Warning: failed to log audit: {e}")


# ==========================================
# FILE INTELLIGENCE DATABASE HELPERS
# ==========================================

_MEM_FILES = {}
_MEM_CONV_FILES = {}
_MEM_CHUNKS = {}


def create_file_record(file_id, user_id, workspace_id, original_name, storage_path, mime_type, file_size):
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO files (id, user_id, workspace_id, original_name, storage_path, mime_type, file_size, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, 'uploading')
            """, (file_id, user_id, workspace_id, original_name, storage_path, mime_type, file_size))
        conn.close()
    except Exception as e:
        print(f"[DB] Using memory fallback for file record: {e}")
        _MEM_FILES[file_id] = {
            'id': file_id,
            'user_id': user_id,
            'workspace_id': workspace_id,
            'original_name': original_name,
            'storage_path': storage_path,
            'mime_type': mime_type,
            'file_size': file_size,
            'status': 'uploading',
            'error_message': None
        }
    return file_id


def update_file_status(file_id, status, error_message=None):
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute("""
                UPDATE files SET status = %s, error_message = %s WHERE id = %s
            """, (status, error_message, file_id))
        conn.close()
    except Exception:
        if file_id in _MEM_FILES:
            _MEM_FILES[file_id]['status'] = status
            _MEM_FILES[file_id]['error_message'] = error_message


def link_file_to_conversation(conversation_id, file_id):
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute("""
                INSERT IGNORE INTO conversation_files (conversation_id, file_id) VALUES (%s, %s)
            """, (conversation_id, file_id))
        conn.close()
    except Exception:
        if conversation_id not in _MEM_CONV_FILES:
            _MEM_CONV_FILES[conversation_id] = []
        if file_id not in _MEM_CONV_FILES[conversation_id]:
            _MEM_CONV_FILES[conversation_id].append(file_id)


def get_file_by_id(file_id, user_id=None):
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            if user_id:
                cursor.execute("SELECT * FROM files WHERE id = %s AND user_id = %s", (file_id, user_id))
            else:
                cursor.execute("SELECT * FROM files WHERE id = %s", (file_id,))
            res = cursor.fetchone()
        conn.close()
        if res:
            return res
    except Exception:
        pass
    return _MEM_FILES.get(file_id)


def get_conversation_files(conversation_id, user_id=None):
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            if user_id:
                cursor.execute("""
                    SELECT f.* FROM files f
                    JOIN conversation_files cf ON f.id = cf.file_id
                    WHERE cf.conversation_id = %s AND f.user_id = %s
                    ORDER BY cf.attached_at ASC
                """, (conversation_id, user_id))
            else:
                cursor.execute("""
                    SELECT f.* FROM files f
                    JOIN conversation_files cf ON f.id = cf.file_id
                    WHERE cf.conversation_id = %s
                    ORDER BY cf.attached_at ASC
                """, (conversation_id,))
            res = cursor.fetchall()
        conn.close()
        if res:
            return res
    except Exception:
        pass

    linked_ids = _MEM_CONV_FILES.get(conversation_id, [])
    return [_MEM_FILES[fid] for fid in linked_ids if fid in _MEM_FILES]


def save_file_chunks(file_id, chunks_data):
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            for chunk in chunks_data:
                emb_json = json.dumps(chunk.get('embedding', []))
                cursor.execute("""
                    INSERT INTO file_chunks 
                    (id, file_id, chunk_index, content, page_number, sheet_name, start_line, end_line, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    chunk['id'],
                    file_id,
                    chunk['chunk_index'],
                    chunk['content'],
                    chunk.get('page_number'),
                    chunk.get('sheet_name'),
                    chunk.get('start_line'),
                    chunk.get('end_line'),
                    emb_json
                ))
        conn.close()
    except Exception as e:
        print(f"[DB] Using memory fallback for chunks: {e}")
        if file_id not in _MEM_CHUNKS:
            _MEM_CHUNKS[file_id] = []
        for chunk in chunks_data:
            c = dict(chunk)
            c['file_id'] = file_id
            _MEM_CHUNKS[file_id].append(c)


def get_chunks_by_file_ids(file_ids):
    if not file_ids:
        return []
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            format_strings = ','.join(['%s'] * len(file_ids))
            cursor.execute(f"""
                SELECT fc.*, f.original_name
                FROM file_chunks fc
                JOIN files f ON f.id = fc.file_id
                WHERE fc.file_id IN ({format_strings})
                ORDER BY fc.file_id, fc.chunk_index ASC
            """, tuple(file_ids))
            rows = cursor.fetchall()
            for r in rows:
                if isinstance(r.get('embedding'), str):
                    try:
                        r['embedding'] = json.loads(r['embedding'])
                    except Exception:
                        r['embedding'] = []
        conn.close()
        if rows:
            return rows
    except Exception:
        pass

    results = []
    for fid in file_ids:
        fname = _MEM_FILES.get(fid, {}).get('original_name', 'Attached File')
        for chunk in _MEM_CHUNKS.get(fid, []):
            c = dict(chunk)
            c['original_name'] = fname
            results.append(c)
    return results


def delete_file_record(file_id, user_id):
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute("SELECT storage_path FROM files WHERE id = %s AND user_id = %s", (file_id, user_id))
            file_rec = cursor.fetchone()
            if file_rec:
                cursor.execute("DELETE FROM files WHERE id = %s AND user_id = %s", (file_id, user_id))
                if file_rec.get('storage_path') and os.path.exists(file_rec['storage_path']):
                    try:
                        os.remove(file_rec['storage_path'])
                    except Exception:
                        pass
                conn.close()
                return True
        conn.close()
    except Exception:
        pass

    if file_id in _MEM_FILES:
        sp = _MEM_FILES[file_id].get('storage_path')
        if sp and os.path.exists(sp):
            try:
                os.remove(sp)
            except Exception:
                pass
        del _MEM_FILES[file_id]
        _MEM_CHUNKS.pop(file_id, None)
        return True

    return False


# --- System Configuration Operations ---
MEMORY_SYSTEM_CONFIGS = {
    'llm_api_url': 'http://122.163.121.176:3041/v1/chat/completions',
    'default_model': 'mistral:latest',
    'secondary_model': 'qwen2.5-coder:1.5b',
    'streaming_protocol': 'Server-Sent Events (SSE)'
}


def get_all_system_configs():
    configs = dict(MEMORY_SYSTEM_CONFIGS)
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute("SELECT key_name, key_value FROM system_configs")
            rows = cursor.fetchall()
            conn.close()
            for row in rows:
                configs[row['key_name']] = row['key_value']
            return configs
    except Exception as e:
        print(f"[DB] system_configs fetch note: {e}")
        return configs


def get_system_config(key_name, default_val=None):
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute("SELECT key_value FROM system_configs WHERE key_name = %s", (key_name,))
            row = cursor.fetchone()
            conn.close()
            if row and row.get('key_value'):
                return row['key_value']
    except Exception:
        pass
    return MEMORY_SYSTEM_CONFIGS.get(key_name, default_val)


def set_system_config(key_name, key_value):
    MEMORY_SYSTEM_CONFIGS[key_name] = key_value
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO system_configs (key_name, key_value)
                VALUES (%s, %s)
                ON DUPLICATE KEY UPDATE key_value = VALUES(key_value);
            """, (key_name, key_value))
        conn.close()
        return True
    except Exception as e:
        print(f"[DB] system_config set note: {e}")
        return False


# --- LLM Provider Management Operations ---
_MEM_LLM_PROVIDERS = [
    {
        "id": 1,
        "name": "Mistral Local",
        "provider_type": "Mistral",
        "model_name": "mistral:latest",
        "endpoint": "http://122.163.121.176:3041/v1/chat/completions",
        "api_key": "NA",
        "timeout": 600,
        "is_active": True
    },
    {
        "id": 2,
        "name": "Qwen Coder Local",
        "provider_type": "Qwen",
        "model_name": "qwen2.5-coder:1.5b",
        "endpoint": "http://122.163.121.176:3041/v1/chat/completions",
        "api_key": "NA",
        "timeout": 600,
        "is_active": True
    }
]


def get_all_llm_providers():
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute("SELECT id, name, provider_type, model_name, endpoint, api_key, timeout, is_active, created_at, updated_at FROM llm_providers ORDER BY id ASC")
            rows = cursor.fetchall()
            conn.close()
            if rows:
                for r in rows:
                    r['is_active'] = bool(r['is_active'])
                return rows
    except Exception as e:
        print(f"[DB] llm_providers fetch note: {e}")
    return list(_MEM_LLM_PROVIDERS)


def add_llm_provider(name, provider_type, model_name, endpoint="", api_key="", timeout=600, is_active=True):
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO llm_providers (name, provider_type, model_name, endpoint, api_key, timeout, is_active) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (name, provider_type, model_name, endpoint, api_key, timeout, is_active)
            )
            pid = cursor.lastrowid
            conn.close()
            return pid
    except Exception as e:
        print(f"[DB] add_llm_provider note: {e}")

    new_id = (max([p['id'] for p in _MEM_LLM_PROVIDERS] or [0])) + 1
    new_p = {
        "id": new_id,
        "name": name,
        "provider_type": provider_type,
        "model_name": model_name,
        "endpoint": endpoint,
        "api_key": api_key,
        "timeout": timeout,
        "is_active": is_active
    }
    _MEM_LLM_PROVIDERS.append(new_p)
    return new_id


def update_llm_provider(provider_id, name, provider_type, model_name, endpoint="", api_key="", timeout=600, is_active=True):
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            if api_key and not api_key.startswith("••••") and not api_key.startswith("mstr***") and not api_key.startswith("AQ.A***"):
                cursor.execute(
                    "UPDATE llm_providers SET name=%s, provider_type=%s, model_name=%s, endpoint=%s, api_key=%s, timeout=%s, is_active=%s WHERE id=%s",
                    (name, provider_type, model_name, endpoint, api_key, timeout, is_active, provider_id)
                )
            else:
                cursor.execute(
                    "UPDATE llm_providers SET name=%s, provider_type=%s, model_name=%s, endpoint=%s, timeout=%s, is_active=%s WHERE id=%s",
                    (name, provider_type, model_name, endpoint, timeout, is_active, provider_id)
                )
            conn.close()
            return True
    except Exception as e:
        print(f"[DB] update_llm_provider note: {e}")

    for p in _MEM_LLM_PROVIDERS:
        if p['id'] == int(provider_id):
            p['name'] = name
            p['provider_type'] = provider_type
            p['model_name'] = model_name
            p['endpoint'] = endpoint
            if api_key and not api_key.startswith("••••") and not api_key.startswith("mstr***") and not api_key.startswith("AQ.A***"):
                p['api_key'] = api_key
            p['timeout'] = timeout
            p['is_active'] = is_active
            return True
    return False


def delete_llm_provider(provider_id):
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute("DELETE FROM llm_providers WHERE id = %s", (provider_id,))
            conn.close()
            return True
    except Exception as e:
        print(f"[DB] delete_llm_provider note: {e}")

    global _MEM_LLM_PROVIDERS
    _MEM_LLM_PROVIDERS = [p for p in _MEM_LLM_PROVIDERS if p['id'] != int(provider_id)]
    return True


def toggle_llm_provider_status(provider_id, is_active):
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute("UPDATE llm_providers SET is_active = %s WHERE id = %s", (is_active, provider_id))
            conn.close()
            return True
    except Exception as e:
        print(f"[DB] toggle_llm_provider_status note: {e}")

    for p in _MEM_LLM_PROVIDERS:
        if p['id'] == int(provider_id):
            p['is_active'] = is_active
            return True
    return False



