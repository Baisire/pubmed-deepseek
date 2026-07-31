"""数据库模块 - 用户管理、用量记录、缓存、对话、API Key。

使用 SQLite 存储，Python 内置 sqlite3，无需额外依赖。
首次调用 init_db() 自动建表并创建默认管理员账号 (admin / admin123)。

v3.0 扩展：
- WAL 模式 + threading.Lock 并发安全
- mesh_cache / citation_cache 缓存表
- chat_sessions / chat_messages 对话持久化
- user_api_keys 用户自定义 API Key（AES-256 加密存储）
- users.tier 用户分层字段
"""

import hashlib
import os
import sqlite3
import threading
from datetime import datetime
from typing import Optional

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pubmed_users.db")

# 并发安全：WAL 模式 + 写锁
_write_lock = threading.Lock()
_wal_initialized = False


def _get_conn() -> sqlite3.Connection:
    global _wal_initialized
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    if not _wal_initialized:
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            _wal_initialized = True
        except sqlite3.Error:
            pass
    return conn


def _hash_password(password: str, salt: str) -> str:
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()


def _generate_salt() -> str:
    return os.urandom(16).hex()


def init_db() -> None:
    """初始化数据库：建表 + 创建默认管理员账号。"""
    with _write_lock:
        with _get_conn() as conn:
            # ---- users 表 ----
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    is_admin INTEGER DEFAULT 0,
                    is_active INTEGER DEFAULT 1,
                    tier TEXT DEFAULT 'free',
                    created_at TEXT DEFAULT (datetime('now', 'localtime'))
                )
                """
            )
            # 兼容旧库：如果没有 tier 字段则添加
            try:
                conn.execute("ALTER TABLE users ADD COLUMN tier TEXT DEFAULT 'free'")
            except sqlite3.OperationalError:
                pass  # 字段已存在

            # ---- usage_log 表 ----
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS usage_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    keyword TEXT NOT NULL,
                    analysis_type TEXT NOT NULL,
                    article_count INTEGER NOT NULL,
                    created_at TEXT DEFAULT (datetime('now', 'localtime')),
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
                """
            )

            # ---- mesh_cache 表（MeSH 术语缓存，永久有效）----
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS mesh_cache (
                    term TEXT PRIMARY KEY,
                    ui TEXT,
                    tree_numbers TEXT,
                    entry_terms TEXT,
                    subheadings TEXT,
                    cached_at TEXT DEFAULT (datetime('now', 'localtime'))
                )
                """
            )

            # ---- citation_cache 表（引文缓存，7 天 TTL）----
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS citation_cache (
                    pmid TEXT PRIMARY KEY,
                    citation_count INTEGER,
                    source TEXT,
                    cached_at TEXT DEFAULT (datetime('now', 'localtime'))
                )
                """
            )

            # ---- chat_sessions 表（对话会话）----
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    title TEXT DEFAULT '新对话',
                    model TEXT DEFAULT 'deepseek-chat',
                    created_at TEXT DEFAULT (datetime('now', 'localtime')),
                    updated_at TEXT DEFAULT (datetime('now', 'localtime')),
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
                """
            )

            # ---- chat_messages 表（对话消息）----
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    tool_calls TEXT,
                    tool_call_id TEXT,
                    created_at TEXT DEFAULT (datetime('now', 'localtime')),
                    FOREIGN KEY (session_id) REFERENCES chat_sessions(id)
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages(session_id)")

            # ---- user_api_keys 表（用户自定义 API Key，加密存储）----
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_api_keys (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    provider TEXT NOT NULL,
                    encrypted_key TEXT NOT NULL,
                    iv TEXT NOT NULL,
                    created_at TEXT DEFAULT (datetime('now', 'localtime')),
                    UNIQUE(user_id, provider),
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
                """
            )

            # 创建默认管理员（如果不存在）+ 确保管理员 tier 为 flagship
            admin = conn.execute(
                "SELECT id, tier FROM users WHERE username = ?", ("admin",)
            ).fetchone()
            if admin is None:
                salt = _generate_salt()
                password_hash = _hash_password("admin123", salt)
                conn.execute(
                    "INSERT INTO users (username, password_hash, salt, is_admin, tier) "
                    "VALUES (?, ?, ?, 1, 'flagship')",
                    ("admin", password_hash, salt),
                )
            elif not admin["tier"] or admin["tier"] == "free":
                # 旧库迁移：管理员 tier 升级为 flagship
                conn.execute(
                    "UPDATE users SET tier = 'flagship' WHERE username = 'admin'"
                )


def create_user(username: str, password: str, is_admin: bool = False) -> tuple[bool, str]:
    """创建新用户。返回 (成功与否, 消息)。"""
    if not username.strip() or not password.strip():
        return False, "用户名和密码不能为空"
    if len(password) < 6:
        return False, "密码长度至少 6 位"

    salt = _generate_salt()
    password_hash = _hash_password(password, salt)

    try:
        with _get_conn() as conn:
            conn.execute(
                "INSERT INTO users (username, password_hash, salt, is_admin) VALUES (?, ?, ?, ?)",
                (username.strip(), password_hash, salt, 1 if is_admin else 0),
            )
        return True, f"用户 {username} 创建成功"
    except sqlite3.IntegrityError:
        return False, f"用户名 {username} 已存在"


def verify_user(username: str, password: str) -> Optional[dict]:
    """验证用户登录。返回用户字典或 None。"""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()

    if row is None:
        return None

    password_hash = _hash_password(password, row["salt"])
    if password_hash != row["password_hash"]:
        return None

    if not row["is_active"]:
        return None

    return {
        "id": row["id"],
        "username": row["username"],
        "is_admin": bool(row["is_admin"]),
        "is_active": bool(row["is_active"]),
        "created_at": row["created_at"],
    }


def get_all_users() -> list[dict]:
    """获取所有用户列表（不含密码）。"""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT id, username, is_admin, is_active, created_at FROM users ORDER BY id"
        ).fetchall()
    return [dict(r) for r in rows]


def delete_user(user_id: int) -> None:
    """删除用户及其用量记录。"""
    with _get_conn() as conn:
        conn.execute("DELETE FROM usage_log WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))


def toggle_user_active(user_id: int, is_active: bool) -> None:
    """启用/禁用用户。"""
    with _get_conn() as conn:
        conn.execute(
            "UPDATE users SET is_active = ? WHERE id = ?",
            (1 if is_active else 0, user_id),
        )


def reset_password(user_id: int, new_password: str) -> tuple[bool, str]:
    """重置用户密码。"""
    if len(new_password) < 6:
        return False, "密码长度至少 6 位"

    salt = _generate_salt()
    password_hash = _hash_password(new_password, salt)

    with _get_conn() as conn:
        conn.execute(
            "UPDATE users SET password_hash = ?, salt = ? WHERE id = ?",
            (password_hash, salt, user_id),
        )
    return True, "密码重置成功"


def log_usage(
    user_id: int, keyword: str, analysis_type: str, article_count: int
) -> None:
    """记录用户的检索历史。"""
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO usage_log (user_id, keyword, analysis_type, article_count) VALUES (?, ?, ?, ?)",
            (user_id, keyword, analysis_type, article_count),
        )


def get_usage_by_user(user_id: int, limit: int = 50) -> list[dict]:
    """获取指定用户的检索历史。"""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT keyword, analysis_type, article_count, created_at "
            "FROM usage_log WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def get_all_usage_stats() -> list[dict]:
    """获取所有用户的用量统计（管理员用）。"""
    with _get_conn() as conn:
        rows = conn.execute(
            """
            SELECT u.id, u.username, u.is_admin, u.is_active,
                   COUNT(ul.id) as total_searches,
                   COALESCE(SUM(ul.article_count), 0) as total_articles,
                   MAX(ul.created_at) as last_search
            FROM users u
            LEFT JOIN usage_log ul ON u.id = ul.user_id
            GROUP BY u.id
            ORDER BY total_searches DESC
            """
        ).fetchall()
    return [dict(r) for r in rows]


def get_recent_usage(limit: int = 20) -> list[dict]:
    """获取最近的检索记录（管理员用）。"""
    with _get_conn() as conn:
        rows = conn.execute(
            """
            SELECT u.username, ul.keyword, ul.analysis_type, ul.article_count, ul.created_at
            FROM usage_log ul
            JOIN users u ON ul.user_id = u.id
            ORDER BY ul.created_at DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


# ============================================================================
# 用户分层 (tier) 管理
# ============================================================================

TIER_FREE = "free"
TIER_BASIC = "basic"
TIER_PRO = "pro"
TIER_FLAGSHIP = "flagship"
TIER_INSTITUTIONAL = "institutional"

ALL_TIERS = [TIER_FREE, TIER_BASIC, TIER_PRO, TIER_FLAGSHIP, TIER_INSTITUTIONAL]


def get_user_tier(user_id: int) -> str:
    """获取用户的 tier 等级。"""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT tier FROM users WHERE id = ?", (user_id,)
        ).fetchone()
    return row["tier"] if row and row["tier"] else TIER_FREE


def set_user_tier(user_id: int, tier: str) -> tuple[bool, str]:
    """设置用户的 tier 等级。"""
    if tier not in ALL_TIERS:
        return False, f"无效的 tier: {tier}"
    with _write_lock:
        with _get_conn() as conn:
            conn.execute(
                "UPDATE users SET tier = ? WHERE id = ?",
                (tier, user_id),
            )
    return True, f"用户 tier 已更新为 {tier}"


def get_daily_usage_count(user_id: int) -> int:
    """获取用户今日检索次数（用于免费版限额）。"""
    with _get_conn() as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) as cnt FROM usage_log
            WHERE user_id = ? AND DATE(created_at) = DATE('now', 'localtime')
            """,
            (user_id,),
        ).fetchone()
    return row["cnt"] if row else 0


# ============================================================================
# 对话会话管理
# ============================================================================

import uuid as _uuid


def create_chat_session(user_id: int, model: str = "deepseek-chat",
                        title: str = "新对话") -> str:
    """创建新的对话会话，返回 session_id。"""
    session_id = _uuid.uuid4().hex[:16]
    with _write_lock:
        with _get_conn() as conn:
            conn.execute(
                "INSERT INTO chat_sessions (id, user_id, title, model) "
                "VALUES (?, ?, ?, ?)",
                (session_id, user_id, title, model),
            )
    return session_id


def list_chat_sessions(user_id: int, limit: int = 50) -> list[dict]:
    """获取用户的对话会话列表（按更新时间倒序）。"""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT id, title, model, created_at, updated_at "
            "FROM chat_sessions WHERE user_id = ? "
            "ORDER BY updated_at DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def get_chat_session(session_id: str) -> Optional[dict]:
    """获取单个会话信息。"""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM chat_sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
    return dict(row) if row else None


def update_chat_session_title(session_id: str, title: str) -> None:
    """更新会话标题。"""
    with _write_lock:
        with _get_conn() as conn:
            conn.execute(
                "UPDATE chat_sessions SET title = ?, "
                "updated_at = datetime('now', 'localtime') WHERE id = ?",
                (title, session_id),
            )


def delete_chat_session(session_id: str) -> None:
    """删除对话会话及其消息。"""
    with _write_lock:
        with _get_conn() as conn:
            conn.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))


def add_chat_message(session_id: str, role: str, content: str,
                     tool_calls: Optional[str] = None,
                     tool_call_id: Optional[str] = None) -> int:
    """添加一条对话消息，返回消息 ID。"""
    with _write_lock:
        with _get_conn() as conn:
            cursor = conn.execute(
                "INSERT INTO chat_messages (session_id, role, content, tool_calls, tool_call_id) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, role, content, tool_calls, tool_call_id),
            )
            # 同时更新会话的 updated_at
            conn.execute(
                "UPDATE chat_sessions SET updated_at = datetime('now', 'localtime') "
                "WHERE id = ?",
                (session_id,),
            )
            msg_id = cursor.lastrowid
    return msg_id


def get_chat_messages(session_id: str, limit: int = 100) -> list[dict]:
    """获取会话的消息列表（按时间正序）。"""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT id, role, content, tool_calls, tool_call_id, created_at "
            "FROM chat_messages WHERE session_id = ? "
            "ORDER BY id ASC LIMIT ?",
            (session_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


# ============================================================================
# 用户 API Key 管理（AES-256 加密存储）
# ============================================================================

import base64 as _base64
import hashlib as _hashlib

_CRYPTO_AVAILABLE = False
try:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives import padding
    _CRYPTO_AVAILABLE = True
except ImportError:
    pass


def _get_key_encryption_key() -> bytes:
    """从数据库位置派生加密密钥（应用级密钥）。"""
    # 注：生产环境应使用独立的密钥管理，这里用路径+固定盐派生
    salt = "pubmed_app_v3_key_secret"
    return _hashlib.sha256((salt + DB_PATH).encode("utf-8")).digest()


def _encrypt_api_key(plain_key: str) -> tuple[str, str]:
    """加密 API Key，返回 (encrypted_b64, iv_b64)。"""
    if not _CRYPTO_AVAILABLE:
        # 降级：简单混淆（不推荐，仅用于无 cryptography 库时）
        key_bytes = plain_key.encode("utf-8")
        return _base64.b64encode(key_bytes).decode(), "none"
    key = _get_key_encryption_key()
    iv = os.urandom(16)
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
    padder = padding.PKCS7(128).padder()
    padded = padder.update(plain_key.encode("utf-8")) + padder.finalize()
    encryptor = cipher.encryptor()
    ct = encryptor.update(padded) + encryptor.finalize()
    return _base64.b64encode(ct).decode(), _base64.b64encode(iv).decode()


def _decrypt_api_key(encrypted_b64: str, iv_b64: str) -> str:
    """解密 API Key。"""
    if iv_b64 == "none" or not _CRYPTO_AVAILABLE:
        return _base64.b64decode(encrypted_b64).decode("utf-8")
    key = _get_key_encryption_key()
    iv = _base64.b64decode(iv_b64)
    ct = _base64.b64decode(encrypted_b64)
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
    decryptor = cipher.decryptor()
    padded = decryptor.update(ct) + decryptor.finalize()
    unpadder = padding.PKCS7(128).unpadder()
    data = unpadder.update(padded) + unpadder.finalize()
    return data.decode("utf-8")


def set_user_api_key(user_id: int, provider: str, api_key: str) -> None:
    """保存用户的 API Key（加密存储）。"""
    encrypted, iv = _encrypt_api_key(api_key)
    with _write_lock:
        with _get_conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO user_api_keys (user_id, provider, encrypted_key, iv) "
                "VALUES (?, ?, ?, ?)",
                (user_id, provider, encrypted, iv),
            )


def get_user_api_key(user_id: int, provider: str) -> Optional[str]:
    """获取用户的 API Key（解密后）。未设置返回 None。"""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT encrypted_key, iv FROM user_api_keys "
            "WHERE user_id = ? AND provider = ?",
            (user_id, provider),
        ).fetchone()
    if not row:
        return None
    try:
        return _decrypt_api_key(row["encrypted_key"], row["iv"])
    except Exception:
        return None


def delete_user_api_key(user_id: int, provider: str) -> None:
    """删除用户的 API Key。"""
    with _write_lock:
        with _get_conn() as conn:
            conn.execute(
                "DELETE FROM user_api_keys WHERE user_id = ? AND provider = ?",
                (user_id, provider),
            )
