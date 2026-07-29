"""数据库模块 - 用户管理与用量记录

使用 SQLite 存储，Python 内置 sqlite3，无需额外依赖。
首次调用 init_db() 自动建表并创建默认管理员账号 (admin / admin123)。
"""

import hashlib
import os
import sqlite3
from datetime import datetime
from typing import Optional

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pubmed_users.db")


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _hash_password(password: str, salt: str) -> str:
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()


def _generate_salt() -> str:
    return os.urandom(16).hex()


def init_db() -> None:
    """初始化数据库：建表 + 创建默认管理员账号。"""
    with _get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                is_admin INTEGER DEFAULT 0,
                is_active INTEGER DEFAULT 1,
                created_at TEXT DEFAULT (datetime('now', 'localtime'))
            )
            """
        )
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

        # 创建默认管理员（如果不存在）
        admin = conn.execute(
            "SELECT id FROM users WHERE username = ?", ("admin",)
        ).fetchone()
        if admin is None:
            salt = _generate_salt()
            password_hash = _hash_password("admin123", salt)
            conn.execute(
                "INSERT INTO users (username, password_hash, salt, is_admin) VALUES (?, ?, ?, 1)",
                ("admin", password_hash, salt),
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
