"""M2 验证脚本：db.py 扩展验证。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import db


def main():
    # 用临时库测试
    test_db = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_m2.db")
    if os.path.exists(test_db):
        os.remove(test_db)
    db.DB_PATH = test_db

    print("1. init_db()...")
    db.init_db()
    print("   OK")

    print("2. 检查表...")
    conn = db._get_conn()
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    tables = [r[0] for r in rows]
    print(f"   {tables}")
    expected = [
        "chat_messages", "chat_sessions", "citation_cache",
        "mesh_cache", "usage_log", "user_api_keys", "users",
    ]
    missing = [t for t in expected if t not in tables]
    assert not missing, f"缺失表: {missing}"
    print("   OK (7 张表)")

    print("3. WAL 模式...")
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    print(f"   journal_mode = {mode}")
    assert mode == "wal"
    print("   OK")

    print("4. users.tier 字段...")
    cols = [r[1] for r in conn.execute("PRAGMA table_info(users)").fetchall()]
    assert "tier" in cols
    print("   OK")

    print("5. 默认管理员 tier=flagship...")
    admin = conn.execute(
        "SELECT username, tier, is_admin FROM users WHERE username='admin'"
    ).fetchone()
    assert admin["tier"] == "flagship"
    assert admin["is_admin"] == 1
    print("   OK")
    conn.close()

    print("6. 并发写入测试 (5 线程 x 10 条)...")
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def worker(i):
        db.set_user_api_key(1, f"provider_{i}", f"key_value_{i}")
        return i

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(worker, i) for i in range(10)]
        for f in as_completed(futures):
            f.result()
    print("   OK (无 database is locked)")

    print("7. API Key 加解密...")
    db.set_user_api_key(1, "deepseek", "sk-test-12345-secret")
    decrypted = db.get_user_api_key(1, "deepseek")
    assert decrypted == "sk-test-12345-secret", f"got: {decrypted}"
    print("   OK")

    print("8. 对话会话 CRUD...")
    sid = db.create_chat_session(1, "deepseek-chat", "测试对话")
    assert sid and len(sid) > 0
    db.add_chat_message(sid, "user", "你好")
    db.add_chat_message(sid, "assistant", "你好！我是医学文献助手。")
    msgs = db.get_chat_messages(sid)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    sessions = db.list_chat_sessions(1)
    assert len(sessions) == 1
    assert sessions[0]["title"] == "测试对话"
    db.update_chat_session_title(sid, "新标题")
    s = db.get_chat_session(sid)
    assert s["title"] == "新标题"
    db.delete_chat_session(sid)
    assert db.get_chat_session(sid) is None
    assert db.get_chat_messages(sid) == []
    print("   OK")

    print("9. tier 管理...")
    ok, msg = db.set_user_tier(1, "pro")
    assert ok
    assert db.get_user_tier(1) == "pro"
    ok, msg = db.set_user_tier(1, "invalid_tier")
    assert not ok
    print("   OK")

    print("10. 每日用量统计...")
    db.log_usage(1, "test", "summary", 5)
    count = db.get_daily_usage_count(1)
    assert count >= 1
    print("   OK")

    # 清理
    if os.path.exists(test_db):
        os.remove(test_db)

    print()
    print("=== M2 数据库扩展验证全部通过 ===")


if __name__ == "__main__":
    main()
