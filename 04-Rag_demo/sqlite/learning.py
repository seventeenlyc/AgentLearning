import sqlite3
from uuid import uuid4

# 连接数据库；文件不存在时会自动创建
conn = sqlite3.connect("demo_session.db")

# 查询结果可以按列名访问，例如 row["title"]
conn.row_factory=sqlite3.Row

try:
    #建表
    conn.execute("""
    CREATE TABLE IF NOT EXISTS demo_session (
    id TEXT PRIMARY KEY NOT NULL,
    title TEXT NOT NULL,
    location TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )""")

    session_id = str(uuid4())

    with conn:
        #增
        conn.execute("""
        INSERT INTO demo_session (id, title, location) VALUES (?, ?, ?)""",
                     (session_id,"学习SQlite","深圳大学"))

        #查
        def search():
            rows = conn.execute("""
                                SELECT id, title, location, created_at
                                FROM demo_session
                                ORDER BY created_at DESC, id DESC
                                """).fetchall()
            for row in rows:
                print(row["id"], row["title"], row["location"], row["created_at"])

        search()
        print("-"*20)
        #改
        conn.execute("""
        UPDATE demo_session
        SET location = ?
        WHERE id = ?
        """,("南方科技大学",session_id))
        search()
        print("-" * 20)
        # 按id查询
        rows = conn.execute("""
                            SELECT *
                            FROM demo_session
                            WHERE id = ?""", (session_id,)).fetchall()
        for row in rows:
            print(row["id"], row["title"], row["location"], row["created_at"])
        print("-" * 20)
        #删
        conn.execute("""
        DELETE FROM demo_session WHERE id = ?""",(session_id,))
        search()
        print("-" * 20)



finally:
    conn.close()
