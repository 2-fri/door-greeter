import sqlite3
import sqlite_vec

def check_db():
    db = sqlite3.connect("faces.db")
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS faces USING vec0(embedding FLOAT[512], value TEXT);")
    cursor = db.execute("SELECT rowid, value FROM faces")
    print("Checking Database...")
    for rowid, description in cursor:
        print(f"ID: {rowid}\n{description}\n")
    db.close()