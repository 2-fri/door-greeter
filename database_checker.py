import sqlite3
import sqlite_vec
import numpy as np

db = sqlite3.connect("faces.db")
db.enable_load_extension(True)
sqlite_vec.load(db)
cursor = db.execute("SELECT rowid, embedding FROM faces")
print("Checking Database...")
for rowid, _ in cursor:
    print(f"ID: {rowid}")