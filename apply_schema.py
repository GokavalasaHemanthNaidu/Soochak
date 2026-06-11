import sqlite3
import os

db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "soochak.db"))
schema_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "db_schema.sql"))

if os.path.exists(db_path):
    print("soochak.db exists, applying migration directly (001_initial is now safe).")

with open(schema_path, "r") as f:
    schema_sql = f.read()

conn = sqlite3.connect(db_path)
cursor = conn.cursor()
try:
    # Phase 3 schema had predictions table with different columns. 
    # Let's drop it to apply the phase 5 one safely.
    cursor.execute("DROP TABLE IF EXISTS predictions;")
    cursor.executescript(schema_sql)
    conn.commit()
    print("Schema successfully applied to soochak.db")
except Exception as e:
    print("Error applying schema:", e)
finally:
    conn.close()
