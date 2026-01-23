import sqlite3
import os

DB_FILE = "jobs.db"

def migrate():
    if not os.path.exists(DB_FILE):
        print(f"Database {DB_FILE} not found. Skipping migration (will be created fresh).")
        return

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    try:
        # Check if column exists
        cursor.execute("PRAGMA table_info(job)")
        columns = [info[1] for info in cursor.fetchall()]
        
        if "is_favorite" in columns:
            print("Column 'is_favorite' already exists.")
        else:
            print("Adding 'is_favorite' column...")
            cursor.execute("ALTER TABLE job ADD COLUMN is_favorite BOOLEAN DEFAULT 0")
            conn.commit()
            print("Added 'is_favorite'.")

        # New fields migration
        new_columns = {
            "llm_model": "TEXT",
            "parent_job_id": "TEXT",
            "temperature": "FLOAT",
            "cfg_scale": "FLOAT",
            "topk": "INTEGER"
        }

        for col_name, col_type in new_columns.items():
            if col_name in columns:
                print(f"Column '{col_name}' already exists.")
            else:
                print(f"Adding '{col_name}' column...")
                cursor.execute(f"ALTER TABLE job ADD COLUMN {col_name} {col_type}")
                conn.commit()
                print(f"Added '{col_name}'.")
            
        print("Migration successful.")
            
    except Exception as e:
        print(f"Migration failed: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    migrate()
