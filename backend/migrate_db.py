import sqlite3

db_path = "backend/jobs.db"

def migrate():
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if column exists to avoid error
        cursor.execute("PRAGMA table_info(job)")
        columns = [info[1] for info in cursor.fetchall()]
        
        if "tags" not in columns:
            print("Adding 'tags' column to 'job' table...")
            cursor.execute("ALTER TABLE job ADD COLUMN tags TEXT")
            conn.commit()
            print("Migration successful! 'tags' column added.")
        else:
            print("'tags' column already exists.")
            
        conn.close()
    except Exception as e:
        print(f"Migration failed: {e}")

if __name__ == "__main__":
    migrate()
