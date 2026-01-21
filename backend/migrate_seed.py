import sqlite3

db_path = "backend/jobs.db"

def migrate():
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if column exists to avoid error
        cursor.execute("PRAGMA table_info(job)")
        columns = [info[1] for info in cursor.fetchall()]
        
        if "seed" not in columns:
            print("Adding 'seed' column to 'job' table...")
            cursor.execute("ALTER TABLE job ADD COLUMN seed INTEGER")
            conn.commit()
            print("Migration successful! 'seed' column added.")
        else:
            print("'seed' column already exists.")
            
        conn.close()
    except Exception as e:
        print(f"Migration failed: {e}")

if __name__ == "__main__":
    migrate()
