import sqlite3
import os

def rename_columns_in_experiments_db(db_name="experiments.db"):
    """
    Renames columns in the 'reagents' and 'protocols' tables
    within the specified SQLite database.

    - In 'reagents' table: renames 'reagents' column to 'content'.
    - In 'protocols'table: renames 'protocol' column to 'content'.

    Args:
        db_name (str): The name of the SQLite database file.
    """

    if not os.path.exists(db_name):
        print(f"Error: Database file '{db_name}' not found.")
        return

    conn = None
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        # Check SQLite version (optional, but good to know)
        # RENAME COLUMN was added in SQLite 3.25.0 (2018-09-15)
        cursor.execute("SELECT sqlite_version();")
        sqlite_version = cursor.fetchone()[0]
        print(f"Connected to SQLite version: {sqlite_version}")

        if tuple(map(int, sqlite_version.split('.'))) < (3, 25, 0):
            print(f"Warning: Your SQLite version ({sqlite_version}) is older than 3.25.0. "
                  "The RENAME COLUMN command might not be available. "
                  "Consider updating SQLite or using an older method for renaming (more complex).")

        # --- Rename column in 'reagents' table ---
        table_reagents = "reagents"
        old_col_reagents = "reagents"
        new_col_content = "content" # Common new name

        try:
            print(f"Attempting to rename column '{old_col_reagents}' to '{new_col_content}' in table '{table_reagents}'...")
            # Note: Table and column names should not be parameterized with '?'
            # For fixed names like these, direct insertion is fine.
            # If names were from user input, sanitization/validation would be crucial.
            cursor.execute(f"ALTER TABLE {table_reagents} RENAME COLUMN {old_col_reagents} TO {new_col_content}")
            print(f"Successfully renamed column in table '{table_reagents}'.")
        except sqlite3.Error as e:
            # Common error: "no such column: reagents" if already renamed or never existed with that name.
            # Or, if the new name "content" already exists as another column.
            print(f"Error renaming column in table '{table_reagents}': {e}")
            if "duplicate column name: content" in str(e):
                 print(f"It seems a column named '{new_col_content}' already exists in '{table_reagents}'.")
            elif f"no such column: {old_col_reagents}" in str(e):
                 print(f"It seems the column '{old_col_reagents}' does not exist in '{table_reagents}'.")


        # --- Rename column in 'protocols' table ---
        table_protocols = "protocols"
        old_col_protocol = "protocol"
        # new_col_content is the same as above

        try:
            print(f"\nAttempting to rename column '{old_col_protocol}' to '{new_col_content}' in table '{table_protocols}'...")
            cursor.execute(f"ALTER TABLE {table_protocols} RENAME COLUMN {old_col_protocol} TO {new_col_content}")
            print(f"Successfully renamed column in table '{table_protocols}'.")
        except sqlite3.Error as e:
            print(f"Error renaming column in table '{table_protocols}': {e}")
            if "duplicate column name: content" in str(e):
                 print(f"It seems a column named '{new_col_content}' already exists in '{table_protocols}'.")
            elif f"no such column: {old_col_protocol}" in str(e):
                 print(f"It seems the column '{old_col_protocol}' does not exist in '{table_protocols}'.")


        # Commit the changes to the database
        conn.commit()
        print("\nChanges have been committed to the database.")

    except sqlite3.Error as e:
        print(f"An overall database error occurred: {e}")
        if conn:
            conn.rollback() # Rollback changes if any overall error before commit
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")

if __name__ == '__main__':
    rename_columns_in_experiments_db()