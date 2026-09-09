import sqlite3

def modify_reagents_table(db_path="experiments.db"):
    """
    Modifies the 'reagents' table in the specified SQLite database.

    For experiments ending with "Cryorecovery", adds "Complete media: " to the 
    beginning of the content.  For those ending with "Cryopreservation", adds
    "Cryopreservation media: " to the end of the content.

    Args:
        db_path (str, optional): The path to the SQLite database file.
            Defaults to "experiments.db".
    """
    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Update records ending with 'Cryorecovery'
        cursor.execute("""
            UPDATE reagents
            SET content = 'Complete media: ' || content
            WHERE experiment LIKE '%Cryorecovery'
        """)

        # Update records ending with 'Cryopreservation'
        cursor.execute("""
            UPDATE reagents
            SET content = 'Cryopreservation media: ' || content 
            WHERE experiment LIKE '%Cryopreservation'
        """)

        # Commit the changes
        conn.commit()
        print("Reagents table updated successfully.")

    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
        conn.rollback()  # Rollback changes on error, to maintain data integrity.
    finally:
        # Close the connection
        if conn: #check if the connection is still valid
            conn.close()

if __name__ == "__main__":
    # Call the function to modify the database.  
    # You can change the path if your database is not named "experiments.db"
    # or is in a different location.
    modify_reagents_table()
