"""
Database Helper Functions for Fruit Ripeness Classifier
Student: Maria Paula Salazar Agudelo
Course: AI Minor - Personal Challenge

WHAT THIS FILE DOES:
This file contains helper functions to work with the SQLite database.
The database saves every prediction made by the model so I can:
- Track prediction history
- Prove testing work for portfolio
- Analyze which fruits the model gets wrong
- Show live camera prediction results

DATABASE FILE: predictions.db (created automatically in project root)

HOW TO USE:
    from scripts.db_helper import init_database, log_prediction

    # First time: Create the database table
    init_database()

    # After each prediction: Save to database
    log_prediction("apple.jpg", "freshapples", 92.5, "camera")

BEGINNER NOTE:
- SQLite = Simple database system (no server needed, just a file)
- SQL = Language to talk to databases
- All SQL queries are explained line by line in this file
"""

import sqlite3
import os
from datetime import datetime

# ============================================================================
# DATABASE CONFIGURATION
# ============================================================================

# Database file location (stored in project root)
DB_PATH = "predictions.db"

# ============================================================================
# HELPER FUNCTION: Initialize Database
# ============================================================================

def init_database():
    """
    Create the predictions table if it doesn't exist.

    WHEN TO USE:
    - Run this ONCE when you first set up the project
    - Safe to run multiple times (won't delete existing data)

    WHAT IT DOES:
    1. Connects to predictions.db (creates file if doesn't exist)
    2. Creates 'predictions' table with 8 columns
    3. Saves and closes connection

    SQL QUERY EXPLAINED:
        CREATE TABLE IF NOT EXISTS predictions (...)
        - CREATE TABLE = Make a new table
        - IF NOT EXISTS = Only if table doesn't already exist
        - predictions = Name of our table

    COLUMNS EXPLAINED:
        id              = Unique number for each prediction (auto-increments: 1, 2, 3...)
        timestamp       = When prediction was made (required)
        image_path      = Where the image is stored (optional)
        true_label      = Actual fruit type if known (optional)
        predicted_label = What model predicted (required)
        confidence      = How confident 0-100% (optional)
        correct         = Was prediction correct? 1=yes, 0=no, NULL=unknown
        source          = Where prediction came from (camera/upload/script)

    EXAMPLE:
        init_database()
        # Creates predictions.db with empty predictions table
    """

    print("Initializing database...")

    # Step 1: Connect to database
    # WHY? Opens connection to database file (creates if doesn't exist)
    conn = sqlite3.connect(DB_PATH)

    # Step 2: Create cursor
    # WHY? A cursor lets us execute SQL commands
    # BEGINNER NOTE: Think of cursor as a "pointer" to run SQL queries
    cursor = conn.cursor()

    # Step 3: Create table (if doesn't exist)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            image_path TEXT,
            true_label TEXT,
            predicted_label TEXT NOT NULL,
            confidence REAL,
            correct INTEGER,
            source TEXT
        )
    ''')

    # Step 4: Save changes
    # WHY? Changes are NOT saved until commit() is called
    # BEGINNER NOTE: Like clicking "Save" in a document
    conn.commit()

    # Step 5: Close connection
    # WHY? Free up resources, good practice
    conn.close()

    print(f"Database initialized: {DB_PATH}")
    print("Table 'predictions' ready to use")


# ============================================================================
# HELPER FUNCTION: Log Prediction
# ============================================================================

def log_prediction(image_path, predicted_label, confidence, source='script', true_label=None):
    """
    Save a prediction to the database.

    PARAMETERS:
        image_path       (str): Path to the image file
        predicted_label  (str): What the model predicted (e.g., "freshapples")
        confidence      (float): Confidence percentage 0-100
        source          (str): Where prediction came from - "camera", "upload", or "script"
        true_label      (str): Optional - actual fruit type if known

    WHAT IT DOES:
    1. Opens database connection
    2. Inserts new row with prediction data
    3. Saves and closes

    SQL QUERY EXPLAINED:
        INSERT INTO predictions (columns...) VALUES (?, ?, ...)
        - INSERT INTO = Add new row to table
        - predictions = Table name
        - (timestamp, image_path, ...) = Which columns to fill
        - VALUES (?, ?, ...) = Placeholders for Python values
        - The ? are replaced by actual values from Python

    EXAMPLE:
        # After making a prediction
        log_prediction(
            image_path="test_images/apple1.jpg",
            predicted_label="freshapples",
            confidence=92.5,
            source="camera"
        )

        # Result: New row added to predictions table
    """

    # Step 1: Connect to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Step 2: Get current timestamp
    # WHY? We want to know WHEN each prediction was made
    # FORMAT: "2025-01-15 14:30:22"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Step 3: Calculate if prediction is correct (if true label is known)
    # 1 = correct, 0 = wrong, None = don't know
    if true_label is not None:
        correct = 1 if predicted_label == true_label else 0
    else:
        correct = None  # Don't know if correct

    # Step 4: Insert prediction into database
    # SQL NOTE: The ? placeholders prevent SQL injection attacks
    # The values in (...) replace the ? in order
    cursor.execute('''
        INSERT INTO predictions
        (timestamp, image_path, true_label, predicted_label, confidence, correct, source)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (timestamp, image_path, true_label, predicted_label, confidence, correct, source))

    # Step 5: Save changes
    conn.commit()

    # Step 6: Get the ID of the row just inserted
    # WHY? Useful to show "Prediction #5 saved"
    prediction_id = cursor.lastrowid

    # Step 7: Close connection
    conn.close()

    print(f"Prediction #{prediction_id} saved to database")
    print(f"  Image: {image_path}")
    print(f"  Predicted: {predicted_label} ({confidence:.1f}% confidence)")
    print(f"  Source: {source}")

    return prediction_id


# ============================================================================
# HELPER FUNCTION: Get Recent Predictions
# ============================================================================

def get_recent_predictions(limit=10):
    """
    Get the most recent predictions from database.

    PARAMETERS:
        limit (int): How many predictions to return (default: 10)

    RETURNS:
        List of dictionaries, each containing one prediction's data

    SQL QUERY EXPLAINED:
        SELECT * FROM predictions ORDER BY timestamp DESC LIMIT ?
        - SELECT * = Get all columns
        - FROM predictions = From this table
        - ORDER BY timestamp DESC = Sort by time, newest first (DESC = descending)
        - LIMIT ? = Only return this many rows

    EXAMPLE:
        recent = get_recent_predictions(5)
        # Returns last 5 predictions as list of dictionaries

        # Print them
        for pred in recent:
            print(f"{pred['timestamp']}: {pred['predicted_label']} ({pred['confidence']}%)")
    """

    # Step 1: Connect to database
    conn = sqlite3.connect(DB_PATH)

    # Step 2: Set row factory to get dictionaries instead of tuples
    # WHY? Easier to use - can access like pred['timestamp'] instead of pred[1]
    # BEGINNER NOTE: This makes results easier to work with
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Step 3: Query for recent predictions
    cursor.execute('''
        SELECT * FROM predictions
        ORDER BY timestamp DESC
        LIMIT ?
    ''', (limit,))

    # Step 4: Fetch all results
    # fetchall() = Get all rows that match the query
    rows = cursor.fetchall()

    # Step 5: Close connection
    conn.close()

    # Step 6: Convert to list of dictionaries
    # WHY? sqlite3.Row objects need to be converted for easy use
    predictions = []
    for row in rows:
        predictions.append({
            'id': row['id'],
            'timestamp': row['timestamp'],
            'image_path': row['image_path'],
            'true_label': row['true_label'],
            'predicted_label': row['predicted_label'],
            'confidence': row['confidence'],
            'correct': row['correct'],
            'source': row['source']
        })

    return predictions


# ============================================================================
# HELPER FUNCTION: Get Accuracy Statistics
# ============================================================================

def get_accuracy_stats():
    """
    Calculate accuracy statistics from predictions where the truth is known.

    WHAT IT DOES:
    1. Counts how many predictions were correct
    2. Counts total predictions with known truth
    3. Calculates accuracy percentage

    RETURNS:
        Dictionary with:
            'correct': Number of correct predictions
            'total': Total predictions with known truth
            'accuracy': Accuracy percentage (0-100)

    SQL QUERIES EXPLAINED:
        Query 1: SELECT COUNT(*) FROM predictions WHERE correct = 1
        - COUNT(*) = Count how many rows
        - WHERE correct = 1 = Only rows where prediction was correct

        Query 2: SELECT COUNT(*) FROM predictions WHERE correct IS NOT NULL
        - WHERE correct IS NOT NULL = Only rows where correctness is known
        - Excludes predictions where the true label wasn't known

    EXAMPLE:
        stats = get_accuracy_stats()
        print(f"Accuracy: {stats['accuracy']:.1f}%")
        print(f"Correct: {stats['correct']} out of {stats['total']}")

        # Output example:
        # Accuracy: 87.5%
        # Correct: 175 out of 200
    """

    # Step 1: Connect to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Step 2: Count correct predictions
    # WHERE correct = 1 means only count the ones that were right
    cursor.execute('SELECT COUNT(*) FROM predictions WHERE correct = 1')
    correct = cursor.fetchone()[0]  # fetchone()[0] gets the first column of first row

    # Step 3: Count total predictions with known truth
    # WHERE correct IS NOT NULL means correctness is known (right or wrong)
    cursor.execute('SELECT COUNT(*) FROM predictions WHERE correct IS NOT NULL')
    total = cursor.fetchone()[0]

    # Step 4: Close connection
    conn.close()

    # Step 5: Calculate accuracy
    # WHY? Accuracy = (correct / total) * 100
    # IF total is 0, set accuracy to 0 (avoid division by zero)
    if total > 0:
        accuracy = (correct / total) * 100
    else:
        accuracy = 0

    # Step 6: Return statistics
    return {
        'correct': correct,
        'total': total,
        'accuracy': accuracy
    }


# ============================================================================
# HELPER FUNCTION: Get Predictions by Source
# ============================================================================

def get_predictions_by_source(source):
    """
    Get all predictions from a specific source (camera, upload, or script).

    PARAMETERS:
        source (str): "camera", "upload", or "script"

    RETURNS:
        List of dictionaries with predictions from that source

    SQL QUERY EXPLAINED:
        SELECT * FROM predictions WHERE source = ?
        - WHERE source = ? = Filter to only rows matching this source

    EXAMPLE:
        camera_preds = get_predictions_by_source("camera")
        print(f"Made {len(camera_preds)} camera predictions")
    """

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute('''
        SELECT * FROM predictions
        WHERE source = ?
        ORDER BY timestamp DESC
    ''', (source,))

    rows = cursor.fetchall()
    conn.close()

    predictions = []
    for row in rows:
        predictions.append({
            'id': row['id'],
            'timestamp': row['timestamp'],
            'image_path': row['image_path'],
            'predicted_label': row['predicted_label'],
            'confidence': row['confidence'],
            'correct': row['correct']
        })

    return predictions


# ============================================================================
# HELPER FUNCTION: Count Predictions per Fruit
# ============================================================================

def count_predictions_per_fruit():
    """
    Count how many predictions were made for each fruit type.

    RETURNS:
        Dictionary mapping fruit names to counts
        Example: {'freshapples': 45, 'freshbanana': 32, ...}

    SQL QUERY EXPLAINED:
        SELECT predicted_label, COUNT(*) as count
        FROM predictions
        GROUP BY predicted_label

        - SELECT predicted_label, COUNT(*) = Get fruit name and count
        - GROUP BY predicted_label = Group all rows by fruit type
        - COUNT(*) as count = Count rows in each group, name it "count"

    EXAMPLE:
        counts = count_predictions_per_fruit()
        for fruit, count in counts.items():
            print(f"{fruit}: {count} predictions")

        # Output example:
        # freshapples: 45 predictions
        # freshbanana: 32 predictions
        # rottenapples: 18 predictions
    """

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        SELECT predicted_label, COUNT(*) as count
        FROM predictions
        GROUP BY predicted_label
        ORDER BY count DESC
    ''')

    rows = cursor.fetchall()
    conn.close()

    # Convert to dictionary
    counts = {}
    for row in rows:
        fruit_name = row[0]  # First column: predicted_label
        count = row[1]       # Second column: count
        counts[fruit_name] = count

    return counts


# ============================================================================
# HELPER FUNCTION: Delete All Predictions (USE WITH CAUTION)
# ============================================================================

def clear_database():
    """
    Delete ALL predictions from database.

    WARNING: This CANNOT be undone!

    USE CASE:
    - Starting fresh testing
    - Clearing test data before real demo

    SQL QUERY EXPLAINED:
        DELETE FROM predictions
        - DELETE FROM = Remove rows
        - predictions = From this table
        - (no WHERE clause = delete ALL rows)

    EXAMPLE:
        # Ask user first!
        response = input("Delete all predictions? (yes/no): ")
        if response.lower() == "yes":
            clear_database()
    """

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Delete all rows
    cursor.execute('DELETE FROM predictions')

    # Get how many were deleted
    deleted_count = cursor.rowcount

    conn.commit()
    conn.close()

    print(f"Database cleared: {deleted_count} predictions deleted")


# ============================================================================
# MAIN - Test Database Functions
# ============================================================================

if __name__ == "__main__":
    """
    Test the database functions.

    RUN THIS TO TEST:
        python scripts/db_helper.py

    WHAT IT DOES:
    1. Creates database table
    2. Adds 3 test predictions
    3. Shows recent predictions
    4. Calculates accuracy
    5. Shows counts per fruit
    """

    print("="*70)
    print("TESTING DATABASE FUNCTIONS")
    print("="*70)
    print()

    # Test 1: Initialize database
    print("TEST 1: Initialize Database")
    print("-" * 70)
    init_database()
    print()

    # Test 2: Add some predictions
    print("TEST 2: Add Test Predictions")
    print("-" * 70)
    log_prediction("test1.jpg", "freshapples", 92.5, "script", "freshapples")
    log_prediction("test2.jpg", "freshbanana", 88.3, "camera", "freshbanana")
    log_prediction("test3.jpg", "rottenapples", 76.2, "upload", "freshapples")  # Wrong prediction
    print()

    # Test 3: Get recent predictions
    print("TEST 3: Get Recent Predictions")
    print("-" * 70)
    recent = get_recent_predictions(5)
    for pred in recent:
        status = "CORRECT" if pred['correct'] == 1 else "WRONG" if pred['correct'] == 0 else "UNKNOWN"
        print(f"{pred['timestamp']}: {pred['predicted_label']} ({pred['confidence']:.1f}%) - {status}")
    print()

    # Test 4: Get accuracy stats
    print("TEST 4: Calculate Accuracy")
    print("-" * 70)
    stats = get_accuracy_stats()
    print(f"Correct predictions: {stats['correct']}")
    print(f"Total predictions: {stats['total']}")
    print(f"Accuracy: {stats['accuracy']:.1f}%")
    print()

    # Test 5: Count per fruit
    print("TEST 5: Count Predictions per Fruit")
    print("-" * 70)
    counts = count_predictions_per_fruit()
    for fruit, count in counts.items():
        print(f"{fruit}: {count} predictions")
    print()

    # Test 6: Get predictions by source
    print("TEST 6: Get Predictions by Source")
    print("-" * 70)
    camera_preds = get_predictions_by_source("camera")
    print(f"Camera predictions: {len(camera_preds)}")
    print()

    print("="*70)
    print("ALL TESTS COMPLETE")
    print("="*70)
    print()
    print("Database file created:", DB_PATH)
    print("You can now use these functions in other scripts!")
