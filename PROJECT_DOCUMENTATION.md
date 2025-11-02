# Fruit Ripeness Classifier - Complete Project Documentation

**Student:** Maria Paula Salazar Agudelo
**Course:** AI Minor - Personal Challenge
**Date:** 2025

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Technologies Used](#technologies-used)
3. [Database System](#database-system)
4. [Project Structure](#project-structure)
5. [How Everything Works](#how-everything-works)
6. [Libraries Explained](#libraries-explained)
7. [Methods and Functions](#methods-and-functions)
8. [SQL Queries Explained](#sql-queries-explained)
9. [Running the Project](#running-the-project)
10. [Troubleshooting](#troubleshooting)

---

## Project Overview

### What is this project?

A machine learning system that looks at a photo of a fruit and tells you if it's:
- **Fresh** (good to eat)
- **Rotten** (throw it away)
- **Unripe** (wait a few days)

**PLUS:** Saves every prediction to a database so you can track your testing history!

### What fruits does it recognize?

- Apples
- Bananas
- Oranges

**Total:** 9 categories (3 fruits × 3 ripeness stages)

---

## Technologies Used

### Programming Language

**Python 3.8+**
- Easy to learn
- Great for AI/ML
- Has excellent database support

### Main Libraries

#### 1. TensorFlow / Keras
**What it is:** Framework for building neural networks

**What I use it for:**
- Building the model
- Training
- Making predictions

#### 2. SQLite3
**What it is:** Lightweight database (no server needed!)

**What I use it for:**
- Save every prediction
- Track testing history
- Generate reports
- Prove what I tested

**Beginner explanation:** Like an Excel file but much more powerful - stores all your data in an organized way

#### 3. NumPy
**What it is:** Mathematical library

**What I use it for:**
- Image processing
- Array operations

#### 4. Matplotlib
**What it is:** Graphing library

**What I use it for:**
- Show images
- Create charts

#### 5. Pillow (PIL)
**What it is:** Image processing

**What I use it for:**
- Load images
- Resize images

#### 6. Flask
**What it is:** Web framework

**What I use it for:**
- Web interface
- Camera access
- Display predictions

---

## Database System

### Why Use a Database?

**Reasons:**
1. **Track History:** See all predictions you've made
2. **Prove Testing:** Show evidence of testing for your project
3. **Live Predictions:** Save camera predictions automatically
4. **Analysis:** Analyze which fruits the model gets wrong
5. **Portfolio Evidence:** Concrete proof of your work

### Database Structure

We use **SQLite** - a simple database that saves to one file.

**Database file:** `predictions.db`

**Table: predictions**

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Auto-increment ID (unique for each prediction) |
| timestamp | TEXT | When prediction was made |
| image_path | TEXT | Path to the image file |
| true_label | TEXT | Actual fruit type (if known) |
| predicted_label | TEXT | What model predicted |
| confidence | REAL | Confidence percentage (0-100) |
| correct | INTEGER | Was prediction correct? (1=yes, 0=no, NULL=unknown) |
| source | TEXT | Where prediction came from (camera/upload/script) |

**Example row:**
```
id: 1
timestamp: 2025-01-15 14:30:22
image_path: test_images/apple1.jpg
true_label: freshapples
predicted_label: freshapples
confidence: 92.5
correct: 1
source: camera
```

### SQL Basics (For This Project)

#### CREATE TABLE
**What it does:** Creates a new table in database

**Our code:**
```sql
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
```

**Explanation line by line:**
- `CREATE TABLE IF NOT EXISTS`: Make table (only if doesn't exist yet)
- `predictions`: Table name
- `id INTEGER PRIMARY KEY AUTOINCREMENT`: Unique ID, auto-increases (1, 2, 3...)
- `timestamp TEXT NOT NULL`: Date/time, required (NOT NULL = must have value)
- `image_path TEXT`: Path to image, optional
- `predicted_label TEXT NOT NULL`: Prediction, required
- `confidence REAL`: Confidence score, number with decimals
- `correct INTEGER`: 1=correct, 0=wrong, NULL=don't know
- `source TEXT`: Where it came from (camera/upload/etc)

#### INSERT
**What it does:** Add a new row to table

**Our code:**
```sql
INSERT INTO predictions
(timestamp, image_path, predicted_label, confidence, source)
VALUES (?, ?, ?, ?, ?)
```

**Explanation:**
- `INSERT INTO predictions`: Add to predictions table
- `(timestamp, image_path, ...)`: Which columns to fill
- `VALUES (?, ?, ?, ?, ?)`: Values to insert (? = placeholder for Python to fill in)

**Python example:**
```python
cursor.execute(
    "INSERT INTO predictions (timestamp, image_path, predicted_label, confidence, source) VALUES (?, ?, ?, ?, ?)",
    (datetime.now(), "apple.jpg", "freshapples", 92.5, "camera")
)
```

#### SELECT
**What it does:** Get data from table

**Examples:**

**Get all predictions:**
```sql
SELECT * FROM predictions
```
`*` means "all columns"

**Get last 10 predictions:**
```sql
SELECT * FROM predictions ORDER BY timestamp DESC LIMIT 10
```
- `ORDER BY timestamp DESC`: Sort by time, newest first
- `LIMIT 10`: Only show 10 rows

**Get only correct predictions:**
```sql
SELECT * FROM predictions WHERE correct = 1
```
`WHERE correct = 1`: Filter to only rows where correct = 1

**Count predictions per fruit:**
```sql
SELECT predicted_label, COUNT(*) as count
FROM predictions
GROUP BY predicted_label
```
- `COUNT(*)`: Count how many rows
- `GROUP BY predicted_label`: Group by each fruit type

#### UPDATE
**What it does:** Change existing data

**Example:**
```sql
UPDATE predictions
SET correct = 1
WHERE id = 5
```
Changes the `correct` column to 1 for row with id=5

#### DELETE
**What it does:** Remove rows

**Example:**
```sql
DELETE FROM predictions WHERE id = 10
```
Deletes row with id=10

---

## Project Structure

```
fruit-classifier-AI-project/
│
├── notebooks/
│   ├── 01_Dataset_Analysis.ipynb
│   ├── 02_Model_Training.ipynb
│   └── 03_Model_Evaluation.ipynb
│
├── scripts/
│   ├── train.py                       # Train the model
│   ├── predict.py                     # Predict + save to database
│   ├── visualize_predictions.py       # Show predictions from database
│   ├── db_helper.py                   # Database functions (beginner-friendly)
│   └── view_history.py                # View prediction history
│
├── models/
│   ├── fruit_classifier.keras         # Trained model
│   ├── class_labels.json              # Class names
│   └── training_config.json           # Training info
│
├── templates/
│   └── index.html                     # Web interface
│
├── predictions.db                 # SQLite database (created automatically)
├── app.py                         # Flask web app
├── TRAINING_GUIDE.md              # Training explained
├── APP_GUIDE.md                   # How to use the app
├── PROJECT_DOCUMENTATION.md       # This file
├── README.md                      # Project overview
├── requirements.txt               # Dependencies
└── .gitignore                     # Git ignore rules
```

---

## How Everything Works

### Flow with Database

```
1. Take/upload fruit image
        ↓
2. Model makes prediction
        ↓
3. Save to database:
   - Image path
   - Prediction
   - Confidence
   - Timestamp
   - Source (camera/upload)
        ↓
4. Show result to user
        ↓
5. Later: View history from database
```

### Why This is Useful

**Scenario 1: Testing for your project**
```
1. Test 100 images
2. All saved to database automatically
3. Generate report: "Tested 100 images, 87% accuracy"
4. Show teacher the database as proof
```

**Scenario 2: Live camera demo**
```
1. Scan fruits with camera
2. Each scan saved to database
3. After demo: Show history of all scans
4. Prove the model works in real-time
```

**Scenario 3: Error analysis**
```
1. Query database for wrong predictions
2. See which fruits get confused
3. Analyze patterns
4. Include in project report
```

---

## Libraries Explained

### sqlite3 (Database Library)

**Import:**
```python
import sqlite3
```

**What it does:** Lets Python talk to SQLite databases

#### `sqlite3.connect(filename)`
**What it does:** Opens/creates a database file

**Beginner explanation:** Like opening an Excel file - if it doesn't exist, creates it

**Example:**
```python
conn = sqlite3.connect('predictions.db')
# Creates predictions.db if it doesn't exist
# Opens it if it already exists
```

#### `cursor = conn.cursor()`
**What it does:** Creates a cursor to execute SQL commands

**Beginner explanation:** A cursor is like a pointer that lets you run SQL commands

**Example:**
```python
cursor = conn.cursor()
cursor.execute("SELECT * FROM predictions")
```

#### `conn.commit()`
**What it does:** Saves changes to database

**IMPORTANT:** Changes are NOT saved until you call commit()!

**Example:**
```python
cursor.execute("INSERT INTO predictions VALUES (...)")
conn.commit()  # NOW it's saved!
```

#### `conn.close()`
**What it does:** Closes database connection

**Example:**
```python
conn.close()  # Always close when done!
```

#### Complete Example:
```python
# 1. Connect to database
conn = sqlite3.connect('predictions.db')
cursor = conn.cursor()

# 2. Run SQL command
cursor.execute("INSERT INTO predictions (predicted_label, confidence) VALUES (?, ?)",
               ("freshapples", 92.5))

# 3. Save changes
conn.commit()

# 4. Close connection
conn.close()
```

---

### Other Libraries (Brief)

#### TensorFlow/Keras
```python
from tensorflow import keras

# Load model
model = keras.models.load_model('model.keras')

# Predict
predictions = model.predict(image_array)
```

#### NumPy
```python
import numpy as np

# Convert image to array
img_array = np.array(image)

# Find maximum
best_class = np.argmax(predictions)
```

#### Pillow (PIL)
```python
from PIL import Image

# Load image
img = Image.open('apple.jpg')

# Resize
img = img.resize((224, 224))
```

#### Matplotlib
```python
import matplotlib.pyplot as plt

# Show image
plt.imshow(image)
plt.show()
```

#### Flask
```python
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')
```

---

## Database Helper Functions

### We create a `db_helper.py` file with these functions:

#### `init_database()`
**What it does:** Creates the predictions table if it doesn't exist

**When to use:** Run once at the start

**Code:**
```python
def init_database():
    """Create predictions table if it doesn't exist"""
    conn = sqlite3.connect('predictions.db')
    cursor = conn.cursor()

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

    conn.commit()
    conn.close()
```

#### `log_prediction(image_path, predicted_label, confidence, source)`
**What it does:** Saves a prediction to database

**Parameters:**
- `image_path`: Where the image is
- `predicted_label`: What model predicted
- `confidence`: How confident (0-100)
- `source`: 'camera', 'upload', or 'script'

**Code:**
```python
def log_prediction(image_path, predicted_label, confidence, source='script'):
    """Save prediction to database"""
    conn = sqlite3.connect('predictions.db')
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO predictions
        (timestamp, image_path, predicted_label, confidence, source)
        VALUES (?, ?, ?, ?, ?)
    ''', (datetime.now().isoformat(), image_path, predicted_label, confidence, source))

    conn.commit()
    conn.close()
```

#### `get_recent_predictions(limit=10)`
**What it does:** Gets the most recent predictions

**Returns:** List of dictionaries

**Code:**
```python
def get_recent_predictions(limit=10):
    """Get most recent predictions"""
    conn = sqlite3.connect('predictions.db')
    cursor = conn.cursor()

    cursor.execute('''
        SELECT * FROM predictions
        ORDER BY timestamp DESC
        LIMIT ?
    ''', (limit,))

    rows = cursor.fetchall()
    conn.close()

    return rows
```

#### `get_accuracy_stats()`
**What it does:** Calculate accuracy from predictions where the truth is known

**Returns:** Dictionary with statistics

**Code:**
```python
def get_accuracy_stats():
    """Calculate accuracy statistics"""
    conn = sqlite3.connect('predictions.db')
    cursor = conn.cursor()

    # Count correct predictions
    cursor.execute('SELECT COUNT(*) FROM predictions WHERE correct = 1')
    correct = cursor.fetchone()[0]

    # Count total predictions with known truth
    cursor.execute('SELECT COUNT(*) FROM predictions WHERE correct IS NOT NULL')
    total = cursor.fetchone()[0]

    conn.close()

    if total > 0:
        accuracy = (correct / total) * 100
    else:
        accuracy = 0

    return {
        'correct': correct,
        'total': total,
        'accuracy': accuracy
    }
```

---

## SQL Queries We Use

### 1. Get All Predictions
```sql
SELECT * FROM predictions
```
**What it returns:** All columns, all rows

### 2. Get Recent 10 Predictions
```sql
SELECT * FROM predictions
ORDER BY timestamp DESC
LIMIT 10
```
**What it returns:** Last 10 predictions, newest first

### 3. Get Camera Predictions Only
```sql
SELECT * FROM predictions
WHERE source = 'camera'
```
**What it returns:** Only predictions from camera

### 4. Count Predictions Per Fruit
```sql
SELECT predicted_label, COUNT(*) as count
FROM predictions
GROUP BY predicted_label
```
**Example result:**
```
freshapples: 45
freshbanana: 32
rottenapples: 18
...
```

### 5. Get Wrong Predictions
```sql
SELECT * FROM predictions
WHERE correct = 0
```
**What it returns:** Only incorrect predictions

### 6. Calculate Accuracy
```sql
SELECT
    COUNT(CASE WHEN correct = 1 THEN 1 END) * 100.0 / COUNT(*) as accuracy
FROM predictions
WHERE correct IS NOT NULL
```
**What it returns:** Accuracy percentage

---

## Running the Project

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Initialize Database
```bash
python scripts/db_helper.py
```
Creates `predictions.db` with empty table

### 3. Train Model
```bash
python scripts/train.py
```

### 4. Make Predictions (saves to database)
```bash
python scripts/predict.py path/to/image.jpg
```

### 5. View History
```bash
python scripts/view_history.py
```
Shows all predictions from database

### 6. Run Web App (with camera)
```bash
python app.py
```
Open browser to `http://localhost:5000`

---

## Database Benefits for Your Project

### For Teacher/Portfolio:

**1. Proof of Testing:**
```sql
SELECT COUNT(*) FROM predictions
-- Result: "Tested 250 images"
```

**2. Show Accuracy:**
```sql
SELECT
    SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*)
FROM predictions
WHERE correct IS NOT NULL
-- Result: "87.5% accuracy on manual tests"
```

**3. Show Usage:**
```sql
SELECT source, COUNT(*)
FROM predictions
GROUP BY source
-- Result:
-- camera: 45
-- upload: 30
-- script: 175
```

**4. Error Analysis:**
```sql
SELECT true_label, predicted_label, COUNT(*)
FROM predictions
WHERE correct = 0
GROUP BY true_label, predicted_label
-- Shows which fruits get confused
```

---

## Summary

**This project uses database to:**
1. ✅ Save every prediction automatically
2. ✅ Track testing history
3. ✅ Prove project work
4. ✅ Generate statistics
5. ✅ Analyze errors

**Database is:**
- Simple (SQLite - one file)
- No server needed
- Beginner-friendly
- Perfect for student projects
- Provides concrete evidence of testing

**All SQL queries explained** at beginner level with examples!

---

**Student:** Maria Paula Salazar Agudelo
**Course:** AI Minor - Personal Challenge
**Date:** 2025
