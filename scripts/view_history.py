"""
Fruit Ripeness Classifier - View Prediction History
Student: Maria Paula Salazar Agudelo
Course: AI Minor - Personal Challenge

WHAT THIS DOES:
Simple script to view all predictions saved in the database.
Shows predictions in a clean, readable format.

WHY THIS IS USEFUL:
- Quick check of testing history
- See what predictions were made
- Verify database is working
- Track testing progress

HOW TO RUN:
    python scripts/view_history.py

    Optional arguments:
    --limit N       Show only last N predictions (default: 50)
    --source TYPE   Show only predictions from source (camera/upload/script)
    --correct       Show only correct predictions
    --wrong         Show only wrong predictions
    --export FILE   Export to text file

REQUIREMENTS:
- Database file: predictions.db
- Python packages: sqlite3
"""

import os
import sys
import argparse
import sqlite3
from datetime import datetime

# Import database helper functions
from db_helper import (
    get_recent_predictions,
    get_accuracy_stats,
    count_predictions_per_fruit,
    get_predictions_by_source
)

# ============================================================================
# CONFIGURATION
# ============================================================================

DB_PATH = "predictions.db"

# ============================================================================
# DISPLAY FUNCTIONS
# ============================================================================

def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*70)
    print(title)
    print("="*70 + "\n")


def display_prediction_summary():
    """
    Display a summary of all predictions in database.

    WHAT IT SHOWS:
    - Total number of predictions
    - Number of predictions per fruit
    - Number of predictions per source
    - Overall accuracy (if available)
    """

    print_section("PREDICTION SUMMARY")

    # Get total count
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM predictions')
    total = cursor.fetchone()[0]
    conn.close()

    print(f"Total predictions: {total}")

    if total == 0:
        print("No predictions in database yet!")
        return

    # Predictions per fruit
    print("\nPredictions per fruit:")
    counts = count_predictions_per_fruit()
    for fruit, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total) * 100
        print(f"  {fruit:<20} {count:>4} ({percentage:>5.1f}%)")

    # Predictions per source
    print("\nPredictions per source:")
    sources = ['camera', 'upload', 'script']
    for source in sources:
        preds = get_predictions_by_source(source)
        count = len(preds)
        if count > 0:
            percentage = (count / total) * 100
            print(f"  {source:<20} {count:>4} ({percentage:>5.1f}%)")

    # Accuracy stats
    stats = get_accuracy_stats()
    if stats['total'] > 0:
        print("\nAccuracy statistics:")
        print(f"  Predictions with known truth: {stats['total']}")
        print(f"  Correct predictions:          {stats['correct']}")
        print(f"  Accuracy:                     {stats['accuracy']:.2f}%")


def display_predictions_table(predictions, show_details=True):
    """
    Display predictions in a formatted table.

    PARAMETERS:
        predictions: List of prediction dictionaries
        show_details: Whether to show detailed view (default: True)

    TABLE FORMAT:
        ID  | Timestamp           | Image Path       | Predicted      | Conf | Correct | Source
        ----+---------------------+------------------+----------------+------+---------+--------
        15  | 2025-01-15 14:30:22 | test/apple1.jpg  | freshapples    | 92.5 | YES     | camera
    """

    if not predictions:
        print("No predictions to display!")
        return

    print(f"Showing {len(predictions)} prediction(s):\n")

    if show_details:
        # Detailed view
        print(f"{'ID':<4} | {'Timestamp':<19} | {'Image Path':<25} | {'Predicted':<15} | {'Conf':<5} | {'Correct':<7} | {'Source':<8}")
        print("-" * 110)

        for pred in predictions:
            # Format fields
            pred_id = pred['id']
            timestamp = pred['timestamp']

            # Shorten image path if too long
            image = pred['image_path'] if pred['image_path'] else "N/A"
            if len(image) > 25:
                image = "..." + image[-22:]

            label = pred['predicted_label'][:15]
            conf = f"{pred['confidence']:.1f}" if pred['confidence'] else "N/A"

            # Format correct status
            if pred['correct'] == 1:
                correct = "YES"
            elif pred['correct'] == 0:
                correct = "NO"
            else:
                correct = "UNKNOWN"

            source = pred['source'] if pred['source'] else "N/A"

            # Print row
            print(f"{pred_id:<4} | {timestamp:<19} | {image:<25} | {label:<15} | {conf:<5} | {correct:<7} | {source:<8}")

    else:
        # Compact view
        print(f"{'ID':<4} | {'Timestamp':<19} | {'Predicted':<15} | {'Confidence':<10}")
        print("-" * 55)

        for pred in predictions:
            pred_id = pred['id']
            timestamp = pred['timestamp']
            label = pred['predicted_label'][:15]
            conf = f"{pred['confidence']:.1f}%" if pred['confidence'] else "N/A"

            print(f"{pred_id:<4} | {timestamp:<19} | {label:<15} | {conf:<10}")


def display_detailed_prediction(prediction_id):
    """
    Display all details for a single prediction.

    PARAMETERS:
        prediction_id (int): ID of prediction to show

    SHOWS:
    - All database fields
    - Formatted in easy-to-read format
    """

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute('SELECT * FROM predictions WHERE id = ?', (prediction_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        print(f"Prediction #{prediction_id} not found in database!")
        return

    print_section(f"PREDICTION DETAILS - ID #{prediction_id}")

    print(f"Timestamp:        {row['timestamp']}")
    print(f"Image Path:       {row['image_path']}")
    print(f"Predicted Label:  {row['predicted_label']}")
    print(f"Confidence:       {row['confidence']:.2f}%" if row['confidence'] else "Confidence:       N/A")
    print(f"True Label:       {row['true_label']}" if row['true_label'] else "True Label:       Unknown")

    if row['correct'] == 1:
        print(f"Correct:          YES")
    elif row['correct'] == 0:
        print(f"Correct:          NO")
    else:
        print(f"Correct:          UNKNOWN")

    print(f"Source:           {row['source']}" if row['source'] else "Source:           N/A")


def export_to_file(predictions, filename):
    """
    Export predictions to a text file.

    PARAMETERS:
        predictions: List of prediction dictionaries
        filename: Path to output file

    WHAT IT DOES:
    1. Creates a formatted text file
    2. Includes all prediction details
    3. Saves to specified location

    WHY THIS IS USEFUL:
    - Share testing results
    - Include in project documentation
    - Backup of prediction history
    """

    print(f"Exporting {len(predictions)} predictions to {filename}...")

    with open(filename, 'w') as f:
        # Header
        f.write("="*70 + "\n")
        f.write("FRUIT RIPENESS CLASSIFIER - PREDICTION HISTORY\n")
        f.write("Student: Maria Paula Salazar Agudelo\n")
        f.write(f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*70 + "\n\n")

        # Summary
        f.write(f"Total predictions: {len(predictions)}\n\n")

        # Table header
        f.write(f"{'ID':<4} | {'Timestamp':<19} | {'Predicted':<15} | {'Conf':<6} | {'Correct':<7} | {'Source':<8}\n")
        f.write("-" * 70 + "\n")

        # Predictions
        for pred in predictions:
            pred_id = pred['id']
            timestamp = pred['timestamp']
            label = pred['predicted_label'][:15]
            conf = f"{pred['confidence']:.1f}%" if pred['confidence'] else "N/A"

            if pred['correct'] == 1:
                correct = "YES"
            elif pred['correct'] == 0:
                correct = "NO"
            else:
                correct = "UNKNOWN"

            source = pred['source'] if pred['source'] else "N/A"

            f.write(f"{pred_id:<4} | {timestamp:<19} | {label:<15} | {conf:<6} | {correct:<7} | {source:<8}\n")

        f.write("\n" + "="*70 + "\n")

    print(f"Exported successfully!")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """
    Main function for view history script.

    COMMAND LINE USAGE:
        python scripts/view_history.py [options]

    EXAMPLES:
        # Show last 50 predictions
        python scripts/view_history.py

        # Show last 10 predictions
        python scripts/view_history.py --limit 10

        # Show only camera predictions
        python scripts/view_history.py --source camera

        # Show only correct predictions
        python scripts/view_history.py --correct

        # Export to file
        python scripts/view_history.py --export history.txt

        # Show details for specific prediction
        python scripts/view_history.py --id 15
    """

    # Setup argument parser
    parser = argparse.ArgumentParser(
        description='View prediction history from database',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--limit', type=int, default=50,
                       help='Number of recent predictions to show (default: 50)')
    parser.add_argument('--source', type=str, choices=['camera', 'upload', 'script'],
                       help='Show only predictions from this source')
    parser.add_argument('--correct', action='store_true',
                       help='Show only correct predictions')
    parser.add_argument('--wrong', action='store_true',
                       help='Show only wrong predictions')
    parser.add_argument('--export', type=str,
                       help='Export predictions to text file')
    parser.add_argument('--id', type=int,
                       help='Show detailed view of specific prediction ID')
    parser.add_argument('--compact', action='store_true',
                       help='Show compact view (less columns)')

    args = parser.parse_args()

    print("="*70)
    print("FRUIT RIPENESS CLASSIFIER - PREDICTION HISTORY")
    print("="*70)
    print("Student: Maria Paula Salazar Agudelo")
    print("Course: AI Minor - Personal Challenge")
    print()

    # Check if database exists
    if not os.path.exists(DB_PATH):
        print(f"ERROR: Database not found: {DB_PATH}")
        print()
        print("Please run a prediction first:")
        print("  python scripts/predict.py test_images/apple1.jpg")
        sys.exit(1)

    # If showing specific prediction
    if args.id:
        display_detailed_prediction(args.id)
        return

    # Show summary
    display_prediction_summary()

    # Get predictions based on filters
    if args.source:
        predictions = get_predictions_by_source(args.source)
        predictions = predictions[:args.limit]
    else:
        predictions = get_recent_predictions(args.limit)

    # Filter by correctness
    if args.correct:
        predictions = [p for p in predictions if p['correct'] == 1]
    elif args.wrong:
        predictions = [p for p in predictions if p['correct'] == 0]

    # Display predictions
    print_section("PREDICTION HISTORY")
    display_predictions_table(predictions, show_details=not args.compact)

    # Export if requested
    if args.export:
        print()
        export_to_file(predictions, args.export)

    print()


# ============================================================================
# RUN THE SCRIPT
# ============================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nView interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nERROR: {str(e)}")
        print()
        print("Troubleshooting:")
        print("  1. Check that database exists: predictions.db")
        print("  2. Check that database has predictions")
        raise
