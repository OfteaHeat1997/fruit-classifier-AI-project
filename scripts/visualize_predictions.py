"""
Fruit Ripeness Classifier - Prediction Visualization Script
Student: Maria Paula Salazar Agudelo
Course: AI Minor - Personal Challenge

WHAT THIS DOES:
Reads predictions from the database and creates visualizations:
1. Bar chart of predictions per fruit type
2. Accuracy statistics (if true labels available)
3. Confidence distribution histogram
4. Predictions by source (camera/upload/script)
5. Gallery view of predicted images

WHY THIS IS USEFUL:
- See which fruits are tested most
- Analyze model accuracy
- Check confidence levels
- Track testing progress
- Create visuals for project portfolio

HOW TO RUN:
    python scripts/visualize_predictions.py

    Optional arguments:
    --limit N          Show last N predictions (default: 50)
    --show-images      Display images with predictions
    --min-confidence N Only show predictions above N% confidence

REQUIREMENTS:
- Database file: predictions.db
- Python packages: matplotlib, numpy, sqlite3

OUTPUT:
- Displays graphs and statistics
- Optionally shows images with predictions
"""

import os
import sys
import argparse
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Import database helper functions
from db_helper import (
    init_database,
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
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_predictions_per_fruit(save_path=None):
    """
    Create a bar chart showing how many predictions per fruit type.

    WHAT IT DOES:
    1. Queries database to count predictions for each fruit
    2. Creates a bar chart
    3. Shows or saves the chart

    WHY THIS IS USEFUL:
    - See which fruits you've tested most
    - Ensure balanced testing across all fruits
    - Show testing coverage in portfolio

    PARAMETERS:
        save_path (str): Optional path to save figure

    EXAMPLE OUTPUT:
        Bar chart with:
        X-axis: Fruit types (freshapples, rottenbanana, etc.)
        Y-axis: Number of predictions
        Bars: Height = how many times predicted
    """

    print("Creating predictions per fruit chart...")

    # Step 1: Get counts from database
    # Returns: {'freshapples': 45, 'freshbanana': 32, ...}
    counts = count_predictions_per_fruit()

    if not counts:
        print("No predictions in database yet!")
        return

    # Step 2: Prepare data for plotting
    # Sort by fruit name for consistent display
    fruits = sorted(counts.keys())
    values = [counts[fruit] for fruit in fruits]

    # Step 3: Create figure
    plt.figure(figsize=(12, 6))

    # Step 4: Create bar chart
    # WHY? Bar chart is best for comparing counts across categories
    bars = plt.bar(fruits, values, color='steelblue', alpha=0.8, edgecolor='black')

    # Step 5: Add value labels on top of bars
    # WHY? Makes it easy to see exact numbers
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10)

    # Step 6: Labels and title
    plt.xlabel('Fruit Type', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Predictions', fontsize=12, fontweight='bold')
    plt.title('Predictions per Fruit Type', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    # Step 7: Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_confidence_distribution(save_path=None):
    """
    Create a histogram showing distribution of confidence scores.

    WHAT IT DOES:
    1. Gets all confidence scores from database
    2. Creates histogram (bins: 0-10%, 10-20%, ... 90-100%)
    3. Shows how confident the model typically is

    WHY THIS IS USEFUL:
    - High confidence = model is sure of predictions
    - Low confidence = model is uncertain (might be wrong)
    - Good model should have mostly high confidence predictions

    PARAMETERS:
        save_path (str): Optional path to save figure

    EXAMPLE OUTPUT:
        Histogram showing:
        X-axis: Confidence ranges (0-10%, 10-20%, etc.)
        Y-axis: How many predictions in that range
        Most predictions should be in 80-100% range for good model
    """

    print("Creating confidence distribution chart...")

    # Step 1: Get all predictions from database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('SELECT confidence FROM predictions WHERE confidence IS NOT NULL')
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print("No predictions with confidence scores in database!")
        return

    # Step 2: Extract confidence values
    confidences = [row[0] for row in rows]

    # Step 3: Create figure
    plt.figure(figsize=(10, 6))

    # Step 4: Create histogram
    # bins=10 means 10 ranges: 0-10, 10-20, ..., 90-100
    # edgecolor makes bars distinct
    plt.hist(confidences, bins=10, range=(0, 100),
             color='coral', alpha=0.8, edgecolor='black')

    # Step 5: Add statistics
    # Calculate mean and median confidence
    mean_conf = np.mean(confidences)
    median_conf = np.median(confidences)

    # Add vertical lines for mean and median
    plt.axvline(mean_conf, color='red', linestyle='--', linewidth=2,
                label=f'Mean: {mean_conf:.1f}%')
    plt.axvline(median_conf, color='blue', linestyle='--', linewidth=2,
                label=f'Median: {median_conf:.1f}%')

    # Step 6: Labels and title
    plt.xlabel('Confidence (%)', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Predictions', fontsize=12, fontweight='bold')
    plt.title('Confidence Distribution', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    # Step 7: Print statistics
    print(f"  Mean confidence: {mean_conf:.1f}%")
    print(f"  Median confidence: {median_conf:.1f}%")
    print(f"  Min confidence: {min(confidences):.1f}%")
    print(f"  Max confidence: {max(confidences):.1f}%")

    # Step 8: Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_predictions_by_source(save_path=None):
    """
    Create a pie chart showing predictions by source (camera/upload/script).

    WHAT IT DOES:
    1. Counts predictions from each source
    2. Creates pie chart
    3. Shows percentage from each source

    WHY THIS IS USEFUL:
    - Track how you're testing (live camera vs batch scripts)
    - Show variety of testing methods in portfolio
    - Understand testing workflow

    PARAMETERS:
        save_path (str): Optional path to save figure

    EXAMPLE OUTPUT:
        Pie chart showing:
        - Camera: 30% (blue)
        - Upload: 20% (green)
        - Script: 50% (orange)
    """

    print("Creating predictions by source chart...")

    # Step 1: Get predictions from database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Count predictions grouped by source
    cursor.execute('''
        SELECT source, COUNT(*) as count
        FROM predictions
        GROUP BY source
    ''')
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print("No predictions in database yet!")
        return

    # Step 2: Prepare data
    sources = [row[0] for row in rows]
    counts = [row[1] for row in rows]

    # Step 3: Create figure
    plt.figure(figsize=(8, 8))

    # Step 4: Create pie chart
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFD966']  # Soft colors
    explode = [0.05] * len(sources)  # Slightly separate slices

    plt.pie(counts, labels=sources, autopct='%1.1f%%',
            colors=colors, explode=explode,
            startangle=90, textprops={'fontsize': 12})

    # Step 5: Title
    plt.title('Predictions by Source', fontsize=14, fontweight='bold', pad=20)

    # Step 6: Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()

    # Step 7: Print details
    print("\nPredictions by source:")
    for source, count in zip(sources, counts):
        print(f"  {source}: {count} predictions")


def show_accuracy_stats():
    """
    Display accuracy statistics from database.

    WHAT IT DOES:
    1. Counts correct vs total predictions
    2. Calculates accuracy percentage
    3. Displays results

    WHY THIS IS USEFUL:
    - Measure model performance
    - Track improvement over time
    - Show results in portfolio

    NOTE:
        Only works for predictions where the true label is known
        (correct column is NOT NULL)
    """

    print("\n" + "="*70)
    print("ACCURACY STATISTICS")
    print("="*70)

    stats = get_accuracy_stats()

    if stats['total'] == 0:
        print("No predictions with known truth in database yet!")
        print("(Accuracy requires both predicted_label and true_label)")
        return

    print(f"Correct predictions: {stats['correct']}")
    print(f"Total predictions:   {stats['total']}")
    print(f"Accuracy:            {stats['accuracy']:.2f}%")
    print()

    # Interpretation
    if stats['accuracy'] >= 85:
        print("Status: EXCELLENT - Model meets project goal (>= 85%)")
    elif stats['accuracy'] >= 70:
        print("Status: GOOD - Model works well (70-85%)")
    elif stats['accuracy'] >= 50:
        print("Status: ACCEPTABLE - Model needs improvement (50-70%)")
    else:
        print("Status: POOR - Model needs significant work (< 50%)")

    print("="*70)


def show_recent_predictions_table(limit=10):
    """
    Display recent predictions in a table format.

    WHAT IT DOES:
    1. Gets last N predictions from database
    2. Displays in formatted table
    3. Shows timestamp, prediction, confidence, correctness

    PARAMETERS:
        limit (int): How many predictions to show

    WHY THIS IS USEFUL:
    - Quick overview of recent testing
    - Check if predictions are being logged
    - Verify database is working

    EXAMPLE OUTPUT:
        ID  | Timestamp           | Predicted      | Confidence | Correct | Source
        ----+---------------------+----------------+------------+---------+--------
        15  | 2025-01-15 14:30:22 | freshapples    | 92.5%      | YES     | camera
        14  | 2025-01-15 14:25:10 | freshbanana    | 88.3%      | YES     | upload
        13  | 2025-01-15 14:20:05 | rottenapples   | 76.2%      | NO      | script
    """

    print("\n" + "="*70)
    print(f"RECENT PREDICTIONS (last {limit})")
    print("="*70)

    predictions = get_recent_predictions(limit)

    if not predictions:
        print("No predictions in database yet!")
        return

    # Print table header
    print(f"{'ID':<4} | {'Timestamp':<19} | {'Predicted':<15} | {'Conf':<6} | {'Correct':<7} | {'Source':<8}")
    print("-" * 70)

    # Print each prediction
    for pred in predictions:
        # Format fields
        pred_id = pred['id']
        timestamp = pred['timestamp']
        label = pred['predicted_label'][:15]  # Truncate if too long
        conf = f"{pred['confidence']:.1f}%" if pred['confidence'] else "N/A"

        # Format correct status
        if pred['correct'] == 1:
            correct = "YES"
        elif pred['correct'] == 0:
            correct = "NO"
        else:
            correct = "UNKNOWN"

        source = pred['source'] if pred['source'] else "N/A"

        # Print row
        print(f"{pred_id:<4} | {timestamp:<19} | {label:<15} | {conf:<6} | {correct:<7} | {source:<8}")

    print("="*70)


def create_all_visualizations(save_dir=None):
    """
    Create all visualizations at once.

    WHAT IT DOES:
    1. Creates predictions per fruit chart
    2. Creates confidence distribution chart
    3. Creates predictions by source chart
    4. Shows accuracy statistics
    5. Shows recent predictions table

    PARAMETERS:
        save_dir (str): Optional directory to save all figures

    WHY THIS IS USEFUL:
    - Generate all visuals at once
    - Save all figures for portfolio
    - Complete analysis of testing history
    """

    print("\n" + "="*70)
    print("CREATING ALL VISUALIZATIONS")
    print("="*70 + "\n")

    # Create save directory if specified
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # 1. Predictions per fruit
    save_path = os.path.join(save_dir, "predictions_per_fruit.png") if save_dir else None
    plot_predictions_per_fruit(save_path)

    # 2. Confidence distribution
    save_path = os.path.join(save_dir, "confidence_distribution.png") if save_dir else None
    plot_confidence_distribution(save_path)

    # 3. Predictions by source
    save_path = os.path.join(save_dir, "predictions_by_source.png") if save_dir else None
    plot_predictions_by_source(save_path)

    # 4. Accuracy stats
    show_accuracy_stats()

    # 5. Recent predictions table
    show_recent_predictions_table(10)

    print("\n" + "="*70)
    print("ALL VISUALIZATIONS COMPLETE")
    print("="*70)


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """
    Main function for visualization script.

    COMMAND LINE USAGE:
        python scripts/visualize_predictions.py [options]

    OPTIONS:
        --limit N           Show last N predictions in table
        --save-dir PATH     Save all figures to this directory
        --fruit-chart       Only show predictions per fruit chart
        --confidence-chart  Only show confidence distribution
        --source-chart      Only show predictions by source chart
        --stats             Only show statistics
        --table             Only show predictions table
    """

    # Setup argument parser
    parser = argparse.ArgumentParser(
        description='Visualize predictions from database',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--limit', type=int, default=50,
                       help='Number of recent predictions to show in table (default: 50)')
    parser.add_argument('--save-dir', type=str,
                       help='Directory to save all figures')
    parser.add_argument('--fruit-chart', action='store_true',
                       help='Only show predictions per fruit chart')
    parser.add_argument('--confidence-chart', action='store_true',
                       help='Only show confidence distribution')
    parser.add_argument('--source-chart', action='store_true',
                       help='Only show predictions by source chart')
    parser.add_argument('--stats', action='store_true',
                       help='Only show accuracy statistics')
    parser.add_argument('--table', action='store_true',
                       help='Only show predictions table')

    args = parser.parse_args()

    print("="*70)
    print("FRUIT RIPENESS CLASSIFIER - PREDICTION VISUALIZATION")
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

    # If specific chart requested, show only that
    if args.fruit_chart:
        plot_predictions_per_fruit(
            os.path.join(args.save_dir, "predictions_per_fruit.png") if args.save_dir else None
        )
    elif args.confidence_chart:
        plot_confidence_distribution(
            os.path.join(args.save_dir, "confidence_distribution.png") if args.save_dir else None
        )
    elif args.source_chart:
        plot_predictions_by_source(
            os.path.join(args.save_dir, "predictions_by_source.png") if args.save_dir else None
        )
    elif args.stats:
        show_accuracy_stats()
    elif args.table:
        show_recent_predictions_table(args.limit)
    else:
        # Show all visualizations
        create_all_visualizations(args.save_dir)

    print()


# ============================================================================
# RUN THE SCRIPT
# ============================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nVisualization interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nERROR: {str(e)}")
        print()
        print("Troubleshooting:")
        print("  1. Check that database exists: predictions.db")
        print("  2. Make sure matplotlib is installed: pip install matplotlib")
        print("  3. Check that database has predictions")
        raise
