# How to View Training Progress with TensorBoard

**Student:** Maria Paula Salazar Agudelo
**Course:** AI Minor - Personal Challenge
**Purpose:** Beginner's guide to monitoring model training in real-time

---

## What is TensorBoard?

**Simple explanation:** TensorBoard is like a "dashboard" that shows how your model is learning while it trains.

**What you can see:**
- 📈 Training accuracy going up over time
- 📉 Loss (errors) going down over time
- 🖼️ Model architecture diagram
- 📊 Weight distributions (histograms)
- ⏱️ Training speed (time per epoch)

**Why use it:**
- See if training is working (accuracy should increase)
- Spot problems early (if accuracy doesn't improve)
- Know when to stop training (when accuracy plateaus)
- Create graphs for your project report

---

## How to Start TensorBoard

### Method 1: Using the Convenience Script (Recommended)

I created a script that starts TensorBoard automatically:

```bash
# Make sure you're in the project folder
cd /mnt/c/Users/maria/Desktop/fruit-classifier-AI-project

# Run the script
bash scripts/start_tensorboard.sh
```

**What happens:**
1. Script starts TensorBoard server
2. Server reads training logs from `logs/fit/` folder
3. Dashboard becomes available at: http://localhost:6006

### Method 2: Manual Command

If you prefer to run it manually:

```bash
# Activate virtual environment first
source venv/bin/activate

# Start TensorBoard (point it to logs directory)
tensorboard --logdir logs/fit --port 6006
```

---

## Opening the Dashboard

After starting TensorBoard:

1. **Open your web browser** (Chrome, Firefox, Edge, etc.)
2. **Go to:** http://localhost:6006
3. **You should see** the TensorBoard dashboard with tabs:
   - **Scalars** - Accuracy and loss graphs
   - **Graphs** - Model architecture
   - **Distributions** - Weight histograms
   - **Histograms** - Weight changes over time

---

## Understanding the Graphs

### 1. Training Accuracy (Scalars Tab)

**What it shows:** How well the model learns the training images

**What to look for:**
- ✅ **Good:** Line goes UP from ~40% → 85%+
- ⚠️ **Problem:** Line stays flat (model not learning)
- ⚠️ **Problem:** Line goes down (something wrong)

**Example interpretation:**
```
Epoch 1:  45% accuracy - Model just started
Epoch 5:  72% accuracy - Model learning patterns
Epoch 10: 85% accuracy - Model doing well
Epoch 15: 87% accuracy - Almost done
Epoch 20: 88% accuracy - Training complete
```

### 2. Validation Accuracy (Scalars Tab)

**What it shows:** How well the model works on NEW images (not in training set)

**What to look for:**
- ✅ **Good:** Close to training accuracy (85-88%)
- ⚠️ **Overfitting:** Much lower than training accuracy
  - Example: Training 95%, Validation 70% = Overfitting!
  - Model memorized training images instead of learning

**Why this matters:**
- Validation accuracy = REAL performance
- This is what your model will get on new fruit photos

### 3. Training Loss (Scalars Tab)

**What it shows:** How "wrong" the predictions are

**What to look for:**
- ✅ **Good:** Line goes DOWN from ~2.0 → 0.3
- Lower = better (means fewer errors)

**Example interpretation:**
```
Epoch 1:  Loss = 2.1  (Many mistakes)
Epoch 5:  Loss = 0.8  (Getting better)
Epoch 10: Loss = 0.3  (Good predictions)
Epoch 15: Loss = 0.2  (Very good)
Epoch 20: Loss = 0.15 (Excellent)
```

### 4. Validation Loss (Scalars Tab)

**What it shows:** Errors on NEW images

**What to look for:**
- ✅ **Good:** Goes down like training loss
- ⚠️ **Overfitting:** Starts going UP while training loss goes down
  - This means: Model is memorizing, not learning!

---

## What Good Training Looks Like

**Graphs you WANT to see:**

1. **Training Accuracy:** Steady increase 45% → 85%+
2. **Validation Accuracy:** Following training accuracy closely
3. **Training Loss:** Steady decrease 2.0 → 0.3
4. **Validation Loss:** Decreasing along with training loss

**Visual example:**
```
Accuracy (%)
    |
100 |                                    ___training
 90 |                            ___----
 80 |                    ___----        ___validation
 70 |            ___----        ___----
 60 |    ___----        ___----
 50 |----        ___----
    |_________________________ Epochs
    0   5   10   15   20
```

---

## What BAD Training Looks Like

### Problem 1: Not Learning

**Symptoms:**
- Accuracy stays at ~11% (random guessing for 9 classes)
- Loss doesn't decrease

**Possible causes:**
- Learning rate too low
- Model architecture wrong
- Data not loading properly

### Problem 2: Overfitting

**Symptoms:**
- Training accuracy: 95%
- Validation accuracy: 65%
- Validation loss INCREASES after epoch 10

**What this means:**
- Model memorized training images
- Doesn't generalize to new images
- Need more data augmentation or regularization

### Problem 3: Underfitting

**Symptoms:**
- Training accuracy stuck at 60%
- Validation accuracy also 60%
- Both plateau early

**What this means:**
- Model too simple for the task
- Need more complex model or more epochs

---

## Real-Time Monitoring During Training

**While your model trains:**

1. Start training in one terminal:
   ```bash
   python scripts/train.py
   ```

2. Start TensorBoard in another terminal:
   ```bash
   bash scripts/start_tensorboard.sh
   ```

3. Open browser to http://localhost:6006

4. **Refresh the page** every few epochs to see updates

**TIP:** Click the refresh icon (⟳) in TensorBoard to update graphs

---

## Stopping TensorBoard

When you're done viewing:

1. Go to the terminal where TensorBoard is running
2. Press `Ctrl + C`
3. Terminal shows: "TensorBoard interrupted"

**Note:** This only stops the viewer, not your training!

---

## Common Issues and Solutions

### Issue 1: "No dashboards are active for the current data set"

**Cause:** No training logs yet

**Solution:**
1. Train your model first: `python scripts/train.py`
2. This creates logs in `logs/fit/`
3. Then start TensorBoard

### Issue 2: "Address already in use (port 6006)"

**Cause:** TensorBoard already running

**Solution:**
```bash
# Find and kill the process
pkill -f tensorboard

# Or use different port
tensorboard --logdir logs/fit --port 6007
```

### Issue 3: Graphs not updating

**Cause:** Browser cache

**Solution:**
1. Click refresh icon in TensorBoard (⟳)
2. Or hard refresh browser (Ctrl + Shift + R)

### Issue 4: Can't access http://localhost:6006

**Cause:** TensorBoard not running or wrong port

**Solution:**
1. Check terminal - TensorBoard should say "TensorBoard 2.x at http://localhost:6006"
2. Try http://127.0.0.1:6006 instead
3. Check firewall settings

---

## Taking Screenshots for Your Report

**For your project documentation:**

1. Train your model completely
2. Open TensorBoard
3. Go to "Scalars" tab
4. Take screenshot of accuracy graphs
5. Go to "Graphs" tab
6. Take screenshot of model architecture

**These screenshots prove:**
- You trained the model yourself
- Model performance (accuracy)
- Model architecture used

---

## Advanced: Comparing Multiple Training Runs

**If you train multiple times:**

TensorBoard shows all runs from `logs/fit/` folder:

```
logs/fit/
├── run_20250130_140000/  (First training)
├── run_20250130_150000/  (Second training)
└── run_20250130_160000/  (Third training)
```

**You can:**
- Compare which hyperparameters worked best
- See if changes improved accuracy
- Track experimentation progress

---

## Summary Checklist

Before submitting your project, verify:

- [ ] Training accuracy reached 85%+
- [ ] Validation accuracy close to training accuracy
- [ ] Loss decreased steadily
- [ ] No signs of overfitting
- [ ] Screenshots of TensorBoard graphs saved
- [ ] Model architecture diagram captured

---

## Quick Reference

**Start TensorBoard:**
```bash
bash scripts/start_tensorboard.sh
```

**Open dashboard:**
```
http://localhost:6006
```

**Stop TensorBoard:**
```
Ctrl + C
```

**Logs location:**
```
logs/fit/
```

---

**Student:** Maria Paula Salazar Agudelo
**Course:** AI Minor - Personal Challenge
**Date:** 2025

**Remember:** TensorBoard is a tool to UNDERSTAND your training, not just watch it. Look for patterns, spot problems early, and use the graphs in your project report!
