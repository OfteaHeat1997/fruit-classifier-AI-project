# Deployment Guide - Fruit Ripeness Classifier

## What We Built Today

```
fruit-classifier-AI-project/
├── models/
│   ├── fruit_classifier.keras      # Original model (13 MB)
│   └── fruit_classifier.tflite     # Mobile-ready model (2.7 MB) ← NEW!
│
├── webapp/                          ← NEW FOLDER!
│   ├── app.py                      # Flask API server (backend)
│   └── templates/
│       └── index.html              # Simple web frontend
│
└── scripts/
    └── convert_to_tflite.py        # Conversion script ← NEW!
```

---

## How to Run the Web App

### Step 1: Start the server
```bash
# In WSL terminal:
cd /mnt/c/Users/maria/Desktop/fruit-classifier-AI-project
source venv/bin/activate  # or use your virtualenv
python webapp/app.py
```

### Step 2: Open in browser
```
http://localhost:5000
```

### Step 3: Test it!
- Drag & drop a fruit image
- See the AI prediction!

---

## Understanding the Architecture

```
┌────────────────────────────────────────────────────────────┐
│                    YOUR APP STRUCTURE                       │
└────────────────────────────────────────────────────────────┘

    FRONTEND                          BACKEND
    (What user sees)                  (AI runs here)

    ┌─────────────┐                  ┌─────────────┐
    │             │    HTTP POST     │             │
    │  Browser    │ ──────────────►  │  Flask API  │
    │  (HTML)     │   /api/predict   │  (Python)   │
    │             │                  │      │      │
    │             │  ◄──────────────  │      ▼      │
    │  Shows:     │    JSON result   │  TFLite     │
    │  "Fresh     │                  │  Model      │
    │   Apple"    │                  │             │
    └─────────────┘                  └─────────────┘

    Later: React/Expo                Same backend!
    (replace HTML)                   (no changes needed)
```

---

## API Reference

### 1. Health Check
```bash
GET http://localhost:5000/api/health
```
Response:
```json
{
  "status": "healthy",
  "model": "fruit_classifier.tflite",
  "classes": ["freshapples", "freshbanana", ...]
}
```

### 2. Make Prediction
```bash
POST http://localhost:5000/api/predict
Content-Type: multipart/form-data
Body: image file
```
Response:
```json
{
  "success": true,
  "prediction": {
    "class": "freshapples",
    "label": "Fresh Apple",
    "confidence": 94.5
  },
  "all_predictions": {
    "freshapples": 94.5,
    "rottenapples": 3.2,
    ...
  }
}
```

---

## Next Steps: React Native / Expo Mobile App

When you're ready to build a mobile app, here's the plan:

### Option A: Expo (Easiest for beginners)

```bash
# 1. Install Node.js first (from nodejs.org)

# 2. Create Expo app
npx create-expo-app FruitClassifierApp
cd FruitClassifierApp

# 3. Install camera library
npx expo install expo-image-picker

# 4. Run on your phone
npx expo start
# Scan QR code with Expo Go app
```

### Option B: React Native CLI (More control)

```bash
npx react-native init FruitClassifierApp
```

### Mobile App Structure (for later)

```
FruitClassifierApp/
├── App.js                 # Main app file
├── src/
│   ├── screens/
│   │   └── HomeScreen.js  # Camera + results
│   ├── components/
│   │   └── ResultCard.js  # Show prediction
│   └── api/
│       └── predict.js     # Call Flask API
```

### Example React Native Code (for later)

```javascript
// src/api/predict.js
const API_URL = 'http://YOUR_COMPUTER_IP:5000';

export async function predictFruit(imageUri) {
  const formData = new FormData();
  formData.append('image', {
    uri: imageUri,
    type: 'image/jpeg',
    name: 'photo.jpg',
  });

  const response = await fetch(`${API_URL}/api/predict`, {
    method: 'POST',
    body: formData,
  });

  return response.json();
}
```

---

## Learning Path Recommendation

```
You are here!
     │
     ▼
┌────────────────┐
│ 1. HTML+Flask  │  ← Simple, understand basics
│    (DONE!)     │
└───────┬────────┘
        │
        ▼
┌────────────────┐
│ 2. Learn React │  ← Web fundamentals
│    basics      │     (reactjs.org tutorial)
└───────┬────────┘
        │
        ▼
┌────────────────┐
│ 3. React       │  ← Build web version
│    (browser)   │
└───────┬────────┘
        │
        ▼
┌────────────────┐
│ 4. Expo/React  │  ← Mobile app!
│    Native      │
└────────────────┘
```

---

## Commands Summary

```bash
# Start web server
python webapp/app.py

# Convert model to TFLite (already done)
python scripts/convert_to_tflite.py

# Test API from command line
curl http://localhost:5000/api/health

# Test prediction from command line
curl -X POST -F "image=@test_image.jpg" http://localhost:5000/api/predict
```

---

## Troubleshooting

### "Cannot connect to server"
- Make sure Flask is running: `python webapp/app.py`
- Check the URL: `http://localhost:5000`

### "Module not found"
```bash
pip install flask flask-cors pillow tensorflow
```

### "Model not found"
- Check that `models/fruit_classifier.tflite` exists
- Run `python scripts/convert_to_tflite.py` if not

---

## Files Created Today

| File | Purpose |
|------|---------|
| `scripts/convert_to_tflite.py` | Convert Keras → TFLite |
| `models/fruit_classifier.tflite` | Mobile-optimized model |
| `webapp/app.py` | Flask API server |
| `webapp/templates/index.html` | Web frontend |
| `DEPLOYMENT_GUIDE.md` | This guide |

---

*Created for Maria Paula's AI Portfolio Project*
