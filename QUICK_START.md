# Quick Start Guide

Stop overthinking it. Here's what you need to know:

---

## 🪟 **Windows Users**

### First Time Setup (5 minutes):
```
Double-click: SETUP_WINDOWS_VENV.bat
```
Wait for it to install packages. Done.

### Running the Project:
```
Double-click: RUN_TRAINING_WINDOWS.bat  (for training)
Double-click: RUN_JUPYTER_WINDOWS.bat   (for notebooks)
```

---

## 🐧 **PyCharm/Linux Users**

### You're Already Set Up!
PyCharm created a working venv at:
```
/home/paula/.virtualenvs/fruit-classifier-AI-project
```

All packages are installed. Just hit the "Run" button in PyCharm.

---

## 📋 **What Files Actually Matter**

Keep these:
- `README.md` - Original project info
- `TRAINING_GUIDE.md` - Detailed training instructions
- `HOW_TO_VIEW_TRAINING_DASHBOARD.md` - Dashboard setup
- `QUICK_START.md` - This file

Everything else is just code and data.

---

## ❓ **Still Getting Errors?**

1. **"ModuleNotFoundError"** → You didn't activate the virtual environment
   - Windows: Use the batch files
   - PyCharm: It auto-activates, just click Run

2. **"The system cannot find the path"** →
   - Windows: Run `SETUP_WINDOWS_VENV.bat` first

3. **"Package not installed"** →
   - Check you're using the right environment
   - Windows needs `venv_windows`
   - PyCharm uses its own venv (already set up)

---

That's it. No more confusion.
