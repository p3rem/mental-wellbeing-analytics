# Code Review Report - Mental Health Project

## 🔴 Critical Errors

### 1. **simple_setup.py - XGBoost Trained on Wrong Target**
**Location:** `simple_setup.py`, lines 68-71

**Issue:** The XGBoost model is being trained on the stress target (`y_train`) instead of a mood target. This means the mood prediction model is actually predicting stress, not mood.

```python
# Current (WRONG):
xgb_mood = xgb.XGBClassifier(n_estimators=100, random_state=42)
xgb_mood.fit(X_train, y_train)  # Using stress target!

# Should be:
# Create mood target first
df['mood_severity_numeric'] = pd.cut(df['mood_score'], bins=[0, 5, 7, 10], labels=[0, 1, 2])
y_mood = df['mood_severity_numeric']
X_train_mood, X_test_mood, y_train_mood, y_test_mood = train_test_split(...)
xgb_mood.fit(X_train_mood, y_train_mood)
```

**Impact:** Mood predictions in the web app will be incorrect.

---

### 2. **simple_setup.py - Missing Mood Target Variable**
**Location:** `simple_setup.py`, line 30

**Issue:** The script creates `high_stress` target but never creates the `mood_severity_numeric` target that XGBoost needs.

**Fix Required:** Add mood target creation similar to `preprocessing.py`:
```python
df['mood_severity_numeric'] = pd.cut(
    df['mood_score'], 
    bins=[0, 5, 7, 10], 
    labels=[0, 1, 2]
).astype(int)
```

---

## ⚠️ Major Issues

### 3. **Inconsistent Prediction Methods**
**Location:** `app.py` vs `src/app.py`

**Issue:** Two different implementations:
- `app.py` (line 300-329): Uses DataFrames with column names
- `src/app.py` (line 145-179): Uses lists

**Problem:** This inconsistency can cause runtime errors if the wrong app is used.

**Recommendation:** Standardize on one approach (DataFrame approach is more robust).

---

### 4. **File Path Dependencies**
**Location:** Multiple files

**Issue:** Hardcoded relative paths that will break if scripts are run from different directories:
- `app.py` line 245: `pd.read_csv('digital_habits_vs_mental_health.csv')`
- `app.py` line 252: `joblib.load('models/rf_stress.joblib')`
- `src/app.py` line 83: Same issue
- All other files have similar hardcoded paths

**Fix:** Use `os.path` or `pathlib` to construct paths relative to script location:
```python
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'digital_habits_vs_mental_health.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
```

---

### 5. **Potential Division by Zero**
**Location:** Multiple files

**Issue:** `screen_sleep_ratio` calculation can divide by zero:
- `app.py` line 282: `screen_sleep_ratio = screen_time / sleep_hours if sleep_hours > 0 else 0`
- `src/app.py` line 126: Same check exists
- `preprocessing.py` line 64: `self.df['screen_sleep_ratio'] = self.df['screen_time_hours'] / self.df['sleep_hours']` - **NO CHECK!**

**Fix:** Add check in `preprocessing.py`:
```python
self.df['screen_sleep_ratio'] = np.where(
    self.df['sleep_hours'] > 0,
    self.df['screen_time_hours'] / self.df['sleep_hours'],
    0
)
```

---

### 6. **Feature List Mismatch Risk**
**Location:** `app.py` line 305-306

**Issue:** Creating DataFrames with specific column names from `models['feature_lists']`, but if the feature creation doesn't match exactly, this will fail.

**Current Code:**
```python
stress_features_df = pd.DataFrame([features], columns=models['feature_lists']['stress'])
```

**Risk:** If `features` dict doesn't contain all keys in `feature_lists['stress']`, or has extra keys, this could cause issues.

**Recommendation:** Validate feature dictionary matches expected features.

---

## 🟡 Minor Issues

### 7. **Unused Import**
**Location:** `app.py` line 11
- `os` is imported but never used

### 8. **Missing Error Handling**
**Location:** `app.py` line 245
- `load_data()` doesn't handle file not found errors gracefully
- Should check if file exists before trying to read

### 9. **Inconsistent Model Loading**
**Location:** `app.py` vs `src/app.py`
- Different error messages and handling
- `app.py` suggests `simple_setup.py`, `src/app.py` suggests `main_analysis.py`

### 10. **Missing Validation**
**Location:** `predict_mental_health` functions
- No validation that user input values are within expected ranges
- No check that all required features are present

### 11. **Hardcoded Values**
**Location:** Multiple files
- Cluster descriptions hardcoded (lines 569-574 in `app.py`)
- Magic numbers throughout (e.g., `contamination=0.1`, `n_clusters=4`)

---

## 📋 Recommendations

### High Priority Fixes:
1. ✅ Fix XGBoost target in `simple_setup.py`
2. ✅ Add mood target creation in `simple_setup.py`
3. ✅ Fix division by zero in `preprocessing.py`
4. ✅ Standardize file paths using `os.path` or `pathlib`

### Medium Priority:
5. Standardize prediction methods between `app.py` and `src/app.py`
6. Add input validation in prediction functions
7. Add better error handling for file operations

### Low Priority:
8. Remove unused imports
9. Add type hints for better code documentation
10. Consider using configuration files for magic numbers

---

## ✅ What's Working Well

1. **Good code organization** - Clear separation of concerns
2. **Comprehensive documentation** - README and INSTRUCTIONS are detailed
3. **Error handling in model loading** - Try-except blocks present
4. **Feature engineering** - Well thought out derived features
5. **Model diversity** - Good mix of supervised and unsupervised learning

---

## 🔧 Quick Fixes Needed

### Fix 1: simple_setup.py - Add Mood Target
```python
# After line 30, add:
df['mood_severity_numeric'] = pd.cut(
    df['mood_score'], 
    bins=[0, 5, 7, 10], 
    labels=[0, 1, 2]
).astype(int)
```

### Fix 2: simple_setup.py - Train XGBoost on Mood
```python
# Replace lines 68-71 with:
X_mood = df[features]
y_mood = df['mood_severity_numeric']
X_train_mood, X_test_mood, y_train_mood, y_test_mood = train_test_split(
    X_mood, y_mood, test_size=0.2, random_state=42, stratify=y_mood
)
xgb_mood = xgb.XGBClassifier(n_estimators=100, random_state=42)
xgb_mood.fit(X_train_mood, y_train_mood)
```

### Fix 3: preprocessing.py - Fix Division by Zero
```python
# Replace line 64 with:
self.df['screen_sleep_ratio'] = np.where(
    self.df['sleep_hours'] > 0,
    self.df['screen_time_hours'] / self.df['sleep_hours'],
    0
)
```

---

**Report Generated:** $(date)
**Files Reviewed:** 8 Python files
**Critical Issues:** 2
**Major Issues:** 4
**Minor Issues:** 5

