# AI Agent Prompt: Build Hourly Bitcoin Direction Prediction Model

You are an AI assistant tasked with rebuilding a Bitcoin price direction prediction model. Your goal is to create a machine learning system that predicts whether Bitcoin's price will go UP or DOWN over the **next 1 hour** (instead of the current 5-minute prediction).

## Current Situation

- Existing notebook: `bitcoin_predict_by_input.ipynb`
- Current model: HistGradientBoostingClassifier using 5-minute prediction horizon
- Current features: 13 technical indicators (return_1, momentum_5, momentum_10, close_vs_ma5, close_vs_ma10, volatility_10, rsi, body, upper_wick, lower_wick, volume, volume_delta, trade_count)
- Data source: Alpaca CryptoHistoricalDataClient (BTC/USD minute bars, 60 days history)

## Your Task

### Step 1: Understand Current Architecture

Read the existing notebook structure to understand:

- How data is fetched from Alpaca API
- How features are engineered using the `build_features()` function
- How the HistGradientBoostingClassifier model is trained and evaluated

### Step 2: Modify Feature Engineering for Hourly Prediction

Update the `build_features()` function to:

- Change `HORIZON_MIN` from 5 to 60 (1 hour = 60 minutes)
- Keep all 13 technical indicators the same (they still provide useful information)
- Change the target variable from:
    - OLD: `df["target_5m"] = (df["close"].shift(-HORIZON_MIN) > df["close"]).astype(int)`
    - NEW: `df["target_1h"] = (df["close"].shift(-HORIZON_MIN) > df["close"]).astype(int)` where HORIZON_MIN=60
- This creates a binary target: 1 if price goes UP in the next hour, 0 if DOWN

### Step 3: Retrain the Model

- Use the exact same HistGradientBoostingClassifier hyperparameters:
    - learning_rate=0.05
    - max_depth=6
    - max_iter=300
    - l2_regularization=0.0
    - random_state=42
- Train on 80% historical data (time-based split, no shuffle)
- Evaluate on 20% test data using: accuracy, ROC AUC, confusion matrix, classification report

### Step 4: Update Prediction Cells

Modify the two prediction cells (latest row + manual price override) to:

- Display the current time in New York timezone **rounded to the nearest hour** (not 5-minute interval)
- Example: 14:37 → 14:00, 15:43 → 15:00
- Keep prediction format the same: "UP/DOWN (P(UP)=XX%)"

### Step 5: Validation

- Ensure model trains without errors
- Check that accuracy is >50% (baseline for binary classification is 50%)
- Verify predictions output is reasonable

## Files to Modify

- **Main file**: `bitcoin_predict_by_input.ipynb`
    - Cell 3 (or equivalent): Change HORIZON_MIN = 60
    - Feature engineering cell: Verify target calculation now uses 60-minute lookahead
    - Model training cell: Re-run with new features
    - Both prediction cells: Update timestamp rounding to hourly

## Expected Output

- Rebuilt notebook with hourly prediction model
- All cells execute without errors
- Model accuracy reported for test set
- Predictions display current hour in NY timezone (e.g., "14:00", "15:00")

## Do NOT

- Change the list of 13 features
- Change the model type (stay with HistGradientBoostingClassifier)
- Use shuffled time splits (maintain temporal order)
- Remove evaluation metrics (keep accuracy, ROC AUC, confusion matrix)

## Success Criteria

1. ✅ HORIZON_MIN changed to 60 consistently throughout
2. ✅ Target variable correctly predicts 1-hour ahead direction
3. ✅ Model trains on 80/20 time-split without errors
4. ✅ Accuracy displayed in evaluation cell
5. ✅ Timestamp displayed as hourly NY time (HH:00)
6. ✅ Manual price override still works with new model
