set -e

# Step 1: Precache data
echo ""
echo "[STEP 1] Precaching data..."
python utils/folds.py --data_path "./assets/data" --n_splits 10
echo "Data precaching completed!"

# Step 2: Compute global statistics
echo ""
echo "[STEP 2] Computing global statistics..."
python utils/compute_global_stats.py --data_path "./assets/data"
echo "Global statistics computed!"

# Step 3: Train model
echo ""
echo "[STEP 3] Training model..."
bash train.sh
echo "Model training completed!"

# Step 4: Evaluate model on test set
echo ""
echo "[STEP 4] Evaluating model on test set (Sherlock1 sessions 11-12)..."
bash evaluate.sh
echo "Test evaluation completed!"