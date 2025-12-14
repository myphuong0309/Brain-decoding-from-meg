set -e

# Step 1: Compute global statistics
echo ""
echo "[STEP 1] Computing global statistics..."
python utils/compute_global_stats.py --data_path "./assets/data"
echo "Global statistics computed!"

# Step 2: Train model
echo ""
echo "[STEP 2] Training model..."
bash train.sh
echo "Model training completed!"

# Step 3: Evaluate model on test set
echo ""
echo "[STEP 3] Evaluating model on test set (Sherlock1 session 12)..."
bash evaluate.sh
echo "Test evaluation completed!"