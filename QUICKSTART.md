# Quick Start: Running on Google Cloud Dataproc

## 🚀 Fast Track (5 minutes setup)

### 1. Install Google Cloud SDK (if not already installed)
```bash
# For Linux/Mac
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# For Windows
# Download from: https://cloud.google.com/sdk/docs/install
```

### 2. Login and Set Project
```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

### 3. Create Storage Bucket
```bash
# Replace with your unique bucket name
BUCKET="your-cs378-bucket"
gsutil mb gs://$BUCKET
```

### 4. Upload Files
```bash
cd /u/arabel/cloud-computing/A8

# Upload code
gsutil cp logistic_regression_cloud.py gs://$BUCKET/code/

# Upload data (choose one):
# For LARGE dataset:
gsutil cp TrainingData.txt gs://$BUCKET/data/

# For SMALL dataset (testing):
gsutil cp SmallTrainingData.txt gs://$BUCKET/data/
```

### 5. Edit and Run the Setup Script
```bash
# Edit the script to set your project ID and bucket name
nano run_on_dataproc.sh
# Update these lines:
#   PROJECT_ID="your-project-id"
#   BUCKET_NAME="your-cs378-bucket"

# Run it!
./run_on_dataproc.sh
```

---

## 📋 What the Script Does

1. ✅ Uploads your Python code to Google Cloud Storage
2. ✅ Creates a Dataproc cluster with 4 workers
3. ✅ Submits your Spark job to the cluster
4. ✅ Shows the output/results
5. ✅ Asks if you want to delete the cluster (say YES to avoid charges!)

---

## 💰 Cost: ~$2-3 for complete run with large dataset

**IMPORTANT:** Delete the cluster after use!

---

## 🔍 Alternative: Manual Run

If the script doesn't work, run these commands manually:

```bash
# Set variables
PROJECT_ID="your-project-id"
BUCKET="your-bucket-name"
CLUSTER="spark-lr-cluster"
REGION="us-central1"

# Create cluster (takes ~2 minutes)
gcloud dataproc clusters create $CLUSTER \
    --region=$REGION \
    --zone=us-central1-a \
    --master-machine-type=n1-standard-4 \
    --num-workers=4 \
    --worker-machine-type=n1-standard-4 \
    --image-version=2.1-debian11

# Submit job (takes ~30-60 minutes for large dataset)
gcloud dataproc jobs submit pyspark \
    gs://$BUCKET/code/logistic_regression_cloud.py \
    --cluster=$CLUSTER \
    --region=$REGION \
    --properties=spark.executor.memory=4g,spark.driver.memory=4g \
    -- gs://$BUCKET/data/TrainingData.txt

# DELETE CLUSTER (don't forget!)
gcloud dataproc clusters delete $CLUSTER --region=$REGION
```

---

## 📊 Expected Output

You'll see all 4 tasks complete:
- **Task 1:** Full batch gradient descent results
- **Task 2:** Mini-batch gradient descent results  
- **Task 3:** MLlib LBFGS results (with full 20K features!)
- **Task 4:** Evaluation metrics (Confusion Matrix, F1 Score)

---

## 🐛 Common Issues

### "Permission denied"
```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

### "Quota exceeded"
- Try fewer workers: `--num-workers=2`
- Or different region: `--region=us-east1`

### "Bucket not found"
```bash
gsutil mb gs://your-bucket-name
gsutil ls  # verify it exists
```

### Job takes forever
- Check in GCP Console: https://console.cloud.google.com/dataproc
- Click on your cluster → Jobs tab → View logs

---

## 📁 Files You Need

- ✅ `logistic_regression_cloud.py` - Main script (created ✓)
- ✅ `run_on_dataproc.sh` - Automated setup (created ✓)
- ✅ `CLOUD_SETUP.md` - Detailed guide (created ✓)
- ✅ `QUICKSTART.md` - This file (created ✓)
- 📥 `TrainingData.txt` - Large dataset (you need to download/upload)

---

## 🎯 Checklist

- [ ] Install gcloud CLI
- [ ] Login: `gcloud auth login`
- [ ] Create GCS bucket
- [ ] Upload Python script to GCS
- [ ] Upload data file to GCS
- [ ] Edit `run_on_dataproc.sh` with your project ID & bucket
- [ ] Run `./run_on_dataproc.sh`
- [ ] Wait for results
- [ ] **DELETE THE CLUSTER!**

---

## 📚 More Help

- Detailed guide: See `CLOUD_SETUP.md`
- Troubleshooting: See `CLOUD_SETUP.md` → Troubleshooting section
- GCP Console: https://console.cloud.google.com/dataproc

---

**Questions?** Check the CLOUD_SETUP.md file for detailed explanations!
