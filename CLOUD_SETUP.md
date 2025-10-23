# Google Cloud Dataproc Setup Guide

This guide will help you run the logistic regression code on Google Cloud Platform using Dataproc (managed Spark clusters).

## Prerequisites

1. **Google Cloud Account** with billing enabled
2. **Google Cloud SDK (gcloud CLI)** installed on your local machine
   - Installation: https://cloud.google.com/sdk/docs/install
3. **Active GCP Project** with Dataproc API enabled

## Quick Start (Automated)

### Step 1: Configure the Script

Edit `run_on_dataproc.sh` and update these variables:

```bash
PROJECT_ID="your-project-id"           # Your GCP project ID
BUCKET_NAME="your-bucket-name"         # GCS bucket name (will be created)
DATA_FILE="TrainingData.txt"           # Or SmallTrainingData.txt for testing
```

### Step 2: Upload Data to Google Cloud Storage

```bash
# Create a bucket (if it doesn't exist)
gsutil mb gs://your-bucket-name

# Create directories
gsutil mkdir gs://your-bucket-name/data
gsutil mkdir gs://your-bucket-name/code

# Upload the training data
gsutil cp TrainingData.txt gs://your-bucket-name/data/
# OR for testing:
gsutil cp SmallTrainingData.txt gs://your-bucket-name/data/
```

### Step 3: Run the Script

```bash
# Make the script executable
chmod +x run_on_dataproc.sh

# Run it
./run_on_dataproc.sh
```

The script will:
1. Upload your Python code to GCS
2. Create a Dataproc cluster (or use existing)
3. Submit the Spark job
4. Display results
5. Optionally delete the cluster

---

## Manual Setup (Step-by-Step)

If you prefer manual control or the automated script doesn't work:

### Step 1: Enable Required APIs

```bash
gcloud services enable dataproc.googleapis.com
gcloud services enable compute.googleapis.com
gcloud services enable storage.googleapis.com
```

### Step 2: Set Your Project

```bash
gcloud config set project YOUR_PROJECT_ID
```

### Step 3: Create a Google Cloud Storage Bucket

```bash
# Replace with your unique bucket name
BUCKET_NAME="your-unique-bucket-name"
gsutil mb gs://$BUCKET_NAME

# Create directories
gsutil mkdir gs://$BUCKET_NAME/data
gsutil mkdir gs://$BUCKET_NAME/code
```

### Step 4: Upload Files to GCS

```bash
# Upload the cloud version of the script
gsutil cp logistic_regression_cloud.py gs://$BUCKET_NAME/code/

# Upload the training data
# For LARGE dataset:
gsutil cp TrainingData.txt gs://$BUCKET_NAME/data/

# For SMALL dataset (testing):
gsutil cp SmallTrainingData.txt gs://$BUCKET_NAME/data/
```

### Step 5: Create a Dataproc Cluster

```bash
CLUSTER_NAME="spark-lr-cluster"
REGION="us-central1"

gcloud dataproc clusters create $CLUSTER_NAME \
    --region=$REGION \
    --zone=us-central1-a \
    --master-machine-type=n1-standard-4 \
    --master-boot-disk-size=100GB \
    --num-workers=4 \
    --worker-machine-type=n1-standard-4 \
    --worker-boot-disk-size=100GB \
    --image-version=2.1-debian11
```

**Cluster Configuration Explained:**
- **Master:** 1 node (n1-standard-4: 4 vCPUs, 15 GB RAM)
- **Workers:** 4 nodes (n1-standard-4: 4 vCPUs, 15 GB RAM each)
- **Total:** 20 vCPUs, 75 GB RAM
- **Cost:** ~$1-2 per hour (delete when done!)

### Step 6: Submit the Spark Job

```bash
# For LARGE dataset:
gcloud dataproc jobs submit pyspark \
    gs://$BUCKET_NAME/code/logistic_regression_cloud.py \
    --cluster=$CLUSTER_NAME \
    --region=$REGION \
    --properties=spark.executor.memory=4g,spark.driver.memory=4g \
    -- gs://$BUCKET_NAME/data/TrainingData.txt

# For SMALL dataset (testing):
gcloud dataproc jobs submit pyspark \
    gs://$BUCKET_NAME/code/logistic_regression_cloud.py \
    --cluster=$CLUSTER_NAME \
    --region=$REGION \
    --properties=spark.executor.memory=4g,spark.driver.memory=4g \
    -- gs://$BUCKET_NAME/data/SmallTrainingData.txt
```

### Step 7: Monitor the Job

You can monitor the job in several ways:

**Option 1: Web Console**
1. Go to: https://console.cloud.google.com/dataproc/jobs
2. Select your project and region
3. Click on your job to see logs and status

**Option 2: Command Line**
```bash
# List recent jobs
gcloud dataproc jobs list --region=$REGION

# Get job details (replace JOB_ID with actual ID)
gcloud dataproc jobs describe JOB_ID --region=$REGION

# View job output
gcloud dataproc jobs wait JOB_ID --region=$REGION
```

**Option 3: SSH to Master Node**
```bash
gcloud compute ssh $CLUSTER_NAME-m --zone=us-central1-a
```

### Step 8: Delete the Cluster (IMPORTANT!)

**Don't forget to delete your cluster to avoid charges!**

```bash
gcloud dataproc clusters delete $CLUSTER_NAME --region=$REGION
```

---

## Cluster Size Recommendations

### For Small Dataset (SmallTrainingData.txt - 3442 docs)
- **Workers:** 2-4 nodes
- **Machine Type:** n1-standard-2 or n1-standard-4
- **Expected Runtime:** 5-10 minutes
- **Cost:** ~$0.50-$1.00 total

### For Large Dataset (TrainingData.txt - 1.3M docs)
- **Workers:** 4-8 nodes
- **Machine Type:** n1-standard-4 or n1-highmem-4
- **Expected Runtime:** 30-60 minutes
- **Cost:** ~$2-$5 total

### Configuration Options

```bash
# SMALL CLUSTER (budget-friendly)
--master-machine-type=n1-standard-2 \
--num-workers=2 \
--worker-machine-type=n1-standard-2

# MEDIUM CLUSTER (recommended for large dataset)
--master-machine-type=n1-standard-4 \
--num-workers=4 \
--worker-machine-type=n1-standard-4

# LARGE CLUSTER (faster processing)
--master-machine-type=n1-highmem-4 \
--num-workers=8 \
--worker-machine-type=n1-highmem-4
```

---

## Troubleshooting

### Issue: "Permission Denied" or "Access Denied"

**Solution:**
```bash
# Ensure you're authenticated
gcloud auth login

# Set the correct project
gcloud config set project YOUR_PROJECT_ID

# Check your permissions
gcloud projects get-iam-policy YOUR_PROJECT_ID
```

### Issue: "Quota Exceeded"

**Solution:**
- Reduce cluster size (fewer workers or smaller machine types)
- Request quota increase in GCP Console
- Try a different region

### Issue: Job Fails with OutOfMemoryError

**Solution:**
```bash
# Increase executor memory
--properties=spark.executor.memory=8g,spark.driver.memory=8g

# OR use larger machine types
--worker-machine-type=n1-highmem-4
```

### Issue: "Bucket Not Found"

**Solution:**
```bash
# Verify bucket exists
gsutil ls

# Create bucket if needed
gsutil mb gs://your-bucket-name

# Verify files are uploaded
gsutil ls gs://your-bucket-name/data/
gsutil ls gs://your-bucket-name/code/
```

### Issue: Job Hangs or Takes Too Long

**Solution:**
- Check job logs in GCP Console
- Verify input file is correct format (one document per line)
- Increase cluster size
- Check Spark UI for executor status

---

## Viewing Results

### Option 1: Job Output in Console
Results are displayed in the job output log in GCP Console

### Option 2: Save Results to GCS
Modify the script to save results:

```python
# At the end of main(), add:
output_path = "gs://your-bucket-name/results/output.txt"
with open("/tmp/results.txt", "w") as f:
    f.write(f"F1 Score: {f1:.4f}\n")
    # ... write other results

# Then use gsutil to copy
import subprocess
subprocess.run(["gsutil", "cp", "/tmp/results.txt", output_path])
```

### Option 3: Download Logs
```bash
# List jobs
gcloud dataproc jobs list --region=$REGION

# Download specific job output
gcloud dataproc jobs describe JOB_ID --region=$REGION > job_output.txt
```

---

## Cost Estimation

**Dataproc Pricing (as of 2024):**
- Compute costs: ~$0.30-$0.50 per vCPU per hour
- Dataproc premium: $0.01 per vCPU per hour
- Storage: $0.02 per GB per month

**Example Costs:**

| Configuration | vCPUs | Runtime | Estimated Cost |
|--------------|-------|---------|----------------|
| Small (2 workers) | 12 | 10 min | $0.50 |
| Medium (4 workers) | 20 | 30 min | $2.00 |
| Large (8 workers) | 36 | 20 min | $3.00 |

**To minimize costs:**
1. ✅ Delete clusters immediately after use
2. ✅ Use preemptible workers: `--num-preemptible-workers=2`
3. ✅ Test with small dataset first
4. ✅ Use appropriate cluster size (don't over-provision)

---

## Files Summary

- **`logistic_regression_cloud.py`** - Cloud-optimized Python script
- **`run_on_dataproc.sh`** - Automated setup and execution script
- **`CLOUD_SETUP.md`** - This guide
- **`logistic_regression_simple.py`** - Local version (for reference)

---

## Next Steps

1. ✅ Test with SmallTrainingData.txt first
2. ✅ Verify results match local execution
3. ✅ Run with full TrainingData.txt
4. ✅ Compare results between Tasks 1-4
5. ✅ Document findings in your README.md
6. ✅ **DELETE YOUR CLUSTER!**

---

## Additional Resources

- [Dataproc Documentation](https://cloud.google.com/dataproc/docs)
- [PySpark on Dataproc](https://cloud.google.com/dataproc/docs/tutorials/python-job)
- [Dataproc Pricing](https://cloud.google.com/dataproc/pricing)
- [GCP Free Tier](https://cloud.google.com/free)

---

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review GCP Console logs
3. Verify all files are uploaded correctly
4. Test with smaller dataset first
5. Check GCP quotas and permissions

**Remember to delete your cluster when done to avoid unexpected charges!**
