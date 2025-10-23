#!/bin/bash
# Complete Cloud Deployment Commands
# Copy and paste these commands one section at a time

echo "=========================================="
echo "GOOGLE CLOUD DATAPROC DEPLOYMENT"
echo "Project: cloud-computing-476020"
echo "=========================================="

# STEP 1: Set project
echo ""
echo "=== STEP 1: Setting Project ==="
gcloud config set project cloud-computing-476020

# STEP 2: Enable APIs (if not already enabled)
echo ""
echo "=== STEP 2: Enabling Required APIs ==="
gcloud services enable dataproc.googleapis.com
gcloud services enable compute.googleapis.com

# STEP 3: Upload Python script to Google Cloud Storage
echo ""
echo "=== STEP 3: Uploading Python Script ==="
# The data is already at gs://cs378n/, so we'll use a temporary bucket for the script
# Or we can submit the script directly without uploading (preferred method)

# STEP 4: Create Dataproc Cluster
echo ""
echo "=== STEP 4: Creating Dataproc Cluster ==="
gcloud dataproc clusters create spark-lr-cluster \
    --region=us-central1 \
    --zone=us-central1-a \
    --master-machine-type=n1-standard-4 \
    --master-boot-disk-size=100GB \
    --num-workers=4 \
    --worker-machine-type=n1-standard-4 \
    --worker-boot-disk-size=100GB \
    --image-version=2.1-debian11 \
    --project=cloud-computing-476020

echo ""
echo "Cluster creation takes ~2-3 minutes..."
echo "Waiting for cluster to be ready..."

# STEP 5: Submit Spark Job
echo ""
echo "=== STEP 5: Submitting Spark Job ==="
echo "This will take 30-60 minutes for the large dataset..."

gcloud dataproc jobs submit pyspark \
    logistic_regression_cloud.py \
    --cluster=spark-lr-cluster \
    --region=us-central1 \
    --properties=spark.executor.memory=4g,spark.driver.memory=4g \
    --project=cloud-computing-476020 \
    -- gs://cs378n/TrainingData.txt

# STEP 6: View Results
echo ""
echo "=== Job Completed! ==="
echo "Results should be displayed above"

# STEP 7: DELETE THE CLUSTER (IMPORTANT!)
echo ""
echo "=========================================="
echo "STEP 6: DELETING CLUSTER"
echo "=========================================="
read -p "Delete the cluster now to avoid charges? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    gcloud dataproc clusters delete spark-lr-cluster \
        --region=us-central1 \
        --project=cloud-computing-476020 \
        --quiet
    echo "Cluster deleted successfully!"
else
    echo ""
    echo "WARNING: Remember to delete the cluster manually:"
    echo "  gcloud dataproc clusters delete spark-lr-cluster --region=us-central1 --project=cloud-computing-476020"
fi

echo ""
echo "=========================================="
echo "DEPLOYMENT COMPLETE!"
echo "=========================================="
