#!/bin/bash
# Script to run logistic regression on Google Cloud Dataproc
# Usage: ./run_on_dataproc.sh

# =============================================================================
# CONFIGURATION - UPDATE THESE VALUES
# =============================================================================

# Your Google Cloud Project ID
PROJECT_ID="your-project-id"

# Google Cloud Storage bucket (without gs:// prefix)
BUCKET_NAME="your-bucket-name"

# Cluster configuration
CLUSTER_NAME="spark-lr-cluster"
REGION="us-central1"
ZONE="us-central1-a"

# Worker configuration
NUM_WORKERS=4
WORKER_MACHINE_TYPE="n1-standard-4"
WORKER_DISK_SIZE="100GB"

# Master configuration
MASTER_MACHINE_TYPE="n1-standard-4"
MASTER_DISK_SIZE="100GB"

# Data file
DATA_FILE="TrainingData.txt"  # Change to SmallTrainingData.txt for testing

# =============================================================================
# DO NOT MODIFY BELOW THIS LINE
# =============================================================================

echo "=========================================="
echo "Google Cloud Dataproc Logistic Regression"
echo "=========================================="
echo ""

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "ERROR: gcloud CLI not found. Please install Google Cloud SDK."
    echo "Visit: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Set the project
echo "Setting project to: $PROJECT_ID"
gcloud config set project $PROJECT_ID

# Upload the Python script to GCS
echo ""
echo "Uploading Python script to gs://$BUCKET_NAME/code/"
gsutil cp logistic_regression_cloud.py gs://$BUCKET_NAME/code/

# Check if data file exists in GCS
echo ""
echo "Checking for data file: gs://$BUCKET_NAME/data/$DATA_FILE"
if ! gsutil ls gs://$BUCKET_NAME/data/$DATA_FILE &> /dev/null; then
    echo "WARNING: Data file not found in GCS."
    echo "Please upload your data file first:"
    echo "  gsutil cp $DATA_FILE gs://$BUCKET_NAME/data/"
    echo ""
    read -p "Do you want to continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if cluster exists
echo ""
echo "Checking if cluster '$CLUSTER_NAME' exists..."
if gcloud dataproc clusters describe $CLUSTER_NAME --region=$REGION &> /dev/null; then
    echo "Cluster '$CLUSTER_NAME' already exists. Using existing cluster."
else
    # Create the cluster
    echo "Creating Dataproc cluster: $CLUSTER_NAME"
    echo "  Region: $REGION"
    echo "  Workers: $NUM_WORKERS x $WORKER_MACHINE_TYPE"
    echo "  Master: $MASTER_MACHINE_TYPE"
    echo ""
    
    gcloud dataproc clusters create $CLUSTER_NAME \
        --region=$REGION \
        --zone=$ZONE \
        --master-machine-type=$MASTER_MACHINE_TYPE \
        --master-boot-disk-size=$MASTER_DISK_SIZE \
        --num-workers=$NUM_WORKERS \
        --worker-machine-type=$WORKER_MACHINE_TYPE \
        --worker-boot-disk-size=$WORKER_DISK_SIZE \
        --image-version=2.1-debian11 \
        --project=$PROJECT_ID
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to create cluster"
        exit 1
    fi
    
    echo "Cluster created successfully!"
fi

# Submit the Spark job
echo ""
echo "=========================================="
echo "Submitting Spark job to cluster..."
echo "=========================================="
echo ""

gcloud dataproc jobs submit pyspark \
    gs://$BUCKET_NAME/code/logistic_regression_cloud.py \
    --cluster=$CLUSTER_NAME \
    --region=$REGION \
    --properties=spark.executor.memory=4g,spark.driver.memory=4g \
    -- gs://$BUCKET_NAME/data/$DATA_FILE

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Job completed successfully!"
    echo "=========================================="
else
    echo ""
    echo "ERROR: Job failed"
    exit 1
fi

# Ask if user wants to delete the cluster
echo ""
read -p "Do you want to delete the cluster now? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Deleting cluster: $CLUSTER_NAME"
    gcloud dataproc clusters delete $CLUSTER_NAME --region=$REGION --quiet
    echo "Cluster deleted."
else
    echo "Cluster left running. Remember to delete it later to avoid charges:"
    echo "  gcloud dataproc clusters delete $CLUSTER_NAME --region=$REGION"
fi

echo ""
echo "Done!"
