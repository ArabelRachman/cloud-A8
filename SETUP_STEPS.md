# Google Cloud Dataproc Setup - Step by Step

## What You Need to Provide

Before we begin, I need the following information from you:

1. **Google Cloud Project ID** - Your GCP project ID (e.g., "my-project-12345")
2. **Google Cloud Storage Bucket Name** - A unique bucket name (e.g., "cs378-yourname-bucket")
3. **Confirm you have:**
   - Google Cloud account with billing enabled
   - `gcloud` CLI installed (or willing to install it)
   - Access to the large TrainingData.txt file

---

## Step 1: Install Google Cloud SDK (if needed)

**Do you have gcloud installed?** Check with:
```bash
gcloud --version
```

**If not installed:**
- **Linux/Mac**: 
  ```bash
  curl https://sdk.cloud.google.com | bash
  exec -l $SHELL
  ```
- **Windows**: Download from https://cloud.google.com/sdk/docs/install

---

## Step 2: Authenticate and Set Project

I need you to run these commands and tell me if they work:

```bash
# Login to Google Cloud
gcloud auth login

# Set your project (REPLACE with your project ID)
gcloud config set project YOUR_PROJECT_ID

# Verify it's set
gcloud config get-value project
```

**Please provide:**
- Your project ID: _______________

---

## Step 3: Enable Required APIs

Run these commands to enable necessary APIs:

```bash
gcloud services enable dataproc.googleapis.com
gcloud services enable compute.googleapis.com  
gcloud services enable storage.googleapis.com
```

Let me know if you get any errors.

---

## Step 4: Create Google Cloud Storage Bucket

```bash
# REPLACE 'your-bucket-name' with a unique name
gsutil mb gs://your-bucket-name

# Verify it was created
gsutil ls
```

**Please provide:**
- Your bucket name: _______________

---

## Step 5: Upload Files to Google Cloud Storage

Once you provide the bucket name, I'll give you the exact commands to run.

You'll need to upload:
1. The Python script (logistic_regression_cloud.py)
2. The large TrainingData.txt file

**Do you have the TrainingData.txt file?**
- [ ] Yes, I have it locally
- [ ] No, I need to download it first
- [ ] If no, where is it? (provide URL/location)

---

## Step 6: Create and Submit Job

After files are uploaded, I'll provide you with:
1. Commands to create the Dataproc cluster
2. Command to submit the Spark job
3. How to monitor progress
4. Command to delete the cluster (important!)

---

## Quick Summary of What We'll Do

1. ✅ Upload `logistic_regression_cloud.py` to `gs://your-bucket/code/`
2. ✅ Upload `TrainingData.txt` to `gs://your-bucket/data/`
3. ✅ Create Dataproc cluster (4 workers, n1-standard-4)
4. ✅ Submit Spark job to cluster
5. ✅ Monitor and view results
6. ✅ **Delete cluster** (to avoid charges!)

---

## Estimated Costs

- **Cluster runtime**: ~30-60 minutes
- **Cost**: ~$2-5 total
- **IMPORTANT**: Remember to delete the cluster after the job completes!

---

## Ready to Start?

Please provide:
1. ✅ Your GCP Project ID: _______________
2. ✅ Your desired bucket name: _______________
3. ✅ Confirm gcloud is installed: [ ] Yes / [ ] No
4. ✅ Do you have TrainingData.txt? [ ] Yes / [ ] No / [ ] Need download URL

Once you provide these, I'll give you the exact commands to run!
