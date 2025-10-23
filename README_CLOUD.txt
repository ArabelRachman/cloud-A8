================================================================================
GOOGLE CLOUD DATAPROC - READY TO DEPLOY
================================================================================

✅ Everything is prepared and ready to run on Google Cloud!

Files Ready:
------------
1. logistic_regression_cloud.py - Optimized for cloud with:
   - Stratified train/test split (both classes in test set)
   - Improved hyperparameters (learning rate: 0.01, lambda: 0.01)
   - 200 iterations for better convergence
   - Balanced mini-batches for Task 2
   - Full 20K dictionary for Task 3 (no memory limits on cloud!)
   - Enhanced evaluation metrics

2. run_on_dataproc.sh - Automated setup script
3. SETUP_STEPS.md - Step-by-step guide (READ THIS FIRST!)
4. QUICKSTART.md - Quick reference
5. CLOUD_SETUP.md - Detailed documentation

================================================================================
WHAT I NEED FROM YOU
================================================================================

Please provide the following information:

1. Google Cloud Project ID: _______________________
   (Find it at: https://console.cloud.google.com/)

2. Google Cloud Storage Bucket Name: _______________________
   (Choose a unique name like: cs378-yourname-bucket)

3. Do you have gcloud CLI installed?
   Run: gcloud --version
   [ ] Yes - version: ___________
   [ ] No - I'll help you install it

4. Do you have the large TrainingData.txt file?
   [ ] Yes, I have it at: ___________
   [ ] No, please provide download URL

================================================================================
NEXT STEPS - Once You Provide the Above
================================================================================

I will give you EXACT commands to:

Step 1: Upload files to Google Cloud Storage
   gsutil cp logistic_regression_cloud.py gs://YOUR-BUCKET/code/
   gsutil cp TrainingData.txt gs://YOUR-BUCKET/data/

Step 2: Create Dataproc cluster
   gcloud dataproc clusters create spark-lr-cluster ...

Step 3: Submit Spark job
   gcloud dataproc jobs submit pyspark ...

Step 4: Monitor progress and view results

Step 5: Delete cluster (IMPORTANT - to avoid charges!)
   gcloud dataproc clusters delete spark-lr-cluster ...

================================================================================
EXPECTED RESULTS
================================================================================

On the LARGE dataset (~1.3 million documents), you should see:

- Task 1: Loss convergence over 200 iterations
- Task 2: Mini-batch GD with balanced sampling
- Task 3: MLlib training with full 20K features
- Task 4: Meaningful F1 score with both classes in test set

Runtime: 30-60 minutes
Cost: ~$2-5 total

================================================================================
READY?
================================================================================

Please reply with:
1. Your Project ID
2. Your Bucket Name
3. gcloud status (installed or not)
4. TrainingData.txt status (have it or need URL)

Then I'll give you the exact commands to copy & paste!

================================================================================

1. cloud-computing-476020
2. i dont need to upload it to my own google cloud, the large data is already hosted on google cloud with this link: Training Dataset Large: https://storage.googleapis.com/cs378n/TrainingData.txt
Test Dataset Large: https://storage.googleapis.com/cs378n/TestingData.txt
3. i dont have cli yet just please install it
4. yes i provided the link for test and training large in number 2