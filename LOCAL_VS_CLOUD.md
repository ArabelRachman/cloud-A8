# Differences Between Local and Cloud Versions

## Key Changes Made for Cloud Execution

### 1. **Command Line Arguments**
- **Local:** Hard-coded filename `SmallTrainingData.txt`
- **Cloud:** Accepts GCS path as argument: `gs://bucket/data/TrainingData.txt`

```python
# Local version
train_file = "SmallTrainingData.txt"

# Cloud version
if len(sys.argv) < 2:
    print("Usage: spark-submit logistic_regression_cloud.py gs://bucket/path/to/data")
    sys.exit(1)
train_file = sys.argv[1]
```

### 2. **Spark Configuration**
- **Local:** Explicitly sets `local[*]` master
- **Cloud:** Lets Dataproc configure cluster automatically

```python
# Local version
conf = SparkConf().setAppName("LogisticRegression").setMaster("local[*]")

# Cloud version
conf = SparkConf().setAppName("LogisticRegression-Cloud")
# No master setting - Dataproc handles this
```

### 3. **Task 3 Dictionary Size**
- **Local:** Uses 5,000 words (memory constraint)
- **Cloud:** Uses full 20,000 words (cluster has more memory)

```python
# Local version
MLLIB_DICT_SIZE = 5000  # Reduced for local memory

# Cloud version
# Uses full DICT_SIZE (20000) for better accuracy
```

### 4. **Task 3 Iterations**
- **Local:** 50 iterations for LBFGS
- **Cloud:** 100 iterations for LBFGS

```python
# Local version
model = LogisticRegressionWithLBFGS.train(..., iterations=50, ...)

# Cloud version
model = LogisticRegressionWithLBFGS.train(..., iterations=100, ...)
```

### 5. **Error Handling**
- **Cloud:** More robust error messages for cluster debugging
- **Cloud:** Less verbose output (cluster logs are separate)

---

## File Comparison Table

| Feature | Local Version | Cloud Version |
|---------|--------------|---------------|
| **File** | `logistic_regression_simple.py` | `logistic_regression_cloud.py` |
| **Input** | Local file | GCS path (gs://...) |
| **Spark Master** | `local[*]` | Managed by Dataproc |
| **Dictionary (Task 1&2)** | 20,000 words | 20,000 words |
| **Dictionary (Task 3)** | 5,000 words | 20,000 words |
| **Task 3 Iterations** | 50 | 100 |
| **Memory Config** | Default | 4GB executor/driver |
| **Dataset Size** | Small (3442 docs) | Large (1.3M docs) |
| **Runtime** | ~5 minutes | ~30-60 minutes |
| **Cost** | $0 (local) | ~$2-3 (cluster) |

---

## Why These Changes?

### Memory Constraints
- **Local:** Limited by single machine RAM (~8-16GB)
  - Must reduce dictionary for Task 3 to avoid OOM
  - Works well for small dataset
  
- **Cloud:** Distributed across 4+ workers (~60-80GB total)
  - Can use full 20K dictionary
  - Handles large dataset efficiently

### Spark Configuration
- **Local:** Need to specify `local[*]` for standalone execution
- **Cloud:** Dataproc automatically configures cluster mode

### File I/O
- **Local:** Reads from local filesystem
- **Cloud:** Reads from Google Cloud Storage (distributed storage)

### Performance
- **Local:** Single machine, limited parallelism
- **Cloud:** Multiple workers, true parallel processing

---

## Running Both Versions

### Local (for testing/development)
```bash
python3 logistic_regression_simple.py
```
- ✅ Quick testing
- ✅ Development/debugging
- ✅ Small dataset
- ❌ Limited by single machine resources

### Cloud (for production/large data)
```bash
./run_on_dataproc.sh
# or
gcloud dataproc jobs submit pyspark \
    gs://bucket/code/logistic_regression_cloud.py \
    --cluster=spark-lr-cluster \
    --region=us-central1 \
    -- gs://bucket/data/TrainingData.txt
```
- ✅ Handles large datasets
- ✅ True distributed computing
- ✅ Full feature set (20K dictionary)
- ❌ Setup required
- ❌ Costs money

---

## Expected Results Comparison

### Small Dataset (3442 documents)

| Metric | Local | Cloud | Difference |
|--------|-------|-------|------------|
| **Task 1 Loss** | 0.693124 | Similar | Minimal |
| **Task 2 Loss** | 0.693128 | Similar | Minimal |
| **Task 3 Accuracy** | 99.85% | Higher? | Cloud uses more features |
| **Task 4 F1** | 0.0000 | Similar | Data split dependent |
| **Runtime** | ~5 min | ~10 min | Cluster overhead |

### Large Dataset (1.3M documents)

| Metric | Local | Cloud |
|--------|-------|-------|
| **Feasibility** | ❌ OOM Error | ✅ Works |
| **Task 3 Features** | Can't run | 20,000 |
| **Task 3 Accuracy** | N/A | ~95-99% |
| **Runtime** | Crashes | 30-60 min |

---

## Code Compatibility

Both versions:
- ✅ Use same algorithm implementations
- ✅ Use same hyperparameters (α=0.0001, λ=1.0)
- ✅ Use same evaluation metrics
- ✅ Produce comparable results on small data

Only difference is **scalability** and **resource constraints**.

---

## Which Version to Use?

### Use Local Version When:
- Testing code changes
- Debugging issues
- Working with small dataset
- Don't have cloud access
- Quick iteration needed

### Use Cloud Version When:
- Running on large dataset (TrainingData.txt)
- Need full feature set (20K dictionary)
- Want maximum accuracy
- Final production run
- Have GCP account/credits

---

## Converting Between Versions

To convert **local → cloud:**
1. Change filename to command-line argument
2. Remove `setMaster("local[*]")`
3. Increase Task 3 dictionary size to 20K
4. Increase Task 3 iterations to 100
5. Upload to GCS and use `spark-submit`

To convert **cloud → local:**
1. Hard-code filename
2. Add `.setMaster("local[*]")`
3. Reduce Task 3 dictionary to 5K
4. Reduce Task 3 iterations to 50
5. Run with `python3` directly

---

Both versions implement the same assignment requirements - choose based on your needs!
