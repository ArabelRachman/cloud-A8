# 1. Student Name: Arabel Rachman
#     Student EID: agr2999

# 2. Student Name: Aidan Liu
#    Student EID: al5445 
  
# Course Name: CS378

# Unique Number 1314

# Date Created: 10/23/2025


import re
import math
from pyspark import SparkContext, SparkConf
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.regression import LabeledPoint

# parameters
DICT_SIZE = 20000
LEARNING_RATE = 0.0001
LAMBDA = 1.0
MAX_ITER = 100
BATCH_SIZE = 1024

def get_doc_info(line):
    match = re.search(r'<doc\s+id\s*=\s*"([^"]+)"', line)
    if match:
        doc_id = match.group(1)
        label = 1 if doc_id.startswith('AU') else 0
        return doc_id, label
    return None, None

def get_words(line):
    line = re.sub(r'<[^>]+>', ' ', line)
    words = re.findall(r'\b[a-z]{2,}\b', line.lower())
    return words

def process_doc(line):
    doc_id, label = get_doc_info(line)
    if doc_id is None:
        return None
    words = get_words(line)
    return (doc_id, label, words)

def to_tf_vector(words, word_dict):
    vec = [0.0] * len(word_dict)
    if not words:
        return vec
    
    total = float(len(words))
    for w in words:
        if w in word_dict:
            vec[word_dict[w]] += 1.0 / total
    return vec

def calc_loss(data, weights_bc, lam):
    n = data.count()
    if n == 0:
        return 0.0
    
    def loss_pt(pt):
        y, x = pt
        theta = sum(w * f for w, f in zip(weights_bc.value, x))
        theta = max(min(theta, 700), -700)
        return -y * theta + math.log(1 + math.exp(theta))
    
    data_loss = data.map(loss_pt).sum() / n
    reg = lam * sum(w * w for w in weights_bc.value)
    return data_loss + reg

def calc_gradient(data, weights_bc, lam):
    n = data.count()
    d = len(weights_bc.value)
    
    def grad_pt(pt):
        y, x = pt
        theta = sum(w * f for w, f in zip(weights_bc.value, x))
        theta = max(min(theta, 700), -700)
        sig = math.exp(theta) / (1 + math.exp(theta))
        return [-x[j] * y + x[j] * sig for j in range(d)]
    
    grad_sum = data.map(grad_pt).reduce(lambda a, b: [x + y for x, y in zip(a, b)])
    return [(g / n + 2 * lam * w) for g, w in zip(grad_sum, weights_bc.value)]

def main():
    train_file = "gs://cs378n/TrainingData.txt"
    test_file = "gs://cs378n/TestingData.txt"
    
    conf = SparkConf().setAppName("LogisticRegression")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("WARN")
    
    # load training data
    print("\nLoading training data...")
    lines = sc.textFile(train_file)
    train_docs = lines.map(process_doc).filter(lambda x: x is not None).cache()
    n_train = train_docs.count()
    print(f"Training documents: {n_train}")
    
    # load test data
    print("Loading test data...")
    test_lines = sc.textFile(test_file)
    test_docs = test_lines.map(process_doc).filter(lambda x: x is not None).cache()
    n_test = test_docs.count()
    print(f"Test documents: {n_test}")
    
    # build dictionary
    print(f"\nBuilding dictionary (top {DICT_SIZE} words)...")
    words = train_docs.flatMap(lambda x: x[2])
    word_counts = words.map(lambda w: (w, 1)).reduceByKey(lambda a, b: a + b)
    top = word_counts.sortBy(lambda x: -x[1]).take(DICT_SIZE)
    word_dict = {w: i for i, (w, c) in enumerate(top)}
    dict_bc = sc.broadcast(word_dict)
    print(f"Dictionary size: {len(word_dict)}")
    
    # convert to vectors
    train_data = train_docs.map(lambda x: (x[1], to_tf_vector(x[2], dict_bc.value))).cache()
    test_data = test_docs.map(lambda x: (x[0], x[1], to_tf_vector(x[2], dict_bc.value))).cache()
    
    # Task 1: batch gradient descent
    print("\n" + "="*70)
    print("TASK 1: Batch Gradient Descent")
    print("="*70)
    
    w = [0.0] * len(word_dict)
    w_bc = sc.broadcast(w)
    
    for i in range(MAX_ITER):
        loss = calc_loss(train_data, w_bc, LAMBDA)
        grad = calc_gradient(train_data, w_bc, LAMBDA)
        w = [wj - LEARNING_RATE * gj for wj, gj in zip(w_bc.value, grad)]
        w_bc = sc.broadcast(w)
        
        if i % 10 == 0 or i == MAX_ITER - 1:
            print(f"Iteration {i}: Loss = {loss:.6f}")
    
    # top 5 words
    idx_to_word = {v: k for k, v in dict_bc.value.items()}
    word_weights = [(idx_to_word[i], w[i]) for i in range(len(w))]
    top5 = sorted(word_weights, key=lambda x: -x[1])[:5]
    
    print("\nTop 5 words (largest coefficients):")
    for word, weight in top5:
        print(f"  {word}: {weight:.6f}")
    
    # Task 2: mini-batch gradient descent
    print("\n" + "="*70)
    print("TASK 2: Mini-Batch Gradient Descent")
    print("="*70)
    
    au_docs = train_docs.filter(lambda x: x[1] == 1)
    wiki_docs = train_docs.filter(lambda x: x[1] == 0)
    n_au = au_docs.count()
    n_wiki = wiki_docs.count()
    
    print(f"AU cases: {n_au}, Wiki: {n_wiki}")
    
    w2 = [0.0] * len(word_dict)
    
    for i in range(MAX_ITER):
        # sample batches
        au_batch = au_docs.sample(False, min(1.0, BATCH_SIZE / max(n_au, 1)), seed=i)
        wiki_batch = wiki_docs.sample(False, min(1.0, BATCH_SIZE / max(n_wiki, 1)), seed=i)
        
        batch = au_batch.union(wiki_batch)
        batch_data = batch.map(lambda x: (x[1], to_tf_vector(x[2], dict_bc.value))).cache()
        
        w2_bc = sc.broadcast(w2)
        loss = calc_loss(batch_data, w2_bc, LAMBDA)
        grad = calc_gradient(batch_data, w2_bc, LAMBDA)
        w2 = [wj - LEARNING_RATE * gj for wj, gj in zip(w2, grad)]
        
        batch_data.unpersist()
        
        if i % 10 == 0 or i == MAX_ITER - 1:
            print(f"Iteration {i}: Loss = {loss:.6f}")
    
    word_weights2 = [(idx_to_word[i], w2[i]) for i in range(len(w2))]
    top5_2 = sorted(word_weights2, key=lambda x: -x[1])[:5]
    
    print("\nTop 5 words (mini-batch):")
    for word, weight in top5_2:
        print(f"  {word}: {weight:.6f}")
    
    # Task 3: Spark MLlib
    print("\n" + "="*70)
    print("TASK 3: Spark MLlib")
    print("="*70)
    
    labeled = train_data.map(lambda x: LabeledPoint(x[0], x[1])).cache()
    
    try:
        model = LogisticRegressionWithLBFGS.train(labeled, iterations=100, regParam=LAMBDA, regType='l2')
        
        preds = labeled.map(lambda p: (p.label, model.predict(p.features)))
        correct = preds.filter(lambda x: x[0] == x[1]).count()
        total = labeled.count()
        acc = correct / total
        
        print(f"Training accuracy: {acc:.4f}")
        
        # top 5 words
        mllib_w = model.weights.toArray()
        mllib_words = [(idx_to_word[i], mllib_w[i]) for i in range(len(mllib_w))]
        mllib_top5 = sorted(mllib_words, key=lambda x: -x[1])[:5]
        
        print("\nTop 5 words (MLlib):")
        for word, weight in mllib_top5:
            print(f"  {word}: {weight:.6f}")
    except Exception as e:
        print(f"MLlib failed: {str(e)}")
    finally:
        labeled.unpersist()
    
    # Task 4: evaluation
    print("\n" + "="*70)
    print("TASK 4: Evaluation")
    print("="*70)
    
    w_final = sc.broadcast(w2)
    
    def predict(x):
        theta = sum(w * f for w, f in zip(w_final.value, x))
        prob = 1.0 / (1.0 + math.exp(-min(max(theta, -700), 700)))
        return 1 if prob >= 0.5 else 0
    
    preds = test_data.map(lambda x: (x[0], x[1], predict(x[2]))).cache()
    
    tp = preds.filter(lambda x: x[1] == 1 and x[2] == 1).count()
    fp = preds.filter(lambda x: x[1] == 0 and x[2] == 1).count()
    tn = preds.filter(lambda x: x[1] == 0 and x[2] == 0).count()
    fn = preds.filter(lambda x: x[1] == 1 and x[2] == 0).count()
    
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 0.001)
    acc = (tp + tn) / max(tp + fp + tn + fn, 1)
    
    print(f"\nAccuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    
    # false positives
    fps = preds.filter(lambda x: x[1] == 0 and x[2] == 1).take(3)
    
    if fps:
        print(f"\nFalse positives (showing {len(fps)}):")
        for i, (doc_id, _, _) in enumerate(fps, 1):
            print(f"  {i}. Document ID: {doc_id}")
    
    print(f"\nDone. F1 Score: {f1:.4f}")
    
    sc.stop()

if __name__ == "__main__":
    main()
