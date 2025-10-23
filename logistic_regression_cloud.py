# -*- coding: utf-8 -*-
"""
Student Name: [Your Name]
Student UT EID: [Your EID]

CS378 - Cloud Computing - Assignment 8
Spark Logistic Regression for Text Classification - CLOUD VERSION

This script implements regularized logistic regression to classify text documents.
Optimized for Google Cloud Dataproc execution with large dataset.
"""

import re
import math
import sys
from pyspark import SparkContext, SparkConf
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.regression import LabeledPoint

# Hyper-parameters
DICT_SIZE = 20000
LEARNING_RATE = 0.01  # Increased for faster convergence
LAMBDA_REG = 0.01  # Low regularization to allow model to learn
MAX_ITERATIONS = 200  # More iterations for better convergence
BATCH_SIZE = 100  # Balanced batch size
DECISION_THRESHOLD = 0.5  # Standard threshold

def extract_doc_info(line):
    """Extract document ID and determine if it's an AU case."""
    match = re.search(r'<doc\s+id\s*=\s*"([^"]+)"', line)
    if match:
        doc_id = match.group(1)
        is_au = 1 if doc_id.startswith('AU') else 0
        return doc_id, is_au
    return None, None

def extract_words(line):
    """Extract all lowercase words from line."""
    # Remove XML tags
    line = re.sub(r'<[^>]+>', ' ', line)
    # Extract words (letters only, lowercase)
    words = re.findall(r'\b[a-z]{2,}\b', line.lower())
    return words

def process_document(line):
    """Process a single document line and return (doc_id, label, words)."""
    doc_id, label = extract_doc_info(line)
    if doc_id is None:
        return None
    words = extract_words(line)
    return (doc_id, label, words)

def words_to_tf_vector(words, dictionary):
    """Convert word list to TF vector using dictionary."""
    tf_vector = [0.0] * len(dictionary)
    if not words:
        return tf_vector
    
    word_count = float(len(words))
    for word in words:
        if word in dictionary:
            idx = dictionary[word]
            tf_vector[idx] += 1.0 / word_count
    
    return tf_vector

def compute_loss(data_rdd, weights_bc, lambda_reg):
    """Compute regularized negative log-likelihood."""
    N = data_rdd.count()
    if N == 0:
        return 0.0
    
    def loss_for_point(point):
        label, features = point
        theta = sum(w * f for w, f in zip(weights_bc.value, features))
        # Clip theta to avoid overflow
        theta = max(min(theta, 700), -700)
        return -label * theta + math.log(1 + math.exp(theta))
    
    data_loss = data_rdd.map(loss_for_point).sum() / N
    reg_loss = lambda_reg * sum(w * w for w in weights_bc.value)
    return data_loss + reg_loss

def compute_gradient(data_rdd, weights_bc, lambda_reg):
    """Compute gradient."""
    N = data_rdd.count()
    dict_size = len(weights_bc.value)
    
    def gradient_for_point(point):
        label, features = point
        theta = sum(w * f for w, f in zip(weights_bc.value, features))
        theta = max(min(theta, 700), -700)  # Clip to avoid overflow
        exp_theta = math.exp(theta)
        factor = exp_theta / (1 + exp_theta)
        return [-features[j] * label + features[j] * factor for j in range(dict_size)]
    
    total_gradient = data_rdd.map(gradient_for_point).reduce(
        lambda a, b: [x + y for x, y in zip(a, b)])
    
    final_gradient = [(g / N + 2 * lambda_reg * w) 
                     for g, w in zip(total_gradient, weights_bc.value)]
    return final_gradient

def main():
    print("="*80)
    print("CS 378 - Assignment 8: Logistic Regression (CLOUD VERSION)")
    print("="*80)
    
    # Data paths (hardcoded - no command line arguments needed)
    train_file = "gs://cs378n/TrainingData.txt"
    test_file = "gs://cs378n/TestingData.txt"
    
    print(f"\nTraining file: {train_file}")
    print(f"Testing file: {test_file}")
    
    # Initialize Spark - let Dataproc configure the cluster settings
    conf = SparkConf().setAppName("LogisticRegression-Cloud")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("WARN")
    
    # Load data from Google Cloud Storage
    print("\n=== Loading Training Data ===")
    lines_rdd = sc.textFile(train_file)
    total_lines = lines_rdd.count()
    print(f"Total training lines loaded: {total_lines}")
    
    # Process ALL training documents
    print("\n=== Processing Training Documents ===")
    train_docs = lines_rdd.map(process_document).filter(lambda x: x is not None).cache()
    num_train = train_docs.count()
    print(f"Total training documents processed: {num_train}")
    
    # Count training classes
    train_au = train_docs.filter(lambda x: x[1] == 1).count()
    train_wiki = train_docs.filter(lambda x: x[1] == 0).count()
    print(f"Training - AU court cases: {train_au}")
    print(f"Training - Wikipedia articles: {train_wiki}")
    
    # Load and process testing data
    print("\n=== Loading Testing Data ===")
    test_lines_rdd = sc.textFile(test_file)
    total_test_lines = test_lines_rdd.count()
    print(f"Total testing lines loaded: {total_test_lines}")
    
    print("\n=== Processing Testing Documents ===")
    test_docs = test_lines_rdd.map(process_document).filter(lambda x: x is not None).cache()
    num_test = test_docs.count()
    print(f"Total testing documents processed: {num_test}")
    
    # Count testing classes
    test_au = test_docs.filter(lambda x: x[1] == 1).count()
    test_wiki = test_docs.filter(lambda x: x[1] == 0).count()
    print(f"Testing - AU court cases: {test_au}")
    print(f"Testing - Wikipedia articles: {test_wiki}")
    
    # Build dictionary from training data only
    print("\n=== Building Dictionary ===")
    train_words = train_docs.flatMap(lambda x: x[2])  # x[2] is words
    word_counts = train_words.map(lambda w: (w, 1)).reduceByKey(lambda a, b: a + b)
    top_words = word_counts.sortBy(lambda x: -x[1]).take(DICT_SIZE)
    dictionary = {word: i for i, (word, count) in enumerate(top_words)}
    dictionary_bc = sc.broadcast(dictionary)
    
    print(f"Dictionary size: {len(dictionary)}")
    if len(dictionary) >= 10:
        print(f"Top 10 words: {[w for w, _ in top_words[:10]]}")
    
    if num_train == 0:
        print("ERROR: No training documents found!")
        sc.stop()
        return
    
    # Convert to TF vectors
    print("\n=== Converting to TF Vectors ===")
    train_data = train_docs.map(
        lambda x: (x[1], words_to_tf_vector(x[2], dictionary_bc.value))
    ).cache()
    
    test_data = test_docs.map(
        lambda x: (x[0], x[1], words_to_tf_vector(x[2], dictionary_bc.value))
    ).cache()
    
    # TASK 1: Full Batch Gradient Descent
    print("\n" + "="*80)
    print("TASK 1: Full Batch Gradient Descent")
    print("="*80)
    
    weights = [0.0] * len(dictionary)
    weights_bc = sc.broadcast(weights)
    
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Regularization: {LAMBDA_REG}")
    print(f"Iterations: {MAX_ITERATIONS}")
    
    for iteration in range(MAX_ITERATIONS):
        loss = compute_loss(train_data, weights_bc, LAMBDA_REG)
        gradient = compute_gradient(train_data, weights_bc, LAMBDA_REG)
        weights = [w - LEARNING_RATE * g for w, g in zip(weights_bc.value, gradient)]
        weights_bc = sc.broadcast(weights)
        
        if iteration % 10 == 0 or iteration == MAX_ITERATIONS - 1:
            print(f"Iteration {iteration:3d}: Loss = {loss:.6f}")
    
    # Top 5 words
    reverse_dict = {v: k for k, v in dictionary_bc.value.items()}
    word_weights = [(reverse_dict[i], weights[i]) for i in range(len(weights))]
    top_5 = sorted(word_weights, key=lambda x: -x[1])[:5]
    
    print("\n=== Top 5 Words (Largest Coefficients) ===")
    for word, weight in top_5:
        print(f"  {word:20s}: {weight:.6f}")
    
    # TASK 2: Mini-Batch Gradient Descent
    print("\n" + "="*80)
    print("TASK 2: Mini-Batch Gradient Descent")
    print("="*80)
    
    au_cases = train_docs.filter(lambda x: x[1] == 1)
    wiki_articles = train_docs.filter(lambda x: x[1] == 0)
    
    au_count = au_cases.count()
    wiki_count = wiki_articles.count()
    
    print(f"AU cases: {au_count}, Wikipedia: {wiki_count}")
    print(f"Batch size per class: {BATCH_SIZE}")
    
    weights2 = [0.0] * len(dictionary)
    
    for iteration in range(MAX_ITERATIONS):
        # Sample balanced batches - same size for both classes
        au_sample = au_cases.sample(False, min(1.0, BATCH_SIZE / max(au_count, 1)), seed=iteration)
        wiki_sample = wiki_articles.sample(False, min(1.0, BATCH_SIZE / max(wiki_count, 1)), seed=iteration)
        
        batch_docs = au_sample.union(wiki_sample)
        batch_data = batch_docs.map(
            lambda x: (x[1], words_to_tf_vector(x[2], dictionary_bc.value))
        ).cache()
        
        weights_bc2 = sc.broadcast(weights2)
        loss = compute_loss(batch_data, weights_bc2, LAMBDA_REG)
        gradient = compute_gradient(batch_data, weights_bc2, LAMBDA_REG)
        weights2 = [w - LEARNING_RATE * g for w, g in zip(weights2, gradient)]
        
        batch_data.unpersist()
        
        if iteration % 20 == 0 or iteration == MAX_ITERATIONS - 1:
            print(f"Iteration {iteration:3d}: Negative LLH = {loss:.6f}")
    
    word_weights2 = [(reverse_dict[i], weights2[i]) for i in range(len(weights2))]
    top_5_2 = sorted(word_weights2, key=lambda x: -x[1])[:5]
    
    print("\n=== Top 5 Words (Mini-Batch) ===")
    for word, weight in top_5_2:
        print(f"  {word:20s}: {weight:.6f}")
    
    # TASK 3: Spark MLlib
    print("\n" + "="*80)
    print("TASK 3: Spark MLlib Logistic Regression")
    print("="*80)
    
    # For cloud execution, we can use the full dictionary
    print(f"Using full dictionary size: {DICT_SIZE}")
    
    labeled_points = train_data.map(lambda x: LabeledPoint(x[0], x[1])).cache()
    print(f"Training with LBFGS, iterations=100, features={DICT_SIZE}")
    
    try:
        model = LogisticRegressionWithLBFGS.train(
            labeled_points,
            iterations=100,
            regParam=LAMBDA_REG,
            regType='l2'
        )
        
        train_preds = labeled_points.map(lambda p: (p.label, model.predict(p.features)))
        train_correct = train_preds.filter(lambda x: x[0] == x[1]).count()
        train_total = labeled_points.count()
        accuracy = train_correct / train_total
        
        print(f"Training samples: {train_total}")
        print(f"Correct predictions: {train_correct}")
        print(f"Training accuracy: {accuracy:.4f}")
        
        # Top 5 words from MLlib model
        mllib_weights = model.weights.toArray()
        mllib_word_weights = [(reverse_dict[i], mllib_weights[i]) for i in range(len(mllib_weights))]
        mllib_top_5 = sorted(mllib_word_weights, key=lambda x: -x[1])[:5]
        
        print("\n=== Top 5 Words (MLlib Model) ===")
        for word, weight in mllib_top_5:
            print(f"  {word:20s}: {weight:.6f}")
            
    except Exception as e:
        print(f"MLlib training failed: {str(e)}")
    
    finally:
        labeled_points.unpersist()
    
    # TASK 4: Evaluation
    print("\n" + "="*80)
    print("TASK 4: Model Evaluation")
    print("="*80)
    
    weights_final_bc = sc.broadcast(weights2)
    
    def predict(features):
        theta = sum(w * f for w, f in zip(weights_final_bc.value, features))
        prob = 1.0 / (1.0 + math.exp(-min(max(theta, -700), 700)))
        return 1 if prob >= DECISION_THRESHOLD else 0
    
    predictions = test_data.map(lambda x: (x[0], x[1], predict(x[2]))).cache()
    
    tp = predictions.filter(lambda x: x[1] == 1 and x[2] == 1).count()
    fp = predictions.filter(lambda x: x[1] == 0 and x[2] == 1).count()
    tn = predictions.filter(lambda x: x[1] == 0 and x[2] == 0).count()
    fn = predictions.filter(lambda x: x[1] == 1 and x[2] == 0).count()
    
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 0.0001)
    accuracy_test = (tp + tn) / max(tp + fp + tn + fn, 1)
    
    print("\n=== Confusion Matrix ===")
    print(f"True Positives:  {tp:6d}")
    print(f"False Positives: {fp:6d}")
    print(f"True Negatives:  {tn:6d}")
    print(f"False Negatives: {fn:6d}")
    
    print("\n=== Metrics ===")
    print(f"Accuracy:  {accuracy_test:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Decision Threshold: {DECISION_THRESHOLD}")
    
    print(f"\n=== Prediction Summary ===")
    total_predicted_au = tp + fp
    total_actual_au = tp + fn
    print(f"Total predicted as AU: {total_predicted_au}")
    print(f"Total actual AU cases: {total_actual_au}")
    print(f"Total test samples: {tp + fp + tn + fn}")
    
    # Analyze false positives
    false_positives = predictions.filter(lambda x: x[1] == 0 and x[2] == 1).take(5)
    
    if false_positives:
        print(f"\n=== False Positive Analysis (Sample) ===")
        print(f"Showing up to 5 false positives:\n")
        
        for i, (doc_id, true_label, pred_label) in enumerate(false_positives, 1):
            print(f"False Positive #{i}: Document ID = {doc_id}")
    
    print("\n" + "="*80)
    print("ALL TASKS COMPLETED")
    print("="*80)
    print(f"\nFinal F1 Score: {f1:.4f}")
    
    sc.stop()

if __name__ == "__main__":
    main()
