// 1. Student Name: Arabel Rachman
   Student EID: agr2999

// 2. Student Name: Aidan Liu
   Student EID: al5445 

// Course Name: CS378

// Unique Number 1314

// Date Created: 10/23/2025

// small dataset
task 1:
  applicant           : 0.000049
  tribunal            : 0.000036
  respondent          : 0.000026
  appellant           : 0.000023
  visa                : 0.000016

task 2: 
AU cases: 60, Wikipedia: 2667
Batch size per class: 1024
Iteration   0: Negative LLH = 0.693147
Iteration  10: Negative LLH = 0.692944
Iteration  20: Negative LLH = 0.692747
Iteration  30: Negative LLH = 0.692558
Iteration  40: Negative LLH = 0.692386
Iteration  50: Negative LLH = 0.692203
Iteration  60: Negative LLH = 0.692040
Iteration  70: Negative LLH = 0.691870
Iteration  80: Negative LLH = 0.691738
Iteration  90: Negative LLH = 0.691590
Iteration  99: Negative LLH = 0.691457

top 5 words:
  applicant           : 0.000134
  tribunal            : 0.000103
  respondent          : 0.000066
  appellant           : 0.000059
  application         : 0.000054

task 3: 
Training with LBFGS, iterations=100, features=5000
Training samples: 2727
Correct predictions: 2727
Training accuracy: 1.0000

task 4: 
Final F1 Score: 0.0384

false positives:
1. Philip Zec
2. Oscar Moreno
3. Federal takeover of Fannie Mae and Freddie Mae 

While these 3 Wikipedia articles don't have any immediate connections to legal systems or Australia, upon looking further we could see why the model was fooled by these articles. Each of the articles contain formal and technical language that mirrors the vocabulary used in the legal documents used to train the model. Words like "House of Commons Debate" or "tribunal" found in these articles could have tricked the model into thinking that they were related to the Australian documents the model was trained on. Furthermore, the formal style and structure of the articles that included citations could have fooled the model further.