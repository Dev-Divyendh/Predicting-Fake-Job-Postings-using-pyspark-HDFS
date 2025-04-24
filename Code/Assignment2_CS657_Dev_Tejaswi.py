from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when, isnan, regexp_replace, lower, trim
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, VectorAssembler
from pyspark.ml.classification import LogisticRegression, LinearSVC, RandomForestClassifier, MultilayerPerceptronClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

"""Loading Data"""

spark = SparkSession.builder.appName("FakeJobPostings").getOrCreate()

df = spark.read.csv("/content/drive/MyDrive/datasets/fake_job_postings.csv", header=True, inferSchema=True)

df.printSchema()

df.show(5)

num_rows = df.count()
num_columns = len(df.columns)
print(f"Number of rows: {num_rows}, Number of columns: {num_columns}")

unique_count = df.select("fraudulent").distinct().count()
print(f"Unique values in 'fraudulent': {unique_count}")

df.select("fraudulent").distinct().show()

"""Label column Processing"""

df_cleaned = df.filter((df['fraudulent'] == 0) | (df['fraudulent'] == 1))

df_cleaned.groupBy('fraudulent').count().show()

"""Missing Values and Column Deletion"""

# Total number of rows
total_rows = df_cleaned.count()

# Calculate the percentage of missing values for each column
missing_values = df_cleaned.select([(count(when(col(c).isNull(), c)) / total_rows).alias(c) for c in df_cleaned.columns])
missing_values.show()

null_threshold = 0.01

#Manually removing columns that do not match the criteria mentioned above
columns_to_drop = ['location', 'department', 'salary_range', 'company_profile',
                   'requirements', 'benefits', 'employment_type',
                   'required_experience', 'required_education', 'industry', 'function']

df_cleaned = df_cleaned.drop(*columns_to_drop)

df_cleaned.printSchema()

num_rows = df_cleaned.count()
num_columns = len(df_cleaned.columns)
print(f"Number of rows: {num_rows}, Number of columns: {num_columns}")

"""cleaning the dataset"""

df_cleaned = df_cleaned.withColumn('description', regexp_replace('description', '[^a-zA-Z\\s]', ''))
df_cleaned = df_cleaned.withColumn('description', regexp_replace('description', '\\s+', ' '))
df_cleaned = df_cleaned.withColumn('description', lower(trim(df_cleaned['description'])))

df_cleaned = df_cleaned.withColumn('title', regexp_replace('description', '[^a-zA-Z\\s]', ''))
df_cleaned = df_cleaned.withColumn('title', regexp_replace('description', '\\s+', ' '))
df_cleaned = df_cleaned.withColumn('title', lower(trim(df_cleaned['description'])))


df_cleaned.show(5)

"""Sampling the data"""

df_cleaned.select("fraudulent").distinct().show(100, truncate=False)

print(f"Number of rows before dropping nulls: {df_cleaned.count()}")
df_cleaned = df_cleaned.dropna()

print(f"Number of rows after dropping nulls: {df_cleaned.count()}")

df_cleaned.groupBy('fraudulent').count().show()

df_cleaned.show(5)

df_cleaned = df_cleaned.withColumn('fraudulent', df_cleaned['fraudulent'].cast('int'))

distinct_keys = df_cleaned.rdd.keyBy(lambda row: row['fraudulent']).map(lambda x: x[0]).distinct().collect()
print(distinct_keys)

fractions = {0: 0.3, 1: 1.0}
rdd_keyed = df_cleaned.rdd.keyBy(lambda row: row['fraudulent'])

rdd_sampled = rdd_keyed.sampleByKey(withReplacement=False, fractions=fractions, seed=42)

df_sampled = rdd_sampled.map(lambda x: x[1]).toDF(df_cleaned.columns)

df_sampled.groupBy('fraudulent').count().show()

df_sampled.show(5)

"""Text-Preprocessing"""

#Tokenization (Title and Description)
tokenizer_title = Tokenizer(inputCol="title", outputCol="title_words")
tokenizer_description = Tokenizer(inputCol="description", outputCol="description_words")

df_tokenized = tokenizer_title.transform(df_sampled)
df_tokenized = tokenizer_description.transform(df_tokenized)

#Removing stopwords
remover_title = StopWordsRemover(inputCol="title_words", outputCol="filtered_title_words")
remover_description = StopWordsRemover(inputCol="description_words", outputCol="filtered_description_words")

df_filtered = remover_title.transform(df_tokenized)
df_filtered = remover_description.transform(df_filtered)

# TF-IDF
hashingTF_title = HashingTF(inputCol="filtered_title_words", outputCol="title_tf", numFeatures=10000)
hashingTF_description = HashingTF(inputCol="filtered_description_words", outputCol="description_tf", numFeatures=10000)

df_tf = hashingTF_title.transform(df_filtered)
df_tf = hashingTF_description.transform(df_tf)

idf_title = IDF(inputCol="title_tf", outputCol="title_features")
idf_description = IDF(inputCol="description_tf", outputCol="description_features")

idf_model_title = idf_title.fit(df_tf)
df_final_title = idf_model_title.transform(df_tf)

idf_model_description = idf_description.fit(df_final_title)
df_final = idf_model_description.transform(df_final_title)

#Dropping the original 'description' and 'title' columns and adding the new feature columns
df_final = df_final.drop("title", "description", "title_words", "description_words", "filtered_title_words", "filtered_description_words", "title_tf", "description_tf")

df_final = df_final.withColumn("telecommuting", col("telecommuting").cast("integer"))
df_final = df_final.withColumn("has_company_logo", col("has_company_logo").cast("integer"))
df_final = df_final.withColumn("has_questions", col("has_questions").cast("integer"))
df_final = df_final.withColumn("fraudulent", col("fraudulent").cast("integer"))

train_df, test_df = df_final.randomSplit([0.7, 0.3], seed=42)

df_final.printSchema()
df_final.show(5, truncate=False)
df_final.select("title_features", "description_features", "fraudulent").show(5, truncate=False)

print(f"Number of rows before dropping nulls: {df_final.count()}")
df_final = df_final.dropna()

print(f"Number of rows after dropping nulls: {df_final.count()}")

"""Train test split"""

train_df, test_df = df_final.randomSplit([0.7, 0.3], seed=42)

print(f"Training set: {train_df.count()} rows")
print(f"Test set: {test_df.count()} rows")

"""Machine Learning Models"""

from pyspark.sql.functions import col, when, count

# Checking for null values in the relevant columns
null_counts = df_final.select(
    *[count(when(col(column).isNull(), column)).alias(column) for column in df_final.columns]
).collect()

print("Count of nulls in each column:")
for row in null_counts:
    for column in row.asDict():
        print(f"{column}: {row[column]}")

# Preparing the features
assembler = VectorAssembler(
    inputCols=["telecommuting", "has_company_logo", "has_questions", "title_features", "description_features"],
    outputCol="features"
)

"""Train test split"""

df_final_assembled = assembler.transform(df_final)
train_df, test_df = df_final_assembled.randomSplit([0.7, 0.3], seed=42)

evaluator_accuracy = MulticlassClassificationEvaluator(labelCol="fraudulent", metricName="accuracy")
evaluator_f1 = MulticlassClassificationEvaluator(labelCol="fraudulent", metricName="f1")
l
def train_and_evaluate_model(model, paramGrid, model_name):
    print(f"Starting training for {model_name}...")

    cv = CrossValidator(estimator=model,
                        estimatorParamMaps=paramGrid,
                        evaluator=evaluator_f1,
                        numFolds=10)

    print(f"Performing cross-validation for {model_name}...")

    cv_model = cv.fit(train_df)

    best_model = cv_model.bestModel
    best_params = best_model.extractParamMap()

    print(f"Best parameters for {model_name}: {best_params}")

    predictions = best_model.transform(test_df)

    accuracy = evaluator_accuracy.evaluate(predictions)
    f1_score = evaluator_f1.evaluate(predictions)

    print(f"Completed training for {model_name}. Accuracy: {accuracy}, F1 Score: {f1_score}\n")

    return accuracy, f1_score, best_params

# Model 1: Logistic Regression
log_reg = LogisticRegression(featuresCol='features', labelCol='fraudulent')
paramGrid_lr = (ParamGridBuilder()
                .addGrid(log_reg.regParam, [0.01, 0.1, 1.0])
                .addGrid(log_reg.maxIter, [10, 20])
                .build())

accuracy_lr, f1_score_lr, best_params_lr = train_and_evaluate_model(log_reg, paramGrid_lr, "Logistic Regression")

# Model 2: Linear SVC
svc = LinearSVC(featuresCol='features', labelCol='fraudulent')
paramGrid_svc = (ParamGridBuilder()
                 .addGrid(svc.regParam, [0.01, 0.1, 1.0])
                 .build())

accuracy_svc, f1_score_svc, best_params_svc = train_and_evaluate_model(svc, paramGrid_svc, "Linear SVC")

# Model 3: Random Forest Classifier
rf = RandomForestClassifier(featuresCol='features', labelCol='fraudulent')
paramGrid_rf = (ParamGridBuilder()
                .addGrid(rf.numTrees, [10, 20, 30])
                .addGrid(rf.maxDepth, [5, 10])
                .build())

accuracy_rf, f1_score_rf, best_params_rf = train_and_evaluate_model(rf, paramGrid_rf, "Random Forest Classifier")

# Output final results
print("Final Results:")
print("Logistic Regression: Accuracy = {}, F1 Score = {}, Best Params = {}".format(accuracy_lr, f1_score_lr, best_params_lr))
print("Linear SVC: Accuracy = {}, F1 Score = {}, Best Params = {}".format(accuracy_svc, f1_score_svc, best_params_svc))
print("Random Forest: Accuracy = {}, F1 Score = {}, Best Params = {}".format(accuracy_rf, f1_score_rf, best_params_rf))

#Model 4: Multi-Layer Perceptron
print("Starting training for Multilayer Perceptron Classifier...")

num_features = len(train_df.select("features").first()[0])
print("Number of features in the dataset:", num_features)

mlp = MultilayerPerceptronClassifier(featuresCol='features', labelCol='fraudulent')
paramGrid_mlp = (ParamGridBuilder()
                 .addGrid(mlp.layers, [[num_features, 5, 4, 3], [num_features, 10, 5, 3]])
                 .build())
evaluator_accuracy = MulticlassClassificationEvaluator(labelCol="fraudulent", metricName="accuracy")
evaluator_f1 = MulticlassClassificationEvaluator(labelCol="fraudulent", metricName="f1")

print("Starting training for Multilayer Perceptron Classifier...")

cv = CrossValidator(estimator=mlp,
                    estimatorParamMaps=paramGrid_mlp,
                    evaluator=evaluator_f1,
                    numFolds=10)

cv_model = cv.fit(train_df)

best_model = cv_model.bestModel
best_params = best_model.extractParamMap()
predictions = best_model.transform(test_df)

accuracy = evaluator_accuracy.evaluate(predictions)
f1_score = evaluator_f1.evaluate(predictions)


print("Completed training for Multilayer Perceptron Classifier.")
print("Best parameters for Multilayer Perceptron Classifier:", best_params)
print("Accuracy:", accuracy)
print("F1 Score:", f1_score)

