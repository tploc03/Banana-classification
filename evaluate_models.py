import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model # type: ignore
from sklearn.metrics import classification_report, accuracy_score

dataset_directory = "raw_to_test/" 
batch_size = 32
img_size = (150, 150)

test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_directory,
    labels='inferred',
    label_mode='int',
    batch_size=batch_size,
    image_size=img_size,
    shuffle=False,
    seed=42
)

y_true = np.concatenate([y for x, y in test_dataset], axis=0)

model_paths = [
    "output/banana_classifier_trained_001.keras",
    # "output/banana_classifier_trained_002.keras",
    # "output/banana_classifier_trained_003.keras",
]
models = [load_model(path) for path in model_paths]

model_results = {}
    
target_names = ['Unripe', 'Ripe', 'Rotten']
for i, model in enumerate(models):
    y_pred_probs = model.predict(test_dataset)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    report = classification_report(y_true, y_pred_classes, target_names=target_names, output_dict=True)
    accuracy = accuracy_score(y_true, y_pred_classes)
    model_results[f"Model_{i+1}"] = {
        "Precision (Unripe)": report['Unripe']['precision'],
        "Precision (Ripe)": report['Ripe']['precision'],
        "Precision (Rotten)": report['Rotten']['precision'],
        "Recall (Unripe)": report['Unripe']['recall'],
        "Recall (Ripe)": report['Ripe']['recall'],
        "Recall (Rotten)": report['Rotten']['recall'],
        "F1-Score (Unripe)": report['Unripe']['f1-score'],
        "F1-Score (Ripe)": report['Ripe']['f1-score'],
        "F1-Score (Rotten)": report['Rotten']['f1-score'],
        "Accuracy": accuracy,
        "Macro Avg Precision": report['macro avg']['precision'],
        "Macro Avg Recall": report['macro avg']['recall'],
        "Macro Avg F1-Score": report['macro avg']['f1-score'],
        "Weighted Avg Precision": report['weighted avg']['precision'],
        "Weighted Avg Recall": report['weighted avg']['recall'],
        "Weighted Avg F1-Score": report['weighted avg']['f1-score']
    }
    

for model_name, metrics in model_results.items():
    print(f"\n{model_name} Evaluation Metrics:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    model = keras.models.load_model('./output/banana_classifier_trained_001.keras')
    model.summary()