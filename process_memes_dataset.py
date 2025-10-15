import json
import csv
from collections import Counter
import os

# --- Configuration ---
INPUT_JSON_PATH = 'EXIST2024_training.json'
OUTPUT_CSV_PATH = 'processed_data_all_labels.csv'

# --- Category Definitions (using original hyphenated format) ---
TASK4_CATEGORIES = ['YES', 'NO']
TASK5_CATEGORIES = sorted(['DIRECT', 'JUDGEMENTAL'])
TASK6_CATEGORIES = sorted([
    'IDEOLOGICAL-INEQUALITY',
    'STEREOTYPING-DOMINANCE',
    'OBJECTIFICATION',
    'SEXUAL-VIOLENCE',
    'MISOGYNY-NON-SEXUAL-VIOLENCE'
])

# --- Helper Functions (formatting step removed) ---

def filter_valid_labels(labels):
    """Filter out invalid placeholder labels from the list only."""
    return [label for label in labels if label != '-']

def create_hard_vector_one_hot(labels, categories):
    """Compute hard labels for single-choice task and generate one-hot vector."""
    vector = [0.0] * len(categories)
    valid_labels = filter_valid_labels(labels)
    if not valid_labels:
        return vector
    
    hard_label = Counter(valid_labels).most_common(1)[0][0]
    if hard_label in categories:
        index = categories.index(hard_label)
        vector[index] = 1.0
    return vector

def create_soft_vector_single_choice(labels, categories, num_annotators):
    """Compute soft labels (voting distribution) vector for single-choice task."""
    vector = [0.0] * len(categories)
    if num_annotators == 0:
        return vector
        
    valid_labels = filter_valid_labels(labels)
    counts = Counter(valid_labels)
    
    for label, count in counts.items():
        if label in categories:
            index = categories.index(label)
            vector[index] = round(count / num_annotators, 4)
    return vector

def create_hard_vector_multi_label(labels_list, categories, num_annotators):
    """Compute hard labels for multi-label task and generate multi-hot vector."""
    vector = [0.0] * len(categories)
    if num_annotators == 0:
        return vector

    counts = Counter()
    for labels in labels_list:
        if labels == ['-']: continue
        # Use original labels directly (deduplicated)
        unique_labels = set(labels)
        counts.update(unique_labels)
        
    threshold = num_annotators / 2
    for i, category in enumerate(categories):
        if counts[category] > threshold:
            vector[i] = 1.0
    return vector

def create_soft_vector_multi_label(labels_list, categories, num_annotators):
    """Compute soft labels (voting distribution) vector for multi-label task."""
    vector = [0.0] * len(categories)
    if num_annotators == 0:
        return vector

    counts = Counter()
    for labels in labels_list:
        if labels == ['-']: continue
        unique_labels = set(labels)
        counts.update(unique_labels)
    
    for i, category in enumerate(categories):
        vector[i] = round(counts[category] / num_annotators, 4)
    return vector

# --- Main Program ---
def main():
    if not os.path.exists(INPUT_JSON_PATH):
        print(f"Error: Input file '{INPUT_JSON_PATH}' not found. Creating sample file...")
        sample_data = {
            "212010": { "id_EXIST": "212010", "number_annotators": 6, "lang": "en", "text": "...", "labels_task4": ["NO", "YES", "YES", "YES", "YES", "YES"], "labels_task5": ["-", "DIRECT", "DIRECT", "DIRECT", "JUDGEMENTAL", "DIRECT"], "labels_task6": [["-"],["OBJECTIFICATION"],["STEREOTYPING-DOMINANCE", "OBJECTIFICATION"],["OBJECTIFICATION"],["STEREOTYPING-DOMINANCE"],["SEXUAL-VIOLENCE"]] }
        }
        with open(INPUT_JSON_PATH, 'w', encoding='utf-8') as f: json.dump(sample_data, f, indent=4)
        print(f"Sample file '{INPUT_JSON_PATH}' has been created. Please replace its content with your complete data and run again.")
        return

    with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f: data = json.load(f)
    with open(OUTPUT_CSV_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'lang', 'text', 'task4_hard', 'task4_soft', 'task5_hard', 'task5_soft', 'task6_hard', 'task6_soft'])

        for item_id, item_data in data.items():
            lang = item_data.get('lang', 'N/A')
            text = item_data.get('text', '').replace('\n', ' ').strip()
            num_annotators = item_data.get('number_annotators', 0)
            
            labels4 = item_data.get('labels_task4', [])
            labels5 = item_data.get('labels_task5', [])
            labels6 = item_data.get('labels_task6', [])

            task4_hard = create_hard_vector_one_hot(labels4, TASK4_CATEGORIES)
            task4_soft = create_soft_vector_single_choice(labels4, TASK4_CATEGORIES, num_annotators)
            
            task5_hard = create_hard_vector_one_hot(labels5, TASK5_CATEGORIES)
            task5_soft = create_soft_vector_single_choice(labels5, TASK5_CATEGORIES, num_annotators)

            task6_hard = create_hard_vector_multi_label(labels6, TASK6_CATEGORIES, num_annotators)
            task6_soft = create_soft_vector_multi_label(labels6, TASK6_CATEGORIES, num_annotators)
            
            writer.writerow([item_id, lang, text, str(task4_hard), str(task4_soft), str(task5_hard), str(task5_soft), str(task6_hard), str(task6_soft)])

    print(f"Processing complete! Data with original label format has been saved to '{OUTPUT_CSV_PATH}'.")

if __name__ == '__main__':
    main()