"""
gender_classification_decision_tree.py
HW3 - Gender Classification using Decision Tree
CSCI 405 / CIS 605

Reads face landmark data, extracts 7 features, and uses
a Decision Tree to classify male vs female.
"""

import math
from pathlib import Path
from typing import List, Dict, Tuple, Any

# only using sklearn for the decision tree and metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score,
                             precision_score, recall_score)


# --- reading .pts files ---

def read_pts_file(file_path: str) -> Dict[str, Any]:
    # reads a single .pts file and returns the points as a list of (x,y) tuples
    result = {
        'file': file_path,
        'version': None,
        'n_points': None,
        'points': []
    }

    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            line = line.strip()

            if line.startswith('version:'):
                result['version'] = int(line.split(':')[1].strip())

            elif line.startswith('n_points:'):
                result['n_points'] = int(line.split(':')[1].strip())

            # once we hit '{' we start reading the actual points
            elif line == '{':
                for j in range(i + 1, len(lines)):
                    point_line = lines[j].strip()
                    if point_line == '}':
                        break
                    if point_line:
                        coords = point_line.split()
                        if len(coords) == 2:
                            try:
                                x = float(coords[0])
                                y = float(coords[1])
                                result['points'].append((x, y))
                            except ValueError:
                                pass
                break

    except Exception as e:
        print(f"Error reading file {file_path}: {e}")

    return result


def read_face_database(database_path: str) -> Dict[str, List[Dict[str, Any]]]:
    # goes through each person folder and reads all their .pts files
    database_path = Path(database_path)

    if not database_path.exists():
        raise FileNotFoundError(f"Database path not found: {database_path}")

    face_data: Dict[str, List[Dict[str, Any]]] = {}

    for person_folder in sorted(database_path.iterdir()):
        if not person_folder.is_dir():
            continue

        person_id = person_folder.name
        face_data[person_id] = []

        for pts_file in sorted(person_folder.glob('*.pts')):
            points_data = read_pts_file(str(pts_file))
            face_data[person_id].append(points_data)

    return face_data


# --- euclidean distance (doing it manually with math.sqrt) ---

def euclidean_distance(point1: Tuple[float, float],
                       point2: Tuple[float, float]) -> float:
    # just the basic distance formula sqrt((x2-x1)^2 + (y2-y1)^2)
    return math.sqrt((point1[0] - point2[0]) ** 2 +
                     (point1[1] - point2[1]) ** 2)


# --- feature extraction (7 features from the 22 landmark points) ---

def extract_features(points: List[Tuple[float, float]]) -> Dict[str, float]:
    # extracts the 7 feature ratios from the landmark points
    # points are 1-indexed in the assignment but 0-indexed in the list
    if len(points) < 22:
        raise ValueError(f"Expected at least 22 points, got {len(points)}")

    features: Dict[str, float] = {}

    try:
        # these two distances get reused a lot as denominators
        # .pts indices are 0-based, so point N in the spec = points[N-1] only
        # if spec is 1-based. Per instructor: the .pts file is already 0-based.
        dist_9_14 = euclidean_distance(points[8], points[13])
        dist_21_22 = euclidean_distance(points[20], points[21])

        # feature 1: eye length ratio
        # take the bigger eye length and divide by dist(9,14)
        # left eye: points 9,10 (0-based); right eye: points 11,12 (0-based)
        left_eye_length = euclidean_distance(points[9], points[10])
        right_eye_length = euclidean_distance(points[11], points[12])
        max_eye_length = max(left_eye_length, right_eye_length)
        features['eye_length_ratio'] = (max_eye_length / dist_9_14
                                        if dist_9_14 != 0 else 0)

        # feature 2: eye distance ratio
        # get the center of each eye then find distance between them
        left_eye_center = ((points[9][0] + points[10][0]) / 2,
                           (points[9][1] + points[10][1]) / 2)
        right_eye_center = ((points[11][0] + points[12][0]) / 2,
                            (points[11][1] + points[12][1]) / 2)
        eye_center_distance = euclidean_distance(left_eye_center,
                                                 right_eye_center)
        features['eye_distance_ratio'] = (eye_center_distance / dist_9_14
                                          if dist_9_14 != 0 else 0)

        # feature 3: nose ratio
        dist_16_17 = euclidean_distance(points[15], points[16])
        features['nose_ratio'] = (dist_16_17 / dist_21_22
                                  if dist_21_22 != 0 else 0)

        # feature 4: lip size ratio
        dist_3_4 = euclidean_distance(points[2], points[3])
        dist_18_19 = euclidean_distance(points[17], points[18])
        features['lip_size_ratio'] = (dist_3_4 / dist_18_19
                                      if dist_18_19 != 0 else 0)

        # feature 5: lip length ratio
        features['lip_length_ratio'] = (dist_3_4 / dist_21_22
                                        if dist_21_22 != 0 else 0)

        # feature 6: eyebrow length ratio
        # take the longer eyebrow: points 4,5 and 6,7 (0-based)
        eyebrow_left = euclidean_distance(points[4], points[5])
        eyebrow_right = euclidean_distance(points[6], points[7])
        max_eyebrow = max(eyebrow_left, eyebrow_right)
        features['eyebrow_length_ratio'] = (max_eyebrow / dist_9_14
                                            if dist_9_14 != 0 else 0)

        # feature 7: aggressive ratio
        dist_11_20 = euclidean_distance(points[10], points[19])
        features['aggressive_ratio'] = (dist_11_20 / dist_21_22
                                        if dist_21_22 != 0 else 0)

    except IndexError as e:
        print(f"Error extracting features: {e}")
        return {}

    return features


def extract_features_as_list(points: List[Tuple[float, float]]) -> List[float]:
    # same as extract_features but returns a list instead of dict
    # easier to pass into sklearn this way
    feat = extract_features(points)
    return [
        feat['eye_length_ratio'],
        feat['eye_distance_ratio'],
        feat['nose_ratio'],
        feat['lip_size_ratio'],
        feat['lip_length_ratio'],
        feat['eyebrow_length_ratio'],
        feat['aggressive_ratio'],
    ]


# --- helper to collect features for all people ---

def collect_all_features(
        face_data: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, Dict[str, List[float]]]:
    # loops through everyone and gets their features for each image
    feature_keys = [
        'eye_length_ratio',
        'eye_distance_ratio',
        'nose_ratio',
        'lip_size_ratio',
        'lip_length_ratio',
        'eyebrow_length_ratio',
        'aggressive_ratio',
    ]

    person_features: Dict[str, Dict[str, List[float]]] = {}

    for person_id, files in face_data.items():
        person_features[person_id] = {k: [] for k in feature_keys}

        for file_data in files:
            if len(file_data['points']) >= 22:
                features = extract_features(file_data['points'])
                for feat_name, value in features.items():
                    person_features[person_id][feat_name].append(value)

    return person_features


# --- makes a nice text table for the report ---

def format_table(rows: List[Dict[str, str]], headers: List[str]) -> str:
    # builds a simple text table with columns lined up
    if not rows:
        return ""

    widths = {h: len(h) for h in headers}
    for row in rows:
        for h in headers:
            widths[h] = max(widths[h], len(str(row.get(h, ""))))

    header_line = "  ".join(h.ljust(widths[h]) for h in headers)
    separator = "  ".join("-" * widths[h] for h in headers)
    data_lines = []
    for row in rows:
        data_lines.append(
            "  ".join(str(row.get(h, "")).ljust(widths[h]) for h in headers)
        )

    return "\n".join([header_line, separator] + data_lines)


# --- decision tree classification ---

def run_decision_tree_classification(
        face_data: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, Any]:
    # trains a decision tree on the training data and tests it
    # training: m-001 to m-003 + w-001 to w-003 (24 images)
    # testing: m-004, m-005, w-004, w-005 (16 images)
    # labels: 0 = male, 1 = female

    train_persons = ['m-001', 'm-002', 'm-003', 'w-001', 'w-002', 'w-003']
    test_persons = ['m-004', 'm-005', 'w-004', 'w-005']

    train_data: List[List[float]] = []
    train_target: List[int] = []
    test_data: List[List[float]] = []
    test_target: List[int] = []

    # go through each person and split into train/test
    for person_id, files in face_data.items():
        # m = male = 0, w = female = 1
        label = 0 if person_id.startswith('m') else 1

        for file_data in files:
            if len(file_data['points']) < 22:
                continue

            feature_vector = extract_features_as_list(file_data['points'])

            if person_id in train_persons:
                train_data.append(feature_vector)
                train_target.append(label)
            elif person_id in test_persons:
                test_data.append(feature_vector)
                test_target.append(label)

    # create and train the decision tree with entropy
    dt = DecisionTreeClassifier(criterion='entropy')
    dt.fit(train_data, train_target)

    # test it
    predictions = dt.predict(test_data)

    # get the metrics
    cm = confusion_matrix(test_target, predictions)
    acc = accuracy_score(test_target, predictions)
    prec = precision_score(test_target, predictions, zero_division=0)
    rec = recall_score(test_target, predictions, zero_division=0)

    return {
        'train_persons': train_persons,
        'test_persons': test_persons,
        'train_size': len(train_data),
        'test_size': len(test_data),
        'predictions': predictions.tolist(),
        'ground_truth': test_target,
        'confusion_matrix': cm,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
    }


# --- report generation ---

def create_report(face_data: Dict[str, List[Dict[str, Any]]]) -> None:
    # generates the results report and saves it to a text file
    R: List[str] = []
    # run the classification
    dt_result = run_decision_tree_classification(face_data)

    R.append("Decision Tree Classification Results")
    R.append("=" * 40)
    R.append(f"Training set: {', '.join(dt_result['train_persons'])} "
             f"({dt_result['train_size']} images)")
    R.append(f"Test set: {', '.join(dt_result['test_persons'])} "
             f"({dt_result['test_size']} images)")
    R.append("")

    cm = dt_result['confusion_matrix']
    R.append("Confusion Matrix:")
    R.append("                  Predicted Male  Predicted Female")
    R.append(f"  Actual Male       {cm[0][0]:>10}  {cm[0][1]:>16}")
    R.append(f"  Actual Female     {cm[1][0]:>10}  {cm[1][1]:>16}")
    R.append("")

    R.append(f"Accuracy  : {dt_result['accuracy']:.4f}  "
             f"({dt_result['accuracy'] * 100:.1f}%)")
    R.append(f"Precision : {dt_result['precision']:.4f}")
    R.append(f"Recall    : {dt_result['recall']:.4f}")
    R.append("")

    # per-image prediction table
    label_map = {0: 'Male', 1: 'Female'}
    pred_rows: List[Dict[str, str]] = []
    idx = 0
    for person_id in dt_result['test_persons']:
        for file_data in face_data[person_id]:
            if len(file_data['points']) < 22:
                continue
            actual = dt_result['ground_truth'][idx]
            pred = dt_result['predictions'][idx]
            correct = "YES" if actual == pred else "NO"
            pred_rows.append({
                'File': Path(file_data['file']).name,
                'Actual': label_map[actual],
                'Predicted': label_map[pred],
                'Correct': correct,
            })
            idx += 1

    pred_headers = ['File', 'Actual', 'Predicted', 'Correct']
    R.append(format_table(pred_rows, pred_headers))
    R.append("")

    # save and print
    report_text = "\n".join(R)

    report_path = Path(__file__).parent / 'decision_tree_report.txt'
    with open(report_path, 'w') as f:
        f.write(report_text)

    print(report_text)
    print(f"\n✓ Report saved to: {report_path}")


# --- main ---

if __name__ == "__main__":
    # path to the database
    database_path = (
        "/Users/austinmoser/PycharmProjects/FaceRecognition/FaceDatabase"
    )

    # read all the pts files
    print("Reading face database...")
    face_data = read_face_database(database_path)

    # generate the report
    print("\n" + "=" * 80)
    print("GENERATING DECISION TREE CLASSIFICATION REPORT...")
    print("=" * 80)
    create_report(face_data)
