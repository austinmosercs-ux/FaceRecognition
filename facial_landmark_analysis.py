"""
facial_landmark_analysis.py

HW2: Gender Classification from Face Images using KNN
CSCI 405 / CIS 605

This program reads 22-point facial landmark data from the AR Face Database,
extracts 7 geometric feature ratios (eye length, eye distance, nose, lip size,
lip length, eyebrow length, aggressive), and then uses a K-Nearest Neighbors
(KNN) classifier from scikit-learn to classify gender (male vs female).

The math (Euclidean distance, feature ratios) is computed manually using
Python's built-in math.sqrt — no external library is used for the distance
calculations themselves.  scikit-learn is used ONLY for the KNN classifier
and the performance metrics (confusion matrix, accuracy, precision, recall).

Data layout
-----------
FaceDatabase/
    m-001/  …  m-005/   (5 males,  4 images each  → 20 male images)
    w-001/  …  w-005/   (5 females, 4 images each → 20 female images)

Each .pts file contains 22 (x, y) landmark coordinates.
"""

import math
from pathlib import Path
from typing import List, Dict, Tuple, Any

# scikit-learn is used ONLY for the KNN classifier and evaluation metrics.
# All distance / feature calculations are done manually with math.sqrt.
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score


# ---------------------------------------------------------------------------
# 1. FILE I/O  –  Reading .pts landmark files
# ---------------------------------------------------------------------------

def read_pts_file(file_path: str) -> Dict[str, Any]:
    """
    Read a single .pts file and return its metadata + landmark points.

    .pts format
    -----------
    version: 1
    n_points: 22
    {
    x1 y1
    x2 y2
    ...
    }

    Args:
        file_path: Full path to the .pts file.

    Returns:
        A dictionary with keys:
            'file'     – the file path (str)
            'version'  – file format version (int)
            'n_points' – declared number of points (int)
            'points'   – list of (x, y) tuples
    """
    result = {
        'file': file_path,
        'version': None,
        'n_points': None,
        'points': []
    }

    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # Walk through the header lines to find version, n_points, and the
        # opening brace that marks the start of coordinate data.
        for i, line in enumerate(lines):
            line = line.strip()

            # Parse "version: 1"
            if line.startswith('version:'):
                result['version'] = int(line.split(':')[1].strip())

            # Parse "n_points: 22"
            elif line.startswith('n_points:'):
                result['n_points'] = int(line.split(':')[1].strip())

            # The opening brace signals the start of (x, y) data
            elif line == '{':
                # Read coordinate lines until the closing brace '}'
                for j in range(i + 1, len(lines)):
                    point_line = lines[j].strip()
                    if point_line == '}':
                        break
                    if point_line:
                        coords = point_line.split()
                        if len(coords) == 2:
                            try:
                                x, y = float(coords[0]), float(coords[1])
                                result['points'].append((x, y))
                            except ValueError:
                                pass  # skip malformed coordinate lines
                break  # done after reading the point block

    except Exception as e:
        print(f"Error reading file {file_path}: {e}")

    return result


def read_face_database(database_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Traverse every person-folder inside the FaceDatabase directory and
    read all .pts files for each person.

    Directory layout expected:
        database_path/
            m-001/   ← person folder (male #1)
                m-001-01.pts
                m-001-02.pts
                ...
            w-003/   ← person folder (female #3)
                w-003-01.pts
                ...

    Args:
        database_path: Path to the top-level FaceDatabase folder.

    Returns:
        Ordered mapping  person_id → [list of pts-file dicts].
        Each pts-file dict is the output of read_pts_file().
    """
    database_path = Path(database_path)

    if not database_path.exists():
        raise FileNotFoundError(f"Database path not found: {database_path}")

    face_data: Dict[str, List[Dict[str, Any]]] = {}

    # Iterate through person folders in sorted order (m-001, m-002, …, w-005)
    for person_folder in sorted(database_path.iterdir()):
        if not person_folder.is_dir():
            continue

        person_id = person_folder.name
        face_data[person_id] = []

        # Read every .pts file inside this person's folder
        for pts_file in sorted(person_folder.glob('*.pts')):
            points_data = read_pts_file(str(pts_file))
            face_data[person_id].append(points_data)

    return face_data


# ---------------------------------------------------------------------------
# 2. DISTANCE CALCULATION  –  Manual Euclidean distance (no library)
# ---------------------------------------------------------------------------

def euclidean_distance(point1: Tuple[float, float],
                       point2: Tuple[float, float]) -> float:
    """
    Compute the Euclidean distance between two 2-D points.

    Formula:  d = sqrt( (x2-x1)^2 + (y2-y1)^2 )

    This is implemented manually using Python's math.sqrt — no external
    numerical library (e.g., NumPy) is used here.

    Args:
        point1: (x, y) coordinates of the first point.
        point2: (x, y) coordinates of the second point.

    Returns:
        The Euclidean distance (float).
    """
    return math.sqrt((point1[0] - point2[0]) ** 2 +
                     (point1[1] - point2[1]) ** 2)


# ---------------------------------------------------------------------------
# 3. FEATURE EXTRACTION  –  7 geometric ratios from 22 landmarks
# ---------------------------------------------------------------------------

def extract_features(points: List[Tuple[float, float]]) -> Dict[str, float]:
    """
    Extract the 7 facial-feature ratios defined in the assignment.

    Landmark numbering is 1-based in the assignment spec; we convert to
    0-based indexing when accessing the list.

    Feature definitions (all distances are Euclidean):
        1. Eye Length Ratio
           = max(dist(pt5,pt6), dist(pt7,pt8)) / dist(pt8, pt13)
        2. Eye Distance Ratio
           = dist(center_left_eye, center_right_eye) / dist(pt8, pt13)
           where center_left_eye  = midpoint of pt5 & pt6
                 center_right_eye = midpoint of pt7 & pt8
        3. Nose Ratio
           = dist(pt15, pt16) / dist(pt20, pt21)
        4. Lip Size Ratio
           = dist(pt2, pt3) / dist(pt17, pt18)
        5. Lip Length Ratio
           = dist(pt2, pt3) / dist(pt20, pt21)
        6. Eyebrow Length Ratio
           = max(dist(pt4,pt5), dist(pt6,pt7)) / dist(pt8, pt13)
        7. Aggressive Ratio
           = dist(pt10, pt19) / dist(pt20, pt21)

    Args:
        points: List of 22 (x, y) tuples (0-indexed).

    Returns:
        Dictionary with 7 feature names → float values.

    Raises:
        ValueError: If fewer than 22 points are provided.
    """
    if len(points) < 22:
        raise ValueError(f"Expected at least 22 points, got {len(points)}")

    features: Dict[str, float] = {}

    try:
        # ----- Shared denominators used by multiple features -----

        # Distance between points 8 and 13 (denominator for features 1, 2, 6)
        dist_8_13 = euclidean_distance(points[7], points[12])  # 1-indexed pts 8 & 13

        # Distance between points 20 and 21 (denominator for features 3, 5, 7)
        dist_20_21 = euclidean_distance(points[19], points[20])  # 1-indexed pts 20 & 21

        # ----- Feature 1: Eye Length Ratio -----
        # "length of eye (maximum of two) over distance between points 8 and 13"
        # Left eye span:  points 5–6  (0-indexed: 4, 5)
        # Right eye span: points 7–8  (0-indexed: 6, 7)
        left_eye_length = euclidean_distance(points[4], points[5])
        right_eye_length = euclidean_distance(points[6], points[7])
        max_eye_length = max(left_eye_length, right_eye_length)
        features['eye_length_ratio'] = (max_eye_length / dist_8_13
                                        if dist_8_13 != 0 else 0)

        # ----- Feature 2: Eye Distance Ratio -----
        # "distance between center of two eyes over distance between points 8 and 13"
        # Center of left eye  = midpoint of points 5 & 6
        # Center of right eye = midpoint of points 7 & 8
        left_eye_center = ((points[4][0] + points[5][0]) / 2,
                           (points[4][1] + points[5][1]) / 2)
        right_eye_center = ((points[6][0] + points[7][0]) / 2,
                            (points[6][1] + points[7][1]) / 2)
        eye_center_distance = euclidean_distance(left_eye_center,
                                                 right_eye_center)
        features['eye_distance_ratio'] = (eye_center_distance / dist_8_13
                                          if dist_8_13 != 0 else 0)

        # ----- Feature 3: Nose Ratio -----
        # "Distance between points 15 and 16 over distance between 20 and 21"
        dist_15_16 = euclidean_distance(points[14], points[15])  # 1-indexed pts 15 & 16
        features['nose_ratio'] = (dist_15_16 / dist_20_21
                                  if dist_20_21 != 0 else 0)

        # ----- Feature 4: Lip Size Ratio -----
        # "Distance between points 2 and 3 over distance between 17 and 18"
        dist_2_3 = euclidean_distance(points[1], points[2])    # 1-indexed pts 2 & 3
        dist_17_18 = euclidean_distance(points[16], points[17])  # 1-indexed pts 17 & 18
        features['lip_size_ratio'] = (dist_2_3 / dist_17_18
                                      if dist_17_18 != 0 else 0)

        # ----- Feature 5: Lip Length Ratio -----
        # "Distance between points 2 and 3 over distance between 20 and 21"
        features['lip_length_ratio'] = (dist_2_3 / dist_20_21
                                        if dist_20_21 != 0 else 0)

        # ----- Feature 6: Eyebrow Length Ratio -----
        # "Distance between points 4 and 5 (or 6 and 7, whichever larger)
        #  over distance between 8 and 13"
        eyebrow_left = euclidean_distance(points[3], points[4])   # 1-indexed pts 4 & 5
        eyebrow_right = euclidean_distance(points[5], points[6])  # 1-indexed pts 6 & 7
        max_eyebrow = max(eyebrow_left, eyebrow_right)
        features['eyebrow_length_ratio'] = (max_eyebrow / dist_8_13
                                            if dist_8_13 != 0 else 0)

        # ----- Feature 7: Aggressive Ratio -----
        # "Distance between points 10 and 19 over distance between 20 and 21"
        dist_10_19 = euclidean_distance(points[9], points[18])  # 1-indexed pts 10 & 19
        features['aggressive_ratio'] = (dist_10_19 / dist_20_21
                                        if dist_20_21 != 0 else 0)

    except IndexError as e:
        print(f"Error extracting features: {e}")
        return {}

    return features


def extract_features_as_list(points: List[Tuple[float, float]]) -> List[float]:
    """
    Convenience wrapper: returns the 7 features as a flat list
    (suitable for feeding into scikit-learn classifiers).

    Order: [eye_length, eye_distance, nose, lip_size, lip_length,
            eyebrow_length, aggressive]

    Args:
        points: 22 (x, y) landmark tuples.

    Returns:
        List of 7 float values.
    """
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


# ---------------------------------------------------------------------------
# 4. FEATURE COLLECTION HELPERS
# ---------------------------------------------------------------------------

def collect_all_features(
        face_data: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, Dict[str, List[float]]]:
    """
    Aggregate feature values for every person across all their image files.

    Args:
        face_data: Output of read_face_database().

    Returns:
        Nested dict:  person_id → feature_name → [list of values per image]
    """
    # The 7 feature keys, in order
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
        # Initialise empty lists for each feature
        person_features[person_id] = {k: [] for k in feature_keys}

        for file_data in files:
            if len(file_data['points']) >= 22:
                features = extract_features(file_data['points'])
                for feat_name, value in features.items():
                    person_features[person_id][feat_name].append(value)

    return person_features


# ---------------------------------------------------------------------------
# 5. TEXT TABLE FORMATTER
# ---------------------------------------------------------------------------

def format_table(rows: List[Dict[str, str]], headers: List[str]) -> str:
    """
    Build a fixed-width plain-text table from a list of row-dicts.

    Args:
        rows:    Each dict maps header-name → cell-value (as string).
        headers: Column names, in display order.

    Returns:
        Multi-line string with header, separator, and data rows.
    """
    if not rows:
        return ""

    # Determine the max width needed for each column
    widths = {h: len(h) for h in headers}
    for row in rows:
        for h in headers:
            widths[h] = max(widths[h], len(str(row.get(h, ""))))

    # Build the header, separator, and data lines
    header_line = "  ".join(h.ljust(widths[h]) for h in headers)
    separator   = "  ".join("-" * widths[h] for h in headers)
    data_lines  = []
    for row in rows:
        data_lines.append(
            "  ".join(str(row.get(h, "")).ljust(widths[h]) for h in headers)
        )

    return "\n".join([header_line, separator] + data_lines)


# ---------------------------------------------------------------------------
# 6. KNN GENDER CLASSIFICATION
# ---------------------------------------------------------------------------

def run_knn_classification(
        face_data: Dict[str, List[Dict[str, Any]]],
        k: int = 3
) -> Dict[str, Any]:
    """
    Train a KNN classifier for gender classification and evaluate it.

    Experiment design (from assignment spec):
        Training set – first 3 males (m-001 … m-003) + first 3 females
                       (w-001 … w-003)  →  24 images
        Testing  set – remaining 2 males (m-004, m-005) + 2 females
                       (w-004, w-005)    →  16 images

    Labels:  0 = male,  1 = female

    The KNN classifier is from scikit-learn (sklearn.neighbors).
    The feature vectors are the 7 manually-computed ratios.

    Args:
        face_data: Output of read_face_database().
        k:         Number of neighbours for KNN (default 3).

    Returns:
        Dictionary containing predictions, ground truth, and all metrics.
    """
    # --- Define which persons go to train vs. test ---
    # Training: 3 males + 3 females  (as specified in the assignment)
    train_persons = ['m-001', 'm-002', 'm-003', 'w-001', 'w-002', 'w-003']
    # Testing: 2 males + 2 females
    test_persons  = ['m-004', 'm-005', 'w-004', 'w-005']

    train_data:   List[List[float]] = []  # feature vectors (7 floats each)
    train_target: List[int]         = []  # gender labels: 0=male, 1=female
    test_data:    List[List[float]] = []
    test_target:  List[int]         = []

    # --- Build feature vectors and labels for every image ---
    for person_id, files in face_data.items():
        # Determine gender label from the person-ID prefix ('m' or 'w')
        label = 0 if person_id.startswith('m') else 1  # 0=male, 1=female

        for file_data in files:
            if len(file_data['points']) < 22:
                continue  # skip files with insufficient landmarks

            # Extract the 7 features (computed manually, no library)
            feature_vector = extract_features_as_list(file_data['points'])

            # Assign to training or testing set based on person ID
            if person_id in train_persons:
                train_data.append(feature_vector)
                train_target.append(label)
            elif person_id in test_persons:
                test_data.append(feature_vector)
                test_target.append(label)

    # --- Train the KNN classifier (scikit-learn) ---
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_data, train_target)

    # --- Predict on the test set ---
    predictions = knn.predict(test_data)

    # --- Compute evaluation metrics (scikit-learn) ---
    cm   = confusion_matrix(test_target, predictions)
    acc  = accuracy_score(test_target, predictions)
    prec = precision_score(test_target, predictions, zero_division=0)
    rec  = recall_score(test_target, predictions, zero_division=0)

    return {
        'k': k,
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


# ---------------------------------------------------------------------------
# 7. REPORT GENERATION
# ---------------------------------------------------------------------------

def create_feature_report(face_data: Dict[str, List[Dict[str, Any]]]) -> None:
    """
    Generate a short text report with KNN gender classification results.

    Contents:
        - Confusion matrix
        - Accuracy, Precision, Recall
        - Per-image prediction table

    The report is printed to stdout AND saved to facial_features_report.txt.

    Args:
        face_data: Output of read_face_database().
    """
    R: List[str] = []  # accumulator for report lines

    R.append("=" * 80)
    R.append("HW2: Gender Classification from Face Images using KNN")
    R.append("CSCI 405 / CIS 605")
    R.append("=" * 80)
    R.append("")

    # Run KNN classification
    knn_result = run_knn_classification(face_data, k=3)

    # Confusion matrix
    cm = knn_result['confusion_matrix']
    R.append("Confusion Matrix:")
    R.append("                  Predicted Male  Predicted Female")
    R.append(f"  Actual Male       {cm[0][0]:>10}  {cm[0][1]:>16}")
    R.append(f"  Actual Female     {cm[1][0]:>10}  {cm[1][1]:>16}")
    R.append("")

    # Performance metrics
    R.append(f"Accuracy  : {knn_result['accuracy']:.4f}  "
             f"({knn_result['accuracy']*100:.1f}%)")
    R.append(f"Precision : {knn_result['precision']:.4f}")
    R.append(f"Recall    : {knn_result['recall']:.4f}")
    R.append("")

    # Per-image prediction table
    label_map = {0: 'Male', 1: 'Female'}
    pred_rows: List[Dict[str, str]] = []
    idx = 0
    for person_id in knn_result['test_persons']:
        for file_data in face_data[person_id]:
            if len(file_data['points']) < 22:
                continue
            actual  = knn_result['ground_truth'][idx]
            pred    = knn_result['predictions'][idx]
            correct = "YES" if actual == pred else "NO"
            pred_rows.append({
                'File':      Path(file_data['file']).name,
                'Actual':    label_map[actual],
                'Predicted': label_map[pred],
                'Correct':   correct,
            })
            idx += 1

    pred_headers = ['File', 'Actual', 'Predicted', 'Correct']
    R.append(format_table(pred_rows, pred_headers))
    R.append("")
    R.append("=" * 80)

    # ---- Write report to file and print to console ----
    report_text = "\n".join(R)

    with open('facial_features_report.txt', 'w') as f:
        f.write(report_text)

    print(report_text)
    print("\n✓ Report saved to: facial_features_report.txt")


# ---------------------------------------------------------------------------
# 8. MAIN  –  Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Path to the face landmark database
    database_path = "/Users/austinmoser/PycharmProjects/FaceRecognition/FaceDatabase"

    # Step 1: Read all .pts files from every person folder
    print("Reading face database...")
    face_data = read_face_database(database_path)

    # Step 2: Generate the consolidated report (features + KNN results)
    print("\n" + "=" * 150)
    print("GENERATING REPORT...")
    print("=" * 150)
    create_feature_report(face_data)
