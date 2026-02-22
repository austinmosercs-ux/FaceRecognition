import math
from pathlib import Path
from typing import List, Dict, Tuple, Any


def read_pts_file(file_path: str) -> Dict[str, Any]:
    """
    Read a .pts file and extract metadata and points.

    Args:
        file_path: Path to the .pts file

    Returns:
        Dictionary containing version, n_points, and points list
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

        # Parse header information
        for i, line in enumerate(lines):
            line = line.strip()

            if line.startswith('version:'):
                result['version'] = int(line.split(':')[1].strip())
            elif line.startswith('n_points:'):
                result['n_points'] = int(line.split(':')[1].strip())
            elif line == '{':
                # Start of points
                # Extract points until we hit the closing brace
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
                                pass
                break

    except Exception as e:
        print(f"Error reading file {file_path}: {e}")

    return result


def read_face_database(database_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Read all .pts files from the face database and extract all points.

    Args:
        database_path: Path to the FaceDatabase folder

    Returns:
        Dictionary mapping person IDs to their face data
    """
    database_path = Path(database_path)

    if not database_path.exists():
        raise FileNotFoundError(f"Database path not found: {database_path}")

    face_data = {}

    # Iterate through all person folders
    for person_folder in sorted(database_path.iterdir()):
        if not person_folder.is_dir():
            continue

        person_id = person_folder.name
        face_data[person_id] = []

        # Read all .pts files in the person's folder
        for pts_file in sorted(person_folder.glob('*.pts')):
            points_data = read_pts_file(str(pts_file))
            face_data[person_id].append(points_data)

    return face_data


def euclidean_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """
    Calculate Euclidean distance between two points.

    Args:
        point1: (x, y) coordinates of first point
        point2: (x, y) coordinates of second point

    Returns:
        Euclidean distance between the two points
    """
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def extract_features(points: List[Tuple[float, float]]) -> Dict[str, float]:
    """
    Extract 7 facial features from facial landmark points.

    Points are indexed from 1-22 (converting to 0-21 for list indexing).

    Args:
        points: List of (x, y) tuples for all facial landmarks (must have at least 22 points)

    Returns:
        Dictionary containing all 7 features with their calculated values
    """
    if len(points) < 22:
        raise ValueError(f"Expected at least 22 points, got {len(points)}")

    features = {}

    # Convert to 0-indexed
    # Points are referenced as 1-based in the feature definitions

    try:
        # Feature 1: Eye length ratio
        # Length of eye (maximum of two) over distance between points 8 and 13
        left_eye_length = euclidean_distance(points[4], points[5])  # points 5 and 6 (1-indexed)
        right_eye_length = euclidean_distance(points[6], points[7])  # points 7 and 8 (1-indexed)
        max_eye_length = max(left_eye_length, right_eye_length)
        dist_8_13 = euclidean_distance(points[7], points[12])  # points 8 and 13 (1-indexed)
        features['eye_length_ratio'] = max_eye_length / dist_8_13 if dist_8_13 != 0 else 0

        # Feature 2: Eye distance ratio
        # Distance between center of two eyes over distance between points 8 and 13
        left_eye_center = ((points[4][0] + points[5][0]) / 2, (points[4][1] + points[5][1]) / 2)
        right_eye_center = ((points[6][0] + points[7][0]) / 2, (points[6][1] + points[7][1]) / 2)
        eye_center_distance = euclidean_distance(left_eye_center, right_eye_center)
        features['eye_distance_ratio'] = eye_center_distance / dist_8_13 if dist_8_13 != 0 else 0

        # Feature 3: Nose ratio
        # Distance between points 15 and 16 over distance between 20 and 21
        dist_15_16 = euclidean_distance(points[14], points[15])  # points 15 and 16 (1-indexed)
        dist_20_21 = euclidean_distance(points[19], points[20])  # points 20 and 21 (1-indexed)
        features['nose_ratio'] = dist_15_16 / dist_20_21 if dist_20_21 != 0 else 0

        # Feature 4: Lip size ratio
        # Distance between points 2 and 3 over distance between 17 and 18
        dist_2_3 = euclidean_distance(points[1], points[2])  # points 2 and 3 (1-indexed)
        dist_17_18 = euclidean_distance(points[16], points[17])  # points 17 and 18 (1-indexed)
        features['lip_size_ratio'] = dist_2_3 / dist_17_18 if dist_17_18 != 0 else 0

        # Feature 5: Lip length ratio
        # Distance between points 2 and 3 over distance between 20 and 21
        features['lip_length_ratio'] = dist_2_3 / dist_20_21 if dist_20_21 != 0 else 0

        # Feature 6: Eye-brow length ratio
        # Distance between points 4 and 5 (or distance between points 6 and 7 whichever is larger)
        # over distance between 8 and 13
        eyebrow_left = euclidean_distance(points[3], points[4])  # points 4 and 5 (1-indexed)
        eyebrow_right = euclidean_distance(points[5], points[6])  # points 6 and 7 (1-indexed)
        max_eyebrow = max(eyebrow_left, eyebrow_right)
        features['eyebrow_length_ratio'] = max_eyebrow / dist_8_13 if dist_8_13 != 0 else 0

        # Feature 7: Aggressive ratio
        # Distance between points 10 and 19 over distance between 20 and 21
        dist_10_19 = euclidean_distance(points[9], points[18])  # points 10 and 19 (1-indexed)
        features['aggressive_ratio'] = dist_10_19 / dist_20_21 if dist_20_21 != 0 else 0

    except IndexError as e:
        print(f"Error extracting features: {e}")
        return {}

    return features


def print_face_database_summary(face_data: Dict[str, List[Dict[str, Any]]]) -> None:
    """
    Print a summary of all extracted face points and features.

    Args:
        face_data: Dictionary of face data from read_face_database
    """
    total_files = 0
    total_points = 0

    for person_id, files in face_data.items():
        print(f"\n{'='*80}")
        print(f"Person ID: {person_id}")
        print(f"{'='*80}")

        for file_data in files:
            total_files += 1
            file_name = Path(file_data['file']).name
            n_points = len(file_data['points'])
            total_points += n_points

            print(f"\n  File: {file_name}")
            print(f"  Version: {file_data['version']}")
            print(f"  Expected Points: {file_data['n_points']}")
            print(f"  Actual Points: {n_points}")

            # Extract and display features
            if n_points >= 22:
                features = extract_features(file_data['points'])
                print(f"\n  EXTRACTED FEATURES:")
                print(f"    1. Eye Length Ratio: {features.get('eye_length_ratio', 'N/A'):.6f}")
                print(f"    2. Eye Distance Ratio: {features.get('eye_distance_ratio', 'N/A'):.6f}")
                print(f"    3. Nose Ratio: {features.get('nose_ratio', 'N/A'):.6f}")
                print(f"    4. Lip Size Ratio: {features.get('lip_size_ratio', 'N/A'):.6f}")
                print(f"    5. Lip Length Ratio: {features.get('lip_length_ratio', 'N/A'):.6f}")
                print(f"    6. Eyebrow Length Ratio: {features.get('eyebrow_length_ratio', 'N/A'):.6f}")
                print(f"    7. Aggressive Ratio: {features.get('aggressive_ratio', 'N/A'):.6f}")
            else:
                print(f"  ERROR: Insufficient points for feature extraction (need 22, have {n_points})")

    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Total Files Read: {total_files}")
    print(f"Total Points Extracted: {total_points}")


def collect_all_features(face_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, List[float]]]:
    """
    Collect all features for each person across all their files.

    Args:
        face_data: Dictionary of face data from read_face_database

    Returns:
        Dictionary mapping person IDs to their feature collections
    """
    person_features = {}

    for person_id, files in face_data.items():
        person_features[person_id] = {
            'eye_length_ratio': [],
            'eye_distance_ratio': [],
            'nose_ratio': [],
            'lip_size_ratio': [],
            'lip_length_ratio': [],
            'eyebrow_length_ratio': [],
            'aggressive_ratio': []
        }

        for file_data in files:
            if len(file_data['points']) >= 22:
                features = extract_features(file_data['points'])
                for feature_name, value in features.items():
                    person_features[person_id][feature_name].append(value)

    return person_features


def format_table(rows: List[Dict[str, str]], headers: List[str]) -> str:
    """
    Format a list of dict rows into a fixed-width text table.
    """
    if not rows:
        return ""

    widths = {header: len(header) for header in headers}
    for row in rows:
        for header in headers:
            widths[header] = max(widths[header], len(str(row.get(header, ""))))

    header_line = "  ".join(header.ljust(widths[header]) for header in headers)
    separator_line = "  ".join("-" * widths[header] for header in headers)

    data_lines = []
    for row in rows:
        data_lines.append("  ".join(str(row.get(header, "")).ljust(widths[header]) for header in headers))

    return "\n".join([header_line, separator_line] + data_lines)


def create_feature_report(face_data: Dict[str, List[Dict[str, Any]]]) -> None:
    """
    Create a single consolidated text report with all facial features analysis.
    """
    person_features = collect_all_features(face_data)

    feature_names = [
        'eye_length_ratio',
        'eye_distance_ratio',
        'nose_ratio',
        'lip_size_ratio',
        'lip_length_ratio',
        'eyebrow_length_ratio',
        'aggressive_ratio'
    ]

    person_ids = sorted(person_features.keys())

    total_files = sum(len(files) for files in face_data.values())
    total_points = sum(len(file_data['points']) for files in face_data.values() for file_data in files)

    # Generate report
    report_lines = []
    report_lines.append("=" * 150)
    report_lines.append("FACIAL FEATURES ANALYSIS REPORT")
    report_lines.append("=" * 150)
    report_lines.append("")
    report_lines.append(f"Total Individuals: {len(person_ids)}")
    report_lines.append(f"Total Files Analyzed: {total_files}")
    report_lines.append(f"Total Facial Landmarks Extracted: {total_points}")
    report_lines.append("")

    # Section 1: Detailed Features by File
    report_lines.append("=" * 150)
    report_lines.append("SECTION 1: DETAILED FACIAL FEATURES BY FILE")
    report_lines.append("=" * 150)
    report_lines.append("")

    all_files_data = []
    for person_id, files in face_data.items():
        for file_data in files:
            if len(file_data['points']) >= 22:
                features = extract_features(file_data['points'])
                all_files_data.append({
                    'Person': person_id,
                    'File': Path(file_data['file']).name,
                    'Eye Length': f"{features['eye_length_ratio']:.4f}",
                    'Eye Distance': f"{features['eye_distance_ratio']:.4f}",
                    'Nose Ratio': f"{features['nose_ratio']:.4f}",
                    'Lip Size': f"{features['lip_size_ratio']:.4f}",
                    'Lip Length': f"{features['lip_length_ratio']:.4f}",
                    'Eyebrow Length': f"{features['eyebrow_length_ratio']:.4f}",
                    'Aggressive': f"{features['aggressive_ratio']:.4f}"
                })

    headers_files = [
        'Person', 'File', 'Eye Length', 'Eye Distance', 'Nose Ratio',
        'Lip Size', 'Lip Length', 'Eyebrow Length', 'Aggressive'
    ]
    report_lines.append(format_table(all_files_data, headers_files))
    report_lines.append("")

    # Section 2: Average Features by Person
    report_lines.append("=" * 150)
    report_lines.append("SECTION 2: AVERAGE FACIAL FEATURES BY PERSON")
    report_lines.append("=" * 150)
    report_lines.append("")

    avg_data = []
    for person_id in person_ids:
        row = {'Person': person_id}
        for feature_name in feature_names:
            values = person_features[person_id][feature_name]
            avg = sum(values) / len(values) if values else 0
            row[feature_name.replace('_', ' ').title()] = f"{avg:.4f}"
        avg_data.append(row)

    headers_avg = ['Person'] + [name.replace('_', ' ').title() for name in feature_names]
    report_lines.append(format_table(avg_data, headers_avg))
    report_lines.append("")

    # Section 3: Gender Comparison
    report_lines.append("=" * 150)
    report_lines.append("SECTION 3: GENDER COMPARISON - AVERAGE FACIAL FEATURES")
    report_lines.append("=" * 150)
    report_lines.append("")

    male_ids = [p for p in person_ids if p.startswith('m')]
    female_ids = [p for p in person_ids if p.startswith('w')]

    gender_data = []
    for feature_name in feature_names:
        row = {'Feature': feature_name.replace('_', ' ').title()}

        male_values = []
        for person_id in male_ids:
            male_values.extend(person_features[person_id][feature_name])
        male_avg = sum(male_values) / len(male_values) if male_values else 0
        row['Male Average'] = f"{male_avg:.4f}"

        female_values = []
        for person_id in female_ids:
            female_values.extend(person_features[person_id][feature_name])
        female_avg = sum(female_values) / len(female_values) if female_values else 0
        row['Female Average'] = f"{female_avg:.4f}"

        diff = male_avg - female_avg
        row['Difference'] = f"{diff:.4f}"

        gender_data.append(row)

    headers_gender = ['Feature', 'Male Average', 'Female Average', 'Difference']
    report_lines.append(format_table(gender_data, headers_gender))
    report_lines.append("")

    report_lines.append("=" * 150)
    report_lines.append("")

    report_text = "\n".join(report_lines)

    with open('facial_features_report.txt', 'w') as f:
        f.write(report_text)

    print(report_text)
    print("\n✓ Report saved to: facial_features_report.txt")


# Example usage
if __name__ == "__main__":
    # Read the face database
    database_path = "/Users/austinmoser/PycharmProjects/FaceRecognition/FaceDatabase"

    print("Reading face database...")
    face_data = read_face_database(database_path)

    # Generate consolidated text report
    print("\n" + "="*150)
    print("GENERATING REPORT...")
    print("="*150)
    create_feature_report(face_data)
