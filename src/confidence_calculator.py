def levenshtein_distance(s1, s2):
    """
    Calculate the Levenshtein Distance between two strings.

    The Levenshtein distance is a measure of the difference between two strings,
    defined as the minimum number of single-character edits (insertions,
    deletions, or substitutions) required to change one string into the other.

    Args:
        s1 (str): The first string.
        s2 (str): The second string.

    Returns:
        int: The Levenshtein Distance between the two strings.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def calculate_confidence(ground_truth, prediction):
    """
    Calculate a confidence score for OCR prediction based on the Levenshtein Distance.

    This function computes a normalized confidence score that represents the
    accuracy of the OCR prediction compared to the ground truth. The score is
    based on the Levenshtein Distance between the predicted text and the
    ground truth, normalized to a range between 0 and 1.

    Args:
        ground_truth (str): The correct text (ground truth).
        prediction (str): The OCR predicted text.

    Returns:
        float: A confidence score between 0 and 1.
    """
    distance = levenshtein_distance(ground_truth, prediction)
    max_len = max(len(ground_truth), len(prediction))
    if max_len == 0:  # Avoid division by zero
        return 1.0
    confidence_score = 1 - (distance / max_len)
    return confidence_score
