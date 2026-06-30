"""
Sorts OCR bounding boxes in the traditional Han reading order:
columns right → left, within each column top → bottom.
"""


def sort(lines: list[str], bbox_meta: list[dict] | None = None) -> list[str]:
    """
    Re-order OCR output lines by Han reading order.

    Args:
        lines:     Raw OCR text lines.
        bbox_meta: Optional list of bbox dicts {"x": ..., "y": ..., "w": ..., "h": ...}
                   corresponding to each line. If None, return lines as-is.

    Returns:
        Lines reordered right-to-left, top-to-bottom.
    """
    if not bbox_meta or len(lines) != len(bbox_meta):
        return lines

    # Filter out items with missing or invalid bounding boxes
    paired = []
    for line, box in zip(lines, bbox_meta):
        if box and all(k in box for k in ["x", "y", "w", "h"]):
            paired.append((line, box))
        else:
            paired.append((line, None))

    # If we have no valid bounding boxes, return lines as-is
    valid_paired = [item for item in paired if item[1] is not None]
    if not valid_paired:
        return lines

    # Compute average box width to use as column clustering threshold
    widths = [box["w"] for _, box in valid_paired]
    avg_w = sum(widths) / len(widths) if widths else 30
    col_threshold = avg_w * 0.8

    # Sort all valid boxes by x-center descending (right-to-left)
    valid_paired.sort(key=lambda item: item[1]["x"] + item[1]["w"] / 2, reverse=True)

    columns = []  # list of columns, each column is a list of (line, box)
    for item in valid_paired:
        line, box = item
        x_c = box["x"] + box["w"] / 2

        # Check if this box belongs to any existing column
        placed = False
        for col in columns:
            col_xs = [b["x"] + b["w"] / 2 for _, b in col]
            col_avg_x = sum(col_xs) / len(col_xs)
            if abs(x_c - col_avg_x) < col_threshold:
                col.append(item)
                placed = True
                break

        if not placed:
            columns.append([item])

    # Sort columns by their average x-center descending (right to left)
    columns.sort(key=lambda col: sum(b["x"] + b["w"] / 2 for _, b in col) / len(col), reverse=True)

    # Within each column, sort items top to bottom (y-coordinate ascending)
    sorted_lines = []
    for col in columns:
        col.sort(key=lambda item: item[1]["y"])
        for line, _ in col:
            sorted_lines.append(line)

    # Append lines that had no bounding box at the end
    for line, box in paired:
        if box is None:
            sorted_lines.append(line)

    return sorted_lines

