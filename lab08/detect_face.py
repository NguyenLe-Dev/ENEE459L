import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw
import warnings
from scipy.ndimage import convolve, gaussian_filter

MODEL_PATH = "version-RFB-320.onnx"

warnings.filterwarnings("ignore", category=UserWarning)

sess_options = ort.SessionOptions()
sess_options.log_severity_level = 3  

# 3. Use only available providers to prevent the provider fallback warning
desired_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
available_providers = [p for p in desired_providers if p in ort.get_available_providers()]

# face detection
def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    if box_scores.ndim != 2 or box_scores.shape[1] != 5:
        raise ValueError("box_scores must have shape (N, 5): [x1, y1, x2, y2, score]")

    if box_scores.shape[0] == 0 or top_k == 0 or candidate_size == 0:
        return box_scores[:0]

    boxes = box_scores[:, :4]
    scores = box_scores[:, 4]

    # Sort ascending so the highest-scoring candidate is at the end.
    order = np.argsort(scores)
    if candidate_size > 0:
        order = order[-candidate_size:]

    picked = []
    while order.size:
        best = order[-1]
        picked.append(best)

        remaining = order[:-1]
        if remaining.size == 0 or (top_k > 0 and len(picked) >= top_k):
            break

        x1 = np.maximum(boxes[best, 0], boxes[remaining, 0])
        y1 = np.maximum(boxes[best, 1], boxes[remaining, 1])
        x2 = np.minimum(boxes[best, 2], boxes[remaining, 2])
        y2 = np.minimum(boxes[best, 3], boxes[remaining, 3])

        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

        best_area = (
            np.maximum(0, boxes[best, 2] - boxes[best, 0])
            * np.maximum(0, boxes[best, 3] - boxes[best, 1])
        )
        remaining_area = (
            np.maximum(0, boxes[remaining, 2] - boxes[remaining, 0])
            * np.maximum(0, boxes[remaining, 3] - boxes[remaining, 1])
        )

        union = best_area + remaining_area - intersection
        iou = np.zeros_like(intersection, dtype=float)
        np.divide(intersection, union, out=iou, where=union > 0)
        
        order = remaining[iou <= iou_threshold]

    return box_scores[picked]
def detect_faces(image_path, output_path="output_detected.png", conf_threshold=0.7):
    with Image.open(image_path) as source:
        image = source.convert("RGB")

    original_width, original_height = image.size
    resized = image.resize((320, 240), Image.BILINEAR)
    input_tensor = np.asarray(resized, dtype=np.float32)
    input_tensor = (input_tensor - 127.0) / 128.0
    input_tensor = np.transpose(input_tensor, (2, 0, 1))[np.newaxis, ...]

    session = ort.InferenceSession(
        MODEL_PATH,
        sess_options=sess_options,
        providers=available_providers,
    )
    input_name = session.get_inputs()[0].name
    scores, boxes = session.run(
        ["scores", "boxes"],
        {input_name: input_tensor},
    )
    scores = np.squeeze(scores, axis=0)
    boxes = np.squeeze(boxes, axis=0)

    face_scores = scores[:, 1]
    keep = face_scores >= conf_threshold
    candidates = np.column_stack((boxes[keep], face_scores[keep]))
    detections = hard_nms(candidates, iou_threshold=0.3)

    draw = ImageDraw.Draw(image)
    for x1, y1, x2, y2, score in detections:
        x1 = int(np.clip(x1 * original_width, 0, original_width - 1))
        y1 = int(np.clip(y1 * original_height, 0, original_height - 1))
        x2 = int(np.clip(x2 * original_width, 0, original_width - 1))
        y2 = int(np.clip(y2 * original_height, 0, original_height - 1))

        draw.rectangle((x1, y1, x2, y2), outline="red", width=2)
        draw.text((x1, max(0, y1 - 12)), f"Face: {score:.2f}", fill="red")

    image.save(output_path)

# Edge detection
def non_maximum_suppression(magnitude, angle):
    magnitude = np.asarray(magnitude)
    angle = np.asarray(angle)
    if magnitude.ndim != 2 or angle.shape != magnitude.shape:
        raise ValueError("magnitude and angle must be same-shaped 2D arrays")

    thinned = np.zeros(magnitude.shape, dtype=np.float32)
    angle_degrees = np.rad2deg(angle) % 180.0
    height, width = magnitude.shape

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            direction = angle_degrees[i, j]
            if direction < 22.5 or direction >= 157.5:
                before, after = magnitude[i, j - 1], magnitude[i, j + 1]
            elif direction < 67.5:
                before, after = magnitude[i + 1, j - 1], magnitude[i - 1, j + 1]
            elif direction < 112.5:
                before, after = magnitude[i - 1, j], magnitude[i + 1, j]
            else:
                before, after = magnitude[i - 1, j - 1], magnitude[i + 1, j + 1]

            if magnitude[i, j] >= before and magnitude[i, j] >= after:
                thinned[i, j] = magnitude[i, j]

    return thinned

def hysteresis_thresholding(img, low_thresh, high_thresh):
    img = np.asarray(img)
    if img.ndim != 2:
        raise ValueError("img must be a 2D array")
    if low_thresh > high_thresh:
        raise ValueError("low_thresh must be less than or equal to high_thresh")

    strong = img >= high_thresh
    weak = (img >= low_thresh) & ~strong
    neighborhood = np.ones((3, 3), dtype=np.uint8)

    while np.any(weak):
        connected_to_strong = convolve(
            strong.astype(np.uint8),
            neighborhood,
            mode="constant",
            cval=0,
        ) > 0
        promoted = weak & connected_to_strong
        if not np.any(promoted):
            break
        strong |= promoted
        weak &= ~promoted

    return strong.astype(np.uint8) * 255

def canny_scipy(image_path, output_path="edges_scipy.png", low=20, high=60):
    with Image.open(image_path) as source:
        grayscale = source.convert("L")
        image = np.asarray(grayscale, dtype=np.float32)

    smoothed = gaussian_filter(image, sigma=1.4)
    sobel_x = np.array(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=np.float32,
    )
    sobel_y = np.array(
        [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
        dtype=np.float32,
    )
    gradient_x = convolve(smoothed, sobel_x)
    gradient_y = convolve(smoothed, sobel_y)

    magnitude = np.hypot(gradient_x, gradient_y)
    max_magnitude = magnitude.max()
    if max_magnitude > 0:
        magnitude = magnitude * (255.0 / max_magnitude)

    angle = np.arctan2(gradient_y, gradient_x)
    thinned = non_maximum_suppression(magnitude, angle)
    edges = hysteresis_thresholding(thinned, low, high)
    Image.fromarray(edges).save(output_path)

if __name__ == "__main__":
    detect_faces("face.jpg")
    canny_scipy("face.jpg")