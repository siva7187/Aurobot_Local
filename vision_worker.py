"""Local webcam emotion detection in a separate OS process."""

import os
import time


def _process_frame(frame, frame_width, cv2_module):
    height, width = frame.shape[:2]
    if width > frame_width:
        scale = frame_width / width
        frame = cv2_module.resize(
            frame, (frame_width, int(height * scale))
        )
    return cv2_module.cvtColor(frame, cv2_module.COLOR_BGR2RGB)


def run(
    result_queue,
    poll_seconds: float = 2.5,
    frame_width: int = 320,
    source: str = "webcam",
    detector_backend: str = "mtcnn",
) -> None:
    """Read local webcam frames and send emotion results to the parent process."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

    if source != "webcam":
        result_queue.put({
            "error": "Only the local webcam source is supported."
        })
        return

    if os.environ.get("AUROBOT_VISION_DEVICE", "auto").lower() == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(
                "[vision_worker] GPU acceleration enabled: "
                f"{[gpu.name for gpu in gpus]}"
            )
        except RuntimeError as exc:
            print(
                "[vision_worker] GPU found but memory-growth "
                f"configuration failed: {exc}"
            )
    else:
        print("[vision_worker] No GPU visible to TensorFlow; using CPU")

    import cv2
    from deepface import DeepFace

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        result_queue.put({"error": "camera_not_accessible"})
        cap.release()
        return

    last_error = None
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            try:
                rgb_frame = _process_frame(frame, frame_width, cv2)
                start = time.time()
                result = DeepFace.analyze(
                    rgb_frame,
                    actions=["emotion"],
                    enforce_detection=False,
                    detector_backend=detector_backend,
                )
                analyze_ms = (time.time() - start) * 1000
                emotion_scores = result[0]["emotion"]
                result_queue.put({
                    "emotion_scores": emotion_scores,
                    "analyze_ms": analyze_ms,
                })
                last_error = None
            except Exception as exc:
                message = str(exc)
                if message != last_error:
                    result_queue.put({"error": message})
                    last_error = message

            time.sleep(poll_seconds)
    finally:
        cap.release()
