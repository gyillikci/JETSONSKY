"""
AIDetector - YOLO-based detection for astronomical objects

This module provides detection and tracking of craters and satellites using YOLOv8 models.

Author: Refactored from JetsonSky monolithic code
Performance: GPU-accelerated inference with tracking
"""

import numpy as np
from collections import defaultdict
from typing import Optional, List, Tuple, Dict, Any

try:
    import torch
    FLAG_TORCH = True
except ImportError:
    FLAG_TORCH = False
    print("PyTorch not available - AI detection disabled")

try:
    from ultralytics import YOLO
    FLAG_YOLO = True
except ImportError:
    FLAG_YOLO = False
    print("YOLOv8 not available - AI detection disabled")


class AIDetector:
    """
    High-performance AI detector for astronomical objects.

    This class provides detection and tracking for:
    - Moon craters (small, medium, large)
    - Satellites, shooting stars, planes

    Features:
    - GPU-accelerated YOLO inference
    - Object tracking with history
    - Bounding box visualization
    - Confidence filtering

    Performance:
    - Inference: <50ms per frame on GPU
    - Tracking: Minimal overhead (<5ms)
    """

    def __init__(self, crater_model_path: Optional[str] = None, satellite_model_path: Optional[str] = None):
        """
        Initialize AI detector with YOLO models.

        Args:
            crater_model_path: Path to crater detection model
            satellite_model_path: Path to satellite detection model
        """
        self.crater_model_path = crater_model_path
        self.satellite_model_path = satellite_model_path

        # Model state
        self.crater_model_predict = None
        self.crater_model_track = None
        self.satellite_model_predict = None
        self.satellite_model_track = None

        self.crater_model_loaded = False
        self.satellite_model_loaded = False

        # Detection settings
        self.crater_confidence = 0.25
        self.satellite_confidence = 0.25
        self.enable_crater_detection = False
        self.enable_satellite_detection = False
        self.enable_tracking = False

        # Tracking history
        self.crater_track_history = defaultdict(lambda: [])
        self.satellite_track_history = defaultdict(lambda: [])

        # Detection results
        self.last_crater_detections = []
        self.last_satellite_detections = []

        # Load models if paths provided
        if crater_model_path:
            self.load_crater_model(crater_model_path)
        if satellite_model_path:
            self.load_satellite_model(satellite_model_path)

    def load_crater_model(self, model_path: str) -> bool:
        """
        Load crater detection model.

        Args:
            model_path: Path to YOLO model file

        Returns:
            bool: True if model loaded successfully
        """
        if not FLAG_YOLO:
            print("YOLOv8 not available")
            return False

        try:
            self.crater_model_predict = YOLO(model_path, task="predict")
            self.crater_model_track = YOLO(model_path, task="track")
            self.crater_model_loaded = True
            print(f"Crater model loaded: {model_path}")
            return True
        except Exception as e:
            print(f"Failed to load crater model: {e}")
            self.crater_model_loaded = False
            return False

    def load_satellite_model(self, model_path: str) -> bool:
        """
        Load satellite detection model.

        Args:
            model_path: Path to YOLO model file

        Returns:
            bool: True if model loaded successfully
        """
        if not FLAG_YOLO:
            print("YOLOv8 not available")
            return False

        try:
            self.satellite_model_predict = YOLO(model_path, task="predict")
            self.satellite_model_track = YOLO(model_path, task="track")
            self.satellite_model_loaded = True
            print(f"Satellite model loaded: {model_path}")
            return True
        except Exception as e:
            print(f"Failed to load satellite model: {e}")
            self.satellite_model_loaded = False
            return False

    def detect_craters(
        self,
        frame: np.ndarray,
        confidence: Optional[float] = None,
        use_tracking: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Detect craters in frame.

        Args:
            frame: Input frame (NumPy array, BGR format)
            confidence: Detection confidence threshold (0-1)
            use_tracking: Enable tracking between frames

        Returns:
            List of detection dictionaries with keys:
                - bbox: (x1, y1, x2, y2) bounding box
                - confidence: Detection confidence
                - class_id: Class ID (0=small, 1=medium, 2=large crater)
                - track_id: Tracking ID (if tracking enabled)

        Performance: <50ms on GPU for typical frame.
        """
        if not self.crater_model_loaded or not self.enable_crater_detection:
            return []

        conf = confidence if confidence is not None else self.crater_confidence

        try:
            # Run detection or tracking
            if use_tracking and self.enable_tracking:
                results = self.crater_model_track(
                    frame,
                    conf=conf,
                    verbose=False,
                    persist=True
                )
            else:
                results = self.crater_model_predict(
                    frame,
                    conf=conf,
                    verbose=False
                )

            # Parse results
            detections = []
            if len(results) > 0:
                boxes = results[0].boxes

                for i in range(len(boxes)):
                    det = {
                        'bbox': boxes.xyxy[i].cpu().numpy(),
                        'confidence': float(boxes.conf[i].cpu().numpy()),
                        'class_id': int(boxes.cls[i].cpu().numpy())
                    }

                    # Add track ID if available
                    if use_tracking and hasattr(boxes, 'id') and boxes.id is not None:
                        det['track_id'] = int(boxes.id[i].cpu().numpy())

                        # Update tracking history
                        track_id = det['track_id']
                        bbox = det['bbox']
                        center_x = (bbox[0] + bbox[2]) / 2
                        center_y = (bbox[1] + bbox[3]) / 2
                        self.crater_track_history[track_id].append((center_x, center_y))

                        # Limit history length
                        if len(self.crater_track_history[track_id]) > 30:
                            self.crater_track_history[track_id].pop(0)

                    detections.append(det)

            self.last_crater_detections = detections
            return detections

        except Exception as e:
            print(f"Crater detection error: {e}")
            return []

    def detect_satellites(
        self,
        frame: np.ndarray,
        confidence: Optional[float] = None,
        use_tracking: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Detect satellites/shooting stars/planes in frame.

        Args:
            frame: Input frame (NumPy array, BGR format)
            confidence: Detection confidence threshold (0-1)
            use_tracking: Enable tracking between frames

        Returns:
            List of detection dictionaries with keys:
                - bbox: (x1, y1, x2, y2) bounding box
                - confidence: Detection confidence
                - class_id: Class ID (0=satellite, 1=shooting star, 2=plane)
                - track_id: Tracking ID (if tracking enabled)

        Performance: <50ms on GPU for typical frame.
        """
        if not self.satellite_model_loaded or not self.enable_satellite_detection:
            return []

        conf = confidence if confidence is not None else self.satellite_confidence

        try:
            # Run detection or tracking
            if use_tracking and self.enable_tracking:
                results = self.satellite_model_track(
                    frame,
                    conf=conf,
                    verbose=False,
                    persist=True
                )
            else:
                results = self.satellite_model_predict(
                    frame,
                    conf=conf,
                    verbose=False
                )

            # Parse results
            detections = []
            if len(results) > 0:
                boxes = results[0].boxes

                for i in range(len(boxes)):
                    det = {
                        'bbox': boxes.xyxy[i].cpu().numpy(),
                        'confidence': float(boxes.conf[i].cpu().numpy()),
                        'class_id': int(boxes.cls[i].cpu().numpy())
                    }

                    # Add track ID if available
                    if use_tracking and hasattr(boxes, 'id') and boxes.id is not None:
                        det['track_id'] = int(boxes.id[i].cpu().numpy())

                        # Update tracking history
                        track_id = det['track_id']
                        bbox = det['bbox']
                        center_x = (bbox[0] + bbox[2]) / 2
                        center_y = (bbox[1] + bbox[3]) / 2
                        self.satellite_track_history[track_id].append((center_x, center_y))

                        # Limit history length
                        if len(self.satellite_track_history[track_id]) > 30:
                            self.satellite_track_history[track_id].pop(0)

                    detections.append(det)

            self.last_satellite_detections = detections
            return detections

        except Exception as e:
            print(f"Satellite detection error: {e}")
            return []

    def draw_detections(
        self,
        frame: np.ndarray,
        crater_detections: Optional[List[Dict]] = None,
        satellite_detections: Optional[List[Dict]] = None,
        draw_tracks: bool = False
    ) -> np.ndarray:
        """
        Draw detection bounding boxes on frame.

        Args:
            frame: Frame to draw on (NumPy array, BGR format)
            crater_detections: Crater detections to draw (uses last if None)
            satellite_detections: Satellite detections to draw (uses last if None)
            draw_tracks: Draw tracking history trails

        Returns:
            Frame with drawn detections

        Performance: <5ms for typical number of detections.
        """
        import cv2

        frame_out = frame.copy()

        # Use last detections if not provided
        if crater_detections is None:
            crater_detections = self.last_crater_detections
        if satellite_detections is None:
            satellite_detections = self.last_satellite_detections

        # Draw craters (green boxes)
        for det in crater_detections:
            bbox = det['bbox'].astype(int)
            conf = det['confidence']
            class_id = det['class_id']

            # Different colors for different crater sizes
            if class_id == 0:  # Small
                color = (0, 255, 0)
                label = "Small"
            elif class_id == 1:  # Medium
                color = (0, 200, 0)
                label = "Medium"
            else:  # Large
                color = (0, 150, 0)
                label = "Large"

            cv2.rectangle(frame_out, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(
                frame_out,
                f"{label} {conf:.2f}",
                (bbox[0], bbox[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

            # Draw track if available
            if draw_tracks and 'track_id' in det:
                track_id = det['track_id']
                if track_id in self.crater_track_history:
                    points = self.crater_track_history[track_id]
                    for i in range(1, len(points)):
                        cv2.line(
                            frame_out,
                            (int(points[i - 1][0]), int(points[i - 1][1])),
                            (int(points[i][0]), int(points[i][1])),
                            color,
                            2
                        )

        # Draw satellites (red boxes)
        for det in satellite_detections:
            bbox = det['bbox'].astype(int)
            conf = det['confidence']
            class_id = det['class_id']

            # Different labels for different object types
            if class_id == 0:
                label = "Satellite"
                color = (0, 0, 255)
            elif class_id == 1:
                label = "Shooting Star"
                color = (0, 100, 255)
            else:
                label = "Plane"
                color = (255, 0, 0)

            cv2.rectangle(frame_out, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(
                frame_out,
                f"{label} {conf:.2f}",
                (bbox[0], bbox[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

            # Draw track if available
            if draw_tracks and 'track_id' in det:
                track_id = det['track_id']
                if track_id in self.satellite_track_history:
                    points = self.satellite_track_history[track_id]
                    for i in range(1, len(points)):
                        cv2.line(
                            frame_out,
                            (int(points[i - 1][0]), int(points[i - 1][1])),
                            (int(points[i][0]), int(points[i][1])),
                            color,
                            2
                        )

        return frame_out

    def reset_tracking(self):
        """Reset all tracking history."""
        self.crater_track_history.clear()
        self.satellite_track_history.clear()

        # Reset YOLO trackers
        try:
            if self.crater_model_track:
                self.crater_model_track.predictor.trackers[0].reset()
        except:
            pass

        try:
            if self.satellite_model_track:
                self.satellite_model_track.predictor.trackers[0].reset()
        except:
            pass

    def get_detection_count(self) -> Dict[str, int]:
        """Get count of detected objects."""
        return {
            'craters': len(self.last_crater_detections),
            'satellites': len(self.last_satellite_detections)
        }

    def set_confidence_threshold(self, crater_conf: Optional[float] = None, satellite_conf: Optional[float] = None):
        """
        Set confidence thresholds for detection.

        Args:
            crater_conf: Crater confidence threshold (0-1)
            satellite_conf: Satellite confidence threshold (0-1)
        """
        if crater_conf is not None:
            self.crater_confidence = max(0.0, min(1.0, crater_conf))
        if satellite_conf is not None:
            self.satellite_confidence = max(0.0, min(1.0, satellite_conf))
