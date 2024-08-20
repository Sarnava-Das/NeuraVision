import cv2
import torch
import numpy as np

# Load the MiDaS model
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)
midas.eval()

# Load YOLO model (assuming you have a YOLO model loaded)
yolo_model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
yolo_model.to(device)
yolo_model.eval()

# Calibration factor (adjust this based on calibration)
calibration_factor = 0.03  # Example factor to convert normalized depth to meters
cm_to_feet = 0.0328084  # Conversion factor from centimeters to feet

# Distance threshold in feet
distance_threshold_feet = 2.5

# Capture video from the camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame for MiDaS
    input_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Apply the MiDaS transformation
    input_frame = midas_transforms(input_frame)
    
    # Ensure the tensor is 4D (batch_size, channels, height, width)
    if input_frame.dim() == 3:
        input_frame = input_frame.unsqueeze(0)

    input_frame = input_frame.to(device)

    with torch.no_grad():
        # Pass through the MiDaS model
        prediction = midas(input_frame)

        # Resize and normalize the prediction to match the original frame
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        depth_map = prediction.cpu().numpy()

    # Invert the depth map if necessary (depends on model output)
    depth_map = np.max(depth_map) - depth_map

    # Normalize depth map to 0-255 for visualization
    depth_map_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # Convert to color map
    depth_map_colored = cv2.applyColorMap(depth_map_vis, cv2.COLORMAP_MAGMA)

    # Run YOLO object detection
    results = yolo_model(frame)
    detections = results.xyxy[0].cpu().numpy()  # (x1, y1, x2, y2, conf, class)

    # Draw detections and calculate distances
    for det in detections:
        x1, y1, x2, y2, conf, cls = map(int, det[:6])
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # Convert depth value to real-world distance in centimeters
        depth_value = depth_map[center_y, center_x]
        distance_cm = depth_value * calibration_factor * 100  # Convert meters to centimeters
        distance_feet = distance_cm * cm_to_feet  # Convert centimeters to feet

        # Ensure the distance is realistic
        distance_feet = max(distance_feet, 0.1)  # Avoid negative or zero distance

        # Check if the object is within the desired distance range
        if distance_feet <= distance_threshold_feet:
            # Draw bounding box and distance text
            label = f"Class {cls}, Dist: {distance_feet:.2f} ft"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the original frame and the depth map
    cv2.imshow('Frame', frame)
    cv2.imshow('Depth Map', depth_map_colored)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
