import streamlit as st
import cv2
import tempfile
import os
import time
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Load YOLOv8 model
model_path = "runs/detect/train/weights/best.pt"

model = YOLO(model_path)

st.set_page_config(page_title="Smoking Detection App", layout="centered")
st.title("🚬 Cigarette/Smoking Detection using YOLOv8")

option = st.radio("Select input type:", ["Image", "Video", "Webcam"])


def draw_results(frame, results):
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])
            label = f"cigarette {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2,
            )
    return frame


if option == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Run detection
        results = model.predict(image, imgsz=640)
        result_image = draw_results(np.array(image.copy()), results)

        st.image(result_image, caption="Detection Result", use_container_width=True)

elif option == "Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
    if uploaded_video:
        # Create a temporary directory to store our files
        temp_dir = tempfile.mkdtemp()
        input_path = os.path.join(temp_dir, "input_video.mp4")
        output_path = os.path.join(temp_dir, "output_video.mp4")

        # Save the uploaded video to the temporary file
        with open(input_path, "wb") as f:
            f.write(uploaded_video.getbuffer())

        # Show original video
        st.video(uploaded_video)
        st.write("Original Video ⬆️")

        # Button to start processing
        if st.button("Start Detection on Video"):
            # Process the video
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                st.error("Error: Could not open video file.")
            else:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                # Always use mp4v codec for MP4 output
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

                # Add progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Process frames
                frame_count = 0
                preview_frame = None

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Update progress
                    frame_count += 1
                    progress = int(frame_count / total_frames * 100)
                    progress_bar.progress(progress)
                    status_text.text(
                        f"Processing frame {frame_count}/{total_frames} ({progress}%)"
                    )

                    # Process frame
                    results = model.predict(frame, imgsz=640)
                    annotated_frame = draw_results(frame.copy(), results)
                    out.write(annotated_frame)

                    # Store last frame for preview
                    preview_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

                # Release resources
                cap.release()
                out.release()

                try:
                    # Check if we need to convert the video format
                    # On some systems, mp4v codec might not play correctly in browsers
                    # Convert to a web-compatible format using FFmpeg if available
                    final_output_path = os.path.join(temp_dir, "final_output.mp4")

                    try:
                        # Try to use FFmpeg for conversion to ensure browser compatibility
                        import subprocess

                        ffmpeg_cmd = [
                            "ffmpeg",
                            "-y",
                            "-i",
                            output_path,
                            "-vcodec",
                            "libx264",
                            "-pix_fmt",
                            "yuv420p",  # Ensure compatibility
                            "-preset",
                            "fast",
                            final_output_path,
                        ]
                        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
                        video_path_to_use = final_output_path
                    except (ImportError, subprocess.SubprocessError, FileNotFoundError):
                        # If FFmpeg fails or isn't available, use the original output
                        st.warning(
                            "FFmpeg conversion not available. Video playback might not work in all browsers."
                        )
                        video_path_to_use = output_path

                    # Read the video as bytes
                    with open(video_path_to_use, "rb") as file:
                        video_bytes = file.read()

                    st.success(
                        "✅ Detection complete. Video ready to view and download."
                    )

                    st.write("Processed Video:")
                    # Use st.video with the bytes of the processed video
                    st.video(video_bytes)

                    st.download_button(
                        label="Download Processed Video (MP4)",
                        data=video_bytes,
                        file_name="processed_video.mp4",
                        mime="video/mp4",
                    )
                except Exception as e:
                    st.error(f"❌ Failed to process or display video: {e}")
                    st.error("Try downloading the video and playing it locally.")

                    # Offer direct download even if playback fails
                    try:
                        with open(output_path, "rb") as file:
                            video_bytes = file.read()

                        st.download_button(
                            label="Download Processed Video (MP4)",
                            data=video_bytes,
                            file_name="processed_video.mp4",
                            mime="video/mp4",
                        )
                    except Exception as download_error:
                        st.error(f"❌ Failed to prepare download: {download_error}")


elif option == "Webcam":
    st.warning(
        "Webcam detection in Streamlit has limitations. This implementation will capture frames until manually stopped."
    )

    # Create placeholders for the webcam interface
    start_button_placeholder = st.empty()
    stop_button_placeholder = st.empty()
    stframe = st.empty()
    status_text = st.empty()

    # Session state to track whether webcam is running
    if "webcam_running" not in st.session_state:
        st.session_state.webcam_running = False

    # Start button
    if (
        start_button_placeholder.button("Start Webcam Detection")
        and not st.session_state.webcam_running
    ):
        st.session_state.webcam_running = True

        # Replace with a message
        start_button_placeholder.text("Webcam active! Press Stop to end detection.")

        # Create a stop button
        stop_pressed = stop_button_placeholder.button(
            "Stop Webcam", key="stop_button_unique"
        )

        cap = cv2.VideoCapture(0)  # Use default camera (index 0)

        if not cap.isOpened():
            st.error(
                "Error: Could not access webcam. Check permissions and connections."
            )
        else:
            # Process frames until stop is pressed
            frame_count = 0

            while (
                cap.isOpened() and not stop_pressed and st.session_state.webcam_running
            ):
                ret, frame = cap.read()
                if not ret:
                    status_text.error("Error: Could not read frame from webcam.")
                    break

                # Process frame with model
                results = model.predict(frame, imgsz=640)
                annotated_frame = draw_results(frame.copy(), results)
                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

                # Display the frame
                stframe.image(
                    rgb_frame, channels="RGB", caption=f"Frame: {frame_count}"
                )
                frame_count += 1

                # Check if stop button was pressed (need to rerun script to detect)
                if stop_button_placeholder.button(
                    "Stop Webcam", key=f"stop_button_{frame_count}"
                ):
                    break

                # Add a small delay
                time.sleep(0.1)

            # Release resources
            cap.release()
            st.session_state.webcam_running = False
            status_text.text("Webcam stopped.")

    # Display message if webcam is already running
    elif st.session_state.webcam_running:
        start_button_placeholder.text("Webcam is already running.")
        if stop_button_placeholder.button("Stop Webcam", key="stop_running_webcam"):
            st.session_state.webcam_running = False
            status_text.text("Webcam stopped.")
