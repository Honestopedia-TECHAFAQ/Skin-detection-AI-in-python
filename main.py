import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import ImageTk, Image
import json

CONFIG = {
    'save_frames': True,
    'save_video': True,
    'save_skin_mask': True
}


def process_image(image):
 
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    result = np.copy(image)

    for (x, y, w, h) in faces:
        center_x = x + w // 2
        center_y = y + h // 2
        radius = int(max(w, h) * 0.8) 
        cv2.circle(result, (center_x, center_y), radius, (0, 0, 0), -1)

    result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    ycrcb = cv2.cvtColor(result, cv2.COLOR_BGR2YCrCb)

    lower_skin_female = np.array([0, 135, 85], dtype=np.uint8)
    upper_skin_female = np.array([150, 180, 135], dtype=np.uint8)

    mask_skin = cv2.inRange(ycrcb, lower_skin_female, upper_skin_female)
    mask_combined = cv2.bitwise_and(mask_skin, result_gray)

    result = np.zeros_like(image)
    result[mask_combined > 0] = [255, 255, 255]

    return result


def process_image_and_save(image_path):
    image = cv2.imread(image_path)
    result = process_image(image)
    cv2.imshow('Skin Mask', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    save_image_to_json(result)
    
    output_image_path = image_path.replace('.jpg', '_processed.jpg')
    cv2.imwrite(output_image_path, result)
    messagebox.showinfo("Download Complete", "Image file saved successfully!")


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    output_path = filedialog.asksaveasfilename(defaultextension=".mp4")
    if not output_path:
        messagebox.showwarning("Download Canceled", "Video file download canceled.")
        return

    if CONFIG['save_video']:
        video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = process_image(frame)

        cv2.imshow('Skin Mask', result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if CONFIG['save_video']:
            video_writer.write(result)

    cap.release()
    cv2.destroyAllWindows()

    json_data = {'frames': []}
    if CONFIG['save_frames']:
        json_data['video_path'] = output_path

    with open(output_path.replace('.mp4', '.json'), 'w') as json_file:
        json.dump(json_data, json_file, indent=4)

    messagebox.showinfo("Download Complete", "Video file saved successfully!")


def process_webcam():
    cap = cv2.VideoCapture(0)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    output_path = filedialog.asksaveasfilename(defaultextension=".mp4")
    if not output_path:
        messagebox.showwarning("Download Canceled", "Video file download canceled.")
        return

    if CONFIG['save_video']:
        video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()

        result = process_image(frame)

        cv2.imshow('Skin Mask', result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if CONFIG['save_video']:
            video_writer.write(result)

    cap.release()
    cv2.destroyAllWindows()

    json_data = {'frames': []}
    if CONFIG['save_frames']:
        json_data['webcam'] = True

    with open(output_path.replace('.mp4', '.json'), 'w') as json_file:
        json.dump(json_data, json_file, indent=4)

    messagebox.showinfo("Download Complete", "Video file saved successfully!")


def select_image():
    root = tk.Tk()
    root.withdraw()
    image_path = filedialog.askopenfilename()

    if image_path:
        process_image_and_save(image_path)


def select_video():
    root = tk.Tk()
    root.withdraw()
    video_path = filedialog.askopenfilename()

    if video_path:
        process_video(video_path)


def save_image_to_json(image):
    json_data = {'image': image.tolist()}

    output_path = filedialog.asksaveasfilename(defaultextension=".json")
    if output_path:
        with open(output_path, 'w') as json_file:
            json.dump(json_data, json_file, indent=4)
        messagebox.showinfo("Download Complete", "JSON file saved successfully!")
    else:
        messagebox.showwarning("Download Canceled", "JSON file download canceled.")


def create_gui():
    root = tk.Tk()
    root.title("Skin Detection App")


    root.configure(bg='#F5F5F5')
    button_style = {'bg': '#4CAF50', 'fg': 'white',
                    'font': ('Helvetica', 12, 'bold')}

    def on_select_image():
        select_image()

    def on_select_video():
        select_video()

    def on_process_webcam():
        process_webcam()

    button_frame = tk.Frame(root, bg='#F5F5F5')
    button_frame.pack(pady=10)

    select_image_button = tk.Button(
        button_frame, text="Select Image", command=on_select_image, **button_style)
    select_image_button.pack(side=tk.LEFT, padx=5)

    select_video_button = tk.Button(
        button_frame, text="Select Video", command=on_select_video, **button_style)
    select_video_button.pack(side=tk.LEFT, padx=5)

    process_webcam_button = tk.Button(
        button_frame, text="Process Webcam", command=on_process_webcam, **button_style)
    process_webcam_button.pack(side=tk.LEFT, padx=5)

    root.mainloop()


create_gui()
