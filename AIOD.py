"""AI Object Detection System - Standalone Desktop Application."""

import tkinter as tk
from tkinter import filedialog, messagebox

from aiod.core.detector import get_detector
from aiod.core.models import DetectionResult
from aiod.utils.image_utils import (
    load_image_for_display,
    bgr_array_to_display_image,
    to_tk_photo_image,
)
from aiod.utils.exceptions import InvalidImageError, ModelError

# Initialize shared YOLO detector
try:
    detector = get_detector()
except Exception as init_err:
    detector = None
    _init_error_message = str(init_err)

selected_image = None


# ==========================================
# MAIN WINDOW
# ==========================================

root = tk.Tk()
root.title("AI Object Detection System")
root.geometry("1100x750")
root.configure(bg="#202020")


# ==========================================
# TITLE
# ==========================================

title = tk.Label(
    root,
    text="AI OBJECT DETECTION SYSTEM",
    font=("Arial", 24, "bold"),
    fg="white",
    bg="#202020",
)
title.pack(pady=15)

subtitle = tk.Label(
    root,
    text="Object Detection using YOLO",
    font=("Arial", 12),
    fg="lightgray",
    bg="#202020",
)
subtitle.pack()


# ==========================================
# IMAGE DISPLAY
# ==========================================

image_frame = tk.Frame(
    root,
    bg="#303030",
    width=800,
    height=430,
)
image_frame.pack(pady=15)
image_frame.pack_propagate(False)

image_label = tk.Label(
    image_frame,
    text="No image selected",
    font=("Arial", 18),
    fg="white",
    bg="#303030",
)
image_label.pack(expand=True)


# ==========================================
# BUTTON FRAME
# ==========================================

button_frame = tk.Frame(root, bg="#202020")
button_frame.pack(pady=10)


# ==========================================
# SELECT IMAGE
# ==========================================

def select_image():
    global selected_image

    file_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[
            ("Image Files", "*.jpg *.jpeg *.png *.webp *.bmp"),
            ("All Files", "*.*"),
        ],
    )

    if not file_path:
        return

    try:
        display_img, _ = load_image_for_display(file_path, max_size=(780, 420))
        photo = to_tk_photo_image(display_img)

        selected_image = file_path
        image_label.config(image=photo, text="")
        image_label.image = photo

        result_label.config(text="Image selected. Click Detect Objects.")
    except InvalidImageError as err:
        messagebox.showerror("Invalid Image", str(err))
    except Exception as err:
        messagebox.showerror("Image Error", f"Could not load image:\n{err}")


# ==========================================
# DETECT OBJECTS
# ==========================================

def detect_objects():
    global detector

    if detector is None:
        messagebox.showerror(
            "Model Error",
            f"YOLO model could not be initialized:\n{_init_error_message}",
        )
        return

    if selected_image is None:
        messagebox.showwarning(
            "No Image",
            "Please select an image first.",
        )
        return

    result_label.config(text="Detecting objects...")
    root.update_idletasks()

    try:
        # Run detection via shared engine
        result: DetectionResult = detector.detect(selected_image, conf=0.25)

        # Render annotated bounding boxes
        if result.annotated_array is not None:
            annotated_display = bgr_array_to_display_image(
                result.annotated_array, max_size=(780, 420)
            )
            photo = to_tk_photo_image(annotated_display)
            image_label.config(image=photo, text="")
            image_label.image = photo

        # Update detection summary label (exact legacy format preserved)
        result_label.config(text=result.formatted_summary)

    except InvalidImageError as err:
        messagebox.showerror("Image Error", str(err))
        result_label.config(text="Detection failed: Invalid image.")
    except ModelError as err:
        messagebox.showerror("Detection Error", str(err))
        result_label.config(text="Detection failed.")
    except Exception as err:
        messagebox.showerror("Error", f"Unexpected error during detection:\n{err}")
        result_label.config(text="Detection failed.")


# ==========================================
# BUTTONS
# ==========================================

select_button = tk.Button(
    button_frame,
    text="SELECT IMAGE",
    command=select_image,
    font=("Arial", 12, "bold"),
    width=20,
    height=2,
)
select_button.grid(row=0, column=0, padx=20)

detect_button = tk.Button(
    button_frame,
    text="DETECT OBJECTS",
    command=detect_objects,
    font=("Arial", 12, "bold"),
    width=20,
    height=2,
)
detect_button.grid(row=0, column=1, padx=20)


# ==========================================
# RESULT LABEL
# ==========================================

result_label = tk.Label(
    root,
    text="Select an image to begin",
    font=("Arial", 12),
    fg="white",
    bg="#202020",
    wraplength=1000,
)
result_label.pack(pady=10)


# ==========================================
# START APPLICATION
# ==========================================

if __name__ == "__main__":
    root.mainloop()
