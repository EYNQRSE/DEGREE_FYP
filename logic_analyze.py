import os
import glob
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps, ImageTk
from ultralytics import YOLO


class WoundProcessor:
    CLASS_NAMES = ["wound", "ruler"]

    def __init__(self, yolo_model_path, treated_model_path, untreated_model_path):
        self.yolo_model = YOLO(yolo_model_path)
        self.treated_model = tf.keras.models.load_model(treated_model_path)
        self.untreated_model = tf.keras.models.load_model(untreated_model_path)

    @staticmethod
    def clear_folder(folder_path):
        files = glob.glob(os.path.join(folder_path, '*'))
        for f in files:
            try:
                os.remove(f)
            except Exception as e:
                print(f"Error deleting file {f}: {e}")

    @staticmethod
    def crop_centered_patch(image_path, center_x, center_y, output_path, size=256):
        img = Image.open(image_path)
        img = ImageOps.exif_transpose(img)
        width, height = img.size
        half = size // 2
        left = int(center_x - half)
        top = int(center_y - half)
        left = max(0, min(left, width - size))
        top = max(0, min(top, height - size))
        right = left + size
        bottom = top + size
        cropped_img = img.crop((left, top, right, bottom))
        cropped_img.save(output_path)
        return output_path

    @staticmethod
    def segment_wound(input_path, output_path):
        img = cv2.imread(input_path)
        mask = np.zeros(img.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        h, w = img.shape[:2]
        rect = (10, 10, w - 20, h - 20)
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        segmented = img.copy()
        segmented[mask2 == 0] = [255, 255, 255]
        cv2.imwrite(output_path, segmented)
        return segmented

    @staticmethod
    def preprocess_for_classification(image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, (224, 224))
        image = tf.cast(image, tf.float32) / 255.0
        return tf.expand_dims(image, axis=0)

    def classify_wound(self, image_path, model_type):
        model = self.treated_model if model_type == "treated" else self.untreated_model
        label_map = {0: "Day 0", 1: "Day 7", 2: "Day 10", 3: "Others"} if model_type == "treated" else {
            0: "Day 0", 1: "Day 7", 2: "Day 10", 3: "Day 15", 4: "Others"}
        img_tensor = self.preprocess_for_classification(image_path)
        pred = model.predict(img_tensor, verbose=0)
        pred_class = int(np.argmax(pred))
        return pred_class, label_map.get(pred_class, f"Unknown {pred_class}")

    @staticmethod
    def get_pixels_per_mm(ruler_image_path):
        img = cv2.imread(ruler_image_path, cv2.IMREAD_GRAYSCALE)
        width_pixels = img.shape[1]
        return width_pixels / 10

    @staticmethod
    def load_binary_mask(path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        _, mask = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        return mask

    @staticmethod
    def calculate_wound_area(mask, pixels_per_mm):
        wound_pixels = np.sum(mask == 0)
        return wound_pixels * (1 / pixels_per_mm) ** 2

    @staticmethod
    def load_resized_image_for_display(image_path, target_width=400, target_height=300):
        img = Image.open(image_path)
        img = ImageOps.exif_transpose(img)  # Correct orientation if needed
        img.thumbnail((target_width, target_height))  # Resize proportionally
        tk_img = ImageTk.PhotoImage(img)
        return tk_img, img  # Return both


    @staticmethod
    def overlay_mask_on_image(cropped_img_path, mask_path, output_path):
        image = cv2.imread(cropped_img_path)
        mask = WoundProcessor.load_binary_mask(mask_path)
        inverted_mask = cv2.bitwise_not(mask)
        mask_3ch = cv2.merge([inverted_mask] * 3)
        red_overlay = np.full_like(image, (0, 0, 255))
        alpha = 0.6
        blended = np.where(mask_3ch == 255,
                           cv2.addWeighted(image, 1 - alpha, red_overlay, alpha, 0),
                           image)
        cv2.imwrite(output_path, blended)
        return output_path
    
    @staticmethod
    def pil_to_tk_image(img, width, height):
        return ImageTk.PhotoImage(img.resize((width, height)))

    
    def create_overlay_with_ruler(self, segmented_path, ruler_path):
        seg_img = Image.open(segmented_path).convert("RGBA")
        ruler_img = Image.open(ruler_path).convert("RGBA")

        # Resize both images to the same size if needed
        if seg_img.size != ruler_img.size:
            ruler_img = ruler_img.resize(seg_img.size)

        # Overlay them
        combined = Image.blend(seg_img, ruler_img, alpha=0.5)
        return combined

    def analyze_image(self, image_path, cropped_folder, gui_callback=None, result_metadata=None):
        os.makedirs(cropped_folder, exist_ok=True)
        results = self.yolo_model.predict(source=image_path, save=False, imgsz=256)
        result = results[0]

        if not hasattr(result, 'boxes') or result.boxes is None or len(result.boxes.xyxy) == 0:
            if gui_callback:
                gui_callback('log', 'No wound or ruler detected.')
            return None, None

        wound_path = ruler_path = None
        img_name = os.path.basename(image_path)

        for box, cls_tensor in zip(result.boxes.xyxy, result.boxes.cls):
            x1, y1, x2, y2 = map(int, box.tolist())
            cls = int(cls_tensor.item())
            label = self.CLASS_NAMES[cls]

            if label == "wound":
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                wound_path = os.path.join(cropped_folder, f"wound_{img_name}")
                self.crop_centered_patch(image_path, cx, cy, wound_path)
                if gui_callback:
                    gui_callback('wound_image', wound_path)

            elif label == "ruler":
                ruler_path = os.path.join(cropped_folder, f"ruler_{img_name}")
                pil_img = Image.open(image_path)
                pil_img = ImageOps.exif_transpose(pil_img)
                pil_crop = pil_img.crop((x1, y1, x2, y2))
                pil_crop.save(ruler_path)
                if gui_callback:
                    gui_callback('ruler_image', ruler_path)
                if result_metadata is not None:
                    result_metadata['ruler_crop_box'] = (x1, y1, x2, y2)  # Save the coordinates used


        if not wound_path or not ruler_path:
            if gui_callback:
                gui_callback('log', 'Missing wound or ruler image.')
            return None, None

        return wound_path, ruler_path

    def segment_image(self, wound_path, segmented_folder, gui_callback=None):
        os.makedirs(segmented_folder, exist_ok=True)

        segmented_path = os.path.join(
            segmented_folder, f"seg_{os.path.basename(wound_path)}"
        )
        overlay_path = os.path.join(
            segmented_folder, f"overlay_{os.path.basename(wound_path)}"
        )

        # --- core processing -------------------------------------------------
        self.segment_wound(wound_path, segmented_path)
        self.overlay_mask_on_image(wound_path, segmented_path, overlay_path)

        # --- GUI previews ----------------------------------------------------
        if gui_callback:
            # Overlay preview
            tk_overlay_img, _ = self.load_resized_image_for_display(
                overlay_path, 400, 300
            )
            gui_callback("overlay_image", tk_overlay_img)

            # Segmented‐mask preview
            tk_seg_img, _ = self.load_resized_image_for_display(
                segmented_path, 400, 300
            )
            gui_callback("segmented_image", tk_seg_img)

        return segmented_path, overlay_path

    def calculate_metrics(self, segmented_path, ruler_path, model_type, gui_callback=None):
        pred_class, pred_label = self.classify_wound(segmented_path, model_type)
        binary_mask = self.load_binary_mask(segmented_path)
        ppm = self.get_pixels_per_mm(ruler_path)
        area = self.calculate_wound_area(binary_mask, ppm)

        if gui_callback:
            gui_callback('ppm', f"{ppm:.2f} pixel/mm")
            gui_callback('classification', pred_label)
            gui_callback('area', f"{area:.2f} mm²")

            # NEW: Send overlay image
            overlay_img = self.create_overlay_with_ruler(segmented_path, ruler_path)
            tk_overlay = WoundProcessor.pil_to_tk_image(overlay_img, width=400, height=300)
            gui_callback('ruler_overlay', tk_overlay)

        return pred_label, area
    
    def generate_result(self, overlay_path, mouse_ref, position, date, classification, area, gui_callback=None):
        result_data = {
            "image_path": overlay_path,
            "mouse_ref": mouse_ref or "N/A",
            "position": position or "N/A",
            "date": date or "N/A",
            "classification": classification,
            "area": area  # store as float
        }

        if gui_callback:
            gui_callback("result_card", result_data)   # ← pass the same dict!

        return result_data

    def run_pipeline(self, image_path, cropped_folder, segmented_folder, model_type, gui_callback=None, mouse_ref=None, position=None, date=None):
        wound_path, ruler_path = self.analyze_image(image_path, cropped_folder, gui_callback)
        if not wound_path or not ruler_path:
            return

        segmented_path, overlay_path = self.segment_image(wound_path, segmented_folder, gui_callback)
        classification, area = self.calculate_metrics(segmented_path, ruler_path, model_type, gui_callback)
        self.generate_result(overlay_path, mouse_ref, position, date, classification, area, gui_callback)
