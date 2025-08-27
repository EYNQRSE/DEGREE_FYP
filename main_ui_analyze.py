import tkinter as tk
from pathlib import Path
import shutil
import numpy as np
from tkinter import ttk, filedialog, scrolledtext
from PIL import Image, ImageTk, ImageDraw, ImageFont, ImageOps
import os
from logic_analyze import WoundProcessor
import glob
import time


# === Initialize Processor ===
processor = WoundProcessor(
    yolo_model_path=r"C:\Users\AMIR\MouseMonitoringSystem\WoundDetection\runs\detect\DetectionV25\weights\best.pt",
    treated_model_path=r"C:\Users\AMIR\MouseMonitoringSystem\WoundClassification\runs\model\model_treatedV8.keras",
    untreated_model_path=r"C:\Users\AMIR\MouseMonitoringSystem\WoundClassification\runs\model\model_untreatedV8.keras"
)

# === Base Window ===
root = tk.Tk()
root.title("Mouse Wound Monitoring System")
root.geometry("1400x800")

# === Left Frame ===
left_frame = tk.Frame(root, width=400)
left_frame.pack(side='left', fill='y', padx=10, pady=10)

original_image_label = tk.Label(left_frame, bg="gray")
original_image_label.pack()

controls_frame = tk.LabelFrame(left_frame, text="Controls & Metadata", padx=10, pady=10)
controls_frame.pack(fill='x', pady=10)

selected_image_path = tk.StringVar()

result_metadata = {
    'segmented_img_path': None,
    'wound_area': None,
    'wound_path': None,     
    'ruler_path': None,     
    'result_card': None,
    'overlay_img_path': None,
    'classification': None,
    'wound_area': None,
    'ruler_crop_box': None, 
}

def select_image():
    path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png")],
        title="Select Image"
    )
    if not path:
        return

    selected_image_path.set(path)  # Store path (if needed)

    # Open image and convert to Tkinter-compatible PhotoImage
    img = Image.open(path)
    img = img.resize((225, 300), Image.LANCZOS)  # You can change size
    img_display = ImageTk.PhotoImage(img)       # ✅ convert to Tk-compatible

    original_image_label.config(image=img_display)
    original_image_label.image = img_display  # ✅ must keep reference

def get_latest_file(folder, pattern="*.jpg"):
    files = glob.glob(os.path.join(folder, pattern))
    if not files:
        return None
    # Sort by modified time
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]

def go_to_next_tab():
    current = notebook.index(notebook.select())
    notebook.select(current + 1)

def gui_callback(key, value):
    """Handle UI updates coming back from the WoundProcessor layer."""
    # --- Cropped wound -----------------------------------------------------------------
    if key == 'wound_image':
        tk_img, _ = WoundProcessor.load_resized_image_for_display(
            value, 400, 300
        )
        cropped_wound_canvas.delete("all")                         # clear old
        x_center = int(cropped_wound_canvas['width'])  // 2
        y_center = int(cropped_wound_canvas['height']) // 2
        cropped_wound_canvas.create_image(
            x_center, y_center, anchor='center', image=tk_img
        )
        cropped_wound_canvas.image = tk_img                        # keep ref

    # --- Cropped ruler -----------------------------------------------------------------
    elif key == 'ruler_image':
        tk_img, _ = WoundProcessor.load_resized_image_for_display(
            value, 400, 300
        )
        cropped_ruler_canvas.delete("all")
        x_center = int(cropped_ruler_canvas['width'])  // 2
        y_center = int(cropped_ruler_canvas['height']) // 2
        cropped_ruler_canvas.create_image(
            x_center, y_center, anchor='center', image=tk_img
        )
        cropped_ruler_canvas.image = tk_img

    # --- Segmented wound (already a PhotoImage coming from logic) ----------------------
    elif key == 'segmented_image':
        segmented_wound_canvas.delete("all")
        x_center = int(segmented_wound_canvas['width'])  // 2
        y_center = int(segmented_wound_canvas['height']) // 2
        segmented_wound_canvas.create_image(
            x_center, y_center, anchor='center', image=value
        )
        segmented_wound_canvas.image = value

    # --- Overlay wound (PhotoImage) ----------------------------------------------------
    elif key == 'overlay_image':
        overlay_wound_canvas.delete("all")
        x_center = int(overlay_wound_canvas['width'])  // 2
        y_center = int(overlay_wound_canvas['height']) // 2
        overlay_wound_canvas.create_image(
            x_center, y_center, anchor='center', image=value
        )
        overlay_wound_canvas.image = value

    # --- Final result card -------------------------------------------------------------
    elif key == "result_card":
        # --- show the image -------------------------------------------------
        tk_img, pil_img = WoundProcessor.load_resized_image_for_display(
            value["image_path"], 400, 300
        )
        result_data["image_label"].config(image=tk_img)
        result_data["image_label"].image = tk_img
        result_data["image_label"].pil   = pil_img

        # --- update text labels --------------------------------------------
        result_data["labels"]["mouse"].config(text=value["mouse_ref"])
        result_data["labels"]["position"].config(text=value["position"])
        result_data["labels"]["date"].config(text=value["date"])
        result_data["labels"]["day"].config(text=value["classification"])
        result_data["labels"]["area"].config(text=f"{value['area']:.2f} mm²")  # ← format here

        # --- store raw metadata for export ---------------------------------
        result_metadata["mouse_ref"]      = value["mouse_ref"]
        result_metadata["position"]       = value["position"]
        result_metadata["date"]           = value["date"]
        result_metadata["classification"] = value["classification"]
        result_metadata["wound_area"]     = float(value["area"])   # ← keep as number
        result_metadata["overlay_img_path"] = value["image_path"]

        result_container.update_idletasks()

    # --- Simple log, metadata, and status updates --------------------------------------
    elif key == 'classification':
        log_text.insert(tk.END, f"Classification: {value}\n")

    elif key == 'area':
        log_text.insert(tk.END, f"Wound Area: {value}\n")
        result_metadata['wound_area'] = value

    elif key == 'segmented_img_path':
        result_metadata['segmented_img_path'] = value

    elif key == 'overlay_img_path':
        result_metadata['overlay_img_path'] = value

    elif key == 'ppm':
        ppm_var.set(f"Pixel per mm: {value}")

    elif key == 'log':
        log_text.insert(tk.END, f"{value}\n")

    elif key == 'ruler_overlay':
        cropped_ruler_canvas.delete("all")
        x_center = int(cropped_ruler_canvas['width'])  // 2
        y_center = int(cropped_ruler_canvas['height']) // 2
        cropped_ruler_canvas.create_image(
            x_center, y_center, anchor='center', image=value
        )
        cropped_ruler_canvas.image = value

# === Right Frame ===
right_frame = tk.Frame(root, width=300)
right_frame.pack(side='right', fill='y', padx=10, pady=10)

log_label = tk.Label(right_frame, text="Logs")
log_label.pack()

log_text = scrolledtext.ScrolledText(right_frame, wrap=tk.WORD, width=40, height=40)
log_text.pack(fill='both', expand=False)

# === Control Widgets ===
def run_pipeline_ui():
    step_start = time.time()
    wound_path, ruler_path = analyze_image_ui(timed=False)
    if not wound_path or not ruler_path:
        return

    segmented_path, overlay_path = segment_image_ui(wound_path,timed=False)
    classification, area = calculate_metrics_ui(segmented_path, ruler_path, timed=False)
    generate_result_ui(overlay_path, classification, area, timed=False)
    duration = time.time() - step_start
    log_text.insert("end", f"[TIME] Quick Analyze took {duration:.2f} sec\n")

def analyze_image_ui(timed=True):
    if timed:
        step_start = time.time()
    image_path = selected_image_path.get()
    if not image_path:
        log_text.insert(tk.END, "No image selected.\n")
        return None, None

    WoundProcessor.clear_folder("output/cropped")
    WoundProcessor.clear_folder("output/segmented")

    cropped_wound_canvas.delete("all")
    cropped_ruler_canvas.delete("all")
    log_text.delete(1.0, tk.END)

    wound_path, ruler_path = processor.analyze_image(image_path, "output/cropped", gui_callback, result_metadata)
    
    if wound_path and ruler_path:
        result_metadata['wound_path'] = wound_path
        result_metadata['ruler_path'] = ruler_path
        log_text.insert(tk.END, f"Wound image saved at: {wound_path}\n")
        log_text.insert(tk.END, f"Ruler image saved at: {ruler_path}\n")
    else:
        log_text.insert(tk.END, "Failed to analyze image. Please check input.\n")
    
    if timed:
        duration = time.time() - step_start
        log_text.insert("end", f"[TIME] Analyze Image took {duration:.2f} sec\n")

    return wound_path, ruler_path

tk.Button(controls_frame, text="Select Image", command=select_image).grid(row=0, column=0, columnspan=2, sticky="w")

model_var = tk.StringVar(value="treated")
tk.Label(controls_frame, text="Model:").grid(row=1, column=0, sticky="w")
tk.Radiobutton(controls_frame, text="Treated", variable=model_var, value="treated").grid(row=1, column=1, sticky="w")
tk.Radiobutton(controls_frame, text="Untreated", variable=model_var, value="untreated").grid(row=2, column=1, sticky="w")

tk.Label(controls_frame, text="Mouse Ref No:").grid(row=3, column=0, sticky="w")
mouse_entry = tk.Entry(controls_frame)
mouse_entry.grid(row=3, column=1)

tk.Label(controls_frame, text="Position:").grid(row=4, column=0, sticky="w")
position_entry = tk.Entry(controls_frame)
position_entry.grid(row=4, column=1)

tk.Label(controls_frame, text="Date:").grid(row=5, column=0, sticky="w")
date_entry = tk.Entry(controls_frame)
date_entry.grid(row=5, column=1)
  
tk.Button(controls_frame, text="Analyze", command=analyze_image_ui).grid(row=6, column=0, columnspan=2, pady=5)
tk.Button(controls_frame, text="Quick Analyze", command=run_pipeline_ui).grid(row=6, column=2, columnspan=2, pady=5)

# === Center Frame ===
center_frame = tk.Frame(root, bg="white")
center_frame.pack(side='left', expand=True, fill='both', padx=10, pady=10)

# Notebook for tabs
notebook = ttk.Notebook(center_frame)
notebook.pack(expand=True, fill='both')

# Create tabs
seg_tab = tk.Frame(notebook, bg="white")
calc_tab = tk.Frame(notebook, bg="white")
res_tab = tk.Frame(notebook, bg="white")

# Add tabs to notebook
notebook.add(seg_tab, text="Segmentation")
notebook.add(calc_tab, text="Calculation")
notebook.add(res_tab, text="Result")

# === Segmentation Tab ===

def segment_image_ui(wound_path, timed=True):
    if timed:
        step_start = time.time()
    segmented_path, overlay_path = processor.segment_image(wound_path, "output/segmented", gui_callback)
    log_text.insert(tk.END, f"Segmented image saved at: {segmented_path}\n")
    log_text.insert(tk.END, f"Overlay image saved at: {overlay_path}\n")
    if segmented_path:
        result_metadata['segmented_img_path'] = segmented_path
    if overlay_path:
        result_metadata['overlay_img_path'] = overlay_path
    if timed:
        duration = time.time() - step_start
        log_text.insert("end", f"[TIME] Segment Image took {duration:.2f} sec\n")
    return segmented_path, overlay_path

def resegment_ui():
    wound_path = result_metadata.get("wound_path")
    if not wound_path or not os.path.exists(wound_path):
        log_text.insert(tk.END, "No valid cropped wound image found for segmentation.\n")
        return
    log_text.insert(tk.END, f"Using wound path for segmentation: {wound_path}\n")
    segment_image_ui(wound_path)

seg_images_frame = tk.Frame(seg_tab)
seg_images_frame.pack(pady=10)

cropped_wound_canvas = tk.Canvas(seg_images_frame, bg="lightgray", width=400, height=300)
cropped_wound_canvas.grid(row=0, column=0, padx=5)

segmented_wound_canvas = tk.Canvas(seg_images_frame, bg="lightgray", width=400, height=300)
segmented_wound_canvas.grid(row=0, column=1, padx=5)

overlay_wound_canvas = tk.Canvas(seg_tab, bg="lightgray", width=800, height=300)
overlay_wound_canvas.pack(pady=10)

seg_button_frame = tk.Frame(seg_tab)
seg_button_frame.pack(pady=5)

button_frame_top = tk.Frame(seg_button_frame)
button_frame_top.pack()

tk.Button(seg_button_frame, text="Resegment", command=resegment_ui).pack(pady=5)
tk.Button(seg_tab, text="Next",  command= go_to_next_tab,width=15).pack(pady=10)

# === Calculation Tab ===
def calculate_metrics_ui(segmented_path, ruler_path,timed=True):
    if timed:
        step_start = time.time()
    model_type = model_var.get()
    classification, area = processor.calculate_metrics(segmented_path, ruler_path, model_type, gui_callback)
    if classification:
        result_metadata['classification'] = classification
    if area:
        result_metadata['wound_area'] = area
    if timed:
        duration = time.time() - step_start
        log_text.insert("end", f"[TIME] Calculate Metrics took {duration:.2f} sec\n")   
    return classification, area

def calculate_area_ui():
    segmented_path = result_metadata.get("segmented_img_path")
    ruler_path = result_metadata.get("ruler_path")

    if not segmented_path or not os.path.exists(segmented_path):
        log_text.insert(tk.END, "No valid segmented image found for area calculation.\n")
        return

    if not ruler_path or not os.path.exists(ruler_path):
        log_text.insert(tk.END, "No valid ruler image found for area calculation.\n")
        return

    calculate_metrics_ui(segmented_path, ruler_path)

calc_content = tk.Frame(calc_tab)
calc_content.pack(expand=False)
calc_content.columnconfigure(0, weight=1)
calc_content.columnconfigure(1, weight=1)

cropped_ruler_canvas = tk.Canvas(calc_content, bg="lightgray", width=400, height=300)
cropped_ruler_canvas.grid(row=0, column=0, padx=10, pady=10)

right_calc_frame = tk.Frame(calc_content)
right_calc_frame.grid(row=0, column=1, padx=10, pady=10)

ppm_var = tk.StringVar(value="Pixel per mm: N/A")  

ppm_label = tk.Label(right_calc_frame, textvariable=ppm_var, bg="white", relief="sunken", width=30)
ppm_label.pack(pady=5)

tk.Button(right_calc_frame, text="Calculate", command=calculate_area_ui).pack(pady=5)

tk.Button(calc_tab, text="Next", command=go_to_next_tab, width=15).pack(pady=(0, 20))

# === Result Tab ===
def create_result_card_image(overlay_path, metadata, save_path):
    """Overlay only Day + Wound Area onto the image and save it"""
    img = Image.open(overlay_path).convert("RGBA")
    img = ImageOps.exif_transpose(img)          # keep orientation

    draw  = ImageDraw.Draw(img)
    font  = ImageFont.load_default()            # swap in truetype if you want
    pad_x, pad_y = 10, 10

    text_lines = [
        f"Day (Classification): {metadata['classification']}",
        f"Wound Area: {metadata['wound_area']:.2f} mm²"
    ]

    for i, line in enumerate(text_lines):
        y = pad_y + i * 18                      # 18 px line‑spacing
        draw.text((pad_x, y), line, fill="white", font=font,
                  stroke_width=1, stroke_fill="black")  # outline for contrast

    img.save(save_path)

def generate_result_ui(overlay_path, classification, area, timed=True):
    if timed:
        step_start = time.time()
    mouse_ref = mouse_entry.get()
    position = position_entry.get()
    date = date_entry.get()
    
    # Ensure the image exists and is properly sized
    if not os.path.exists(overlay_path):
        log_text.insert(tk.END, f"[ERROR] Overlay image not found at {overlay_path}\n")
        if timed:
            duration = time.time() - step_start
            log_text.insert("end", f"[TIME] Generate Result took {duration:.2f} sec\n")
        return
    
    # Process the image with consistent dimensions
    try:
        img = Image.open(overlay_path)
        img = img.resize((400, 300), Image.LANCZOS)  # Force exact size
        temp_path = "output/result_display.png"
        img.save(temp_path)

        if timed:
            duration = time.time() - step_start
            log_text.insert("end", f"[TIME] Generate Result took {duration:.2f} sec\n")
        
        return processor.generate_result(temp_path, mouse_ref, position, date, classification, area, gui_callback)
    except Exception as e:
        log_text.insert(tk.END, f"[ERROR] Failed to process result image: {e}\n")

# === Final Result Container (includes card + buttons) ===
result_container = tk.Frame(res_tab, bg="white")
result_container.pack(fill="both", expand=True, padx=10, pady=10)

# === Final Result Card ===
final_result_card = tk.Frame(result_container, bg="white", relief="groove", bd=2)
final_result_card.pack(fill="both", expand=True, padx=10, pady=10)

# Image frame with flexible sizing
image_frame = tk.Frame(final_result_card, bg="white")
image_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

# Result image label without fixed dimensions
result_image_label = tk.Label(image_frame, bg="gray")
result_image_label.pack(fill="both", expand=True)

# Right: Description
desc_frame = tk.Frame(final_result_card, bg="white")
desc_frame.grid(row=0, column=1, sticky="nsew", padx=20)

result_labels = {}
for i, (label_text, key) in enumerate([
    ("Mouse Ref No:", "mouse"),
    ("Position:", "position"),
    ("Date:", "date"),
    ("Day (Classification):", "day"),
    ("Wound Area:", "area")
]):
    tk.Label(desc_frame, text=label_text, anchor="w", bg="white", font=("Arial", 10, "bold")).grid(row=i, column=0, sticky="w")
    result_labels[key] = tk.Label(desc_frame, text="N/A", bg="white", anchor="w", width=25)
    result_labels[key].grid(row=i, column=1, sticky="w")

# Configure grid weights
final_result_card.grid_columnconfigure(0, weight=1)
final_result_card.grid_columnconfigure(1, weight=1)

# Store references globally if needed
result_data = {
    "image_label": result_image_label,
    "labels": result_labels
}

# === Buttons Below the Card ===
def generate_report():
    """Generates the result card using the same function as the full pipeline"""
    try:
        # Get the latest overlay image path from metadata
        overlay_path = result_metadata.get('overlay_img_path')
        
        if not overlay_path or not os.path.exists(overlay_path):
            log_text.insert(tk.END, "[ERROR] No processed image found. Run analysis first.\n")
            return

        # Get classification and area from previous results (or calculate fresh)
        classification = result_metadata.get("classification")
        area = result_metadata.get("wound_area")
   
        # Regenerate the result card
        generate_result_ui(overlay_path, classification, area)
        log_text.insert(tk.END, "[SUCCESS] Result card regenerated\n")
        
    except Exception as e:
        log_text.insert(tk.END, f"[ERROR] Report generation failed: {e}\n")

def import_result_image():
    # === Save full result card with metadata ===
    overlay_path = result_metadata.get("overlay_img_path")
    if not overlay_path or not os.path.exists(overlay_path):
        log_text.insert(tk.END, "[ERROR] No overlay image available to save.\n")
        return

    card_path = filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=[("PNG files", "*.png")],
        title="Save Result Card As"
    )
    if card_path:
        try:
            create_result_card_image(overlay_path, result_metadata, card_path)
            log_text.insert(tk.END, f"[SUCCESS] Result card saved to: {card_path}\n")
        except Exception as e:
            log_text.insert(tk.END, f"[ERROR] Failed to save result card: {e}\n")

# Button frame
res_button_frame = tk.Frame(result_container, bg="white")
res_button_frame.pack(fill="x", pady=(0, 10))

# Buttons
tk.Button(res_button_frame, 
          text="Generate Result", 
          width=20, 
          command=generate_report).pack(side="left", padx=10, expand=True)

tk.Button(res_button_frame, 
          text="Export Image", 
          width=20, 
          command=import_result_image).pack(side="left", padx=10, expand=True)

# === Main Loop ===
root.mainloop()
