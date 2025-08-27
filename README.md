# 🧪 AI-Driven Wound Monitoring System for Diabetic Mice

This project is a desktop-based wound monitoring system developed to assist researchers in tracking the healing progress of diabetic wounds in lab mice. It uses computer vision and AI (YOLO and CNN) to detect, segment, classify, and calculate wound areas from images in a semi-automated workflow.

---

## 📋 Features

- 🖼️ **Automatic Wound Detection** using YOLOv8
- ✂️ **Image Cropping** based on bounding box
- 🔍 **Image Segmentation** using GrabCut
- 🧠 **Healing Day Classification** using a trained CNN (Day 0, 7, 10, 15, Others)
- 📐 **Wound Area Calculation**
- 🖥️ **User Interface** with Tkinter
- 📄 **Final Annotated Output** with classification and wound size overlaid

---

## 🛠️ System Requirements

- **Operating System**: Windows 10/11, Linux (tested), macOS (Tkinter setup needed)
- **Python Version**: 3.8 or newer

### Python Dependencies

Install all dependencies using:

```bash
pip install -r requirements.txt

wound-monitoring-system/
│
├── main_ui.py             # GUI application (entry point)
├── logic_analyze.py       # Image processing and AI logic
├── models/                # Trained CNN model (classifier.tflite or .h5)
├── sample_images/         # Sample test images
├── results/               # Output images with annotations
├── requirements.txt       # Python dependencies
└── README.md              # This file
