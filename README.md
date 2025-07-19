# food_and_drink01-PT

## 🍔 Food & Drink Detection Model – YOLOv8 (.pt)

Thank you for purchasing our custom-trained YOLOv8 model for food & drink detection. This model is trained to detect 15 common items found in restaurants, bars, and dining scenes.

---

### 📦 Included in this Download

- `food_& drink.pt` – PyTorch model for object detection  
- `food_& drink-seg.pt` – (optional) segmentation model  
- `dataset.yaml` – class labels and structure  
- `test_webcam.py` – script to test detection via webcam  
- `seg_webcam.py` – script to test segmentation model  
- `yolo_train.py` – script used for training  
- `README_FoodDetection_YOLOv8.md` – this file

---

### 🧠 Model Info

- Framework: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)  
- Format: PyTorch `.pt`  
- Classes: 15 (listed below)  
- Trained on custom dataset (`images/train`, `images/val`) with clear labeling  
- Resolution: recommended input 640x640 px  

---

### 🏷️ Class Names

```yaml
0: Singha Beer
1: Sangsom
2: Sangsom Box
3: Beer glass
4: Heineken Beer
5: Chang Beer
6: Beerlao
7: French fries
8: Tomato ketchup
9: Table
10: Tissue
11: Ice bucket
12: Plate
13: Budweiser Beer
14: Corona Extra Beer


