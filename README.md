# เทรนโมเดลจำแนกท่าทางมือ (Hand Gesture Classification)

โครงงานรายวิชา “ระบบปฏิบัติการ Deep Learning (01204466)” — คณะวิศวกรรมศาสตร์ ภาควิชาวิศวกรรมคอมพิวเตอร์ มหาวิทยาลัยเกษตรศาสตร์ ภาคเรียนที่ 1 ปีการศึกษา 2568  
ผู้สอน: ผู้ช่วยศาสตราจารย์ ดร.ภารุจ รัตนวรพันธุ์

**ผู้จัดทำ**  
- นางสาวเปรมมิกา เนียมเปรม — รหัสนิสิต 6610505489  
- นางสาววริษฐา ภิรมย์เกียรติ์ — รหัสนิสิต 6610505551

---

## สารบัญ
1. [หัวข้อ Final Project](#หัวข้อ-final-project)  
   1.1 [ทำไมต้องใช้ Deep Learning](#ทำไมต้องใช้-deep-learning)  
   1.2 [สถาปัตยกรรม Deep Learning](#สถาปัตยกรรม-deep-learning)  
   1.3 [อธิบายโค้ด](#อธิบายโค้ด)  
   1.4 [วิธีการเทรนโมเดล](#วิธีการเทรนโมเดล)  
   1.5 [Dataset](#dataset)  
   1.6 [การประเมินผล & Metrics](#การประเมินผล--metrics)  
   1.7 [รูปภาพประกอบผลลัพธ์](#รูปภาพประกอบผลลัพธ์)  
2. [บทความอ้างอิงและงานที่เกี่ยวข้อง](#บทความอ้างอิงและงานที่เกี่ยวข้อง)  
3. [เปอร์เซ็นงานของแต่ละคน](#เปอร์เซ็นงานของแต่ละคน)  
4. [ภาคผนวก](#ภาคผนวก)

---

## หัวข้อ Final Project
จำแนกท่ามือ 6 คลาส `{dislike, fist, like, ok, palm, peace}` เพื่อนำไปใช้กับระบบสั่งงานไร้สัมผัส/AR/VR และการเข้าถึงสำหรับผู้ใช้พิเศษ ความท้าทายคือท่าที่คล้ายกัน (เช่น ok vs like), ความหลากหลายของแสง/พื้นหลัง และมือที่มีขนาดเล็กในภาพ จึงออกแบบทั้งสถาปัตยกรรม-การฝึก-การประเมินอย่างรอบด้าน

### ทำไมต้องใช้ Deep Learning
- รองรับความแปรผันสูงของภาพจริง (แสง เป้าหมายเล็ก มุมกล้อง พื้นหลัง คน/อุปกรณ์ต่างกัน)  
- CNN เรียนรู้ฟีเจอร์ลำดับชั้น + ใช้ **pretrained** ช่วยลดข้อมูลที่ต้องใช้และทำให้ converge เร็ว  
- ข้อเด่น: ความแม่นยำ/ยืดหยุ่นสูง, transfer learning, end-to-end  
- ข้อด้อย: ต้องใช้ GPU/เวลา, ต้องคุมคุณภาพ/อคติข้อมูล, ต้องดูแล pipeline ให้ดี

### สถาปัตยกรรม Deep Learning
**ชนิดโมเดล:** CNN (Transfer Learning)  
**Backbone:** ResNet18 (pretrained ImageNet) ใช้ `conv1→bn1→relu→maxpool→layer1→layer2→layer3→layer4`  
**Head (ออกแบบเอง):**  
`Conv2d(512→256,k3,s1,p1) → BN → ReLU → SEBlock(r=16) → GAP → Dropout(0.2) → Linear(256→6) → Softmax (infer)`  

- **SEBlock:** GAP → FC(256→16)→ReLU → FC(16→256)→Sigmoid แล้วคูณกับฟีเจอร์เดิมแบบ channel-wise  
- เป้าหมายคือเน้นช่องสัญญาณที่สำคัญต่อท่ามือ และลด overfitting ด้วย GAP+Dropout ก่อนชั้น FC

---

## อธิบายโค้ด
- **Data pipeline:** `build_transforms`, `build_dataloaders`  
  - Train: RandomResizedCrop, HorizontalFlip, ColorJitter, Normalize(ImageNet)  
  - Val/Test: Resize/CenterCrop + Normalize  
  - ใช้ `ImageFolder + DataLoader` (มี `pin_memory`, `persistent_workers`)  
- **Model:** `ConvBNAct`, `SEBlock`, `GestureHead`, `GestureResNet18Custom`, `build_model`, `set_backbone_trainable`  
- **Train loop:** `run_epoch`, `train_model`  
  - Phase-1: freeze backbone → เทรนเฉพาะหัว  
  - Phase-2: unfreeze ทั้งโมเดล  
  - Optimizer: AdamW, Scheduler: ReduceLROnPlateau (ดู `val_loss`)  
  - AMP (mixed precision) + gradient clipping + early stopping (เซฟ `best.pt`)  
  - เก็บ `history.json` และวาดกราฟ loss/acc  
- **Evaluate & Report:** `evaluate_and_report`  
  - Accuracy, Macro P/R/F1, per-class report  
  - Confusion Matrix (counts/normalized), `classification_report_test.(txt|csv)`, `test_summary.txt`  
- **(ตัวเลือก)** Export `model.onnx` สำหรับรันไทม์อื่น ๆ

---

## วิธีการเทรนโมเดล
1. เตรียมโครงสร้างโฟลเดอร์ภาพ (เช่น HaGRID 30k ใน Google Drive) และสร้าง train/val/test  
2. ตั้งค่าหลัก: `IMG_SIZE=224, BATCH=32, EPOCHS=15, LR=3e-4, WEIGHT_DECAY=1e-4, MID_CH=256, DROP_P=0.2, FREEZE_EPOCHS=3, EARLY_PATIENCE=5, CLIP_NORM=1.0, USE_PRETRAINED=True`  
3. สร้าง DataLoader และโมเดล แล้วเรียก `train_model(...)` (freeze→unfreeze)  
4. ประเมินด้วย `evaluate_and_report(...)` → ได้รายงาน/กราฟ/CM  
5. (ทางเลือก) Export `model.onnx`

---

## Dataset
- แหล่งข้อมูล: **HaGRID Sample 30k 384p (Kaggle)**  
- ใช้เฉพาะ 6 คลาส: like, dislike, fist, ok, palm, peace  
- วิธีสปลิต:  
  - วิธีที่ 1: ตั้ง `TEST=0.20` แล้วแบ่งครึ่งของ test ไปเป็น val → โดยรวมได้ประมาณ train≈80% / val≈10% / test≈10%  
  - วิธีที่ 2: สุ่ม 80/10/10 และคัดลอกไฟล์จริงเก็บถาวร  
- ออกแบบโฟลเดอร์ให้ใช้งานกับ `ImageFolder` ได้ทันที (train/val/test)

---

## การประเมินผล & Metrics
- **Loss:** Cross-Entropy (รายงาน train/val loss เป็นกราฟ)  
- **Metrics:**  
  - Accuracy (train/val/test)  
  - Macro Precision/Recall/F1 (รวมทั้ง per-class)  
- **Visualization:**  
  - Confusion Matrix (counts/normalized)  
  - กราฟ loss/acc  
  - ตาราง per-class metrics  
  - (เสริม) PR Curve / PR AUC (macro AP)  
- **Artifacts ใน `/content/run_outputs`:**  
  `loss_curve.png`, `acc_curve.png`, `classification_report_test.txt/csv`, `cm_test_counts.png`, `cm_test_norm.png`, `test_summary.txt`, `best.pt`, `classes.json`, `history.json`, `train_summary.txt`

---

## รูปภาพประกอบผลลัพธ์
- รูปสถาปัตยกรรม (CNN + Custom SE Head)  
- กราฟ training/validation loss & accuracy  
- Confusion matrix (counts และ normalized)

---

## บทความอ้างอิงและงานที่เกี่ยวข้อง
- ResNet-18 (backbone มาตรฐานใน `torchvision.models`)  
- Squeeze-and-Excitation (SE) Block  
- ONNX Export (สำหรับ inference ข้ามเฟรมเวิร์ก/รันไทม์)  
- Dataset: HaGRID Sample 30k 384p (Kaggle)

---

## เปอร์เซ็นงานของแต่ละคน
**ฝ่าย Data — 50% (ตามที่ตกลง)**  
- สปลิตข้อมูล/จัดโครงสร้าง ImageFolder — 15%  
- Data hygiene & balancing เบื้องต้น — 8%  
- Data pipeline (transforms, dataloaders, workers) — 10%  
- Augmentation (RandomResizedCrop/Flip/ColorJitter) — 7%  
- วิเคราะห์ข้อมูล (class distribution, ตัวอย่างภาพ) — 5%  
- เอกสารฝั่ง Data — 5%

**ฝ่าย Model — 50% (ตามที่ตกลง)**  
- ออกแบบหัวคัสตอม (Conv3×3→BN→ReLU→SE→GAP→Dropout→FC) — 15%  
- ประกอบกับ ResNet18 (`build_model`, `set_backbone_trainable`) — 8%  
- เทรน (freeze→unfreeze, AdamW, ReduceLROnPlateau, AMP, clip, early stop) — 12%  
- ประเมินผล (Accuracy, Macro-P/R/F1, CM, per-class) — 8%  
- วิชวลไลซ์ (loss/acc, PR/mAP, misclassified, ECE/Grad-CAM) — 5%  
- เอกสารฝั่ง Model — 2%

---

## ภาคผนวก
### คำสั่งรัน (Colab/Notebook)
1) เมานต์ไดรฟ์/ติดตั้งไลบรารี แล้วตั้ง `DATA_ROOT` ให้ถูกต้อง  
2) รันส่วนสร้างสปลิตจนได้ `train/val/test`  
3) รันตามลำดับ: Config → Data Pipeline → Model → Train Loop → Evaluation → (Export ONNX)

### ไฟล์ที่ต้องส่ง
- `final_report.pdf` (ไฟล์รายงาน)  
- ลิงก์ GitHub repository (โค้ด/โน้ตบุ๊ก + อาร์ติแฟกต์ที่สำคัญ)
