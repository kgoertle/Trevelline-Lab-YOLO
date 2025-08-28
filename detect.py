import argparse, time, csv, platform, sys
from pathlib import Path
from datetime import datetime
import numpy as np
import cv2
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent

# ------------ CLASS COLORS --------------
CLASS_NAMES = ["M", "F", "Feeder", "Main_Perch", "Sky_Perch", "Wooden_Perch", "Nesting_Box"]
CLASS_COLORS = {
    "M": (255, 200, 180), "F": (255, 192, 203), "Feeder": (144, 238, 144),
    "Main_Perch": (50, 50, 50), "Sky_Perch": (200, 200, 200),
    "Wooden_Perch": (60, 105, 165), "Nesting_Box": (69, 98, 99)
}
TARGETS = ["Feeder", "Main_Perch", "Sky_Perch", "Nesting_Box", "Wooden_Perch"]

# ---------------- BOX SMOOTHING -----------------
class BoxSmoother:
    def __init__(self, max_history=0, alpha=1.0, dist_thresh=None):
        self.max_history, self.alpha, self.dist_thresh = max_history, alpha, dist_thresh
        self.history = []

    def smooth(self, boxes):
        new_history, smoothed = [], []
        for box in boxes:
            x1,y1,x2,y2,cls = box
            matched = next(((hx1,hy1,hx2,hy2,hcls) for hx1,hy1,hx2,hy2,hcls in self.history
                            if cls==hcls and (self.dist_thresh is None or 
                            np.linalg.norm(np.array([(x1+x2)/2,(y1+y2)/2]) -
                                           np.array([(hx1+hx2)/2,(hy1+hy2)/2])) < self.dist_thresh)), None)
            if matched:
                hx1,hy1,hx2,hy2,_ = matched
                x1 = int(self.alpha*x1 + (1-self.alpha)*hx1)
                y1 = int(self.alpha*y1 + (1-self.alpha)*hy1)
                x2 = int(self.alpha*x2 + (1-self.alpha)*hx2)
                y2 = int(self.alpha*y2 + (1-self.alpha)*hy2)
            smoothed.append([x1,y1,x2,y2,cls])
            new_history.append([x1,y1,x2,y2,cls])
        self.history = new_history[-self.max_history:]
        return smoothed

# ---------------- CSV UPDATE -----------------
def update_csv_file(csv_dir, interactions, start, end):
    file = csv_dir / f"{start.strftime('%d-%m_%H-%M-%S')}_to_{end.strftime('%d-%m_%H-%M-%S')}.csv"
    with open(file,'w',newline='') as f:
        w = csv.writer(f)
        w.writerow(["Bird","Object","Start Time","End Time","Duration (s)","Frames"])
        for (bname,oname), data in interactions.items():
            w.writerow([
                bname, oname,
                datetime.fromtimestamp(data["start_time"]).strftime("%H:%M:%S"),
                datetime.fromtimestamp(data.get("end_time", time.time())).strftime("%H:%M:%S"),
                f"{data.get('duration',0):.2f}", data.get("frames",0)
            ])
    return file

# ---------------- LOAD TFLITE -----------------
def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    return interpreter, interpreter.get_input_details(), interpreter.get_output_details()

# ---------------- PREPROCESS -----------------
def preprocess(frame, input_details):
    h, w = input_details[0]['shape'][1:3]
    # Resize preserving aspect ratio with padding
    fh, fw = frame.shape[:2]
    scale = min(w/fw, h/fh)
    nw, nh = int(fw*scale), int(fh*scale)
    resized = cv2.resize(frame, (nw, nh))
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[(h-nh)//2:(h-nh)//2+nh, (w-nw)//2:(w-nw)//2+nw] = resized
    dtype = input_details[0]['dtype']

    if dtype in [np.uint8, np.int8]:
        scale_val, zero_point = input_details[0].get('quantization', (1.0, 0))
        if scale_val is None: scale_val = 1.0
        if zero_point is None: zero_point = 0
        img = canvas.astype(np.float32) / 255.0
        img = (img / scale_val + zero_point).round().astype(dtype)
        return np.expand_dims(img, axis=0)
    else:
        img = canvas.astype(np.float32) / 255.0
        return np.expand_dims(img, axis=0)

# ---------------- RUN DETECTION -----------------
def run_detection(interpreter, input_details, output_details, source, output_dir, smoother, test_detect=False):
    IS_LINUX = platform.system() == "Linux"

    if str(source).isdigit():
        source = int(source)
        cap = cv2.VideoCapture(source)
    elif isinstance(source, str) and "picamera" in source.lower():
        if not IS_LINUX: raise RuntimeError("[ERROR] Pi Camera only supported on Linux.")
        from picamera2 import Picamera2
        cap = Picamera2()
        cap.configure(cap.create_video_configuration(main={"format": 'RGB888', "size": (640,480)}))
        cap.start()
    else:
        cap = cv2.VideoCapture(str(source))

    if not cap or (not IS_LINUX and not cap.isOpened()):
        print(f"[ERROR] Could not open source {source}!")
        return

    out_file = output_dir / f"{datetime.now().strftime('%m-%d-%Y_%H-%M-%S')}.avi"
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out_writer = cv2.VideoWriter(str(out_file), fourcc, 30, (1280,720))

    interaction_dir = output_dir / "interaction-metrics"
    interaction_dir.mkdir(parents=True, exist_ok=True)

    interactions, threshold = {}, (5 if test_detect else 60)
    save_interval, next_save_time = (60 if test_detect else 3600), time.time() + (60 if test_detect else 3600)
    session_start = datetime.now()
    prev_time, fps_smooth, frame_count, interaction_counter = time.time(), 0, 0, 0

    while True:
        ret, frame = (cap.read() if not isinstance(source,int) else (True, cap.capture_array()))
        if not ret or frame is None: break

        img = preprocess(frame, input_details)
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()

        boxes   = interpreter.get_tensor(output_details[0]['index'])[0]
        classes = interpreter.get_tensor(output_details[1]['index'])[0].astype(int)
        scores  = interpreter.get_tensor(output_details[2]['index'])[0]

        combined = [[int(x1),int(y1),int(x2),int(y2),cls,float(conf)] 
                    for (x1,y1,x2,y2),cls,conf in zip(boxes,classes,scores)]
        smoothed_boxes = smoother.smooth([[x1,y1,x2,y2,cls] for x1,y1,x2,y2,cls,_ in combined])

        # Draw boxes
        for i, (x1,y1,x2,y2,cls) in enumerate(smoothed_boxes):
            cname, conf = CLASS_NAMES[cls], combined[i][5]
            color = CLASS_COLORS.get(cname,(0,255,0))
            label=f"{cname} {conf:.2f}"
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            (w,h),_ = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.5,1)
            cv2.rectangle(frame,(x1,y1-h-4),(x1+w,y1),color,-1)
            cv2.putText(frame,label,(x1,y1-2),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)

        # Interaction detection
        now = time.time()
        active_keys = set()
        M_or_F = [b for b in smoothed_boxes if CLASS_NAMES[b[4]] in ["M","F"]]
        for bx1,by1,bx2,by2,bcls in M_or_F:
            bname = CLASS_NAMES[bcls]
            for ox1,oy1,ox2,oy2,ocls in smoothed_boxes:
                oname = CLASS_NAMES[ocls]
                if bname==oname or oname not in TARGETS: continue
                ix1,iy1,ix2,iy2 = max(bx1,ox1), max(by1,oy1), min(bx2,ox2), min(by2,oy2)
                key=(bname,oname)
                if ix2>ix1 and iy2>iy1:
                    active_keys.add(key)
                    if key not in interactions:
                        interactions[key]={"start_time":now,"frames":1,"active":False,"duration":0}
                    else:
                        interactions[key]["frames"]+=1
                        interactions[key]["duration"]=now-interactions[key]["start_time"]
                    if interactions[key]["frames"]==threshold:
                        interactions[key]["active"]=True
                        interaction_counter+=1

        # Deactivate keys not currently interacting
        for key in list(interactions.keys()):
            if key not in active_keys and interactions[key]["active"]:
                interactions[key]["active"]=False
            if key not in active_keys:
                interactions.pop(key)

        if now>=next_save_time:
            update_csv_file(interaction_dir, interactions, session_start, datetime.now())
            next_save_time = now+save_interval

        fps_smooth = 0.9*fps_smooth + 0.1*(1/(time.time()-prev_time)); prev_time=time.time()
        frame_count+=1
        print(f"\rFrames: {frame_count} | Interactions: {interaction_counter} | FPS: {fps_smooth:.1f}", end="")
        out_writer.write(frame)

    # Cleanup
    if isinstance(source,int) or str(source).isdigit(): cap.release()
    elif "picamera" in str(source).lower(): cap.stop()
    out_writer.release()
    final_csv = update_csv_file(interaction_dir, interactions, session_start, datetime.now())
    print(f"\n[SAVE] Final interaction metrics saved to: {final_csv}")
    print(f"[SAVE] Detection results saved to: {out_file}")

# ---------------- MAIN -----------------
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="TFLite detection for Raspberry Pi 3 ARM64 systems.")
    parser.add_argument("--source", required=True, help="Video source (file, camera index, or 'picamera').")
    parser.add_argument("--smooth", type=float, default=1.0)
    parser.add_argument("--dist-thresh", type=float, default=None)
    parser.add_argument("--max-history", type=int, default=0)
    parser.add_argument("--test-detect", action="store_true")
    args = parser.parse_args()

    # Auto-select TFLite model in priority order
    candidates = [
        BASE_DIR / "best_int8.tflite",
        BASE_DIR / "best_float32.tflite",
        BASE_DIR / "best_float16.tflite"
    ]
    tflite_model = next((m for m in candidates if m.exists()), None)

    if not tflite_model:
        print("[ERROR] No suitable TFLite model found (expected one of: best_int8.tflite, best_float32.tflite, best_float16.tflite).")
        sys.exit(1)

    print(f"[INFO] Using TFLite model: {tflite_model.name}")

    interpreter, input_details, output_details = load_tflite_model(tflite_model)
    smoother = BoxSmoother(max_history=args.max_history, alpha=args.smooth, dist_thresh=args.dist_thresh)

    output_base = BASE_DIR / ("logs/test-runs" if args.test_detect else "logs/runs")
    output_base.mkdir(parents=True, exist_ok=True)

    run_detection(interpreter, input_details, output_details, args.source, output_base, smoother, args.test_detect)
