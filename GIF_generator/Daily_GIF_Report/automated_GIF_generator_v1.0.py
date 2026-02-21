# IceCube Upgrade GIF file generator
# Version: Automated_GIF_generator_v1.0 (includes email, new dir arrival and new gif files)
#published: 11 Feb 2026
#Author : Shouvik Mondal(smondal@icecube.wisc.edu)
#Based on Seowon Choi's([schoi1@icecube.wisc.edu] / [choi940927@gmail.com]) scripts

import os
import glob
import cv2
import numpy as np
import re
import csv
import tarfile
import shutil
import time
import stat
import smtplib
from email.mime.text import MIMEText
from PIL import Image

# === IMPORT ICUCamera(does most of the image processing) ===
#keep this in the same directory as this script to be safe
try:
    import ICUCamera as icuc
except ImportError:
    icuc = None
    print("[WARNING] ICUCamera.py not found or missing dependencies.")

# ==================== CONFIGURATION ====================
root_dir = "/data/exp/IceCube/2026/internal-system/upgrade-camera"
output_base_dir = "/data/ana/Calibration/upgrade-camera/GIF_reports"
CSV_FILENAME = "comm_modules - Sheet1.csv" #where all the port and strings informations are(got it from Rumman)

# --- NOTIFICATION SETTINGS ---
ENABLE_EMAIL = True
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

# WHO IS SENDING? (Must use App Password if Gmail)
SENDER_EMAIL = "try_with_your_email@gmail.com"
SENDER_PASSWORD =  "abcd efgh ijkl mnop"   #"PUT_YOUR_16_CHAR_APP_PASSWORD_HERE" 

# WHO IS RECEIVING? (List of emails)
RECEIVER_EMAILS = [
    "collegelevelphysics@gmail.com", 
    "shouvik.mondal@utah.edu"
]

# Settings
FRAME_DURATION_MS = 500 #in ms
EXCLUDED_DATES = [""] # if you want to exclude any run folder e.g "20260123" 
FONT_SCALE = 1.2       
FONT_THICKNESS = 2 #must be an integer     
TEXT_COLOR = (0, 0, 255) 
BG_COLOR = (0, 0, 0)     #black background
# =======================================================

def send_summary_email(new_dirs, generated_gifs):
    if not ENABLE_EMAIL: return
    if not new_dirs and not generated_gifs: return # Don't spam if nothing happened

    count_gifs = len(generated_gifs)
    count_dirs = len(new_dirs)
    
    subject = f"[ICUCamera_GIF_generator_v1.0] Processed {count_gifs} GIFs | Found {count_dirs} New Folders"
    
    body = f"ICUcamera_GIF_Report:\n\n"
    
    if new_dirs:
        body += f"=== NEW DATA ARRIVED ({count_dirs}) ===\n"
        for d in new_dirs:
            body += f" [NEW] {d}\n"
        body += "\n"
        
    if generated_gifs:
        body += f"=== GIFS GENERATED/UPDATED ({count_gifs}) ===\n"
        for g in generated_gifs:
            body += f" [OK] {g}\n"
        body += "\n"
    else:
        body += "No GIFs needed updates this run.\n"

    body += f"\nOutput Location: {output_base_dir}"

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = SENDER_EMAIL
    msg['To'] = ", ".join(RECEIVER_EMAILS)

    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        # Send to list of recipients
        server.sendmail(SENDER_EMAIL, RECEIVER_EMAILS, msg.as_string())
        server.quit()
        print(f"  [Notification] Summary email sent to {len(RECEIVER_EMAILS)} recipients.")
    except Exception as e:
        print(f"  [Notification Failed] Could not send email: {e}")

def check_for_new_directories_at_pole(root_path):
    history_file = os.path.join(os.getcwd(), "directory_history.txt")
    try:
        current_dirs = set([d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))])
    except Exception:
        return []

    known_dirs = set()
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            known_dirs = set(f.read().splitlines())

    new_arrivals = list(current_dirs - known_dirs)
    new_arrivals.sort()

    if new_arrivals:
        print(f"  [New Data] Detected {len(new_arrivals)} new folders.")
        with open(history_file, 'w') as f:
            for d in sorted(list(current_dirs)):
                f.write(f"{d}\n")
    
    return new_arrivals

def find_csv_file(start_path, filename): #keep the csv file in the same directory as well
    candidate = os.path.join(os.getcwd(), filename)
    if os.path.exists(candidate): return candidate
    candidate = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    if os.path.exists(candidate): return candidate
    return None

def load_depth_map(filename): #extract information from the csv file like string, port number etc.
    mapping = {}
    csv_path = find_csv_file(os.getcwd(), filename)
    if not csv_path:
        return mapping
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                port = row.get('Port Number', '').strip()
                depth = row.get('Depth', '').strip()
                if not port or not depth: continue
                for s in range(87, 94):
                    s_str = str(s)
                    if row.get(s_str, '').strip():
                        mapping[(s_str, port)] = depth
    except Exception: pass
    return mapping

def asinh_stretch(x, p_black=0.5, p_white=99.9, a=15.0): #we can add more function, take a look at ICUCamera.py
    x = x.astype(np.float32)
    black = np.percentile(x, p_black)
    white = np.percentile(x, p_white)
    x = (x - black) / max(white - black, 1e-6)
    x = np.clip(x, 0, 1)
    y = np.arcsinh(a * x) / np.arcsinh(a)
    return (y * 255).astype(np.uint8)

def clahe(bgr_img, clipLimit=2.0):
    lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(8,8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

def draw_text_with_bg(img, text, x, y, font_scale, thickness, color, bg_color):
    (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv2.rectangle(img, (x - 5, y - text_h - 5), (x + text_w + 5, y + 5), bg_color, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

#This prevents our entire script from crashing just because one temporary file couldn't be deleted.(faced a lot before!)

def on_rm_error(func, path, exc_info):
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception: pass 

def scan_and_group_files(root_path):
    print(f"Scanning {root_path} for archives...")
    grouped_tasks = {} 
    all_tars = glob.glob(os.path.join(root_path, "**", "*.tar.gz"), recursive=True)
    
    # Flexible Regex for _trial0 etc.[before it was showing error for this "trial_o" string--> now its fixed]
    pattern = re.compile(r"^(.*)_(\d{8}.*)\.tar\.gz$")

    for tar_path in all_tars:
        filename = os.path.basename(tar_path)
        match = pattern.match(filename)
        if match:
            prefix = match.group(1) 
            timestamp_str = match.group(2) 
            date_str = timestamp_str[:8] 
            
            if prefix not in grouped_tasks:
                grouped_tasks[prefix] = []
            grouped_tasks[prefix].append({
                "path": tar_path,
                "date": date_str,
                "timestamp": timestamp_str,
                "filename": filename
            })
    return grouped_tasks

def process_gif_task(prefix, file_list, output_root, depth_map):
    s_match = re.search(r"string(\d+)", prefix, re.IGNORECASE)
    r_match = re.search(r"Run_([A-Za-z0-9]+)", prefix, re.IGNORECASE)
    
    string_num = s_match.group(1) if s_match else "Unknown"
    run_type = r_match.group(1) if r_match else "Unknown"

    string_dir = os.path.join(output_root, f"String_{string_num}")
    run_dir = os.path.join(string_dir, f"Run_{run_type}")
    if not os.path.exists(run_dir): os.makedirs(run_dir)
        
    save_path = os.path.join(run_dir, f"{prefix}.gif")
    
    files_by_date = {}
    for f in file_list:
        if f['date'] in EXCLUDED_DATES: continue
        if f['date'] not in files_by_date: files_by_date[f['date']] = []
        files_by_date[f['date']].append(f)
    
    unique_files = []
    for d in sorted(files_by_date.keys()):
        daily = sorted(files_by_date[d], key=lambda x: x['timestamp'])
        unique_files.append(daily[0])
    
    num_new_frames = len(unique_files)
    if num_new_frames < 2: return None

    # === NEW UPDATE & REPORTING ===
    action_taken = None
    if os.path.exists(save_path):
        try:
            with Image.open(save_path) as img:
                existing_frames = getattr(img, 'n_frames', 1)
            
            if num_new_frames <= existing_frames:
                return None # No update needed
            else:
                print(f"  [UPDATING] {prefix}: {existing_frames} -> {num_new_frames} frames")
                action_taken = f"{prefix}.gif (Updated: {existing_frames}->{num_new_frames} frames)"
        except Exception:
            print(f"  [Reprocessing] {prefix} (Corrupt file)")
            action_taken = f"{prefix}.gif (Repaired)"
    else:
         print(f"  [Creating] {prefix} ({num_new_frames} frames)...")
         action_taken = f"{prefix}.gif (Created: {num_new_frames} frames)"
    # ================================

    temp_dir = f"temp_{string_num}_{run_type}_{os.getpid()}" #temporary directory to process the images in the background
    if not os.path.exists(temp_dir): os.makedirs(temp_dir)
    
    frames = []
    try:
        extracted_paths = []
        for item in unique_files:
            try:
                with tarfile.open(item['path'], "r:gz") as tar:
                    raw_member = None
                    for m in tar.getmembers():
                        if m.name.endswith(".raw"):
                            raw_member = m
                            break
                    if raw_member:
                        raw_member.name = os.path.basename(raw_member.name)
                        try:
                            tar.extract(raw_member, path=temp_dir, filter='data')
                        except TypeError:
                            tar.extract(raw_member, path=temp_dir)
                        extracted_paths.append(os.path.join(temp_dir, raw_member.name))
            except Exception: pass

        p_match_inner = re.search(r"port(\d+)", prefix, re.IGNORECASE)
        port_num = p_match_inner.group(1) if p_match_inner else "Unknown"
        depth_val = depth_map.get((string_num, port_num))

        for filepath in extracted_paths:
            if not icuc: break
            try:
                result = icuc.Raw2Npy(filepath)
            except Exception: continue
                
            if result is None: continue
            _, raw_data = result
            
            R = raw_data[0::2, 0::2]
            G1 = raw_data[0::2, 1::2]
            G2 = raw_data[1::2, 0::2]
            B = raw_data[1::2, 1::2]
            
            bayer = asinh_stretch(raw_data, p_black=0.5, a=40.0)
            bgr = cv2.cvtColor(bayer, cv2.COLOR_BAYER_BG2BGR)
            final_bgr = clahe(bgr, clipLimit=3.0)
            
            h, w, _ = final_bgr.shape
            fname = os.path.basename(filepath)
            match = re.search(r"(\d{8}-\d{2}-\d{2}-\d{2})", fname)
            if match:
                ts_str = match.group(1)
                date_text = f"{ts_str[:4]}-{ts_str[4:6]}-{ts_str[6:8]} {ts_str[9:11]}:{ts_str[12:14]}"
                draw_text_with_bg(final_bgr, date_text, 30, 50, FONT_SCALE, FONT_THICKNESS, TEXT_COLOR, BG_COLOR)
            
            if depth_val:
                depth_text = f"Depth: {depth_val}m"
                (tw, th), _ = cv2.getTextSize(depth_text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS)
                draw_text_with_bg(final_bgr, depth_text, w - tw - 30, 50, FONT_SCALE, FONT_THICKNESS, TEXT_COLOR, BG_COLOR)
            
            info_text = f"Str: {string_num} | Port: {port_num}"
            (tw, th), _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS)
            draw_text_with_bg(final_bgr, info_text, w - tw - 30, 100, FONT_SCALE, FONT_THICKNESS, TEXT_COLOR, BG_COLOR)

            final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(final_rgb)
            frames.append(pil_img.convert("P", palette=Image.ADAPTIVE))

        if frames:
            durations = [FRAME_DURATION_MS] * len(frames)
            frames[0].save(save_path, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)

    finally:
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir, onerror=on_rm_error)
            except Exception:
                time.sleep(1.0)
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except: pass

    return action_taken

def main():
    print("=== ICUcamera_GIF_generator_automated(v1.0) ===")
    
    # 1. Check for New Directories (and remember/store them)
    new_dirs = check_for_new_directories_at_pole(root_dir)

    depth_map = load_depth_map(CSV_FILENAME)
    tasks = scan_and_group_files(root_dir)
    
    if not tasks:
        print(f"No files found in {root_dir}")
        return
    
    print(f"Checking {len(tasks)} configurations...")
    
    # 2. Process GIFs and collect a list of what changed
    generated_gifs = []
    count = 0
    total = len(tasks)
    
    for prefix in sorted(tasks.keys()):
        count += 1
        if count % 10 == 0: print(f"  [Scan] {count}/{total} checked...", end="\r")
        
        # The function now returns the filename IF it did something
        result = process_gif_task(prefix, tasks[prefix], output_base_dir, depth_map)
        if result:
            generated_gifs.append(result)

    print(f"\nScan Complete. Updates: {len(generated_gifs)}")

    # 3. Send Summary Email (Only if something happened)
    send_summary_email(new_dirs, generated_gifs)

if __name__ == "__main__":
    main()
