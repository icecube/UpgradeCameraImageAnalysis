# ==============================================================================
# IceCube Upgrade Image Analyzer(VISUAL PURPOSE ONLY)
# Version: v1.0
# Written by: Shouvik Mondal(smondal@icecube.wisc.edu) based on Seowon Choi's[schoi1@icecube.wisc.edu] scripts

# ==============================================================================


#how to install the libraries : pip install opencv-python numpy Pillow matplotlib tkinterdnd2
#Built-in libraries in python: tkinter (The GUI framework), os, tarfile, gzip, io (File handling)
#for linux users if you get error installing "tkinter" library try: sudo apt-get install python3-tk
#Python 3.9 or 3.10 is recommended to smoothly run this script

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinterdnd2 import DND_FILES, TkinterDnD
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import tarfile
import gzip
import io

# Plotting Libraries
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- parameters for ICUcameras (from ICUCamera.py) ---
RAW_W = 1312
RAW_H = [979, 993]
#EXPECTED_PIXELS = RAW_W * RAW_H
#EXPECTED_BYTES = EXPECTED_PIXELS * 2
EXPECTED_BYTES = [h * RAW_W * 2 for h in RAW_H]

class IceCube_Upgrade_Image_Analyzer(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.title("IceCube Upgrade Image Analyzer")
        self.geometry("1650x950")
        self.configure(bg="#202020")

        # --- Scientific Data States ---
        self.raw_16bit = None       # The pure sensor data (gray)
        self.display_rgb = None     # The visual image
        #self.vignette_mask = None   # For lens correction
        self.base_rgb = None        # Base converted RGB
        
        # --- Zoom & Pan Mode ---
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        
        # --- Parameters ---
        self.params = {
            'gamma': tk.DoubleVar(value=1.4),       
            'clahe': tk.DoubleVar(value=2.0),       
            'brightness': tk.DoubleVar(value=0),
            'contrast': tk.DoubleVar(value=1.0),
            'saturation': tk.DoubleVar(value=1.0),
            'sharpness': tk.DoubleVar(value=0.0),
            #'vignette': tk.BooleanVar(value=False), 
            'threshold': tk.IntVar(value=0)
        }
        self.view_mode = tk.StringVar(value="RGB")

        self._init_ui()
        #self._generate_vignette_mask() 

    def _init_ui(self):
        # 1. LEFT: Controls & Info
        panel_l = tk.Frame(self, width=340, bg="#2b2b2b", padx=3, pady=3)
        panel_l.pack(side=tk.LEFT, fill=tk.Y)
        panel_l.pack_propagate(False)
        
        tk.Label(panel_l, text="Parameters to change", fg="#00e676", bg="#2b2b2b", font=("Arial", 14, "bold")).pack(pady=5)
        
        # Drag & Drop Zone
        self.drop_box = tk.Label(panel_l, text="[ drag.TAR.GZ/.Raw files here ]", bg="#444", fg="#fff", height=3, relief="sunken")
        self.drop_box.pack(fill=tk.X, pady=3)
        self.drop_box.drop_target_register(DND_FILES)
        self.drop_box.dnd_bind('<<Drop>>', lambda e: self.load_content(e.data.strip('{}')))
        self.drop_box.bind("<Button-1>", lambda e: self.open_file())

        # Save Button
        btn_save = tk.Button(panel_l, text="💾 Save Current Image", command=self.save_image, 
                             bg="#2196F3", fg="white", font=("Arial", 10, "bold"), pady=3)
        btn_save.pack(fill=tk.X, pady=(0, 3))

        # Metadata Display
        self.meta_txt = tk.Text(panel_l, height=8, bg="#1a1a1a", fg="#00ff00", font=("Consolas", 8))
        self.meta_txt.pack(fill=tk.X, pady=3)
        self.meta_txt.insert(tk.END, "System Ready.\nUse Mouse Wheel to Zoom.\nRight-Click to Reset.")

        # --- Channel Selector ---
        lbl_ch = tk.Label(panel_l, text="View Channel:", fg="#ccc", bg="#2b2b2b", anchor="w")
        lbl_ch.pack(fill=tk.X, pady=(3, 0))
        
        frm_ch = tk.Frame(panel_l, bg="#2b2b2b")
        frm_ch.pack(fill=tk.X, pady=2)
        modes = ["RGB", "Red", "Green", "Green 1", "Green 2", "Blue", "Gray", "Heatmap"]
        
        for i, mode in enumerate(modes):
            rb = tk.Radiobutton(frm_ch, text=mode, variable=self.view_mode, value=mode, 
                                command=self.process_pipeline, bg="#2b2b2b", fg="white", selectcolor="#444", 
                                indicatoron=0, width=8)
            
            rb.grid(row=i//3, column=i%3, padx=1, pady=1, sticky="ew")

        # --- Sliders (With Numerical Values) ---
        tk.Label(panel_l, text="Image Adjustments:", fg="#ccc", bg="#2b2b2b", anchor="w").pack(fill=tk.X, pady=(5, 0))
        
        self._add_control(panel_l, "Gamma", 'gamma', 0.1, 4.0)
        self._add_control(panel_l, "CLAHE", 'clahe', 0.0, 10.0)
        self._add_control(panel_l, "Contrast", 'contrast', 0.5, 3.0)
        self._add_control(panel_l, "Brightness", 'brightness', -100, 100)
        self._add_control(panel_l, "Saturation", 'saturation', 0.0, 3.0)
        self._add_control(panel_l, "Sharpness", 'sharpness', 0.0, 10.0)
        self._add_control(panel_l, "Threshold", 'threshold', 0, 255)
        
        # Vignette Toggle
        #tk.Checkbutton(panel_l, text="Apply Vignette Correction", variable=self.params['vignette'], 
                       #bg="#2b2b2b", fg="#fff", selectcolor="#444", command=self.process_pipeline).pack(pady=3)

        # 2. CENTER: Image Viewer with zooming option
        self.panel_c = tk.Frame(self, bg="#111")
        self.panel_c.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Replaced standard Label with Canvas for smooth interaction
        self.canvas_img = tk.Canvas(self.panel_c, bg="#111", highlightthickness=0)
        self.canvas_img.pack(fill=tk.BOTH, expand=True)

        # Event Bindings
        self.canvas_img.bind("<MouseWheel>", self.zoom_image)      # Windows/Mac Zoom
        self.canvas_img.bind("<Button-4>", self.zoom_image)        # Linux Zoom Up
        self.canvas_img.bind("<Button-5>", self.zoom_image)        # Linux Zoom Down
        self.canvas_img.bind("<ButtonPress-1>", self.start_pan)    # Start Pan
        self.canvas_img.bind("<B1-Motion>", self.pan_image)        # Panning
        self.canvas_img.bind("<Button-3>", self.reset_zoom)        # Reset (Right Click)
        self.canvas_img.bind("<Motion>", self.inspect_pixel)       # Inspector

        # 3. RIGHT: Scientific Dashboard
        panel_r = tk.Frame(self, width=450, bg="#2b2b2b")
        panel_r.pack(side=tk.RIGHT, fill=tk.Y)
        panel_r.pack_propagate(False)
        
        tk.Label(panel_r, text="Real-time Analysis", fg="#29b6f6", bg="#2b2b2b", font=("Arial", 12, "bold")).pack(pady=3)
        
        # Matplotlib Canvas
        self.fig = Figure(figsize=(4, 5), dpi=100, facecolor="#2b2b2b")
        self.ax_hist = self.fig.add_subplot(211)
        self.ax_fft = self.fig.add_subplot(212)
        self.canvas = FigureCanvasTkAgg(self.fig, master=panel_r)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=3)

        # Stats Overlay
        self.stats_lbl = tk.Label(panel_r, text="Stats: N/A", bg="#2b2b2b", fg="#ffcc00", justify=tk.LEFT, font=("Consolas", 10))
        self.stats_lbl.pack(fill=tk.X, pady=3, padx=3)

    # --- UPDATED CONTROLS ---
    def _add_control(self, parent, label, var_key, min_v, max_v):
        f = tk.Frame(parent, bg="#2b2b2b")
        f.pack(fill=tk.X, pady=2)
        
        tk.Label(f, text=label, fg="#ccc", bg="#2b2b2b", width=12, anchor="w").pack(side=tk.LEFT)
        
        # Numerical Value Display
        current_val = self.params[var_key].get()
        val_text = f"{int(current_val)}" if var_key == 'threshold' else f"{current_val:.2f}"
        
        val_lbl = tk.Label(f, text=val_text, fg="#00e676", bg="#2b2b2b", width=5, font=("Consolas", 9))
        val_lbl.pack(side=tk.RIGHT)
        
        def on_slide(val):
            v_float = float(val)
            if var_key == 'threshold':
                val_lbl.config(text=f"{int(v_float)}")
            else:
                val_lbl.config(text=f"{v_float:.2f}")
            self.process_pipeline()

        ttk.Scale(f, from_=min_v, to=max_v, variable=self.params[var_key], 
                  command=on_slide).pack(side=tk.LEFT, fill=tk.X, expand=True)

    #def _generate_vignette_mask(self):
        #Y, X = np.ogrid[:RAW_H, :RAW_W]
        #center_y, center_x = RAW_H / 2, RAW_W / 2
        #dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        #max_dist = np.sqrt(center_x**2 + center_y**2)
        #self.vignette_mask = 1 + (0.5 * (dist_from_center / max_dist)**2)

    # --- ZOOM & PAN LOGIC ---
    def start_pan(self, event):
        self.last_mouse_x = event.x
        self.last_mouse_y = event.y

    def pan_image(self, event):
        if self.display_rgb is None: return
        dx = event.x - self.last_mouse_x
        dy = event.y - self.last_mouse_y
        self.pan_x += dx
        self.pan_y += dy
        self.last_mouse_x = event.x
        self.last_mouse_y = event.y
        self.update_view()

    def zoom_image(self, event):
        if self.display_rgb is None: return
        
        # Zoom Factor
        if event.num == 5 or event.delta < 0:
            factor = 0.9  # Out
        else:
            factor = 1.1  # In

        new_zoom = self.zoom_level * factor
        # Limit zoom range
        if 0.1 < new_zoom < 20.0:
            self.zoom_level = new_zoom
            self.update_view()

    def reset_zoom(self, event):
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.update_view()

    # --- VIEW PANEL ---
    def update_view(self):
        if self.display_rgb is None: return
        
        h, w = self.display_rgb.shape[:2]
        
        # Base scale to fit window
        win_w = self.canvas_img.winfo_width()
        win_h = self.canvas_img.winfo_height()
        if win_w < 10: win_w = 800
        
        base_scale = min(win_w/w, win_h/h)
        
        # Apply Zoom
        final_scale = base_scale * self.zoom_level
        new_w = int(w * final_scale)
        new_h = int(h * final_scale)
        
        # Resize
        resized = cv2.resize(self.display_rgb, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        # Convert to TkImage
        pil_img = Image.fromarray(resized)
        self.tk_img = ImageTk.PhotoImage(pil_img) # Keep Reference
        
        # Draw on Canvas (Center + Pan Offset)
        center_x = win_w // 2 + self.pan_x
        center_y = win_h // 2 + self.pan_y
        
        self.canvas_img.delete("all")
        self.canvas_img.create_image(center_x, center_y, anchor="center", image=self.tk_img)

        # Store params for reverse-mapping
        self.current_draw_params = {
            'scale': final_scale,
            'tx': center_x - (new_w // 2),
            'ty': center_y - (new_h // 2)
        }

    # --- PIXEL VIEWER ---
    def inspect_pixel(self, event):
        if self.raw_16bit is None or not hasattr(self, 'current_draw_params'): return
        try:
            p = self.current_draw_params
            
            # 1. Mouse relative to image top-left corner on screen
            rel_x = event.x - p['tx']
            rel_y = event.y - p['ty']
            
            # 2. Scale back to original image coordinates
            x = int(rel_x / p['scale'])
            y = int(rel_y / p['scale'])
            
            h, w = self.raw_16bit.shape
            if 0 <= x < w and 0 <= y < h:
                val = self.raw_16bit[y, x]
                # Preserve existing stats text, update VIEWING part
                base_text = self.stats_lbl.cget("text").split("\n\n")[0]
                self.stats_lbl.config(text=f"{base_text}\n\n[ Pixel Inspection]\nX:{x} Y:{y}\nRaw Value: {val}")
        except: pass

    # --- IO & PROCESSING ---
    def open_file(self):
        f = filedialog.askopenfilename(filetypes=[("IceCube Data", "*.tar.gz *.tar *.RAW *.raw"), ("Images", "*.png *.jpg")])
        if f: self.load_content(f)

    def save_image(self):
        if self.display_rgb is None: 
            messagebox.showwarning("Warning", "No processed image to save.")
            return
        
        f = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg"), ("BMP Image", "*.bmp"), ("TIFF Image", "*.tiff")]
        )
        if f:
            try:
                save_img = cv2.cvtColor(self.display_rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f, save_img)
                messagebox.showinfo("Success", f"Image saved to:\n{f}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not save file:\n{e}")

    def asinh_stretch(self, img, p_black=0.5, p_white=99.9, a=15.0):
        """
        Updated from ICUCamera.py: compresses dark noise and stretches bright signal.
        """
        # Working in float to avoid clipping errors
        img = img.astype(np.float32)
        
        # Find the Noise Floor (Black Point)
        # p_black=0.5 means we consider the bottom 0.5% of pixels as "background noise"
        black = np.percentile(img, p_black)
        white = np.percentile(img, p_white)
        
        # SUBTRACT the noise floor (This kills the gray edge effect!)
        img = (img - black) / max(white - black, 1e-6)
        
        # Clip negative values (anything below the floor becomes 0)
        img = np.clip(img, 0, 1)
        
        # Apply the Asinh Curve (Dynamic Range Compression)
        img = np.arcsinh(a * img) / np.arcsinh(a)
        
        # Convert back to 8-bit for display
        return (img * 255).astype(np.uint8)

    def load_content(self, path):
        self.meta_txt.delete(1.0, tk.END)
        self.meta_txt.insert(tk.END, f"File: {os.path.basename(path)}\n")
        
        try:
            raw_data = None
            
            # 1. Archive Extraction
            if path.endswith((".tar.gz", ".tar", ".tgz")):
                with tarfile.open(path, "r:*") as tar:
                    target = next((m for m in tar.getmembers() if m.name.lower().endswith(".raw")), None)
                    # If no .raw extension, check if file size matches ANY valid candidate size
                    if not target:
                        for m in tar.getmembers():
                         # We allow a small 5000 byte buffer for headers/footers
                            if any(abs(m.size - size) < 5000 for size in EXPECTED_BYTES):
                                target = m
                                break
                    
                    if target:
                        self.meta_txt.insert(tk.END, f"Found: {target.name}\nSize: {target.size} bytes\n")
                        raw_data = tar.extractfile(target).read()
                    else:
                        raise ValueError("No .RAW file found in archive.")
            
            # 2. Direct File Reading
            elif path.lower().endswith((".raw", ".gz",".RAW",".tar.gz")):
                with open(path, "rb") as f:
                    raw_data = f.read()

            # 3. Decompression & Header Handling
            if raw_data:
                if len(raw_data) > 12 and raw_data[12:14] == b'\x1f\x8b':
                    self.meta_txt.insert(tk.END, "Info: Aggregate Format (Header+Gzip)\n")
                    import gzip
                    raw_data = gzip.decompress(raw_data[12:])
                elif raw_data[:2] == b'\x1f\x8b':
                     import gzip
                     raw_data = gzip.decompress(raw_data)

                # --- BUFFER CORRECTION ---
                if len(raw_data) % 2 != 0:
                    self.meta_txt.insert(tk.END, f"Debug: Trimmed odd byte.\n")
                    raw_data = raw_data[:-1]

                #Read data as 1D array
                arr = np.frombuffer(raw_data, dtype=np.uint16)
                
                #Iterative Shape Matching (Logic from ICUCamera.py)
                matched_shape = None
                
                for h in RAW_H:
                    expected_pixels = RAW_W * h
                    
                    #Exact match
                    if arr.size == expected_pixels:
                        matched_shape = (h, RAW_W)
                        break
                    
                    # Match with small header offset,
                    # If data is slightly larger (up to 2048 pixels extra), we assume it's a header and take the end.
                    if arr.size > expected_pixels and arr.size < expected_pixels + 2048:
                        self.meta_txt.insert(tk.END, f"Info: Trimmed {arr.size - expected_pixels} header pixels.\n")
                        arr = arr[-expected_pixels:] # Keeping only the image data at the end of processing
                        matched_shape = (h, RAW_W)
                        break

                if matched_shape is None:
                    raise ValueError(f"Could not fit data (Pixels: {arr.size}) into widths {RAW_W} x heights {RAW_H}")

                # 3. Apply the detected shape
                current_h, current_w = matched_shape
                self.meta_txt.insert(tk.END, f"Geometry: {current_w}x{current_h}\n")
                
                arr = arr >> 4 
                self.raw_16bit = arr.reshape(matched_shape)
                
                #bayer_img = (self.raw_16bit/16).astype(np.uint8) ## We divide by 16 to fit 0-4095 into 0-255 range.
                bayer_img = self.asinh_stretch(self.raw_16bit, p_black=0.5)
                self.base_rgb = cv2.cvtColor(bayer_img, cv2.COLOR_BAYER_BG2RGB)
                
                self.process_pipeline()
                
                # RESET VIEW ON LOAD
                self.zoom_level = 1.0
                self.pan_x = 0
                self.pan_y = 0
                self.update_view()
                self.meta_txt.insert(tk.END, "Success: Loaded.\n")

            else:
                self.base_rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
                self.raw_16bit = cv2.cvtColor(self.base_rgb, cv2.COLOR_RGB2GRAY).astype(np.uint16)
                self.process_pipeline()
                self.zoom_level = 1.0
                self.pan_x = 0
                self.pan_y = 0
                self.update_view()

        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load data:\n{str(e)}")

    def get_gray(self, npy, pedestal=235.0, sat_thr=4090, dead_value=None):
            # This script corrects the saturated blue pixels using the other channels.
    # crop to even shape
        """
        get_gray: Recovers saturated pixels using channel correlation.
        Returns a single 2D float array (channel B reconstruction).
        """
        # Crop to even shape to ensure 2x2 bayer blocks fit
        H, W = npy.shape
        npy = npy[:H - (H % 2), :W - (W % 2)]   # To make sure even shape between 4 channels

        # Extract Raw Bayer Channels
        B  = npy[1::2, 1::2].astype(np.float32)
        G1 = npy[0::2, 1::2].astype(np.float32)
        G2 = npy[1::2, 0::2].astype(np.float32)
        R  = npy[0::2, 0::2].astype(np.float32)

        # Identify Saturated Pixels
        B_sat  = B  >= sat_thr
        G_bad  = (G1 >= sat_thr) | (G2 >= sat_thr)   # if any of G1/G2 is sat, G is bad
        R_sat  = R  >= sat_thr

        # Prepare Reference Channels
        G_use = 0.5 * (G1 + G2) # Average Green
        P = float(pedestal)

        # Weights (Scaling factors to normalize R/G to match B)
        # Note: These weights might need tuning for your specific flasher/LED
        weights = {"G": 10.0, "R": 25.0} 
        
        B_from_R = weights["R"] * (R - P)
        B_from_G = weights["G"] * (G_use - P)

        B_new = B.copy()
        
        # case 1: B sat, but G & R usable -> mix
        m = B_sat & (~G_bad) & (~R_sat)
        B_new[m] = 0.5 * (B_from_G[m] + B_from_R[m])

        #case 2: B & G bad, R usable -> R only
        m = B_sat & (G_bad) & (~R_sat)
        B_new[m] = B_from_R[m]

        # case 3: B sat, G usable, R sat -> G only  
        # Don't expect this case to be happen.
        # If this case happens, check the channel division first, and then illuminate conditions.
        m = B_sat & (~G_bad) & (R_sat)
        B_new[m] = B_from_G[m]

        # Case 4: Everything is saturated -> Mark as dead/max;B sat, G bad, R sat -> unrecoverable marker
        m = B_sat & (G_bad) & (R_sat)
        if dead_value is None:
            B_new[m] = weights["R"] * (4094.0 - P) # Estimate max possible
        else:
            B_new[m] = dead_value

        return B_new

    # --- PIPELINE ---
    def process_pipeline(self):
        if self.base_rgb is None: return
        #img = self.base_rgb.astype(np.float32)
        mode = self.view_mode.get()

        # 1. --- Standard View(preocessed RGB) vs Raw Channel(R,G1,G2,B) View ---
        if mode in ["Green 1", "Green 2", "Gray"] and self.raw_16bit is not None:
            raw_sub = None
            # EXTRACT RAW SUB-CHANNELS (RGGB Pattern)
            # Green 1 is at Row 0, Col 1 (and every alternate)
            # Green 2 is at Row 1, Col 0 (and every alternate)
            
            if mode == "Green 1":
                raw_sub = self.raw_16bit[0::2, 1::2] 
            elif mode == "Green 2": # Green 2
                raw_sub = self.raw_16bit[1::2, 0::2]
            elif mode == "Gray":
                raw_sub = self.get_gray(self.raw_16bit)
                
            # Stretch the 12-bit data to 8-bit for display
            sub_8bit = self.asinh_stretch(raw_sub, p_black=0.5)
            
            # RESIZE back to full resolution so it fits the canvas(GUI)
            # INTER_NEAREST keeps the "pixelated" raw look
            full_h, full_w = self.raw_16bit.shape
            sub_resized = cv2.resize(sub_8bit, (full_w, full_h), interpolation=cv2.INTER_NEAREST)
            
            # Create the display image (3 channels)
            if mode == "Gray":
                # For Gray, R=G=B
                img = cv2.merge([sub_resized, sub_resized, sub_resized])
            else:
                # For Green channels, make it look Green
                zeros = np.zeros_like(sub_resized)
                img = cv2.merge([zeros, sub_resized, zeros]) # R=0, G=Val, B=0
            
            img = img.astype(np.float32)
        else:
            # Standard Processing (RGB, Red, Blue, etc.)
            img = self.base_rgb.astype(np.float32)



        # 2. Gamma
        gamma = self.params['gamma'].get()
        if gamma != 1.0:
            img = 255 * (img / 255.0) ** (1 / gamma)

        img = np.clip(img, 0, 255).astype(np.uint8)

        # 3. CLAHE
        clip = self.params['clahe'].get()
        if clip > 0:
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
            l = clahe.apply(l)
            img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)

        # 4. Saturation
        sat = self.params['saturation'].get()
        if sat != 1.0:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 1] *= sat
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        # 5. Brightness/Contrast
        alpha = self.params['contrast'].get()
        beta = self.params['brightness'].get()
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        # 6. Sharpness
        sharp = self.params['sharpness'].get()
        if sharp > 0:
            blurred = cv2.GaussianBlur(img, (0, 0), 3, borderType=cv2.BORDER_REPLICATE)
            img = cv2.addWeighted(img, 1.0 + sharp/2.0, blurred, -sharp/2.0, 0)
        
        # 7. Threshold Overlay
        thresh = self.params['threshold'].get()
        if thresh > 0:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # img is 8-bit (0-255)
            mask = gray > thresh                   # thresh is 0-255
            img[mask] = [255, 0, 0] 

        # 8. Channel Views
        mode = self.view_mode.get()
        if mode == "Red":
            img[:, :, 1] = 0; img[:, :, 2] = 0
        elif mode == "Green":
            img[:, :, 0] = 0; img[:, :, 2] = 0 # This is the "Combined Green" from the demosaiced image
        elif mode == "Blue":
            img[:, :, 0] = 0; img[:, :, 1] = 0
        #elif mode == "Gray":
            #gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            #img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        elif mode == "Heatmap":
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
            img = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        self.display_rgb = img
        self.update_view()
        self.update_dashboard()

    # --- Dashboard ---
    def update_dashboard(self):
        if self.raw_16bit is None: return
        
        flat_data = self.raw_16bit.flatten()
        
        # Histogram
        self.ax_hist.clear()
        self.ax_hist.hist(flat_data, bins=50, color='#00e676', log=True, alpha=0.8)
        self.ax_hist.set_title("Histogram of Pixel Values from RAW", color='white', fontsize=8)
        self.ax_hist.tick_params(colors='white', labelsize=7)
        self.ax_hist.set_facecolor('#2b2b2b')

        # FFT
        self.ax_fft.clear()
        f = np.fft.fft2(self.raw_16bit)
        fshift = np.fft.fftshift(f)
        mag = 20 * np.log(np.abs(fshift) + 1)
        self.ax_fft.imshow(mag, cmap='inferno')
        self.ax_fft.set_title("Frequency Spectrum (FFT) from RAW", color='white', fontsize=8)
        self.ax_fft.axis('off')
        
        self.fig.tight_layout()
        self.canvas.draw()
        
        # Stats
        mean_val = np.mean(flat_data)
        max_val = np.max(flat_data)
        sat_count = np.sum(flat_data >= 4095) 
        self.stats_lbl.config(text=f"Mean Intensity: {mean_val:.2f}\nMax Peak: {max_val}\nSaturated Pixels: {sat_count}")

if __name__ == "__main__":
    app = IceCube_Upgrade_Image_Analyzer()
    app.mainloop()
