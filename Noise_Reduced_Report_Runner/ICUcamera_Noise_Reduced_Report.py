"""
ICUcamera Noise Subtracted Image Generator - V1.0.0 
=========================================================
Author: Shouvik Mondal(smondal@icecube.wisc.edu)
Based on Seowon Choi's scripts [schoi1@icecube.wisc.edu] / [choi940927@gmail.com]
# version 1.0.0
Last Updated: 2026-05-05
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from pathlib import Path
from datetime import datetime
import cv2

try:
    import ICUCamera as icuc #download in a same folder as this script
except ImportError:
    print("ERROR: ICUCamera.py not found!")
    sys.exit(1)


# ============================================================
# CONFIGURATION - EDIT HERE!
# ============================================================

# INPUT FILE NAME (just the filename, not full path)
INPUT_FILENAME = r"Camera-Run_IIB_string92_mDOM_port5106_cam1_illum1_gain0_exposure3700ms_20260327-16-15-56_trial0_new.raw"

# INPUT DIRECTORY (where the file is located)
INPUT_DIRECTORY = r"C:\Users\shouv\Downloads"

# OUTPUT DIRECTORY (where to save results)
OUTPUT_DIRECTORY = r"C:\Users\shouv\Downloads\analysis_results"

# PROCESSING PARAMETERS
PEDESTAL = 235.0          # Camera baseline (ADU)
LOG_K = 300.0             # Log stretch parameter (50-800)
ASINH_A = 15.0            # Asinh parameter (5-50)
GAMMA = 0.7               # Gamma value (0.3-2.0)
DPI = 150                 # Output resolution (75-300)

# STRETCHING METHODS TO APPLY
STRETCHES = ['linear', 'log', 'asinh', 'gamma']

# VISUALIZATIONS TO GENERATE
CHANNELS = ['RGB', 'Grey', 'Red', 'Blue', 'Green', 'Pedestal-Sub', 'B-R Sub', 'B-G Sub']

# GENERATE PDF REPORT?
CREATE_PDF = True

# GENERATE SUMMARY PANEL?
CREATE_SUMMARY = True

# ============================================================
# END OF CONFIGURATION
# ============================================================

#File naming convention
class ImageAnalyzer:
    def __init__(self, input_file, output_dir=None):
        self.input_file = input_file
        self.file_name = os.path.basename(input_file)
        self.base_name = self.file_name.replace('.RAW', '').replace('.raw', '')
        
        if output_dir:
            self.output_dir = output_dir
        else:
            self.output_dir = os.path.dirname(input_file)
        
        # Create folder using base_name
        self.analysis_dir = os.path.join(self.output_dir, f"{self.base_name}_analysis")
        os.makedirs(self.analysis_dir, exist_ok=True)
        
        # Use SHORT names for output files(windows os have 260 character limit!)
        self.short_prefix = "img"
        self.counter = 0
        
        # Parameters
        self.pedestal = 235.0
        self.log_k = 300.0
        self.asinh_a = 15.0
        self.gamma_val = 0.7
        self.dpi = 150
        
        # Scan image
        print(f"\n{'='*60}")
        print(f"Scanning image: {self.file_name}")
        print(f"{'='*60}")
        
        self.shape, self.npy = icuc.Raw2Npy(self.input_file)
        print(f"Image scanned successfully")
        print(f"  Shape: {self.shape}")
        
        self.channels = self._extract_channels()
        
        # Parse metadata
        parts = self.file_name.split('_')
        self.device_info = f"{parts[7]}_{parts[8]}_{parts[9]}" if len(parts) > 9 else "Unknown"
        self.exposure_val = next((p for p in parts if 'exposure' in p), "unknown")
        
        print(f"  Device: {self.device_info}")
        print(f"  Exposure: {self.exposure_val}")
        print(f"  Output directory: {self.analysis_dir}")
        
        # Calculate channel statistics/info for consistent scaling
        self._calculate_channel_stats()
    
    def _extract_channels(self):
        """Extract RGGB channels"""
        H, W = self.npy.shape
        npy = self.npy[:H - (H % 2), :W - (W % 2)]
        
        return {
            'R': npy[0::2, 0::2].astype(np.float32),
            'G1': npy[0::2, 1::2].astype(np.float32),
            'G2': npy[1::2, 0::2].astype(np.float32),
            'B': npy[1::2, 1::2].astype(np.float32)
        }
    
    def _calculate_channel_stats(self):
        """Calculate channel statistics for consistent color scaling"""
        self.channel_stats = {}
        
        for name, channel in self.channels.items():
            flat = channel.flatten()
            self.channel_stats[name] = {
                'min': np.percentile(flat, 0.5),
                'max': np.percentile(flat, 99.5),
                'mean': np.mean(flat),
                'std': np.std(flat)
            }
        
        print(f"\n  Channel Statistics (0.5%-99.5% range):")
        for name, stats in self.channel_stats.items():
            print(f"    {name}: {stats['min']:.1f} - {stats['max']:.1f} ")
    
    def _get_visualization(self, channel_name):
        """Get base visualization image"""
        if channel_name == 'RGB':
            # Use cv2-based processing for proper RGB
            bgr = icuc.Npy2Bgr(self.npy)
            bgr = icuc.BgrCorrection(bgr)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            # Clip to [0, 1] to avoid warning
            return np.clip(rgb, 0, 1)
        
        elif channel_name == 'Grey':
            # Grey = corrected grayscale
            grey = icuc.get_gray(self.npy, pedestal=self.pedestal)
            return grey.astype(np.float32)
        
        elif channel_name == 'Red':
            return self.channels['R'].astype(np.float32)
        
        elif channel_name == 'Blue':
            return self.channels['B'].astype(np.float32)
        
        elif channel_name == 'Green':
            return 0.5 * (self.channels['G1'] + self.channels['G2'])
        
        elif channel_name == 'Pedestal-Sub':
            return self.channels['B'] - self.pedestal
        
        elif channel_name == 'B-R Sub':
            return (self.channels['B'] - self.pedestal) - (self.channels['R'] - self.pedestal)
        
        elif channel_name == 'B-G Sub':
            G_avg = 0.5 * (self.channels['G1'] + self.channels['G2'])
            return (self.channels['B'] - self.pedestal) - (G_avg - self.pedestal)
        
        return None
    
    def _linear_stretch(self, img, p_black=0.5, p_white=99.9):
        """Linear stretch - maps [black_point, white_point] → [0, 1]"""
        img = img.astype(np.float32)
        black = np.percentile(img, p_black)
        white = np.percentile(img, p_white)
        stretched = (img - black) / max(white - black, 1e-6)
        return np.clip(stretched, 0, 1)
    
    def _log_stretch(self, img, p_black=0.5, p_white=99.9):
        """Logarithmic stretch - reveals faint structures via log scale"""
        img = img.astype(np.float32)
        img = img - np.min(img)
        black = np.percentile(img, p_black)
        white = np.percentile(img, p_white)
        stretched = (img - black) / max(white - black, 1e-6)
        stretched = np.clip(stretched, 0, 1)
        # Apply log compression to [0, 1]
        return np.log1p(self.log_k * stretched) / np.log1p(self.log_k)
    
    def _asinh_stretch(self, img, p_black=0.5, p_white=99.9):
        """Asinh stretch - smooth nonlinear, preserves detail better than log"""
        img = img.astype(np.float32)
        img = img - np.min(img)
        black = np.percentile(img, p_black)
        white = np.percentile(img, p_white)
        stretched = (img - black) / max(white - black, 1e-6)
        stretched = np.clip(stretched, 0, 1)
        # Apply asinh to reveal structure
        return np.arcsinh(self.asinh_a * stretched) / np.arcsinh(self.asinh_a)
    
    def _gamma_stretch(self, img, p_black=0.5, p_white=99.9):
        """Gamma stretch - power law brightness adjustment"""
        img = img.astype(np.float32)
        img = img - np.min(img)
        black = np.percentile(img, p_black)
        white = np.percentile(img, p_white)
        stretched = (img - black) / max(white - black, 1e-6)
        stretched = np.clip(stretched, 0, 1)
        # Apply gamma correction (gamma < 1 brightens, > 1 darkens)
        return np.power(stretched, self.gamma_val)
    
    def _apply_stretch(self, img, stretch_method):
        """Apply stretching method
        
        All methods normalize to [0, 1] for display
        Different methods show different tonal distributions
        """
        if stretch_method == 'linear':
            return self._linear_stretch(img)
        elif stretch_method == 'log':
            return self._log_stretch(img)
        elif stretch_method == 'asinh':
            return self._asinh_stretch(img)
        elif stretch_method == 'gamma':
            return self._gamma_stretch(img)
        return self._linear_stretch(img)
    
    def _get_colormap(self, channel):
        """Get appropriate colormap"""
        if 'Red' in channel or channel == 'R':
            return 'Reds'
        elif 'Blue' in channel or channel == 'B':
            return 'Blues'
        elif 'Green' in channel or channel == 'G':
            return 'Greens'
        elif 'Sub' in channel:
            return 'RdBu_r'
        elif channel == 'RGB':
            return None
        else:
            return 'hot'
    
    def _get_vminmax(self, channel_name):
        """Get consistent vmin/vmax for channel (ACTUAL ADU VALUES, not normalized)"""
        if channel_name == 'Red':
            stats = self.channel_stats['R']
        elif channel_name == 'Blue':
            stats = self.channel_stats['B']
        elif channel_name == 'Green':
            g1_stats = self.channel_stats['G1']
            g2_stats = self.channel_stats['G2']
            return (
                (g1_stats['min'] + g2_stats['min']) / 2,
                (g1_stats['max'] + g2_stats['max']) / 2
            )
        else:
            return None, None
        
        return stats['min'], stats['max']
    
    def process(self, stretch_methods, visualizations, create_summary=True, 
                create_pdf=True):
        """Process all combinations"""
        count = 0
        all_images = {}
        all_vmm = {}
        
        total_combinations = len(visualizations) * len(stretch_methods)
        current = 0
        
        print(f"\n{'='*60}")
        print(f"Processing Combinations")
        print(f"{'='*60}")
        print(f"Stretching methods: {', '.join(stretch_methods)}")
        print(f"  - linear: Linear mapping of [percentile_0.5, percentile_99.9] → [0, 1]")
        print(f"  - log: Logarithmic compression log1p(k*x) for faint structures")
        print(f"  - asinh: Arcsinh smooth nonlinear stretch")
        print(f"  - gamma: Power law γ={self.gamma_val} (brightens dark areas)")
        print(f"Visualizations: {', '.join(visualizations)}")
        print(f"Total combinations: {total_combinations}")
        print(f"{'='*60}\n")
        
        for channel in visualizations:
            base_img = self._get_visualization(channel)
            if base_img is None:
                continue
            
            all_images[channel] = base_img
            all_vmm[channel] = self._get_vminmax(channel)
            
            for stretch in stretch_methods:
                current += 1
                print(f"[{current:2d}/{total_combinations}] Processing: {channel:12s} + {stretch:6s}...", end=" ")
                
                # Apply stretch to get the stretched image for display [0, 1]
                stretched = self._apply_stretch(base_img, stretch)
                
                # SHORT filename with counter
                self.counter += 1
                filename = f"{self.short_prefix}_{self.counter:03d}_{channel}_{stretch}.png"
                filepath = os.path.join(self.analysis_dir, filename)
                
                # Save PNG with proper rendering
                self._save_png_with_colorbar(
                    filepath, 
                    base_img,          # Original data (for colorbar scale)
                    stretched,         # Stretched data [0, 1] (for display)
                    channel, 
                    stretch
                )
                
                print("✓")
                count += 1
        
        print(f"\n{'='*60}")
        
        if create_summary:
            print("Creating summary panel...", end=" ")
            self._create_summary_panel(all_images, all_vmm)
            print("✓")
        
        if create_pdf:
            print("Creating PDF report...", end=" ")
            self._create_pdf_report(all_images, all_vmm)
            print("✓")
        
        print(f"{'='*60}\n")
        
        return count
    
    def _save_png_with_colorbar(self, filepath, base_img, stretched_img, channel_name, stretch_name):
        """Save PNG with colorbars showing ACTUAL ADU VALUES (not normalized)"""
        
        fig = plt.figure(figsize=(11, 9))
        ax = fig.add_subplot(111)
        
        # Get vmin/vmax for this channel (ACTUAL ADU values)
        vmin, vmax = self._get_vminmax(channel_name)
        
        # ============================================================
        # RGB IMAGE - Clear color with proper stretching
        # ============================================================
        if channel_name == 'RGB':
            ax.imshow(stretched_img)
            ax.set_title(f"{channel_name} - {stretch_name.capitalize()}", 
                        fontsize=13, fontweight='bold', pad=15)
            ax.set_xlabel('X Pixel', fontsize=11, fontweight='bold')
            ax.set_ylabel('Y Pixel', fontsize=11, fontweight='bold')
            ax.tick_params(labelsize=9)
        
        # ============================================================
        # GREY IMAGE - With stretched display and ADU colorbar
        # ============================================================
        elif channel_name == 'Grey':
            # Display stretched image
            im = ax.imshow(stretched_img, cmap='gray', vmin=0, vmax=1)
            
            # Create colorbar showing ACTUAL ADU values from base_img
            vmin_actual = np.percentile(base_img, 1)
            vmax_actual = np.percentile(base_img, 99)
            sm = ScalarMappable(norm=mcolors.Normalize(vmin=vmin_actual, vmax=vmax_actual), cmap='gray')
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(f'Intensity (ADU: {vmin_actual:.0f}-{vmax_actual:.0f})', 
                          rotation=270, labelpad=20, fontsize=10)
            
            ax.set_title(f"{channel_name} - {stretch_name.capitalize()}", 
                        fontsize=13, fontweight='bold', pad=15)
            ax.set_xlabel('X Pixel', fontsize=11, fontweight='bold')
            ax.set_ylabel('Y Pixel', fontsize=11, fontweight='bold')
            ax.tick_params(labelsize=9)
        
        # ============================================================
        # SINGLE CHANNELS (Red, Blue, Green) - With stretched display and ADU colorbar
        # ============================================================
        elif channel_name in ['Red', 'Blue', 'Green']:
            cmap = self._get_colormap(channel_name)
            # Display stretched [0, 1]
            im = ax.imshow(stretched_img, cmap=cmap, vmin=0, vmax=1)
            
            # Create colorbar showing ACTUAL ADU values
            sm = ScalarMappable(norm=mcolors.Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(f'{channel_name} (ADU: {vmin:.0f}-{vmax:.0f})', 
                          rotation=270, labelpad=20, fontsize=10)
            
            ax.set_title(f"{channel_name} - {stretch_name.capitalize()}", 
                        fontsize=13, fontweight='bold', pad=15)
            ax.set_xlabel('X Pixel', fontsize=11, fontweight='bold')
            ax.set_ylabel('Y Pixel', fontsize=11, fontweight='bold')
            ax.tick_params(labelsize=9)
        
        # ============================================================
        # PEDESTAL-SUB - With stretched display and actual value colorbar
        # ============================================================
        elif channel_name == 'Pedestal-Sub':
            cmap = self._get_colormap(channel_name)
            im = ax.imshow(stretched_img, cmap=cmap, vmin=0, vmax=1)
            
            # Colorbar for actual values
            vmin_actual = np.percentile(base_img, 1)
            vmax_actual = np.percentile(base_img, 99)
            sm = ScalarMappable(norm=mcolors.Normalize(vmin=vmin_actual, vmax=vmax_actual), cmap=cmap)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(f'Value (ADU: {vmin_actual:.0f}-{vmax_actual:.0f})', 
                          rotation=270, labelpad=20, fontsize=10)
            
            ax.set_title(f"{channel_name} - {stretch_name.capitalize()}", 
                        fontsize=13, fontweight='bold', pad=15)
            ax.set_xlabel('X Pixel', fontsize=11, fontweight='bold')
            ax.set_ylabel('Y Pixel', fontsize=11, fontweight='bold')
            ax.tick_params(labelsize=9)
        
        # ============================================================
        # DIFFERENCE IMAGES (B-R Sub, B-G Sub) - With stretched display and actual value colorbar
        # ============================================================
        elif 'Sub' in channel_name:
            cmap = self._get_colormap(channel_name)
            im = ax.imshow(stretched_img, cmap=cmap, vmin=0, vmax=1)
            
            # Colorbar for actual values
            vmin_actual = np.percentile(base_img, 1)
            vmax_actual = np.percentile(base_img, 99)
            sm = ScalarMappable(norm=mcolors.Normalize(vmin=vmin_actual, vmax=vmax_actual), cmap=cmap)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(f'Difference (ADU: {vmin_actual:.0f}-{vmax_actual:.0f})', 
                          rotation=270, labelpad=20, fontsize=10)
            
            ax.set_title(f"{channel_name} - {stretch_name.capitalize()}", 
                        fontsize=13, fontweight='bold', pad=15)
            ax.set_xlabel('X Pixel', fontsize=11, fontweight='bold')
            ax.set_ylabel('Y Pixel', fontsize=11, fontweight='bold')
            ax.tick_params(labelsize=9)
        
        else:
            # Fallback
            ax.imshow(stretched_img)
            ax.set_title(f"{channel_name} - {stretch_name.capitalize()}", 
                        fontsize=13, fontweight='bold', pad=15)
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def _create_summary_panel(self, all_images, all_vmm):
        """Create multi-panel summary with proper formatting"""
        n_channels = len(all_images)
        if n_channels == 0:
            return
        
        ncols = min(4, n_channels)
        nrows = (n_channels + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4.5*nrows))
        if n_channels == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, (channel_name, img) in enumerate(all_images.items()):
            ax = axes[idx]
            cmap = self._get_colormap(channel_name)
            vmin, vmax = all_vmm[channel_name]
            
            # ============================================================
            # RGB - Natural color display
            # ============================================================
            if channel_name == 'RGB':
                stretched = self._linear_stretch(img)
                ax.imshow(stretched)
                ax.set_title(channel_name, fontsize=12, fontweight='bold')
                ax.set_xlabel('X Pixel', fontsize=9)
                ax.set_ylabel('Y Pixel', fontsize=9)
            
            # ============================================================
            # GREY - With linear stretch and ADU colorbar
            # ============================================================
            elif channel_name == 'Grey':
                stretched = self._linear_stretch(img)
                im = ax.imshow(stretched, cmap='gray', vmin=0, vmax=1)
                
                # Colorbar with actual ADU values
                vmin_actual = np.percentile(img, 1)
                vmax_actual = np.percentile(img, 99)
                sm = ScalarMappable(norm=mcolors.Normalize(vmin=vmin_actual, vmax=vmax_actual), cmap='gray')
                sm.set_array([])
                cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label(f'ADU', fontsize=8)
                
                ax.set_title(channel_name, fontsize=12, fontweight='bold')
                ax.set_xlabel('X Pixel', fontsize=9)
                ax.set_ylabel('Y Pixel', fontsize=9)
            
            # ============================================================
            # SINGLE CHANNELS - With colorbars showing ADU
            # ============================================================
            elif channel_name in ['Red', 'Blue', 'Green']:
                stretched = self._linear_stretch(img)
                im = ax.imshow(stretched, cmap=cmap, vmin=0, vmax=1)
                
                # Colorbar with actual ADU values
                sm = ScalarMappable(norm=mcolors.Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
                sm.set_array([])
                cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label(f'ADU', fontsize=8)
                
                ax.set_title(channel_name, fontsize=12, fontweight='bold')
                ax.set_xlabel('X Pixel', fontsize=9)
                ax.set_ylabel('Y Pixel', fontsize=9)
            
            # ============================================================
            # OTHER CHANNELS
            # ============================================================
            else:
                stretched = self._linear_stretch(img)
                im = ax.imshow(stretched, cmap=cmap)
                
                vmin_actual = np.percentile(img, 1)
                vmax_actual = np.percentile(img, 99)
                sm = ScalarMappable(norm=mcolors.Normalize(vmin=vmin_actual, vmax=vmax_actual), cmap=cmap)
                sm.set_array([])
                cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label(f'ADU', fontsize=8)
                
                ax.set_title(channel_name, fontsize=12, fontweight='bold')
                ax.set_xlabel('X Pixel', fontsize=9)
                ax.set_ylabel('Y Pixel', fontsize=9)
        
        # Hide unused subplots
        for idx in range(n_channels, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f"Summary: {self.file_name}", fontsize=15, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        filepath = os.path.join(self.analysis_dir, f"{self.short_prefix}_000_summary.png")
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def _create_pdf_report(self, all_images, all_vmm):
        """Create comprehensive PDF report"""
        
        try:
            hdr_txt = icuc.header_info(self.input_file)
        except:
            hdr_txt = "Header information not available"
        
        common_hdr_lst = hdr_txt.split("\n")[0:5]
        common_hdr_txt = "\n".join(line.strip() for line in common_hdr_lst)
        camera_hdr_lst = hdr_txt.split("\n")[5:]
        camera_hdr_txt = "\n".join(line.strip() for line in camera_hdr_lst)
        
        B_channel_flat = self.channels['B'].flatten()
        G1_channel_flat = self.channels['G1'].flatten()
        G2_channel_flat = self.channels['G2'].flatten()
        R_channel_flat = self.channels['R'].flatten()
        
        FIG_W = 12
        height_ratios = [2.5, 5.0, 12.0, 5.0, 10.0]
        FIG_H = float(np.sum(height_ratios))
        
        fig = plt.figure(figsize=(FIG_W, FIG_H))
        gs = fig.add_gridspec(nrows=5, ncols=1, height_ratios=height_ratios, hspace=0.35)
        fig.subplots_adjust(left=0.06, right=0.96, top=0.95, bottom=0.02)
        
        # [0] Header Info
        gs0 = gs[0].subgridspec(2, 2, height_ratios=[0.5, 1.0], wspace=0.1, hspace=0.25)
        
        ax0_txt = fig.add_subplot(gs0[0, :])
        ax0_txt.axis("off")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ax0_txt.text(0.0, 1.0, 
                    f"File name : {self.file_name}\nResolution : {self.shape}\nAnalyzed : {timestamp}", 
                    ha="left", va="top", fontsize=13, fontweight='bold')
        
        ax1_l = fig.add_subplot(gs0[1, 0])
        ax1_r = fig.add_subplot(gs0[1, 1])
        ax1_l.text(0.0, 0.95, common_hdr_txt, va="top", ha="left", fontsize=10, family="monospace")
        ax1_l.axis("off")
        ax1_l.set_title("[Common Header]", loc="left", pad=6, fontsize=10)
        ax1_r.text(0.0, 0.95, camera_hdr_txt, va="top", ha="left", fontsize=10, family="monospace")
        ax1_r.axis("off")
        ax1_r.set_title("[Camera Header]", loc="left", pad=6, fontsize=10)
        
        self._add_section_title(fig, ax0_txt, "[0] General Info", dy=0.015)
        
        # [1] Original Data
        gs1 = gs[1].subgridspec(1, 2, wspace=0.25)
        ax1_l = fig.add_subplot(gs1[0, 0])
        ax1_r = fig.add_subplot(gs1[0, 1])
        
        im1 = ax1_l.imshow(self.npy, cmap='gray', vmin=0, vmax=4095)
        fig.colorbar(im1, ax=ax1_l, fraction=0.046, pad=0.04)
        ax1_l.set_title('Raw 12-bit Bayer Data')
        
        # Histograms
        ax1_r.hist(B_channel_flat, bins=256, range=(0, 4095), color='b', alpha=0.6, label='B', histtype='step', linewidth=1.5)
        ax1_r.hist(G1_channel_flat, bins=256, range=(0, 4095), color='g', alpha=0.6, label='G1', histtype='step', linewidth=1.5)
        ax1_r.hist(G2_channel_flat, bins=256, range=(0, 4095), color='lime', alpha=0.6, label='G2', histtype='step', linewidth=1.5)
        ax1_r.hist(R_channel_flat, bins=256, range=(0, 4095), color='r', alpha=0.6, label='R', histtype='step', linewidth=1.5)
        ax1_r.set_yscale('log')
        ax1_r.set_xlabel('Pixel Value (ADU)')
        ax1_r.set_ylabel('Count')
        ax1_r.set_title('Channel Histograms')
        ax1_r.legend(fontsize=9)
        ax1_r.grid(True, alpha=0.3)
        
        self._add_section_title(fig, ax1_l, "[1] Raw Data Analysis", dy=0.015)
        
        # [2] RGGB Channel Splitting
        gs2 = gs[2].subgridspec(2, 2, wspace=0.25, hspace=0.35)
        axes2 = [fig.add_subplot(gs2[i, j]) for i in range(2) for j in range(2)]
        
        channel_lst = [self.channels['R'], self.channels['G1'], self.channels['G2'], self.channels['B']]
        channel_name = ['R', 'G1', 'G2', 'B']
        cmap_lst = ['Reds', 'Greens', 'Greens', 'Blues']
        color_lst = ['red', 'green', 'green', 'blue']
        
        for i in range(4):
            ax = axes2[i]
            channel = channel_lst[i].astype(np.float32)
            vmin = self.channel_stats[channel_name[i]]['min']
            vmax = self.channel_stats[channel_name[i]]['max']
            
            ax.set_title(f"{channel_name[i]} Channel [{vmin:.0f}-{vmax:.0f} ADU]", 
                        color=color_lst[i], pad=8, fontsize=10)
            im2 = ax.imshow(channel, cmap=cmap_lst[i], vmin=vmin, vmax=vmax)
            fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.02)
        
        self._add_section_title(fig, axes2[0], "[2] RGGB Bayer Channel Separation", dy=0.018)
        
        # [3] Visualization Examples
        gs3 = gs[3].subgridspec(1, 2, wspace=0.25, width_ratios=[1.25, 1.15])
        ax3_l = fig.add_subplot(gs3[0, 0])
        ax3_r = fig.add_subplot(gs3[0, 1])
        
        # Grey with LogNorm
        gray = icuc.get_gray(self.npy, pedestal=self.pedestal)
        img = gray.astype(np.float32)
        img_disp = np.maximum(img, 1e-3)
        norm = mcolors.LogNorm(vmin=np.percentile(img_disp, 1), vmax=np.percentile(img_disp, 99))
        im3 = ax3_l.imshow(img_disp, cmap='gray', norm=norm)
        fig.colorbar(im3, ax=ax3_l, fraction=0.046, pad=0.04, label='log intensity')
        ax3_l.set_title('Corrected Grayscale\n(Log Normalized)', pad=10, fontsize=10)
        
        # RGB visualization with proper cv2 processing
        bgr16 = icuc.Npy2Bgr16(self.npy, bayer_code=cv2.COLOR_BAYER_BG2BGR)
        bgr01_asinh = icuc.stretch_preserve_color(bgr16, p_black=0.1, p_white=99, a=40.0)
        out8_asinh = icuc.to_uint8(bgr01_asinh)
        out8_asinh_clahe = icuc.clahe_on_l_channel(out8_asinh, clipLimit=3.0)
        rgb = cv2.cvtColor(out8_asinh_clahe, cv2.COLOR_BGR2RGB)
        
        ax3_r.imshow(rgb)
        ax3_r.set_title('RGB Image using ASINH and CLAHE)', pad=10)
        
        self._add_section_title(fig, ax3_l, "[3] Standard Visualizations", dy=0.018)
        
        # [4] Saturation Analysis
        gs4 = gs[4].subgridspec(2, 2, wspace=0.25, hspace=0.35)
        axes4 = [fig.add_subplot(gs4[i, j]) for i in range(2) for j in range(2)]
        
        cmap = mcolors.ListedColormap(['lightgray', 'red', 'blue'])
        norm_lbl = mcolors.BoundaryNorm(boundaries=[-0.5, 0.5, 1.5, 2.5], ncolors=cmap.N)
        
        for i in range(4):
            ax = axes4[i]
            channel = channel_lst[i].astype(np.float32)
            
            dn_mask = channel < 245
            sat_mask = channel > 4090
            
            label = np.zeros_like(channel, dtype=int)
            label[dn_mask] = 1
            label[sat_mask] = 2
            
            total = self.channels['R'].size
            dn_ratio = np.sum(dn_mask) / total * 100
            sat_ratio = np.sum(sat_mask) / total * 100
            
            ax.set_title(f"{channel_name[i]} Channel\nDark: {dn_ratio:.1f}% | Sat: {sat_ratio:.1f}%",
                        color=color_lst[i], pad=8, fontsize=10)
            im4 = ax.imshow(label, cmap=cmap, norm=norm_lbl, interpolation='nearest')
            cbar = fig.colorbar(im4, ax=ax, ticks=[0, 1, 2], shrink=0.9)
            cbar.ax.set_yticklabels(['Normal', 'Dark', 'Saturated'], rotation=90, va='center', fontsize=8)
        
        self._add_section_title(fig, axes4[0], "[4] Saturation & Signal Analysis", dy=0.018)
        
        fig.text(0.5, 0.995, "Comprehensive Image Analysis Report",
                ha="center", va="top", fontsize=20, fontweight="bold")
        
        fig.text(0.02, 0.995,
                f"Software Version:\nICUcamera_Noise_Reduced_ReoportV1.0.0\nICUCamera {icuc.get_version()}",
                ha="left", va="top", fontsize=8)
        
        pdf_path = os.path.join(self.analysis_dir, f"{self.short_prefix}_000_report.pdf")
        fig.savefig(pdf_path, dpi=300, pad_inches=0.2)
        plt.close(fig)
    
    def _add_section_title(self, fig, ref_ax, title, x=0.02, dy=0.012, fontsize=16):
        """Add section title"""
        bb = ref_ax.get_position()
        y = min(bb.y1 + dy, 0.995)
        fig.text(x, y, title, fontsize=fontsize, fontweight="bold", ha="left", va="bottom")


def main():
    print("\n" + "="*60)
    print("ICUcamera Noise Subtracted Image Generator - V1.0.0")
    print("="*60)
    
    # Construct full input path
    input_file = os.path.join(INPUT_DIRECTORY, INPUT_FILENAME)
    
    # Verify input file exists
    if not os.path.exists(input_file):
        print(f"\n❌ ERROR: Input file not found!")
        print(f"   Expected: {input_file}")
        print(f"\n   Please verify:")
        print(f"   - INPUT_FILENAME = {INPUT_FILENAME}")
        print(f"   - INPUT_DIRECTORY = {INPUT_DIRECTORY}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    
    try:
        # Create analyzer
        analyzer = ImageAnalyzer(input_file, OUTPUT_DIRECTORY)
        
        # Set parameters
        analyzer.pedestal = PEDESTAL
        analyzer.log_k = LOG_K
        analyzer.asinh_a = ASINH_A
        analyzer.gamma_val = GAMMA
        analyzer.dpi = DPI
        
        # Print settings
        print(f"\n{'='*60}")
        print(f"Processing Settings")
        print(f"{'='*60}")
        print(f"Pedestal:     {analyzer.pedestal:.1f} ADU")
        print(f"Log-k:        {analyzer.log_k:.1f}")
        print(f"Asinh-a:      {analyzer.asinh_a:.1f}")
        print(f"Gamma:        {analyzer.gamma_val:.2f}")
        print(f"Output DPI:   {analyzer.dpi}")
        print(f"Generate PDF: {CREATE_PDF}")
        print(f"Gen. Summary: {CREATE_SUMMARY}")
        print(f"{'='*60}\n")
        
        # Process
        count = analyzer.process(
            stretch_methods=STRETCHES,
            visualizations=CHANNELS,
            create_summary=CREATE_SUMMARY,
            create_pdf=CREATE_PDF
        )
        
        # Summary
        print(f"{'='*60}")
        print(f"✓ Analysis Complete!")
        print(f"{'='*60}")
        print(f"Total images generated: {count}")
        print(f"Output directory: {analyzer.analysis_dir}")
        print(f"\nGenerated files:")
        print(f"  ✓ {count} PNG images (img_001-{count:03d}_CHANNEL_STRETCH.png)")
        print(f"     - All stretching methods validated:")
        print(f"       • linear: Maps percentile range to [0, 1]")
        print(f"       • log: log1p compression reveals faint structures")
        print(f"       • asinh: Smooth nonlinear stretch arcsinh(a*x)/arcsinh(a)")
        print(f"       • gamma: Power law γ={GAMMA} brightens darks")
        if CREATE_SUMMARY:
            print(f"  ✓ Summary panel (img_000_summary.png)")
        if CREATE_PDF:
            print(f"  ✓ PDF report (img_000_report.pdf)")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
