import os

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_TEMP_DIR = os.path.join(_PROJECT_ROOT, "temp")
os.makedirs(_TEMP_DIR, exist_ok=True)
os.environ["GRADIO_TEMP_DIR"] = _TEMP_DIR


import preprocess_lib as pp
import config
import cv2
import tempfile
import gradio as gr
import numpy as np
import traceback
import matplotlib
import glob
matplotlib.use('AGG')
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

def process_image(file, auto_mode, need_denoise, need_bc, need_ic,
                  need_deconv, need_clahe,
                  d_large, d_small,
                  rball, poly,
                  clahe_flu, clahe_bri,
                  rl_iter, psf_sigma,
                  clahe_tile, save_flag, progress = gr.Progress()):
    if not file:
        return None, "Upload an image to start, format could be 'tif' or 'tiff'"

    if not isinstance(file, list): #stack multiple images into a list
        file = [file]

    param = config.make_params(
        DENOISE_KERNEL_LARGE=(int(d_large), int(d_large)),
        DENOISE_KERNEL_SMALL=(int(d_small), int(d_small)),
        ROLLING_BALL_RADIUS=int(rball),
        POLYFIT_DOWNSAMPLE=float(poly),
        CLAHE_L=(float(clahe_flu), float(clahe_bri)),
        RL_ITERATIONS=int(rl_iter),
        PSF_SIGMA=float(psf_sigma),
        CLAHE_TILE_GRID=(int(clahe_tile), int(clahe_tile)),
    )

    if auto_mode:
        manual_tags = None
    else:
        manual_tags = {
            'need_denoise':                bool(need_denoise),
            'need_background_correction':  bool(need_bc),
            'need_illumination_correction': bool(need_ic),
            'need_deconvolution':          bool(need_deconv),
            'need_clahe':                  bool(need_clahe),
        }

    gallery, logs, tag_cache = [], [], {}
    mode_str = "Auto" if auto_mode else "Manual"
    logs.append(f"━━━━━━━━━━━━━━━━━━━━━━")
    logs.append(f"[Mode] {mode_str}")
    table_rows =[]

    for f in progress.tqdm(file, desc = "Processing"):
        try:
            path = getattr(f, 'name', None) or str(f)
            name = os.path.basename(path)

            img = cv2.imread(path)
            if img is None:
                logs.append(f"{name}: cannot read file")
                continue
            img = cv2.resize(img, config.TARGET_SIZE, interpolation=cv2.INTER_AREA)

            out, tags = pp.preprocessing(img, auto=auto_mode,
                                         manual_tags=manual_tags, param=param)
            tag_cache[path] = tags.copy()

            steps = tags.get('executed_steps', [])
            img_type = tags.get('image_type', '?')
            m = tags.get('eval_metrics', {})
            #evaluation restrive
            sh_o, sh_p = m.get('sharpness_original', 0), m.get('sharpness_processed', 0)
            ns_o, ns_p = m.get('noise_original', 0),     m.get('noise_processed', 0)
            ct_o, ct_p = m.get('contrast_original', 0),  m.get('contrast_processed', 0)
            en_o, en_p = m.get('entropy_original', 0),   m.get('entropy_processed', 0)
            
            sh_r = m.get('sharpness_ratio', 0)
            ns_r = m.get('noise_ratio', 0)
            ct_r = m.get('contrast_ratio', 0)
            en_r = m.get('entropy_ratio', 0)


            #gallery set
            # Image compare
            ori = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8)
            pro = (np.clip(out, 0, 1) * 255).astype(np.uint8)
            compare = np.hstack([ori, pro])

            gallery.append((compare, name))


            #matplotlib
            ori_gray = cv2.cvtColor(ori, cv2.COLOR_RGB2GRAY)
            pro_gray = cv2.cvtColor(pro, cv2.COLOR_RGB2GRAY)
            fig, ax  = plt.subplots(figsize = (6, 3))
            ax.hist(ori_gray.ravel(), bins=128, alpha=0.5, color='blue', label='Original')
            ax.hist(pro_gray.ravel(), bins=128, alpha=0.5, color='red', label='Processed')
            ax.set_title(f"{name} — Histogram")
            ax.legend()
            plt.tight_layout()
            plt.xlabel("pixel intensity")
            plt.ylabel("pixel population")
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=80)
            plt.close(fig)
            buf.seek(0)
            hist_img = np.array(Image.open(buf))
            gallery.append((hist_img, f"{name} - Histogram"))


            # Saving option
            for old_file in glob.glob(os.path.join(os.path.dirname(__file__), "tmp*.csv")):
                try: os.remove(old_file)
                except: pass

            if save_flag:
                pp.save_preproiamge({path:out}, config.OUTPUT_DIR)
                logs.append(f"saved to {config.OUTPUT_DIR}")


            # metrics visualization
            table_rows.append([name, img_type,
                               " → ".join(steps) if steps else "none",
                               f"{sh_o:.1f} → {sh_p:.1f} ({sh_r:.2f}x)",
                               f"{ns_o:.4f} → {ns_p:.4f} ({ns_r:.2f}x)",
                               f"{ct_o:.1f} → {ct_p:.1f} ({ct_r:.2f}x)",
                               f"{en_o:.3f} → {en_p:.3f} ({en_r:.2f}x)"])

            #logs
            logs.append(f"")
            logs.append(f" {name}")
            logs.append(f"   type: {img_type}, size: {img.shape[:2]}")
            logs.append(f"   clearness: {m.get('sharpness_ratio', 0):.2f}x  |  "
                        f"noise: {m.get('noise_ratio', 0):.2f}x  |  "
                        f"contrast: {m.get('contrast_ratio', 0):.2f}x")

            

          
        except Exception as e:
            logs.append(f" {os.path.basename(str(f))}: {e}")
            logs.append(traceback.format_exc())

    logs.append(f"━━━━━━━━━━━━━━━━━━━━━━")
    logs.append(f" Done: {len(table_rows)}/{len(file)} images")
    return tag_cache, gallery, "\n".join(logs), table_rows



def reset_parameters():
    """Reset all config at once"""
    return (
        config.DENOISE_KERNEL_LARGE[0],   # d_large
        config.DENOISE_KERNEL_SMALL[0],   # d_small
        config.ROLLING_BALL_RADIUS,       # rball
        config.POLYFIT_DOWNSAMPLE,        # poly
        config.CLAHE_L[0],                # clahe_flu
        config.CLAHE_L[1],                # clahe_bri
        config.RL_ITERATIONS,             # rl_iter
        config.PSF_SIGMA,                 # psf_sigma
        config.CLAHE_TILE_GRID[0],        # clahe_tile
    )

def export_report(tags_cache):
    """store the tempo tags into file.cvs"""
    if not tags_cache:
        return None
    tmp = tempfile.NamedTemporaryFile(
        mode='w', suffix='.csv', delete=False,
        dir=os.path.dirname(os.path.abspath(__file__)),
        encoding='utf-8-sig',         
    )
    pp.prepro_report(tags_cache, tmp.name)
    print(f"[Report] exported to {tmp.name}")

    return tmp.name



with gr.Blocks(title="Preprocessing", theme = gr.themes.Soft()) as demo:
    gr.Markdown("# Preprocessing")
    
    with gr.Row():
        with gr.Column(scale=1):
            # 
            files_input = gr.File(label="Upload image here", file_count="multiple", file_types=[".tif", ".tiff"])
            #auto cehck 
            auto_mode = gr.Checkbox(label="Auto mode (recommended)", value=True)
            
            #Manual Panel    
            with gr.Group(visible=False) as manual_panel:

                # 1. Denoise
                with gr.Group():
                    need_denoise = gr.Checkbox(label="🔹 Denoise", value=True)
                    with gr.Row(visible=True) as g_dn:
                        d_large = gr.Slider(1, 15, value=config.DENOISE_KERNEL_LARGE[0],
                                            step=2, label="Kernel (large)")
                        d_small = gr.Slider(1, 9, value=config.DENOISE_KERNEL_SMALL[0],
                                            step=2, label="Kernel (small)")
                need_denoise.change(lambda v: gr.Row(visible=v),
                                    inputs=need_denoise, outputs=g_dn)

                # 2. Background Correction
                with gr.Group():
                    need_bc = gr.Checkbox(label="🔹 Background Correction", value=False)
                    with gr.Row(visible=False) as g_bc:
                        rball = gr.Slider(10, 100, value=config.ROLLING_BALL_RADIUS,
                                          step=5, label="Rolling ball radius")
                        poly = gr.Slider(0.1, 1.0, value=config.POLYFIT_DOWNSAMPLE,
                                         step=0.05, label="Polyfit downsample")
                need_bc.change(lambda v: gr.Row(visible=v),
                               inputs=need_bc, outputs=g_bc)

                # 3. Illumination Correction
                with gr.Group():
                    need_ic = gr.Checkbox(label="🔹 Illumination Correction", value=False)
                    with gr.Row(visible=False) as g_ic:
                        clahe_flu = gr.Slider(0.5, 5.0, value=config.CLAHE_L[0],
                                              step=0.1, label="CLAHE clip (flu)")
                        clahe_bri = gr.Slider(0.5, 5.0, value=config.CLAHE_L[1],
                                              step=0.1, label="CLAHE clip (bri)")
                need_ic.change(lambda v: gr.Row(visible=v),
                               inputs=need_ic, outputs=g_ic)

                # 4. Deconvolution
                with gr.Group():
                    need_deconv = gr.Checkbox(label="🔹 Deconvolution", value=False)
                    with gr.Row(visible=False) as g_deconv:
                        rl_iter = gr.Slider(1, 30, value=config.RL_ITERATIONS,
                                            step=1, label="RL iterations")
                        psf_sigma = gr.Slider(0.5, 5.0, value=config.PSF_SIGMA,
                                              step=0.1, label="PSF sigma")
                need_deconv.change(lambda v: gr.Row(visible=v),
                                   inputs=need_deconv, outputs=g_deconv)

                # 5. CLAHE 
                with gr.Group():
                    need_clahe = gr.Checkbox(label="🔹 CLAHE", value=True)
                    with gr.Row(visible=True) as g_cl:
                        clahe_tile = gr.Slider(4, 16, value=config.CLAHE_TILE_GRID[0],
                                               step=2, label="Tile grid")
                need_clahe.change(lambda v: gr.Row(visible=v),
                                  inputs=need_clahe, outputs=g_cl)

            auto_mode.change(lambda c: gr.Group(visible=not c),
                             inputs=auto_mode, outputs=manual_panel)

            save_flag = gr.Checkbox(label = "Save the processed images", value =  False)

            # Reset
            reset_btn = gr.Button("Reset all to defaults", variant="secondary")

            run_btn = gr.Button("Run", variant="primary")

            #Export logs
            state_tags =gr.State({})
            export_btn= gr.Button("Export report (CSV)", variant="secondary")
            report_file = gr.File(label="Download report", interactive=False)

        with gr.Column(scale=2):
            gallery_out = gr.Gallery(label="Left: Original | Right: Processed",
                                     columns=1, rows=3, height=500)
            metrics_table = gr.Dataframe(                
                headers=["Image", "Type", "Steps", "Clearness", "Noise", "Contrast", "Entropy"],
                datatype=["str", "str", "str", "str", "str", "str","str"],
                label="Metrics",
                interactive=False,
            )
            log_out = gr.Textbox(label="Logs", lines=8)

    
    
    run_btn.click(
    fn=process_image,
    inputs=[files_input, auto_mode,                        
        need_denoise, need_bc, need_ic,                
        need_deconv, need_clahe,                       
        d_large, d_small,                              
        rball, poly,                                   
        clahe_flu, clahe_bri,                          
        rl_iter, psf_sigma,                            
        clahe_tile, save_flag,                                   
    ],
    outputs=[state_tags, gallery_out, log_out, metrics_table],
    )

    reset_btn.click(
        fn=reset_parameters,
        inputs=[],
        outputs=[d_large, d_small, rball, poly,
                 clahe_flu, clahe_bri, rl_iter, psf_sigma, clahe_tile],
    )


    export_btn.click(
        fn=export_report,
        inputs=[state_tags],
        outputs=[report_file],
    )

import shutil
if os.path.isdir(_TEMP_DIR):
    shutil.rmtree(_TEMP_DIR, ignore_errors=True)
    os.makedirs(_TEMP_DIR, exist_ok=True)
 
if __name__ == "__main__":
    demo.queue(default_concurrency_limit=4).launch(inbrowser=True, debug=True)
