import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pywt
import time
from skimage.metrics import structural_similarity as ssim_metric 
from skimage.metrics import peak_signal_noise_ratio as psnr_metric 
from scipy.fftpack import dct, idct
import io
import base64
import os

def load_image(image_file):
    try:
        img = Image.open(image_file).convert('L') 
        return np.array(img)
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None

def calculate_metrics(original, compressed):
    """Calculates PSNR and SSIM between two images."""
    original = original.astype(np.float32)
    compressed = compressed.astype(np.float32)

    if original.shape != compressed.shape:
        h, w = original.shape
        compressed = np.resize(compressed, (h, w))

    if np.array_equal(original, compressed):
        psnr_value = float('inf') 
        ssim_value = 1.0
    else:
        data_range = 255.0
        psnr_value = psnr_metric(original, compressed, data_range=data_range)
        win_size = min(7, original.shape[0], original.shape[1])
        if win_size < 7:
             st.warning(f"SSIM window size reduced to {win_size} due to small image size.")
        if win_size % 2 == 0:
            win_size -=1 

        if win_size >= 1:
             ssim_value = ssim_metric(original, compressed, data_range=data_range, win_size=win_size, channel_axis=None)
        else:
             ssim_value = 0.0 
             st.error("Cannot compute SSIM, image dimensions too small.")

    return psnr_value, ssim_value

def get_jpeg_quantization_matrix(quality):
    Q_lum = np.array([ 
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ], dtype=np.float32)

    if quality <= 0: quality = 1
    if quality > 100: quality = 100

    if quality < 50:
        scale = 5000 / quality
    else:
        scale = 200 - quality * 2

    Q = np.clip(np.round((Q_lum * scale) / 100.0), 1, 255)
    return Q.astype(np.float32)

def dct_compression(image, quality_factor=50, perceptual=True):
    start_time = time.time()
    image = image.astype(np.float32)
    h, w = image.shape
    block_size = 8

    h_pad = h + (block_size - h % block_size) % block_size
    w_pad = w + (block_size - w % block_size) % block_size
    padded = np.zeros((h_pad, w_pad), dtype=np.float32)
    padded[:h, :w] = image

    result = np.zeros_like(padded)

    if perceptual:
        Q = get_jpeg_quantization_matrix(quality_factor)
    else:
        Q = np.ones((block_size, block_size), dtype=np.float32)
        quality_scale = max(1, (100 - quality_factor) / 10.0) 
        for i in range(block_size):
            for j in range(block_size):
                 Q[i, j] = 1 + (i + j) * quality_scale
        Q = np.clip(Q, 1, 255)

    nonzero_count = 0
    total_coeffs = 0

    for i in range(0, h_pad, block_size):
        for j in range(0, w_pad, block_size):
            block = padded[i:i+block_size, j:j+block_size] - 128.0  # Level shift

            dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')

            quantized = np.round(dct_block / Q)
            nonzero_count += np.count_nonzero(quantized)
            total_coeffs += block_size * block_size

            dequantized = quantized * Q

            idct_block = idct(idct(dequantized.T, norm='ortho').T, norm='ortho')

            result[i:i+block_size, j:j+block_size] = idct_block + 128.0

    compression_ratio = total_coeffs / max(nonzero_count, 1) 

    result_unpadded = result[:h, :w]
    compressed_image = np.clip(result_unpadded, 0, 255).astype(np.uint8)

    execution_time = time.time() - start_time
    return compressed_image, compression_ratio, execution_time

def wavelet_compression(image, threshold_percent=10, wavelet='haar', level=3):
    start_time = time.time()
    image = image.astype(np.float32)
    original_shape = image.shape

    # Decompose
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    
    # Rebuild structure for modification convenience
    coeffs_array, coeff_slices = pywt.coeffs_to_array(coeffs)

    # Flatten detail coefficients to find threshold
    detail_coeffs = coeffs_array.copy()
    
    # Get the approximation band (LL) slice
    approx_slice = coeff_slices[0]
    
    # Create a mask for all detail coefficients (everything except approximation)
    detail_mask = np.ones_like(coeffs_array, dtype=bool)
    detail_mask[approx_slice[0], approx_slice[1]] = False
    
    # Extract only the detail coefficients
    details_only = coeffs_array[detail_mask]
    
    # Count total and non-zero coefficients before thresholding
    total_coeffs = coeffs_array.size
    nonzero_before_thresholding = np.count_nonzero(coeffs_array)
    
    # Calculate threshold based on percentile of absolute detail coefficient values
    if details_only.size > 0:
        threshold = np.percentile(np.abs(details_only), 100 - threshold_percent)
    else:
        threshold = 0  # No details to threshold
    
    # Apply thresholding to detail coefficients only (hard thresholding)
    coeffs_array_thresh = coeffs_array.copy()
    coeffs_array_thresh[detail_mask] = pywt.threshold(details_only, threshold, mode='hard')
    
    # Count non-zero coefficients after thresholding
    nonzero_after = np.count_nonzero(coeffs_array_thresh)
    
    # Reconstruct coefficients structure
    coeffs_modified = pywt.array_to_coeffs(coeffs_array_thresh, coeff_slices, output_format='wavedec2')
    
    # Reconstruct image
    reconstructed = pywt.waverec2(coeffs_modified, wavelet)
    
    # Crop to original size (wavelet transforms can introduce padding)
    if reconstructed.shape != original_shape:
        reconstructed = reconstructed[:original_shape[0], :original_shape[1]]
    
    compressed_image = np.clip(reconstructed, 0, 255).astype(np.uint8)
    
    # Calculate simplified compression ratio
    compression_ratio = total_coeffs / max(nonzero_after, 1)  # Avoid division by zero
    execution_time = time.time() - start_time
    
    return compressed_image, compression_ratio, execution_time

def subjective_quality_score(original, compressed):
    psnr_val, ssim_val = calculate_metrics(original, compressed)

    if psnr_val > 40 and ssim_val > 0.97:
        score = 5
        category = "Excellent - No visible compression artifacts"
    elif psnr_val > 35 and ssim_val > 0.94:
        score = 4
        category = "Good - Compression artifacts not noticeable in normal viewing"
    elif psnr_val > 30 and ssim_val > 0.90:
        score = 3
        category = "Fair - Minor artifacts visible upon close inspection"
    elif psnr_val > 25 and ssim_val > 0.80:
        score = 2
        category = "Poor - Visible artifacts affecting viewing experience"
    else:
        score = 1
        category = "Bad - Severe artifacts significantly degrading image quality"

    return score, category

def visualize_artifacts(original, compressed, zoom_ratio=4):
    h, w = original.shape
    zh, zw = h // zoom_ratio, w // zoom_ratio
    start_h = (h - zh) // 2
    start_w = (w - zw) // 2

    original_zoom = original[start_h : start_h + zh, start_w : start_w + zw]
    compressed_zoom = compressed[start_h : start_h + zh, start_w : start_w + zw]

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle("Compression Artifact Visualization", fontsize=16)

    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
 
    rect = plt.Rectangle((start_w, start_h), zw, zh, linewidth=1, edgecolor='r', facecolor='none')
    axes[0, 0].add_patch(rect)

    axes[0, 1].imshow(compressed, cmap='gray')
    axes[0, 1].set_title('Compressed Image')
    axes[0, 1].axis('off')
   
    rect = plt.Rectangle((start_w, start_h), zw, zh, linewidth=1, edgecolor='r', facecolor='none')
    axes[0, 1].add_patch(rect)

    axes[1, 0].imshow(original_zoom, cmap='gray')
    axes[1, 0].set_title('Original (Zoomed Region)')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(compressed_zoom, cmap='gray')
    axes[1, 1].set_title('Compressed (Zoomed Region)')
    axes[1, 1].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    return fig

def experiment_compression_quality(image, quality_range, method='dct', perceptual=True, wavelet_type='haar', wavelet_level=3):
    results = {
        'quality_parameter': quality_range,
        'compression_ratio': [],
        'psnr': [],
        'ssim': [],
        'execution_time': [],
        'subjective_score': [],
        'quality_category': []
    }

    for quality in quality_range:
        if method == 'dct':
            compressed, ratio, exec_time = dct_compression(
                image, quality_factor=quality, perceptual=perceptual)
            threshold_percent_actual = None
        else:  # wavelet
            # For wavelet, we need to invert the quality parameter to get threshold percent
            threshold_percent_actual = 100 - quality
            compressed, ratio, exec_time = wavelet_compression(
                image, threshold_percent=threshold_percent_actual, wavelet=wavelet_type, level=wavelet_level)

        psnr_val, ssim_val = calculate_metrics(image, compressed)
        subj_score, category = subjective_quality_score(image, compressed)

        results['compression_ratio'].append(ratio)
        results['psnr'].append(psnr_val)
        results['ssim'].append(ssim_val)
        results['execution_time'].append(exec_time * 1000)  # Convert to milliseconds
        results['subjective_score'].append(subj_score)
        results['quality_category'].append(category)

    return results

def plot_comparison(results_dict, metric='psnr', title=None, figsize=(10, 6)):
    plt.figure(figsize=figsize)
    
    for method_name, data in results_dict.items():
        if metric in data:
            plt.plot(data['compression_ratio'], data[metric], marker='o', linestyle='-', label=method_name)
    
    metric_name = metric.upper() if metric in ['psnr', 'ssim'] else metric.replace('_', ' ').title()
    
    plt.xlabel('Compression Ratio')
    plt.ylabel(metric_name)
    
    if title:
        plt.title(title)
    else:
        plt.title(f'Compression Ratio vs {metric_name}')
    
    plt.grid(True)
    plt.legend()
    
    return plt

def get_image_download_link(img, filename, text="Download Compressed Image"):
    """Generate a download link for an image"""
    buffered = io.BytesIO()
    Image.fromarray(img).save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:image/png;base64,{img_str}" download="{filename}">📥 {text}</a>'
    return href

# --- Streamlit App UI ---
st.set_page_config(layout="wide", page_title="Image Compression Comparison")

st.title("🖼️ Comparative Study of Lossy Image Compression")
st.write("""
Explore and compare two fundamental lossy image compression techniques:
Discrete Cosine Transform (DCT), similar to JPEG, and Wavelet Transform, similar to JPEG 2000.
Upload an image and adjust the compression parameters to see the trade-offs
between compression ratio and visual quality (measured by PSNR and SSIM).
""")

# --- Sidebar for Controls ---
st.sidebar.header("⚙️ Controls")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "tif", "bmp", "pgm"])

original_image = None
if uploaded_file is not None:
    original_image = load_image(uploaded_file)
    if original_image is not None:
        st.sidebar.image(original_image, caption="Original Uploaded Image", use_container_width=True)
else:
    st.sidebar.info("Upload an image to begin.")

st.sidebar.markdown("---") 

compression_method = st.sidebar.selectbox(
    "Select Compression Method:",
    ("DCT (JPEG-like)", "Wavelet (JPEG2000-like)")
)

# --- Compression Parameters ---
quality_factor = 50
threshold_percent = 50
perceptual_dct = True
wavelet_type = 'haar'
wavelet_level = 3

if compression_method == "DCT (JPEG-like)":
    quality_factor = st.sidebar.slider("Quality Factor (1-100)", 1, 100, 50, help="Lower value means higher compression, lower quality.")
    perceptual_dct = st.sidebar.checkbox("Use Perceptual Quantization (JPEG standard)", value=True, help="Uses standard JPEG quantization tables. If unchecked, uses a simpler linear quantization.")
    st.sidebar.markdown("---")
elif compression_method == "Wavelet (JPEG2000-like)":
    threshold_percent = st.sidebar.slider("Detail Coefficients to Keep (%)", 1, 100, 50, help="Percentage of highest-magnitude detail coefficients to retain. Lower value means higher compression.")
    threshold_percent_actual = 100 - threshold_percent 

    wavelet_type = st.sidebar.selectbox("Wavelet Type:", ['haar', 'db4', 'sym4', 'bior2.2'], index=0)
    wavelet_level = st.sidebar.slider("Decomposition Level", 1, 5, 3, help="Number of wavelet decomposition levels.")
    st.sidebar.markdown("---")

# --- Action Buttons ---
compress_button = st.sidebar.button("🚀 Compress Image", disabled=(original_image is None))
st.sidebar.markdown("---")

# --- Full Analysis Button and Parameters ---
st.sidebar.header("📊 Full Analysis")
st.sidebar.write("Compare multiple compression methods with various settings.")

run_analysis = st.sidebar.button("Run Full Analysis", disabled=(original_image is None))
st.sidebar.markdown("---")

# --- Main Panel for Results ---
if original_image is None:
    st.info("Please upload an image using the sidebar to see the results.")
else:
    st.subheader("Original Image")
    # Increased image width from 400 to 600
    st.image(original_image, caption=f"Original ({original_image.shape[1]}x{original_image.shape[0]})", use_container_width=False, width=600)

    if compress_button:
        st.subheader("Compressed Image & Analysis")
        compressed_image = None
        compression_ratio = 0.0
        execution_time = 0.0

        with st.spinner(f"Applying {compression_method} compression..."):
            if compression_method == "DCT (JPEG-like)":
                compressed_image, compression_ratio, execution_time = dct_compression(
                    original_image, quality_factor=quality_factor, perceptual=perceptual_dct
                )
                param_value = f"Quality={quality_factor}, Perceptual={perceptual_dct}"
            elif compression_method == "Wavelet (JPEG2000-like)":
                compressed_image, compression_ratio, execution_time = wavelet_compression(
                    original_image, threshold_percent=threshold_percent_actual, wavelet=wavelet_type, level=wavelet_level
                )
                param_value = f"{threshold_percent}% Kept, Wavelet='{wavelet_type}', Level={wavelet_level}"

        if compressed_image is not None:
            # Increased image width from 400 to 600
            st.image(compressed_image, caption=f"Compressed ({compression_method} | {param_value})", use_container_width=False, width=600)
            
            # Add download button for compressed image
            filename = f"compressed_{compression_method.split()[0].lower()}_{int(time.time())}.png"
            st.markdown(get_image_download_link(compressed_image, filename), unsafe_allow_html=True)

            # --- Metrics ---
            st.markdown("#### Performance Metrics")
            psnr_val, ssim_val = calculate_metrics(original_image, compressed_image)
            subj_score, category = subjective_quality_score(original_image, compressed_image)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Compression Ratio", f"{compression_ratio:.2f} : 1", help="Ratio of total coefficients to non-zero quantized coefficients (higher is more compressed). Simplified measure.")
            col2.metric("PSNR (dB)", f"{psnr_val:.2f}", help="Peak Signal-to-Noise Ratio (higher is better quality). 'inf' means identical.")
            col3.metric("SSIM", f"{ssim_val:.4f}", help="Structural Similarity Index (closer to 1 is better quality).")
            col4.metric("Time (ms)", f"{execution_time * 1000:.2f}", help="Execution time for the compression function.")

            # --- Quality Assessment ---
            st.markdown("#### Subjective Quality Assessment")
            st.info(f"**Quality Score: {subj_score}/5** - {category}")

            # --- Artifact Visualization ---
            st.markdown("#### Artifact Visualization (Zoomed)")
            with st.spinner("Generating artifact visualization..."):
                 fig_artifacts = visualize_artifacts(original_image, compressed_image)
                 st.pyplot(fig_artifacts)

    # --- Full Analysis Section ---
    if run_analysis:
        st.header("📊 Full Compression Analysis")
        
        with st.spinner("Running comprehensive analysis of different compression methods..."):
            # Define quality parameters for each method
            dct_qualities = [90, 75, 50, 25, 10, 5]
            wavelet_qualities = [90, 75, 50, 25, 10, 5]  # These will be inverted for threshold
            
            # Run experiments
            results = {
                "DCT (Perceptual)": experiment_compression_quality(
                    original_image, dct_qualities, method='dct', perceptual=True),
                "DCT (Simple)": experiment_compression_quality(
                    original_image, dct_qualities, method='dct', perceptual=False),
                "Wavelet (Haar)": experiment_compression_quality(
                    original_image, wavelet_qualities, method='wavelet', wavelet_type='haar', wavelet_level=3),
                "Wavelet (DB4)": experiment_compression_quality(
                    original_image, wavelet_qualities, method='wavelet', wavelet_type='db4', wavelet_level=3)
            }
            
            # Display results in tabular format
            st.subheader("Compression Method Comparison")
            
            tabs = st.tabs(["PSNR Comparison", "SSIM Comparison", "Subjective Quality", "Execution Time"])
            
            with tabs[0]:
                st.write("#### PSNR vs Compression Ratio")
                fig_psnr = plot_comparison(results, 'psnr', 'PSNR vs Compression Ratio')
                st.pyplot(fig_psnr.gcf())
                
            with tabs[1]:
                st.write("#### SSIM vs Compression Ratio")
                fig_ssim = plot_comparison(results, 'ssim', 'SSIM vs Compression Ratio')
                st.pyplot(fig_ssim.gcf())
                
            with tabs[2]:
                st.write("#### Subjective Quality Score vs Compression Ratio")
                fig_subj = plot_comparison(results, 'subjective_score', 'Subjective Quality vs Compression Ratio')
                st.pyplot(fig_subj.gcf())
                
            with tabs[3]:
                st.write("#### Execution Time vs Compression Ratio")
                fig_time = plot_comparison(results, 'execution_time', 'Execution Time vs Compression Ratio')
                st.pyplot(fig_time.gcf())
            
            # Display detailed results in a table
            st.subheader("Detailed Results")
            
            # Create a more detailed table for each method
            for method_name, method_results in results.items():
                st.write(f"#### {method_name}")
                
                # Create a dataframe for this method's results
                data = {
                    "Quality Parameter": method_results['quality_parameter'],
                    "Compression Ratio": [f"{cr:.2f}" for cr in method_results['compression_ratio']],
                    "PSNR (dB)": [f"{p:.2f}" for p in method_results['psnr']],
                    "SSIM": [f"{s:.4f}" for s in method_results['ssim']],
                    "Subjective Score": method_results['subjective_score'],
                    "Quality Category": method_results['quality_category'],
                    "Execution Time (ms)": [f"{t:.2f}" for t in method_results['execution_time']]
                }
                
                # Display as a table
                st.table(data)

# --- Footer / Info ---
st.markdown("---")
st.markdown("""
* **DCT (JPEG-like):** Compresses images in 8x8 blocks using the Discrete Cosine Transform and quantization. Prone to blocking artifacts at low quality.
* **Wavelet (JPEG2000-like):** Compresses images using multi-resolution wavelet decomposition and coefficient thresholding. Often yields better quality at high compression but can cause blurring. Less prone to blocking.
* **Compression Ratio:** Calculated as *Total Coefficients / Non-Zero Coefficients*. This is a simplified metric; real JPEG/JPEG2000 use entropy coding (e.g., Huffman, Arithmetic) for further compression.
* **PSNR/SSIM:** Standard metrics for objective image quality assessment compared to the original.
* **Subjective Quality Score:** A perceptual quality assessment on a 1-5 scale based on PSNR and SSIM thresholds.
""")