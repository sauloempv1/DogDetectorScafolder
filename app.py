import os
import time
import tempfile
import io
import numpy as np
import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import torch

st.set_page_config(page_title="Dog Detector (YOLOv8)", layout="wide")

# ---------- Display config ----------
# Maximum display width in pixels for images/GIFs/videos shown in the UI.
# Adjust if needed to fit your page/layout.
DISPLAY_MAX_WIDTH = 800

# ---------- Sidebar ----------
st.sidebar.header("⚙️ Configuração")
default_weights = "notebooks/runs_dog/det_dog_tuned (Best)/weights/best.pt"
weights_path = st.sidebar.text_input("Caminho do modelo (.pt)", value=default_weights)
conf = st.sidebar.slider("Confidence", 0.05, 0.85, 0.25, 0.01)
iou  = st.sidebar.slider("IoU (NMS)", 0.2, 0.9, 0.5, 0.01)
device_cpu = st.sidebar.checkbox("Forçar CPU (desmarca p/ usar GPU se houver)", value=False)
imgsz = st.sidebar.selectbox("Tamanho de entrada", [416, 512, 640, 800], index=2)
show_labels = st.sidebar.checkbox("Mostrar rótulos", True)
show_conf = st.sidebar.checkbox("Mostrar confiança", True)

# ---------- Model load ----------
@st.cache_resource(show_spinner=True)
def load_model(path):
    return YOLO(path)

if not os.path.isfile(weights_path):
    st.error(f"Arquivo de pesos não encontrado: {weights_path}")
    st.stop()

model = load_model(weights_path)
st.success(f"Modelo carregado: {weights_path}")

def get_device():
    if device_cpu:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"

st.sidebar.write("Usando dispositivo:", get_device())

# ---------- Helpers ----------
def resize_for_display(img_bgr: np.ndarray, max_width: int = DISPLAY_MAX_WIDTH) -> np.ndarray:
    """Resize an image to fit max_width (preserving aspect ratio). Returns BGR np.ndarray."""
    if img_bgr is None:
        return None
    h, w = img_bgr.shape[:2]
    if w <= max_width or max_width <= 0:
        return img_bgr
    scale = max_width / w
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

def annotate_and_count(result):
    """Use result.plot() to draw boxes and count class 0 (dog)."""
    try:
        annotated = result.plot()
    except Exception:
        annotated = result.orig_img if hasattr(result, "orig_img") else None
    if result.boxes is None:
        return annotated, 0
    cls = result.boxes.cls.detach().cpu().numpy() if getattr(result.boxes, "cls", None) is not None else np.array([])
    dog_count = int((cls == 0).sum())
    return annotated, dog_count

def run_inference_on_image(img_input):
    """Accept numpy BGR, bytes or file-like and return (annotated_bgr, count)."""
    img_bgr = None
    if isinstance(img_input, np.ndarray):
        img_bgr = img_input
    elif isinstance(img_input, (bytes, bytearray)):
        arr = np.frombuffer(img_input, np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    elif hasattr(img_input, "read"):
        data = img_input.read()
        arr = np.frombuffer(data, np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    else:
        raise ValueError("Entrada inválida para run_inference_on_image (esperado ndarray/bytes/file-like).")

    if img_bgr is None:
        return None, 0

    dev = get_device()
    results = model.predict(
        source=img_bgr,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=dev,
        half=(dev != "cpu"),
        verbose=False,
        stream=False,
        show_labels=show_labels,
        show_conf=show_conf
    )
    r = results[0]
    annotated, count = annotate_and_count(r)
    return annotated, count

# ---------- UI ----------
tab_img, tab_video, tab_cam = st.tabs(["🖼️ Imagem", "🎬 Vídeo", "📷 Webcam"])

# ---------- Image tab ----------
with tab_img:
    st.subheader("🖼️ Upload de Imagem")
    up = st.file_uploader("Envie uma imagem", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if up is not None:
        up_bytes = up.getvalue()
        img_bgr = cv2.imdecode(np.frombuffer(up_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img_bgr is None:
            st.error("Falha ao ler a imagem.")
        else:
            with st.spinner("Processando..."):
                annotated, count = run_inference_on_image(img_bgr)
            if annotated is None:
                st.warning("Nenhuma anotação gerada, mostrando imagem original.")
                annotated = img_bgr

            # Display image resized to fit DISPLAY_MAX_WIDTH
            display_img = resize_for_display(annotated, DISPLAY_MAX_WIDTH)
            col1, col2 = st.columns([3,1])
            with col1:
                try:
                    st.image(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB), channels="RGB", width=display_img.shape[1])
                except Exception:
                    st.image(up_bytes, width=min(DISPLAY_MAX_WIDTH, 700))
            with col2:
                st.metric("🐶 Dogs no frame", count)

            out_default_name = f"{os.path.splitext(up.name)[0]}_annotated"
            out_name = st.text_input("Nome do arquivo (sem extensão) para download", value=out_default_name, key="img_out_name")

            is_success, buf = cv2.imencode(".png", annotated)
            if is_success:
                st.download_button(
                    label="⬇️ Baixar imagem anotada (PNG)",
                    data=buf.tobytes(),
                    file_name=f"{out_name}.png",
                    mime="image/png"
                )
            else:
                st.warning("Falha ao gerar PNG para download.")

# ---------- Video tab ----------
with tab_video:
    st.subheader("🎬 Vídeo (arquivo)")
    vup = st.file_uploader("Envie um vídeo (mp4/mov/avi/mkv)", type=["mp4","mov","avi","mkv"])
    if vup is not None:
        out_default_gif = f"{os.path.splitext(vup.name)[0]}_annotated"
        out_name_video = st.text_input("Nome do GIF para download (sem extensão)", value=out_default_gif, key="gif_name_input")

        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(vup.name)[1])
        tfile.write(vup.read()); tfile.flush()
        cap = cv2.VideoCapture(tfile.name)
        if not cap.isOpened():
            st.error("Não consegui abrir o vídeo.")
        else:
            stframe = st.empty()
            cnt_placeholder = st.empty()
            fps_placeholder = st.empty()

            st.info("Vídeo carregado. Configure o GIF e clique em 'Processar e gerar GIF anotado'.")

            max_frames = st.slider("Número máximo de frames a processar (amostragem para GIF)", 10, 1000, 200)
            desired_fps = st.slider("FPS do GIF resultante", 1, 30, 10)
            max_width = st.number_input("Largura máxima do GIF (px) — 0 = usar original", min_value=0, max_value=1920, value=640, step=16)
            # We will cap max_width to DISPLAY_MAX_WIDTH to avoid UI overflow
            max_width = min(max_width, DISPLAY_MAX_WIDTH) if max_width > 0 else 0

            generate_gif_btn = st.button("Processar e gerar GIF anotado")
            create_video_btn = st.checkbox("Também gerar vídeo MP4 anotado (pode ser lento)")

            if generate_gif_btn:
                frames = []
                annotated_frames_for_video = []
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                step = max(1, frame_count // max_frames) if frame_count > 0 else 1
                idx = 0
                processed = 0
                pbar = st.progress(0)
                dev = get_device()
                prev = time.time()

                try:
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        if idx % step == 0:
                            results = model.predict(
                                source=frame,
                                imgsz=imgsz,
                                conf=conf,
                                iou=iou,
                                device=dev,
                                half=(dev != "cpu"),
                                verbose=False,
                                stream=False,
                                show_labels=show_labels,
                                show_conf=show_conf
                            )
                            r = results[0]
                            annotated, count = annotate_and_count(r)
                            if annotated is None:
                                annotated = frame
                            # resize to user's max_width but don't exceed DISPLAY_MAX_WIDTH
                            if max_width and annotated.shape[1] > max_width:
                                scale = max_width / annotated.shape[1]
                                annotated = cv2.resize(annotated, (int(annotated.shape[1] * scale), int(annotated.shape[0] * scale)), interpolation=cv2.INTER_AREA)

                            if annotated.shape[1] > DISPLAY_MAX_WIDTH:
                                annotated = resize_for_display(annotated, DISPLAY_MAX_WIDTH)

                            now = time.time()
                            fps = 1.0 / max(1e-6, (now - prev))
                            prev = now
                            try:
                                cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
                            except Exception:
                                pass

                            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                            frames.append(annotated_rgb)
                            if create_video_btn:
                                annotated_frames_for_video.append(annotated)
                            processed += 1
                            pbar.progress(min(100, int(processed * 100 / max_frames)))
                            stframe.image(annotated_rgb, channels="RGB", width=annotated_rgb.shape[1])
                            cnt_placeholder.metric("🐶 Dogs no frame", count)
                            fps_placeholder.metric("⚡ FPS (aprox.)", f"{fps:.1f}")
                            if processed >= max_frames:
                                break
                        idx += 1
                finally:
                    cap.release()
                    try:
                        os.remove(tfile.name)
                    except Exception:
                        pass

                if len(frames) == 0:
                    st.error("Nenhum frame processado. Verifique o arquivo ou reduza as configurações.")
                else:
                    pil_frames = [Image.fromarray(f) for f in frames]
                    gif_buffer = io.BytesIO()
                    pil_frames[0].save(
                        gif_buffer, format="GIF", save_all=True, append_images=pil_frames[1:], loop=0,
                        duration=max(20, int(1000/desired_fps))
                    )
                    gif_buffer.seek(0)

                    # Show GIF scaled to DISPLAY_MAX_WIDTH
                    st.image(gif_buffer.getvalue(), caption="GIF anotado", width=min(DISPLAY_MAX_WIDTH, pil_frames[0].width))

                    # download direct (in-memory)
                    if gif_buffer.getbuffer().nbytes > 0:
                        st.download_button(
                            label="⬇️ Baixar GIF anotado (direto)",
                            data=gif_buffer.getvalue(),
                            file_name=f"{out_name_video}.gif",
                            mime="image/gif"
                        )
                    else:
                        st.warning("GIF vazio; não foi possível habilitar o download.")

                    # save to server and provide download (fallback)
                    out_dir = os.path.join("outputs", "gifs")
                    os.makedirs(out_dir, exist_ok=True)
                    gif_server_path = os.path.join(out_dir, f"{out_name_video}.gif")
                    with open(gif_server_path, "wb") as f:
                        f.write(gif_buffer.getvalue())
                    st.success(f"GIF salvo no servidor: {gif_server_path}")

                    try:
                        with open(gif_server_path, "rb") as fh:
                            st.download_button(
                                label="⬇️ Baixar GIF anotado (do servidor)",
                                data=fh.read(),
                                file_name=os.path.basename(gif_server_path),
                                mime="image/gif"
                            )
                    except Exception as e:
                        st.warning(f"Erro ao ler o GIF salvo: {e}")

                    if create_video_btn and annotated_frames_for_video:
                        mp4_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                        h, w = annotated_frames_for_video[0].shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        out_vid = cv2.VideoWriter(mp4_temp.name, fourcc, desired_fps, (w, h))
                        for f in annotated_frames_for_video:
                            out_vid.write(f)
                        out_vid.release()
                        with open(mp4_temp.name, "rb") as fh:
                            mp4_data = fh.read()
                        st.download_button(
                            label="⬇️ Baixar vídeo anotado (MP4)",
                            data=mp4_data,
                            file_name=f"{os.path.splitext(vup.name)[0]}_annotated.mp4",
                            mime="video/mp4"
                        )
                        out_dir_vid = os.path.join("outputs", "videos")
                        os.makedirs(out_dir_vid, exist_ok=True)
                        mp4_server_path = os.path.join(out_dir_vid, f"{os.path.splitext(vup.name)[0]}_annotated.mp4")
                        with open(mp4_server_path, "wb") as f:
                            f.write(mp4_data)
                        st.success(f"MP4 salvo no servidor: {mp4_server_path}")
                        try:
                            os.remove(mp4_temp.name)
                        except Exception:
                            pass

            st.success("Pronto. Ajuste parâmetros e gere novamente se quiser.")

# ---------- Webcam tab ----------
with tab_cam:
    st.subheader("📷 Webcam (tempo real)")
    st.caption("⚠️ No WSL2 a webcam do Windows não aparece por padrão. Para webcam, rode o app no Windows nativo ou use `streamlit-webrtc`.")
    run = st.checkbox("Iniciar webcam", value=False)
    cam_index = st.number_input("Índice da câmera (0 = padrão)", min_value=0, max_value=4, value=0, step=1)
    if run:
        dev = get_device()
        cap = cv2.VideoCapture(int(cam_index))
        if not cap.isOpened():
            st.error("Não consegui abrir a webcam.")
        else:
            stframe = st.empty(); cnt_placeholder = st.empty(); fps_placeholder = st.empty()
            prev = time.time()
            stop = st.button("Parar")
            while run and not stop:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Sem frame da câmera.")
                    break
                results = model.predict(
                    source=frame, imgsz=imgsz, conf=conf, iou=iou,
                    device=dev, half=(dev != "cpu"), verbose=False, stream=False,
                    show_labels=show_labels, show_conf=show_conf
                )
                r = results[0]
                annotated, count = annotate_and_count(r)
                if annotated is None:
                    annotated = frame
                # keep webcam small enough to not break layout
                display_frame = resize_for_display(annotated, DISPLAY_MAX_WIDTH)
                now = time.time()
                fps = 1.0 / max(1e-6, (now - prev))
                prev = now
                try:
                    cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
                except Exception:
                    pass
                stframe.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), channels="RGB", width=display_frame.shape[1])
                cnt_placeholder.metric("🐶 Dogs no frame", count)
                fps_placeholder.metric("⚡ FPS", f"{fps:.1f}")
            cap.release()
            st.info("Webcam parada.")
# end of file
