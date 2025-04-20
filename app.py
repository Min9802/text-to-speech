# This app is adopted from https://github.com/coqui-ai/TTS/blob/dev/TTS/demos/xtts_ft_demo/xtts_demo.py
# With some modifications to fit the viXTTS model
import argparse
import hashlib
import logging
import os
import string
import subprocess
import sys
import tempfile
from datetime import datetime

import gradio as gr
import soundfile as sf
import torch
import torchaudio
from huggingface_hub import hf_hub_download, snapshot_download
from underthesea import sent_tokenize
from unidecode import unidecode
from vinorm import TTSnorm

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

XTTS_MODEL = None
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "model")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
FILTER_SUFFIX = "_DeepFilterNet3.wav"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def auto_load_model(checkpoint_dir="model/", repo_id="capleaf/viXTTS", use_deepspeed=False):
    """Automatically load model at startup if files exist"""
    global XTTS_MODEL
    clear_gpu_cache()
    os.makedirs(checkpoint_dir, exist_ok=True)

    required_files = ["model.pth", "config.json", "vocab.json", "speakers_xtts.pth"]
    files_in_dir = os.listdir(checkpoint_dir)
    
    # If all required files exist, load the model
    if all(file in files_in_dir for file in required_files):
        print("Model files found. Auto-loading model...")
        xtts_config = os.path.join(checkpoint_dir, "config.json")
        config = XttsConfig()
        config.load_json(xtts_config)
        XTTS_MODEL = Xtts.init_from_config(config)
        
        # Check if CUDA is available and disable DeepSpeed on CPU-only systems
        if not torch.cuda.is_available() and use_deepspeed:
            use_deepspeed = False
            message = "DeepSpeed requires CUDA. Falling back to CPU mode..."
            print(message)
        
        try:
            XTTS_MODEL.load_checkpoint(
                config, checkpoint_dir=checkpoint_dir, use_deepspeed=use_deepspeed
            )
            if torch.cuda.is_available():
                XTTS_MODEL.cuda()
                
            message = "Model loaded automatically on startup!"
            print(message)
            return True, message
        except Exception as e:
            # If loading with DeepSpeed fails, try again without it
            if use_deepspeed:
                message = f"Error loading with DeepSpeed: {str(e)}. Trying without DeepSpeed..."
                print(message)
                clear_gpu_cache()
                XTTS_MODEL = Xtts.init_from_config(config)
                XTTS_MODEL.load_checkpoint(
                    config, checkpoint_dir=checkpoint_dir, use_deepspeed=False
                )
                if torch.cuda.is_available():
                    XTTS_MODEL.cuda()
                    
                message = "Model loaded automatically without DeepSpeed!"
                print(message)
                return True, message
            else:
                message = f"Error auto-loading model: {str(e)}"
                print(message)
                return False, message
    else:
        message = "Not all model files present. Model will need to be loaded manually or downloaded."
        print(message)
    
    return False, message


def load_model(checkpoint_dir="model/", repo_id="capleaf/viXTTS", use_deepspeed=False, progress=gr.Progress()):
    global XTTS_MODEL
    clear_gpu_cache()
    os.makedirs(checkpoint_dir, exist_ok=True)

    required_files = ["model.pth", "config.json", "vocab.json", "speakers_xtts.pth"]
    files_in_dir = os.listdir(checkpoint_dir)
    if not all(file in files_in_dir for file in required_files):
        progress(0, desc="Checking model files")
        yield f"Missing model files! Downloading from {repo_id}..."
        progress(0.2, desc=f"Downloading model files from {repo_id}")
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=checkpoint_dir,
        )
        progress(0.5, desc="Downloading speaker embeddings")
        hf_hub_download(
            repo_id="coqui/XTTS-v2",
            filename="speakers_xtts.pth",
            local_dir=checkpoint_dir,
        )
        progress(0.6, desc="Model download finished")
        yield f"Model download finished..."

    progress(0.7, desc="Loading model configuration")
    xtts_config = os.path.join(checkpoint_dir, "config.json")
    config = XttsConfig()
    config.load_json(xtts_config)
    XTTS_MODEL = Xtts.init_from_config(config)
    yield "Loading model..."
    
    # Check if CUDA is available and disable DeepSpeed on CPU-only systems
    if not torch.cuda.is_available() and use_deepspeed:
        use_deepspeed = False
        progress(0.75, desc="DeepSpeed requires CUDA. Falling back to CPU mode...")
        yield "DeepSpeed requires CUDA. Falling back to CPU mode..."
    
    try:
        progress(0.8, desc="Loading model weights")
        XTTS_MODEL.load_checkpoint(
            config, checkpoint_dir=checkpoint_dir, use_deepspeed=use_deepspeed
        )
        if torch.cuda.is_available():
            progress(0.9, desc="Moving model to GPU")
            XTTS_MODEL.cuda()
            
        progress(1.0, desc="Model loaded successfully")
        print("Model Loaded!")
        yield "Model Loaded!"
    except Exception as e:
        # If loading with DeepSpeed fails, try again without it
        if use_deepspeed:
            progress(0.8, desc=f"Error loading with DeepSpeed: {str(e)}. Trying without DeepSpeed...")
            yield f"Error loading with DeepSpeed: {str(e)}. Trying without DeepSpeed..."
            clear_gpu_cache()
            XTTS_MODEL = Xtts.init_from_config(config)
            progress(0.9, desc="Loading model without DeepSpeed")
            XTTS_MODEL.load_checkpoint(
                config, checkpoint_dir=checkpoint_dir, use_deepspeed=False
            )
            if torch.cuda.is_available():
                XTTS_MODEL.cuda()
                
            progress(1.0, desc="Model loaded successfully without DeepSpeed")
            print("Model Loaded without DeepSpeed!")
            yield "Model Loaded without DeepSpeed!"
        else:
            progress(1.0, desc=f"Error loading model: {str(e)}")
            yield f"Error loading model: {str(e)}"
            raise


# Define dictionaries to store cached results
cache_queue = []
speaker_audio_cache = {}
filter_cache = {}
conditioning_latents_cache = {}


def invalidate_cache(cache_limit=50):
    """Invalidate the cache for the oldest key"""
    if len(cache_queue) > cache_limit:
        key_to_remove = cache_queue.pop(0)
        print("Invalidating cache", key_to_remove)
        if os.path.exists(key_to_remove):
            os.remove(key_to_remove)
        if os.path.exists(key_to_remove.replace(".wav", "_DeepFilterNet3.wav")):
            os.remove(key_to_remove.replace(".wav", "_DeepFilterNet3.wav"))
        if key_to_remove in filter_cache:
            del filter_cache[key_to_remove]
        if key_to_remove in conditioning_latents_cache:
            del conditioning_latents_cache[key_to_remove]


def generate_hash(data):
    hash_object = hashlib.md5()
    hash_object.update(data)
    return hash_object.hexdigest()


def get_file_name(text, max_char=50):
    filename = text[:max_char]
    filename = filename.lower()
    filename = filename.replace(" ", "_")
    filename = filename.translate(
        str.maketrans("", "", string.punctuation.replace("_", ""))
    )
    filename = unidecode(filename)
    current_datetime = datetime.now().strftime("%m%d%H%M%S")
    filename = f"{current_datetime}_{filename}"
    return filename


def normalize_vietnamese_text(text):
    try:
        text = (
            TTSnorm(text, unknown=False, lower=False, rule=True)
            .replace("..", ".")
            .replace("!.", "!")
            .replace("?.", "?")
            .replace(" .", ".")
            .replace(" ,", ",")
            .replace('"', "")
            .replace("'", "")
            .replace("AI", "Ây Ai")
            .replace("A.I", "Ây Ai")
        )
    except UnicodeEncodeError:
        # If TTSnorm fails due to encoding issues, perform basic normalization without TTSnorm
        print("Warning: TTSnorm encoding error with Vietnamese text, using basic normalization instead")
        text = text.replace("..", ".").replace("!.", "!").replace("?.", "?").replace(" .", ".").replace(" ,", ",").replace('"', "").replace("'", "").replace("AI", "Ây Ai").replace("A.I", "Ây Ai")
    return text


def calculate_keep_len(text, lang):
    """Simple hack for short sentences"""
    if lang in ["ja", "zh-cn"]:
        return -1

    word_count = len(text.split())
    num_punct = text.count(".") + text.count("!") + text.count("?") + text.count(",")

    if word_count < 5:
        return 15000 * word_count + 2000 * num_punct
    elif word_count < 10:
        return 13000 * word_count + 2000 * num_punct
    return -1


def run_tts(lang, tts_text, speaker_audio_file, use_deepfilter, normalize_text, progress=gr.Progress()):
    global filter_cache, conditioning_latents_cache, cache_queue

    if XTTS_MODEL is None:
        return "You need to run the previous step to load the model !!", None

    if not speaker_audio_file:
        return "You need to provide reference audio!!!", None
        
    progress(0, desc="Starting TTS process")

    # Use the file name as the key, since it's suppose to be unique 💀
    speaker_audio_key = speaker_audio_file
    if not speaker_audio_key in cache_queue:
        cache_queue.append(speaker_audio_key)
        invalidate_cache()

    # Check if filtered reference is cached
    if use_deepfilter and speaker_audio_key in filter_cache:
        progress(0.1, desc="Using cached filtered audio...")
        print("Using filter cache...")
        speaker_audio_file = filter_cache[speaker_audio_key]
    elif use_deepfilter:
        progress(0.1, desc="Filtering reference audio...")
        print("Running filter...")
        subprocess.run(
            [
                "deepFilter",
                speaker_audio_file,
                "-o",
                os.path.dirname(speaker_audio_file),
            ]
        )
        filter_cache[speaker_audio_key] = speaker_audio_file.replace(
            ".wav", FILTER_SUFFIX
        )
        speaker_audio_file = filter_cache[speaker_audio_key]

    # Check if conditioning latents are cached
    cache_key = (
        speaker_audio_key,
        XTTS_MODEL.config.gpt_cond_len,
        XTTS_MODEL.config.max_ref_len,
        XTTS_MODEL.config.sound_norm_refs,
    )
    if cache_key in conditioning_latents_cache:
        progress(0.2, desc="Using cached conditioning latents...")
        print("Using conditioning latents cache...")
        gpt_cond_latent, speaker_embedding = conditioning_latents_cache[cache_key]
    else:
        progress(0.2, desc="Computing conditioning latents...")
        print("Computing conditioning latents...")
        gpt_cond_latent, speaker_embedding = XTTS_MODEL.get_conditioning_latents(
            audio_path=speaker_audio_file,
            gpt_cond_len=XTTS_MODEL.config.gpt_cond_len,
            max_ref_length=XTTS_MODEL.config.max_ref_len,
            sound_norm_refs=XTTS_MODEL.config.sound_norm_refs,
        )
        conditioning_latents_cache[cache_key] = (gpt_cond_latent, speaker_embedding)

    if normalize_text and lang == "vi":
        progress(0.3, desc="Normalizing Vietnamese text...")
        tts_text = normalize_vietnamese_text(tts_text)

    # Split text by sentence
    if lang in ["ja", "zh-cn"]:
        sentences = tts_text.split("。")
    else:
        sentences = sent_tokenize(tts_text)

    from pprint import pprint
    pprint(sentences)
    
    progress(0.4, desc="Generating speech...")

    wav_chunks = []
    for i, sentence in enumerate(sentences):
        if sentence.strip() == "":
            continue
        
        progress_val = 0.4 + (0.5 * i / max(len(sentences), 1))
        progress(progress_val, desc=f"Processing sentence {i+1}/{len(sentences)}...")
        
        wav_chunk = XTTS_MODEL.inference(
            text=sentence,
            language=lang,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            # The following values are carefully chosen for viXTTS
            temperature=0.3,
            length_penalty=1.0,
            repetition_penalty=10.0,
            top_k=30,
            top_p=0.85,
            enable_text_splitting=True,
        )

        keep_len = calculate_keep_len(sentence, lang)
        wav_chunk["wav"] = wav_chunk["wav"][:keep_len]

        wav_chunks.append(torch.tensor(wav_chunk["wav"]))

    progress(0.9, desc="Finalizing audio...")
    out_wav = torch.cat(wav_chunks, dim=0).unsqueeze(0)
    gr_audio_id = os.path.basename(os.path.dirname(speaker_audio_file))
    out_path = os.path.join(OUTPUT_DIR, f"{get_file_name(tts_text)}_{gr_audio_id}.wav")
    print("Saving output to ", out_path)
    torchaudio.save(out_path, out_wav, 24000)
    
    progress(1.0, desc="Done!")
    # Return the path to the saved audio file instead of numpy array
    return "Speech generated !", out_path


# Define a logger to redirect
class Logger:
    def __init__(self, filename="log.out"):
        self.log_file = filename
        self.terminal = sys.stdout
        # Open log file with UTF-8 encoding to support Vietnamese characters
        self.log = open(self.log_file, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return False


# Redirect stdout and stderr to a file
sys.stdout = Logger()
sys.stderr = sys.stdout


logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def read_logs():
    sys.stdout.flush()
    with open(sys.stdout.log_file, "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""viXTTS inference app\n\n""",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Port to run the gradio. Default: 5003",
        default=5003,
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        help="Path to the checkpoint directory. This directory must contain 04 files: model.pth, config.json, vocab.json and speakers_xtts.pth",
        default=None,
    )

    parser.add_argument(
        "--reference_audio",
        type=str,
        help="Path to the reference audio file.",
        default=None,
    )

    args = parser.parse_args()
    if args.model_dir:
        MODEL_DIR = os.path.abspath(args.model_dir)

    REFERENCE_AUDIO = os.path.join(SCRIPT_DIR, "assets", "vixtts_sample_female.wav")
    if args.reference_audio:
        REFERENCE_AUDIO = os.abspath(args.reference_audio)
    
    # Get available reference audio samples
    SAMPLES_DIR = os.path.join(MODEL_DIR, "samples")
    sample_files = []
    if os.path.exists(SAMPLES_DIR):
        sample_files = [os.path.join(SAMPLES_DIR, f) for f in os.listdir(SAMPLES_DIR) if f.endswith('.wav')]
        sample_files.sort()

    # Update auto_load_model call to capture and use the returned status and message
    success, load_message = auto_load_model(checkpoint_dir=MODEL_DIR, repo_id="capleaf/viXTTS", use_deepspeed=True)
    css="""
    .textarea_resize TextArea {
        resize: vertical;
    }
    """
    with gr.Blocks(
        theme=gr.themes.Origin(
            spacing_size=gr.themes.sizes.spacing_sm, 
            radius_size=gr.themes.sizes.radius_lg
        ), 
        title="Text to Speech", 
        css=css
    ) as app:
        intro = """
        # Text to Speech
        Visit viXTTS on HuggingFace: [viXTTS](https://huggingface.co/capleaf/viXTTS)
        """
        gr.Markdown(intro)
        with gr.Row():
            with gr.Column() as col1:
                repo_id = gr.Textbox(
                    label="HuggingFace Repo ID",
                    value="capleaf/viXTTS",
                )
                checkpoint_dir = gr.Textbox(
                    label="viXTTS model directory",
                    value=MODEL_DIR,
                )

                use_deepspeed = gr.Checkbox(
                    value=True, label="Use DeepSpeed for faster inference"
                )

                progress_load = gr.Label(label="Progress:", value="Model Loaded!" if success else load_message)
                load_btn = gr.Button(
                    value="Load viXTTS model", variant="primary"
                )

            with gr.Column() as col2:
                tts_text = gr.TextArea(
                    label="Text Prompt (Văn bản cần đọc).",
                    info="Mỗi câu nên từ 10 từ trở lên.",
                    value="Xin chào, tôi là một công cụ chuyển đổi văn bản thành giọng nói tiếng Việt.",
                    placeholder="Nhập văn bản cần đọc ở đây...",
                    show_copy_button=True,
                    elem_classes="textarea_resize"
                )
                
                # Add dropdown to select reference audio from samples directory
                if sample_files:
                    # Create display names without file extensions
                    sample_names = [os.path.basename(f) for f in sample_files]
                    sample_display_names = [os.path.splitext(name)[0] for name in sample_names]
                    
                    # Create a mapping from display names to actual file names
                    sample_name_mapping = dict(zip(sample_display_names, sample_names))
                    
                    reference_sample_selector = gr.Dropdown(
                        label="Select sample reference voice (Chọn giọng mẫu)",
                        choices=sample_display_names,
                        value=sample_display_names[0] if sample_display_names else None,
                        info="Select a reference voice from model/samples"
                    )

                tts_language = gr.Dropdown(
                    label="Language (ngôn ngữ)",
                    value="vi",
                    choices=[
                        "vi",
                        "en",
                        "es",
                        "fr",
                        "de",
                        "it",
                        "pt",
                        "pl",
                        "tr",
                        "ru",
                        "nl",
                        "cs",
                        "ar",
                        "zh",
                        "hu",
                        "ko",
                        "ja",
                    ],
                )

                use_filter = gr.Checkbox(
                    label="Denoise Reference Audio",
                    value=True,
                )

                normalize_text = gr.Checkbox(
                    label="Normalize Input Text",
                    value=True,
                )

                speaker_reference_audio = gr.Audio(
                    label="Reference Audio (Giọng mẫu)",
                    value=REFERENCE_AUDIO,
                    type="filepath",
                )
                
                tts_btn = gr.Button(value="Generate & Speak", variant="primary")

            with gr.Column() as col3:
                progress_gen = gr.Label(label="Progress:")
                tts_output_audio = gr.Audio(label="Generated Audio", type="filepath", autoplay=True)

        load_btn.click(
            fn=load_model,
            inputs=[checkpoint_dir, repo_id, use_deepspeed],
            outputs=[progress_load],
        )

        tts_btn.click(
            fn=run_tts,
            inputs=[
                tts_language,
                tts_text,
                speaker_reference_audio,
                use_filter,
                normalize_text,
            ],
            outputs=[progress_gen, tts_output_audio],
        )

        # Add event handler for sample reference selector
        if sample_files:
            def update_reference_audio(display_name):
                if not display_name:
                    return REFERENCE_AUDIO
                # Get the actual filename with extension using the mapping
                file_name = sample_name_mapping[display_name]
                sample_path = os.path.join(SAMPLES_DIR, file_name)
                return sample_path
                
            reference_sample_selector.change(
                fn=update_reference_audio,
                inputs=[reference_sample_selector],
                outputs=[speaker_reference_audio],
            )

    app.launch(favicon_path="assets/favicon.ico",share=False, debug=False, server_port=args.port, server_name="0.0.0.0", show_api=True)
