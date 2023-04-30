import argparse
import string
import numpy as np
import librosa
from bark import SAMPLE_RATE, generate_audio, preload_models
import os
import datetime
import soundfile as sf
import re
from collections import defaultdict, namedtuple
from scipy import signal
from scipy.io import wavfile
import random

FileData = namedtuple("FileData", ["filename", "name", "desc"])
sample_rate = 24000 


SUPPORTED_LANGS = [
    ("English", "en"),
    ("German", "de"),
    ("Spanish", "es"),
    ("French", "fr"),
    ("Hindi", "hi"),
    ("Italian", "it"),
    ("Japanese", "ja"),
    ("Korean", "ko"),
    ("Polish", "pl"),
    ("Portuguese", "pt"),
    ("Russian", "ru"),
    ("Turkish", "tr"),
    ("Chinese", "zh"),
]

def random_name(length=10):
    # Generate a random string of the specified length
    result = ''.join(random.choices(string.ascii_letters, k=length))
    return result + '.wav'

def read_npz_files(directory):
    return [f for f in os.listdir(directory) if f.endswith(".npz")]

def extract_name_and_desc(filepath):
    with np.load(filepath) as data:
        name = data.get('name', '')
        desc = data.get('desc', '')
        return name, desc

def categorize_files(files, directory):
    categorized_files = defaultdict(list)
    lang_dict = {code: lang for lang, code in SUPPORTED_LANGS}
    
    for file in files:
        name, desc = extract_name_and_desc(os.path.join(directory, file))
        match = re.match(r"([a-z]{2}|\w+)_", file)
        if match:
            prefix = match.group(1)
            if prefix in lang_dict:
                categorized_files[lang_dict[prefix]].append(FileData(file, name, desc))
            else:
                categorized_files[prefix.capitalize()].append(FileData(file, name, desc))
        else:
            categorized_files["Other"].append(FileData(file, name, desc))

    return categorized_files

# this is a mess but whatever
def print_speakers_list(categorized_files):
    print("Available history prompts:")
    for category, files in categorized_files.items():
        sorted_files = sorted(files, key=lambda x: (re.search(r"_\w+(_\d+)?\.npz$", x.filename) and re.search(r"_\w+(_\d+)?\.npz$", x.filename).group()[:-4], x.filename))
        print(f"\n  {category}:")
        for file_data in sorted_files:
            name_display = f'  "{file_data.name}"' if file_data.name else ''
            desc_display = f'{file_data.desc}' if file_data.desc else ''
            print(f"    {file_data.filename[:-4]} {name_display} {desc_display}")

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
history_prompt_dir = os.path.join(CUR_PATH, "bark", "assets", "prompts")

npz_files = read_npz_files(history_prompt_dir)
categorized_files = categorize_files(npz_files, history_prompt_dir)
ALLOWED_PROMPTS = {file[:-4] for file in npz_files}



def estimate_spoken_time(text, wpm=150, time_limit=14):
    # Remove text within square brackets
    text_without_brackets = re.sub(r'\[.*?\]', '', text)
    
    words = text_without_brackets.split()
    word_count = len(words)
    time_in_seconds = (word_count / wpm) * 60
    
    if time_in_seconds > time_limit:
        return True, time_in_seconds
    else:
        return False, time_in_seconds


def save_npz_file(filepath, x_semantic_continued, coarse_prompt, fine_prompt, output_dir=None):
    np.savez(filepath, semantic_prompt=x_semantic_continued, coarse_prompt=coarse_prompt, fine_prompt=fine_prompt)
    print(f"speaker file for this clip saved to {filepath}")

def split_text(text, split_words=0, split_lines=0):
    if split_words > 0:
        words = text.split()
        chunks = [' '.join(words[i:i + split_words]) for i in range(0, len(words), split_words)]
    elif split_lines > 0:
        lines = [line for line in text.split('\n') if line.strip()]
        chunks = ['\n'.join(lines[i:i + split_lines]) for i in range(0, len(lines), split_lines)]
    else:
        chunks = [text]
    return chunks

def save_audio_to_file(filepath, audio_array, sample_rate=24000, format='WAV', subtype='PCM_16', output_dir=None):
    sf.write(filepath, audio_array, sample_rate, format=format, subtype=subtype)
    print(f"Saved audio to {filepath}")
    
def zero_crossing_rate(audio, frame_length=1024, hop_length=512):
    zero_crossings = np.sum(np.abs(np.diff(np.sign(audio))) / 2)
    total_frames = 1 + int((len(audio) - frame_length) / hop_length)
    return zero_crossings / total_frames
    
def compute_spectral_contrast(audio_data, sample_rate, n_bands=6, fmin=200.0):
    spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sample_rate, n_bands=n_bands, fmin=fmin)
    return np.mean(spectral_contrast)
    
def compute_average_bass_energy(audio_data, sample_rate, max_bass_freq=250):
    stft = librosa.stft(audio_data)
    power_spectrogram = np.abs(stft)**2
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=stft.shape[0])
    bass_mask = frequencies <= max_bass_freq
    bass_energy = power_spectrogram[np.ix_(bass_mask, np.arange(power_spectrogram.shape[1]))].mean()
    return bass_energy
    
def generate_audio_with_zcr_check(text, history_prompt, sample_rate, zcr_threshold=80, spectral_threshold=23, bass_energy_threshold=150, max_attempts=10, **kwargs):
    attempts = 0
    output_directory = "output_array"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    base = kwargs.pop("base", None)

    while attempts < max_attempts:
        if attempts > 0 and base is not None:
            # Reset the base model token
            print(f"Reset the base model token Regenerating...")
            base = None
        
        audio_array, x = generate_audio(text, history_prompt, base=base, **kwargs)
        zcr = zero_crossing_rate(audio_array)
        spectral_contrast = compute_spectral_contrast(audio_array, sample_rate)
        bass_energy = compute_average_bass_energy(audio_array, sample_rate)
        print(f"Attempt {attempts + 1}: ZCR = {zcr}, Spectral Contrast = {spectral_contrast:.2f}, Bass Energy = {bass_energy:.2f}")
      
        # Save the audio array to the output_array directory with a random name for debugging 
        #output_file = os.path.join(output_directory, f"audio_{zcr:.2f}_sc{spectral_contrast:.2f}_be{bass_energy:.2f}.wav")
        #wavfile.write(output_file, sample_rate, audio_array)
        #print(f"Saved audio array to {output_file}")
        
        if zcr < zcr_threshold and spectral_contrast < spectral_threshold and bass_energy < bass_energy_threshold:
            print(f"Audio passed ZCR, Spectral Contrast, and Bass Energy thresholds. No need to regenerate.")
            break
        else:
            print(f"Audio failed ZCR, Spectral Contrast, and/or Bass Energy thresholds. Regenerating...")

        attempts += 1

    if attempts == max_attempts:
        print("Reached maximum attempts. Returning the last generated audio.")
    
    return audio_array, x, zcr, spectral_contrast, bass_energy






def gen_and_save_audio(text_prompt, history_prompt=None, text_temp=0.7, waveform_temp=0.7, filename="", output_dir="bark_samples", split_by_words=0, split_by_lines=0, stable_mode=False, confused_travolta_mode=False, iteration=1, zcr_threshold=70):
    def generate_unique_filename(base_filename):
        name, ext = os.path.splitext(base_filename)
        unique_filename = base_filename
        counter = 1
        while os.path.exists(unique_filename):
            unique_filename = f"{name}_{counter}{ext}"
            counter += 1
        return unique_filename
    orig_history_prompt = history_prompt
    saveit = True if history_prompt is None else False
    if iteration == 1:
        print(f"Full Prompt: {text_prompt}")
        if args.history_prompt:
            print(f"  Using speaker: {history_prompt}")
        else:
            print(f"  No speaker. Randomly generating a speaker.")
 
    text_chunks = split_text(text_prompt, split_by_words, split_by_lines)
    
    base = None
    npzbase = None
    audio_arr_chunks = []

    # Should output each audio chunk to disk midway so you at least a partial output if a long process crashes.
    for i, chunk in enumerate(text_chunks):
        print(f"Processing chunk {i + 1}/{len(text_chunks)}: {chunk}")
        longer_than_14_seconds, estimated_time = estimate_spoken_time(chunk)
        print(f"Current text chunk ballpark estimate: {estimated_time:.2f} seconds.")
        if longer_than_14_seconds:
            print(f"Text Prompt could be too long, might want to try a shorter one or try splitting tighter.")

        audio_array, x, zcr, spectral_contrast, bass_energy = generate_audio_with_zcr_check(chunk, history_prompt, sample_rate, zcr_threshold=zcr_threshold, text_temp=text_temp, waveform_temp=waveform_temp, base=base, confused_travolta_mode=confused_travolta_mode)

        print(f"Zero-crossing rate: {zcr}")
        
        if saveit is True and npzbase is None:
            npzbase = x
        if stable_mode:
            base = x if (base is None and history_prompt is None) else base
        else:
            base = x
            history_prompt = None
        audio_arr_chunks.append(audio_array)
        
    concatenated_audio_arr = np.concatenate(audio_arr_chunks)

    if not filename:
        date_str = datetime.datetime.now().strftime("%Y-%m-%d-%H")
        truncated_text = text_prompt.replace("WOMAN:", "").replace("MAN:", "")[:15].strip().replace(" ", "_")
        filename = f"{truncated_text}-history_prompt-{orig_history_prompt}-text_temp-{text_temp}-waveform_temp-{waveform_temp}-{date_str}.wav"
        filename = generate_unique_filename(filename)

    # Create output directory if it doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
    else:
        filepath = filename

    i = 1
    name, ext = os.path.splitext(filepath)
    while os.path.exists(filepath):
        filepath = f"{name}_{i}{ext}"
        i += 1

    if saveit is True:
        save_npz_file(f'{filepath}.npz', npzbase[0], npzbase[1], npzbase[2], output_dir=output_dir)

    save_audio_to_file(filepath, concatenated_audio_arr, SAMPLE_RATE, output_dir=output_dir)



# If there's no text_prompt passed on the command line, process this list instead.
# If you use an entir song, make sure you set --split_by_lines. 
text_prompts = []

text_prompt = """
    ♪ We're no strangers to love ♪
    ♪ You know the rules and so do I (do I) ♪
    ♪ A full commitment's what I'm thinking of ♪
    ♪ You wouldn't get this from any other guy ♪
"""
text_prompts.append(text_prompt)

text_prompt = """
    In the beginning the Universe was created. This has made a lot of people very angry and been widely regarded as a bad move.
"""
text_prompts.append(text_prompt)

text_prompt = """
    A common mistake that people make when trying to design something completely foolproof is to underestimate the ingenuity of complete fools.
"""
text_prompts.append(text_prompt)


def main(args):
  
    if args.list_speakers:
        print_speakers_list(categorized_files)
    else:
        if args.text_prompt:
            text_prompts_to_process = [args.text_prompt]
        elif args.prompt_file:
            with open(args.prompt_file, "r", encoding="utf-8") as f:
                if args.prompt_file_separator:
                    text_prompts_to_process = f.read().split(args.prompt_file_separator)
                else:
                    text_prompts_to_process = [f.read()]

            text_prompts_to_process = [prompt for prompt in text_prompts_to_process if prompt.strip()]

            print(f"Processing prompts from file: {args.prompt_file}")
            print(f"Number of prompts after splitting: {len(text_prompts_to_process)}")

        else:
            print("No text prompt provided. Using the prompts defined in this python file instead.")
            text_prompts_to_process = text_prompts
        if args.history_prompt: 
            history_prompt = args.history_prompt
        else:
            history_prompt = None
        text_temp = args.text_temp if args.text_temp else 0.7
        waveform_temp = args.waveform_temp if args.waveform_temp else 0.7
        stable_mode = args.stable_mode if args.stable_mode else False
        confused_travolta_mode = args.confused_travolta_mode if args.confused_travolta_mode else False
        filename = args.filename if args.filename else ""
        output_dir = args.output_dir if args.output_dir else "bark_samples"

        print("Loading Bark models...")
        
        if args.use_smaller_models:
            print("Using smaller models.")
            preload_models(use_smaller_models=True)
        else:
            preload_models()

        print("Models loaded.")

        for idx, prompt in enumerate(text_prompts_to_process, start=1):
            print(f"Processing prompt {idx} of {len(text_prompts_to_process)}:")
            
            split_by_words = args.split_by_words if args.split_by_words else 0
            split_by_lines = args.split_by_lines if args.split_by_lines else 0

            if args.iterations > 1: 
                for iteration in range(1, args.iterations + 1):
                    print(f"Iteration {iteration} of {args.iterations}.")
                    gen_and_save_audio(prompt, history_prompt, text_temp, waveform_temp, filename, output_dir, split_by_words, split_by_lines, stable_mode, confused_travolta_mode, iteration=iteration)
            else:
                gen_and_save_audio(prompt, history_prompt, text_temp, waveform_temp, filename, output_dir, split_by_words, split_by_lines, stable_mode, confused_travolta_mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
        (This grew into a bit more than a BARK CLI wrapper.)
you manually split your text, which I neglected to provide an easy way to do (seperate token?).

        """, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--text_prompt", help="Text prompt. If not provided, a set of default prompts will be used defined in this file.")
    parser.add_argument("--history_prompt", default=None, help="Optional. Choose a speaker from the list of languages: . Use --list_speakers to see all available options.")
    parser.add_argument("--text_temp", type=float, help="Text temperature. Default is 0.7.")
    parser.add_argument("--waveform_temp", type=float, help="Waveform temperature. Default is 0.7.")
    parser.add_argument("--filename", help="Output filename. If not provided, a unique filename will be generated based on the text prompt and other parameters.")
    parser.add_argument("--output_dir", help="Output directory. Default is 'bark_samples'.")
    parser.add_argument("--list_speakers", action="store_true", help="List all preset speaker options instead of generating audio.")
    parser.add_argument("--use_smaller_models", action="store_true", help="Use for GPUS with less than 10GB of memory, or for more speed.")
    parser.add_argument("--iterations", type=int, default=1, help="Number of iterations. Default is 1.")
    parser.add_argument("--split_by_words", type=int, default=0, help="Breaks text_prompt into <14 second audio clips every x words")
    parser.add_argument("--split_by_lines", type=int, default=0, help="Breaks text_prompt into <14 second audio clips every x lines")
    parser.add_argument("--stable_mode", action="store_true", help="Choppier and not as natural sounding, but much more stable for very long audio files.")
    parser.add_argument("--confused_travolta_mode", default=False, action="store_true", help="Just for fun. Try it and you'll understand.")

    parser.add_argument("--prompt_file", help="Optional. The path to a file containing the text prompt. Overrides the --text_prompt option if provided.")
    parser.add_argument("--prompt_file_separator", help="Optional. The separator used to split the content of the prompt_file into multiple text prompts.")
    
    args = parser.parse_args()
    main(args)
