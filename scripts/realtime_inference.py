import argparse
import os
from omegaconf import OmegaConf
import numpy as np
import cv2
import torch
import glob
import pickle
import sys
from tqdm import tqdm
import copy
import json
from transformers import WhisperModel
import librosa
import subprocess
import tempfile

from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.utils import datagen
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs
from musetalk.utils.blending import get_image_prepare_material, get_image_blending
from musetalk.utils.utils import load_all_model
from musetalk.utils.audio_processor import AudioProcessor

import shutil
import threading
import queue
import time
import matplotlib.pyplot as plt


def fast_check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except:
        return False


def video2imgs(vid_path, save_path, ext='.png', cut_frame=10000000):
    cap = cv2.VideoCapture(vid_path)
    count = 0
    while True:
        if count > cut_frame:
            break
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(f"{save_path}/{count:08d}.png", frame)
            count += 1
        else:
            break


def osmakedirs(path_list):
    for path in path_list:
        os.makedirs(path) if not os.path.exists(path) else None


class AudioChunker:
    """Class for splitting audio into chunks based on pauses"""
    def __init__(self, min_silence_len=700, silence_thresh=-40, keep_silence=300):
        """
        Initialize AudioChunker.
        
        Args:
            min_silence_len: Minimum silence length in ms to be considered a pause
            silence_thresh: Silence threshold in dB
            keep_silence: Amount of silence to keep at chunk boundaries (ms)
        """
        self.min_silence_len = min_silence_len  # in ms
        self.silence_thresh = silence_thresh  # in dB
        self.keep_silence = keep_silence  # in ms
        
    def detect_pauses(self, audio_path):
        """
        Detect pauses in audio and return chunk boundaries.
        
        Returns:
            List of (start_time, end_time) tuples in seconds
        """
        # Load audio file
        y, sr = librosa.load(audio_path, sr=None)
        
        # Convert parameters from ms to samples
        min_silence_samples = int(self.min_silence_len * sr / 1000)
        keep_silence_samples = int(self.keep_silence * sr / 1000)
        
        # Calculate amplitude in dB
        db = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        mean_db = np.mean(db, axis=0)
        
        # Find silence regions
        is_silence = mean_db < self.silence_thresh
        
        # Find transitions
        transitions = np.where(np.diff(is_silence.astype(int)))[0]
        
        # Find silence regions longer than min_silence_samples
        silence_regions = []
        for i in range(0, len(transitions), 2):
            if i+1 < len(transitions):
                if transitions[i+1] - transitions[i] >= min_silence_samples:
                    silence_regions.append((transitions[i], transitions[i+1]))
        
        # Convert silence regions to chunk boundaries
        if not silence_regions:
            # If no silence detected, return the whole audio as one chunk
            return [(0, len(y)/sr)]
        
        chunk_boundaries = []
        start = 0
        
        for silence_start, silence_end in silence_regions:
            chunk_end = silence_start + keep_silence_samples
            chunk_boundaries.append((start/sr, chunk_end/sr))
            start = silence_end - keep_silence_samples
        
        # Add the last chunk
        if start < len(y):
            chunk_boundaries.append((start/sr, len(y)/sr))
        
        return chunk_boundaries
    
    def split_audio(self, audio_path, output_dir):
        """
        Split audio file into chunks based on pauses and save them.
        
        Args:
            audio_path: Path to input audio file
            output_dir: Directory to save audio chunks
            
        Returns:
            List of paths to audio chunks
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Get chunk boundaries
        chunk_boundaries = self.detect_pauses(audio_path)
        
        # Split audio into chunks
        chunk_paths = []
        for i, (start_time, end_time) in enumerate(chunk_boundaries):
            output_path = os.path.join(output_dir, f"chunk_{i:03d}.wav")
            
            # Use ffmpeg to extract chunk
            cmd = [
                "ffmpeg",
                "-y",  # Overwrite output files
                "-i", audio_path,
                "-ss", str(start_time),  # Start time
                "-to", str(end_time),    # End time
                "-c:a", "pcm_s16le",     # Use PCM format
                output_path
            ]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            chunk_paths.append(output_path)
        
        return chunk_paths


class LiveOutput:
    """Class for displaying output frames in real-time"""
    def __init__(self, window_name="MuseTalk Realtime Output"):
        self.window_name = window_name
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 640, 480)
        
    def show_frame(self, frame):
        """Display a frame and handle key presses"""
        cv2.imshow(self.window_name, frame)
        key = cv2.waitKey(1)
        if key == 27:  # ESC key
            return False
        return True
    
    def close(self):
        """Close the display window"""
        cv2.destroyWindow(self.window_name)


@torch.no_grad()
class Avatar:
    def __init__(self, avatar_id, video_path, bbox_shift, batch_size, preparation):
        self.avatar_id = avatar_id
        self.video_path = video_path
        self.bbox_shift = bbox_shift
        # Choose base path based on version
        if args.version == "v15":
            self.base_path = f"./results/{args.version}/avatars/{avatar_id}"
        else:  # v1
            self.base_path = f"./results/avatars/{avatar_id}"
            
        self.avatar_path = self.base_path
        self.full_imgs_path = f"{self.avatar_path}/full_imgs"
        self.coords_path = f"{self.avatar_path}/coords.pkl"
        self.latents_out_path = f"{self.avatar_path}/latents.pt"
        self.video_out_path = f"{self.avatar_path}/vid_output/"
        self.mask_out_path = f"{self.avatar_path}/mask"
        self.mask_coords_path = f"{self.avatar_path}/mask_coords.pkl"
        self.avatar_info_path = f"{self.avatar_path}/avator_info.json"
        self.avatar_info = {
            "avatar_id": avatar_id,
            "video_path": video_path,
            "bbox_shift": bbox_shift,
            "version": args.version
        }
        self.preparation = preparation
        self.batch_size = batch_size
        self.idx = 0
        self.display = LiveOutput(f"MuseTalk - {avatar_id}")
        self.init()

    def init(self):
        if self.preparation:
            if os.path.exists(self.avatar_path):
                response = input(f"{self.avatar_id} exists, Do you want to re-create it ? (y/n)")
                if response.lower() == "y":
                    shutil.rmtree(self.avatar_path)
                    print("*********************************")
                    print(f"  creating avator: {self.avatar_id}")
                    print("*********************************")
                    osmakedirs([self.avatar_path, self.full_imgs_path, self.video_out_path, self.mask_out_path])
                    self.prepare_material()
                else:
                    self.input_latent_list_cycle = torch.load(self.latents_out_path)
                    with open(self.coords_path, 'rb') as f:
                        self.coord_list_cycle = pickle.load(f)
                    input_img_list = glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
                    input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                    self.frame_list_cycle = read_imgs(input_img_list)
                    with open(self.mask_coords_path, 'rb') as f:
                        self.mask_coords_list_cycle = pickle.load(f)
                    input_mask_list = glob.glob(os.path.join(self.mask_out_path, '*.[jpJP][pnPN]*[gG]'))
                    input_mask_list = sorted(input_mask_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                    self.mask_list_cycle = read_imgs(input_mask_list)
            else:
                print("*********************************")
                print(f"  creating avator: {self.avatar_id}")
                print("*********************************")
                osmakedirs([self.avatar_path, self.full_imgs_path, self.video_out_path, self.mask_out_path])
                self.prepare_material()
        else:
            if not os.path.exists(self.avatar_path):
                print(f"{self.avatar_id} does not exist, you should set preparation to True")
                sys.exit()

            with open(self.avatar_info_path, "r") as f:
                avatar_info = json.load(f)

            if avatar_info['bbox_shift'] != self.avatar_info['bbox_shift']:
                response = input(f" 【bbox_shift】 is changed, you need to re-create it ! (c/continue)")
                if response.lower() == "c":
                    shutil.rmtree(self.avatar_path)
                    print("*********************************")
                    print(f"  creating avator: {self.avatar_id}")
                    print("*********************************")
                    osmakedirs([self.avatar_path, self.full_imgs_path, self.video_out_path, self.mask_out_path])
                    self.prepare_material()
                else:
                    sys.exit()
            else:
                self.input_latent_list_cycle = torch.load(self.latents_out_path)
                with open(self.coords_path, 'rb') as f:
                    self.coord_list_cycle = pickle.load(f)
                input_img_list = glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
                input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                self.frame_list_cycle = read_imgs(input_img_list)
                with open(self.mask_coords_path, 'rb') as f:
                    self.mask_coords_list_cycle = pickle.load(f)
                input_mask_list = glob.glob(os.path.join(self.mask_out_path, '*.[jpJP][pnPN]*[gG]'))
                input_mask_list = sorted(input_mask_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                self.mask_list_cycle = read_imgs(input_mask_list)

    def prepare_material(self):
        print("preparing data materials ... ...")
        with open(self.avatar_info_path, "w") as f:
            json.dump(self.avatar_info, f)

        if os.path.isfile(self.video_path):
            video2imgs(self.video_path, self.full_imgs_path, ext='png')
        else:
            print(f"copy files in {self.video_path}")
            files = os.listdir(self.video_path)
            files.sort()
            files = [file for file in files if file.split(".")[-1] == "png"]
            for filename in files:
                shutil.copyfile(f"{self.video_path}/{filename}", f"{self.full_imgs_path}/{filename}")
        input_img_list = sorted(glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]')))

        print("extracting landmarks...")
        coord_list, frame_list = get_landmark_and_bbox(input_img_list, self.bbox_shift)
        input_latent_list = []
        idx = -1
        # maker if the bbox is not sufficient
        coord_placeholder = (0.0, 0.0, 0.0, 0.0)
        for bbox, frame in zip(coord_list, frame_list):
            idx = idx + 1
            if bbox == coord_placeholder:
                continue
            x1, y1, x2, y2 = bbox
            if args.version == "v15":
                y2 = y2 + args.extra_margin
                y2 = min(y2, frame.shape[0])
                coord_list[idx] = [x1, y1, x2, y2]  # Update coord_list's bbox
            crop_frame = frame[y1:y2, x1:x2]
            resized_crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            latents = vae.get_latents_for_unet(resized_crop_frame)
            input_latent_list.append(latents)

        self.frame_list_cycle = frame_list + frame_list[::-1]
        self.coord_list_cycle = coord_list + coord_list[::-1]
        self.input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
        self.mask_coords_list_cycle = []
        self.mask_list_cycle = []

        for i, frame in enumerate(tqdm(self.frame_list_cycle)):
            cv2.imwrite(f"{self.full_imgs_path}/{str(i).zfill(8)}.png", frame)

            x1, y1, x2, y2 = self.coord_list_cycle[i]
            if args.version == "v15":
                mode = args.parsing_mode
            else:
                mode = "raw"
            mask, crop_box = get_image_prepare_material(frame, [x1, y1, x2, y2], fp=fp, mode=mode)

            cv2.imwrite(f"{self.mask_out_path}/{str(i).zfill(8)}.png", mask)
            self.mask_coords_list_cycle += [crop_box]
            self.mask_list_cycle.append(mask)

        with open(self.mask_coords_path, 'wb') as f:
            pickle.dump(self.mask_coords_list_cycle, f)

        with open(self.coords_path, 'wb') as f:
            pickle.dump(self.coord_list_cycle, f)

        torch.save(self.input_latent_list_cycle, os.path.join(self.latents_out_path))

    def process_frames(self, res_frame_queue, video_len, save_dir=None):
        """Process generated frames and display/save them"""
        frames_processed = 0
        
        # Create save directory if needed
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
        while frames_processed < video_len:
            try:
                res_frame = res_frame_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue

            bbox = self.coord_list_cycle[self.idx % (len(self.coord_list_cycle))]
            ori_frame = copy.deepcopy(self.frame_list_cycle[self.idx % (len(self.frame_list_cycle))])
            x1, y1, x2, y2 = bbox
            
            try:
                res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
            except:
                frames_processed += 1
                continue
                
            mask = self.mask_list_cycle[self.idx % (len(self.mask_list_cycle))]
            mask_crop_box = self.mask_coords_list_cycle[self.idx % (len(self.mask_coords_list_cycle))]
            combine_frame = get_image_blending(ori_frame, res_frame, bbox, mask, mask_crop_box)

            # Display frame
            if not self.display.show_frame(combine_frame):
                print("Display closed by user. Exiting...")
                break
                
            # Save frame if requested
            if save_dir:
                cv2.imwrite(f"{save_dir}/{str(frames_processed).zfill(8)}.png", combine_frame)
                
            frames_processed += 1
            self.idx = (self.idx + 1) % len(self.coord_list_cycle)
            
        return frames_processed

    def inference_chunk(self, audio_path, fps, out_dir=None):
        """Process a single audio chunk with real-time display"""
        print(f"Processing chunk: {os.path.basename(audio_path)}")
        
        # Extract audio features
        start_time = time.time()
        whisper_input_features, librosa_length = audio_processor.get_audio_feature(
            audio_path, 
            weight_dtype=weight_dtype
        )
        
        whisper_chunks = audio_processor.get_whisper_chunk(
            whisper_input_features,
            device,
            weight_dtype,
            whisper,
            librosa_length,
            fps=fps,
            audio_padding_length_left=args.audio_padding_length_left,
            audio_padding_length_right=args.audio_padding_length_right,
        )
        print(f"Processing audio feature for {audio_path} took {(time.time() - start_time) * 1000:.2f}ms")
        
        # Set up for real-time processing
        video_num = len(whisper_chunks)
        res_frame_queue = queue.Queue(maxsize=30)  # Limit queue size for more real-time feel
        
        # Start frame processing thread
        process_thread = threading.Thread(
            target=self.process_frames, 
            args=(res_frame_queue, video_num, out_dir)
        )
        process_thread.start()
        
        # Generate frames
        gen = datagen(whisper_chunks, self.input_latent_list_cycle, self.batch_size)
        start_time = time.time()
        
        # Process audio in batches and generate frames
        for i, (whisper_batch, latent_batch) in enumerate(tqdm(
            gen, 
            total=int(np.ceil(float(video_num) / self.batch_size)),
            desc="Generating frames"
        )):
            audio_feature_batch = pe(whisper_batch.to(device))
            latent_batch = latent_batch.to(device=device, dtype=unet.model.dtype)

            pred_latents = unet.model(
                latent_batch,
                timesteps,
                encoder_hidden_states=audio_feature_batch
            ).sample
            
            pred_latents = pred_latents.to(device=device, dtype=vae.vae.dtype)
            recon = vae.decode_latents(pred_latents)
            
            for res_frame in recon:
                # Add frames to queue, blocking if queue is full
                res_frame_queue.put(res_frame, block=True)
        
        # Wait for processing to complete
        process_thread.join()
        
        processing_time = time.time() - start_time
        print(f"Processed {video_num} frames in {processing_time:.2f}s " +
              f"({video_num/processing_time:.2f} fps)")
        
        return video_num

    def inference_with_pauses(self, audio_path, out_vid_name=None, fps=25, save_output=False):
        """Process audio by splitting it at pauses and showing each chunk in real-time"""
        print(f"Starting real-time inference on {audio_path}")
        
        # Create temp directory for chunks
        temp_dir = tempfile.mkdtemp()
        chunks_dir = os.path.join(temp_dir, "chunks")
        os.makedirs(chunks_dir, exist_ok=True)
        
        # Create output directory if saving
        frames_dir = None
        if save_output:
            frames_dir = os.path.join(self.avatar_path, "frames")
            os.makedirs(frames_dir, exist_ok=True)
        
        try:
            # Split audio at pauses
            chunker = AudioChunker(
                min_silence_len=args.min_silence_len,
                silence_thresh=args.silence_thresh,
                keep_silence=args.keep_silence
            )
            
            # Visualize speech segments for debugging if requested
            if args.visualize_segments:
                self.visualize_speech_segments(audio_path, chunker)
            
            audio_chunks = chunker.split_audio(audio_path, chunks_dir)
            print(f"Split audio into {len(audio_chunks)} chunks based on pauses")
            
            # Process each chunk with real-time display
            total_frames = 0
            chunk_count = len(audio_chunks)
            
            for i, chunk_path in enumerate(audio_chunks):
                print(f"\nProcessing chunk {i+1}/{chunk_count}")
                
                # Create directory for this chunk's frames if saving
                chunk_frames_dir = None
                if frames_dir:
                    chunk_frames_dir = os.path.join(frames_dir, f"chunk_{i:03d}")
                    os.makedirs(chunk_frames_dir, exist_ok=True)
                
                # Process this audio chunk
                frames = self.inference_chunk(chunk_path, fps, chunk_frames_dir)
                total_frames += frames
                
                # Small pause between chunks to make pauses more natural
                time.sleep(0.5)
            
            print(f"\nFinished processing all {chunk_count} chunks ({total_frames} total frames)")
            
            # Combine all chunks into final video if requested
            if save_output and out_vid_name:
                self.combine_chunks_to_video(frames_dir, audio_path, out_vid_name, fps)
                
        finally:
            # Clean up temporary files
            shutil.rmtree(temp_dir)
            
    def combine_chunks_to_video(self, frames_dir, audio_path, out_vid_name, fps):
        """Combine all processed frames into a final video with audio"""
        print("Combining chunks into final video...")
        
        # Create output directory
        os.makedirs(self.video_out_path, exist_ok=True)
        
        # Find all frame directories
        chunk_dirs = sorted(glob.glob(os.path.join(frames_dir, "chunk_*")))
        
        # Create temp directory for all frames
        all_frames_dir = os.path.join(self.avatar_path, "all_frames")
        os.makedirs(all_frames_dir, exist_ok=True)
        
        # Copy all frames to a single directory
        frame_index = 0
        for chunk_dir in chunk_dirs:
            chunk_frames = sorted(glob.glob(os.path.join(chunk_dir, "*.png")))
            for frame_path in chunk_frames:
                shutil.copy(
                    frame_path, 
                    os.path.join(all_frames_dir, f"{frame_index:08d}.png")
                )
                frame_index += 1
        
        # Convert frames to video
        output_vid = os.path.join(self.video_out_path, f"{out_vid_name}.mp4")
        
        # Create video without audio first
        temp_video = os.path.join(self.avatar_path, "temp.mp4")
        cmd_img2video = (
            f"ffmpeg -y -v warning -r {fps} -f image2 "
            f"-i {all_frames_dir}/%08d.png -vcodec libx264 "
            f"-vf format=yuv420p -crf 18 {temp_video}"
        )
        print(cmd_img2video)
        os.system(cmd_img2video)
        
        # Add audio to video
        cmd_combine_audio = (
            f"ffmpeg -y -v warning -i {audio_path} -i {temp_video} {output_vid}"
        )
        print(cmd_combine_audio)
        os.system(cmd_combine_audio)
        
        # Clean up
        os.remove(temp_video)
        shutil.rmtree(all_frames_dir)
        
        print(f"Final video saved to {output_vid}")
        
    def visualize_speech_segments(self, audio_path, chunker):
        """Visualize the speech segments and pauses"""
        # Load audio
        y, sr = librosa.load(audio_path, sr=None)
        
        # Get chunk boundaries
        chunk_boundaries = chunker.detect_pauses(audio_path)
        
        # Calculate waveform envelope
        S = np.abs(librosa.stft(y))
        envelope = np.mean(S, axis=0)
        times = librosa.times_like(envelope, sr=sr)
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(times, envelope, alpha=0.5)
        
        # Add colored regions for speech segments
        for i, (start, end) in enumerate(chunk_boundaries):
            plt.axvspan(start, end, color=f"C{i % 10}", alpha=0.3, 
                        label=f"Chunk {i}")
        
        plt.title("Speech Segments Detection")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        
        # Save the visualization
        os.makedirs("visualizations", exist_ok=True)
        plt.savefig(f"visualizations/{os.path.basename(audio_path)}_segments.png")
        plt.close()
        
        print(f"Visualization saved to visualizations/{os.path.basename(audio_path)}_segments.png")


if __name__ == "__main__":
    '''
    Modified script to process audio in chunks based on pauses for more realistic real-time experience.
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default="v15", choices=["v1", "v15"], help="Version of MuseTalk: v1 or v15")
    parser.add_argument("--ffmpeg_path", type=str, default="./ffmpeg-4.4-amd64-static/", help="Path to ffmpeg executable")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--vae_type", type=str, default="sd-vae", help="Type of VAE model")
    parser.add_argument("--unet_config", type=str, default="./models/musetalk/musetalk.json", help="Path to UNet configuration file")
    parser.add_argument("--unet_model_path", type=str, default="./models/musetalk/pytorch_model.bin", help="Path to UNet model weights")
    parser.add_argument("--whisper_dir", type=str, default="./models/whisper", help="Directory containing Whisper model")
    parser.add_argument("--inference_config", type=str, default="configs/inference/realtime.yaml")
    parser.add_argument("--bbox_shift", type=int, default=0, help="Bounding box shift value")
    parser.add_argument("--result_dir", default='./results', help="Directory for output results")
    parser.add_argument("--extra_margin", type=int, default=10, help="Extra margin for face cropping")
    parser.add_argument("--fps", type=int, default=25, help="Video frames per second")
    parser.add_argument("--audio_padding_length_left", type=int, default=2, help="Left padding length for audio")
    parser.add_argument("--audio_padding_length_right", type=int, default=2, help="Right padding length for audio")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size for inference")
    parser.add_argument("--output_vid_name", type=str, default=None, help="Name of output video file")
    parser.add_argument("--use_saved_coord", action="store_true", help='Use saved coordinates to save time')
    parser.add_argument("--saved_coord", action="store_true", help='Save coordinates for future use')
    parser.add_argument("--parsing_mode", default='jaw', help="Face blending parsing mode")
    parser.add_argument("--left_cheek_width", type=int, default=90, help="Width of left cheek region")
    parser.add_argument("--right_cheek_width", type=int, default=90, help="Width of right cheek region")
    parser.add_argument("--save_output", action="store_true", help="Save output frames and video")
    
    # New parameters for pause-based chunking
    parser.add_argument("--min_silence_len", type=int, default=500, 
                        help="Minimum silence length in ms to be considered a pause")
    parser.add_argument("--silence_thresh", type=float, default=-35, 
                        help="Silence threshold in dB")
    parser.add_argument("--keep_silence", type=int, default=200, 
                        help="Amount of silence to keep at chunk boundaries (ms)")
    parser.add_argument("--visualize_segments", action="store_true",
                        help="Visualize the detected speech segments")

    args = parser.parse_args()

    # Configure ffmpeg path
    if not fast_check_ffmpeg():
        print("Adding ffmpeg to PATH")
        # Choose path separator based on operating system
        path_separator = ';' if sys.platform == 'win32' else ':'
        os.environ["PATH"] = f"{args.ffmpeg_path}{path_separator}{os.environ['PATH']}"
        if not fast_check_ffmpeg():
            print("Warning: Unable to find ffmpeg, please ensure ffmpeg is properly installed")

    # Set computing device
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    # Load model weights
    vae, unet, pe = load_all_model(
        unet_model_path=args.unet_model_path,
        vae_type=args.vae_type,
        unet_config=args.unet_config,
        device=device
    )
    timesteps = torch.tensor([0], device=device)

    pe = pe.half().to(device)
    vae.vae = vae.vae.half().to(device)
    unet.model = unet.model.half().to(device)

    # Initialize audio processor and Whisper model
    audio_processor = AudioProcessor(feature_extractor_path=args.whisper_dir)
    weight_dtype = unet.model.dtype
    whisper = WhisperModel.from_pretrained(args.whisper_dir)
    whisper = whisper.to(device=device, dtype=weight_dtype).eval()
    whisper.requires_grad_(False)

    # Initialize face parser with configurable parameters based on version
    if args.version == "v15":
        fp = FaceParsing(
            left_cheek_width=args.left_cheek_width,
            right_cheek_width=args.right_cheek_width
        )
    else:  # v1
        fp = FaceParsing()

    inference_config = OmegaConf.load(args.inference_config)
    print(inference_config)

    for avatar_id in inference_config:
        data_preparation = inference_config[avatar_id]["preparation"]
        video_path = inference_config[avatar_id]["video_path"]
        if args.version == "v15":
            bbox_shift = 0
        else:
            bbox_shift = inference_config[avatar_id]["bbox_shift"]
        avatar = Avatar(
            avatar_id=avatar_id,
            video_path=video_path,
            bbox_shift=bbox_shift,
            batch_size=args.batch_size,
            preparation=data_preparation)

        audio_clips = inference_config[avatar_id]["audio_clips"]
        for audio_num, audio_path in audio_clips.items():
            print("Processing audio:", audio_path)
            # Use the new pause-based inference method
            avatar.inference_with_pauses(
                audio_path,
                out_vid_name=audio_num if args.save_output else None,
                fps=args.fps,
                save_output=args.save_output
            )
            
    print("All processing complete!")
