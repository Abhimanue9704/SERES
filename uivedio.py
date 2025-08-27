import customtkinter as ctk
import vlc
import os
from customtkinter import CTkFont
import subprocess
import sys
import ctypes
import threading
import numpy as np
import cv2
from PIL import Image, ImageTk
import tkinter as tk
from uibutton import SpeakButton
class NDIViewerUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("SERES")
        self.geometry("1920x1080")

        # Configure the appearance mode and color theme
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        # Initialize NDI-related attributes
        self.ndi_lib = None
        self.my_ndi_recv = None
        self.running = True
        self.ndi_thread = None

        # Initialize VLC-related attributes
        self.instance = None
        self.player = None
        self.is_playing = False
        self.custom_font=None
        # Initialize UI flag
        self.ui_setup_complete = False
        self.speak_button_instance=None
        self.speak_button=None
        self.clear_button=None
        self.button_frame=None
        self.main_frame=None
        # Create a frame for the video
        self.video_frame = ctk.CTkFrame(self, fg_color="black")
        self.video_frame.pack(fill="both", expand=True)

        # Initialize Canvas
        self.canvas = tk.Canvas(self.video_frame, width=1920, height=1080, bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        # Mark UI setup as complete
        self.ui_setup_complete = True

        # Initialize VLC
        self.setup_vlc()
        
        # Set the video path
        video_path = r'E:\ser\TTS\YouCut_20250122_190407777.mp4'
        if os.path.exists(video_path):
            self.load_video(video_path)
        else:
            print("Failed to open video file")
            self.cleanup_video()
            self.setup_main_ui()

        self.clear_button_state_file()
        # Setup NDI
        self.setup_ndi()

    def setup_ndi(self):
        # Load NDI SDK Library
        self.ndi_lib = ctypes.CDLL(r"C:\Program Files\NDI\NDI 5 SDK\Bin\x64\Processing.NDI.Lib.x64.dll")
        
        if not self.ndi_lib.NDIlib_initialize():
            raise RuntimeError("Failed to initialize NDI.")
            
        # Define NDI structures
        class NDIlib_recv_instance_t(ctypes.Structure):
            pass

        class NDIlib_source_t(ctypes.Structure):
            _fields_ = [
                ("p_ndi_name", ctypes.c_char_p),
                ("p_url_address", ctypes.c_char_p)
            ]

        self.NDIlib_video_frame_v2_t = type('NDIlib_video_frame_v2_t', (ctypes.Structure,),
            {'_fields_': [
                ("xres", ctypes.c_int),
                ("yres", ctypes.c_int),
                ("FourCC", ctypes.c_int),
                ("frame_rate_N", ctypes.c_int),
                ("frame_rate_D", ctypes.c_int),
                ("picture_aspect_ratio", ctypes.c_float),
                ("frame_format_type", ctypes.c_int),
                ("timecode", ctypes.c_longlong),
                ("p_data", ctypes.POINTER(ctypes.c_uint8)),
                ("line_stride_in_bytes", ctypes.c_int),
                ("p_metadata", ctypes.c_char_p),
                ("timestamp", ctypes.c_longlong)
            ]})

        # Create receiver
        class NDIlib_recv_create_v3_t(ctypes.Structure):
            _fields_ = [
                ("source_to_connect_to", NDIlib_source_t),
                ("color_format", ctypes.c_int),
                ("bandwidth", ctypes.c_int),
                ("allow_video_fields", ctypes.c_bool),
                ("p_ndi_recv_name", ctypes.c_char_p)
            ]

        recv_create_desc = NDIlib_recv_create_v3_t()
        recv_create_desc.color_format = 0  # BGRX_BGRA
        recv_create_desc.bandwidth = 0
        recv_create_desc.allow_video_fields = False
        recv_create_desc.p_ndi_recv_name = b"Python NDI Viewer"

        self.ndi_lib.NDIlib_recv_create_v3.restype = ctypes.POINTER(NDIlib_recv_instance_t)
        self.my_ndi_recv = self.ndi_lib.NDIlib_recv_create_v3(ctypes.byref(recv_create_desc))

        # Connect to source
        my_source = NDIlib_source_t()
        my_source.p_ndi_name = b"DESKTOP-NR0AHGQ (Unreal Engine Output)"  # The source to receive ndi streaming input from
        self.ndi_lib.NDIlib_recv_connect(self.my_ndi_recv, ctypes.byref(my_source))

        # Start video thread
        self.ndi_thread = threading.Thread(target=self.update_ndi_frame)
        self.ndi_thread.daemon = True
        self.ndi_thread.start()

    def update_ndi_frame(self):
        while self.running:
            video_frame = self.NDIlib_video_frame_v2_t()
            
            result = self.ndi_lib.NDIlib_recv_capture_v2(
                self.my_ndi_recv,
                ctypes.byref(video_frame),
                None,
                None,
                1500
            )

            if result == 1 and video_frame.p_data:
                # Convert NDI frame to numpy array
                frame_buffer = np.ctypeslib.as_array(
                    video_frame.p_data,
                    shape=(video_frame.yres, video_frame.line_stride_in_bytes // 4, 4)
                )
                
                frame_copy = frame_buffer.copy()
                frame_bgr = cv2.cvtColor(frame_copy, cv2.COLOR_BGRA2BGR)
                
                # Resize frame to 1920x1080
                frame_resized = cv2.resize(frame_bgr, (1920, 1080), interpolation=cv2.INTER_AREA)
                
                # Convert to PIL format
                image = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
                photo = ImageTk.PhotoImage(image=image)
                
                # Update canvas in main thread
                if self.ui_setup_complete:
                    self.after(0, self.update_canvas, photo)
                
                # Free NDI frame
                self.ndi_lib.NDIlib_recv_free_video_v2(self.my_ndi_recv, ctypes.byref(video_frame))

    def update_canvas(self, photo):
        """Update canvas in main thread"""
        if hasattr(self, 'canvas'):
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, image=photo, anchor="nw")
            self.canvas.image = photo  # Keep reference

    def setup_vlc(self):
        """Initialize VLC instance and player"""
        try:
            self.instance = vlc.Instance()
            self.player = self.instance.media_player_new()
        except Exception as e:
            print(f"Failed to initialize VLC: {e}")
            self.instance = None
            self.player = None

    def load_video(self, video_path):
        """Load and prepare video for playback"""
        try:
            if self.player and self.instance:
                media = self.instance.media_new(video_path)
                self.player.set_media(media)
                
                # Get the window ID
                if os.name == "nt":  # Windows
                    self.player.set_hwnd(self.video_frame.winfo_id())
                else:  # Linux/Mac
                    self.player.set_xwindow(self.video_frame.winfo_id())
                
                # Bind window resize event
                self.bind("<Configure>", self.on_resize)
                
                # Start playback after a short delay
                self.after(100, self.start_playback)
                self.is_playing = True
        except Exception as e:
            print(f"Failed to load video: {e}")
            self.cleanup_video()
            self.setup_main_ui()

    def cleanup_video(self):
        """Safely cleanup VLC resources"""
        try:
            if hasattr(self, 'player') and self.player:
                if self.is_playing:
                    self.player.stop()
                    self.is_playing = False
                self.player.release()
                self.player = None
            
            if hasattr(self, 'instance') and self.instance:
                self.instance.release()
                self.instance = None
            
            if hasattr(self, 'video_frame') and self.video_frame:
                self.video_frame.destroy()
        except Exception as e:
            print(f"Error during cleanup: {e}")

    def start_playback(self):
        """Start video playback"""
        if self.player:
            self.player.play()
            self.is_playing = True
            self.check_video_end()

    def check_video_end(self):
        """Check if video has ended"""
        if not self.player:
            return
            
        if self.player.get_state() == vlc.State.Ended:
            self.cleanup_video()
            self.setup_main_ui()
        elif self.is_playing:
            self.after(100, self.check_video_end)

    def on_resize(self, event):
        """Handle window resize events"""
        pass  # VLC handles resizing automatically
    
    def neuro_player_thread(self):
        program_path = "E:/NeuroSync_Player/wave_to_face.py"  # Change to your actual path
        subprocess.Popen(program_path, shell=True)

    def speak_emotion(self):
        self.speak_button.configure(text="GETTING READY...", fg_color="#e74c3c")

        thread = threading.Thread(target=self.neuro_player_thread, daemon=True)
        thread.start()
        
        # script_path = "E:/NeuroSync_Player/wave_to_face.py"
        
        # if not os.path.exists(script_path):
        #     print(f"Error: Could not find {script_path}")
        #     return
        # try:
        #     # Using Python executable to run the script
        #     python_path = sys.executable
        #     subprocess.run([python_path, script_path], check=True)
        #     print("Response received successfully.")
        # except subprocess.CalledProcessError as e:
        #     print(f"Error running script: {e}")
        # except Exception as e:
        #     print(f"Unexpected error: {e}")

    def clear_file(self):
        file_name = r"E:\ser\TTS\chat_history.txt"
        """Clears the contents of a file."""
        with open(file_name, "w") as file:  # Open in write mode to overwrite
            pass  # Writing nothing effectively clears the file

    def clear_button_state_file(self):
        file_name = r"E:\ser\TTS\button_state.txt"
        """Clears the contents of a file."""
        with open(file_name, "w") as file:  # Open in write mode to overwrite
            pass

    def neuro_api_thread(self):
        program_path = r"E:\NeuroSync_Local_API\neurosync_local_api.py"  # Change to your actual path
        subprocess.Popen([sys.executable, program_path], shell=True)    

    def setup_main_ui(self):
        """Set up the main UI components"""
        # Main container layout
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Grid configuration
        self.main_frame.columnconfigure(0, weight=3)
        self.main_frame.columnconfigure(1, weight=1)
        self.main_frame.rowconfigure(0, weight=1)
        self.main_frame.rowconfigure(1, weight=0)

        # Canvas for video
        self.canvas = ctk.CTkCanvas(self.main_frame, width=1920, height=1080, bg="black", highlightthickness=0)
        self.canvas.grid(row=0, column=0, padx=(0, 10), pady=10, sticky="nsew")

        self.custom_font = CTkFont(family="OCR A Extended", size=18)

        # Process button
        self.button_frame = ctk.CTkFrame(self.main_frame)
        self.button_frame.grid(row=1, column=0, columnspan=2, pady=(10, 20), sticky="n")

        # Configure columns in button_frame
        self.button_frame.columnconfigure(0, weight=1)
        self.button_frame.columnconfigure(1, weight=1)

        # Speak button
        self.speak_button_instance = SpeakButton(self.button_frame, font=self.custom_font)

        self.speak_button=self.speak_button_instance.speak_button
        self.clear_button=self.speak_button_instance.clear_button

        thread = threading.Thread(target=self.neuro_api_thread, daemon=True)
        thread.start()

    def on_closing(self):
        """Handle application cleanup and closing"""
        # Stop NDI thread
        self.running = False
        if self.ndi_thread:
            self.ndi_thread.join()

        # Cleanup NDI resources
        if self.my_ndi_recv and self.ndi_lib:
            self.ndi_lib.NDIlib_recv_destroy(self.my_ndi_recv)
            self.ndi_lib.NDIlib_destroy()

        # Cleanup video
        self.cleanup_video()
        self.quit()
        self.destroy()

if __name__ == "__main__":
    app = NDIViewerUI()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()