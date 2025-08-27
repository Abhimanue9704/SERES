import os
import customtkinter as ctk
import threading
import time
import sys

class SpeakButton:
    def __init__(self, parent, font):
        """Initialize the Speak Button inside the given parent widget."""
        self.speak_button = ctk.CTkButton(
            parent,
            text="PRESS TO SPEAK",
            font=font,
            fg_color="#e74c3c",
            hover_color="#b03a2e",
            command=self.speak_emotion
        )
        self.speak_button.grid(row=0, column=0, padx=10)  # Keep same layout

        self.clear_button = ctk.CTkButton(
            parent,
            text="CLEAR",
            font=font,
            fg_color="#e74c3c",
            hover_color="#b03a2e",
            command=self.clear_file
        )
        self.clear_button.grid(row=0, column=1, padx=10)
        self.running = True
        threading.Thread(target=self.check_state, daemon=True).start()
    
    def check_state(self):
        state_file = "button_state.txt"
        last_state = ""
        
        while self.running:
            try:
                if os.path.exists(state_file):
                    with open(state_file, 'r') as f:
                        state = f.read().strip()
                        if state != last_state:
                            self.speak_button.after(0, self.update_button, state)
                            last_state = state
            except:
                pass
            time.sleep(0.1)  # Check every 100ms
    
    def update_button(self, state):
        if state == "LISTENING":
            self.speak_button.configure(text="LISTENING...", fg_color="#27ae60",hover_color="#27ae60")
        elif state == "RESET":
            self.speak_button.configure(text="PRESS TO SPEAK", fg_color="#e74c3c")
    
    def speak_emotion(self):
        """Handle button press"""
        # Show "GETTING READY" immediately
        self.speak_button.configure(text="GETTING READY...", fg_color="#f39c12",hover_color="#f39c12")
        
        # Launch your script
        script_path = "E:/NeuroSync_Player/wave_to_face.py"
        if not os.path.exists(script_path):
            print(f"Error: Could not find {script_path}")
            return
            
        import subprocess
        try:
            subprocess.Popen([sys.executable, script_path])
        except Exception as e:
            print(f"Error: {e}")
            self.speak_button.configure(text="PRESS TO SPEAK", fg_color="#e74c3c")

    # wave_to_face.py
    def update_button_state(state):
        with open("button_state.txt", "w") as f:
            f.write(state)

    def clear_file(self):
        file_name = r"E:\ser\TTS\chat_history.txt"
        """Clears the contents of a file."""
        with open(file_name, "w") as file:  # Open in write mode to overwrite
            pass  # Writing nothing effectively clears the file    