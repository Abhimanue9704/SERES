import torch
from TTS.api import TTS
from meta_ai_api import MetaAI
import speech_recognition as sr
import sys
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import torch
import librosa
import numpy as np
import os
import soundfile as sf
import noisereduce as nr
from scipy import signal
from uibutton import SpeakButton
from uivedio import NDIViewerUI
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")  
import re
from pydub import AudioSegment
import concurrent.futures
import threading

class ComplexEmotionPredictor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
        self.speaker_wav_path = "E:\\ser\\cloneSF.wav"
        self.complex_emotion_name=""
        self.model_dir = "E:/ser/wav2vec2-results/checkpoint-240"  
        if not os.path.exists(self.model_dir):
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_dir)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(self.model_dir)
        self.output_dir="E:/ser/TTS"
        self.model.eval()

        # Map the indices to basic emotions
        self.basic_emotions = {
            0: "neutral",
            1: "calm",
            2: "happy",
            3: "sad",
            4: "angry",
            5: "fear",
            6: "disgust",
            7: "surprise"
        }

        # Define comprehensive complex emotions
        self.complex_emotions = {
            0: "frustrated",      # angry + sad
            1: "anxious",         # fear + neutral
            2: "excited",         # happy + surprise
            3: "melancholic",     # sad + neutral
            4: "content",         # calm + happy
            5: "annoyed",         # angry + disgust
            6: "worried",         # fear + sad
            7: "amused",          # happy + surprise
            8: "overwhelmed",     # fear + surprise + angry
            9: "nostalgic",       # happy + sad + neutral
            10: "depressed",      # sustained sad + low variance
            11: "grateful",       # happy + calm + surprise
            12: "guilty",         # sad + disgust + fear
            13: "proud",          # happy + surprise + neutral
            14: "grief",          # intense sad + neutral + low positive
            15: "hopeless",       # sad + fear + low positive
            16: "apathetic",      # neutral + sad + low variance
            17: "heartbroken",    # sad + angry + low positive
            18: "lonely",         # sad + neutral + low social
            19: "despondent",     # sad + fear + neutral + low variance
            20: "numb",           # neutral + low variance
            21: "devastated",     # intense sad + angry + fear
            22: "desolate",       # sad + fear + low positive
            23: "miserable",      # sad + disgust + low positive
            24: "remorseful",     # sad + disgust + neutral
            25: "pissed off"      #very angry+surprised  
        }

    def calculate_complex_emotion(self, probabilities):
        """
        Enhanced algorithm to determine complex emotions based on basic emotion probabilities
        Returns the complex emotion index
        """
        # Get indices for easier reference
        neutral, calm, happy, sad, angry, fear, disgust, surprise = range(8)

        # Calculate emotional metrics
        emotional_flatness = 1.0 - np.std(probabilities)  # High value indicates flat emotional state
        total_negative = probabilities[sad] + probabilities[fear] + probabilities[disgust] + probabilities[angry]
        total_positive = probabilities[happy] + probabilities[calm]

        # Enhanced mapping rules
        if probabilities[angry] > 0.2 and probabilities[sad] > 0.15:
            return 0  # frustrated
        elif probabilities[fear] > 0.2 and probabilities[neutral] > 0.15:
            return 1  # anxious
        elif probabilities[happy] > 0.2 and probabilities[surprise] > 0.15:
            return 2  # excited
        elif probabilities[sad] > 0.25 and probabilities[neutral] > 0.15:
            return 3  # melancholic
        elif probabilities[calm] > 0.2 and probabilities[happy] > 0.15:
            return 4  # content
        elif probabilities[angry] > 0.2 and probabilities[disgust] > 0.15 or probabilities[disgust]>0.2 and probabilities[angry]>0.1:
            return 5  # annoyed
        elif probabilities[fear] > 0.2 and probabilities[sad] > 0.15:
            return 6  # worried
        elif probabilities[happy] > 0.2 and probabilities[surprise] > 0.15:
            return 7  # amused
        elif probabilities[fear] > 0.15 and probabilities[surprise] > 0.15 and probabilities[angry] > 0.15:
            return 8  # overwhelmed
        elif probabilities[happy] > 0.15 and probabilities[sad] > 0.15 and probabilities[neutral] > 0.15:
            return 9  # nostalgic
        elif probabilities[happy] > 0.15 and probabilities[calm] > 0.15 and probabilities[surprise] > 0.15:
            return 11  # grateful
        elif probabilities[sad] > 0.15 and probabilities[disgust] > 0.15 and probabilities[fear] > 0.15:
            return 12  # guilty
        elif probabilities[happy] > 0.15 and probabilities[surprise] > 0.15 and probabilities[neutral] > 0.15:
            return 13  # proud
        elif probabilities[sad] > 0.15 and probabilities[disgust] > 0.15 and probabilities[neutral] > 0.15:
            return 24  # remorseful

        # Depression and related states
        if probabilities[sad] > 0.25 and emotional_flatness > 0.7:
            return 10  # depressed
        elif probabilities[sad] > 0.3 and total_positive < 0.1:
            return 14  # grief
        elif probabilities[sad] > 0.25 and probabilities[fear] > 0.15 and total_positive < 0.15:
            return 15  # hopeless
        elif probabilities[neutral] > 0.25 and probabilities[sad] > 0.15 and emotional_flatness > 0.6:
            return 16  # apathetic
        elif probabilities[sad] > 0.2 and probabilities[angry] > 0.15 and total_positive < 0.1:
            return 17  # heartbroken
        elif probabilities[sad] > 0.25 and probabilities[neutral] > 0.15 and total_positive < 0.15:
            return 18  # lonely
        elif probabilities[sad] > 0.2 and probabilities[fear] > 0.15 and emotional_flatness > 0.6:
            return 19  # despondent
        elif emotional_flatness > 0.8 and probabilities[angry] < 0.15 and probabilities[disgust] < 0.15:
            return 20  # numb
        elif probabilities[sad] > 0.25 and probabilities[angry] > 0.15 and probabilities[fear] > 0.15:
            return 21  # devastated
        elif probabilities[sad] > 0.25 and probabilities[fear] > 0.15 and total_positive < 0.1:
            return 22  # desolate
        elif probabilities[sad] > 0.2 and probabilities[disgust] > 0.15 and total_positive < 0.1:
            return 23  # miserable
        elif probabilities[angry]>0.3 and probabilities[surprise]>0.1:
            return 25  #Pissed off

        # Default case with no reward
        return np.random.randint(25)

    def get_complex_emotion_name(self, index):
        """
        Get the name of the complex emotion based on its index
        """
        return self.complex_emotions[index]

    # Emotion mapping based on your trained model
    emotion_mapping = {
        "neutral": 0,
        "calm": 1,
        "happy": 2,
        "sad": 3,
        "angry": 4,
        "fear": 5,
        "disgust": 6,
        "surprise": 7
    }

    reverse_emotion_mapping = {v: k for k, v in emotion_mapping.items()}

    # Load the processor and model
    # model_dir = "E:/ser/wav2vec2-results/checkpoint-240"  # Replace with your checkpoint directory

    # if not os.path.exists(model_dir):
    #     raise FileNotFoundError(f"Model directory not found: {model_dir}")

    # processor = Wav2Vec2Processor.from_pretrained(model_dir)
    # model = Wav2Vec2ForSequenceClassification.from_pretrained(model_dir)
    # model.eval()

    # Load and preprocess the audio file
    def load_audio(self,file_path, target_sr=16000):
        try:
            audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
            return audio
        except Exception as e:
            print(f"Error loading audio file: {file_path}, Error: {e}")
            return None

    # Predict emotion from an audio file
    def predict_emotion(self,file_path):
        # Preprocess the audio
        audio = self.load_audio(file_path)
        if audio is None or len(audio) == 0:
            print(f"Audio preprocessing failed for: {file_path}")
            return None, None

        # Use the processor to prepare inputs for the model
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)

        # Forward pass through the model
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Extract logits and compute probabilities
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1).squeeze().numpy()

        # Get the predicted label
        predicted_label = np.argmax(probabilities)

        return predicted_label, probabilities

    def get_emotion_name(self,label):
        return self.reverse_emotion_mapping.get(label, "Unknown")

    def send_request(self,prompt):
        """
        Sends a text prompt to MetaAI and gets the response.

        :param prompt: The user input prompt to send to MetaAI.
        :return: The response from MetaAI.
        """
        try:
            ai = MetaAI()
            response = ai.prompt(message=prompt)
            return response.get('message', 'No response')
        except Exception as e:
            print(f"Error sending request to MetaAI: {e}")
            return None

    def get_voice_input(self,flag):
        """
        Captures continuous voice input from the microphone and converts it to text.

        :return: The recognized text from the voice input.
        """

        recognizer = sr.Recognizer()
        with sr.Microphone(sample_rate=16000) as source:
            SpeakButton.update_button_state("LISTENING")
            print("Listening... Please speak your prompt. Stop speaking to end recording.")
            try:
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source, timeout=None, phrase_time_limit=None)
                # # Test with a single audio file
                # audio = "angry1.wav"
                # audio = "neutral1.wav"
                # audio = "neutral_me.wav"
                # audio = "angry_me.wav"
                # audio = "fear_me.wav"
                # audio = "sad_me.wav"
                # audio = "disgust_me.wav"
                # raw_audio, original_sample_rate = sf.read(audio, dtype='int16')
                # print(original_sample_rate)
                # Replace with the path to your audio file
                # Convert AudioData to raw audio (NumPy array)

                # Convert speech to text
                text = recognizer.recognize_google(audio)

                raw_audio = np.frombuffer(audio.get_raw_data(), dtype=np.int16)
                # original_sample_rate = 44100  # Original microphone sample rate
                original_sample_rate = 16000
                # Save original audio
                # output_filename_original = "step1_original_audio.wav"
                # sf.write(output_filename_original, raw_audio, original_sample_rate)
                # print(f"Original audio saved to {output_filename_original}")
                
                # Convert to float32 for better processing
                raw_audio_float = raw_audio.astype(np.float32) / 32768.0
                
                # Apply noise reduction
                # First 1000ms of audio is usually silence - use it as noise profile
                noise_sample = raw_audio_float[:int(original_sample_rate)]
                reduced_noise = nr.reduce_noise(
                    y=raw_audio_float,
                    sr=original_sample_rate,
                    y_noise=noise_sample,  # Changed from noise_clip to y_noise
                    prop_decrease=1.0,
                    n_std_thresh_stationary=1.5,
                    stationary=True,
                    n_fft=2048,
                    win_length=2048,
                    hop_length=512,
                    n_jobs=-1
                )
                
                # Save the noise-reduced audio at original sample rate
                output_filename_clean = "step2_noise_reduced.wav"
                sf.write(output_filename_clean, reduced_noise, original_sample_rate)
                print(f"Noise-reduced audio saved to {output_filename_clean}")
                
                # Resample to 16kHz for Wav2Vec2
                target_sample_rate = 16000
                
                # Calculate resampling ratio
                resampling_ratio = target_sample_rate / original_sample_rate
                
                # Calculate new length
                new_length = int(len(reduced_noise) * resampling_ratio)
                
                # Resample using high-quality polyphase filtering
                resampled_audio = signal.resample(reduced_noise, new_length)
                
                # Save the final resampled audio
                output_filename_resampled = "step3_resampled_16k.wav"
                sf.write(output_filename_resampled, resampled_audio, target_sample_rate)
                print(f"Resampled audio saved to {output_filename_resampled}")

                if(flag==0):
                    # Predict emotion from raw audio
                    label, probabilities = self.predict_emotion_from_raw_audio(raw_audio, target_sample_rate)
                    if label is not None:
                        emotion = self.get_emotion_name(label)
                        print(f"Predicted Emotion: {emotion}")
                        print(f"Probabilities: {probabilities}")
                        predictor = ComplexEmotionPredictor()
                        probabilities = np.array(probabilities)
                        labels = list(self.emotion_mapping.keys())
                        # Plot the bar chart
                        plt.figure(figsize=(10, 5))
                        plt.bar(labels, probabilities, color='skyblue')
                        plt.xlabel("Emotions")
                        plt.ylabel("Probability")
                        plt.title("Emotion Probabilities from Classifier")
                        plt.ylim(0, 1)  # Probability values are between 0 and 1
                        plt.grid(axis='y', linestyle='--', alpha=0.7)
                        plt.savefig("emotion_probabilities.png")
                        plt.show()
                        complex_emotion_index = predictor.calculate_complex_emotion(probabilities)
                        self.complex_emotion_name = predictor.get_complex_emotion_name(complex_emotion_index)
                        print(f"Predicted complex emotion: {self.complex_emotion_name}")
                    else:
                        print("Failed to process the audio data.")
                
                
                SpeakButton.update_button_state("RESET")
                if(flag==0):
                    text = text + f" The emotional state of the speaker is: {self.complex_emotion_name}"
                print(f"Recognized Text: {text}")
                return text
            except sr.UnknownValueError:
                SpeakButton.update_button_state("RESET")
                print("Sorry, I could not understand your speech.")
                return None
            except sr.RequestError as e:
                print(f"Speech recognition service error: {e}")
                return None

    def predict_emotion_from_raw_audio(self,raw_audio, sample_rate):
        """
        Predicts the emotion from raw audio data.

        :param raw_audio: NumPy array containing raw audio data.
        :param sample_rate: The sample rate of the audio data.
        :return: Predicted label and probabilities.
        """
        try:
            # Normalize raw audio data
            audio_float = raw_audio.astype(np.float32) / np.iinfo(np.int16).max
            
            # Process the audio with Wav2Vec2Processor
            inputs = self.processor(audio_float, sampling_rate=sample_rate, return_tensors="pt", padding=True)
            
            # Forward pass through the model
            with torch.no_grad():
                outputs = self.model(**inputs)

            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1).squeeze().numpy()
            predicted_label = np.argmax(probabilities)
            return predicted_label, probabilities
        except Exception as e:
            print(f"Error during emotion prediction: {e}")
            return None, None

    def generate_speech_response(self,response_text):
        # """
        # Generates a speech file from the given text using TTS.

        # :param response_text: The text to be converted to speech.
        # """
        # try:
        #     # device = "cpu"
        #     # tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        #     output_path = "E:\ser\TTS\output_response.wav"
        #     # speaker_wav_path = "E:\\ser\\clone.wav"  # Update to the correct path for your speaker WAV file

        #     self.tts.tts_to_file(
        #         text=response_text,
        #         speaker_wav=self.speaker_wav_path,
        #         language="en",
        #         file_path=output_path
        #     )
        #     print(f"Speech response saved to {output_path}")
        # except Exception as e:
        #     print(f"Error generating speech response: {e}")
        # print("Process writing over.")
        
        try:
            text_chunks = [chunk.strip() for chunk in self.split_text_with_regex(response_text) if chunk.strip()]
            print(text_chunks)
            
            # Create a lock for thread-safe access to the TTS model
            tts_lock = threading.Lock()
            
            # Function that acquires the lock before using the TTS model
            def generate_with_lock(chunk, index):
                with tts_lock:
                    return self.generate_speech_chunk(chunk, index)
            
            chunk_paths = [None] * len(text_chunks)  # Initialize with None values
            
            # Process chunks in parallel with thread safety
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(text_chunks))) as executor:
                # Submit all tasks and store futures with their indices
                futures = {}
                for i, chunk in enumerate(text_chunks):
                    future = executor.submit(generate_with_lock, chunk, i)
                    futures[future] = i
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(futures):
                    index = futures[future]
                    try:
                        result = future.result()
                        if result:
                            chunk_paths[index] = result
                    except Exception as e:
                        print(f"Error processing chunk {index}: {e}")
            
            # Remove None values
            chunk_paths = [path for path in chunk_paths if path is not None]
            
            # Merge the generated speech files
            final_output_path = "E:\\ser\\TTS\\output_response.wav"
            self.merge_audio_chunks(chunk_paths, final_output_path)
        except Exception as e:
            print(f"Error in generate_speech_response: {e}")
        
        # try:
        #     # Split the text into chunks
        #     text_chunks = [chunk.strip() for chunk in self.split_text_with_regex(response_text) if chunk.strip()]
        #     print(text_chunks)
            
        #     # Create a class method that can be pickled (for ProcessPoolExecutor)
        #     def process_speech_chunk(args):
        #         chunk_text, chunk_index = args
        #         output_path = os.path.join(self.output_dir, f"chunk_{chunk_index}.wav")
                
        #         # Create a new TTS instance for this process
        #         device = "cuda" if torch.cuda.is_available() else "cpu"
        #         local_tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
                
        #         try:
        #             local_tts.tts_to_file(
        #                 text=chunk_text,
        #                 speaker_wav=self.speaker_wav_path,
        #                 language="en",
        #                 file_path=output_path
        #             )
        #             return (chunk_index, output_path)  # Return both index and path
        #         except Exception as e:
        #             print(f"Error generating chunk {chunk_index}: {e}")
        #             return (chunk_index, None)
            
        #     # Create a list of arguments for each chunk
        #     chunk_args = [(chunk, i) for i, chunk in enumerate(text_chunks)]
            
        #     # Use ThreadPoolExecutor since it's simpler and doesn't need pickling
        #     results = []
        #     with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        #         for result in executor.map(process_speech_chunk, chunk_args):
        #             results.append(result)
            
        #     # Sort results by index to maintain original order
        #     results.sort(key=lambda x: x[0])
            
        #     # Extract paths, filtering out None values
        #     chunk_paths = [path for _, path in results if path is not None]
            
        #     # Merge the audio files
        #     final_output_path = "E:\\ser\\TTS\\output_response.wav"
        #     if chunk_paths:
        #         self.merge_audio_chunks(chunk_paths, final_output_path)
        #         print(f"Final speech response saved to {final_output_path}")
        #     else:
        #         print("No speech chunks were successfully generated")
                
        # except Exception as e:
        #     print(f"Error in generate_speech_response: {e}")

    def save_chat_history(self,chat, file_name):
        """Save the chat history to a file."""
        with open(file_name, "a") as file:  # Append mode to add to the file
            file.write(chat + "\n")

    def retrieve_chat_history(self,file_name):
        """Retrieve chat history from a file."""
        if os.path.exists(file_name):
            with open(file_name, "r") as file:
                return file.read()
        return ""

    def split_text_with_regex(self, text):
        """Splits text into smaller chunks using regex."""
        return re.split(r'(?<=[.!?]) +', text)

    def generate_speech_chunk(self, text_chunk, index):
        """Generates speech for a text chunk and saves it as a temporary file."""
        chunk_path = os.path.join(self.output_dir, f"chunk_{index}.wav")
        try:
            self.tts.tts_to_file(
                text=text_chunk,
                speaker_wav=self.speaker_wav_path,
                language="en",
                file_path=chunk_path
            )
            return chunk_path  # Return the path of the generated file
        except Exception as e:
            print(f"Error generating speech chunk {index}: {e}")
            return None

    def merge_audio_chunks(self, chunk_paths, output_path):
        """Merges multiple audio chunks into a single audio file."""
        combined = AudioSegment.empty()
        for chunk_path in chunk_paths:
            try:
                audio = AudioSegment.from_wav(chunk_path)
                combined += audio
            except Exception as e:
                print(f"Error processing audio chunk {chunk_path}: {e}")

        # Export the final combined audio
        combined.export(output_path, format="wav")
        print(f"Final speech response saved to {output_path}")

        # Clean up temporary chunk files
        for chunk_path in chunk_paths:
            os.remove(chunk_path)
    
if __name__ == "__main__":
    # Get voice input
    speechInstance=ComplexEmotionPredictor()
    file=r"E:\ser\TTS\chat_history.txt"
    output_path = "E:\ser\TTS\output_response.wav"
    if os.path.exists(output_path):
            os.remove(output_path)
    chat_history = speechInstance.retrieve_chat_history(file)
    flag=0
    if(len(chat_history)!=0):
        flag=1
    user_prompt = speechInstance.get_voice_input(flag)
    if user_prompt==None:
        sys.exit()
    chat="User:"+user_prompt+"\nResponse:"
    if user_prompt:
        if(flag==0):
            modified_prompt = (
            "Respond like an emotional support agent and more like a human in a few setences not too long but moderate"
            f"This is the prompt: {user_prompt}."
        )
        else:
            modified_prompt = (
            "Respond like an emotional support agent and more like a human in a few setences not too long but moderate"
            f"This is the prompt: {user_prompt}."
            f"The previous conversations: {chat_history}"
        )
        response = speechInstance.send_request(modified_prompt)
        if response:
            print("Response:", response)
            chat=chat+response
            print(chat)
            speechInstance.save_chat_history(chat,file)
            # response1=speechInstance.split_text_with_regex(response)
            # print(response1)
            speechInstance.generate_speech_response(response)
        else:
            chat=""
            print("No response received from MetaAI.")   
    else:
        print("No valid input received.")

# sad
# Enter the prompt:i'm really depressed , no one really wants to talk to me and am having really bad social anxiety that i can't talk to people properly

# disgust
# Enter the prompt:Why is it that no matter how clearly I explain things, step by step, with examples and all the patience in the world
# , you still manage to completely ignore what I said and do the exact opposite