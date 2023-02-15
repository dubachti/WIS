import tkinter as tk
from tkinter import ttk
import sounddevice as sd
import numpy as np
from embedding import Embedding

# todos 
#   - catch if multiple recording with same name
#   - catch short recordings

class RecorderGUI:
    def __init__(self, master):

        self.master = master
        self.master.title("Voice Recorder")
        
        # Create GUI elements
        style = ttk.Style()
        style.configure("TLabel", font=("Helvetica", 18))
        style.configure(".", background="#edf0f2")
        style.map(".", background=[("active", "#0072d9")])
        
        self.record_button = ttk.Button(self.master, text="Record", command=self.record)
        self.record_button.pack(pady=20)

        self.stop_button = ttk.Button(self.master, text="Stop", command=self.stop, state=tk.DISABLED)
        self.stop_button.pack(pady=20)
        self.save_button = ttk.Button(self.master, text="Save", command=self.save, state=tk.DISABLED, style="TButton")
        self.save_button.pack(pady=20)
        
        self.filename_label = ttk.Label(self.master, text="Enter Name:", font=("Helvetica", 12))
        self.filename_label.pack(pady=20)
        self.filename_entry = ttk.Entry(self.master, font=("Helvetica", 12), background="white", foreground="black")
        self.filename_entry.pack(pady=0)

        self.predict_button = ttk.Button(self.master, text="Predict", command=self.predict, state=tk.DISABLED, style="TButton")
        self.predict_button.pack(pady=20)

        self.embediding_box = tk.Text(self.master, height=6, width=40, font=("Helvetica",12), background="white", state='normal')
        self.embediding_box.pack(expand=True, pady=20)
        self.embediding_box.insert('end', f"{'Index':<20} Name\n")
        self.embediding_box.configure(state='disabled')

        self.prediction_box = tk.Text(self.master, height=6, width=40, font=("Helvetica",12), background="white", state='disabled')
        self.prediction_box.pack(expand=True, pady=20)
        
        # Initialize recording variables
        self.recording = False
        self.frames = []

        # Initialite model
        self.embedding = Embedding()
    
    def record(self):
        self.recording = True
        self.frames = []
        
        # Start recording
        self.stream = sd.InputStream(callback=self.record_callback, channels=1, samplerate=22050)
        self.stream.start()
        
        # Update GUI elements
        self.record_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.save_button.config(state=tk.DISABLED)
        self.filename_entry.config(state=tk.DISABLED)
        self.predict_button.config(state=tk.DISABLED)
    
    def stop(self):
        self.recording = False
        
        # Stop recording
        self.stream.stop()
        self.stream.close()
        
        # Update GUI elements
        self.record_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.save_button.config(state=tk.NORMAL)
        self.filename_entry.config(state=tk.NORMAL)
        self.predict_button.config(state=tk.NORMAL)
    
    def save(self):
        name = self.filename_entry.get()
        y, sr = np.array(self.frames), self.stream.samplerate
        self.embedding.add_embedding(name, y, sr)
        
        self.embediding_box.configure(state='normal')
        self.embediding_box.insert('end', f"{self.embedding.num_embeddings():<20} {name}\n")
        self.embediding_box.configure(state='disabled')

    
    def record_callback(self, indata, frames, time, status):
        # Callback function for recording
        if self.recording:
            self.frames.extend(indata.copy())

    def predict(self):
        y, sr = np.array(self.frames), self.stream.samplerate
        name, distances = self.embedding.find_speaker(y, sr)

        distances.sort(key= lambda x: x[1])
        msg = ''.join(f"{n:<20} {val.item():5f} \n" for n, val in distances)

        self.prediction_box.configure(state='normal')
        self.prediction_box.delete('1.0','end')
        self.prediction_box.insert('end', f"{'Name':<20} L2 dist.\n")
        self.prediction_box.insert('end', msg)
        self.prediction_box.configure(state='disabled')

    
root = tk.Tk()
root.geometry("400x400")

app = RecorderGUI(root)
root.mainloop()
