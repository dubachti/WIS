import tkinter as tk
from tkinter import ttk
import sounddevice as sd
import numpy as np
from embedding import Embedding

class RecorderGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Speaker Recognition")
        style = ttk.Style()
        style.configure("TLabel", font=("Helvetica", 18))
        style.configure(".", background="#edf0f2")
        style.map(".", background=[("active", "#0072d9")])
        
        # record
        self.record_button = ttk.Button(self.master, text="Record", command=self.record)
        self.record_button.grid(row=0, column=0)

        # stop
        self.stop_button = ttk.Button(self.master, text="Stop", command=self.stop, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=1)

        # name entry
        self.filename_label = ttk.Label(self.master, text="Enter Name:", font=("Helvetica", 12))
        self.filename_label.grid(row=1, column=0)
        self.filename_entry = ttk.Entry(self.master, font=("Helvetica", 12), background="white", foreground="black")
        self.filename_entry.grid(row=1, column=1)

        # save
        self.save_button = ttk.Button(self.master, text="Save", command=self.save, state=tk.DISABLED, style="TButton")
        self.save_button.grid(row=2, column=0)

        # predict
        self.predict_button = ttk.Button(self.master, text="Predict", command=self.predict, state=tk.DISABLED, style="TButton")
        self.predict_button.grid(row=2, column=1)

        # list of speakers
        self.embediding_box_label = ttk.Label(self.master, text="Known Speakers:", font=("Helvetica", 12))
        self.embediding_box_label.grid(row=3, column=0)
        self.embediding_box = tk.Text(self.master, height=6, width=40, font=("Helvetica",12), background="white", state='disabled')
        self.embediding_box.grid(row=4, column=0, columnspan=2)

        # output
        self.prediction_box_label = ttk.Label(self.master, text="Output:", font=("Helvetica", 12))
        self.prediction_box_label.grid(row=5, column=0)
        self.prediction_box = tk.Text(self.master, height=6, width=40, font=("Helvetica",12), background="white", state='disabled')
        self.prediction_box.grid(row=6, column=0, columnspan=2)
        
        # Initialize recording variables
        self.recording = False
        self.frames = []

        # Initialize model
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
        if self.embedding.num_embeddings():
            self.predict_button.config(state=tk.NORMAL)
    
    def save(self):
        name = self.filename_entry.get()
        y, sr = np.array(self.frames), self.stream.samplerate
        if len(y) < 65536: return # recording not long enough
        self.embedding.add_embedding(name, y, sr)
        
        self.embediding_box.configure(state='normal')
        self.embediding_box.insert('end', f"{self.embedding.num_embeddings()} \t {name}\n")
        self.embediding_box.configure(state='disabled')

    
    def record_callback(self, indata, frames, time, status):
        # Callback function for recording
        if self.recording:
            self.frames.extend(indata.copy())

    def predict(self):
        y, sr = np.array(self.frames), self.stream.samplerate
        if len(y) < 65536: 
            name = '-'
            msg = '--- recording not long enough ---'
        else:
            distances = self.embedding.find_speaker(y, sr)
            name = distances[0][0]
            msg = ''.join(f"{n} \t {val.item():.3f} \n" for n, val in distances)

        self.prediction_box.configure(state='normal')
        self.prediction_box.delete('1.0','end')
        self.prediction_box.insert('end', f"Prediction: \t {name}\n \n")
        self.prediction_box.insert('end', f"Name \t L2 dist.\n")
        self.prediction_box.insert('end', msg)
        self.prediction_box.configure(state='disabled')

def main(): 
    root = tk.Tk()
    root.geometry("290x305")
    RecorderGUI(root)
    root.mainloop()

if __name__ == '__main__': main()
