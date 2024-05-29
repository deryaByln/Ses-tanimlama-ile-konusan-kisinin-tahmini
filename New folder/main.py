import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import speech_recognition as sr
import pyaudio
import wave
import os
from pydub import AudioSegment
import threading

import matplotlib.pyplot as plt

def extract_mfcc(audio_file):
    """ Ses dosyasından MFCC özelliklerini çıkarır ve histogram oluşturur. """
    try:
        y, sr = librosa.load(audio_file)

        # Ses ön işleme
        # Örnek: Gürültü azaltma (Ortalama çıkarma)
        y = y - np.mean(y)
        
        # MFCC özelliklerini çıkar
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs = np.mean(mfccs.T, axis=0)

        # Histogram oluştur ve kaydet
        plt.figure(figsize=(6, 4))
        plt.hist(mfccs, bins=20)
        plt.title(f"MFCC Histogramı - {os.path.basename(audio_file)}")
        plt.xlabel("MFCC Değerleri")
        plt.ylabel("Frekans")
        plt.savefig(f"{audio_file[:-4]}_histogram.png")
        plt.close()  # Figure'ü kapat

        return mfccs
    except Exception as e:
        messagebox.showerror("Hata", f"Ses dosyası işlenirken bir hata oluştu: {e}")
        return None

def train_model():
    """ SVM modelini eğitir. """
    X = []  # MFCC özellikleri
    y = []  # Konuşmacı etiketleri

    # "_Ses" ile biten klasörleri bul
    speaker_folders = [
        f for f in os.listdir('.') if os.path.isdir(f) and f.endswith("_Ses")
    ]

    # Konuşmacıları sırayla derya, melih, dilara olarak belirle
    speaker_mapping = {"derya": 0, "melih": 1, "dilara": 2}
    for folder_name in speaker_folders:
        speaker_name = folder_name[:-4].lower()  # "_Ses" kısmını kaldır ve küçük harfe çevir
        folder_path = os.path.join(".", folder_name)

        for filename in os.listdir(folder_path):
            if filename.endswith(".wav"):
                file_path = os.path.join(folder_path, filename)
                mfccs = extract_mfcc(file_path)
                if mfccs is not None:
                    X.append(mfccs)
                    y.append(speaker_mapping[speaker_name])
                    print(f"Dosya: {filename}, Konuşmacı: {speaker_name}")
    if len(X) > 0:
        # Veri setini eğitim ve test kümelerine ayır
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # SVM modelini eğit
        svm_model = SVC(kernel='linear')
        svm_model.fit(X_train, y_train)

        # Test verileri üzerinde modelin doğruluğunu değerlendir
        y_pred = svm_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Sınıflandırma raporunu al
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # F1-Score'u (macro average) al
        f1_score = report['macro avg']['f1-score']

        messagebox.showinfo("Model Eğitimi", f"Model başarıyla eğitildi.\nDoğruluk (ACC): {accuracy}\nF1-Score (FM): {f1_score}")
        return svm_model
    else:
        messagebox.showerror("Hata", "Model eğitimi için yeterli veri bulunamadı.")
        return None

def transcribe_audio(audio_file):
    """ Sesi metne dönüştürür. """
    r = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = r.record(source)
        try:
            text = r.recognize_google(audio, language="tr-TR")
            return text
        except sr.UnknownValueError:
            return "Google Speech Recognition sesi anlayamadı"
        except sr.RequestError as e:
            return f"Google Speech Recognition isteğini işleyemiyor; {e}"

def count_words(text):
    """ Metindeki kelime sayısını hesaplar. """
    words = text.split()
    return len(words)

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Konuşmacı Tanıma Sistemi")
        self.geometry("600x400")
        self.configure(bg='#f0f0f0')
        self.svm_model = None
        self.recording = False
        self.audio_frames = []
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.create_widgets()

    def create_widgets(self):
        # Sol taraftaki butonlar
        left_frame = tk.Frame(self, bg="#f0f0f0")
        left_frame.pack(side=tk.LEFT, padx=20, pady=20)

        self.train_button = tk.Button(left_frame, text="Modeli Eğit", command=self.train_model, width=15, height=2, relief=tk.GROOVE, borderwidth=3)
        self.train_button.pack(pady=10)

        self.record_button = tk.Button(left_frame, text="Kaydı Başlat", command=self.start_recording, width=15, height=2, relief=tk.GROOVE, borderwidth=3)
        self.record_button.pack(pady=10)

        self.stop_button = tk.Button(left_frame, text="Kaydı Durdur", command=self.stop_recording, state="disabled", width=15, height=2, relief=tk.GROOVE, borderwidth=3)
        self.stop_button.pack(pady=10)

        self.analyze_button = tk.Button(left_frame, text="Sesi Analiz Et", command=self.analyze_recording, state="disabled", width=15, height=2, relief=tk.GROOVE, borderwidth=3)
        self.analyze_button.pack(pady=10)

        # Sağ taraftaki sonuçlar
        right_frame = tk.Frame(self, bg="#f0f0f0")
        right_frame.pack(side=tk.RIGHT, padx=20, pady=20)

        self.speaker_label = tk.Label(right_frame, text="Konuşmacı: ", bg="#f0f0f0", font=("Arial", 14))
        self.speaker_label.pack(pady=10)

        self.word_count_label = tk.Label(right_frame, text="Kelime Sayısı: ", bg="#f0f0f0", font=("Arial", 10)) 
        self.word_count_label.pack(pady=10)

        self.progress = ttk.Progressbar(right_frame, orient="horizontal", length=250, mode="determinate")
        self.progress.pack(pady=10)

        self.text_label = tk.Text(right_frame, wrap=tk.WORD, width=40, height=5)  # Metin kutusu
        self.text_label.pack(pady=10)

    def train_model(self):
        self.progress['value'] = 0
        self.update()
        self.svm_model = train_model()
        self.progress['value'] = 100
        self.update()
        if self.svm_model:
            messagebox.showinfo("Başarılı", "Model eğitimi tamamlandı!")

    def start_recording(self):
        self.recording = True
        self.audio_frames = []
        self.stream = self.p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)
        self.record_button.config(state="disabled")
        self.stop_button.config(state="normal")
        threading.Thread(target=self.record).start()

    def record(self):
        while self.recording:
            data = self.stream.read(1024)
            self.audio_frames.append(data)

    def stop_recording(self):
        self.recording = False
        self.stream.stop_stream()
        self.stream.close()
        self.record_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.analyze_button.config(state="normal")
        wf = wave.open("kaydedilen_ses.wav", 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(44100)
        wf.writeframes(b''.join(self.audio_frames))
        wf.close()
        messagebox.showinfo("Başarılı", "Kayıt tamamlandı!")

    def analyze_recording(self):
        if self.svm_model:
            try:
                self.progress['value'] = 0
                self.update()
                speaker_id = self.predict_speaker("kaydedilen_ses.wav")
                if speaker_id is not None:
                    self.progress['value'] = 33
                    self.update()
                    speaker_names = {0: "derya", 1: "melih", 2: "dilara"}
                    self.speaker_label.config(text=f"Konuşmacı: {speaker_names[speaker_id]}")
                    self.progress['value'] = 66
                    self.update()

                    text = transcribe_audio("kaydedilen_ses.wav")
                    self.text_label.delete("1.0", tk.END)  # Önceki metni sil
                    self.text_label.insert(tk.END, text) # Metni ekle
                    self.progress['value'] = 100
                    self.update()

                    # Kelime sayımını ayrı bir thread'de başlat
                    threading.Thread(target=self.count_and_display_words, args=(text,)).start()

                else:
                    messagebox.showerror("Hata", "Ses dosyası analiz edilemedi.")
                    self.progress['value'] = 100
                    self.update()
            except Exception as e:
                messagebox.showerror("Hata", f"Ses dosyası analiz edilirken bir hata oluştu: {e}")
                self.progress['value'] = 100
                self.update()
        else:
            messagebox.showwarning("Uyarı", "Lütfen önce modeli eğitin.")
            
    def count_and_display_words(self, text):
        """ Kelime sayısını hesaplar ve GUI'de görüntüler. """
        word_count = count_words(text)
        self.word_count_label.config(text=f"Kelime Sayısı: {word_count}")

    def predict_speaker(self, audio_file):
        if self.svm_model:
            try:
                mfccs = extract_mfcc(audio_file)
                if mfccs is not None:
                    speaker_id = self.svm_model.predict([mfccs])[0]
                    return speaker_id
                else:
                    return None
            except Exception as e:
                messagebox.showerror("Hata", f"Ses dosyası tahmin edilirken bir hata oluştu: {e}")
                return None
        else:
            messagebox.showwarning("Uyarı", "Lütfen önce modeli eğitin.")
            return None

if __name__ == "__main__":
    app = App()
    app.mainloop()
