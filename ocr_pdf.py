import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QFileDialog, QLabel, QProgressBar
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QMovie
from pdf2image import convert_from_path
import pytesseract
import re
import subprocess
import spacy
import cv2
import numpy as np
import traceback  # Hata izlemeleri için eklenen modül

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# spaCy modelini yükleyelim (Türkçe transformer tabanlı model)
nlp = spacy.load("tr_core_news_trf")

class PDFConverterThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(str)

    def __init__(self, pdf_path):
        super().__init__()
        self.pdf_path = pdf_path
        self.nlp = None

    def run(self):
        try:
            # spaCy modelini burada yükleyin
            self.nlp = spacy.load("tr_core_news_trf")
            
            images = convert_from_path(self.pdf_path)
            output_text = ""

            for i, image in enumerate(images):
                # Görseli gri tonlamaya çevirelim (önce numpy array'e dönüştürülmeli)
                gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

                # Görseli daha iyi OCR için adaptive threshold uygulayalım
                thresh_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

                text = pytesseract.image_to_string(thresh_image, lang='tur')
                cleaned_text = self.clean_text(text)

                # spaCy ile metin işleme
                doc_spacy = self.nlp(cleaned_text)
                corrected_text = " ".join([token.text for token in doc_spacy])

                output_text += f"Sayfa {i+1} için çıkarılan metin:\n"
                output_text += corrected_text + "\n\n"

                # Progress signal
                self.progress.emit(int((i + 1) / len(images) * 100))

                # Görseli de kaydedelim
                image_path = os.path.splitext(self.pdf_path)[0] + f'_page_{i+1}.jpg'
                image.save(image_path, 'JPEG')

            txt_path = os.path.splitext(self.pdf_path)[0] + '_ocr_results.txt'
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(output_text)
            self.finished.emit(txt_path)

        except Exception as e:
            # Hatanın detaylı bilgisini almak için traceback kullanıyoruz
            error_message = f"Hata: {e}\n{traceback.format_exc()}"
            print(error_message)  # Hata mesajını konsola yazdır
            self.finished.emit(error_message)

    @staticmethod
    def clean_text(text):
        """Metindeki yasadışı karakterleri temizler."""
        ILLEGAL_CHARACTERS_RE = re.compile(r'[\000-\010]|[\013-\014]|[\016-\037]')
        return ILLEGAL_CHARACTERS_RE.sub('', text)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PDF to Word Converter")
        self.setGeometry(100, 100, 400, 250)

        # Layout oluşturma
        layout = QVBoxLayout()

        # Butonlar ve label ekleme
        self.select_button = QPushButton("PDF Dosyası Seç")
        self.select_button.clicked.connect(self.select_pdf)
        layout.addWidget(self.select_button)

        self.status_label = QLabel("Henüz bir dosya seçilmedi.")
        layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.progress_label = QLabel()
        self.progress_label.setVisible(False)
        layout.addWidget(self.progress_label)

        self.open_file_button = QPushButton("Dosyayı Aç")
        self.open_file_button.clicked.connect(self.open_file)
        self.open_file_button.setVisible(False)
        layout.addWidget(self.open_file_button)

        self.dancing_label = QLabel(self)
        self.dancing_label.setAlignment(Qt.AlignCenter)
        self.movie = QMovie("path_to_gif.gif")  # Add the correct path to your GIF file
        self.dancing_label.setMovie(self.movie)
        self.dancing_label.setVisible(False)
        layout.addWidget(self.dancing_label)

        # Ana widget
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.output_file_path = None

    def select_pdf(self):
        pdf_paths, _ = QFileDialog.getOpenFileNames(self, "Bir veya daha fazla PDF dosyası seçin", "", "PDF files (*.pdf)")
        if pdf_paths:
            self.status_label.setText(f"{len(pdf_paths)} dosya seçildi")
            self.convert_pdfs(pdf_paths)

    def convert_pdfs(self, pdf_paths):
        # UI kontrolü
        self.progress_bar.setVisible(True)
        self.progress_label.setVisible(True)
        self.select_button.setEnabled(False)
        self.open_file_button.setVisible(False)
        self.dancing_label.setVisible(True)
        self.movie.start()

        self.output_file_paths = []
        self.total_files = len(pdf_paths)
        self.current_file_index = 0

        # İlk PDF dönüştürme işlemi için yeni thread başlat
        self.convert_pdf(pdf_paths[self.current_file_index], pdf_paths)

    def convert_pdf(self, pdf_path, pdf_paths):
        self.thread = PDFConverterThread(pdf_path)
        self.thread.progress.connect(self.update_progress)
        self.thread.finished.connect(lambda result: self.conversion_finished(result, pdf_paths))
        self.thread.start()

    def update_progress(self, value):
        self.progress_bar.setValue(value)
        self.progress_label.setText(f"%{value} Tamamlandı")

    def conversion_finished(self, result, pdf_paths):
        # UI kontrolü
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        self.select_button.setEnabled(True)
        self.dancing_label.setVisible(False)
        self.movie.stop()

        if result.endswith('.txt'):
            self.output_file_paths.append(result)
            self.status_label.setText(f"Dönüştürme tamamlandı. Sonuç: {result}")
            self.output_file_path = result
            self.open_file_button.setVisible(True)
        else:
            self.status_label.setText(f"Hata oluştu: {result}")
            print(result)  # Hata mesajını da ekrana yazdır

        # Bir sonraki dosyayı dönüştürme
        self.current_file_index += 1
        if self.current_file_index < self.total_files:
            self.convert_pdf(pdf_paths[self.current_file_index], pdf_paths)

    def open_file(self):
        # Çıktı dosyasını platforma göre aç
        if self.output_file_path and os.path.exists(self.output_file_path):
            if sys.platform == "win32":
                os.startfile(self.output_file_path)
            elif sys.platform == "darwin":
                subprocess.call(["open", self.output_file_path])
            else:
                subprocess.call(["xdg-open", self.output_file_path])

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
