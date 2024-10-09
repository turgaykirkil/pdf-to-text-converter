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
import traceback
import sqlite3

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# spaCy modelini yükleyelim (Türkçe model)
nlp = spacy.load("tr_core_news_trf")

class PDFConverterThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(str)
    progress_value = pyqtSignal(int)  # **Her sayfanın ilerlemesini iletmek için yeni sinyal**

    def __init__(self, pdf_path):
        super().__init__()
        self.pdf_path = pdf_path
        self.nlp = None

    def run(self):
        try:
            self.nlp = spacy.load("tr_core_news_trf")
            images = convert_from_path(self.pdf_path)
            output_text = ""

            total_pages = len(images)

            for i, image in enumerate(images):
                # Görseli numpy array'e dönüştür
                img_np = np.array(image)

                # Görseli gri tonlamaya çevir
                gray_image = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

                # Adaptive threshold uygula
                thresh_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

                # Sayfayı sütunlara ayır
                columns = self.split_into_columns(thresh_image)

                page_text = ""
                for col in columns:
                    # Her sütun için OCR uygula
                    col_text = pytesseract.image_to_string(col, lang='tur')
                    cleaned_text = self.clean_text(col_text)

                    # Ön işleme tabi tut (OCR hatalarını düzeltme)
                    preprocessed_text = self.preprocess_text(cleaned_text)

                    page_text += preprocessed_text + "\n\n"

                output_text += f"Sayfa {i+1} için çıkarılan metin:\n"
                output_text += page_text + "\n\n"

                # Progress signal
                progress_percentage = int((i + 1) / total_pages * 100)
                self.progress.emit(progress_percentage)
                self.progress_value.emit(progress_percentage)  # **Yeni sinyal**

            # Metni parse et ve veritabanına kaydet
            parsed_data = self.parse_text(output_text)
            self.save_to_database(parsed_data)

            txt_path = os.path.splitext(self.pdf_path)[0] + '_ocr_results.txt'
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(output_text)
            self.finished.emit(txt_path)

        except Exception as e:
            error_message = f"Hata: {e}\n{traceback.format_exc()}"
            print(error_message)
            self.finished.emit(error_message)

    def split_into_columns(self, image):
        height, width = image.shape
        mid = width // 2
        left_column = image[:, :mid]
        right_column = image[:, mid:]
        return [left_column, right_column]

    @staticmethod
    def clean_text(text):
        """Metindeki yasadışı karakterleri temizler."""
        ILLEGAL_CHARACTERS_RE = re.compile(r'[\000-\010]|[\013-\014]|[\016-\037]')
        return ILLEGAL_CHARACTERS_RE.sub('', text)

    @staticmethod
    def preprocess_text(text):
        """OCR hatalarını düzeltmek için metni ön işler."""
        corrections = {
            # OCR hataları düzeltmeleri (örneğin, 'l' yerine 'I' gibi)
        }
        for wrong, correct in corrections.items():
            text = text.replace(wrong, correct)
        return text

    # Metni parse eden fonksiyonlar
    def parse_text(self, text):
        """Metni parse eder ve verileri çıkarır."""
        # Metni ilanlara böl
        announcements = self.split_announcements(text)
        parsed_announcements = []
        for announcement in announcements:
            data = self.parse_announcement(announcement)
            parsed_announcements.append(data)
        return parsed_announcements

    def split_announcements(self, text):
        """Metni ilanlara böler."""
        pattern = r"(?:T\.C\.|TC)[ ]?.+?T[İI]CARET S[İI]C[İI]L[İI] M[ÜU]D[ÜU]RL[ÜU][ĞG][ÜU]['’]?[N]?[D]?EN"
        matches = list(re.finditer(pattern, text, re.DOTALL | re.IGNORECASE))
        announcements = []
        for i in range(len(matches)):
            start = matches[i].start()
            if i + 1 < len(matches):
                end = matches[i + 1].start()
            else:
                end = len(text)
            announcement_text = text[start:end].strip()
            announcements.append(announcement_text)
        return announcements

    def parse_announcement(self, announcement_text):
        """Her ilandan verileri çıkarır."""
        data = {}

        # Şehir
        city_match = re.search(
            r"(?:T\.C\.|TC)[ ]?(.+?)\s+T[İI][CÇ]ARET S[İI]C[İI]L[İI]\s+M[ÜU][DÐ][ÜU]RL[ÜU][ĞG][ÜU]['’]?[N]?[D]?EN",
            announcement_text,
            re.DOTALL | re.IGNORECASE
        )
        if city_match:
            data['city'] = city_match.group(1).strip()

        # İlan Sıra No
        ilan_no_match = re.search(
            r"İlan Sıra No\s*[:：]?\s*(\d+)",
            announcement_text,
            re.IGNORECASE
        )
        if ilan_no_match:
            data['ilan_sira_no'] = ilan_no_match.group(1).strip()

        # MERSİS No
        mersis_no_match = re.search(
            r"MERS[İI]S No\s*[:：]?\s*(\d+)",
            announcement_text,
            re.IGNORECASE
        )
        if mersis_no_match:
            data['mersis_no'] = mersis_no_match.group(1).strip()

        # Ticaret Sicil / Dosya No
        ticaret_sicil_no_match = re.search(
            r"Ticaret Sicil[ / Dosya]* No\s*[:：]?\s*(.+)",
            announcement_text,
            re.IGNORECASE
        )
        if ticaret_sicil_no_match:
            data['ticaret_sicil_no'] = ticaret_sicil_no_match.group(1).strip()

        # Ticaret Unvanı
        ticaret_unvani_match = re.search(
            r"Ticaret Unvan[ıi]\s*[:：]?\s*(.*?)(?:\n|$)([A-ZİĞÜŞÖÇ].+)",
            announcement_text,
            re.IGNORECASE
        )
        if ticaret_unvani_match:
            first_line = ticaret_unvani_match.group(1).strip()
            second_line = ticaret_unvani_match.group(2).strip()
            if first_line:
                data['ticaret_unvani'] = f"{first_line} {second_line}".strip()
            else:
                data['ticaret_unvani'] = second_line
        else:
            # Alternatif olarak, sadece bir satır sonraki ifadeyi al
            ticaret_unvani_match = re.search(
                r"Ticaret Unvan[ıi]\s*[:：]?\s*(.+)",
                announcement_text,
                re.IGNORECASE
            )
            if ticaret_unvani_match:
                data['ticaret_unvani'] = ticaret_unvani_match.group(1).strip()

        # Adres
        adres_match = re.search(
            r"Adres\s*[:：]?\s*(.+?)(?=\n\n|Yukarıda|Tescil Edilen Hususlar|$)",
            announcement_text,
            re.DOTALL | re.IGNORECASE
        )
        if adres_match:
            data['adres'] = adres_match.group(1).strip()

        # Tescil Edilen Hususlar
        tescil_hususlar_match = re.search(
            r"Tescil Edilen Hususlar\s*[:：]?\s*(.+?)(?:\n|$)",
            announcement_text,
            re.IGNORECASE
        )
        if tescil_hususlar_match:
            data['tescil_edilen_hususlar'] = tescil_hususlar_match.group(1).strip()

        # Tescile Delil Olan Belgeler
        tescil_belgeler_match = re.search(
            r"Tescile Delil Olan Belgeler\s*[:：]?\s*(.+?)(?=\n|$)",
            announcement_text,
            re.DOTALL | re.IGNORECASE
        )
        if tescil_belgeler_match:
            data['tescile_delil_olan_belgeler'] = tescil_belgeler_match.group(1).strip()

        # Detaylar
        details_match = re.search(
            r"(?:Tescil Edilen Hususlar.*?)(?:\n\n|\n)(.+)",
            announcement_text,
            re.DOTALL | re.IGNORECASE
        )
        if details_match:
            data['details'] = details_match.group(1).strip()
        else:
            data['details'] = ''

        # Şahıs Bilgilerini Çıkar
        data['persons'] = self.parse_persons(announcement_text)

        # Şirket ile ilgili maskelenmiş kimlik numarası ve ad-soyad varsa, bunları da persons'a ekle
        company_persons = self.extract_company_persons(announcement_text)
        if company_persons:
            if 'persons' not in data:
                data['persons'] = []
            data['persons'].extend(company_persons)

        # Özel durumlar için detaylı parça alma
        if 'KONKORDATO' in announcement_text.upper():
            data['konkordato'] = self.parse_konkordato_details(announcement_text)

        if 'PAY DEVRİ' in announcement_text.upper():
            data['pay_devri'] = self.parse_pay_devri_details(announcement_text)

        return data

    def parse_persons(self, announcement_text):
        """Şahısların kimlik numaraları, isimleri ve adreslerini çıkarır."""
        persons = []

        # Regex deseni
        pattern = r"Türkiye Cumhuriyeti Uyruklu\s+(\d{3}\*{4,6}\d{2,3} Kimlik No'lu),?\s*([A-ZÇŞĞÜÖİ\s/]+) adresinde ikamet eden,?\s*([A-ZÇŞĞÜÖİ\s']+)"
        matches = re.findall(pattern, announcement_text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            person_data = {
                'kimlik_no': match[0].strip(),
                'adres': match[1].strip(),
                'isim': match[2].strip()
            }
            persons.append(person_data)

        return persons

    def extract_company_persons(self, announcement_text):
        """Şirket ile ilgili maskelenmiş kimlik numarası ve ad-soyad bilgilerini çıkarır."""
        persons = []

        # Yönetim kurulu üyeleri ve diğer yetkilileri yakalamak için regex
        pattern = r"(\d{3}\*{4,6}\d{2,3} Kimlik No'lu),?\s*([A-ZÇŞĞÜÖİ\s/]+) adresinde ikamet eden,?\s*([A-ZÇŞĞÜÖİ\s']+)"
        matches = re.findall(pattern, announcement_text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            person_data = {
                'kimlik_no': match[0].strip(),
                'adres': match[1].strip(),
                'isim': match[2].strip()
            }
            persons.append(person_data)

        # Ayrıca, 'Kimlik Numaralı' ifadesiyle verilen kişileri yakala
        pattern2 = r"(\d{3}\*{4,6}\d{2,3} Kimlik Numaralı)\s+([A-ZÇŞĞÜÖİ\s']+)"
        matches2 = re.findall(pattern2, announcement_text, re.DOTALL | re.IGNORECASE)
        for match in matches2:
            person_data = {
                'kimlik_no': match[0].strip(),
                'isim': match[1].strip(),
                'adres': ''
            }
            persons.append(person_data)

        return persons

    def parse_konkordato_details(self, announcement_text):
        data = {}
        # Mahkeme Kararı Tarihi
        mahkeme_karari_tarihi_match = re.search(
            r"\d+\. ASL[İI]YE HUKUK MAHKEMES[İI]'n[ıi]n (\d{1,2}\.\d{1,2}\.\d{4}) tarihli karar[ıi] ile",
            announcement_text,
            re.IGNORECASE
        )
        if mahkeme_karari_tarihi_match:
            data['mahkeme_karari_tarihi'] = mahkeme_karari_tarihi_match.group(1)

        # Başlangıç Tarihi
        baslangic_tarihi_match = re.search(
            r"Başlangıç Tarihi\s*[:：]?\s*(.+)",
            announcement_text,
            re.IGNORECASE
        )
        if baslangic_tarihi_match:
            data['baslangic_tarihi'] = baslangic_tarihi_match.group(1).strip()

        # Bitiş Tarihi
        bitis_tarihi_match = re.search(
            r"Bitiş Tarihi\s*[:：]?\s*(.+)",
            announcement_text,
            re.IGNORECASE
        )
        if bitis_tarihi_match:
            data['bitis_tarihi'] = bitis_tarihi_match.group(1).strip()

        # Konkordato Komiseri
        komiser_match = re.search(
            r"(\d{3}\*{4,6}\d{2,3} Kimlik No'lu),?\s*([A-ZÇŞĞÜÖİ\s/]+) adresinde ikamet eden,?\s*([A-ZÇŞĞÜÖİ\s']+);\s*(\d{1,2}\.\d{1,2}\.\d{4}) tarihine kadar Konkordato Komiseri olarak atanmıştır",
            announcement_text,
            re.DOTALL | re.IGNORECASE
        )
        if komiser_match:
            data['komiser_kimlik_no'] = komiser_match.group(1).strip()
            data['komiser_adres'] = komiser_match.group(2).strip()
            data['komiser_adi'] = komiser_match.group(3).strip()
            data['komiser_gorev_bitis_tarihi'] = komiser_match.group(4).strip()

            # Konkordato komiserini persons listesine ekle
            data.setdefault('persons', []).append({
                'kimlik_no': data['komiser_kimlik_no'],
                'isim': data['komiser_adi'],
                'adres': data['komiser_adres']
            })

        return data

    def parse_pay_devri_details(self, announcement_text):
        data = {}

        # Devir İşlemi
        devir_eden_match = re.search(
            r"Şirket Ortaklarından (\d{3}\*{4,6}\d{2,3} Kimlik Numaralı) ([A-ZÇŞĞÜÖİ\s']+) (\d[\d\.,]+ TL) sermaye karşılığı (\d+) adet payını hukuki ve mali yükümlülükleri ile (\d{3}\*{4,6}\d{2,3} Kimlik Numaralı) ([A-ZÇŞĞÜÖİ\s']+)'e devretmiştir",
            announcement_text,
            re.DOTALL | re.IGNORECASE
        )
        if devir_eden_match:
            data['devir_eden_kimlik_no'] = devir_eden_match.group(1).strip()
            data['devir_eden_adi'] = devir_eden_match.group(2).strip()
            data['devredilen_tutar'] = devir_eden_match.group(3)
            data['devredilen_pay_adedi'] = devir_eden_match.group(4)
            data['devir_alici_kimlik_no'] = devir_eden_match.group(5).strip()
            data['devir_alici_adi'] = devir_eden_match.group(6).strip()

            # Devir eden ve alan kişileri persons tablosuna ekle
            data.setdefault('persons', []).extend([
                {
                    'kimlik_no': data['devir_eden_kimlik_no'],
                    'isim': data['devir_eden_adi'],
                    'adres': ''
                },
                {
                    'kimlik_no': data['devir_alici_kimlik_no'],
                    'isim': data['devir_alici_adi'],
                    'adres': ''
                }
            ])

        # Yeni Ortaklık Yapısı
        shareholding_matches = re.findall(
            r"([A-ZÇŞĞÜÖİ\s']+)\s*:\s*Beheri ([\d\.,]+) Türk Lirası değerinde (\d+) adet paya karşılık gelen ([\d\.,]+) Türk Lirası",
            announcement_text,
            re.DOTALL | re.IGNORECASE
        )
        shareholders = []
        for match in shareholding_matches:
            shareholder = {
                'adi': match[0].strip(),
                'beher_pay_degeri': match[1],
                'pay_adedi': match[2],
                'toplam_tutar': match[3]
            }
            shareholders.append(shareholder)
            # Ortakları persons tablosuna ekle
            data.setdefault('persons', []).append({
                'kimlik_no': '',  # Kimlik numarası yoksa boş bırakıyoruz
                'isim': shareholder['adi'],
                'adres': ''
            })
        data['shareholders'] = shareholders

        return data

    # Veritabanına kayıt eden fonksiyon
    def save_to_database(self, parsed_announcements):
        """Verileri veritabanına kaydeder."""
        conn = sqlite3.connect('company_records.db')
        cursor = conn.cursor()

        # Tablo oluşturma
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS companies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mersis_no TEXT,
            ticaret_sicil_no TEXT,
            ticaret_unvani TEXT,
            adres TEXT,
            city TEXT
        )
        ''')

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS announcements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_id INTEGER,
            ilan_sira_no TEXT,
            tescil_edilen_hususlar TEXT,
            tescile_delil_olan_belgeler TEXT,
            details_text TEXT,
            FOREIGN KEY(company_id) REFERENCES companies(id)
        )
        ''')

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS persons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_id INTEGER,
            kimlik_no TEXT,
            isim TEXT,
            adres TEXT,
            FOREIGN KEY(company_id) REFERENCES companies(id)
        )
        ''')

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS konkordato (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_id INTEGER,
            mahkeme_karari_tarihi TEXT,
            baslangic_tarihi TEXT,
            bitis_tarihi TEXT,
            komiser_kimlik_no TEXT,
            komiser_adres TEXT,
            komiser_adi TEXT,
            komiser_gorev_bitis_tarihi TEXT,
            FOREIGN KEY(company_id) REFERENCES companies(id)
        )
        ''')

        for data in parsed_announcements:
            # Şirketi kaydet
            cursor.execute('''
            INSERT INTO companies (mersis_no, ticaret_sicil_no, ticaret_unvani, adres, city)
            VALUES (?, ?, ?, ?, ?)
            ''', (
                data.get('mersis_no'),
                data.get('ticaret_sicil_no'),
                data.get('ticaret_unvani'),
                data.get('adres'),
                data.get('city')
            ))
            company_id = cursor.lastrowid

            # İlanı kaydet
            cursor.execute('''
            INSERT INTO announcements (company_id, ilan_sira_no, tescil_edilen_hususlar, tescile_delil_olan_belgeler, details_text)
            VALUES (?, ?, ?, ?, ?)
            ''', (
                company_id,
                data.get('ilan_sira_no'),
                data.get('tescil_edilen_hususlar'),
                data.get('tescile_delil_olan_belgeler'),
                data.get('details')
            ))

            # Şahısları kaydet
            if 'persons' in data:
                for person in data['persons']:
                    cursor.execute('''
                    INSERT INTO persons (company_id, kimlik_no, isim, adres)
                    VALUES (?, ?, ?, ?)
                    ''', (
                        company_id,
                        person.get('kimlik_no'),
                        person.get('isim'),
                        person.get('adres')
                    ))

            # Konkordato detaylarını kaydet
            if 'konkordato' in data:
                konkordato = data['konkordato']
                cursor.execute('''
                INSERT INTO konkordato (company_id, mahkeme_karari_tarihi, baslangic_tarihi, bitis_tarihi, komiser_kimlik_no, komiser_adres, komiser_adi, komiser_gorev_bitis_tarihi)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    company_id,
                    konkordato.get('mahkeme_karari_tarihi'),
                    konkordato.get('baslangic_tarihi'),
                    konkordato.get('bitis_tarihi'),
                    konkordato.get('komiser_kimlik_no'),
                    konkordato.get('komiser_adres'),
                    konkordato.get('komiser_adi'),
                    konkordato.get('komiser_gorev_bitis_tarihi')
                ))

        conn.commit()
        conn.close()

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
        self.movie = QMovie("path_to_gif.gif")  # Doğru yolu ekleyin
        self.dancing_label.setMovie(self.movie)
        self.dancing_label.setVisible(False)
        layout.addWidget(self.dancing_label)

        # Ana widget
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.output_file_path = None

        # Thread'leri tutmak için bir liste
        self.threads = []

        # **Toplam ilerlemeyi takip etmek için değişkenler**
        self.total_progress = 0
        self.current_pdf_progress = 0

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

        # **Toplam ilerlemeyi sıfırlıyoruz**
        self.total_progress = 0

        # İlk PDF dönüştürme işlemi için yeni thread başlat
        self.convert_pdf(pdf_paths[self.current_file_index], pdf_paths)

    def convert_pdf(self, pdf_path, pdf_paths):
        thread = PDFConverterThread(pdf_path)
        thread.progress.connect(self.update_progress)
        # thread'i parametre olarak geçiyoruz
        thread.finished.connect(lambda result, t=thread: self.conversion_finished(result, pdf_paths, t))
        thread.start()
        # Thread'i listeye ekliyoruz
        self.threads.append(thread)

        # **Şu anki PDF'nin ilerlemesini sıfırlıyoruz**
        self.current_pdf_progress = 0

        # **Durum bilgisini güncelliyoruz**
        self.status_label.setText(f"İşlenen dosya: {os.path.basename(pdf_path)} ({self.current_file_index + 1}/{self.total_files})")

    def update_progress(self, value):
        # **Şu anki PDF'nin ilerlemesini güncelliyoruz**
        self.current_pdf_progress = value

        # **Toplam ilerlemeyi hesaplıyoruz**
        total_progress = ((self.current_file_index + self.current_pdf_progress / 100) / self.total_files) * 100
        self.progress_bar.setValue(int(total_progress))
        self.progress_label.setText(f"%{int(total_progress)} Tamamlandı")

    def conversion_finished(self, result, pdf_paths, thread):
        # Thread'i listeden kaldırıyoruz
        self.threads.remove(thread)

        if result.endswith('.txt'):
            self.output_file_paths.append(result)
            # **Her dosya işlendiğinde durum bilgisini güncelliyoruz**
            # self.status_label.setText(f"Dönüştürme tamamlandı. Sonuç: {result}")
            self.output_file_path = result
            # self.open_file_button.setVisible(True)
        else:
            self.status_label.setText(f"Hata oluştu: {result}")
            print(result)  # Hata mesajını da ekrana yazdır

        # Bir sonraki dosyayı dönüştürme
        self.current_file_index += 1
        if self.current_file_index < self.total_files:
            self.convert_pdf(pdf_paths[self.current_file_index], pdf_paths)
        else:
            # Tüm dosyalar işlendiğinde UI elemanlarını güncelliyoruz
            self.progress_bar.setVisible(False)
            self.progress_label.setVisible(False)
            self.select_button.setEnabled(True)
            self.dancing_label.setVisible(False)
            self.movie.stop()
            self.status_label.setText("Tüm dosyaların dönüştürme işlemi tamamlandı.")
            self.open_file_button.setVisible(True)

    def open_file(self):
        # Çıktı dosyalarını platforma göre aç
        if self.output_file_paths:
            for file_path in self.output_file_paths:
                if os.path.exists(file_path):
                    if sys.platform == "win32":
                        os.startfile(file_path)
                    elif sys.platform == "darwin":
                        subprocess.call(["open", file_path])
                    else:
                        subprocess.call(["xdg-open", file_path])

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
