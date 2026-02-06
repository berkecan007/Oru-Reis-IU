import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
import random
import numpy as np
import threading
import time
from collections import deque
import cv2
from PIL import Image, ImageTk, ImageDraw
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# Harita kütüphanesi kontrolü
try:
    import tkintermapview  # type: ignore
except ImportError:
    tkintermapview = None

class SystemControlInterface:
    def __init__(self, root):
        self.root = root
        self.root.title("SUALTI ARACI SİSTEM KONTROL ARAYÜZÜ")
        self.root.geometry("1300x850")
        self.root.configure(bg="#1a1a2e")
        
        # Harita konumu için değişkenler
        self.map_widget = None
        self.map_marker = None
        self.map_path = None
        self.location_points = deque([], maxlen=400)
        self.location_status_var = tk.StringVar(value="Konum simülasyonu hazır.")
        self.vehicle_icon = self.create_vehicle_icon()
        
        # Veri depoları
        self.pressure_data = deque([1013.25] * 50, maxlen=100)  # hPa
        self.depth_data = deque([0] * 50, maxlen=100)  # metre
        self.time_data = deque(range(50), maxlen=100)
        
        # Kamera başlatma
        self.camera_active = False
        self.cap = None
        
        # Ana konteyner
        self.main_container = tk.Frame(root, bg="#1a1a2e")
        self.main_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Üst başlık
        self.create_header()
        
        # Ana içerik alanı
        self.create_main_content()
        
        # Alt bilgi çubuğu
        self.create_footer()
        
        # Veri güncelleme başlat
        self.update_time()
        self.start_sensor_simulation()
        self.start_location_updates()
        self.init_camera()
    
    def create_header(self):
        header_frame = tk.Frame(self.main_container, bg="#162447", height=70)
        header_frame.pack(fill="x", pady=(0, 10))
        
        # Sol tarafta başlık
        title_label = tk.Label(header_frame, text="⚓ SUALTI ARACI KONTROL SİSTEMİ", 
                             font=("Arial", 18, "bold"), 
                             bg="#162447", fg="#00ffff")
        title_label.pack(side="left", padx=20)
        
        # Sağ tarafta saat ve durum
        status_frame = tk.Frame(header_frame, bg="#162447")
        status_frame.pack(side="right", padx=20)
        
        self.time_label = tk.Label(status_frame, text="", 
                                 font=("Arial", 12, "bold"), 
                                 bg="#162447", fg="#00ff00")
        self.time_label.pack(side="right", padx=(10, 0))
        
        self.status_indicator = tk.Label(status_frame, text="● ÇALIŞIYOR", 
                                       font=("Arial", 11, "bold"),
                                       bg="#162447", fg="#00ff00")
        self.status_indicator.pack(side="right", padx=10)
    
    def create_main_content(self):
        content_frame = tk.Frame(self.main_container, bg="#1a1a2e")
        content_frame.pack(fill="both", expand=True)
        
        # Sol Panel - Sistem Detayları ve Grafikler
        self.create_left_panel(content_frame)
        
        # Orta Panel - Kamera Görüntüsü ve Kontroller
        self.create_center_panel(content_frame)
        
        # Sağ Panel - Motor Kontrol
        self.create_right_panel(content_frame)
    
    def create_left_panel(self, parent):
        left_frame = tk.Frame(parent, bg="#0f3460", width=350,
                            relief="ridge", borderwidth=2)
        left_frame.pack(side="left", fill="y", padx=(0, 10))
        left_frame.pack_propagate(False)
        
        # Sistem Detayları
        sys_details_frame = tk.LabelFrame(left_frame, text="⚙️ SİSTEM DURUMU", 
                                        font=("Arial", 12, "bold"),
                                        bg="#0f3460", fg="#e6e6e6",
                                        padx=15, pady=15)
        sys_details_frame.pack(fill="x", padx=10, pady=10)
        
        # Gerçek sensör değerleri (simülasyon)
        self.sensor_values = {
            "sıcaklık": tk.StringVar(value="24.5°C"),
            "nem": tk.StringVar(value="45%"),
            "ivme_x": tk.StringVar(value="0.12g"),
            "ivme_y": tk.StringVar(value="0.08g"),
            "ivme_z": tk.StringVar(value="0.95g"),
            "manyetik": tk.StringVar(value="52.3µT"),
            "gyro": tk.StringVar(value="0.05°/s")
        }
        
        details = [
            ("🌡️ Sıcaklık:", self.sensor_values["sıcaklık"]),
            ("💧 Nem:", self.sensor_values["nem"]),
            ("📈 İvme X:", self.sensor_values["ivme_x"]),
            ("📈 İvme Y:", self.sensor_values["ivme_y"]),
            ("📈 İvme Z:", self.sensor_values["ivme_z"]),
            ("🧲 Manyetik:", self.sensor_values["manyetik"]),
            ("🔄 Gyro:", self.sensor_values["gyro"])
        ]
        
        for label, var in details:
            detail_frame = tk.Frame(sys_details_frame, bg="#0f3460")
            detail_frame.pack(fill="x", pady=4)
            
            tk.Label(detail_frame, text=label, font=("Arial", 10),
                    bg="#0f3460", fg="#b3b3cc", width=12, anchor="w").pack(side="left")
            value_label = tk.Label(detail_frame, textvariable=var, 
                                 font=("Arial", 10, "bold"),
                                 bg="#0f3460", fg="#00ff00")
            value_label.pack(side="left")
        
        # Basınç Grafiği
        pressure_frame = tk.LabelFrame(left_frame, text="📊 BASINÇ (hPa)", 
                                     font=("Arial", 12, "bold"),
                                     bg="#0f3460", fg="#e6e6e6",
                                     padx=10, pady=10)
        pressure_frame.pack(fill="x", padx=10, pady=10)
        
        self.fig_pressure = Figure(figsize=(3.5, 2.5), dpi=80, facecolor='#0f3460')
        self.ax_pressure = self.fig_pressure.add_subplot(111)
        self.ax_pressure.set_facecolor('#0f3460')
        self.ax_pressure.tick_params(colors='white')
        self.ax_pressure.set_ylabel('Basınç (hPa)', color='white')
        self.ax_pressure.set_xlabel('Zaman (s)', color='white')
        self.line_pressure, = self.ax_pressure.plot([], [], 'y-', linewidth=2)
        
        self.canvas_pressure = FigureCanvasTkAgg(self.fig_pressure, pressure_frame)
        self.canvas_pressure.draw()
        self.canvas_pressure.get_tk_widget().pack(fill="both", expand=True)
        
        # Derinlik Grafiği
        depth_frame = tk.LabelFrame(left_frame, text="🌊 DERİNLİK (m)", 
                                  font=("Arial", 12, "bold"),
                                  bg="#0f3460", fg="#e6e6e6",
                                  padx=10, pady=10)
        depth_frame.pack(fill="x", padx=10, pady=10)
        
        self.fig_depth = Figure(figsize=(3.5, 2.5), dpi=80, facecolor='#0f3460')
        self.ax_depth = self.fig_depth.add_subplot(111)
        self.ax_depth.set_facecolor('#0f3460')
        self.ax_depth.tick_params(colors='white')
        self.ax_depth.set_ylabel('Derinlik (m)', color='white')
        self.ax_depth.set_xlabel('Zaman (s)', color='white')
        self.line_depth, = self.ax_depth.plot([], [], 'c-', linewidth=2)
        
        self.canvas_depth = FigureCanvasTkAgg(self.fig_depth, depth_frame)
        self.canvas_depth.draw()
        self.canvas_depth.get_tk_widget().pack(fill="both", expand=True)
    
    def create_center_panel(self, parent):
        """Orta paneli eşit parçalı (Kamera/Harita) oluşturur"""
        center_frame = tk.Frame(parent, bg="#1a1a2e")
        center_frame.pack(side="left", fill="both", expand=True, padx=10)
        
        # --- KAMERA GÖRÜNTÜSÜ (ÜST YARI) ---
        # ÖNEMLİ DÜZELTME: Frame boyutunu sabitliyoruz
        self.camera_frame = tk.LabelFrame(center_frame, text="📷 KAMERA GÖRÜNTÜSÜ", 
                                   font=("Arial", 12, "bold"),
                                   bg="#0f3460", fg="#e6e6e6",
                                   padx=5, pady=5)
        self.camera_frame.pack(side="top", fill="both", expand=True, pady=(0, 5))
        
        # !!! KRİTİK NOKTA !!!
        # pack_propagate(False) diyerek, içeriğin (resmin) çerçeveyi büyütmesini engelliyoruz.
        self.camera_frame.pack_propagate(False)
        
        # Kamera görüntü alanı (Label)
        self.camera_label = tk.Label(self.camera_frame, bg="#000000", 
                                   text="Kamera başlatılıyor...",
                                   font=("Arial", 14), fg="white")
        self.camera_label.pack(fill="both", expand=True)
        
        # --- HARİTA / GÖREV ALANI (ALT YARI) ---
        task_frame = tk.LabelFrame(center_frame, text="🚗 ARAÇ CANLI KONUMU", 
                                 font=("Arial", 12, "bold"),
                                 bg="#0f3460", fg="#e6e6e6",
                                 padx=5, pady=5)
        # expand=True ve fill="both" ile diğer %50 yer kaplaması sağlanır
        task_frame.pack(side="top", fill="both", expand=True, pady=(5, 0))
        
        self.create_map_section(task_frame)

    def create_map_section(self, parent):
        """Harita alanını parent içine yerleştirir"""
        map_frame = tk.Frame(parent, bg="#0f3460")
        map_frame.pack(fill="both", expand=True, pady=(5, 0))

        status_label = tk.Label(map_frame, textvariable=self.location_status_var,
                              font=("Arial", 10), bg="#0f3460", fg="#00ff00", anchor="w")
        status_label.pack(fill="x", pady=(0, 6), side="top")

        if tkintermapview:
            self.map_widget = tkintermapview.TkinterMapView(
                map_frame, corner_radius=0)
            self.map_widget.set_tile_server("https://a.tile.openstreetmap.org/{z}/{x}/{y}.png")
            self.map_widget.set_zoom(15)
            self.map_widget.set_position(41.0082, 28.9784)  # İstanbul başlangıç
            self.map_widget.pack(fill="both", expand=True)
        else:
            tk.Label(map_frame,
                     text="Harita için 'pip install tkintermapview' kurun.\n"
                          "Şimdilik harita yüklenemedi.",
                     font=("Arial", 11, "bold"),
                     bg="#0f3460", fg="#ffb347",
                     justify="left").pack(fill="both", expand=True, pady=8)
    
    def create_right_panel(self, parent):
        right_frame = tk.Frame(parent, bg="#0f3460", width=350,
                             relief="ridge", borderwidth=2)
        right_frame.pack(side="right", fill="y", padx=(10, 0))
        right_frame.pack_propagate(False)
        
        # Motor Kontrol
        motor_frame = tk.LabelFrame(right_frame, text="🚀 MOTOR KONTROL", 
                                  font=("Arial", 12, "bold"),
                                  bg="#0f3460", fg="#e6e6e6",
                                  padx=15, pady=15)
        motor_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Motor durum göstergesi
        status_frame = tk.Frame(motor_frame, bg="#0f3460")
        status_frame.pack(fill="x", pady=(0, 15))
        
        self.motor_status = tk.Label(status_frame, text="⚡ MOTORLAR HAZIR",
                                   font=("Arial", 11, "bold"),
                                   bg="#2c3e50", fg="#00ff00",
                                   padx=15, pady=10)
        self.motor_status.pack(fill="x")
        
        # Motor hız kontrolü
        speed_frame = tk.Frame(motor_frame, bg="#0f3460")
        speed_frame.pack(fill="x", pady=10)
        
        tk.Label(speed_frame, text="🎚️ MOTOR HIZI:", font=("Arial", 10, "bold"),
                bg="#0f3460", fg="#e6e6e6").pack(anchor="w", pady=(0, 5))
        
        self.speed_var = tk.IntVar(value=50)
        speed_scale = tk.Scale(speed_frame, from_=0, to=100,
                             variable=self.speed_var,
                             orient="horizontal",
                             length=250,
                             bg="#0f3460", fg="#00ff00",
                             highlightthickness=0,
                             troughcolor="#2c3e50",
                             command=self.update_motor_speed)
        speed_scale.pack(fill="x")
        
        self.speed_label = tk.Label(speed_frame, text="%50", 
                                  font=("Arial", 12, "bold"),
                                  bg="#0f3460", fg="#00ffff")
        self.speed_label.pack(pady=5)
        
        # Yön kontrolü
        tk.Label(motor_frame, text="🧭 YÖN KONTROLÜ:", 
                font=("Arial", 10, "bold"),
                bg="#0f3460", fg="#e6e6e6").pack(anchor="w", pady=(10, 5))
        
        direction_frame = tk.Frame(motor_frame, bg="#0f3460")
        direction_frame.pack(pady=10)
        
        # Joystick benzeri düğmeler
        directions = [
            ("↖", 0, 0), ("↑", 0, 1), ("↗", 0, 2),
            ("←", 1, 0), ("●", 1, 1), ("→", 1, 2),
            ("↙", 2, 0), ("↓", 2, 1), ("↘", 2, 2)
        ]
        
        for text, row, col in directions:
            if text == "●":  # Ortadaki dur butonu
                btn_color = "#e74c3c"
                cmd = lambda: self.move_direction("DUR")
            else:
                btn_color = "#3498db"
                cmd = lambda t=text: self.move_direction(t)
            
            btn = tk.Button(direction_frame, text=text,
                          font=("Arial", 14, "bold"),
                          bg=btn_color, fg="white",
                          width=4, height=2,
                          command=cmd)
            btn.grid(row=row, column=col, padx=3, pady=3)
        
        # Özel hareketler
        special_frame = tk.Frame(motor_frame, bg="#0f3460")
        special_frame.pack(fill="x", pady=15)
        
        moves = {
            "🔄 DÖNÜŞ": "360° dönüş yapılıyor",
            "📏 YÜKSEL": "Yüzeye yükseliyor",
            "📐 DAL": "Derinliğe dalıyor",
            "⚖️ DENGE": "Dengeleme yapılıyor"
        }
        
        for text, desc in moves.items():
            # Renkleri manuel sırayla veriyoruz
            color = "#9b59b6" 
            if "YÜKSEL" in text: color = "#2ecc71"
            elif "DAL" in text: color = "#3498db"
            elif "DENGE" in text: color = "#f39c12"

            btn = tk.Button(special_frame, text=text,
                          font=("Arial", 9, "bold"),
                          bg=color, fg="white",
                          padx=10, pady=6,
                          command=lambda t=text: self.special_move(t))
            btn.pack(side="left", padx=2, expand=True, fill="x")
        
        # Acil durum butonu
        emergency_btn = tk.Button(motor_frame, text="🚨 ACİL DURDUR",
                                font=("Arial", 11, "bold"),
                                bg="#e74c3c", fg="white",
                                padx=20, pady=10,
                                command=self.emergency_stop)
        emergency_btn.pack(fill="x", pady=(15, 5))

        # Otonom ve Hedef Takip
        modes_frame = tk.Frame(motor_frame, bg="#0f3460")
        modes_frame.pack(fill="x", pady=(0, 10))
        tk.Button(modes_frame, text="🚀 OTONOM MOD",
                  font=("Arial", 10, "bold"),
                  bg="#9b59b6", fg="white",
                  padx=10, pady=8,
                  command=lambda: self.start_task("🚀 OTONOM MOD")).pack(side="left", expand=True, fill="x", padx=4)
        tk.Button(modes_frame, text="🎯 HEDEF TAKİP",
                  font=("Arial", 10, "bold"),
                  bg="#3498db", fg="white",
                  padx=10, pady=8,
                  command=lambda: self.start_task("🎯 HEDEF TAKİP")).pack(side="left", expand=True, fill="x", padx=4)
    
    def create_footer(self):
        footer_frame = tk.Frame(self.main_container, bg="#162447", height=40)
        footer_frame.pack(fill="x", pady=(10, 0))
        
        # Sistem bilgileri
        info_frame = tk.Frame(footer_frame, bg="#162447")
        info_frame.pack(fill="both", expand=True)
        
        # Bağlantı durumu
        self.connection_label = tk.Label(info_frame, text="🔗 Bağlantı: AKTİF",
                                       font=("Arial", 9, "bold"),
                                       bg="#162447", fg="#00ff00")
        self.connection_label.pack(side="left", padx=20)
        
        # Veri akışı
        self.data_label = tk.Label(info_frame, text="📊 Veri Akışı: 125 Hz",
                                  font=("Arial", 9),
                                  bg="#162447", fg="#00ffff")
        self.data_label.pack(side="left", padx=20)
        
        # Batarya durumu
        battery_frame = tk.Frame(info_frame, bg="#162447")
        battery_frame.pack(side="right", padx=20)
        
        tk.Label(battery_frame, text="🔋 Batarya:", font=("Arial", 9),
                bg="#162447", fg="#e6e6e6").pack(side="left")
        
        self.battery_var = tk.StringVar(value="92%")
        tk.Label(battery_frame, textvariable=self.battery_var,
                font=("Arial", 9, "bold"),
                bg="#162447", fg="#00ff00").pack(side="left")
    
    def init_camera(self):
        """Kamerayı başlat"""
        try:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.camera_active = True
                self.start_camera_stream()
            else:
                self.camera_active = False
                self.camera_label.config(text="❌ Kamera bulunamadı veya erişilemiyor.")
                if self.cap:
                    self.cap.release()
        except Exception as e:
            self.camera_active = False
            self.camera_label.config(text=f"⚠️ Kamera hatası: {str(e)[:50]}")
            if self.cap:
                self.cap.release()
    
    def start_camera_stream(self):
        """Kamera görüntüsünü göster (Stabil Boyutlandırma)"""
        if self.camera_active and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # OpenCV BGR -> RGB dönüşümü
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # --- ÇÖZÜM: TAŞMAYI ENGELLEYEN BOYUTLANDIRMA ---
                # Çerçevenin (LabelFrame) boyutunu alıyoruz (Resmin konacağı yer)
                container_w = self.camera_frame.winfo_width()
                container_h = self.camera_frame.winfo_height()
                
                # Pencere henüz yüklenmediyse standart boyut kullan
                if container_w < 10 or container_h < 10:
                    container_w, container_h = 640, 480
                
                # Görüntüyü çerçevenin içine sığacak şekilde küçült (Boşluk payı bırak)
                # 20 piksel boşluk bırakıyoruz ki sınırları zorlamasın
                w = container_w - 20 
                h = container_h - 20
                
                if w > 10 and h > 10:
                    frame = cv2.resize(frame, (w, h))
                    img = Image.fromarray(frame)
                    imgtk = ImageTk.PhotoImage(image=img)
                    self.camera_label.imgtk = imgtk
                    self.camera_label.config(image=imgtk, text="")
            
            # 30 ms sonra tekrar çağır
            self.root.after(30, self.start_camera_stream)
        elif self.camera_active:
            self.camera_label.config(text="Kamera görüntüsü alınamıyor")
    
    def toggle_camera(self):
        """Kamerayı aç/kapat"""
        if self.cap and self.cap.isOpened():
            self.camera_active = not self.camera_active
            if self.camera_active:
                self.start_camera_stream()
            else:
                self.camera_label.config(image="", text="Kamera durduruldu")
        else:
            self.init_camera()
    
    def capture_image(self):
        """Fotoğraf çek"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"capture_{timestamp}.png"
        
        frame = None
        cap = None
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                ret, frame = cap.read()
        finally:
            if cap and cap.isOpened():
                cap.release()

        if frame is not None:
            cv2.imwrite(filename, frame)
            messagebox.showinfo("Başarılı", f"Fotoğraf kaydedildi: {filename}")
        else:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "SIMULASYON GORUNTUSU", (150, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imwrite(filename, frame)
            messagebox.showinfo("Simülasyon", f"Simüle edilmiş fotoğraf kaydedildi: {filename}")
    
    def create_vehicle_icon(self, size=28):
        """Haritada araç için basit simge"""
        try:
            img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            draw.ellipse((2, 2, size - 2, size - 2), fill=(0, 200, 255, 230))
            draw.polygon([(size * 0.5, size * 0.05),
                          (size * 0.75, size * 0.5),
                          (size * 0.25, size * 0.5)],
                          fill=(255, 255, 255, 240))
            return ImageTk.PhotoImage(img)
        except Exception as e:
            print(f"Simge oluşturulamadı: {e}")
            return None

    def start_location_updates(self):
        """Sensör verisi yoksa bile konumu simüle eder ve haritayı günceller"""
        if not tkintermapview:
            return

        self.current_lat = 41.0082
        self.current_lon = 28.9784

        def loc_thread():
            while True:
                try:
                    self.current_lat += random.uniform(-0.00025, 0.00025)
                    self.current_lon += random.uniform(-0.00025, 0.00025)
                    self.root.after(0, lambda lat=self.current_lat, lon=self.current_lon:
                                    self.update_location_on_map(lat, lon))
                    time.sleep(1.0)
                except Exception as e:
                    print(f"Konum simülasyon hatası: {e}")
                    time.sleep(2)

        threading.Thread(target=loc_thread, daemon=True).start()

    def update_location_on_map(self, lat, lon):
        """Haritada marker ve izi günceller"""
        if not self.map_widget:
            return

        if not self.map_marker:
            self.map_marker = self.map_widget.set_marker(lat, lon,
                                                       text="Araç",
                                                       icon=self.vehicle_icon)
        else:
            self.map_marker.set_position(lat, lon)

        self.location_points.append((lat, lon))
        if self.map_path:
            self.map_path.delete()
        if len(self.location_points) > 1:
            self.map_path = self.map_widget.set_path(list(self.location_points))

        self.map_widget.set_position(lat, lon)
        self.location_status_var.set(f"Lat: {lat:.6f}  Lon: {lon:.6f} (simüle)")

    def start_sensor_simulation(self):
        """Sensör verilerini simüle et"""
        def sensor_thread():
            while True:
                try:
                    # Basınç simülasyonu
                    current_time = time.time()
                    pressure = 1013.25 + 50 * np.sin(current_time * 0.5) + random.uniform(-2, 2)
                    depth = 50 + 30 * np.sin(current_time * 0.3) + random.uniform(-1, 1)
                    
                    self.pressure_data.append(pressure)
                    self.depth_data.append(depth)
                    self.time_data.append(len(self.time_data))
                    
                    # Grafikleri güncelle (Veri güncellemesi)
                    self.update_graphs()
                    
                    # Sensör değerlerini güncelle
                    self.update_sensor_values()
                    
                    # Batarya simülasyonu
                    battery = max(10, 100 - (current_time % 100))
                    self.battery_var.set(f"{battery:.0f}%")
                    
                    time.sleep(0.5)
                    
                except Exception as e:
                    print(f"Sensör hatası: {e}")
                    time.sleep(1)
        
        thread = threading.Thread(target=sensor_thread, daemon=True)
        thread.start()
    
    def update_graphs(self):
        """Grafikleri güncelle"""
        try:
            # Sadece verileri güncellemek performans için daha iyidir
            # Ancak matplotlib-tkagg entegrasyonunda clear() en temiz yöntemlerden biridir
            # İleri seviye optimizasyon için set_ydata kullanılabilir.
            
            # Basınç
            self.ax_pressure.clear()
            self.ax_pressure.set_facecolor('#0f3460')
            self.ax_pressure.tick_params(colors='white')
            self.ax_pressure.set_ylabel('Basınç (hPa)', color='white')
            self.ax_pressure.set_xlabel('Zaman (s)', color='white')
            
            data_to_show = min(50, len(self.pressure_data))
            x_data = list(range(data_to_show))
            y_data = list(self.pressure_data)[-data_to_show:]
            
            self.ax_pressure.plot(x_data, y_data, 'y-', linewidth=2)
            self.ax_pressure.set_ylim(950, 1050)
            
            # Derinlik
            self.ax_depth.clear()
            self.ax_depth.set_facecolor('#0f3460')
            self.ax_depth.tick_params(colors='white')
            self.ax_depth.set_ylabel('Derinlik (m)', color='white')
            self.ax_depth.set_xlabel('Zaman (s)', color='white')
            
            y_depth = list(self.depth_data)[-data_to_show:]
            self.ax_depth.plot(x_data, y_depth, 'c-', linewidth=2)
            self.ax_depth.set_ylim(0, 100)
            
            self.canvas_pressure.draw()
            self.canvas_depth.draw()
            
        except Exception as e:
            print(f"Grafik güncelleme hatası: {e}")
    
    def update_sensor_values(self):
        """Sensör değerlerini güncelle"""
        temp = 20 + 5 * np.sin(time.time() * 0.2) + random.uniform(-0.5, 0.5)
        humidity = 40 + 10 * np.sin(time.time() * 0.1) + random.uniform(-2, 2)
        
        self.sensor_values["sıcaklık"].set(f"{temp:.1f}°C")
        self.sensor_values["nem"].set(f"{humidity:.0f}%")
        
        ax = 0.1 * np.sin(time.time())
        ay = 0.08 * np.sin(time.time() * 1.2)
        az = 0.95 + 0.05 * np.sin(time.time() * 0.5)
        
        self.sensor_values["ivme_x"].set(f"{ax:.3f}g")
        self.sensor_values["ivme_y"].set(f"{ay:.3f}g")
        self.sensor_values["ivme_z"].set(f"{az:.3f}g")
        
        mag = 50 + 5 * np.sin(time.time() * 0.3)
        self.sensor_values["manyetik"].set(f"{mag:.1f}µT")
        
        gyro = 0.05 * np.sin(time.time())
        self.sensor_values["gyro"].set(f"{gyro:.2f}°/s")
    
    def update_time(self):
        """Saati güncelle"""
        now = datetime.now().strftime("%H:%M:%S - %d.%m.%Y")
        self.time_label.config(text=f"🕒 {now}")
        self.root.after(1000, self.update_time)
    
    def update_motor_speed(self, value):
        """Motor hızını güncelle"""
        self.speed_label.config(text=f"%{value}")
    
    def move_direction(self, direction):
        """Yön hareketi"""
        directions = {
            "↖": "SOL-YUKARI", "↑": "YUKARI", "↗": "SAĞ-YUKARI",
            "←": "SOL", "●": "DURDU", "→": "SAĞ",
            "↙": "SOL-AŞAĞI", "↓": "AŞAĞI", "↘": "SAĞ-AŞAĞI",
            "DUR": "DURDU"
        }
        
        if direction in directions:
            self.motor_status.config(text=f"🏃 {directions[direction]}", fg="#f39c12")
            messagebox.showinfo("Hareket", f"Araç {directions[direction]} yönünde hareket ediyor")
    
    def special_move(self, move_type):
        """Özel hareket"""
        moves = {
            "🔄 DÖNÜŞ": "360° dönüş yapılıyor",
            "📏 YÜKSEL": "Yüzeye yükseliyor",
            "📐 DAL": "Derinliğe dalıyor",
            "⚖️ DENGE": "Dengeleme yapılıyor"
        }
        
        if move_type in moves:
            messagebox.showinfo("Özel Hareket", moves[move_type])
    
    def emergency_stop(self):
        """Acil durdur"""
        self.motor_status.config(text="🚨 ACİL DURDURULDU", fg="#e74c3c")
        self.speed_var.set(0)
        self.speed_label.config(text="%0")
        messagebox.showwarning("Acil Durum", "Tüm motorlar acil durduruldu!")
    
    def start_task(self, task_name):
        """Görev başlat"""
        tasks = {
            "🚀 OTONOM MOD": "Otonom mod başlatıldı",
            "🎯 HEDEF TAKİP": "Hedef takip modu aktif"
        }
        
        if task_name in tasks:
            messagebox.showinfo("Görev", tasks[task_name])
    
    def on_closing(self):
        """Pencere kapanırken kaynakları serbest bırak"""
        if self.cap:
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = SystemControlInterface(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()