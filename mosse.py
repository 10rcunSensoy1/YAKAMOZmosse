import cv2
import time
import torch



class TrackingMosse:
	def __init__(self):
		# YOLO modelini yükleme
		self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='deneme.onnx', force_reload=True)
		
		# Video yakalama
		self.cap = cv2.VideoCapture('CAKIR.mp4')
		self.mosse_tracker = None
		self.tracking_active = False
		self.iha_class_id = 0  # İHA'nın sınıf ID'si
		self.confidence = 0  # Tahmin doğruluk oranı
		self.fps = 30  # Başlangıç FPS değeri
		self.inside = False  # İHA'nın içerde olup olmadığını belirten bayrak
		self.run()
	
	def run(self):
		self.prev_time = time.time()
		self.frame_count = 0
		self.fps_display = 0
		
		while self.cap.isOpened():
			ret, frame = self.cap.read()
			if not ret:
				break
			
			# Çerçeveyi boyutlandırma
			frame = cv2.resize(frame, (640, 640))  # Çözünürlüğü artırmadan çalışıyoruz
			
			# Takip alanını çizme (YOLO karesi)
			self.draw_tracking_area(frame)
			
			# MOSSE ile takip
			if self.tracking_active and self.mosse_tracker is not None:
				self.track_mosse(frame)
			else:
				# YOLO ile tespit
				self.run_yolo(frame)
			
			# FPS'i hesapla ve göster
			self.show_fps(frame)
			
			cv2.imshow('YOLOv5 ve MOSSE', frame)
			
			key = cv2.waitKey(int(1000 / self.fps)) & 0xFF  # Dinamik FPS ayarı
			if key == ord('q'):
				break
			elif key == ord('r'):
				self.reset_tracking(frame)
		
		self.cap.release()
		cv2.destroyAllWindows()
	
	def draw_tracking_area(self, frame):
		# Orta bölgede bir kare çiz (YOLO karesi)
		square_size = 250  # Kare boyutunu standardize ediyoruz
		height, width, _ = frame.shape
		
		self.top_left_x = (width - square_size) // 2
		self.top_left_y = (height - square_size) // 2
		self.bottom_right_x = self.top_left_x + square_size
		self.bottom_right_y = self.top_left_y + square_size
		
		# Mavi YOLO karesini çiz
		cv2.rectangle(frame, (self.top_left_x, self.top_left_y), (self.bottom_right_x, self.bottom_right_y),
					  (255, 0, 0), 2)
	
	def track_mosse(self, frame):
		if self.mosse_tracker is not None:
			success, bbox = self.mosse_tracker.update(frame)
			if success:
				x, y, w, h = [int(v) for v in bbox]
				
				# Güncellenmiş yeşil MOSSE kutusu
				cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
				
				# MOSSE karesi üzerinde İHA etiketi ve doğruluk oranı göster
				label = f"IHA ({self.confidence * 100:.1f}%)"
				cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
				
				# İHA'nın kare içinde olup olmadığını kontrol et ve durumu göster
				self.check_inside(frame, (x, y, w, h))
			else:
				# MOSSE takibi başarısız olursa, yeniden YOLO tespiti ile MOSSE karesini sıfırla
				self.run_yolo(frame)
	
	def run_yolo(self, frame):
		# YOLO tahminleri
		results = self.model(frame)  # Modelin giriş boyutunu artırmadan kullanıyoruz
		detections = results.xyxy[0].cpu().numpy()  # YOLO sonuçları numpy dizisine çevriliyor
		
		for det in detections:
			x1, y1, x2, y2, conf, cls = det[:6]  # İlk 6 değeri alıyoruz
			
			if int(cls) == self.iha_class_id and conf > 0.3:  # İHA sınıfı ve daha düşük güven skoru kontrolü
				bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
				
				# MOSSE izleyiciyi başlat ve yeniden odaklan
				self.mosse_tracker = cv2.legacy.TrackerMOSSE_create()
				self.mosse_tracker.init(frame, bbox)
				self.tracking_active = True
				self.confidence = conf  # Güven oranını kaydet
				
				# YOLO karesini İHA'nın merkezine yönlendir
				self.center_yolo_to_iha(bbox)
				break
	
	def center_yolo_to_iha(self, bbox):
		x, y, w, h = bbox
		iha_center_x = x + w // 2
		iha_center_y = y + h // 2
		
		# Kare merkezini hesapla
		square_center_x = (self.top_left_x + self.bottom_right_x) // 2
		square_center_y = (self.top_left_y + self.bottom_right_y) // 2
		
		# Merkezin farkını hesapla
		offset_x = iha_center_x - square_center_x
		offset_y = iha_center_y - square_center_y
		
		# YOLO karesinin merkezini İHA'nın merkezine kaydır
		self.top_left_x += offset_x
		self.top_left_y += offset_y
		self.bottom_right_x += offset_x
		self.bottom_right_y += offset_y
	
	def check_inside(self, frame, bbox):
		x, y, w, h = bbox
		center_x = x + w // 2
		center_y = y + h // 2
		
		# İHA'nın YOLO karesi içinde olup olmadığını kontrol et
		if (self.top_left_x <= center_x <= self.bottom_right_x) and (
				self.top_left_y <= center_y <= self.bottom_right_y):
			cv2.putText(frame, "INSIDE", (self.top_left_x + 10, self.top_left_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
						(0, 255, 0), 2)
			self.fps = 15  # İHA içerideyse FPS'i 15'e düşür
			self.inside = True
		else:
			cv2.putText(frame, "OUTSIDE", (self.top_left_x + 10, self.top_left_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
						(0, 0, 255), 2)
			self.show_arrow(frame, center_x, center_y)
			self.fps = 30  # İHA dışarıdaysa FPS'i 30'a çıkar
			self.inside = False
			
			# MOSSE karesini yeniden İHA üzerine odaklamak için YOLO tespiti çalıştır
			self.run_yolo(frame)
	
	def show_arrow(self, frame, center_x, center_y):
		# MOSSE karesi YOLO karesinin tamamen dışındaysa bir ok işareti göster
		height, width, _ = frame.shape
		arrow_color = (0, 0, 255)  # Kırmızı renk
		
		if center_x < self.top_left_x:  # Sol tarafta
			cv2.arrowedLine(frame, (center_x + 20, center_y), (self.top_left_x + 20, center_y), arrow_color, 2)
		elif center_x > self.bottom_right_x:  # Sağ tarafta
			cv2.arrowedLine(frame, (center_x - 20, center_y), (self.bottom_right_x - 20, center_y), arrow_color, 2)
		if center_y < self.top_left_y:  # Üst tarafta
			cv2.arrowedLine(frame, (center_x, center_y + 20), (center_x, self.top_left_y + 20), arrow_color, 2)
		elif center_y > self.bottom_right_y:  # Alt tarafta
			cv2.arrowedLine(frame, (center_x, center_y - 20), (center_x, self.bottom_right_y - 20), arrow_color, 2)
	
	def show_fps(self, frame):
		self.frame_count += 1
		current_time = time.time()
		elapsed_time = current_time - self.prev_time
		
		if elapsed_time >= 1.0:
			self.fps_display = self.frame_count / elapsed_time
			self.frame_count = 0
			self.prev_time = current_time
		
		cv2.putText(frame, f'FPS: {int(self.fps_display)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
		
		def reset_tracking(self, frame):
			# Mevcut MOSSE izleyiciyi sıfırla
			self.mosse_tracker = None
			self.tracking_active = False
			
			# YOLO tespitini yeniden çalıştırarak İHA'ya yeniden odaklan
			self.run_yolo(frame)
		
		# Takip sistemi sınıfını başlat
	
tracking_system = TrackingMosse()
