class Config:
	use_webcam = True  # If False, file given by vide_path will be used  
	video_path = 'test.mp4'

	face_detection_model = 'hog'  # Options: 'hog' (faster, less accuracy), 'cnn' (slower, more accuracy)
