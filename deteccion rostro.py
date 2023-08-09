import cv2

# Cargamos el clasificador preentrenado para detección de rostros
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Iniciamos la cámara web
cap = cv2.VideoCapture(0)

while True:
    # Capturamos un frame de la cámara
    ret, frame = cap.read()

    # Convertimos el frame a escala de grises (requerido para la detección de rostros)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectamos rostros en la imagen
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Dibujamos un rectángulo alrededor de cada rostro detectado
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Mostramos la imagen resultante con las detecciones
    cv2.imshow('Detección de Rostros', frame)

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberamos los recursos y cerramos la ventana
cap.release()
cv2.destroyAllWindows()
