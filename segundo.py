import cv2
import math
import time
import mediapipe as mp
import serial

# ==========================================
# 1. CONFIGURACIÓN DEL PUERTO SERIAL (COM4)
# ==========================================
PUERTO_SERIAL = "COM4"
BAUD_RATE = 9600
USAR_SERIAL = True 

if USAR_SERIAL:
    try:
        ser = serial.Serial(PUERTO_SERIAL, BAUD_RATE, timeout=0.1)
        time.sleep(2)
        print(f" Conectado exitosamente al puerto {PUERTO_SERIAL}")
    except Exception as e:
        print(f" Error al abrir el puerto serial: {e}")
        print("Continuando en modo SOLO VISUAL...")
        USAR_SERIAL = False

# ==========================================
# 2. INICIALIZACIÓN DE MEDIAPIPE Y CÁMARA
# ==========================================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1, 
    refine_landmarks=True, 
    min_detection_confidence=0.5
)

def get_dist(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

cap = cv2.VideoCapture(0)
estado_previo = "NEUTRO"

print("\n--- DETECTOR CON TRISTEZA OPTIMIZADA (COM4) ---")
print("Presiona 'q' para salir.\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    emotion = "NEUTRO"
    comando_char = 'N'

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            lm = face_landmarks.landmark

            # 1. Distancia de Referencia (Ancho de los Ojos)
            eye_dist = get_dist(lm[33], lm[263])
            if eye_dist == 0:
                continue

            # 2. Mediciones Generales
            mouth_height = get_dist(lm[13], lm[14])
            mouth_width = get_dist(lm[61], lm[291])
            entrecejo_dist = get_dist(lm[55], lm[285])

            # --- PARÁMETROS ESPECÍFICOS PARA TRISTEZA ---
            # Cejas internas (55, 285) vs Cejas externas (70, 300)
            # En tristeza, la ceja interna sube más que la externa en el eje Y
            inclinacion_ceja_izq = lm[55].y - lm[70].y
            inclinacion_ceja_der = lm[285].y - lm[300].y
            inclinacion_triste = (inclinacion_ceja_izq + inclinacion_ceja_der) / 2.0

            # Caída de comisuras relativa a la altura media de la cara
            comisuras_caidas = (lm[61].y > lm[17].y - 0.005) or (lm[291].y > lm[17].y - 0.005)

            # --- RATIOS NORMALIZADOS ---
            mar = mouth_height / mouth_width            # Alto/Ancho boca
            smile_ratio = mouth_width / eye_dist        # Sonrisa
            entrecejo_ratio = entrecejo_dist / eye_dist # Enojado

            # --- ÁRBOLES DE DECISIÓN ---
            
            # 1. SORPRENDIDO (Aprobado)
            if mar > 0.45:
                emotion = "SORPRENDIDO"
                comando_char = 'O'

            # 2. FELIZ (Aprobado)
            elif smile_ratio > 0.72:
                emotion = "FELIZ"
                comando_char = 'H'

            # 3. ENOJADO (Aprobado)
            elif entrecejo_ratio < 0.28:
                emotion = "ENOJADO"
                comando_char = 'A'

            # 4. TRISTE (Mejorado: Cejas en V invertida O comisuras ligeramente caídas)
            elif inclinacion_triste < -0.010 or comisuras_caidas:
                emotion = "TRISTE"
                comando_char = 'S'

            # 5. NEUTRO
            else:
                emotion = "NEUTRO"
                comando_char = 'N'

            # Mostrar datos para calibración en pantalla
            cv2.putText(frame, f"Triste Brow Y: {inclinacion_triste:.3f}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    # --- COMUNICACIÓN SERIAL ---
    if emotion != estado_previo:
        print(f" GESTO DETECTADO: {emotion} -> Enviando '{comando_char}'")
        if USAR_SERIAL:
            ser.write(comando_char.encode())
        estado_previo = emotion

    # --- FEEDBACK VISUAL ---
    colores = {
        "FELIZ": (0, 255, 0),
        "SORPRENDIDO": (255, 255, 0),
        "ENOJADO": (0, 0, 255),
        "TRISTE": (255, 0, 255),
        "NEUTRO": (200, 200, 200)
    }

    cv2.putText(frame, f"GESTO: {emotion}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, colores.get(emotion, (250,250,250)), 2)
    cv2.imshow("Control de Gestos - COM4", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
if USAR_SERIAL:
    ser.close()
