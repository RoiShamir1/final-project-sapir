import cv2

rtmp_url = "rtmp://10.186.183.143/live/argus"

print("⏳ מנסה להתחבר לזרם הווידאו מהרחפן...")
cap = cv2.VideoCapture(rtmp_url)

if not cap.isOpened():
    print("❌ שגיאה: לא ניתן להתחבר לזרם. ודא ש-MonaServer פועל ושהשלט משדר לאותה רשת.")
    exit()

print("✅ חיבור הצליח! לחץ על 'q' בחלון הווידאו כדי לסגור אותו.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ איבדנו את החיבור לרחפן.")
        break

    cv2.imshow("Argus - Live Drone Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()