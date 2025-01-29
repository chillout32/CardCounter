import mss
import cv2
import numpy as np
from screeninfo import get_monitors
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model
model = load_model('my_model.keras')

card_values = {
    '2': 1,
    '3': 1,
    '4': 1,
    '5': 1,
    '6': 1,
    '7': 0,
    '8': 0,
    '9': 0,
    '10': -1,
    'J': -1,
    'Q': -1,
    'K': -1,
    'A': -1
}

roi_width = 1200
roi_height = 670

# Monitor Selector
monitor = get_monitors()[0]
screen_width = monitor.width
screen_height = monitor.height

roi_top = (screen_height - roi_height) // 2 - 20
roi_left = (screen_width - roi_width) // 2 + 30

roi = {"top": roi_top, "left": roi_left, "width": roi_width, "height": roi_height}

cv2.namedWindow("Live Feed with Border", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Live Feed with Border", roi["width"], roi["height"])

running_count = 0
counted_cards = set()

def preprocess_image(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    img = img / 255.0  # Normalize to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def predict_card(frame):
    img = preprocess_image(frame)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)  # Get the predicted class index
    class_labels = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']  # Update according to your training labels
    predicted_card = class_labels[predicted_class]
    return predicted_card

def find_card_regions(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    card_regions = []

    for contour in contours:
        # Ignore small contours
        if cv2.contourArea(contour) < 1000:
            continue

        # Get bounding box around contour
        x, y, w, h = cv2.boundingRect(contour)
        card_region = frame[y:y+h, x:x+w]
        card_regions.append(card_region)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return card_regions, frame

# Monitor capturing
with mss.mss() as sct:
    while True:
        screen = np.array(sct.grab(roi))

        found_cards, debug_frame = find_card_regions(screen)

        for card_region in found_cards:
            predicted_card = predict_card(card_region)
            if predicted_card not in counted_cards:
                running_count += card_values.get(predicted_card, 0)
                counted_cards.add(predicted_card)

        print(f"Running Count: {running_count}")
        
        if running_count > 5:
            print("Advice: Stand")
        elif running_count < -5:
            print("Advice: Hit")
        else:
            print("Advice: Wait")

        cv2.imshow("Live Feed with Border", debug_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cv2.destroyAllWindows()
