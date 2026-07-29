import cv2
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

print("Loading BLIP model...")

processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

print("Model Loaded!")

cap = cv2.VideoCapture(0)

caption = ""

frame_count = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # Generate a caption every 30 frames
    if frame_count % 30 == 0:

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(rgb)

        inputs = processor(image, return_tensors="pt")

        out = model.generate(**inputs)

        caption = processor.decode(
            out[0],
            skip_special_tokens=True
        )

    cv2.putText(
        frame,
        caption,
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.imshow("Live AI Caption Generator", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()