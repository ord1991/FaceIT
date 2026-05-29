import os
import cv2
import uuid
import threading
import time
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from database import User, init_db, get_db
from face_engine import FaceEngine
import numpy as np

app = FastAPI()

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    # Basic CSP: allow self, jQuery CDN, and inline styles (needed for some UI elements)
    # Note: 'unsafe-inline' for script-src is avoided if possible,
    # but some inline event handlers in templates may require it or need refactoring.
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' https://code.jquery.com 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:;"
    )
    return response

# Ensure directories exist
os.makedirs("faces", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Initialize DB
init_db()

# Initialize Face Engine
engine = FaceEngine()

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
# Also mount faces for easy access to images if needed
app.mount("/faces", StaticFiles(directory="faces"), name="faces")
templates = Jinja2Templates(directory="templates")

# Lock for thread safety
unknown_faces_lock = threading.Lock()

# Camera handling
class Camera:
    def __init__(self):
        self.video = None
        self.is_mock = os.getenv("MOCK_MODE", "false").lower() == "true"
        self.mock_image = None
        if self.is_mock:
            print("Running in MOCK MODE")
            # Create a dummy image if none exists or load one
            if os.path.exists("mock_face.jpg"):
                self.mock_image = cv2.imread("mock_face.jpg")
            else:
                self.mock_image = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(self.mock_image, "MOCK CAMERA", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
                cv2.imwrite("mock_face.jpg", self.mock_image)
        else:
            self.video = cv2.VideoCapture(0)

    def get_frame(self):
        if self.is_mock:
            return True, self.mock_image.copy()
        else:
            return self.video.read()

    def __del__(self):
        if self.video:
            self.video.release()

camera = Camera()

# Latest unknown face detections for quick-tagging
# Stores {id: {"frame": cropped_frame, "embedding": embedding, "timestamp": time}}
unknown_faces = {}

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

def generate_frames():
    global unknown_faces
    while True:
        success, frame = camera.get_frame()
        if not success:
            break

        processed_frame, results = engine.process_frame(frame)

        # Update unknown faces
        for res in results:
            if res["status"] == "Unknown":
                x, y, w, h = res["box"]
                # Ensure box is within frame
                h_img, w_img = frame.shape[:2]
                x, y = max(0, x), max(0, y)
                w, h = min(w, w_img - x), min(h, h_img - y)

                face_crop = frame[y:y+h, x:x+w]
                if face_crop.size > 0:
                    with unknown_faces_lock:
                        # Check if we already have this unknown face recently to avoid duplicates
                        is_duplicate = False
                        for existing in unknown_faces.values():
                            dist = engine.cosine_distance(np.array(res["embedding"]), np.array(existing["embedding"]))
                            if dist < engine.threshold:
                                is_duplicate = True
                                # Update timestamp to keep it fresh
                                existing["timestamp"] = time.time()
                                break

                        if is_duplicate:
                            continue

                        face_id = str(uuid.uuid4())
                        # We only keep the latest few unknowns to avoid memory/disk bloat
                        to_remove_path = None
                        if len(unknown_faces) >= 50:
                            oldest_id = min(unknown_faces.keys(), key=lambda k: unknown_faces[k]["timestamp"])
                            to_remove_path = unknown_faces[oldest_id]["image_url"].lstrip('/')
                            del unknown_faces[oldest_id]

                        # Add new entry
                        crop_path = f"static/unknown_{face_id}.jpg"
                        unknown_faces[face_id] = {
                            "id": face_id,
                            "image_url": f"/{crop_path}",
                            "embedding": res["embedding"],
                            "timestamp": time.time()
                        }

                    # Perform I/O outside the lock to minimize contention
                    if to_remove_path and os.path.exists(to_remove_path):
                        os.remove(to_remove_path)
                    cv2.imwrite(crop_path, face_crop)

        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        # Yield a tiny bit to reduce CPU usage
        time.sleep(0.01)

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/users")
async def list_users(db: Session = Depends(get_db)):
    users = db.query(User).all()
    return [{"id": u.id, "name": u.name, "status": u.status, "image_path": u.image_path} for u in users]

@app.post("/users/add")
def add_user(
    name: str = Form(...),
    status: str = Form(...),
    face_id: str = Form(None), # From quick-tagging
    file: UploadFile = File(None), # From upload
    db: Session = Depends(get_db)
):
    embedding = None
    image_path = None

    with unknown_faces_lock:
        if face_id and face_id in unknown_faces:
            # Quick-tagging from unknown_faces
            unknown = unknown_faces[face_id]
            embedding = unknown["embedding"]
            # Save the crop permanently
            image_name = f"{uuid.uuid4()}.jpg"
            image_path = f"faces/{image_name}"
            # The crop was saved as static/unknown_... we move/copy it
            source_path = unknown["image_url"].lstrip('/')
            if os.path.exists(source_path):
                os.rename(source_path, image_path)
            del unknown_faces[face_id]

    if embedding is None and file:
        # From file upload
        contents = file.file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Get embedding
        temp_path = f"temp_{uuid.uuid4()}.jpg"
        cv2.imwrite(temp_path, img)
        embedding = engine.get_embedding(temp_path)

        if embedding:
            image_name = f"{uuid.uuid4()}.jpg"
            image_path = f"faces/{image_name}"
            cv2.imwrite(image_path, img)
            os.remove(temp_path)
        else:
            os.remove(temp_path)
            raise HTTPException(status_code=400, detail="No face detected in uploaded image")

    if embedding is None:
        # Capture from current frame?
        # For simplicity, UI can send the current "Unknown" face_id.
        # If they want to capture "NOW", we'd need another mechanism.
        # Let's assume they use quick-tagging or upload.
        raise HTTPException(status_code=400, detail="No face source provided")

    new_user = User(name=name, status=status, image_path=image_path, embedding=embedding)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    # Reload embeddings in engine
    engine.load_known_users()

    return {"status": "success", "user_id": new_user.id}

@app.post("/users/update_status")
async def update_status(user_id: int = Form(...), status: str = Form(...), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.status = status
    db.commit()
    engine.load_known_users()
    return {"status": "success"}

@app.get("/unknown_faces")
async def get_unknown_faces():
    # Return list of recent unknown faces
    # Filter out old ones (e.g., older than 5 minutes for better user experience)
    now = time.time()
    recent = []
    with unknown_faces_lock:
        to_delete = []
        for k, v in unknown_faces.items():
            if now - v["timestamp"] > 300:
                to_delete.append(k)
            else:
                recent.append({"id": v["id"], "image_url": v["image_url"]})

        for k in to_delete:
            path = unknown_faces[k]["image_url"].lstrip('/')
            if os.path.exists(path):
                os.remove(path)
            del unknown_faces[k]

    return recent

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
