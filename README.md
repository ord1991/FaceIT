# FaceIT - Real-time Face Recognition Access Control

FaceIT is a high-performance, real-time face recognition system designed for access control. It is built using Python, FastAPI, and the DeepFace library, optimized to run efficiently on CPU (Intel i7) without requiring a discrete GPU.

## Features

- **Real-time Monitoring**: Live MJPEG video stream with dynamic bounding boxes and status labels.
- **Face Recognition**: Powered by DeepFace with the VGG-Face model and OpenCV backend.
- **Classification Logic**:
  - **Approved (Green)**: Recognized user with 'approved' status.
  - **Blacklisted (Red)**: Recognized user with 'blacklisted' status.
  - **Unknown (Yellow)**: Face detected but not matched in the database.
- **User Management**:
  - Register new users via image upload.
  - **Quick-tagging**: Register detected "Unknown" faces directly from the dashboard.
  - Toggle user status between Approved and Blacklisted.
- **Modern Dashboard**: A responsive, RTL-supported (Hebrew-friendly) web interface.
- **Mock Mode**: Capability to test the system and UI using a static image instead of a live webcam.

## Tech Stack

- **Backend**: Python, FastAPI
- **Recognition Engine**: DeepFace (VGG-Face, Cosine Similarity)
- **Database**: SQLite with SQLAlchemy ORM
- **Frontend**: HTML5, CSS3 (RTL), Vanilla JS, jQuery, Jinja2 Templates
- **Image Processing**: OpenCV

## Project Structure

```text
├── main.py             # FastAPI application, camera handling, and API routes
├── face_engine.py      # FaceEngine class for embeddings and frame processing
├── database.py         # SQLAlchemy models and session management
├── templates/
│   └── index.html      # Dashboard template
├── static/
│   ├── style.css       # Frontend styling (RTL support)
│   └── script.js       # Frontend logic and AJAX calls
├── faces/              # Permanent storage for registered user face crops (ignored by git)
└── requirements.txt    # Project dependencies
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ord1991/FaceIT
   cd FaceIT
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare directories**:
   The application will automatically create `faces/` and `static/` directories on first run.

## Usage

### Running the application
To start the server:
```bash
python main.py
```
The dashboard will be available at `http://localhost:8000`.

### Mock Mode
If you don't have a webcam connected or wish to test the UI with a static image, set the `MOCK_MODE` environment variable:
```bash
export MOCK_MODE=true
python main.py
```
This will use `mock_face.jpg` as the video source.

## Configuration

- **Distance Threshold**: The default cosine distance threshold for VGG-Face is set to `0.4` in `face_engine.py`.
- **Database**: The system uses a local SQLite database file named `faceit.db`.

## License

MIT
