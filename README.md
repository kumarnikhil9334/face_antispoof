# Face Liveness Detection (Anti-Spoofing)

A real-time face liveness detection app. It tells whether the person in front of the camera is **REAL**, or a **FAKE** (photo, phone screen, print).

Built with a **FastAPI** backend and a **React (Vite)** webcam UI, on top of the pre-trained [Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing) models.

![Demo](docs/demo.png)

## How it works

```
Camera -> React (frame capture every 1.5s) -> POST /predict -> FastAPI
       -> face detection -> 2 anti-spoof models -> {label, is_real, score} -> UI
```

## Project structure

```
backend/Silent-Face-Anti-Spoofing/   FastAPI app (app.py), models, source
frontend/anti-spoof-ui/              React (Vite) webcam UI
```

## Run locally (Windows)

**Requirements:** Python 3.11, Node.js

**Backend**
```bash
cd backend/Silent-Face-Anti-Spoofing
py -3.11 -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pip install fastapi uvicorn python-multipart
python -m uvicorn app:app --reload --port 8000
```
API docs: http://localhost:8000/docs

**Frontend** (new terminal)
```bash
cd frontend/anti-spoof-ui
npm install
npm run dev
```
Open http://localhost:5173, allow camera access, and click **Start Live Check**.

## API

`POST /predict` with an image file (`multipart/form-data`, field `file`)

```json
{ "is_real": true, "label": "REAL", "score": 0.97 }
```

## Limitations

- Accuracy drops in low light.
- Designed for a single face per frame.
- Uses a pre-trained model; no custom training done.

## Credits

Model and original code: [minivision-ai/Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing) (Apache 2.0 License). Backend API and React UI built by Nikhil Kumar.