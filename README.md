# Face Anti-Spoofing

The project is split into two independently runnable applications:

- `frontend/anti-spoof-ui` — React + Vite user interface
- `backend/Silent-Face-Anti-Spoofing` — FastAPI API and anti-spoofing models

## Run locally on Windows

Open two terminals from this project directory.

In the first terminal, start the API:

```powershell
npm run dev:api
```

In the second terminal, start the frontend:

```powershell
npm run dev
```

Open the Vite URL printed in the frontend terminal. The Vite development server
proxies `/api/predict` to the backend at `http://127.0.0.1:8000/predict`.

The backend command uses the existing virtual environment at
`backend/Silent-Face-Anti-Spoofing/venv`. If you create a new environment, install
the dependencies from that backend directory with
`python -m pip install -r requirements.txt`, then run Uvicorn from the same
directory with `python -m uvicorn app:app --host 127.0.0.1 --port 8000`.

## Frontend checks

```powershell
npm run build
npm run lint
```
