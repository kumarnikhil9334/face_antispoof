import { useCallback, useEffect, useRef, useState } from 'react'
import './App.css'

const API_URL = import.meta.env.VITE_API_URL || '/api/predict'
const CHECK_INTERVAL_MS = 1500

function App() {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const streamRef = useRef(null)
  const intervalRef = useRef(null)
  const requestRef = useRef(null)
  const runningRef = useRef(false)
  const requestInFlightRef = useRef(false)
  const [isStarting, setIsStarting] = useState(false)
  const [isChecking, setIsChecking] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')

  const captureAndPredict = useCallback(async () => {
    if (!runningRef.current || requestInFlightRef.current) return

    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas || !video.videoWidth || !video.videoHeight) return

    requestInFlightRef.current = true
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    const context = canvas.getContext('2d')
    if (!context) {
      requestInFlightRef.current = false
      setError('Could not capture a frame from the camera.')
      return
    }

    context.drawImage(video, 0, 0, canvas.width, canvas.height)
    const frame = await new Promise((resolve) => {
      canvas.toBlob(resolve, 'image/jpeg', 0.85)
    })
    if (!frame) {
      requestInFlightRef.current = false
      setError('Could not encode the camera frame.')
      return
    }

    const controller = new AbortController()
    requestRef.current = controller
    const formData = new FormData()
    formData.append('file', frame, 'frame.jpg')

    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        body: formData,
        signal: controller.signal,
      })
      const data = await response.json()
      if (!response.ok) {
        throw new Error(data.error || `Prediction failed (${response.status}).`)
      }

      if (runningRef.current) {
        setResult(data)
        setError('')
      }
    } catch (requestError) {
      if (requestError.name !== 'AbortError' && runningRef.current) {
        setError(requestError.message || 'Could not reach the prediction API.')
      }
    } finally {
      requestInFlightRef.current = false
      if (requestRef.current === controller) requestRef.current = null
    }
  }, [])

  const stopLiveCheck = useCallback(() => {
    runningRef.current = false
    window.clearInterval(intervalRef.current)
    intervalRef.current = null
    requestRef.current?.abort()
    requestRef.current = null
    streamRef.current?.getTracks().forEach((track) => track.stop())
    streamRef.current = null
    if (videoRef.current) videoRef.current.srcObject = null
    setIsChecking(false)
  }, [])

  const startLiveCheck = async () => {
    setIsStarting(true)
    setError('')
    setResult(null)

    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error('Camera access is not available in this browser.')
      }
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { facingMode: 'user' },
      })
      streamRef.current = stream
      videoRef.current.srcObject = stream
      await videoRef.current.play()

      runningRef.current = true
      setIsChecking(true)
      void captureAndPredict()
      intervalRef.current = window.setInterval(captureAndPredict, CHECK_INTERVAL_MS)
    } catch (cameraError) {
      stopLiveCheck()
      setError(cameraError.message || 'Could not start the camera.')
    } finally {
      setIsStarting(false)
    }
  }

  useEffect(() => {
    return () => {
      runningRef.current = false
      window.clearInterval(intervalRef.current)
      requestRef.current?.abort()
      streamRef.current?.getTracks().forEach((track) => track.stop())
    }
  }, [])

  const resultClass = result?.is_real ? 'real' : 'fake'

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="/" aria-label="Liveness Lab home">
          <span className="brand-mark" aria-hidden="true">L</span>
          <span>Liveness Lab</span>
        </a>
        <span className="api-indicator">
          <span className="indicator-dot" />
          Local API · :8000
        </span>
      </header>

      <section className="intro">
        <p className="eyebrow">REAL-TIME FACE ANALYSIS</p>
        <h1>Is this face <span>real?</span></h1>
        <p className="intro-copy">
          Start a live check to analyze camera frames for presentation attacks.
        </p>
      </section>

      <section className="workspace" aria-label="Live face anti-spoofing check">
        <div className="camera-card">
          <div className="card-heading">
            <div>
              <p className="card-label">CAMERA FEED</p>
              <p className="card-description">Position your face in the frame</p>
            </div>
            <span className={`live-badge${isChecking ? ' active' : ''}`}>
              <span className="indicator-dot" />
              {isChecking ? 'LIVE' : 'STANDBY'}
            </span>
          </div>

          <div className="video-frame">
            <video ref={videoRef} autoPlay muted playsInline />
            {!isChecking && (
              <div className="camera-placeholder">
                <span className="camera-icon" aria-hidden="true">◉</span>
                <span>Camera preview will appear here</span>
              </div>
            )}
            {isChecking && <div className="face-guide" aria-hidden="true" />}
          </div>
          <canvas ref={canvasRef} className="capture-canvas" aria-hidden="true" />

          <button
            className={`start-button${isChecking ? ' stop-button' : ''}`}
            type="button"
            onClick={isChecking ? stopLiveCheck : startLiveCheck}
            disabled={isStarting}
          >
            {isStarting
              ? 'Opening camera…'
              : isChecking
                ? 'Stop Live Check'
                : 'Start Live Check'}
          </button>
          <p className="camera-note">Camera frames are processed locally by your API.</p>
        </div>

        <aside className="result-card" aria-live="polite">
          <p className="card-label">ANALYSIS RESULT</p>
          {result ? (
            <div className={`result-content ${resultClass}`}>
              <span className="result-icon" aria-hidden="true">
                {result.is_real ? '✓' : '!'}
              </span>
              <p className="result-label">{result.label}</p>
              <p className="result-score">{(result.score * 100).toFixed(1)}%</p>
              <p className="result-caption">Model confidence</p>
            </div>
          ) : (
            <div className="result-empty">
              <span className="result-empty-icon" aria-hidden="true">◎</span>
              <p>{isChecking ? 'Waiting for first frame…' : 'No analysis yet'}</p>
              <span>Start the camera to see a result.</span>
            </div>
          )}
          {error && <p className="error-message" role="alert">{error}</p>}
          <div className="result-legend">
            <span><i className="legend-dot real-dot" /> REAL</span>
            <span><i className="legend-dot fake-dot" /> FAKE</span>
          </div>
        </aside>
      </section>

      <footer className="footer">
        <span>Silent Face Anti-Spoofing</span>
        <span>Local development · localhost</span>
      </footer>
    </main>
  )
}

export default App
