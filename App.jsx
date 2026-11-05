
import { useEffect, useState } from "react";
import {
  Camera, Radio, ShieldAlert, Activity,
  Wifi, WifiOff, Boxes
} from "lucide-react";
import "./App.css";

const API = "http://127.0.0.1:5000";

export default function App() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    let active = true;

    async function update() {
      try {
        const response = await fetch(`${API}/api/status`);
        if (!response.ok) throw new Error("API error");

        const result = await response.json();

        if (active) {
          setData(result);
          setError(false);
        }
      } catch {
        if (active) setError(true);
      }
    }

    update();
    const timer = setInterval(update, 700);

    return () => {
      active = false;
      clearInterval(timer);
    };
  }, []);

  const proximity = data?.proximity ?? "UNKNOWN";

  return (
    <main className="dashboard">
      <header className="topbar">
        <div className="brand">
          <div className="brand-icon"><Activity /></div>
          <div>
            <h1>VISION<span>GUARD</span></h1>
            <p>AI OBJECT DETECTION & PROXIMITY MONITOR</p>
          </div>
        </div>

        <div className="live-status">
          <span className="live-dot" />
          {error ? "BACKEND OFFLINE" : "SYSTEM MONITOR"}
        </div>
      </header>

      <section className="intro">
        <div>
          <p className="eyebrow">INTELLIGENT MONITORING SYSTEM</p>
          <h2>Perception Dashboard</h2>
          <p className="subtext">
            Live object classification and ultrasonic obstacle monitoring.
          </p>
        </div>

        <div className={`alert-badge ${proximity.toLowerCase()}`}>
          <ShieldAlert size={20} />
          {proximity}
        </div>
      </section>

      {error && (
        <div className="error-banner">
          Backend unavailable. Start the Python app.py server.
        </div>
      )}

      <section className="metrics">
        <article className="metric">
          <div className="metric-heading">
            <Camera /> <span>CAMERA</span>
          </div>
          <h3>{data?.camera_connected ? "Connected" : "Offline"}</h3>
          <p>Phone video stream</p>
        </article>

        <article className="metric">
          <div className="metric-heading">
            <Boxes /> <span>OBJECTS</span>
          </div>
          <h3>{data?.object_count ?? "--"}</h3>
          <p>Objects currently detected</p>
        </article>

        <article className="metric">
          <div className="metric-heading">
            <Radio /> <span>DISTANCE</span>
          </div>
          <h3>
            {data?.distance_cm != null
              ? `${data.distance_cm} cm`
              : "--"}
          </h3>
          <p>Ultrasonic sensor measurement</p>
        </article>

        <article className="metric">
          <div className="metric-heading">
            {data?.esp32_connected ? <Wifi /> : <WifiOff />}
            <span>ESP32</span>
          </div>
          <h3>{data?.esp32_connected ? "Connected" : "Offline"}</h3>
          <p>Sensor communication</p>
        </article>
      </section>

      <section className="main-grid">
        <article className="panel camera-panel">
          <div className="panel-heading">
            <div>
              <h3>Live AI Vision</h3>
              <p>YOLO object detection feed</p>
            </div>
            <span className="tag">LIVE FEED</span>
          </div>

          <div className="video-container">
            {data?.camera_connected && !error ? (
              <img src={`${API}/video_feed`} alt="Live AI camera feed" />
            ) : (
              <div className="placeholder">
                <Camera size={40} />
                <p>Waiting for camera stream...</p>
              </div>
            )}
          </div>
        </article>

        <article className="panel distance-panel">
          <div className="panel-heading">
            <div>
              <h3>Proximity Monitor</h3>
              <p>HC-SR04 ultrasonic sensor</p>
            </div>
            <Radio size={20} />
          </div>

          <div className={`distance-display ${proximity.toLowerCase()}`}>
            <strong>
              {data?.distance_cm != null
                ? data.distance_cm
                : "--"}
            </strong>
            <span>centimetres</span>
          </div>

          <div className="distance-scale">
            <div className="scale-track">
              <div
                className="scale-fill"
                style={{
                  width: data?.distance_cm == null
                    ? "0%"
                    : `${Math.min(data.distance_cm, 150) / 150 * 100}%`
                }}
              />
            </div>
            <div className="scale-labels">
              <span>0 cm</span><span>75 cm</span><span>150+ cm</span>
            </div>
          </div>

          <div className={`alert-message ${proximity.toLowerCase()}`}>
            <ShieldAlert size={19} />
            <div>
              <strong>{proximity}</strong>
              <p>
                {proximity === "DANGER"
                  ? "Obstacle detected within 30 cm."
                  : proximity === "WARNING"
                  ? "Obstacle detected within 100 cm."
                  : proximity === "CLEAR"
                  ? "No nearby obstacle within the warning range."
                  : "Waiting for a valid sensor reading."}
              </p>
            </div>
          </div>

          <p className="sensor-note">
            The ESP32 controls the buzzer independently.
          </p>
        </article>
      </section>

      <section className="panel objects-panel">
        <div className="panel-heading">
          <div>
            <h3>Detected Object Classification</h3>
            <p>YOLO class labels and model confidence</p>
          </div>
          <span className="tag">
            {data?.objects?.length ?? 0} DETECTIONS
          </span>
        </div>

        {data?.objects?.length > 0 ? (
          <div className="object-list">
            {data.objects.map((object, index) => (
              <div className="object-row" key={`${object.class}-${index}`}>
                <div className="object-symbol"><Boxes /></div>
                <div className="object-name">
                  <strong>{object.class}</strong>
                  <span>Detected by YOLO</span>
                </div>
                <div className="confidence">
                  <div className="confidence-track">
                    <div style={{ width: `${object.confidence}%` }} />
                  </div>
                  <span>{object.confidence}% confidence</span>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="empty-state">
            No objects detected in the current frame.
          </div>
        )}
      </section>

      <footer>
        VISIONGUARD · AI PERCEPTION + ULTRASONIC SAFETY MONITORING
        <span>Last update: {data?.updated_at ?? "--"}</span>
      </footer>
    </main>
  );
}
