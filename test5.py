import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
from tkinter.scrolledtext import ScrolledText
import sqlite3
import cv2
import threading
import queue
import datetime
import time
import sys
from fast_alpr import ALPR


DB_FILE = "gate_logs.db"
COOLDOWN = 10  # seconds
last_seen = {}  # {plate: last_detection_time}
stop_event = threading.Event()
alpr_lock = threading.Lock()  # protect ALPR calls
last_seen_lock = threading.Lock()
_last_seen_cleanup = 0  # timestamp for periodic cleanup


# --- Queues and camera list ---
frame_raw_queues = {}
frame_disp_queues = {}
capture_threads = []
process_threads = []

# --- Default camera keys ---
CAM_KEYS = ("Entrance", "Exit")

# --- DB init (authorized_people, logs, settings) ---
def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS authorized_people (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        plate TEXT NOT NULL UNIQUE
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        plate TEXT NOT NULL,
        name TEXT,
        camera TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        authorized INTEGER DEFAULT 0
    )
    """)
    

        
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY,
        value TEXT
    )
    """)
    # default camera urls (only inserted if not present)
    cursor.execute("INSERT OR IGNORE INTO settings (key, value) VALUES ('Entrance', 'rtsp://admin:admin@192.168.1.35:554/stream')")
    cursor.execute("INSERT OR IGNORE INTO settings (key, value) VALUES ('Exit', 'rtsp://admin:admin@192.168.1.35:554/stream2')")
    conn.commit()
    conn.close()

init_db()

# --- Helper DB functions ---

class StdoutRedirector:
    def __init__(self, widget):
        self.widget = widget

    def write(self, string):
        if self.widget.winfo_exists():  # check widget still exists
            self.widget.insert(tk.END, string)
            self.widget.see(tk.END)
        else:
            # fallback to console if widget destroyed
            print(string, end='')

    def flush(self):
        pass

def get_setting(key):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("SELECT value FROM settings WHERE key=?", (key,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else ""

def set_setting(key, value):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, value))
    conn.commit()
    conn.close()

# --- Logger ---
def log_event(plate, camera):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM authorized_people WHERE plate = ?", (plate,))
    row = cursor.fetchone()
    name = row[0] if row else "Unknown"
    authorized = 1 if row else 0
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        "INSERT INTO logs (plate, name, camera, timestamp, authorized) VALUES (?, ?, ?, ?, ?)",
        (plate, name, camera, timestamp, authorized),
    )
    conn.commit()
    conn.close()
    print(f"📌 Logged: {plate} - {name} ({camera}) @ {timestamp}")
    return bool(authorized), name


def clear_last_seen_older_than(seconds=300):
    """Cleanup old entries to keep last_seen small."""
    now = time.time()
    for p in list(last_seen.keys()):
        if now - last_seen[p] > seconds:
            del last_seen[p]

# --- Gate (stub) ---
def open_gate(camera):
    print(f"🚪 Gate OPEN triggered at {camera}!")

# --- ALPR instance (shared) ---
alpr = ALPR(
    detector_model="yolo-v9-t-384-license-plate-end2end",
    ocr_model="cct-xs-v1-global-model"
)

# --- Capture / Process functions ---
def capture_camera(rtsp_url, cam_name):
    """Continuously read frames from camera, auto-reconnect if stream fails."""
    cap = None
    while not stop_event.is_set():
        if cap is None or not cap.isOpened():
            print(f"[INFO] Connecting to {cam_name}...")
            cap = cv2.VideoCapture(rtsp_url)

            if not cap.isOpened():
                print(f"[ERROR] Cannot open {cam_name}. Retrying in 5 seconds...")
                time.sleep(5)
                continue

            print(f"[INFO] {cam_name} connected!")

        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"[WARN] Lost connection to {cam_name}, restarting in 3s...")
            cap.release()
            cap = None
            time.sleep(3)
            continue

        # replace old raw frame
        q = frame_raw_queues[cam_name]
        try:
            if q.full():
                q.get_nowait()
        except queue.Empty:
            pass
        q.put(frame)

    if cap:
        cap.release()


def process_camera(cam_name):
    q_raw = frame_raw_queues[cam_name]
    q_disp = frame_disp_queues[cam_name]
    motion_check_interval = 3  # seconds
    last_motion_check = 0
    motion_cooldown = 0
    cooldown_time = 3
    frame_count = 0
    skip_rate = 5
    global _last_seen_cleanup

    # --- Background subtractor (keep per camera) ---
    backsub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

    while not stop_event.is_set():
        try:
            frame = q_raw.get(timeout=0.2)
        except queue.Empty:
            continue

        frame_count += 1
        now = time.time()

        # --- Motion detection using background subtractor ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small_gray = cv2.resize(gray, (0, 0), fx=0.25, fy=0.25)  # downscale

        fg_mask = backsub.apply(small_gray)

        # clean up noise
        fg_mask = cv2.medianBlur(fg_mask, 5)

        motion_level = cv2.countNonZero(fg_mask)
        motion_detected = False  # NEW FLAG

        if now - last_motion_check >= motion_check_interval:
            last_motion_check = now

            if motion_level > 2500:  # adjust sensitivity for downscaled size
                motion_cooldown = now
                motion_detected = True  # <-- first frame after motion
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG] Motion={motion_level} on {cam_name}")

            run_alpr = (now - motion_cooldown <= cooldown_time)
            

            # --- Run ALPR only on full frame ---
            if run_alpr and (frame_count % skip_rate == 0 or motion_detected):
                

                with alpr_lock:
                    try:
                        results = alpr.predict(frame)  # full frame
                    except Exception as e:
                        print("ALPR predict error:", e)
                        results = []

                for res in results:
                    raw_plate = res.ocr.text.strip()

                    # Normalize plate: uppercase + only alphanumeric chars
                    plate = "".join(ch for ch in raw_plate.upper() if ch.isalnum())

                    confidence = res.ocr.confidence

                    if confidence > 0.95 and plate:
                        now = time.time()

                        # thread-safe check/update of cooldown
                        with last_seen_lock:
                            last = last_seen.get(plate)

                            if last is not None and (now - last) < COOLDOWN:
                                # still in cooldown, skip
                                continue

                            # not in cooldown: record detection time immediately
                            last_seen[plate] = now

                        # log and open gate (DB log uses normalized plate)
                        authorized, _ = log_event(plate, cam_name)
                        if authorized:
                            open_gate(cam_name)

                # Only draw overlays if showing feed
                if SHOW_FEED:
                    with alpr_lock:
                        try:
                            annotated = alpr.draw_predictions(frame)
                        except Exception:
                            annotated = frame
                else:
                    annotated = None
            else:
                # Only keep raw frame if showing feed
                annotated = frame if SHOW_FEED else None

            # --- push frame to display queue (only if SHOW_FEED) ---
            if SHOW_FEED and annotated is not None:
                try:
                    if q_disp.full():
                        q_disp.get_nowait()
                except queue.Empty:
                    pass
                q_disp.put(annotated)

            # periodic cleanup (run once every 30 seconds)
            if now - _last_seen_cleanup > 30:
                with last_seen_lock:
                    cutoff = now - 600  # remove entries older than 10 minutes
                    for p in list(last_seen.keys()):
                        if last_seen[p] < cutoff:
                            del last_seen[p]
                _last_seen_cleanup = now



def toggle_feed():
    global SHOW_FEED
    SHOW_FEED = var_show_feed.get()
    if not SHOW_FEED:
        cv2.destroyAllWindows()  # close any open feed windows immediately

class RedirectText(object):
    def __init__(self, text_widget):
        self.output = text_widget

    def write(self, string):
        try:
            if not self.output.winfo_exists():
                # widget destroyed -> fallback to stdout
                sys.__stdout__.write(string)
                return
        except Exception:
            sys.__stdout__.write(string)
            return

        # schedule insert in main thread
        def _append():
            try:
                self.output.insert(tk.END, string)
                self.output.see(tk.END)
            except Exception:
                # if widget got destroyed between scheduling and execution
                pass

        try:
            self.output.after(0, _append)
        except Exception:
            # fallback
            sys.__stdout__.write(string)

    def flush(self):
        pass



def open_activity_window():
    global log_window, log_text
    if 'log_window' in globals() and log_window.winfo_exists():
        log_window.lift()  # bring existing window to front
        return

    # Create a new top-level window
    log_window = tk.Toplevel(root)
    log_window.title("Activity Log")
    log_window.geometry("600x400")

    # Add a scrollable text widget
    log_text = ScrolledText(log_window, wrap=tk.WORD, state="normal")
    log_text.pack(expand=True, fill="both")

    # Redirect stdout (print) to this text box
    sys.stdout = RedirectText(log_text)
    sys.stderr = RedirectText(log_text)

    print("✅ Activity window started...")
    print("All future prints will appear here!")

    # Restore normal stdout/stderr when window is closed
    def on_close():
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        log_window.destroy()
    log_window.protocol("WM_DELETE_WINDOW", on_close)

# --- GUI functions ---
def manage_plates():
    window = tk.Toplevel(root)
    window.title("Manage Authorized Plates")
    window.geometry("500x400")

    tree = ttk.Treeview(window, columns=("id", "name", "plate"), show="headings")
    tree.heading("id", text="ID")
    tree.heading("name", text="Name")
    tree.heading("plate", text="Plate")
    tree.column("id", width=40)
    tree.pack(fill=tk.BOTH, expand=True)

    def load_data():
        for i in tree.get_children():
            tree.delete(i)
        conn = sqlite3.connect(DB_FILE)
        cur = conn.cursor()
        cur.execute("SELECT id, name, plate FROM authorized_people ORDER BY id DESC")
        for row in cur.fetchall():
            tree.insert("", tk.END, values=row)
        conn.close()

    def add_entry():
        name = simpledialog.askstring("Input", "Enter name:", parent=window)
        plate = simpledialog.askstring("Input", "Enter plate:", parent=window)
        if not (name and plate):
            return
        conn = sqlite3.connect(DB_FILE)
        cur = conn.cursor()
        try:
            cur.execute("INSERT INTO authorized_people (name, plate) VALUES (?, ?)", (name, plate))
            conn.commit()
            messagebox.showinfo("Success", f"Added {name} - {plate}", parent=window)
            load_data()
        except sqlite3.IntegrityError:
            messagebox.showerror("Error", "Plate already exists!", parent=window)
        conn.close()

    def delete_entry():
        sel = tree.selection()
        if not sel:
            return
        item = tree.item(sel[0])
        pid = item["values"][0]
        if not messagebox.askyesno("Confirm", "Delete selected entry?", parent=window):
            return
        conn = sqlite3.connect(DB_FILE)
        cur = conn.cursor()
        cur.execute("DELETE FROM authorized_people WHERE id=?", (pid,))
        conn.commit()
        conn.close()
        load_data()

    btns = tk.Frame(window)
    btns.pack(fill=tk.X)
    tk.Button(btns, text="➕ Add", command=add_entry).pack(side=tk.LEFT, padx=5, pady=5)
    tk.Button(btns, text="🗑️ Delete", command=delete_entry).pack(side=tk.LEFT, padx=5, pady=5)

    load_data()

def view_logs():
    window = tk.Toplevel(root)
    window.title("Logs")
    window.geometry("1100x600")

    # --- Filters ---
    filter_frame = tk.LabelFrame(window, text="🔍 Search Filters")
    filter_frame.pack(fill=tk.X, padx=8, pady=8)

    tk.Label(filter_frame, text="Name:").grid(row=0, column=0, padx=4, pady=4, sticky="e")
    entry_name = tk.Entry(filter_frame, width=20)
    entry_name.grid(row=0, column=1, padx=4, pady=4)

    tk.Label(filter_frame, text="Plate:").grid(row=0, column=2, padx=4, pady=4, sticky="e")
    entry_plate = tk.Entry(filter_frame, width=20)
    entry_plate.grid(row=0, column=3, padx=4, pady=4)

    tk.Label(filter_frame, text="From (YYYY-MM-DD):").grid(row=1, column=0, padx=4, pady=4, sticky="e")
    entry_from = tk.Entry(filter_frame, width=20)
    entry_from.grid(row=1, column=1, padx=4, pady=4)

    tk.Label(filter_frame, text="To (YYYY-MM-DD):").grid(row=1, column=2, padx=4, pady=4, sticky="e")
    entry_to = tk.Entry(filter_frame, width=20)
    entry_to.grid(row=1, column=3, padx=4, pady=4)

    tk.Label(filter_frame, text="Authorized:").grid(row=2, column=0, padx=4, pady=4, sticky="e")
    combo_auth = ttk.Combobox(filter_frame, values=["All", "Authorized", "Unauthorized"], state="readonly", width=18)
    combo_auth.current(0)  # default = All
    combo_auth.grid(row=2, column=1, padx=4, pady=4)

    # --- Treeview ---
    columns = ("id", "plate", "name", "camera", "timestamp", "authorized")
    tree = ttk.Treeview(window, columns=columns, show="headings")
    for c in columns:
        tree.heading(c, text=c.capitalize())
        tree.column(c, width=160 if c != "id" else 60)
    tree.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

    # --- Load logs with filters ---
    def load_logs(limit=500):
        for r in tree.get_children():
            tree.delete(r)

        conn = sqlite3.connect(DB_FILE)
        cur = conn.cursor()

        query = "SELECT id, plate, name, camera, timestamp, authorized FROM logs WHERE 1=1"
        params = []

        # filters
        if entry_name.get().strip():
            query += " AND name LIKE ?"
            params.append(f"%{entry_name.get().strip()}%")
        if entry_plate.get().strip():
            query += " AND plate LIKE ?"
            params.append(f"%{entry_plate.get().strip()}%")
        if entry_from.get().strip():
            query += " AND date(timestamp) >= date(?)"
            params.append(entry_from.get().strip())
        if entry_to.get().strip():
            query += " AND date(timestamp) <= date(?)"
            params.append(entry_to.get().strip())
        if combo_auth.get() == "Authorized":
            query += " AND authorized = 1"
        elif combo_auth.get() == "Unauthorized":
            query += " AND authorized = 0"

        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        cur.execute(query, params)
        for row in cur.fetchall():
            # Convert authorized int to text
            row = list(row)
            row[5] = "Yes" if row[5] == 1 else "No"
            tree.insert("", tk.END, values=row)

        conn.close()

    # --- Buttons ---
    toolbar = tk.Frame(window)
    toolbar.pack(fill=tk.X, padx=8, pady=4)

    tk.Button(toolbar, text="🔍 Search", command=load_logs).pack(side=tk.LEFT, padx=5)
    tk.Button(toolbar, text="🔄 Refresh", command=lambda: load_logs()).pack(side=tk.LEFT, padx=5)
    tk.Button(toolbar, text="Export CSV", command=lambda: export_logs_csv(window)).pack(side=tk.LEFT, padx=5)

    load_logs()


   

def export_logs_csv(parent):
    import csv
    fname = simpledialog.askstring("Save CSV", "Enter filename (e.g. logs.csv):", parent=parent)
    if not fname:
        return
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("SELECT plate, name, camera, timestamp FROM logs ORDER BY id DESC")
    rows = cur.fetchall()
    conn.close()
    with open(fname, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["plate", "name", "camera", "timestamp"])
        writer.writerows(rows)
    messagebox.showinfo("Exported", f"Saved to {fname}", parent=parent)

def camera_settings():
    window = tk.Toplevel(root)
    window.title("Camera Settings")
    window.geometry("700x160")

    tk.Label(window, text="Entrance Camera URL:").pack(anchor="w", padx=8)
    entrance_entry = tk.Entry(window, width=90)
    entrance_entry.pack(padx=8, pady=4)

    tk.Label(window, text="Exit Camera URL:").pack(anchor="w", padx=8)
    exit_entry = tk.Entry(window, width=90)
    exit_entry.pack(padx=8, pady=4)

    # load current
    entrance_entry.insert(0, get_setting("Entrance"))
    exit_entry.insert(0, get_setting("Exit"))

    def save():
        set_setting("Entrance", entrance_entry.get().strip())
        set_setting("Exit", exit_entry.get().strip())
        messagebox.showinfo("Saved", "Camera URLs saved.", parent=window)

    tk.Button(window, text="💾 Save", command=save).pack(pady=8)

# --- Start/Stop system and frame updater ---
def start_system():
    if stop_event.is_set():
        # was stopped previously — reset
        stop_event.clear()
    # initialize queues using URLs from DB
    global frame_raw_queues, frame_disp_queues, capture_threads, process_threads
    frame_raw_queues = {}
    frame_disp_queues = {}
    capture_threads = []
    process_threads = []

    for key in CAM_KEYS:
        frame_raw_queues[key] = queue.Queue(maxsize=1)
        frame_disp_queues[key] = queue.Queue(maxsize=1)

    # spawn capture/process threads
    for key in CAM_KEYS:
        url = get_setting(key)
        tcap = threading.Thread(target=capture_camera, args=(url, key), daemon=True)
        tproc = threading.Thread(target=process_camera, args=(key,), daemon=True)
        tcap.start()
        tproc.start()
        capture_threads.append(tcap)
        process_threads.append(tproc)

    # disable start, enable stop
    btn_start.config(state=tk.DISABLED)
    btn_stop.config(state=tk.NORMAL)

    # start display updater (runs in main thread via after)
    root.after(30, update_frames)
    messagebox.showinfo("System", "ALPR system started.\nUse Stop System to terminate.")

def stop_system():
    stop_event.set()
    # give threads a moment to exit
    time.sleep(0.2)
    btn_start.config(state=tk.NORMAL)
    btn_stop.config(state=tk.DISABLED)
    # Destroy any OpenCV windows
    if SHOW_FEED:
        cv2.destroyAllWindows()
    messagebox.showinfo("System", "ALPR system stopped.")

def update_frames():
    """Run in main thread with after(). Shows frames from display queues."""
    if stop_event.is_set():
        return
    for key in CAM_KEYS:
        q = frame_disp_queues.get(key)
        if q and not q.empty():
            try:
                frame = q.get_nowait()
            except queue.Empty:
                frame = None
            if frame is not None and SHOW_FEED:
                cv2.imshow(f"ALPR - {key}", frame)
    # re-schedule
    root.after(30, update_frames)

# --- Main window UI ---
root = tk.Tk()
root.title("Gate Control System")
root.geometry("420x380")
SHOW_FEED = True
var_show_feed = tk.BooleanVar(value=True)

tk.Label(root, text="Gate Control System", font=("Segoe UI", 14)).pack(pady=8)

btn_frame = tk.Frame(root)
btn_frame.pack(pady=12)

btn_manage = tk.Button(btn_frame, text="👥 Manage Plates", width=22, command=manage_plates)
btn_settings = tk.Button(btn_frame, text="🎥 Camera Settings", width=22, command=camera_settings)
btn_logs = tk.Button(btn_frame, text="📜 View Logs", width=22, command=view_logs)
btn_start = tk.Button(btn_frame, text="▶ Start Camera System", width=22, command=start_system)
btn_stop = tk.Button(btn_frame, text="⏹ Stop System", width=22, command=stop_system, state=tk.DISABLED)
chk_show_feed = tk.Checkbutton(root, text="Show Camera Feed", variable=var_show_feed, command=toggle_feed)
chk_show_feed.pack(pady=5)
btn_open_log = tk.Button(root, text="Open Activity Window", command=open_activity_window)
btn_open_log.pack(pady=20)

btn_manage.grid(row=0, column=0, padx=8, pady=6)
btn_settings.grid(row=0, column=1, padx=8, pady=6)
btn_logs.grid(row=1, column=0, padx=8, pady=6)
btn_start.grid(row=1, column=1, padx=8, pady=6)
btn_stop.grid(row=2, column=0, columnspan=2, pady=10)

# ensure clean shutdown if window closed
def on_closing():
    if messagebox.askokcancel("Quit", "Stop system and exit?"):
        stop_event.set()
        time.sleep(0.2)
        root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)
start_system()
root.mainloop()
