import sqlite3
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from datetime import datetime

DB_FILE = "gate_logs.db"

def load_logs(plate_filter="", from_date="", to_date=""):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    query = "SELECT plate, name, camera, timestamp FROM logs WHERE 1=1"
    params = []

    # Plate search
    if plate_filter:
        query += " AND plate LIKE ?"
        params.append(f"%{plate_filter}%")

    # Date filters
    if from_date:
        query += " AND date(timestamp) >= date(?)"
        params.append(from_date)
    if to_date:
        query += " AND date(timestamp) <= date(?)"
        params.append(to_date)

    query += " ORDER BY id DESC"
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    return rows

def refresh_table():
    plate_filter = plate_entry.get().strip()
    from_date = from_date_entry.get().strip()
    to_date = to_date_entry.get().strip()

    try:
        # Validate dates
        if from_date:
            datetime.strptime(from_date, "%Y-%m-%d")
        if to_date:
            datetime.strptime(to_date, "%Y-%m-%d")
    except ValueError:
        messagebox.showerror("Date Error", "Dates must be in YYYY-MM-DD format")
        return

    for row in tree.get_children():
        tree.delete(row)

    for plate, name, camera, timestamp in load_logs(plate_filter, from_date, to_date):
        tree.insert("", "end", values=(plate, name, camera, timestamp))

# GUI setup
root = tk.Tk()
root.title("Gate Access Logs")
root.geometry("1000x600")

# Filters frame
filter_frame = tk.Frame(root)
filter_frame.pack(pady=10)

tk.Label(filter_frame, text="Plate:").grid(row=0, column=0, padx=5)
plate_entry = tk.Entry(filter_frame)
plate_entry.grid(row=0, column=1, padx=5)

tk.Label(filter_frame, text="From (YYYY-MM-DD):").grid(row=0, column=2, padx=5)
from_date_entry = tk.Entry(filter_frame)
from_date_entry.grid(row=0, column=3, padx=5)

tk.Label(filter_frame, text="To (YYYY-MM-DD):").grid(row=0, column=4, padx=5)
to_date_entry = tk.Entry(filter_frame)
to_date_entry.grid(row=0, column=5, padx=5)

search_button = tk.Button(filter_frame, text="🔍 Search", command=refresh_table)
search_button.grid(row=0, column=6, padx=10)

# Table
columns = ("Plate", "Name", "Camera", "Timestamp")
tree = ttk.Treeview(root, columns=columns, show="headings")

for col in columns:
    tree.heading(col, text=col)
    tree.column(col, width=200)

tree.pack(fill="both", expand=True)

# Load initial data
refresh_table()

root.mainloop()
