import os
import io
import contextlib
import webbrowser
import threading
import multiprocessing
import uuid
import time
import numpy as np
from flask import Flask, render_template, request, send_file, jsonify
from chaos_benchmark import build_dataset

# Determine the directory of this file
WEB_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__, 
    template_folder=os.path.join(WEB_DIR, 'templates'),
    static_folder=os.path.join(WEB_DIR, 'static')
)

jobs = {}

def generation_worker(job_id, params):
    def cb(sys_name, label, counts):
        jobs[job_id]["progress"][sys_name] = counts.copy()
        
    def cancel_check():
        return jobs.get(job_id, {}).get("cancel", False)

    try:
        df = build_dataset(
            rows_per_class=params["rows"],
            max_attempts_per_class=params["attempts"],
            systems=params["systems"],
            workers=params.get("workers"),
            seed=42, # static seed for the benchmark UI
            verbose=False,
            progress_callback=cb,
            cancel_check=cancel_check
        )
        if cancel_check():
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = "Cancelled by user"
        else:
            jobs[job_id]["status"] = "complete"
            jobs[job_id]["result"] = df
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)

@app.route('/')
def index():
    cpu_count = multiprocessing.cpu_count()
    return render_template('index.html', cpu_count=cpu_count)

@app.route('/api/start', methods=['POST'])
def start_generation():
    # Cancel any currently running jobs to prevent multiple parallel background generations
    for j_id, j_data in jobs.items():
        if j_data["status"] == "running":
            j_data["cancel"] = True

    data = request.json
    rows_per_class = int(data.get('rows_per_class', 20))
    max_attempts = int(data.get('max_attempts', 100))
    workers = int(data.get('workers', multiprocessing.cpu_count() - 1))
    systems = data.get('systems', None)
    
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "running",
        "progress": {sys: {"Stable": 0, "Periodic": 0, "Chaotic": 0} for sys in (systems or [])},
        "result": None,
        "error": None,
        "config": {"rows_per_class": rows_per_class}
    }
    
    thread = threading.Thread(target=generation_worker, args=(job_id, {
        "rows": rows_per_class,
        "attempts": max_attempts,
        "systems": systems,
        "workers": workers
    }))
    thread.start()
    
    return jsonify({"job_id": job_id})

@app.route('/api/progress/<job_id>', methods=['GET'])
def get_progress(job_id):
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404
        
    job = jobs[job_id]
    response = {
        "status": job["status"],
        "progress": job["progress"]
    }
    
    if job["status"] == "complete":
        df = job["result"]
        # return a preview of first 3 rows, converting NaN to None for JSON
        preview = df.head(3).replace({np.nan: None}).to_dict(orient='records')
        response["preview"] = preview
    elif job["status"] == "error":
        response["error"] = job["error"]
        
    return jsonify(response)

@app.route('/api/download/<job_id>', methods=['GET'])
def download(job_id):
    if job_id not in jobs or jobs[job_id]["status"] != "complete":
        return "Job not ready", 400
        
    df = jobs[job_id]["result"]
    csv_data = df.to_csv(index=False)
    rows_per_class = jobs[job_id]["config"]["rows_per_class"]
    
    return send_file(
        io.BytesIO(csv_data.encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'chaos_dataset_{rows_per_class}_rows.csv'
    )

def main():
    port = 5000
    url = f"http://127.0.0.1:{port}"
    print(f"\nStarting Chaos Benchmark Web UI at {url}")
    print("Press CTRL+C to quit.\n")
    
    # Open the browser in a short delay so the server has time to bind
    threading.Timer(1.25, lambda: webbrowser.open(url)).start()
    
    # Run the server without debug mode to prevent reloading issues
    app.run(host='127.0.0.1', port=port, debug=False)

@app.route('/api/cancel/<job_id>', methods=['POST'])
def cancel_job(job_id):
    if job_id in jobs:
        jobs[job_id]["cancel"] = True
        return jsonify({"status": "cancelling"})
    return jsonify({"error": "not found"}), 404


if __name__ == '__main__':
    main()
