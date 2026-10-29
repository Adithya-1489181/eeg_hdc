import json
import tempfile
import traceback
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

from inference import SeizureAnalyzer

MAX_UPLOAD_MB = 800

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


@app.errorhandler(413)
def handle_too_large(e):
    return jsonify({
        "error": f"File too large. The server accepts recordings up to "
                 f"{MAX_UPLOAD_MB} MB; please upload a shorter clip.",
    }), 413


@app.errorhandler(HTTPException)
def handle_http_exception(e):
    return jsonify({"error": e.description}), e.code

analyzer = SeizureAnalyzer()
models_loaded = False
classifier_available = False


def _init_models():
    global models_loaded, classifier_available
    try:
        analyzer.load_models()
        models_loaded = True
        classifier_available = analyzer.classifier is not None
    except Exception as e:
        print(f"Warning: Could not load models: {e}")
        models_loaded = False
        classifier_available = False


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    return jsonify({
        "models_loaded": models_loaded,
        "classifier_available": classifier_available,
    })


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file.filename.lower().endswith(".edf"):
        return jsonify({"error": "Only .edf files are supported"}), 400

    line_freq = float(request.form.get("line_freq", 50.0))
    threshold = float(request.form.get("threshold", 0.5))
    min_event_s = float(request.form.get("min_event_s", 8.0))

    tmp = None
    try:
        suffix = Path(file.filename).suffix
        tmp = tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix, dir=str(Path(tempfile.gettempdir()))
        )
        file.save(tmp.name)
        tmp.close()

        results = analyzer.analyze(
            tmp.name, line_freq=line_freq, detection_threshold=threshold,
            min_event_s=min_event_s,
        )
        return jsonify(results)

    except ValueError as e:
        return jsonify({"error": str(e)}), 422
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500
    finally:
        if tmp:
            Path(tmp.name).unlink(missing_ok=True)


_init_models()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
