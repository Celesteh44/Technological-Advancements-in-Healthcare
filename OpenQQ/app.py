from flask import Flask, render_template_string, request, jsonify, Blueprint
import sqlite3
import os
import openai
from PIL import Image
import torch
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel

app = Flask(__name__)

# --- Local Cache Setup (SQLite) ---
DB_PATH = 'cache.db'

FAQ_DOC = {
    "What is OpenEMR?": "OpenEMR is a free and open-source electronic health records and medical practice management application.",
    "What is OpenMRS?": "OpenMRS is a platform that provides a customizable EMR system for low-resource environments."
}

HEALTHCARE_TECH = [
    {"name": "OpenMRS", "description": "Customizable EMR system.", "url": "https://openmrs.org/"},
    {"name": "OpenEMR", "description": "Open-source EHR and practice management.", "url": "https://www.open-emr.org/"},
    {"name": "GNU Health", "description": "Health and hospital information system.", "url": "https://www.gnuhealth.org/"}
]

def init_db():
    if not os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''CREATE TABLE cache (prompt TEXT PRIMARY KEY, response TEXT)''')
        c.execute('''CREATE TABLE tech (name TEXT, description TEXT, url TEXT)''')
        c.executemany('INSERT INTO tech VALUES (?, ?, ?)', [(d['name'], d['description'], d['url']) for d in HEALTHCARE_TECH])
        conn.commit()
        conn.close()

init_db()

# --- AI Blueprint ---
ai_routes = Blueprint('ai_routes', __name__)

@ai_routes.route('/api/ask', methods=['POST'])
def ask():
    prompt = request.json.get('prompt')
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT response FROM cache WHERE prompt = ?", (prompt,))
    row = c.fetchone()
    if row:
        return jsonify({"response": row[0]})

    results = c.execute("SELECT name, description FROM tech WHERE name LIKE ? OR description LIKE ?", (f"%{prompt}%", f"%{prompt}%")).fetchall()
    if results:
        response = '\n'.join([f"{r[0]}: {r[1]}" for r in results])
    else:
        openai.api_key = os.getenv("OPENAI_API_KEY")
        try:
            result = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
            response = result['choices'][0]['message']['content']
        except Exception as e:
            response = f"[Error calling model: {e}]"

    c.execute("INSERT INTO cache (prompt, response) VALUES (?, ?)", (prompt, response))
    conn.commit()
    conn.close()

    return jsonify({"response": response})

@ai_routes.route('/api/autotag', methods=['POST'])
def autotag():
    file = request.files.get('image')
    if not file:
        return jsonify({"error": "No image uploaded."}), 400

    image = Image.open(file.stream).convert("RGB")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    inputs = processor(text=["robot", "sensor", "healthcare", "technology"], images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    tags = ["robot", "sensor", "healthcare", "technology"]
    top_tag = tags[probs.argmax().item()]

    return jsonify({"label": top_tag})

app.register_blueprint(ai_routes)

# --- HTML Template ---
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="Explore the latest technologies revolutionizing healthcare information systems and patient care.">
    <title>Innovations in Healthcare Technology</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #fdf1f7;
            color: #333;
            margin: 0;
            padding: 0;
        }
        header {
            background: #ffc0cb;
            color: #880e4f;
            padding: 20px;
            text-align: center;
        }
        main {
            padding: 20px;
            max-width: 900px;
            margin: auto;
        }
        section {
            margin-bottom: 40px;
        }
        h2 {
            color: #880e4f;
        }
        footer {
            text-align: center;
            padding: 20px;
            background-color: #ffc0cb;
            color: white;
        }
        img {
            max-width: 100%;
            height: auto;
            margin: 20px 0;
            border-radius: 8px;
        }
        a {
            color: #d81b60;
            text-decoration: none;
        }
        a:hover {
            text-decoration: underline;
        }
        textarea, input[type='text'] {
            width: 100%;
            height: 100px;
            margin-top: 20px;
            padding: 10px;
            font-size: 1em;
        }
            @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        details {
            animation: fadeIn 0.6s ease-in-out both;
        }
        details:hover {
            transform: scale(1.02);
            transition: transform 0.3s ease;
        }
        img {
            transition: transform 0.4s ease-in-out;
        }
        img:hover {
            transform: scale(1.05);
        }
        </style>
</head>
<body>
    <header>
        <h1>Technological Advancements in Healthcare</h1>
        <p>Empowering Patients and Providers with Modern Innovations</p>
        <img src="/static/healthcare_header.png" alt="Healthcare Technology Coding Illustration" onerror="this.onerror=null;this.src='/static/fallback.jpg';">
    </header>
    <main>
        <section>
            <h2>Search Healthcare Technologies</h2>
            <input type="text" id="promptInput" placeholder="Search the database or ask a question...">
            <button onclick="submitPrompt()">Submit</button>
            <pre id="responseOutput"></pre>
        </section>
        <section>
            <h2>Upload and Auto-Tag 3D Asset Image</h2>
            <input type="file" id="imageInput" accept="image/*">
            <button onclick="submitImage()">Upload & Tag</button>
            <p id="tagResult"></p>
        </section>
    <section>
            <h2>Latest Innovations in Healthcare Technology</h2>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                <details open style="background: #fff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); transition: all 0.3s ease-in-out;">
                    <summary><strong>Smart Wearables</strong></summary>
                    Devices like <a href="https://www.apple.com/apple-watch/" target="_blank">Apple Watch</a> and <a href="https://www.fitbit.com/global/us/home" target="_blank">Fitbit</a> detect heart abnormalities and blood oxygen levels.<br>
                    <img src="/static/smart_wearables.jpg" alt="Smart Watch" onerror="this.onerror=null;this.src='/static/fallback.jpg';" width="300">
                </details>
                <details style="background: #fff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                    <summary><strong>AI-Powered Diagnostics</strong></summary>
                    Tools like <a href="https://health.google/health-research/" target="_blank">Google Health AI</a> diagnose diabetic retinopathy and skin conditions.<br>
                    <img src="/static/ai_diagnostics.jpg" alt="AI Diagnostics" onerror="this.onerror=null;this.src='/static/fallback.jpg';" width="300">
                </details>
                <details style="background: #fff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                    <summary><strong>Telehealth 2.0</strong></summary>
                    Enhanced platforms integrate real-time data from home devices. <a href="https://www.cdc.gov/telehealth/" target="_blank">Learn More</a><br>
                    <img src="/static/telehealth.jpg" alt="Telehealth" onerror="this.onerror=null;this.src='/static/fallback.jpg';" width="300">
                </details>
                <details style="background: #fff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                    <summary><strong>Personalized Genomic Medicine</strong></summary>
                    <a href="https://www.genome.gov/health/Genomics-and-Medicine" target="_blank">Genome sequencing</a> helps tailor treatments to individuals.<br>
                    <img src="/static/genomics.jpg" alt="Genomics" onerror="this.onerror=null;this.src='/static/fallback.jpg';" width="300">
                </details>
                <details style="background: #fff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                    <summary><strong>Robotic Surgery</strong></summary>
                    Robotics like the <a href="https://www.intuitive.com/en-us/products-and-services/da-vinci" target="_blank">da Vinci system</a> enhance surgical precision.<br>
                    <img src="/static/robotic_surgery.jpg" alt="Robotic Surgery" onerror="this.onerror=null;this.src='/static/fallback.jpg';" width="300">
                </details>
                <details style="background: #fff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                    <summary><strong>IoMT</strong></summary>
                    <a href="https://www.ibm.com/blogs/internet-of-things/iomt-healthcare/" target="_blank">Internet of Medical Things</a> improves monitoring and hospital efficiency.<br>
                    <img src="/static/iomt.jpg" alt="IoMT" onerror="this.onerror=null;this.src='/static/fallback.jpg';" width="300">
                </details>
                <details style="background: #fff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                    <summary><strong>Blockchain Health Records</strong></summary>
                    <a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7204202/" target="_blank">Blockchain technology</a> ensures secure and transparent record sharing.<br>
                    <img src="/static/blockchain.jpg" alt="Blockchain in Health" onerror="this.onerror=null;this.src='/static/fallback.jpg';" width="300">
                </details>
                <details style="background: #fff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                    <summary><strong>3D Printing in Medicine</strong></summary>
                    <a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6628311/" target="_blank">3D printing</a> enables customized prosthetics and surgical models.<br>
                    <img src="/static/3d_printing.jpg" alt="3D Printing Medical" onerror="this.onerror=null;this.src='/static/fallback.jpg';" width="300">
                </details>
            </div>
        </section>
    </main>
    <footer>
        <p>&copy; 2025 HealthTech Insights. All rights reserved.</p>
    </footer>
    <script>
        async function submitPrompt() {
            const prompt = document.getElementById('promptInput').value;
            const responseElem = document.getElementById('responseOutput');
            responseElem.textContent = 'Loading...';
            const res = await fetch('/api/ask', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({prompt})
            });
            const data = await res.json();
            responseElem.textContent = data.response;
        }

        async function submitImage() {
            const input = document.getElementById('imageInput');
            const resultElem = document.getElementById('tagResult');
            if (!input.files.length) {
                resultElem.textContent = 'Please select an image to upload.';
                return;
            }
            const formData = new FormData();
            formData.append('image', input.files[0]);
            resultElem.textContent = 'Tagging...';
            const res = await fetch('/api/autotag', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();
            resultElem.textContent = `Predicted Tag: ${data.label}`;
        }
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
