with open("api/main.py", "r") as f:
    content = f.read()

old = "from fastapi import FastAPI, HTTPException\n"
new = "from fastapi import FastAPI, HTTPException\nfrom fastapi.responses import HTMLResponse\nimport os as _os\n"

content = content.replace(old, new)

old = "@app.get('/')\ndef root():\n    return {'status': 'running', 'message': 'Multi-Agent Research Pipeline API'}\n"
new = "@app.get('/', response_class=HTMLResponse)\ndef root():\n    ui_path = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), 'ui.html')\n    with open(ui_path, 'r') as f:\n        return f.read()\n"

content = content.replace(old, new)

with open("api/main.py", "w") as f:
    f.write(content)
print("api/main.py updated to serve ui.html")
