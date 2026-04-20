with open("ui.html", "r") as f:
    content = f.read()

old = "  const API = 'http://localhost:8000';"
new = "  const API = window.location.hostname === 'localhost' ? 'http://localhost:8000' : 'https://' + window.location.hostname.replace('--', '-').replace('.hf.space', '') + '-' + window.location.hostname.split('.')[1] + '.hf.space';"

content = content.replace(old, new)

with open("ui.html", "w") as f:
    f.write(content)
print("ui.html API URL updated")
