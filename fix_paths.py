import os

with open("api/main.py", "r") as f:
    content = f.read()

content = content.replace("os.makedirs('reports', exist_ok=True)", "os.makedirs('/tmp/reports', exist_ok=True)")
content = content.replace("'reports/{filename}'", "'/tmp/reports/{filename}'")
content = content.replace("f'reports/{filename}'", "f'/tmp/reports/{filename}'")
content = content.replace("os.listdir('reports')", "os.listdir('/tmp/reports')")
content = content.replace("f'reports/{base_name}", "f'/tmp/reports/{base_name}")

with open("api/main.py", "w") as f:
    f.write(content)
print("api/main.py paths updated")

with open("utils/logger.py", "r") as f:
    content = f.read()

content = content.replace("os.makedirs('logs', exist_ok=True)", "os.makedirs('/tmp/logs', exist_ok=True)")
content = content.replace("f'logs/session_{self.session_id}.json'", "f'/tmp/logs/session_{self.session_id}.json'")

with open("utils/logger.py", "w") as f:
    f.write(content)
print("utils/logger.py paths updated")
