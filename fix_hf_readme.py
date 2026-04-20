with open("README.md", "r") as f:
    content = f.read()

old_header_end = content.find("# Multi Agent Research Pipeline")
clean_content = content[old_header_end:]

new_header = """---
title: Multi Agent Research Pipeline
emoji: 🔬
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
---

"""

with open("README.md", "w") as f:
    f.write(new_header + clean_content)
print("README.md fixed")
