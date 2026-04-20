with open("README.md", "r") as f:
    existing = f.read()

header = """---
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
    f.write(header + existing)
print("README.md updated with HuggingFace header")
