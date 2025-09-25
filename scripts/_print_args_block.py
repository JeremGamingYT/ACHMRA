from pathlib import Path
text = Path("polymer/training/achmra/pipeline.py").read_text()
idx = text.find("args_kwargs")
if idx == -1:
    print("not found")
else:
    print(text[idx:idx+400])
