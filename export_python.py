import os

ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(ROOT, "projet_python.txt")

EXCLUDE_DIRS = {".git", "venv", "__pycache__"}

with open(OUTPUT, "w", encoding="utf-8") as out:
    for dirpath, dirnames, filenames in os.walk(ROOT):
        # enlever les dossiers à exclure
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]

        for filename in filenames:
            if not filename.endswith((".py", ".ipynb")):
                continue

            full_path = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(full_path, ROOT)

            out.write(f"### FILE: {rel_path}\n")
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    out.write(f.read())
            except UnicodeDecodeError:
                out.write("[UNICODE ERROR: impossible de lire ce fichier]\n")

            out.write("\n\n")

print(f"Export terminé dans {OUTPUT}")
