from pathlib import Path
import glob

output_folder = "artifacts/InfoVQA/mineru/dev"
hybrid_auto = 0

def main():
    folders = [folder for folder in Path(output_folder).iterdir()]
    for folder in folders:
        sub_folder = [f for f in folder.iterdir()]
        mode = sub_folder[0].name
        if mode.lower() == "hybrid_auto":
            global hybrid_auto
            hybrid_auto += 1
    print(hybrid_auto)
    return

if __name__ == "__main__":
    main()