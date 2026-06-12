# How to update your GitHub repository

After downloading this upgraded project, copy these files into your local repository folder.

```bash
git clone https://github.com/mohammadali-mousavireineh/IoT-Paper.git
cd IoT-Paper
```

Copy the new files into this folder, then run:

```bash
git add .
git commit -m "Polish project structure and documentation"
git push origin main
```

Recommended next rename:

```bash
git mv Untitled-1.ipynb notebooks/01_original_experiment.ipynb
git commit -m "Move original notebook into notebooks folder"
git push origin main
```

If the CSV is large, avoid pushing it repeatedly. Use `.gitignore`, Git LFS, or a release asset.
