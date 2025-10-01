# Lab 01 — Version Control Activity

Maps to: **DAT5501 Version Control Activity** (Git + VS Code).

## What to do
- Create a basic Python file (e.g., `src/hello.py`) and print something.
- Commit, branch, make a change, merge.

## Commands (quick path)
```bash
git init
git add .
git commit -m "lab01: init"
git switch -c feature/update-hello
echo 'print("Hello, Amelia!")' > src/hello.py
git add src/hello.py && git commit -m "feat: hello"
git switch -
git merge feature/update-hello
```

## Tests
- Optional: add a tiny test to ensure a function returns expected text.