# DAT5501 – Lab 3: Continuous Integration (CI)

## Lab work completed in a separate repository available on GitHub

## Overview  
This lab focused on **Continuous Integration (CI)** using GitHub Actions.  
We automated the testing workflow so that unit tests run automatically on each push or pull request.

## Key Steps  
- Created a `.github/workflows/ci.yml` pipeline file.  
- Configured workflow to set up Python, install dependencies, and run pytest.  
- Pushed changes to GitHub and verified that the CI job triggered.  
- Broke the function intentionally to see the workflow fail.  
- Fixed the function and confirmed the workflow passed again.

*CircleCI does runs within this portfolio. 