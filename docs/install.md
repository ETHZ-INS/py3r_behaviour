# Install py3r-behaviour

## Pre-install requirements
A virtual environment with:

- Python >= 3.12 
- pip >= 21.3 
- git

## Windows
copy and run this command in Command Prompt or Powershell

```## 
powershell -Command "$repo='ETH-INS/py3r_behaviour'; \
$latest = Invoke-RestMethod -Uri \"https://api.github.com/repos/$repo/releases/latest\"; \
$tag = $latest.tag_name; \
pip install --upgrade git+https://github.com/$repo.git@$tag"


```

## Linux/Mac OS
copy and run this command in Terminal
```bash
repo="ETH-INS/py3r_behaviour"
latest_tag=$(curl -s https://api.github.com/repos/$repo/releases/latest | grep -Po '"tag_name": "\K.*?(?=")')
pip install --upgrade git+https://github.com/$repo.git@$latest_tag


```