# Install py3r-behaviour

## Pre-install requirements
A virtual environment with:

- Python >= 3.12 
- pip >= 21.3 
- git

## Windows
copy and run this command in Powershell

```## 
$repo='ETHZ-INS/py3r_behaviour'; $latest = Invoke-RestMethod -Uri "https://api.github.com/repos/$repo/releases/latest"; $tag = $latest.tag_name; pip install --upgrade git+https://github.com/$repo.git@$tag


```

## Linux/Mac OS
copy and run this command in Terminal
```bash
repo="ETHZ-INS/py3r_behaviour"; latest_tag=$(curl -sL -o /dev/null -w '%{url_effective}' "https://github.com/$repo/releases/latest" | sed 's#.*/tag/##' | tr -d '\r'); [ -n "$latest_tag" ] || { echo "No GitHub release found"; exit 1; }; pip install --upgrade "git+https://github.com/$repo.git@$latest_tag"


```