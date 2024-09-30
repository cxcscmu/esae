from source import workspace
from pathlib import Path

esCert = Path(workspace, "elasticsearch-8.15.1/config/certs/http_ca.crt")
esUser, esAuth = "elastic", "1Wh07+RG8x+a2zNKC*D9"

workspace = Path(workspace, "interpret")
workspace.mkdir(mode=0o770, parents=True, exist_ok=True)
