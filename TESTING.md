# Testing

The local test suite covers the contracts around the scientific pipeline and
web application without requiring the large dataset or trained model.

Run it from the project root:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\venv\Scripts\python.exe -m pytest -q
```

The tests cover:

- Legacy VTK binary read/write round trips
- Follicle maturity boundaries and volume measurements
- Detection ROI and minimum-size filtering
- Flask page, upload validation, analysis response, and slice retrieval

GitHub Actions runs the same suite on every push and pull request.