# AI Disclosure

## Tools used

Claude (Anthropic).

## How I used it

**1. Template code for all four parts**

I used Claude to write a starting version of the code: `eta_app.py` and its tests (Part A), `eta_mlflow.py`
and `test_gate.py` (Part B), `eta_ray.py` and `verify_ray.py` (Part C), and `smoke_eta.py` (Part D). The code
follows the Lecture 10 notebooks in the course repo (REST, MLflow and Ray Serve exercises) and
`Monitoring/prom_client.py`.

**2. Formatting the Markdown files and the report**

I used it to format and tidy up `README.md` and this file so they're readable, and to draft the written
answers in the report from the results of my own runs

## Conclusion

Claude gave me a working starting point and saved a lot of time on environment and VM issues. I ran
everything myself on my VM: the tests, the load tests with one and two workers, the MLflow registry, gate
and shadow test, the Ray canary checks and the smoke test. All numbers in the report come from my own runs,
and I checked that the output files match the report.
