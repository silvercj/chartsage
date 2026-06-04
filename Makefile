.PHONY: dev test test-e2e test-pdf smoke clean qa

# Backend dev server (run from project root)
dev:
	cd src/api && uvicorn main:app --reload --port 8000

# Fast: unit + integration (no real Claude calls)
test:
	pytest tests/unit tests/integration -v

# Slow: opt-in real Claude smoke tests
test-e2e:
	RUN_E2E=true pytest tests/e2e -v -m e2e

# Boot the API + smoke a single fixture
smoke:
	@cd src/api && (uvicorn main:app --port 8001 &) && sleep 2 && \
		curl -f -X POST http://localhost:8001/generate-report \
			-F "file=@../../tests/e2e/fixtures/sales.csv" \
			| tee /tmp/chartsage_smoke.json \
		&& echo "OK" \
		&& pkill -f "uvicorn main:app --port 8001"

test-pdf:
	RUN_PDF_TESTS=true pytest tests/unit/test_pdf_export.py -v

clean:
	rm -rf src/api/__pycache__ src/api/logs/*.log
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# QA / Eval Harness — run the corpus through the real pipeline + validators + judge.
# Flags pass through, e.g.: make qa ARGS="--only synthetic --no-judge --limit 3"
qa:
	venv/bin/python qa/run_eval.py $(ARGS)
