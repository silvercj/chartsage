.PHONY: dev test test-e2e smoke clean

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

clean:
	rm -rf src/api/__pycache__ src/api/logs/*.log
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
