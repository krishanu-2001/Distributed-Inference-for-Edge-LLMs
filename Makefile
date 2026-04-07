.PHONY: run run-sglang run-sglang-no-launch test kill webapp

run:
	python -m src.main --config config/default.yaml

run-sglang:
	PYTHONPATH=sglang/python:$$PYTHONPATH python -m src.sglang_main --config config/sglang.yaml

run-sglang-no-launch:
	PYTHONPATH=sglang/python:$$PYTHONPATH python -m src.sglang_main --config config/sglang.yaml --no-launch

test:
	python -m tests.test_prefix_trie

kill:
	@for port in 8100 8101 8102 8103; do \
		pid=$$(lsof -ti :$$port 2>/dev/null); \
		if [ -n "$$pid" ]; then \
			kill -9 $$pid && echo "Killed pid $$pid on port $$port"; \
		fi; \
	done

webapp:
	uvicorn webapp.app:app --port 9000 --reload
