PY := .venv/bin/python
export PYTHONPATH := src

.PHONY: setup demo eval test clean

# Tạo venv, cài thư viện, tạo index từ data/lesson_plans (cần GEMINI_API_KEY trong .env)
setup:
	python3 -m venv .venv
	$(PY) -m pip install -r requirements.txt
	$(PY) -m t3.cli setup

# Chạy kịch bản mẫu: câu trả lời được, từ chối, hỏi lại, agent ghi note + verify. Mỗi lần chạy ghi một file trong logs/
demo:
	$(PY) -m t3.cli demo

# Chạy testset: dense so với hybrid, quét ngưỡng, ghi eval_out/results.csv
eval:
	$(PY) -m t3.run_eval

# Test offline (embedder giả, LLM giả), không cần key
test:
	$(PY) -m pytest -q

clean:
	rm -rf .venv data/index .pytest_cache
