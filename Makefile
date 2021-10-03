.PHONY: install train eval predict api test clean

install:
	pip install -r requirements.txt

train:
	python -m src.train --config configs/voc.yaml

eval:
	python -m src.eval --config configs/voc.yaml

predict:
	python -m src.predict --image samples/test.jpg --weights checkpoints/best.pt

api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest -q

clean:
	rm -rf __pycache__ .pytest_cache mlruns/ runs/ outputs/
