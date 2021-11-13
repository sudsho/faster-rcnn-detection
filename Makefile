.PHONY: install dev train predict api test lint docker clean

install:
	pip install -r requirements.txt

dev: install
	pip install pytest flake8

train:
	python -m src.train --config configs/voc.yaml

predict:
	python -m src.predict --image samples/test.jpg --weights checkpoints/voc/best.pt --save out.jpg

api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest -q

lint:
	flake8 src tests --max-line-length=100 --ignore=E501,W503,E203

docker:
	docker build -t faster-rcnn-detection:dev .

clean:
	rm -rf __pycache__ .pytest_cache mlruns/ runs/ outputs/
