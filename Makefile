.PHONY: zotero-index zotero-query web-backend web-frontend web-test-backend web-test-frontend

zotero-index:
	python scripts/zotero_index.py --refresh

zotero-query:
	python scripts/zotero_index.py --query "$(q)"

web-backend:
	uvicorn web.backend.app:app --reload --host 127.0.0.1 --port 8000

web-frontend:
	npm --prefix web/frontend run dev

web-test-backend:
	pytest -q tests/test_web_backend_api.py

web-test-frontend:
	npm --prefix web/frontend test
