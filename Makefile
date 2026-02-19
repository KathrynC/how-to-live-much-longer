.PHONY: zotero-index zotero-query

zotero-index:
	python scripts/zotero_index.py --refresh

zotero-query:
	python scripts/zotero_index.py --query "$(q)"
