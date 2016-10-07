DOCS=docs

.PHONY: docs tests

docs:
	cd $(DOCS) && $(MAKE) clean && $(MAKE) html

docs-online: docs
	ghp-import -np $(DOCS)/_build/html -r origin

tests:
	py.test tests

clean:
	rm -rf dist

sdist:
	python3 setup.py sdist

release: docs-online
	python3 setup.py sdist upload
