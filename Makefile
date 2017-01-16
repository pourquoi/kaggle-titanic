.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')

#################################################################################
# COMMANDS                                                                      #
#################################################################################

requirements:
	/Users/pourquoi/Library/DataScienceStudio/dss_home/pip install -q -r requirements.txt

data: requirements
	python src/data/make_dataset.py

clean:
	find . -name "*.pyc" -exec rm {} \;

lint:
	python -m flake8 --exclude=lib/,bin/ .

sync_data_to_s3:
	python -m aws s3 sync data/ s3://$(BUCKET)/data/

sync_data_from_s3:
	python -m aws s3 sync s3://$(BUCKET)/data/ data/

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################
