# Template for final project

# Service of extracting pdf files and fine-tuning open source LLM's on them

Structure:
- main.py - entry-point, contains all logic for api and pipelines of project
- data_preparator - contains all classes and methods for data ETL pipeline
- model - contains all models related classes and methods
- utils - useful methods
- config.py - contains constants and configurations needed to maintainability
- Dockerfile - allows to run project in docker container with all requirements, needed for project