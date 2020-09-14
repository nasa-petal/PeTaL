#https://stackoverflow.com/a/57886655
FROM python:3.7-slim-buster as base

ENV PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONHASHSEED=random \
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100 \
  POETRY_VERSION=1.0.5

# System deps:
RUN pip install "poetry==$POETRY_VERSION"

RUN mkdir /code

# Copy only requirements to cache them in docker layer
WORKDIR /code
#https://stackoverflow.com/a/46801962
COPY pyproject.toml poetry.loc[k] /code/

# Project initialization:
RUN poetry config virtualenvs.create false \
  && poetry install --no-interaction --no-ansi

RUN pip install torch==1.5.1+cpu torchvision==0.6.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Creating folders, and files for a project:
COPY . /code



#RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python
#RUN $HOME/.poetry/bin/poetry install
#RUN $HOME/.poetry/bin/poetry run pip install torch==1.5.1+cpu torchvision==0.6.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
#RUN pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
#COPY . /code/