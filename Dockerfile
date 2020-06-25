FROM python:3
ENV PYTHONUNBUFFERED 1
RUN mkdir /code
WORKDIR /code
COPY requirements.txt /code/
#RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python
#RUN $HOME/.poetry/bin/poetry install
#RUN $HOME/.poetry/bin/poetry run pip install torch==1.5.1+cpu torchvision==0.6.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
COPY . /code/