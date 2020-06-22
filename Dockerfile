FROM python:3

WORKDIR $HOME/Documents/localwrk/git/petal

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN ./run config/default.json

CMD [ "python", "manage.py runserver" ]



FROM python:3
ENV PYTHONUNBUFFERED 1
RUN mkdir /code
WORKDIR /code
COPY requirements.txt /code/
RUN pip install -r requirements.txt
COPY . /code/