
from petal import app
# from petal import args

# DEBUG = args['debug']

if __name__ == '__main__':
    # app.run(host='0.0.0.0', debug=DEBUG)
    app.run(host='0.0.0.0', debug=True)

    app.config.update(
        TEMPLATES_AUTO_RELOAD = True
    )












