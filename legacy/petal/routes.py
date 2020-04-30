from flask import Flask, session, url_for, escape, redirect, Response, jsonify, json, request, render_template, send_from_directory
from petal import app
from petal import graph
from petal import bird
from petal import qBIRD 
#from petal import bird_to_cluster (file is gone)
# from petal import bird_to_biomole
from petal import ONT_PATH, DATA_PATH, LIT_PATH, ont, DEBUG, resnet, geonet, cluster
from werkzeug.utils import secure_filename
import base64
import io
import random, json
app.secret_key = "any random string"
app.config["APPLICATION_ROOT"] = "/petal"


@app.route('/profile/')
@app.route('/profile')
@app.route('/')
def home():
    title = 'Home'
    return render_template('home.html', title=title)


@app.route('/profile/<ont_class>', methods = ['GET', 'POST'])
def profile(ont_class):
    profile = graph.create_profile(ont, ont_class, debug=DEBUG, 
    data_path=DATA_PATH, lit_path=LIT_PATH)
    return render_template('profile.html', profile=profile)


@app.route('/map')
def map():
    return render_template('map.html')

@app.route('/BioMole')
def lander():
    return render_template('BioMole-wrapper.html')

@app.route('/BioMole-inframe')
def index2():
    return render_template('BioMole.html')

@app.route('/data/<path:path>')
def send_data(path):
    return send_from_directory('data', path)


@app.route('/vision', methods=['GET', 'POST'])
def classify():
    if request.method == 'POST':
        top = 5
        message = request.get_json(force=True)
        encoded_img = message['img'].split(',')[-1]
        decoded_img = base64.b64decode(encoded_img)
        img = io.BytesIO(decoded_img)
        response = {'ontology':resnet.classify(img, top=top)}
        response['pattern'] = geonet.classify(img)
        # Add links to ontology
        for c in range(top):
            response['ontology'][c]['class'] = \
            graph.to_ont(ont, response['ontology'][c]['label'])
        return jsonify(response)
    return render_template('vision.html')


@app.route('/graph-2d', methods=['GET', 'POST'])
def d3_graph():
    graph.build_graph_json(ont)
    return render_template('graph-2d.html')


@app.route('/graph-3d', methods=['GET', 'POST'])
def real3d_graph():
    return render_template('graph-3d.html')


@app.route('/test', methods=['GET', 'POST'])
def test():
    return render_template('test.html')


@app.route('/scatter', methods=['GET', 'POST'])
def scatter():
    return render_template('scatter.html')


@app.route('/tsne', methods=['GET', 'POST'])
def tsne():
    return render_template('tsne.html')


@app.route('/cluster', methods=['GET', 'POST'])
def d3_cluster():
    if request.method == 'POST':
        response = request.get_json(force=True)
        df0 = response['df0']
        cluster_count = int(response["numClusters"])
        selected_clusters = response["selectedClusters"]

        if df0 is not "":
            if not selected_clusters:
                data, df0, lda_out, mapping, titles = cluster.recluster(df0, cluster_count)
            else:
                serialized_data = response["preparedData"]
                lda_out = response["lda_out"]
                data, df0, lda_out, mapping, titles = cluster.subset(df0, serialized_data, lda_out, cluster_count, selected_clusters)
        else:
            file_text = response['file']
            data, df0, lda_out, mapping, titles = cluster.create_cluster(file_text, cluster_count)

        return jsonify([df0, data, lda_out, mapping, titles])
    return render_template('cluster.html')


@app.route('/bird_e2b', methods=['GET', 'POST'])
def bird_e2b():
    if request.method == 'POST':
        tertiary = request.get_json(force=True)
        file_name = 'petal/data/bird/' + tertiary + '.txt'
        papers = bird.get_papers(file_name)
        
        return render_template('abstract.html', papers=papers)
    return render_template('bird_e2b.html')

@app.route('/bird_searchable', methods=['GET', 'POST'])
def bird_searchable():
    if request.method == 'POST':
        tertiary = request.get_json(force=True)
        file_name = 'petal/data/bird/' + tertiary + '.txt'
        papers = bird.get_papers(file_name)
        return render_template('abstract2.html', papers=papers)
    return render_template('bird_searchable.html')

@app.route('/qBIRD', methods=['GET', 'POST'])
def bird_with_qbird():
    if request.method == 'POST':
        nlp = qBIRD.load_nlp()
        terms = qBIRD.load_dict("petal/static/js/NTRS_data.js")

        question = request.get_json(force=True)
        exact_matches, exact_synonym_matches, partial_matches = qBIRD.get_eng_terms(question, terms, nlp)
        all_matches = exact_matches | exact_synonym_matches | partial_matches
        print ("all_matches",len(all_matches))
        print ("exact_matches", len(exact_matches))
        print ("partial_matches", len(partial_matches))
        print ("exact_synonym_matches", len(exact_synonym_matches))
        all_papers =  []
        for a_match in exact_matches:
            file_name = 'petal/data/bird/' + a_match + '.txt'
            papers = bird.get_papers(file_name)
            if papers: 
                all_papers.extend(papers)
    

        return render_template('abstract2.html', papers=all_papers)
    return render_template('qBIRD.html')



@app.route('/bird_nav', methods=['GET', 'POST'])
def bird_nav():
    if request.method == 'POST':
        tertiary = request.get_json(force=True)
        papers = bird.get_papers(tertiary)
        return render_template('abstract.html', papers=papers)
    return render_template('bird_nav.html')

@app.route('/bird2cluster', methods=['GET', 'POST'])
def bird2cluster_tool():
    tertiary = request.get_json(force=True)
    file_name = 'petal/data/bird/' + tertiary + '.txt'   # Edit 9/6/2019 by Kei
    papers = bird.get_papers(file_name)
    converted_papers = bird_to_cluster.convert(papers) #no bird to cluster file
    return converted_papers


@app.route('/bird2biomole', methods=['GET', 'POST'])
def bird2biomole_tool():
    tertiary = request.get_json(force=True)
    papers = bird.get_papers(tertiary)
    converted_papers = bird_to_biomole.convert(papers)
    return converted_papers


@app.route('/getNodeData/<ont_class>', methods=['GET', 'POST'])
def getNodeData(ont_class):
    profile = graph.create_profile(ont, ont_class, debug=DEBUG, 
    data_path=DATA_PATH, lit_path=LIT_PATH)
    return jsonify(profile)


# For Drag and Drop
app.config["UPLOAD_FOLDER"] = "petal/static/uploads"


@app.route("/sendfile", methods=["POST"])
def send_file():
    print("send file function called")
    fileob = request.files["file2upload"]
    filename = secure_filename(fileob.filename)
    save_path = "{}/{}".format(app.config["UPLOAD_FOLDER"], filename)
    fileob.save(save_path)
    return "successful_upload"
