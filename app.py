# import the necessary packages
from keras.models import load_model
from PIL import Image
import numpy as np
import flask
import io

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None

def load_model_from_file():
    # load the  Keras model
    global model
    model = load_model('models/famosos_cnn.h5')

def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = np.asarray(image)

    # return the processed image
    return image

@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}
    load_model_from_file()

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            class_names = ['Adam Sandler', 'Adele', 'Ashton Kutcher', 'Bella Thorne']
            images_predict = []
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # preprocess the image and prepare it for classification
            image = prepare_image(image, target=(100, 100))
            images_predict.append(image)
            images_predict = np.asarray(images_predict)
            

            # classify the input image and then initialize the list
            # of predictions to return to the client
            preds = model.predict(images_predict)
            print(preds)
            #results = imagenet_utils.decode_predictions(preds)
            data["predictions"] = []

            # loop over the results and add them to the list of
            # returned predictions
            for i in range(len(preds[0])):
                r = {"type": class_names[i], "probability": float(preds[0][i])}
                data["predictions"].append(r)

            # indicate that the request was a success
            data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    load_model()
    app.run()