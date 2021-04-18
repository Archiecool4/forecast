from flask import Flask, render_template

from predict import get_model, predict_images

import tensorflow as tf 

import numpy as np

import matplotlib.pyplot as plt

import requests

app = Flask(__name__)

generator = get_model()

with open('token.txt', 'r') as f:
    access_token = f.readline().strip()

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/forecast', methods=['GET', 'POST'])
def forecast():
    return render_template('forecast.html', access_token=access_token)

@app.route('/map/<long>/<lat>/<zoom>')
def get_map_image(long, lat, zoom):
    url = 'https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/' + \
        long + ',' + lat + ',' + zoom + '/500x500?access_token=' + access_token

    r = requests.get(url, allow_redirects=False)
    with open('static/holder.png', 'wb') as f:
        f.write(r.content)

    img = tf.cast(tf.image.decode_jpeg(
        tf.io.read_file('static/holder.png')), tf.float32)

    imgs = predict_images(np.expand_dims(img, 0), generator)

    for i, im in enumerate(imgs):
        plt.subplot(3, 2, i + 1)
        plt.imshow(im)
        plt.axis('off')
    plt.savefig('static/holder.png')

    return render_template('forecast.html', sample='''
    <img src="/static/holder.png" width=100% />
    ''', access_token=access_token)
    

if __name__ == "__main__":
    app.run()
