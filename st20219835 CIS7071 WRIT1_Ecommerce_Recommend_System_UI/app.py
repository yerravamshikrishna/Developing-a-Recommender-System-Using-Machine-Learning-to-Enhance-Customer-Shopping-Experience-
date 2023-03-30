from flask import Flask, render_template, request
import pandas as pd
import pickle

from model_templates import ContentBasedRecommendorSystem, NearestNeighboursContentBasedRecommendorSystem
app = Flask(__name__)

# Load the saved model
model_engine = pd.read_pickle("nn_engine.pkl")

# Load the product data from CSV
products_df = pd.read_pickle("df_copy.pkl")



# Define the route for the homepage
@app.route('/')
def index():
    # Get a sample of 20 products
    sample_products = products_df.sample(100)[['Title','Specifications']].dropna().sample(30)
    # Render the index template and pass in the sample products
    return render_template('index.html', products=sample_products)

# Define the route for product recommendations
@app.route('/<product_id>', methods=["GET"])
def product_recommendations(product_id):
    print(product_id)
    products = get_recommendations(product_id)
    return render_template("./recommend.html", products=products )

# Define a function to get product recommendations for a given ID
def get_recommendations(product_id):
    # Use the trained model to generate recommendations for the given ID
    scores, recommended_ids = model_engine.predict(product_id)
    # Get the product details for the recommended IDs from the products dataframe
    recommended_products = products_df.loc[recommended_ids]
    return recommended_products[['Title','Specifications']]


# main driver function
if __name__ == '__main__':
    # on the local development server.
    app.run(debug=True)
