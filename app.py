from flask import Flask, render_template, url_for
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

app = Flask(__name__, static_folder='static', template_folder='templates')

# Load innovation data
innovation_data = pd.read_csv('data/innovation_data.csv')

# Create a linear regression model to predict innovation score
X = innovation_data[['rd_spend', 'product_launch_success']]  # use rd_spend and product_launch_success as features
y = innovation_data['product_launch_success']  # use product_launch_success as the target variable

model = LinearRegression()
model.fit(X, y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/visualize')
def visualize():
    # Create the bar chart
    plt.switch_backend('Agg')  # Switch to Agg backend
    fig = plt.figure(figsize=(10, 6))
    sns.barplot(x='company_name', y='rd_spend', data=innovation_data)
    plt.title('Total R&D Spend by Company')
    plt.xlabel('Company Name')
    plt.ylabel('R&D Spend ($)')
    plt.savefig('static/bar_chart.png')
    plt.close(fig)  # Close the figure

    # Create the line plot
    plt.switch_backend('Agg')  # Switch to Agg backend
    fig = plt.figure(figsize=(10, 6))
    sns.lineplot(x='market_share', y='product_launch_success', data=innovation_data)
    plt.title('Relationship Between Market Share and Product Launch Success')
    plt.xlabel('Market Share (%)')
    plt.ylabel('Product Launch Success (%)')
    plt.savefig('static/line_plot.png')
    plt.close(fig)  # Close the figure

    bar_chart_url = url_for('static', filename='bar_chart.png')
    scatter_plot_url = url_for('static', filename='line_plot.png')

    return render_template('visualize.html', 
                           bar_chart_url=bar_chart_url, 
                           scatter_plot_url=scatter_plot_url)


@app.route('/predict')
def predict():
    # Predict the best company for innovation
    predicted_innovation_scores = model.predict(X)
    best_company_index = predicted_innovation_scores.argmax()
    best_company_name = innovation_data.iloc[best_company_index]['company_name']
    return render_template('predict.html', prediction=best_company_name, market_share=innovation_data.iloc[best_company_index]['market_share'])

if __name__ == '__main__':
    app.run(debug=True)