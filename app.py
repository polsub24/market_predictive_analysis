from flask import Flask, render_template, url_for
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__, static_folder='static')

# Load innovation data
innovation_data = pd.read_csv('data/innovation_data.csv')

@app.route('/')
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

    # Create the scatter plot
    plt.switch_backend('Agg')  # Switch to Agg backend
    fig = plt.figure(figsize=(10, 6))
    sns.scatterplot(x='market_share', y='product_launch_success', data=innovation_data)
    plt.title('Relationship Between Market Share and Product Launch Success')
    plt.xlabel('Market Share (%)')
    plt.ylabel('Product Launch Success (%)')
    plt.savefig('static/scatter_plot.png')
    plt.close(fig)  # Close the figure

    return render_template('visualize.html', 
                           bar_chart=url_for('static', filename='bar_chart.png'), 
                           scatter_plot=url_for('static', filename='scatter_plot.png'))


if __name__ == '__main__':
    app.run(debug=True)