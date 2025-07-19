# Edunet-GreenSkills

A machine learning project for smart irrigation using sensor data to control pumps.

## Project Structure

- `greenskill.ipynb`: Main Jupyter notebook for data analysis and ML modeling.
- `dataset.csv`: Sensor and pump data for training/testing.
- `README.md`: Project documentation.

## Getting Started

1. **Install dependencies**  
   Run the following in a notebook cell:


2. **Load the dataset**  
The notebook loads `dataset.csv` and performs basic EDA.

3. **Preprocessing**  
- Removes unnecessary columns.
- Defines features (`sensor_0` to `sensor_19`) and labels (pump columns).

4. **Modeling**  
- Uses RandomForestClassifier with MultiOutputClassifier for multi-label pump prediction.

## Usage

Open `greenskill.ipynb` in Jupyter or VS Code and run the cells sequentially.

## Next Steps

- Add feature scaling and hyperparameter tuning.
- Implement model saving/loading with `joblib`.
- Add visualizations for feature importance and prediction results.

