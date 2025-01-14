# Neural Network Training Project

This repository contains a basic implementation of a single-layer neural network (perceptron) using **Python** and **NumPy**. The project showcases how to train a neural network using the **backpropagation method**.

## 🧩 **Project Structure**
```
📁 AI2
├── src
│   └── AI2.py             # Main code file with the neural network implementation
├── README.md              # Documentation file (this file)
├── requirements.txt       # Dependencies
└── .gitignore             # Git ignore file
```

## 🚀 **How to Run the Project**
1. Clone this repository:
   ```bash
   git clone https://github.com/QUA77I20/AI2.git
   ```

2. Navigate to the project folder:
   ```bash
   cd AI2
   ```

3. Run the Python script:
   ```bash
   python src/AI2.py
   ```

## ⚙️ **Functions and Methods**
### `sigmoid(x)`
The sigmoid function is used to map any real value to the range (0, 1).

### Training Process
- The network is trained using a simple dataset of binary inputs and outputs.
- The **backpropagation method** is used to adjust the synaptic weights based on the error between the expected and actual outputs.

## 📈 **Training Example**
Initial random weights:
```
[ 0.5, -0.3, 0.8 ]
```

After training:
```
[ 1.2, -0.6, 0.9 ]
```

## 🧪 **Testing**
The network is tested with new inputs to predict the output.

Example test input:
```
New input: [1, 1, 0]
Predicted output: 0.89
```

## 📂 **Future Improvements**
- Implement multi-layer perceptron (MLP).
- Add error visualization (e.g., matplotlib graphs).
- Optimize the backpropagation algorithm.

## 📄 **License**
This project is open-source and available under the MIT License.

