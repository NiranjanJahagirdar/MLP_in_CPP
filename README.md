# Multilayer Perceptron using C++

Assignment 2 for the course, Foundations of Data Science, taught by Dr. Navneet Goyal in Second Semester 2020-21 at BITS Pilani.

Team Members

-Pranay Khariwal    			2017B3A70565P
-Niranjan Jahagirdar			2017B3A70454P 

-----------------------------------------------------------------------------------


To Run The Code

    • g++ mlp.cpp
    • ./a.out

-----------------------------------------------------------------------------------

Press 1: To run preloaded model with defined parameters
Press 0: To give your own input parameters (Learning Rate,Activation Function, No of hidden layers and dimensions of each hidden Layer)

User has to enter
    • Activation Function
    • Learning Rate
    • No of Hidden Layers
    • Dimension of Each Hidden Layer
    • No of Training Epochs

Input and Output dimension is fixed as we are using Haberman dataset.

NOTE:

Input and output dimensions can be changed as per users need based on whatever data they may chose to run this model on.

-----------------------------------------------------------------------------------

Activation Functions:

    • Sigmoid 
    • Tanh
    • Relu (Rectified Linear Units) 
    • LeakyRelu
    • ELU (Exponential Linear Units)

and their respective derivatives

-----------------------------------------------------------------------------------


Summary:


    • Its a fully connected MLP.
    • It incorporates the following features.
        ◦ variable no of hidden layers with different number of hidden neurons
        ◦ various activation function as per users need
        ◦ variable input and output dimensions
        ◦ Error printing after an interval of 1000 iterations
        
  
-----------------------------------------------------------------------------------


Functions

•	FeedForward
•	Backpropagation
•	Error calculations (MAE)
•	Initialisation of layer and neuron Data Structure 
•	Activation Functions & their derivatives.
•	Other Supporting and printing Functions.

