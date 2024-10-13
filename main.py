import numpy as np
import tensorflow as tf
import pygame as pg
def main():
    test = False
    #Sets up Neural network dimenstions for ex: [2, 3, 2] has 2 inputs a hidden layer with 3 nodes and 2 outputs.

    board = [784, 16, 16, 10]

    #Gets the training data for the number set

    data = getdata(20000)

    #Initializes the values of the biases and weights

    biases = [np.zeros((board[x+1])) for x in range(len(board)-1)]
    weights = [xavier_initialization(board[x], board[x+1]) for x in range(len(board)-1)]


    #Calculates the cost of running the network on the first test case

    for case in range(20000):
        evaluated =  evaluateNetwork(weights, biases, board, data[0][case])
        costlist = getcostlist(data, evaluated, case)
        costsum = np.sum(costlist)
        #Prints the cost
        #print(costlist)
        #print(costsum)

        backpropogate(weights, biases, costlist, evaluated)
        
    #print(costsum)
    if test:
        pg.init()

        screen = pg.display.set_mode((1120, 1120))

        screen.fill((0,0,0))
        mx, my = 0,0
        mousedown = False
        newb = np.zeros(784)
        while True:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    return
                elif event.type == pg.MOUSEBUTTONDOWN:
                    mousedown = True
                    mx, my = event.pos
                elif event.type == pg.MOUSEBUTTONUP:
                    mousedown = False
                elif event.type == pg.MOUSEMOTION and mousedown:
                    mx, my = event.pos
            if mousedown:
                if mx > 0 and mx < 1120 and my > 0 and my < 1120:
                    pg.draw.rect(screen, (255, 255, 255), (np.floor(mx/40)*40, np.floor(my/40)*40, 40, 40))
                    if newb[round(np.floor(mx/40)+(np.floor(my/40)*28))] == 1:
                        continue
                    newb[round(np.floor(mx/40)+(np.floor(my/40)*28))] = 1
                    evaluated =  evaluateNetwork(weights, biases, board, newb)
                    # Create a list of tuples (value, index)
                    indexed_values = [(value, index) for index, value in enumerate(evaluated[len(evaluated)-1])]

                    # Sort the list of tuples based on the values in descending order
                    sorted_indexed_values = sorted(indexed_values, key=lambda x: x[0], reverse=True)

                    # Extract the sorted values and their original indices
                    sorted_values = [value for value, index in sorted_indexed_values]
                    original_indices = [index for value, index in sorted_indexed_values]
                    outstr = ""
                    for b, c in enumerate(original_indices):
                        outstr += str(c) + ": " + str(sorted_values[b]) + "    "
                    print(outstr)
                    
                    pg.display.flip()
                    #return
    else:
        data2 = getdata(22001, 20000)
        avgc = 0
        for j in range(2000):
            evaluated =  evaluateNetwork(weights, biases, board, data2[0][j])
            maxn = 0
            maxi = 0
            for g,h in enumerate(evaluated[len(evaluated)-1]):
                if h > maxn:
                    maxn = h
                    maxi = g
            if maxi == data2[1][j]:
                avgc += 1
        print(str(avgc/2000))
        pg.init()

        screen = pg.display.set_mode((1120, 1120))

        screen.fill((0,0,0))
        mx, my = 0,0
        mousedown = False
        newb = np.zeros(784)
        while True:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    return
                elif event.type == pg.MOUSEBUTTONDOWN:
                    mousedown = True
                    mx, my = event.pos
                elif event.type == pg.MOUSEBUTTONUP:
                    mousedown = False
                elif event.type == pg.MOUSEMOTION and mousedown:
                    mx, my = event.pos
            if mousedown:
                if mx > 0 and mx < 1120 and my > 0 and my < 1120:
                    pg.draw.rect(screen, (255, 255, 255), (np.floor(mx/40)*40, np.floor(my/40)*40, 40, 40))
                    if newb[round(np.floor(mx/40)+(np.floor(my/40)*28))] == 1:
                        continue
                    newb[round(np.floor(mx/40)+(np.floor(my/40)*28))] = 1
                    evaluated =  evaluateNetwork(weights, biases, board, newb)
                    # Create a list of tuples (value, index)
                    indexed_values = [(value, index) for index, value in enumerate(evaluated[len(evaluated)-1])]

                    # Sort the list of tuples based on the values in descending order
                    sorted_indexed_values = sorted(indexed_values, key=lambda x: x[0], reverse=True)

                    # Extract the sorted values and their original indices
                    sorted_values = [value for value, index in sorted_indexed_values]
                    original_indices = [index for value, index in sorted_indexed_values]
                    outstr = ""
                    for b, c in enumerate(original_indices):
                        outstr += str(c) + ": " + str(sorted_values[b]) + "    "
                    print(outstr)
                    
                    pg.display.flip()
                    #return

def backpropogate(weights, biases, costlist, evaluated, learning_rate=0.004):
    # Initialize gradients for weights and biases
    grad_w = [np.zeros(w.shape) for w in weights]
    grad_b = [np.zeros(b.shape) for b in biases]

    # Convert costlist to numpy array for further computations
    costlist = np.array(costlist)

    # Calculate the delta for the output layer
    delta = (evaluated[-1] - costlist) * activation_derivative(evaluated[-1])

    # Gradients for the last layer
    grad_w[-1] = np.outer(delta, evaluated[-2])
    grad_b[-1] = delta

    # Backpropagate through the layers
    for l in range(2, len(evaluated)):
        delta = np.dot(weights[-l+1].T, delta) * activation_derivative(evaluated[-l])
        grad_w[-l] = np.outer(delta, evaluated[-l-1])
        grad_b[-l] = delta

    # Update weights and biases
    for i in range(len(weights)):
        weights[i] -= learning_rate * grad_w[i]
        biases[i] -= learning_rate * grad_b[i]



    return

def getdata(limit=-1, after=0):

    #Returns the data from the mnist dataset in the format: [[Imgdata 1, Imgdata 2, ...], [CorrectNumber 1, CorrectNumber 2, ...]]

    mnist = tf.keras.datasets.mnist.load_data()
    out = []
    out2 = []
    for count, img in enumerate(mnist[0][0]):
        if count == limit:
            break
        if count < after:
            continue
        out.append(np.array([j for sub in img for j in sub]))
        out2.append(mnist[0][1][count])
    return [out, out2]

def getcostlist(data, evaluated, count):
    #Generates a list using the correct value

    correct = np.zeros(10)
    correct[data[1][count]] = 1

    #calculates the cost list and returns it

    return ((evaluated[len(evaluated)-1] - correct)**2)

def evaluateNetwork(weights, biases, board, inp):

    #Defines a list inputs which contains initially just the first input layer

    inputs = [inp]
    for layer in range(len(board)-1):

        #runs the evaluation function on the previous inputs and appends it to the input list

        inputs.append(activation(np.dot(weights[layer], inputs[layer]) + biases[layer]))

    #Returns the inputs list that now contains all evaluated values of every layer in the network

    return inputs

def xavier_initialization(fan_in, fan_out):

    #determines optimal random weight initialization values

    return np.random.randn(fan_out, fan_in) * np.sqrt(1. / fan_in)

def costDeterminer(wanted, got):
    return (got-wanted)**2

def activation(inp):

    #Sigmoid activation function

    return 1 / (1 + np.exp(-inp))

def activation_derivative(inp):
    # Derivative of sigmoid activation function
    return inp * (1 - inp)

if __name__ == "__main__":
    main()