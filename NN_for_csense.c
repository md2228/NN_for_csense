//#include <stdlib.h>
//#include <stdio.h>
//#include <math.h>

// Simple NN that can learn xor (modified for compatibility with csense)
// Credit for the original: https://www.youtube.com/watch?v=LA4I3cWkp1E 

//unsigned long int next = 1;
//unsigned long int RAND_MAX = 32767;

int my_rand(unsigned int next) // RAND_MAX assumed to be 32767
{
    next = next * 1103515245 + 12345;
    //printf("Next = %ld ", next);
    return (unsigned int)(next/65536) % 32768;
}

double my_exp(double exp) 
{
    double base_e = 2.7182818;
	double result = 1.0;

    if((int)exp > 0)
	    for( ;(int)exp > 0; exp--) result *= base_e;
    else
        for( ;(int)exp < 0; exp++) result *= base_e;

    //printf("Result = %f , Base_e = %f ", result, base_e);
	
    return result;
}

double sigmoid(double x) { return (double)(1.0 / (1.0 + my_exp(-x))); }
double dSigmoid(double x) { return (double)(x * (1.0 - x)); }

double init_weights(unsigned int next, unsigned long int RAND_MAX) { return (((double)my_rand(next)) / (double)RAND_MAX); }

// void shuffle(int *array, size_t n) { // Original uses size_t for n, i and j; we'll use unsigned long long
void shuffle(int *array, unsigned long long n, unsigned int next, unsigned long int RAND_MAX) { // Unused: Can't use an array's address
    if (n > 1) {
        unsigned long long i;
        int *p;
        for (i = 0; i < (n - 1); i++) {
            next = my_rand(next);
            unsigned long long j = i + my_rand(next) / (RAND_MAX / (n - i) + 1);
            // The below three lines are from the original
            //int t = array[j];
            //array[j] = array[i];
            //array[i] = t;
            // This modification fixes "Indexing an access object is not supported in ADA!"
            int t = *(array + j);
            p = array + j;
            *p = *(array + i);
            p = array + i;
            *p = t;
        }
    }
    //return n;
}

int main(int argc, char **argv) {

    // These four were #define constants in the original
    const short numInputs = 2;
    const short numHiddenNodes = 2;
    const short numOutputs = 1;
    const short numTrainingSets = 4;

    // These five indexes are declared whithin each loop in the original
    int i = 0;
    int j = 0;
    int epoch = 0;
    int x = 0;
    int k = 0;
    
    // These two were globals in the original
    unsigned int next = 1;
    unsigned long int RAND_MAX = 32767;

    const double lr = 0.1f;

    double hiddenLayer[2]; // numHiddenNodes
    double outputLayer[1]; // numOutputs
    
    double hiddenLayerBias[2]; // numHiddenNodes
    double outputLayerBias[1]; // numOutputs

    double hiddenWeights[2][2]; // numInputs, numHiddenNodes
    double outputWeights[2][1]; // numHiddenNodes, numOutputs

    double training_inputs[4][2] = {{0.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 1.0f}}; // numTrainingSets, numInputs
    double training_outputs[4][1] = {{0.0f}, {1.0f}, {1.0f}, {0.0f}}; // numTrainingSets, numOutputs

    for (i = 0; i < numInputs; i++){
        for (j = 0; j < numHiddenNodes; j++){
            next = my_rand(next);
            hiddenWeights[i][j] = init_weights(next, RAND_MAX);
        }
    }

    
    for (i = 0; i < numInputs; i++){
        for (j = 0; j < numOutputs; j++){
            next = my_rand(next);
            outputWeights[i][j] = init_weights(next, RAND_MAX);
        }
    }

    
    for (i = 0; i < numOutputs; i++){
        next = my_rand(next);
        outputLayerBias[i] = init_weights(next, RAND_MAX);
    }
    
    //int trainingSetOrder[] = {0, 1 ,2 ,3}; // Original
    int trainingSetOrder[4]; // Attempt to fix "Unexpected reach to static local "V023_trainingSetOrder" with no corresponding global!"
    trainingSetOrder[0] = 0;
    trainingSetOrder[1] = 1;
    trainingSetOrder[2] = 2;
    trainingSetOrder[3] = 3;


    int numberOfEpochs = 10000;

    // Train the Neural Network for a number of epochs
    for(epoch = 0; epoch < numberOfEpochs; epoch++) {

        //int shuffles_n_return = 0; // Dud
        //int *p = &trainingSetOrder[0]; // Error:  Taking the address of a variable is not allowed without an external memory.

        next = my_rand(next);
        // Passing an address of a non access object is not supported (trainingSetOrder)
        //shuffle(trainingSetOrder, numTrainingSets, next, RAND_MAX); 
        //shuffles_n_return = shuffle(trainingSetOrder, numTrainingSets, next, RAND_MAX);

        /* shuffle (BEGIN) */
        unsigned long long u;
        for (u = 0; u < (numTrainingSets - 1); u++) {
            next = my_rand(next);
            unsigned long long w = u + my_rand(next) / (RAND_MAX / (numTrainingSets - u) + 1);
            int t = trainingSetOrder[w];
            trainingSetOrder[w] = trainingSetOrder[u];
            trainingSetOrder[u] = t;
        }
        /* shuffle (END) */

        for(x = 0; x < numTrainingSets; x++) {

            i = trainingSetOrder[i];

            // Forward pass

            // Compute hidden layer activation
            for (j = 0; j < numHiddenNodes; j++) {

                double activation = hiddenLayerBias[j];

                for (k = 0 ; k < numInputs; k++) {
                    activation += training_inputs[i][k] * hiddenWeights[k][j];
                }

                hiddenLayer[j] = sigmoid(activation);
            }

            // Compute output layer activation
            for (j = 0; j < numOutputs; j++) {

                double activation = outputLayerBias[j];

                for (k = 0 ; k < numHiddenNodes; k++) {
                    activation += hiddenLayer[k] * outputWeights[k][j];
                }

                outputLayer[j] = sigmoid(activation);
            }

            /**
            printf("Input: %g %g\tOutput: %g\tPredicted output: %g \n", training_inputs[i][0],
                    training_inputs[i][1], outputLayer[0], training_outputs[i][0]);
            **/
            // Backpropagation

            // Compute change in output weights

            double deltaOutput[1]; // numOutputs

            for(j = 0; j < numOutputs; j++) {
                double error = (training_outputs[i][j] - outputLayer[j]);
                deltaOutput[j] = error * dSigmoid(outputLayer[j]);
            }

            // Compute change in hidden weights

            double deltaHidden[2]; // numHiddenNodes

            for(j = 0; j < numHiddenNodes; j++) {
                double error = 0.0f;

                for(k = 0; k < numOutputs; k++) {
                    error += deltaOutput[k] * outputWeights[j][k];
                }
                deltaHidden[j] = error * dSigmoid(hiddenLayer[j]);
            }

            // Apply change in output weights
            
            for(j = 0; j < numOutputs; j++) {
                outputLayerBias[j] += deltaOutput[j] * lr;

                for(k = 0; k < numHiddenNodes; k++) {
                    outputWeights[k][j] += hiddenLayer[k] * deltaOutput[j] * lr;
                }
            }

            
            // Apply change in hidden weights
            
            for(j = 0; j < numHiddenNodes; j++) {
                hiddenLayerBias[j] += deltaHidden[j] * lr;

                for(k = 0; k < numInputs; k++) {
                    hiddenWeights[k][j] += training_inputs[i][k] * deltaHidden[j] * lr;
                }
            }

        }

    }

    // Print final weight after done training
    //fputs("Final hidden weights \n[", stdout); // The original video uses fputs
    /**
    printf("Final hidden weights \n[");
    for(j = 0; j < numHiddenNodes; j++) {
        printf("[  ");
        for(k = 0; k < numInputs; k++) {
            printf("%f ", hiddenWeights[k][j]);
        }
        printf("  ]");
    }

    printf("]\nFinal hidden biases \n[ ");
    for(j = 0; j < numHiddenNodes; j++) {
        printf("%f ", hiddenLayerBias[j]);
    }
            
    printf("]\nFinal output weights \n[");
    for(j = 0; j < numOutputs; j++) {
        printf("[  ");
        for(k = 0; k < numHiddenNodes; k++) {
            printf("%f ", outputWeights[k][j]);
        }
        printf("  ]");
    }

    printf("]\nFinal output biases \n[ ");
    for(j = 0; j < numOutputs; j++) {
        printf("%f ", outputLayerBias[j]);
    }
    printf("]\n");
    **/
   
    return 0;
}