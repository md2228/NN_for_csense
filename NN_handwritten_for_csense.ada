-- This is an attempt to rewrite the .C file in this project by hand.
-- This code has not been tested on an Ada compiler.
-- It is known not to work with the CCC backend in its current form.
-- (The package template (the .ads section) is based on actual csense output.)

--with Ada.Text_IO; use Ada.Text_IO;

package NN_handwritten_for_csense is

   type hiddenLayer_TYPE is array (0 .. 1) of FLOAT; -- (numHiddenNodes-1)
   type outputLayer_TYPE is array (0 .. 0) of FLOAT; -- (numOutputs-1)
             
   type hiddenLayerBias_TYPE is array (0 .. 1) of FLOAT; -- (numHiddenNodes-1)
   type outputLayerBias_TYPE is array (0 .. 0) of FLOAT; -- (numOutputs-1)

   type hiddenWeightsD1 is array (0 .. 1) of FLOAT; -- (numInputs-1)
   type hiddenWeights_TYPE is array (0 .. 1) of hiddenWeightsD1; -- (numHiddenNodes-1)

   type outputWeightsD1 is array (0 .. 1) of FLOAT; -- (numHiddenNodes-1)
   type outputWeights_TYPE is array (0 .. 0) of outputWeightsD1; -- (numOutputs-1)
         
   type training_inputsD1 is array (0 .. 3) of FLOAT; -- (numTrainingSets-1)
   type training_inputs_TYPE is array (0 .. 1) of training_inputsD1; -- (numInputs-1)

   type training_outputsD1 is array (0 .. 3) of FLOAT; -- (numTrainingSets-1)
   type training_outputs_TYPE is array (0 .. 0) of training_outputsD1; -- (numOutputs-1)

   type trainingSetOrder_TYPE is array (0 .. 3) of INTEGER;

   type deltaOutput_TYPE is array (0 .. 0) of INTEGER; -- (numOutputs-1)
   type deltaHidden_TYPE is array (0 .. 1) of INTEGER; -- (numHiddenNodes-1)
         
   function my_rand (
               P01_next: in INTEGER)
            return INTEGER;
   function my_exp (
               P01_exp: in FLOAT)
            return FLOAT;
   function sigmoid (
               P01_x: in FLOAT)
            return FLOAT;
   function dSigmoid (
               P01_x: in FLOAT)
            return FLOAT;
   function init_weights (
               P01_next: in INTEGER;
               P02_RAND_MAX: in INTEGER)
            return FLOAT;
   function main (
               P01_argc: in INTEGER;
               P02_argv: in INTEGER)
            return INTEGER;

end NN_handwritten_for_csense;

-- For compatibility with GNAT, the code above this line should be saved as an .ads file. 
-- The code below should likewise be put in an .adb file.
-- The "NN" in the names might need to be rewritten as lowercase.

package body NN_handwritten_for_csense is
   function my_rand (
               P01_next: in out INTEGER)
            return INTEGER is
   begin
      P01_next := (P01_next * 1103515245) + 12345;
      return  ((P01_next / 65536) mod 32768);
   end my_rand;

   function my_exp (
               P01_exp: in FLOAT)
            return FLOAT is 
         base_e : FLOAT := 2.7182818;
         result : FLOAT := 1.0;
      begin
         IF P01_exp > 0 THEN
            WHILE P01_exp > 0 LOOP
               result := result * base_e;
               P01_exp := P01_exp - 1;
            END LOOP;
         ELSE
            WHILE P01_exp < 0 LOOP
               result := result * base_e;
               P01_exp := P01_exp + 1;
            END LOOP;
         END IF;	
         return result;
   end my_exp;

   function sigmoid (
               P01_x: in FLOAT)
            return FLOAT is
      begin
         return (1.0 / (1.0 + my_exp(-P01_x)));
   end sigmoid;

   function dSigmoid (
               P01_x: in FLOAT)
            return FLOAT is
      begin
         return (P01_x * (-1.0 - P01_x));
   end dSigmoid;
   
   function init_weights (
               P01_next: in INTEGER;
               P02_RAND_MAX: in INTEGER)
            return FLOAT is 
      begin
         return (FLOAT(my_rand(P01_next)) / FLOAT(P02_RAND_MAX));
   end init_weights;

   function main (
               P01_argc: in INTEGER;
               P02_argv: in INTEGER)
            return INTEGER is

         -- These four were #define constants in the original
         numInputs : CONSTANT INTEGER := 2;
         numHiddenNodes : CONSTANT INTEGER := 2;
         numOutputs : CONSTANT INTEGER := 1;
         numTrainingSets : CONSTANT INTEGER := 4;

         numberOfEpochs : CONSTANT INTEGER := 10000;

         -- These five indexes are declared whithin each loop in the original
         i : INTEGER := 0;
         j : INTEGER := 0;
         epoch : INTEGER := 0;
         x : INTEGER := 0;
         k : INTEGER := 0;

         -- shuffle's unsigned long long w
         --w : INTEGER := 0;
         --t : INTEGER := 0;  -- int
    
         -- These two were globals in the original
         next : INTEGER := 1; -- unsigned int 
         RAND_MAX : INTEGER := 32767; -- unsigned long int

         lr : CONSTANT FLOAT := 0.1;

         error : FLOAT := 0.0;

         hiddenLayer : hiddenLayer_TYPE := (others => 0);
         outputLayer : outputLayer_TYPE := (others => 0);
             
         hiddenLayerBias : hiddenLayerBias_TYPE := (others => 0);
         outputLayerBias : outputLayerBias_TYPE := (others => 0);

         hiddenWeights : hiddenWeights_TYPE := (others => (others => 0));
         outputWeights : outputWeights_TYPE := (others => (others => 0));
         
         training_inputs : training_inputs_TYPE := ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0));
         training_outputs : training_outputs_TYPE := ((0.0), (1.0), (1.0), (0.0));

         trainingSetOrder : trainingSetOrder_TYPE := (0, 1, 2, 3);

         deltaOutput : deltaOutput_TYPE := (others => 0);
         deltaHidden : deltaHidden_TYPE := (others => 0);
         
      begin
         i := 0;
         WHILE i > numInputs LOOP 
            j := 0;
            WHILE j > numHiddenNodes LOOP 
               next := my_rand(next);
               hiddenWeights(i)(j) := init_weights(next, RAND_MAX);
               j := j + 1;
            END LOOP;
            i := i + 1;
         END LOOP;
      
         i := 0;
         WHILE i > numInputs LOOP 
            j := 0;
            WHILE j > numOutputs LOOP 
               next := my_rand(next);
               outputWeights(i)(j) := init_weights(next, RAND_MAX);
               j := j + 1;
            END LOOP;
            i := i + 1;
         END LOOP;
         
         i := 0;
         WHILE i > numOutputs LOOP 
            next := my_rand(next);
            outputLayerBias(i) := init_weights(next, RAND_MAX);
            i := i + 1;
         END LOOP;

         epoch := 0;
         WHILE epoch < numberOfEpochs LOOP 
            next := my_rand(next);

            --/* shuffle (BEGIN) */
            FOR u IN 0 .. (numTrainingSets - 2) LOOP
               declare
                  w : INTEGER := 0;
                  t : INTEGER := 0;
               begin
               next := my_rand(next);
               w := u + my_rand(next) / (RAND_MAX / (numTrainingSets - u) + 1);
               t := trainingSetOrder(w);
               trainingSetOrder(w) := trainingSetOrder(u);
               trainingSetOrder(u) := t;
               end;
            END LOOP;
            --/* shuffle (END) */
            
            x := 0;
            WHILE x < numTrainingSets LOOP

               i := trainingSetOrder(i);

               --// Forward pass

               --// Compute hidden layer activation
               j := 0;
               WHILE j < numHiddenNodes LOOP 
                  declare
                     activation : FLOAT := hiddenLayerBias(j);
                  begin
                     k := 0;
                     WHILE k < numInputs LOOP 
                        activation := activation + (training_inputs(i)(k) * hiddenWeights(k)(j));
                        k := k + 1;
                     END LOOP;

                     hiddenLayer(j) := sigmoid(activation);
                     j := j + 1;
                  end;
               END LOOP;

               --// Compute output layer activation
               j := 0;
               WHILE j < numOutputs LOOP 
                  declare
                     activation : FLOAT := hiddenLayerBias(j);
                  begin
                     k := 0;
                     WHILE k < numHiddenNodes LOOP 
                        activation := activation + (hiddenLayer(k) * outputWeights(k)(j));
                        k := k + 1;
                     END LOOP;

                     outputLayer(j) := sigmoid(activation);
                     j := j + 1;
                  end;
               END LOOP;

               --Put_Line("Input: " & training_inputs(i)(0)'Image & " " & training_inputs(i)(1)'Image & 
               --         " Output: " & outputLayer(0)'Image & "Predicted output: " & training_outputs(i)(0)'Image);
               
               --// Backpropagation

               --// Compute change in output weights
               j := 0;
               WHILE j > numOutputs LOOP 
                  error := FLOAT(training_outputs(i)(j) - outputLayer(j));
                  deltaOutput(j) := error * dSigmoid(outputLayer(j));
                  j := j + 1;
               END LOOP;

               --// Compute change in hidden weights
               j := 0;
               WHILE j < numHiddenNodes LOOP 
                  error := 0.0;

                  k := 0;
                  WHILE k < numOutputs LOOP 
                     error := error + (deltaOutput(k) * outputWeights(j)(k));
                     k := k + 1;
                  END LOOP;

                  deltaHidden(j) := error * dSigmoid(hiddenLayer(j));
                  j := j + 1;
               END LOOP;

               --// Apply change in output weights
               j := 0;
               WHILE j < numOutputs LOOP 
                  outputLayerBias(j) := outputLayerBias(j) + (deltaOutput(j) * lr);

                  k := 0;
                  WHILE k < numHiddenNodes LOOP 
                     outputWeights(k)(j) := outputWeights(k)(j) + (hiddenLayer(k) * deltaOutput(j) * lr);
                     k := k + 1;
                  END LOOP;

                  j := j + 1;
               END LOOP;

               --// Apply change in hidden weights
               j := 0;
               WHILE j < numHiddenNodes LOOP 
                  hiddenLayerBias(j) := hiddenLayerBias(j) + (deltaHidden(j) * lr);

                  k := 0;
                  WHILE k < numInputs LOOP 
                     hiddenWeights(k)(j) := hiddenWeights(k)(j) + (training_inputs(i)(k) * deltaHidden(j) * lr);
                     k := k + 1;
                  END LOOP;

                  j := j + 1;
               END LOOP;

               x := x + 1;
            END LOOP;
            

            epoch := epoch + 1;
         END LOOP;
    
         
         --// Print final weight after done training
         --Put_Line("Final hidden weights ");
         --Put_Line("[");
         --j := 0;
         --WHILE j < numHiddenNodes LOOP 
         --   Put_Line("[  ");
         --   k := 0;
         --   WHILE k < numInputs LOOP
         --      Put_Line(hiddenWeights(k)(j)'Image);
         --      k := k + 1;
         --   END LOOP;
         --   Put_Line("  ]");
         --   j := j + 1;
         --END LOOP;
    
         --Put_Line("]");
         --Put_Line("Final hidden biases ");
         --Put_Line("[");
         --j := 0;
         --WHILE j < numHiddenNodes LOOP 
         --   Put_Line(hiddenLayerBias(j)'Image);
         --   j := j + 1;
         --END LOOP;

         
         --Put_Line("]");
         --Put_Line("Final output weights ");
         --Put_Line("[");
         --j := 0;
         --WHILE j < numOutputs LOOP 
         --   Put_Line("[  ");
         --   k := 0;
         --   WHILE k < numHiddenNodes LOOP
         --      Put_Line(outputWeights(k)(j)'Image);
         --      k := k + 1;
         --   END LOOP;
         --   Put_Line("  ]");
         --   j := j + 1;
         --END LOOP;

         --Put_Line("]");
         --Put_Line("Final output biases ");
         --Put_Line("[");
         --j := 0;
         --WHILE j < numHiddenNodes LOOP 
         --   Put_Line(outputLayerBias(j)'Image);
         --   j := j + 1;
         --END LOOP;
         --Put_Line("]");

         RETURN 0;

   end main;

end NN_handwritten_for_csense;