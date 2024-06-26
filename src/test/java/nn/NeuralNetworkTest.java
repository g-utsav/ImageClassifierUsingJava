package nn;

import static org.junit.Assert.*;

import java.util.Random;

import org.junit.Test;

import nn.matrix.Matrix;
import nn.neuralnetwork.Approximator;
import nn.neuralnetwork.BatchResult;
import nn.neuralnetwork.Engine;
import nn.neuralnetwork.LossFunctions;
import nn.neuralnetwork.RunningAverages;
import nn.neuralnetwork.Transform;
import nn.neuralnetwork.Util;

public class NeuralNetworkTest {

	private Random random = new Random();
	
	@Test
	public void testTrainEngin() {
		int inputRows = 500;
		int cols = 32;
		int outputRows = 10;
		
		
		Engine engine = new Engine();
		engine.add(Transform.DENSE, 100, inputRows);
		engine.add(Transform.RELU, null);
		engine.add(Transform.DENSE, outputRows);
		engine.add(Transform.SOFTMAX);
		
		RunningAverages runningAverages = new RunningAverages(2, 2, (callNumber, averages) -> {
//			assertTrue(averages[0] < 6);
			System.out.printf("%d.  Loss : %.3f  -- Percent Correct : %.2f \n",callNumber, averages[0], averages[1]);
		});
		
//		System.exit(0);
		double initialLearningRate = 0.02;
		double learningRate = initialLearningRate;
		double iterations = 20;
		for(int i=0; i<iterations; i++) {
//			Matrix input = Util.generateInputMatrix(inputRows, cols);
//			Matrix expected = Util.generateTrainableExpectedMatrix(outputRows, input);
			
			var tm = Util.generateTrainingMatrices(inputRows, outputRows, cols);
			var input = tm.getInput();
			var expected = tm.getOutpur();
			
			BatchResult batchResult = engine.runForewards(input);
	//		engine.evaluate(batchResult, expected);
	//		double loss1 = batchResult.getLoss();
		
			engine.runBackWard(batchResult, expected);	
			engine.adjust(batchResult, learningRate);
			
	//		batchResult = engine.runForewards(input);
			engine.evaluate(batchResult, expected);
			
			
			runningAverages.add(batchResult.getLoss(), batchResult.getPercentCorrect());
//			double loss2 = batchResult.getLoss();
//			double percentCorrect = batchResult.getPercentCorrect();
//			
//			System.out.printf("Loss : %.3f, %% Correct : %.2f \n",loss2, percentCorrect);
			
			learningRate -= (initialLearningRate/iterations);
		}
//		System.out.println(loss1 + " " + loss2);
//		System.out.println(percentCorrect);
		
	}
	
	@Test
	public void testWeightGradient() {
		int inputRows = 4;
		int outputRows = 5;
		
		Matrix weight = new Matrix(outputRows, inputRows, i->random.nextGaussian());
		Matrix input = Util.generateInputMatrix(inputRows, 1);
		Matrix expected  = Util.generateExpectedMatrix(outputRows, 1);
		
		Matrix output = weight.multiply(input).softMax();
		
		Matrix loss = LossFunctions.crossEntropy(expected, output);
				
		Matrix calculatedError = output.apply((index, value) -> value - expected.get(index));
		
		Matrix calculatedWeightGraidient = calculatedError.multiply(input.transpose());
		
		Matrix approximatedWeightGraidents = Approximator.wightGradient(weight, w ->{
			Matrix out = w.multiply(input).softMax();
			return LossFunctions.crossEntropy(expected, out);
		});
		
		calculatedWeightGraidient.setTolerance(0.0001);
		
		assertTrue(calculatedWeightGraidient.equals(approximatedWeightGraidents));
		
//		System.out.println(input);
//		System.out.println(weight);
//		System.out.println(output);
//		System.out.println(expected);
//		System.out.println(loss);
//		System.out.println(calculatedError);
//		System.out.println(calculatedWeightGraidient);
//		System.out.println(approximatedWeightGraidents);
	}
	
	@Test
	public void testEngine() {
		int inputRows = 5;
		int cols = 6;
		int outputRows = 4;
		
		Engine engine = new Engine();
		
		engine.add(Transform.DENSE, 8,5);
		engine.add(Transform.RELU);
		engine.add(Transform.DENSE, 5);
		engine.add(Transform.RELU);
		engine.add(Transform.DENSE, 4);
		engine.add(Transform.SOFTMAX);

		engine.setStoreInputError(true);
		
		Matrix input = Util.generateInputMatrix(inputRows, cols);
		Matrix expected  = Util.generateExpectedMatrix(outputRows, cols);
		
		
		Matrix approximatedError = Approximator.gradient(input, in -> {
			BatchResult batchResult = engine.runForewards(in);
			return LossFunctions.crossEntropy(expected, batchResult.getOutput());
		});
		
		BatchResult batchResult = engine.runForewards(input);
		engine.runBackWard(batchResult, expected);
		
		Matrix calculatedError = batchResult.getInputError();
		
//		System.out.println(calculatedError);
//		System.out.println(approximatedError);
		
		calculatedError.setTolerance(0.0001);
		
		assertTrue(calculatedError.equals(approximatedError));
//		System.out.println(input);
//		System.out.println(engine);
//		System.out.println(output);
	}
	
	@Test
	public void testBackPropReLu() {
		
		interface NeuralNet{
			Matrix apply(Matrix m);
		}
		
		final int inputRows = 4;
		final int cols = 5;
		final int outputRows = 4;
		
		Matrix input = new Matrix(inputRows, cols, i->random.nextGaussian());
		
		Matrix expected = new Matrix(outputRows, cols, i->0);
		
		Matrix weights = new Matrix(outputRows, inputRows, i->random.nextGaussian());
		Matrix biases = new Matrix(outputRows, 1, i->random.nextGaussian());
		
		for(int col=0; col<expected.getCols(); col++) {
			int randomRow = random.nextInt(outputRows);
			expected.set(randomRow, col, 1);
		}		
		
		NeuralNet neuralNet = m -> {
			Matrix out = m.apply((index, value) -> value > 0 ? value : 0);
			out = weights.multiply(out);
			out = out.modify((row, col, value) -> value + biases.get(row));
			out = out.softMax();
//			return weights.multiply(m).modify((row, col, value) -> value + biases.get(row)).softMax();
			return out;
		};
		
		Matrix softMaxOutput = neuralNet.apply(input);
		
		Matrix approximatedResult = Approximator.gradient(input, in -> {
			//in =  weights.multiply(in).modify((row, col, value) -> value + biases.get(row));
//			Matrix a = LossFunction.crossEntropy(expected, in.softMax());
			Matrix out = neuralNet.apply(in);
			Matrix a = LossFunctions.crossEntropy(expected, out);
			return a;
		});
		
		Matrix calculatedResult = softMaxOutput.apply((index, value) -> value - expected.get(index));
		
		calculatedResult = weights.transpose().multiply(calculatedResult);
		calculatedResult = calculatedResult.apply((index, value) -> input.get(index) > 0 ? value : 0);
		
//		System.out.println(calculatedResult);
//		System.out.println(approximatedResult);
		
		assertTrue(approximatedResult.equals(calculatedResult));
	}

	@Test
	public void testBackPropWeights() {
		
		interface NeuralNet{
			Matrix apply(Matrix m);
		}
		
		final int inputRows = 4;
		final int cols = 5;
		final int outputRows = 4;
		
		Matrix input = new Matrix(inputRows, cols, i->random.nextGaussian());
		
		Matrix expected = new Matrix(outputRows, cols, i->0);
		
		Matrix weights = new Matrix(outputRows, inputRows, i->random.nextGaussian());
		Matrix biases = new Matrix(outputRows, 1, i->random.nextGaussian());
		
		for(int col=0; col<expected.getCols(); col++) {
			int randomRow = random.nextInt(outputRows);
			expected.set(randomRow, col, 1);
		}		
		
		NeuralNet neuralNet = m -> weights.multiply(m).modify((row, col, value) -> value + biases.get(row)).softMax();
		
		Matrix softMaxOutput = neuralNet.apply(input);
		
		Matrix approximatedResult = Approximator.gradient(input, in -> {
			//in =  weights.multiply(in).modify((row, col, value) -> value + biases.get(row));
//			Matrix a = LossFunction.crossEntropy(expected, in.softMax());
			Matrix out = neuralNet.apply(in);
			Matrix a = LossFunctions.crossEntropy(expected, out);
			return a;
		});
		
		Matrix calculatedResult = softMaxOutput.apply((index, value) -> value - expected.get(index));
		
		calculatedResult = weights.transpose().multiply(calculatedResult);
		
//		System.out.println(calculatedResult);
//		System.out.println(approximatedResult);
		
		assertTrue(approximatedResult.equals(calculatedResult));
	}

	@Test
	public void testSoftMaxCrossEntropyGradient() {
		final int rows = 4;
		final int cols = 5;
		
		Matrix input = new Matrix(rows, cols, i->random.nextGaussian());
		
		Matrix expected = new Matrix(rows, cols, i->0);
		
		for(int col=0; col<expected.getCols(); col++) {
			int randomRow = random.nextInt(rows);
			expected.set(randomRow, col, 1);
		}		
		
		Matrix softMaxOutput = input.softMax();
		
		Matrix result = Approximator.gradient(input, in -> {
			Matrix a = LossFunctions.crossEntropy(expected, in.softMax());
			return a;
		});
		
		result.forEach((index, value) -> {
			double softMaxValue = softMaxOutput.get(index);
			double expectedValue = expected.get(index);
			
			assertTrue(Math.abs(value - (softMaxValue - expectedValue)) < 0.000001);
			
//			System.out.println(value+" , "+(softMaxValue-expectedValue));
		});
		
//		input.forEach((index, value)->{
//			double resultValue = result.get(index);
//			double expectedValue = expected.get(index);
//			
//			if(expectedValue < 0.001) {
//				assertTrue(Math.abs(resultValue) < 0.01);
//			}else {
//				assertTrue(Math.abs(resultValue - (-1.0/value)) < 0.1);
//			}
//		});
//		
//		System.out.println(input);
//		System.out.println(expected);
//		System.out.println(result);
	}
	
	@Test
	public void testApproximator() {
		final int rows = 4;
		final int cols = 5;
		
		Matrix input = new Matrix(rows, cols, i->random.nextGaussian()).softMax();
		
		Matrix expected = new Matrix(rows, cols, i->0);
		
		for(int col=0; col<expected.getCols(); col++) {
			int randomRow = random.nextInt(rows);
			expected.set(randomRow, col, 1);
		}		
		
		Matrix result = Approximator.gradient(input, in -> {
			Matrix a = LossFunctions.crossEntropy(expected, in);
			return a;
		});
		
		input.forEach((index, value)->{
			double resultValue = result.get(index);
			double expectedValue = expected.get(index);
			
			if(expectedValue < 0.001) {
				assertTrue(Math.abs(resultValue) < 0.01);
			}else {
				assertTrue(Math.abs(resultValue - (-1.0/value)) < 0.1);
			}
		});;
		
//		System.out.println(input);
//		System.out.println(expected);
//		System.out.println(result);
	}

	@Test
	public void testCrossEntropy() {
		double[] expectedValues = {1,0,0,0,0,1,0,1,0};
		
		Matrix expected = new Matrix(3,3, i -> expectedValues[i]);
		
		Matrix actual  = new Matrix(3,3,i->0.001*i*i).softMax();
		
		Matrix result = LossFunctions.crossEntropy(expected, actual);
		
		
		actual.forEach((row,col,index,value)->{
			double expectedValue = expected.get(index);
			
			double loss = result.get(col);
			
			if(expectedValue > 0.9) {
				assertTrue(Math.abs(Math.log(value)+loss) < 0.001);
			}
		});
		
//		System.out.println(result);
//		System.out.println(expected);
//		System.out.println(actual);
	}
	
	
	
	//@Test
	public void testTemp() {
		
		int inputSize = 5;
		int layer1Size = 6;
		int layer2Size = 4;
		
		Matrix input = new Matrix(inputSize,1, i->random.nextGaussian());
		
		Matrix layer1Weight = new Matrix(layer1Size, input.getRows(), i->random.nextGaussian());
		Matrix layer1Biases = new Matrix(layer1Size, 1, i->random.nextGaussian());
		
		Matrix layer2Weight = new Matrix(layer2Size, layer1Weight.getRows(), i->random.nextGaussian());
		Matrix layer2Biases = new Matrix(layer2Size, 1, i->random.nextGaussian());
		
		var output = input;
		System.out.println(output);
		
		output = layer1Weight.multiply(output);
		System.out.println(output);
		
		output = output.modify((row, col, value) -> value + layer1Biases.get(row));
		System.out.println(output);
		
		output = output.modify(value -> value > 0 ? value : 0);
		System.out.println(output);
		
		output = layer2Weight.multiply(output);
		System.out.println(output);
		
		output = output.modify((row, col, value) -> value + layer2Biases.get(row));
		System.out.println(output);
		
		output = output.softMax();
		System.out.println(output);
	}
	
	@Test
	public void testAddBiases() {
		Matrix input = new Matrix(3,3,i->(i+1));
		Matrix weight = new Matrix(3,3,i->(i+1));
		Matrix biases = new Matrix(3,1,i->(i+1));
		
		double[] expectedValues = { +31.00000,+37.00000,+43.00000,+68.00000,+83.00000,+98.00000,+105.00000,+129.00000,+153.00000};
		
		Matrix expected  = new Matrix(3,3, i->expectedValues[i]);
		
		Matrix result = weight.multiply(input).modify((row, col, value) -> value + biases.get(row));
		
		assertTrue(expected.equals(result));
		
//		System.out.println(input);
//		System.out.println(weight);
//		System.out.println(biases);
//		System.out.println(result);

	}

	@Test
	public void testReLu() {
		
		final var numberNeurons = 5;
		final var numberInputs = 6;
		final var inputSize = 4;
		
		Matrix input = new Matrix(inputSize,numberInputs,i->random.nextDouble());
		Matrix weight = new Matrix(numberNeurons,inputSize,i->random.nextGaussian());
		Matrix biases = new Matrix(numberNeurons,1,i-> random.nextGaussian());
		
		Matrix result1 = weight.multiply(input).modify((row, col, value) -> value + biases.get(row));
		Matrix result2 = weight.multiply(input).modify((row, col, value) -> value + biases.get(row)).modify(value -> value > 0 ? value : 0);
		
		result2.forEach((index,value)->{
//			System.out.println(index+", "+value);
			double originalValue = result1.get(index);
			
			if(originalValue > 0) {
				assertTrue(Math.abs(originalValue-value) < 0.000001);
			}else {
				assertTrue(Math.abs(value) < 0.000001);
			}
		});
		
//		System.out.println(input);
//		System.out.println(weight);
//		System.out.println(biases);
//		System.out.println(result1);
//		System.out.println(result2);
	}
}
