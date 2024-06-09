package nn;

import nn.loader.Loader;
import nn.loader.test.TestLoader;
import nn.neuralnetwork.NeuralNetwork;
import nn.neuralnetwork.Transform;

public class App {

	public static void main(String[] args) {
		int inputRows = 100;	
		int outputRows = 3;
		
		NeuralNetwork neuralNetwork = new NeuralNetwork();
		neuralNetwork.add(Transform.DENSE, 100, inputRows);
		neuralNetwork.add(Transform.RELU);
		neuralNetwork.add(Transform.DENSE, 100);
		neuralNetwork.add(Transform.RELU);
		neuralNetwork.add(Transform.DENSE, outputRows);
		neuralNetwork.add(Transform.SOFTMAX);
		
		neuralNetwork.setEpochs(20);
		neuralNetwork.setLearningRate(0.02, 0.001);
		neuralNetwork.setThreads(20);
		System.out.println(neuralNetwork);
		
		Loader trainLoader = new TestLoader(60_000, 32);
		Loader testLoader = new TestLoader(10_000, 32);
		
		neuralNetwork.fit(trainLoader, testLoader);
		
	}
	
}
