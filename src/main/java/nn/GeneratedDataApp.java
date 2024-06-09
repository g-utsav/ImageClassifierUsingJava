package nn;

import nn.loader.Loader;
import nn.loader.test.TestLoader;
import nn.neuralnetwork.NeuralNetwork;
import nn.neuralnetwork.Transform;

public class GeneratedDataApp {

	public static void main(String[] args) {
		
		String fileName = "neural1.net";
		
		NeuralNetwork neuralNetwork = NeuralNetwork.load(fileName);
		
		if(neuralNetwork == null) {
			System.out.println("Unable to load Neural Network from Saved file.Creating a new Instance. ");
			
			int inputRows = 100;	
			int outputRows = 3;
			
			neuralNetwork = new NeuralNetwork();
			neuralNetwork.add(Transform.DENSE, 100, inputRows);
			neuralNetwork.add(Transform.RELU);
			neuralNetwork.add(Transform.DENSE, 100);
			neuralNetwork.add(Transform.RELU);
			neuralNetwork.add(Transform.DENSE, outputRows);
			neuralNetwork.add(Transform.SOFTMAX);
			
			neuralNetwork.setEpochs(20);
			neuralNetwork.setLearningRate(0.02, 0.001);
			neuralNetwork.setThreads(20);
		}else {
			System.out.println("Loaded from "+fileName);
		}
		
		System.out.println(neuralNetwork);
		
		
		Loader trainLoader = new TestLoader(60_000, 32);
		Loader testLoader = new TestLoader(10_000, 32);
		
		neuralNetwork.fit(trainLoader, testLoader);
		
		if(neuralNetwork.save(fileName)) {
			System.out.println("Saved to "+fileName);
		}else {
			System.out.println("Unable to save to "+fileName);
		}
		
	}
	
}
