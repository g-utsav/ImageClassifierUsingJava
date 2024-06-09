package nn;

import java.io.File;
import java.io.IOException;

import nn.loader.BatchData;
import nn.loader.Loader;
import nn.loader.MetaData;
import nn.loader.test.TestLoader;
import nn.neuralnetwork.NeuralNetwork;
import nn.neuralnetwork.Transform;
import nn.neuralnetwork.loader.image.ImageLoader;

public class App {
	public static void main(String[] args) {

		final String fileName = "mnistNeural0.net";
		
		if(args.length == 0) {
			System.out.println("Usage : [app] <MNIST DATA DIRECTORY>");
			return;
		}

		String directory = args[0];

		if(!new File(directory).isDirectory()) {
			System.out.println("'"+ directory+"' is not directory.");
			return;
		}
		
		final String trainImages = String.format("%s%s%s", directory, File.separator, "train-images.idx3-ubyte");
		final String trainLabels = String.format("%s%s%s", directory, File.separator, "train-labels.idx1-ubyte");
		final String testImages = String.format("%s%s%s", directory, File.separator, "t10k-images.idx3-ubyte");
		final String testLables = String.format("%s%s%s", directory, File.separator, "t10k-labels.idx1-ubyte");
		
		Loader trainLoader = new ImageLoader(trainImages, trainLabels, 32);
		Loader testLoader = new ImageLoader(testImages, testLables, 32);
		
		MetaData metaData = trainLoader.open();
		int inputSize = metaData.getInputSize();
		int outputSize = metaData.getExpectedSize();
		trainLoader.close();
		
		NeuralNetwork neuralNetwork = NeuralNetwork.load(fileName);
		
		if(neuralNetwork == null) {
			System.out.println("Unable to load Neural Network from Saved file.Creating a new Instance. ");
			
			neuralNetwork = new NeuralNetwork();
			neuralNetwork.setScaleInitialWeights(0.2);
			neuralNetwork.setEpochs(20);
			neuralNetwork.setLearningRate(0.02, 0.001);
			neuralNetwork.setThreads(20);

			neuralNetwork.add(Transform.DENSE, 200, inputSize);
			neuralNetwork.add(Transform.RELU);
			neuralNetwork.add(Transform.DENSE, 100);
			neuralNetwork.add(Transform.RELU);
//			neuralNetwork.add(Transform.DENSE, 50);
//			neuralNetwork.add(Transform.RELU);
			neuralNetwork.add(Transform.DENSE, outputSize);
			neuralNetwork.add(Transform.SOFTMAX);
			
		}else {
			System.out.println("Loaded from "+fileName);
		}
		
		System.out.println(neuralNetwork);

		neuralNetwork.fit(trainLoader, testLoader);
		
		if(neuralNetwork.save(fileName)) {
			System.out.println("Saved to "+fileName);
		}else {
			System.out.println("Unable to save to "+fileName);
		}
		
	}
}
