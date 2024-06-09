package nn.neuralnetwork.loader.image;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;

import javax.imageio.ImageIO;

import nn.loader.BatchData;
import nn.loader.Loader;
import nn.loader.MetaData;
import nn.neuralnetwork.NeuralNetwork;

public class GeneratePridictedImageWriter {
	public static void main(String[] args) {
		if(args.length == 0) {
			System.out.println("Usage : [app] <MNIST DATA DIRECTORY>");
			return;
		}
		
		String directory = args[0];
		
		if(!new File(directory).isDirectory()) {
			System.out.println("'"+ directory+"' is not directory.");
			return;
		}
		
		new GeneratePridictedImageWriter().run(directory);
	}
	
	private int convertOneHotToInt(double[] labelData, int offSet, int oneHotSize) {
		double maxValue = 0;
		int maxIndex = 0;
		
		for(int i = 0; i < oneHotSize; i++) {
			if(labelData[offSet + i] > maxValue) {
				maxValue = labelData[offSet + i];
				maxIndex = i;
			}
		}
		return maxIndex;
		
//		for(int i=0; i<oneHotSize; i++) {
//			if(Math.abs(labelData[offSet + i] - 1) < 0.001) {
//				return i;
//			}
//		}
//		throw new RuntimeException("Invalid one hot vector.");
	}

	private void run(String directory) {
//		final String trainImages = String.format("%s%s%s", directory, File.separator, "train-images.idx3-ubyte");
//		final String trainLabels = String.format("%s%s%s", directory, File.separator, "train-labels.idx1-ubyte");
		final String testImages = String.format("%s%s%s", directory, File.separator, "t10k-images.idx3-ubyte");
		final String testLables = String.format("%s%s%s", directory, File.separator, "t10k-labels.idx1-ubyte");
		
		int batchSize = 900;
		
//		ImageLoader trainLoader = new ImageLoader(trainImages, trainLabels, batchSize);
		ImageLoader testLoader = new ImageLoader(testImages, testLables, batchSize);
		
		ImageMetaData metaData = testLoader.open();
		
		NeuralNetwork neuralNetwork = NeuralNetwork.load("mnistNeural0.net");
		
		int imageWidth = metaData.getWidth();
		int imageHeight = metaData.getHeight();
		
		int labelSize = metaData.getExpectedSize();
		
		for(int i=0; i<metaData.getNumberBatches(); i++) {
			BatchData batchData = testLoader.readBatch();
			
			var numberImages = metaData.getItemsRead();
			int horizontalImages = (int)Math.sqrt(numberImages);
			
			while(numberImages % horizontalImages != 0) {
				++horizontalImages;
			}
			
			int verticalImages = numberImages / horizontalImages;
			
			int canvasWidth = horizontalImages * imageWidth;
			int canvasHeight = verticalImages * imageHeight;
			
			
			String montagePath = String.format("%s%s%s%smontage%d.jpg",directory,File.separator,"images",File.separator, i);
			System.out.println("Writing images "+montagePath);
			
			var montage = new BufferedImage(canvasWidth, canvasHeight ,BufferedImage.TYPE_INT_RGB);
			
			double[] pixelData = batchData.getInputBatch();
			double[] labelData = batchData.getExpectedBatch();
			
			int imageSize = imageWidth * imageHeight; 
			boolean[] correct = new boolean[numberImages];
			for(int n = 0; n < numberImages; n++) {
				double[] singleImage = Arrays.copyOfRange(pixelData, n * imageSize, (n + 1) * imageSize);
				double[] singleLabel = Arrays.copyOfRange(labelData, n * labelSize, (n + 1) * labelSize);
				
				double[] predictedLabel = neuralNetwork.predict(singleImage);
				
				int predicted = convertOneHotToInt(predictedLabel, 0, labelSize);
				int actual = convertOneHotToInt(singleLabel, 0, labelSize);
				
				correct[n] = predicted == actual;
			}
			
			for(int pixelIndex=0; pixelIndex<pixelData.length; pixelIndex++) {
				int imageNumber = pixelIndex / imageSize;
				int pixelNumber = pixelIndex % imageSize;
				
				int montageRow = imageNumber / horizontalImages;
				int montageCol = imageNumber % horizontalImages;
				
				int pixelRow = pixelNumber / imageWidth;
				int pixelCol = pixelNumber % imageWidth;
				
				int x = montageCol * imageWidth + pixelCol;
				int y = montageRow * imageHeight + pixelRow;
				
				double pixelValue = pixelData[pixelIndex];
				int color = (int)(0x100 * pixelValue);
				
				int pixelColor = 0;
				
				if(correct[imageNumber]) {
					pixelColor = (color << 16) + (color << 8); //+ (color);					
				}else {
					pixelColor = (color << 16);// + (color << 8) + (color);
				}
						
				
				montage.setRGB(x, y, pixelColor);
			}
			
			try {
				ImageIO.write(montage, "jpg", new File(montagePath));
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			
			StringBuilder sb = new StringBuilder();
			for(int labelIndex = 0; labelIndex < numberImages; labelIndex++) {
				if(labelIndex % horizontalImages == 0) {
					sb.append("\n");
				}
				
				int label = convertOneHotToInt(labelData, labelIndex * labelSize, labelSize);
				sb.append(String.format("%d ", label));
			}
			String labelPath = String.format("%s%s%s%slabels%d.txt",directory,File.separator,"images",File.separator, i);
			System.out.println("Writing labels "+labelPath);
			try {
				FileWriter fw = new FileWriter(labelPath);
				fw.write(sb.toString());
				fw.flush();
				fw.close();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			
		}
		
		testLoader.close();
	}
}
