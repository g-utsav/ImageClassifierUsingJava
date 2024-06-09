package nn.neuralnetwork;

import java.util.stream.DoubleStream;

public class RunningAverages {

	private int nCalls = 0;
	private double[][] values;
	private Callback callback;
	private int pos = 0;
	
	public interface Callback{
		public void apply(int callNumber, double[] averages);
	}
	
	public RunningAverages(int numberAverages, int windowSize, Callback callback) {
		this.callback = callback;
		values = new double[numberAverages][windowSize];
		
//		System.out.println(values.length);
//		System.out.println(values[0].length);
	}
	
	public void add(double ...args) {
		for(int i=0; i<values.length; i++) {
			values[i][pos] = args[i];
		}
		if(++pos == values[0].length) {
			double[] averages = new double[values.length];
			for(int i=0; i<values.length; i++) {
				averages[i] = DoubleStream.of(values[i]).average().getAsDouble();
			}
			callback.apply(++nCalls, averages);
			pos = 0;
		}
	}
}
