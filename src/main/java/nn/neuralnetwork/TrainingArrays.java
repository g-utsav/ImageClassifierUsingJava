package nn.neuralnetwork;

public class TrainingArrays {
	private double[] input;
	private double[] outpur;
	
	public TrainingArrays(double[] input, double[] outpur) {
		this.input = input;
		this.outpur = outpur;
	}

	public double[] getInput() {
		return input;
	}

	public void setInput(double[] input) {
		this.input = input;
	}

	public double[] getOutput() {
		return outpur;
	}

	public void setOutpur(double[] outpur) {
		this.outpur = outpur;
	}	
}
