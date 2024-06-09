package nn.neuralnetwork;

import nn.matrix.Matrix;

public class TrainingMatrices {
	private Matrix input;
	private Matrix outpur;
	
	public TrainingMatrices(Matrix input, Matrix outpur) {
		this.input = input;
		this.outpur = outpur;
	}

	public Matrix getInput() {
		return input;
	}

	public void setInput(Matrix input) {
		this.input = input;
	}

	public Matrix getOutpur() {
		return outpur;
	}

	public void setOutpur(Matrix outpur) {
		this.outpur = outpur;
	}
	
	
	
}
