package nn.neuralnetwork;

import java.util.LinkedList;

import nn.matrix.Matrix;

public class BatchResult {

	private LinkedList<Matrix> io = new LinkedList<Matrix>();
	private LinkedList<Matrix> weightErrors = new LinkedList<Matrix>();
	private LinkedList<Matrix> weightInputs = new LinkedList<Matrix>();
	private Matrix inputError;
	private double loss;
	private double percentCorrect;
	
	public void addWeightInput(Matrix input) {
		weightInputs.add(input);
	}
	
	public LinkedList<Matrix> getWeightInputs(){
		return this.weightInputs;
	}
	
	public LinkedList<Matrix> getIo(){
		return io;
	}
	
	public void addIo(Matrix m) {
		io.add(m);
	}

	public Matrix getOutput() {
		return io.getLast();
	}
	
	public LinkedList<Matrix> getWeightErrors() {
		return weightErrors;
	}

	public void addWeightErrors(Matrix weightErrors) {
		this.weightErrors.addFirst(weightErrors);
	}

	public Matrix getInputError() {
		return inputError;
	}

	public void setInputError(Matrix inputError) {
		this.inputError = inputError;
	}

	public void setLoss(double loss) {
		this.loss = loss;
	}
	
	public double getLoss() {
		return this.loss;
	}

	public void setPercentCorrect(double percentCorrect) {
		this.percentCorrect = percentCorrect;
	}
	
	public double getPercentCorrect() {
		return this.percentCorrect;
	}
	
}
