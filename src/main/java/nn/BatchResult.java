package nn;

import java.util.LinkedList;

import nn.matrix.Matrix;

public class BatchResult {

	private LinkedList<Matrix> io = new LinkedList<Matrix>();
	private LinkedList<Matrix> weightErrors = new LinkedList<Matrix>();
	private Matrix inputError;
	
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
	
	
}
