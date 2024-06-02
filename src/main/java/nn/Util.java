package nn;

import java.util.Random;

import nn.matrix.Matrix;

public class Util {
	
	private static Random random = new Random();
	
	public static  Matrix generateInputMatrix(int rows, int cols) {
		return new Matrix(rows, cols, i->random.nextGaussian());
	}
	
	public static  Matrix generateExpectedMatrix(int rows, int cols) {
		Matrix expected = new Matrix(rows, cols, i->0);
		
		for(int col=0; col<expected.getCols(); col++) {
			int randomRow = random.nextInt(rows);
			expected.set(randomRow, col, 1);
		}	
		
		return expected;
	}
}
