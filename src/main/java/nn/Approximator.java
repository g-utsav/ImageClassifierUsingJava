package nn;

import java.util.function.Function;

import nn.matrix.Matrix;

public class Approximator {

	public static Matrix gradient(Matrix input, Function<Matrix, Matrix> transform) {
		
		final double INC = 0.0000001;
		
		Matrix loss1 = transform.apply(input);
	
		assert loss1.getCols() == input.getCols() : "Input/loss columns not equal";
		assert loss1.getRows() == 1 : "Transform doesnot return 1 single row";
		
//		System.out.println(input);
		
//		System.out.println("loss " +loss1);
		
		Matrix result = new Matrix(input.getRows(), input.getCols(), i->0);
		
		input.forEach((row, col, index, value)->{
			
			Matrix incremented = input.addIncrement(row, col, INC);
			
			Matrix loss2 = transform.apply(incremented);
			
			double rate = (loss2.get(col) - loss1.get(col))/INC;
			
			result.set(row, col, rate);
			
//			System.out.printf("%+12.5f", value);
//			
//			if(col == input.getCols()-1) {
//				System.out.println();
//			}
		});
		
		return result;
	}
	
}
