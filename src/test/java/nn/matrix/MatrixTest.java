package nn.matrix;

import static org.junit.Assert.*;

import java.util.Random;

import org.junit.Test;

public class MatrixTest {

	private Random random = new Random();
	static int rows = 3;
	static int cols = 3;
	
	@Test
	public void testGetGreatestRowNumber() {
		double[] values = {2,-6,2,7,2,-6,10,-1,1};
		Matrix m = new Matrix(3,3, i->values[i]);
		double[] expectedValues = {2,1,0};
		Matrix expected = new Matrix(1,3,i->expectedValues[i]);
		Matrix result = m.getGreatestRowNumbers();
		
		assertTrue(expected.equals(result));
		
//		System.out.println(expected);
//		System.out.println(m);
//		System.out.println(result);
	}
	
	@Test
	public void testAverageColumn() {
		int rows = 7;
		int cols = 5;
		Matrix m = new Matrix(rows, cols, i ->2*i-3);
		double averageIndex = (cols - 1)/2.0;
		Matrix expected = new Matrix(rows , 1);
		expected.modify((row, col, value)->2*(row*cols+averageIndex)-3);
		
		Matrix result = m.averageColumn();
		
		assertTrue(expected.equals(result));
		
//		System.out.println(m);
//		System.out.println(result);
//		System.out.println(expected);
	}
	
	@Test
	public void testTranspose() {
		Matrix m = new Matrix(2,3,i->i);
		Matrix result = m.transpose();
		double[] expectedValue = {0,3,1,4,2,5};
		
		Matrix expected = new Matrix(3,2, i->expectedValue[i]);
		
		assertTrue(expected.equals(result));
		
//		System.out.println(m);
//		System.out.println(result);
	}
	
	@Test
	public void testAddIncrement() {
		Matrix m = new Matrix(5,8, i->random.nextGaussian());
		
		int row = 3;
		int col = 2;
		
		double inc  = 5.001;
		
		Matrix result = m.addIncrement(row, col, inc);
		
//		System.out.println(m);
//		System.out.println(result);
				
	}
	
	@Test
	public void testSoftMax() {
		Matrix m = new Matrix(5,8, i->random.nextGaussian());
		
		Matrix result = m.softMax();
		
//		System.out.println(m);
//		System.out.println(result);
		
		double[] colSums = new double[8];
		result.forEach((row, col, value) -> {
			assertTrue(value >= 0 && value <=1);
			
			colSums[col] += value;
		});
		
		for(var sum : colSums) {
			assertTrue(Math.abs(sum - 1.0) < 0.000001);			
		}
		
	}
	
	@Test
	public void testEquals() {
		Matrix  m1 = new Matrix(rows, cols, i->1.5*(i-6));
		Matrix  m2 = new Matrix(rows, cols, i->1.5*(i-6));
		Matrix  m3 = new Matrix(rows, cols, i->0.5*(i-6));
		
		assertTrue(m1.equals(m2));
		
		assertFalse(m1.equals(m3));
	}
	
	@Test
	public void testAddMatrices() {
		Matrix m1 = new Matrix(rows, cols, i->i);
		Matrix m2 = new Matrix(rows, cols, i->1.5*i);
		Matrix expected = new Matrix(rows, cols, i->2.5*i);
	
		Matrix result = m1.apply((index, value)->value+m2.get(index));
		
		assertTrue(expected.equals(result));
		
//		System.out.println(m1);
//		System.out.println(m2);
//		System.out.println(expected);
//		System.out.println(result);
	}
	
	@Test
	public void testMultiply() {
		Matrix  m1 = new Matrix(2, 3, i->i);
		Matrix  m2 = new Matrix(3, 2, i->i);
		
		double[] expectedValues = {10,13,28,40};
		Matrix expected = new Matrix(2,2,i->expectedValues[i]);
		
		Matrix result = m1.multiply(m2);
		
		assertTrue(expected.equals(result));
		
//		System.out.println(m1.toString());
//		System.out.println(m2.toString());
//		System.out.println(result.toString());
	}
	
	@Test
	public void testSumCOlumns() {
		Matrix m1 = new Matrix(4,5, i -> i);
		
		Matrix result = m1.sumColumns();
		
//		System.out.println(m1);
//		System.out.println(result);
		
		double[] expectedValues = {+30.00000, +34.00000, +38.00000, +42.00000, +46.00000};
		Matrix expected = new Matrix(1,5, i->expectedValues[i]);
		
		assertTrue(expected.equals(result));
	}
	
	@Test
	public void testMultiplySpeend() {
		
		int rows = 500;
		int cols = 500;
		int mid = 50;
		
		Matrix  m1 = new Matrix(rows, mid, i->i);
		Matrix  m2 = new Matrix(mid, cols, i->i);
		for(int i=0; i<0; i++)m1.multiply(m2);
		
		int sum = 0;
		for(int i=0; i<0; i++) {
			var start = System.currentTimeMillis();
			m1.multiply(m2);
			var end = System.currentTimeMillis();
			sum += end-start;
		}
		System.out.printf("Matrix multiplication time take: %dms\n",sum/30);
	}
	
	@Test
	public void multiplyDouble() {
		
		Matrix  m = new Matrix(rows, cols, i->1.5*(i-6));
		
		double x = 0.5;
		
		Matrix expected  = new Matrix(rows, cols, i->x*1.5*(i-6));
		
		Matrix result = m.apply((value)->x*value);
		
		assertTrue(result.equals(expected));
		
		System.out.println();
		
		assertTrue(Math.abs(Math.abs(result.get(1))-Math.abs(expected.get(1)))<0.00001);
	}
	
	@Test
	public void testToString() {
		
		
		Matrix m = new Matrix(rows, cols, i->i*2);
		
		String text = m.toString();
//		System.out.println(text);
		
		double[] expected = new double[rows*cols];
		
		for(var i=0; i<expected.length; i++) {
			expected[i] = i*2;
		}
		
		var rowsText = text.split("\n");
		
		assertTrue(rowsText.length == rows);
		
		int index = 0;
		
		for(var row : rowsText) {
			var values = row.split("\\s+");
			for(var textValue : values) {
				if(textValue.length() == 0)continue;
				
				var doubleValue = Double.valueOf(textValue);
				
				assertTrue(Math.abs(doubleValue - expected[index]) < 0.0001);
				
				index++;
			}
		}
	}

}
