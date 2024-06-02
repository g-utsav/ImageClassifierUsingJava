package nn.matrix;

import java.util.Arrays;
import java.util.Objects;

public class Matrix {
	
	private static final String NUMBER_FORMAT = "%+12.7f";
	private  double tolerance = 0.000001;
	private int rows;
	private int cols;
	private double[] a;
	
	public interface Producer{
		double producer(int index);
	}
	
	public interface ValueProducer{
		double producer(double value);
	}
	
	public interface IndexValueProducer{
		double producer(int index, double value);
	}
	
	public interface IndexValueConsumer{
		void consume(int index, double value);
	}
	
	public interface RowColValueConsumer{
		void consume(int row, int col, double value);
	}
	
	public interface RowColIndexValueConsumer{
		void consume(int row, int col, int index, double value);
	}
	
	public interface RowColProducer{
		double producer(int row, int col, double value);
	}
	
	public Matrix(int rows, int cols) {
		this.rows = rows;
		this.cols = cols;
		
		a = new double[rows*cols];
	}
	
	public Matrix(int rows, int cols, Producer producer) {
		this(rows,cols);
		
		for(int i=0; i<a.length; i++) {
			a[i] = producer.producer(i);
		}
	}
	
	public Matrix apply(ValueProducer producer) {
		var result = new Matrix(rows, cols);
		
		for(int i=0; i<a.length; i++) {
			result.a[i] = producer.producer(a[i]);
		}
		
		return result;
	}
	
	public Matrix apply(IndexValueProducer producer) {
		var result = new Matrix(rows, cols);
		
		for(int i=0; i<a.length; i++) {
			result.a[i] = producer.producer(i,a[i]);
		}
		
		return result;
	}
	
	public double get(int index) {
		return a[index];
	}
	
	public Matrix sumColumns() {
		Matrix result = new Matrix(1, cols);
		
		int index = 0;
		for(int row = 0; row < rows; row++) {
			for(int col = 0; col < cols; col++) {
				result.a[col] += a[index++];
			}
		}
		
		return result;
	}
	
	public Matrix softMax() {
		Matrix result  = new Matrix(rows, cols, i -> Math.exp(a[i]));
		
		Matrix colSum = result.sumColumns();
		
		result.modify((row, col, value) -> {
			return value/colSum.get(col);
		});
		
		return result;
	}
	
	public int getRows() {
		return rows;
	}
	
	public int getCols() {
		return cols;
	}
	
	public Matrix multiply(Matrix m) {
		Matrix result = new Matrix(rows, m.cols);
		
		assert cols == m.rows : "Cannont multiply, Wrong number of rows vs cols";
		/*
		 *row, col, n -> 16ms
		 *row, n, col -> 9ms
		 *col, row, n -> 17ms
		 *col, n, row -> 28ms
		 *n, row, col -> 9ms
		 *n, col, row -> 29ms
		 */	
		for(var row = 0; row<result.rows; row++) {
			for(var n=0; n<cols; n++) {
				for(var col =0; col<result.cols; col++) {
					result.a[row*result.cols + col] += a[row * cols + n] * m.a[col + n * m.cols];
				}
			}
		}
		
		return result;
	}
	
	public Matrix modify(RowColProducer producer) {
		var index = 0;
		for(var row = 0; row < rows; ++row) {
			for(var col = 0; col < cols; ++col) {
				a[index] = producer.producer(row, col, a[index]);
				++index;
			}
		}
		
		return this;
	}
	
	public Matrix modify(ValueProducer producer) {
		for(int i=0; i<a.length; i++) {
			a[i] = producer.producer(a[i]);
		}
		return this;
	}
		
	public void forEach(IndexValueConsumer consumer) {
		for(int i=0; i<a.length; i++) {
			consumer.consume(i, a[i]);
		}
	}
	
	public void forEach(RowColValueConsumer consumer) {
		int index = 0;
		for(int row = 0; row < rows; row++) {
			for(int col = 0; col < cols; col++) {
				consumer.consume(row, col, a[index++]);
			}
		}
	}
	
	public void forEach(RowColIndexValueConsumer consumer) {
		int index = 0;
		for(int row = 0; row < rows; row++) {
			for(int col = 0; col < cols; col++) {
				consumer.consume(row, col, index, a[index++]);
			}
		}
	}
	
	public Matrix addIncrement(int row, int col, double increment) {
		Matrix result = apply((index,value)->a[index]);
		
		double originalValue = get(row,col);
		double newValue = originalValue + increment;
		
		result.set(row, col, newValue);
		
		return result;
	}
	
	public void set(int row, int col, double value) {
		a[row * cols + col] = value;
	}
	
	public double get(int row, int col) {
		return a[row * cols + col];
	}
	
	public Matrix transpose() {
		Matrix result = new Matrix(cols, rows);
		
		for(int i = 0; i < a.length; i++) {
			int row = i / cols;
			int col = i % cols;
			
			result.a[col * rows + row] = a[i];
		}
		
		return result;
	}
	
	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + Arrays.hashCode(a);
		result = prime * result + Objects.hash(cols, rows);
		return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		Matrix other = (Matrix) obj;
		
		for(int i=0; i<a.length; i++) {
			if(Math.abs(a[i]-other.a[i]) > tolerance) return false;
		}
		
		return true;
	}
	
	public void setTolerance(double tolerance) {
		this.tolerance = tolerance;
	}

	public String toString() {
		StringBuilder sb = new StringBuilder();
		
		int index = 0;
		for(int row=0; row<rows; row++) {
			for(int col=0; col<cols; col++) {	
				sb.append(String.format(NUMBER_FORMAT, a[index]));
				
				index++;
			}
			sb.append("\n");
		}
		
		return sb.toString();
	}
	
	public String toString(boolean showValue) {
		if(showValue) {
			return toString();
		}else {
			return rows +"x"+ cols;
		}
	}
}
