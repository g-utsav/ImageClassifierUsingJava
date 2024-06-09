package nn.neuralnetwork.loader.test;

import static org.junit.Assert.*;

import org.junit.Test;

import nn.loader.BatchData;
import nn.loader.Loader;
import nn.loader.MetaData;
import nn.loader.test.TestLoader;
import nn.matrix.Matrix;

public class TestTestLoader {

	@Test
	public void test() {
		
		int batchSize = 33;
		Loader testLoader = new TestLoader(600, batchSize);
		
		MetaData metaData = testLoader.open();
		
		int numberItems = metaData.getNumberItems();
		int lastBatchSize = numberItems % batchSize;
		int numberBatches = metaData.getNumberBatches();
		
		for(int i=0; i<numberBatches; i++) {
			BatchData batchData = testLoader.readBatch();
			
			assertTrue(batchData != null);
			
			int itemsRead = metaData.getItemsRead();
			
			int inputSize = metaData.getInputSize();
			int expectedSize = metaData.getExpectedSize();
			Matrix input = new Matrix(inputSize, itemsRead, batchData.getInputBatch());
			Matrix expected = new Matrix(expectedSize, itemsRead, batchData.getExpectedBatch());
			
			assertTrue(input.sum() != 0);
			assertTrue(expected.sum() == itemsRead);
			
			if(i == numberBatches-1) {
				assertTrue(itemsRead == lastBatchSize);
			}else {
				assertTrue(itemsRead == batchSize);				
			}
		}
	}

}
