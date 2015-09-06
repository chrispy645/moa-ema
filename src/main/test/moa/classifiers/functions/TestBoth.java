package moa.classifiers.functions;

import java.util.Arrays;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class TestBoth {
	
	public static void main(String[] args) throws Exception {	
		DataSource ds = new DataSource("datasets/simple2_sparse.arff");
		Instances data = ds.getDataSet();
		data.setClassIndex( data.numAttributes() - 1);
		SparseEMA sparseEma = new SparseEMA();
		for(int x = 0; x < data.numInstances(); x++) {
			sparseEma.trainOnInstance(data.get(x));
		}
		TestEMA.printArray( sparseEma.getWeightMatrix() );
		
		System.out.println("");
		
		ds = new DataSource("datasets/simple2.arff");
		data = ds.getDataSet();
		data.setClassIndex( data.numAttributes() - 1);
		EMA ema = new EMA();
		for(int x = 0; x < data.numInstances(); x++) {
			ema.trainOnInstance(data.get(x));
		}		
		TestEMA.printArray( ema.getWeightMatrix() );
		
		System.out.println( Arrays.deepEquals(ema.getWeightMatrix(), sparseEma.getWeightMatrix()) );
	}

}
