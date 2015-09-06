package moa.classifiers.functions;

import java.util.Arrays;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class TestBoth {
	
	public static void main(String[] args) throws Exception {
		
		DataSource ds = new DataSource("datasets/simple2_sparse.arff");
		Instances data = ds.getDataSet();
		data.setClassIndex( data.numAttributes() - 1);
		double[][] sparsePreds = new double[ data.numInstances() ][];
		SparseEMA sparseEma = new SparseEMA();
		// train
		for(int x = 0; x < data.numInstances(); x++) {
			sparseEma.trainOnInstance(data.get(x));
		}
		// predict
		for(int x = 0; x < data.numInstances(); x++) {
			sparsePreds[x] = sparseEma.getVotesForInstance(data.get(x));
		}
		
		//TestEMA.printArray( sparseEma.getWeightMatrix() );
		
		System.out.println("");
		
		ds = new DataSource("datasets/simple2.arff");
		data = ds.getDataSet();
		data.setClassIndex( data.numAttributes() - 1);
		double[][] densePreds = new double[ data.numInstances() ][];
		EMA ema = new EMA();
		// train
		for(int x = 0; x < data.numInstances(); x++) {
			ema.trainOnInstance(data.get(x));
		}
		// predict
		for(int x = 0; x < data.numInstances(); x++) {
			densePreds[x] = ema.getVotesForInstance(data.get(x));
		}
		
		//TestEMA.printArray( ema.getWeightMatrix() );
		
		// see if weight matrices are the same
		System.out.println( Arrays.deepEquals(ema.getWeightMatrix(), sparseEma.getWeightMatrix()) );
		
		// see if preds are the same
		for(int x = 0; x < data.numInstances(); x++) {
			System.out.println( Arrays.equals(sparsePreds[x], densePreds[x]) );
		}
	}

}
