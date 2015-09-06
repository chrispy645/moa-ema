package moa.classifiers.functions;

import java.util.ArrayList;
import java.util.Arrays;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;
import weka.core.converters.ConverterUtils.DataSource;

public class TestSparseEMA {
	
	private static void deleteme(Instances data) {
		System.out.println(data);	
		Instance inst = data.get(2);
		System.out.println( inst.numValues() ); // should be 3
		for(int x = 0; x < inst.numValues(); x++) {
			if( inst.index(x) == inst.classIndex() ) {
				continue;
			}
			System.out.print( inst.index(x) + "," + inst.valueSparse(x) + "  " );
		}
		System.out.println("");
	}
	
	public static void main(String[] args) throws Exception {
		
		DataSource ds = new DataSource("datasets/simple2_sparse.arff"); // sparse version works for dense?
		
		//DataSource ds = new DataSource("/Users/cjb60/github/590-cyber-sec/data/saul-data-ema-arff/computer-scientists-scientist-17.arff");
		Instances data = ds.getDataSet();
		data.setClassIndex( data.numAttributes() - 1);
		
		
		//deleteme(data);
		//System.exit(0);
		
		SparseEMA ema = new SparseEMA();
		for(Instance inst : data) {
			ema.trainOnInstance(inst);
		}
		
		TestEMA.printArray( ema.getWeightMatrix() );
		
		
		
	}

}
